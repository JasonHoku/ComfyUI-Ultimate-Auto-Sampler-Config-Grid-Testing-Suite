"""
Generation Orchestrator - Main Entry Point
Coordinates the entire grid generation workflow
"""

import os
import json
import time
import re
import gc
import threading
import torch
import hashlib
import folder_paths
import comfy.sd  # Required for async workers
from comfy.model_management import InterruptProcessingException

from .trigger_words import collect_unique_prompts_with_triggers, build_prompt_with_triggers, clear_trigger_caches, omit_without_lookup_warning
from .regional_prompt import append_to_base, validate_regional_prompts, has_regions
from .batch_encoding import batch_encode_prompts, encode_prompt_with_combinators, pinned_clip, DEFAULT_TE_BATCH_SIZE
from .manifest_utils import load_existing_manifest, save_manifest, item_filename
from .model_loader import (
    load_checkpoint, load_loras, cleanup_model_references,
    get_latent_channels, get_latent_downscale_ratio, make_empty_latent, load_loras_for_preencoding,
    print_incompatible_loras_summary, load_diffusion_model_and_clip,
    load_vae_by_name
)
from .lora_utils import expand_lora_folder
from .image_generation import (
    generate_image, generate_image_attn_safe, flush_batch_with_vae, flush_batch_with_remote_vae,
    create_image_metadata, decode_latent_with_vae, pil_to_image_array, calculate_eta, print_generation_progress
)
from .config_utils import sanitize_session_name
from .html_generator import get_html_template
from .conditioning_cache import ConditioningCache
from .remote_vae import RemoteVAEDecodeWorker, is_remote_vae_available, get_endpoint_names, INSTALL_INSTRUCTIONS
from . import distribution
from . import image_metrics

try:
    from server import PromptServer
except ImportError:
    PromptServer = None


def setup_session_directories(session_name):
    """Create session directories and return paths."""
    base_dir = os.path.join(folder_paths.get_output_directory(), "benchmarks", session_name)
    img_dir = os.path.join(base_dir, "images")
    manifest_path = os.path.join(base_dir, "manifest.json")
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    return {
        "base": base_dir,
        "images": img_dir,
        "manifest": manifest_path
    }


def initialize_remote_vae(remote_vae_endpoint, img_dir, manifest_path, existing_data, session_name, unique_id, image_format="webp"):
    """Initialize remote VAE worker if enabled."""
    if not remote_vae_endpoint or remote_vae_endpoint == "None":
        return None

    if remote_vae_endpoint in ["SD", "SDXL", "Flux", "HunyuanVideo"]:
        # Pass the endpoint NAME through to the worker; the companion plugin
        # (ComfyUI-USCG-RemoteVAE) resolves the name to the actual URL.
        actual_endpoint = remote_vae_endpoint
        print(f"[GridTester] 🌐 Using {remote_vae_endpoint} remote VAE (resolved by companion plugin)")
    elif remote_vae_endpoint == "Auto (Experimental)":
        print(f"[GridTester] 🌐 Auto mode selected - worker will initialize on first flush")
        return None
    else:
        actual_endpoint = remote_vae_endpoint
        print(f"[GridTester] 🌐 Using custom endpoint: {actual_endpoint}")

    worker = RemoteVAEDecodeWorker(
        endpoint=actual_endpoint,
        img_dir=img_dir,
        manifest_path=manifest_path,
        existing_data=existing_data,
        session_name=session_name,
        unique_id=unique_id,
        image_format=image_format
    )
    print(f"[GridTester] 🌐 Remote VAE worker started")
    return worker


REMOTE_VAE_PREFIX = "remote:"

def is_remote_vae(vae_string):
    """Check if a VAE value is a per-config remote URL (e.g. 'remote:http://...')."""
    return isinstance(vae_string, str) and vae_string.startswith(REMOTE_VAE_PREFIX)

def normalize_config_vae(vae):
    """A config's ``vae`` value as the loop sees it: None / "" / "None" mean "Default" (use the
    checkpoint's bundled VAE, or the sampler node's remote VAE when one is set)."""
    if not vae or str(vae) in ("None", "none"):
        return "Default"
    return vae


def configs_needing_work(job_plan, status_fn):
    """Configs (each once, in plan order) with at least one plan entry still 'todo'.

    This is the pre-encoding filter: prompts are only encoded for jobs that will actually run, so a
    resumed run skips the text-encoder work for everything already on disk.
    """
    needed, seen = [], set()
    for pj in job_plan:
        if id(pj["conf"]) in seen:
            continue
        if status_fn(pj) == "todo":
            seen.add(id(pj["conf"]))
            needed.append(pj["conf"])
    return needed


def extract_remote_vae_url(vae_string):
    """Extract the URL from a remote VAE string like 'remote:http://...'.

    If the URL is empty:
      - When the companion plugin is NOT installed, raise RuntimeError with
        install instructions (the most likely user state — Builder UI shows
        install card so URL field is empty).
      - When the companion IS installed, raise ValueError asking the user to
        pick a preset or enter a URL.
    """
    url = vae_string[len(REMOTE_VAE_PREFIX):]
    if not url:
        if not is_remote_vae_available():
            raise RuntimeError(INSTALL_INSTRUCTIONS)
        raise ValueError(
            "[GridTester] Per-config remote VAE URL is empty.\n"
            "Please provide a valid endpoint URL in the config builder's VAE section."
        )
    return url


def _flush_pending_batch(pending_batch, current_vae_is_remote, current_remote_vae_url,
                         per_config_remote_workers, use_remote_vae, remote_vae_worker,
                         loaded_vae, paths, existing_data, session_name, manifest_path, unique_id,
                         config_overrides_vae=False, image_format="webp"):
    """Flush pending batch using the appropriate VAE decode method.

    Handles three-way dispatch:
      1. Per-config remote VAE (current_vae_is_remote) — uses per-config worker
      2. Global remote VAE (use_remote_vae) — uses global remote_vae_worker
         ONLY when config_overrides_vae is False (config VAE is "Default")
      3. Local VAE — uses loaded_vae for local decode

    Config Builder VAE settings take priority over the sampler node's
    remote_vae_endpoint when a config explicitly sets a non-Default VAE.
    """
    if not pending_batch:
        return
    if current_vae_is_remote and current_remote_vae_url:
        worker = per_config_remote_workers.get(current_remote_vae_url)
        if worker:
            flush_batch_with_remote_vae(pending_batch, worker, existing_data, session_name)
        else:
            print(f"[GridTester] ⚠️ No remote worker for {current_remote_vae_url}, falling back to local VAE")
            flush_batch_with_vae(pending_batch, loaded_vae, paths["images"], existing_data, session_name, manifest_path, unique_id, image_format=image_format)
    elif not config_overrides_vae and use_remote_vae and remote_vae_worker:
        flush_batch_with_remote_vae(pending_batch, remote_vae_worker, existing_data, session_name)
    else:
        flush_batch_with_vae(pending_batch, loaded_vae, paths["images"], existing_data, session_name, manifest_path, unique_id, image_format=image_format)


def _cleanup_per_config_remote_workers(per_config_remote_workers):
    """Shut down all per-config remote VAE workers."""
    for url, worker in per_config_remote_workers.items():
        try:
            print(f"[GridTester] 🌐 Waiting for per-config remote VAE ({url})...")
            worker.wait_completion()
            worker.stop()
        except Exception as e:
            print(f"[GridTester] ⚠️ Error stopping per-config remote worker ({url}): {e}")


def calculate_clip_hash(clip_model):
    """Calculate a hash of the CLIP model for cache validation."""
    try:
        if hasattr(clip_model, 'state_dict'):
            state_dict = clip_model.state_dict()
            model_signature = str([(k, tuple(v.shape)) for k, v in list(state_dict.items())[:10]])
        elif hasattr(clip_model, 'cond_stage_model'):
            model_signature = str(type(clip_model.cond_stage_model))
        else:
            model_signature = str(type(clip_model))
        return hashlib.md5(model_signature.encode()).hexdigest()[:16]
    except:
        return "unknown"


def check_if_job_completed(existing_items, conf, seed, width, height, batch_idx, positive_prompt, negative_prompt, has_optional_inputs=False, *, preview=False):
    """Independent check to see if a specific generation job already exists.

    When has_optional_inputs is True, we cannot reliably detect changes from
    upstream nodes (models, LoRAs, conditionings, latents), so we disable
    skip detection entirely and return -1 (no match) to force re-generation.

    ``preview`` (keyword-only): preview-pass items (``item["preview"] is True``) and
    full items never match each other. A preview job matches on the FULL width/height
    stored in ``full_width``/``full_height`` (the preview's own latent size depends on
    the model's latent ratio, which is unknown before the model loads); callers pass a
    conf whose steps/cfg are the preview values. Legacy items have no ``preview`` key
    and count as full. The distributed companion calls this positionally, so full
    behaviour is the default.
    """
    # When optional inputs are connected, we can't reliably match on model,
    # lora, prompts, or latents because those may come from upstream nodes
    # whose changes we can't detect. Disable skip detection entirely.
    if has_optional_inputs:
        return -1

    FLOAT_TOLERANCE = 0.0001

    for idx, item in enumerate(existing_items):
        if bool(item.get("preview", False)) != preview: continue
        if item.get("seed") != seed: continue
        if preview:
            if item.get("full_width", item.get("width")) != width: continue
            if item.get("full_height", item.get("height")) != height: continue
        else:
            if item.get("width") != width: continue
            if item.get("height") != height: continue
        if item.get("batch_idx", 0) != batch_idx: continue
        if item.get("sampler") != conf["sampler"]: continue
        if item.get("scheduler") != conf["scheduler"]: continue

        # Check attention mode (default = no attention_mode key or "default")
        item_attn = item.get("attention_mode", "default")
        conf_attn = conf.get("attention_mode", "default")
        if item_attn != conf_attn: continue

        try:
            if abs(float(item.get("steps")) - float(conf["steps"])) > FLOAT_TOLERANCE: continue
            if abs(float(item.get("cfg")) - float(conf["cfg"])) > FLOAT_TOLERANCE: continue
            if abs(float(item.get("denoise")) - float(conf["denoise"])) > FLOAT_TOLERANCE: continue
        except (ValueError, TypeError):
            continue

        # Standard matching - check model, lora, and prompts
        if item.get("model") != conf["model"]: continue
        if item.get("positive", "").strip() != positive_prompt.strip(): continue
        if item.get("negative", "").strip() != negative_prompt.strip(): continue

        item_lora = item.get("lora", "None")
        conf_lora = conf.get("lora_expanded", "None")
        if item_lora != conf_lora: continue

        # Check model_type and text_encoders — different text encoders produce
        # different conditioning even with the same model file, so they must
        # NOT be considered duplicate jobs
        item_model_type = item.get("model_type", "checkpoint")
        conf_model_type = conf.get("model_type", "checkpoint")
        if item_model_type != conf_model_type: continue

        item_te = item.get("text_encoders", [])
        conf_te = conf.get("text_encoders", [])
        if item_te != conf_te: continue

        # Check clip_type — different clip types produce different conditioning
        item_clip_type = item.get("clip_type", "stable_diffusion")
        conf_clip_type = conf.get("clip_type", "stable_diffusion")
        if item_clip_type != conf_clip_type: continue

        # Check clip_skip — different clip_skip values produce different conditioning
        item_clip_skip = item.get("clip_skip", 0)
        conf_clip_skip = conf.get("clip_skip", 0)
        if item_clip_skip != conf_clip_skip: continue

        # Check VAE — different VAEs produce different decoded images.
        # Normalize "None"/empty to "Default" so legacy configs that emit
        # "vae": "None" don't appear distinct from configs that omit the
        # key entirely.
        def _norm_vae(v):
            if not v or str(v) in ("None", "none"):
                return "Default"
            return v
        item_vae = _norm_vae(item.get("vae", "Default"))
        conf_vae = _norm_vae(conf.get("vae", "Default"))
        if item_vae != conf_vae: continue

        return idx

    return -1


def _ltx_sort_components(x):
    """Return tuple of LTX-specific sort components (None for non-LTX entries)."""
    if x.get('model_type') != 'ltx_video':
        return (None, None, None, None)
    cm = x.get('clip_models') or []
    return (
        tuple(cm),
        x.get('vae_video', ''),
        x.get('vae_audio', ''),
        x.get('latent_upscaler', ''),
    )


def get_model_cache_key(conf):
    """Generate a cache key that uniquely identifies the model+clip combination."""
    model_type = conf.get("model_type", "checkpoint")
    if model_type == "checkpoint":
        return conf["model"]
    elif model_type == "ltx_video":
        clip_models = conf.get("clip_models", ["", ""])
        clip_a = clip_models[0] if len(clip_models) > 0 else ""
        clip_b = clip_models[1] if len(clip_models) > 1 else ""
        return (
            "ltx::" + conf["model"] + "::" + clip_a + "::" + clip_b + "::" +
            conf.get("vae_video", "") + "::" + conf.get("vae_audio", "") + "::" +
            conf.get("latent_upscaler", "")
        )
    else:
        te_key = "|".join(sorted(conf.get("text_encoders", [])))
        return f"{conf['model']}::{model_type}::{te_key}"


def _build_ltx_manifest_entry(conf, gen_result, output_filename, gen_index=None, session_name=""):
    """Build a manifest entry for an LTX video. Mirrors image-gen entry shape
    but with media_type='video' and LTX-specific fields.

    The `file` key is the ComfyUI /view? URL the dashboard uses to fetch the mp4
    via Comfy's standard view endpoint — same format the image-gen path uses
    (see image_generation.py:844)."""
    import random
    mp4_filename = output_filename + ".mp4"
    file_url = f"/view?filename={mp4_filename}&type=output&subfolder=benchmarks/{session_name}/images"
    # Unique timestamp-based id (matches image_generation.py:828 pattern). Used by
    # the dashboard for sorting, position tracking, and DOM identity.
    item_id = int(time.time() * 100000) + random.randint(0, 1000)
    return {
        "id": item_id,
        "gen_index": gen_index,
        "media_type": "video",
        # The dashboard's loader reads `file` (ComfyUI /view? URL) for the actual
        # video src. image_path/video_path stored as duplicates for any code that
        # needs the absolute disk path.
        "file": file_url,
        "image_path": gen_result["video_path"],
        "video_path": gen_result["video_path"],
        "preview_path": gen_result.get("preview_path"),
        "width": gen_result["width"],
        "height": gen_result["height"],
        "duration_seconds": gen_result["duration_seconds"],
        "frame_rate": gen_result["fps"],
        "frames": gen_result["frames"],
        "model_type": "ltx_video",
        "model": conf["model"],
        "clip_models": conf.get("clip_models", []),
        "vae_video": conf.get("vae_video", ""),
        "vae_audio": conf.get("vae_audio", ""),
        "latent_upscaler": conf.get("latent_upscaler", ""),
        "sampler_stage1": conf.get("sampler_stage1", ""),
        "sigmas_stage1": conf.get("sigmas_stage1", ""),
        "sampler_stage2": conf.get("sampler_stage2", ""),
        "sigmas_stage2": conf.get("sigmas_stage2", ""),
        "image_strength_stage1": conf.get("image_strength_stage1", 0.8),
        "image_strength_stage2": conf.get("image_strength_stage2", 1.0),
        "img_compression": conf.get("img_compression", 18),
        "input_image": conf.get("input_image"),
        "audio_mode": conf.get("audio_mode", "on"),
        "cfg": conf.get("cfg", 1.0),
        "seed": conf.get("seed"),
        "positive": conf.get("positive", ""),
        "negative": conf.get("negative", ""),
        "lora": conf.get("lora", "None"),
        "duration": gen_result.get("duration", 0),
        "favorited": False,
        "rejected": False,
        "note": "",
    }


def load_model_by_type(conf, ckpt_name, use_remote_vae, optional_model, optional_clip, optional_vae,
                       optional_positive, optional_negative, loaded_clip, loaded_vae, model_cache):
    """Dispatch to correct loader based on model_type in config."""
    model_type = conf.get("model_type", "checkpoint")
    target = conf["model"]

    if model_type == "checkpoint":
        return load_checkpoint(
            target, ckpt_name, use_remote_vae,
            optional_model, optional_clip, optional_vae,
            optional_positive, optional_negative,
            loaded_clip, loaded_vae, model_cache=model_cache
        )
    else:
        # diffusion_model or gguf
        return load_diffusion_model_and_clip(
            model_name=target,
            model_type=model_type,
            text_encoder_paths=conf.get("text_encoders", []),
            clip_type_str=conf.get("clip_type", "stable_diffusion"),
            gguf_options=conf.get("gguf_options"),
            use_remote_vae=use_remote_vae,
            optional_model=optional_model,
            optional_clip=optional_clip,
            optional_vae=optional_vae,
            model_cache=model_cache
        )


def run_deferred_upscales(
    deferred_queue, upscale_settings, session_settings,
    loaded_vae, patched_model, patched_clip, conditioning_cache,
    paths, existing_data, session_name, unique_id,
    PromptServer, config_overrides_vae=None
):
    """
    Process deferred upscale jobs after generation completes.
    Runs all pipeline chains for each queued image.
    Called BEFORE finalization cleanup so VAE/model/CLIP are still loaded.

    Queue entries: config, width, height, seed, batch_idx, actual_positive_prompt,
    actual_negative_prompt, gen_index, config_fp and optionally base_item (the exact
    manifest item to upscale; skips the identity search). The caller supplies a
    model/VAE/CLIP/conditioning_cache that matches every job in the queue.
    """
    if not deferred_queue:
        return

    from .image_generation import upscale_image, decode_latent_with_vae, create_image_metadata
    import itertools as up_itertools
    import random as up_random
    from PIL import Image as PILImage
    import comfy.model_management as mm
    import numpy as np
    import torch

    pipelines = upscale_settings.get("pipelines", [])
    if not pipelines:
        return

    total_jobs = len(deferred_queue)
    print(f"\n[GridTester] 🔄 Starting deferred upscale phase: {total_jobs} images")

    upscale_durations = []

    for job_idx, job in enumerate(deferred_queue):
        # Check for interrupt
        if mm.processing_interrupted():
            print(f"\n[GridTester] 🛑 INTERRUPTED during deferred upscaling")
            break

        conf = job["config"]
        pipe_w = job["width"]
        pipe_h = job["height"]
        current_seed = job["seed"]
        actual_positive_prompt = job["actual_positive_prompt"]
        actual_negative_prompt = job["actual_negative_prompt"]

        # Distributed callers hand over the exact manifest item; local entries search by identity
        base_item = job.get("base_item")
        if base_item is None:
            # Find the base image in the manifest by matching identity
            for item in existing_data.get("items", []):
                if (item.get("seed") == current_seed and
                    item.get("width") == pipe_w and
                    item.get("height") == pipe_h and
                    not item.get("upscaled") and
                    item.get("positive") == actual_positive_prompt):
                    base_item = item
                    break

        if not base_item:
            print(f"[GridTester] ⚠️ Deferred upscale #{job_idx+1}: base image not found in manifest, skipping")
            continue

        # generated items carry only a /view URL ("file"); upscaled/scanned ones also have "filename"
        base_filename = item_filename(base_item)
        if not base_filename:
            print(f"[GridTester] ⚠️ Deferred upscale #{job_idx+1}: no filename in manifest, skipping")
            continue

        base_image_path = os.path.join(paths["images"], base_filename)
        if not os.path.exists(base_image_path):
            print(f"[GridTester] ⚠️ Deferred upscale #{job_idx+1}: file not found: {base_filename}, skipping")
            continue

        # Check if already upscaled (resume support)
        already_upscaled = False
        _job_fp = job.get("config_fp")
        for item in existing_data.get("items", []):
            if not item.get("upscaled"):
                continue
            if _job_fp:
                # keyed on the base cell's identity (sampler/cfg/... included): a fixed-seed grid has
                # many cells sharing seed + prompt, and each one needs its own upscale
                if item.get("base_fp") == _job_fp:
                    already_upscaled = True
                    break
                continue
            # legacy manifests / queue entries without a fingerprint: seed + prompt, and the same base size
            if (item.get("seed") == current_seed and
                item.get("positive") == actual_positive_prompt and
                item.get("width") != pipe_w and
                item.get("base_width", pipe_w) == pipe_w and
                item.get("base_height", pipe_h) == pipe_h):
                already_upscaled = True
                break
        if already_upscaled:
            print(f"[GridTester] ⏭️ Deferred upscale #{job_idx+1}: already upscaled, skipping")
            continue

        # Load base image from disk and VAE-encode back to latent
        try:
            pil_image = PILImage.open(base_image_path).convert("RGB")
            img_array = np.array(pil_image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            encoded = loaded_vae.encode(img_tensor[:, :, :, :3])
            result_latent = {"samples": encoded}
        except Exception as e:
            print(f"[GridTester] ⚠️ Deferred upscale #{job_idx+1}: failed to load/encode: {e}")
            continue

        # Look up conditioning from cache
        final_positive = conditioning_cache["positive"].get(actual_positive_prompt)
        final_negative = conditioning_cache["negative"].get(actual_negative_prompt)

        if final_positive is None or final_negative is None:
            print(f"[GridTester] ⚠️ Deferred upscale #{job_idx+1}: conditioning not in cache, re-encoding")
            try:
                clip_skip = conf.get("clip_skip", 0)
                with pinned_clip(patched_clip):  # one encoder load for both prompts
                    if final_positive is None:
                        final_positive = encode_prompt_with_combinators(patched_clip, actual_positive_prompt, clip_skip)
                        conditioning_cache["positive"][actual_positive_prompt] = final_positive
                    if final_negative is None:
                        final_negative = encode_prompt_with_combinators(patched_clip, actual_negative_prompt, clip_skip)
                        conditioning_cache["negative"][actual_negative_prompt] = final_negative
            except Exception as e:
                print(f"[GridTester] ⚠️ Deferred upscale #{job_idx+1}: re-encode failed: {e}, skipping")
                continue

        # HiRes prompt adjustment
        hires_positive_cond = final_positive
        hires_prompt_active = False
        hires_prompt_behavior_rt = ""
        hires_prompt_text_rt = ""
        if upscale_settings.get("hires_prompt_adjust") and upscale_settings.get("hires_prompt_text", "").strip():
            hires_prompt_behavior_rt = upscale_settings.get("hires_prompt_behavior", "append_end")
            hires_prompt_text_rt = upscale_settings["hires_prompt_text"].strip()
            if hires_prompt_behavior_rt == "prepend":
                adjusted_prompt = hires_prompt_text_rt + " " + actual_positive_prompt
            elif hires_prompt_behavior_rt == "append_end":
                adjusted_prompt = append_to_base(actual_positive_prompt, hires_prompt_text_rt)
            elif hires_prompt_behavior_rt == "replace":
                adjusted_prompt = hires_prompt_text_rt
            else:
                adjusted_prompt = actual_positive_prompt
            hires_cond = conditioning_cache["positive"].get(adjusted_prompt)
            if hires_cond is None:
                # not cached (fresh per-model cache): encode it, keep the base conditioning on failure
                try:
                    with pinned_clip(patched_clip):
                        hires_cond = encode_prompt_with_combinators(patched_clip, adjusted_prompt, conf.get("clip_skip", 0))
                    conditioning_cache["positive"][adjusted_prompt] = hires_cond
                except Exception as e:
                    print(f"[GridTester] ⚠️ Deferred upscale #{job_idx+1}: HiRes prompt encode failed: {e}, using base prompt")
                    hires_cond = None
            if hires_cond is not None:
                hires_positive_cond = hires_cond
                hires_prompt_active = True

        job_start_time = time.time()
        upscale_combo_idx = 0

        for pipeline_idx, pipeline in enumerate(pipelines):
            if pipeline.get("active", True) is False:
                continue
            pipeline_name = pipeline.get("name", f"Pipeline {pipeline_idx + 1}")
            pipeline_steps = pipeline.get("steps", [])
            if not pipeline_steps:
                continue

            pipe_latent = result_latent
            pipe_w_current = pipe_w
            pipe_h_current = pipe_h

            expanded_steps = []
            for step in pipeline_steps:
                if step.get("active", True) is False:
                    continue
                repeat = max(1, int(step.get("repeat", 1)))
                for _ in range(repeat):
                    expanded_steps.append(step)

            for step_idx, ucfg in enumerate(expanded_steps):
                mode = ucfg.get("mode", "hires_only")
                show_hires = mode in ("hires_only", "model_then_hires")
                show_model = mode in ("model_only", "model_then_hires")

                raw_ratios = str(ucfg.get("upscale_ratios", "1.5"))
                ratios = [float(r.strip()) for r in raw_ratios.split(",") if r.strip()] or [1.5]
                raw_denoise = str(ucfg.get("hires_denoise", "0.3"))
                denoises = [float(d.strip()) for d in raw_denoise.split(",") if d.strip()] or [0.3]
                models = ucfg.get("upscale_models", []) or [""]

                if show_hires and show_model:
                    combos = list(up_itertools.product(models, ratios, denoises))
                elif show_hires:
                    combos = list(up_itertools.product([""], ratios, denoises))
                elif show_model:
                    combos = list(up_itertools.product(models, [1.0], [0.0]))
                else:
                    combos = []

                for combo in combos:
                    up_model, up_ratio, up_denoise = combo
                    single_config = {
                        "mode": mode,
                        "upscale_model": up_model,
                        "upscale_ratio": up_ratio,
                        "hires_denoise": up_denoise,
                        "hires_steps": ucfg.get("hires_steps", 0),
                        "tiled_vae": ucfg.get("tiled_vae", False),
                        "tile_size": ucfg.get("tile_size", 512),
                        "upscale_size": ucfg.get("upscale_size", "2.0"),
                        "resize_method": ucfg.get("resize_method", "bilinear"),
                        "hires_tiled_sampling": ucfg.get("hires_tiled_sampling", False),
                        "hires_tile_width": ucfg.get("hires_tile_width", 512),
                        "hires_tile_height": ucfg.get("hires_tile_height", 512),
                        "hires_mask_blur": ucfg.get("hires_mask_blur", 8),
                        "hires_tile_padding": ucfg.get("hires_tile_padding", 32),
                        "hires_force_uniform_tiles": ucfg.get("hires_force_uniform_tiles", False)
                    }

                    up_positive = hires_positive_cond if (show_hires and hires_prompt_active) else final_positive
                    upscale_result, upscale_duration = upscale_image(
                        pipe_latent, loaded_vae, patched_model, single_config,
                        conf, up_positive, final_negative, pipe_w_current, pipe_h_current
                    )

                    is_last_step = step_idx == len(expanded_steps) - 1
                    is_last_combo = combo == combos[-1]
                    is_final_output = is_last_step and is_last_combo

                    up_metrics_sink = {}
                    up_metrics_img = None
                    if isinstance(upscale_result, dict) and "samples" in upscale_result:
                        upscaled_pil = decode_latent_with_vae(
                            loaded_vae, upscale_result["samples"],
                            metrics_sink=up_metrics_sink if is_final_output else None)
                        up_metrics_img = upscaled_pil
                        up_w, up_h = upscaled_pil.size
                        if is_final_output:
                            upscale_id = int(time.time() * 100000) + up_random.randint(0, 1000)
                            upscaled_filename = f"img_{upscale_id}_upscaled.webp"
                            upscaled_pil.save(
                                os.path.join(paths["images"], upscaled_filename),
                                format="WEBP", quality=95
                            )
                    elif isinstance(upscale_result, PILImage.Image):
                        up_metrics_img = upscale_result
                        up_w, up_h = upscale_result.size
                        if is_final_output:
                            upscale_id = int(time.time() * 100000) + up_random.randint(0, 1000)
                            upscaled_filename = f"img_{upscale_id}_upscaled.webp"
                            upscale_result.save(
                                os.path.join(paths["images"], upscaled_filename),
                                format="WEBP", quality=95
                            )
                    else:
                        continue

                    if not is_final_output:
                        if isinstance(upscale_result, dict) and "samples" in upscale_result:
                            pipe_latent = upscale_result
                        pipe_w_current = up_w
                        pipe_h_current = up_h
                        continue

                    # Final output: create manifest entry
                    upscaled_meta = create_image_metadata(
                        conf, up_w, up_h, upscale_duration, current_seed, job.get("batch_idx", 0),
                        actual_positive_prompt, actual_negative_prompt,
                        gen_index=job.get("gen_index", 0)
                    )
                    upscaled_meta["id"] = upscale_id
                    upscaled_meta["filename"] = upscaled_filename
                    upscaled_meta["upscaled"] = True
                    upscaled_meta["base_width"], upscaled_meta["base_height"] = pipe_w, pipe_h
                    upscaled_meta["upscale_pipeline"] = pipeline_name
                    if job.get("config_fp"):
                        from .config_utils import config_fingerprint as _cfp
                        upscaled_meta["base_fp"] = job["config_fp"]
                        upscaled_meta["config_fp"] = _cfp(conf, pipe_w, pipe_h, conf.get("seed"), pipeline_name)
                    upscaled_meta["upscale_mode"] = mode
                    upscaled_meta["upscale_ratio"] = up_ratio
                    upscaled_meta["upscale_denoise"] = up_denoise
                    if up_model:
                        upscaled_meta["upscale_model"] = up_model
                    if hires_prompt_active:
                        upscaled_meta["hires_prompt_behavior"] = hires_prompt_behavior_rt
                        upscaled_meta["hires_prompt_text"] = hires_prompt_text_rt

                    image_metrics.finalize(upscaled_meta, up_metrics_img, up_metrics_sink)
                    existing_data["items"].append(upscaled_meta)
                    upscale_combo_idx += 1

        if upscale_combo_idx > 0:
            save_manifest(paths["manifest"], existing_data)
            if PromptServer is not None:
                try:
                    PromptServer.instance.send_sync("ultimate_grid.update_data", {
                        "node": unique_id,
                        "session_name": session_name,
                        "new_items": [existing_data["items"][-1]]
                    })
                except Exception:
                    pass

        job_duration = time.time() - job_start_time
        upscale_durations.append(job_duration)

        # Progress reporting
        if PromptServer is not None:
            try:
                avg_dur = sum(upscale_durations) / len(upscale_durations)
                remaining = (total_jobs - (job_idx + 1)) * avg_dur
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                eta_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
                PromptServer.instance.send_sync("ultimate_grid.progress", {
                    "node": unique_id,
                    "session_name": session_name,
                    "current_job": job_idx + 1,
                    "total_jobs": total_jobs,
                    "progress_pct": int(((job_idx + 1) / total_jobs) * 100),
                    "eta_str": eta_str,
                    "avg_duration": round(avg_dur, 1),
                    "last_duration": round(job_duration, 1),
                    "phase": "upscaling",
                })
            except Exception:
                pass

        print(f"[GridTester] 🔄 Deferred upscale {job_idx+1}/{total_jobs} complete ({job_duration:.1f}s)")

    print(f"[GridTester] ✅ Deferred upscale phase complete: {len(upscale_durations)}/{total_jobs} images upscaled")


# Item fields that describe THIS image rather than its config (stripped when a config is rebuilt from an item).
_ITEM_ONLY_KEYS = (
    "id", "file", "filename", "favorited", "rejected", "note", "duration", "gen_index",
    "config_fp", "base_fp", "metrics", "upscaled", "base_width", "base_height",
)


def _conf_from_item(item):
    """Best-effort job config rebuilt from a manifest item (fallback when no distributed job matches)."""
    conf = {k: v for k, v in item.items() if k not in _ITEM_ONLY_KEYS}
    conf["positive"] = item.get("config_positive", item.get("positive", ""))
    conf["negative"] = item.get("config_negative", item.get("negative", ""))
    conf.setdefault("lora_expanded", item.get("lora", "None"))
    return conf


def _build_distributed_upscale_queue(items, pre_run_ids, jobs):
    """
    Build the deferred-upscale queue for a distributed run.

    items: existing_data["items"] (newest first); pre_run_ids: ids present before the run
    (only this run's images are upscaled); jobs: [(config, input_job)] of COMPLETED jobs.
    Worker-inserted items carry only a /view URL (no filename, no config_fp), so each item is
    matched back to its job config with the same rule resume uses (check_if_job_completed).
    Returns entries shaped like the local deferred queue plus ``base_item``.
    """
    from .config_utils import config_fingerprint

    by_key = {}
    for cfg, ij in jobs:
        by_key.setdefault((cfg.get("model"), cfg.get("sampler"), cfg.get("scheduler")), []).append((cfg, ij))

    queue = []
    for item in reversed(items):  # oldest first = generation order
        if item.get("id") in pre_run_ids:
            continue
        if item.get("upscaled") or item.get("preview") or item.get("media_type") == "video":
            continue
        if item.get("model_type") == "ltx_video":
            continue
        if not item_filename(item):
            print(f"[Distribution] ⚠️ Upscale: item {item.get('id')} has no filename, skipping")
            continue

        w, h = item.get("width"), item.get("height")
        seed = item.get("seed")
        bidx = item.get("batch_idx", 0)
        pos, neg = item.get("positive", ""), item.get("negative", "")
        matches = []
        for cfg, _ij in by_key.get((item.get("model"), item.get("sampler"), item.get("scheduler")), []):
            try:
                if check_if_job_completed([item], cfg, seed, w, h, bidx, pos, neg) == 0:
                    matches.append(cfg)
            except Exception:
                continue
        if matches:
            cfg = next((c for c in matches if c.get("seed") == seed), matches[0])
        else:
            cfg = _conf_from_item(item)

        queue.append({
            "config": cfg, "width": w, "height": h, "seed": seed,
            "batch_idx": bidx, "actual_positive_prompt": pos,
            "actual_negative_prompt": neg, "gen_index": item.get("gen_index", 0),
            "config_fp": config_fingerprint(cfg, w, h, cfg.get("seed")),
            "base_item": item,
        })
    return queue


def _run_distributed_upscales(
    queue, upscale_settings, session_settings, paths, existing_data, session_name, unique_id,
    ckpt_name, use_remote_vae, optional_model, optional_clip, optional_vae,
    optional_positive, optional_negative, model_cache, incompatible_loras
):
    """
    Run a distributed run's upscales on the master, one (model, LoRA, VAE) group at a time.
    Each group loads its own model and gets a fresh conditioning cache (conditioning from another
    model's CLIP is invalid), then everything is released before the next group.
    """
    import comfy.model_management as mm

    def _norm_vae(c):
        v = c.get("vae", "Default")
        return "Default" if (not v or str(v) in ("None", "none")) else v

    groups = {}
    for entry in queue:
        c = entry["config"]
        key = (get_model_cache_key(c), c.get("lora_expanded", c.get("lora", "None")), _norm_vae(c))
        groups.setdefault(key, []).append(entry)

    if optional_positive is not None or optional_negative is not None:
        print("[Distribution] ℹ️ Positive/negative inputs are connected: upscale conditioning comes from the prompts, not those inputs")

    PromptServer = None
    try:
        from server import PromptServer as _PS
        PromptServer = _PS
    except Exception:
        pass

    for (model_key, lora_str, vae_name), group in groups.items():
        if mm.processing_interrupted():
            print("[Distribution] 🛑 INTERRUPTED during distributed upscales")
            break
        cfg0 = group[0]["config"]
        loaded_model = loaded_clip = loaded_vae = patched_model = patched_clip = None
        cache = {"positive": {}, "negative": {}}
        try:
            loaded_model, loaded_clip, loaded_vae = load_model_by_type(
                cfg0, ckpt_name, use_remote_vae, optional_model, optional_clip, optional_vae,
                optional_positive, optional_negative, None, None, model_cache=model_cache
            )
            if loaded_vae is None and optional_vae is not None:
                loaded_vae = optional_vae
            if vae_name != "Default":
                if is_remote_vae(vae_name):
                    print(f"[Distribution] ℹ️ Remote VAE '{vae_name}': upscale encode/decode stays local with the model's VAE")
                else:
                    loaded_vae = load_vae_by_name(vae_name)
            patched_model, patched_clip, should_skip = load_loras(
                loaded_model, loaded_clip, lora_str, model_key, incompatible_loras, model_cache=model_cache
            )
            if should_skip:
                print(f"[Distribution] ⚠️ LoRA incompatible for {model_key}; skipping {len(group)} upscales")
                continue
            if loaded_vae is None:
                print(f"[Distribution] ⚠️ No local VAE for {model_key}; skipping {len(group)} upscales")
                continue
            run_deferred_upscales(
                group, upscale_settings, session_settings, loaded_vae, patched_model, patched_clip,
                cache, paths, existing_data, session_name, unique_id, PromptServer
            )
        except Exception as e:
            if isinstance(e, InterruptProcessingException):
                raise
            print(f"[Distribution] ⚠️ Upscale group {model_key} failed: {e}")
        finally:
            patched_model, patched_clip = cleanup_model_references(patched_model, patched_clip, cache)
            loaded_model = loaded_clip = loaded_vae = None
            gc.collect()


def _expand_lora_weight_arrays(expanded_configs):
    """
    Expand LoRA weight array bracket notation into Cartesian product of configs.

    Input format per lora part: "name:[0.5, 0.8]:1.0" or "name:0.5:[0.7, 1.0]"
    Weight arrays use brackets in the STRENGTH fields (after first colon),
    while folder random syntax uses brackets in the NAME field (before first colon).

    If multiple loras in the same config have weight arrays, produces a Cartesian
    product of all combinations:
      lora1:[0.5, 1.0] + lora2:[0.7, 1.0] → 4 configs
    """
    import re
    from itertools import product as itertools_product

    result = []
    weight_array_pattern = re.compile(r'\[([^\]]+)\]')

    for conf in expanded_configs:
        lora_str = conf.get("lora", "None")
        if lora_str == "None" or ":" not in lora_str:
            result.append(conf)
            continue

        lora_parts = lora_str.split(" + ")
        # Check each part for weight arrays (brackets in strength fields, not name field)
        parts_with_arrays = []
        has_weight_arrays = False

        for part in lora_parts:
            part = part.strip()
            colon_idx = part.find(":")
            if colon_idx < 0:
                # No strength specified — plain lora name
                parts_with_arrays.append([(part, None, None)])
                continue

            name = part[:colon_idx]
            strength_str = part[colon_idx + 1:]

            # If brackets are in the name portion, it's folder syntax — skip
            if "[" in name:
                parts_with_arrays.append([(part, None, None)])
                continue

            # Parse strength fields for bracket arrays
            # Format: "model_str:clip_str" where either can be "[val1, val2]"
            strength_parts = strength_str.split(":")
            model_strs = strength_parts[0] if len(strength_parts) > 0 else "1.0"
            clip_strs = strength_parts[1] if len(strength_parts) > 1 else model_strs

            # Extract arrays from model strength
            model_match = weight_array_pattern.search(model_strs)
            if model_match:
                model_vals = [v.strip() for v in model_match.group(1).split(",")]
                has_weight_arrays = True
            else:
                model_vals = [model_strs.strip()]

            # Extract arrays from clip strength
            clip_match = weight_array_pattern.search(clip_strs)
            if clip_match:
                clip_vals = [v.strip() for v in clip_match.group(1).split(",")]
                has_weight_arrays = True
            else:
                clip_vals = [clip_strs.strip()]

            # Build all combinations for this lora part
            combos = []
            for mv in model_vals:
                for cv in clip_vals:
                    combos.append((name, mv, cv))
            parts_with_arrays.append(combos)

        if not has_weight_arrays:
            result.append(conf)
            continue

        # Cartesian product across all lora parts
        for combo in itertools_product(*parts_with_arrays):
            new_lora_parts = []
            for item in combo:
                name, mv, cv = item
                if mv is None:
                    # Passthrough (folder syntax or plain name)
                    new_lora_parts.append(name)
                else:
                    new_lora_parts.append(f"{name}:{mv}:{cv}")

            new_conf = conf.copy()
            new_conf["lora"] = " + ".join(new_lora_parts)
            result.append(new_conf)

    if len(result) != len(expanded_configs):
        print(f"[GridTester] 🎯 LoRA weight arrays expanded: {len(expanded_configs)} → {len(result)} configs")

    return result


def run_generation_loop(
    self,
    ckpt_name, positive_text, negative_text, seed, denoise, vae_batch_size,
    overwrite_existing, flush_batch_every, configs_json, resolutions_json,
    session_name, unique_id, add_random_seeds_to_gens, lora_triggerwords_mode,
    remote_vae_endpoint, save_conditioning_cache_to_file, enable_model_cache,
    optional_model, optional_clip, optional_vae,
    optional_positive, optional_negative, optional_latent,
    distribution_config=None,
    session_settings=None
):
    """Main generation loop orchestrator."""

    # Clear stale trigger word caches from previous runs (within same ComfyUI session).
    # Without this, @lru_cache and _build_prompt_cache can serve outdated results
    # if loras_tags.json was updated by a CivitAI fetch in a prior run.
    clear_trigger_caches()

    from .model_cache import ModelCache

    # Initialize Model Cache (or disable completely if user requested)
    if enable_model_cache:
        model_cache = ModelCache(
            max_models=1,        # Adjust for your VRAM
            max_lora_sets=2,    # Adjust for your VRAM
            max_lora_files=30,   # Adjust for your VRAM
            enable_preload=True,   # Preloading enabled when cache is on
            async_preload=True,    # Always async when preloading
            cache_device='cpu',
            verbose=True
        )
    else:
        # Completely disable entire cache system
        print("[GridTester] ⚠️ Model cache DISABLED - all models will load from disk (slower but saves RAM/VRAM)")
        model_cache = None

    # ==== SETUP ====
    session_name = sanitize_session_name(session_name)
    paths = setup_session_directories(session_name)
    existing_data = load_existing_manifest(paths["manifest"])
    existing_data["session_name"] = session_name
    
    from .config_utils import (
        parse_json_with_error, parse_float_input, parse_prompt_input_nested, derive_extra_seeds,
        expand_configs, prepare_input_jobs, latent_job_pixel_size
    )
    from .config_utils import (
        JOB_ORDERS, build_job_plan, order_job_plan, count_switches, config_fingerprint,
        preview_dims, build_preview_index, should_skip_for_previews, build_full_fp_set,
        run_meta_extras, job_order_note
    )

    # ==== JOB ORDER / PREVIEW PASS SETTINGS (defaults reproduce the old behaviour) ====
    _ss = session_settings or {}
    job_order = _ss.get("job_order", "as_written")
    if job_order not in JOB_ORDERS:
        job_order = "as_written"
    pv = _ss.get("preview_pass") or {}
    preview_on = bool(pv.get("enabled"))
    if preview_on and optional_latent is not None:
        print(f"[GridTester] ⚠️ Preview pass ignored: a latent input is connected (its size is fixed)")
        preview_on = False
    try:
        pv_steps = max(1, int(pv.get("steps", 8)))
    except (TypeError, ValueError):
        pv_steps = 8
    try:
        pv_scale = min(1.0, max(0.05, float(pv.get("scale", 0.5))))
    except (TypeError, ValueError):
        pv_scale = 0.5
    try:
        pv_cfg = None if pv.get("cfg_override") in (None, "") else float(pv.get("cfg_override"))
    except (TypeError, ValueError):
        pv_cfg = None
    skip_rejected_previews = bool(_ss.get("skip_rejected_previews", True))
    only_favorited_previews = bool(_ss.get("only_favorited_previews", False))

    raw_configs = parse_json_with_error(configs_json, "configs")
    denoise_values = parse_float_input(denoise)
    resolutions = parse_json_with_error(resolutions_json, "resolutions")
    
    pos_prompts = parse_prompt_input_nested(positive_text)
    neg_prompts = parse_prompt_input_nested(negative_text)
    
    extra_seeds = []
    if add_random_seeds_to_gens > 0:
        # Deterministic per base seed (see config_utils.derive_extra_seeds) so a
        # re-run / resumed run matches already-rendered images by seed.
        extra_seeds = derive_extra_seeds(seed, add_random_seeds_to_gens)
    
    expanded = expand_configs(raw_configs, pos_prompts, neg_prompts, denoise_values, seed, extra_seeds, ckpt_name)
    expanded.sort(key=lambda x: (
        x.get('model_type', 'checkpoint'),
        x['model'],
        *_ltx_sort_components(x),
        tuple(x.get('text_encoders', [])),
        x.get('vae', 'Default'),
        x['lora'],
        x.get('attention_mode', 'default'),
        x['positive'],
        x['negative'],
    ))
    
    # ==== EXPAND LORA WEIGHT ARRAYS (Cartesian product) ====
    # Format: "lora:[0.5, 0.8]:1.0" → expand into separate configs per strength combo
    expanded = _expand_lora_weight_arrays(expanded)

    # ==== EXPAND LORA FOLDERS ====
    print(f"[GridTester] 🎲 Expanding LoRA folders and random selections...")
    for conf in expanded:
        if conf["lora"] != "None":
            lora_parts = conf["lora"].split(" + ")
            expanded_parts = []
            for part in lora_parts:
                part = part.strip()
                if "[" in part and "]" in part:
                    expanded_lora = expand_lora_folder(part, seed=conf.get("seed"))
                    if expanded_lora:
                        if isinstance(expanded_lora, list):
                            expanded_parts.extend(expanded_lora)
                        else:
                            expanded_parts.append(expanded_lora)
                else:
                    expanded_parts.append(part)
            conf["lora_expanded"] = " + ".join(expanded_parts) if expanded_parts else "None"
            conf["lora"] = conf["lora_expanded"]
        else:
            conf["lora_expanded"] = "None"
    print(f"[GridTester] ✅ LoRA expansion complete")

    # ==== VAE VALIDATION FOR NON-CHECKPOINT MODELS ====
    # Only error if non-checkpoint model has no VAE source (no optional_vae AND no per-config VAE).
    # LTX video configs are exempt — they declare vae_video and vae_audio in their own LTX
    # block, validated separately by preflight_ltx() at gen time.
    use_remote_vae = remote_vae_endpoint and remote_vae_endpoint != "None"
    _omit_warn = omit_without_lookup_warning(expanded, lora_triggerwords_mode)
    if _omit_warn:
        print(f"[GridTester] ⚠️ {_omit_warn}")
    needs_vae = any(
        c.get("model_type", "checkpoint") not in ("checkpoint", "ltx_video")
        and normalize_config_vae(c.get("vae", "Default")) == "Default"
        for c in expanded
    )
    if needs_vae and not use_remote_vae and optional_vae is None:
        raise ValueError(
            "GGUF and diffusion models do not include a bundled VAE.\n"
            "You must either:\n"
            "  1. Connect a VAE to the optional_vae input on the sampler node, or\n"
            "  2. Enable remote VAE decoding via remote_vae_endpoint, or\n"
            "  3. Set a specific VAE in the config builder for those configs\n"
        )

    # ==== REGISTER SMART CACHE SCHEDULE ====
    if model_cache:
        model_cache.register_schedule(expanded)

    input_jobs = prepare_input_jobs(optional_latent, resolutions)

    # Count total jobs: configs with per-config resolutions provide their own
    # width/height and don't multiply with the global input_jobs.
    configs_with_res = sum(1 for c in expanded if c.get("resolution") is not None)
    configs_without_res = len(expanded) - configs_with_res
    total_jobs = configs_with_res + (configs_without_res * len(input_jobs))

    # ==== FLAT JOB PLAN (+ optional reload-minimising order) ====
    # One entry per job, in exactly the order the old nested loop visited them
    # (resolution outer, configs inner) unless job_order says otherwise.
    _plan_as_written = build_job_plan(expanded, input_jobs)
    job_plan = order_job_plan(_plan_as_written, job_order, get_model_cache_key)
    if len(job_plan) != total_jobs:
        print(f"[GridTester] ⚠️ Job plan has {len(job_plan)} entries but total_jobs is {total_jobs}")
    if job_order == "minimize_reloads":
        _sw_before = count_switches(_plan_as_written, get_model_cache_key)
        _sw_after = count_switches(job_plan, get_model_cache_key)
        print(f"[GridTester] 🧭 Job order: minimize_reloads | model loads {_sw_before['model_loads']} -> {_sw_after['model_loads']}, "
              f"LoRA switches {_sw_before['lora_switches']} -> {_sw_after['lora_switches']}, "
              f"VAE switches {_sw_before['vae_switches']} -> {_sw_after['vae_switches']}")
        _note = job_order_note(_sw_before, _sw_after)
        if _note:
            print(f"[GridTester] ℹ️ Job order: {_note}")

    print(f"{'='*80}")
    print(f"[GridTester] 🚀 GENERATION START")
    if job_order != "as_written":
        print(f"[GridTester] 🧭 Job order: {job_order} (Start At Job numbers follow this order)")
    if configs_with_res > 0:
        print(f"[GridTester] 📋 {configs_with_res} configs with per-config resolutions + "
              f"{configs_without_res} configs × {len(input_jobs)} resolutions = {total_jobs} total jobs")
    else:
        print(f"[GridTester] 📋 {len(expanded)} configs × {len(input_jobs)} resolutions = {total_jobs} total jobs")
    print(f"{'='*80}")
    
    # ==== OPTIONAL INPUT DETECTION ====
    # Track whether any optional inputs are connected - this affects skip/resume logic
    has_optional_inputs = any([
        optional_model is not None,
        optional_clip is not None,
        optional_vae is not None,
        optional_positive is not None,
        optional_negative is not None,
        optional_latent is not None
    ])
    _meta_run = run_meta_extras(session_settings, has_optional_inputs)

    if has_optional_inputs and not overwrite_existing:
        print(f"[GridTester] ⚠️ Optional inputs connected — job skip/resume DISABLED.")
        print(f"[GridTester] ⚠️ Changes to upstream nodes (models, LoRAs, prompts, latents) connected via optional inputs")
        print(f"[GridTester] ⚠️ cannot be automatically detected, so all jobs will be re-generated.")
        print(f"[GridTester] ⚠️ To use resume mode, disconnect optional inputs and use the built-in config fields instead.")

    # ==== DISTRIBUTED PROCESSING BRANCH ====
    # If distribution_config is provided and enabled, delegate to distributed processing
    # which handles the generation loop, then jump to finalization
    dist_enabled = (
        distribution_config
        and isinstance(distribution_config, dict) 
        and distribution_config.get("enabled")
        and distribution_config.get("worker_urls")
    )

    if distribution_config:
        print(f"[GridTester] 🌐 Distribution check: type={type(distribution_config).__name__}, "
              f"enabled={distribution_config.get('enabled') if isinstance(distribution_config, dict) else 'N/A'}, "
              f"workers={len(distribution_config.get('worker_urls', [])) if isinstance(distribution_config, dict) else 'N/A'}, "
              f"dist_enabled={dist_enabled}")
    else:
        print(f"[GridTester] ℹ️ distribution_config is None/empty, running normal generation")

    # ==== PREVIEW PASS: no upscaling at all (one change point covers inline, deferred, HiRes pre-encode) ====
    session_settings_full = session_settings  # pre-override: a legacy distributed companion ignores the preview pass
    if preview_on:
        session_settings = {**(session_settings or {}), "upscaling": {"enabled": False}}
        print(f"[GridTester] 🔎 Preview pass: {pv_steps} steps, x{pv_scale} scale"
              f"{'' if pv_cfg is None else f', cfg {pv_cfg}'}, upscales skipped")
        if any(c.get("model_type") == "ltx_video" for c in expanded):
            print(f"[GridTester] ℹ️ Preview pass does not apply to LTX video configs - they render normally")

    if dist_enabled:
        print(f"[GridTester] 🌐 ENTERING DISTRIBUTED MODE with {len(distribution_config.get('worker_urls', []))} worker(s)")
        # Upscaling in distributed mode: workers generate base images, master runs upscales after collection
        if session_settings and session_settings.get("upscaling", {}).get("enabled", False):
            print(f"[GridTester] ℹ️ Upscaling enabled — workers will generate base images, master will run upscales after all generation completes.")
        return _run_distributed_generation(
            self, distribution_config, expanded, input_jobs, existing_data,
            overwrite_existing, has_optional_inputs, lora_triggerwords_mode,
            session_name, paths, unique_id, total_jobs, seed, ckpt_name,
            positive_text, negative_text, vae_batch_size, configs_json,
            resolutions_json, flush_batch_every, use_remote_vae,
            remote_vae_endpoint, save_conditioning_cache_to_file,
            enable_model_cache, optional_model, optional_clip, optional_vae,
            optional_positive, optional_negative, optional_latent, model_cache,
            session_settings=session_settings,
            job_order=job_order, preview_on=preview_on, pv_steps=pv_steps, pv_scale=pv_scale,
            pv_cfg=pv_cfg, meta_run=_meta_run, job_plan=job_plan,
            skip_rejected_previews=skip_rejected_previews, only_favorited_previews=only_favorited_previews,
            upscale_session_settings=session_settings_full
        )

    # The schedule above used `expanded`; the main loop walks job_plan, so the smart cache needs
    # plan positions (identical to the old behaviour when there is one resolution and as_written).
    if model_cache:
        model_cache.register_schedule([pj["conf"] for pj in job_plan])

    # ==== OPTIONAL CONDITIONING SETUP ====
    if optional_positive or optional_negative:
        pos_hash = hashlib.md5(str(optional_positive).encode()).hexdigest()[:16] if optional_positive else None
        neg_hash = hashlib.md5(str(optional_negative).encode()).hexdigest()[:16] if optional_negative else None
    else:
        pos_hash, neg_hash = None, None

    # ==== REMOTE VAE SETUP (use_remote_vae already defined above for VAE validation) ====

    try:
        if PromptServer is not None:
            pbar = PromptServer.instance.progress_bar_pool.get_progress_bar(unique_id)
        else:
            pbar = None
    except:
        pbar = None
    
    # ==== STATE VARIABLES ====
    loaded_model, loaded_clip, loaded_vae = None, None, None
    patched_model, patched_clip = None, None
    cached_model_key = None
    cached_lora_key = None
    cached_lora_cache_key = None
    cached_vae_key = None  # Track which VAE is currently loaded
    default_model_vae = None  # Track the model's bundled/default VAE for reverting
    conditioning_cache = {"positive": {}, "negative": {}}
    incompatible_loras = {}
    pending_batch = []
    current_job = 0
    new_completed = 0  # New generations completed in this run (excludes pre-existing skips)
    total_generated = 0
    gen_index_offset = len(existing_data.get("items", []))  # Sequential index for deterministic sort ordering
    skipped_count = 0
    start_at_job = int(session_settings.get("start_at_job", 0)) if session_settings else 0
    if start_at_job > 0:
        print(f"[GridTester] ⏭️ Start At Job #{start_at_job} — skipping earlier jobs")
    image_format = session_settings.get("image_format", "webp") if session_settings else "webp"
    if image_format != "webp":
        print(f"[GridTester] 🖼️ Image save format: {image_format}")
    job_durations = []
    deferred_upscale_queue = []
    eta_start_time = time.time()

    # Initialize remote VAE worker
    remote_vae_worker = None
    if use_remote_vae and expanded:
        remote_vae_worker = initialize_remote_vae(
            remote_vae_endpoint,
            paths["images"],
            paths["manifest"],
            existing_data,
            session_name,
            unique_id,
            image_format=image_format
        )
    
    # Per-config remote VAE state
    per_config_remote_workers = {}   # Keyed by URL, reused across configs
    current_vae_is_remote = False
    current_remote_vae_url = None
    config_overrides_vae = False  # True when config explicitly sets a non-Default VAE (overrides sampler node's remote_vae_endpoint)

    # ==== SKIP PREDICATE (single source of truth for "does this plan entry still need work?") ====
    # `_pstate` holds the preview index (rebuilt lazily whenever the item list changes, i.e. after
    # every flush/save, so dashboard rejects/favourites merged in by save_manifest are seen mid-run).
    _pstate = {"sig": None, "pindex": {}, "full_fps": set()}

    def _have_history():
        return bool(existing_data.get("items")) or bool((existing_data.get("meta") or {}).get("rejected_preview_fps"))

    def _refresh_pstate():
        items = existing_data.get("items", [])
        sig = (len(items), id(items[0]) if items else None)
        if sig == _pstate["sig"]:
            return
        _pstate["sig"] = sig
        _pstate["pindex"] = build_preview_index(items, (existing_data.get("meta") or {}).get("rejected_preview_fps", []))
        _pstate["full_fps"] = build_full_fp_set(items) if preview_on else set()

    def _effective_conf(c):
        """The conf a PREVIEW job actually samples with (fewer steps, optional cfg override)."""
        return {**c, "steps": pv_steps, "cfg": c["cfg"] if pv_cfg is None else pv_cfg}

    def _promotion_status(c, fp):
        """'rejected_preview' / 'no_favorited_preview' (full run) or 'full_exists' (preview run) or None."""
        if has_optional_inputs or c.get("model_type") == "ltx_video" or fp is None:
            return None
        _refresh_pstate()
        if preview_on:
            return "full_exists" if fp in _pstate["full_fps"] else None
        return should_skip_for_previews(fp, _pstate["pindex"], skip_rejected_previews, only_favorited_previews)

    def _job_status(pj):
        """'exists' | 'rejected_preview' | 'no_favorited_preview' | 'full_exists' | 'todo'."""
        c = pj["conf"]
        is_pv = preview_on and c.get("model_type") != "ltx_video"
        pos_, _ = build_prompt_with_triggers(c, lora_triggerwords_mode)
        if check_if_job_completed(
            existing_data["items"], _effective_conf(c) if is_pv else c, c["seed"],
            pj["w"], pj["h"], pj["batch_idx"], pos_, c["negative"],
            has_optional_inputs=has_optional_inputs, preview=is_pv
        ) != -1:
            return "exists"
        if has_optional_inputs or c.get("model_type") == "ltx_video":
            return "todo"
        return _promotion_status(c, config_fingerprint(c, pj["w"], pj["h"], c["seed"])) or "todo"

    # ==== PRE-ENCODING STAGE ====
    unique_model_keys = set(get_model_cache_key(conf) for conf in expanded)

    # LTX video runs use their own dual-CLIP encoder inside ltx_video_generate(),
    # so skip the standard image-gen pre-encoding entirely when every config is LTX.
    all_ltx = expanded and all(c.get("model_type") == "ltx_video" for c in expanded)

    if not all_ltx and not (optional_positive and optional_negative) and expanded and len(unique_model_keys) == 1:
        first_conf = job_plan[0]["conf"] if job_plan else expanded[0]
        target_model_name = first_conf["model"]

        print(f"[GridTester] ✅ Single model detected ({target_model_name}) - enabling pre-encoding")

        loaded_model, loaded_clip, loaded_vae = load_model_by_type(
            first_conf, ckpt_name, use_remote_vae,
            optional_model, optional_clip, optional_vae,
            optional_positive, optional_negative, None, None, model_cache=model_cache
        )

        # VAE fallback for non-checkpoint models
        if loaded_vae is None and optional_vae is not None:
            loaded_vae = optional_vae
        # Remember this model's default VAE for reverting later
        default_model_vae = loaded_vae

        # Per-config VAE: if first config specifies a VAE, load it.
        # Config Builder VAE settings take priority over sampler node's
        # remote_vae_endpoint. Normalize "None"/empty to "Default" — a
        # fresh Builder UI with no VAE selected should mean "use the
        # checkpoint's bundled VAE", not "load a model file named None".
        target_vae = normalize_config_vae(first_conf.get("vae", "Default"))
        if target_vae != "Default":
            config_overrides_vae = True
            if is_remote_vae(target_vae):
                url = extract_remote_vae_url(target_vae)
                current_vae_is_remote = True
                current_remote_vae_url = url
                print(f"[GridTester] 🌐 Using per-config remote VAE: {url}")
                if url not in per_config_remote_workers:
                    per_config_remote_workers[url] = RemoteVAEDecodeWorker(
                        endpoint=url, img_dir=paths["images"],
                        manifest_path=paths["manifest"],
                        existing_data=existing_data,
                        session_name=session_name, unique_id=unique_id,
                        image_format=image_format
                    )
            else:
                print(f"[GridTester] 🎨 Loading per-config VAE: {target_vae}")
                loaded_vae = load_vae_by_name(target_vae)
            cached_vae_key = target_vae
        else:
            cached_vae_key = "Default"
        
        if first_conf["lora_expanded"] != "None":
            patched_model, patched_clip = load_loras_for_preencoding(
                loaded_model, loaded_clip, first_conf["lora_expanded"]
            )
        else:
            patched_model, patched_clip = loaded_model, loaded_clip
        
        clip_hash = calculate_clip_hash(patched_clip)
        cond_cache = ConditioningCache(
            cache_dir=paths["base"], 
            clip_hash=clip_hash, 
            enable_disk_cache=save_conditioning_cache_to_file)
        
        # Filter configs to only collect prompts from jobs that actually need to run
        # This avoids wasting time encoding prompts for jobs that will be skipped in resume mode
        print(f"[GridTester] 🧠 Collecting unique prompts...")
        if not overwrite_existing and _have_history():
            configs_needing_work_ = configs_needing_work(job_plan, _job_status)
            skippable = len(expanded) - len(configs_needing_work_)
            if skippable > 0:
                print(f"[GridTester] ⏭️ {skippable}/{len(expanded)} configs already completed, encoding only {len(configs_needing_work_)} needed")

            if not configs_needing_work_:
                print(f"[GridTester] ⏭️ All configs already completed, skipping encoding entirely")
                unique_positives, unique_negatives = set(), set()
            else:
                unique_positives, unique_negatives = collect_unique_prompts_with_triggers(
                    configs_needing_work_, lora_triggerwords_mode
                )
        else:
            unique_positives, unique_negatives = collect_unique_prompts_with_triggers(
                expanded, lora_triggerwords_mode
            )

        # Stop with a clear message on malformed COUPLE regional prompts (instead of a later
        # "Encoding not found" failure)
        validate_regional_prompts(sorted(set(unique_positives) | set(unique_negatives)))

        # Collect adjusted HiRes prompts for pre-encoding (if hires prompt adjustment is enabled)
        upscale_settings = session_settings.get("upscaling", {}) if session_settings else {}
        if upscale_settings.get("enabled") and upscale_settings.get("hires_prompt_adjust") and upscale_settings.get("hires_prompt_text", "").strip():
            hires_behavior = upscale_settings.get("hires_prompt_behavior", "append_end")
            hires_text = upscale_settings["hires_prompt_text"].strip()
            hires_adjusted_positives = set()
            for base_prompt in unique_positives:
                if hires_behavior == "prepend":
                    adjusted = hires_text + " " + base_prompt
                elif hires_behavior == "append_end":
                    adjusted = append_to_base(base_prompt, hires_text)
                elif hires_behavior == "replace":
                    adjusted = hires_text
                else:
                    adjusted = base_prompt
                hires_adjusted_positives.add(adjusted)
            # Add adjusted prompts to unique set so they get batch-encoded
            unique_positives = unique_positives | hires_adjusted_positives
            print(f"[GridTester] 🔍 Added {len(hires_adjusted_positives)} HiRes-adjusted prompts for pre-encoding")

        clip_skip = first_conf.get("clip_skip", 0)
        if clip_skip != 0:
            print(f"[GridTester] 🔧 Using clip_skip={clip_skip}")

        # Text-encoder batching (several prompts per TE forward). Session-level Run Setting;
        # te_batch_verify is a hidden dev key (never emitted by the Builder) used by the live check.
        try:
            te_batch_size = max(1, int((session_settings or {}).get("te_batch_size", DEFAULT_TE_BATCH_SIZE)))
        except (TypeError, ValueError):
            te_batch_size = DEFAULT_TE_BATCH_SIZE
        te_batch_verify = bool((session_settings or {}).get("te_batch_verify", False))
        if te_batch_size > 1:
            print(f"[GridTester] ⚙️ te_batch_size={te_batch_size} (text-encoder batching)")

        try:
            conditioning_cache = batch_encode_prompts(
                patched_clip, unique_positives, unique_negatives, cond_cache, clip_skip, enable_disk_cache=save_conditioning_cache_to_file,
                te_batch_size=te_batch_size, te_batch_verify=te_batch_verify
            )
        except InterruptProcessingException:
            print(f"\n[GridTester] 🛑 INTERRUPTED during pre-encoding - Stopping all jobs")
            loaded_model, loaded_clip, loaded_vae = None, None, None
            patched_model, patched_clip = None, None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Re-raise so ComfyUI knows execution was interrupted (not completed)
            raise

        cached_model_key = get_model_cache_key(first_conf)
        cached_lora_key = first_conf["lora_expanded"]
        if model_cache:
            cached_lora_cache_key = model_cache._get_lora_cache_key(cached_model_key, first_conf["lora_expanded"])
        else:
            cached_lora_cache_key = None
        latent_channels = get_latent_channels(loaded_model, optional_latent)
        latent_ratio = get_latent_downscale_ratio(loaded_model)
    elif len(unique_model_keys) > 1:
        print(f"[GridTester] ⚠️ Multiple models detected ({len(unique_model_keys)} different models) - pre-encoding DISABLED")
        cond_cache = None
        latent_channels = 4
        latent_ratio = 8  # refreshed per model at the model-switch site below
    else:
        print(f"[GridTester] ℹ️ Using optional conditioning, skipping pre-encoding")
        cond_cache = None
        latent_channels = 4
        latent_ratio = 8
    
    # ==== FULL RUN SEED BEHAVIOR (PRE-RUN) ====
    # Apply "random_before" full run seed behavior: randomize seed before the entire session
    for conf_idx_pre, conf_pre in enumerate(expanded):
        if conf_pre.get("full_run_seed_behavior") == "random_before":
            import random
            conf_pre["seed"] = random.randint(0, 2**63 - 1)
            print(f"[GridTester] 🎲 Full run random_before: config {conf_idx_pre} seed → {conf_pre['seed']}")

    # ==== PRE-FLIGHT SCAN: count already-existing configs ====
    # Walk the same input_jobs × expanded iteration that the main loop uses,
    # applying per-config resolution overrides, and call check_if_job_completed
    # with the same arguments.  This gives us an "existing_count" we can show
    # in the running ETA line so the user knows how much real work is left.
    # Only useful when overwrite_existing is False (otherwise nothing is skipped).
    existing_count = 0
    _pf_ran = False
    if not overwrite_existing and not has_optional_inputs and _have_history():
        _pf_ran = True
        _pf_total = total_jobs
        print(f"[GridTester] 🔍 Pre-flight scan: checking which of {_pf_total} configs already exist...")
        _pf_checked = 0
        _pf_reasons = {}
        # Same job_plan the main loop walks (per-config resolutions already applied by build_job_plan)
        for _pf_pj in job_plan:
            _pf_st = _job_status(_pf_pj)
            if _pf_st != "todo":
                existing_count += 1
                _pf_reasons[_pf_st] = _pf_reasons.get(_pf_st, 0) + 1
            _pf_checked += 1
            if _pf_checked % 1000 == 0:
                print(f"[GridTester] 🔍  ...scanned {_pf_checked}/{_pf_total} ({existing_count} exist so far)")
        _pf_new = _pf_total - existing_count
        print(f"[GridTester] ✅ Pre-flight: {existing_count}/{_pf_total} already exist; {_pf_new} new to generate")
        if _pf_reasons.get("rejected_preview"):
            print(f"[GridTester] ⏭️ {_pf_reasons['rejected_preview']} cells skipped because their preview was rejected")
        if _pf_reasons.get("no_favorited_preview"):
            print(f"[GridTester] ⏭️ {_pf_reasons['no_favorited_preview']} cells skipped because their preview is not favorited")
        if _pf_reasons.get("full_exists"):
            print(f"[GridTester] ⏭️ {_pf_reasons['full_exists']} cells skipped in the preview pass because a full image already exists")

    # Send initial dashboard progress so the user sees "X/N already done"
    # even before the first new gen completes. Clears any stale values from a prior run.
    if PromptServer is not None and total_jobs > 0:
        try:
            _initial_pct = int((existing_count / total_jobs) * 100)
            PromptServer.instance.send_sync("ultimate_grid.progress", {
                "node": unique_id,
                "session_name": session_name,
                "current_job": existing_count,
                "total_jobs": total_jobs,
                "progress_pct": _initial_pct,
                "eta_str": "—",
                "finish_time": "",
                "avg_duration": 0,
                "last_duration": 0,
            })
        except Exception:
            pass

    # ==== MAIN GENERATION LOOP ====
    print(f"\n{'='*80}\n")

    _ltx_preflight_done = [False]  # Mutable cell so inner branch can set it

    _promo_skips = 0
    for plan_idx, pj in enumerate(job_plan):
        # The per-config resolution override (and its batch_idx reset) now lives in build_job_plan,
        # so w/h/batch_idx are fresh for every entry and can no longer leak into later configs.
        conf, conf_idx, job_idx, job = pj["conf"], pj["conf_idx"], pj["job_idx"], pj["job"]
        w, h, batch_idx = pj["w"], pj["h"], pj["batch_idx"]

        # ==== CHECK FOR INTERRUPT ====
        # ==== UPDATE CURRENT STEP ====
        if model_cache:
            model_cache.set_current_step(plan_idx)

        try:
            import comfy.model_management as mm
            if mm.processing_interrupted():
                print(f"\n[GridTester] 🛑 INTERRUPTED - Stopping all jobs")

                if pending_batch:
                    _flush_pending_batch(pending_batch, current_vae_is_remote, current_remote_vae_url,
                                         per_config_remote_workers, use_remote_vae, remote_vae_worker,
                                         loaded_vae, paths, existing_data, session_name, paths["manifest"], unique_id,
                                         config_overrides_vae=config_overrides_vae, image_format=image_format)
                    pending_batch = []

                _cleanup_per_config_remote_workers(per_config_remote_workers)
                if remote_vae_worker:
                    remote_vae_worker.wait_completion()
                    remote_vae_worker.stop()

                existing_data["meta"] = {
                    "positive": positive_text,
                    "negative": negative_text,
                    "model": ckpt_name,
                    "seed": seed,
                    "vae_batch_size": vae_batch_size,
                    "configs_json": configs_json,
                    "resolutions_json": resolutions_json,
                    "image_format": image_format,
                    "job_order": job_order,
                    "preview_pass": {"enabled": preview_on, "steps": pv_steps, "scale": pv_scale, "cfg_override": pv_cfg},
                    "rejected_preview_fps": list((existing_data.get("meta") or {}).get("rejected_preview_fps", [])),
                    **_meta_run
                }
                save_manifest(paths["manifest"], existing_data)

                loaded_model, loaded_clip, loaded_vae = None, None, None
                patched_model, patched_clip = None, None
                conditioning_cache.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Re-raise so ComfyUI knows execution was interrupted (not completed)
                raise InterruptProcessingException()
        except InterruptProcessingException:
            raise
        except:
            pass
            
        current_seed = conf["seed"]
        if conf.get("seed_behavior") == "randomize":
            import random
            current_seed = random.randint(0, 2**63 - 1)
        # Fingerprint of this cell (configured seed, FULL resolution); LTX items are not fingerprinted.
        job_fp = None
        if conf.get("model_type") != "ltx_video":
            job_fp = config_fingerprint(conf, w, h, conf["seed"])
            if not preview_on and not has_optional_inputs and conf.get("seed_behavior") == "randomize":
                _refresh_pstate()
                _pe = _pstate["pindex"].get(job_fp)
                if _pe and _pe.get("seed") is not None:
                    current_seed = _pe["seed"]  # re-render the survivor with the seed the user approved
        current_job += 1
            
        if pbar:
            try:
                pbar.update_absolute(current_job, total_jobs)
            except:
                pass

        # LTX Video dispatch — must run BEFORE any image-gen-specific code that
        # accesses conf['sampler']/conf['steps'] (LTX configs use sampler_stage1/2).
        if conf.get("model_type") == "ltx_video":
            from .ltx_video_generation import preflight_ltx, load_ltx_models, ltx_video_generate

            if has_regions(conf.get("positive", "")) or has_regions(conf.get("negative", "")):
                print("[GridTester] ⚠️ COUPLE regional prompts are not supported for LTX video; text is encoded literally")

            # Pre-flight on first LTX config of this run (idempotent — cached check internally)
            if not _ltx_preflight_done[0]:
                try:
                    preflight_ltx(conf)
                    _ltx_preflight_done[0] = True
                    # Free image-gen caches to make room for LTX VRAM
                    try:
                        from . import model_cache as mc
                        if hasattr(mc, "clear_image_caches"):
                            mc.clear_image_caches()
                    except Exception:
                        pass
                except RuntimeError as e:
                    print(f"[GridTester] LTX preflight FAILED: {e}")
                    raise

            # Override conf width/height with the current loop's resolution.
            # The loop's `w, h` comes from the sampler node's resolutions widget
            # (or a per-config override). LTX configs don't have width/height in
            # their builder UI yet, so the cartesian-expansion defaults (446x576)
            # would otherwise stick.
            conf["width"] = w
            conf["height"] = h

            # Filename uses gen index + timestamp suffix to guarantee uniqueness
            # per run. Skip-on-exists is intentionally NOT enabled for LTX (v1) —
            # the resume/match logic image-gen uses doesn't yet understand LTX
            # config fields (sigmas, strengths, audio_mode, etc.) so it produced
            # false-positive skips on real config changes (e.g., LoRA strength,
            # seed). Until we have a proper LTX-aware manifest match, every LTX
            # gen runs fresh. Users can manually delete unwanted mp4s.
            _ts_suffix = str(int(time.time() * 1000) % 10_000_000)
            ltx_output_filename = f"{conf_idx:06d}_seed{conf.get('seed', 0)}_{w}x{h}_dur{conf.get('duration_seconds', 5)}s_{conf.get('frame_rate', 25)}fps_{_ts_suffix}"
            ltx_output_path = os.path.join(paths["images"], ltx_output_filename + ".mp4")

            progress_pct = int((current_job / total_jobs) * 100)
            print(f"[GridTester] 📊 {current_job}/{total_jobs} ({progress_pct}%) | "
                  f"LTX {conf.get('duration_seconds', 5)}s @ {conf.get('frame_rate', 25)}fps | {conf.get('width')}x{conf.get('height')}")

            try:
                ltx_models = load_ltx_models(conf)
                gen_result = ltx_video_generate(conf, ltx_models, ltx_output_path)
            except Exception as e:
                print(f"[GridTester] LTX gen FAILED for config {conf_idx}: {e}")
                raise

            # Build manifest entry and persist
            item = _build_ltx_manifest_entry(conf, gen_result, ltx_output_filename, gen_index=conf_idx, session_name=session_name)
            existing_data["items"].insert(0, item)
            save_manifest(paths["manifest"], existing_data)

            # Skip the standard image-gen path
            continue

        progress_pct = int((current_job / total_jobs) * 100)
        _is_pv = preview_on  # LTX configs never reach this point (dispatched above)
        eff_conf = _effective_conf(conf) if _is_pv else conf
        if _is_pv:
            print(f"[GridTester] 📊 {current_job}/{total_jobs} ({progress_pct}%) | "
                  f"{conf['sampler']} @ PREVIEW {pv_steps} steps | {w}x{h} x{pv_scale}")
        else:
            print(f"[GridTester] 📊 {current_job}/{total_jobs} ({progress_pct}%) | "
                  f"{conf['sampler']} @ {conf['steps']} steps | {w}x{h}")
            
        actual_positive_prompt, lora_triggers = build_prompt_with_triggers(
            conf, lora_triggerwords_mode
        )
        actual_negative_prompt = conf["negative"]
            
        # ==== START AT JOB # (skip earlier jobs) ====
        if start_at_job > 0 and current_job < start_at_job:
            skipped_count += 1
            continue

        # ==== PREVIEW PROMOTION SKIP ====
        # Full run: skip cells whose preview was rejected (or not favorited). Preview run: skip cells
        # that already have a full image. A rejection is an explicit user signal, so overwrite_existing
        # does not override it; it still overrides the "full image exists" skip.
        _promo_why = _promotion_status(conf, job_fp)
        if _promo_why and not (_promo_why == "full_exists" and overwrite_existing):
            skipped_count += 1
            if _promo_why != "full_exists":
                _promo_skips += 1
                if _promo_skips % 50 == 0:
                    print(f"[GridTester] ⏭️ skipped {_promo_skips} jobs whose preview was rejected")
            continue

        # ==== CHECK EXISTING MATCH ====
        match_index = check_if_job_completed(
            existing_data["items"],
            eff_conf,
            current_seed,
            w, h,
            batch_idx,
            actual_positive_prompt,
            actual_negative_prompt,
            has_optional_inputs=has_optional_inputs,
            preview=_is_pv
        )

        if match_index != -1:
            if not overwrite_existing:
                skipped_count += 1
                continue
            else:
                old_item = existing_data["items"][match_index]
                try:
                    old_fname_match = re.search(r'filename=([^&]+)', old_item["file"])
                    if old_fname_match:
                        old_file_path = os.path.join(paths["images"], old_fname_match.group(1))
                        if os.path.exists(old_file_path):
                            os.remove(old_file_path)
                except Exception as e:
                    print(f"[GridTester] ⚠️ Warning: Could not delete old file: {e}")
                    
                existing_data["items"].pop(match_index)
            
        # ==== MODEL SWITCHING ====
        target_model_name = conf["model"]
        target_model_key = get_model_cache_key(conf)
        if target_model_key != cached_model_key:
            if cached_model_key is not None:
                patched_model, patched_clip = cleanup_model_references(
                    patched_model, patched_clip, conditioning_cache
                )
                # CRITICAL FIX: Restore dict structure after clear
                conditioning_cache["positive"] = {}
                conditioning_cache["negative"] = {}

            loaded_model, loaded_clip, loaded_vae = load_model_by_type(
                conf, ckpt_name, use_remote_vae,
                optional_model, optional_clip, optional_vae,
                optional_positive, optional_negative, loaded_clip, loaded_vae, model_cache=model_cache
            )
            # VAE fallback for non-checkpoint models
            if loaded_vae is None and optional_vae is not None:
                loaded_vae = optional_vae
            # Remember this model's default VAE for reverting later
            default_model_vae = loaded_vae
            cached_vae_key = "Default"
            current_vae_is_remote = False
            current_remote_vae_url = None
            config_overrides_vae = False

            cached_model_key = target_model_key
            cached_lora_key = None
            cached_lora_cache_key = None
            latent_channels = get_latent_channels(loaded_model, optional_latent)
            latent_ratio = get_latent_downscale_ratio(loaded_model)
            model_switched = True
        else:
            model_switched = False

        # A connected latent's pixel size = latent cells x this model's ratio, known only now.
        if job.get("latent_cells"):
            w, h = latent_job_pixel_size(job, latent_ratio)
            if job_fp is not None:
                job_fp = config_fingerprint(conf, w, h, conf["seed"])

        # ==== PER-CONFIG VAE SWITCHING ====
        # Config Builder VAE settings take priority over sampler node's remote_vae_endpoint
        target_vae = normalize_config_vae(conf.get("vae", "Default"))
        if target_vae != cached_vae_key:
            # Flush pending batch before switching VAE (they need current VAE for decoding)
            if pending_batch:
                _flush_pending_batch(pending_batch, current_vae_is_remote, current_remote_vae_url,
                                     per_config_remote_workers, use_remote_vae, remote_vae_worker,
                                     loaded_vae, paths, existing_data, session_name, paths["manifest"], unique_id,
                                     config_overrides_vae=config_overrides_vae, image_format=image_format)
                pending_batch = []

            if target_vae == "Default":
                # Revert to model's bundled/default VAE
                loaded_vae = default_model_vae
                current_vae_is_remote = False
                current_remote_vae_url = None
                config_overrides_vae = False
                print(f"[GridTester] 🎨 Reverting to Default VAE")
            elif is_remote_vae(target_vae):
                url = extract_remote_vae_url(target_vae)
                current_vae_is_remote = True
                current_remote_vae_url = url
                config_overrides_vae = True
                loaded_vae = None  # No local VAE needed
                print(f"[GridTester] 🌐 Using per-config remote VAE: {url}")
                if url not in per_config_remote_workers:
                    per_config_remote_workers[url] = RemoteVAEDecodeWorker(
                        endpoint=url, img_dir=paths["images"],
                        manifest_path=paths["manifest"],
                        existing_data=existing_data,
                        session_name=session_name, unique_id=unique_id,
                        image_format=image_format
                    )
            else:
                current_vae_is_remote = False
                current_remote_vae_url = None
                config_overrides_vae = True
                print(f"[GridTester] 🎨 Loading per-config VAE: {target_vae}")
                loaded_vae = load_vae_by_name(target_vae)
            cached_vae_key = target_vae

        # ==== LORA SWITCHING ====
        current_lora_string = conf["lora_expanded"]
            
        if model_cache:
            current_cache_key = model_cache._get_lora_cache_key(target_model_key, current_lora_string)
            need_to_load = (current_cache_key != cached_lora_cache_key) or patched_model is None
        else:
            need_to_load = (current_lora_string != cached_lora_key) or patched_model is None
            
        if need_to_load:
            patched_model, patched_clip, should_skip = load_loras(
                loaded_model, loaded_clip, current_lora_string,
                target_model_key, incompatible_loras, model_cache=model_cache
            )
                
            if should_skip:
                skipped_count += 1
                continue
                
            cached_lora_key = current_lora_string
            if model_cache:
                cached_lora_cache_key = current_cache_key
                
            # Batch encode logic for multi-model switching
            # Because of short-circuit, if model_switched is True, conditioning_cache['positive'] isn't checked initially
            # But inside the loop, we access it.
            if model_switched or not conditioning_cache["positive"]:
                model_unique_positives = []
                model_unique_negatives = []
                _seen_positives = set()
                _seen_negatives = set()

                for future_idx in range(plan_idx, len(job_plan)):
                    future_conf = job_plan[future_idx]["conf"]
                    if get_model_cache_key(future_conf) != target_model_key:
                        break  # the contiguous run of this model in the plan
                    if get_model_cache_key(future_conf) == target_model_key:
                        future_positive, _ = build_prompt_with_triggers(
                            future_conf, lora_triggerwords_mode
                        )
                        # Skip encoding prompts for configs that are already completed
                        # (avoids wasting time on prompts only used by skipped jobs)
                        if not overwrite_existing and _have_history():
                            # this plan entry (conf @ resolution) is already done or skipped
                            all_done = _job_status(job_plan[future_idx]) != "todo"
                            if all_done:
                                continue
                        if future_positive not in _seen_positives:
                            _seen_positives.add(future_positive)
                            model_unique_positives.append(future_positive)
                        future_neg = future_conf["negative"]
                        if future_neg not in _seen_negatives:
                            _seen_negatives.add(future_neg)
                            model_unique_negatives.append(future_neg)

                if model_unique_positives:
                    print(f"[GridTester] 🧠 Batch encoding {len(model_unique_positives)} prompts for {target_model_name}")

                    clip_skip = conf.get("clip_skip", 0)

                    try:
                        with pinned_clip(patched_clip, model_unique_positives + model_unique_negatives):
                            for prompt in model_unique_positives:
                                # Check for interrupt before each prompt encoding
                                if mm.processing_interrupted():
                                    print(f"\n[GridTester] 🛑 INTERRUPTED during positive encoding - Stopping all encoding")
                                    raise InterruptProcessingException()

                                if prompt not in conditioning_cache["positive"]:
                                    conditioning_cache["positive"][prompt] = encode_prompt_with_combinators(patched_clip, prompt, clip_skip)

                            for prompt in model_unique_negatives:
                                # Check for interrupt before each prompt encoding
                                if mm.processing_interrupted():
                                    print(f"\n[GridTester] 🛑 INTERRUPTED during negative encoding - Stopping all encoding")
                                    raise InterruptProcessingException()

                                if prompt not in conditioning_cache["negative"]:
                                    conditioning_cache["negative"][prompt] = encode_prompt_with_combinators(patched_clip, prompt, clip_skip)

                    except InterruptProcessingException:
                        print(f"\n[GridTester] 🛑 INTERRUPTED during encoding - Stopping all jobs")

                        if pending_batch:
                            _flush_pending_batch(pending_batch, current_vae_is_remote, current_remote_vae_url,
                                                 per_config_remote_workers, use_remote_vae, remote_vae_worker,
                                                 loaded_vae, paths, existing_data, session_name, paths["manifest"], unique_id,
                                                 config_overrides_vae=config_overrides_vae, image_format=image_format)
                            pending_batch = []

                        _cleanup_per_config_remote_workers(per_config_remote_workers)
                        if remote_vae_worker:
                            remote_vae_worker.wait_completion()
                            remote_vae_worker.stop()

                        existing_data["meta"] = {
                            "positive": positive_text,
                            "negative": negative_text,
                            "model": ckpt_name,
                            "seed": seed,
                            "vae_batch_size": vae_batch_size,
                            "configs_json": configs_json,
                            "resolutions_json": resolutions_json,
                            "image_format": image_format,
                            "job_order": job_order,
                            "preview_pass": {"enabled": preview_on, "steps": pv_steps, "scale": pv_scale, "cfg_override": pv_cfg},
                            "rejected_preview_fps": list((existing_data.get("meta") or {}).get("rejected_preview_fps", [])),
                            **_meta_run
                        }
                        save_manifest(paths["manifest"], existing_data)

                        loaded_model, loaded_clip, loaded_vae = None, None, None
                        patched_model, patched_clip = None, None
                        conditioning_cache.clear()
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        # Re-raise so ComfyUI knows execution was interrupted (not completed)
                        raise

                    print(f"[GridTester] ✅ Encoded {len(conditioning_cache['positive'])} positive, {len(conditioning_cache['negative'])} negative")

                model_switched = False
            if cond_cache:
                cond_cache.set_lora_config(conf['lora_expanded'])
            
        # Get conditioning
        if optional_positive:
            final_positive = optional_positive
        else:
            full_positive = actual_positive_prompt
            final_positive = conditioning_cache["positive"].get(full_positive)
            if final_positive is None:
                cached_keys = list(conditioning_cache["positive"].keys())
                cached_preview = [k[:40] for k in cached_keys[:5]]
                raise RuntimeError(
                    f"[GridTester] ❌ BUG: Encoding not found for positive prompt: {full_positive[:80]}...\n"
                    f"  model_key={target_model_key}, lora={current_lora_string}\n"
                    f"  Cache has {len(cached_keys)} entries: {cached_preview}\n"
                    f"  model_switched={model_switched}, need_to_load={need_to_load}"
                )

        if optional_negative:
            final_negative = optional_negative
        else:
            final_negative = conditioning_cache["negative"].get(conf["negative"])
            if final_negative is None:
                cached_keys = list(conditioning_cache["negative"].keys())
                cached_preview = [k[:40] for k in cached_keys[:5]]
                raise RuntimeError(
                    f"[GridTester] ❌ BUG: Encoding not found for negative prompt: {conf['negative'][:80]}...\n"
                    f"  model_key={target_model_key}, lora={current_lora_string}\n"
                    f"  Cache has {len(cached_keys)} entries: {cached_preview}\n"
                    f"  model_switched={model_switched}, need_to_load={need_to_load}"
                )
            
        # =========================================================
        # ==== 🚀 ASYNC LOOK-AHEAD PRE-FETCHING (CORRECTED) ====
        # =========================================================
        if model_cache is not None and model_cache.async_preload:
            # Check if next job exists (the plan carries it directly, no index arithmetic)
            if plan_idx + 1 < len(job_plan):
                next_idx = plan_idx + 1

                if next_idx < len(job_plan):
                    next_conf = job_plan[next_idx]["conf"]
                        
                    # RESOLVE cache keys for next conf
                    next_model_key = get_model_cache_key(next_conf)
                    next_lora = next_conf["lora_expanded"]

                    # CASE 1: Same Base, Different LoRA
                    if next_model_key == target_model_key and next_lora != current_lora_string:
                        def _preload_lora_worker():
                            return load_loras(
                                loaded_model, loaded_clip, next_lora,
                                target_model_name, {}, model_cache=None
                            )
                        model_cache.preload_lora_model(target_model_key, next_lora, _preload_lora_worker)

                    # CASE 2: Different Base Model
                    elif next_model_key != target_model_key:
                        next_model_type = next_conf.get("model_type", "checkpoint")
                        next_model_name = next_conf["model"]
                        print(f"[GridTester] 🔮 Pre-loading Base Model: {next_model_name} ({next_model_type})")

                        if next_model_type == "checkpoint":
                            next_model_resolved = ckpt_name if next_model_name == "Default" else next_model_name
                            def _preload_base_worker(path_to_load=next_model_resolved):
                                try:
                                    import comfy.sd
                                    ckpt_path = folder_paths.get_full_path("checkpoints", path_to_load)
                                    if use_remote_vae:
                                        out = comfy.sd.load_checkpoint_guess_config(
                                            ckpt_path, output_vae=False, output_clip=True,
                                            embedding_directory=folder_paths.get_folder_paths("embeddings")
                                        )
                                        return out[0], out[1], None
                                    else:
                                        out = comfy.sd.load_checkpoint_guess_config(
                                            ckpt_path, output_vae=True, output_clip=True,
                                            embedding_directory=folder_paths.get_folder_paths("embeddings")
                                        )
                                        return out[0], out[1], out[2]
                                except Exception as e:
                                    print(f"[GridTester] ❌ Async Worker Error: {e}")
                                    return None, None, None
                            model_cache.preload_base_model(next_model_key, _preload_base_worker)
                        else:
                            # GGUF/diffusion model preload
                            def _preload_diff_worker(conf_to_load=next_conf):
                                try:
                                    return load_diffusion_model_and_clip(
                                        model_name=conf_to_load["model"],
                                        model_type=conf_to_load.get("model_type"),
                                        text_encoder_paths=conf_to_load.get("text_encoders", []),
                                        clip_type_str=conf_to_load.get("clip_type", "stable_diffusion"),
                                        gguf_options=conf_to_load.get("gguf_options"),
                                        use_remote_vae=use_remote_vae,
                                    )
                                except Exception as e:
                                    print(f"[GridTester] ❌ Async Worker Error: {e}")
                                    return None, None, None
                            model_cache.preload_base_model(next_model_key, _preload_diff_worker)
        # =========================================================

        # Preview pass renders a snapped, scaled-down latent; full jobs use w x h unchanged.
        gen_w, gen_h = preview_dims(w, h, pv_scale, latent_ratio) if _is_pv else (w, h)

        # Generate image
        if job["latent"] is not None:
            latent_in = {"samples": job["latent"]["samples"].clone()}
        else:
            # Pixels-per-cell comes from the model's latent format (8 for SD/SDXL/Flux.1,
            # 16 for the Flux.2 family incl. MOSS-Image). Hardcoding 8 here made 16x
            # models render at 2x size and ~10x slower.
            latent_in = make_empty_latent(latent_channels, gen_w, gen_h, latent_ratio)
            
        result_latent = None
        attn_fallback = None
        try:
            attention_mode = conf.get("attention_mode", "default")
            if attention_mode != "default":
                print(f"[GridTester] 🧠 Using attention mode: {attention_mode}")

            # Log active extra options
            extra_opts = []
            if conf.get("model_sampling_override", "none") != "none":
                extra_opts.append(f"model_sampling={conf['model_sampling_override']}")
            if conf.get("use_advanced_sampling", False):
                extra_opts.append(f"advanced({conf.get('advanced_guider','cfg_guider')}/{conf.get('advanced_scheduler','basic')})")
            if conf.get("flux_guidance_value", 0) and float(conf.get("flux_guidance_value", 0)) > 0:
                extra_opts.append(f"flux_guidance={conf['flux_guidance_value']}")
            if extra_opts:
                print(f"[GridTester] ⚙️ Extra options: {', '.join(extra_opts)}")

            result_latent, duration, attn_fallback = generate_image_attn_safe(
                patched_model, current_seed, eff_conf["steps"], eff_conf["cfg"],
                conf["sampler"], conf["scheduler"], final_positive, final_negative,
                latent_in, conf["denoise"], attention_mode=attention_mode,
                model_sampling_override=conf.get("model_sampling_override", "none"),
                model_sampling_shift=conf.get("model_sampling_shift", 1.73),
                model_sampling_flux_max_shift=conf.get("model_sampling_flux_max_shift", 1.15),
                model_sampling_flux_base_shift=conf.get("model_sampling_flux_base_shift", 0.5),
                use_advanced_sampling=conf.get("use_advanced_sampling", False),
                advanced_guider=conf.get("advanced_guider", "cfg_guider"),
                advanced_scheduler=conf.get("advanced_scheduler", "basic"),
                flux_guidance_value=conf.get("flux_guidance_value", 0.0),
                use_deep_shrink=conf.get("use_deep_shrink", False),
                deep_shrink_block_number=conf.get("deep_shrink_block_number", 3),
                deep_shrink_downscale_factor=conf.get("deep_shrink_downscale_factor", 2.0),
                deep_shrink_start_percent=conf.get("deep_shrink_start_percent", 0.0),
                deep_shrink_end_percent=conf.get("deep_shrink_end_percent", 0.35),
                deep_shrink_downscale_after_skip=conf.get("deep_shrink_downscale_after_skip", True),
                deep_shrink_downscale_method=conf.get("deep_shrink_downscale_method", "bicubic"),
                deep_shrink_upscale_method=conf.get("deep_shrink_upscale_method", "bicubic"),
                use_slg=conf.get("use_slg", False),
                slg_double_layers=conf.get("slg_double_layers", ""),
                slg_single_layers=conf.get("slg_single_layers", "22"),
                slg_start_percent=conf.get("slg_start_percent", 0.0),
                slg_end_percent=conf.get("slg_end_percent", 0.5),
                width=gen_w,
                height=gen_h
            )

            # ==== UPSCALING (pipeline-based with sequential steps) ====
            upscale_produced = False
            save_pre_upscale = False
            total_upscale_duration = 0  # Track cumulative upscale time for duration metric
            if session_settings and session_settings.get("upscaling", {}).get("enabled", False) and result_latent is not None:
                from .image_generation import upscale_image
                import itertools as upscale_itertools
                import random as upscale_random
                from PIL import Image as PILImage

                upscale_settings_rt = session_settings["upscaling"]
                upscale_pipelines = upscale_settings_rt.get("pipelines", [])
                upscale_combo_idx = 0
                save_pre_upscale = upscale_settings_rt.get("save_pre_upscale", False)

                # If deferring upscales to end, skip inline processing and queue the job
                run_upscales_at_end = upscale_settings_rt.get("run_upscales_at_end", False)
                if run_upscales_at_end:
                    deferred_upscale_queue.append({
                        "config": conf.copy(),
                        "width": w, "height": h,
                        "seed": current_seed,
                        "batch_idx": batch_idx,
                        "actual_positive_prompt": actual_positive_prompt,
                        "actual_negative_prompt": actual_negative_prompt,
                        "gen_index": gen_index_offset + total_generated,
                        "config_fp": job_fp,
                    })
                    # Clear pipelines so the inline loop below becomes a no-op
                    upscale_pipelines = []

                # Compute HiRes-adjusted conditioning if enabled
                hires_positive_cond = final_positive  # Default: same as base
                hires_prompt_active = False
                hires_prompt_behavior_rt = ""
                hires_prompt_text_rt = ""
                if upscale_settings_rt.get("hires_prompt_adjust") and upscale_settings_rt.get("hires_prompt_text", "").strip():
                    hires_prompt_behavior_rt = upscale_settings_rt.get("hires_prompt_behavior", "append_end")
                    hires_prompt_text_rt = upscale_settings_rt["hires_prompt_text"].strip()
                    if hires_prompt_behavior_rt == "prepend":
                        adjusted_prompt = hires_prompt_text_rt + " " + actual_positive_prompt
                    elif hires_prompt_behavior_rt == "append_end":
                        adjusted_prompt = append_to_base(actual_positive_prompt, hires_prompt_text_rt)
                    elif hires_prompt_behavior_rt == "replace":
                        adjusted_prompt = hires_prompt_text_rt
                    else:
                        adjusted_prompt = actual_positive_prompt
                    # Look up pre-encoded conditioning from cache
                    hires_cond = conditioning_cache["positive"].get(adjusted_prompt)
                    if hires_cond is not None:
                        hires_positive_cond = hires_cond
                        hires_prompt_active = True
                        print(f"[GridTester] 🔍 Using HiRes-adjusted prompt ({hires_prompt_behavior_rt}): {adjusted_prompt[:80]}...")
                    else:
                        print(f"[GridTester] ⚠️ HiRes-adjusted prompt not found in cache, using original prompt")

                for pipeline_idx, pipeline in enumerate(upscale_pipelines):
                    # Skip inactive pipelines (safety check — should already be filtered)
                    if pipeline.get("active", True) is False:
                        continue

                    pipeline_name = pipeline.get("name", f"Pipeline {pipeline_idx + 1}")
                    pipeline_steps = pipeline.get("steps", [])
                    if not pipeline_steps:
                        continue

                    # Each pipeline starts from the original base image
                    pipe_latent = result_latent
                    pipe_w = w
                    pipe_h = h
                    pipeline_history = []  # Accumulate per-step metadata for compound manifest entries

                    # Flatten steps with repeat: a step with repeat=3 runs 3 times sequentially
                    expanded_steps = []
                    for step in pipeline_steps:
                        if step.get("active", True) is False:
                            continue
                        repeat = max(1, int(step.get("repeat", 1)))
                        for _ in range(repeat):
                            expanded_steps.append(step)

                    for step_idx, ucfg in enumerate(expanded_steps):
                        mode = ucfg.get("mode", "hires_only")
                        show_hires = mode in ("hires_only", "model_then_hires")
                        show_model = mode in ("model_only", "model_then_hires")

                        # --- Florence2 Hi-Res Fix mode (inline path) ---
                        if mode == "florence2_hires":
                            from .florence2_hires import (
                                run_florence2_step, preflight_florence2,
                                build_florence2_manifest_entry, build_florence2_no_detection_entry,
                            )
                            from PIL import Image as PILImage
                            import numpy as _np
                            import torch as _torch

                            # Preflight once per job (cached on the job object via closure proxy —
                            # use a module-global guard since `job` doesn't exist in this scope)
                            global _FLORENCE2_INLINE_PREFLIGHTED
                            if not globals().get("_FLORENCE2_INLINE_PREFLIGHTED", False):
                                preflight_florence2()
                                _FLORENCE2_INLINE_PREFLIGHTED = True

                            # Decode current latent to PIL for Florence2 input
                            src_pil = decode_latent_with_vae(loaded_vae, pipe_latent["samples"])
                            src_arr = _np.array(src_pil.convert("RGB")).astype(_np.float32) / 255.0
                            src_tensor = _torch.from_numpy(src_arr).unsqueeze(0)

                            # Build the unified step config — merge step.florence2 sub-object
                            # with top-level sampling fields from the step.
                            _hd = str(ucfg.get("hires_denoise", "0.45")).split(",")[0].strip() or "0.45"
                            f2_config = {
                                **(ucfg.get("florence2", {}) or {}),
                                "hires_denoise": float(_hd),
                                "hires_steps": int(ucfg.get("hires_steps", 15) or 15),
                                "cfg": float(ucfg.get("cfg", 1.5) or 1.5),
                                "sampler": ucfg.get("sampler", "euler"),
                                "scheduler": ucfg.get("scheduler", "simple"),
                                # In inline mode the model+lora are already loaded for this
                                # job — forcing from_builder avoids a redundant reload that
                                # from_manifest would trigger.
                                "model_source": "from_builder",
                            }

                            # Minimal item dict for run_florence2_step's manifest-extras path
                            item_for_florence2 = {
                                "model": conf.get("model", ""),
                                "lora_expanded": conf.get("loras_expanded", ""),
                                "seed": current_seed,
                            }

                            step_result = run_florence2_step(
                                source_image=src_tensor,
                                item=item_for_florence2,
                                step_config=f2_config,
                                fallback_handles=(patched_model, patched_clip, loaded_vae),
                                ckpt_cache={},  # no caching needed; from_builder uses fallback
                                conditioning_cache=conditioning_cache,
                                positive_prompt=actual_positive_prompt,
                                negative_prompt=actual_negative_prompt,
                                clip_skip=conf.get("clip_skip", 0),
                                # Inline path forces model_source=from_builder above, so
                                # this is informational; passing it for consistency.
                                session_model_name=conf.get("model", ""),
                            )

                            is_last_step = step_idx == len(expanded_steps) - 1
                            is_final_output = is_last_step

                            if step_result.get("status") == "no_detection":
                                if is_final_output:
                                    # Append sentinel entry pointing back at the base image.
                                    # `meta` is not yet assigned at this point in the function —
                                    # it lives at ~L2324, AFTER the upscale loop. Use
                                    # create_image_metadata directly (matches the normal
                                    # hires/model upscale path at L2184).
                                    sentinel_id = int(time.time() * 100000) + upscale_random.randint(0, 1000)
                                    sentinel_meta = create_image_metadata(
                                        conf, pipe_w, pipe_h,
                                        float(step_result.get("duration", 0)),
                                        current_seed, batch_idx,
                                        actual_positive_prompt, actual_negative_prompt,
                                        gen_index=gen_index_offset + total_generated,
                                    )
                                    sentinel_meta.update({
                                        "id": sentinel_id,
                                        "base_width": w, "base_height": h,
                                        "base_fp": job_fp,
                                        "config_fp": config_fingerprint(conf, w, h, conf["seed"], pipeline_name),
                                        "upscaled": False,
                                        "florence2_no_detection": True,
                                        "florence2_text_input": step_result.get("florence2_text_input", ""),
                                        "florence2_model": step_result.get("florence2_model", ""),
                                        "note": f"Florence2 found no '{step_result.get('florence2_text_input', '')}' in image",
                                    })
                                    # Insert at beginning + set upscale_produced (matches L2255-2256
                                    # canonical pattern). Without upscale_produced=True, the base
                                    # image at L2337 would ALSO get saved as a duplicate.
                                    existing_data["items"].insert(0, sentinel_meta)
                                    upscale_produced = True
                                    save_manifest(paths["manifest"], existing_data)
                                    upscale_combo_idx += 1
                                continue  # skip the normal combo loop

                            total_upscale_duration += float(step_result.get("duration", 0))

                            if is_final_output:
                                # Save the Florence2-fixed image and update manifest.
                                # `meta` is not yet assigned at this point in the function;
                                # build the entry via create_image_metadata directly (matches
                                # the normal hires/model upscale path at L2184).
                                upscale_id = int(time.time() * 100000) + upscale_random.randint(0, 1000)
                                upscaled_filename = f"img_{upscale_id}.webp"
                                filepath = os.path.join(paths["images"], upscaled_filename)
                                step_result["image_pil"].save(filepath, format="WEBP", quality=95)

                                upscaled_meta = create_image_metadata(
                                    conf, step_result["image_width"], step_result["image_height"],
                                    float(step_result.get("duration", 0)) + total_upscale_duration,
                                    current_seed, batch_idx,
                                    actual_positive_prompt, actual_negative_prompt,
                                    gen_index=gen_index_offset + total_generated,
                                )
                                upscaled_meta.update({
                                    "id": upscale_id,
                                    "base_fp": job_fp,
                                    "config_fp": config_fingerprint(conf, w, h, conf["seed"], pipeline_name),
                                    "file": f"/view?filename={upscaled_filename}&type=output&subfolder=benchmarks/{session_name}/images",
                                    "filename": upscaled_filename,
                                    "rejected": False,
                                    "upscaled": True,
                                    "base_width": w, "base_height": h,
                                    "upscale_source": "inline",
                                    "upscale_pipeline": pipeline_name,
                                    "upscale_mode": "florence2_hires",
                                    "florence2_model": step_result.get("florence2_model"),
                                    "florence2_text_input": step_result.get("florence2_text_input"),
                                    "florence2_target_megapixels": step_result.get("florence2_target_megapixels"),
                                    "florence2_crop_padding": step_result.get("florence2_crop_padding"),
                                    "florence2_grow_expand": step_result.get("florence2_grow_expand"),
                                    "florence2_feather": step_result.get("florence2_feather"),
                                    "florence2_output_mask_select": step_result.get("florence2_output_mask_select"),
                                    "florence2_detection_count": step_result.get("florence2_detection_count"),
                                    "florence2_bbox": step_result.get("florence2_bbox"),
                                    "hires_denoise": f2_config["hires_denoise"],
                                })
                                image_metrics.finalize(upscaled_meta, step_result["image_pil"])
                                # Insert at beginning + set upscale_produced (matches L2255-2256
                                # canonical pattern). Without upscale_produced=True, the base
                                # image at L2337 would ALSO get saved as a duplicate.
                                existing_data["items"].insert(0, upscaled_meta)
                                upscale_produced = True
                                save_manifest(paths["manifest"], existing_data)
                                if PromptServer is not None:
                                    try:
                                        PromptServer.instance.send_sync("ultimate_grid.update_data", {
                                            "session_name": session_name,
                                            "new_items": [upscaled_meta]
                                        })
                                    except Exception:
                                        pass
                                upscale_combo_idx += 1
                            else:
                                # Chained — re-encode result_pil for next step's pipe_latent
                                _next_arr = _np.array(step_result["image_pil"].convert("RGB")).astype(_np.float32) / 255.0
                                _next_tensor = _torch.from_numpy(_next_arr).unsqueeze(0)
                                pipe_latent = {"samples": loaded_vae.encode(_next_tensor[:, :, :, :3])}
                                pipe_w = step_result["image_width"]
                                pipe_h = step_result["image_height"]
                            continue  # Skip the normal combo loop for this step

                        # --- SeedVR2 upscale mode ---
                        if mode == "seedvr2":
                            from .image_generation import seedvr2_upscale
                            from PIL import Image as PILImage
                            sv_config = ucfg.get("seedvr2", {})
                            # Decode current latent to PIL for SeedVR2 input
                            # decode_latent_with_vae(vae, latent_samples) already returns a PIL image
                            pil_img = decode_latent_with_vae(loaded_vae, pipe_latent["samples"])
                            result_pil, up_w, up_h, sv_duration = seedvr2_upscale(pil_img, sv_config)
                            total_upscale_duration += sv_duration
                            is_last_step = step_idx == len(expanded_steps) - 1
                            is_final_output = is_last_step
                            if is_final_output:
                                # Save the upscaled image and update manifest
                                upscale_id = int(time.time() * 100000) + upscale_random.randint(0, 1000)
                                upscaled_filename = f"img_{upscale_id}.webp"
                                filepath = os.path.join(paths["images"], upscaled_filename)
                                result_pil.save(filepath, quality=80)
                                # `meta` is not yet assigned at this point in the function
                                # (same as the Florence2 branch): build via create_image_metadata.
                                upscaled_meta = create_image_metadata(
                                    conf, up_w, up_h, sv_duration + total_upscale_duration,
                                    current_seed, batch_idx,
                                    actual_positive_prompt, actual_negative_prompt,
                                    gen_index=gen_index_offset + total_generated,
                                )
                                upscaled_meta.update({
                                    "id": upscale_id,
                                    "base_fp": job_fp,
                                    "config_fp": config_fingerprint(conf, w, h, conf["seed"], pipeline_name),
                                    "file": f"/view?filename={upscaled_filename}&type=output&subfolder=benchmarks/{session_name}/images",
                                    "filename": upscaled_filename,
                                    "width": up_w, "height": up_h,
                                    "duration": sv_duration,
                                    "upscaled": True,
                                    "base_width": w, "base_height": h,
                                    "upscale_source": "inline",
                                    "upscale_pipeline": pipeline_name,
                                    "upscale_mode": "seedvr2",
                                    "upscale_model": sv_config.get("dit_model", ""),
                                })
                                image_metrics.finalize(upscaled_meta, result_pil)
                                # insert(0) + upscale_produced: newest-first manifest order, and
                                # without the flag the base image would ALSO be saved (duplicate)
                                existing_data["items"].insert(0, upscaled_meta)
                                upscale_produced = True
                                save_manifest(paths["manifest"], existing_data)
                                if PromptServer is not None:
                                    try:
                                        PromptServer.instance.send_sync("ultimate_grid.update_data", {
                                            "session_name": session_name,
                                            "new_items": [upscaled_meta]
                                        })
                                    except Exception:
                                        pass
                                upscale_combo_idx += 1
                            else:
                                # Re-encode for next step in chain (VAE.encode takes BHWC)
                                import torch as seedvr_torch
                                result_tensor = seedvr_torch.from_numpy(pil_to_image_array(result_pil))
                                pipe_latent = loaded_vae.encode(result_tensor[:, :, :, :3])
                                pipe_latent = {"samples": pipe_latent}
                                pipe_w_current = up_w
                                pipe_h_current = up_h
                            continue  # Skip the normal combo loop for this step

                        # Parse multi-value fields
                        raw_ratios = str(ucfg.get("upscale_ratios", "1.5"))
                        ratios = [float(r.strip()) for r in raw_ratios.split(",") if r.strip()] or [1.5]
                        raw_denoise = str(ucfg.get("hires_denoise", "0.3"))
                        denoises = [float(d.strip()) for d in raw_denoise.split(",") if d.strip()] or [0.3]
                        models = ucfg.get("upscale_models", []) or [""]

                        # Build Cartesian product of multi-value fields
                        if show_hires and show_model:
                            combos = list(upscale_itertools.product(models, ratios, denoises))
                        elif show_hires:
                            combos = list(upscale_itertools.product([""], ratios, denoises))
                        elif show_model:
                            combos = list(upscale_itertools.product(models, [1.0], [0.0]))
                        else:
                            combos = []

                        for combo in combos:
                            up_model_name, up_ratio, up_denoise = combo

                            # Skip combos requiring a model when none is selected
                            if show_model and not up_model_name:
                                upscale_combo_idx += 1
                                continue

                            # Build single-value upscaling config for upscale_image()
                            # For model_only mode, use upscale_ratio for upscale_size so the
                            # model's native output is preserved (not resized back to 1x)
                            effective_size = up_ratio if up_ratio > 1.0 else 2.0
                            single_config = {
                                "mode": mode,
                                "upscale_ratio": up_ratio,
                                "hires_denoise": up_denoise,
                                "hires_steps": ucfg.get("hires_steps", 0),
                                "tiled_vae": ucfg.get("tiled_vae", False),
                                "tile_size": ucfg.get("tile_size", 512),
                                "tile_overlap": ucfg.get("tile_overlap", 64),
                                "temporal_size": ucfg.get("temporal_size", 512),
                                "temporal_overlap": ucfg.get("temporal_overlap", 64),
                                "upscale_model": up_model_name,
                                "upscale_size": effective_size,
                                "resize_method": ucfg.get("resize_method", "bilinear"),
                                "hires_tiled_sampling": ucfg.get("hires_tiled_sampling", False),
                                "hires_tile_width": ucfg.get("hires_tile_width", 512),
                                "hires_tile_height": ucfg.get("hires_tile_height", 512),
                                "hires_mask_blur": ucfg.get("hires_mask_blur", 8),
                                "hires_tile_padding": ucfg.get("hires_tile_padding", 32),
                                "hires_force_uniform_tiles": ucfg.get("hires_force_uniform_tiles", False)
                            }

                            # Steps within a pipeline chain sequentially (each feeds into the next)
                            # Use HiRes-adjusted conditioning for modes that involve hires fix
                            up_positive = hires_positive_cond if (show_hires and hires_prompt_active) else final_positive
                            upscale_result, upscale_duration = upscale_image(
                                pipe_latent, loaded_vae, patched_model, single_config,
                                conf, up_positive, final_negative, pipe_w, pipe_h
                            )
                            total_upscale_duration += upscale_duration

                            # Determine if this is the final output (only the last step's last combo in the pipeline)
                            is_last_step = step_idx == len(expanded_steps) - 1
                            is_last_combo = combo == combos[-1]
                            is_final_output = is_last_step and is_last_combo

                            # Generate timestamp-based ID (same convention as normal images)
                            upscale_id = int(time.time() * 100000) + upscale_random.randint(0, 1000)
                            upscaled_filename = f"img_{upscale_id}_upscaled.webp"
                            up_metrics_sink = {}
                            up_metrics_img = None

                            if isinstance(upscale_result, dict) and "samples" in upscale_result:
                                # HiRes modes return latent — decode to get dimensions
                                upscaled_pil = decode_latent_with_vae(
                                    loaded_vae, upscale_result["samples"],
                                    metrics_sink=up_metrics_sink if is_final_output else None)
                                up_metrics_img = upscaled_pil
                                up_w, up_h = upscaled_pil.size
                                # Only save to disk if this is a final output
                                if is_final_output:
                                    upscaled_pil.save(
                                        os.path.join(paths["images"], upscaled_filename),
                                        "WEBP", quality=80
                                    )
                                # Feed this latent to the next step in the pipeline
                                pipe_latent = upscale_result
                                pipe_w = up_w
                                pipe_h = up_h
                            elif isinstance(upscale_result, PILImage.Image):
                                # Model-only returns PIL directly
                                up_metrics_img = upscale_result
                                up_w, up_h = upscale_result.size
                                # Only save to disk if this is a final output
                                if is_final_output:
                                    upscale_result.save(
                                        os.path.join(paths["images"], upscaled_filename),
                                        "WEBP", quality=80
                                    )
                                # Encode PIL back to latent for next step in pipeline
                                import numpy as np_stack
                                import torch as torch_stack
                                pil_np = np_stack.array(upscale_result).astype(np_stack.float32) / 255.0
                                pil_tensor = torch_stack.from_numpy(pil_np).unsqueeze(0)
                                pipe_latent = {"samples": loaded_vae.encode(pil_tensor[:, :, :, :3])}
                                pipe_w = up_w
                                pipe_h = up_h
                            else:
                                upscale_combo_idx += 1
                                continue

                            # Collect this step's upscale info for pipeline history
                            step_info = {
                                "mode": mode,
                                "ratio": up_ratio,
                                "denoise": up_denoise,
                                "model": up_model_name or "",
                                "resize_method": ucfg.get("resize_method", "bilinear"),
                                "hires_steps": ucfg.get("hires_steps", 0),
                                "tiled_vae": ucfg.get("tiled_vae", False),
                                "tile_size": ucfg.get("tile_size", 512),
                                "tiled_sampling": ucfg.get("hires_tiled_sampling", False),
                                "tile_w": ucfg.get("hires_tile_width", 512),
                                "tile_h": ucfg.get("hires_tile_height", 512),
                            }
                            if mode == "model_only":
                                step_info["upscale_size"] = ucfg.get("upscale_size", "2.0")
                            pipeline_history.append(step_info)

                            if is_final_output:
                                upscaled_meta = create_image_metadata(
                                    conf, up_w, up_h, duration + total_upscale_duration, current_seed, batch_idx,
                                    actual_positive_prompt, actual_negative_prompt,
                                    gen_index=gen_index_offset + total_generated
                                )
                                upscaled_meta["id"] = upscale_id
                                upscaled_meta["upscaled"] = True
                                upscaled_meta["base_width"], upscaled_meta["base_height"] = w, h
                                upscaled_meta["upscale_pipeline"] = pipeline_name
                                upscaled_meta["base_fp"] = job_fp
                                upscaled_meta["config_fp"] = config_fingerprint(conf, w, h, conf["seed"], pipeline_name)

                                # Include HiRes prompt adjustment info in manifest
                                if hires_prompt_active:
                                    upscaled_meta["hires_prompt_behavior"] = hires_prompt_behavior_rt
                                    upscaled_meta["hires_prompt_text"] = hires_prompt_text_rt

                                # Multiple steps in pipeline: save arrays of all steps
                                if len(pipeline_history) > 1:
                                    upscaled_meta["upscale_stacked"] = True
                                    upscaled_meta["upscale_mode"] = [s["mode"] for s in pipeline_history]
                                    upscaled_meta["upscale_ratio"] = [s["ratio"] for s in pipeline_history]
                                    upscaled_meta["upscale_denoise"] = [s["denoise"] for s in pipeline_history]
                                    models = [s["model"] for s in pipeline_history if s["model"]]
                                    if models:
                                        upscaled_meta["upscale_model"] = models
                                    upscaled_meta["upscale_resize_method"] = [s["resize_method"] for s in pipeline_history]
                                    upscaled_meta["upscale_hires_steps"] = [s["hires_steps"] for s in pipeline_history]
                                    # Tiling info: only include steps that have it enabled
                                    tiled_vae_steps = [s["tile_size"] for s in pipeline_history if s["tiled_vae"]]
                                    if tiled_vae_steps:
                                        upscaled_meta["upscale_tiled_vae"] = True
                                        upscaled_meta["upscale_tile_size"] = tiled_vae_steps
                                    tiled_sampling_steps = [f'{s["tile_w"]}x{s["tile_h"]}' for s in pipeline_history if s["tiled_sampling"]]
                                    if tiled_sampling_steps:
                                        upscaled_meta["upscale_tiled_sampling"] = True
                                        upscaled_meta["upscale_tile_w"] = [s["tile_w"] for s in pipeline_history if s["tiled_sampling"]]
                                        upscaled_meta["upscale_tile_h"] = [s["tile_h"] for s in pipeline_history if s["tiled_sampling"]]
                                else:
                                    # Single step pipeline
                                    upscaled_meta["upscale_mode"] = mode
                                    upscaled_meta["upscale_ratio"] = up_ratio
                                    upscaled_meta["upscale_denoise"] = up_denoise
                                    if up_model_name:
                                        upscaled_meta["upscale_model"] = up_model_name
                                    upscaled_meta["upscale_resize_method"] = ucfg.get("resize_method", "bilinear")
                                    upscaled_meta["upscale_hires_steps"] = ucfg.get("hires_steps", 0)
                                    if ucfg.get("tiled_vae", False):
                                        upscaled_meta["upscale_tiled_vae"] = True
                                        upscaled_meta["upscale_tile_size"] = ucfg.get("tile_size", 512)
                                    if ucfg.get("hires_tiled_sampling", False):
                                        upscaled_meta["upscale_tiled_sampling"] = True
                                        upscaled_meta["upscale_tile_w"] = ucfg.get("hires_tile_width", 512)
                                        upscaled_meta["upscale_tile_h"] = ucfg.get("hires_tile_height", 512)
                                    if mode == "model_only":
                                        upscaled_meta["upscale_size"] = ucfg.get("upscale_size", "2.0")

                                upscaled_meta["file"] = f"/view?filename={upscaled_filename}&type=output&subfolder=benchmarks/{session_name}/images"
                                upscaled_meta["rejected"] = False
                                image_metrics.finalize(upscaled_meta, up_metrics_img, up_metrics_sink)
                                # Insert at beginning (same order as flush_batch_with_vae)
                                existing_data["items"].insert(0, upscaled_meta)
                                upscale_produced = True

                                # Save manifest immediately so dashboard can pick it up
                                save_manifest(paths["manifest"], existing_data)

                                # Send live update to dashboard (same as flush_batch_with_vae)
                                if PromptServer is not None:
                                    try:
                                        manifest_meta = existing_data.get("meta", {})
                                        PromptServer.instance.send_sync("ultimate_grid.update", {
                                            "node": unique_id,
                                            "session_name": session_name,
                                            "new_items": [upscaled_meta],
                                            "meta": manifest_meta
                                        })
                                    except Exception:
                                        pass

                                print(f"[GridTester] 🔍 Saved upscaled image: {upscaled_filename} "
                                      f"(pipeline={pipeline_name}, mode={mode}, ratio={up_ratio}, denoise={up_denoise}"
                                      f"{', model=' + up_model_name if up_model_name else ''})")
                            else:
                                print(f"[GridTester] 🔍 Pipeline '{pipeline_name}' intermediate step {step_idx + 1}/{len(expanded_steps)} "
                                      f"(mode={mode}, ratio={up_ratio}, denoise={up_denoise}"
                                      f"{', model=' + up_model_name if up_model_name else ''})")

                            upscale_combo_idx += 1

                # Save manifest after all pipeline combos
                if upscale_combo_idx > 0:
                    save_manifest(paths["manifest"], existing_data)

            # Include upscale time in total duration for ETA accuracy
            total_duration = duration + total_upscale_duration
            new_completed += 1
            job_durations.append(total_duration)
            # With the pre-flight scan, jobs that will be skipped later must not count as remaining work
            eta_info = calculate_eta(job_durations, (existing_count + new_completed) if _pf_ran else current_job, total_jobs)
            if eta_info:
                print_generation_progress(current_job, total_jobs, conf, w, h, total_duration, eta_info)
                # Compact single-line ETA summary — uses effective progress (existing + newly completed)
                _effective_job = existing_count + new_completed
                _eta_pct = int((_effective_job / total_jobs) * 100)
                if eta_info['hours'] > 0:
                    _eta_str = f"{eta_info['hours']}h {eta_info['minutes']}m"
                elif eta_info['minutes'] > 0:
                    _eta_str = f"{eta_info['minutes']}m {eta_info['seconds']}s"
                else:
                    _eta_str = f"{eta_info['seconds']}s"
                _eta_line = (f"[GridTester] 📊 job {_effective_job}/{total_jobs} ({_eta_pct}%) | "
                             f"ETA: {_eta_str} | ~{eta_info['finish_formatted']} | "
                             f"{eta_info['avg_duration']:.1f}s/job")
                if existing_count > 0:
                    _eta_line += f" | {existing_count} existing"
                print(_eta_line)
                # Send progress to dashboard frontend
                if PromptServer is not None:
                    try:
                        progress_pct = int((_effective_job / total_jobs) * 100)
                        if eta_info['hours'] > 0:
                            eta_str = f"{eta_info['hours']}h {eta_info['minutes']}m"
                        elif eta_info['minutes'] > 0:
                            eta_str = f"{eta_info['minutes']}m {eta_info['seconds']}s"
                        else:
                            eta_str = f"{eta_info['seconds']}s"
                        PromptServer.instance.send_sync("ultimate_grid.progress", {
                            "node": unique_id,
                            "session_name": session_name,
                            "current_job": _effective_job,
                            "total_jobs": total_jobs,
                            "progress_pct": progress_pct,
                            "eta_str": eta_str,
                            "finish_time": eta_info['finish_formatted'],
                            "avg_duration": round(eta_info['avg_duration'], 1),
                            "last_duration": round(total_duration, 1)
                        })
                    except Exception:
                        pass

            # Skip saving the base (non-upscaled) image when upscaling produced results
            # — only the upscaled final version(s) should appear in the output
            # Unless save_pre_upscale is enabled, in which case save both
            if not upscale_produced or save_pre_upscale:
                meta = create_image_metadata(
                    eff_conf, gen_w, gen_h, total_duration, current_seed, batch_idx,
                    actual_positive_prompt, actual_negative_prompt,
                    gen_index=gen_index_offset + total_generated
                )
                meta["config_fp"] = job_fp
                if attn_fallback:
                    meta["attention_mode"] = "default"
                    meta["attention_fallback"] = attn_fallback
                if _is_pv:
                    meta.update({
                        "preview": True, "preview_of": job_fp,
                        "full_steps": conf["steps"], "full_cfg": conf["cfg"],
                        "full_width": w, "full_height": h, "preview_scale": pv_scale,
                    })
                if pos_hash or neg_hash:
                    meta["conditioning_pos_hash"] = pos_hash
                    meta["conditioning_neg_hash"] = neg_hash

                pending_batch.append((result_latent["samples"].clone(), meta))
            total_generated += 1

            # ==== GPU COOLDOWN (if enabled in session settings) ====
            if session_settings and session_settings.get("cooldown", {}).get("enabled", False):
                cooldown_config = session_settings["cooldown"]
                cooldown_every_n = int(cooldown_config.get("every_n", 1))
                if total_generated > 0 and total_generated % cooldown_every_n == 0:
                    cooldown_seconds = int(cooldown_config.get("seconds", 5))
                    clear_vram = cooldown_config.get("clear_vram", False)
                    print(f"[GridTester] ❄️ GPU Cooldown: pausing {cooldown_seconds}s after {total_generated} generations")
                    if clear_vram:
                        import comfy.model_management as mm_cooldown
                        mm_cooldown.soft_empty_cache()
                        mm_cooldown.unload_all_models()
                        print(f"[GridTester] ❄️ VRAM cleared")
                        cached_model_key = None  # Force model reload on next iteration
                    import time as time_cooldown
                    time_cooldown.sleep(cooldown_seconds)
                    print(f"[GridTester] ❄️ Cooldown complete, resuming generation")

        except Exception as e:
            import comfy.model_management
            if isinstance(e, InterruptProcessingException):
                print(f"\n[GridTester] 🛑 INTERRUPTED during generation - Stopping all jobs")
                if result_latent is not None:
                    del result_latent
                result_latent = None

                if pending_batch:
                    _flush_pending_batch(pending_batch, current_vae_is_remote, current_remote_vae_url,
                                         per_config_remote_workers, use_remote_vae, remote_vae_worker,
                                         loaded_vae, paths, existing_data, session_name, paths["manifest"], unique_id,
                                         config_overrides_vae=config_overrides_vae, image_format=image_format)
                    pending_batch = []

                _cleanup_per_config_remote_workers(per_config_remote_workers)
                if remote_vae_worker:
                    remote_vae_worker.wait_completion()
                    remote_vae_worker.stop()

                existing_data["meta"] = {
                    "positive": positive_text,
                    "negative": negative_text,
                    "model": ckpt_name,
                    "seed": seed,
                    "vae_batch_size": vae_batch_size,
                    "configs_json": configs_json,
                    "resolutions_json": resolutions_json,
                    "image_format": image_format,
                    "job_order": job_order,
                    "preview_pass": {"enabled": preview_on, "steps": pv_steps, "scale": pv_scale, "cfg_override": pv_cfg},
                    "rejected_preview_fps": list((existing_data.get("meta") or {}).get("rejected_preview_fps", [])),
                    **_meta_run
                }
                save_manifest(paths["manifest"], existing_data)

                loaded_model, loaded_clip, loaded_vae = None, None, None
                patched_model, patched_clip = None, None
                conditioning_cache.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Re-raise so ComfyUI knows execution was interrupted (not completed)
                raise
            else:
                print(f"[GridTester] ❌ Generation failed: {e}")
                if result_latent is not None:
                    del result_latent
                continue
            
        if result_latent is not None:
            del result_latent
        result_latent = None
        del latent_in
        latent_in = None
            
        if current_job % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        threshold = vae_batch_size if flush_batch_every <= 0 else flush_batch_every
        if len(pending_batch) >= threshold:
            _flush_pending_batch(pending_batch, current_vae_is_remote, current_remote_vae_url,
                                 per_config_remote_workers, use_remote_vae, remote_vae_worker,
                                 loaded_vae, paths, existing_data, session_name, paths["manifest"], unique_id,
                                 config_overrides_vae=config_overrides_vae, image_format=image_format)
            pending_batch = []

    # ==== FINALIZATION ====
    if pending_batch:
        _flush_pending_batch(pending_batch, current_vae_is_remote, current_remote_vae_url,
                             per_config_remote_workers, use_remote_vae, remote_vae_worker,
                             loaded_vae, paths, existing_data, session_name, paths["manifest"], unique_id,
                             config_overrides_vae=config_overrides_vae, image_format=image_format)

    # ==== DEFERRED UPSCALE PHASE (must run before cleanup so VAE/model/CLIP are still loaded) ====
    if deferred_upscale_queue and session_settings and session_settings.get("upscaling", {}).get("run_upscales_at_end", False):
        upscale_settings_deferred = session_settings["upscaling"]
        run_deferred_upscales(
            deferred_upscale_queue, upscale_settings_deferred, session_settings,
            loaded_vae, patched_model, patched_clip, conditioning_cache,
            paths, existing_data, session_name, unique_id,
            PromptServer, config_overrides_vae=config_overrides_vae
        )

    # Shut down per-config remote VAE workers
    _cleanup_per_config_remote_workers(per_config_remote_workers)

    if remote_vae_worker:
        print(f"[GridTester] 🌐 Waiting for remote VAE...")
        remote_vae_worker.wait_completion()
        remote_vae_worker.stop()
    
    print_incompatible_loras_summary(incompatible_loras)
    
    if skipped_count > 0:
        print(f"[GridTester] ⏭️ Skipped {skipped_count} configs")

    # ==== FULL RUN SEED BEHAVIOR (POST-RUN) ====
    # These modify the seed for the NEXT queue/run, not the current one.
    # The new seeds are stored in the manifest meta so the user can see/use them.
    next_run_seeds = {}
    for conf_idx_post, conf_post in enumerate(expanded):
        frb = conf_post.get("full_run_seed_behavior", "fixed")
        if frb == "random_after":
            import random
            new_seed = random.randint(0, 2**63 - 1)
            print(f"[GridTester] 🎲 Full run random_after: config {conf_idx_post} next seed → {new_seed}")
            conf_post["seed"] = new_seed
            next_run_seeds[str(conf_idx_post)] = new_seed
        elif frb == "increment_after":
            conf_post["seed"] = conf_post["seed"] + 1
            print(f"[GridTester] ➕ Full run increment_after: config {conf_idx_post} next seed → {conf_post['seed']}")
            next_run_seeds[str(conf_idx_post)] = conf_post["seed"]
        elif frb == "decrement_after":
            conf_post["seed"] = conf_post["seed"] - 1
            print(f"[GridTester] ➖ Full run decrement_after: config {conf_idx_post} next seed → {conf_post['seed']}")
            next_run_seeds[str(conf_idx_post)] = conf_post["seed"]
    # Persist next-run seeds to the node's seed widget via PromptServer
    if next_run_seeds and PromptServer is not None:
        try:
            PromptServer.instance.send_sync("ultimate_grid.next_run_seeds", {
                "node": unique_id,
                "seeds": next_run_seeds
            })
        except Exception:
            pass

    existing_data["meta"] = {
        "positive": positive_text,
        "negative": negative_text,
        "model": ckpt_name,
        "seed": seed,
        "vae_batch_size": vae_batch_size,
        "configs_json": configs_json,
        "resolutions_json": resolutions_json,
        "image_format": image_format,
        "job_order": job_order,
        "preview_pass": {"enabled": preview_on, "steps": pv_steps, "scale": pv_scale, "cfg_override": pv_cfg},
        "rejected_preview_fps": list((existing_data.get("meta") or {}).get("rejected_preview_fps", [])),
        **_meta_run
    }
    save_manifest(paths["manifest"], existing_data)

    print(f"[GridTester] 🧹 Cleaning up...")
    loaded_model, loaded_clip, loaded_vae = None, None, None
    patched_model, patched_clip = None, None
    conditioning_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
     
    html = get_html_template(session_name, existing_data, unique_id)
      
    if job_durations:
        total_elapsed = time.time() - eta_start_time
        total_hours = int(total_elapsed // 3600)
        total_minutes = int((total_elapsed % 3600) // 60)
        total_seconds = int(total_elapsed % 60)
        avg_per_job = sum(job_durations) / len(job_durations)
        
        print(f"\n{'='*80}")
        print(f"[GridTester] 🎉 COMPLETE!")
        print(f"[GridTester] ✅ {total_generated} images generated")
        print(f"[GridTester] ⏱️  {total_hours}h {total_minutes}m {total_seconds}s total")
        print(f"[GridTester] 📊 {avg_per_job:.1f}s average per job")
        print(f"{'='*80}\n")

        # Send completion event to dashboard
        if PromptServer is not None:
            try:
                PromptServer.instance.send_sync("ultimate_grid.progress", {
                    "node": unique_id,
                    "session_name": session_name,
                    "current_job": total_jobs,
                    "total_jobs": total_jobs,
                    "progress_pct": 100,
                    "eta_str": "Done",
                    "finish_time": time.strftime("%H:%M:%S"),
                    "avg_duration": round(avg_per_job, 1),
                    "last_duration": 0,
                    "complete": True,
                    "total_elapsed": f"{total_hours}h {total_minutes}m {total_seconds}s",
                    "total_generated": total_generated
                })
            except Exception:
                pass
    
    return (html,)


# === Conditioning Serialization Helpers ===
# These convert ComfyUI conditioning tensors to/from JSON-serializable dicts
# for transmitting pre-encoded conditionings from master to workers.
# Pattern mirrors ConditioningCache._conditioning_to_dict/_dict_to_conditioning
# but handles multi-entry conditionings (AND combinator creates multiple entries).

def _conditioning_to_serializable(conditioning):
    """
    Convert a ComfyUI conditioning to a JSON-serializable list of dicts.

    Args:
        conditioning: ComfyUI format [[cond_tensor, {"pooled_output": pooled_tensor}], ...]
                      May have multiple entries when AND combinator is used.

    Returns:
        list of dicts, each with base64-encoded tensor data:
        [{"cond": {"data": ..., "shape": ..., "dtype": ...}, "pooled": {...}}, ...]
    """
    import base64
    import numpy as np

    entries = []
    for cond_tensor, extra in conditioning:
        cond_np = cond_tensor.cpu().numpy()
        entry = {
            "cond": {
                "data": base64.b64encode(cond_np.tobytes()).decode('utf-8'),
                "shape": list(cond_np.shape),
                "dtype": str(cond_np.dtype)
            }
        }

        # Serialize ALL keys from the extra dict, not just pooled_output.
        # Models like Wan/T5 rely on attention_mask and other keys for proper sampling.
        if isinstance(extra, dict):
            extra_serialized = {}
            for key, val in extra.items():
                if val is None:
                    extra_serialized[key] = {"type": "none"}
                elif isinstance(val, torch.Tensor):
                    val_np = val.cpu().numpy()
                    extra_serialized[key] = {
                        "type": "tensor",
                        "data": base64.b64encode(val_np.tobytes()).decode('utf-8'),
                        "shape": list(val_np.shape),
                        "dtype": str(val_np.dtype)
                    }
                else:
                    # Scalar or other JSON-serializable value
                    extra_serialized[key] = {"type": "value", "data": val}
            entry["extra"] = extra_serialized

        entries.append(entry)

    return entries


def _serializable_to_conditioning(entries):
    """
    Convert a serialized list of dicts back to ComfyUI conditioning format.

    Args:
        entries: list of dicts with base64-encoded tensor data (from _conditioning_to_serializable)

    Returns:
        ComfyUI conditioning format [[cond_tensor, {"pooled_output": pooled_tensor}], ...]
    """
    import base64
    import numpy as np

    conditioning = []
    for entry in entries:
        # Reconstruct main conditioning tensor
        cond_data = entry["cond"]
        cond_bytes = base64.b64decode(cond_data["data"])
        cond_np = np.frombuffer(cond_bytes, dtype=np.dtype(cond_data["dtype"])).reshape(
            tuple(cond_data["shape"])
        )
        # .copy() is required because np.frombuffer returns a read-only array,
        # and torch.from_numpy cannot send read-only arrays to GPU
        cond_tensor = torch.from_numpy(cond_np.copy())

        # Reconstruct ALL extra dict keys (attention_mask, pooled_output, etc.)
        extra_dict = {}
        if "extra" in entry:
            for key, val_info in entry["extra"].items():
                if val_info.get("type") == "none":
                    extra_dict[key] = None
                elif val_info.get("type") == "tensor":
                    val_bytes = base64.b64decode(val_info["data"])
                    val_np = np.frombuffer(val_bytes, dtype=np.dtype(val_info["dtype"])).reshape(
                        tuple(val_info["shape"])
                    )
                    extra_dict[key] = torch.from_numpy(val_np.copy())
                elif val_info.get("type") == "value":
                    extra_dict[key] = val_info["data"]
        elif "pooled" in entry:
            # Backwards compatibility with old serialization format
            pooled_data = entry["pooled"]
            pooled_bytes = base64.b64decode(pooled_data["data"])
            pooled_np = np.frombuffer(pooled_bytes, dtype=np.dtype(pooled_data["dtype"])).reshape(
                tuple(pooled_data["shape"])
            )
            extra_dict["pooled_output"] = torch.from_numpy(pooled_np.copy())

        conditioning.append([cond_tensor, extra_dict])

    return conditioning


def _preencode_all_conditionings(
    expanded, lora_triggerwords_mode, ckpt_name, use_remote_vae,
    optional_model, optional_clip, optional_vae,
    optional_positive, optional_negative, model_cache
):
    """
    Pre-encode all unique prompts across all (model, LoRA) combinations.

    Groups expanded configs by (model_key, lora_key), loads model+LoRA for each group,
    encodes all unique prompts, serializes the conditionings using base64 encoding,
    then cleans up.

    Returns:
        dict: {
            "model_key|lora_key": {
                "positive": {prompt_text: serialized_entries_list, ...},
                "negative": {prompt_text: serialized_entries_list, ...}
            },
            ...
        }
        Empty dict if encoding fails or no configs to encode.
    """
    from collections import defaultdict
    from .model_loader import load_loras_for_preencoding

    # Group configs by (model_key, lora_key)
    groups = defaultdict(list)
    for conf in expanded:
        model_key = get_model_cache_key(conf)
        lora_key = conf.get("lora_expanded", conf.get("lora", "None"))
        group_key = f"{model_key}|{lora_key}"
        groups[group_key].append(conf)

    print(f"[Distribution] 🧠 Found {len(groups)} unique (model, LoRA) group(s) to pre-encode")

    encoded_all = {}

    for group_key, configs in groups.items():
        parts = group_key.split("|", 1)
        lora_key = parts[1] if len(parts) > 1 else "None"
        first_conf = configs[0]

        print(f"[Distribution] 🧠 Pre-encoding group: model={first_conf['model']}, lora={lora_key}")

        # Collect unique prompts for this group
        unique_positives = set()
        unique_negatives = set()
        for conf in configs:
            try:
                actual_positive, _ = build_prompt_with_triggers(conf, lora_triggerwords_mode)
            except Exception:
                actual_positive = conf.get("positive", "")
            actual_negative = conf.get("negative", "")
            unique_positives.add(actual_positive)
            unique_negatives.add(actual_negative)

        validate_regional_prompts(sorted(unique_positives | unique_negatives))

        print(f"[Distribution] 🧠   {len(unique_positives)} unique positive, "
              f"{len(unique_negatives)} unique negative prompt(s)")

        try:
            # Load model + CLIP for this group
            loaded_model, loaded_clip, loaded_vae = load_model_by_type(
                first_conf, ckpt_name, use_remote_vae,
                optional_model, optional_clip, optional_vae,
                optional_positive, optional_negative, None, None,
                model_cache=model_cache
            )

            # Apply LoRA to get patched CLIP
            if lora_key != "None":
                patched_model, patched_clip = load_loras_for_preencoding(
                    loaded_model, loaded_clip, lora_key
                )
            else:
                patched_model, patched_clip = loaded_model, loaded_clip

            clip_skip = first_conf.get("clip_skip", 0)

            serialized_positive = {}
            serialized_negative = {}

            # One encoder load for the whole group (pinned_clip); ComfyUI would otherwise
            # re-prepare the text encoder on every encode.
            with pinned_clip(patched_clip, list(unique_positives) + list(unique_negatives)):
                for prompt in unique_positives:
                    cond = encode_prompt_with_combinators(patched_clip, prompt, clip_skip)
                    serialized_positive[prompt] = _conditioning_to_serializable(cond)

                for prompt in unique_negatives:
                    cond = encode_prompt_with_combinators(patched_clip, prompt, clip_skip)
                    serialized_negative[prompt] = _conditioning_to_serializable(cond)

            encoded_all[group_key] = {
                "positive": serialized_positive,
                "negative": serialized_negative
            }

            print(f"[Distribution] 🧠   Encoded {len(serialized_positive)} positive, "
                  f"{len(serialized_negative)} negative conditioning(s)")

            # Cleanup patched references
            del patched_model, patched_clip

        except Exception as e:
            print(f"[Distribution] ⚠️ Pre-encoding failed for group {group_key}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return encoded_all


def _send_distribution_status(manager):
    """Send distribution status update to the dashboard."""
    try:
        status = manager.get_status()
        status["active"] = manager.is_active
        import server as _server
        _server.PromptServer.instance.send_sync("ultimate_grid.distribution_status", {
            "node": manager.unique_id,
            "session_name": manager.session_name,
            **status
        })
    except Exception:
        pass


def _distributed_plan_entries(job_plan, existing_data, *, overwrite_existing, has_optional_inputs,
                              lora_triggerwords_mode, preview=None, skip_rejected=True, only_favorited=False):
    """Turn the (already ordered) job plan into the entries the distributed manager queues.

    The master decides everything stateful here, with the same rules and order as the normal loop
    (existing-image skip, preview promotion skip, survivor seed, fingerprints), so workers stay stateless
    renderers and a preview/full pass behaves the same with or without workers.

    ``preview`` is None (full pass) or ``{"steps", "scale", "cfg"}`` (cfg may be None = keep).
    Returns ``(entries, counts)``; an entry is ``{"config", "input_job", "extra"}`` where ``extra`` holds
    ``stamp`` (manifest fields the master adds on collection), optional ``render`` (``preview_scale``) and
    optional ``seed`` (survivor seed to re-render). ``counts`` = {"exists", "promotion", "full_exists"}.
    """
    from .config_utils import (
        config_fingerprint, build_preview_index, build_full_fp_set, should_skip_for_previews
    )
    items = existing_data.get("items", [])
    meta = existing_data.get("meta") or {}
    pindex = build_preview_index(items, meta.get("rejected_preview_fps", []))
    full_fps = build_full_fp_set(items) if preview else set()
    entries = []
    counts = {"exists": 0, "promotion": 0, "full_exists": 0}

    for pj in job_plan:
        conf, w, h, batch_idx = pj["conf"], pj["w"], pj["h"], pj["batch_idx"]
        is_ltx = conf.get("model_type") == "ltx_video"
        is_pv = bool(preview) and not is_ltx
        fp = None if is_ltx else config_fingerprint(conf, w, h, conf["seed"])
        randomize = conf.get("seed_behavior") == "randomize"

        survivor_seed = None
        if not is_pv and not has_optional_inputs and fp is not None and randomize:
            pe = pindex.get(fp)
            if pe and pe.get("seed") is not None:
                survivor_seed = pe["seed"]  # re-render the survivor with the seed the user approved

        # Preview promotion skip. A rejection is an explicit user signal, so overwrite_existing does not
        # override it; it still overrides the "full image exists" skip.
        why = None
        if not (has_optional_inputs or is_ltx or fp is None):
            if is_pv:
                why = "full_exists" if fp in full_fps else None
            else:
                why = should_skip_for_previews(fp, pindex, skip_rejected, only_favorited)
        if why and not (why == "full_exists" and overwrite_existing):
            counts["full_exists" if why == "full_exists" else "promotion"] += 1
            continue

        eff_conf = conf
        if is_pv:
            eff_conf = {**conf, "steps": preview["steps"],
                        "cfg": conf["cfg"] if preview.get("cfg") is None else preview["cfg"]}

        # A randomized seed never matches an old item (the normal loop rolls a fresh one before the check),
        # except a survivor's approved seed.
        if not overwrite_existing and (not randomize or survivor_seed is not None):
            pos_, _ = build_prompt_with_triggers(conf, lora_triggerwords_mode)
            if check_if_job_completed(
                items, eff_conf, survivor_seed if survivor_seed is not None else conf["seed"],
                w, h, batch_idx, pos_, conf["negative"],
                has_optional_inputs=has_optional_inputs, preview=is_pv
            ) != -1:
                counts["exists"] += 1
                continue

        extra = {}
        if fp is not None:
            stamp = {"config_fp": fp}
            if is_pv:
                stamp.update({
                    "preview": True, "preview_of": fp,
                    "full_steps": conf["steps"], "full_cfg": conf["cfg"],
                    "full_width": w, "full_height": h, "preview_scale": preview["scale"],
                })
            extra["stamp"] = stamp
        if is_pv:
            extra["render"] = {"preview_scale": preview["scale"]}
        if survivor_seed is not None:
            extra["seed"] = survivor_seed
        entries.append({
            "config": eff_conf,
            "input_job": {"label": pj["job"].get("label", f"{w}x{h}"), "width": w, "height": h,
                          "batch_idx": batch_idx},
            "extra": extra,
        })
    return entries, counts


def _distributed_metrics_pass(existing_data, pre_ids, images_dir, image_path_for=None):
    """Metrics for images that workers rendered. Run once on the master after collection.

    Workers measure the builtins on the decoded GPU tensor (same numbers the master gets for its own images,
    when their companion/main code is new enough); adapters (aesthetic scorer ...) only ever run here, so
    workers do not need the scorer pack installed. An item with no metrics at all (older worker) gets
    builtins from the saved file and its adapters. Master-rendered items already have everything and are
    left alone. Returns ``{"adapters": n, "builtins_from_file": m}``; no-op without an active metrics run.
    """
    from . import image_metrics
    spec = image_metrics.current_spec()
    out = {"adapters": 0, "builtins_from_file": 0}
    if not spec:
        return out
    if image_path_for is None:
        from .metrics_runner import _image_path_for as image_path_for
    from PIL import Image
    adapter_names = list(spec.get("adapters") or [])
    for item in existing_data.get("items", []):
        if item.get("id") in pre_ids or item.get("upscaled") or item.get("media_type") == "video":
            continue
        have = item.get("metrics") or {}
        need_builtins = bool(spec.get("builtins")) and not have
        need_adapters = any(a not in have for a in adapter_names)
        if not (need_builtins or need_adapters):
            continue
        path = image_path_for(item, images_dir)
        if path is None:
            continue
        try:
            with Image.open(path) as im:
                pil = im.convert("RGB")
        except Exception as e:
            print(f"[Distribution] ⚠️ metrics skipped for item {item.get('id')}: {e}")
            continue
        if need_builtins:
            image_metrics.attach(item, pil)
            out["builtins_from_file"] += 1
        else:
            image_metrics.attach_adapters(item, pil)
        if need_adapters:
            out["adapters"] += 1
    return out


def _populate_distributed_manager(manager, job_plan, expanded, input_jobs, existing_data, overwrite_existing,
                                  has_optional_inputs, lora_triggerwords_mode, *, job_order="as_written",
                                  preview_on=False, pv_steps=8, pv_scale=0.5, pv_cfg=None,
                                  skip_rejected=True, only_favorited=False):
    """Queue the run on the distributed manager: the master's own plan when the companion supports it
    (populate_from_plan), else the companion's legacy nested loop (job order / preview ignored)."""
    if job_plan is not None and hasattr(manager, "populate_from_plan"):
        # The master decides order, skips, preview promotion and fingerprints; workers just render.
        _pv = {"steps": pv_steps, "scale": pv_scale, "cfg": pv_cfg} if preview_on else None
        entries, _counts = _distributed_plan_entries(
            job_plan, existing_data, overwrite_existing=overwrite_existing,
            has_optional_inputs=has_optional_inputs, lora_triggerwords_mode=lora_triggerwords_mode,
            preview=_pv, skip_rejected=skip_rejected, only_favorited=only_favorited
        )
        print(f"[Distribution] 📋 Plan: {len(entries)} job(s) to render | skipped: {_counts['exists']} already exist, "
              f"{_counts['promotion']} rejected/unfavorited previews, {_counts['full_exists']} already full")
        if job_order != "as_written":
            print(f"[Distribution] 🧭 Job order {job_order}: jobs are claimed in the master's plan order")
        if preview_on:
            print(f"[GridTester] 🔎 Distributed preview pass: {pv_steps} steps, x{pv_scale} scale")
        _spec = image_metrics.current_spec()
        manager.populate_from_plan(
            entries, run_extras={"metrics_builtins": list(_spec.get("builtins") or [])} if _spec else {})
    else:
        if (job_order != "as_written" or preview_on
                or any(it.get("preview") is True for it in existing_data.get("items", []))):
            print(f"[GridTester] ⚠️ job_order / preview pass / preview promotion need an updated ComfyUI-USCG-Distributed "
                  f"companion - ignored in distributed mode")
        manager.populate_jobs(
            expanded, input_jobs, existing_data,
            overwrite_existing, has_optional_inputs, lora_triggerwords_mode
        )



def _distributed_run_meta(existing_data, positive_text, negative_text, ckpt_name, seed, vae_batch_size,
                          configs_json, resolutions_json, image_format, job_order, preview_on, pv_steps,
                          pv_scale, pv_cfg, meta_run):
    """Manifest meta for a distributed run: the same keys the normal loop writes, so the preview
    tombstones (rejected_preview_fps), job order and the run-API session_settings survive. The distributed
    path used to overwrite meta with a short dict, which erased all of them."""
    return {
        "positive": positive_text, "negative": negative_text,
        "model": ckpt_name, "seed": seed,
        "vae_batch_size": vae_batch_size,
        "configs_json": configs_json, "resolutions_json": resolutions_json,
        "image_format": image_format,
        "job_order": job_order,
        "preview_pass": {"enabled": preview_on, "steps": pv_steps, "scale": pv_scale, "cfg_override": pv_cfg},
        "rejected_preview_fps": list((existing_data.get("meta") or {}).get("rejected_preview_fps", [])),
        **(meta_run or {})
    }


def _run_distributed_generation(
    self, distribution_config, expanded, input_jobs, existing_data,
    overwrite_existing, has_optional_inputs, lora_triggerwords_mode,
    session_name, paths, unique_id, total_jobs, seed, ckpt_name,
    positive_text, negative_text, vae_batch_size, configs_json,
    resolutions_json, flush_batch_every, use_remote_vae,
    remote_vae_endpoint, save_conditioning_cache_to_file,
    enable_model_cache, optional_model, optional_clip, optional_vae,
    optional_positive, optional_negative, optional_latent, model_cache,
    session_settings=None,
    job_order="as_written", preview_on=False, pv_steps=8, pv_scale=0.5, pv_cfg=None,
    meta_run=None, job_plan=None, skip_rejected_previews=True, only_favorited_previews=False,
    upscale_session_settings=None
):
    """
    Run generation in distributed mode.

    The master acts as both coordinator and local worker:
    - Creates a DistributionManager with the full job list
    - Notifies remote workers to start pulling jobs via API
    - Processes jobs locally by claiming from its own manager
    - Waits for all remote workers to finish
    - Returns the final HTML dashboard
    """
    if not distribution.is_distribution_available():
        raise RuntimeError(distribution.INSTALL_INSTRUCTIONS)

    # JobState is needed for direct job-state manipulation in timeout/cleanup paths.
    JobState = distribution._require_companion().usrv_dist_manager.JobState

    worker_urls = distribution_config.get("worker_urls", [])
    claim_timeout = distribution_config.get("claim_timeout", 600)

    # Inject trigger word mode into configs so workers can use it
    for conf in expanded:
        conf["_lora_triggerwords_mode"] = lora_triggerwords_mode

    # === Create and populate distribution manager ===
    manager = distribution.create_manager(
        session_name=session_name,
        paths=paths,
        unique_id=unique_id,
        existing_data=existing_data,
        job_completed_check=check_if_job_completed,
        claim_timeout_seconds=claim_timeout,
        build_prompt_with_triggers=build_prompt_with_triggers,
        get_model_cache_key=get_model_cache_key,
    )

    # Pass session-level settings (cooldown, etc.) to manager for worker passthrough
    if session_settings:
        manager._session_settings = session_settings

    # Extract image format for local master saves
    image_format = session_settings.get("image_format", "webp") if session_settings else "webp"
    if image_format != "webp":
        print(f"[GridTester] 🖼️ Image save format (distributed master): {image_format}")

    # Ids present before this run: only images made by this run get upscaled in the post-phase
    _pre_run_ids = {it.get("id") for it in existing_data.get("items", [])}
    if optional_latent is not None:
        print("[Distribution] ⚠️ Latent input: remote workers render empty latents at the placeholder size")

    def _write_meta():
        existing_data["meta"] = _distributed_run_meta(
            existing_data, positive_text, negative_text, ckpt_name, seed, vae_batch_size,
            configs_json, resolutions_json, image_format, job_order, preview_on, pv_steps,
            pv_scale, pv_cfg, meta_run)

    _pre_ids = {it.get("id") for it in existing_data.get("items", [])}
    _populate_distributed_manager(
        manager, job_plan, expanded, input_jobs, existing_data, overwrite_existing, has_optional_inputs,
        lora_triggerwords_mode, job_order=job_order, preview_on=preview_on, pv_steps=pv_steps,
        pv_scale=pv_scale, pv_cfg=pv_cfg, skip_rejected=skip_rejected_previews,
        only_favorited=only_favorited_previews
    )

    # A legacy companion (no populate_from_plan) ignores the preview pass, so its workers render FULL
    # images: the master must still upscale them, with the settings from before the preview override.
    if (preview_on and upscale_session_settings is not None
            and not (job_plan is not None and hasattr(manager, "populate_from_plan"))):
        session_settings = upscale_session_settings
        print(f"[GridTester] ⚠️ Legacy distributed companion ignores the preview pass: workers render full "
              f"images, so the master keeps the upscale phase enabled")

    # Make manager accessible to API endpoints
    distribution.set_active_manager(manager)
    _send_distribution_status(manager)

    # === Master pre-encoding phase (if enabled) ===
    # When "Use Master Text Encoding" is on, pre-encode ALL unique prompts for all
    # (model, LoRA) combinations. Workers will receive these conditionings with their
    # job claims and skip CLIP encoding entirely.
    use_master_encoding = distribution_config.get("use_master_encoding", False)
    if use_master_encoding and not (optional_positive and optional_negative):
        print(f"[Distribution] 🧠 MASTER PRE-ENCODING: Encoding all unique prompts for workers...")

        encoded_conditionings = _preencode_all_conditionings(
            expanded, lora_triggerwords_mode, ckpt_name, use_remote_vae,
            optional_model, optional_clip, optional_vae,
            optional_positive, optional_negative, model_cache
        )

        if encoded_conditionings:
            manager.set_encoded_conditionings(encoded_conditionings)
            print(f"[Distribution] 🧠 Pre-encoding complete, conditionings will be sent with job claims")
        else:
            print(f"[Distribution] ⚠️ Pre-encoding produced no results, workers will encode locally")
    elif use_master_encoding and (optional_positive and optional_negative):
        print(f"[Distribution] ⚠️ Master encoding disabled: optional conditioning inputs override prompts")

    # Early exit if all jobs were already completed (resume with nothing to do)
    initial_status = manager.get_status()
    if initial_status["total"] == 0:
        print(f"[Distribution] ⏭️ All jobs already completed, nothing to distribute")
        manager.deactivate()
        distribution.clear_active_manager()

        _write_meta()
        save_manifest(paths["manifest"], existing_data)

        html = get_html_template(session_name, existing_data, unique_id)
        return (html,)

    # === Notify remote workers ===
    master_url = distribution.get_master_url()
    print(f"[Distribution] 🌐 Master URL: {master_url}")
    print(f"[Distribution] 🌐 Notifying {len(worker_urls)} worker(s)...")

    sync_models = distribution_config.get("sync_models_to_workers", False)
    if sync_models:
        manager.sync_models_to_workers = True
        print(f"[Distribution] ☁️ Model sync enabled — workers will download missing models from master")

    worker_results = distribution.notify_workers_to_start(worker_urls, master_url, session_name, sync_models_to_workers=sync_models)
    successful_workers = sum(1 for _, ok, _ in worker_results if ok)
    print(f"[Distribution] ✅ {successful_workers}/{len(worker_urls)} workers started")

    # === Master local processing state ===
    loaded_model, loaded_clip, loaded_vae = None, None, None
    patched_model, patched_clip = None, None
    cached_model_key = None
    cached_lora_key = None
    cached_vae_key = None
    default_model_vae = None
    conditioning_cache = {"positive": {}, "negative": {}}
    incompatible_loras = {}
    pending_batch = []
    pending_job_ids = []  # Track job_ids for items in pending_batch
    total_generated = 0
    gen_index_offset = len(existing_data.get("items", []))
    job_durations = []
    eta_start_time = time.time()
    master_jobs_processed = 0

    # Remote VAE setup for master's local processing
    remote_vae_worker = None
    if use_remote_vae and expanded:
        remote_vae_worker = initialize_remote_vae(
            remote_vae_endpoint, paths["images"], paths["manifest"],
            existing_data, session_name, unique_id,
            image_format=image_format
        )
    per_config_remote_workers = {}
    current_vae_is_remote = False
    current_remote_vae_url = None
    config_overrides_vae = False  # True when config explicitly sets a non-Default VAE (overrides sampler node's remote_vae_endpoint)

    try:
        if PromptServer is not None:
            pbar = PromptServer.instance.progress_bar_pool.get_progress_bar(unique_id)
        else:
            pbar = None
    except:
        pbar = None

    dist_total = manager.get_status()["total"]

    # Ensure last_prompt_id exists for sampling callbacks (some ComfyUI versions
    # may not have it set when running inside the distributed code path)
    try:
        if PromptServer is not None and PromptServer.instance is not None:
            if not hasattr(PromptServer.instance, 'last_prompt_id'):
                PromptServer.instance.last_prompt_id = f"dist_master_{unique_id}"
    except Exception:
        pass

    # === Master local processing loop ===
    # Master claims jobs from its own distribution manager and processes them
    # using the same model loading, encoding, and generation code paths as the
    # normal (non-distributed) generation loop.
    try:
        while True:
            # Check for interrupt
            try:
                import comfy.model_management as mm
                if mm.processing_interrupted():
                    print(f"\n[Distribution/Master] 🛑 INTERRUPTED")
                    raise InterruptProcessingException()
            except InterruptProcessingException:
                raise
            except:
                pass

            # Claim next job from manager
            batch = manager.claim_batch_for_local(1)
            if not batch:
                break  # No more pending jobs for master

            claimed_job = batch[0]
            conf = claimed_job["config"]
            input_job = claimed_job["input_job"]
            job_id = claimed_job["job_id"]
            w = input_job["width"]
            h = input_job["height"]
            batch_idx = input_job.get("batch_idx", 0)
            # Main-USCG extras the master queued with this job (seed to re-render / preview render override /
            # manifest stamp). Absent for an old plan-less queue.
            ux = claimed_job.get("uscg") or {}
            current_seed = conf["seed"]
            if ux.get("seed") is not None:
                current_seed = ux["seed"]  # re-render the survivor with the seed the user approved
            elif conf.get("seed_behavior") == "randomize":
                import random
                current_seed = random.randint(0, 2**63 - 1)

            master_jobs_processed += 1
            completed_total = manager.total_completed
            print(f"[Distribution/Master] 📊 Processing job {job_id} | "
                  f"{conf['sampler']} @ {conf['steps']} steps | {w}x{h} | "
                  f"Progress: {completed_total}/{dist_total}")

            if pbar:
                try:
                    pbar.update_absolute(completed_total, dist_total)
                except:
                    pass

            # Build prompts with trigger words
            actual_positive_prompt, _ = build_prompt_with_triggers(conf, lora_triggerwords_mode)
            actual_negative_prompt = conf["negative"]

            # --- Model Switching ---
            target_model_key = get_model_cache_key(conf)
            if target_model_key != cached_model_key:
                if cached_model_key is not None:
                    patched_model, patched_clip = cleanup_model_references(
                        patched_model, patched_clip, conditioning_cache
                    )
                    conditioning_cache = {"positive": {}, "negative": {}}

                loaded_model, loaded_clip, loaded_vae = load_model_by_type(
                    conf, ckpt_name, use_remote_vae,
                    optional_model, optional_clip, optional_vae,
                    optional_positive, optional_negative, loaded_clip, loaded_vae,
                    model_cache=model_cache
                )
                if loaded_vae is None and optional_vae is not None:
                    loaded_vae = optional_vae
                default_model_vae = loaded_vae
                cached_model_key = target_model_key
                cached_lora_key = None
                cached_vae_key = "Default"
                current_vae_is_remote = False
                current_remote_vae_url = None
                config_overrides_vae = False

            # --- VAE Switching ---
            # Config Builder VAE settings take priority over sampler node's remote_vae_endpoint
            target_vae = normalize_config_vae(conf.get("vae", "Default"))
            if target_vae != cached_vae_key:
                # Flush pending batch before switching VAE
                if pending_batch:
                    _flush_pending_batch(
                        pending_batch, current_vae_is_remote, current_remote_vae_url,
                        per_config_remote_workers, use_remote_vae, remote_vae_worker,
                        loaded_vae, paths, existing_data, session_name,
                        paths["manifest"], unique_id,
                        config_overrides_vae=config_overrides_vae, image_format=image_format
                    )
                    for jid in pending_job_ids:
                        manager.complete_job(jid)
                    pending_batch = []
                    pending_job_ids = []

                if target_vae == "Default":
                    loaded_vae = default_model_vae
                    current_vae_is_remote = False
                    current_remote_vae_url = None
                    config_overrides_vae = False
                elif is_remote_vae(target_vae):
                    url = extract_remote_vae_url(target_vae)
                    current_vae_is_remote = True
                    current_remote_vae_url = url
                    config_overrides_vae = True
                    loaded_vae = None
                    if url not in per_config_remote_workers:
                        per_config_remote_workers[url] = RemoteVAEDecodeWorker(
                            endpoint=url, img_dir=paths["images"],
                            manifest_path=paths["manifest"],
                            existing_data=existing_data,
                            session_name=session_name, unique_id=unique_id,
                            image_format=image_format
                        )
                else:
                    current_vae_is_remote = False
                    current_remote_vae_url = None
                    config_overrides_vae = True
                    loaded_vae = load_vae_by_name(target_vae)
                cached_vae_key = target_vae

            # --- LoRA Switching ---
            current_lora_string = conf.get("lora_expanded", conf.get("lora", "None"))
            if current_lora_string != cached_lora_key or patched_model is None:
                patched_model, patched_clip, should_skip = load_loras(
                    loaded_model, loaded_clip, current_lora_string,
                    target_model_key, incompatible_loras, model_cache=model_cache
                )
                if should_skip:
                    manager.fail_job(job_id, "LoRA incompatible")
                    continue
                cached_lora_key = current_lora_string

            # --- Prompt Encoding ---
            # One encoder load covers both prompts (pinned_clip); nothing is loaded when both are cached.
            need_pos = not optional_positive and actual_positive_prompt not in conditioning_cache["positive"]
            need_neg = not optional_negative and actual_negative_prompt not in conditioning_cache["negative"]
            if need_pos or need_neg:
                clip_skip = conf.get("clip_skip", 0)
                with pinned_clip(patched_clip):
                    if need_pos:
                        conditioning_cache["positive"][actual_positive_prompt] = \
                            encode_prompt_with_combinators(patched_clip, actual_positive_prompt, clip_skip)
                    if need_neg:
                        conditioning_cache["negative"][actual_negative_prompt] = \
                            encode_prompt_with_combinators(patched_clip, actual_negative_prompt, clip_skip)
            final_positive = optional_positive if optional_positive else conditioning_cache["positive"][actual_positive_prompt]
            final_negative = optional_negative if optional_negative else conditioning_cache["negative"][actual_negative_prompt]

            # --- Create Latent ---
            latent_channels = get_latent_channels(loaded_model, optional_latent)
            latent_ratio = get_latent_downscale_ratio(loaded_model)
            if optional_latent is not None:
                # pixel size of a connected latent = cells x this model's ratio
                w = int(optional_latent["samples"].shape[3]) * latent_ratio
                h = int(optional_latent["samples"].shape[2]) * latent_ratio
            gen_w, gen_h = w, h
            _pv_scale = (ux.get("render") or {}).get("preview_scale")
            if _pv_scale and optional_latent is None:
                from .config_utils import preview_dims
                gen_w, gen_h = preview_dims(w, h, _pv_scale, latent_ratio)
            if optional_latent is not None:
                latent_in = {"samples": optional_latent["samples"].clone()}
            else:
                latent_in = make_empty_latent(latent_channels, gen_w, gen_h, latent_ratio)

            # --- Generate Image ---
            try:
                attention_mode = conf.get("attention_mode", "default")

                # Log active extra options
                extra_opts = []
                if conf.get("model_sampling_override", "none") != "none":
                    extra_opts.append(f"model_sampling={conf['model_sampling_override']}")
                if conf.get("use_advanced_sampling", False):
                    extra_opts.append(f"advanced({conf.get('advanced_guider','cfg_guider')}/{conf.get('advanced_scheduler','basic')})")
                if conf.get("flux_guidance_value", 0) and float(conf.get("flux_guidance_value", 0)) > 0:
                    extra_opts.append(f"flux_guidance={conf['flux_guidance_value']}")
                if extra_opts:
                    print(f"[GridTester] ⚙️ Extra options: {', '.join(extra_opts)}")

                result_latent, duration, attn_fallback = generate_image_attn_safe(
                    patched_model, current_seed, conf["steps"], conf["cfg"],
                    conf["sampler"], conf["scheduler"], final_positive, final_negative,
                    latent_in, conf["denoise"], attention_mode=attention_mode,
                    model_sampling_override=conf.get("model_sampling_override", "none"),
                    model_sampling_shift=conf.get("model_sampling_shift", 1.73),
                    model_sampling_flux_max_shift=conf.get("model_sampling_flux_max_shift", 1.15),
                    model_sampling_flux_base_shift=conf.get("model_sampling_flux_base_shift", 0.5),
                    use_advanced_sampling=conf.get("use_advanced_sampling", False),
                    advanced_guider=conf.get("advanced_guider", "cfg_guider"),
                    advanced_scheduler=conf.get("advanced_scheduler", "basic"),
                    flux_guidance_value=conf.get("flux_guidance_value", 0.0),
                    use_deep_shrink=conf.get("use_deep_shrink", False),
                    deep_shrink_block_number=conf.get("deep_shrink_block_number", 3),
                    deep_shrink_downscale_factor=conf.get("deep_shrink_downscale_factor", 2.0),
                    deep_shrink_start_percent=conf.get("deep_shrink_start_percent", 0.0),
                    deep_shrink_end_percent=conf.get("deep_shrink_end_percent", 0.35),
                    deep_shrink_downscale_after_skip=conf.get("deep_shrink_downscale_after_skip", True),
                    deep_shrink_downscale_method=conf.get("deep_shrink_downscale_method", "bicubic"),
                    deep_shrink_upscale_method=conf.get("deep_shrink_upscale_method", "bicubic"),
                    use_slg=conf.get("use_slg", False),
                    slg_double_layers=conf.get("slg_double_layers", ""),
                    slg_single_layers=conf.get("slg_single_layers", "22"),
                    slg_start_percent=conf.get("slg_start_percent", 0.0),
                    slg_end_percent=conf.get("slg_end_percent", 0.5),
                    width=gen_w,
                    height=gen_h
                )

                job_durations.append(duration)
                eta_info = calculate_eta(job_durations, completed_total + 1, dist_total)
                if eta_info:
                    print_generation_progress(
                        completed_total + 1, dist_total, conf, w, h, duration, eta_info
                    )

                meta = create_image_metadata(
                    conf, gen_w, gen_h, duration, current_seed, batch_idx,
                    actual_positive_prompt, actual_negative_prompt,
                    gen_index=gen_index_offset + total_generated
                )
                if ux.get("stamp"):
                    meta.update(ux["stamp"])
                meta["dist_worker"] = "master"
                if attn_fallback:
                    meta["attention_mode"] = "default"
                    meta["attention_fallback"] = attn_fallback

                pending_batch.append((result_latent["samples"].clone(), meta))
                pending_job_ids.append(job_id)
                total_generated += 1

                del result_latent, latent_in

            except InterruptProcessingException:
                raise
            except Exception as e:
                print(f"[Distribution/Master] ❌ Generation failed: {e}")
                manager.fail_job(job_id, str(e))
                del latent_in
                continue

            # --- Flush batch at threshold ---
            threshold = vae_batch_size if flush_batch_every <= 0 else flush_batch_every
            if len(pending_batch) >= threshold:
                _flush_pending_batch(
                    pending_batch, current_vae_is_remote, current_remote_vae_url,
                    per_config_remote_workers, use_remote_vae, remote_vae_worker,
                    loaded_vae, paths, existing_data, session_name,
                    paths["manifest"], unique_id,
                    config_overrides_vae=config_overrides_vae, image_format=image_format
                )
                for jid in pending_job_ids:
                    manager.complete_job(jid)
                pending_batch = []
                pending_job_ids = []
                _send_distribution_status(manager)

            # Periodic cleanup
            if master_jobs_processed % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Flush remaining master batch
        if pending_batch:
            _flush_pending_batch(
                pending_batch, current_vae_is_remote, current_remote_vae_url,
                per_config_remote_workers, use_remote_vae, remote_vae_worker,
                loaded_vae, paths, existing_data, session_name,
                paths["manifest"], unique_id,
                config_overrides_vae=config_overrides_vae, image_format=image_format
            )
            for jid in pending_job_ids:
                manager.complete_job(jid)
            pending_batch = []
            pending_job_ids = []
            _send_distribution_status(manager)

    except InterruptProcessingException:
        # Flush any remaining batch before cleanup
        if pending_batch:
            _flush_pending_batch(
                pending_batch, current_vae_is_remote, current_remote_vae_url,
                per_config_remote_workers, use_remote_vae, remote_vae_worker,
                loaded_vae, paths, existing_data, session_name,
                paths["manifest"], unique_id,
                config_overrides_vae=config_overrides_vae, image_format=image_format
            )
            for jid in pending_job_ids:
                manager.complete_job(jid)

        # Stop remote workers and deactivate manager
        distribution.stop_all_workers(worker_urls)
        manager.deactivate()
        distribution.clear_active_manager()

        _cleanup_per_config_remote_workers(per_config_remote_workers)
        if remote_vae_worker:
            remote_vae_worker.wait_completion()
            remote_vae_worker.stop()

        _write_meta()
        save_manifest(paths["manifest"], existing_data)

        loaded_model, loaded_clip, loaded_vae = None, None, None
        patched_model, patched_clip = None, None
        conditioning_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Re-raise so ComfyUI knows execution was interrupted (not completed)
        raise

    # === Wait for remote workers to finish ===
    # Max wait timeout prevents infinite loops if workers crash without cleanup.
    # After timeout, remaining claimed jobs are reclaimed and processed on master.
    dist_wait_timeout = distribution_config.get("wait_timeout", 600)  # Default 10 minutes
    if manager.has_pending_or_claimed:
        print(f"[Distribution] ⏳ Waiting for remote workers to finish (timeout: {dist_wait_timeout}s)...")
        wait_start = time.time()
        while manager.has_pending_or_claimed:
            # Release timed-out jobs from dead workers (was only called during claim_job)
            manager.release_timed_out_jobs()

            # Check max wait timeout — reclaim remaining jobs for master processing
            elapsed_wait = time.time() - wait_start
            if elapsed_wait > dist_wait_timeout:
                status = manager.get_status()
                remaining = status['pending'] + status['claimed']
                print(f"[Distribution] ⏰ Wait timeout ({dist_wait_timeout}s) reached with {remaining} jobs remaining — reclaiming for master")
                # Force-reclaim all remaining claimed jobs
                with manager._lock:
                    for job in manager._jobs.values():
                        if job.state == JobState.CLAIMED and job.worker_id != "master":
                            job.state = JobState.PENDING
                            job.worker_id = None
                            job.claimed_at = None
                            manager._pending_queue.append(job.job_id)
                break

            # Check for interrupt while waiting
            try:
                import comfy.model_management as mm
                if mm.processing_interrupted():
                    print(f"\n[Distribution] 🛑 INTERRUPTED while waiting for workers")
                    distribution.stop_all_workers(worker_urls)
                    manager.deactivate()
                    distribution.clear_active_manager()

                    _cleanup_per_config_remote_workers(per_config_remote_workers)
                    if remote_vae_worker:
                        remote_vae_worker.wait_completion()
                        remote_vae_worker.stop()

                    _write_meta()
                    save_manifest(paths["manifest"], existing_data)

                    loaded_model, loaded_clip, loaded_vae = None, None, None
                    patched_model, patched_clip = None, None
                    conditioning_cache.clear()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    raise InterruptProcessingException()
            except InterruptProcessingException:
                raise
            except:
                pass

            time.sleep(5)
            _send_distribution_status(manager)
            status = manager.get_status()
            print(f"[Distribution] ⏳ Pending: {status['pending']}, "
                  f"Claimed: {status['claimed']}, "
                  f"Completed: {status['completed']}/{status['total']}")

    # === Handle remaining jobs after wait timeout ===
    if manager.has_pending_or_claimed:
        status = manager.get_status()
        remaining = status['pending'] + status['claimed']
        print(f"[Distribution] ⚠️ {remaining} job(s) still incomplete after wait timeout — marking as failed and proceeding to finalization")
        # Mark remaining jobs as failed so we don't block forever
        with manager._lock:
            for job in manager._jobs.values():
                if job.state in (JobState.PENDING, JobState.CLAIMED):
                    job.state = JobState.COMPLETED  # Mark done to unblock
                    print(f"[Distribution] ⚠️ Abandoned job {job.job_id} (was {job.state})")

    # === Finalization ===
    distribution.stop_all_workers(worker_urls)
    manager.deactivate()

    # Don't clear_active_manager() immediately — a slow worker may still
    # be in the middle of submitting a result. The submit_result route only needs
    # manager != None (it does NOT check is_active). Deactivation already prevents
    # new job claims (claim_job returns 503). Delay the cleanup with a daemon thread
    # that waits for workers to go silent before clearing the manager.
    def _delayed_manager_cleanup():
        # Wait up to 5 minutes for all workers to stop submitting.
        # Check every 10s whether any registered workers still have recent heartbeats.
        for _ in range(30):  # 30 × 10s = 5 minutes max
            time.sleep(10)
            if not manager or not manager._workers:
                break
            now = time.time()
            any_alive = False
            with manager._lock:
                for w in manager._workers.values():
                    if now - w.get("last_heartbeat", 0) < 30:
                        any_alive = True
                        break
            if not any_alive:
                break
        distribution.clear_active_manager()

    threading.Thread(target=_delayed_manager_cleanup, daemon=True).start()

    _cleanup_per_config_remote_workers(per_config_remote_workers)
    if remote_vae_worker:
        print(f"[Distribution] 🌐 Waiting for remote VAE...")
        remote_vae_worker.wait_completion()
        remote_vae_worker.stop()

    print_incompatible_loras_summary(incompatible_loras)

    # === Metrics for worker-rendered images (adapters only run here; see _distributed_metrics_pass) ===
    if image_metrics.current_spec():
        _mres = _distributed_metrics_pass(existing_data, _pre_ids, paths["images"])
        if _mres["adapters"] or _mres["builtins_from_file"]:
            save_manifest(paths["manifest"], existing_data)
            print(f"[Distribution] 📏 measured {_mres['adapters']} worker image(s) on master "
                  f"({_mres['builtins_from_file']} builtins from file)")

    # === Run deferred upscales on master after all distributed generation is complete ===
    if session_settings and session_settings.get("upscaling", {}).get("enabled", False):
        upscale_settings = session_settings.get("upscaling", {})
        print(f"[Distribution] 🔍 Running upscales on master for distributed results...")
        # Release the master's model first so the per-group loads below have the VRAM
        patched_model, patched_clip = cleanup_model_references(patched_model, patched_clip, conditioning_cache)
        conditioning_cache = {"positive": {}, "negative": {}}
        loaded_model, loaded_clip, loaded_vae = None, None, None
        gc.collect()
        try:
            with manager._lock:
                _done_jobs = [(j.config, j.input_job) for j in manager._jobs.values() if j.state == JobState.COMPLETED]
            deferred_upscale_queue = _build_distributed_upscale_queue(
                existing_data.get("items", []), _pre_run_ids, _done_jobs
            )
            if deferred_upscale_queue:
                print(f"[Distribution] 🔍 {len(deferred_upscale_queue)} images from this run to upscale")
                _run_distributed_upscales(
                    deferred_upscale_queue, upscale_settings, session_settings, paths, existing_data,
                    session_name, unique_id, ckpt_name, use_remote_vae,
                    optional_model, optional_clip, optional_vae, optional_positive, optional_negative,
                    model_cache, incompatible_loras
                )
            else:
                print(f"[Distribution] ℹ️ No items to upscale")
        except InterruptProcessingException:
            raise
        except Exception as e:
            print(f"[Distribution] Upscale phase failed: {e}")

    _write_meta()
    save_manifest(paths["manifest"], existing_data)

    print(f"[Distribution] 🧹 Cleaning up...")
    loaded_model, loaded_clip, loaded_vae = None, None, None
    patched_model, patched_clip = None, None
    conditioning_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    html = get_html_template(session_name, existing_data, unique_id)

    final_status = manager.get_status()
    if job_durations:
        total_elapsed = time.time() - eta_start_time
        total_hours = int(total_elapsed // 3600)
        total_minutes = int((total_elapsed % 3600) // 60)
        total_seconds = int(total_elapsed % 60)
        avg_per_job = sum(job_durations) / len(job_durations)

        print(f"\n{'='*80}")
        print(f"[Distribution] 🎉 DISTRIBUTED GENERATION COMPLETE!")
        print(f"[Distribution] ✅ {final_status.get('completed', total_generated)} total images generated")
        print(f"[Distribution] 📊 Master processed: {total_generated} images ({avg_per_job:.1f}s avg)")
        print(f"[Distribution] ⏱️  {total_hours}h {total_minutes}m {total_seconds}s total")
        if final_status.get('failed', 0) > 0:
            print(f"[Distribution] ❌ {final_status['failed']} jobs failed permanently")

        # Per-worker breakdown
        workers = final_status.get("workers", {})
        if workers:
            print(f"[Distribution] 👥 Worker breakdown:")
            print(f"[Distribution]   📌 master: {total_generated} jobs completed")
            for wid, winfo in workers.items():
                completed = winfo.get("jobs_completed", 0)
                failed = winfo.get("jobs_failed", 0)
                duplicated = winfo.get("jobs_duplicated", 0)
                parts = []
                if failed > 0:
                    parts.append(f"{failed} failed")
                if duplicated > 0:
                    parts.append(f"{duplicated} late/duplicate")
                extra_str = f" ({', '.join(parts)})" if parts else ""
                print(f"[Distribution]   📌 {wid}: {completed} jobs completed{extra_str}")

        print(f"{'='*80}\n")

        # Send completion event to dashboard
        if PromptServer is not None:
            try:
                PromptServer.instance.send_sync("ultimate_grid.progress", {
                    "node": unique_id,
                    "session_name": session_name,
                    "current_job": dist_total,
                    "total_jobs": dist_total,
                    "progress_pct": 100,
                    "eta_str": "Done",
                    "finish_time": time.strftime("%H:%M:%S"),
                    "avg_duration": round(avg_per_job, 1),
                    "last_duration": 0,
                    "complete": True,
                    "total_elapsed": f"{total_hours}h {total_minutes}m {total_seconds}s",
                    "total_generated": final_status.get("completed", total_generated)
                })
            except Exception:
                pass

    return (html,)