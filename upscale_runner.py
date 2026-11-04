"""
Dashboard Upscale Runner
Background thread for upscaling images triggered from the dashboard.
Loads models via ComfyUI APIs, runs upscale pipeline chains,
saves results and updates manifests.
"""

import os
import time
import json
import threading
import uuid

import torch
import numpy as np
from PIL import Image as PILImage

from .model_loader import load_checkpoint, load_vae_by_name
from .batch_encoding import encode_prompt_with_combinators, encode_pair_cached
from .image_generation import upscale_image, decode_latent_with_vae, create_image_metadata
from .manifest_utils import save_manifest, base_dims_of
from . import image_metrics

import comfy.model_management as mm

# Active upscale jobs (module-level state)
_upscale_jobs = {}


def start_upscale_job(session_name, image_ids, upscale_config, all_favorited=False):
    """
    Start an async upscale job. Returns (job_id, error) immediately.
    The actual work runs in a background thread.
    """
    import folder_paths
    benchmarks_dir = os.path.realpath(os.path.join(folder_paths.get_output_directory(), "benchmarks"))
    session_dir = os.path.join(benchmarks_dir, session_name)
    manifest_path = os.path.join(session_dir, "manifest.json")
    images_dir = os.path.join(session_dir, "images")

    if not os.path.exists(manifest_path):
        return None, "Manifest not found"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    items = manifest_data.get("items", [])
    meta = manifest_data.get("meta", {})

    # Resolve image list
    if all_favorited:
        target_items = [item for item in items if item.get("favorited")]
    else:
        id_set = set(image_ids) if image_ids else set()
        target_items = [item for item in items if item.get("id") in id_set]

    if not target_items:
        return None, "No matching images found"

    job_id = f"upscale_{uuid.uuid4().hex[:12]}"
    job = {
        "id": job_id,
        "status": "running",
        "completed": 0,
        "total": len(target_items),
        "current_image": "",
        "error": None,
        "cancel": threading.Event(),
    }
    _upscale_jobs[job_id] = job

    thread = threading.Thread(
        target=_run_upscale_thread,
        args=(job, target_items, upscale_config, meta, manifest_data, manifest_path, images_dir, session_name),
        daemon=True
    )
    thread.start()

    return job_id, None


def get_upscale_status(job_id):
    """Get status of an upscale job."""
    job = _upscale_jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    return {
        "status": job["status"],
        "completed": job["completed"],
        "total": job["total"],
        "current_image": job.get("current_image", ""),
        "error": job.get("error"),
    }


def cancel_upscale_job(job_id):
    """Signal an upscale job to stop."""
    job = _upscale_jobs.get(job_id)
    if job and job["status"] == "running":
        job["cancel"].set()
        return True
    return False


def _run_upscale_thread(job, target_items, upscale_config, meta, manifest_data, manifest_path, images_dir, session_name):
    """
    Background thread that runs the upscale pipeline on each image.
    Loads the session's model/VAE/CLIP, encodes prompts, runs pipeline chains.
    """
    import itertools as up_itertools
    import random as up_random

    try:
        pipelines = upscale_config.get("pipelines", [])
        if not pipelines:
            job["status"] = "error"
            job["error"] = "No upscale pipelines configured"
            return

        # Resolve image format from manifest meta (written by orchestrator)
        _raw_fmt = meta.get("image_format", "webp")
        _fmt = (_raw_fmt or "webp").lower().strip()
        if _fmt not in ("webp", "png", "jpg", "jpeg"):
            _fmt = "webp"
        _ext = "jpg" if _fmt in ("jpg", "jpeg") else _fmt

        # Load model, VAE, CLIP for conditioning
        model_name = meta.get("model", "") or ""
        vae_name = meta.get("vae", "")

        # If meta.model is empty/missing (legacy manifests, manifests written before
        # meta.model was recorded), fall back to the first non-empty item.model in
        # the target set. Items always store their own model name.
        if not model_name:
            for _it in target_items:
                _it_model = (_it.get("model", "") or "").strip()
                if _it_model:
                    model_name = _it_model
                    print(f"[DashboardUpscale] meta.model is empty; using first item's model: {model_name}")
                    break

        # Validate the checkpoint exists before calling load_checkpoint. Without
        # this guard, a missing file ends up at comfy/utils.py:load_torch_file(None,
        # ...) and crashes with the cryptic "'NoneType' object has no attribute
        # 'lower'". Common cause: manifest copied from another machine where the
        # checkpoint isn't installed; or manifest entry has model=None.
        try:
            import folder_paths as _fp_check
            ckpt_path_check = _fp_check.get_full_path("checkpoints", model_name) if model_name else None
        except Exception:
            ckpt_path_check = None
        if not model_name or ckpt_path_check is None:
            job["status"] = "error"
            job["error"] = (
                f"Checkpoint '{model_name or '(empty)'}' not found in "
                f"ComfyUI/models/checkpoints/. The session manifest references it but "
                f"the file is not installed on this machine. Copy the .safetensors "
                f"file over, or regenerate the images on this machine first."
            )
            print(f"[DashboardUpscale] ❌ {job['error']}")
            return

        # VRAM hygiene before load_checkpoint: the previous workflow's model handles
        # may still be Python-referenced and won't be evicted by ComfyUI's LRU on a
        # new load_checkpoint call. soft_empty_cache + gc.collect releases those
        # before we add the upscale session's checkpoint on top. Critical for
        # Florence2 mode on small-VRAM cards (otherwise prior SDXL gen + new SDXL
        # session + Florence2 = OOM during beam search).
        try:
            import gc as _gc_pre
            _gc_pre.collect()
            mm.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"[DashboardUpscale] 🔄 Loading model: {model_name}")
        loaded_model, loaded_clip, loaded_vae = load_checkpoint(
            target_model_name=model_name,
            ckpt_name=model_name,  # Same as target since we're loading directly
            use_remote_vae=False,
        )

        if vae_name and vae_name != "None" and vae_name != "Default":
            print(f"[DashboardUpscale] 🔄 Loading VAE: {vae_name}")
            loaded_vae = load_vae_by_name(vae_name)

        patched_model = loaded_model
        patched_clip = loaded_clip

        # Conditioning cache for this job
        conditioning_cache = {"positive": {}, "negative": {}}

        # Get PromptServer for sending events
        PromptServer = None
        try:
            import server as comfy_server
            PromptServer = comfy_server.PromptServer
            # Ensure last_prompt_id exists — KSampler progress callbacks require it
            if PromptServer.instance is not None and not hasattr(PromptServer.instance, 'last_prompt_id'):
                PromptServer.instance.last_prompt_id = f"dashboard_upscale_{job['id']}"
        except Exception:
            pass

        for img_idx, item in enumerate(target_items):
            # Never inherit the source's metrics; recompute them only when the source had
            # them. This thread must not read generation-run state, so the spec is explicit.
            item_metrics_spec = image_metrics.spec_from_item(item)
            if job["cancel"].is_set():
                print(f"[DashboardUpscale] 🛑 Cancelled — clearing VRAM...")
                try:
                    mm.soft_empty_cache()
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                job["status"] = "cancelled"
                return

            # Extract filename from manifest item — may be in "filename" or parsed from "file" URL
            filename = item.get("filename", "")
            if not filename and item.get("file"):
                import re
                fname_match = re.search(r'filename=([^&]+)', item["file"])
                if fname_match:
                    filename = fname_match.group(1)
            job["current_image"] = filename
            image_path = os.path.join(images_dir, filename)

            if not os.path.exists(image_path):
                print(f"[DashboardUpscale] ⚠️ File not found: {filename}, skipping")
                job["completed"] += 1
                continue

            # Load image and VAE-encode to latent
            try:
                pil_image = PILImage.open(image_path).convert("RGB")
                img_array = np.array(pil_image).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                encoded = loaded_vae.encode(img_tensor[:, :, :, :3])
                result_latent = {"samples": encoded}
            except Exception as e:
                print(f"[DashboardUpscale] ⚠️ Failed to load/encode {filename}: {e}")
                job["completed"] += 1
                continue

            # Get conditioning
            pos_prompt = item.get("positive", meta.get("positive", ""))
            neg_prompt = item.get("negative", meta.get("negative", ""))
            clip_skip = item.get("clip_skip", 0)

            # One encoder load for the pair (ComfyUI re-prepares the TE on every encode otherwise)
            final_positive, final_negative = encode_pair_cached(
                patched_clip, conditioning_cache, pos_prompt, neg_prompt, clip_skip
            )

            # HiRes prompt adjustment
            hires_positive_cond = final_positive
            hires_prompt_active = False
            hires_prompt_behavior_rt = ""
            hires_prompt_text_rt = ""
            if upscale_config.get("hires_prompt_adjust") and upscale_config.get("hires_prompt_text", "").strip():
                hires_prompt_behavior_rt = upscale_config.get("hires_prompt_behavior", "append_end")
                hires_prompt_text_rt = upscale_config["hires_prompt_text"].strip()
                if hires_prompt_behavior_rt == "prepend":
                    adjusted = hires_prompt_text_rt + " " + pos_prompt
                elif hires_prompt_behavior_rt == "append_end":
                    adjusted = pos_prompt + " " + hires_prompt_text_rt
                elif hires_prompt_behavior_rt == "replace":
                    adjusted = hires_prompt_text_rt
                else:
                    adjusted = pos_prompt
                if adjusted not in conditioning_cache["positive"]:
                    conditioning_cache["positive"][adjusted] = encode_prompt_with_combinators(patched_clip, adjusted, clip_skip)
                hires_positive_cond = conditioning_cache["positive"][adjusted]
                hires_prompt_active = True

            pipe_w = item.get("width", 512)
            pipe_h = item.get("height", 512)
            upscale_combo_idx = 0
            total_upscale_duration = 0

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

                # Flatten steps with repeat
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

                    # --- SeedVR2 upscale mode ---
                    if mode == "seedvr2":
                        from .image_generation import seedvr2_upscale
                        sv_config = ucfg.get("seedvr2", {})
                        # Convert source PIL image directly (no latent decoding needed)
                        pil_input = PILImage.open(os.path.join(images_dir, filename))
                        result_pil, up_w, up_h, sv_duration = seedvr2_upscale(pil_input, sv_config)
                        total_upscale_duration += sv_duration

                        is_last_step = step_idx == len(expanded_steps) - 1
                        if is_last_step:
                            # Save the upscaled image
                            upscale_id = int(time.time() * 100000) + up_random.randint(0, 1000)
                            upscaled_filename = f"img_{upscale_id}_upscaled.{_ext}"
                            _up_path = os.path.join(images_dir, upscaled_filename)
                            if _fmt == "png":
                                result_pil.save(_up_path, format="PNG")
                            elif _fmt in ("jpg", "jpeg"):
                                result_pil.save(_up_path, format="JPEG", quality=95)
                            else:
                                result_pil.save(_up_path, format="WEBP", quality=95)

                            # Create manifest entry
                            upscaled_meta = {
                                k: v for k, v in item.items()
                                if k not in ("id", "gen_index", "file", "filename", "upscaled", "width", "height", "duration",
                                             "upscale_source", "upscale_pipeline", "upscale_mode",
                                             "upscale_ratio", "upscale_denoise", "upscale_model", "metrics")
                            }
                            upscaled_meta.update({
                                "id": upscale_id,
                                "gen_index": len(manifest_data["items"]),
                                "file": f"/view?filename={upscaled_filename}&type=output&subfolder=benchmarks/{session_name}/images",
                                "filename": upscaled_filename,
                                "width": up_w, "height": up_h,
                                "duration": round(sv_duration, 2),
                                "upscaled": True,
                                **base_dims_of(item),
                                "upscale_source": "dashboard",
                                "upscale_pipeline": pipeline_name,
                                "upscale_mode": "seedvr2",
                                "upscale_model": sv_config.get("dit_model", ""),
                            })
                            image_metrics.finalize(upscaled_meta, result_pil, spec=item_metrics_spec)
                            manifest_data["items"].insert(0, upscaled_meta)
                            upscale_combo_idx += 1
                        else:
                            # For chained steps, the next step gets the PIL result
                            # Re-read from saved temp or keep in memory
                            pil_input = result_pil
                            pipe_w_current = up_w
                            pipe_h_current = up_h
                        continue  # Skip the normal combo loop

                    # --- Florence2 Hi-Res Fix mode ---
                    if mode == "florence2_hires":
                        from .florence2_hires import (
                            run_florence2_step, preflight_florence2,
                            build_florence2_manifest_entry, build_florence2_no_detection_entry,
                        )
                        import numpy as _np

                        # Preflight once per job (cached)
                        if not job.get("_florence2_preflighted"):
                            preflight_florence2()
                            job["_florence2_preflighted"] = True

                        # Merge top-level step sampling fields into a unified config dict.
                        # hires_denoise on the step may be "0.45, 0.65" — Florence2 uses just one
                        # value per pass; take the first.
                        _hd = str(ucfg.get("hires_denoise", "0.45")).split(",")[0].strip() or "0.45"
                        f2_config = {
                            **(ucfg.get("florence2", {}) or {}),
                            "hires_denoise": float(_hd),
                            "hires_steps": int(ucfg.get("hires_steps", 15) or 15),
                            "cfg": float(ucfg.get("cfg", 1.5) or 1.5),
                            "sampler": ucfg.get("sampler", "euler"),
                            "scheduler": ucfg.get("scheduler", "simple"),
                        }

                        # Source image: load fresh from disk as tensor (1, H, W, 3)
                        src_pil = PILImage.open(os.path.join(images_dir, filename)).convert("RGB")
                        src_arr = _np.array(src_pil).astype(_np.float32) / 255.0
                        src_tensor = torch.from_numpy(src_arr).unsqueeze(0)

                        # Per-job checkpoint+lora cache (lazy init)
                        if "_florence2_ckpt_cache" not in job:
                            job["_florence2_ckpt_cache"] = {}

                        step_result = run_florence2_step(
                            source_image=src_tensor,
                            item=item,
                            step_config=f2_config,
                            fallback_handles=(patched_model, patched_clip, loaded_vae),
                            ckpt_cache=job["_florence2_ckpt_cache"],
                            conditioning_cache=conditioning_cache,
                            positive_prompt=pos_prompt,
                            negative_prompt=neg_prompt,
                            clip_skip=clip_skip,
                            # Pass the session's already-loaded model name so Florence2 can
                            # short-circuit duplicate loads when item.model matches it.
                            session_model_name=meta.get("model", ""),
                        )

                        is_last_step = step_idx == len(expanded_steps) - 1

                        if step_result.get("status") == "no_detection":
                            sentinel_id = int(time.time() * 100000) + up_random.randint(0, 1000)
                            entry = build_florence2_no_detection_entry(
                                step_result, item,
                                sentinel_id=sentinel_id,
                                current_index=len(manifest_data["items"]),
                                base_dims=base_dims_of(item),
                            )
                            manifest_data["items"].insert(0, entry)
                            upscale_combo_idx += 1
                            total_upscale_duration += float(step_result.get("duration", 0))
                            continue

                        total_upscale_duration += float(step_result.get("duration", 0))

                        if is_last_step:
                            upscale_id = int(time.time() * 100000) + up_random.randint(0, 1000)
                            upscaled_filename = f"img_{upscale_id}_upscaled.{_ext}"
                            _up_path = os.path.join(images_dir, upscaled_filename)
                            result_pil = step_result["image_pil"]
                            if _fmt == "png":
                                result_pil.save(_up_path, format="PNG")
                            elif _fmt in ("jpg", "jpeg"):
                                result_pil.save(_up_path, format="JPEG", quality=95)
                            else:
                                result_pil.save(_up_path, format="WEBP", quality=95)

                            entry = build_florence2_manifest_entry(
                                step_result, item,
                                session_name=session_name,
                                pipeline_name=pipeline_name,
                                upscale_id=upscale_id,
                                upscaled_filename=upscaled_filename,
                                current_index=len(manifest_data["items"]),
                                hires_denoise=f2_config["hires_denoise"],
                                base_dims=base_dims_of(item),
                            )
                            image_metrics.finalize(entry, result_pil, spec=item_metrics_spec)
                            manifest_data["items"].insert(0, entry)
                            upscale_combo_idx += 1
                        else:
                            pipe_w_current = step_result["image_width"]
                            pipe_h_current = step_result["image_height"]
                        continue  # Skip the normal combo loop

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
                            "mode": mode, "upscale_model": up_model,
                            "upscale_ratio": up_ratio, "hires_denoise": up_denoise,
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
                            item, up_positive, final_negative, pipe_w_current, pipe_h_current
                        )

                        is_last_step = step_idx == len(expanded_steps) - 1
                        is_last_combo = combo == combos[-1]
                        is_final_output = is_last_step and is_last_combo

                        up_metrics_sink = {}
                        up_metrics_img = None
                        if isinstance(upscale_result, dict) and "samples" in upscale_result:
                            up_metrics_img = upscaled_pil = decode_latent_with_vae(
                                loaded_vae, upscale_result["samples"],
                                metrics_sink=up_metrics_sink if (is_final_output and item_metrics_spec) else None,
                                metrics_spec=item_metrics_spec)
                            up_w, up_h = upscaled_pil.size
                            if is_final_output:
                                upscale_id = int(time.time() * 100000) + up_random.randint(0, 1000)
                                upscaled_filename = f"img_{upscale_id}_upscaled.{_ext}"
                                _up_path = os.path.join(images_dir, upscaled_filename)
                                if _fmt == "png":
                                    upscaled_pil.save(_up_path, format="PNG")
                                elif _fmt in ("jpg", "jpeg"):
                                    upscaled_pil.save(_up_path, format="JPEG", quality=95)
                                else:
                                    upscaled_pil.save(_up_path, format="WEBP", quality=95)
                        elif isinstance(upscale_result, PILImage.Image):
                            up_metrics_img = upscale_result
                            up_w, up_h = upscale_result.size
                            if is_final_output:
                                upscale_id = int(time.time() * 100000) + up_random.randint(0, 1000)
                                upscaled_filename = f"img_{upscale_id}_upscaled.{_ext}"
                                _up_path = os.path.join(images_dir, upscaled_filename)
                                if _fmt == "png":
                                    upscale_result.save(_up_path, format="PNG")
                                elif _fmt in ("jpg", "jpeg"):
                                    upscale_result.save(_up_path, format="JPEG", quality=95)
                                else:
                                    upscale_result.save(_up_path, format="WEBP", quality=95)
                        else:
                            continue

                        if not is_final_output:
                            if isinstance(upscale_result, dict) and "samples" in upscale_result:
                                pipe_latent = upscale_result
                            pipe_w_current = up_w
                            pipe_h_current = up_h
                            continue

                        # Create manifest entry — copy source item fields, override with upscale info
                        upscaled_meta = {
                            k: v for k, v in item.items()
                            if k not in ("id", "gen_index", "file", "filename", "upscaled", "width", "height", "duration",
                                         "upscale_source", "upscale_pipeline", "upscale_mode",
                                         "upscale_ratio", "upscale_denoise", "upscale_model", "metrics")
                        }
                        # gen_index: place upscaled images at the end (newest) since they were just created
                        # Use current manifest length so each new upscale gets the next sequential position
                        upscale_gen_index = len(manifest_data["items"])

                        upscaled_meta.update({
                            "id": upscale_id,
                            "gen_index": upscale_gen_index,
                            "file": f"/view?filename={upscaled_filename}&type=output&subfolder=benchmarks/{session_name}/images",
                            "filename": upscaled_filename,
                            "width": up_w,
                            "height": up_h,
                            "duration": round(upscale_duration, 2),
                            "upscaled": True,
                            **base_dims_of(item),
                            "upscale_source": "dashboard",
                            "upscale_pipeline": pipeline_name,
                            "upscale_mode": mode,
                            "upscale_ratio": up_ratio,
                            "upscale_denoise": up_denoise,
                        })
                        if up_model:
                            upscaled_meta["upscale_model"] = up_model
                        if hires_prompt_active:
                            upscaled_meta["hires_prompt_behavior"] = hires_prompt_behavior_rt
                            upscaled_meta["hires_prompt_text"] = hires_prompt_text_rt

                        image_metrics.finalize(upscaled_meta, up_metrics_img, up_metrics_sink, spec=item_metrics_spec)
                        manifest_data["items"].insert(0, upscaled_meta)
                        upscale_combo_idx += 1

            if upscale_combo_idx > 0:
                save_manifest(manifest_path, manifest_data)
                # Send all new items to dashboard (may be multiple from different pipelines/combos)
                if PromptServer is not None:
                    try:
                        new_items = manifest_data["items"][-upscale_combo_idx:]
                        PromptServer.instance.send_sync("ultimate_grid.update_data", {
                            "session_name": session_name,
                            "new_items": new_items
                        })
                    except Exception:
                        pass

            job["completed"] = img_idx + 1

            # Send progress
            if PromptServer is not None:
                try:
                    PromptServer.instance.send_sync("ultimate_grid.progress", {
                        "session_name": session_name,
                        "current_job": img_idx + 1,
                        "total_jobs": len(target_items),
                        "progress_pct": int(((img_idx + 1) / len(target_items)) * 100),
                        "eta_str": "",
                        "avg_duration": 0,
                        "last_duration": 0,
                        "phase": "upscaling",
                    })
                except Exception:
                    pass

            print(f"[DashboardUpscale] 🔄 Upscaled {img_idx+1}/{len(target_items)}: {filename}")

        # Done — clear VRAM and RAM
        print(f"[DashboardUpscale] 🧹 Clearing VRAM and RAM...")
        try:
            mm.soft_empty_cache()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as cleanup_err:
            print(f"[DashboardUpscale] ⚠️ Cleanup warning: {cleanup_err}")

        job["status"] = "complete"
        print(f"[DashboardUpscale] ✅ Complete: {job['completed']}/{job['total']} images upscaled")

    except Exception as e:
        import traceback
        traceback.print_exc()
        job["status"] = "error"
        job["error"] = str(e)
        print(f"[DashboardUpscale] ❌ Error: {e}")
    finally:
        # Always attempt cleanup on any exit path
        try:
            mm.soft_empty_cache()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
