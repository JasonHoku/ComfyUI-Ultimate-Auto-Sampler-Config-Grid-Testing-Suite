"""
Main SamplerGridTester Node for ComfyUI
Orchestrates grid generation across multiple configurations
"""

import re
import torch
import json
import os
import time
import random
import hashlib
import uuid
import folder_paths
import nodes
import comfy.utils
import comfy.sd
import comfy.samplers
import comfy.model_management
from PIL import Image
import numpy as np

# Import from split modules
from .remote_vae import (
    detect_model_type,
    RemoteVAEDecodeWorker,
)
from .lora_utils import load_and_save_tags
from .config_utils import (
    parse_json_with_error,
    parse_float_input,
    parse_string_input,
    expand_configs,
    prepare_input_jobs,
    sanitize_session_name,
    normalize_str,
    get_files_from_folder,
    parse_lora_definition
)

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

from .html_generator import get_html_template


class SamplerGridTester:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "configs_json": ("STRING", {
                    "multiline": True,
                    "default": '[{"sampler": "euler", "scheduler": "normal", "steps": 20, "cfg": 7.0}]',
                    "tooltip": "Configs JSON (typically wired from UltimateConfigBuilder). Carries all run settings, prompts, models, LoRAs, etc. via _session_settings.",
                }),
            },
            "optional": {
                "optional_model": ("MODEL",),
                "optional_clip": ("CLIP",),
                "optional_vae": ("VAE",),
                "optional_positive": ("CONDITIONING",),
                "optional_negative": ("CONDITIONING",),
                "optional_latent": ("LATENT",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dashboard_html",)
    FUNCTION = "run_tests"
    CATEGORY = "sampling/testing"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Force re-execution when optional inputs are connected.

        ComfyUI's standard IS_CHANGED uses input hash comparisons to decide
        whether to skip execution. However, for complex objects like MODEL, CLIP,
        VAE, and CONDITIONING passed via optional inputs, the hash doesn't reliably
        change when the upstream node's internal state changes (e.g., user swaps
        model in a loader node).

        When ANY optional input is connected, we return a unique value each time
        to force ComfyUI to always re-execute this node, ensuring the grid tester
        picks up upstream changes to models, LoRAs, prompts, etc.

        When NO optional inputs are connected, we return a deterministic hash
        based on the text/numeric inputs so ComfyUI can properly cache.
        """
        optional_keys = [
            "optional_model", "optional_clip", "optional_vae",
            "optional_positive", "optional_negative", "optional_latent"
        ]

        # Check if any optional input is connected (not None and present in kwargs)
        has_optional = any(
            kwargs.get(key) is not None for key in optional_keys
        )

        if has_optional:
            # Force re-execution every time - optional inputs can't be reliably hashed
            return float("NaN")

        # No optional inputs - return deterministic hash so ComfyUI can cache properly
        # Hash the text/numeric inputs that we CAN reliably track
        hash_parts = []
        for key in sorted(kwargs.keys()):
            if key not in optional_keys and key != "unique_id":
                val = kwargs.get(key)
                if val is not None:
                    hash_parts.append(f"{key}={val}")

        return hashlib.md5("|".join(hash_parts).encode()).hexdigest()

    # --- HELPER METHODS ---
    def is_float_equal(self, a, b, tolerance=1e-5):
        """Robust float comparison"""
        try:
            return abs(float(a) - float(b)) < tolerance
        except:
            return str(a) == str(b)


    def hash_conditioning(self, conditioning):
        """Create a hash of conditioning tensor for change detection"""
        if conditioning is None:
            return "none"
        
        try:
            tensor = conditioning[0][0]
            tensor_bytes = tensor.cpu().numpy().tobytes()
            hash_obj = hashlib.md5(tensor_bytes)
            return hash_obj.hexdigest()[:16]
        except Exception as e:
            print(f"[GridTester] Warning: Could not hash conditioning: {e}")
            return "unknown"


    def get_latent_channels(self, model, optional_latent):
        """Detect the correct number of latent channels for the model"""
        # First, check if we have an optional_latent to extract from
        if optional_latent is not None:
            channels = optional_latent["samples"].shape[1]
            print(f"[GridTester] 🔍 Detected {channels} latent channels from optional_latent")
            return channels
        
        # Try to detect from model
        if model is not None:
            try:
                if hasattr(model, 'model') and hasattr(model.model, 'latent_format'):
                    latent_format = model.model.latent_format
                    if hasattr(latent_format, 'latent_channels'):
                        channels = latent_format.latent_channels
                        print(f"[GridTester] 🔍 Detected {channels} latent channels from model.latent_format")
                        return channels
                
                if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                    diff_model = model.model.diffusion_model
                    if hasattr(diff_model, 'in_channels'):
                        channels = diff_model.in_channels
                        print(f"[GridTester] 🔍 Detected {channels} latent channels from diffusion_model.in_channels")
                        return channels
            except Exception as e:
                print(f"[GridTester] ⚠️ Could not detect latent channels: {e}")
        
        # Default to 4 (SD1.5/SDXL)
        print(f"[GridTester] 🔍 Using default 4 latent channels (SD1.5/SDXL)")
        return 4


    def find_existing_match(self, existing_items, conf, w, h, current_seed, batch_idx, match_keys):
        """Returns the index of matching item, or -1 if not found"""
        for idx, item in enumerate(existing_items):
            is_match = True
            for k in match_keys:
                val_conf = conf.get(k)
                
                # Override with current job values
                if k == "width": 
                    val_conf = w
                elif k == "height": 
                    val_conf = h
                elif k == "seed": 
                    val_conf = current_seed
                elif k == "batch_idx": 
                    val_conf = batch_idx
                
                val_item = item.get(k)
                
                # Handle model defaults
                if k == "model":
                    if val_item is None:
                        if val_conf != "Default":
                            is_match = False
                            break
                    elif val_conf == "Default" and val_item is None:
                        continue
                
                # Float comparison
                if isinstance(val_conf, float) or isinstance(val_item, float):
                    if not self.is_float_equal(val_conf, val_item):
                        is_match = False
                        break
                
                # String comparison
                elif isinstance(val_conf, str) and isinstance(val_item, str):
                    if normalize_str(val_conf) != normalize_str(val_item):
                        is_match = False
                        break
                
                # Direct comparison
                elif val_item != val_conf:
                    is_match = False
                    break
            
            if is_match:
                return idx
        
        return -1


    def run_tests(self, configs_json,
                optional_model=None, optional_clip=None, optional_vae=None,
                optional_positive=None, optional_negative=None, optional_latent=None,
                unique_id=None):

        # === Defaults for removed widgets (all moved to Builder UI) ===
        # These are read from _session_settings below when present, otherwise
        # the hardcoded defaults here are used.
        # ckpt_name: fall back to the first available checkpoint so that a
        # standalone node (no Builder UI wired) with "model": "Default" does
        # not crash the orchestrator with get_full_path(None).
        _ckpt_list = folder_paths.get_filename_list("checkpoints")
        ckpt_name = _ckpt_list[0] if _ckpt_list else None
        positive_text = ""
        negative_text = ""
        seed = 0
        denoise = "1.0"
        vae_batch_size = 1
        resolutions_json = '[[1024, 1024]]'
        session_name = "my_session"
        overwrite_existing = False
        flush_batch_every = 1
        add_random_seeds_to_gens = 0
        lora_triggerwords_mode = "None"
        remote_vae_endpoint = "None"
        save_conditioning_cache_to_file = False
        enable_model_cache = False

        # Import the generation logic from the separate module
        from .generation_orchestrator import run_generation_loop

        # Disable cache saving if any optional inputs are connected
        # (changes in models/LoRAs cannot be reliably detected from optional inputs)
        if optional_model is not None or optional_clip is not None or optional_positive is not None or optional_negative is not None:
            if save_conditioning_cache_to_file:
                print("[GridTester] ⚠️ save_conditioning_cache_to_file disabled: optional inputs connected (changes cannot be reliably detected)")
            save_conditioning_cache_to_file = False

        # Extract distribution config from configs_json (embedded as _distribution key)
        dist_config = None
        session_settings = None
        try:
            import json
            parsed = json.loads(configs_json)
            if isinstance(parsed, dict) and "_distribution" in parsed:
                dist_config = parsed["_distribution"]
                session_settings = parsed.get("_session_settings")
                if session_settings:
                    print(f"[GridTester] ⚙️ Session settings extracted: {list(session_settings.keys())}")
                # Replace configs_json with just the configs array for downstream
                configs_json = json.dumps(parsed["configs"], indent=2, ensure_ascii=False)
                print(f"[GridTester] 🌐 Distribution config extracted: {len(dist_config.get('worker_urls', []))} worker(s), enabled={dist_config.get('enabled')}")
            elif isinstance(parsed, dict) and "configs" in parsed:
                # Configs wrapped in dict but no distribution settings — unwrap for downstream
                session_settings = parsed.get("_session_settings")
                if session_settings:
                    print(f"[GridTester] ⚙️ Session settings extracted: {list(session_settings.keys())}")
                configs_json = json.dumps(parsed["configs"], indent=2, ensure_ascii=False)
                print(f"[GridTester] ℹ️ No distribution settings in configs_json")
            else:
                print(f"[GridTester] ℹ️ No distribution settings in configs_json")
        except Exception as e:
            print(f"[GridTester] ⚠️ Error parsing configs_json for distribution: {e}")

        # Builder UI Run Settings override widget values when present.
        # When a user sets these in the Builder UI, _session_settings carries
        # them through configs_json and they take precedence over what's
        # wired to the SamplerGridTester widgets. Generator widgets are still
        # the fallback for backward compat.
        if session_settings:
            if "overwrite_existing" in session_settings:
                overwrite_existing = bool(session_settings["overwrite_existing"])
                print(f"[GridTester] ⚙️ overwrite_existing overridden by Builder UI: {overwrite_existing}")
            if "flush_batch_every" in session_settings:
                try:
                    flush_batch_every = int(session_settings["flush_batch_every"])
                    print(f"[GridTester] ⚙️ flush_batch_every overridden by Builder UI: {flush_batch_every}")
                except (TypeError, ValueError):
                    pass
            if "lora_triggerwords_mode" in session_settings:
                lora_triggerwords_mode = str(session_settings["lora_triggerwords_mode"])
                print(f"[GridTester] ⚙️ lora_triggerwords_mode overridden by Builder UI: {lora_triggerwords_mode}")
            if "save_conditioning_cache_to_file" in session_settings:
                save_conditioning_cache_to_file = bool(session_settings["save_conditioning_cache_to_file"])
                print(f"[GridTester] ⚙️ save_conditioning_cache_to_file overridden by Builder UI: {save_conditioning_cache_to_file}")
            if "enable_model_cache" in session_settings:
                enable_model_cache = bool(session_settings["enable_model_cache"])
                print(f"[GridTester] ⚙️ enable_model_cache overridden by Builder UI: {enable_model_cache}")
            if "vae_batch_size" in session_settings:
                try:
                    vae_batch_size = int(session_settings["vae_batch_size"])
                    print(f"[GridTester] ⚙️ vae_batch_size overridden by Builder UI: {vae_batch_size}")
                except (TypeError, ValueError):
                    pass
            if "session_name" in session_settings:
                session_name = str(session_settings["session_name"])
                print(f"[GridTester] ⚙️ session_name overridden by Builder UI: {session_name}")
            if "add_random_seeds_to_gens" in session_settings:
                try:
                    add_random_seeds_to_gens = int(session_settings["add_random_seeds_to_gens"])
                    print(f"[GridTester] ⚙️ add_random_seeds_to_gens overridden by Builder UI: {add_random_seeds_to_gens}")
                except (TypeError, ValueError):
                    pass

        return run_generation_loop(
            self,
            ckpt_name, positive_text, negative_text, seed, denoise, vae_batch_size,
            overwrite_existing, flush_batch_every, configs_json, resolutions_json,
            session_name, unique_id, add_random_seeds_to_gens, lora_triggerwords_mode,
            remote_vae_endpoint, save_conditioning_cache_to_file, enable_model_cache,
            optional_model, optional_clip, optional_vae,
            optional_positive, optional_negative, optional_latent,
            distribution_config=dist_config,  # Extracted from configs_json
            session_settings=session_settings  # Extracted from configs_json
        )


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SamplerGridTester": SamplerGridTester
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerGridTester": "Sampler Grid Tester"
}