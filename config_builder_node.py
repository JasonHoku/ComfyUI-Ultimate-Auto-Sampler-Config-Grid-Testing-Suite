"""
Ultimate Config Builder - Complete HTML UI Version
ALL data stored in single widget (lora_config)
Python reads everything from that widget
"""

import os
import sys
import json
import time
import builtins
import folder_paths
from typing import List, Dict, Any
import server
from aiohttp import web
import hashlib
from .civitai import civitai_fetch_by_hash


def safe_print(*args, **kwargs):
    """
    Windows-safe print that survives colorama/stdout corruption.
    Falls back to raw sys.__stdout__ if the wrapped stdout is broken,
    and silently drops the message if even that fails. This prevents
    OSError [Errno 22] from Windows console bugs crashing node execution.
    """
    try:
        builtins.print(*args, **kwargs)
    except (OSError, ValueError):
        try:
            msg = " ".join(str(a) for a in args) + kwargs.get("end", "\n")
            sys.__stdout__.write(msg)
            sys.__stdout__.flush()
        except Exception:
            pass  # Drop silently — don't crash the node over a log line


# Shadow the module-level `print` so every print() call in this file routes
# through safe_print. Without this, sibling methods (process_lora_array,
# lookup_triggers_endpoint, etc.) still hit the broken stdout and crash with
# OSError [Errno 22] on Windows — the local `print = safe_print` in
# generate_config only covers that one function.
print = safe_print




# =============================================================================
# INLINE LORA_UTILS FUNCTIONS (for compatibility)
# =============================================================================

def load_json_from_file(file_path):
    """Load JSON data from file"""
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print(f"[ConfigBuilder] Error decoding JSON in file: {file_path}")
        return None


def save_dict_to_json(data_dict, file_path):
    """Save dictionary to JSON file"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)
            print(f"[ConfigBuilder] Data saved to {file_path}")
    except Exception as e:
        print(f"[ConfigBuilder] Error saving JSON to file: {e}")


def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def get_model_version_info(hash_value):
    """Fetch model version info from Civitai API using hash"""
    return civitai_fetch_by_hash(hash_value)


def load_and_save_tags(lora_name, force_fetch=False, auto_fetch=True):
    """
    Load trigger tags for a LoRA, fetching from Civitai API if necessary.
    Caches results to loras_tags.json.
    
    Args:
        lora_name: Name of the LoRA file (may include path with / or \\)
        force_fetch: Force fetch from API even if cached
        auto_fetch: If False, don't hash uncached LoRAs (just save empty entry)
        
    Returns:
        List of trigger words/tags
    """
    output_dir = folder_paths.get_output_directory()
    json_tags_path = os.path.join(output_dir, "benchmarks", "loras_tags.json")
    
    lora_tags = load_json_from_file(json_tags_path)
    
    # Normalize the lora_name to use forward slashes for cache lookup
    normalized_name = lora_name.replace("\\", "/")
    
    # Try to find in cache
    output_tags = None
    found_in_cache = False
    
    if lora_tags is not None:
        # Try original name first
        if lora_name in lora_tags:
            output_tags = lora_tags[lora_name]
            found_in_cache = True
        # If not found, try normalized name
        elif normalized_name != lora_name and normalized_name in lora_tags:
            output_tags = lora_tags[normalized_name]
            found_in_cache = True
        # If still not found, try backslash version
        else:
            backslash_name = normalized_name.replace("/", "\\")
            if backslash_name in lora_tags:
                output_tags = lora_tags[backslash_name]
                found_in_cache = True
    
    # If found in cache, return immediately
    if found_in_cache:
        return output_tags if output_tags is not None else []
    
    # NOT IN CACHE - Need to fetch from API
    lora_path = folder_paths.get_full_path("loras", lora_name)
    
    # Check if lora_path is valid before attempting to hash
    if lora_path is None:
        print(f"[ConfigBuilder] ⚠️ LoRA file not found: {lora_name}")
        return []
    
    # Only hash if force_fetch is True OR (not in cache AND auto_fetch is True)
    if force_fetch or auto_fetch:
        print(f"[ConfigBuilder] 🔄 Fetching tags for {lora_name}")
        try:
            LORAsha256 = calculate_sha256(lora_path)
            model_info = get_model_version_info(LORAsha256)
            
            if model_info is not None and "trainedWords" in model_info:
                print(f"[ConfigBuilder] ✅ Tags found for {lora_name}")
                if lora_tags is None:
                    lora_tags = {}
                lora_tags[normalized_name] = model_info["trainedWords"]
                save_dict_to_json(lora_tags, json_tags_path)
                return model_info["trainedWords"]
            else:
                print(f"[ConfigBuilder] No tags found for {lora_name}")
                if lora_tags is None:
                    lora_tags = {}
                lora_tags[normalized_name] = []
                save_dict_to_json(lora_tags, json_tags_path)
                return []
        except Exception as e:
            print(f"[ConfigBuilder] Error processing {lora_name}: {e}")
            return []
    else:
        # Not in cache and auto_fetch=False
        if lora_tags is None:
            lora_tags = {}
        lora_tags[normalized_name] = []
        save_dict_to_json(lora_tags, json_tags_path)
        return []


class UltimateConfigBuilder:
    """
    Config builder with complete HTML UI.
    All data is stored in the lora_config widget as a single JSON object.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        sessions = cls.get_available_sessions()
        
        return {
            "required": {
                # Session Management (hidden, controlled by HTML)
                "session_name": ("STRING", {
                    "default": "my_test_session",
                    "multiline": False
                }),
                "load_session": (sessions, {
                    "default": sessions[0] if sessions else "None"
                }),
                
                # Sampler Settings (hidden, controlled by HTML)
                "samplers": ("STRING", {
                    "default": "euler, dpmpp_2m",
                    "multiline": False
                }),
                "schedulers": ("STRING", {
                    "default": "normal, karras",
                    "multiline": False
                }),
                "steps": ("STRING", {
                    "default": "20, 30",
                    "multiline": False
                }),
                "cfg": ("STRING", {
                    "default": "7.0",
                    "multiline": False
                }),
                
                # LoRA Configuration (ACTUAL DATA STORAGE - contains EVERYTHING)
                "lora_config": ("STRING", {
                    "default": cls.get_default_config(),
                    "multiline": True
                }),
                
                # Options (hidden, controlled by HTML)
                "include_none": ("BOOLEAN", {
                    "default": False
                }),
            },
            "optional": {
                "model": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            }
        }
    
    # session_name is carried through configs_json's _session_settings now,
    # so the dedicated output socket is redundant. Removed to simplify the
    # node interface.
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("configs_json",)
    FUNCTION = "generate_config"
    CATEGORY = "sampling/testing"
    OUTPUT_NODE = True
    
    @staticmethod
    def get_default_config():
        """Return default complete configuration"""
        config = {
            "session_name": "my_test_session",
            "include_none": False,
            "global_positive_groups": [],
            "global_negative": "",
            "config_arrays": [
                {
                    "name": "Config 1",
                    "samplers": "euler, dpmpp_2m",
                    "schedulers": "normal, karras",
                    "steps": "20, 30",
                    "cfg": "7.0",
                    "model": "",
                    "loras": ["None"],
                    "lora_omit_triggers": [],
                    "lora_triggerwords_append_settings": {},
                    "combine": True,
                    "positive_prompt_groups": [],
                    "negative_prompt": "",
                    "use_custom_prompts": False
                }
            ]
        }
        return json.dumps(config, indent=2, ensure_ascii=False)
    
    @staticmethod
    def get_available_sessions() -> List[str]:
        """Scan benchmarks folder for available sessions, sorted by newest first"""
        sessions = ["None"]
        try:
            output_dir = folder_paths.get_output_directory()
            benchmarks_dir = os.path.join(output_dir, "benchmarks")
            
            if os.path.exists(benchmarks_dir):
                session_items = []
                for item in os.listdir(benchmarks_dir):
                    item_path = os.path.join(benchmarks_dir, item)
                    manifest_path = os.path.join(item_path, "manifest.json")
                    
                    if os.path.isdir(item_path) and os.path.exists(manifest_path):
                        # Get modification time for sorting
                        mtime = os.path.getmtime(item_path)
                        session_items.append((item, mtime))
                
                # Sort by modification time (newest first)
                session_items.sort(key=lambda x: x[1], reverse=True)
                sessions.extend([item[0] for item in session_items])
        except Exception as e:
            print(f"[ConfigBuilder] Warning: Could not scan sessions: {e}")
        
        return sessions
    
    @staticmethod
    def expand_lora_folders(lora_list: List[str]) -> List[str]:
        """
        Expand folder references to individual LoRA files.
        
        Args:
            lora_list: List of LoRA strings (may include folders)
            
        Returns:
            List with folders expanded to individual files
        """
        expanded = []
        available_loras = folder_paths.get_filename_list("loras")
        
        for lora_str in lora_list:
            if not lora_str or lora_str == "None":
                continue
            
            # Parse out the name (before any : strength modifiers)
            lora_name = lora_str.split(":")[0]
            
            # Check if it's a folder reference
            is_folder = lora_name.endswith("/") or lora_name.endswith("/*")
            
            if is_folder:
                # Remove trailing / or /*
                folder_name = lora_name.rstrip("/*").rstrip("/")
                
                # Normalize to forward slashes for comparison
                folder_prefix = folder_name.replace("\\", "/") + "/"
                
                # Find all LoRAs in this folder
                for lora_file in available_loras:
                    normalized_file = lora_file.replace("\\", "/")
                    if normalized_file.startswith(folder_prefix):
                        expanded.append(lora_file)
                
                lora_count = len([l for l in available_loras if l.replace('\\', '/').startswith(folder_prefix)])
                print(f"[ConfigBuilder] Expanded folder '{lora_name}' to {lora_count} LoRAs")
            else:
                # Regular LoRA file
                expanded.append(lora_str)
        
        return expanded
    
    @staticmethod
    def lookup_lora_triggers(lora_list: List[str]) -> Dict[str, List[str]]:
        """
        Lookup trigger words for a list of LoRAs.
        
        Args:
            lora_list: List of LoRA strings (may include strengths and folders)
            
        Returns:
            Dict mapping lora name to list of trigger words
        """
        # First, expand any folder references
        expanded_list = UltimateConfigBuilder.expand_lora_folders(lora_list)
        
        trigger_map = {}
        
        for lora_str in expanded_list:
            if not lora_str or lora_str == "None":
                continue
                
            # Handle combined LoRAs (e.g., "lora1 + lora2")
            if " + " in lora_str:
                parts = lora_str.split(" + ")
                for part in parts:
                    part = part.strip()
                    if part and part != "None":
                        lora_name = part.split(":")[0]
                        if lora_name not in trigger_map:
                            try:
                                triggers = load_and_save_tags(
                                    lora_name, 
                                    force_fetch=False,
                                    auto_fetch=True
                                )
                                trigger_map[lora_name] = triggers if triggers else []
                            except Exception as e:
                                print(f"[ConfigBuilder] Error fetching triggers for {lora_name}: {e}")
                                trigger_map[lora_name] = []
            else:
                # Single LoRA
                lora_name = lora_str.split(":")[0]
                if lora_name and lora_name != "None" and lora_name not in trigger_map:
                    try:
                        triggers = load_and_save_tags(
                            lora_name,
                            force_fetch=False,
                            auto_fetch=True
                        )
                        trigger_map[lora_name] = triggers if triggers else []
                    except Exception as e:
                        print(f"[ConfigBuilder] Error fetching triggers for {lora_name}: {e}")
                        trigger_map[lora_name] = []
        
        return trigger_map
    
    @staticmethod
    def parse_int_list(value: str) -> List[int]:
        """Parse comma-separated integers"""
        items = UltimateConfigBuilder.parse_comma_list(value)
        result = []
        for item in items:
            try:
                # Cast to float first to handle strings like "20.0", then to int
                result.append(int(float(item)))
            except ValueError:
                print(f"[ConfigBuilder] Warning: Could not parse integer '{item}'")
        return result

    @staticmethod
    def parse_comma_list(value) -> List[str]:
        """Parse comma-separated string or pass through list"""
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if not value or str(value).strip() == "":
            return []
        return [item.strip() for item in str(value).split(",") if item.strip()]

    @staticmethod
    def parse_number_list(value: str) -> List[float]:
        """Parse comma-separated numbers"""
        items = UltimateConfigBuilder.parse_comma_list(value)
        result = []
        for item in items:
            try:
                result.append(float(item))
            except ValueError:
                print(f"[ConfigBuilder] Warning: Could not parse '{item}'")
        return result

    @staticmethod
    def process_lora_array(config_array: Dict, include_none: bool) -> List[str]:
        """
        Process a SINGLE config array and return its lora strings.
        
        Args:
            config_array: Single config array dict from config_arrays
            include_none: Whether to include "None" in results
            
        Returns:
            List of lora strings for this config array
        """
        array_name = config_array.get("name", "Unnamed Config")
        
        # FIX: Force combine to True. 
        # The UI defaults new configs to 'combine: false' but the Preview 
        # treats them as combined. We enforce True here to match the Preview/Stack behavior.
        combine = True 
        
        loras = config_array.get("loras", [])
        lora_bypass_states = config_array.get("lora_bypass_states", {})

        # Filter out bypassed loras, then convert to strings
        lora_strings = []
        for lora in loras:
            if not lora or lora == "None":
                continue
            lora_str = str(lora)
            # Extract lora name (path before first colon) to check bypass state
            lora_name = lora_str.split(":")[0] if ":" in lora_str else lora_str
            if lora_bypass_states.get(lora_name, False):
                continue  # Skip bypassed
            lora_strings.append(lora_str)

        # Apply weight arrays (bracket notation) for "Compare Strengths" entries.
        # The Builder UI stores per-LoRA strength arrays in a side-channel
        # (config_array.lora_weight_arrays[name + "_model" / "_clip"]). Rewrite
        # each lora string from "name:m:c" -> "name:[m_arr]:[c_arr|c]" so the
        # orchestrator's _expand_lora_weight_arrays can fan them out at runtime.
        # SYNC WARNING: must match convertStateToConfigs() in conf-builder-utilities.js.
        weight_arrays = config_array.get("lora_weight_arrays", {}) or {}

        def _apply_weight_array(lora_str: str) -> str:
            name, _sep, rest = lora_str.partition(":")
            parts = rest.split(":") if rest else []
            model_str = parts[0] if parts else "1.00"
            clip_str = parts[1] if len(parts) > 1 else model_str
            model_arr = weight_arrays.get(name + "_model")
            clip_arr = weight_arrays.get(name + "_clip")
            if model_arr and len(model_arr) > 1:
                model_part = "[" + ", ".join(str(v) for v in model_arr) + "]"
                if clip_arr and len(clip_arr) > 1:
                    # Legacy/edge case: separate clip array → preserves Cartesian behavior
                    # for backward-compat with workflows saved before the locked-semantics
                    # simplification (2026-04-28).
                    clip_part = "[" + ", ".join(str(v) for v in clip_arr) + "]"
                    return f"{name}:{model_part}:{clip_part}"
                # Locked semantics: omit the clip part entirely.
                # parse_lora_definition (config_utils.py) defaults clip = model when no
                # third segment is present, so "name:[a,b,c]" expands to N configs each
                # with model=clip=value. This avoids the Cartesian explosion that
                # "name:[m]:[c]" would trigger in expand_lora_stack.
                return f"{name}:{model_part}"
            if clip_arr and len(clip_arr) > 1:
                # Edge: clip-only array (no model array) — emit as before.
                return f"{name}:{model_str}:[" + ", ".join(str(v) for v in clip_arr) + "]"
            return lora_str

        lora_strings = [_apply_weight_array(s) for s in lora_strings]

        # Add combined version if requested
        if combine and len(lora_strings) > 1:
            # Folder refs ("name/") are excluded from stacking — they expand at
            # runtime into multiple loras. Detect by the NAME portion (before the
            # first colon) ending in "/", not the whole string, so that
            # "folder/:[a,b]:[c,d]" (folder ref + strength array) is also excluded.
            stackable = [s for s in lora_strings if not s.split(":", 1)[0].endswith("/")]
            if len(stackable) > 1:
                # When combine is true, ONLY return the combined version
                combined = " + ".join(stackable)
                lora_strings = [combined]
                print(f"[ConfigBuilder] {array_name}: Combined {len(stackable)} LoRAs into stack")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_strings = []
        for item in lora_strings:
            if item not in seen:
                seen.add(item)
                unique_strings.append(item)
        
        print(f"[ConfigBuilder] {array_name}: Processed {len(unique_strings)} LoRA configs")
        return unique_strings
    
    @staticmethod
    def state_to_configs_json(state: dict) -> str:
        """
        Pure transformer: builder UI state -> configs_json string.

        Single source of truth. Called by:
          - generate_config()              (run-time, via lora_config widget)
          - /configbuilder/preview         (preview endpoint, via POST body)

        Both code paths share this function. By construction, the preview
        and the actual node output cannot disagree.

        Args:
            state: parsed builder UI state (the JSON object stored in the
                   lora_config widget — config_arrays, prompts, etc.)
        Returns:
            The configs_json string (top-level: {"configs": [...], "_distribution": ..., "_session_settings": ...}).
        """
        # SINGLE SOURCE OF TRUTH for builder UI state -> configs_json.

        actual_include_none = state.get("include_none", False)
        config_arrays = state.get("config_arrays", [])

        # Global prompts (used when per-config prompts are not defined)
        global_positive_groups = state.get("global_positive_groups", [])
        global_negative = state.get("global_negative", "")

        if not config_arrays:
            config_arrays = [{
                "name": "Config 1",
                "samplers": ["euler"],
                "schedulers": ["normal"],
                "steps": "20",
                "cfg": "7.0",
                "model": "",
                "loras": ["None"],
                "lora_omit_triggers": [],
                "lora_triggerwords_append_settings": {},
                "combine": True,
                "positive_prompt_groups": [],
                "negative_prompt": "",
                "use_custom_prompts": False
            }]

        configs_output = []
        total_lora_configs = 0

        for config_array in config_arrays:
            # Parse values from this config array
            sampler_list = UltimateConfigBuilder.parse_comma_list(config_array.get("samplers", "euler"))
            scheduler_list = UltimateConfigBuilder.parse_comma_list(config_array.get("schedulers", "normal"))
            steps_list = UltimateConfigBuilder.parse_int_list(config_array.get("steps", "20"))
            cfg_list = UltimateConfigBuilder.parse_number_list(config_array.get("cfg", "7.0"))
            models_raw = config_array.get("models", ["None"])
            omit_triggers = config_array.get("lora_omit_triggers", [])
            lora_triggerwords_append_settings = config_array.get("lora_triggerwords_append_settings", {})

            # Extra Model & Sampling Options
            model_sampling_override = config_array.get("model_sampling_override", "none")
            model_sampling_shift = config_array.get("model_sampling_shift", "1.73")
            model_sampling_flux_max_shift = config_array.get("model_sampling_flux_max_shift", "1.15")
            model_sampling_flux_base_shift = config_array.get("model_sampling_flux_base_shift", "0.5")
            use_advanced_sampling = config_array.get("use_advanced_sampling", False)
            advanced_guider = config_array.get("advanced_guider", "cfg_guider")
            advanced_scheduler = config_array.get("advanced_scheduler", "basic")
            use_flux_guidance = config_array.get("use_flux_guidance", False)
            flux_guidance_value = config_array.get("flux_guidance_value", "3.5")

            # Process models - handle both object format {path, type} and legacy string format
            model_strings = []
            model_type = "checkpoint"  # default
            model_bypass_states = config_array.get("model_bypass_states", {})
            for m in models_raw:
                if isinstance(m, dict):
                    # Trust the user's type selection even if no file is picked yet
                    # (otherwise switching to LTX/diffusion_model type without yet
                    # selecting a file silently falls back to checkpoint).
                    if m.get("type"):
                        model_type = m.get("type", "checkpoint")
                    path = m.get("path", "")
                    if path and path != "None" and not model_bypass_states.get(path, False):
                        model_strings.append(str(path))
                elif isinstance(m, str) and m and m != "None" and not model_bypass_states.get(m, False):
                    model_strings.append(str(m))

            # Process loras for this config
            lora_strings = UltimateConfigBuilder.process_lora_array(config_array, actual_include_none)
            total_lora_configs += len(lora_strings)

            # Create ONE config for this array
            config = {
                "sampler": sampler_list if len(sampler_list) > 1 else sampler_list[0] if sampler_list else "euler",
                "scheduler": scheduler_list if len(scheduler_list) > 1 else scheduler_list[0] if scheduler_list else "normal",
                "steps": steps_list if len(steps_list) > 1 else steps_list[0] if steps_list else 20,
                "cfg": cfg_list if len(cfg_list) > 1 else cfg_list[0] if cfg_list else 7.0,
                "lora": lora_strings if len(lora_strings) > 1 else lora_strings[0] if lora_strings else "None",
                "model": model_strings if len(model_strings) > 1 else model_strings[0] if model_strings else "None"
            }

            # Always include all fields with defaults to prevent manifest data loss
            # when fields are toggled off in the UI and the session is reloaded
            seed_behavior = config_array.get("seed_behavior", "fixed")
            config["seed_behavior"] = seed_behavior

            # Full run seed behavior (applied before/after entire grid test session)
            full_run_seed_behavior = config_array.get("full_run_seed_behavior", "fixed")
            config["full_run_seed_behavior"] = full_run_seed_behavior

            # Full run seed (overrides node seed when > 0)
            full_run_seed = config_array.get("full_run_seed", 0)
            config["full_run_seed"] = int(full_run_seed) if full_run_seed else 0

            # Process VAEs — filter out bypassed (unchecked) entries AND any
            # empty / "None" placeholders. If nothing remains, OMIT the "vae"
            # key entirely so the orchestrator falls back to the default
            # behavior (no per-config VAE override). Emitting "vae": "None"
            # made the generator try to load a model file literally named
            # "None", which fails.
            vaes_raw = config_array.get("vaes", ["None"])
            vae_bypass_states = config_array.get("vae_bypass_states", {}) or {}
            vae_strings = [
                str(v) for v in vaes_raw
                if v and v != "None" and not vae_bypass_states.get(str(v), False)
            ]
            if vae_strings:
                config["vae"] = vae_strings if len(vae_strings) > 1 else vae_strings[0]

            # Always include model_type and related fields
            config["model_type"] = model_type
            if model_type != "checkpoint":
                text_encoders = config_array.get("text_encoders", [])
                if text_encoders:
                    config["text_encoders"] = [te for te in text_encoders if te and te != "None"]
                clip_type = config_array.get("clip_type", "")
                if clip_type:
                    config["clip_type"] = clip_type
                if model_type == "gguf":
                    gguf_options = config_array.get("gguf_options", {})
                    if gguf_options:
                        config["gguf_options"] = gguf_options
                if model_type == "ltx_video":
                    # Flatten configArray.ltx_video into top-level fields for the orchestrator.
                    # Mirrors the JS-side flattening in conf-builder-utilities.convertStateToConfigs.
                    ltx = config_array.get("ltx_video") or {}
                    if ltx.get("clip_models"):
                        config["clip_models"] = ltx["clip_models"]
                    if ltx.get("vae_video"):
                        config["vae_video"] = ltx["vae_video"]
                    if ltx.get("vae_audio"):
                        config["vae_audio"] = ltx["vae_audio"]
                    if ltx.get("latent_upscaler"):
                        config["latent_upscaler"] = ltx["latent_upscaler"]
                    if ltx.get("duration_seconds") is not None:
                        config["duration_seconds"] = ltx["duration_seconds"]
                    if ltx.get("frame_rate") is not None:
                        config["frame_rate"] = ltx["frame_rate"]
                    if ltx.get("sampler_stage1"):
                        config["sampler_stage1"] = ltx["sampler_stage1"]
                    if ltx.get("sampler_stage2"):
                        config["sampler_stage2"] = ltx["sampler_stage2"]
                    if ltx.get("sigmas_stage1"):
                        config["sigmas_stage1"] = ltx["sigmas_stage1"]
                    if ltx.get("sigmas_stage2"):
                        config["sigmas_stage2"] = ltx["sigmas_stage2"]
                    if ltx.get("input_image") is not None:
                        config["input_image"] = ltx["input_image"]
                    if ltx.get("image_strength_stage1") is not None:
                        config["image_strength_stage1"] = ltx["image_strength_stage1"]
                    if ltx.get("image_strength_stage2") is not None:
                        config["image_strength_stage2"] = ltx["image_strength_stage2"]
                    if ltx.get("img_compression") is not None:
                        config["img_compression"] = ltx["img_compression"]
                    if ltx.get("audio_mode"):
                        config["audio_mode"] = ltx["audio_mode"]

            # Always include omit triggers (empty list if none)
            config["lora_omit_triggers"] = omit_triggers if omit_triggers else []

            # Always include trigger append settings (empty dict if none)
            config["lora_triggerwords_append_settings"] = lora_triggerwords_append_settings if lora_triggerwords_append_settings else {}

            # Per-config resolutions (override sampler's resolutions_json)
            raw_resolutions = config_array.get("resolutions", [])
            if raw_resolutions and len(raw_resolutions) > 0:
                # Convert "WxH" strings to [W, H] arrays for config_utils.expand_configs()
                parsed_res = []
                for r in raw_resolutions:
                    if isinstance(r, str) and "x" in r:
                        parts = r.split("x")
                        parsed_res.append([int(parts[0]), int(parts[1])])
                    elif isinstance(r, (list, tuple)) and len(r) == 2:
                        parsed_res.append([int(r[0]), int(r[1])])
                if parsed_res:
                    config["resolutions"] = parsed_res

            # Attention mode(s) for testing different attention implementations
            attention_modes = config_array.get("attention_modes", ["default"])
            if isinstance(attention_modes, list):
                filtered = [a for a in attention_modes if a and a != "default"]
                if filtered:
                    config["attention_mode"] = filtered if len(filtered) > 1 else filtered[0]
                else:
                    config["attention_mode"] = "default"
            elif isinstance(attention_modes, str):
                config["attention_mode"] = attention_modes
            else:
                config["attention_mode"] = "default"

            # Model prompt prefix/suffix (quality tags prepended/appended to prompts)
            model_prompt_prefix = config_array.get("model_prompt_prefix", "")
            config["model_prompt_prefix"] = model_prompt_prefix.strip() if model_prompt_prefix else ""
            model_prompt_suffix = config_array.get("model_prompt_suffix", "")
            config["model_prompt_suffix"] = model_prompt_suffix.strip() if model_prompt_suffix else ""

            # Always include model sampling options with defaults
            config["model_sampling_override"] = model_sampling_override if model_sampling_override else "none"
            if model_sampling_override and model_sampling_override != "none":
                if model_sampling_override == "flux":
                    config["model_sampling_flux_max_shift"] = model_sampling_flux_max_shift
                    config["model_sampling_flux_base_shift"] = model_sampling_flux_base_shift
                elif model_sampling_override == "flux2":
                    config["model_sampling_shift"] = model_sampling_shift if model_sampling_shift else "2.02"
                else:
                    config["model_sampling_shift"] = model_sampling_shift
            config["use_advanced_sampling"] = use_advanced_sampling or False
            if use_advanced_sampling:
                config["advanced_guider"] = advanced_guider
                config["advanced_scheduler"] = advanced_scheduler
            config["use_flux_guidance"] = use_flux_guidance or False
            if use_flux_guidance:
                config["flux_guidance_value"] = flux_guidance_value

            # Deep Shrink (Kohya / PatchModelAddDownscale) — patches the UNet
            # to downscale features at a specific block during early diffusion.
            # Only emit detail params when toggle is on, to keep configs_json clean.
            use_deep_shrink = config_array.get("use_deep_shrink", False)
            config["use_deep_shrink"] = bool(use_deep_shrink)
            if use_deep_shrink:
                config["deep_shrink_block_number"] = int(config_array.get("deep_shrink_block_number", 3))
                config["deep_shrink_downscale_factor"] = float(config_array.get("deep_shrink_downscale_factor", 2.0))
                config["deep_shrink_start_percent"] = float(config_array.get("deep_shrink_start_percent", 0.0))
                config["deep_shrink_end_percent"] = float(config_array.get("deep_shrink_end_percent", 0.35))
                config["deep_shrink_downscale_after_skip"] = bool(config_array.get("deep_shrink_downscale_after_skip", True))
                config["deep_shrink_downscale_method"] = str(config_array.get("deep_shrink_downscale_method", "bicubic"))
                config["deep_shrink_upscale_method"] = str(config_array.get("deep_shrink_upscale_method", "bicubic"))

            # ==== PROMPT HANDLING ====
            # Priority: per-config > global > node inputs (omitted = use node inputs)
            use_custom = config_array.get("use_custom_prompts", False)
            per_config_positive_groups = config_array.get("positive_prompt_groups", [])
            per_config_negative = config_array.get("negative_prompt", "")

            if use_custom and per_config_positive_groups:
                # Per-config prompts override everything
                # Store as nested array format for parse_prompt_input_nested() compatibility
                config["positive"] = per_config_positive_groups
                if per_config_negative:
                    config["negative"] = per_config_negative
                config["_prompt_source"] = "custom"
            elif global_positive_groups:
                # Global prompts override node inputs
                config["positive"] = global_positive_groups
                if global_negative:
                    config["negative"] = global_negative
                config["_prompt_source"] = "global"
            # If neither, omit "positive"/"negative" keys - node inputs will be used as fallback

            # --- CONFIG SCHEMA VALIDATION ---
            # Sanity check for typos / dropped fields. Only includes keys this
            # function ALWAYS sets unconditionally — fields that are filled in
            # later by the orchestrator (seed, denoise) or only set for
            # specific model types (clip_type, text_encoders, gguf_options,
            # ltx_*) are intentionally excluded so they don't fire false alarms.
            _EXPECTED_CONFIG_KEYS = {
                "sampler", "scheduler", "steps", "cfg", "seed_behavior",
                "model", "model_type", "lora", "vae",
                "model_sampling_override", "use_advanced_sampling", "use_flux_guidance",
                "model_prompt_prefix", "model_prompt_suffix", "attention_mode",
                "use_deep_shrink",
            }
            missing = _EXPECTED_CONFIG_KEYS - set(config.keys())
            if missing:
                print(f"[ConfigBuilder] ⚠️ Config schema drift: missing keys {missing} in config for '{config.get('model', '?')}'")

            configs_output.append(config)

        # Build the output object with configs and optional distribution settings
        output_obj = {"configs": configs_output}

        # Embed distribution config if enabled
        if state.get("distribution_enabled") and state.get("worker_urls"):
            output_obj["_distribution"] = {
                "enabled": True,
                "worker_urls": [u for u in state["worker_urls"] if u and u.strip()],
                "claim_timeout": state.get("claim_timeout", 600),
                "use_master_encoding": state.get("use_master_encoding", False),
                "sync_models_to_workers": state.get("sync_models_to_workers", False)
            }

        # Embed session-level settings (upscaling, cooldown) if enabled
        session_settings = {}
        upscaling_data = state.get("upscaling", {})
        if upscaling_data and upscaling_data.get("enabled", False):
            # Filter out inactive pipelines and inactive steps within pipelines
            pipelines = upscaling_data.get("pipelines", [])
            active_pipelines = []
            for p in pipelines:
                if p.get("active", True) is False:
                    continue
                active_steps = [s for s in p.get("steps", []) if s.get("active", True) is not False]
                if active_steps:
                    active_pipelines.append({**p, "steps": active_steps})
            if active_pipelines:
                session_settings["upscaling"] = {
                    "enabled": True,
                    "save_pre_upscale": upscaling_data.get("save_pre_upscale", False),
                    "run_upscales_at_end": upscaling_data.get("run_upscales_at_end", False),
                    "hires_prompt_adjust": upscaling_data.get("hires_prompt_adjust", False),
                    "hires_prompt_behavior": upscaling_data.get("hires_prompt_behavior", "append_end"),
                    "hires_prompt_text": upscaling_data.get("hires_prompt_text", ""),
                    "pipelines": active_pipelines
                }
        cooldown_data = state.get("cooldown", {})
        if cooldown_data and cooldown_data.get("enabled", False):
            session_settings["cooldown"] = cooldown_data
        # Start At Job # (skip to a specific job number) — always emit.
        try:
            session_settings["start_at_job"] = int(state.get("start_at_job", 0))
        except (TypeError, ValueError):
            session_settings["start_at_job"] = 0
        # Image save format — always emit so Builder UI is authoritative.
        session_settings["image_format"] = str(state.get("image_format", "webp"))
        # Builder UI is authoritative for all run settings — emit every field.
        # Type coercion is defensive; the Builder UI sends correct types but
        # workflows saved before the Builder UI had these fields might have
        # missing or string-typed values that need normalization.
        session_settings["overwrite_existing"] = bool(state.get("overwrite_existing", False))
        try:
            session_settings["flush_batch_every"] = int(state.get("flush_batch_every", 1))
        except (TypeError, ValueError):
            session_settings["flush_batch_every"] = 1
        session_settings["lora_triggerwords_mode"] = str(state.get("lora_triggerwords_mode", "None"))
        session_settings["save_conditioning_cache_to_file"] = bool(state.get("save_conditioning_cache_to_file", False))
        session_settings["enable_model_cache"] = bool(state.get("enable_model_cache", False))
        try:
            session_settings["vae_batch_size"] = int(state.get("vae_batch_size", 1))
        except (TypeError, ValueError):
            session_settings["vae_batch_size"] = 1

        # session_name moves into session_settings too so the Generator's
        # session_name widget can be removed (Phase 2).
        session_settings["session_name"] = str(state.get("session_name", "my_session"))
        try:
            session_settings["add_random_seeds_to_gens"] = int(state.get("add_random_seeds_to_gens", 0))
        except (TypeError, ValueError):
            session_settings["add_random_seeds_to_gens"] = 0

        if session_settings:
            output_obj["_session_settings"] = session_settings

        json_output = json.dumps(output_obj, indent=2, ensure_ascii=False)
        return json_output

    def generate_config(
        self,
        session_name,
        load_session,
        samplers,
        schedulers,
        steps,
        cfg,
        lora_config,
        include_none,
        model=""
    ):
        """
        Generate configuration.

        NOTE: All widget parameters are IGNORED!
        The actual data comes from the lora_config widget which contains everything.
        """
        # Use Windows-safe print for all logging in this function. Prevents
        # OSError [Errno 22] from colorama/stdout corruption crashing the node.
        print = safe_print

        print(f"\n{'='*80}")
        print(f"[ConfigBuilder] 🎯 Generating Configuration")
        print(f"{'='*80}")

        # Parse the COMPLETE state from lora_config widget
        try:
            state = json.loads(lora_config)
        except json.JSONDecodeError as e:
            print(f"[ConfigBuilder] ⚠️ Error parsing lora_config: {e}")
            print(f"[ConfigBuilder] Using default config")
            state = json.loads(self.get_default_config())

        actual_session_name = state.get("session_name", session_name)

        # ============================================================================
        # SINGLE SOURCE OF TRUTH: state_to_configs_json is also called by
        # POST /configbuilder/preview, so the preview cannot disagree with this.
        # ============================================================================
        json_output = UltimateConfigBuilder.state_to_configs_json(state)

        # Brief summary log (computed from the helper's output)
        try:
            parsed = json.loads(json_output)
            n_configs = len(parsed.get("configs", []))
        except Exception:
            n_configs = 0
        print(f"[ConfigBuilder] 📊 Session: {actual_session_name}")
        print(f"[ConfigBuilder] 📊 Configs: {n_configs}")
        print(f"{'='*80}\n")

        return (json_output,)


# API endpoint for trigger word lookup
@server.PromptServer.instance.routes.post("/configbuilder/lookup_triggers")
async def lookup_triggers_endpoint(request):
    """API endpoint to lookup trigger words for LoRAs"""
    try:
        from . import civitai as _civitai  # local import keeps top of file clean
        data = await request.json()
        lora_list = data.get("loras", [])

        print(f"[ConfigBuilder] 🔍 Lookup request for {len(lora_list)} LoRAs")

        trigger_map = UltimateConfigBuilder.lookup_lora_triggers(lora_list)

        print(f"[ConfigBuilder] ✅ Found triggers for {len(trigger_map)} LoRAs")

        return web.json_response({
            "triggers": trigger_map,
            "civitai_available": _civitai.is_civitai_available(),
        })
    except Exception as e:
        print(f"[ConfigBuilder] ❌ Error in lookup_triggers endpoint: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({
            "error": str(e)
        }, status=500)


# API endpoint for detailed LoRA metadata lookup
@server.PromptServer.instance.routes.post("/configbuilder/lookup_lora_metadata")
async def lookup_lora_metadata_endpoint(request):
    """API endpoint to lookup full metadata for a specific LoRA from CivitAI"""
    try:
        data = await request.json()
        lora_name = data.get("lora_name", "")
        force_refresh = data.get("force_refresh", False)

        if not lora_name:
            return web.json_response({
                "error": "No LoRA name provided"
            }, status=400)
        
        print(f"[ConfigBuilder] 🔍 Full metadata lookup request for: {lora_name}")

        # Check disk cache first to avoid expensive SHA256 hashing
        output_dir = folder_paths.get_output_directory()
        model_data_dir = os.path.join(output_dir, "benchmarks", "model-data", lora_name.replace("/", "_").replace("\\", "_").replace(".safetensors", ""))
        metadata_file = os.path.join(model_data_dir, "metadata.json")

        if os.path.exists(metadata_file) and not force_refresh:
            try:
                cached = load_json_from_file(metadata_file)
                if cached and cached.get("name"):
                    # Get file modification time for cache date display
                    cache_mtime = os.path.getmtime(metadata_file)
                    cache_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cache_mtime))
                    print(f"[ConfigBuilder] ✅ Using cached metadata for: {lora_name} (cached on {cache_date})")
                    return web.json_response({
                        "metadata": cached,
                        "saved_to": metadata_file,
                        "cached": True,
                        "cache_date": cache_date
                    })
            except Exception:
                pass  # Cache miss or corrupt file - fall through to fresh lookup

        # Get the full path to the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)

        if lora_path is None:
            return web.json_response({
                "error": f"LoRA file not found: {lora_name}"
            }, status=404)

        # Calculate the hash (expensive - only when not cached)
        lora_hash = calculate_sha256(lora_path)
        short_hash = lora_hash[:10]  # First 10 characters for short hash

        print(f"[ConfigBuilder] 📊 Hash calculated: {short_hash}")

        # Fetch metadata from CivitAI
        model_info = get_model_version_info(lora_hash)
        
        if model_info is None:
            return web.json_response({
                "error": "No metadata found on CivitAI",
                "hash": lora_hash,
                "short_hash": short_hash
            }, status=404)
        
        # Extract relevant information
        metadata = {
            "name": model_info.get("name", "Unknown"),
            "model_name": model_info.get("model", {}).get("name", "Unknown") if isinstance(model_info.get("model"), dict) else "Unknown",
            "trained_words": model_info.get("trainedWords", []),
            "base_model": model_info.get("baseModel", "Unknown"),
            "description": model_info.get("description", ""),
            "tags": model_info.get("model", {}).get("tags", []) if isinstance(model_info.get("model"), dict) else [],
            "images": [],
            "url": f"https://civitai.com/models/{model_info.get('modelId', '')}" if model_info.get("modelId") else "",
            "hash": lora_hash,
            "short_hash": short_hash,
            "file_path": lora_path,
            "stats": model_info.get("stats", {}),
            "creator": model_info.get("creator", {}).get("username", "Unknown") if isinstance(model_info.get("creator"), dict) else "Unknown"
        }
        
        # Extract images
        if "images" in model_info and isinstance(model_info["images"], list):
            for img in model_info["images"][:5]:  # Limit to first 5 images
                if isinstance(img, dict) and "url" in img:
                    metadata["images"].append({
                        "url": img["url"],
                        "nsfw": img.get("nsfw", "None"),
                        "width": img.get("width", 0),
                        "height": img.get("height", 0)
                    })
        
        # Save metadata to file
        output_dir = folder_paths.get_output_directory()
        model_data_dir = os.path.join(output_dir, "benchmarks", "model-data", lora_name.replace("/", "_").replace("\\", "_").replace(".safetensors", ""))
        os.makedirs(model_data_dir, exist_ok=True)
        
        metadata_file = os.path.join(model_data_dir, "metadata.json")
        save_dict_to_json(metadata, metadata_file)
        
        print(f"[ConfigBuilder] ✅ Metadata saved to: {metadata_file}")
        
        # Compare fresh tags with stored loras_tags.json (only on force_refresh)
        tags_changed = False
        old_tags = []
        new_tags = metadata.get("trained_words", [])
        if force_refresh and new_tags:
            json_tags_path = os.path.join(output_dir, "benchmarks/loras_tags.json")
            if os.path.exists(json_tags_path):
                lora_tags = load_json_from_file(json_tags_path) or {}
                normalized = lora_name.replace("\\", "/")
                backslash = lora_name.replace("/", "\\")
                old_tags = lora_tags.get(lora_name, lora_tags.get(normalized, lora_tags.get(backslash, [])))
                if old_tags is None:
                    old_tags = []
                # Compare sorted lists to detect any difference
                if sorted(old_tags) != sorted(new_tags):
                    tags_changed = True
                    print(f"[ConfigBuilder] ⚠️ Tags changed for {lora_name}: {old_tags} -> {new_tags}")

        return web.json_response({
            "metadata": metadata,
            "saved_to": metadata_file,
            "tags_changed": tags_changed,
            "old_tags": old_tags,
            "new_tags": new_tags
        })

    except Exception as e:
        print(f"[ConfigBuilder] ❌ Error in lookup_lora_metadata endpoint: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({
            "error": str(e)
        }, status=500)


# API endpoint for detailed Model/Checkpoint metadata lookup
@server.PromptServer.instance.routes.post("/configbuilder/lookup_model_metadata")
async def lookup_model_metadata_endpoint(request):
    """API endpoint to lookup full metadata for a model/checkpoint from CivitAI"""
    try:
        data = await request.json()
        model_name = data.get("model_name", "")
        model_type = data.get("model_type", "checkpoint")
        force_refresh = data.get("force_refresh", False)

        if not model_name:
            return web.json_response({
                "error": "No model name provided"
            }, status=400)

        print(f"[ConfigBuilder] 🔍 Full metadata lookup request for model: {model_name} (type: {model_type})")

        # Check disk cache first to avoid expensive SHA256 hashing
        output_dir = folder_paths.get_output_directory()
        model_data_dir = os.path.join(output_dir, "benchmarks", "model-data", model_name.replace("/", "_").replace("\\", "_").replace(".safetensors", "").replace(".ckpt", "").replace(".gguf", ""))
        metadata_file = os.path.join(model_data_dir, "metadata.json")

        if os.path.exists(metadata_file) and not force_refresh:
            try:
                cached = load_json_from_file(metadata_file)
                if cached and cached.get("name"):
                    # Get file modification time for cache date display
                    cache_mtime = os.path.getmtime(metadata_file)
                    cache_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cache_mtime))
                    print(f"[ConfigBuilder] ✅ Using cached metadata for model: {model_name} (cached on {cache_date})")
                    return web.json_response({
                        "metadata": cached,
                        "saved_to": metadata_file,
                        "cached": True,
                        "cache_date": cache_date
                    })
            except Exception:
                pass  # Cache miss or corrupt file - fall through to fresh lookup

        # Resolve the full path based on model type
        model_path = None
        if model_type == "gguf":
            try:
                model_path = folder_paths.get_full_path("unet_gguf", model_name)
            except (KeyError, Exception):
                pass
            if model_path is None:
                model_path = folder_paths.get_full_path("diffusion_models", model_name)
        elif model_type == "diffusion_model":
            model_path = folder_paths.get_full_path("diffusion_models", model_name)
        else:
            model_path = folder_paths.get_full_path("checkpoints", model_name)

        if model_path is None:
            return web.json_response({
                "error": f"Model file not found: {model_name}"
            }, status=404)

        # Calculate the hash (expensive - only when not cached)
        model_hash = calculate_sha256(model_path)
        short_hash = model_hash[:10]

        print(f"[ConfigBuilder] 📊 Hash calculated: {short_hash}")

        # Fetch metadata from CivitAI
        model_info = get_model_version_info(model_hash)

        if model_info is None:
            return web.json_response({
                "error": "No metadata found on CivitAI",
                "hash": model_hash,
                "short_hash": short_hash
            }, status=404)

        # Extract relevant information
        metadata = {
            "name": model_info.get("name", "Unknown"),
            "model_name": model_info.get("model", {}).get("name", "Unknown") if isinstance(model_info.get("model"), dict) else "Unknown",
            "trained_words": model_info.get("trainedWords", []),
            "base_model": model_info.get("baseModel", "Unknown"),
            "description": model_info.get("description", ""),
            "tags": model_info.get("model", {}).get("tags", []) if isinstance(model_info.get("model"), dict) else [],
            "images": [],
            "url": f"https://civitai.com/models/{model_info.get('modelId', '')}" if model_info.get("modelId") else "",
            "hash": model_hash,
            "short_hash": short_hash,
            "file_path": model_path,
            "stats": model_info.get("stats", {}),
            "creator": model_info.get("creator", {}).get("username", "Unknown") if isinstance(model_info.get("creator"), dict) else "Unknown"
        }

        # Extract images
        if "images" in model_info and isinstance(model_info["images"], list):
            for img in model_info["images"][:5]:
                if isinstance(img, dict) and "url" in img:
                    metadata["images"].append({
                        "url": img["url"],
                        "nsfw": img.get("nsfw", "None"),
                        "width": img.get("width", 0),
                        "height": img.get("height", 0)
                    })

        # Save metadata to file
        output_dir = folder_paths.get_output_directory()
        model_data_dir = os.path.join(output_dir, "benchmarks", "model-data", model_name.replace("/", "_").replace("\\", "_").replace(".safetensors", "").replace(".ckpt", "").replace(".gguf", ""))
        os.makedirs(model_data_dir, exist_ok=True)

        metadata_file = os.path.join(model_data_dir, "metadata.json")
        save_dict_to_json(metadata, metadata_file)

        print(f"[ConfigBuilder] ✅ Model metadata saved to: {metadata_file}")

        return web.json_response({
            "metadata": metadata,
            "saved_to": metadata_file
        })

    except Exception as e:
        print(f"[ConfigBuilder] ❌ Error in lookup_model_metadata endpoint: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({
            "error": str(e)
        }, status=500)


# Fast count endpoint — returns just file counts per category for change detection
@server.PromptServer.instance.routes.get("/configbuilder/model_counts")
async def get_model_counts_endpoint(request):
    """Return file counts per model category. Very fast — uses folder_paths internal cache."""
    try:
        counts = {}
        for cat in ["checkpoints", "diffusion_models", "text_encoders", "vae", "loras"]:
            try:
                counts[cat] = len(folder_paths.get_filename_list(cat))
            except (KeyError, Exception):
                counts[cat] = 0
        for cat in ["unet_gguf", "clip_gguf", "upscale_models"]:
            try:
                counts[cat] = len(folder_paths.get_filename_list(cat))
            except (KeyError, Exception):
                counts[cat] = 0
        return web.json_response(counts)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

# API endpoint to get all model lists for unified model selector
@server.PromptServer.instance.routes.get("/configbuilder/model_lists")
async def get_model_lists_endpoint(request):
    """Return all model lists for the config builder's unified model selector."""
    try:
        checkpoints = folder_paths.get_filename_list("checkpoints")
        diffusion_models = folder_paths.get_filename_list("diffusion_models")
        text_encoders = folder_paths.get_filename_list("text_encoders")
        try:
            clip_models = folder_paths.get_filename_list("clip")
        except (KeyError, Exception):
            clip_models = []
        # GGUF lists - may not exist if ComfyUI-GGUF is not installed
        try:
            unet_gguf = folder_paths.get_filename_list("unet_gguf")
        except (KeyError, Exception):
            unet_gguf = []
        try:
            clip_gguf = folder_paths.get_filename_list("clip_gguf")
        except (KeyError, Exception):
            clip_gguf = []

        # CLIP type options: prefer ComfyUI runtime node definitions, fallback to known defaults.
        clip_types = [
            "stable_diffusion", "stable_cascade", "sd3", "stable_audio",
            "mochi", "ltxv", "pixart", "cosmos", "lumina2", "krea2", "wan",
            "hidream", "chroma", "ace", "flux", "flux2"
        ]
        dual_clip_types = [
            "sdxl", "sd3", "flux", "flux2", "hunyuan_video", "hidream",
            "hunyuan_image", "hunyuan_video_15"
        ]
        try:
            import nodes

            def _extract_type_options(node_name):
                node_cls = nodes.NODE_CLASS_MAPPINGS.get(node_name)
                if not node_cls or not hasattr(node_cls, "INPUT_TYPES"):
                    return None
                input_types = node_cls.INPUT_TYPES() or {}
                required = input_types.get("required") or {}
                type_spec = required.get("type")
                if isinstance(type_spec, (list, tuple)) and len(type_spec) > 0:
                    options = type_spec[0]
                    if isinstance(options, (list, tuple)):
                        return [str(x) for x in options]
                return None

            runtime_clip_types = _extract_type_options("CLIPLoader")
            runtime_dual_clip_types = _extract_type_options("DualCLIPLoader")
            if runtime_clip_types:
                clip_types = runtime_clip_types
            if runtime_dual_clip_types:
                dual_clip_types = runtime_dual_clip_types
        except Exception:
            pass

        # Upscale model list
        try:
            upscale_models = folder_paths.get_filename_list("upscale_models")
        except (KeyError, Exception):
            upscale_models = []

        # Latent upscale model list (separate folder; used by LatentUpscaleModelLoader,
        # e.g. LTX 2.3 spatial upscaler)
        try:
            latent_upscale_models = folder_paths.get_filename_list("latent_upscale_models")
        except (KeyError, Exception):
            latent_upscale_models = []

        # VAE list
        vae_list = folder_paths.get_filename_list("vae")

        # Sampler and scheduler lists from ComfyUI core
        import comfy.samplers
        sampler_names = list(comfy.samplers.KSampler.SAMPLERS)
        scheduler_names = list(comfy.samplers.KSampler.SCHEDULERS)

        return web.json_response({
            "checkpoints": checkpoints,
            "diffusion_models": diffusion_models,
            "unet_gguf": unet_gguf,
            "text_encoders": text_encoders,
            "clip": clip_models,
            "clip_gguf": clip_gguf,
            "clip_types": clip_types,
            "dual_clip_types": dual_clip_types,
            "vae": vae_list,
            "upscale_models": upscale_models,
            "latent_upscale_models": latent_upscale_models,
            "samplers": sampler_names,
            "schedulers": scheduler_names
        })
    except Exception as e:
        print(f"[ConfigBuilder] Error in model_lists endpoint: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


# API endpoint to refresh model/lora lists (triggered by "Update Node Definitions")
@server.PromptServer.instance.routes.post("/configbuilder/refresh_models")
async def refresh_models_endpoint(request):
    """
    API endpoint to signal frontend to clear its caches.
    Called when ComfyUI updates node definitions.
    """
    try:
        print(f"[ConfigBuilder] 🔄 Refresh signal received - clearing frontend caches")
        
        return web.json_response({
            "status": "ok",
            "message": "Frontend caches should be cleared"
        })
    except Exception as e:
        print(f"[ConfigBuilder] ❌ Error in refresh_models endpoint: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({ 
            "error": str(e)
        }, status=500)


@server.PromptServer.instance.routes.get("/configbuilder/get_lora_triggers")
async def get_lora_triggers_endpoint(request):
    """Get trigger words for a specific LoRA from loras_tags.json"""
    try:
        lora_name = request.query.get("lora_name", "")
        if not lora_name:
            return web.json_response({"error": "Missing lora_name"}, status=400)

        json_tags_path = os.path.join(folder_paths.get_output_directory(), "benchmarks/loras_tags.json")
        triggers = []
        if os.path.exists(json_tags_path):
            from .lora_utils import load_json_from_file
            lora_tags = load_json_from_file(json_tags_path) or {}
            # Try exact match, normalized, and backslash variants
            normalized = lora_name.replace("\\", "/")
            backslash = lora_name.replace("/", "\\")
            triggers = lora_tags.get(lora_name, lora_tags.get(normalized, lora_tags.get(backslash, [])))

        return web.json_response({"lora_name": lora_name, "triggers": triggers})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/configbuilder/save_lora_triggers")
async def save_lora_triggers_endpoint(request):
    """Save edited trigger words for a LoRA to loras_tags.json"""
    try:
        data = await request.json()
        lora_name = data.get("lora_name", "")
        triggers = data.get("triggers", [])

        if not lora_name:
            return web.json_response({"error": "Missing lora_name"}, status=400)

        json_tags_path = os.path.join(folder_paths.get_output_directory(), "benchmarks/loras_tags.json")
        from .lora_utils import load_json_from_file, save_dict_to_json
        lora_tags = {}
        if os.path.exists(json_tags_path):
            lora_tags = load_json_from_file(json_tags_path) or {}

        # Normalize the name for consistent storage
        normalized = lora_name.replace("\\", "/")
        lora_tags[normalized] = triggers

        save_dict_to_json(lora_tags, json_tags_path)

        # Clear the trigger word LRU cache so changes take effect immediately
        from .trigger_words import clear_trigger_caches
        clear_trigger_caches()

        print(f"[ConfigBuilder] ✏️ Saved {len(triggers)} trigger words for: {normalized}")
        return web.json_response({"status": "saved", "lora_name": normalized, "triggers": triggers})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/configbuilder/preview")
async def preview_endpoint(request):
    """
    Preview endpoint — runs state_to_configs_json on the POSTed builder UI
    state and returns the resulting configs_json string. The Builder UI's
    JSON Preview panel calls this. Single source of truth: this endpoint and
    generate_config() share state_to_configs_json, so they cannot disagree.
    """
    try:
        body = await request.json()
        state = body.get("state")
        if not isinstance(state, dict):
            return web.json_response(
                {"error": "Missing or invalid 'state' object in request body"},
                status=400,
            )
        configs_json = UltimateConfigBuilder.state_to_configs_json(state)
        return web.json_response({"configs_json": configs_json})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)