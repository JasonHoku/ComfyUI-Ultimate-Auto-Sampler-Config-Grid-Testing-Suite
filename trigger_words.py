"""
LoRA Trigger Word Management
Handles fetching, filtering, and processing of LoRA trigger words from Civitai
"""

import re
from functools import lru_cache
from .lora_utils import load_and_save_tags
from .config_utils import parse_lora_definition
from .regional_prompt import join_prompt_parts


def _should_omit_trigger(trigger, omit_normalized):
    """
    Check if a trigger word should be omitted using token-based word boundary matching.
    Uses regex word boundaries so "blue" omits "blue" but not "blueberry".
    Falls back to exact match for multi-word omit terms.

    Args:
        trigger: The cleaned trigger string (lowercase)
        omit_normalized: List of normalized omit terms (lowercase, stripped)

    Returns:
        True if the trigger should be omitted
    """
    for omit_term in omit_normalized:
        # Use word boundary regex: ensures "blue" matches "blue" but not "blueberry"
        if re.search(r'\b' + re.escape(omit_term) + r'\b', trigger, re.IGNORECASE):
            return True
    return False


@lru_cache(maxsize=128)
def _get_filtered_lora_triggers_cached(lora_string, omit_tuple):
    """
    Cached internal implementation. Takes hashable args (tuple instead of list).
    """
    active_loras = parse_lora_definition(lora_string)
    trigger_list = []

    # Normalize omit list: strip whitespace and trailing commas, lowercase
    omit_normalized = [str(t).lower().strip().rstrip(',').strip() for t in omit_tuple]

    for lora_def in active_loras:
        lname, lstr_m, lstr_c = lora_def
        try:
            civitai_tags_list = load_and_save_tags(lname, force_fetch=False)
            if len(civitai_tags_list) > 0:
                for tags in civitai_tags_list:
                    # Clean the trigger: strip whitespace and trailing commas
                    cleaned_tags = tags.strip().rstrip(',').strip()

                    # Check if this trigger should be omitted (token-based word boundary matching)
                    if not _should_omit_trigger(cleaned_tags.lower(), omit_normalized):
                        trigger_list.append(cleaned_tags)
        except Exception as e:
            pass

    return trigger_list


def get_filtered_lora_triggers(lora_string, omit_list, lookup_triggers=True):
    """
    Get LoRA trigger words with filtering applied.

    Args:
        lora_string: LoRA definition string (e.g., "lora1.safetensors:0.8:0.6 + lora2.safetensors:1.0:1.0")
        omit_list: List of trigger words to omit
        lookup_triggers: Whether to lookup triggers from Civitai

    Returns:
        List of filtered trigger words
    """
    if not lookup_triggers or lora_string == "None":
        return []

    # Convert list to tuple for hashable cache key
    return list(_get_filtered_lora_triggers_cached(lora_string, tuple(omit_list)))


def get_trigger_placement_for_lora(lora_name, config, lora_triggerwords_mode):
    """
    Determine where to place trigger words for a specific LoRA.
    
    Args:
        lora_name: The LoRA filename (e.g., "XL/Tmp-Trendy/lora1.safetensors")
        config: Configuration dictionary
        lora_triggerwords_mode: The global mode (None, Append To End, Append To Start, Read From Config)
        
    Returns:
        str: "start", "end", or "none"
    """
    if lora_triggerwords_mode == "None":
        return "none"
    elif lora_triggerwords_mode == "Append To End":
        return "end"
    elif lora_triggerwords_mode == "Append To Start":
        return "start"
    elif lora_triggerwords_mode == "Read From Config":
        # Check if config has lora_triggerwords_append_settings
        settings = config.get("lora_triggerwords_append_settings", {})
        
        # Normalize the lora_name path (handle both / and \)
        normalized_lora = lora_name.replace('\\', '/')
        
        # print(f"[DEBUG] lora_name: {lora_name}")
        # print(f"[DEBUG] normalized_lora: {normalized_lora}")
        # print(f"[DEBUG] settings: {settings}")
        
        # Try exact match first (with original path)
        if lora_name in settings:
            placement = settings[lora_name].lower()
            if placement in ["start", "end"]:
                # print(f"[DEBUG] Exact match (original): {placement}")
                return placement
        
        # Try exact match with normalized path
        if normalized_lora != lora_name and normalized_lora in settings:
            placement = settings[normalized_lora].lower()
            if placement in ["start", "end"]:
                # print(f"[DEBUG] Exact match (normalized): {placement}")
                return placement
        
        # Try matching without extension (normalized)
        lora_base = normalized_lora.replace('.safetensors', '').replace('.ckpt', '').replace('.pt', '')
        if lora_base in settings:
            placement = settings[lora_base].lower()
            if placement in ["start", "end"]:
                # print(f"[DEBUG] Base name match: {placement}")
                return placement
        
        # Try matching against folders (check if lora is in any configured folder)
        for setting_key, placement in settings.items():
            # Normalize the setting key as well
            normalized_key = setting_key.replace('\\', '/')
            
            # print(f"[DEBUG] Checking folder: '{normalized_key}' vs '{normalized_lora}'")
            
            # Check if this is a folder setting (ends with /)
            if normalized_key.endswith('/'):
                # Check if the normalized lora path starts with this normalized folder path
                if normalized_lora.startswith(normalized_key):
                    placement_lower = placement.lower()
                    if placement_lower in ["start", "end"]:
                        # print(f"[DEBUG] Folder match! '{normalized_lora}' starts with '{normalized_key}' -> {placement_lower}")
                        return placement_lower
        
        # Default to end if not specified in config
        # print(f"[DEBUG] No match found, defaulting to 'end'")
        return "end"
    
    return "none"


def collect_unique_prompts_with_triggers(expanded_configs, lora_triggerwords_mode):
    """
    Collect all unique prompts from configs, applying LoRA trigger words where needed.
    
    Args:
        expanded_configs: List of expanded configuration dictionaries
        lora_triggerwords_mode: Mode for trigger word placement
        
    Returns:
        tuple: (unique_positives set, unique_negatives set)
    """
    unique_positives = set()
    unique_negatives = set()
    
    for conf in expanded_configs:
        full_positive = conf["positive"]

        if lora_triggerwords_mode != "None" and conf["lora"] != "None":
            omit_list = conf.get("lora_omit_triggers", [])
            trigger_list = get_filtered_lora_triggers(
                conf["lora"],
                omit_list,
                lookup_triggers=True
            )

            if not trigger_list:
                print(f"[GridTester] ℹ️ No trigger words found for LoRA: {conf['lora'][:60]}")

            if trigger_list:
                # Parse the lora string to get individual loras
                active_loras = parse_lora_definition(conf["lora"])

                # Separate triggers by placement
                start_triggers = []
                end_triggers = []

                for lora_def in active_loras:
                    lname, _, _ = lora_def
                    placement = get_trigger_placement_for_lora(lname, conf, lora_triggerwords_mode)

                    # Get triggers specific to this lora
                    try:
                        lora_triggers = load_and_save_tags(lname, force_fetch=False)
                        omit_normalized = [str(t).lower().strip().rstrip(',').strip() for t in omit_list]

                        for tag in lora_triggers:
                            cleaned_tag = tag.strip().rstrip(',').strip()
                            if not _should_omit_trigger(cleaned_tag.lower(), omit_normalized):
                                if placement == "start":
                                    start_triggers.append(cleaned_tag)
                                elif placement == "end":
                                    end_triggers.append(cleaned_tag)
                    except:
                        pass

                # Build the full prompt with triggers in correct positions
                parts = []
                if start_triggers:
                    parts.append(", ".join(start_triggers))
                parts.append(conf['positive'])
                if end_triggers:
                    parts.append(", ".join(end_triggers))

                full_positive = join_prompt_parts(parts, 1 if start_triggers else 0)

                # Show omitted triggers only once during pre-encoding
                if omit_list:
                    all_triggers = []
                    active_loras = parse_lora_definition(conf["lora"])
                    for lora_def in active_loras:
                        lname, _, _ = lora_def
                        try:
                            tags = load_and_save_tags(lname, force_fetch=False)
                            all_triggers.extend([t.strip().rstrip(',').strip() for t in tags])
                        except:
                            pass

                    omitted = set(all_triggers) - set(start_triggers + end_triggers)
                    if omitted:
                        print(f"[GridTester] 🚫 Omitted triggers: {', '.join(omitted)}")

        # Apply model-specific prompt prefix/suffix (wraps entire prompt+triggers)
        full_positive = _apply_model_prompt_affixes(full_positive, conf)

        unique_positives.add(full_positive)
        unique_negatives.add(conf["negative"])
    
    return unique_positives, unique_negatives


_build_prompt_cache = {}


def clear_trigger_caches():
    """
    Clear all trigger word caches. Must be called at the start of each generation run
    to ensure trigger words are re-read from loras_tags.json (which may have been updated
    by a previous run's CivitAI fetch or manual edit).

    Without this, @lru_cache on _get_filtered_lora_triggers_cached and the module-level
    _build_prompt_cache dict persist stale results across runs within the same ComfyUI session.
    """
    global _build_prompt_cache
    _build_prompt_cache.clear()
    _get_filtered_lora_triggers_cached.cache_clear()
    print("[GridTester] 🔄 Trigger word caches cleared")


def _apply_model_prompt_affixes(prompt, config):
    """
    Apply model-specific prompt prefix and suffix to a prompt string.

    These are quality tags that should always wrap the prompt for specific model families,
    e.g. "score_9, score_8_up" for Pony or "masterpiece, best quality" for SD1.5.

    Args:
        prompt: The assembled prompt string (may already include LoRA triggers)
        config: Configuration dictionary with optional model_prompt_prefix/suffix

    Returns:
        The prompt with prefix/suffix applied
    """
    prefix = config.get("model_prompt_prefix", "").strip()
    suffix = config.get("model_prompt_suffix", "").strip()

    if not prefix and not suffix:
        return prompt

    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(prompt)
    if suffix:
        parts.append(suffix)

    return join_prompt_parts(parts, 1 if prefix else 0)


def build_prompt_with_triggers(config, lora_triggerwords_mode):
    """
    Build final prompt with LoRA triggers and model-specific prefix/suffix applied.
    Results are cached based on relevant config fields to avoid redundant work.

    Args:
        config: Configuration dictionary
        lora_triggerwords_mode: Mode for trigger word placement

    Returns:
        tuple: (final_prompt, trigger_string)
    """
    if config["lora"] == "None" or lora_triggerwords_mode == "None":
        prompt = _apply_model_prompt_affixes(config["positive"], config)
        return prompt, ""

    omit_list = config.get("lora_omit_triggers", [])
    append_settings = config.get("lora_triggerwords_append_settings", {})

    # Build a hashable cache key from relevant config fields
    cache_key = (
        config["lora"],
        config["positive"],
        tuple(omit_list),
        tuple(sorted(append_settings.items())) if append_settings else (),
        lora_triggerwords_mode,
        config.get("model_prompt_prefix", ""),
        config.get("model_prompt_suffix", "")
    )

    if cache_key in _build_prompt_cache:
        return _build_prompt_cache[cache_key]

    # Parse the lora string to get individual loras
    active_loras = parse_lora_definition(config["lora"])

    # Separate triggers by placement
    start_triggers = []
    end_triggers = []

    for lora_def in active_loras:
        lname, _, _ = lora_def
        placement = get_trigger_placement_for_lora(lname, config, lora_triggerwords_mode)
        # Get triggers specific to this lora
        try:
            lora_triggers = load_and_save_tags(lname, force_fetch=False)
            omit_normalized = [str(t).lower().strip().rstrip(',').strip() for t in omit_list]

            for tag in lora_triggers:
                cleaned_tag = tag.strip().rstrip(',').strip()
                if not _should_omit_trigger(cleaned_tag.lower(), omit_normalized):
                    if placement == "start":
                        start_triggers.append(cleaned_tag)
                    elif placement == "end":
                        end_triggers.append(cleaned_tag)
        except:
            pass

    # Build the trigger string for return value
    all_triggers = start_triggers + end_triggers
    trigger_string = ""
    if all_triggers:
        trigger_string = ", " + ", ".join(all_triggers)
        print(f"[GridTester] 🏷️ Trigger words for {config['lora'][:40]}: {', '.join(all_triggers[:5])}" +
              (f" (+{len(all_triggers)-5} more)" if len(all_triggers) > 5 else ""))

    # Build the full prompt with triggers in correct positions
    parts = []
    if start_triggers:
        parts.append(", ".join(start_triggers))
    parts.append(config['positive'])
    if end_triggers:
        parts.append(", ".join(end_triggers))

    final_prompt = join_prompt_parts(parts, 1 if start_triggers else 0)

    # Apply model-specific prefix/suffix (wraps the entire prompt+triggers)
    final_prompt = _apply_model_prompt_affixes(final_prompt, config)

    result = (final_prompt, trigger_string)

    # Store in cache (limit size to prevent memory issues)
    if len(_build_prompt_cache) > 256:
        _build_prompt_cache.clear()
    _build_prompt_cache[cache_key] = result

    return result