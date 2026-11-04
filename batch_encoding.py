"""
Batch CLIP Encoding Module
Handles efficient batch encoding of prompts with caching.
Supports prompt combinators: AND, CAT, AVG(), BREAK (native).
"""

import contextlib
import re
import time
import torch
import gc
import comfy.model_management


# ============================================================
# === KEEP THE TEXT ENCODER RESIDENT ACROSS A RUN OF ENCODES ===
# ============================================================

@contextlib.contextmanager
def pinned_clip(clip_model, sample_texts=None):
    """
    Load a CLIP/text encoder ONCE and keep ComfyUI from re-preparing it on every
    encode inside the block.

    comfy.sd.CLIP.encode_from_tokens() calls self.load_model() ->
    comfy.model_management.load_models_gpu([patcher]) on EVERY encode. Since the
    dynamic VRAM loader (comfy_aimdo, ComfyUI 0.4x+) load_models_gpu() re-runs
    ModelPatcherDynamic.load() even for a model that is already resident: a Python
    walk over every module of the encoder plus free_memory()/get_free_memory()
    bookkeeping. That is the repeated

        Model MOSSImageTEModel_ prepared for dynamic VRAM loading. 7002MB Staged. ...

    log line, once per prompt (N prompts -> N+1 lines). For a ~500-module LLM text
    encoder (Qwen3.5-9B for MOSS-Image, Qwen2.5-VL for Qwen-Image, Qwen3-VL for
    Krea 2) the re-prepare costs about as much as the encode itself; SDXL's small
    CLIPs hide it, which is why SDXL pre-encoding always looked fast.

    Inside this context the encoder is loaded once through the normal load_model()
    path (sized by the longest of ``sample_texts`` so any memory estimate still
    runs) and ``clip_model.load_model`` is shadowed by a no-op returning the
    already-loaded patcher. Nothing in a plain encode loop touches model
    management, so the encoder stays resident. The shadow is removed on exit, even
    on exceptions, so the CLIP behaves normally afterwards (sampling may then evict
    it as usual). Nesting is safe: the inner block restores the outer shadow.
    """
    if clip_model is None or not hasattr(clip_model, "load_model"):
        yield clip_model
        return

    tokens = {}
    if sample_texts:
        try:
            longest = max(sample_texts, key=len)
            tokens = clip_model.tokenize(longest)
        except Exception:
            tokens = {}
    try:
        patcher = clip_model.load_model(tokens)
    except TypeError:
        patcher = clip_model.load_model()  # very old ComfyUI: load_model() took no tokens
    if patcher is None:
        patcher = getattr(clip_model, "patcher", None)

    inst = vars(clip_model)
    had_shadow = "load_model" in inst
    previous = inst.get("load_model")
    clip_model.load_model = lambda tokens={}: patcher
    try:
        yield clip_model
    finally:
        if had_shadow:
            clip_model.load_model = previous
        else:
            inst.pop("load_model", None)


def encode_pair_cached(clip_model, cache, positive, negative, clip_skip=0):
    """
    Fill cache["positive"][positive] / cache["negative"][negative] if missing,
    loading the encoder at most once for the pair, and return (pos_cond, neg_cond).
    Does not touch the encoder at all when both are already cached.
    """
    need_pos = positive not in cache["positive"]
    need_neg = negative not in cache["negative"]
    if need_pos or need_neg:
        with pinned_clip(clip_model):
            if need_pos:
                cache["positive"][positive] = encode_prompt_with_combinators(clip_model, positive, clip_skip)
            if need_neg:
                cache["negative"][negative] = encode_prompt_with_combinators(clip_model, negative, clip_skip)
    return cache["positive"][positive], cache["negative"][negative]


# ============================================================
# === PROMPT COMBINATOR ENCODING (AND / CAT / AVG / BREAK) ===
# ============================================================

def _encode_single(clip_model, text, clip_skip=0):
    """
    Encode a single text segment. BREAK is handled natively by clip.tokenize().

    Args:
        clip_model: CLIP model instance
        text: Prompt text (may contain BREAK keyword)
        clip_skip: CLIP layer skip value

    Returns:
        Conditioning list: [[cond_tensor, pooled_dict]]
    """
    original_layer = None
    if clip_skip != 0 and hasattr(clip_model.cond_stage_model, 'clip_layer'):
        original_layer = clip_model.cond_stage_model.clip_layer
        clip_model.cond_stage_model.set_clip_options({"layer": clip_skip})

    tokens = clip_model.tokenize(text)  # BREAK handled natively here
    pooled_dict = clip_model.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    cond = pooled_dict.pop("cond")
    result = [[cond, pooled_dict]]

    if original_layer is not None:
        clip_model.cond_stage_model.set_clip_options({"layer": original_layer})

    return result


def _parse_and_segments(prompt):
    """
    Split prompt on AND keyword. Returns list of (text, weight) tuples.

    Examples:
        "a cat AND a dog"           -> [("a cat", 1.0), ("a dog", 1.0)]
        "a cat :1.5 AND a dog :0.5" -> [("a cat", 1.5), ("a dog", 0.5)]
        "a cat AND a dog :0"        -> [("a cat", 1.0), ("a dog", 0.0)]
    """
    parts = re.split(r'\s+AND\s+', prompt)
    segments = []
    for part in parts:
        part = part.strip()
        # Check for trailing weight like "cat :1.5" (space before colon distinguishes from (word:weight) syntax)
        weight_match = re.match(r'^(.*?)\s+:(\d+\.?\d*)\s*$', part)
        if weight_match:
            segments.append((weight_match.group(1).strip(), float(weight_match.group(2))))
        else:
            segments.append((part, 1.0))
    return segments


def _parse_avg_segments(text):
    """
    Split text on AVG(weight) markers. Returns list of (text, avg_weight) tuples.
    The first segment has weight=None (it's the base). Subsequent segments have their AVG weight.

    Examples:
        "realistic AVG(0.3) anime"  -> [("realistic", None), ("anime", 0.3)]
        "a AVG() b"                 -> [("a", None), ("b", 0.5)]  (default weight 0.5)
        "no avg here"               -> [("no avg here", None)]
    """
    parts = re.split(r'\s*AVG\((\d*\.?\d*)\)\s*', text)
    if len(parts) == 1:
        return [(text, None)]  # No AVG found

    segments = []
    for i in range(0, len(parts), 2):
        segment_text = parts[i].strip() if i < len(parts) else ""
        if not segment_text:
            continue
        if i == 0:
            # First segment is the base — no weight
            segments.append((segment_text, None))
        else:
            # Subsequent segments get the weight from the preceding capture group
            weight_str = parts[i - 1] if i - 1 < len(parts) else ""
            weight = float(weight_str) if weight_str else 0.5
            segments.append((segment_text, weight))

    return segments if segments else [(text, None)]


def _parse_cat_segments(text):
    """
    Split text on CAT keyword. Returns list of text segments.

    Example:
        "a cat CAT in a garden" -> ["a cat", "in a garden"]
    """
    return [s.strip() for s in re.split(r'\s+CAT\s+', text) if s.strip()]


def _conditioning_combine(cond1, cond2):
    """
    ConditioningCombine: list concatenation.
    Each conditioning entry becomes a separate guidance signal during sampling.
    Same as ComfyUI's ConditioningCombine node (nodes.py).
    """
    return cond1 + cond2


def _conditioning_concat(cond_to, cond_from):
    """
    ConditioningConcat: tensor concatenation along dim 1.
    Extends the token context window by appending from-conditioning tokens.
    Same as ComfyUI's ConditioningConcat node (nodes.py).
    """
    out = []
    cond_from_tensor = cond_from[0][0]
    for i in range(len(cond_to)):
        t1 = cond_to[i][0]
        tw = torch.cat((t1, cond_from_tensor), 1)
        n = [tw, cond_to[i][1].copy()]
        out.append(n)
    return out


def _conditioning_average(cond_to, cond_from, strength):
    """
    ConditioningAverage: weighted blend of conditioning tensors.
    strength=1.0 means 100% cond_to, strength=0.0 means 100% cond_from.
    Same as ComfyUI's ConditioningAverage node (nodes.py).
    """
    out = []
    cond_from_tensor = cond_from[0][0]
    pooled_from = cond_from[0][1].get("pooled_output", None)

    for i in range(len(cond_to)):
        t1 = cond_to[i][0]
        pooled_to = cond_to[i][1].get("pooled_output", pooled_from)

        t0 = cond_from_tensor[:, :t1.shape[1]]
        if t0.shape[1] < t1.shape[1]:
            t0 = torch.cat([t0, torch.zeros((1, t1.shape[1] - t0.shape[1], t1.shape[2]))], dim=1)

        tw = torch.mul(t1, strength) + torch.mul(t0, 1.0 - strength)
        t_to = cond_to[i][1].copy()

        if pooled_from is not None and pooled_to is not None:
            t_to["pooled_output"] = torch.mul(pooled_to, strength) + torch.mul(pooled_from, 1.0 - strength)
        elif pooled_from is not None:
            t_to["pooled_output"] = pooled_from

        out.append([tw, t_to])
    return out


def encode_prompt_with_combinators(clip_model, prompt, clip_skip=0):
    """
    Encode a prompt that may contain AND, CAT, or AVG() combinators.

    Processing order (matching comfyui-prompt-control):
      1. Split on AND (creates separate conditioning entries via ConditioningCombine)
      2. Within each AND-segment, process AVG(weight) (weighted blend via ConditioningAverage)
      3. Within each segment, process CAT (tensor concat via ConditioningConcat)
      4. BREAK is handled natively by clip.tokenize() within each segment

    Syntax:
      - "prompt1 AND prompt2"           → ConditioningCombine
      - "prompt1 AND prompt2 :0.5"      → AND with weight scaling
      - "prompt1 CAT prompt2"           → ConditioningConcat
      - "prompt1 AVG(0.3) prompt2"      → ConditioningAverage (0.3 = 30% prompt2)
      - "prompt1 BREAK prompt2"         → Separate 77-token chunks (native)

    Args:
        clip_model: CLIP model instance
        prompt: Prompt text possibly containing combinators
        clip_skip: CLIP layer skip value

    Returns:
        Conditioning list: [[cond_tensor, pooled_dict], ...]
    """
    # Fast path: if no combinators present, use direct encoding
    if ' AND ' not in prompt and ' CAT ' not in prompt and 'AVG(' not in prompt:
        return _encode_single(clip_model, prompt, clip_skip)

    # 1. Split on AND
    and_segments = _parse_and_segments(prompt)

    all_conditionings = []
    for segment_text, and_weight in and_segments:
        if and_weight == 0:
            continue  # Skip zero-weighted AND segments

        # 2. Process AVG() within this AND-segment
        avg_segments = _parse_avg_segments(segment_text)

        if len(avg_segments) > 1:
            # Has AVG — encode first segment as base, then blend with subsequent segments
            result = _encode_single(clip_model, avg_segments[0][0], clip_skip)
            for avg_text, avg_weight in avg_segments[1:]:
                next_cond = _encode_single(clip_model, avg_text, clip_skip)
                w = avg_weight if avg_weight is not None else 0.5
                result = _conditioning_average(result, next_cond, w)
        else:
            # 3. Process CAT within this segment
            cat_parts = _parse_cat_segments(segment_text)
            if len(cat_parts) > 1:
                result = _encode_single(clip_model, cat_parts[0], clip_skip)
                for cat_text in cat_parts[1:]:
                    next_cond = _encode_single(clip_model, cat_text, clip_skip)
                    result = _conditioning_concat(result, next_cond)
            else:
                result = _encode_single(clip_model, segment_text, clip_skip)

        # Apply AND weight by scaling the conditioning tensor
        if and_weight != 1.0:
            result = [[torch.mul(r[0], and_weight), r[1]] for r in result]

        all_conditionings.append(result)

    # 4. Combine all AND segments via ConditioningCombine
    if not all_conditionings:
        return _encode_single(clip_model, "", clip_skip)

    final = all_conditionings[0]
    for additional in all_conditionings[1:]:
        final = _conditioning_combine(final, additional)

    return final


def batch_encode_with_cache(clip_model, prompts, cond_cache, prompt_type="positive", batch_size=64, clip_skip=0):
    """
    Batch encode prompts while checking persistent cache first.
    Only encodes prompts that aren't already cached.
    
    Args:
        clip_model: CLIP model to use for encoding
        prompts: Set or list of prompt strings to encode
        cond_cache: ConditioningCache instance
        prompt_type: "positive" or "negative"
        batch_size: Number of prompts to encode at once
        clip_skip: Number of CLIP layers to skip from the end (0 = use last layer, -1 = skip 1 layer, -2 = skip 2 layers)
        
    Returns:
        dict: Mapping of prompt text to conditioning tensors
    """
    results = {}
    prompts_to_encode = []
    

    # Check cache first
    print(f"[GridTester] 🔍 Checking cache for {len(prompts)} {prompt_type} prompts...")
    for prompt in prompts:
        cached = cond_cache.get(prompt, prompt_type)
        if cached is not None:
            results[prompt] = cached
        else:
            prompts_to_encode.append(prompt)
    
    cache_hits = len(results)
    cache_misses = len(prompts_to_encode)
    
    print(f"[GridTester] 📊 Cache: {cache_hits} hits, {cache_misses} misses")
    
    # Batch encode uncached prompts
    if prompts_to_encode:
        clip_skip_msg = f" (clip_skip={clip_skip})" if clip_skip != 0 else ""
        print(f"[GridTester] 🚀 Batch encoding {len(prompts_to_encode)} {prompt_type} prompts{clip_skip_msg} (batch_size={batch_size})")
        
        # The encoder is loaded ONCE by pinned_clip (below) and stays resident for
        # the whole loop; without it ComfyUI re-prepares it on every single encode.
        _t_enc_start = time.time()
        
        prompts_list = list(prompts_to_encode)
        total_batches = (len(prompts_list) + batch_size - 1) // batch_size
        
        with torch.no_grad(), pinned_clip(clip_model, prompts_list):
            for batch_idx in range(0, len(prompts_list), batch_size):
                batch_prompts = prompts_list[batch_idx:batch_idx + batch_size]
                current_batch = (batch_idx // batch_size) + 1
                
                # Encode batch
                for prompt in batch_prompts:
                    # Check for interrupt before each prompt encoding
                    if comfy.model_management.processing_interrupted():
                        print(f"\n[GridTester] 🛑 INTERRUPTED during {prompt_type} encoding - Stopping all encoding")
                        raise comfy.model_management.InterruptProcessingException()

                    try:
                        # Encode with combinator support (AND/CAT/AVG/BREAK)
                        conditioning = encode_prompt_with_combinators(clip_model, prompt, clip_skip)
                        results[prompt] = conditioning
                        cond_cache.set(prompt, conditioning, prompt_type)

                    except comfy.model_management.InterruptProcessingException:
                        print(f"\n[GridTester] 🛑 INTERRUPTED during {prompt_type} encoding - Stopping all encoding")
                        raise  # Re-raise to stop all encoding

                    except Exception as e:
                        print(f"[GridTester] ⚠️ Failed to encode {prompt_type} prompt: {e}")
                        print(f"[GridTester] ⚠️ Prompt was: {prompt[:80]}...")
                        import traceback
                        traceback.print_exc()
                        # Store None so we can detect this downstream with a clear error
                        results[prompt] = None
                
                # Progress
                encoded_count = min(batch_idx + batch_size, len(prompts_list))
                if current_batch % 5 == 0 or current_batch == total_batches:
                    print(f"[GridTester]   Batch {current_batch}/{total_batches} | Encoded {encoded_count}/{len(prompts_list)}")
        
        _enc_elapsed = time.time() - _t_enc_start
        _per_prompt_ms = (_enc_elapsed / len(prompts_list) * 1000.0) if prompts_list else 0.0
        print(f"[GridTester] ✅ Batch encoding complete! {len(prompts_list)} prompts in {_enc_elapsed:.1f}s "
              f"({_per_prompt_ms:.0f} ms/prompt, encoder prepared once)")
    
    return results


def batch_encode_prompts(patched_clip, unique_positives, unique_negatives, cond_cache, clip_skip=0, enable_disk_cache=True):
    """
    Batch encode both positive and negative prompts.
    
    Args:
        patched_clip: Patched CLIP model
        unique_positives: Set of unique positive prompts
        unique_negatives: Set of unique negative prompts
        cond_cache: ConditioningCache instance
        clip_skip: Number of CLIP layers to skip from the end
        enable_disk_cache: Whether to save cache to disk (passed through to cache)
        
    Returns:
        dict: conditioning_cache with "positive" and "negative" keys
    """
    conditioning_cache = {"positive": {}, "negative": {}}
    
    print(f"[GridTester] 🧠 Found {len(unique_positives)} unique positive prompts")
    print(f"[GridTester] 🧠 Found {len(unique_negatives)} unique negative prompts")
    
    # Use ComfyUI's model management to prevent memory leak warnings during batch encoding
    # The model needs to stay loaded for the entire encoding loop
    with torch.no_grad():
        # Encode all positive prompts
        print(f"[GridTester] 🧠 Encoding {len(unique_positives)} unique positive prompts...")
        conditioning_cache["positive"] = batch_encode_with_cache(
            patched_clip, 
            unique_positives, 
            cond_cache, 
            prompt_type="positive",
            batch_size=16,
            clip_skip=clip_skip
        )

        conditioning_cache["negative"] = batch_encode_with_cache(
            patched_clip, 
            unique_negatives, 
            cond_cache, 
            prompt_type="negative",
            batch_size=16,
            clip_skip=clip_skip
        )
    
    # Save and print cache stats (only saves if disk cache is enabled)
    if cond_cache is not None:
        cond_cache.save()
        # cond_cache.print_stats() # for debugging
    
    # Final cleanup after all encoding is complete
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"[GridTester] ✅ All prompts pre-encoded!")
    print(f"[GridTester] 💾 Cache size: {len(conditioning_cache['positive'])} positive, {len(conditioning_cache['negative'])} negative")
    
    return conditioning_cache