"""
Batch CLIP Encoding Module
Handles efficient batch encoding of prompts with caching.
Supports prompt combinators: AND, CAT, AVG(), BREAK (native).
"""

import contextlib
import numbers
import re
import time
import weakref
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
# === SEVERAL PROMPTS THROUGH ONE TEXT-ENCODER FORWARD ===
# ============================================================
# MOSS-Image's Qwen3.5-9B encoder spends ~0.55 s per forward in fixed overhead
# (24 Gated DeltaNet layers, each a 63-iteration Python loop that doesn't depend on
# the batch size). Encoding B prompts in ONE forward amortises it.
#
# Design "batched forward, per-prompt replay": the INNER model's ``encode`` (the call
# ClipTokenWeightEncoder.encode_token_weights makes) is shadowed by a replay object.
# The first call runs the real encode on the right-padded batch; every call returns
# that prompt's slice, shaped exactly like a one-row forward. The existing
# ``_encode_single`` then runs per prompt, so template stripping, mask popping,
# clip_skip, device moves and dict building are the unmodified single-prompt code.

# Default for the ``te_batch_size`` Run Setting. 1 = off (today's per-prompt path).
DEFAULT_TE_BATCH_SIZE = 1

# clip_name -> family label. An allowlist: a family is added only after the live check passes.
TE_BATCH_FAMILIES = {"qwen35_9b": "MOSS-Image", "qwen3vl_4b": "Krea 2", "qwen25_7b": "Qwen-Image"}
TE_BATCH_MAX_TOKENS = 4096        # cap on rows * longest-row tokens per forward (memory guard)
TE_BATCH_SELFCHECK_RELL2 = 1e-3   # batched vs single cond tolerance

_TE_BATCH_DISABLED = weakref.WeakSet()
_TE_BATCH_DISABLED_IDS = set()    # fallback for objects that cannot be weak-referenced


class _TEBatchUnsupported(Exception):
    """The encoder's output didn't have the shape the batched path relies on."""


def _te_disabled(clip_model):
    try:
        return clip_model in _TE_BATCH_DISABLED
    except TypeError:
        return id(clip_model) in _TE_BATCH_DISABLED_IDS


def _te_disable(clip_model):
    try:
        _TE_BATCH_DISABLED.add(clip_model)
    except TypeError:
        _TE_BATCH_DISABLED_IDS.add(id(clip_model))


def _te_batch_target(clip_model):
    """-> ((inner, clip_name, pad_id), None) when this CLIP can batch, else (None, reason)."""
    csm = getattr(clip_model, "cond_stage_model", None)
    clip_attr = getattr(csm, "clip", None)
    clip_name = getattr(csm, "clip_name", None)
    if not isinstance(clip_attr, str) or not isinstance(clip_name, str):
        return None, "multi-encoder or unsupported CLIP wrapper"
    if clip_name not in TE_BATCH_FAMILIES:
        return None, f"{clip_name} is not a verified batchable encoder"
    inner = getattr(csm, clip_attr, None)
    if inner is None or not callable(getattr(inner, "encode", None)) or not callable(getattr(inner, "process_tokens", None)):
        return None, "encoder has no encode/process_tokens"
    if getattr(inner, "enable_attention_masks", None) is not True or getattr(inner, "return_attention_masks", None) is not True:
        return None, "encoder does not use attention masks"
    special = getattr(inner, "special_tokens", None) or {}
    if special.get("end") is not None:
        return None, "encoder has an end token"
    pad = special.get("pad")
    if not isinstance(pad, numbers.Integral) or isinstance(pad, bool):
        return None, "encoder has no pad token"
    if getattr(clip_model, "use_clip_schedule", False):
        return None, "scheduled CLIP"
    if "encode" in vars(inner):
        return None, "encoder already wrapped"
    return (inner, clip_name, int(pad)), None


def _batchable_ids(clip_model, text, clip_name, pad_id):
    """-> ((tokens_dict, ids_list), None) or (None, reason). Tokenises ONCE; the dict is reused for the encode."""
    if _has_combinators(text) or "BREAK" in text:
        return None, "combinator"
    tokens = clip_model.tokenize(text)
    if list(tokens) != [clip_name] or len(tokens[clip_name]) != 1:
        return None, "multi-section"
    ids = []
    for item in tokens[clip_name][0]:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            return None, "unexpected token entry"
        t, w = item[0], item[1]
        if not isinstance(t, numbers.Integral) or isinstance(t, bool):
            return None, "embedding or image token"
        if w != 1.0:
            return None, "weighted"
        ids.append(int(t))
    if not ids:
        return None, "empty"
    if pad_id in ids:
        return None, "pad id inside prompt"
    return (tokens, ids), None


def _own(t):
    return t.clone() if hasattr(t, "clone") else t.copy()


class _ReplayEncode:
    """Stands in for ``inner.encode``: one real forward on the padded batch, then per-prompt slices."""

    def __init__(self, real_encode, id_lists, pad_id):
        self.real = real_encode
        self.id_lists = id_lists
        self.pad = pad_id
        self.index = {tuple(ids): i for i, ids in enumerate(id_lists)}
        self.T = max(len(ids) for ids in id_lists)
        self.out = None
        self.forwards = 0
        self.misses = 0

    def __call__(self, to_encode):
        key = tuple(to_encode[0]) if len(to_encode) == 1 else None
        if key not in self.index:
            self.misses += 1  # safety net: unknown input runs the normal single path
            return self.real(to_encode)
        if self.out is None:
            padded = [list(ids) + [self.pad] * (self.T - len(ids)) for ids in self.id_lists]
            self.out = self.real(padded)
            self.forwards += 1
        return self._slice(self.index[key])

    def _slice(self, i):
        out = self.out
        z, pooled = out[0], out[1]
        B = len(self.id_lists)
        L = len(self.id_lists[i])
        if len(z.shape) < 2 or z.shape[0] != B or z.shape[-2] != self.T:
            raise _TEBatchUnsupported(f"unexpected output shape {tuple(z.shape)}")
        if pooled is not None:
            raise _TEBatchUnsupported("pooled output")  # EOS-position pooled would move with padding
        z_i = _own(z[i:i + 1, ..., :L, :])
        if len(out) == 2:
            return (z_i, None)
        extra_i = {}
        for k, v in out[2].items():
            if k == "attention_mask":
                if int(v[i].sum()) != L:
                    raise _TEBatchUnsupported("attention mask length mismatch")
                extra_i[k] = _own(v[i:i + 1, :L])
            elif hasattr(v, "shape") and len(v.shape) and v.shape[0] == B:
                extra_i[k] = _own(v[i:i + 1])
            else:
                raise _TEBatchUnsupported(f"extra output {k}")
        return (z_i, None, extra_i)


def _tensor_stats(a, b):
    """-> (||a-b||, ||b||, max|a-b|) for torch tensors or numpy arrays."""
    if hasattr(a, "detach"):
        a32, b32 = a.detach().float(), b.detach().float()
        d = a32 - b32
        return float(d.norm()), float(b32.norm()), (float(d.abs().max()) if d.numel() else 0.0)
    import numpy as np
    a64, b64 = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = a64 - b64
    return float(np.linalg.norm(d)), float(np.linalg.norm(b64)), (float(np.abs(d).max()) if d.size else 0.0)


def _compare_conds(batched, single):
    """-> (ok, reason, worst_relL2, worst_maxabs, shapes_equal). Compares keys, shapes, dtypes and values."""
    if len(batched) != len(single):
        return False, "conditioning length differs", 0.0, 0.0, False
    worst_rel = worst_abs = 0.0
    for (ca, da), (cb, db) in zip(batched, single):
        pairs = [("cond", ca, cb)]
        if sorted(da) != sorted(db):
            return False, f"keys differ {sorted(da)} vs {sorted(db)}", worst_rel, worst_abs, False
        for k in da:
            pairs.append((k, da[k], db[k]))
        for name, a, b in pairs:
            if hasattr(a, "shape") or hasattr(b, "shape"):
                if not (hasattr(a, "shape") and hasattr(b, "shape")) or tuple(a.shape) != tuple(b.shape):
                    return False, f"{name} shape differs", worst_rel, worst_abs, False
                if str(a.dtype) != str(b.dtype):
                    return False, f"{name} dtype differs", worst_rel, worst_abs, False
                dn, bn, mx = _tensor_stats(a, b)
                worst_rel = max(worst_rel, dn / max(bn, 1e-12))
                worst_abs = max(worst_abs, mx)
            elif a != b:
                return False, f"{name} value differs", worst_rel, worst_abs, True
    ok = worst_rel <= TE_BATCH_SELFCHECK_RELL2
    return ok, ("" if ok else f"relL2 {worst_rel:.2e} > {TE_BATCH_SELFCHECK_RELL2:g}"), worst_rel, worst_abs, True


def _is_oom(e):
    oom = getattr(comfy.model_management, "OOM_EXCEPTION", None)
    if isinstance(oom, type) and isinstance(e, oom):
        return True
    return "out of memory" in str(e).lower()


def encode_prompts_batched(clip_model, prompts, batch_size, clip_skip=0, verify=False):
    """
    Encode as many ``prompts`` as possible several-at-a-time in one text-encoder forward.

    Returns (results: dict prompt -> conditioning, leftovers: list[str], stats: dict).
    ``leftovers`` are prompts the caller must still encode the normal way (ineligible
    prompt/encoder, singleton batches, failures). ``batch_size <= 1`` returns everything as
    leftovers and touches nothing. The first batch of every call is self-checked against a
    single-prompt encode of its shortest prompt (most padding); on mismatch batching is
    disabled for this CLIP and every prompt is returned as a leftover.
    """
    prompts = list(prompts)
    stats = {"batched": 0, "forwards": 0, "max_rows": 0, "reason": None, "verify": None}
    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError):
        batch_size = 1
    if batch_size <= 1 or not prompts:
        return {}, prompts, stats
    if _te_disabled(clip_model):
        stats["reason"] = "disabled earlier in this run (self-check or unsupported output)"
        return {}, prompts, stats
    target, reason = _te_batch_target(clip_model)
    if target is None:
        stats["reason"] = reason
        return {}, prompts, stats
    inner, clip_name, pad_id = target

    # eligibility + dedupe by token ids
    tok = {}
    by_ids = {}
    for p in prompts:
        got, _why = _batchable_ids(clip_model, p, clip_name, pad_id)
        if got is None:
            continue
        tok[p] = got[0]
        by_ids.setdefault(tuple(got[1]), []).append(p)
    order = sorted(by_ids, key=len, reverse=True)  # longest first, stable

    results = {}
    singles = set()   # prompts that go the legacy way (singleton batches / OOM at 1)
    state = {"bs": batch_size, "forwards": 0, "max_rows": 0, "checked": False, "worst_rel": 0.0,
             "worst_abs": 0.0, "shapes_equal": True, "verified": 0, "verify_fail": 0}

    def _fail(why):
        _te_disable(clip_model)
        print(f"[GridTester] ⚠️ TE batching self-check failed ({why}) - encoding per prompt")
        return {}, prompts, stats

    def _interrupted():
        if comfy.model_management.processing_interrupted():
            print(f"\n[GridTester] 🛑 INTERRUPTED during text-encoder batching - Stopping all encoding")
            raise comfy.model_management.InterruptProcessingException()

    def _run(batch):
        """Encode ``batch`` (list of ids tuples) in one forward. Returns {prompt: cond}."""
        id_lists = [list(ids) for ids in batch]
        replay = _ReplayEncode(inner.encode, id_lists, pad_id)
        local = {}
        inner.encode = replay
        try:
            for ids in batch:
                _interrupted()
                p = by_ids[ids][0]
                local[p] = _encode_single(clip_model, p, clip_skip, tokens=tok[p])
        finally:
            # always un-shadow, whatever happened (incl. CUDA OOM / interrupt)
            vars(inner).pop("encode", None)
            replay.out = None
        if replay.misses:
            raise _TEBatchUnsupported("replay miss")
        state["forwards"] += replay.forwards
        state["max_rows"] = max(state["max_rows"], len(batch))
        return local

    def _single_ref(p):
        return _encode_single(clip_model, p, clip_skip, tokens=tok[p])

    def _accept(batch, local):
        """Self-check (first batch) / verify (dev), then merge. Returns a failure reason or None."""
        skip = None
        if not state["checked"]:
            skip = by_ids[batch[-1]][0]  # shortest prompt of the first batch = most padding
            ref = _single_ref(skip)
            ok, why, rel, mx, same = _compare_conds(local[skip], ref)
            if not ok:
                return why
            local[skip] = ref
            state["checked"] = True
            state["worst_rel"] = max(state["worst_rel"], rel)
            state["worst_abs"] = max(state["worst_abs"], mx)
        if verify:
            for ids in batch:
                p = by_ids[ids][0]
                state["verified"] += 1
                if p == skip:
                    continue
                ok, why, rel, mx, same = _compare_conds(local[p], _single_ref(p))
                state["worst_rel"] = max(state["worst_rel"], rel)
                state["worst_abs"] = max(state["worst_abs"], mx)
                state["shapes_equal"] = state["shapes_equal"] and same
                if not ok:
                    state["verify_fail"] += 1
        results.update(local)
        return None

    def _process(batch):
        try:
            local = _run(batch)
        except comfy.model_management.InterruptProcessingException:
            raise
        except _TEBatchUnsupported:
            raise
        except Exception as e:
            if not _is_oom(e):
                raise
            try:
                comfy.model_management.soft_empty_cache()
            except Exception:
                pass
            if len(batch) <= 1:
                singles.update(by_ids[batch[0]])
                return None
            half = len(batch) // 2
            print(f"[GridTester] ⚠️ TE batch OOM at B={len(batch)} - retrying at B={half}")
            state["bs"] = min(state["bs"], half)
            for part in (batch[:half], batch[half:]):
                if len(part) <= 1:
                    singles.update(by_ids[part[0]])
                else:
                    bad = _process(part)
                    if bad:
                        return bad
            return None
        return _accept(batch, local)

    pending = list(order)
    try:
        while pending:
            _interrupted()
            tmax = len(pending[0])
            n = 1
            while n < state["bs"] and n < len(pending) and (n + 1) * tmax <= TE_BATCH_MAX_TOKENS:
                n += 1
            batch, pending = pending[:n], pending[n:]
            if len(batch) == 1:
                singles.update(by_ids[batch[0]])
                continue
            bad = _process(batch)
            if bad:
                return _fail(bad)
    except _TEBatchUnsupported as e:
        return _fail(f"unsupported encoder output: {e}")

    # prompts that tokenised identically to a batched one share its conditioning
    for plist in by_ids.values():
        if plist[0] in results:
            for other in plist[1:]:
                results[other] = results[plist[0]]
    leftovers = [p for p in prompts if p not in results]
    stats["batched"] = len(results)
    stats["forwards"] = state["forwards"]
    stats["max_rows"] = state["max_rows"]
    if verify and state["verified"]:
        print(f"[GridTester] 🧪 TE batch verify: {state['verified']} prompts, worst relL2={state['worst_rel']:.2e}, "
              f"worst max|Δ|={state['worst_abs']:.2e}, shapes equal: {'yes' if state['shapes_equal'] else 'no'}"
              f"{'' if not state['verify_fail'] else ', FAILED: ' + str(state['verify_fail'])}")
        stats["verify"] = {"prompts": state["verified"], "worst_rel": state["worst_rel"],
                           "worst_abs": state["worst_abs"], "shapes_equal": state["shapes_equal"],
                           "failed": state["verify_fail"]}
    return results, leftovers, stats


# ============================================================
# === PROMPT COMBINATOR ENCODING (AND / CAT / AVG / BREAK) ===
# ============================================================

def _encode_single(clip_model, text, clip_skip=0, tokens=None):
    """
    Encode a single text segment. BREAK is handled natively by clip.tokenize().

    Args:
        clip_model: CLIP model instance
        text: Prompt text (may contain BREAK keyword)
        clip_skip: CLIP layer skip value
        tokens: Pre-tokenized ``clip_model.tokenize(text)`` result (the batched
            text-encoder path tokenises once to check eligibility and reuses it)

    Returns:
        Conditioning list: [[cond_tensor, pooled_dict]]
    """
    original_layer = None
    if clip_skip != 0 and hasattr(clip_model.cond_stage_model, 'clip_layer'):
        original_layer = clip_model.cond_stage_model.clip_layer
        clip_model.cond_stage_model.set_clip_options({"layer": clip_skip})

    try:
        if tokens is None:
            tokens = clip_model.tokenize(text)  # BREAK handled natively here
        pooled_dict = clip_model.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    finally:
        # restore even when the encode raises (e.g. CUDA OOM that the batched path retries)
        if original_layer is not None:
            clip_model.cond_stage_model.set_clip_options({"layer": original_layer})
    cond = pooled_dict.pop("cond")
    result = [[cond, pooled_dict]]

    return result


def _has_combinators(prompt):
    """True when the prompt uses AND / CAT / AVG() (the slow path of encode_prompt_with_combinators)."""
    return ' AND ' in prompt or ' CAT ' in prompt or 'AVG(' in prompt


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
    if not _has_combinators(prompt):
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


def batch_encode_with_cache(clip_model, prompts, cond_cache, prompt_type="positive", batch_size=64, clip_skip=0,
                            te_batch_size=DEFAULT_TE_BATCH_SIZE, te_batch_verify=False):
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
        te_batch_size: Encode up to this many eligible prompts per text-encoder forward
            (MOSS-Image / Krea 2 / Qwen-Image only; 1 = off, the per-prompt path)
        te_batch_verify: Dev only. Re-encode every batched prompt singly and print the worst difference
        
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
        
        prompts_all = list(prompts_to_encode)
        prompts_list = prompts_all
        te_stats = None
        
        with torch.no_grad(), pinned_clip(clip_model, prompts_all):
            if te_batch_size and te_batch_size > 1:
                # Several prompts per text-encoder forward; whatever can't be batched comes
                # back as leftovers and takes the unchanged per-prompt loop below.
                try:
                    batched, prompts_list, te_stats = encode_prompts_batched(
                        clip_model, prompts_all, te_batch_size, clip_skip, te_batch_verify)
                except comfy.model_management.InterruptProcessingException:
                    raise
                except Exception as e:
                    print(f"[GridTester] ⚠️ TE batching failed ({e}) - encoding per prompt")
                    batched, prompts_list, te_stats = {}, prompts_all, None
                for prompt, conditioning in batched.items():
                    results[prompt] = conditioning
                    cond_cache.set(prompt, conditioning, prompt_type)
                if te_stats and te_stats.get("reason"):
                    print(f"[GridTester] ℹ️ TE batching off for this encoder ({te_stats['reason']})")
            total_batches = (len(prompts_list) + batch_size - 1) // batch_size
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
        _per_prompt_ms = (_enc_elapsed / len(prompts_all) * 1000.0) if prompts_all else 0.0
        _te_note = ""
        if te_stats and te_stats.get("batched"):
            _te_note = (f"; {te_stats['batched']} batched in {te_stats['forwards']} forwards of <={te_stats['max_rows']}, "
                        f"{len(prompts_list)} per-prompt")
            if te_stats.get("verify"):
                v = te_stats["verify"]
                _te_note += f", verify worst relL2={v['worst_rel']:.1e}"
        print(f"[GridTester] ✅ Batch encoding complete! {len(prompts_all)} prompts in {_enc_elapsed:.1f}s "
              f"({_per_prompt_ms:.0f} ms/prompt, encoder prepared once{_te_note})")
    
    return results


def batch_encode_prompts(patched_clip, unique_positives, unique_negatives, cond_cache, clip_skip=0, enable_disk_cache=True,
                         te_batch_size=DEFAULT_TE_BATCH_SIZE, te_batch_verify=False):
    """
    Batch encode both positive and negative prompts.
    
    Args:
        patched_clip: Patched CLIP model
        unique_positives: Set of unique positive prompts
        unique_negatives: Set of unique negative prompts
        cond_cache: ConditioningCache instance
        clip_skip: Number of CLIP layers to skip from the end
        enable_disk_cache: Whether to save cache to disk (passed through to cache)
        te_batch_size: Prompts per text-encoder forward (1 = off), see batch_encode_with_cache
        te_batch_verify: Dev only, see batch_encode_with_cache
        
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
            clip_skip=clip_skip,
            te_batch_size=te_batch_size,
            te_batch_verify=te_batch_verify
        )

        conditioning_cache["negative"] = batch_encode_with_cache(
            patched_clip, 
            unique_negatives, 
            cond_cache, 
            prompt_type="negative",
            batch_size=16,
            clip_skip=clip_skip,
            te_batch_size=te_batch_size,
            te_batch_verify=te_batch_verify
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