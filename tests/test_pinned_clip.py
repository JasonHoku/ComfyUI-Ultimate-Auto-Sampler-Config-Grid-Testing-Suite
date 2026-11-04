"""The text encoder is prepared once per encode loop, not once per prompt.

REGRESSION (2026-11): comfy.sd.CLIP.encode_from_tokens() calls load_model() ->
model_management.load_models_gpu() on EVERY encode, and ComfyUI's dynamic VRAM
loader re-runs the patcher's load() walk even for a resident model (the repeated
"Model MOSSImageTEModel_ prepared for dynamic VRAM loading" log line, N prompts ->
N+1 lines). For the 7 GB Qwen3.5-9B encoder that was ~0.9 s per prompt, which made
USCG's pre-encode of 60 prompts take about a minute. pinned_clip() loads the
encoder once and shadows load_model for the duration of the loop.
"""
import contextlib
import importlib.util
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# tests/conftest.py registers a STUB `batch_encoding` for the orchestrator's relative
# imports, so load the real module file under a private name (same trick as
# test_latent_geometry.py does for model_loader).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)
_mm = sys.modules.get("comfy.model_management")
if _mm is not None and not hasattr(_mm, "processing_interrupted"):
    _mm.processing_interrupted = lambda: False
if _mm is not None and "comfy" in sys.modules and not hasattr(sys.modules["comfy"], "model_management"):
    sys.modules["comfy"].model_management = _mm  # `import comfy.model_management` on stubs does not wire the attribute
_spec = importlib.util.spec_from_file_location(f"{_PKG}.batch_encoding_real", os.path.join(_ROOT, "batch_encoding.py"))
be = importlib.util.module_from_spec(_spec)
be.__package__ = _PKG
_spec.loader.exec_module(be)
if not hasattr(be.torch, "no_grad"):
    # torch stub (no real torch in this interpreter): the fast path only needs these
    be.torch = types.SimpleNamespace(no_grad=contextlib.nullcontext,
                                     cuda=types.SimpleNamespace(is_available=lambda: False))


# ---- fakes -----------------------------------------------------------------

class FakeCLIP:
    """Mimics comfy.sd.CLIP: encode_from_tokens() calls self.load_model() every time."""

    def __init__(self):
        self.patcher = object()
        self.load_calls = 0
        self.load_tokens = []
        self.encode_calls = 0
        self.cond_stage_model = types.SimpleNamespace()  # no clip_layer -> clip_skip branch is skipped

    def tokenize(self, text):
        return {"qwen": [[(len(text), 1.0)]]}

    def load_model(self, tokens={}):
        self.load_calls += 1
        self.load_tokens.append(tokens)
        return self.patcher

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        self.load_model(tokens)
        self.encode_calls += 1
        return {"cond": f"cond{self.encode_calls}", "pooled_output": "pooled"}


class OldCLIP(FakeCLIP):
    """Ancient ComfyUI: load_model() took no arguments."""

    def load_model(self):
        self.load_calls += 1
        return self.patcher

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        self.load_model()
        self.encode_calls += 1
        return {"cond": "c", "pooled_output": "p"}


class MemoryCache:
    def __init__(self):
        self.store = {}

    def get(self, prompt, prompt_type):
        return self.store.get((prompt_type, prompt))

    def set(self, prompt, cond, prompt_type):
        self.store[(prompt_type, prompt)] = cond


TEXTS = ["a cat", "a dog on a very long sunny beach with umbrellas", "sky", "tea", "a bird"]


# ---- pinned_clip -----------------------------------------------------------

def test_many_encodes_load_the_encoder_once():
    clip = FakeCLIP()
    with be.pinned_clip(clip, TEXTS):
        for t in TEXTS:
            be._encode_single(clip, t)
    assert clip.encode_calls == len(TEXTS)
    assert clip.load_calls == 1, "encoder must be prepared once for the whole loop"


def test_shadow_is_removed_after_the_block():
    clip = FakeCLIP()
    with be.pinned_clip(clip, TEXTS):
        be._encode_single(clip, "x")
    assert "load_model" not in vars(clip)
    be._encode_single(clip, "y")
    assert clip.load_calls == 2, "normal per-encode load_model behaviour is back"


def test_shadow_is_removed_on_exception():
    clip = FakeCLIP()
    with pytest.raises(RuntimeError):
        with be.pinned_clip(clip, TEXTS):
            raise RuntimeError("interrupted")
    assert "load_model" not in vars(clip)
    be._encode_single(clip, "y")
    assert clip.load_calls == 2


def test_nested_blocks_restore_the_outer_shadow():
    clip = FakeCLIP()
    with be.pinned_clip(clip, TEXTS):
        outer = vars(clip)["load_model"]
        with be.pinned_clip(clip):
            be._encode_single(clip, "inner")
        assert vars(clip)["load_model"] is outer
        be._encode_single(clip, "outer")
    assert "load_model" not in vars(clip)
    # the inner block's load_model call hits the outer shadow, so the encoder is
    # loaded exactly once for the whole nest and never per encode
    assert clip.load_calls == 1
    assert clip.encode_calls == 2


def test_single_load_is_sized_by_the_longest_prompt():
    clip = FakeCLIP()
    with be.pinned_clip(clip, TEXTS):
        pass
    longest = max(TEXTS, key=len)
    assert clip.load_tokens == [clip.tokenize(longest)]


def test_none_and_non_clip_pass_through():
    with be.pinned_clip(None) as c:
        assert c is None
    plain = object()
    with be.pinned_clip(plain) as c:
        assert c is plain


def test_old_load_model_signature_still_works():
    clip = OldCLIP()
    with be.pinned_clip(clip, TEXTS):
        for t in TEXTS:
            be._encode_single(clip, t)
    assert clip.load_calls == 1
    assert "load_model" not in vars(clip)


# ---- batch_encode_with_cache / encode_pair_cached --------------------------

def test_batch_encode_with_cache_prepares_the_encoder_once(capsys):
    clip = FakeCLIP()
    prompts = [f"prompt number {i} with some words" for i in range(10)]
    results = be.batch_encode_with_cache(clip, prompts, MemoryCache(), prompt_type="positive", batch_size=4)
    assert clip.encode_calls == 10
    assert clip.load_calls == 1
    assert set(results) == set(prompts)
    cond = results[prompts[0]]
    assert isinstance(cond, list) and cond[0][0] == "cond1" and cond[0][1] == {"pooled_output": "pooled"}
    out = capsys.readouterr().out
    assert "10 prompts in" in out and "ms/prompt" in out
    assert "load_model" not in vars(clip)


def test_batch_encode_with_cache_skips_the_encoder_on_full_cache_hit():
    clip = FakeCLIP()
    cache = MemoryCache()
    for p in TEXTS:
        cache.set(p, [["cached", {}]], "positive")
    results = be.batch_encode_with_cache(clip, TEXTS, cache, prompt_type="positive")
    assert clip.load_calls == 0 and clip.encode_calls == 0
    assert all(results[p] == [["cached", {}]] for p in TEXTS)


def test_encode_pair_cached_loads_once_and_never_on_hits():
    clip = FakeCLIP()
    cache = {"positive": {}, "negative": {}}
    pos, neg = be.encode_pair_cached(clip, cache, "a cat", "blurry")
    assert clip.load_calls == 1 and clip.encode_calls == 2
    assert pos is cache["positive"]["a cat"] and neg is cache["negative"]["blurry"]
    pos2, neg2 = be.encode_pair_cached(clip, cache, "a cat", "blurry")
    assert (pos2, neg2) == (pos, neg)
    assert clip.load_calls == 1, "both cached -> encoder not touched"
    be.encode_pair_cached(clip, cache, "a dog", "blurry")
    assert clip.load_calls == 2 and clip.encode_calls == 3, "only the missing half is encoded"
