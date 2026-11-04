"""Several prompts through ONE text-encoder forward (batch_encoding.encode_prompts_batched).

The batched path must hand back exactly what the per-prompt path returns (same dict
keys, shapes, dtypes, values), fall back for anything it cannot batch safely, and
never leave the inner encoder's ``encode`` shadowed. Fake numpy tensors, no torch.
"""
import contextlib
import importlib.util
import os
import sys
import types

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)
_mm = sys.modules.get("comfy.model_management")
if _mm is not None and "comfy" in sys.modules and not hasattr(sys.modules["comfy"], "model_management"):
    sys.modules["comfy"].model_management = _mm
_spec = importlib.util.spec_from_file_location(f"{_PKG}.batch_encoding_real_te", os.path.join(_ROOT, "batch_encoding.py"))
be = importlib.util.module_from_spec(_spec)
be.__package__ = _PKG
_spec.loader.exec_module(be)
if not hasattr(be.torch, "no_grad"):
    be.torch = types.SimpleNamespace(no_grad=contextlib.nullcontext,
                                     cuda=types.SimpleNamespace(is_available=lambda: False))


class _Interrupt(Exception):
    pass


class _MM:
    """Replaces be.comfy.model_management for the duration of a test."""

    def __init__(self):
        self.interrupt_on_call = None
        self.calls = 0
        self.empty_cache_calls = 0
        self.OOM_EXCEPTION = MemoryError
        self.InterruptProcessingException = _Interrupt

    def processing_interrupted(self):
        self.calls += 1
        return self.interrupt_on_call is not None and self.calls >= self.interrupt_on_call

    def soft_empty_cache(self):
        self.empty_cache_calls += 1


@pytest.fixture(autouse=True)
def mm(monkeypatch):
    fake = _MM()
    monkeypatch.setattr(be.comfy, "model_management", fake, raising=False)
    be._TE_BATCH_DISABLED.clear()
    be._TE_BATCH_DISABLED_IDS.clear()
    return fake


PAD = 0
VEC = np.array([1.0, 2.0, 3.0, 4.0])


class FakeInner:
    special_tokens = {"pad": PAD}
    enable_attention_masks = True
    return_attention_masks = True

    def __init__(self, pooled=None, bias_when_batched=0.0, oom_over=None, fail=None):
        self.calls = []            # list of row lists passed to the REAL encode
        self.layers = []
        self.owner = None
        self.pooled = pooled
        self.bias = bias_when_batched
        self.oom_over = oom_over
        self.fail = fail

    def process_tokens(self, *a, **kw):  # presence only
        raise AssertionError("not used")

    def encode(self, rows):
        T = len(rows[0])
        assert all(len(r) == T for r in rows), "rows must be equal length"
        self.calls.append([list(r) for r in rows])
        if self.owner is not None:
            self.layers.append(getattr(self.owner, "clip_layer", None))
        if self.fail:
            raise self.fail
        if self.oom_over is not None and len(rows) > self.oom_over:
            raise MemoryError("CUDA out of memory")
        ids = np.array(rows, dtype=np.float64)
        z = np.cumsum(ids, axis=1)[:, :, None] * VEC[None, None, :]
        if len(rows) > 1:
            z = z + self.bias
        mask = np.ones(ids.shape, dtype=np.int64)
        for b, r in enumerate(rows):
            if PAD in r:
                mask[b, r.index(PAD):] = 0
        pooled = None if self.pooled is None else np.zeros((len(rows), 2))
        return z.astype(np.float32), pooled, {"attention_mask": mask}

    def encode_token_weights(self, token_weight_pairs):
        to_encode = [[t[0] for t in section] for section in token_weight_pairs]
        o = self.encode(to_encode)
        return o[0], o[1], {"attention_mask": o[2]["attention_mask"]}


class FakeCSM:
    def __init__(self, inner, clip_name="qwen35_9b", with_layer=False):
        self.clip = clip_name
        self.clip_name = clip_name
        setattr(self, clip_name, inner)
        inner.owner = self
        if with_layer:
            self.clip_layer = -1

    def set_clip_options(self, opts):
        if "layer" in opts:
            self.clip_layer = opts["layer"]

    def encode_token_weights(self, tokens):
        return getattr(self, self.clip).encode_token_weights(tokens[self.clip_name])


class FakeKreaCSM(FakeCSM):
    """Outer post-processing: strip a 3-token template prefix and pop the all-ones mask."""

    def encode_token_weights(self, tokens):
        z, pooled, extra = super().encode_token_weights(tokens)
        extra = dict(extra)
        assert int(extra.pop("attention_mask").min()) == 1
        return z[:, 3:], pooled, extra


class FakeCLIP:
    def __init__(self, csm, clip_name="qwen35_9b"):
        self.cond_stage_model = csm
        self.clip_name = clip_name
        self.patcher = object()

    def tokenize(self, text):
        t = text.strip()
        if " BREAK " in t:
            parts = t.split(" BREAK ")
            return {self.clip_name: [[(ord(c) % 50 + 1, 1.0) for c in p] for p in parts]}
        if t.startswith("(") and t.endswith(")"):
            return {self.clip_name: [[(ord(c) % 50 + 1, 1.2) for c in t[1:-1]]]}
        return {self.clip_name: [[(PAD if c == "#" else ord(c) % 50 + 1, 1.0) for c in t]]}

    def load_model(self, tokens={}):
        return self.patcher

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        out = self.cond_stage_model.encode_token_weights(tokens)
        d = {"cond": out[0], "pooled_output": out[1]}
        if len(out) > 2:
            d.update(out[2])
        return d


def make(krea=False, with_layer=False, **inner_kw):
    inner = FakeInner(**inner_kw)
    csm = (FakeKreaCSM if krea else FakeCSM)(inner, with_layer=with_layer)
    return FakeCLIP(csm), inner, csm


def single(clip, text, skip=0):
    return be.encode_prompt_with_combinators(clip, text, skip)


def assert_same(a, b):
    assert len(a) == len(b)
    (ca, da), (cb, db) = a[0], b[0]
    assert ca.shape == cb.shape and ca.dtype == cb.dtype
    assert np.array_equal(ca, cb)
    assert sorted(da) == sorted(db)
    for k in da:
        if hasattr(da[k], "shape"):
            assert da[k].shape == db[k].shape and np.array_equal(da[k], db[k])
        else:
            assert da[k] == db[k]


def multi_row_calls(inner):
    return [c for c in inner.calls if len(c) > 1]


PROMPTS5 = ["abcdefghij", "abcdefgh", "abcdef", "abcd", "ab"]


# ---- 1/2: grouping, slicing, padding -----------------------------------------

def test_batches_of_four_with_singleton_leftover_and_equal_results():
    clip, inner, _ = make()
    results, leftovers, stats = be.encode_prompts_batched(clip, PROMPTS5, 4)
    assert [len(c) for c in multi_row_calls(inner)] == [4]
    assert leftovers == ["ab"]                       # singleton batch -> legacy path
    assert stats["batched"] == 4 and stats["forwards"] == 1 and stats["max_rows"] == 4
    for p in PROMPTS5[:4]:
        ref = single(clip, p)
        assert_same(results[p], ref)
        assert results[p][0][0].shape == (1, len(p), 4)
        assert results[p][0][1]["attention_mask"].shape == (1, len(p))
        assert results[p][0][1]["attention_mask"].min() == 1


def test_rows_are_right_padded_longest_first():
    clip, inner, _ = make()
    be.encode_prompts_batched(clip, PROMPTS5[::-1], 4)
    rows = multi_row_calls(inner)[0]
    assert all(len(r) == 10 for r in rows)
    lens = [len([t for t in r if t != PAD]) for r in rows]
    assert lens == [10, 8, 6, 4]
    for r, n in zip(rows, lens):
        assert r[n:] == [PAD] * (10 - n) and PAD not in r[:n]


# ---- 3: outer post-processing survives per item -----------------------------------

def test_krea_like_outer_strip_and_mask_pop_per_item():
    clip, inner, _ = make(krea=True)
    prompts = ["abcdefgh", "abcdef", "abcd"]
    results, leftovers, _ = be.encode_prompts_batched(clip, prompts, 4)
    assert leftovers == []
    for p in prompts:
        ref = single(clip, p)
        assert_same(results[p], ref)
        assert results[p][0][0].shape == (1, len(p) - 3, 4)
        assert "attention_mask" not in results[p][0][1]


# ---- 4: fallbacks ----------------------------------------------------------------

@pytest.mark.parametrize("bad", ["(weighted)", "a AND b cc", "a CAT b cc", "a AVG(0.5) b", "abc BREAK def", "ab#cd"])
def test_ineligible_prompts_never_enter_a_batched_forward(bad):
    clip, inner, _ = make()
    prompts = [bad, "abcdefgh", "abcdef", "abcd"]
    results, leftovers, _ = be.encode_prompts_batched(clip, prompts, 4)
    assert bad in leftovers and bad not in results
    for p in prompts[1:]:
        assert p in results
    badids = [t[0] for t in clip.tokenize(bad)[clip.clip_name][0]]
    for rows in multi_row_calls(inner):
        assert badids not in rows


def test_multi_key_tokens_dict_falls_back():
    clip, inner, _ = make()
    orig = clip.tokenize
    clip.tokenize = lambda t: dict(orig(t), extra=[[(1, 1.0)]])
    results, leftovers, stats = be.encode_prompts_batched(clip, PROMPTS5, 4)
    assert results == {} and leftovers == PROMPTS5 and inner.calls == []


def test_non_allowlisted_clip_name():
    inner = FakeInner()
    clip = FakeCLIP(FakeCSM(inner, clip_name="t5xxl"), clip_name="t5xxl")
    results, leftovers, stats = be.encode_prompts_batched(clip, PROMPTS5, 4)
    assert results == {} and leftovers == PROMPTS5 and "t5xxl" in stats["reason"] and inner.calls == []


def test_encoder_without_masks_or_with_end_token_is_excluded():
    clip, inner, _ = make()
    inner.enable_attention_masks = False
    assert be.encode_prompts_batched(clip, PROMPTS5, 4)[0] == {}
    inner.enable_attention_masks = True
    inner.special_tokens = {"pad": PAD, "end": 7}
    assert be.encode_prompts_batched(clip, PROMPTS5, 4)[0] == {}
    assert inner.calls == []


def test_multi_encoder_wrapper_and_clip_schedule_are_excluded():
    clip, inner, csm = make()
    del csm.clip
    assert be.encode_prompts_batched(clip, PROMPTS5, 4)[0] == {}
    clip2, inner2, _ = make()
    clip2.use_clip_schedule = True
    assert be.encode_prompts_batched(clip2, PROMPTS5, 4)[0] == {}


def test_pooled_output_disables_batching_for_the_run(capsys):
    clip, inner, _ = make(pooled=True)
    results, leftovers, _ = be.encode_prompts_batched(clip, PROMPTS5, 4)
    assert results == {} and leftovers == PROMPTS5
    assert "encode" not in vars(inner)
    n = len(inner.calls)
    results, leftovers, stats = be.encode_prompts_batched(clip, PROMPTS5, 4)
    assert results == {} and leftovers == PROMPTS5 and len(inner.calls) == n and "disabled" in stats["reason"]


# ---- 5: off means off ------------------------------------------------------------

def test_batch_size_one_touches_nothing():
    clip, inner, _ = make()
    results, leftovers, stats = be.encode_prompts_batched(clip, PROMPTS5, 1)
    assert results == {} and leftovers == PROMPTS5 and inner.calls == []
    assert "encode" not in vars(inner)


# ---- 6: self-check -----------------------------------------------------------------

def test_self_check_failure_falls_back_and_stays_disabled(capsys):
    clip, inner, _ = make(bias_when_batched=100.0)
    results, leftovers, _ = be.encode_prompts_batched(clip, PROMPTS5, 4)
    assert results == {} and leftovers == PROMPTS5
    assert "self-check failed" in capsys.readouterr().out
    assert "encode" not in vars(inner)
    # the per-prompt path is unaffected and matches a never-batched clip
    ref_clip, _, _ = make()
    for p in PROMPTS5:
        assert_same(single(clip, p), single(ref_clip, p))
    n = len(inner.calls)
    assert be.encode_prompts_batched(clip, PROMPTS5, 4)[0] == {}
    assert len(inner.calls) == n


# ---- 7: OOM ------------------------------------------------------------------------

def test_oom_halves_the_batch_and_still_gives_correct_results(capsys, mm):
    clip, inner, _ = make(oom_over=2)
    prompts = ["abcdefghijkl"[:n] for n in range(12, 4, -1)]  # 8 distinct lengths
    results, leftovers, _ = be.encode_prompts_batched(clip, prompts, 8)
    out = capsys.readouterr().out
    assert "OOM at B=8 - retrying at B=4" in out and "OOM at B=4 - retrying at B=2" in out
    assert leftovers == [] and mm.empty_cache_calls >= 1
    assert "encode" not in vars(inner)
    for p in prompts:
        assert_same(results[p], single(clip, p))


# ---- 8: the shadow never leaks ----------------------------------------------------------

def test_shadow_removed_after_success():
    clip, inner, _ = make()
    be.encode_prompts_batched(clip, PROMPTS5, 4)
    assert "encode" not in vars(inner)


def test_shadow_removed_after_an_exception():
    clip, inner, _ = make(fail=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        be.encode_prompts_batched(clip, PROMPTS5, 4)
    assert "encode" not in vars(inner)


def test_shadow_removed_after_oom_that_never_recovers():
    clip, inner, _ = make(oom_over=0)   # every forward raises
    results, leftovers, _ = be.encode_prompts_batched(clip, PROMPTS5, 4)
    assert results == {} and sorted(leftovers) == sorted(PROMPTS5)
    assert "encode" not in vars(inner)


@pytest.mark.parametrize("k", [1, 3])
def test_shadow_removed_after_interrupt(mm, k):
    clip, inner, _ = make()
    mm.interrupt_on_call = k   # k=1: before install; k=3: inside the shadowed block
    with pytest.raises(_Interrupt):
        be.encode_prompts_batched(clip, PROMPTS5, 4)
    assert "encode" not in vars(inner)


# ---- 9: clip_skip -----------------------------------------------------------------------

def test_clip_skip_applies_to_the_batched_forward_and_is_restored():
    clip, inner, csm = make(with_layer=True)
    results, _, _ = be.encode_prompts_batched(clip, PROMPTS5, 4, clip_skip=-2)
    batched_layers = [lay for c, lay in zip(inner.calls, inner.layers) if len(c) > 1]
    assert batched_layers == [-2]
    assert csm.clip_layer == -1


def test_clip_layer_is_restored_when_the_encode_raises():
    clip, inner, csm = make(with_layer=True, fail=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        be._encode_single(clip, "abc", -2)
    assert csm.clip_layer == -1


# ---- 10: duplicates ---------------------------------------------------------------------

def test_prompts_with_identical_tokens_share_one_row():
    clip, inner, _ = make()
    prompts = ["abcdef", " abcdef ", "abcd", "ab"]
    results, leftovers, _ = be.encode_prompts_batched(clip, prompts, 4)
    assert leftovers == []
    rows = multi_row_calls(inner)[0]
    assert len(rows) == 3
    assert_same(results["abcdef"], results[" abcdef "])


# ---- 11: token cap ----------------------------------------------------------------------

def test_token_cap_prevents_oversized_batches():
    clip, inner, _ = make()
    long_prompts = ["a" * 3000, "b" * 3000, "c" * 3000]
    results, leftovers, _ = be.encode_prompts_batched(clip, long_prompts, 8)
    assert multi_row_calls(inner) == []          # 2 x 3000 > TE_BATCH_MAX_TOKENS: nothing batched
    assert sorted(leftovers) == sorted(long_prompts)


def test_token_cap_still_allows_batches_that_fit():
    clip, inner, _ = make()
    prompts = ["a" * 2000, "b" * 2000, "c" * 2000]
    results, leftovers, _ = be.encode_prompts_batched(clip, prompts, 8)
    sizes = [len(c) for c in multi_row_calls(inner)]
    assert sizes == [2]
    assert len(results) == 2 and len(leftovers) == 1


def test_verify_mode_reports_every_prompt(capsys):
    clip, inner, _ = make()
    results, _, stats = be.encode_prompts_batched(clip, PROMPTS5[:4], 4, verify=True)
    assert stats["verify"]["prompts"] == 4 and stats["verify"]["shapes_equal"] and stats["verify"]["failed"] == 0
    assert "TE batch verify: 4 prompts" in capsys.readouterr().out
