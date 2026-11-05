"""COUPLE prompts through the real batch_encoding.encode_prompt_with_combinators.

Fake CLIP: tokenize() returns the text, encode_from_tokens() returns {"cond": text, "pooled_output": None},
so every conditioning entry is traceable by its text. CAT / AVG need real tensors (requires_torch).
"""
import importlib.util
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)
sys.path.insert(0, _ROOT)
_mm = sys.modules.get("comfy.model_management")
if _mm is not None and "comfy" in sys.modules and not hasattr(sys.modules["comfy"], "model_management"):
    sys.modules["comfy"].model_management = _mm
_spec = importlib.util.spec_from_file_location(f"{_PKG}.batch_encoding_real_regional",
                                               os.path.join(_ROOT, "batch_encoding.py"))
be = importlib.util.module_from_spec(_spec)
be.__package__ = _PKG
_spec.loader.exec_module(be)

import importlib  # noqa: E402

rp = importlib.import_module(f"{_PKG}.regional_prompt")


class FakeCLIP:
    def __init__(self):
        self.tokenized = []

    def tokenize(self, text):
        self.tokenized.append(text)
        return text

    def encode_from_tokens(self, tokens, return_pooled=True, return_dict=True):
        return {"cond": tokens, "pooled_output": None}


def _tags(out):
    return [e[1].get(rp.REGION_KEY) for e in out]


def test_base_and_two_regions_in_order_with_tags():
    out = be.encode_prompt_with_combinators(FakeCLIP(), "base COUPLE a | b")
    assert [e[0] for e in out] == ["base", "a", "b"]
    assert _tags(out) == [None, rp.region_tag(0, 2, "cols", 1.0), rp.region_tag(1, 2, "cols", 1.0)]


def test_rows_layout_and_weight_in_tag():
    out = be.encode_prompt_with_combinators(FakeCLIP(), "base COUPLE(rows) top | bottom :1.5")
    assert _tags(out)[1:] == [rp.region_tag(0, 2, "rows", 1.0), rp.region_tag(1, 2, "rows", 1.5)]


def test_and_inside_region_gives_entries_with_same_tag():
    out = be.encode_prompt_with_combinators(FakeCLIP(), "base COUPLE a AND a2 | b")
    assert [e[0] for e in out] == ["base", "a", "a2", "b"]
    assert _tags(out)[1] == _tags(out)[2] == rp.region_tag(0, 2, "cols", 1.0)
    assert _tags(out)[3] == rp.region_tag(1, 2, "cols", 1.0)


def test_empty_base_has_no_base_entry():
    out = be.encode_prompt_with_combinators(FakeCLIP(), "COUPLE a | b")
    assert [e[0] for e in out] == ["a", "b"]
    assert all(t is not None for t in _tags(out))


def test_empty_and_zero_regions_keep_slot_but_add_no_entry():
    out = be.encode_prompt_with_combinators(FakeCLIP(), "base COUPLE a | | c :0 | d")
    assert [e[0] for e in out] == ["base", "a", "d"]
    assert [t["index"] for t in _tags(out)[1:]] == [0, 3]
    assert all(t["count"] == 4 for t in _tags(out)[1:])


def test_base_with_break_goes_through_tokenizer_untouched():
    clip = FakeCLIP()
    out = be.encode_prompt_with_combinators(clip, "x BREAK y COUPLE a | b")
    assert out[0][0] == "x BREAK y"


def test_tags_are_independent_dicts_and_pooled_kept():
    out = be.encode_prompt_with_combinators(FakeCLIP(), "COUPLE a AND b | c")
    assert out[0][1][rp.REGION_KEY] is not out[1][1][rp.REGION_KEY]
    assert "pooled_output" in out[0][1]


def test_has_combinators_true_for_couple():
    assert be._has_combinators("x COUPLE a | b")
    assert not be._has_combinators("x couple a | b")
    assert not be._has_combinators("plain prompt")


def test_batchable_ids_hands_couple_back_as_combinator():
    assert be._batchable_ids(object(), "x COUPLE a | b", "clip_l", 0) == (None, "combinator")


@pytest.mark.parametrize("p", ["x COUPLE only one", "x COUPLE(diag) a | b", "x COUPLE a | b COUPLE c | d"])
def test_invalid_prompt_raises(p):
    with pytest.raises(rp.RegionalPromptError):
        be.encode_prompt_with_combinators(FakeCLIP(), p)


def test_plain_prompt_still_uses_single_encode():
    clip = FakeCLIP()
    out = be.encode_prompt_with_combinators(clip, "1girl, couple, solo")
    assert [e[0] for e in out] == ["1girl, couple, solo"]
    assert rp.REGION_KEY not in out[0][1]


@pytest.mark.requires_torch
def test_cat_and_avg_inside_regions():
    import torch

    class TClip(FakeCLIP):
        def encode_from_tokens(self, tokens, return_pooled=True, return_dict=True):
            return {"cond": torch.full((1, 2, 3), float(len(tokens))), "pooled_output": None}

    out = be.encode_prompt_with_combinators(TClip(), "base COUPLE aa CAT bbb | c AVG(0.5) dddd")
    assert len(out) == 3
    assert out[0][1].get(rp.REGION_KEY) is None
    assert out[1][1][rp.REGION_KEY]["index"] == 0 and out[2][1][rp.REGION_KEY]["index"] == 1
    assert out[1][0].shape[1] == 4  # CAT concatenates along the token axis
    assert torch.allclose(out[2][0], torch.full((1, 2, 3), 2.5))  # AVG 0.5 of len 1 and len 4
