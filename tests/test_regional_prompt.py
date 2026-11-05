"""COUPLE regional prompt parser, join helpers and mask builder (regional_prompt.py).

Parser/join tests are pure Python (stub safe). Mask tests need real torch.
"""
import importlib
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)
sys.path.insert(0, _ROOT)
rp = importlib.import_module(f"{_PKG}.regional_prompt")


# ---------------------------------------------------------------- parser

def test_basic_two_way_split():
    s = rp.parse_regional_prompt("2girls, park COUPLE red hair | blue hair")
    assert s.base == "2girls, park"
    assert s.axis == "cols"
    assert [(r.text, r.weight) for r in s.regions] == [("red hair", 1.0), ("blue hair", 1.0)]


def test_no_marker_returns_none():
    assert rp.parse_regional_prompt("just a prompt | with pipe") is None
    assert rp.has_regions("just a prompt") is False


def test_base_empty_or_missing():
    for p in ("COUPLE a | b", "  COUPLE a | b"):
        s = rp.parse_regional_prompt(p)
        assert s.base == ""
        assert [r.text for r in s.regions] == ["a", "b"]


@pytest.mark.parametrize("layout,axis", [
    ("(cols)", "cols"), ("(COLS)", "cols"), ("(v)", "cols"), ("(vertical)", "cols"), ("(columns)", "cols"),
    ("(rows)", "rows"), ("(h)", "rows"), ("(Horizontal)", "rows"), ("()", "cols"), ("", "cols"),
])
def test_layout_aliases(layout, axis):
    assert rp.parse_regional_prompt(f"x COUPLE{layout} a | b").axis == axis


def test_layout_with_spaces_inside_parens():
    assert rp.parse_regional_prompt("x COUPLE( rows ) a | b").axis == "rows"


def test_region_weights():
    s = rp.parse_regional_prompt("x COUPLE a :0.8 | b :0 | c")
    assert [(r.text, r.weight) for r in s.regions] == [("a", 0.8), ("b", 0.0), ("c", 1.0)]
    s = rp.parse_regional_prompt("x COUPLE a | b :2")
    assert s.regions[1].weight == 2.0


def test_weight_needs_space_before_colon():
    s = rp.parse_regional_prompt("x COUPLE (a:1.2) | b")
    assert s.regions[0].text == "(a:1.2)" and s.regions[0].weight == 1.0


def test_weight_belongs_to_region_with_and():
    s = rp.parse_regional_prompt("x COUPLE a AND b :0.5 | c")
    assert s.regions[0].text == "a AND b" and s.regions[0].weight == 0.5


@pytest.mark.parametrize("br", ["()", "[]", "{}", "<>"])
def test_pipe_inside_brackets_not_split(br):
    o, c = br
    s = rp.parse_regional_prompt(f"x COUPLE {o}a|b{c} tail | right")
    assert len(s.regions) == 2
    assert s.regions[0].text == f"{o}a|b{c} tail"


def test_nested_brackets():
    s = rp.parse_regional_prompt("x COUPLE ((a|b)|c) | d")
    assert [r.text for r in s.regions] == ["((a|b)|c)", "d"]


def test_dangling_break_and_commas_removed():
    s = rp.parse_regional_prompt("park, BREAK COUPLE BREAK a, | , b BREAK")
    assert s.base == "park"
    assert s.regions[0].text == "a"
    assert s.regions[1].text == "b BREAK"  # only a LEADING BREAK of a region is dangling


def test_base_only_break_vanishes():
    assert rp.parse_regional_prompt("BREAK COUPLE a | b").base == ""
    assert rp.parse_regional_prompt(", BREAK COUPLE a | b").base == ""


def test_inner_break_and_combinators_kept_verbatim():
    s = rp.parse_regional_prompt("x BREAK y COUPLE a AND b | c CAT d AVG(0.3) e BREAK f")
    assert s.base == "x BREAK y"
    assert s.regions[0].text == "a AND b"
    assert s.regions[1].text == "c CAT d AVG(0.3) e BREAK f"


def test_lowercase_couple_is_plain_text():
    p = "1girl, couple, holding hands | not a split"
    assert rp.parse_regional_prompt(p) is None
    assert not rp.has_regions(p)


@pytest.mark.parametrize("p", ["aCOUPLE a | b", "COUPLEx a | b", "x COUPLES a | b", "x(COUPLE) a | b"])
def test_couple_must_be_its_own_word(p):
    assert rp.parse_regional_prompt(p) is None


def test_marker_at_end_is_an_error():
    with pytest.raises(rp.RegionalPromptError):
        rp.parse_regional_prompt("x COUPLE")


def test_four_regions_ok():
    assert len(rp.parse_regional_prompt("COUPLE a | b | c | d").regions) == 4


def test_empty_region_keeps_slot_with_base():
    s = rp.parse_regional_prompt("base COUPLE a | | c")
    assert [r.text for r in s.regions] == ["a", "", "c"]


def test_zero_weight_region_with_base_ok():
    s = rp.parse_regional_prompt("base COUPLE a | b :0")
    assert s.regions[1].weight == 0.0


# ---------------------------------------------------------------- errors

@pytest.mark.parametrize("p", [
    "x COUPLE only one region",
    "x COUPLE a | b | c | d | e",
    "x COUPLE(diag) a | b",
    "x COUPLE(cols, 0.5) a | b",
    "x COUPLE(cols a | b",
    "x COUPLE a | b COUPLE c | d",
    "COUPLE a | b COUPLE c",
    "x COUPLE | ",
    "x COUPLE  |  |  ",
    "x COUPLE a :0 | b :0",
    "COUPLE a | ",
    "COUPLE a | | c",
    "COUPLE a | b :0",
])
def test_errors(p):
    with pytest.raises(rp.RegionalPromptError):
        rp.parse_regional_prompt(p)


def test_error_is_valueerror():
    assert issubclass(rp.RegionalPromptError, ValueError)


def test_validate_lists_every_bad_prompt():
    bad1 = "x COUPLE only"
    bad2 = "y COUPLE(zz) a | b " + "z" * 100
    good = "x COUPLE a | b"
    with pytest.raises(rp.RegionalPromptError) as ei:
        rp.validate_regional_prompts({bad1, bad2, good, "plain prompt"})
    msg = str(ei.value)
    assert "x COUPLE only" in msg and "y COUPLE(zz)" in msg
    assert good not in msg.replace("x COUPLE only", "")
    assert "z" * 90 not in msg  # cut to 80 chars
    assert "(2)" in msg


def test_validate_ok_and_empty():
    rp.validate_regional_prompts(["a", "b COUPLE c | d"])
    rp.validate_regional_prompts([])
    rp.validate_regional_prompts(None)


# --------------------------------------------------------- join helpers

_PREFIXES = [None, "trigger, words"]
_SUFFIXES = [None, "masterpiece, best quality"]


def _old_join(parts):
    return ", ".join(parts)


@pytest.mark.parametrize("body", ["1girl, solo", "a AND b BREAK c", "lowercase couple | pipe", "", "x COUPLEy a | b"])
@pytest.mark.parametrize("prefix", _PREFIXES)
@pytest.mark.parametrize("suffix", _SUFFIXES)
def test_join_prompt_parts_identical_without_couple(body, prefix, suffix):
    parts = ([prefix] if prefix else []) + [body] + ([suffix] if suffix else [])
    idx = 1 if prefix else 0
    assert rp.join_prompt_parts(parts, idx) == _old_join(parts)
    assert rp.join_prompt_parts(parts, idx, sep=" ") == " ".join(parts)


def test_join_prompt_parts_out_of_range_index():
    assert rp.join_prompt_parts(["a", "b"], 5) == "a, b"
    assert rp.join_prompt_parts([], 0) == ""


@pytest.mark.parametrize("prefix", _PREFIXES)
@pytest.mark.parametrize("suffix", _SUFFIXES)
def test_join_prompt_parts_with_couple_puts_text_in_base(prefix, suffix):
    body = "park, sunny COUPLE(rows) red | blue :1.3"
    parts = ([prefix] if prefix else []) + [body] + ([suffix] if suffix else [])
    idx = 1 if prefix else 0
    out = rp.join_prompt_parts(parts, idx)
    exp_base = ", ".join(([prefix] if prefix else []) + ["park, sunny"] + ([suffix] if suffix else []))
    assert out == exp_base + " COUPLE(rows) red | blue :1.3"
    s = rp.parse_regional_prompt(out)
    assert s.base == exp_base and s.axis == "rows"
    assert [(r.text, r.weight) for r in s.regions] == [("red", 1.0), ("blue", 1.3)]


def test_join_prompt_parts_with_empty_base():
    out = rp.join_prompt_parts(["trig", "COUPLE a | b", "suffix"], 1)
    assert out == "trig, suffix COUPLE a | b"
    assert rp.join_prompt_parts(["COUPLE a | b"], 0) == "COUPLE a | b"
    assert rp.join_prompt_parts(["COUPLE(h) a | b"], 0) == "COUPLE(h) a | b"


@pytest.mark.parametrize("prompt,text", [
    ("1girl, solo", "detailed"), ("", "x"), ("a AND b", "c CAT d"), ("couple | x", "y"), ("p", ""),
])
def test_append_to_base_identical_without_couple(prompt, text):
    assert rp.append_to_base(prompt, text) == prompt + " " + text


def test_append_to_base_goes_into_base():
    assert rp.append_to_base("park COUPLE a | b", "extra") == "park extra COUPLE a | b"
    assert rp.append_to_base("COUPLE a | b", "extra") == "extra COUPLE a | b"
    assert rp.append_to_base("park BREAK COUPLE a | b", "extra") == "park extra COUPLE a | b"


def test_plain_prompt_text():
    assert rp.plain_prompt_text("park COUPLE(rows) red hair :1.3 | blue hair") == "park, red hair, blue hair"
    assert rp.plain_prompt_text("x COUPLE a | | b") == "x, a, b"
    assert rp.plain_prompt_text("no couple here") == "no couple here"
    assert rp.plain_prompt_text("x COUPLE only") == "x COUPLE only"  # invalid: unchanged


def test_region_tag_is_json_safe():
    import json
    t = rp.region_tag(1, 3, "rows", 1.5)
    assert t == {"v": 1, "index": 1, "count": 3, "axis": "rows", "weight": 1.5}
    assert json.loads(json.dumps(t)) == t


def test_apply_regional_masks_without_tags_needs_no_torch():
    cond = [["t", {"pooled_output": None}]]
    assert rp.apply_regional_masks(cond, (1, 4, 8, 8)) is cond
    assert rp.apply_regional_masks(None, (1, 4, 8, 8)) is None


# ------------------------------------------------------------------ masks

@pytest.mark.requires_torch
@pytest.mark.parametrize("axis", ["cols", "rows"])
@pytest.mark.parametrize("count,h,w", [(2, 8, 10), (3, 7, 9), (4, 6, 5), (3, 104, 152)])
def test_build_region_masks_partition(axis, count, h, w):
    import torch
    m = rp.build_region_masks(count, axis, h, w)
    assert m.shape == (count, h, w) and m.dtype == torch.float32
    assert set(m.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(m.sum(0), torch.ones(h, w))
    if axis == "cols":
        assert torch.equal(m, m[:, :1, :].expand(count, h, w))
    else:
        assert torch.equal(m, m[:, :, :1].expand(count, h, w))


@pytest.mark.requires_torch
def test_build_region_masks_order_and_odd_width():
    m = rp.build_region_masks(3, "cols", 2, 7)
    idx = m.argmax(0)[0].tolist()
    assert idx == sorted(idx)  # left to right
    assert idx[0] == 0 and idx[-1] == 2
    r = rp.build_region_masks(2, "rows", 6, 3)
    assert r[0, :3].all() and r[1, 3:].all()


@pytest.mark.requires_torch
def test_build_region_masks_frame():
    m = rp.build_region_masks(2, "cols", 4, 8, frame=(0.0, 0.5, 1.0, 1.0))  # right half of the image
    assert not m[0].any() and m[1].all()
    m = rp.build_region_masks(2, "cols", 4, 8, frame=(0.0, 0.25, 1.0, 0.75))  # straddles the split
    assert m[0, :, :4].all() and m[1, :, 4:].all()
    m = rp.build_region_masks(2, "rows", 8, 4, frame=(0.5, 0.0, 1.0, 1.0))  # bottom half
    assert not m[0].any() and m[1].all()


@pytest.mark.requires_torch
def test_apply_regional_masks_4d():
    import torch
    base = ["tb", {"pooled_output": None}]
    r0 = ["t0", {"pooled_output": None, rp.REGION_KEY: rp.region_tag(0, 2, "cols", 1.0)}]
    r1 = ["t1", {"pooled_output": None, rp.REGION_KEY: rp.region_tag(1, 2, "cols", 1.3)}]
    cond = [base, r0, r1]
    snapshot = [(e[0], dict(e[1])) for e in cond]
    out = rp.apply_regional_masks(cond, torch.Size([1, 4, 10, 12]))
    assert [e[0] for e in out] == ["tb", "t0", "t1"]
    assert "mask" not in out[0][1]
    for e, w in ((out[1], 1.0), (out[2], 1.3)):
        d = e[1]
        assert d["mask"].shape == (1, 10, 12) and d["mask"].dtype == torch.float32
        assert d["mask_strength"] == w and d["set_area_to_bounds"] is False
        assert rp.REGION_KEY not in d
    assert torch.equal(out[1][1]["mask"] + out[2][1]["mask"], torch.ones(1, 10, 12))
    for (t, d), e in zip(snapshot, cond):  # input untouched
        assert e[0] == t and e[1] == d


@pytest.mark.requires_torch
def test_apply_regional_masks_5d_latent():
    import torch
    r0 = ["t0", {rp.REGION_KEY: rp.region_tag(0, 2, "rows", 1.0)}]
    r1 = ["t1", {rp.REGION_KEY: rp.region_tag(1, 2, "rows", 1.0)}]
    out = rp.apply_regional_masks([r0, r1], torch.Size([1, 16, 3, 8, 6]))
    assert out[0][1]["mask"].shape == (1, 3, 8, 6)
    assert torch.equal(out[0][1]["mask"] + out[1][1]["mask"], torch.ones(1, 3, 8, 6))


@pytest.mark.requires_torch
def test_apply_regional_masks_drops_entries_outside_frame_and_keeps_and_entries():
    import torch
    base = ["tb", {}]
    a = rp.region_tag(0, 2, "cols", 1.0)
    b = rp.region_tag(1, 2, "cols", 1.0)
    cond = [base, ["a1", {rp.REGION_KEY: a}], ["a2", {rp.REGION_KEY: a}], ["b1", {rp.REGION_KEY: b}]]
    out = rp.apply_regional_masks(cond, torch.Size([1, 4, 8, 8]), frame=(0.0, 0.5, 1.0, 1.0))
    assert [e[0] for e in out] == ["tb", "b1"]
    out = rp.apply_regional_masks(cond, torch.Size([1, 4, 8, 8]))
    assert [e[0] for e in out] == ["tb", "a1", "a2", "b1"]


@pytest.mark.requires_torch
def test_apply_regional_masks_unchanged_object_without_tags():
    import torch
    cond = [["t", {"pooled_output": None}]]
    assert rp.apply_regional_masks(cond, torch.Size([1, 4, 8, 8])) is cond
