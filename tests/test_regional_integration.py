"""Issue #11 (COUPLE regional prompts): wiring guards.

The mask maths and the parser live in regional_prompt.py (tested in
test_regional_prompt.py). These tests make sure every sampling path, every
text-appending site and every validation point actually uses it.
"""
import importlib.util
import os
import re
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def _src(name):
    with open(os.path.join(_ROOT, name), encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n")


def _func_body(src, name):
    m = re.search(rf"^def {name}\(.*?(?=^(?:def |class |@)|\Z)", src, re.S | re.M)
    assert m, f"{name} not found"
    return m.group(0)


def test_generate_image_masks_both_conds_before_sampling():
    body = _func_body(_src("image_generation.py"), "generate_image")
    assert "region_frame=None" in body.split("):", 1)[0]
    pos = body.index("apply_regional_masks(positive_conditioning")
    neg = body.index("apply_regional_masks(negative_conditioning")
    assert "region_frame" in body[pos:pos + 200] and "region_frame" in body[neg:neg + 200]
    flux = body.index("Flux Guidance")
    t0 = body.index("t0 = time.time()")
    assert flux < pos < t0 and flux < neg < t0
    for sampler in ("common_ksampler", "guider.sample"):
        if sampler in body:
            assert body.index(sampler) > max(pos, neg), sampler


def test_tiled_hires_passes_tile_frame():
    body = _func_body(_src("image_generation.py"), "tiled_hires_sample")
    assert "region_frame=(y_start / lat_h, x_start / lat_w, y_end / lat_h, x_end / lat_w)" in body


def test_florence2_hires_passes_crop_frame():
    src = _src("florence2_hires.py")
    call = src[src.index("hires_latent_dict, _ = generate_image("):]
    call = call[:call.index("\n    )")]
    assert "region_frame=" in call


@pytest.mark.parametrize("fname", ["generation_orchestrator.py", "upscale_runner.py"])
def test_every_append_end_branch_uses_append_to_base(fname):
    lines = _src(fname).split("\n")
    hits = [i for i, l in enumerate(lines) if '== "append_end"' in l]
    assert hits, "no append_end branches found"
    for i in hits:
        assert "append_to_base(" in lines[i + 1], f"{fname}:{i + 2} appends without append_to_base"
    # the old hand-rolled concatenation must be gone
    assert not re.search(r'\+ " " \+ hires_(prompt_)?text(_rt)?\b', "\n".join(lines))


def test_orchestrator_validates_regional_prompts_in_both_places():
    src = _src("generation_orchestrator.py")
    assert "validate_regional_prompts" in src.split("def ", 1)[0]  # imported at module top
    main = src.index("collect_unique_prompts_with_triggers(\n                expanded")
    assert "validate_regional_prompts(" in src[main:main + 1200]
    body = _func_body(src, "_preencode_all_conditionings")
    assert "validate_regional_prompts(" in body
    assert body.index("validate_regional_prompts(") < body.index("load_model_by_type(")


def test_ltx_branch_warns_about_couple():
    src = _src("generation_orchestrator.py")
    i = src.index("from .ltx_video_generation import preflight_ltx")
    assert "COUPLE regional prompts are not supported for LTX video" in src[i:i + 800]


def test_trigger_words_and_metrics_use_helpers():
    tw = _src("trigger_words.py")
    assert tw.count("join_prompt_parts(") >= 3
    assert '", ".join(parts)' not in tw
    assert "plain_prompt_text" in _src("image_metrics.py")


@pytest.fixture
def ig(monkeypatch):
    import torch
    if not hasattr(torch, "inference_mode"):
        monkeypatch.setattr(torch, "inference_mode", lambda *a, **k: (lambda f: f), raising=False)
    pkg = os.path.basename(_ROOT)
    spec = importlib.util.spec_from_file_location(f"{pkg}.image_generation_real", os.path.join(_ROOT, "image_generation.py"))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"image_generation.py not importable under the stubs: {e}")
    return mod


def test_attn_safe_forwards_region_frame(ig, monkeypatch):
    seen = []

    def fake_generate(*args, **kwargs):
        seen.append(kwargs)
        return "lat", 1.0

    monkeypatch.setattr(ig, "generate_image", fake_generate)
    frame = (0.0, 0.5, 1.0, 1.0)
    out = ig.generate_image_attn_safe(1, 2, attention_mode="default", region_frame=frame)
    assert out == ("lat", 1.0, None)
    assert seen[0]["region_frame"] == frame
