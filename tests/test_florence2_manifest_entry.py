"""Tests for build_florence2_manifest_entry and build_florence2_no_detection_entry."""
import pytest


def test_manifest_entry_carries_florence2_fields():
    from florence2_hires import build_florence2_manifest_entry

    step_result = {
        "status": "ok",
        "image_width": 1024,
        "image_height": 1024,
        "duration": 6.34,
        "florence2_model": "microsoft/Florence-2-base",
        "florence2_text_input": "face",
        "florence2_target_megapixels": 1.0,
        "florence2_crop_padding": 64,
        "florence2_grow_expand": 32,
        "florence2_feather": "128/128/128/128",
        "florence2_output_mask_select": "",
        "florence2_model_source": "from_manifest",
        "florence2_detection_count": 1,
        "florence2_bbox": [342, 218, 318, 412],
    }
    item = {
        "id": 100,
        "model": "novaAnimeXL_ilV160.safetensors",
        "lora": "Sexy-IL-v11:1.0",
        "lora_expanded": "Sexy-IL-v11.safetensors:1.0:1.0",
        "positive": "masterpiece, 1girl, face",
        "negative": "worst quality",
        "vae": "",
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 10,
        "cfg": 1.5,
        "seed": 645173509154809,
    }
    entry = build_florence2_manifest_entry(
        step_result, item,
        session_name="my_session",
        pipeline_name="Pipeline 1",
        upscale_id=1747584382000,
        upscaled_filename="img_1747584382000_upscaled.webp",
        current_index=47,
        hires_denoise=0.65,
    )

    assert entry["id"] == 1747584382000
    assert entry["gen_index"] == 47
    assert entry["filename"] == "img_1747584382000_upscaled.webp"
    assert "benchmarks/my_session/images" in entry["file"]
    assert entry["width"] == 1024 and entry["height"] == 1024
    assert entry["upscaled"] is True
    assert entry["upscale_source"] == "dashboard"
    assert entry["upscale_pipeline"] == "Pipeline 1"
    assert entry["upscale_mode"] == "florence2_hires"
    assert entry["florence2_model"] == "microsoft/Florence-2-base"
    assert entry["florence2_text_input"] == "face"
    assert entry["florence2_target_megapixels"] == 1.0
    assert entry["florence2_detection_count"] == 1
    assert entry["florence2_bbox"] == [342, 218, 318, 412]
    assert entry["hires_denoise"] == 0.65
    # Carried-through item fields
    assert entry["model"] == "novaAnimeXL_ilV160.safetensors"
    assert entry["lora"] == "Sexy-IL-v11:1.0"
    assert entry["positive"] == "masterpiece, 1girl, face"
    assert entry["seed"] == 645173509154809


def test_manifest_entry_omits_inherited_upscale_keys():
    """Source item's old upscale_mode etc. shouldn't leak into the new entry."""
    from florence2_hires import build_florence2_manifest_entry

    step_result = {
        "image_width": 512, "image_height": 512, "duration": 1.0,
        "florence2_model": "microsoft/Florence-2-base",
        "florence2_text_input": "face",
    }
    item = {
        "id": 99,
        "model": "ckpt.safetensors",
        "upscaled": True,
        "upscale_mode": "model_then_hires",
        "upscale_ratio": 2.0,
        "upscale_denoise": 0.3,
        "upscale_model": "old_upscaler.safetensors",
    }
    entry = build_florence2_manifest_entry(
        step_result, item, session_name="s", pipeline_name="p",
        upscale_id=1, upscaled_filename="img.webp", current_index=0,
        hires_denoise=0.5,
    )

    # The new entry's upscale_mode should be florence2_hires, not the inherited one
    assert entry["upscale_mode"] == "florence2_hires"
    # Old upscale_* keys should NOT be present
    assert "upscale_ratio" not in entry
    assert "upscale_denoise" not in entry
    assert "upscale_model" not in entry


def test_no_detection_entry_marks_no_detection_flag():
    from florence2_hires import build_florence2_no_detection_entry

    step_result = {
        "status": "no_detection",
        "florence2_text_input": "face",
        "florence2_model": "microsoft/Florence-2-base",
    }
    item = {
        "id": 100,
        "filename": "img_orig.webp",
        "file": "/view?filename=img_orig.webp&type=output",
    }
    entry = build_florence2_no_detection_entry(
        step_result, item, sentinel_id=1747584400000, current_index=50
    )

    assert entry["id"] == 1747584400000
    assert entry["filename"] == "img_orig.webp"  # points to original
    assert entry["upscaled"] is False
    assert entry["florence2_no_detection"] is True
    assert entry["florence2_text_input"] == "face"
    assert "no 'face'" in entry["note"]


def test_no_detection_entry_does_not_carry_upscale_metadata():
    """No-detection entry is a sentinel — no width/height/duration."""
    from florence2_hires import build_florence2_no_detection_entry

    step_result = {
        "florence2_text_input": "dragon",
        "florence2_model": "microsoft/Florence-2-base",
    }
    item = {"id": 1, "filename": "x.webp", "file": "x.webp"}
    entry = build_florence2_no_detection_entry(
        step_result, item, sentinel_id=2, current_index=0
    )
    # Sentinel entries don't have width/height/duration
    assert "width" not in entry
    assert "height" not in entry
    assert "duration" not in entry
    # They don't claim to be an upscale output
    assert entry["upscaled"] is False
    assert "upscale_mode" not in entry


def test_base_dims_kwarg_lands_in_both_builders():
    from florence2_hires import build_florence2_manifest_entry, build_florence2_no_detection_entry

    step = {"status": "ok", "image_width": 1536, "image_height": 1536, "florence2_text_input": "face"}
    kw = dict(session_name="s", pipeline_name="P", upscale_id=1, upscaled_filename="f.webp",
              current_index=0, hires_denoise=0.5)
    item = {"id": 9, "file": "/view?x", "filename": "x.webp"}
    plain = build_florence2_manifest_entry(step, item, **kw)
    assert "base_width" not in plain and "base_height" not in plain
    dims = {"base_width": 1024, "base_height": 1024}
    withd = build_florence2_manifest_entry(step, item, base_dims=dims, **kw)
    assert withd["base_width"] == 1024 and withd["base_height"] == 1024
    assert {k: v for k, v in withd.items() if not k.startswith("base_")} == plain

    s_plain = build_florence2_no_detection_entry(step, item, sentinel_id=3, current_index=0)
    assert "base_width" not in s_plain
    s_dims = build_florence2_no_detection_entry(step, item, sentinel_id=3, current_index=0, base_dims=dims)
    assert s_dims["base_width"] == 1024 and s_dims["florence2_no_detection"] is True
