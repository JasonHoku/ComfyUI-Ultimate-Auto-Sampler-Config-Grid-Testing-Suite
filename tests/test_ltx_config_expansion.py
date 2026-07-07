"""Tests for LTX field cartesian expansion."""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_utils import expand_configs


def _ltx_base():
    return {
        "model_type": "ltx_video",
        "model": "ltx_model.safetensors",
        "clip_models": ["c1.safetensors", "c2.safetensors"],
        "vae_video": "vv.safetensors",
        "vae_audio": "va.safetensors",
        "latent_upscaler": "lup.safetensors",
        "positive": "a video",
        "negative": "ugly",
        "width": 446,
        "height": 576,
        "duration_seconds": 5,
        "frame_rate": 25,
        "sampler_stage1": "euler_ancestral_cfg_pp",
        "sampler_stage2": "euler_cfg_pp",
        "sigmas_stage1": "1.0, 0.5, 0.0",
        "sigmas_stage2": "0.85, 0.4, 0.0",
        "cfg": 1.0,
        "seed": 42,
        "input_image": None,
        "image_strength_stage1": 0.8,
        "image_strength_stage2": 1.0,
        "img_compression": 18,
        "audio_mode": "on",
        "lora": "None",
    }


def _call(base):
    """Wrap expand_configs with the dummy outer args it needs for non-LTX paths."""
    return expand_configs(
        raw_configs=[base],
        pos_prompts=[base.get("positive", "")],
        neg_prompts=[base.get("negative", "")],
        denoise_values=[1.0],
        seed=base.get("seed", 0),
        extra_seeds=[],
        ckpt_name=None,
    )


def test_single_value_returns_one_config():
    result = _call(_ltx_base())
    assert len(result) == 1
    assert result[0]["model_type"] == "ltx_video"
    assert result[0]["model"] == "ltx_model.safetensors"


def test_sigmas_stage2_array_expands():
    base = _ltx_base()
    base["sigmas_stage2"] = ["0.85, 0.4, 0.0", "0.9, 0.5, 0.0"]
    result = _call(base)
    assert len(result) == 2
    sigmas_seen = sorted(c["sigmas_stage2"] for c in result)
    assert sigmas_seen == ["0.85, 0.4, 0.0", "0.9, 0.5, 0.0"]


def test_audio_mode_both_expands_to_on_off():
    base = _ltx_base()
    base["audio_mode"] = "both"
    result = _call(base)
    assert len(result) == 2
    modes_seen = sorted(c["audio_mode"] for c in result)
    assert modes_seen == ["off", "on"]


def test_audio_mode_array_accepted():
    base = _ltx_base()
    base["audio_mode"] = ["on", "off"]
    result = _call(base)
    assert len(result) == 2


def test_duration_array_expands():
    base = _ltx_base()
    base["duration_seconds"] = [5, 10]
    result = _call(base)
    assert len(result) == 2


def test_image_strength_arrays_cross_with_sigmas():
    base = _ltx_base()
    base["image_strength_stage1"] = [0.6, 0.8]
    base["sigmas_stage2"] = ["0.85, 0.4, 0.0", "0.9, 0.5, 0.0"]
    result = _call(base)
    assert len(result) == 4  # 2 x 2


def test_input_image_array_expands():
    base = _ltx_base()
    base["input_image"] = ["a.png", "b.png", "c.png"]
    result = _call(base)
    assert len(result) == 3


def test_clip_models_2tuple_does_not_expand():
    """clip_models is a [name1, name2] tuple, not 2 sweep values."""
    base = _ltx_base()
    base["clip_models"] = ["g.safetensors", "tp.safetensors"]
    result = _call(base)
    assert len(result) == 1
    assert result[0]["clip_models"] == ["g.safetensors", "tp.safetensors"]


def test_clip_models_list_of_lists_expands():
    """[[c1a, c2a], [c1b, c2b]] sweeps 2 CLIP pairs."""
    base = _ltx_base()
    base["clip_models"] = [["g.safetensors", "tp1.safetensors"], ["g.safetensors", "tp2.safetensors"]]
    result = _call(base)
    assert len(result) == 2


def test_input_image_folder_expands_to_images(tmp_path):
    """Folder-form input_image expands to one config per image file in the folder."""
    folder = tmp_path / "imgs"
    folder.mkdir()
    (folder / "a.png").write_text("dummy")
    (folder / "b.png").write_text("dummy")
    (folder / "c.png").write_text("dummy")

    base = _ltx_base()
    base["input_image"] = str(folder) + "/"
    result = _call(base)
    assert len(result) == 3
    paths = sorted(c["input_image"] for c in result)
    assert all(p.endswith((".png", ".jpg", ".jpeg", ".webp")) for p in paths)
    assert all(os.path.basename(p) in ("a.png", "b.png", "c.png") for p in paths)


def test_input_image_folder_filters_non_images(tmp_path):
    """Non-image files in the folder are skipped."""
    folder = tmp_path / "imgs"
    folder.mkdir()
    (folder / "a.png").write_text("dummy")
    (folder / "notes.txt").write_text("dummy")
    (folder / "b.jpg").write_text("dummy")

    base = _ltx_base()
    base["input_image"] = str(folder) + "/"
    result = _call(base)
    assert len(result) == 2
    extensions = sorted(os.path.splitext(c["input_image"])[1] for c in result)
    assert extensions == [".jpg", ".png"]


def test_input_image_folder_with_backslash(tmp_path):
    """Windows-style backslash separator also triggers folder expansion."""
    folder = tmp_path / "imgs"
    folder.mkdir()
    (folder / "a.png").write_text("dummy")

    base = _ltx_base()
    # Windows path with trailing backslash
    base["input_image"] = str(folder) + "\\"
    result = _call(base)
    assert len(result) == 1


def test_input_image_folder_nonexistent_passes_through(tmp_path):
    """Non-existent folder path is left as a single string (preflight will catch it later)."""
    base = _ltx_base()
    base["input_image"] = str(tmp_path / "does_not_exist") + "/"
    result = _call(base)
    # Treated as single string since the folder doesn't exist; not expanded
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Prompt field handling — Config Builder nested array format
# ---------------------------------------------------------------------------

def test_positive_plain_string_passes_through():
    """A plain string positive prompt yields exactly one config with that string."""
    base = _ltx_base()
    base["positive"] = "a cat on a rooftop"
    result = _call(base)
    assert len(result) == 1
    assert result[0]["positive"] == "a cat on a rooftop"


def test_positive_config_builder_nested_array_two_prompts():
    """Config Builder wraps prompts in [["p1", "p2"]]. Each inner string → a separate run."""
    base = _ltx_base()
    base["positive"] = [["prompt one", "prompt two"]]
    result = _call(base)
    assert len(result) == 2
    prompts = sorted(c["positive"] for c in result)
    assert prompts == ["prompt one", "prompt two"]


def test_positive_config_builder_single_prompt_in_nested_array():
    """[["single prompt"]] collapses to a single run."""
    base = _ltx_base()
    base["positive"] = [["only one prompt"]]
    result = _call(base)
    assert len(result) == 1
    assert result[0]["positive"] == "only one prompt"


def test_positive_flat_array_two_strings():
    """A flat list of two strings produces two separate runs."""
    base = _ltx_base()
    base["positive"] = ["option A", "option B"]
    result = _call(base)
    assert len(result) == 2
    prompts = sorted(c["positive"] for c in result)
    assert prompts == ["option A", "option B"]


def test_positive_is_always_a_string_after_expansion():
    """Regardless of input format, every expanded config must have a str positive."""
    base = _ltx_base()
    base["positive"] = [["p1", "p2"]]
    result = _call(base)
    for c in result:
        assert isinstance(c["positive"], str), "positive must be a plain string after expansion"


def test_negative_config_builder_nested_array():
    """Nested array negative follows the same expansion logic as positive."""
    base = _ltx_base()
    base["negative"] = [["bad quality", "blurry, ugly"]]
    result = _call(base)
    assert len(result) == 2
    negs = sorted(c["negative"] for c in result)
    assert negs == ["bad quality", "blurry, ugly"]


def test_positive_and_negative_nested_arrays_cross():
    """Two positive prompts × two negative prompts → 4 configs (Cartesian product)."""
    base = _ltx_base()
    base["positive"] = [["pos1", "pos2"]]
    base["negative"] = [["neg1", "neg2"]]
    result = _call(base)
    assert len(result) == 4
    combos = sorted((c["positive"], c["negative"]) for c in result)
    assert combos == [
        ("pos1", "neg1"),
        ("pos1", "neg2"),
        ("pos2", "neg1"),
        ("pos2", "neg2"),
    ]


def test_input_image_string_no_trailing_slash_unchanged(tmp_path):
    """A regular file path (no trailing slash) is NOT folder-expanded."""
    base = _ltx_base()
    base["input_image"] = "/tmp/single.png"
    result = _call(base)
    assert len(result) == 1
    assert result[0]["input_image"] == "/tmp/single.png"



# ---------------------------------------------------------------------------
# Prompt field handling — Config Builder nested array format
# ---------------------------------------------------------------------------

def test_positive_plain_string_passes_through():
    """A plain string positive prompt yields exactly one config with that string."""
    base = _ltx_base()
    base["positive"] = "a cat on a rooftop"
    result = _call(base)
    assert len(result) == 1
    assert result[0]["positive"] == "a cat on a rooftop"


def test_positive_config_builder_nested_array_two_prompts():
    """Config Builder wraps prompts in [["p1", "p2"]]. Each inner string → a separate run."""
    base = _ltx_base()
    base["positive"] = [["prompt one", "prompt two"]]
    result = _call(base)
    assert len(result) == 2
    prompts = sorted(c["positive"] for c in result)
    assert prompts == ["prompt one", "prompt two"]


def test_positive_config_builder_single_prompt_in_nested_array():
    """[["single prompt"]] collapses to a single run."""
    base = _ltx_base()
    base["positive"] = [["only one prompt"]]
    result = _call(base)
    assert len(result) == 1
    assert result[0]["positive"] == "only one prompt"


def test_positive_flat_array_two_strings():
    """A flat list of two strings produces two separate runs."""
    base = _ltx_base()
    base["positive"] = ["option A", "option B"]
    result = _call(base)
    assert len(result) == 2
    prompts = sorted(c["positive"] for c in result)
    assert prompts == ["option A", "option B"]


def test_positive_is_always_a_string_after_expansion():
    """Regardless of input format, every expanded config must have a str positive."""
    base = _ltx_base()
    base["positive"] = [["p1", "p2"]]
    result = _call(base)
    for c in result:
        assert isinstance(c["positive"], str), "positive must be a plain string after expansion"


def test_negative_config_builder_nested_array():
    """Nested array negative follows the same expansion logic as positive."""
    base = _ltx_base()
    base["negative"] = [["bad quality", "blurry, ugly"]]
    result = _call(base)
    assert len(result) == 2
    negs = sorted(c["negative"] for c in result)
    assert negs == ["bad quality", "blurry, ugly"]


def test_positive_and_negative_nested_arrays_cross():
    """Two positive prompts × two negative prompts → 4 configs (Cartesian product)."""
    base = _ltx_base()
    base["positive"] = [["pos1", "pos2"]]
    base["negative"] = [["neg1", "neg2"]]
    result = _call(base)
    assert len(result) == 4
    combos = sorted((c["positive"], c["negative"]) for c in result)
    assert combos == [
        ("pos1", "neg1"),
        ("pos1", "neg2"),
        ("pos2", "neg1"),
        ("pos2", "neg2"),
    ]