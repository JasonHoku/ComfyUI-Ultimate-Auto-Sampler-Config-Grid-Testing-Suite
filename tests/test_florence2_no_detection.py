"""Tests for the no-detection short-circuit path in run_florence2_step()."""
import torch
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.requires_torch


def _make_image(w=512, h=512):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


def test_empty_mask_returns_no_detection(monkeypatch):
    """Florence2Run returns empty mask -> return dict with status=no_detection."""
    class FakeF2R:
        FUNCTION = "encode"

        def encode(self, **kwargs):
            empty_mask = torch.zeros((1, 512, 512), dtype=torch.float32)
            empty_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (empty_img, empty_mask, "", {})

    class FakeLoader:
        FUNCTION = "loadmodel"

        def loadmodel(self, **kwargs):
            return ("model_handle",)

    import nodes
    monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", {
        "Florence2Run": FakeF2R,
        "DownloadAndLoadFlorence2Model": FakeLoader,
    }, raising=False)

    import florence2_hires
    florence2_hires._FLORENCE2_MODEL_CACHE.clear()

    step_config = {
        "florence2_model": "microsoft/Florence-2-base",
        "text_input": "face",
        "output_mask_select": "",
        "target_megapixels": 1.0,
        "crop_padding": 64,
        "min_crop_resolution": 256,
        "max_crop_resolution": 1536,
        "grow_expand": 32,
        "feather_left": 128,
        "feather_top": 128,
        "feather_right": 128,
        "feather_bottom": 128,
        "model_source": "from_builder",
        "on_no_detection": "skip",
        "hires_denoise": 0.45,
        "hires_steps": 15,
        "cfg": 1.5,
        "sampler": "euler",
        "scheduler": "simple",
    }
    item = {"model": "any.safetensors"}
    fallback_handles = ("model", "clip", "vae")

    result = florence2_hires.run_florence2_step(
        source_image=_make_image(),
        item=item,
        step_config=step_config,
        fallback_handles=fallback_handles,
        ckpt_cache={},
        conditioning_cache={"positive": {}, "negative": {}},
        positive_prompt="",
        negative_prompt="",
        clip_skip=0,
    )
    assert result["status"] == "no_detection"
    assert result["florence2_text_input"] == "face"
    assert result["florence2_model"] == "microsoft/Florence-2-base"


def test_out_of_range_mask_select_returns_no_detection(monkeypatch):
    """Florence2 finds 1 region, user requests index '5' -> no_detection."""
    class FakeF2R:
        FUNCTION = "encode"

        def encode(self, **kwargs):
            # Single non-empty mask returned
            mask = torch.zeros((1, 512, 512), dtype=torch.float32)
            mask[0, 100:200, 100:200] = 1.0
            img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (img, mask, "", {})

    class FakeLoader:
        FUNCTION = "loadmodel"

        def loadmodel(self, **kwargs):
            return ("model_handle",)

    import nodes
    monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", {
        "Florence2Run": FakeF2R,
        "DownloadAndLoadFlorence2Model": FakeLoader,
    }, raising=False)

    import florence2_hires
    florence2_hires._FLORENCE2_MODEL_CACHE.clear()

    step_config = {
        "florence2_model": "microsoft/Florence-2-base",
        "text_input": "face",
        "output_mask_select": "5",  # OOR for single detection
        "target_megapixels": 1.0,
        "crop_padding": 64,
        "min_crop_resolution": 256,
        "max_crop_resolution": 1536,
        "grow_expand": 32,
        "feather_left": 128, "feather_top": 128, "feather_right": 128, "feather_bottom": 128,
        "model_source": "from_builder",
        "on_no_detection": "skip",
        "hires_denoise": 0.45, "hires_steps": 15,
        "cfg": 1.5, "sampler": "euler", "scheduler": "simple",
    }

    result = florence2_hires.run_florence2_step(
        source_image=_make_image(),
        item={"model": "any.safetensors"},
        step_config=step_config,
        fallback_handles=("model", "clip", "vae"),
        ckpt_cache={},
        conditioning_cache={"positive": {}, "negative": {}},
        positive_prompt="",
        negative_prompt="",
        clip_skip=0,
    )
    assert result["status"] == "no_detection"
