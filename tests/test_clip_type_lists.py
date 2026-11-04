"""The Builder's CLIP Type dropdown is fed by the installed ComfyUI's loaders.

The list used to be a hardcoded copy of ComfyUI's CLIPLoader options, so every
new model family (Krea 2 in PR #25, then MOSS-Image) needed a plugin release
before it could be selected. Now the static list is only the fallback.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nodes
from config_builder_node import get_clip_type_lists, _FALLBACK_CLIP_TYPES, _FALLBACK_DUAL_CLIP_TYPES


class _Loader:
    def __init__(self, types_):
        self._types = types_

    def INPUT_TYPES(self):
        return {"required": {"clip_name": (["a.safetensors"],), "type": (self._types,)}}


def test_fallback_has_moss_image_and_keeps_stable_diffusion_first():
    assert _FALLBACK_CLIP_TYPES[0] == "stable_diffusion"
    assert "moss_image" in _FALLBACK_CLIP_TYPES
    assert "krea2" in _FALLBACK_CLIP_TYPES
    assert "sdxl" in _FALLBACK_DUAL_CLIP_TYPES


def test_without_comfy_loaders_returns_fallback(monkeypatch):
    monkeypatch.delattr(nodes, "NODE_CLASS_MAPPINGS", raising=False)
    clip_types, dual = get_clip_type_lists()
    assert clip_types == list(_FALLBACK_CLIP_TYPES)
    assert dual == list(_FALLBACK_DUAL_CLIP_TYPES)


def test_live_loader_types_are_appended_without_duplicates(monkeypatch):
    monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", {
        "CLIPLoader": _Loader(["stable_diffusion", "moss_image", "zz_future_type"]),
        "DualCLIPLoader": _Loader(["sdxl", "zz_future_dual"]),
    }, raising=False)
    clip_types, dual = get_clip_type_lists()
    assert clip_types[0] == "stable_diffusion"
    assert clip_types.count("moss_image") == 1
    assert "zz_future_type" in clip_types
    assert dual.count("sdxl") == 1 and "zz_future_dual" in dual


def test_broken_loader_falls_back_quietly(monkeypatch):
    monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", {"CLIPLoader": object()}, raising=False)
    clip_types, _ = get_clip_type_lists()
    assert "moss_image" in clip_types
