"""Latent geometry follows the model's latent format, not a hardcoded 8.

REGRESSION (2026-11): MOSS-Image uses the Flux.2 latent format (128 channels,
16 pixels per latent cell). The orchestrator built every empty latent as
``h // 8, w // 8``, so a 832x1216 job sampled a latent with 4x the cells
(the DiT ran on 4x the tokens) and decoded to 1664x2432 -- about 10x slower
than the official ComfyUI template and twice the requested size.
"""
import enum
import sys
import os
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

# tests/conftest.py registers a lightweight `model_loader` STUB (for the
# orchestrator's relative imports), so load the real module file here under a
# private name, the same way conftest loads remote_vae / distribution / civitai.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)
if not hasattr(sys.modules.get(f"{_PKG}.lora_utils", object()), "LoRAFileNotFoundError"):
    # the lora_utils stub lacks the exception class model_loader imports
    setattr(sys.modules[f"{_PKG}.lora_utils"], "LoRAFileNotFoundError", type("LoRAFileNotFoundError", (Exception,), {}))
if "comfy" in sys.modules and "comfy.sd" in sys.modules:
    sys.modules["comfy"].sd = sys.modules["comfy.sd"]  # `import comfy.sd` on stubs does not wire the attribute
_spec = importlib.util.spec_from_file_location(f"{_PKG}.model_loader_real", os.path.join(_ROOT, "model_loader.py"))
model_loader = importlib.util.module_from_spec(_spec)
model_loader.__package__ = _PKG
_spec.loader.exec_module(model_loader)
get_latent_downscale_ratio = model_loader.get_latent_downscale_ratio
empty_latent_shape = model_loader.empty_latent_shape
resolve_clip_type = model_loader.resolve_clip_type


# ---- fakes -----------------------------------------------------------------

def _model(ratio=None, via_config=False, config_name="SDXL", no_inner=False):
    """A fake ModelPatcher: .model.latent_format or .model.model_config.latent_format."""
    latent_format = types.SimpleNamespace(latent_channels=128 if ratio == 16 else 4)
    if ratio is not None:
        latent_format.spacial_downscale_ratio = ratio
    model_config = type(config_name, (), {})()
    if via_config:
        model_config.latent_format = latent_format
        inner = types.SimpleNamespace(model_config=model_config)
    else:
        inner = types.SimpleNamespace(latent_format=latent_format, model_config=model_config)
    if no_inner:
        return types.SimpleNamespace()
    return types.SimpleNamespace(model=inner)


class FakeCLIPType(enum.Enum):
    STABLE_DIFFUSION = 1
    FLUX = 6
    FLUX2 = 25
    KREA2 = 30
    MOSS_IMAGE = 38


# ---- downscale ratio ---------------------------------------------------------

def test_ratio_from_latent_format_flux2_family():
    assert get_latent_downscale_ratio(_model(ratio=16)) == 16


def test_ratio_from_model_config_latent_format():
    assert get_latent_downscale_ratio(_model(ratio=16, via_config=True)) == 16


def test_ratio_sd_family_is_8():
    assert get_latent_downscale_ratio(_model(ratio=8)) == 8


def test_ratio_defaults_to_8_when_format_has_no_attribute():
    # LatentFormat subclasses that never set the attribute (older ComfyUI)
    assert get_latent_downscale_ratio(_model(ratio=None)) == 8


def test_ratio_defaults_to_8_for_unusable_model():
    assert get_latent_downscale_ratio(_model(no_inner=True)) == 8
    assert get_latent_downscale_ratio(None) == 8


# ---- empty latent shape -------------------------------------------------------

def test_empty_latent_shape_moss_832x1216():
    # The bug: this used to be [1, 128, 152, 104] (4x the cells)
    assert empty_latent_shape(128, 832, 1216, 16) == [1, 128, 76, 52]


def test_empty_latent_shape_sdxl_unchanged():
    assert empty_latent_shape(4, 1024, 1024, 8) == [1, 4, 128, 128]


def test_empty_latent_shape_floors_like_comfy():
    assert empty_latent_shape(128, 1000, 1000, 16) == [1, 128, 62, 62]


def test_empty_latent_shape_batch_and_default_ratio():
    assert empty_latent_shape(4, 512, 768, batch_size=2) == [2, 4, 96, 64]
    assert empty_latent_shape(4, 512, 768, downscale_ratio=None) == [1, 4, 96, 64]


# ---- clip type resolution -----------------------------------------------------

def _patch_cliptype(monkeypatch):
    monkeypatch.setattr(model_loader.comfy.sd, "CLIPType", FakeCLIPType, raising=False)


def test_unknown_clip_type_falls_back_with_warning(monkeypatch, capsys):
    _patch_cliptype(monkeypatch)
    member, name = resolve_clip_type("bogus_type")
    assert member is FakeCLIPType.STABLE_DIFFUSION and name == "stable_diffusion"
    assert "Unknown clip_type 'bogus_type'" in capsys.readouterr().out


def test_explicit_clip_type_is_used_as_is(monkeypatch):
    _patch_cliptype(monkeypatch)
    member, name = resolve_clip_type("moss_image", _model(ratio=16, config_name="SDXL"))
    assert member is FakeCLIPType.MOSS_IMAGE and name == "moss_image"


def test_default_clip_type_follows_model_family(monkeypatch, capsys):
    """MOSS-Image loaded with stable_diffusion got the generic Qwen3.5 encoder
    (wrong chat template). The default now follows the detected model config."""
    _patch_cliptype(monkeypatch)
    member, name = resolve_clip_type("stable_diffusion", _model(ratio=16, config_name="MOSSImage"))
    assert member is FakeCLIPType.MOSS_IMAGE and name == "moss_image"
    assert "MOSSImage" in capsys.readouterr().out


def test_default_clip_type_stays_for_sd_models(monkeypatch):
    _patch_cliptype(monkeypatch)
    for family in ("SDXL", "SD15", "ZImage"):
        member, name = resolve_clip_type("stable_diffusion", _model(ratio=8, config_name=family))
        assert member is FakeCLIPType.STABLE_DIFFUSION and name == "stable_diffusion"


def test_explicit_non_default_wins_over_model_family(monkeypatch):
    _patch_cliptype(monkeypatch)
    member, name = resolve_clip_type("flux2", _model(ratio=16, config_name="MOSSImage"))
    assert member is FakeCLIPType.FLUX2 and name == "flux2"


def test_empty_clip_type_means_default(monkeypatch):
    _patch_cliptype(monkeypatch)
    assert resolve_clip_type("")[1] == "stable_diffusion"
    assert resolve_clip_type(None)[1] == "stable_diffusion"
    assert resolve_clip_type(None, _model(ratio=16, config_name="Krea2"))[1] == "krea2"
