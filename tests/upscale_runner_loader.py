"""Load the REAL upscale_runner.py under the test stubs (the root conftest registers a stub module of the same
name, and the sibling stubs lack upscale_image / seedvr2_upscale)."""
import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)
_CACHE = {}


def load_runner():
    if "mod" in _CACHE:
        return _CACHE["mod"]
    ig = sys.modules.get(f"{_PKG}.image_generation")
    if ig is not None:
        for name in ("upscale_image", "seedvr2_upscale", "decode_latent_with_vae", "create_image_metadata"):
            if not hasattr(ig, name):
                setattr(ig, name, lambda *a, **k: None)
    spec = importlib.util.spec_from_file_location(f"{_PKG}.upscale_runner_real", os.path.join(_ROOT, "upscale_runner.py"))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _CACHE["mod"] = mod
    return mod
