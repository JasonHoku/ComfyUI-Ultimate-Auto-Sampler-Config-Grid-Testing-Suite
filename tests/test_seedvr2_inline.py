"""Inline SeedVR2 step: decode_latent_with_vae(vae, samples) returns a PIL image and
the chained re-encode needs a BHWC float tensor (VAE.encode layout, no movedim)."""
import importlib.util
import inspect
import os
import re
import sys

import numpy as np
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def _load_helper():
    src = open(os.path.join(_ROOT, "image_generation.py"), encoding="utf-8").read()
    m = re.search(r"^def pil_to_image_array\(.*?(?=^(?:def |@|#))", src, re.S | re.M)  # stop at the next top-level def/decorator/comment
    ns = {}
    exec(m.group(0), ns)
    return ns["pil_to_image_array"]


def test_pil_to_image_array_is_bhwc_float01():
    f = _load_helper()
    img = Image.new("RGB", (5, 3), (255, 0, 51))
    arr = f(img)
    assert arr.shape == (1, 3, 5, 3) and arr.dtype == np.float32
    assert np.allclose(arr[0, 0, 0], [1.0, 0.0, 0.2])


def test_pil_to_image_array_converts_non_rgb():
    arr = _load_helper()(Image.new("RGBA", (4, 2), (0, 255, 0, 128)))
    assert arr.shape == (1, 2, 4, 3)


def test_orchestrator_seedvr2_branch_call_shape():
    src = open(os.path.join(_ROOT, "generation_orchestrator.py"), encoding="utf-8").read()
    i = src.index('if mode == "seedvr2":')
    branch = src[i:i + 6000]
    assert 'decode_latent_with_vae(loaded_vae, pipe_latent["samples"])' in branch
    assert "movedim" not in branch
    assert "dict(meta)" not in branch
