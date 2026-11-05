"""Sobel edge metric (item 25): exact values on synthetic images, numpy reference and torch parity."""
import importlib
import os

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)
im = importlib.import_module(f"{_PKG}.image_metrics")


def _gray(plane):
    return np.repeat(np.asarray(plane, dtype=np.float32)[..., None], 3, axis=-1)


def _step(lo=0.0, hi=1.0, h=64, w=64, col=32):
    p = np.full((h, w), lo, dtype=np.float32)
    p[:, col:] = hi
    return _gray(p)


def _blur5(a):
    k = np.ones(5) / 5
    out = a.copy()
    for ax in (0, 1):
        out = np.apply_along_axis(lambda v: np.convolve(np.pad(v, 2, mode="edge"), k, mode="valid"), ax, out)
    return out


def test_step_edge_vertical_and_horizontal():
    a = _step()
    assert im.compute_metrics_numpy(a, ["sobel"])["sobel"] == pytest.approx(2 / 62, abs=1e-4)
    t = np.transpose(a, (1, 0, 2)).copy()
    assert im.compute_metrics_numpy(t, ["sobel"])["sobel"] == pytest.approx(2 / 62, abs=1e-4)


def test_half_contrast_step():
    got = im.compute_metrics_numpy(_step(0.25, 0.75), ["sobel"])["sobel"]
    assert got == pytest.approx(1 / 62, abs=1e-4)


def test_flat_is_zero_and_checkerboard_is_zero_but_sharp():
    flat = _gray(np.full((32, 32), 0.5))
    assert im.compute_metrics_numpy(flat, ["sobel"])["sobel"] == 0
    yy, xx = np.indices((32, 32))
    chk = _gray(((yy + xx) % 2).astype(np.float32))
    m = im.compute_metrics_numpy(chk, ["sobel", "sharpness"])
    assert m["sobel"] == 0
    assert m["sharpness"] > 1e6


def test_noise_beats_blur_and_uint8_equals_float():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64, 3)).astype(np.float32)
    assert im.compute_metrics_numpy(a, ["sobel"])["sobel"] > im.compute_metrics_numpy(_blur5(a), ["sobel"])["sobel"]
    u8 = (a * 255).round().astype(np.uint8)
    assert im.compute_metrics_numpy(u8, ["sobel"]) == im.compute_metrics_numpy(u8.astype(np.float32) / 255.0, ["sobel"])


def test_too_small_has_no_sobel_key():
    assert "sobel" not in im.compute_metrics_numpy(np.zeros((2, 8, 3), dtype=np.float32), ["sobel", "luma"])
    assert "sobel" in im.compute_metrics_numpy(np.zeros((3, 3, 3), dtype=np.float32), ["sobel"])


def test_keys_and_spec_order():
    a = _step()
    assert set(im.compute_metrics_numpy(a, ["sobel"])) == {"sobel"}
    assert im.normalize_spec({"enabled": True, "builtins": ["sobel", "luma"]})["builtins"] == ["luma", "sobel"]


def test_downscale_invariance():
    big = _step(h=64, w=2048, col=1024)
    small = _step(h=32, w=1024, col=512)
    assert im.compute_metrics_numpy(big, ["sobel"]) == im.compute_metrics_numpy(small, ["sobel"])


def test_torch_parity():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "conv2d"):
        pytest.skip("torch is stubbed")
    rng = np.random.default_rng(5)
    cases = [_step()] + [rng.random(s).astype(np.float32) for s in ((96, 128, 3), (1500, 1100, 3))]
    for a in cases:
        got = im.compute_metrics_torch(torch.from_numpy(a)[None], ["sobel"])[0]
        ref = im.compute_metrics_numpy(a, ["sobel"])
        assert set(got) == set(ref) == {"sobel"}
        assert got["sobel"] == pytest.approx(ref["sobel"], rel=1e-4, abs=1e-4)
