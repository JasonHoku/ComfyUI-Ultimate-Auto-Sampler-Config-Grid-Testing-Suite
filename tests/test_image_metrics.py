"""Exact-value tests for image_metrics (numpy reference path; torch parity skips without real torch)."""
import importlib.util
import math
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)
_spec = importlib.util.spec_from_file_location(f"{_PKG}.image_metrics_real", os.path.join(_ROOT, "image_metrics.py"))
im = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(im)

ALL = im.BUILTIN_METRICS


def _checker(h, w, block=1):
    yy, xx = np.indices((h, w))
    v = (((yy // block) + (xx // block)) % 2).astype(np.float32)
    return np.repeat(v[..., None], 3, axis=-1)


def test_flat_grey():
    m = im.compute_metrics_numpy(np.full((64, 64, 3), 0.5, np.float32))
    assert m["sharpness"] == 0 and m["colorfulness"] == 0 and m["entropy"] == 0
    assert m["contrast"] == 0 and m["saturation"] == 0
    assert m["luma"] == pytest.approx(0.5, abs=1e-4)
    assert set(m) == set(ALL)


def test_checkerboard():
    m = im.compute_metrics_numpy(_checker(64, 64))
    assert m["sharpness"] == pytest.approx(16 * 255 ** 2, rel=1e-6)   # 1040400
    assert m["entropy"] == pytest.approx(1.0, abs=1e-4)
    assert m["contrast"] == pytest.approx(0.5, abs=1e-4)
    assert m["luma"] == pytest.approx(0.5, abs=1e-4)
    assert m["colorfulness"] == 0


def test_pure_red():
    a = np.zeros((32, 32, 3), np.float32)
    a[..., 0] = 1.0
    m = im.compute_metrics_numpy(a)
    assert m["saturation"] == pytest.approx(1.0)
    assert m["colorfulness"] == pytest.approx(0.3 * math.sqrt(255 ** 2 + 127.5 ** 2), abs=1e-3)
    assert m["luma"] == pytest.approx(0.299, abs=1e-4)


def test_noise_sharper_than_blur_and_uint8_equals_float():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64, 3)).astype(np.float32)
    k = np.ones(5) / 5
    blur = a.copy()
    for ax in (0, 1):
        blur = np.apply_along_axis(lambda v: np.convolve(np.pad(v, 2, mode="edge"), k, mode="valid"), ax, blur)
    assert im.compute_metrics_numpy(a)["sharpness"] > 5 * im.compute_metrics_numpy(blur)["sharpness"]
    u8 = (a * 255).round().astype(np.uint8)
    assert im.compute_metrics_numpy(u8) == im.compute_metrics_numpy(u8.astype(np.float32) / 255.0)


def test_downscale_matches_base_and_constant_unchanged():
    big = _checker(1024, 2048, block=2)
    small = _checker(512, 1024, block=1)
    assert im.compute_metrics_numpy(big) == im.compute_metrics_numpy(small)
    c = np.full((2048, 1500, 3), 0.25, np.float32)
    m = im.compute_metrics_numpy(c)
    assert m["luma"] == pytest.approx(0.25, abs=1e-4) and m["sharpness"] == 0


def test_rounding_and_nan():
    rng = np.random.default_rng(1)
    m = im.compute_metrics_numpy(rng.random((48, 48, 3)).astype(np.float32))
    for v in m.values():
        assert round(v, 4) == v
    bad = np.full((16, 16, 3), np.nan, np.float32)
    assert im.compute_metrics_numpy(bad) == {}


def test_key_subset_and_chw():
    a = np.random.default_rng(2).random((40, 56, 3)).astype(np.float32)
    assert set(im.compute_metrics_numpy(a, ["luma", "entropy", "bogus"])) == {"luma", "entropy"}
    assert im.compute_metrics_numpy(np.transpose(a, (2, 0, 1))) == im.compute_metrics_numpy(a)


def test_normalize_spec():
    assert im.normalize_spec(None) is None
    assert im.normalize_spec({}) is None
    assert im.normalize_spec({"enabled": False, "builtins": ["luma"]}) is None
    assert im.normalize_spec({"enabled": True, "builtins": [], "adapters": []}) is None
    assert im.normalize_spec({"enabled": True, "builtins": ["bogus"]}) is None
    assert im.normalize_spec({"enabled": True, "builtins": ["luma", "bogus", "entropy"]}) == \
        {"builtins": ["entropy", "luma"], "adapters": []}
    # idempotent on a normalized spec; disabled registry entries are dropped
    s = im.normalize_spec({"enabled": True, "builtins": ["luma"], "adapters": ["aesthetic"]})
    assert s == {"builtins": ["luma"], "adapters": []}
    assert im.normalize_spec(s) == s
    assert im.normalize_spec({"enabled": True, "builtins": {"luma": True, "entropy": False}}) == \
        {"builtins": ["luma"], "adapters": []}


def test_torch_parity():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "conv2d"):
        pytest.skip("torch is stubbed")
    rng = np.random.default_rng(3)
    for shape in ((96, 128, 3), (1500, 1100, 3)):
        a = rng.random(shape).astype(np.float32)
        t = torch.from_numpy(a)[None]
        got = im.compute_metrics_torch(t)[0]
        ref = im.compute_metrics_numpy(a)
        assert set(got) == set(ref)
        for k in ref:
            assert got[k] == pytest.approx(ref[k], rel=1e-4, abs=1e-4), k
        chw = im.compute_metrics_torch(t.permute(0, 3, 1, 2))[0]
        assert chw == got
