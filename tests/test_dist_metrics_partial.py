"""_distributed_metrics_pass: a worker that omitted a builtin (older USCG, no sobel) gets just that key from the file."""
import importlib
import os
import sys

import numpy as np
import pytest
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_PKG = os.path.basename(_ROOT)

import generation_orchestrator as go  # preloaded by tests/conftest.py

im = importlib.import_module(f"{_PKG}.image_metrics")


@pytest.fixture(autouse=True)
def _clean():
    im._stack.clear()
    yield
    im._stack.clear()


def test_missing_builtin_filled_from_file_without_touching_existing(tmp_path, monkeypatch):
    arr = np.zeros((16, 16, 3), np.uint8)
    arr[:, 8:] = 255
    Image.fromarray(arr).save(tmp_path / "w.png")
    monkeypatch.setattr(im, "current_spec", lambda: {"builtins": ["sharpness", "sobel"], "adapters": []})
    items = [{"id": 5, "filename": "w.png", "metrics": {"sharpness": 2.0}}]
    path_for = lambda item, d: str(tmp_path / item["filename"])
    out = go._distributed_metrics_pass({"items": items}, set(), str(tmp_path), path_for)
    assert out == {"adapters": 0, "builtins_from_file": 1}
    assert items[0]["metrics"]["sharpness"] == 2.0
    assert items[0]["metrics"]["sobel"] > 0
