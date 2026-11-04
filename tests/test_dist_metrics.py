"""_distributed_metrics_pass: adapters (and missing builtins) run once on the master for worker images."""
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


@pytest.fixture
def calls(monkeypatch):
    log = {"attach": [], "adapters": []}

    def attach(meta, image, **kw):
        log["attach"].append(meta["id"])
        meta["metrics"] = {"sharpness": 1.0, "aesthetic": 5.0}

    def attach_adapters(meta, image):
        log["adapters"].append(meta["id"])
        meta.setdefault("metrics", {})["aesthetic"] = 5.0

    monkeypatch.setattr(im, "attach", attach)
    monkeypatch.setattr(im, "attach_adapters", attach_adapters)
    return log


def _img(tmp_path, name):
    p = tmp_path / name
    Image.fromarray(np.full((16, 16, 3), 100, np.uint8)).save(p)
    return str(p)


def _path_for(tmp_path):
    return lambda item, d: str(tmp_path / item["filename"]) if (tmp_path / item.get("filename", "")).is_file() else None


def test_noop_without_a_metrics_run(tmp_path, calls):
    data = {"items": [{"id": 1, "filename": "a.png"}]}
    _img(tmp_path, "a.png")
    assert go._distributed_metrics_pass(data, set(), str(tmp_path), _path_for(tmp_path)) == \
        {"adapters": 0, "builtins_from_file": 0}
    assert calls == {"attach": [], "adapters": []}


def test_only_new_items_and_only_what_is_missing(tmp_path, calls):
    for n in ("old", "worker_new", "worker_old", "master", "up", "vid"):
        _img(tmp_path, f"{n}.png")
    im.begin_run({"enabled": True, "builtins": ["sharpness"], "adapters": ["aesthetic"]})
    items = [
        {"id": 1, "filename": "old.png"},                                                     # before the run
        {"id": 2, "filename": "worker_new.png", "metrics": {"sharpness": 2.0}},               # builtins from worker
        {"id": 3, "filename": "worker_old.png"},                                              # no metrics at all
        {"id": 4, "filename": "master.png", "metrics": {"sharpness": 3.0, "aesthetic": 6.0}}, # master: complete
        {"id": 5, "filename": "up.png", "upscaled": True},                                    # upscaled: skipped
        {"id": 6, "filename": "vid.png", "media_type": "video"},                              # video: skipped
    ]
    out = go._distributed_metrics_pass({"items": items}, {1}, str(tmp_path), _path_for(tmp_path))
    assert calls["adapters"] == [2]            # adapters only for the worker image that has builtins
    assert calls["attach"] == [3]              # builtins + adapters from file for the one with nothing
    assert out == {"adapters": 2, "builtins_from_file": 1}
    assert items[1]["metrics"] == {"sharpness": 2.0, "aesthetic": 5.0}   # worker's builtins kept untouched
    assert items[3]["metrics"] == {"sharpness": 3.0, "aesthetic": 6.0}


def test_builtins_only_spec_never_runs_adapters(tmp_path, calls):
    _img(tmp_path, "a.png")
    im.begin_run({"enabled": True, "builtins": ["sharpness"], "adapters": []})
    items = [{"id": 1, "filename": "a.png", "metrics": {"sharpness": 2.0}}]
    out = go._distributed_metrics_pass({"items": items}, set(), str(tmp_path), _path_for(tmp_path))
    assert calls == {"attach": [], "adapters": []} and out["adapters"] == 0


def test_missing_file_is_skipped_quietly(tmp_path, calls):
    im.begin_run({"enabled": True, "builtins": ["sharpness"], "adapters": ["aesthetic"]})
    items = [{"id": 1, "filename": "gone.png"}]
    out = go._distributed_metrics_pass({"items": items}, set(), str(tmp_path), _path_for(tmp_path))
    assert out == {"adapters": 0, "builtins_from_file": 0}
    assert "metrics" not in items[0]
