"""Phase 2: background 'compute metrics for this session' job."""
import importlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pytest
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)
mr = importlib.import_module(f"{_PKG}.metrics_runner")
# conftest registers a lock-less STUB manifest_utils; load the real one for the runner's lock
_mu_spec = importlib.util.spec_from_file_location(f"{_PKG}.manifest_utils_real", os.path.join(_ROOT, "manifest_utils.py"))
_mu = importlib.util.module_from_spec(_mu_spec)
_mu.__package__ = _PKG
_mu_spec.loader.exec_module(_mu)
im = importlib.import_module(f"{_PKG}.image_metrics")


def _save(path, value, size=(40, 30)):
    Image.fromarray(np.full((size[1], size[0], 3), value, np.uint8)).save(path)


@pytest.fixture
def session(tmp_path, monkeypatch):
    out = tmp_path / "out"
    sdir = out / "benchmarks" / "s1"
    (sdir / "images").mkdir(parents=True)
    monkeypatch.setattr(sys.modules["folder_paths"], "get_output_directory", lambda: str(out), raising=False)
    for n, v in (("a.webp", 10), ("b.webp", 200), ("c.png", 90)):
        _save(sdir / "images" / n, v)
    outside = tmp_path / "secret.png"
    _save(outside, 50)
    items = [
        {"id": 1, "file": "/view?filename=a.webp&type=output&subfolder=benchmarks/s1/images", "favorited": True, "note": "keep"},
        {"id": 2, "file": "/view?filename=b.webp&type=output&subfolder=benchmarks/s1/images", "metrics": {"luma": 9.0}},
        {"id": 3, "filename": "c.png", "file": "/view?filename=c.png&type=output"},
        {"id": 4, "file": "/view?filename=a.webp", "media_type": "video"},
        {"id": 5, "file": "/view?filename=..%2F..%2F..%2Fsecret.png"},
        {"id": 6, "filename": "../../../secret.png"},
        {"id": 7, "filename": "missing.webp"},
        {"id": 8, "florence2_no_detection": True, "filename": "a.webp"},
    ]
    mp = sdir / "manifest.json"
    mp.write_text(json.dumps({"session_name": "s1", "meta": {}, "items": items}), encoding="utf-8")
    monkeypatch.setattr(mr, "manifest_utils", _mu)
    mr._jobs.clear()
    return mp


def _wait(job_id):
    for _ in range(200):
        st = mr.get_metrics_status(job_id)
        if st["status"] != "running":
            return st
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def _items(mp):
    return {i["id"]: i for i in json.loads(mp.read_text(encoding="utf-8"))["items"]}


def test_computes_only_missing_and_safe_items(session):
    job_id, err = mr.start_metrics_job("s1")
    assert err is None
    st = _wait(job_id)
    assert st["status"] == "complete"
    it = _items(session)
    assert set(it[1]["metrics"]) == set(im.BUILTIN_METRICS)
    assert it[1]["metrics"]["luma"] == pytest.approx(10 / 255, abs=1e-3)
    assert it[3]["metrics"]["luma"] == pytest.approx(90 / 255, abs=1e-3)
    assert it[2]["metrics"] == {"luma": 9.0}                          # existing metrics untouched
    for skipped in (4, 5, 6, 7, 8):
        assert "metrics" not in it[skipped]
    assert it[1]["favorited"] is True and it[1]["note"] == "keep"     # other fields intact
    assert st["total"] == 2


def test_overwrite_recomputes_existing(session):
    job_id, err = mr.start_metrics_job("s1", overwrite=True)
    assert err is None and _wait(job_id)["status"] == "complete"
    assert _items(session)[2]["metrics"]["luma"] == pytest.approx(200 / 255, abs=1e-3)


def test_nothing_to_do_and_bad_session(session):
    job_id, _ = mr.start_metrics_job("s1")
    _wait(job_id)
    assert mr.start_metrics_job("s1")[0] is None                       # everything measured now
    assert mr.start_metrics_job("nope")[1] == "Manifest not found"
    assert mr.start_metrics_job("..")[0] is None


def test_flush_reloads_disk_so_concurrent_edits_survive(session):
    # a user edit lands on disk between the job's initial read and its flush
    mr._flush(str(session), {1: {"luma": 0.5}}, False)
    data = json.loads(session.read_text(encoding="utf-8"))
    for i in data["items"]:
        if i["id"] == 3:
            i["favorited"] = True
    session.write_text(json.dumps(data), encoding="utf-8")
    mr._flush(str(session), {3: {"luma": 0.7}}, False)
    it = _items(session)
    assert it[3]["favorited"] is True and it[3]["metrics"] == {"luma": 0.7} and it[1]["metrics"] == {"luma": 0.5}
    assert not os.path.exists(str(session) + ".metrics.tmp")


def test_cancel_stops_job(session, monkeypatch):
    orig = im.compute_metrics_numpy

    def slow(*a, **k):
        time.sleep(0.15)
        return orig(*a, **k)
    monkeypatch.setattr(im, "compute_metrics_numpy", slow)
    job_id, _ = mr.start_metrics_job("s1")
    assert mr.cancel_metrics_job(job_id) is True
    st = _wait(job_id)
    assert st["status"] == "cancelled" and st["completed"] <= st["total"]
    assert mr.cancel_metrics_job(job_id) is False


def test_second_job_for_same_session_rejected(session, monkeypatch):
    orig = im.compute_metrics_numpy
    monkeypatch.setattr(im, "compute_metrics_numpy", lambda *a, **k: (time.sleep(0.2), orig(*a, **k))[1])
    job_id, _ = mr.start_metrics_job("s1")
    assert mr.start_metrics_job("s1")[1].startswith("A metrics job is already running")
    mr.cancel_metrics_job(job_id)
    _wait(job_id)
