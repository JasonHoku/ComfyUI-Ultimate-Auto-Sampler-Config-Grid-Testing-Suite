"""metrics_runner fill_missing: backfill only the metric keys an item lacks (item 25, Sobel on older sessions)."""
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
_mu_spec = importlib.util.spec_from_file_location(f"{_PKG}.manifest_utils_real_fm", os.path.join(_ROOT, "manifest_utils.py"))
_mu = importlib.util.module_from_spec(_mu_spec)
_mu.__package__ = _PKG
_mu_spec.loader.exec_module(_mu)
im = importlib.import_module(f"{_PKG}.image_metrics")

OLD_SIX = ("sharpness", "entropy", "colorfulness", "contrast", "luma", "saturation")


@pytest.fixture
def session(tmp_path, monkeypatch):
    out = tmp_path / "out"
    sdir = out / "benchmarks" / "s1"
    (sdir / "images").mkdir(parents=True)
    monkeypatch.setattr(sys.modules["folder_paths"], "get_output_directory", lambda: str(out), raising=False)
    for n, v in (("a.webp", 10), ("b.webp", 200), ("c.png", 90)):
        Image.fromarray(np.full((30, 40, 3), v, np.uint8)).save(sdir / "images" / n)
    old = {k: 1.5 for k in OLD_SIX}
    old["aesthetic"] = 6.5
    items = [
        {"id": 1, "file": "/view?filename=a.webp&type=output&subfolder=benchmarks/s1/images"},
        {"id": 2, "file": "/view?filename=b.webp&type=output&subfolder=benchmarks/s1/images", "metrics": {"luma": 9.0}},
        {"id": 3, "file": "/view?filename=c.png&type=output&subfolder=benchmarks/s1/images", "metrics": old},
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


def test_fill_missing_keeps_existing_and_adds_the_rest(session):
    job_id, err = mr.start_metrics_job("s1", fill_missing=True)
    assert err is None
    st = _wait(job_id)
    assert st["status"] == "complete" and st["total"] == 3
    it = _items(session)
    assert it[2]["metrics"]["luma"] == 9.0
    assert set(it[2]["metrics"]) == set(im.BUILTIN_METRICS)
    assert set(it[1]["metrics"]) == set(im.BUILTIN_METRICS)


def test_fill_missing_adds_only_sobel_to_old_items(session):
    job_id, _ = mr.start_metrics_job("s1", fill_missing=True)
    _wait(job_id)
    m = _items(session)[3]["metrics"]
    assert {k: m[k] for k in OLD_SIX} == {k: 1.5 for k in OLD_SIX}
    assert m["aesthetic"] == 6.5 and "sobel" in m


def test_second_fill_missing_run_has_nothing_to_do(session):
    job_id, _ = mr.start_metrics_job("s1", fill_missing=True)
    _wait(job_id)
    jid, err = mr.start_metrics_job("s1", fill_missing=True)
    assert jid is None and err.startswith("Nothing to compute")


def test_default_mode_still_skips_items_with_metrics(session):
    job_id, _ = mr.start_metrics_job("s1")
    assert _wait(job_id)["total"] == 1


def test_flush_merge_never_overwrites_disk_keys_and_keeps_edits(session):
    data = json.loads(session.read_text(encoding="utf-8"))
    for i in data["items"]:
        if i["id"] == 3:
            i["favorited"] = True
            i["metrics"]["luma"] = 0.42
    session.write_text(json.dumps(data), encoding="utf-8")
    mr._flush(str(session), {3: {"luma": 0.99, "sobel": 0.1}}, False, merge=True)
    m = _items(session)[3]
    assert m["favorited"] is True
    assert m["metrics"]["luma"] == 0.42 and m["metrics"]["sobel"] == 0.1 and m["metrics"]["aesthetic"] == 6.5
