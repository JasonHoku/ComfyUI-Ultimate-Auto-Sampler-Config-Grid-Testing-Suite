"""A generation run's manifest save must not wipe per-image metrics that were computed
for an existing session (metrics_runner writes them straight to disk while the run's
in-memory copy of those items has no `metrics`). Disk wins for an item's metrics when
the in-memory item has none; freshly computed in-memory metrics are kept.

Loads the real manifest_utils.py via importlib (tests/conftest.py registers a stub
under the package name), like test_comparisons_manifest.py does.
"""
import importlib.util
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("manifest_utils_real_metrics", os.path.join(_ROOT, "manifest_utils.py"))
mu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mu)


def _item(i, **extra):
    d = {"id": i, "file": f"/view?filename=img_{i}.webp", "favorited": False, "rejected": False, "note": ""}
    d.update(extra)
    return d


def test_disk_metrics_survive_a_stale_in_memory_save(tmp_path):
    path = str(tmp_path / "manifest.json")
    disk = {"session_name": "s", "meta": {}, "items": [
        _item(2, metrics={"sharpness": 12.5, "entropy": 7.1}),
        _item(1, metrics={"sharpness": 3.0}),
    ]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(disk, f)

    # the generation run's copy predates the metrics job: same items, no metrics, plus a new image
    stale = {"session_name": "s", "meta": {}, "items": [_item(3), _item(2), _item(1)]}
    mu.save_manifest(path, stale)

    with open(path, encoding="utf-8") as f:
        after = json.load(f)
    by_id = {it["id"]: it for it in after["items"]}
    assert set(by_id) == {1, 2, 3}, "the new image is kept"
    assert by_id[2]["metrics"] == {"sharpness": 12.5, "entropy": 7.1}
    assert by_id[1]["metrics"] == {"sharpness": 3.0}
    assert "metrics" not in by_id[3]


def test_fresh_in_memory_metrics_are_not_overwritten_by_disk(tmp_path):
    path = str(tmp_path / "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"session_name": "s", "meta": {}, "items": [_item(1, metrics={"sharpness": 1.0})]}, f)
    fresh = {"session_name": "s", "meta": {}, "items": [_item(1, metrics={"sharpness": 99.0})]}
    mu.save_manifest(path, fresh)
    with open(path, encoding="utf-8") as f:
        after = json.load(f)
    assert after["items"][0]["metrics"] == {"sharpness": 99.0}
