"""Stable ids for scanned items: a re-scan keeps ids, votes, favorites, notes and metrics.

Loads the real manifest_utils.py / directory_scanner.py via importlib (tests/conftest.py
registers stubs under the package name).
"""
import copy
import importlib.util
import json
import os
import sys

import pytest
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)


def _load(name, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_ROOT, name))
    m = importlib.util.module_from_spec(spec)
    m.__package__ = _PKG
    spec.loader.exec_module(m)
    return m


mu = _load("manifest_utils.py", "manifest_utils_scanids")


# ---- T1 -------------------------------------------------------------------

def test_stable_item_id_deterministic_and_in_range():
    a = mu.stable_item_id("scan:images/img_1.webp")
    assert a == mu.stable_item_id("scan:images/img_1.webp")
    assert type(a) is int
    assert 2 ** 52 <= a < 2 ** 53
    assert a != mu.stable_item_id("scan:images/img_2.webp")
    assert a != mu.stable_item_id("scan:images/img_1.webp", 1)


def test_generated_ids_are_below_scan_range():
    import time
    assert int(time.time() * 100000) + 1000 < mu.SCAN_ID_BASE


def test_scan_relpath_forward_slashes(tmp_path):
    p = os.path.join(str(tmp_path), "images", "a.png")
    assert mu.scan_relpath(p, str(tmp_path)) == "images/a.png"


def _items(rels):
    return [{"source_relpath": r, "source_file": os.path.basename(r)} for r in rels]


def test_assign_scan_ids_resolves_clash_deterministically(monkeypatch):
    monkeypatch.setattr(mu, "stable_item_id", lambda key, salt=0: mu.SCAN_ID_BASE + 7 + salt)
    rels = ["b/z.png", "a/y.png", "c.png"]
    outs = []
    for order in (rels, list(reversed(rels)), [rels[1], rels[2], rels[0]]):
        items = mu.assign_scan_ids(_items(order), reserved={mu.SCAN_ID_BASE + 7})
        ids = {i["source_relpath"]: i["id"] for i in items}
        assert len(set(ids.values())) == 3
        assert mu.SCAN_ID_BASE + 7 not in ids.values()
        outs.append(ids)
    assert outs[0] == outs[1] == outs[2]


def test_assign_scan_ids_same_for_any_input_order():
    rels = [f"images/{i}.png" for i in range(20)]
    a = {i["source_relpath"]: i["id"] for i in mu.assign_scan_ids(_items(rels))}
    b = {i["source_relpath"]: i["id"] for i in mu.assign_scan_ids(_items(reversed(rels)))}
    assert a == b


def _old_manifest():
    return {
        "session_name": "s",
        "meta": {"rejected_preview_fps": ["fp1"]},
        "comparisons": [{"a": 11, "b": 22, "winner": "a", "ts": 5, "mode": "random"},
                        {"a": 99, "b": 98, "winner": "tie", "ts": 6, "mode": "random"}],
        "items": [
            {"id": 11, "source_relpath": "images/a.png", "source_file": "a.png", "favorited": True,
             "note": "keep me", "metrics": {"s": 1}},
            {"id": 22, "source_file": "b.png", "rejected": True, "notes": "legacy note"},   # legacy scan: no relpath
        ],
    }


def _new_manifest():
    return {"session_name": "s", "meta": {"x": 1}, "items": [
        {"id": mu.stable_item_id("scan:images/a.png"), "source_relpath": "images/a.png", "source_file": "a.png",
         "favorited": False, "rejected": False},
        {"id": mu.stable_item_id("scan:b.png"), "source_relpath": "b.png", "source_file": "b.png",
         "favorited": False, "rejected": False},
        {"id": mu.stable_item_id("scan:c.png"), "source_relpath": "c.png", "source_file": "c.png",
         "favorited": False, "rejected": False},
    ]}


def test_merge_rescan_reuses_ids_and_carries_user_data():
    old = _old_manifest()
    old_comparisons = json.dumps(old["comparisons"], sort_keys=True)
    new = mu.merge_rescan(old, _new_manifest())
    by = {i["source_relpath"]: i for i in new["items"]}
    assert by["images/a.png"]["id"] == 11                      # by source_relpath
    assert by["images/a.png"]["favorited"] is True
    assert by["images/a.png"]["note"] == "keep me"
    assert by["images/a.png"]["metrics"] == {"s": 1}
    assert by["b.png"]["id"] == 22                             # by legacy source_file
    assert by["b.png"]["rejected"] is True and by["b.png"]["notes"] == "legacy note"
    assert by["c.png"]["id"] == mu.stable_item_id("scan:c.png")  # new file: stable id
    assert json.dumps(new["comparisons"], sort_keys=True) == old_comparisons   # votes incl. vanished files
    assert new["meta"]["rejected_preview_fps"] == ["fp1"] and new["meta"]["x"] == 1


def test_merge_rescan_reused_id_pushes_clashing_new_file_to_next_salt():
    old = {"items": [{"id": mu.stable_item_id("scan:c.png"), "source_relpath": "gone.png", "source_file": "gone.png"}]}
    new = {"items": [{"id": mu.stable_item_id("scan:c.png"), "source_relpath": "c.png", "source_file": "c.png"},
                     {"id": mu.stable_item_id("scan:gone.png"), "source_relpath": "gone.png", "source_file": "gone.png"}]}
    mu.merge_rescan(old, new)
    ids = [i["id"] for i in new["items"]]
    assert len(set(ids)) == 2
    assert new["items"][1]["id"] == old["items"][0]["id"]
    assert new["items"][0]["id"] == mu.stable_item_id("scan:c.png", 1)


def test_merge_rescan_no_old_manifest_is_noop():
    new = _new_manifest()
    before = copy.deepcopy(new)
    assert mu.merge_rescan(None, new) == before


def test_write_rescan_manifest_round_trip(tmp_path):
    path = str(tmp_path / "manifest.json")
    mu.write_rescan_manifest(path, _new_manifest())           # first scan: no old file
    with open(path) as f:
        first = json.load(f)
    # the user acts on the session between scans
    first["items"][0]["favorited"] = True
    first["comparisons"] = [{"a": first["items"][0]["id"], "b": first["items"][1]["id"], "winner": "b", "ts": 1, "mode": "random"}]
    with open(path, "w") as f:
        json.dump(first, f)
    mu.write_rescan_manifest(path, _new_manifest())           # re-scan
    with open(path) as f:
        second = json.load(f)
    assert [i["id"] for i in second["items"]] == [i["id"] for i in first["items"]]
    assert second["items"][0]["favorited"] is True
    assert second["comparisons"] == first["comparisons"]
    assert not os.path.exists(path + ".tmp")


# ---- T3 -------------------------------------------------------------------

def test_user_changes_survive_save_of_rescanned_copy(tmp_path):
    path = str(tmp_path / "manifest.json")
    mu.write_rescan_manifest(path, _new_manifest())
    with open(path) as f:
        disk = json.load(f)
    disk["items"][1]["favorited"] = True
    disk["items"][1]["note"] = "n"
    with open(path, "w") as f:
        json.dump(disk, f)
    stale = _new_manifest()                                   # a re-scanned copy in memory, no user fields
    mu.merge_manifest_user_changes(path, stale)
    by = {i["source_relpath"]: i for i in stale["items"]}
    assert by["b.png"]["favorited"] is True and by["b.png"]["note"] == "n"


# ---- T2: the scanner ------------------------------------------------------

@pytest.fixture
def scanner(monkeypatch):
    import types
    fake_mp = types.ModuleType(f"{_PKG}.metadata_packer")        # the real one needs piexif (not in CI)
    fake_mp.extract_metadata_from_image = lambda path: {}
    monkeypatch.setitem(sys.modules, f"{_PKG}.manifest_utils", mu)
    monkeypatch.setitem(sys.modules, f"{_PKG}.metadata_packer", fake_mp)
    return _load("directory_scanner.py", f"{_PKG}.directory_scanner_real")


def _mk(tmp_path):
    d = tmp_path / "pics"
    (d / "images").mkdir(parents=True)
    for rel in ("one.png", "two.png", "images/three.png"):
        Image.new("RGB", (8, 8), (10, 20, 30)).save(d / rel)
    return d


def test_scan_twice_gives_identical_ids(scanner, tmp_path):
    d = _mk(tmp_path)
    items1, _ = scanner.scan_directory_for_images(str(d), view_subfolder="benchmarks/s/external_images")
    for n, p in enumerate(sorted(d.rglob("*.png"))):
        os.utime(p, (1, 1000 + 97 * n))                         # change mtimes between scans
    items2, _ = scanner.scan_directory_for_images(str(d), view_subfolder="benchmarks/s/external_images")
    m1 = {i["source_relpath"]: i["id"] for i in items1}
    m2 = {i["source_relpath"]: i["id"] for i in items2}
    assert set(m1) == {"one.png", "two.png", "images/three.png"}
    assert m1 == m2
    assert all(2 ** 52 <= v < 2 ** 53 for v in m1.values()) and len(set(m1.values())) == 3


def test_manifest_sourced_item_without_id_gets_stable_id(scanner, tmp_path):
    d = _mk(tmp_path)
    with open(d / "manifest.json", "w") as f:
        json.dump({"items": [{"file": "/view?filename=one.png&type=output&subfolder=x", "model": "m", "positive": "p"}]}, f)
    items, stats = scanner.scan_directory_for_images(str(d))
    one = [i for i in items if i.get("source_relpath") == "one.png"][0]
    assert one["id"] == mu.stable_item_id("scan:one.png")
