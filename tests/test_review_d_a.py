"""Review-d findings 3 + 6: session-namespaced scan ids, merge_rescan basename fallback."""
import importlib.util
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)


def _load(name, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_ROOT, name))
    m = importlib.util.module_from_spec(spec)
    m.__package__ = _PKG
    spec.loader.exec_module(m)
    return m


mu = _load("manifest_utils.py", "manifest_utils_review_d_a")


def _items(*rels):
    return [{"source_relpath": r, "source_file": r.split("/")[-1]} for r in rels]


def test_ids_differ_across_sessions_for_same_relpath():
    a = mu.assign_scan_ids(_items("ComfyUI_00001_.png"), namespace="sess-a")[0]["id"]
    b = mu.assign_scan_ids(_items("ComfyUI_00001_.png"), namespace="sess-b")[0]["id"]
    assert a != b


def test_same_session_same_relpath_is_stable():
    a = mu.assign_scan_ids(_items("x/ComfyUI_00001_.png"), namespace="sess-a")[0]["id"]
    b = mu.assign_scan_ids(_items("x/ComfyUI_00001_.png"), namespace="sess-a")[0]["id"]
    assert a == b
    assert a == mu.stable_item_id("scan:sess-a:x/ComfyUI_00001_.png")


def test_no_namespace_keeps_legacy_key():
    got = mu.assign_scan_ids(_items("c.png"))[0]["id"]
    assert got == mu.stable_item_id("scan:c.png")


def test_merge_rescan_same_basename_no_duplicate_ids():
    # legacy old items: no source_relpath, only source_file; one old item, two new same-basename files
    old = {"items": [{"id": 111, "source_file": "a.png", "favorited": True}]}
    new = {"session_name": "s", "items": [
        {"id": mu.stable_item_id("scan:s:a.png"), "source_relpath": "a.png", "source_file": "a.png"},
        {"id": mu.stable_item_id("scan:s:images/a.png"), "source_relpath": "images/a.png", "source_file": "a.png"},
    ]}
    mu.merge_rescan(old, new)
    ids = [i["id"] for i in new["items"]]
    assert len(set(ids)) == 2
    assert ids.count(111) == 1
    assert sum(1 for i in new["items"] if i.get("favorited")) == 1


def test_non_unique_basenames_do_not_match_via_source_file():
    old = {"items": [
        {"id": 1, "source_file": "a.png", "favorited": True},
        {"id": 2, "source_file": "a.png"},
    ]}
    new = {"session_name": "s", "items": [
        {"id": mu.stable_item_id("scan:s:a.png"), "source_relpath": "a.png", "source_file": "a.png"},
    ]}
    mu.merge_rescan(old, new)
    assert new["items"][0]["id"] not in (1, 2)
    assert not new["items"][0].get("favorited")


def test_merge_rescan_fresh_ids_use_session_namespace():
    old = {"items": []}
    new = {"session_name": "s", "items": _items("a.png")}
    new["items"][0]["id"] = mu.stable_item_id("scan:s:a.png")
    mu.merge_rescan(old, new)
    assert new["items"][0]["id"] == mu.stable_item_id("scan:s:a.png")
