"""Pairwise-review votes stored in manifest["comparisons"].

Loads the real manifest_utils.py (tests/conftest.py registers a stub under the
package name) via importlib, like test_pinned_clip.py does.
"""
import copy
import importlib.util
import json
import os
import threading

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("manifest_utils_real", os.path.join(_ROOT, "manifest_utils.py"))
mu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mu)


def V(a, b, winner="a", ts=1000, **kw):
    d = {"a": a, "b": b, "winner": winner, "ts": ts}
    d.update(kw)
    return d


def _write(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def _read(path):
    with open(path) as f:
        return json.load(f)


def _manifest():
    return {"session_name": "s", "meta": {"model": "m.safetensors"},
            "items": [{"id": 3, "favorited": True, "note": "n", "file": "/view?x=3"},
                      {"id": 2, "rejected": True, "file": "/view?x=2"},
                      {"id": 1, "file": "/view?x=1"}]}


# ---- validate_comparison ----------------------------------------------------
def test_validate_rejects_bad_entries():
    assert mu.validate_comparison(V(1, 1)) is None
    assert mu.validate_comparison(V(True, 2)) is None
    assert mu.validate_comparison(V("1", 2)) is None
    assert mu.validate_comparison(V(1.0, 2)) is None
    assert mu.validate_comparison(V(1, 2, winner="x")) is None
    assert mu.validate_comparison(V(1, 2, ts=-1)) is None
    assert mu.validate_comparison(V(1, 2, ts="5")) is None
    assert mu.validate_comparison("nope") is None


def test_validate_cleans_fields():
    v = mu.validate_comparison(V(1, 2, "tie", mode="x" * 100, blind=False, junk=1))
    assert v == {"a": 1, "b": 2, "winner": "tie", "ts": 1000, "mode": "x" * 32, "blind": False}
    assert mu.validate_comparison(V(1, 2))["mode"] == "matched"
    assert "blind" not in mu.validate_comparison(V(1, 2, blind="yes"))
    assert mu.validate_comparison(V(180000000000000, 2))["a"] == 180000000000000


# ---- merge_comparison_lists -------------------------------------------------
def test_merge_append_dedupe_remove():
    m, added, removed = mu.merge_comparison_lists([], [V(1, 2), V(1, 2), V(3, 4, ts=5)], [])
    assert (len(m), added, removed) == (2, 2, 0)
    m2, added, removed = mu.merge_comparison_lists(m, [V(1, 2)], [])
    assert (len(m2), added) == (2, 0)
    m3, added, removed = mu.merge_comparison_lists(m, [], ["1000:1:2"])
    assert (len(m3), removed) == (1, 1) and m3[0]["a"] == 3
    m4, added, removed = mu.merge_comparison_lists(m, [], ["nope"])
    assert (m4, removed) == (m, 0)


def test_merge_preserves_invalid_existing():
    junk = {"weird": True}
    m, added, removed = mu.merge_comparison_lists([junk, V(1, 2)], [V(5, 6)], ["1000:1:2"])
    assert m[0] == junk and len(m) == 2 and (added, removed) == (1, 1)


# ---- update_manifest_comparisons -------------------------------------------
def test_update_roundtrip_leaves_items_meta_alone(tmp_path):
    p = str(tmp_path / "manifest.json")
    orig = _manifest()
    _write(p, orig)
    r = mu.update_manifest_comparisons(p, append=[V(1, 2), V(2, 3, ts=7)])
    assert r["added"] == 2 and r["total"] == 2
    d = _read(p)
    assert d["items"] == orig["items"] and d["meta"] == orig["meta"] and len(d["comparisons"]) == 2
    mu.update_manifest_comparisons(p, remove_keys=["1000:1:2"])
    d = _read(p)
    assert d["items"] == orig["items"] and len(d["comparisons"]) == 1
    r = mu.update_manifest_comparisons(p, clear=True)
    d = _read(p)
    assert d["comparisons"] == [] and d["items"] == orig["items"] and r["removed"] == 1
    assert not os.path.exists(p + ".tmp")


def test_update_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mu.update_manifest_comparisons(str(tmp_path / "nope.json"), append=[V(1, 2)])


def test_update_corrupt_json_raises_and_keeps_file(tmp_path):
    p = str(tmp_path / "manifest.json")
    with open(p, "w") as f:
        f.write('{"items": [ {"id": 1}, ')
    before = open(p).read()
    with pytest.raises(ValueError):  # JSONDecodeError subclasses ValueError
        mu.update_manifest_comparisons(p, append=[V(1, 2)])
    assert open(p).read() == before


def test_update_creates_comparisons_when_absent(tmp_path):
    p = str(tmp_path / "manifest.json")
    _write(p, _manifest())
    assert "comparisons" not in _read(p)
    mu.update_manifest_comparisons(p, append=[V(1, 2)])
    assert len(_read(p)["comparisons"]) == 1


# ---- regression: other writers must not drop / resurrect votes -------------
def test_stale_in_memory_manifest_saved_after_vote_keeps_vote(tmp_path):
    p = str(tmp_path / "manifest.json")
    _write(p, _manifest())
    stale = copy.deepcopy(_manifest())            # the generator's start-of-run copy: no votes
    mu.update_manifest_comparisons(p, append=[V(1, 2), V(2, 3, ts=7)])
    stale["items"].insert(0, {"id": 4, "file": "/view?x=4"})
    mu.save_manifest(p, stale)
    d = _read(p)
    assert [i["id"] for i in d["items"]] == [4, 3, 2, 1]
    assert len(d["comparisons"]) == 2


def test_stale_copy_with_old_votes_does_not_resurrect_after_clear(tmp_path):
    p = str(tmp_path / "manifest.json")
    m = _manifest()
    m["comparisons"] = [V(1, 2)]
    _write(p, m)
    stale = copy.deepcopy(m)
    mu.update_manifest_comparisons(p, clear=True)
    mu.save_manifest(p, stale)
    assert _read(p)["comparisons"] == []


def test_stale_copy_does_not_resurrect_undone_vote(tmp_path):
    p = str(tmp_path / "manifest.json")
    m = _manifest()
    m["comparisons"] = [V(1, 2), V(2, 3, ts=7)]
    _write(p, m)
    stale = copy.deepcopy(m)
    mu.update_manifest_comparisons(p, remove_keys=["7:2:3"])
    mu.save_manifest(p, stale)
    assert [c["ts"] for c in _read(p)["comparisons"]] == [1000]


def test_preserve_disk_comparisons_noop_without_field():
    t = {"comparisons": [V(1, 2)]}
    mu.preserve_disk_comparisons({"items": []}, t)
    assert t["comparisons"] == [V(1, 2)]
    mu.preserve_disk_comparisons({"comparisons": []}, t)
    assert t["comparisons"] == []


# ---- concurrency ------------------------------------------------------------
def test_concurrent_saves_and_votes_lose_nothing(tmp_path):
    p = str(tmp_path / "manifest.json")
    base = _manifest()
    _write(p, base)
    state = copy.deepcopy(base)   # the generator's long-lived in-memory dict
    errors = []
    N = 50

    def gen():
        try:
            for i in range(N):
                state["items"].insert(0, {"id": 1000 + i, "file": f"/view?x={i}"})
                mu.save_manifest(p, state)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def vote():
        try:
            for i in range(N):
                mu.update_manifest_comparisons(p, append=[V(1, 2, ts=i + 1)])
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=gen), threading.Thread(target=vote)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    d = _read(p)
    assert len(d["items"]) == 3 + N
    assert len(d["comparisons"]) == N
