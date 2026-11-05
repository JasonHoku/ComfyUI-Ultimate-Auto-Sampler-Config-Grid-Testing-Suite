"""Claim-time re-check of previews rejected while a distributed full pass runs."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generation_orchestrator as go  # preloaded by tests/conftest.py

FP = "fp:abc"
EXTRA = {"stamp": {"config_fp": FP}}


def _preview(**kw):
    p = {"id": "p1", "preview": True, "preview_of": FP, "seed": 1}
    p.update(kw)
    return p


def _write(path, items, meta=None, bump=0):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"items": items, "meta": meta or {}}, f)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + bump))


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _make(tmp_path, items=(), meta=None, disk_items=None, disk_meta=None, **kw):
    mp = str(tmp_path / "manifest.json")
    _write(mp, list(items if disk_items is None else disk_items), disk_meta if disk_meta is not None else meta)
    mem = {"items": list(items), "meta": meta or {}}
    kw.setdefault("skip_rejected", True)
    kw.setdefault("only_favorited", False)
    clock = _Clock()
    chk = go._make_reject_recheck(mp, mem, min_interval=2.0, clock=clock, **kw)
    return mp, chk, clock


def test_not_rejected_returns_none(tmp_path):
    _, chk, _ = _make(tmp_path, [_preview()])
    assert chk(EXTRA, {}) is None


def test_rewritten_manifest_with_rejection_is_seen(tmp_path):
    mp, chk, clock = _make(tmp_path, [_preview()])
    assert chk(EXTRA, {}) is None
    _write(mp, [_preview(rejected=True)], bump=5_000_000_000)
    clock.t = 10.0
    assert chk(EXTRA, {}) == "rejected_preview"


def test_disk_tombstone(tmp_path):
    _, chk, _ = _make(tmp_path, [_preview()], disk_meta={"rejected_preview_fps": [FP]})
    assert chk(EXTRA, {}) == "rejected_preview"


def test_in_memory_tombstone(tmp_path):
    _, chk, _ = _make(tmp_path, [_preview()], meta={"rejected_preview_fps": [FP]}, disk_meta={})
    assert chk(EXTRA, {}) == "rejected_preview"


def test_only_favorited_with_unfavorited_preview(tmp_path):
    _, chk, _ = _make(tmp_path, [_preview()], skip_rejected=False, only_favorited=True)
    assert chk(EXTRA, {}) == "no_favorited_preview"


def test_preview_jobs_and_unstamped_jobs_never_dropped(tmp_path):
    _, chk, _ = _make(tmp_path, [_preview(rejected=True)])
    assert chk({"stamp": {"config_fp": FP}, "render": {"preview_scale": 0.5}}, {}) is None
    assert chk({}, {}) is None
    assert chk(None, {}) is None
    assert chk({"stamp": {}}, {}) is None


def test_corrupt_file_keeps_old_index(tmp_path):
    mp, chk, clock = _make(tmp_path, [_preview(rejected=True)])
    assert chk(EXTRA, {}) == "rejected_preview"
    with open(mp, "w", encoding="utf-8") as f:
        f.write("{not json")
    clock.t = 10.0
    assert chk(EXTRA, {}) == "rejected_preview"


def test_no_reread_until_min_interval(tmp_path):
    mp, chk, clock = _make(tmp_path, [_preview()])
    assert chk(EXTRA, {}) is None
    _write(mp, [_preview(rejected=True)], bump=5_000_000_000)
    clock.t = 1.0  # < min_interval: still the cached index
    assert chk(EXTRA, {}) is None
    clock.t = 2.5
    assert chk(EXTRA, {}) == "rejected_preview"


# -- _install_claim_filter ----------------------------------------------------------------------------------

class _Mgr:
    def __init__(self):
        self.filter = None

    def set_claim_filter(self, fn):
        self.filter = fn


def _install(mgr, tmp_path, **kw):
    base = dict(preview_on=False, has_optional_inputs=False, skip_rejected=True, only_favorited=False)
    base.update(kw)
    return go._install_claim_filter(
        mgr, {"manifest": str(tmp_path / "m.json")}, {"items": [], "meta": {}}, **base)


def test_installed_for_a_full_pass(tmp_path):
    m = _Mgr()
    assert _install(m, tmp_path) is True
    assert callable(m.filter)


def test_not_installed_for_preview_pass_or_optional_inputs_or_no_rules(tmp_path):
    for kw in ({"preview_on": True}, {"has_optional_inputs": True},
               {"skip_rejected": False, "only_favorited": False}):
        m = _Mgr()
        assert _install(m, tmp_path, **kw) is False
        assert m.filter is None


def test_manager_without_set_claim_filter_is_fine(tmp_path, capsys):
    assert _install(object(), tmp_path) is False
    assert "update" in capsys.readouterr().out.lower()
