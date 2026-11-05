"""LTX jobs in distributed mode: the master renders them, they never reach workers."""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generation_orchestrator as go  # preloaded by tests/conftest.py
from config_utils import build_job_plan


@pytest.fixture(autouse=True)
def _prompts(monkeypatch):
    monkeypatch.setattr(go, "build_prompt_with_triggers", lambda conf, mode: (conf["positive"], ""))


def _conf(**kw):
    c = {"model": "m", "lora": "None", "lora_expanded": "None", "sampler": "euler", "scheduler": "normal",
         "steps": 30, "cfg": 7.0, "denoise": 1.0, "seed": 5, "positive": "p", "negative": "n"}
    c.update(kw)
    return c


def _ltx(**kw):
    c = {"model_type": "ltx_video", "model": "ltx", "seed": 9, "positive": "video", "negative": "",
         "duration_seconds": 5, "frame_rate": 25}
    c.update(kw)
    return c


def _job(w=1024, h=1024):
    return {"label": f"{w}x{h}", "width": w, "height": h, "batch_idx": 0}


class _OldManager:
    def populate_jobs(self, *a):
        raise AssertionError("must not queue anything")


class _NewManager:
    def __init__(self):
        self.entries = None

    def populate_from_plan(self, entries, run_extras=None):
        self.entries = entries


def _populate(mgr, confs, items=()):
    plan = build_job_plan(confs, [_job()])
    return go._populate_distributed_manager(
        mgr, plan, confs, [_job()], {"items": list(items), "meta": {}}, False, False, "off")


def test_ltx_split_out_and_never_queued():
    mgr = _NewManager()
    confs = [_conf(seed=1), _ltx(), _conf(seed=2)]
    ret = _populate(mgr, confs)
    assert [e["config"]["seed"] for e in mgr.entries] == [1, 2]
    assert all(e["config"].get("model_type") != "ltx_video" for e in mgr.entries)
    assert len(ret["ltx_entries"]) == 1
    assert ret["ltx_entries"][0]["conf_idx"] == 1
    assert ret["plan_total"] == 3
    assert all("conf_idx" in e for e in mgr.entries)


def test_ltx_not_skipped_as_existing():
    ltx = _ltx()
    item = {"media_type": "video", "model_type": "ltx_video", "seed": 9, "width": 1024, "height": 1024,
            "batch_idx": 0, "positive": "video", "negative": "", "model": "ltx"}
    plan = build_job_plan([ltx], [_job()])
    entries, counts = go._distributed_plan_entries(
        plan, {"items": [item], "meta": {}}, overwrite_existing=False, has_optional_inputs=False,
        lora_triggerwords_mode="off")
    assert len(entries) == 1
    assert counts["exists"] == 0


def test_legacy_manager_with_ltx_raises():
    with pytest.raises(RuntimeError, match="rendered on the master"):
        go._populate_distributed_manager(
            _OldManager(), None, [_conf(), _ltx()], [_job()], {"items": []}, False, False, "off")


def test_render_ltx_job_inserts_item_and_preflights_once(monkeypatch, tmp_path):
    pkg = go.__package__
    calls = {"preflight": 0, "load": 0, "gen": 0, "saved": 0}
    stub = types.ModuleType(f"{pkg}.ltx_video_generation")

    def preflight(conf):
        calls["preflight"] += 1

    def load(conf):
        calls["load"] += 1
        return object()

    def gen(conf, models, out_path):
        calls["gen"] += 1
        return {"path": out_path}

    stub.preflight_ltx, stub.load_ltx_models, stub.ltx_video_generate = preflight, load, gen
    monkeypatch.setitem(sys.modules, f"{pkg}.ltx_video_generation", stub)
    monkeypatch.setattr(go, "_build_ltx_manifest_entry",
                        lambda conf, res, name, gen_index=None, session_name="": {"id": name, "media_type": "video"})
    monkeypatch.setattr(go, "save_manifest", lambda path, data: calls.__setitem__("saved", calls["saved"] + 1))
    paths = {"images": str(tmp_path), "manifest": str(tmp_path / "manifest.json")}
    data = {"items": [{"id": "old"}]}
    flag = [False]

    a = go._render_ltx_job(_ltx(), 640, 480, 3, paths, data, "sess", flag, dist_worker="master")
    b = go._render_ltx_job(_ltx(), 640, 480, 4, paths, data, "sess", flag)

    assert calls["preflight"] == 1 and calls["load"] == 2 and calls["gen"] == 2 and calls["saved"] == 2
    assert a["dist_worker"] == "master"
    assert "dist_worker" not in b
    assert data["items"][0] is b and data["items"][1] is a
    assert data["items"][2] == {"id": "old"}
