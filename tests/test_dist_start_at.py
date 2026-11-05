"""Distributed mode honours Start At Job: plan positions before N are skipped by the master."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generation_orchestrator as go  # preloaded by tests/conftest.py
from config_utils import build_job_plan, order_job_plan


@pytest.fixture(autouse=True)
def _prompts(monkeypatch):
    monkeypatch.setattr(go, "build_prompt_with_triggers", lambda conf, mode: (conf["positive"], ""))


def _conf(**kw):
    c = {"model": "m", "lora": "None", "lora_expanded": "None", "sampler": "euler", "scheduler": "normal",
         "steps": 30, "cfg": 7.0, "denoise": 1.0, "seed": 5, "positive": "p", "negative": "n"}
    c.update(kw)
    return c


def _job(w=1024, h=1024):
    return {"label": f"{w}x{h}", "width": w, "height": h, "batch_idx": 0}


def _plan(confs):
    return build_job_plan(confs, [_job()])


def _full_item(conf):
    return {"seed": conf["seed"], "width": 1024, "height": 1024, "batch_idx": 0, "sampler": conf["sampler"],
            "scheduler": conf["scheduler"], "steps": conf["steps"], "cfg": conf["cfg"], "denoise": 1.0,
            "model": conf["model"], "positive": conf["positive"], "negative": conf["negative"], "lora": "None"}


def _run(plan, items=(), **kw):
    kw.setdefault("overwrite_existing", False)
    kw.setdefault("has_optional_inputs", False)
    kw.setdefault("lora_triggerwords_mode", "off")
    return go._distributed_plan_entries(plan, {"items": list(items), "meta": {}}, **kw)


def _five():
    return [_conf(seed=i, positive=f"p{i}") for i in range(1, 6)]


def test_start_at_three_skips_first_two():
    entries, counts = _run(_plan(_five()), start_at_job=3)
    assert [e["config"]["seed"] for e in entries] == [3, 4, 5]
    assert counts["start_at"] == 2


@pytest.mark.parametrize("n", [0, 1])
def test_zero_and_one_skip_nothing(n):
    entries, counts = _run(_plan(_five()), start_at_job=n)
    assert [e["config"]["seed"] for e in entries] == [1, 2, 3, 4, 5]
    assert counts.get("start_at", 0) == 0


def test_larger_than_plan_skips_everything():
    entries, counts = _run(_plan(_five()), start_at_job=99)
    assert entries == []
    assert counts["start_at"] == 5


def test_job_before_n_that_exists_counts_as_start_at():
    confs = _five()
    entries, counts = _run(_plan(confs), [_full_item(confs[0])], start_at_job=3)
    assert counts["start_at"] == 2
    assert counts["exists"] == 0
    assert [e["config"]["seed"] for e in entries] == [3, 4, 5]


def test_positions_follow_the_ordered_plan():
    confs = [_conf(seed=1, model="A"), _conf(seed=2, model="B"), _conf(seed=3, model="A"),
             _conf(seed=4, model="B")]
    plan = order_job_plan(_plan(confs), "minimize_reloads", go.get_model_cache_key)
    seeds = [pj["conf"]["seed"] for pj in plan]
    entries, counts = _run(plan, start_at_job=3)
    assert [e["config"]["seed"] for e in entries] == seeds[2:]
    assert counts["start_at"] == 2


class _OldManager:
    def __init__(self):
        self.called = None

    def populate_jobs(self, *a):
        self.called = ("populate_jobs", a)


class _NewManager(_OldManager):
    def populate_from_plan(self, entries, run_extras=None):
        self.called = ("populate_from_plan", entries, run_extras)


def _populate(mgr, plan, **kw):
    confs = [pj["conf"] for pj in plan]
    return go._populate_distributed_manager(mgr, plan, confs, [_job()], {"items": []}, False, False, "off", **kw)


def test_legacy_manager_warns_that_start_at_is_ignored(capsys):
    mgr = _OldManager()
    ret = _populate(mgr, _plan(_five()), start_at_job=3)
    assert mgr.called[0] == "populate_jobs"
    assert "Start At Job" in capsys.readouterr().out
    assert ret["queued"] is None and ret["plan_total"] is None


def test_populate_returns_plan_total_and_queued():
    mgr = _NewManager()
    ret = _populate(mgr, _plan(_five()), start_at_job=3)
    assert ret["plan_total"] == 5
    assert [e["config"]["seed"] for e in ret["queued"]] == [3, 4, 5]
    assert ret["counts"]["start_at"] == 2
    assert ret["ltx_entries"] == []
    assert [e["config"]["seed"] for e in mgr.called[1]] == [3, 4, 5]
