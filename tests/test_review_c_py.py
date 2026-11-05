"""Review fixes (python side): manifest append safety, breaker completion, IS_CHANGED for linked inputs, Start At Job #."""
import asyncio
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import workflow_sweep_runner as runner  # noqa: E402
from test_workflow_sweep_runner import (  # noqa: E402,F401
    env, entry, make, drive, _real_manifest_utils, _cfg_body)
from test_workflow_sweep_generator import sampler_node  # noqa: E402,F401


def test_append_items_corrupt_manifest_raises_and_is_untouched(tmp_path):
    mu = _real_manifest_utils()
    path = tmp_path / "manifest.json"
    path.write_text('{"session_name": "s", "items": [{"id": 1}', encoding="utf-8")   # half-written
    before = path.read_bytes()
    with pytest.raises(RuntimeError) as ei:
        mu.append_items(str(path), [{"id": 2}], session_name="s")
    assert str(path) in str(ei.value)
    assert path.read_bytes() == before


def test_append_items_creates_missing_and_keeps_existing(tmp_path):
    mu = _real_manifest_utils()
    path = tmp_path / "sub" / "manifest.json"
    data = mu.append_items(str(path), [{"id": 1}], session_name="s")
    assert [i["id"] for i in data["items"]] == [1]
    mu.append_items(str(path), [{"id": 2}], session_name="s")
    assert [i["id"] for i in json.loads(path.read_text())["items"]] == [2, 1]
    assert not (tmp_path / "sub" / "manifest.json.tmp").exists()


def _fail(sw, bridge):
    drive(sw, bridge)
    for pid in list(bridge.queue):
        bridge.start(pid)
    for pid in list(bridge.running):
        bridge.finish(pid, "error")
    drive(sw, bridge)


def test_breaker_does_not_pause_when_no_jobs_remain(env):
    sw = make(env, [entry((1, 2, 3))])
    for _ in range(10):
        if not runner.tick(sw):
            break
        for pid in list(env.bridge.queue):
            env.bridge.start(pid)
        for pid in list(env.bridge.running):
            env.bridge.finish(pid, "error")
    assert sw.state == "completed" and "3 job(s) failed" in sw.reason


def test_paused_sweep_with_nothing_left_finishes(env):
    sw = make(env, [entry((1, 2, 3, 4))])
    sw.state, sw.reason = "paused", "3 jobs in a row failed; last: x"
    for j in sw.jobs:
        j["status"] = "error"
    assert runner.tick(sw) is False
    assert sw.state == "completed" and "4 job(s) failed" in sw.reason
    # and it no longer blocks a new sweep
    runner.prepare_sweep([entry((9,))], {"session_name": "wf_test"}, None, None, env.bridge, class_info=None)


def test_paused_sweep_with_pending_jobs_stays_paused(env):
    sw = make(env, [entry((1, 2, 3, 4))])
    sw.state = "paused"
    for j in sw.jobs[:3]:
        j["status"] = "error"
    assert runner.tick(sw) is True and sw.state == "paused"


def test_is_changed_nan_when_configs_json_is_linked(sampler_node):
    cls = sampler_node.SamplerGridTester
    assert cls.IS_CHANGED() != cls.IS_CHANGED()            # linked input: absent from kwargs
    assert cls.IS_CHANGED(steps=20) != cls.IS_CHANGED(steps=20)


def test_start_at_job_n_skips_n_minus_1(env):
    sw = make(env, [entry((1, 2, 3, 4))], {"start_at_job": 3})
    assert [j["status"] for j in sw.jobs] == ["skipped", "skipped", "pending", "pending"]
    runner.reset_registry()
    sw = make(env, [entry((1, 2, 3, 4))], {"start_at_job": 1})
    assert [j["status"] for j in sw.jobs] == ["pending"] * 4


def test_write_spec_retries_permission_error(env, monkeypatch):
    sw = make(env)
    real, calls = os.replace, {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("open in dashboard")
        return real(a, b)
    monkeypatch.setattr(runner.os, "replace", flaky)
    runner._write_spec(sw)
    assert calls["n"] >= 3


def test_dry_run_and_api_start_still_work_via_executor(env):
    body = _cfg_body([entry()], "wf_test")
    out = asyncio.run(runner.dry_run(body, bridge=env.bridge, class_info=None))
    assert out["total"] == 6
    status, res = asyncio.run(runner.api_start(body, bridge=env.bridge, class_info=None))
    assert status == 200 and res["total"] == 6
