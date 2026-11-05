"""Mixed runs: standard configs + Workflow Sweep entries in ONE Generator run (standard loop first, then the sweep)."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_workflow_sweep_generator import (  # noqa: E402,F401
    PKG, _FakeBridge, runner, sampler_node, wf_config)

STD = {"model_type": "checkpoint", "model": "x"}


class _BadBridge(_FakeBridge):
    async def validate(self, prompt, partial):
        return False, {"type": "prompt_outputs_failed_validation", "message": "nope"}, {}


@pytest.fixture
def env(sampler_node, tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda t: ["a.safetensors"], raising=False)
    monkeypatch.setattr(runner, "TICK", 0.05)
    monkeypatch.setattr(sampler_node, "get_html_template", lambda *a, **k: "<html>")
    monkeypatch.setattr(sys.modules["comfy"], "model_management", sys.modules["comfy.model_management"], raising=False)
    runner.reset_registry()
    bridge = _FakeBridge()
    runner.set_bridge(bridge)
    calls = []
    state = {"total": 3, "hook": None, "raise": None}
    go = sys.modules[f"{PKG}.generation_orchestrator"]

    def fake_loop(self, *args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs, "active": runner._active_sweep()})
        kwargs["run_info"]["total_jobs"] = state["total"]
        if state["hook"]:
            state["hook"]()
        if state["raise"]:
            raise state["raise"]
        return ("<loop>",)

    monkeypatch.setattr(go, "run_generation_loop", fake_loop, raising=False)
    yield {"calls": calls, "state": state, "bridge": bridge, "mod": sampler_node}
    sw = runner._active_sweep()
    if sw is not None:
        runner.cancel(sw.id)
        if sw.thread:
            sw.thread.join(5)
    runner.set_bridge(None)
    runner.reset_registry()


def cfg(configs, **settings):
    s = {"session_name": "mixed_s", "start_at_job": 0}
    s.update(settings)
    return json.dumps({"configs": configs, "_session_settings": s})


def run(env, configs, **settings):
    return env["mod"].SamplerGridTester().run_tests(cfg(configs, **settings), unique_id="9")


def test_split_configs_keeps_order_dict_or_list(sampler_node):
    a, w1, b, w2 = dict(STD, name="a"), wf_config(), dict(STD, name="b"), wf_config()
    std, wf = sampler_node._split_configs([a, w1, b, w2])
    assert std == [a, b] and wf == [w1, w2]
    assert sampler_node._split_configs({"configs": [w1, a]}) == ([a], [w1])
    assert sampler_node._split_configs(None) == ([], [])


def test_mixed_run_loop_first_then_sweep_registered(env):
    out = run(env, [STD, wf_config()])
    assert out == ("<loop>",)
    assert len(env["calls"]) == 1
    call = env["calls"][0]
    assert call["active"] is None                      # nothing registered while the standard loop runs
    kw = call["kwargs"]
    assert json.loads(call["args"][8]) == [STD]        # configs_json = standard entries only
    full = json.loads(kw["meta_configs_json"])
    assert [c["model_type"] for c in full] == ["checkpoint", "workflow"]
    assert kw["next_phase"] == "sweep"
    sw = runner._active_sweep()
    assert sw is not None and sw.session == "mixed_s" and sw.origin_node == "9" and len(sw.jobs) == 2


def test_mixed_run_keeps_builder_order_in_meta(env):
    run(env, [wf_config(), STD])
    full = json.loads(env["calls"][0]["kwargs"]["meta_configs_json"])
    assert [c["model_type"] for c in full] == ["workflow", "checkpoint"]


@pytest.mark.parametrize("sa,expected", [(0, 0), (2, 0), (3, 0), (4, 1), (6, 3)])
def test_start_at_job_is_global_standard_first(env, sa, expected):
    run(env, [STD, wf_config()], start_at_job=sa)
    (sw,) = list(runner._sweeps.values())
    assert sw.settings["start_at_job"] == expected


def test_start_at_one_past_standard_skips_nothing(env):
    run(env, [STD, wf_config()], start_at_job=4)
    sw = runner._active_sweep()
    assert not [j for j in sw.jobs if j["status"] == "skipped"]


def test_overwrite_and_random_seeds_reach_the_sweep(env):
    run(env, [STD, wf_config()], overwrite_existing=True, add_random_seeds_to_gens=1)
    sw = runner._active_sweep()
    assert sw.settings["overwrite_existing"] is True
    assert len(sw.jobs) == 3                            # 2 seeds + 1 derived extra seed


def test_interrupt_during_standard_phase_registers_no_sweep(env):
    from comfy.model_management import InterruptProcessingException
    env["state"]["raise"] = InterruptProcessingException()
    with pytest.raises(InterruptProcessingException):
        run(env, [STD, wf_config()])
    assert runner._active_sweep() is None and not runner._sweeps


def test_preflight_busy_refuses_before_any_standard_job(env):
    other = wf_config()
    other["workflow_bindings"][0]["values"] = [5, 6]
    runner.start_sweep([other], {"session_name": "mixed_s"}, origin_node="1")
    with pytest.raises(RuntimeError, match="NOT started"):
        run(env, [STD, wf_config()])
    assert env["calls"] == []


def test_preflight_validation_failure_refuses_before_any_standard_job(env):
    runner.set_bridge(_BadBridge())
    with pytest.raises(ValueError, match="nothing ran"):
        run(env, [STD, wf_config()])
    assert env["calls"] == []


def test_sweep_busy_after_the_loop_says_standard_jobs_were_saved(env):
    other = wf_config()
    other["workflow_bindings"][0]["values"] = [5, 6]
    env["state"]["hook"] = lambda: runner.start_sweep([other], {"session_name": "mixed_s"}, origin_node="1")
    with pytest.raises(RuntimeError, match="Standard jobs finished"):
        run(env, [STD, wf_config()])


def test_preflight_sweep_registers_nothing(env):
    info = runner.preflight_sweep([wf_config()], {"session_name": "pf"}, origin_node="1")
    assert info["total"] == 2 and info["to_run"] == 2
    assert not runner._sweeps and env["bridge"].enqueued == []


def test_busy_conflict_ignores_an_identical_sweep(env):
    sw = runner.start_sweep([wf_config()], {"session_name": "bc"}, origin_node="1")
    fps = [j["fingerprint"] for j in sw.jobs]
    assert runner.busy_conflict("bc", fps) is None
    assert runner.busy_conflict("bc", ["other"]) is sw
    assert runner.busy_conflict("elsewhere", fps) is sw


def test_pure_run_helpers():
    go = sys.modules[f"{PKG}.generation_orchestrator"]
    meta = go._standard_run_meta({"meta": {"rejected_preview_fps": ["a"]}}, "p", "n", "m", 1, 2, "FULL", "[[1,1]]",
                                 "webp", "as_written", False, 8, 0.5, None, {"x": 1})
    assert meta["configs_json"] == "FULL" and meta["rejected_preview_fps"] == ["a"] and meta["x"] == 1
    assert meta["preview_pass"] == {"enabled": False, "steps": 8, "scale": 0.5, "cfg_override": None}
    done = go._completion_payload("1", "s", 3, 1.234, "0h 0m 5s", 3)
    assert done["complete"] is True and "next_phase" not in done and done["avg_duration"] == 1.2
    assert go._completion_payload("1", "s", 3, 1.0, "x", 3, "sweep")["next_phase"] == "sweep"
