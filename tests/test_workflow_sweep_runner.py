"""Workflow Sweep runner: window, results, errors, breaker, interrupt/lost, cancel, resume, bridge mirror."""
import asyncio
import json
import os
import sys
import threading
import time
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import workflow_sweep as wfs  # noqa: E402
import workflow_sweep_runner as runner  # noqa: E402


def _real_manifest_utils():
    # conftest registers a stub manifest_utils; load the real file for update_meta
    import importlib.util
    spec = importlib.util.spec_from_file_location("manifest_utils_real_sweep", os.path.join(ROOT, "manifest_utils.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load(name):
    with open(os.path.join(ROOT, "tests", "fixtures", name), "r", encoding="utf-8") as f:
        return json.load(f)


def entry(values=(1, 2, 3, 4, 5, 6), **extra):
    e = {"model_type": "workflow", "name": "c", "workflow": load("image_moss_image_t2i.json"),
         "workflow_output": {"node_id": "38", "output_index": 0},
         "workflow_bindings": [{"id": "b", "kind": "literal", "targets": [{"node_id": "32", "input": "seed"}],
                                "values": list(values)}]}
    e.update(extra)
    return e


class FakeBridge:
    """Synchronous stand-in for ComfyBridge with a hand-driven queue/history."""

    def __init__(self):
        self.queue, self.running, self.history, self.meta = [], [], {}, {}
        self.enqueued, self.events, self.deleted, self.interrupted = [], [], [], []
        self.reject = None  # callable(prompt) -> error string | None
        self.n = 0

    def run_on_loop(self, coro):
        return asyncio.run(coro)

    async def validate(self, prompt, partial):
        err = self.reject(prompt) if self.reject else None
        if err:
            return False, {"type": "prompt_outputs_failed_validation", "message": err}, \
                {"29": {"class_type": "UNETLoader", "errors": [{"message": err, "details": "unet_name"}]}}
        return True, None, {}

    async def enqueue(self, prompt, partial, client_id):
        ok, error, node_errors = await self.validate(prompt, partial)
        if not ok:
            return None, error, node_errors
        self.n += 1
        pid = f"pid-{self.n}"
        collector = [n for n in prompt.values() if n["class_type"] == "USCGSweepCollect"][0]
        self.meta[pid] = json.loads(collector["inputs"]["job_json"])
        self.queue.append(pid)
        self.enqueued.append((pid, prompt, partial, client_id))
        return pid, None, {}

    def queue_state(self):
        return set(self.running), set(self.queue)

    def history_entry(self, pid):
        return self.history.get(pid)

    def delete_queued(self, pid):
        if pid in self.queue:
            self.queue.remove(pid)
            self.deleted.append(pid)
            return True
        return False

    def interrupt(self, pid):
        self.interrupted.append(pid)
        return pid in self.running

    def send_sync(self, event, data):
        self.events.append((event, data))

    # ---- test controls
    def start(self, pid):
        self.queue.remove(pid)
        self.running.append(pid)

    def finish(self, pid, kind="success", collect=True, ts=(1000, 3500)):
        if pid in self.queue:
            self.queue.remove(pid)
        if pid in self.running:
            self.running.remove(pid)
        msgs = [["execution_start", {"prompt_id": pid, "timestamp": ts[0]}]]
        if kind == "success":
            msgs.append(["execution_success", {"prompt_id": pid, "timestamp": ts[1]}])
            status = "success"
        elif kind == "error":
            msgs.append(["execution_error", {"prompt_id": pid, "node_id": "32", "node_type": "KSampler",
                                              "exception_message": "CUDA out of memory", "timestamp": ts[1]}])
            status = "error"
        else:
            msgs.append(["execution_interrupted", {"prompt_id": pid, "timestamp": ts[1]}])
            status = "error"
        self.history[pid] = {"status": {"status_str": status, "completed": kind == "success", "messages": msgs},
                             "outputs": {}}
        if kind == "success" and collect:
            meta = self.meta[pid]
            runner.on_collected(meta["sweep_id"], meta["job_index"], [{"id": 1}])

    def progress_events(self):
        return [d for e, d in self.events if e == "ultimate_grid.progress"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(runner, "manifest_utils", _real_manifest_utils(), raising=False)
    runner.reset_registry()
    bridge = FakeBridge()
    yield types.SimpleNamespace(bridge=bridge, tmp=tmp_path)
    runner.reset_registry()


def make(env, configs=None, settings=None, origin="node7", thread=False):
    settings = {"session_name": "wf_test", **(settings or {})}
    sw = runner.prepare_sweep(configs or [entry()], settings, origin, None, env.bridge, class_info=None)
    runner.launch(sw, start_thread=thread)
    return sw


def drive(sw, bridge, n=1):
    for _ in range(n):
        if not runner.tick(sw):
            return False
    return True


def complete_all(sw, bridge, limit=100):
    for _ in range(limit):
        for pid in list(bridge.queue):
            bridge.start(pid)
        for pid in list(bridge.running):
            bridge.finish(pid)
        if not runner.tick(sw):
            return
    raise AssertionError("sweep did not finish")


def test_window_respected_and_refilled(env):
    sw = make(env)
    drive(sw, env.bridge)
    assert len(env.bridge.queue) == runner.WINDOW == 2
    drive(sw, env.bridge, 3)
    assert len(env.bridge.enqueued) == 2        # nothing new while two are in flight
    pid = env.bridge.queue[0]
    env.bridge.start(pid)
    env.bridge.finish(pid)
    drive(sw, env.bridge)
    assert len(env.bridge.enqueued) == 3        # the freed slot is refilled
    assert len(sw.in_flight()) == 2


def test_enqueued_prompt_shape_and_synthetic_client_id(env):
    sw = make(env)
    drive(sw, env.bridge)
    pid, prompt, partial, client_id = env.bridge.enqueued[0]
    collector_id = [k for k, v in prompt.items() if v["class_type"] == "USCGSweepCollect"][0]
    assert partial == [collector_id]            # the template's SaveImage / PreviewAny are not executed
    assert client_id == f"uscg-sweep-{sw.id}"
    job_json = json.loads(prompt[collector_id]["inputs"]["job_json"])
    assert job_json["sweep_id"] == sw.id and job_json["job_index"] == 0
    assert "class_type" not in json.dumps(job_json)      # job conf only, never the template
    assert prompt[collector_id]["inputs"]["session_name"] == "wf_test"
    assert prompt["32"]["inputs"]["seed"] == 1


def test_run_other_outputs_executes_everything(env):
    sw = make(env, [entry(workflow_run_other_outputs=True)])
    drive(sw, env.bridge)
    assert env.bridge.enqueued[0][2] is None


def test_success_path_progress_events_and_completion(env):
    sw = make(env, [entry((1, 2, 3))])
    complete_all(sw, env.bridge)
    assert sw.state == "completed"
    assert [j["status"] for j in sw.jobs] == ["done"] * 3
    assert sw.jobs[0]["duration"] == pytest.approx(2.5)   # from ComfyUI's execution_start/success timestamps
    events = env.bridge.progress_events()
    assert events[0]["current_job"] == 0 and events[0]["total_jobs"] == 3 and events[0]["phase"] == "sweep"
    assert all(e["node"] == "node7" and e["session_name"] == "wf_test" for e in events)
    last = events[-1]
    assert last["complete"] is True and last["total_generated"] == 3 and last["eta_str"] == "Done"
    assert any(e["current_job"] == 2 and not e.get("complete") for e in events)
    status = runner.get_status(sw.id)
    assert status["done"] == 3 and status["errors"] == 0 and status["state"] == "completed"
    assert runner.get_status(session="wf_test")["sweep_id"] == sw.id


def test_spec_file_written_without_secrets(env):
    sw = make(env, [entry((1, 2))])
    complete_all(sw, env.bridge)
    path = env.tmp / "benchmarks" / "wf_test" / "workflow_sweeps" / f"{sw.id}.json"
    spec = json.loads(path.read_text())
    assert spec["state"] == "completed" and len(spec["jobs"]) == 2
    assert spec["jobs"][0]["subs"] == [["32", "seed", 1]] and spec["jobs"][0]["prompt_id"] == "pid-1"
    assert list(spec["templates"].values())[0]["output"] == ["38", 0]


def test_manifest_created_with_meta(env):
    make(env)
    data = json.loads((env.tmp / "benchmarks" / "wf_test" / "manifest.json").read_text())
    assert data["session_name"] == "wf_test" and data["items"] == [] and data["meta"]["image_format"] == "webp"


def test_job_error_recorded_and_sweep_continues(env):
    sw = make(env, [entry((1, 2, 3))])
    drive(sw, env.bridge)
    p1, p2 = env.bridge.queue[:2]
    env.bridge.start(p1)
    env.bridge.finish(p1, "error")
    env.bridge.start(p2)
    env.bridge.finish(p2)
    complete_all(sw, env.bridge)
    assert [j["status"] for j in sw.jobs] == ["error", "done", "done"]
    assert "CUDA out of memory" in sw.jobs[0]["error"] and "KSampler" in sw.jobs[0]["error"]
    assert sw.state == "completed" and "1 job" in sw.reason
    status = runner.get_status(sw.id)
    assert status["errors"] == 1 and status["last_errors"][0]["index"] == 0
    assert sw.consecutive_errors == 0            # reset by the later success


def test_three_consecutive_errors_pause(env):
    sw = make(env, [entry(range(1, 9))])
    for _ in range(3):
        drive(sw, env.bridge)
        for pid in list(env.bridge.queue):
            env.bridge.start(pid)
        for pid in list(env.bridge.running):
            env.bridge.finish(pid, "error")
            break
        drive(sw, env.bridge)
        if sw.state == "paused":
            break
    drive(sw, env.bridge)
    assert sw.state == "paused" and "in a row failed" in sw.reason and "CUDA" in sw.reason


def test_invalid_job_continues_and_first_invalid_fails_start(env):
    env.bridge.reject = lambda prompt: "Value not in list: unet_name" if prompt["32"]["inputs"]["seed"] == 2 else None
    sw = make(env, [entry((1, 2, 3))])
    complete_all(sw, env.bridge)
    assert [j["status"] for j in sw.jobs] == ["done", "invalid", "done"]
    assert "node 29 (UNETLoader" in sw.jobs[1]["error"] and "Value not in list" in sw.jobs[1]["error"]
    # job 1 invalid at start -> the sweep is refused with the node errors (route answers 400)
    env.bridge.reject = lambda prompt: "model missing"
    sw2 = runner.prepare_sweep([entry((1, 2))], {"session_name": "other"}, None, None, env.bridge, class_info=None)
    runner.reset_registry()
    with pytest.raises(ValueError, match="first job is not valid.*model missing"):
        asyncio.run(runner.validate_first(sw2))


def test_interrupted_pauses_and_resume_requeues(env):
    sw = make(env, [entry((1, 2, 3))])
    drive(sw, env.bridge)
    p1 = env.bridge.queue[0]
    env.bridge.start(p1)
    env.bridge.finish(p1, "interrupted")
    drive(sw, env.bridge)
    assert sw.state == "paused" and "interrupted" in sw.reason
    assert sw.jobs[0]["status"] == "interrupted"
    # the second (still queued) prompt is pulled while paused
    drive(sw, env.bridge)
    assert env.bridge.deleted and sw.jobs[1]["status"] == "pending"
    ok, state = runner.resume(sw.id)
    assert ok and state == "running" and sw.jobs[0]["status"] == "pending"
    complete_all(sw, env.bridge)
    assert [j["status"] for j in sw.jobs] == ["done"] * 3 and sw.state == "completed"


def test_lost_prompt_pauses(env, monkeypatch):
    monkeypatch.setattr(runner, "LOST_GRACE", -1)
    sw = make(env, [entry((1, 2, 3))])
    drive(sw, env.bridge)
    env.bridge.queue.clear()                       # user cleared ComfyUI's queue
    drive(sw, env.bridge)
    assert sw.state == "paused" and "removed from ComfyUI's queue" in sw.reason
    assert {j["status"] for j in sw.jobs[:2]} == {"lost"}
    runner.resume(sw.id)
    assert [j["status"] for j in sw.jobs[:2]] == ["pending", "pending"]


def test_success_without_collected_item_is_an_error(env, monkeypatch):
    monkeypatch.setattr(runner, "COLLECT_GRACE", -1)
    sw = make(env, [entry((1,))])
    drive(sw, env.bridge)
    pid = env.bridge.queue[0]
    env.bridge.start(pid)
    env.bridge.finish(pid, collect=False)
    drive(sw, env.bridge, 2)
    assert sw.jobs[0]["status"] == "error" and "collector" in sw.jobs[0]["error"]


def test_cancel_deletes_queued_interrupts_running_and_marks_rest(env):
    sw = make(env, [entry(range(1, 6))])
    drive(sw, env.bridge)
    p1, p2 = env.bridge.queue[:2]
    env.bridge.start(p1)
    drive(sw, env.bridge)
    assert sw.jobs[0]["status"] == "running"
    ok, state = runner.cancel(sw.id)
    assert ok and state == "cancelled"
    assert drive(sw, env.bridge) is False
    assert env.bridge.deleted == [p2] and env.bridge.interrupted == [p1]
    assert {j["status"] for j in sw.jobs} == {"cancelled"}
    assert runner.cancel(sw.id)[0] is False        # already finished
    assert runner.get_status(sw.id)["state"] == "cancelled"


def test_pause_pulls_queued_but_lets_running_finish_then_resume(env):
    sw = make(env, [entry((1, 2, 3))])
    drive(sw, env.bridge)
    p1, p2 = env.bridge.queue[:2]
    env.bridge.start(p1)
    drive(sw, env.bridge)
    assert runner.pause(sw.id) == (True, "paused")
    drive(sw, env.bridge)
    assert env.bridge.deleted == [p2] and sw.jobs[0]["status"] == "running" and sw.jobs[1]["status"] == "pending"
    env.bridge.finish(p1)
    drive(sw, env.bridge)
    assert sw.jobs[0]["status"] == "done" and sw.state == "paused" and not env.bridge.queue
    assert runner.pause(sw.id)[0] is False
    runner.resume(sw.id)
    complete_all(sw, env.bridge)
    assert sw.state == "completed"


def _write_manifest(env, items):
    d = env.tmp / "benchmarks" / "wf_test"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"session_name": "wf_test", "items": items, "meta": {"x": 1}}))


def test_resume_skips_fingerprints_in_manifest(env):
    plan = wfs.plan_workflow_configs([entry((1, 2, 3))])
    done_fps = [plan["jobs"][0]["wf_fingerprint"], plan["jobs"][2]["wf_fingerprint"]]
    _write_manifest(env, [{"id": 11, "wf_fingerprint": done_fps[0], "gen_index": 4},
                          {"id": 12, "wf_fingerprint": done_fps[1]}, {"id": 13}])
    sw = make(env, [entry((1, 2, 3))])
    assert [j["status"] for j in sw.jobs] == ["skipped", "pending", "skipped"]
    assert sw.skipped_existing == 2 and sw.to_run == 1
    assert sw.jobs[1]["gen_index"] == 5 + 1       # continues after the highest existing gen_index
    complete_all(sw, env.bridge)
    assert len(env.bridge.enqueued) == 1 and env.bridge.enqueued[0][1]["32"]["inputs"]["seed"] == 2
    first_event = env.bridge.progress_events()[0]
    assert first_event["current_job"] == 2 and first_event["total_jobs"] == 3   # "2/3 already done"


def test_overwrite_existing_reruns_and_carries_replace_ids(env):
    plan = wfs.plan_workflow_configs([entry((1, 2))])
    _write_manifest(env, [{"id": 11, "wf_fingerprint": plan["jobs"][0]["wf_fingerprint"]}])
    sw = make(env, [entry((1, 2))], {"overwrite_existing": True})
    assert [j["status"] for j in sw.jobs] == ["pending", "pending"] and sw.jobs[0]["replace_ids"] == [11]
    drive(sw, env.bridge)
    jj = env.bridge.meta[env.bridge.enqueued[0][0]]
    assert jj["replace_item_ids"] == [11]


def test_start_at_job_skips_first_n(env):
    sw = make(env, [entry((1, 2, 3, 4))], {"start_at_job": 2})
    assert [j["status"] for j in sw.jobs] == ["skipped", "skipped", "pending", "pending"]
    assert sw.skipped_existing == 0


def test_all_done_sweep_completes_immediately(env):
    plan = wfs.plan_workflow_configs([entry((1,))])
    _write_manifest(env, [{"id": 11, "wf_fingerprint": plan["jobs"][0]["wf_fingerprint"]}])
    sw = make(env, [entry((1,))])
    assert drive(sw, env.bridge) is False and sw.state == "completed"


def test_one_active_sweep_and_409(env):
    sw = make(env)
    with pytest.raises(runner.SweepBusy):
        runner.prepare_sweep([entry()], {"session_name": "x"}, None, None, env.bridge, class_info=None)
    status, body = asyncio.run(runner.api_start(
        {"configs_json": json.dumps({"configs": [entry()], "_session_settings": {"session_name": "x"}})},
        bridge=env.bridge, class_info=None))
    assert status == 409 and body["sweep_id"] == sw.id
    runner.cancel(sw.id)
    drive(sw, env.bridge)
    runner.prepare_sweep([entry()], {"session_name": "x"}, None, None, env.bridge, class_info=None)  # free again


def test_api_start_ok_and_bad_input(env):
    body = {"configs_json": json.dumps({"configs": [entry((1, 2))], "_session_settings": {"session_name": "api"}}),
            "node_id": "5"}
    status, out = asyncio.run(runner.api_start(body, bridge=env.bridge, class_info=None))
    assert status == 200 and out["total"] == 2 and out["to_run"] == 2
    runner.cancel(out["sweep_id"])
    time.sleep(runner.TICK * 3)
    status, out = asyncio.run(runner.api_start({"configs_json": json.dumps({"configs": [{"model_type": "checkpoint"}]})},
                                               bridge=env.bridge, class_info=None))
    assert status == 400 and "cannot be mixed" in out["error"]
    status, out = asyncio.run(runner.api_start({}, bridge=env.bridge, class_info=None))
    assert status == 400


def test_start_sweep_returns_while_jobs_pending_and_never_blocks(env):
    t0 = time.time()
    sw = runner.start_sweep([entry(range(1, 50))], {"session_name": "nb"}, "n1", None, env.bridge, class_info=None)
    assert time.time() - t0 < 2 and sw.state == "running"
    time.sleep(runner.TICK * 3)                    # the feeder thread fills the window on its own
    assert len(env.bridge.enqueued) == runner.WINDOW
    assert runner.cancel(sw.id)[0]
    sw.thread.join(5)
    assert not sw.thread.is_alive() and sw.state == "cancelled"


def test_dry_run_shape_and_no_queueing(env):
    body = {"configs_json": json.dumps({"configs": [entry((1, 2, 3))], "_session_settings": {"session_name": "dry"}}),
            "limit": 2}
    out = asyncio.run(runner.dry_run(body, bridge=env.bridge, class_info=None))
    assert out["total"] == 3 and out["to_run"] == 3 and len(out["jobs"]) == 2
    assert out["validation"]["ok"] is True and out["axes"][0]["count"] == 3
    j = out["jobs"][0]
    assert j["prompt"]["32"]["inputs"]["seed"] == 1 and j["partial_execution_targets"]
    assert any(n["class_type"] == "USCGSweepCollect" for n in j["prompt"].values())
    assert env.bridge.enqueued == [] and runner._active_sweep() is None
    env.bridge.reject = lambda p: "bad model"
    out = asyncio.run(runner.dry_run(body, bridge=env.bridge, class_info=None))
    assert out["validation"]["ok"] is False and "bad model" in out["validation"]["error"]


def test_dry_run_allowed_while_a_sweep_is_active(env):
    make(env)
    body = {"configs_json": json.dumps({"configs": [entry((1,))], "_session_settings": {"session_name": "d2"}})}
    assert asyncio.run(runner.dry_run(body, bridge=env.bridge, class_info=None))["total"] == 1


def test_configs_from_body_uses_builder_state_transformer():
    state = {"session_name": "st", "config_arrays": [{
        "name": "WF", "kind": "workflow",
        "workflow": {"api_prompt": load("sdxl_default_api.json"), "output": {"node_id": "8", "output_index": 0},
                     "bindings": []}}]}
    configs, settings = runner.configs_from_body({"state": state})
    assert configs[0]["model_type"] == "workflow" and settings["session_name"] == "st"
    with pytest.raises(ValueError):
        runner.configs_from_body({})


def test_classify_history_entries():
    ok = {"status": {"status_str": "success", "messages": [["execution_start", {"timestamp": 1}],
                                                            ["execution_success", {"timestamp": 3}]]}}
    assert runner._classify(ok)[0] == "success"
    err = {"status": {"status_str": "error", "messages": [["execution_error", {"node_id": "3", "node_type": "K",
                                                                                 "exception_message": "boom"}]]}}
    kind, detail = runner._classify(err)
    assert kind == "error" and "boom" in detail["error"] and "K #3" in detail["error"]
    assert runner._classify({"status": {"status_str": "error", "messages": [["execution_interrupted", {}]]}})[0] == "interrupted"
    assert runner._classify({"status": None})[0] == "unknown"
    assert runner._classify(None)[0] == "unknown"


def test_summarize_errors():
    msg = runner.summarize_errors({"message": "Prompt outputs failed validation", "details": "x"},
                                  {"29": {"class_type": "UNETLoader", "errors": [{"message": "Value not in list",
                                                                                   "details": "unet_name: 'a'"}]}},
                                  load("image_moss_image_t2i.json"))
    assert "Prompt outputs failed validation" in msg and "node 29 (UNETLoader / Load Diffusion Model)" in msg


# ------------------------------------------------------------------ ComfyBridge mirrors server.py post_prompt

class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _fake_comfy(monkeypatch, with_sensitive=True, validate_args=3, valid=True, handlers=None, replace=True):
    calls = {"validate": [], "replace": [], "handlers": []}

    class PS:
        number = 5.0
        prompt_queue = _Queue()
        loop = None

        def trigger_on_prompt(self, json_data):
            calls["handlers"].append(dict(json_data))
            if handlers:
                json_data = handlers(json_data)
            return json_data

        class node_replace_manager:  # noqa: N801
            @staticmethod
            def apply_replacements(prompt):
                calls["replace"].append(prompt)
                prompt["9999"] = {"class_type": "Replaced", "inputs": {}}

    ps = PS()
    if not replace:
        del PS.node_replace_manager
    server_mod = sys.modules["server"]
    monkeypatch.setattr(server_mod, "PromptServer", types.SimpleNamespace(instance=ps), raising=False)
    monkeypatch.setattr(server_mod, "valid_workflow_metadata", lambda jd: jd.get("workflow_metadata"), raising=False)
    monkeypatch.setattr(server_mod, "workflow_metadata_from_prompt", lambda ed: None, raising=False)

    ex = types.ModuleType("execution")
    if validate_args == 3:
        async def validate_prompt(prompt_id, prompt, partial_execution_list):
            calls["validate"].append((prompt_id, partial_execution_list))
            return (valid, None if valid else {"message": "bad"}, ["38x"], {} if valid else {"3": {"errors": []}})
    elif validate_args == 2:
        def validate_prompt(prompt_id, prompt):
            calls["validate"].append((prompt_id, None))
            return (valid, None, ["38x"], {})
    else:
        def validate_prompt(prompt):
            calls["validate"].append((None, None))
            return (valid, None, ["38x"], {})
    ex.validate_prompt = validate_prompt
    if with_sensitive:
        ex.SENSITIVE_EXTRA_DATA_KEYS = ("auth_token_comfy_org", "api_key_comfy_org")
    monkeypatch.setitem(sys.modules, "execution", ex)
    return ps, calls


def test_bridge_enqueue_mirrors_post_prompt(monkeypatch):
    ps, calls = _fake_comfy(monkeypatch)
    prompt = {"1": {"class_type": "X", "inputs": {}}}
    pid, err, node_errors = asyncio.run(runner.ComfyBridge().enqueue(prompt, ["1"], "cid"))
    assert err is None and node_errors == {}
    (item,) = ps.prompt_queue.items
    assert len(item) == 6                              # (number, prompt_id, prompt, extra_data, outputs, sensitive)
    number, prompt_id, queued_prompt, extra, outputs, sensitive = item
    assert number == 5.0 and ps.number == 6.0          # number taken, then incremented
    assert prompt_id == pid and len(pid) == 36
    assert queued_prompt is prompt and "9999" in prompt  # node replacements applied before validation
    assert extra["client_id"] == "cid" and isinstance(extra["create_time"], int)
    assert outputs == ["38x"] and sensitive == {}
    assert calls["validate"] == [(pid, ["1"])]
    assert calls["handlers"][0]["prompt"] is prompt and calls["handlers"][0]["partial_execution_targets"] == ["1"]


def test_bridge_enqueue_honours_on_prompt_handler_and_strips_sensitive(monkeypatch):
    def handler(jd):
        jd["extra_data"] = {"api_key_comfy_org": "SECRET", "other": 1}
        jd["number"] = 2
        return jd
    ps, _ = _fake_comfy(monkeypatch, handlers=handler)
    asyncio.run(runner.ComfyBridge().enqueue({"1": {"class_type": "X", "inputs": {}}}, None, "c"))
    number, _, _, extra, _, sensitive = ps.prompt_queue.items[0]
    assert number == 2.0 and ps.number == 5.0          # an explicit number does not consume the counter
    assert sensitive == {"api_key_comfy_org": "SECRET"} and "api_key_comfy_org" not in extra and extra["other"] == 1


def test_bridge_enqueue_older_server_shapes(monkeypatch):
    ps, calls = _fake_comfy(monkeypatch, with_sensitive=False, validate_args=2, replace=False)
    pid, err, _ = asyncio.run(runner.ComfyBridge().enqueue({"1": {"class_type": "X", "inputs": {}}}, None, "c"))
    assert err is None and len(ps.prompt_queue.items[0]) == 5 and calls["validate"][0][0] == pid
    ps, calls = _fake_comfy(monkeypatch, validate_args=1)
    asyncio.run(runner.ComfyBridge().enqueue({"1": {"class_type": "X", "inputs": {}}}, None, "c"))
    assert len(ps.prompt_queue.items[0]) == 6


def test_bridge_enqueue_invalid_prompt_is_not_queued(monkeypatch):
    ps, _ = _fake_comfy(monkeypatch, valid=False)
    pid, err, node_errors = asyncio.run(runner.ComfyBridge().enqueue({"1": {"class_type": "X", "inputs": {}}}, None, "c"))
    assert pid is None and err == {"message": "bad"} and "3" in node_errors and ps.prompt_queue.items == []


def test_bridge_validate_does_not_queue(monkeypatch):
    ps, calls = _fake_comfy(monkeypatch)
    ok, err, ne = asyncio.run(runner.ComfyBridge().validate({"1": {"class_type": "X", "inputs": {}}}, ["1"]))
    assert ok and ps.prompt_queue.items == [] and len(calls["validate"]) == 1


def test_run_on_loop_refuses_the_loop_thread_and_works_from_others(monkeypatch):
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    ps, _ = _fake_comfy(monkeypatch)
    ps.loop = loop
    bridge = runner.ComfyBridge()

    async def double(x):
        return x * 2
    assert bridge.run_on_loop(double(21)) == 42

    async def from_loop():
        c = double(1)
        with pytest.raises(RuntimeError, match="event loop thread"):
            bridge.run_on_loop(c)
        return "refused"
    assert asyncio.run_coroutine_threadsafe(from_loop(), loop).result(5) == "refused"
    loop.call_soon_threadsafe(loop.stop)
    t.join(5)


def test_bridge_queue_helpers(monkeypatch):
    removed = []

    class Q:
        def get_current_queue_volatile(self):
            return [(0, "r1", {}, {}, [])], [(1, "q1", {}, {}, [], {}), (2, "q2", {}, {}, [], {})]

        def get_history(self, pid, map_function=None):
            return {pid: map_function({"status": {"status_str": "success"}, "outputs": {}, "prompt": "big"})} \
                if pid == "h1" else {}

        def delete_queue_item(self, fn):
            removed.append(fn((1, "q1")))
            return True

        def interrupt_if_running(self, pid):
            return pid == "r1"
    ps, _ = _fake_comfy(monkeypatch)
    ps.prompt_queue = Q()
    b = runner.ComfyBridge()
    assert b.queue_state() == ({"r1"}, {"q1", "q2"})
    assert b.history_entry("h1") == {"status": {"status_str": "success"}, "outputs": {}}   # no deep copy of prompt
    assert b.history_entry("nope") is None
    assert b.delete_queued("q1") is True and removed == [True]
    assert b.interrupt("r1") is True and b.interrupt("q1") is False


# --------------------------------------------------------------------------- attach to the identical sweep

def _cfg_body(configs, session):
    return {"configs_json": json.dumps({"configs": configs, "_session_settings": {"session_name": session}})}


def test_same_sweep_attaches_and_queues_nothing(env):
    sw = runner.start_sweep([entry()], {"session_name": "wf_test"}, "n1", None, env.bridge, class_info=None)
    n = len(env.bridge.enqueued)
    again = runner.start_sweep([entry()], {"session_name": "wf_test"}, "n2", None, env.bridge, class_info=None)
    assert again is sw
    assert len(runner._sweeps) == 1 and len(env.bridge.enqueued) == n
    runner.cancel(sw.id)
    sw.thread.join(5)


def test_api_start_same_sweep_returns_attached_200(env):
    sw = make(env)
    assert runner.pause(sw.id)[0]
    status, body = asyncio.run(runner.api_start(_cfg_body([entry()], "wf_test"), bridge=env.bridge, class_info=None))
    assert status == 200 and body["attached"] is True and body["sweep_id"] == sw.id
    assert body["state"] == "paused" and sw.state == "paused"      # attach never changes the sweep state
    assert len(runner._sweeps) == 1 and env.bridge.enqueued == []


def test_different_sweep_409_with_session_and_message(env):
    sw = make(env)
    # different values
    status, body = asyncio.run(runner.api_start(_cfg_body([entry((7, 8))], "wf_test"), bridge=env.bridge, class_info=None))
    assert status == 409 and body["sweep_id"] == sw.id and body["session_name"] == "wf_test"
    assert "NOT started" in body["error"] and "already active" in body["error"]
    # same workflow, different session
    status, body = asyncio.run(runner.api_start(_cfg_body([entry()], "other_session"), bridge=env.bridge, class_info=None))
    assert status == 409
    # same workflow and session but a different start_at_job / overwrite_existing is NOT the same sweep
    for extra in ({"start_at_job": 2}, {"overwrite_existing": True}):
        with pytest.raises(runner.SweepBusy) as ei:
            runner.prepare_sweep([entry()], {"session_name": "wf_test", **extra}, None, None, env.bridge, class_info=None)
        assert not isinstance(ei.value, runner.SweepAlreadyRunning)
    with pytest.raises(runner.SweepBusy) as ei:
        runner.prepare_sweep([entry((7, 8))], {"session_name": "wf_test"}, None, None, env.bridge, class_info=None)
    assert not isinstance(ei.value, runner.SweepAlreadyRunning) and "NOT started" in str(ei.value)
    assert len(runner._sweeps) == 1


def test_extra_seeds_deterministic_and_attach(env):
    sw = make(env, settings={"add_random_seeds_to_gens": 2})
    raw = {"configs": [entry()], "_session_settings": {"session_name": "wf_test", "add_random_seeds_to_gens": 2}}
    status, body = asyncio.run(runner.api_start({"configs_json": json.dumps(raw)}, bridge=env.bridge, class_info=None))
    assert status == 200 and body["attached"] is True and body["sweep_id"] == sw.id


def test_launch_race_same_sweep(env):
    a = runner.prepare_sweep([entry()], {"session_name": "race"}, None, None, env.bridge, class_info=None)
    b = runner.prepare_sweep([entry()], {"session_name": "race"}, None, None, env.bridge, class_info=None)
    runner.launch(a, start_thread=False)
    with pytest.raises(runner.SweepAlreadyRunning) as ei:
        runner.launch(b, start_thread=False)
    assert ei.value.sweep is a and b.id not in runner._sweeps


def test_pause_and_resume_emit_progress(env):
    sw = make(env)
    env.bridge.events.clear()
    runner.pause(sw.id)
    runner.resume(sw.id)
    states = [d["state"] for d in env.bridge.progress_events()]
    assert states == ["paused", "running"]


# --------------------------------------------------------------------------- job_prompt + sweep meta

def test_job_prompt_matches_apply_subs_and_has_no_collector(env):
    sw = make(env, [entry((1, 2, 3))])
    status, body = runner.job_prompt("wf_test", sw.id, 1)
    job = sw.jobs[1]
    tpl = sw.templates[job["conf"]["wf_template_hash"]]
    assert status == 200 and body["prompt"] == wfs.apply_subs(tpl["prompt"], job["conf"]["wf_subs"])
    assert "USCGSweepCollect" not in json.dumps(body["prompt"])
    assert body["fingerprint"] == job["fingerprint"] and body["job_index"] == 1 and body["params"] is not None
    # from the spec file, after the in-memory registry is gone
    runner.reset_registry()
    status2, body2 = runner.job_prompt("wf_test", sw.id, "1")
    assert status2 == 200 and body2["prompt"] == body["prompt"] and body2["fingerprint"] == body["fingerprint"]


def test_job_prompt_validation_and_404(env):
    sw = make(env)
    for sid, idx in (("../x", 0), ("abc", 0), (sw.id, -1), (sw.id, "x"), (sw.id, "1.5"), ("", 0)):
        assert runner.job_prompt("wf_test", sid, idx)[0] == 400, (sid, idx)
    assert runner.job_prompt("wf_test", "deadbeef0000", 0)[0] == 404
    assert runner.job_prompt("wf_test", sw.id, 99)[0] == 404
    assert runner.job_prompt("some_other_session", sw.id, 0)[0] == 404   # wrong session never leaks another's job


def test_new_manifest_meta_carries_configs_json(env):
    sw = make(env)
    meta = wfs.read_manifest_items(wfs.session_paths("wf_test")["manifest"])["meta"]
    parsed = json.loads(meta["configs_json"])
    assert parsed["configs"][0]["model_type"] == "workflow" and parsed["_session_settings"]["session_name"] == "wf_test"
    assert meta["workflow_sweep"] is True


def test_existing_sweep_manifest_meta_refreshed_but_standard_untouched(env):
    paths = wfs.session_paths("wf_test")
    os.makedirs(paths["images"], exist_ok=True)
    old = {"session_name": "wf_test", "meta": {"workflow_sweep": True, "configs_json": ""}, "items": [{"id": 5, "file": "x"}]}
    with open(paths["manifest"], "w") as f:
        json.dump(old, f)
    make(env)
    data = wfs.read_manifest_items(paths["manifest"])
    assert json.loads(data["meta"]["configs_json"])["configs"] and data["items"] == [{"id": 5, "file": "x"}]
    runner.reset_registry()
    std = {"session_name": "std_sess", "meta": {"configs_json": "{}"}, "items": []}
    p2 = wfs.session_paths("std_sess")
    os.makedirs(p2["images"], exist_ok=True)
    with open(p2["manifest"], "w") as f:
        json.dump(std, f)
    before = open(p2["manifest"], "rb").read()
    make(env, settings={"session_name": "std_sess"})
    assert open(p2["manifest"], "rb").read() == before


def test_update_meta_preserves_items_written_since(tmp_path):
    mu = _real_manifest_utils()
    path = str(tmp_path / "manifest.json")
    with open(path, "w") as f:
        json.dump({"meta": {"a": 1}, "items": []}, f)
    with open(path, "w") as f:                      # another writer adds an item after the caller last looked
        json.dump({"meta": {"a": 1}, "items": [{"id": 9}]}, f)
    mu.update_meta(path, {"configs_json": "z"})
    data = json.load(open(path))
    assert data["items"] == [{"id": 9}] and data["meta"] == {"a": 1, "configs_json": "z"}


def _job_seeds(sw):
    return sorted(j["conf"]["seed"] for j in sw.jobs)


def test_extra_seeds_use_derive_extra_seeds_and_base_seed(env):
    cu = wfs._cu()
    one = lambda **s: runner.prepare_sweep([entry((1,))], {"session_name": "s", "add_random_seeds_to_gens": 2, **s},
                                           None, None, env.bridge, class_info=None)
    a, b = one(), one()
    assert _job_seeds(a) == _job_seeds(b) == sorted({1, *cu.derive_extra_seeds(0, 2)})
    assert _job_seeds(one(seed=5)) != _job_seeds(a)


def test_resume_with_extra_seeds_skips_everything(env):
    sw = make(env, [entry((1,))], {"add_random_seeds_to_gens": 2})
    complete_all(sw, env.bridge)
    assert sw.state == "completed" and len(sw.jobs) > 1
    # the (fake) collector wrote no items: record what a real run leaves in the manifest
    _write_manifest(env, [{"id": 100 + j["index"], "wf_fingerprint": j["fingerprint"]} for j in sw.jobs])
    runner.reset_registry()
    again = runner.prepare_sweep([entry((1,))], {"session_name": "wf_test", "add_random_seeds_to_gens": 2},
                                 None, None, env.bridge, class_info=None)
    assert again.to_run == 0 and again.skipped_existing == len(again.jobs) > 1
