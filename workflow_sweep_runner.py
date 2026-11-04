"""
Workflow Sweep runner: feeds a sweep's jobs to ComfyUI's own queue, one prompt per job.

A sweep is prepared synchronously (expand, fingerprint, skip jobs already in the manifest), then a
daemon thread keeps at most WINDOW sweep prompts queued/running and follows each through ComfyUI's
history. Every enqueue repeats server.py's post_prompt in-process (on_prompt handlers, node
replacements, awaited validate_prompt, number, extra_data, queue tuple), so a queued sweep prompt is
exactly what ComfyUI would have queued from its UI.

Hard rules honoured here:
  * no network code (the enqueue is an in-process call on the server event loop),
  * the executor is never blocked: nothing in this module (or the collector node) waits on a prompt
    it queued; the Generator node starts a sweep and returns immediately,
  * `run_on_loop` is never called from the loop thread, and no lock is held across it.

All loop-bound ComfyUI access goes through `ComfyBridge`, so the runner is testable with a fake.
"""

import asyncio
import inspect
import json
import os
import re
import threading
import time
import traceback
import uuid

try:
    from . import workflow_sweep as wfs
    from . import manifest_utils
except ImportError:  # imported as a top-level module (tests)
    import workflow_sweep as wfs
    import manifest_utils


WINDOW = 2                 # max sweep prompts queued/running at once
TICK = 0.25                # seconds between runner polls
LOST_GRACE = 5.0           # a prompt missing from queue, running set and history this long is "lost"
COLLECT_GRACE = 5.0        # success without a collected item for this long is an error
BREAKER = 3                # consecutive error/invalid jobs that pause the sweep
SPEC_EVERY = 10            # spec file rewrite cadence (terminal jobs)

ACTIVE_STATES = ("preparing", "running", "paused")
TERMINAL_JOB = ("done", "error", "invalid", "skipped", "cancelled")


class SweepBusy(Exception):
    """Another (different) workflow sweep is still active (route -> 409)."""
    def __init__(self, sweep_id, session=None, sweep=None):
        if sweep is not None:
            with sweep.lock:
                detail = f"session '{sweep.session}', sweep {sweep_id}, {sweep.count('done')}/{sweep.to_run} done, {sweep.state}"
            session = sweep.session
        else:
            detail = f"session '{session}', sweep {sweep_id}" if session else f"sweep {sweep_id}"
        super().__init__(f"A workflow sweep is already active ({detail}). This run was NOT started. "
                         f"Pause or cancel it in the Builder's Workflow Sweep panel (or POST "
                         f"/config_tester/sweep/cancel {{sweep_id}}), or wait for it to finish.")
        self.sweep_id = sweep_id
        self.session = session


class SweepAlreadyRunning(SweepBusy):
    """The identical sweep is already active: callers attach to it instead of failing."""
    def __init__(self, sweep):
        super().__init__(sweep.id, sweep=sweep)
        self.sweep = sweep


def summarize_errors(error, node_errors, prompt=None):
    """One readable line from ComfyUI's validate_prompt error + node_errors."""
    parts = []
    if isinstance(error, dict):
        msg = error.get("message") or error.get("type") or "invalid prompt"
        if error.get("details"):
            msg += f" ({error['details']})"
        parts.append(msg)
    elif error:
        parts.append(str(error))
    for nid, ne in (node_errors or {}).items():
        ne = ne if isinstance(ne, dict) else {}
        ct = ne.get("class_type") or ((prompt or {}).get(str(nid)) or {}).get("class_type") or "?"
        title = wfs.node_title(prompt, nid) if prompt else ct
        label = ct if title == ct else f"{ct} / {title}"
        msgs = "; ".join(f"{e.get('message', '')} {e.get('details', '')}".strip() for e in (ne.get("errors") or []))
        parts.append(f"node {nid} ({label}): {msgs or 'invalid'}")
    return (" | ".join(parts) or "invalid prompt")[:1500]


# --------------------------------------------------------------------------- bridge to ComfyUI

class ComfyBridge:
    """Thin wrapper over PromptServer.instance / execution / nodes (lazy imports, version guards)."""

    def _ps(self):
        import server
        return server.PromptServer.instance

    def _execution(self):
        import execution
        return execution

    def run_on_loop(self, coro, timeout=300):
        """Run a coroutine on the server event loop from a NON-loop thread and wait for it."""
        ps = self._ps()
        loop = ps.loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            coro.close()
            raise RuntimeError("run_on_loop was called from the server event loop thread (would deadlock)")
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)

    async def _validate(self, prompt_id, prompt, partial):
        execution = self._execution()
        fn = execution.validate_prompt
        n = len(inspect.signature(fn).parameters)
        if n >= 3:
            result = fn(prompt_id, prompt, partial)
        elif n == 2:
            result = fn(prompt_id, prompt)
        else:
            result = fn(prompt)
        if inspect.isawaitable(result):
            result = await result
        return result  # (valid, error, outputs, node_errors)

    async def validate(self, prompt, partial_targets):
        """Validate without queueing. Returns (ok, error, node_errors)."""
        valid = await self._validate(str(uuid.uuid4()), prompt, partial_targets)
        return bool(valid[0]), valid[1], valid[3] if len(valid) > 3 else {}

    async def enqueue(self, prompt, partial_targets, client_id):
        """Mirror of server.py post_prompt (ComfyUI 0.42.5). Returns (prompt_id|None, error, node_errors)."""
        ps = self._ps()
        execution = self._execution()
        json_data = {"prompt": prompt, "client_id": client_id}
        if partial_targets is not None:
            json_data["partial_execution_targets"] = partial_targets
        if hasattr(ps, "trigger_on_prompt"):
            json_data = ps.trigger_on_prompt(json_data)

        if "number" in json_data:
            number = float(json_data["number"])
        else:
            number = ps.number
            if "front" in json_data and json_data["front"]:
                number = -number
            ps.number += 1

        if "prompt" not in json_data:
            return None, {"type": "no_prompt", "message": "No prompt provided"}, {}
        prompt = json_data["prompt"]
        prompt_id = str(uuid.uuid4())
        partial = json_data.get("partial_execution_targets")

        manager = getattr(ps, "node_replace_manager", None)
        if manager is not None and hasattr(manager, "apply_replacements"):
            manager.apply_replacements(prompt)

        valid = await self._validate(prompt_id, prompt, partial)
        extra_data = json_data.get("extra_data", {})
        if not isinstance(extra_data, dict):
            extra_data = {}
        if "client_id" in json_data:
            extra_data["client_id"] = json_data["client_id"]

        # workflow metadata for websocket messages (0.42.x); absent on older servers
        extra_data.pop("workflow_metadata", None)
        try:
            import server as _server
            valid_meta = getattr(_server, "valid_workflow_metadata", None)
            meta_from_prompt = getattr(_server, "workflow_metadata_from_prompt", None)
            metadata = valid_meta(json_data) if valid_meta else None
            if metadata is None and meta_from_prompt:
                metadata = meta_from_prompt(extra_data)
            if metadata is not None:
                extra_data["workflow_metadata"] = metadata
        except Exception:
            pass

        if not valid[0]:
            return None, valid[1], (valid[3] if len(valid) > 3 else {})

        outputs_to_execute = valid[2]
        sensitive_keys = getattr(execution, "SENSITIVE_EXTRA_DATA_KEYS", None)
        extra_data["create_time"] = int(time.time() * 1000)
        if sensitive_keys is not None:  # 6-tuple queue items (0.42.x)
            sensitive = {}
            for key in sensitive_keys:
                if key in extra_data:
                    sensitive[key] = extra_data.pop(key)
            ps.prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive))
        else:
            ps.prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute))
        return prompt_id, None, (valid[3] if len(valid) > 3 else {})

    def queue_state(self):
        """(running prompt ids, queued prompt ids), one atomic snapshot."""
        running, queued = self._ps().prompt_queue.get_current_queue_volatile()
        return {item[1] for item in running}, {item[1] for item in queued}

    def history_entry(self, prompt_id):
        """{status:{status_str, messages}, outputs} or None when ComfyUI has no history for it."""
        q = self._ps().prompt_queue

        def slim(p):
            return {"status": p.get("status"), "outputs": p.get("outputs")}
        try:
            got = q.get_history(prompt_id, map_function=slim)
        except TypeError:  # older servers: no map_function
            got = q.get_history(prompt_id)
        return (got or {}).get(prompt_id)

    def delete_queued(self, prompt_id):
        return bool(self._ps().prompt_queue.delete_queue_item(lambda item: item[1] == prompt_id))

    def interrupt(self, prompt_id):
        q = self._ps().prompt_queue
        if hasattr(q, "interrupt_if_running"):
            return bool(q.interrupt_if_running(prompt_id))
        running, _ = self.queue_state()
        if prompt_id in running:
            import nodes
            nodes.interrupt_processing()
            return True
        return False

    def send_sync(self, event, data):
        try:
            self._ps().send_sync(event, data)
        except Exception:
            pass


_bridge = None


def get_bridge():
    global _bridge
    if _bridge is None:
        _bridge = ComfyBridge()
    return _bridge


def set_bridge(bridge):
    """Tests inject a FakeBridge."""
    global _bridge
    _bridge = bridge


# --------------------------------------------------------------------------- sweep model

class Sweep:
    def __init__(self, sweep_id, session, jobs, templates, origin_node, client_id, settings, warnings, axes, bridge):
        self.id = sweep_id
        self.session = session
        self.jobs = jobs
        self.templates = templates
        self.origin_node = origin_node
        self.client_id = client_id
        self.settings = settings            # {image_format, overwrite_existing, ...}
        self.warnings = warnings
        self.axes = axes
        self.bridge = bridge
        self.state = "preparing"
        self.reason = ""
        self.created_at = time.time()
        self.started_at = None
        self.finished_at = None
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.thread = None
        self.needs_cleanup = False
        self.consecutive_errors = 0
        self.durations = []
        self.last_errors = []
        self.terminal_since_spec = 0
        self.eta_start = None

    # counters (computed from job states so they can never drift)
    def count(self, *statuses):
        return sum(1 for j in self.jobs if j["status"] in statuses)

    @property
    def skipped_existing(self):
        return sum(1 for j in self.jobs if j["status"] == "skipped" and j.get("skip_reason") == "existing")

    @property
    def to_run(self):
        return sum(1 for j in self.jobs if j["status"] != "skipped")

    def in_flight(self):
        return [j for j in self.jobs if j["status"] in ("enqueued", "running")]


_sweeps = {}
_registry_lock = threading.RLock()


def reset_registry():
    """Tests only."""
    with _registry_lock:
        _sweeps.clear()


def _active_sweep():
    with _registry_lock:
        for sw in _sweeps.values():
            if sw.state in ACTIVE_STATES:
                return sw
    return None


def _same_sweep(busy, session, fingerprints, start_at=0, overwrite=False):
    """True when `busy` is the sweep the caller is trying to start: same session, same ordered job
    fingerprints, same start_at_job and overwrite_existing."""
    return (busy.session == session
            and [j["fingerprint"] for j in busy.jobs] == list(fingerprints)
            and int(busy.settings.get("start_at_job", 0) or 0) == int(start_at or 0)
            and bool(busy.settings.get("overwrite_existing", False)) == bool(overwrite))


def _find(sweep_id=None, session=None):
    with _registry_lock:
        if sweep_id:
            return _sweeps.get(str(sweep_id))
        if session:
            matches = [s for s in _sweeps.values() if s.session == session]
            return max(matches, key=lambda s: s.created_at) if matches else None
    return None


# --------------------------------------------------------------------------- prepare / validate / launch

def _manifest_state(manifest_path):
    """({fingerprint: [item ids]}, next gen_index) from the manifest on disk."""
    data = wfs.read_manifest_items(manifest_path)
    by_fp, max_gen = {}, -1
    for it in data.get("items") or []:
        if not isinstance(it, dict):
            continue
        fp = it.get("wf_fingerprint")
        if fp and it.get("id") is not None:
            by_fp.setdefault(fp, []).append(it["id"])
        g = it.get("gen_index")
        if isinstance(g, int) and g > max_gen:
            max_gen = g
    return by_fp, max_gen + 1


def prepare_sweep(configs, session_settings=None, origin_node=None, client_id=None, bridge=None,
                  class_info=wfs.comfy_class_info, check_busy=True):
    """
    Expand + fingerprint + skip finished jobs. Raises ValueError (readable) or SweepBusy.
    Pure apart from reading the session manifest.
    """
    settings = dict(session_settings or {})
    if not isinstance(configs, list) or not configs:
        raise ValueError("There are no configs to run.")
    extra_seeds = []
    try:
        n_extra = int(settings.get("add_random_seeds_to_gens", 0) or 0)
    except (TypeError, ValueError):
        n_extra = 0
    if n_extra > 0:
        # deterministic (same helper and base seed 0 as the Generator): a re-run or resume yields the same jobs
        extra_seeds = wfs._cu().derive_extra_seeds(settings.get("seed", 0), n_extra)
    plan = wfs.plan_workflow_configs(configs, extra_seeds, class_info=class_info)

    paths = wfs.session_paths(settings.get("session_name") or "workflow_sweep")
    overwrite = bool(settings.get("overwrite_existing", False))
    try:
        start_at = max(0, int(settings.get("start_at_job", 0) or 0))
    except (TypeError, ValueError):
        start_at = 0
    busy = _active_sweep() if check_busy else None
    if busy is not None:
        fps = [c["wf_fingerprint"] for c in plan["jobs"]]
        if _same_sweep(busy, paths["session"], fps, start_at, overwrite):
            raise SweepAlreadyRunning(busy)
        raise SweepBusy(busy.id, sweep=busy)
    by_fp, gen_base = _manifest_state(paths["manifest"])

    jobs = []
    for idx, conf in enumerate(plan["jobs"]):
        job = {"index": idx, "fingerprint": conf["wf_fingerprint"], "conf": conf, "status": "pending",
               "prompt_id": None, "error": None, "gen_index": gen_base + idx, "replace_ids": [],
               "enqueued_at": None, "started_at": None, "finished_at": None, "duration": None,
               "collected": False, "missing_since": None, "success_seen_at": None}
        if idx < start_at:
            job["status"], job["skip_reason"] = "skipped", "start_at_job"
        elif conf["wf_fingerprint"] in by_fp:
            if overwrite:
                job["replace_ids"] = list(by_fp[conf["wf_fingerprint"]])
            else:
                job["status"], job["skip_reason"] = "skipped", "existing"
        jobs.append(job)

    sw = Sweep(uuid.uuid4().hex[:12], paths["session"], jobs, plan["templates"], origin_node,
               client_id or None, {"image_format": str(settings.get("image_format", "webp")),
                                   "overwrite_existing": overwrite, "start_at_job": start_at,
                                   "total_jobs": len(jobs)},
               list(plan["warnings"]), plan["axes"], bridge or get_bridge())
    if sw.client_id is None:
        sw.client_id = f"uscg-sweep-{sw.id}"
    sw.source_configs = configs
    return sw


def build_job_request(sw, job):
    """(prompt, partial_execution_targets) for one job."""
    conf = job["conf"]
    tpl = sw.templates[conf["wf_template_hash"]]
    job_json = {"conf": conf, "sweep_id": sw.id, "job_index": job["index"], "gen_index": job["gen_index"],
                "origin_node": sw.origin_node, "replace_item_ids": job["replace_ids"],
                "image_format": sw.settings.get("image_format", "webp")}
    prompt, cid = wfs.build_job_prompt(tpl["prompt"], conf["wf_subs"], tpl["output"],
                                       {"session_name": sw.session,
                                        "job_json": json.dumps(job_json, ensure_ascii=False)},
                                       tpl.get("media", "image"), tpl.get("frames_from"))
    partial = None if tpl["run_other_outputs"] else [cid]
    return prompt, partial


async def validate_first(sw):
    """Validate the first job that will run (on the event loop). Raises ValueError with node errors."""
    job = next((j for j in sw.jobs if j["status"] == "pending"), None)
    if job is None:
        return
    prompt, partial = build_job_request(sw, job)
    ok, error, node_errors = await sw.bridge.validate(prompt, partial)
    if not ok:
        raise ValueError("The first job is not valid: " + summarize_errors(error, node_errors, prompt))


def _configs_json(sw):
    """Builder-loadable configs for the manifest meta (the workflow entries, so Load Session can restore them)."""
    cfgs = getattr(sw, "source_configs", None)
    if not cfgs:
        return ""
    return json.dumps({"configs": cfgs, "_session_settings": {"session_name": sw.session}})


def _new_meta(sw):
    first = sw.jobs[0]["conf"] if sw.jobs else {}
    return {"positive": first.get("positive", ""), "negative": first.get("negative", ""),
            "model": first.get("model", ""), "seed": first.get("seed", 0) if isinstance(first.get("seed", 0), int) else 0,
            "vae_batch_size": 1, "configs_json": _configs_json(sw), "resolutions_json": "[]",
            "image_format": sw.settings.get("image_format", "webp"), "workflow_sweep": True}


def ensure_manifest(sw):
    """Create the session manifest when missing (existing manifests are left alone)."""
    paths = wfs.session_paths(sw.session)
    os.makedirs(paths["images"], exist_ok=True)
    if not os.path.exists(paths["manifest"]):
        data = {"session_name": sw.session, "meta": _new_meta(sw), "items": []}
        with open(paths["manifest"], "w") as f:
            json.dump(data, f, indent=4)
        return data
    data = wfs.read_manifest_items(paths["manifest"])
    cj = _configs_json(sw)
    if cj and (data.get("meta") or {}).get("workflow_sweep"):
        try:
            manifest_utils.update_meta(paths["manifest"], {"configs_json": cj})
        except Exception as e:
            print(f"[GridTester] ⚠️ Could not refresh the sweep session's saved configs: {e}")
    return data


def launch(sw, start_thread=True):
    """Register the sweep and start its feeder thread (tests drive tick() themselves with start_thread=False)."""
    with _registry_lock:
        busy = _active_sweep()
        if busy is not None and busy is not sw:
            if _same_sweep(busy, sw.session, [j["fingerprint"] for j in sw.jobs],
                           sw.settings.get("start_at_job", 0), sw.settings.get("overwrite_existing", False)):
                raise SweepAlreadyRunning(busy)
            raise SweepBusy(busy.id, sweep=busy)
        _sweeps[sw.id] = sw
    ensure_manifest(sw)
    with sw.lock:
        sw.state = "running"
        sw.started_at = sw.eta_start = time.time()
    print(f"[GridTester] 🧪 Workflow sweep {sw.id}: {sw.to_run} job(s) to run, "
          f"{sw.skipped_existing} already in the manifest (session '{sw.session}')")
    for w in sw.warnings:
        print(f"[GridTester] ⚠️ {w}")
    _write_spec(sw)
    _progress(sw)
    if start_thread:
        sw.thread = threading.Thread(target=_run, args=(sw,), name=f"uscg-sweep-{sw.id}", daemon=True)
        sw.thread.start()
    return sw


def start_sweep(configs, session_settings=None, origin_node=None, client_id=None, bridge=None,
                class_info=wfs.comfy_class_info):
    """Prepare + validate job 1 on the loop + launch. For a NON-loop thread (the Generator node).
    Returns immediately after launch; never waits for any sweep prompt."""
    try:
        sw = prepare_sweep(configs, session_settings, origin_node, client_id, bridge, class_info)
        sw.bridge.run_on_loop(validate_first(sw))
        return launch(sw)
    except SweepAlreadyRunning as e:
        if e.sweep.state == "paused":
            ok, msg = resume(e.sweep.id)
            print(f"[GridTester] ▶️ Same workflow sweep was paused ({e.sweep.id}, session "
                  f"'{e.sweep.session}'): resumed ({msg})")
            return e.sweep
        print(f"[GridTester] 🔁 Same workflow sweep already running ({e.sweep.id}, session "
              f"'{e.sweep.session}'): attaching, nothing re-queued")
        return e.sweep


# --------------------------------------------------------------------------- feeder thread

def _run(sw):
    try:
        while True:
            alive = tick(sw)
            if not alive:
                break
            sw.wake.wait(TICK)
            sw.wake.clear()
    except Exception as e:  # never let the feeder die silently
        traceback.print_exc()
        with sw.lock:
            sw.state, sw.reason = "failed", f"internal error: {e}"
            sw.finished_at = time.time()
        _write_spec(sw)
        _progress(sw)


def _fail_job(sw, job, status, error, params=None):
    job["status"] = status
    job["error"] = error
    job["finished_at"] = time.time()
    sw.last_errors.append({"index": job["index"], "status": status, "error": error,
                           "params": job["conf"].get("wf_params")})
    del sw.last_errors[:-20]
    sw.consecutive_errors += 1
    sw.terminal_since_spec += 1
    print(f"[GridTester] ❌ Sweep job {job['index'] + 1}/{len(sw.jobs)} {status}: {error}")
    if sw.consecutive_errors >= BREAKER and sw.state == "running":
        sw.state = "paused"
        sw.reason = f"{BREAKER} jobs in a row failed; last: {error}"


def _classify(history):
    """-> (kind, detail) with kind in success|error|interrupted|unknown, from a ComfyUI history entry."""
    status = (history or {}).get("status") or {}
    messages = status.get("messages") or []
    names = {}
    for m in messages:
        if isinstance(m, (list, tuple)) and len(m) == 2:
            names[m[0]] = m[1] if isinstance(m[1], dict) else {}
    if "execution_interrupted" in names:
        return "interrupted", names["execution_interrupted"]
    if "execution_error" in names or status.get("status_str") == "error":
        d = names.get("execution_error", {})
        text = d.get("exception_message") or "execution error"
        if d.get("node_type") or d.get("node_id"):
            text = f"{d.get('node_type', '?')} #{d.get('node_id', '?')}: {text}"
        return "error", {"error": text.strip()[:1500], "start": names.get("execution_start"),
                         "end": names.get("execution_error")}
    if status.get("status_str") == "success" or "execution_success" in names:
        return "success", {"start": names.get("execution_start"), "end": names.get("execution_success")}
    return "unknown", {}


def _poll(sw, inflight):
    b = sw.bridge
    now = time.time()
    running_ids, queued_ids = b.queue_state()   # snapshot FIRST, then history (race-free ordering)
    for job in inflight:
        pid = job["prompt_id"]
        if pid in running_ids:
            job["missing_since"] = None
            if job["status"] == "enqueued":
                job["status"], job["started_at"] = "running", now
            continue
        if pid in queued_ids:
            job["missing_since"] = None
            continue
        history = b.history_entry(pid)
        if history is None:
            job["missing_since"] = job["missing_since"] or now
            if now - job["missing_since"] > LOST_GRACE:
                with sw.lock:
                    job["status"] = "lost"
                    job["error"] = "the prompt disappeared from ComfyUI's queue and history (queue cleared?)"
                    if sw.state == "running":
                        sw.state, sw.reason = "paused", "a sweep prompt was removed from ComfyUI's queue; resume to re-run it"
                print(f"[GridTester] ⚠️ Sweep job {job['index'] + 1} lost; sweep paused")
            continue
        kind, detail = _classify(history)
        with sw.lock:
            if kind == "success":
                job["success_seen_at"] = job["success_seen_at"] or now
                if job["collected"]:
                    _finish_done(sw, job, detail, now)
                elif now - job["success_seen_at"] > COLLECT_GRACE:
                    _fail_job(sw, job, "error", "the job ran but the collector produced no output")
                    _after_terminal(sw)
            elif kind == "error":
                _fail_job(sw, job, "error", detail["error"])
                _after_terminal(sw)
            elif kind == "interrupted":
                job["status"], job["error"] = "interrupted", "interrupted in ComfyUI"
                if sw.state == "running":
                    sw.state, sw.reason = "paused", "a job was interrupted in ComfyUI; resume to re-run it"
                print(f"[GridTester] ⏸️ Sweep job {job['index'] + 1} interrupted; sweep paused")
            else:
                job["missing_since"] = job["missing_since"] or now
                if now - job["missing_since"] > LOST_GRACE:
                    _fail_job(sw, job, "error", "ComfyUI finished the prompt without a recognisable status")
                    _after_terminal(sw)


def _duration_from(detail, fallback):
    try:
        start, end = detail.get("start") or {}, detail.get("end") or {}
        d = (float(end["timestamp"]) - float(start["timestamp"])) / 1000.0
        if d >= 0:
            return d
    except (KeyError, TypeError, ValueError):
        pass
    return fallback


def _finish_done(sw, job, detail, now):
    job["status"] = "done"
    job["finished_at"] = now
    fallback = (now - job["started_at"]) if job["started_at"] else 0.0
    job["duration"] = _duration_from(detail, fallback)
    sw.durations.append(job["duration"])
    sw.consecutive_errors = 0
    sw.terminal_since_spec += 1
    _after_terminal(sw)


def _after_terminal(sw):
    _progress(sw)
    if sw.terminal_since_spec >= SPEC_EVERY:
        sw.terminal_since_spec = 0
        _write_spec(sw)


def _enqueue_job(sw, job):
    try:
        prompt, partial = build_job_request(sw, job)
    except Exception as e:
        with sw.lock:
            _fail_job(sw, job, "invalid", f"could not build the prompt: {e}")
            _after_terminal(sw)
        return
    try:
        pid, error, node_errors = sw.bridge.run_on_loop(sw.bridge.enqueue(prompt, partial, sw.client_id))
    except Exception as e:
        with sw.lock:
            _fail_job(sw, job, "invalid", f"could not queue the prompt: {e}")
            _after_terminal(sw)
        return
    with sw.lock:
        if pid is None:
            _fail_job(sw, job, "invalid", summarize_errors(error, node_errors, prompt))
            _after_terminal(sw)
        else:
            job["status"], job["prompt_id"], job["enqueued_at"] = "enqueued", pid, time.time()
            job["missing_since"] = None


def _do_cleanup(sw):
    """Cancel: drop our queued prompts, interrupt the running one, mark the rest cancelled."""
    b = sw.bridge
    for job in list(sw.in_flight()):
        pid = job["prompt_id"]
        try:
            if job["status"] == "enqueued" and b.delete_queued(pid):
                pass
            else:
                b.interrupt(pid)
        except Exception as e:
            print(f"[GridTester] ⚠️ Sweep cancel could not remove {pid}: {e}")
    with sw.lock:
        for job in sw.jobs:
            if job["status"] in ("pending", "enqueued", "running", "interrupted", "lost"):
                job["status"] = "cancelled"
        sw.needs_cleanup = False
        sw.finished_at = time.time()
    _write_spec(sw)
    _progress(sw)


def _finish(sw):
    with sw.lock:
        errors = sw.count("error", "invalid")
        sw.state = "completed"
        sw.reason = f"{errors} job(s) failed" if errors else ""
        sw.finished_at = time.time()
    print(f"[GridTester] 🎉 Workflow sweep {sw.id} complete: {sw.count('done')} image(s), {sw.count('error', 'invalid')} failed")
    _write_spec(sw)
    _progress(sw, complete=True)


def tick(sw):
    """One feeder iteration. Returns False when the sweep is over (thread should exit)."""
    b = sw.bridge
    with sw.lock:
        if sw.state in ("completed", "failed") or (sw.state == "cancelled" and not sw.needs_cleanup):
            return False
        cleanup = sw.state == "cancelled" and sw.needs_cleanup
        inflight = sw.in_flight()
    if cleanup:
        _do_cleanup(sw)
        return False

    if inflight:
        _poll(sw, inflight)

    with sw.lock:
        state = sw.state
        queued = [j for j in sw.in_flight() if j["status"] == "enqueued"]
    if state == "paused":
        for job in queued:               # pull what has not started; a started job finishes
            try:
                pulled = b.delete_queued(job["prompt_id"])
            except Exception:
                pulled = False
            if pulled:
                with sw.lock:
                    job.update(status="pending", prompt_id=None, enqueued_at=None, missing_since=None)
        return True

    while True:
        with sw.lock:
            if sw.state != "running":
                break
            nxt = next((j for j in sw.jobs if j["status"] == "pending"), None)
            if nxt is None or len(sw.in_flight()) >= WINDOW:
                break
        _enqueue_job(sw, nxt)

    with sw.lock:
        if sw.state != "running":
            return True
        if not sw.in_flight() and not any(j["status"] == "pending" for j in sw.jobs):
            finished = True
        else:
            finished = False
    if finished:
        _finish(sw)
        return False
    return True


# --------------------------------------------------------------------------- public controls

def pause(sweep_id):
    sw = _find(sweep_id)
    if sw is None:
        return False, "unknown sweep"
    with sw.lock:
        if sw.state != "running":
            return False, f"sweep is {sw.state}"
        sw.state, sw.reason = "paused", "paused by user"
    sw.wake.set()
    _progress(sw)
    return True, "paused"


def resume(sweep_id):
    sw = _find(sweep_id)
    if sw is None:
        return False, "unknown sweep"
    with sw.lock:
        if sw.state != "paused":
            return False, f"sweep is {sw.state}"
        for job in sw.jobs:
            if job["status"] in ("interrupted", "lost"):  # failed jobs stay failed; these two never really ran
                job.update(status="pending", prompt_id=None, error=None, missing_since=None, success_seen_at=None)
        sw.consecutive_errors = 0
        sw.state, sw.reason = "running", ""
    sw.wake.set()
    _progress(sw)
    return True, "running"


def cancel(sweep_id):
    sw = _find(sweep_id)
    if sw is None:
        return False, "unknown sweep"
    with sw.lock:
        if sw.state not in ACTIVE_STATES:
            return False, f"sweep is {sw.state}"
        sw.state, sw.reason, sw.needs_cleanup = "cancelled", "cancelled by user", True
    sw.wake.set()
    return True, "cancelled"


def on_collected(sweep_id, job_index, items):
    """Called by the collector node after it saved a job's image(s). False when the sweep is unknown."""
    sw = _find(sweep_id)
    if sw is None:
        return False
    try:
        job = sw.jobs[int(job_index)]
    except (IndexError, TypeError, ValueError):
        return False
    with sw.lock:
        job["collected"] = True
        job["items"] = [it.get("id") for it in (items or []) if isinstance(it, dict)]
    return True


def job_started_at(sweep_id, job_index):
    sw = _find(sweep_id)
    if sw is None:
        return None
    try:
        return sw.jobs[int(job_index)]["started_at"]
    except (IndexError, TypeError, ValueError):
        return None


def get_status(sweep_id=None, session=None):
    sw = _find(sweep_id, session)
    if sw is None:
        return None
    with sw.lock:
        done = sw.count("done")
        errors = sw.count("error", "invalid")
        recent = sw.durations[-10:]
        avg = sum(recent) / len(recent) if recent else 0.0
        remaining = sw.count("pending", "enqueued", "running", "interrupted", "lost")
        return {"sweep_id": sw.id, "session_name": sw.session, "state": sw.state, "reason": sw.reason,
                "total": len(sw.jobs), "to_run": sw.to_run, "skipped_existing": sw.skipped_existing,
                "done": done, "errors": errors, "in_flight": len(sw.in_flight()), "pending": sw.count("pending"),
                "eta_str": _eta_str(avg * remaining) if recent and sw.state == "running" else "",
                "avg_duration": round(avg, 1), "warnings": list(sw.warnings),
                "last_errors": list(sw.last_errors)}


_SWEEP_ID_RE = re.compile(r"^[0-9a-f]{6,32}$")


def job_prompt(session_name, sweep_id, job_index):
    """(http_status, body) for GET /config_tester/sweep/job_prompt: the API prompt of one sweep job
    exactly as it ran (templates + subs), minus the USCG collector node. Takes ids, never paths: the
    session name is sanitised by session_paths and the sweep id is regex-checked before it touches a path."""
    sweep_id = str(sweep_id or "")
    if not _SWEEP_ID_RE.match(sweep_id):
        return 400, {"error": "sweep_id must be 6-32 lowercase hex characters."}
    try:
        idx = int(str(job_index).strip())
        if idx < 0 or not re.match(r"^\d+$", str(job_index).strip()):
            raise ValueError
    except (TypeError, ValueError):
        return 400, {"error": "job_index must be an integer >= 0."}
    spec = None
    sw = _find(sweep_id)
    if sw is not None and sw.session == wfs.session_paths(session_name)["session"]:
        with sw.lock:
            if idx >= len(sw.jobs):
                return 404, {"error": f"Sweep {sweep_id} has no job {idx}."}
            job = sw.jobs[idx]
            tpl = sw.templates.get(job["conf"].get("wf_template_hash"))
            if tpl is None:
                return 404, {"error": "The job's workflow template is not available."}
            subs, params, fp = job["conf"].get("wf_subs") or [], job["conf"].get("wf_params"), job["fingerprint"]
            template, output = tpl["prompt"], tpl["output"]
    else:
        path = os.path.join(wfs.session_paths(session_name)["sweeps"], sweep_id + ".json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                spec = json.load(f)
        except (OSError, ValueError):
            return 404, {"error": f"No sweep {sweep_id} was recorded for session '{session_name}'."}
        jobs = spec.get("jobs") or []
        if idx >= len(jobs):
            return 404, {"error": f"Sweep {sweep_id} has no job {idx}."}
        job = jobs[idx]
        tpl = (spec.get("templates") or {}).get(job.get("template"))
        if tpl is None:
            return 404, {"error": "The job's workflow template is not in the sweep spec."}
        subs, params, fp = job.get("subs") or [], job.get("params"), job.get("fingerprint")
        template, output = tpl["prompt"], tpl.get("output")
    try:
        prompt = wfs.apply_subs(template, subs)
    except ValueError as e:
        return 404, {"error": str(e)}
    return 200, {"sweep_id": sweep_id, "job_index": idx, "fingerprint": fp, "params": params,
                 "output": output, "prompt": prompt}


# --------------------------------------------------------------------------- progress + spec file

def _eta_str(seconds):
    seconds = int(max(0, seconds))
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")


def _progress(sw, complete=False):
    """ultimate_grid.progress for the Dashboard's ETA bar (phase 'sweep')."""
    try:
        with sw.lock:
            total = len(sw.jobs)
            done = sw.count("done")
            errors = sw.count("error", "invalid")
            current = sw.skipped_existing + done
            remaining = sw.count("pending", "enqueued", "running", "interrupted", "lost")
            recent = sw.durations[-10:]
            avg = sum(recent) / len(recent) if recent else 0.0
            payload = {
                "node": sw.origin_node, "session_name": sw.session, "phase": "sweep", "sweep_id": sw.id,
                "state": sw.state, "errors": errors, "current_job": current, "total_jobs": total,
                "progress_pct": int((current + errors) / total * 100) if total else 100,
                "eta_str": _eta_str(avg * remaining) if recent and remaining else "—",
                "finish_time": time.strftime("%H:%M:%S", time.localtime(time.time() + avg * remaining)) if recent and remaining else "",
                "avg_duration": round(avg, 1), "last_duration": round(sw.durations[-1], 1) if sw.durations else 0,
            }
            if complete:
                elapsed = time.time() - (sw.eta_start or sw.created_at)
                payload.update({"complete": True, "eta_str": "Done", "progress_pct": 100,
                                "total_elapsed": f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s",
                                "total_generated": done})
        sw.bridge.send_sync("ultimate_grid.progress", payload)
    except Exception:
        pass


def _write_spec(sw):
    """benchmarks/<session>/workflow_sweeps/<sweep_id>.json: templates, bindings and the job list, so any
    executed prompt can be reconstructed. Never served by any route."""
    try:
        paths = wfs.session_paths(sw.session)
        os.makedirs(paths["sweeps"], exist_ok=True)
        with sw.lock:
            spec = {
                "sweep_id": sw.id, "session_name": sw.session, "state": sw.state, "reason": sw.reason,
                "created_at": sw.created_at, "settings": sw.settings, "axes": [a for c in sw.axes for a in c["axes"]], "warnings": sw.warnings,
                "templates": {h: {"prompt": t["prompt"], "output": t["output"],
                                  "run_other_outputs": t["run_other_outputs"],
                                  "media": t.get("media", "image"), "frames_from": t.get("frames_from")}
                              for h, t in sw.templates.items()},
                "jobs": [{"index": j["index"], "fingerprint": j["fingerprint"], "status": j["status"],
                          "prompt_id": j["prompt_id"], "error": j["error"], "duration": j["duration"],
                          "template": j["conf"].get("wf_template_hash"), "subs": j["conf"].get("wf_subs"),
                          "params": j["conf"].get("wf_params")} for j in sw.jobs],
            }
        path = os.path.join(paths["sweeps"], f"{sw.id}.json")
        with open(path + ".tmp", "w") as f:
            json.dump(spec, f, indent=1)
        os.replace(path + ".tmp", path)
    except Exception as e:
        print(f"[GridTester] ⚠️ Could not write the sweep spec file: {e}")


# --------------------------------------------------------------------------- request-body helpers (routes)

def configs_from_body(body):
    """(configs list, session_settings) from {state} (Builder state, through the single source of truth)
    or {configs_json}. Raises ValueError."""
    if not isinstance(body, dict):
        raise ValueError("The request body must be a JSON object.")
    if isinstance(body.get("state"), dict):
        try:
            from .config_builder_node import UltimateConfigBuilder
        except ImportError:
            from config_builder_node import UltimateConfigBuilder
        raw = UltimateConfigBuilder.state_to_configs_json(body["state"])
    elif body.get("configs_json") is not None:
        raw = body["configs_json"]
    else:
        raise ValueError("Send {state} (the Builder state) or {configs_json}.")
    parsed = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    if isinstance(parsed, dict):
        return parsed.get("configs") or [], parsed.get("_session_settings") or {}
    return parsed, {}


async def dry_run(body, bridge=None, class_info=wfs.comfy_class_info):
    """Concrete first prompts (with collector) + validation of the first job, nothing is queued."""
    configs, settings = configs_from_body(body)
    try:
        limit = max(1, min(20, int(body.get("limit", 3))))
    except (TypeError, ValueError):
        limit = 3
    sw = prepare_sweep(configs, settings, None, None, bridge, class_info, check_busy=False)
    jobs_out = []
    for job in sw.jobs[:limit]:
        prompt, partial = build_job_request(sw, job)
        jobs_out.append({"index": job["index"], "fingerprint": job["fingerprint"], "skip": job["status"] == "skipped",
                         "params": job["conf"].get("wf_params"), "partial_execution_targets": partial,
                         "prompt": prompt})
    validation = {"ok": True, "error": None, "node_errors": {}}
    first = next((j for j in sw.jobs if j["status"] == "pending"), None)
    if first is not None:
        prompt, partial = build_job_request(sw, first)
        try:
            ok, error, node_errors = await sw.bridge.validate(prompt, partial)
            validation = {"ok": ok, "error": summarize_errors(error, node_errors, prompt) if not ok else None,
                          "node_errors": node_errors if not ok else {}}
        except Exception as e:
            validation = {"ok": False, "error": f"validation could not run: {e}", "node_errors": {}}
    return {"total": len(sw.jobs), "to_run": sw.to_run, "skipped_existing": sw.skipped_existing,
            "axes": [a for c in sw.axes for a in c["axes"]], "warnings": sw.warnings, "validation": validation, "jobs": jobs_out}


async def api_start(body, bridge=None, class_info=wfs.comfy_class_info):
    """POST /config_tester/sweep/start. Returns (http_status, json_body). Runs on the event loop."""
    try:
        configs, settings = configs_from_body(body)
        sw = prepare_sweep(configs, settings, body.get("node_id"), body.get("client_id"), bridge, class_info)
        await validate_first(sw)
        launch(sw)
    except SweepAlreadyRunning as e:
        sw = e.sweep
        return 200, {**(get_status(sw.id) or {}), "attached": True, "sweep_id": sw.id, "total": len(sw.jobs),
                     "to_run": sw.to_run, "skipped_existing": sw.skipped_existing, "warnings": sw.warnings}
    except SweepBusy as e:
        return 409, {"error": str(e), "sweep_id": e.sweep_id, "session_name": e.session}
    except ValueError as e:
        return 400, {"error": str(e)}
    return 200, {"sweep_id": sw.id, "total": len(sw.jobs), "to_run": sw.to_run,
                 "skipped_existing": sw.skipped_existing, "warnings": sw.warnings}
