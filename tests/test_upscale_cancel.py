"""Dashboard upscale: cancel at every granularity, job bookkeeping, per-session guard, B1/B4/B5 fixes."""
import os
import re
import sys
import threading
import time

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from tests.upscale_runner_loader import load_runner  # noqa: E402

ur = load_runner()


def _job(session="s", total=3, status="running"):
    return {"id": "upscale_t", "session_name": session, "status": status, "completed": 0, "total": total,
            "outputs": 0, "current_image": "", "current_step": "", "error": None,
            "cancel": threading.Event(), "started_at": time.time(), "finished_at": None}


@pytest.fixture(autouse=True)
def _clean_jobs():
    ur._upscale_jobs.clear()
    yield
    ur._upscale_jobs.clear()


# ---- driver -----------------------------------------------------------------
def test_cancel_before_item_two_keeps_item_one():
    job = _job(total=3)
    done_calls, progress, finished = [], [], []

    def process(item, produced):
        produced.append({"id": item * 10})
        if item == 1:
            job["cancel"].set()           # user presses Cancel while item 1 finishes

    state = ur._drive_upscale_items(job, [1, 2, 3], process,
                                    lambda it, p: done_calls.append((it, list(p))),
                                    lambda d, t: progress.append((d, t)),
                                    finished.append)
    assert state == "cancelled" and job["status"] == "cancelled"
    assert job["completed"] == 1 and job["outputs"] == 1
    assert done_calls == [(1, [{"id": 10}])]
    assert progress == [(1, 3)]
    assert finished == ["cancelled"]
    assert job["finished_at"] is not None


def test_cancel_mid_item_keeps_partial_entries():
    job = _job(total=2)
    done_calls, finished = [], []

    def process(item, produced):
        produced.append({"id": 1})
        produced.append({"id": 2})
        raise ur.UpscaleCancelled()

    state = ur._drive_upscale_items(job, ["a", "b"], process,
                                    lambda it, p: done_calls.append(list(p)), None, finished.append)
    assert state == "cancelled" and job["status"] == "cancelled"
    assert job["completed"] == 0              # the interrupted image is not counted as done...
    assert job["outputs"] == 2                # ...but what it already produced is kept
    assert done_calls == [[{"id": 1}, {"id": 2}]]
    assert finished == ["cancelled"]


def test_complete_run_reports_complete():
    job = _job(total=2)
    finished = []
    state = ur._drive_upscale_items(job, [1, 2], lambda it, p: p.append({"id": it}), None, None, finished.append)
    assert state == "complete" and job["status"] == "complete"
    assert job["completed"] == 2 and job["outputs"] == 2
    assert finished == ["complete"]


def test_error_propagates_after_partial_entries_are_handed_over():
    job = _job(total=2)
    kept = []

    def process(item, produced):
        produced.append({"id": 1})
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        ur._drive_upscale_items(job, [1, 2], process, lambda it, p: kept.append(list(p)))
    assert kept == [[{"id": 1}]]
    assert job["outputs"] == 1


def test_cancel_set_before_start_does_no_work():
    job = _job()
    job["cancel"].set()
    calls = []
    assert ur._drive_upscale_items(job, [1, 2], lambda i, p: calls.append(i)) == "cancelled"
    assert calls == [] and job["completed"] == 0


# ---- cancel / status API ----------------------------------------------------
def test_cancel_upscale_job_states():
    job = _job()
    ur._upscale_jobs["upscale_t"] = job
    assert ur.cancel_upscale_job("upscale_t") == {"cancelled": True, "status": "cancelling"}
    assert job["cancel"].is_set()
    assert ur.cancel_upscale_job("upscale_t") == {"cancelled": True, "status": "cancelling"}   # idempotent
    job["status"] = "complete"
    assert ur.cancel_upscale_job("upscale_t") == {"cancelled": False, "status": "complete"}
    assert ur.cancel_upscale_job("nope") == {"cancelled": False, "status": "not_found"}


def test_status_fields_and_cancelling():
    job = _job()
    job.update(completed=1, outputs=4, current_image="img_1.webp", current_step="Pipeline 1 · step 2/3 · combo 1/4")
    ur._upscale_jobs["upscale_t"] = job
    st = ur.get_upscale_status("upscale_t")
    assert st == {"status": "running", "completed": 1, "total": 3, "outputs": 4, "current_image": "img_1.webp",
                  "current_step": "Pipeline 1 · step 2/3 · combo 1/4", "error": None}
    job["cancel"].set()
    assert ur.get_upscale_status("upscale_t")["status"] == "cancelling"
    job["status"] = "cancelled"
    assert ur.get_upscale_status("upscale_t")["status"] == "cancelled"
    assert ur.get_upscale_status("missing") == {"status": "not_found"}


def test_list_jobs_filters_by_session():
    a, b = _job("one"), _job("two")
    a["id"], b["id"] = "ja", "jb"
    ur._upscale_jobs.update({"ja": a, "jb": b})
    assert [j["job_id"] for j in ur.list_upscale_jobs("one")] == ["ja"]
    assert {j["job_id"] for j in ur.list_upscale_jobs()} == {"ja", "jb"}


# ---- guard + pruning ----------------------------------------------------------
def test_per_session_guard_blocks_a_second_job():
    ur._upscale_jobs["upscale_t"] = _job("sess")
    assert ur.start_upscale_job("sess", [1], {"pipelines": []}) == (None, ur.ALREADY_RUNNING_MSG)
    ur._upscale_jobs["upscale_t"]["cancel"].set()      # still winding down => still blocked
    assert ur.start_upscale_job("sess", [1], {"pipelines": []})[1] == ur.ALREADY_RUNNING_MSG
    ur._upscale_jobs["upscale_t"]["status"] = "cancelled"
    # a finished job no longer blocks; the missing manifest is the next error
    assert ur.start_upscale_job("sess", [1], {"pipelines": []})[1] == "Manifest not found"
    # a different session was never blocked
    ur._upscale_jobs["upscale_t"]["status"] = "running"
    assert ur.start_upscale_job("other", [1], {"pipelines": []})[1] == "Manifest not found"


def test_prune_jobs_drops_only_old_finished_jobs():
    now = 10_000.0
    old_done = _job(status="complete"); old_done["finished_at"] = now - 7200
    fresh_done = _job(status="error"); fresh_done["finished_at"] = now - 10
    running = _job(); running["started_at"] = now - 99999
    ur._upscale_jobs.update({"old": old_done, "fresh": fresh_done, "run": running})
    ur._prune_jobs(max_age=3600, now=now)
    assert set(ur._upscale_jobs) == {"fresh", "run"}


# ---- cancellable model -----------------------------------------------------------
class _FakePatcher:
    def __init__(self, options=None):
        self.model_options = dict(options or {})
        self.clones = 0

    def clone(self):
        c = _FakePatcher(self.model_options)
        c.clones = self.clones + 1
        return c

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn


def _args():
    return {"input": "x", "timestep": "t", "c": {"k": 1}}


def test_cancellable_model_passes_through_until_cancelled():
    ev = threading.Event()
    base = _FakePatcher()
    m = ur._cancellable_model(base, ev)
    assert m is not base and "model_function_wrapper" not in base.model_options   # original untouched
    wrapper = m.model_options["model_function_wrapper"]
    calls = []
    out = wrapper(lambda inp, t, **c: calls.append((inp, t, c)) or "eps", _args())
    assert out == "eps" and calls == [("x", "t", {"k": 1})]
    ev.set()
    with pytest.raises(ur.UpscaleCancelled):
        wrapper(lambda *a, **k: calls.append("late"), _args())
    assert calls == [("x", "t", {"k": 1})]


def test_cancellable_model_chains_existing_wrapper():
    ev = threading.Event()
    seen = []

    def prev(apply_model, args):
        seen.append(args["input"])
        return "prev-result"

    m = ur._cancellable_model(_FakePatcher({"model_function_wrapper": prev}), ev)
    wrapper = m.model_options["model_function_wrapper"]
    assert wrapper(lambda *a, **k: "direct", _args()) == "prev-result" and seen == ["x"]
    ev.set()
    with pytest.raises(ur.UpscaleCancelled):
        wrapper(lambda *a, **k: "direct", _args())
    assert seen == ["x"]


# ---- whole thread: events (B1 + B4) --------------------------------------------------
def test_thread_sends_new_items_once_and_final_progress(monkeypatch, tmp_path):
    srv = sys.modules["server"]
    events = []

    class _PS:
        def send_sync(self, ev, payload):
            events.append((ev, payload))

    monkeypatch.setattr(srv.PromptServer, "instance", _PS(), raising=False)
    saved = []
    monkeypatch.setattr(ur, "save_manifest", lambda path, data: saved.append(len(data["items"])))
    monkeypatch.setattr(ur, "load_checkpoint", lambda **kw: (_FakePatcher(), object(), object()))
    monkeypatch.setattr(ur, "_clean_vram", lambda: None)
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_full_path", lambda f, n: "x", raising=False)
    manifest = {"items": [{"id": 1}, {"id": 2}, {"id": 3}], "meta": {"model": "m.safetensors"}}

    def fake_one(ctx, item, produced):
        entry = {"id": 100 + item["id"], "upscaled": True}
        ctx["manifest_data"]["items"].insert(0, entry)          # the real code inserts at index 0
        produced.append(entry)
        if item["id"] == 2:
            ctx["job"]["cancel"].set()

    monkeypatch.setattr(ur, "_upscale_one_item", fake_one)
    job = _job(total=3)
    ur._run_upscale_thread(job, list(manifest["items"]), {"pipelines": [{"steps": [{}]}]}, manifest["meta"],
                           manifest, str(tmp_path / "m.json"), str(tmp_path), "s")
    assert job["status"] == "cancelled" and job["completed"] == 2 and job["outputs"] == 2
    updates = [p for ev, p in events if ev == "ultimate_grid.update_data"]
    assert [[i["id"] for i in u["new_items"]] for u in updates] == [[101], [102]]     # exactly the new ones, not the oldest
    final = [p for ev, p in events if ev == "ultimate_grid.progress"][-1]
    assert final["state"] == "cancelled" and final["phase"] == "upscaling"
    assert final["current_job"] == 2 and final["total_jobs"] == 3 and final["progress_pct"] == 66

    # a completed run ends with the "complete" progress event the ETA bar understands
    events.clear()
    manifest2 = {"items": [{"id": 1}], "meta": {"model": "m.safetensors"}}
    job2 = _job(total=1)
    ur._run_upscale_thread(job2, list(manifest2["items"]), {"pipelines": [{"steps": [{}]}]}, manifest2["meta"],
                           manifest2, str(tmp_path / "m2.json"), str(tmp_path), "s")
    assert job2["status"] == "complete"
    final2 = [p for ev, p in events if ev == "ultimate_grid.progress"][-1]
    assert final2["complete"] is True and final2["total_generated"] == 1 and final2["total_elapsed"].endswith("s")


def test_missing_checkpoint_sets_error_and_finished(monkeypatch, tmp_path):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_full_path", lambda f, n: None, raising=False)
    job = _job(total=1)
    ur._run_upscale_thread(job, [{"id": 1}], {"pipelines": [{"steps": [{}]}]}, {"model": "gone.safetensors"},
                           {"items": []}, str(tmp_path / "m.json"), str(tmp_path), "s")
    assert job["status"] == "error" and "gone.safetensors" in job["error"] and job["finished_at"]


# ---- source guards -----------------------------------------------------------------
def _src(name):
    with open(os.path.join(_ROOT, name), encoding="utf-8") as f:
        return f.read()


def test_runner_never_uses_the_global_interrupt():
    src = _src("upscale_runner.py")
    assert "interrupt_current_processing" not in src


def test_runner_has_no_tail_slicing_of_new_items():
    src = _src("upscale_runner.py")
    assert '"items"][-' not in src          # B1: new items live at index 0, never at the tail


def test_cancel_checks_exist_at_every_granularity():
    src = _src("upscale_runner.py")
    body = src[src.index("def _upscale_one_item"):]
    assert body.count("_check_cancel(job)") >= 6
    for anchor in ("for pipeline_idx, pipeline in enumerate(pipelines):", "for step_idx, ucfg in enumerate(expanded_steps):",
                   "for combo_idx, combo in enumerate(combos):"):
        i = body.index(anchor)
        assert "_check_cancel(job)" in body[i:i + 120], anchor


def test_florence2_step_takes_model_wrap():
    import inspect
    sys.path.insert(0, _ROOT)
    src = _src("florence2_hires.py")
    assert re.search(r"def run_florence2_step\([^)]*model_wrap=None", src, re.S)
    assert "model_wrap(model_handle)" in src
