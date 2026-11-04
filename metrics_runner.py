"""
Phase 2 of the image-metrics feature: "Compute metrics for this session".

A background job (same shape as upscale_runner: job dict + daemon thread +
status/cancel helpers) that measures the items of an EXISTING session that have
no `metrics` yet. Pure CPU (PIL -> numpy), no VRAM, never touches the GPU.

Manifest writes happen under manifest_utils._manifest_lock: the manifest is
RELOADED from disk, the new `metrics` are set by item id, and the file is
replaced atomically, so a stale full copy is never written back and a user's
concurrent favorites/notes survive. Do not run it on a session that a
generation run is currently writing: that run saves its own in-memory copy of
the manifest and would drop metrics written meanwhile.
"""
import json
import os
import re
import threading
import uuid

import numpy as np
from PIL import Image

from . import image_metrics
from . import manifest_utils

_jobs = {}
FLUSH_EVERY = 25


def _image_path_for(item, images_dir):
    """Resolve the item's image file INSIDE images_dir (None for videos/sentinels/anything else)."""
    if item.get("media_type") == "video" or item.get("florence2_no_detection"):
        return None
    name = item.get("filename")
    if not name:
        m = re.search(r'[?&]filename=([^&]+)', str(item.get("file") or ""))
        name = m.group(1) if m else None
    if not name or name != os.path.basename(name) or "%" in name:
        return None
    if not name.lower().endswith((".webp", ".png", ".jpg", ".jpeg")):
        return None
    root = os.path.realpath(images_dir)
    path = os.path.realpath(os.path.join(root, name))
    if os.path.dirname(path) != root or not os.path.isfile(path):
        return None
    return path


def start_metrics_job(session_name, overwrite=False, keys=None):
    """Start the job. Returns (job_id, error). session_name must already be sanitized by the caller."""
    import folder_paths
    benchmarks_dir = os.path.realpath(os.path.join(folder_paths.get_output_directory(), "benchmarks"))
    session_dir = os.path.realpath(os.path.join(benchmarks_dir, session_name))
    if os.path.dirname(session_dir) != benchmarks_dir:
        return None, "Invalid session"
    manifest_path = os.path.join(session_dir, "manifest.json")
    images_dir = os.path.join(session_dir, "images")
    if not os.path.exists(manifest_path):
        return None, "Manifest not found"

    spec = image_metrics.normalize_spec({"enabled": True, "builtins": list(keys or image_metrics.BUILTIN_METRICS)})
    if not spec:
        return None, "No valid metric names"

    with manifest_utils._manifest_lock:
        with open(manifest_path, "r", encoding="utf-8") as f:
            items = json.load(f).get("items", [])
    targets = [it["id"] for it in items if "id" in it and (overwrite or not it.get("metrics"))
               and _image_path_for(it, images_dir)]
    if not targets:
        return None, "Nothing to compute (every image already has metrics, or there are no images)"
    for jid, j in list(_jobs.items()):
        if j["status"] == "running" and j["session"] == session_name:
            return None, "A metrics job is already running for this session"

    job_id = f"metrics_{uuid.uuid4().hex[:12]}"
    job = {"id": job_id, "session": session_name, "status": "running", "completed": 0, "total": len(targets),
           "failed": 0, "error": None, "cancel": threading.Event()}
    _jobs[job_id] = job
    threading.Thread(target=_run_metrics_thread,
                     args=(job, targets, spec["builtins"], overwrite, manifest_path, images_dir, session_name),
                     daemon=True).start()
    return job_id, None


def get_metrics_status(job_id):
    job = _jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    return {k: job[k] for k in ("status", "completed", "total", "failed", "error")}


def cancel_metrics_job(job_id):
    job = _jobs.get(job_id)
    if job and job["status"] == "running":
        job["cancel"].set()
        return True
    return False


def _flush(manifest_path, results, overwrite):
    """Merge `results` ({id: metrics}) into the manifest on disk under the lock; atomic replace."""
    if not results:
        return
    with manifest_utils._manifest_lock:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for it in data.get("items", []):
            m = results.get(it.get("id"))
            if m and (overwrite or not it.get("metrics")):
                it["metrics"] = m
        tmp = manifest_path + ".metrics.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        os.replace(tmp, manifest_path)


def _send_progress(job, session_name):
    try:
        from server import PromptServer
        PromptServer.instance.send_sync("ultimate_grid.metrics_progress", {
            "session_name": session_name, "job_id": job["id"], "status": job["status"],
            "completed": job["completed"], "total": job["total"],
        })
    except Exception:
        pass


def _run_metrics_thread(job, target_ids, keys, overwrite, manifest_path, images_dir, session_name):
    pending = {}
    try:
        with manifest_utils._manifest_lock:
            with open(manifest_path, "r", encoding="utf-8") as f:
                by_id = {it.get("id"): it for it in json.load(f).get("items", [])}
        for tid in target_ids:
            if job["cancel"].is_set():
                break
            it = by_id.get(tid)
            path = _image_path_for(it, images_dir) if it else None
            try:
                if path is None:
                    raise ValueError("no image file")
                with Image.open(path) as im:
                    arr = np.asarray(im.convert("RGB"))
                m = image_metrics.compute_metrics_numpy(arr, keys)
                if m:
                    pending[tid] = m
                else:
                    job["failed"] += 1
            except Exception as e:
                job["failed"] += 1
                print(f"[GridTester] ⚠️ metrics skipped for item {tid}: {e}")
            job["completed"] += 1
            if len(pending) >= FLUSH_EVERY:
                _flush(manifest_path, pending, overwrite)
                pending = {}
                _send_progress(job, session_name)
        _flush(manifest_path, pending, overwrite)
        job["status"] = "cancelled" if job["cancel"].is_set() else "complete"
        print(f"[GridTester] 📏 metrics job {job['status']}: {job['completed']}/{job['total']} images, {job['failed']} skipped")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"[GridTester] ❌ metrics job failed: {e}")
    _send_progress(job, session_name)
