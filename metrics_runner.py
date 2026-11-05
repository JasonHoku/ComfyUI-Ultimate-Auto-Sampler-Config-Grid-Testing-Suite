"""
Phase 2 of the image-metrics feature: "Compute metrics for this session".

A background job (same shape as upscale_runner: job dict + daemon thread +
status/cancel helpers) that measures the items of an EXISTING session that have
no `metrics` yet. Pure CPU (PIL -> numpy), no VRAM, never touches the GPU.

Manifest writes happen under manifest_utils._manifest_lock: the manifest is
RELOADED from disk, the new `metrics` are set by item id, and the file is
replaced atomically, so a stale full copy is never written back and a user's
concurrent favorites/notes survive. A generation run writing the same session
keeps these metrics too: every manifest writer copies an item's on-disk
`metrics` when its in-memory copy has none (manifest_utils, flush_batch_with_vae,
the remote VAE worker).
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


def start_metrics_job(session_name, overwrite=False, keys=None, fill_missing=False):
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
    if overwrite:
        targets = [it["id"] for it in items if "id" in it and _image_path_for(it, images_dir)]
    elif fill_missing:
        # Opt-in: items that have SOME metrics but lack a requested key (e.g. Sobel on older sessions)
        targets = [it["id"] for it in items if "id" in it
                   and any(k not in (it.get("metrics") or {}) for k in spec["builtins"])
                   and _image_path_for(it, images_dir)]
    else:
        targets = [it["id"] for it in items if "id" in it and not it.get("metrics")
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
                     args=(job, targets, spec["builtins"], overwrite, manifest_path, images_dir, session_name, fill_missing),
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


def _flush(manifest_path, results, overwrite, merge=False):
    """Merge `results` ({id: metrics}) into the manifest on disk under the lock; atomic replace.

    merge=True (fill_missing): add only keys the item on disk does not have (disk wins; adapter keys survive).
    """
    if not results:
        return
    with manifest_utils._manifest_lock:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for it in data.get("items", []):
            m = results.get(it.get("id"))
            if m and merge:
                cur = it.get("metrics") or {}
                it["metrics"] = {**cur, **{k: v for k, v in m.items() if k not in cur}}
            elif m and (overwrite or not it.get("metrics")):
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


def _run_metrics_thread(job, target_ids, keys, overwrite, manifest_path, images_dir, session_name, fill_missing=False):
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
                need = [k for k in keys if k not in (it.get("metrics") or {})] if fill_missing else keys
                m = image_metrics.compute_metrics_numpy(arr, need) if need else {}
                if m:
                    pending[tid] = m
                else:
                    job["failed"] += 1
            except Exception as e:
                job["failed"] += 1
                print(f"[GridTester] ⚠️ metrics skipped for item {tid}: {e}")
            job["completed"] += 1
            if len(pending) >= FLUSH_EVERY:
                _flush(manifest_path, pending, overwrite, fill_missing)
                pending = {}
                _send_progress(job, session_name)
        _flush(manifest_path, pending, overwrite, fill_missing)
        job["status"] = "cancelled" if job["cancel"].is_set() else "complete"
        print(f"[GridTester] 📏 metrics job {job['status']}: {job['completed']}/{job['total']} images, {job['failed']} skipped")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"[GridTester] ❌ metrics job failed: {e}")
    _send_progress(job, session_name)
