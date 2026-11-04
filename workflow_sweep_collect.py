"""
USCGSweepCollect: the output node a Workflow Sweep splices onto each job's image socket.

It saves the image(s) into the USCG session folder (benchmarks/<session>/images) and appends manifest
items with the standard fields, so the Dashboard, favourites, filters and export keep working. It never
waits on anything and never writes outside the benchmarks folder (the session name is sanitised with
the same rule as every other route). `job_json` carries the job's conf only, never the workflow template.
"""

import json
import os
import time

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

try:
    from . import workflow_sweep as wfs
except ImportError:  # top-level import (tests)
    import workflow_sweep as wfs


def _append_items(*args, **kwargs):
    try:
        from .manifest_utils import append_items
    except ImportError:
        from manifest_utils import append_items
    return append_items(*args, **kwargs)


def _runner():
    try:
        from . import workflow_sweep_runner as runner
    except ImportError:
        import workflow_sweep_runner as runner
    return runner


def _tensor_to_array(images, b):
    """images[b] -> uint8 HxWxC numpy array (SaveImage's conversion), duck-typed so no torch import is needed."""
    arr = images[b]
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return np.clip(255.0 * np.asarray(arr), 0, 255).astype(np.uint8)


class USCGSweepCollect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "session_name": ("STRING", {"default": "workflow_sweep"}),
                "job_json": ("STRING", {"multiline": True, "default": "{}"}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "collect"
    CATEGORY = "sampling/testing"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # every job must run its collector, whatever the cache says about its ancestors
        return float("NaN")

    def collect(self, images, session_name="workflow_sweep", job_json="{}", prompt=None, unique_id=None):
        paths = wfs.session_paths(session_name)
        session = paths["session"]
        os.makedirs(paths["images"], exist_ok=True)

        try:
            job = json.loads(job_json or "{}")
            job = job if isinstance(job, dict) else {}
        except (TypeError, ValueError):
            job = {}
        conf = job.get("conf") if isinstance(job.get("conf"), dict) else {}
        sweep_id, job_index = job.get("sweep_id"), job.get("job_index")

        fmt = str(job.get("image_format") or "webp").lower().strip()
        if fmt not in ("webp", "png", "jpg", "jpeg"):
            fmt = "webp"
        ext = "jpg" if fmt in ("jpg", "jpeg") else fmt

        started = None
        if sweep_id is not None and job_index is not None:
            started = _runner().job_started_at(sweep_id, job_index)
        duration = round(time.time() - started, 2) if started else 0

        gen_index = job.get("gen_index")
        items, files = [], []
        for b in range(len(images)):
            arr = _tensor_to_array(images, b)
            img = Image.fromarray(arr)
            item = wfs.build_collect_item(conf, arr.shape[1], arr.shape[0], duration, b, gen_index)
            item["id"] = wfs.new_item_id()
            filename = f"img_{item['id']}.{ext}"
            filepath = os.path.join(paths["images"], filename)
            if fmt == "png":
                info = PngInfo()
                if prompt is not None:  # same text chunk SaveImage writes: the image can be dragged back into ComfyUI
                    info.add_text("prompt", json.dumps(prompt))
                img.save(filepath, format="PNG", pnginfo=info)
            elif ext == "jpg":
                img.save(filepath, format="JPEG", quality=95)
            else:
                img.save(filepath, quality=80)
            item["file"] = f"/view?filename={filename}&type=output&subfolder=benchmarks/{session}/images"
            if sweep_id is not None:
                item["wf_sweep_id"] = sweep_id
            if job_index is not None:
                item["wf_job_index"] = job_index
            items.append(item)
            files.append({"filename": filename, "subfolder": f"benchmarks/{session}/images", "type": "output"})

        manifest = _append_items(paths["manifest"], items, remove_ids=job.get("replace_item_ids") or (),
                                 session_name=session)
        try:  # live dashboard update (same event flush_batch_with_vae sends)
            from server import PromptServer
            PromptServer.instance.send_sync("ultimate_grid.update", {
                "node": job.get("origin_node"), "session_name": session, "new_items": items,
                "meta": (manifest or {}).get("meta", {})})
        except Exception:
            pass
        if sweep_id is not None:
            try:
                _runner().on_collected(sweep_id, job_index, items)
            except Exception:
                pass
        print(f"[GridTester] 🧪 Sweep collected {len(items)} image(s) into '{session}'")
        return {"ui": {"images": files}}


def _save_mp4(video, path, prompt=None):
    """Write a VIDEO to `path` as mp4/h264 with the core `VideoInput.save_to` (what SaveVideo.execute calls;
    SaveVideo itself needs executor-populated hidden inputs, so it cannot be invoked from a node body)."""
    try:
        from comfy_api.latest._input_impl import VideoContainer as _Container
    except ImportError:
        try:
            from comfy_api.latest._input.video_types import VideoContainer as _Container
        except ImportError:
            _Container = None
    fmt = "mp4"
    if _Container is not None:
        try:
            fmt = _Container("mp4")
        except Exception:
            fmt = "mp4"
    metadata = {"prompt": prompt} if prompt else None
    try:
        video.save_to(path, format=fmt, codec="auto", metadata=metadata)
    except TypeError:
        video.save_to(path)  # older Comfy: differently named kwargs


class USCGSweepCollectVideo:
    """Video twin of USCGSweepCollect: one VIDEO per job, saved as mp4 with a middle-frame poster."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "session_name": ("STRING", {"default": "workflow_sweep"}),
                "job_json": ("STRING", {"multiline": True, "default": "{}"}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "collect"
    CATEGORY = "sampling/testing"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def collect(self, video, session_name="workflow_sweep", job_json="{}", prompt=None, unique_id=None):
        paths = wfs.session_paths(session_name)
        session = paths["session"]
        os.makedirs(paths["images"], exist_ok=True)

        try:
            job = json.loads(job_json or "{}")
            job = job if isinstance(job, dict) else {}
        except (TypeError, ValueError):
            job = {}
        conf = job.get("conf") if isinstance(job.get("conf"), dict) else {}
        sweep_id, job_index = job.get("sweep_id"), job.get("job_index")

        started = None
        if sweep_id is not None and job_index is not None:
            started = _runner().job_started_at(sweep_id, job_index)
        duration = round(time.time() - started, 2) if started else 0

        item_id = wfs.new_item_id()
        stem = f"vid_{item_id}"
        subfolder = f"benchmarks/{session}/images"
        filename = f"{stem}.mp4"
        filepath = os.path.join(paths["images"], filename)
        _save_mp4(video, filepath, prompt)

        components = None
        try:  # frames + fps + poster; decoding a file-backed VIDEO into RAM is the known cost of this
            components = video.get_components()
        except Exception as e:
            print(f"[GridTester] ⚠️ Sweep video: could not read the frames back ({e}); poster skipped")
        frames_t = getattr(components, "images", None)
        try:
            frames = int(video.get_frame_count())
        except Exception:
            frames = int(len(frames_t)) if frames_t is not None else 0
        try:
            fps = float(video.get_frame_rate())
        except Exception:
            fps = float(getattr(components, "frame_rate", 0) or 0)
        try:
            width, height = (int(x) for x in video.get_dimensions())
        except Exception:
            width, height = 0, 0
            if frames_t is not None and len(frames_t):
                first = _tensor_to_array(frames_t, 0)
                height, width = first.shape[0], first.shape[1]

        preview_path = None
        if frames_t is not None and len(frames_t):
            try:  # middle frame, like the LTX previews
                arr = _tensor_to_array(frames_t, len(frames_t) // 2)
                preview_path = os.path.join(paths["images"], f"{stem}.preview.png")
                Image.fromarray(arr).convert("RGB").save(preview_path)
            except Exception as e:
                preview_path = None
                print(f"[GridTester] ⚠️ Sweep video: poster could not be written ({e})")

        item = wfs.build_collect_item(conf, width, height, duration, 0, job.get("gen_index"))
        item["id"] = item_id
        item["media_type"] = "video"
        item["file"] = f"/view?filename={filename}&type=output&subfolder={subfolder}"
        item["image_path"] = filepath
        item["video_path"] = filepath
        if preview_path:
            item["preview_path"] = preview_path
            item["poster"] = f"/view?filename={stem}.preview.png&type=output&subfolder={subfolder}"
        item["frames"] = frames
        item["frame_rate"] = int(fps) if fps and float(fps).is_integer() else (round(fps, 3) if fps else 0)
        item["duration_seconds"] = round(frames / fps, 2) if fps else 0
        if sweep_id is not None:
            item["wf_sweep_id"] = sweep_id
        if job_index is not None:
            item["wf_job_index"] = job_index
        items = [item]

        manifest = _append_items(paths["manifest"], items, remove_ids=job.get("replace_item_ids") or (),
                                 session_name=session)
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("ultimate_grid.update", {
                "node": job.get("origin_node"), "session_name": session, "new_items": items,
                "meta": (manifest or {}).get("meta", {})})
        except Exception:
            pass
        if sweep_id is not None:
            try:
                _runner().on_collected(sweep_id, job_index, items)
            except Exception:
                pass
        print(f"[GridTester] 🧪 Sweep collected 1 video into '{session}'")
        return {"ui": {"images": [{"filename": filename, "subfolder": subfolder, "type": "output"}],
                       "animated": (True,)}}


NODE_CLASS_MAPPINGS = {"USCGSweepCollect": USCGSweepCollect, "USCGSweepCollectVideo": USCGSweepCollectVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"USCGSweepCollect": "USCG Sweep Collect",
                              "USCGSweepCollectVideo": "USCG Sweep Collect Video"}
