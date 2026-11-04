"""USCGSweepCollect node: saves images + manifest items inside the benchmarks folder, notifies the runner."""
import functools
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.basename(ROOT)
sys.path.insert(0, ROOT)

import workflow_sweep as wfs  # noqa: E402
import workflow_sweep_collect as collect_mod  # noqa: E402
import workflow_sweep_runner as runner  # noqa: E402


def _load_real(name, private):
    spec = importlib.util.spec_from_file_location(f"{PKG}.{private}", os.path.join(ROOT, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = PKG
    spec.loader.exec_module(mod)
    return mod


class FakeTensor:
    """Quacks like a torch IMAGE batch [B,H,W,C] float 0..1 (cpu().numpy())."""

    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        return FakeTensor(self.arr[i])

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


@pytest.fixture
def env(tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path), raising=False)
    mu = _load_real("manifest_utils", "manifest_utils_real2")
    monkeypatch.setattr(collect_mod, "_append_items", mu.append_items)
    torch = sys.modules.get("torch")
    patched = torch is not None and not hasattr(torch, "inference_mode")
    if patched:
        torch.inference_mode = lambda *a, **k: (lambda f: f)
    try:
        cim = _load_real("image_generation", "image_generation_real2").create_image_metadata
    finally:
        if patched:
            del torch.inference_mode
    real_build = wfs.build_collect_item
    monkeypatch.setattr(wfs, "build_collect_item", functools.partial(real_build, create_image_metadata=cim))
    runner.reset_registry()
    yield tmp_path
    runner.reset_registry()


def images(n=1, h=16, w=24):
    return FakeTensor(np.random.rand(n, h, w, 3).astype(np.float32))


CONF = {"model_type": "workflow", "model": "m", "sampler": "euler", "scheduler": "simple", "steps": 20,
        "cfg": 4.0, "seed": 3, "denoise": 1.0, "positive": "a teapot", "negative": "", "wf_fingerprint": "ab" * 10}


def job_json(**extra):
    d = {"conf": CONF, "sweep_id": "sw1", "job_index": 0, "gen_index": 7, "origin_node": "12", "image_format": "webp"}
    d.update(extra)
    return json.dumps(d)


def test_node_definition_matches_plan():
    cls = collect_mod.USCGSweepCollect
    spec = cls.INPUT_TYPES()
    assert spec["required"]["images"] == ("IMAGE",) and "session_name" in spec["required"]
    assert spec["required"]["job_json"][1]["multiline"] is True
    assert spec["hidden"] == {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}
    assert cls.RETURN_TYPES == () and cls.OUTPUT_NODE is True and cls.FUNCTION == "collect"
    assert cls.IS_CHANGED() != cls.IS_CHANGED()     # NaN: never cached
    assert collect_mod.NODE_CLASS_MAPPINGS["USCGSweepCollect"] is cls


def test_collect_saves_image_manifest_item_and_ui_output(env):
    out = collect_mod.USCGSweepCollect().collect(images(2), "mysession", job_json(), prompt={"1": {}}, unique_id="40")
    base = env / "benchmarks" / "mysession"
    manifest = json.loads((base / "manifest.json").read_text())
    assert len(manifest["items"]) == 2 and len(out["ui"]["images"]) == 2
    item = manifest["items"][0]
    assert item["model_type"] == "workflow" and item["width"] == 24 and item["height"] == 16   # from the tensor
    assert item["gen_index"] == 7 and item["seed"] == 3 and item["denoise"] == 1
    assert item["wf_sweep_id"] == "sw1" and item["wf_job_index"] == 0
    assert item["favorited"] is False and item["rejected"] is False and item["note"] == ""
    assert item["file"].startswith("/view?filename=img_") and item["file"].endswith(
        "&type=output&subfolder=benchmarks/mysession/images")
    assert {m["batch_idx"] for m in manifest["items"]} == {0, 1}
    assert len({m["id"] for m in manifest["items"]}) == 2
    for f in out["ui"]["images"]:
        assert f["subfolder"] == "benchmarks/mysession/images" and f["type"] == "output"
        assert Image.open(base / "images" / f["filename"]).size == (24, 16)


def test_collect_png_embeds_prompt_chunk_and_formats(env):
    prompt = {"3": {"class_type": "KSampler", "inputs": {}}}
    out = collect_mod.USCGSweepCollect().collect(images(), "fmt", job_json(image_format="png"), prompt=prompt)
    img = Image.open(env / "benchmarks" / "fmt" / "images" / out["ui"]["images"][0]["filename"])
    assert img.format == "PNG" and json.loads(img.text["prompt"]) == prompt
    out = collect_mod.USCGSweepCollect().collect(images(), "fmt", job_json(image_format="jpg"))
    assert out["ui"]["images"][0]["filename"].endswith(".jpg")
    out = collect_mod.USCGSweepCollect().collect(images(), "fmt", job_json(image_format="weird"))
    assert out["ui"]["images"][0]["filename"].endswith(".webp")


def test_collect_sanitises_session_name_and_stays_in_benchmarks(env):
    collect_mod.USCGSweepCollect().collect(images(), "../../escape/..\\x", job_json())
    assert (env / "benchmarks" / "escapex" / "manifest.json").exists()
    assert not (env.parent / "escape").exists()
    written = [p for p in env.rglob("*") if p.is_file()]
    assert all(str(p).startswith(str(env / "benchmarks")) for p in written)


def test_collect_works_without_a_sweep_manual_use(env):
    out = collect_mod.USCGSweepCollect().collect(images(), "manual", "{}")
    item = json.loads((env / "benchmarks" / "manual" / "manifest.json").read_text())["items"][0]
    assert item["model_type"] == "workflow" and "wf_sweep_id" not in item and len(out["ui"]["images"]) == 1
    collect_mod.USCGSweepCollect().collect(images(), "manual", "not json at all")   # garbage job_json is tolerated


def test_collect_appends_to_fresh_disk_manifest_preserving_favorites(env):
    d = env / "benchmarks" / "fav"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"session_name": "fav", "meta": {"k": 1},
                                                  "items": [{"id": 1, "favorited": True}]}))
    collect_mod.USCGSweepCollect().collect(images(), "fav", job_json())
    m = json.loads((d / "manifest.json").read_text())
    assert [i.get("id") for i in m["items"]][-1] == 1 and m["items"][-1]["favorited"] is True
    assert m["meta"] == {"k": 1} and len(m["items"]) == 2


def test_collect_replace_item_ids_removes_old_item_and_file(env):
    d = env / "benchmarks" / "rep"
    (d / "images").mkdir(parents=True)
    (d / "images" / "img_old.webp").write_bytes(b"x")
    (d / "manifest.json").write_text(json.dumps({"items": [
        {"id": 99, "file": "/view?filename=img_old.webp&type=output&subfolder=benchmarks/rep/images"}]}))
    collect_mod.USCGSweepCollect().collect(images(), "rep", job_json(replace_item_ids=[99]))
    m = json.loads((d / "manifest.json").read_text())
    assert [i["id"] for i in m["items"]] != [99] and all(i["id"] != 99 for i in m["items"]) and len(m["items"]) == 1
    assert not (d / "images" / "img_old.webp").exists()


def test_collect_notifies_runner_and_dashboard(env, monkeypatch):
    events = []
    import server
    ps = type("PS", (), {"send_sync": lambda self, e, d: events.append((e, d))})()
    monkeypatch.setattr(server.PromptServer, "instance", ps, raising=False)
    called = []
    monkeypatch.setattr(runner, "on_collected", lambda sid, idx, items: called.append((sid, idx, items)) or True)
    collect_mod.USCGSweepCollect().collect(images(), "notify", job_json())
    assert called[0][0] == "sw1" and called[0][1] == 0 and len(called[0][2]) == 1
    (event, data), = events
    assert event == "ultimate_grid.update" and data["node"] == "12" and data["session_name"] == "notify"
    assert len(data["new_items"]) == 1 and isinstance(data["meta"], dict)


def test_collect_duration_uses_runner_start_time(env, monkeypatch):
    monkeypatch.setattr(runner, "job_started_at", lambda sid, idx: time.time() - 10)
    collect_mod.USCGSweepCollect().collect(images(), "dur", job_json())
    item = json.loads((env / "benchmarks" / "dur" / "manifest.json").read_text())["items"][0]
    assert 9.5 < item["duration"] < 12


def test_package_registers_collector_node_and_sweep_routes():
    pkg = sys.modules.get(PKG)
    mappings = getattr(pkg, "NODE_CLASS_MAPPINGS", None)
    assert mappings is not None and "USCGSweepCollect" in mappings
    assert pkg.NODE_DISPLAY_NAME_MAPPINGS["USCGSweepCollect"] == "USCG Sweep Collect"
    for fn in ("sweep_parse", "sweep_dry_run", "sweep_start", "sweep_status", "sweep_pause", "sweep_resume", "sweep_cancel"):
        assert callable(getattr(pkg, fn)), fn


# --------------------------------------------------------------------------- USCGSweepCollectVideo

class FakeVideo:
    """Quacks like comfy_api VideoInput: get_components / get_dimensions / get_frame_count / save_to."""

    def __init__(self, frames=8, h=16, w=24, fps=8, kwargs_ok=True):
        from fractions import Fraction
        self.frames, self.h, self.w, self.fps = frames, h, w, Fraction(fps)
        self.kwargs_ok = kwargs_ok
        self.saved = []

    def get_components(self):
        class C:
            pass
        c = C()
        c.images = images(self.frames, self.h, self.w)
        c.frame_rate = self.fps
        return c

    def get_dimensions(self):
        return self.w, self.h

    def get_frame_count(self):
        return self.frames

    def get_frame_rate(self):
        return self.fps

    def save_to(self, path, **kwargs):
        if kwargs and not self.kwargs_ok:
            raise TypeError("old comfy: no kwargs")
        self.saved.append((path, kwargs))
        with open(path, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42fakevideo")


def test_video_node_definition():
    cls = collect_mod.USCGSweepCollectVideo
    spec = cls.INPUT_TYPES()
    assert spec["required"]["video"] == ("VIDEO",) and "session_name" in spec["required"]
    assert spec["hidden"] == {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}
    assert cls.RETURN_TYPES == () and cls.OUTPUT_NODE is True and cls.FUNCTION == "collect"
    assert cls.IS_CHANGED() != cls.IS_CHANGED()
    assert collect_mod.NODE_CLASS_MAPPINGS["USCGSweepCollectVideo"] is cls
    assert collect_mod.NODE_DISPLAY_NAME_MAPPINGS["USCGSweepCollectVideo"] == "USCG Sweep Collect Video"


def test_video_collect_saves_mp4_poster_and_ltx_shaped_item(env):
    video = FakeVideo(frames=8, fps=8)
    out = collect_mod.USCGSweepCollectVideo().collect(video, "vsession", job_json(), prompt={"1": {"class_type": "X"}})
    base = env / "benchmarks" / "vsession"
    manifest = json.loads((base / "manifest.json").read_text())
    assert len(manifest["items"]) == 1
    item = manifest["items"][0]
    assert item["media_type"] == "video" and item["model_type"] == "workflow"
    assert item["file"].startswith("/view?filename=vid_") and item["file"].endswith(
        ".mp4&type=output&subfolder=benchmarks/vsession/images")
    mp4 = base / "images" / f"vid_{item['id']}.mp4"
    assert mp4.read_bytes().startswith(b"\x00\x00\x00\x18ftyp")
    assert item["image_path"] == item["video_path"] == str(mp4)
    preview = base / "images" / f"vid_{item['id']}.preview.png"
    assert item["preview_path"] == str(preview) and Image.open(preview).size == (24, 16)
    assert item["poster"] == f"/view?filename=vid_{item['id']}.preview.png&type=output&subfolder=benchmarks/vsession/images"
    assert item["frames"] == 8 and item["frame_rate"] == 8 and item["duration_seconds"] == 1.0
    assert item["width"] == 24 and item["height"] == 16
    assert item["wf_sweep_id"] == "sw1" and item["wf_job_index"] == 0 and item["gen_index"] == 7
    assert item["seed"] == 3 and item["favorited"] is False and item["rejected"] is False
    # core save_to is called with mp4 + the workflow as metadata
    (path, kwargs), = video.saved
    assert path == str(mp4) and str(kwargs["format"]).lower().endswith("mp4") and kwargs["codec"] == "auto"
    assert kwargs["metadata"] == {"prompt": {"1": {"class_type": "X"}}}
    assert out["ui"]["images"] == [{"filename": mp4.name, "subfolder": "benchmarks/vsession/images", "type": "output"}]
    assert out["ui"]["animated"] == (True,)


def test_video_collect_fractional_fps_and_old_comfy_save_fallback(env):
    video = FakeVideo(frames=12, fps=12, kwargs_ok=False)
    collect_mod.USCGSweepCollectVideo().collect(video, "vold", job_json(), prompt=None)
    item = json.loads((env / "benchmarks" / "vold" / "manifest.json").read_text())["items"][0]
    assert item["frame_rate"] == 12 and item["duration_seconds"] == 1.0
    assert video.saved[0][1] == {}                       # fell back to save_to(path)
    video = FakeVideo(frames=10, fps=0)
    video.get_frame_rate = lambda: 7.5
    collect_mod.USCGSweepCollectVideo().collect(video, "vold", job_json(job_index=1))
    item = json.loads((env / "benchmarks" / "vold" / "manifest.json").read_text())["items"][0]
    assert item["frame_rate"] == 7.5 and item["duration_seconds"] == 1.33


def test_video_collect_sanitises_session_and_stays_in_benchmarks(env):
    collect_mod.USCGSweepCollectVideo().collect(FakeVideo(), "../../escape/..\\x", job_json())
    assert (env / "benchmarks" / "escapex" / "manifest.json").exists()
    written = [p for p in env.rglob("*") if p.is_file()]
    assert written and all(str(p).startswith(str(env / "benchmarks")) for p in written)


def test_video_collect_notifies_runner_and_replaces_old_item(env, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "on_collected", lambda sid, idx, items: calls.append((sid, idx, items)))
    collect_mod.USCGSweepCollectVideo().collect(FakeVideo(), "vrep", job_json())
    manifest = json.loads((env / "benchmarks" / "vrep" / "manifest.json").read_text())
    first = manifest["items"][0]
    assert calls and calls[0][0] == "sw1" and calls[0][2][0]["id"] == first["id"]
    old_mp4 = env / "benchmarks" / "vrep" / "images" / f"vid_{first['id']}.mp4"
    assert old_mp4.exists()
    collect_mod.USCGSweepCollectVideo().collect(FakeVideo(), "vrep", job_json(replace_item_ids=[first["id"]]))
    items = json.loads((env / "benchmarks" / "vrep" / "manifest.json").read_text())["items"]
    assert len(items) == 1 and items[0]["id"] != first["id"]
    assert not old_mp4.exists()                          # the previous mp4 is deleted with its item


def test_video_collect_without_a_sweep_and_with_garbage_job_json(env):
    collect_mod.USCGSweepCollectVideo().collect(FakeVideo(), "vman", "not json")
    item = json.loads((env / "benchmarks" / "vman" / "manifest.json").read_text())["items"][0]
    assert item["media_type"] == "video" and "wf_sweep_id" not in item
