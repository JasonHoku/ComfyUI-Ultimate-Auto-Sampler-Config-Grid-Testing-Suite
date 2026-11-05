"""Workflow Sweep: SaveWEBM / SaveAnimatedWEBP / SaveAnimatedPNG as frame sources (spliced CreateVideo)."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import workflow_sweep as ws  # noqa: E402
import workflow_sweep_runner as runner  # noqa: E402
from test_workflow_sweep_video import VideoBridge, load, lit  # noqa: E402,F401

FIXTURES = {"SaveWEBM": "video_webm_api.json",
            "SaveAnimatedWEBP": "video_animated_webp_api.json",
            "SaveAnimatedPNG": "video_animated_png_api.json"}
SINK_PARAMS = [pytest.param(c, f, id=c) for c, f in FIXTURES.items()]


def sink_entry(fixture, bindings=(), prompt=None):
    return {"model_type": "workflow", "name": "s", "workflow": prompt or load(fixture),
            "workflow_output": {"node_id": "1", "output_index": 0, "media": "video", "frames_from": "2"},
            "workflow_bindings": list(bindings)}


def createvideo(prompt):
    return [n for n in prompt.values() if n["class_type"] == "CreateVideo"][0]["inputs"]


@pytest.fixture
def vbridge(tmp_path, monkeypatch):
    import folder_paths
    import test_workflow_sweep_runner as rt
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(runner, "manifest_utils", rt._real_manifest_utils(), raising=False)
    runner.reset_registry()
    yield VideoBridge()
    runner.reset_registry()


@pytest.mark.parametrize("cls,fixture", SINK_PARAMS)
def test_sink_is_a_frames_source_and_the_default(cls, fixture):
    r = ws.parse_for_ui(load(fixture), class_info=None)
    assert r["video_sources"] == [{"node_id": "1", "output_index": 0, "type": "IMAGE", "media": "video",
                                   "kind": "frames", "frames_from": "2", "picked_from": "2"}]
    assert r["default_output"] == {"node_id": "1", "output_index": 0, "media": "video", "frames_from": "2"}


@pytest.mark.parametrize("cls,fixture", SINK_PARAMS)
def test_find_output_nodes_lists_the_sinks_without_a_registry(cls, fixture):
    outs = ws.find_output_nodes(load(fixture), class_info=None)
    assert [(o["node_id"], o["class_type"], o["source"]) for o in outs] == [("2", cls, ["1", 0])]


@pytest.mark.parametrize("cls,fixture", SINK_PARAMS)
def test_build_job_prompt_splices_createvideo_from_the_sink_fps(cls, fixture):
    p = load(fixture)
    p["2"]["inputs"]["fps"] = 12
    prompt, cid = ws.build_job_prompt(p, [], ["1", 0], {}, "video", "2")
    cv_ids = [k for k, n in prompt.items() if n["class_type"] == "CreateVideo"]
    assert len(cv_ids) == 1
    inputs = prompt[cv_ids[0]]["inputs"]
    assert inputs["fps"] == 12.0 and isinstance(inputs["fps"], float)
    assert inputs["images"] == ["1", 0] and "audio" not in inputs
    assert prompt[cid]["class_type"] == "USCGSweepCollectVideo" and prompt[cid]["inputs"]["video"] == [cv_ids[0], 0]
    assert prompt["2"] == p["2"]                      # the sink itself is untouched


def test_swept_sink_fps_changes_fingerprints_and_createvideo_fps():
    plan = ws.plan_workflow_entry(sink_entry("video_webm_api.json", [lit("b", "2", "fps", [12, 24])]))
    assert len({j["wf_fingerprint"] for j in plan["jobs"]}) == 2
    assert not any("not upstream" in w for w in plan["warnings"]), plan["warnings"]
    tpl = plan["template"]
    fps = []
    for j in plan["jobs"]:
        prompt, _ = ws.build_job_prompt(tpl["prompt"], j["wf_subs"], tpl["output"], {}, "video", tpl["frames_from"])
        fps.append(createvideo(prompt)["fps"])
    assert fps == [12.0, 24.0]


def test_fps_is_clamped_to_createvideo_range_with_a_warning():
    p = load("video_webm_api.json")
    p["2"]["inputs"]["fps"] = 0.5
    plan = ws.plan_workflow_entry(sink_entry("", prompt=p))
    assert any("0.5" in w and "outside CreateVideo's 1–120" in w and "uses 1." in w for w in plan["warnings"]), plan["warnings"]
    tpl = plan["template"]
    prompt, _ = ws.build_job_prompt(tpl["prompt"], [], tpl["output"], {}, "video", "2")
    assert createvideo(prompt)["fps"] == 1.0
    p["2"]["inputs"]["fps"] = 240
    prompt, _ = ws.build_job_prompt(p, [], ["1", 0], {}, "video", "2")
    assert createvideo(prompt)["fps"] == 120.0
    assert ws._sink_fps({"class_type": "SaveWEBM", "inputs": {"fps": 240}}) == (120.0, True)
    assert ws._sink_fps({"class_type": "SaveWEBM", "inputs": {"fps": 24}}) == (24.0, False)
    assert ws._sink_fps({"class_type": "SaveWEBM", "inputs": {}}) == (24.0, False)       # default fps


def test_swept_out_of_range_fps_gives_one_combined_warning():
    plan = ws.plan_workflow_entry(sink_entry("video_webm_api.json", [lit("b", "2", "fps", [0.5, 12, 240])]))
    clamp = [w for w in plan["warnings"] if "outside CreateVideo" in w]
    assert len(clamp) == 1 and "0.5" in clamp[0] and "240" in clamp[0]


def test_linked_fps_is_forwarded_unchanged():
    p = load("video_webm_api.json")
    p["3"] = {"class_type": "PrimitiveFloat", "inputs": {"value": 12}}
    p["2"]["inputs"]["fps"] = ["3", 0]
    assert ws._sink_fps(p["2"]) == (["3", 0], False)
    prompt, _ = ws.build_job_prompt(p, [], ["1", 0], {}, "video", "2")
    assert createvideo(prompt)["fps"] == ["3", 0]


def test_one_image_feeding_vhs_and_webm_gives_two_distinct_sources():
    p = load("video_vhs_api.json")
    p["3"] = {"class_type": "SaveWEBM", "inputs": {"images": ["1", 0], "filename_prefix": "x", "codec": "vp9", "fps": 12, "crf": 32}}
    srcs = ws.find_video_sources(p)
    assert [(s["frames_from"], s["kind"]) for s in srcs] == [("2", "frames"), ("3", "frames")]   # VHS first
    fps = []
    for fr in ("2", "3"):
        e = sink_entry("", prompt=p)
        e["workflow_output"]["frames_from"] = fr
        fps.append(ws.plan_workflow_entry(e)["jobs"][0]["wf_fingerprint"])
    assert fps[0] != fps[1]


def test_vhs_still_reads_frame_rate_and_audio():
    p = load("video_vhs_api.json")
    p["3"] = {"class_type": "LoadAudio", "inputs": {"audio": "a.wav"}}
    p["2"]["inputs"]["audio"] = ["3", 0]
    p["2"]["inputs"]["frame_rate"] = 16
    prompt, _ = ws.build_job_prompt(p, [], ["1", 0], {}, "video", "2")
    cv = createvideo(prompt)
    assert cv["fps"] == 16.0 and cv["audio"] == ["3", 0]


def test_runner_queues_a_webp_sweep(vbridge):
    sw = runner.prepare_sweep([sink_entry("video_animated_webp_api.json", [lit("b", "2", "fps", [6, 12])])],
                              {"session_name": "wf_sink"}, None, None, vbridge, class_info=None)
    runner.launch(sw, start_thread=False)
    assert runner.tick(sw)
    pid, prompt, partial, _ = vbridge.enqueued[0]
    classes = sorted(n["class_type"] for n in prompt.values())
    assert "USCGSweepCollectVideo" in classes and "CreateVideo" in classes
    status, body = runner.job_prompt("wf_sink", sw.id, 0)
    assert status == 200
    assert "USCGSweepCollect" not in json.dumps(body["prompt"]) and "CreateVideo" not in json.dumps(body["prompt"])
