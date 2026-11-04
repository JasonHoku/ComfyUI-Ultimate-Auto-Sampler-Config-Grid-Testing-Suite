"""Workflow Sweep video outputs: parse (sources, default), plan (media, fingerprints), job prompt (collector, CreateVideo splice)."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import workflow_sweep as ws  # noqa: E402
import workflow_sweep_runner as runner  # noqa: E402
import test_workflow_sweep_runner as rt  # noqa: E402  (FakeBridge)
from config_builder_node import UltimateConfigBuilder  # noqa: E402


def load(name):
    with open(os.path.join(ROOT, "tests", "fixtures", name), "r", encoding="utf-8") as f:
        return json.load(f)


def vhs_entry(bindings=(), prompt=None, **out):
    output = {"node_id": "1", "output_index": 0, "media": "video", "frames_from": "2"}
    output.update(out)
    return {"model_type": "workflow", "name": "v", "workflow": prompt or load("video_vhs_api.json"),
            "workflow_output": output, "workflow_bindings": list(bindings)}


def core_entry(bindings=(), **out):
    output = {"node_id": "2", "output_index": 0, "media": "video"}
    output.update(out)
    return {"model_type": "workflow", "name": "v", "workflow": load("video_core_api.json"),
            "workflow_output": output, "workflow_bindings": list(bindings)}


def lit(bid, nid, inp, values):
    return {"id": bid, "kind": "literal", "targets": [{"node_id": nid, "input": inp}], "values": list(values)}


# ------------------------------------------------------------------------- parse

def test_core_fixture_video_sources_and_default():
    r = ws.parse_for_ui(load("video_core_api.json"), class_info=None)
    v = r["video_sources"]
    assert v[0] == {"node_id": "2", "output_index": 0, "type": "VIDEO", "media": "video", "kind": "video", "picked_from": "3"}
    assert r["image_sources"] == []
    assert r["default_output"] == {"node_id": "2", "output_index": 0, "media": "video"}
    assert not any("No image" in w for w in r["warnings"])


def test_vhs_fixture_is_a_frames_source():
    r = ws.parse_for_ui(load("video_vhs_api.json"), class_info=None)
    assert r["video_sources"] == [{"node_id": "1", "output_index": 0, "type": "IMAGE", "media": "video",
                                   "kind": "frames", "frames_from": "2", "picked_from": "2"}]
    assert r["default_output"] == {"node_id": "1", "output_index": 0, "media": "video", "frames_from": "2"}
    assert isinstance(r["suggestions"], list)


def test_savevideo_outranks_vhs_and_other_video_returners():
    p = load("video_core_api.json")
    p["4"] = {"class_type": "VHS_VideoCombine", "inputs": {"images": ["1", 0], "frame_rate": 8}}
    order = [(s["kind"], s["node_id"]) for s in ws.find_video_sources(p)]
    assert order == [("video", "2"), ("frames", "1")]


def test_image_workflows_keep_image_default():
    for name in ("sdxl_default_api.json", "image_moss_image_t2i.json"):
        r = ws.parse_for_ui(load(name), class_info=None)
        assert r["video_sources"] == []
        assert r["default_output"] == r["image_sources"][0]
        assert "media" not in r["default_output"]
    # a SaveImage feeder wins even when the graph also has a video output
    p = load("sdxl_default_api.json")
    p["20"] = {"class_type": "CreateVideo", "inputs": {"images": ["8", 0], "fps": 8}}
    p["21"] = {"class_type": "SaveVideo", "inputs": {"video": ["20", 0]}}
    r = ws.parse_for_ui(p, class_info=None)
    assert r["default_output"]["node_id"] == "8" and "media" not in r["default_output"]
    assert r["video_sources"][0]["node_id"] == "20"


def test_vhs_latent_and_meta_batch_are_refused_with_a_warning():
    for key, link in (("vae", ["4", 0]), ("meta_batch", ["5", 0])):
        p = load("video_vhs_api.json")
        p["2"]["inputs"][key] = link
        r = ws.parse_for_ui(p, class_info=None)
        assert r["video_sources"] == []
        assert any("skipped as a video source" in w and key in w for w in r["warnings"]), r["warnings"]
        with pytest.raises(ValueError, match="cannot be used as a video source"):
            ws.plan_workflow_entry(vhs_entry(prompt=p))


def test_registry_video_return_types_are_listed():
    p = {"1": {"class_type": "LoadVideo", "inputs": {"file": "x.mp4"}}}

    def info(ct):
        return {"missing": False, "return_types": ["VIDEO", "AUDIO"], "output_node": False, "inputs": {}}

    r = ws.parse_for_ui(p, class_info=info)
    assert r["video_sources"] == [{"node_id": "1", "output_index": 0, "type": "VIDEO", "media": "video", "kind": "video"}]
    assert r["default_output"]["media"] == "video"


def test_no_output_at_all_warns():
    r = ws.parse_for_ui({"1": {"class_type": "Foo", "inputs": {}}}, class_info=None)
    assert r["default_output"] is None and any("No image or video output" in w for w in r["warnings"])


def test_normalize_output_default_and_validation():
    p = ws.parse_api_workflow(load("video_core_api.json"))
    assert ws.normalize_output({}, p) == ("2", 0, "video", None)
    assert ws.normalize_output({"workflow_output": {"node_id": "2", "output_index": 0}}, p) == ("2", 0, "image", None)
    with pytest.raises(ValueError, match="Unknown output media"):
        ws.normalize_output({"workflow_output": {"node_id": "2", "media": "gif"}}, p)
    with pytest.raises(ValueError, match="frames node 9"):
        ws.normalize_output({"workflow_output": {"node_id": "1", "media": "video", "frames_from": "9"}}, p)
    # frames_from is meaningless for images and is dropped
    assert ws.normalize_output({"workflow_output": {"node_id": "1", "frames_from": "2"}}, p) == ("1", 0, "image", None)


# ------------------------------------------------------------------------- plan / fingerprints

def test_plan_marks_jobs_as_video():
    plan = ws.plan_workflow_entry(core_entry([lit("b", "2", "fps", [8, 12])]))
    assert len(plan["jobs"]) == 2
    for j in plan["jobs"]:
        assert j["media_type"] == "video" and j["wf_output_media"] == "video" and "wf_frames_from" not in j
    assert plan["template"]["media"] == "video" and plan["template"]["frames_from"] is None
    plan = ws.plan_workflow_entry(vhs_entry())
    j = plan["jobs"][0]
    assert j["media_type"] == "video" and j["wf_frames_from"] == "2" and j["wf_output"] == "1:0"
    assert plan["template"]["frames_from"] == "2"


def test_image_plan_has_no_video_keys():
    e = {"model_type": "workflow", "name": "c", "workflow": load("sdxl_default_api.json"), "workflow_bindings": []}
    j = ws.plan_workflow_entry(e)["jobs"][0]
    assert j["media_type"] == "image" and "wf_output_media" not in j and "wf_frames_from" not in j


def test_golden_image_hashes_unchanged():
    """Pinned on main 6b60163: image templates/fingerprints must stay byte-identical (resume of old sessions)."""
    wf = load("sdxl_default_api.json")
    plan = ws.plan_workflow_entry({"model_type": "workflow", "name": "c", "workflow": wf, "workflow_bindings": []})
    assert plan["template"]["output"] == ["8", 0]
    assert plan["template"]["hash"] == "7a10272fe5cd263c"
    assert plan["jobs"][0]["wf_fingerprint"] == "e829b0b70d514835454e"
    plan = ws.plan_workflow_entry({"model_type": "workflow", "name": "c", "workflow": wf,
                                   "workflow_bindings": [lit("b1", "3", "cfg", [2, 3])]})
    assert [j["wf_fingerprint"] for j in plan["jobs"]] == ["121321289f16ba908470", "ac9174b0bdd5d8ac1900"]


def test_video_and_image_fingerprints_differ_for_the_same_socket():
    p = load("video_core_api.json")
    img = ws.fingerprint(p, ("2", 0))
    vid = ws.fingerprint(p, ("2", 0), "video")
    assert img != vid and len(vid) == 20
    assert ws.fingerprint(p, ("2", 0), "video", "2") != vid          # frames_from is part of it
    assert ws.template_hash(p, ("2", 0), False, "video") != ws.template_hash(p, ("2", 0), False)
    assert ws.fingerprint(p, ("2", 0), "image", "2") == img            # ignored for images


def test_fps_axis_changes_the_fingerprint_and_vhs_frame_rate_is_upstream():
    plan = ws.plan_workflow_entry(core_entry([lit("b", "2", "fps", [8, 12])]))
    assert len({j["wf_fingerprint"] for j in plan["jobs"]}) == 2 and not plan["warnings"]
    plan = ws.plan_workflow_entry(vhs_entry([lit("b", "2", "frame_rate", [8, 16])]))
    assert len({j["wf_fingerprint"] for j in plan["jobs"]}) == 2
    assert not any("not upstream" in w for w in plan["warnings"]), plan["warnings"]
    plan = ws.plan_workflow_entry(vhs_entry([lit("b", "1", "color", [0, 255])]))
    assert not any("not upstream" in w for w in plan["warnings"])


def test_image_and_video_entries_plan_together_but_not_with_standard_configs():
    img = {"model_type": "workflow", "name": "i", "workflow": load("sdxl_default_api.json"), "workflow_bindings": []}
    plan = ws.plan_workflow_configs([img, core_entry()])
    assert {t["media"] for t in plan["templates"].values()} == {"image", "video"}
    with pytest.raises(ValueError, match="cannot be mixed"):
        ws.plan_workflow_configs([core_entry(), {"model_type": "sdxl"}])


# ------------------------------------------------------------------------- build_job_prompt

def test_core_video_collector_is_fed_directly():
    t = ws.plan_workflow_entry(core_entry())["template"]
    p, cid = ws.build_job_prompt(t["prompt"], [], t["output"], {"session_name": "s"}, t["media"], t["frames_from"])
    assert p[cid]["class_type"] == "USCGSweepCollectVideo" and p[cid]["inputs"]["video"] == ["2", 0]
    assert p[cid]["inputs"]["session_name"] == "s" and "images" not in p[cid]["inputs"]
    assert set(p) - set(t["prompt"]) == {cid}


def test_vhs_frames_get_a_createvideo_splice():
    plan = ws.plan_workflow_entry(vhs_entry([lit("b", "2", "frame_rate", [8, 16])]))
    t = plan["template"]
    seen = set()
    for job in plan["jobs"]:
        p, cid = ws.build_job_prompt(t["prompt"], job["wf_subs"], t["output"], {}, t["media"], t["frames_from"])
        added = set(p) - set(t["prompt"])
        assert len(added) == 2 and cid in added
        cv = (added - {cid}).pop()
        assert p[cv]["class_type"] == "CreateVideo" and int(cv) < int(cid)
        assert p[cv]["inputs"]["images"] == ["1", 0] and "audio" not in p[cv]["inputs"]
        assert p[cid]["class_type"] == "USCGSweepCollectVideo" and p[cid]["inputs"]["video"] == [cv, 0]
        assert isinstance(p[cv]["inputs"]["fps"], float)
        seen.add(p[cv]["inputs"]["fps"])
        # the VHS node itself is untouched
        assert p["2"]["inputs"]["frame_rate"] == float(p[cv]["inputs"]["fps"])
    assert seen == {8.0, 16.0}


def test_vhs_splice_forwards_link_fps_and_audio():
    p = load("video_vhs_api.json")
    p["7"] = {"class_type": "PrimitiveInt", "inputs": {"value": 12}}
    p["8"] = {"class_type": "LoadAudio", "inputs": {"audio": "a.wav"}}
    p["2"]["inputs"]["frame_rate"] = ["7", 0]
    p["2"]["inputs"]["audio"] = ["8", 0]
    t = ws.plan_workflow_entry(vhs_entry(prompt=p))["template"]
    out, cid = ws.build_job_prompt(t["prompt"], [], t["output"], {}, "video", "2")
    cv = out[out[cid]["inputs"]["video"][0]]
    assert cv["inputs"]["fps"] == ["7", 0] and cv["inputs"]["audio"] == ["8", 0]


def test_collector_ids_are_unique_and_numeric():
    t = ws.plan_workflow_entry(vhs_entry())["template"]
    p, cid = ws.build_job_prompt(t["prompt"], [], t["output"], {}, "video", "2")
    assert cid.isdigit() and len(p) == 4


def test_image_path_of_build_job_prompt_is_unchanged():
    t = load("sdxl_default_api.json")
    p, cid = ws.build_job_prompt(t, [], ("8", 0), {})
    assert p[cid]["class_type"] == "USCGSweepCollect" and p[cid]["inputs"] == {"images": ["8", 0]}


# ------------------------------------------------------------------------- runner integration

class VideoBridge(rt.FakeBridge):
    async def enqueue(self, prompt, partial, client_id):
        ok, error, node_errors = await self.validate(prompt, partial)
        if not ok:
            return None, error, node_errors
        self.n += 1
        pid = f"pid-{self.n}"
        collector = [n for n in prompt.values() if n["class_type"].startswith("USCGSweepCollect")][0]
        self.meta[pid] = json.loads(collector["inputs"]["job_json"])
        self.queue.append(pid)
        self.enqueued.append((pid, prompt, partial, client_id))
        return pid, None, {}


@pytest.fixture
def vbridge(tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(runner, "manifest_utils", rt._real_manifest_utils(), raising=False)
    runner.reset_registry()
    yield VideoBridge()
    runner.reset_registry()


def test_runner_queues_video_collector_and_job_prompt_route_excludes_it(vbridge):
    sw = runner.prepare_sweep([vhs_entry([lit("b", "2", "frame_rate", [8, 16])])], {"session_name": "wf_vid"},
                              None, None, vbridge, class_info=None)
    runner.launch(sw, start_thread=False)
    assert runner.tick(sw)
    pid, prompt, partial, _ = vbridge.enqueued[0]
    classes = sorted(n["class_type"] for n in prompt.values())
    assert "USCGSweepCollectVideo" in classes and "CreateVideo" in classes
    cid = [k for k, n in prompt.items() if n["class_type"] == "USCGSweepCollectVideo"][0]
    assert partial == [cid]
    status, body = runner.job_prompt("wf_vid", sw.id, 0)
    assert status == 200
    assert "USCGSweepCollect" not in json.dumps(body["prompt"]) and "CreateVideo" not in json.dumps(body["prompt"])
    tpl = sw.templates[sw.jobs[0]["conf"]["wf_template_hash"]]
    assert body["prompt"] == ws.apply_subs(tpl["prompt"], sw.jobs[0]["conf"]["wf_subs"])
    with open(os.path.join(ws.session_paths("wf_vid")["sweeps"], sw.id + ".json")) as f:
        spec = json.load(f)
    assert {t["media"] for t in spec["templates"].values()} == {"video"}
    assert {t["frames_from"] for t in spec["templates"].values()} == {"2"}


# ------------------------------------------------------------------------- Builder emission

def _array(output):
    return {"name": "WF", "kind": "workflow",
            "workflow": {"source_name": "t.json", "api_prompt": {"3": {"class_type": "KSampler", "inputs": {}}},
                         "output": output, "run_other_outputs": False, "bindings": []},
            "positive_prompt_groups": [], "negative_prompt": "", "use_custom_prompts": False}


def _emit(output):
    state = {"session_name": "s", "include_none": False, "global_positive_groups": [], "global_negative": "",
             "config_arrays": [_array(output)]}
    return json.loads(UltimateConfigBuilder.state_to_configs_json(state))["configs"][0]


def test_builder_keeps_media_keys_only_for_video():
    assert _emit({"node_id": "8", "output_index": 0})["workflow_output"] == {"node_id": "8", "output_index": 0}
    assert _emit({"node_id": "8", "output_index": 0, "picked_from": "9"})["workflow_output"] == {"node_id": "8", "output_index": 0}
    assert _emit({"node_id": "8", "output_index": 0, "media": "image", "frames_from": "2"})["workflow_output"] == {"node_id": "8", "output_index": 0}
    assert _emit({"node_id": "1", "output_index": 0, "media": "video", "frames_from": 2})["workflow_output"] == \
        {"node_id": "1", "output_index": 0, "media": "video", "frames_from": "2"}
    assert _emit({"node_id": "2", "media": "video"})["workflow_output"] == {"node_id": "2", "output_index": 0, "media": "video"}
    assert _emit({})["workflow_output"] == {}
