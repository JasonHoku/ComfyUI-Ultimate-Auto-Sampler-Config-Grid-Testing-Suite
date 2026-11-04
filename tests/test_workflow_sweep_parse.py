"""Workflow Sweep: parsing an API workflow, describing it, suggesting bindings."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import workflow_sweep as ws  # noqa: E402

FIX = os.path.join(ROOT, "tests", "fixtures")


def load(name):
    with open(os.path.join(FIX, name), "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def moss():
    return ws.parse_api_workflow(load("image_moss_image_t2i.json"))


@pytest.fixture
def sdxl():
    return ws.parse_api_workflow(load("sdxl_default_api.json"))


def test_parse_accepts_string_and_object():
    raw = load("image_moss_image_t2i.json")
    assert ws.parse_api_workflow(json.dumps(raw)).keys() == ws.parse_api_workflow(raw).keys()


def test_parse_rejects_ui_format():
    ui = {"nodes": [{"id": 1}], "links": []}
    with pytest.raises(ValueError, match=r"Export \(API\)"):
        ws.parse_api_workflow(ui)


def test_parse_rejects_garbage():
    for bad in ("", "{not json", "[]", '{"1": {"inputs": {}}}'):
        with pytest.raises(ValueError):
            ws.parse_api_workflow(bad)


def test_parse_unwraps_prompt_request_body():
    raw = load("sdxl_default_api.json")
    assert set(ws.parse_api_workflow({"prompt": raw, "client_id": "x"})) == set(raw)


def test_dotted_v3_input_names_kept_verbatim(moss):
    table = {n["id"]: n for n in ws.describe_nodes(moss)}
    names = [i["name"] for i in table["28"]["inputs"]]
    assert "sampling_mode.temperature" in names
    assert "sampling_mode.seed" in names


def test_links_flagged_in_node_table(moss):
    table = {n["id"]: n for n in ws.describe_nodes(moss)}
    by_name = {i["name"]: i for i in table["32"]["inputs"]}
    assert by_name["model"]["is_link"] and by_name["model"]["link"] == ["40", 0]
    assert not by_name["cfg"]["is_link"] and by_name["cfg"]["value"] == 2.5
    assert by_name["cfg"]["type"] == "FLOAT"


def test_describe_uses_class_info_for_combo_options(moss):
    def info(ct):
        if ct == "KSampler":
            return {"missing": False, "return_types": ["LATENT"], "output_node": False,
                    "inputs": {"sampler_name": {"type": "COMBO", "options": ["euler", "er_sde"], "required": True}}}
        return {"missing": ct == "NoSuchNode", "return_types": [], "output_node": False, "inputs": {}}
    table = {n["id"]: n for n in ws.describe_nodes(moss, info)}
    sampler = {i["name"]: i for i in table["32"]["inputs"]}["sampler_name"]
    assert sampler["options"] == ["euler", "er_sde"] and sampler["type"] == "COMBO"


def test_image_source_moss_is_vae_decode(moss):
    sources = ws.find_image_sources(moss)
    assert sources[0] == {"node_id": "38", "output_index": 0, "type": "IMAGE", "picked_from": "9"}
    assert ws.default_output(moss)["node_id"] == "38"


def test_image_source_uses_registry_image_outputs(sdxl):
    def info(ct):
        return {"missing": False, "return_types": ["IMAGE"] if ct == "VAEDecode" else [], "output_node": False, "inputs": {}}
    assert [s["node_id"] for s in ws.find_image_sources(sdxl, info)] == ["8"]


def test_output_nodes_listed(moss):
    outs = {o["node_id"]: o for o in ws.find_output_nodes(moss)}
    assert outs["9"]["source"] == ["38", 0]
    assert "34" in outs  # PreviewAny


def _sug(suggestions, kind, field=None):
    return [s for s in suggestions if s["kind"] == kind and (field is None or s["field"] == field)]


def test_moss_suggestions(moss):
    s = ws.suggest_bindings(moss, {"node_id": "38", "output_index": 0})
    # positive text hides behind PreviewAny/Switch links: found on the primitive string node 26
    assert _sug(s, "positive")[0]["targets"] == [{"node_id": "26", "input": "value"}]
    assert _sug(s, "negative")[0]["targets"] == [{"node_id": "37", "input": "text"}]
    assert _sug(s, "literal", "seed")[0]["targets"] == [{"node_id": "32", "input": "seed"}]
    assert _sug(s, "literal", "model")[0]["targets"] == [{"node_id": "29", "input": "unet_name"}]
    assert _sug(s, "literal", "cfg")[0]["targets"] == [{"node_id": "32", "input": "cfg"}]
    assert _sug(s, "literal", "vae")[0]["targets"] == [{"node_id": "33", "input": "vae_name"}]
    # the system prompt node (27) must never be offered as the positive prompt
    assert all(t["node_id"] != "27" for x in _sug(s, "positive") for t in x["targets"])


def test_sdxl_suggestions(sdxl):
    s = ws.suggest_bindings(sdxl, {"node_id": "8", "output_index": 0})
    assert _sug(s, "positive")[0]["targets"] == [{"node_id": "6", "input": "text"}]
    assert _sug(s, "negative")[0]["targets"] == [{"node_id": "7", "input": "text"}]
    assert _sug(s, "literal", "model")[0]["targets"][0]["input"] == "ckpt_name"
    assert _sug(s, "lora")[0]["slots"] == [{"node_id": "10"}]
    res = _sug(s, "resolution")[0]
    assert res["width"] == {"node_id": "5", "input": "width"} and res["height"]["input"] == "height"


def test_known_fields_from_moss_template(moss):
    f = ws.known_fields_from_template(moss, "38")
    assert f["sampler"] == "er_sde" and f["scheduler"] == "simple" and f["steps"] == 25
    assert f["cfg"] == 2.5 and f["vae"] == "flux2-vae.safetensors"
    assert f["model"] == "moss_image_sft_int8_convrot.safetensors"
    assert f["clip_type"] == "moss_image"
    assert f["text_encoders"] == ["qwen3.5_9b_w4a8_gptq.safetensors"]
    assert f["denoise"] == 1 and isinstance(f["denoise"], int)
    assert f["negative"].startswith("worst quality")
    assert f["lora"] == "None"


def test_known_fields_report_active_lora(sdxl):
    sdxl["10"]["inputs"].update({"lora_name": "x.safetensors", "strength_model": 0.8, "strength_clip": 0.5})
    assert ws.known_fields_from_template(sdxl, "8")["lora"] == "x.safetensors:0.8:0.5"
