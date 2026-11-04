"""Workflow Sweep: binding expansion (product, pairing, seeds, LoRA slots, resolution, ordering)."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import workflow_sweep as ws  # noqa: E402
from config_utils import expand_configs  # noqa: E402


def load(name):
    with open(os.path.join(ROOT, "tests", "fixtures", name), "r", encoding="utf-8") as f:
        return json.load(f)


def moss_entry(bindings, **extra):
    e = {"model_type": "workflow", "name": "Config 1", "workflow": load("image_moss_image_t2i.json"),
         "workflow_source": "image_moss_image_t2i.json", "workflow_output": {"node_id": "38", "output_index": 0},
         "workflow_run_other_outputs": False, "workflow_bindings": bindings}
    e.update(extra)
    return e


def sdxl_entry(bindings, **extra):
    e = {"model_type": "workflow", "name": "SDXL", "workflow": load("sdxl_default_api.json"),
         "workflow_output": {"node_id": "8", "output_index": 0}, "workflow_bindings": bindings}
    e.update(extra)
    return e


def lit(bid, nid, inp, values, **kw):
    return {"id": bid, "kind": "literal", "targets": [{"node_id": nid, "input": inp}], "values": values, **kw}


def prompt_binding(kind, nid, inp):
    return {"id": kind, "kind": kind, "targets": [{"node_id": nid, "input": inp}]}


def test_product_counts_and_substitution():
    entry = moss_entry([lit("b1", "32", "cfg", [2.5, 4, 6]), lit("b2", "32", "seed", [1, 2])])
    jobs = ws.expand_workflow_entry(entry)
    assert len(jobs) == 6
    assert {(j["cfg"], j["seed"]) for j in jobs} == {(c, s) for c in (2.5, 4, 6) for s in (1, 2)}
    # cfg 4 is coerced to the template's float type, seed stays int
    assert all(isinstance(j["cfg"], float) and isinstance(j["seed"], int) for j in jobs)
    assert len({j["wf_fingerprint"] for j in jobs}) == 6


def test_known_fields_filled_from_template_when_unbound():
    j = ws.expand_workflow_entry(moss_entry([lit("b1", "32", "cfg", [3])]))[0]
    assert j["sampler"] == "er_sde" and j["scheduler"] == "simple" and j["steps"] == 25
    assert j["vae"] == "flux2-vae.safetensors" and j["model_type"] == "workflow" and j["media_type"] == "image"
    assert j["wf_output"] == "38:0" and j["lora"] == "None" and j["lora_expanded"] == "None"
    assert j["wf_params"] == {"KSampler.cfg": 3.0}


def test_empty_axis_is_an_error():
    with pytest.raises(ValueError, match="no values"):
        ws.expand_workflow_entry(moss_entry([lit("b1", "32", "cfg", [])]))


def test_no_bindings_yields_single_job():
    assert len(ws.expand_workflow_entry(moss_entry([]))) == 1


def test_coercion_rules():
    assert ws.coerce_like(7.0, 7) == 7 and isinstance(ws.coerce_like(7.0, 7), int)
    assert ws.coerce_like("7", 1) == 7
    assert ws.coerce_like(4, 2.5) == 4.0 and isinstance(ws.coerce_like(4, 2.5), float)
    assert ws.coerce_like("false", True) is False and ws.coerce_like("true", False) is True
    assert ws.coerce_like(5, "euler") == "5"
    assert ws.coerce_like("x", ["1", 0]) == "x"  # template was a link: untouched


def test_unbound_value_type_stable_in_fingerprint():
    a = ws.expand_workflow_entry(moss_entry([lit("b1", "32", "steps", [20])]))[0]["wf_fingerprint"]
    b = ws.expand_workflow_entry(moss_entry([lit("b1", "32", "steps", [20.0])]))[0]["wf_fingerprint"]
    assert a == b


def test_prompt_pairing_rule():
    both = [prompt_binding("positive", "26", "value"), prompt_binding("negative", "37", "text")]
    # 2 positive x 1 negative -> product
    assert len(ws.expand_workflow_entry(moss_entry(both, positive=["a", "b"], negative="x"))) == 2
    # equal lengths > 1 -> 1-to-1 pairing
    jobs = ws.expand_workflow_entry(moss_entry(both, positive=["a", "b"], negative=["x", "y"]))
    assert sorted((j["positive"], j["negative"]) for j in jobs) == [("a", "x"), ("b", "y")]
    # unequal -> product
    assert len(ws.expand_workflow_entry(moss_entry(both, positive=["a", "b", "c"], negative=["x", "y"]))) == 6


def test_prompt_groups_cartesian_like_standard_configs():
    entry = moss_entry([prompt_binding("positive", "26", "value")], positive=[["a", "b"], ["1", "2"]])
    assert sorted(j["positive"] for j in ws.expand_workflow_entry(entry)) == ["a, 1", "a, 2", "b, 1", "b, 2"]


def test_prompt_binding_without_prompts_errors_with_fallback_option():
    entry = moss_entry([prompt_binding("positive", "26", "value")])
    with pytest.raises(ValueError, match="no positive prompts"):
        ws.expand_workflow_entry(entry)
    jobs = ws.expand_workflow_entry(entry, None, ["fallback prompt"], [""])
    assert jobs[0]["positive"] == "fallback prompt"


def test_unbound_prompts_come_from_template():
    j = ws.expand_workflow_entry(moss_entry([lit("b1", "32", "cfg", [3])]))[0]
    assert j["positive"].startswith("A bimbo girl")  # text of template node 26
    assert j["negative"] == "worst quality, bad anatomy, watermark, jpeg artifact"


def test_extra_seeds_appended_to_seed_axis():
    jobs = ws.expand_workflow_entry(moss_entry([lit("b1", "32", "seed", [1, 2])]), extra_seeds=[777, 1])
    assert sorted(j["seed"] for j in jobs) == [1, 2, 777]  # duplicate seed 1 dropped


def test_extra_seeds_without_seed_axis_warn_and_name_inputs():
    plan = ws.plan_workflow_entry(moss_entry([lit("b1", "32", "cfg", [3, 4])]), extra_seeds=[5])
    assert len(plan["jobs"]) == 2
    assert any("KSampler.seed" in w for w in plan["warnings"])


def test_lora_slot_mapping_and_none():
    b = {"id": "l", "kind": "lora", "slots": [{"node_id": "10"}], "values": ["x.safetensors:0.8", "None"]}
    plan = ws.plan_workflow_entry(sdxl_entry([b]))
    by_lora = {j["lora"]: j for j in plan["jobs"]}
    assert set(by_lora) == {"x.safetensors:0.8", "None"}
    subs = {(n, i): v for n, i, v in by_lora["x.safetensors:0.8"]["wf_subs"]}
    assert subs[("10", "lora_name")] == "x.safetensors"
    assert subs[("10", "strength_model")] == 0.8 and subs[("10", "strength_clip")] == 0.8
    none_subs = {(n, i): v for n, i, v in by_lora["None"]["wf_subs"]}
    assert none_subs[("10", "strength_model")] == 0.0 and none_subs[("10", "strength_clip")] == 0.0
    assert ("10", "lora_name") not in none_subs  # name left as in the template


def test_lora_weight_arrays_expand():
    b = {"id": "l", "kind": "lora", "slots": [{"node_id": "10"}], "values": ["x.safetensors:[0.5, 1.0]:1.0"]}
    jobs = ws.expand_workflow_entry(sdxl_entry([b]))
    assert sorted(j["lora"] for j in jobs) == ["x.safetensors:0.5:1.0", "x.safetensors:1.0:1.0"]


def test_lora_too_many_parts_errors_and_second_slot_zeroed():
    wf = load("sdxl_default_api.json")
    wf["11"] = {"inputs": {"lora_name": "n.safetensors", "strength_model": 1.0, "strength_clip": 1.0,
                           "model": ["10", 0], "clip": ["10", 1]}, "class_type": "LoraLoader",
                "_meta": {"title": "Load LoRA"}}
    wf["3"]["inputs"]["model"] = ["11", 0]
    wf["6"]["inputs"]["clip"] = ["11", 1]
    wf["7"]["inputs"]["clip"] = ["11", 1]
    slots = [{"node_id": "10"}, {"node_id": "11"}]
    ok = ws.expand_workflow_entry(sdxl_entry([{"id": "l", "kind": "lora", "slots": slots,
                                               "values": ["a.safetensors:1:1"]}], workflow=wf))
    s = {(n, i): v for n, i, v in ok[0]["wf_subs"]}
    assert s[("10", "lora_name")] == "a.safetensors" and s[("11", "strength_model")] == 0.0
    with pytest.raises(ValueError, match="stacks 2 LoRAs but only 1"):
        ws.expand_workflow_entry(sdxl_entry([{"id": "l", "kind": "lora", "slots": [{"node_id": "10"}],
                                              "values": ["a.safetensors + b.safetensors"]}]))


def test_resolution_split():
    b = {"id": "r", "kind": "resolution", "width": {"node_id": "5", "input": "width"},
         "height": {"node_id": "5", "input": "height"}, "values": ["1024x1024", "832x1216"]}
    jobs = ws.expand_workflow_entry(sdxl_entry([b]))
    assert sorted((j["width"], j["height"]) for j in jobs) == [(832, 1216), (1024, 1024)]
    with pytest.raises(ValueError, match="WxH"):
        ws.expand_workflow_entry(sdxl_entry([dict(b, values=["big"])]))


def test_linked_input_needs_override():
    with pytest.raises(ValueError, match="override link"):
        ws.expand_workflow_entry(moss_entry([lit("b", "38", "samples", ["x"])]))
    replaced = ws.expand_workflow_entry(moss_entry([lit("b", "38", "samples", ["x"], override_link=True)]))
    assert replaced[0]["wf_subs"] == [["38", "samples", "x"]]
    assert len(ws.expand_workflow_entry(moss_entry([lit("b", "31", "clip_name", ["a", "b"])]))) == 2


def test_two_bindings_writing_same_input_error():
    with pytest.raises(ValueError, match="both write"):
        ws.expand_workflow_entry(moss_entry([lit("a", "32", "cfg", [1]), lit("b", "32", "cfg", [2])]))


def test_shared_targets_get_same_value():
    b = {"id": "s", "kind": "literal", "field": "seed",
         "targets": [{"node_id": "32", "input": "seed"}, {"node_id": "28", "input": "sampling_mode.seed"}],
         "values": [9]}
    j = ws.expand_workflow_entry(moss_entry([b]))[0]
    assert {(n, i) for n, i, _ in j["wf_subs"]} == {("32", "seed"), ("28", "sampling_mode.seed")}


def test_binding_off_the_output_path_warns():
    wf = load("sdxl_default_api.json")
    wf["20"] = {"inputs": {"text": "orphan", "clip": ["4", 1]}, "class_type": "CLIPTextEncode",
                "_meta": {"title": "orphan"}}
    plan = ws.plan_workflow_entry(sdxl_entry([lit("b", "20", "text", ["a", "b"])], workflow=wf))
    assert any("no effect" in w for w in plan["warnings"])


def test_cleared_field_still_substitutes_but_feeds_no_axis_field():
    j = ws.expand_workflow_entry(moss_entry([lit("b", "32", "cfg", [9], field=None)]))[0]
    assert j["wf_params"] == {"KSampler.cfg": 9.0}
    assert j["wf_subs"] == [["32", "cfg", 9.0]]


def test_axis_order_loaders_outermost_and_adjacent():
    bindings = [lit("s", "3", "cfg", [1, 2]),
                lit("m", "4", "ckpt_name", ["a.safetensors", "b.safetensors"]),
                {"id": "r", "kind": "resolution", "width": {"node_id": "5", "input": "width"},
                 "height": {"node_id": "5", "input": "height"}, "values": ["512x512", "768x768"]}]
    plan = ws.plan_workflow_entry(sdxl_entry(bindings))
    labels = [a["label"] for a in plan["axes"]]
    assert labels[0] == "Load Checkpoint.ckpt_name"
    assert labels[-1] == "KSampler.cfg"
    models = [j["model"] for j in plan["jobs"]]
    assert len(plan["jobs"]) == 8
    # loader values adjacent: exactly one change a -> b across all 8 jobs
    assert sum(1 for x, y in zip(models, models[1:]) if x != y) == 1


def test_job_cap():
    with pytest.raises(ValueError, match="more than"):
        ws.expand_workflow_entry(moss_entry([lit("b", "32", "seed", list(range(300))),
                                             lit("d", "29", "unet_name", [f"m{i}" for i in range(600)])]))


def test_expand_configs_branch_matches_direct_expansion():
    entry = moss_entry([lit("b1", "32", "cfg", [2.5, 4])])
    out = expand_configs([entry], [""], [""], [1.0], 0, [], None)
    assert [j["wf_fingerprint"] for j in out] == [j["wf_fingerprint"] for j in ws.expand_workflow_entry(entry)]


def test_mixed_standard_and_workflow_rejected_by_planner():
    with pytest.raises(ValueError, match="cannot be mixed"):
        ws.plan_workflow_configs([moss_entry([]), {"model_type": "checkpoint"}])


def test_plan_workflow_configs_templates_and_order():
    plan = ws.plan_workflow_configs([moss_entry([lit("b", "32", "cfg", [1, 2])]), sdxl_entry([])])
    assert len(plan["jobs"]) == 3 and len(plan["templates"]) == 2
    assert plan["jobs"][2]["wf_template_hash"] in plan["templates"]
