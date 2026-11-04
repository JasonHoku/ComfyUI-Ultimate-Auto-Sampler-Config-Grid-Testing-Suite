"""Workflow Sweep: prompt fidelity (template + substitutions + collector), fingerprints."""
import copy
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import workflow_sweep as ws  # noqa: E402


def load(name):
    with open(os.path.join(ROOT, "tests", "fixtures", name), "r", encoding="utf-8") as f:
        return json.load(f)


def entry(bindings):
    return {"model_type": "workflow", "name": "c", "workflow": load("image_moss_image_t2i.json"),
            "workflow_output": {"node_id": "38", "output_index": 0}, "workflow_bindings": bindings,
            "positive": ["p1", "p2"]}


BINDINGS = [
    {"id": "b1", "kind": "literal", "targets": [{"node_id": "32", "input": "cfg"}], "values": [2.5, 4]},
    {"id": "b2", "kind": "literal", "targets": [{"node_id": "32", "input": "seed"}], "values": [1, 2]},
    {"id": "b4", "kind": "positive", "targets": [{"node_id": "26", "input": "value"}]},
]


def test_every_job_minus_collector_equals_template_plus_subs():
    template = load("image_moss_image_t2i.json")
    plan = ws.plan_workflow_entry(entry(BINDINGS))
    out = plan["template"]["output"]
    assert len(plan["jobs"]) == 8
    for job in plan["jobs"]:
        prompt, cid = ws.build_job_prompt(plan["template"]["prompt"], job["wf_subs"], out,
                                          {"session_name": "s", "job_json": "{}"})
        expected = copy.deepcopy(template)
        for nid, inp, val in job["wf_subs"]:
            expected[nid]["inputs"][inp] = val
        rest = {k: v for k, v in prompt.items() if k != cid}
        assert rest == expected                      # keys, inputs and _meta all intact
        assert set(prompt) - set(template) == {cid}  # nothing else added
        assert prompt[cid]["class_type"] == "USCGSweepCollect"
        assert prompt[cid]["inputs"]["images"] == ["38", 0]


def test_template_object_is_never_mutated():
    plan = ws.plan_workflow_entry(entry(BINDINGS))
    snapshot = copy.deepcopy(plan["template"]["prompt"])
    ws.build_job_prompt(plan["template"]["prompt"], plan["jobs"][0]["wf_subs"], plan["template"]["output"], {})
    assert plan["template"]["prompt"] == snapshot


def test_collector_id_numeric_and_unused():
    template = load("image_moss_image_t2i.json")
    _, cid = ws.build_job_prompt(template, [], ("38", 0), {})
    assert cid == "41" and cid not in template
    p = {"7": {"class_type": "X", "inputs": {}}, "8:1": {"class_type": "Y", "inputs": {}}}
    assert ws.collector_node_id(p) == "8"
    assert ws.collector_node_id({"a": {"class_type": "X", "inputs": {}}}) == "1"


def test_unknown_target_node_errors():
    with pytest.raises(ValueError, match="not in the workflow"):
        ws.apply_subs(load("sdxl_default_api.json"), [("999", "x", 1)])


def test_fingerprint_stable_across_key_order_and_meta_edits():
    p = load("sdxl_default_api.json")
    f1 = ws.fingerprint(p, ("8", 0))
    reordered = {k: p[k] for k in reversed(list(p))}
    for node in reordered.values():
        node["inputs"] = {k: node["inputs"][k] for k in reversed(list(node["inputs"]))}
    assert ws.fingerprint(reordered, ("8", 0)) == f1
    edited = copy.deepcopy(p)
    edited["3"]["_meta"]["title"] = "renamed"
    assert ws.fingerprint(edited, ("8", 0)) == f1
    assert len(f1) == 20


def test_fingerprint_changes_on_any_input_or_output_change():
    p = load("sdxl_default_api.json")
    f1 = ws.fingerprint(p, ("8", 0))
    q = copy.deepcopy(p)
    q["7"]["inputs"]["text"] = "different"  # even an unbound value is a different job
    assert ws.fingerprint(q, ("8", 0)) != f1
    assert ws.fingerprint(p, ("8", 1)) != f1


def test_fingerprint_independent_of_collector_inputs():
    plan = ws.plan_workflow_entry(entry(BINDINGS))
    job = plan["jobs"][0]
    tp, out = plan["template"]["prompt"], plan["template"]["output"]
    a, ca = ws.build_job_prompt(tp, job["wf_subs"], out, {"job_json": "a"})
    b, cb = ws.build_job_prompt(tp, job["wf_subs"], out, {"job_json": "b"})
    a.pop(ca)
    b.pop(cb)
    assert ws.fingerprint(a, ("38", 0)) == ws.fingerprint(b, ("38", 0)) == job["wf_fingerprint"]


def test_template_hash_covers_output_and_run_other():
    p = load("sdxl_default_api.json")
    h = ws.template_hash(p, ("8", 0), False)
    assert ws.template_hash(p, ("8", 0), True) != h and ws.template_hash(p, ("3", 0), False) != h


def test_job_conf_carries_no_template():
    plan = ws.plan_workflow_entry(entry(BINDINGS))
    blob = json.dumps(plan["jobs"][0])
    assert "workflow" not in json.loads(blob) and "class_type" not in blob
