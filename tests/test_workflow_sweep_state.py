"""Builder state -> configs_json for Workflow Sweep arrays (kind: "workflow") + standard regression."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config_builder_node import UltimateConfigBuilder  # noqa: E402


def _standard_array(**extra):
    arr = {
        "name": "Test", "samplers": ["euler"], "schedulers": ["normal"], "steps": "20", "cfg": "7.0",
        "models": ["None"], "vaes": ["None"], "text_encoders": [], "clip_type": "stable_diffusion",
        "gguf_options": {}, "loras": ["a:1.0:1.0"], "lora_omit_triggers": [],
        "lora_triggerwords_append_settings": {}, "lora_bypass_states": {}, "lora_strength_lock": {},
        "lora_weight_arrays": {}, "model_bypass_states": {}, "vae_bypass_states": {}, "te_bypass_states": {},
        "combine": False, "positive_prompt_groups": [], "negative_prompt": "", "use_custom_prompts": False,
        "model_prompt_prefix": "", "model_prompt_suffix": "", "attention_modes": ["default"],
    }
    arr.update(extra)
    return arr


def _workflow_array(**extra):
    arr = {
        "name": "WF", "kind": "workflow",
        "workflow": {
            "source_name": "t.json",
            "api_prompt": {"3": {"class_type": "KSampler", "inputs": {"seed": 1, "cfg": 7.0}}},
            "output": {"node_id": "8", "output_index": 0},
            "run_other_outputs": False,
            "bindings": [{"id": "b1", "kind": "literal", "targets": [{"node_id": "3", "input": "cfg"}],
                          "values": [2, 3]}],
        },
        "positive_prompt_groups": [], "negative_prompt": "", "use_custom_prompts": False,
    }
    arr.update(extra)
    return arr


def _state(*arrays, **extra):
    state = {"session_name": "wf_s", "include_none": False, "global_positive_groups": [], "global_negative": "",
             "config_arrays": list(arrays)}
    state.update(extra)
    return state


def _configs(state):
    return json.loads(UltimateConfigBuilder.state_to_configs_json(state))


def test_workflow_array_emits_workflow_config_shape():
    parsed = _configs(_state(_workflow_array()))
    cfg = parsed["configs"][0]
    assert cfg["model_type"] == "workflow" and cfg["name"] == "WF"
    assert cfg["workflow"]["3"]["class_type"] == "KSampler"
    assert cfg["workflow_source"] == "t.json"
    assert cfg["workflow_output"] == {"node_id": "8", "output_index": 0}
    assert cfg["workflow_run_other_outputs"] is False
    assert cfg["workflow_bindings"][0]["values"] == [2, 3]
    assert "sampler" not in cfg and "lora" not in cfg  # standard-only keys are not emitted
    assert "_session_settings" in parsed


def test_workflow_array_prompt_priority():
    custom = _workflow_array(use_custom_prompts=True, positive_prompt_groups=[["custom"]], negative_prompt="cn")
    cfg = _configs(_state(custom, global_positive_groups=[["global"]]))["configs"][0]
    assert cfg["positive"] == [["custom"]] and cfg["negative"] == "cn" and cfg["_prompt_source"] == "custom"
    cfg = _configs(_state(_workflow_array(), global_positive_groups=[["global"]], global_negative="gn"))["configs"][0]
    assert cfg["positive"] == [["global"]] and cfg["negative"] == "gn" and cfg["_prompt_source"] == "global"
    assert "positive" not in _configs(_state(_workflow_array()))["configs"][0]


def test_workflow_array_without_output_emits_empty_output():
    arr = _workflow_array()
    arr["workflow"]["output"] = {}
    assert _configs(_state(arr))["configs"][0]["workflow_output"] == {}


def test_mixed_arrays_keep_builder_order():
    cfgs = _configs(_state(_standard_array(), _workflow_array()))["configs"]
    assert [c["model_type"] for c in cfgs] == ["checkpoint", "workflow"]


def test_standard_array_unchanged_by_kind_standard():
    plain = _configs(_state(_standard_array()))
    tagged = _configs(_state(_standard_array(kind="standard")))
    assert plain == tagged and plain["configs"][0]["model_type"] == "checkpoint"


def test_emitted_workflow_config_expands_through_expand_configs():
    from config_utils import expand_configs
    arr = _workflow_array()
    arr["workflow"]["output"] = {"node_id": "3", "output_index": 0}  # the mini template has only a sampler
    cfg = _configs(_state(arr))["configs"]
    jobs = expand_configs(cfg, [""], [""], [1.0], 0, [], None)
    assert sorted(j["cfg"] for j in jobs) == [2.0, 3.0]
