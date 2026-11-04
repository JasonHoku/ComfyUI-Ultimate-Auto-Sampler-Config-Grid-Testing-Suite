"""Skip Layer Guidance (SkipLayerGuidanceDiTSimple) config plumbing.

Builder UI state -> state_to_configs_json -> expand_configs must carry the
SLG fields the same way Deep Shrink's travel: a boolean toggle that is always
emitted, detail fields only when the toggle is on, and per-config passthrough
in the expanded configs the orchestrator hands to generate_image.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_builder_node import UltimateConfigBuilder
from config_utils import expand_configs


def _state(extra_array_fields=None):
    arr = {
        "name": "Test",
        "samplers": ["euler"],
        "schedulers": ["normal"],
        "steps": "20",
        "cfg": "7.0",
        "models": ["None"],
        "vaes": ["None"],
        "text_encoders": [],
        "clip_type": "stable_diffusion",
        "gguf_options": {},
        "loras": ["None"],
        "lora_omit_triggers": [],
        "lora_triggerwords_append_settings": {},
        "lora_bypass_states": {},
        "lora_strength_lock": {},
        "lora_weight_arrays": {},
        "model_bypass_states": {},
        "vae_bypass_states": {},
        "te_bypass_states": {},
        "combine": False,
        "positive_prompt_groups": [],
        "negative_prompt": "",
        "use_custom_prompts": False,
        "model_prompt_prefix": "",
        "model_prompt_suffix": "",
        "attention_modes": ["default"],
    }
    if extra_array_fields:
        arr.update(extra_array_fields)
    return {
        "session_name": "test_session",
        "include_none": False,
        "global_positive_groups": [],
        "global_negative": "",
        "config_arrays": [arr],
    }


def _configs(state):
    return json.loads(UltimateConfigBuilder.state_to_configs_json(state))["configs"]


def test_slg_off_emits_only_the_toggle():
    configs = _configs(_state())
    assert len(configs) == 1
    c = configs[0]
    assert c["use_slg"] is False
    assert "slg_single_layers" not in c and "slg_end_percent" not in c


def test_slg_on_emits_detail_fields_with_moss_template_defaults():
    configs = _configs(_state({"use_slg": True}))
    c = configs[0]
    assert c["use_slg"] is True
    assert c["slg_double_layers"] == ""
    assert c["slg_single_layers"] == "22"
    assert c["slg_start_percent"] == 0.0
    assert c["slg_end_percent"] == 0.5


def test_slg_on_emits_user_values():
    configs = _configs(_state({
        "use_slg": True, "slg_double_layers": "7, 8, 9", "slg_single_layers": "",
        "slg_start_percent": "0.1", "slg_end_percent": "0.3",
    }))
    c = configs[0]
    assert c["slg_double_layers"] == "7, 8, 9"
    assert c["slg_single_layers"] == ""
    assert c["slg_start_percent"] == 0.1 and c["slg_end_percent"] == 0.3


def _expand(entry):
    return expand_configs(
        raw_configs=[entry],
        pos_prompts=["a cat"],
        neg_prompts=[""],
        denoise_values=[1.0],
        seed=0,
        extra_seeds=[],
        ckpt_name="ckpt.safetensors",
    )


def test_expand_passes_slg_fields_through():
    out = _expand({"sampler": "euler", "scheduler": "normal", "steps": 20, "cfg": 7.0,
                   "model": "Default", "use_slg": True, "slg_single_layers": "22", "slg_end_percent": 0.5})
    assert len(out) == 1
    c = out[0]
    assert c["use_slg"] is True
    assert c["slg_single_layers"] == "22"
    assert c["slg_double_layers"] == ""
    assert c["slg_start_percent"] == 0.0
    assert c["slg_end_percent"] == 0.5


def test_expand_defaults_slg_off():
    out = _expand({"sampler": "euler", "scheduler": "normal", "steps": 20, "cfg": 7.0, "model": "Default"})
    assert out[0]["use_slg"] is False
