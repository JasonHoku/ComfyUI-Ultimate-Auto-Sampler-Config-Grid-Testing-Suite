"""Golden tests for UltimateConfigBuilder.state_to_configs_json.

This is the single transformer that both the runtime (generate_config) and
the preview endpoint (/configbuilder/preview) call. Drift between preview
and runtime is impossible by construction because both call this function.
"""
import json
from config_builder_node import UltimateConfigBuilder


def make_state(loras=None, lora_weight_arrays=None, lora_bypass_states=None,
               models=None, samplers=None, schedulers=None, steps="20", cfg="7.0",
               extra_array_fields=None):
    """Build a minimal valid state with one config_array for tests."""
    arr = {
        "name": "Test",
        "samplers": samplers or ["euler"],
        "schedulers": schedulers or ["normal"],
        "steps": steps,
        "cfg": cfg,
        "models": models or ["None"],
        "vaes": ["None"],
        "text_encoders": [],
        "clip_type": "stable_diffusion",
        "gguf_options": {},
        "loras": loras if loras is not None else ["None"],
        "lora_omit_triggers": [],
        "lora_triggerwords_append_settings": {},
        "lora_bypass_states": lora_bypass_states or {},
        "lora_strength_lock": {},
        "lora_weight_arrays": lora_weight_arrays or {},
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


def test_lora_strength_arrays_render_as_brackets():
    """REGRESSION: Compare Strengths emits single-bracket form for locked semantics.

    Bug history:
    - 2026-04-28 (morning): JS preview emitted brackets but Python emitted
      scalar 1.00:1.00 because Python ignored lora_weight_arrays side-channel.
    - 2026-04-28 (afternoon): Both sides emitted "name:[m]:[c]" (two arrays),
      which the orchestrator interpreted as Cartesian N×M configs instead of
      N. Locked-mode fix: emit only "name:[m]" (no clip part) so the orchestrator
      expands to N configs each with model=clip=value.

    State setup uses ONLY lora_weight_arrays[name + "_model"] (no _clip), which
    is the data shape the Builder UI now writes for compare strengths.
    """
    state = make_state(
        loras=["Sexy-IL-v11/:1.00:1.00"],
        lora_weight_arrays={
            "Sexy-IL-v11/_model": [0, 1, 2, 5, 10],
        },
    )
    json_str = UltimateConfigBuilder.state_to_configs_json(state)
    parsed = json.loads(json_str)
    loras = [c["lora"] for c in parsed["configs"]]
    # Locked semantics (default): single bracket array, no clip part.
    # The orchestrator expands "name:[a,b,c]" to N configs each with
    # model=clip=value (parse_lora_definition defaults clip to model
    # when the third segment is missing).
    expected = "Sexy-IL-v11/:[0, 1, 2, 5, 10]"
    assert any(l == expected for l in loras), \
        f"Expected exact lora string {expected!r} in output. Got: {loras}"


def test_single_lora_passes_through_verbatim():
    """A single LoRA with scalar strengths shows up unchanged in configs_json."""
    state = make_state(loras=["myLora:0.80:0.50"])
    json_str = UltimateConfigBuilder.state_to_configs_json(state)
    parsed = json.loads(json_str)
    assert len(parsed["configs"]) == 1, parsed["configs"]
    assert parsed["configs"][0]["lora"] == "myLora:0.80:0.50"


def test_multiple_loras_combine_with_plus_separator():
    """combine: True (forced) joins multiple stackable loras with ' + '.

    Note: process_lora_array forces combine=True regardless of the field's
    value in the config_array (see config_builder_node.py 'combine = True').
    """
    state = make_state(loras=["lora1:0.5:1.00", "lora2:0.7:1.00"])
    json_str = UltimateConfigBuilder.state_to_configs_json(state)
    parsed = json.loads(json_str)
    assert parsed["configs"][0]["lora"] == "lora1:0.5:1.00 + lora2:0.7:1.00"


def test_folder_reference_left_unexpanded():
    """Folder refs ('myFolder/') stay literal — orchestrator expands at runtime."""
    state = make_state(loras=["myFolder/"])
    json_str = UltimateConfigBuilder.state_to_configs_json(state)
    parsed = json.loads(json_str)
    lora = parsed["configs"][0]["lora"]
    assert lora == "myFolder/", f"Folder ref should not be expanded here, got: {lora}"


def test_folder_with_strength_array_not_combined_with_other_loras():
    """A folder ref with strength arrays must keep its trailing '/' AND
    bracket form, AND must NOT be combined with stackable loras (the combine
    block uses 'name part ends with /' to detect folder refs)."""
    state = make_state(
        loras=["myFolder/", "actualLora:0.5:1.00"],
        lora_weight_arrays={
            "myFolder/_model": [0.5, 0.8, 1.0],
        },
    )
    json_str = UltimateConfigBuilder.state_to_configs_json(state)
    parsed = json.loads(json_str)
    # Multiple loras (one folder, one stackable, no stacking happens
    # because there's only one stackable), so the per-config "lora" field
    # is a LIST — not a combined string.
    lora = parsed["configs"][0]["lora"]
    assert isinstance(lora, list), f"Expected list of separate loras, got: {lora!r}"
    folder_entry = [s for s in lora if s.startswith("myFolder/")]
    assert folder_entry, f"Folder ref missing: {lora}"
    assert "[0.5, 0.8, 1.0]" in folder_entry[0], f"Bracket form missing: {folder_entry[0]}"
    assert "actualLora:0.5:1.00" in lora, f"Stackable lora missing: {lora}"


def test_bypassed_lora_is_excluded():
    """A LoRA listed in lora_bypass_states with True is filtered out."""
    state = make_state(
        loras=["lora1:0.5:1.00", "lora2:0.7:1.00"],
        lora_bypass_states={"lora1": True},
    )
    json_str = UltimateConfigBuilder.state_to_configs_json(state)
    parsed = json.loads(json_str)
    lora = parsed["configs"][0]["lora"]
    # Only lora2 survives the bypass filter.
    assert lora == "lora2:0.7:1.00", f"Expected only lora2 to survive bypass, got: {lora}"


def test_ltx_video_config_model_type_propagates():
    """An LTX video config_array (model with type='ltx_video') reaches
    configs_json with model_type=='ltx_video'.
    See project memory: model_type lives PER-MODEL inside config_array.models[i].
    """
    state = make_state(
        models=[{"path": "ltx-2.3.safetensors", "type": "ltx_video"}],
        extra_array_fields={
            "ltx_video": {
                "clip_models": ["gemma_3_12B.safetensors", "ltx-2.3_text_projection.safetensors"],
                "vae_video": "ltx_video_vae.safetensors",
                "vae_audio": "ltx_audio_vae.safetensors",
                "latent_upscaler": "ltx_latent_upscaler.safetensors",
                "duration_seconds": 4,
                "frame_rate": 24,
                "sampler_stage1": "euler",
                "sigmas_stage1": "1.0,0.5,0.0",
                "image_strength_stage1": 0.0,
                "sampler_stage2": "euler",
                "sigmas_stage2": "0.5,0.25,0.0",
                "image_strength_stage2": 0.0,
                "img_compression": 75,
                "input_image": "",
                "audio_mode": "on",
            },
        },
    )
    json_str = UltimateConfigBuilder.state_to_configs_json(state)
    parsed = json.loads(json_str)
    cfg = parsed["configs"][0]
    assert cfg["model_type"] == "ltx_video", f"Expected model_type=ltx_video, got: {cfg.get('model_type')!r}"


def test_legacy_dual_array_preserves_cartesian_form():
    """Backward compat: if a workflow was saved with separate _model and _clip
    arrays (pre-2026-04-28-pm format), the bracket-fold should still emit
    "name:[m]:[c]" so the orchestrator can Cartesian-expand it. New UI never
    writes _clip, but legacy state should still round-trip correctly."""
    state = make_state(
        loras=["legacyLora:1:1"],
        lora_weight_arrays={
            "legacyLora_model": [0.5, 1.0],
            "legacyLora_clip": [0.7, 1.0],
        },
    )
    json_str = UltimateConfigBuilder.state_to_configs_json(state)
    parsed = json.loads(json_str)
    loras = [c["lora"] for c in parsed["configs"]]
    expected = "legacyLora:[0.5, 1.0]:[0.7, 1.0]"
    assert any(l == expected for l in loras), \
        f"Expected legacy dual-array form {expected!r}. Got: {loras}"


def _te_batch_size_for(**overrides):
    state = make_state()
    state.update(overrides)
    parsed = json.loads(UltimateConfigBuilder.state_to_configs_json(state))
    return parsed["_session_settings"]["te_batch_size"]


def test_te_batch_size_defaults_off_and_is_clamped():
    """te_batch_size is a session-level Run Setting: default 1 (off), junk/<1 -> 1."""
    assert _te_batch_size_for() == 1
    assert _te_batch_size_for(te_batch_size=8) == 8
    assert _te_batch_size_for(te_batch_size="8") == 8
    assert _te_batch_size_for(te_batch_size="junk") == 1
    assert _te_batch_size_for(te_batch_size=None) == 1
    assert _te_batch_size_for(te_batch_size=0) == 1
    assert _te_batch_size_for(te_batch_size=-3) == 1
# ---- job order + preview pass run settings -----------------------------------

def _ss(**state_extra):
    state = make_state()
    state.update(state_extra)
    return json.loads(UltimateConfigBuilder.state_to_configs_json(state))["_session_settings"]


def test_preview_and_job_order_defaults_are_emitted():
    ss = _ss()
    assert ss["job_order"] == "as_written"
    assert ss["preview_pass"] == {"enabled": False, "steps": 8, "scale": 0.5, "cfg_override": None}
    assert ss["skip_rejected_previews"] is True
    assert ss["only_favorited_previews"] is False


def test_preview_settings_are_coerced():
    ss = _ss(job_order="minimize_reloads",
             preview_pass={"enabled": 1, "steps": "12", "scale": "0.33", "cfg_override": "3.5"},
             skip_rejected_previews=0, only_favorited_previews=1)
    assert ss["job_order"] == "minimize_reloads"
    assert ss["preview_pass"] == {"enabled": True, "steps": 12, "scale": 0.33, "cfg_override": 3.5}
    assert ss["skip_rejected_previews"] is False
    assert ss["only_favorited_previews"] is True


def test_preview_bad_types_fall_back_and_clamp():
    ss = _ss(preview_pass={"enabled": True, "steps": "abc", "scale": "x", "cfg_override": "nope"})
    assert ss["preview_pass"] == {"enabled": True, "steps": 8, "scale": 0.5, "cfg_override": None}
    ss = _ss(preview_pass={"steps": 0, "scale": 7})
    assert ss["preview_pass"]["steps"] == 1 and ss["preview_pass"]["scale"] == 1.0
    assert _ss(preview_pass={"scale": 0})["preview_pass"]["scale"] == 0.05
    assert _ss(preview_pass="garbage")["preview_pass"]["enabled"] is False


def test_cfg_override_blank_is_null_and_zero_is_kept():
    assert _ss(preview_pass={"cfg_override": ""})["preview_pass"]["cfg_override"] is None
    assert _ss(preview_pass={"cfg_override": None})["preview_pass"]["cfg_override"] is None
    assert _ss(preview_pass={"cfg_override": 0})["preview_pass"]["cfg_override"] == 0.0


def test_unknown_job_order_becomes_as_written():
    assert _ss(job_order="alphabetical")["job_order"] == "as_written"


def test_checkpoint_without_vae_omits_key_and_reports_no_drift(capsys):
    state = make_state(models=["m.safetensors"])
    out = json.loads(UltimateConfigBuilder.state_to_configs_json(state))
    assert all("vae" not in c for c in out["configs"])
    assert "schema drift" not in capsys.readouterr().out.lower()


def test_picked_vae_is_still_emitted(capsys):
    state = make_state(models=["m.safetensors"], extra_array_fields={"vaes": ["my.vae.safetensors"]})
    out = json.loads(UltimateConfigBuilder.state_to_configs_json(state))
    assert any(c.get("vae") for c in out["configs"])
    assert "schema drift" not in capsys.readouterr().out.lower()
