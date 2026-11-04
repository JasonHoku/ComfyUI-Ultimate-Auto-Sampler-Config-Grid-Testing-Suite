"""Validation of the shipped examples/*.json workflows (no torch, no ComfyUI).

Checks the graph is well formed, the Builder state restores/emits correctly for
each model family, and the image counts promised in examples/README.md are real.
"""
import ast
import glob
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_builder_node import UltimateConfigBuilder
from config_utils import expand_configs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX_DIR = os.path.join(ROOT, "examples")
FILES = sorted(glob.glob(os.path.join(EX_DIR, "*.json")))

# Images each example expands to (must match the "**N images**" line in examples/README.md).
EXPECTED_IMAGES = {
    "uscg-sdxl-illustrious.json": 16,
    "uscg-anima.json": 4,
    "uscg-z-image-turbo.json": 2,
    "uscg-krea2-turbo.json": 2,
    "uscg-flux2-klein-9b.json": 4,  # Advanced Sampling on + off (expand_configs grids both)
    "uscg-moss-image.json": 2,
}

# Every top-level key of the Builder's new-state object (web/conf_builder/conf-builder-main.js,
# "Initialize default state structure"). A saved state missing one would silently fall back to defaults.
STATE_KEYS = {
    "session_name", "config_name", "auto_save", "include_none", "label_mode", "global_positive_groups",
    "global_negative", "distribution_enabled", "worker_urls", "claim_timeout", "use_master_encoding",
    "config_arrays", "upscaling", "cooldown", "image_format", "overwrite_existing", "flush_batch_every",
    "lora_triggerwords_mode", "save_conditioning_cache_to_file", "enable_model_cache", "vae_batch_size",
    "add_random_seeds_to_gens",
}

# Filenames the official template set uses, per example (placeholders are only valid if listed here).
ALLOWED_FILES = {
    "uscg-sdxl-illustrious.json": {"sd_xl_base_1.0.safetensors"},
    "uscg-anima.json": {"anima-base-v1.0.safetensors", "qwen_3_06b_base.safetensors", "qwen_image_vae.safetensors"},
    "uscg-z-image-turbo.json": {"z_image_turbo_bf16.safetensors", "qwen_3_4b.safetensors", "ae.safetensors"},
    "uscg-krea2-turbo.json": {"krea2_turbo_fp8_scaled.safetensors", "qwen3vl_4b_fp8_scaled.safetensors", "qwen_image_vae.safetensors"},
    "uscg-flux2-klein-9b.json": {"flux-2-klein-9b-fp8.safetensors", "qwen_3_8b_fp8mixed.safetensors", "flux2-vae.safetensors"},
    "uscg-moss-image.json": {"moss_image_sft_int8_convrot.safetensors", "qwen3.5_9b_w4a8_gptq.safetensors", "flux2-vae_small.safetensors"},
}


def _node_class_names():
    tree = ast.parse(open(os.path.join(ROOT, "__init__.py"), encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "NODE_CLASS_MAPPINGS" for t in node.targets):
            return {k.value for k in node.value.keys}
    raise AssertionError("NODE_CLASS_MAPPINGS not found")


def _load(path):
    return json.load(open(path, encoding="utf-8"))


def _state(wf):
    builder = next(n for n in wf["nodes"] if n["type"] == "UltimateConfigBuilder")
    return json.loads(builder["widgets_values"][6])


def test_example_files_present():
    assert {os.path.basename(f) for f in FILES} == set(EXPECTED_IMAGES)


@pytest.mark.parametrize("path", FILES, ids=os.path.basename)
def test_graph_is_well_formed(path):
    wf = _load(path)
    assert {"version", "nodes", "links"} <= set(wf)
    known = _node_class_names() | {"MarkdownNote", "Note"}
    types = [n["type"] for n in wf["nodes"]]
    assert set(types) <= known
    for t in ("UltimateConfigBuilder", "UltimateSamplerGrid", "UltimateGridDashboard"):
        assert types.count(t) == 1

    node_ids = [n["id"] for n in wf["nodes"]]
    assert len(set(node_ids)) == len(node_ids)
    link_ids = [l[0] for l in wf["links"]]
    assert len(set(link_ids)) == len(link_ids)
    assert wf["last_node_id"] >= max(node_ids)
    assert wf["last_link_id"] >= max(link_ids)

    by_id = {n["id"]: n for n in wf["nodes"]}
    for lid, src, src_slot, dst, dst_slot, typ in wf["links"]:
        assert lid in by_id[src]["outputs"][src_slot]["links"]
        assert by_id[dst]["inputs"][dst_slot]["link"] == lid
        assert by_id[src]["outputs"][src_slot]["type"] == typ == by_id[dst]["inputs"][dst_slot]["type"]
    for n in wf["nodes"]:
        for i in n["inputs"]:
            if i.get("link") is not None:
                assert i["link"] in link_ids
        for o in n["outputs"]:
            for l in o.get("links") or []:
                assert l in link_ids

    builder = next(n for n in wf["nodes"] if n["type"] == "UltimateConfigBuilder")
    gen = next(n for n in wf["nodes"] if n["type"] == "UltimateSamplerGrid")
    dash = next(n for n in wf["nodes"] if n["type"] == "UltimateGridDashboard")
    assert any(l[1] == builder["id"] and l[2] == 0 and l[3] == gen["id"] and gen["inputs"][l[4]]["name"] == "configs_json" for l in wf["links"])
    assert any(l[1] == gen["id"] and l[2] == 0 and l[3] == dash["id"] and dash["inputs"][l[4]]["name"] == "dashboard_html" for l in wf["links"])


@pytest.mark.parametrize("path", FILES, ids=os.path.basename)
def test_builder_state_restorable(path):
    wf = _load(path)
    builder = next(n for n in wf["nodes"] if n["type"] == "UltimateConfigBuilder")
    assert len(builder["widgets_values"]) == 9
    state = _state(wf)
    assert state["config_arrays"]
    assert STATE_KEYS <= set(state)
    dash = next(n for n in wf["nodes"] if n["type"] == "UltimateGridDashboard")
    assert dash["widgets_values"][0] == state["session_name"]


@pytest.mark.parametrize("path", FILES, ids=os.path.basename)
def test_emits_and_counts(path, capsys):
    name = os.path.basename(path)
    state = _state(_load(path))
    out = json.loads(UltimateConfigBuilder.state_to_configs_json(state))
    # state_to_configs_json lists "vae" in _EXPECTED_CONFIG_KEYS but omits it on purpose when no VAE is
    # picked (the SDXL checkpoint has its own), so that one key may be reported; anything else is real drift.
    drift = [l for l in capsys.readouterr().out.splitlines() if "Config schema drift" in l]
    assert all("missing keys {'vae'}" in l for l in drift), drift
    assert out["_session_settings"]["add_random_seeds_to_gens"] == 0
    count = len(expand_configs(out["configs"], [""], [""], [1.0], 0, [], None))
    assert count == EXPECTED_IMAGES[name]
    readme = open(os.path.join(EX_DIR, "README.md"), encoding="utf-8").read()
    assert f"**{count} images**" in readme


@pytest.mark.parametrize("path", FILES, ids=os.path.basename)
def test_no_local_paths_and_known_filenames(path):
    name = os.path.basename(path)
    text = open(path, encoding="utf-8").read()
    assert "\\\\" not in text.replace("\\n", "")  # no escaped backslashes (Windows paths)
    assert not re.search(r"[A-Za-z]:[\\/]", text.replace("\\n", " "))
    assert "Users" not in text
    state = _state(_load(path))
    used = set()
    for a in state["config_arrays"]:
        for m in a["models"]:
            used.add(m["path"] if isinstance(m, dict) else m)
        used.update(a["text_encoders"])
        used.update(v for v in a["vaes"] if v != "None")
    assert used <= ALLOWED_FILES[name]


def _cfgs(key):
    return json.loads(UltimateConfigBuilder.state_to_configs_json(_state(_load(os.path.join(EX_DIR, key)))))["configs"]


def test_family_settings():
    sdxl = _cfgs("uscg-sdxl-illustrious.json")[0]
    assert sdxl["model_type"] == "checkpoint"

    anima = _cfgs("uscg-anima.json")[0]
    assert anima["model_type"] == "diffusion_model"
    assert anima["clip_type"] == "stable_diffusion"
    assert anima["text_encoders"] == ["qwen_3_06b_base.safetensors"]
    assert anima["vae"] == "qwen_image_vae.safetensors"

    z = _cfgs("uscg-z-image-turbo.json")[0]
    assert z["clip_type"] == "lumina2"
    assert z["model_sampling_override"] == "aura_flow"

    assert _cfgs("uscg-krea2-turbo.json")[0]["clip_type"] == "krea2"

    klein = _cfgs("uscg-flux2-klein-9b.json")[0]
    assert klein["clip_type"] == "flux2"
    assert klein["advanced_scheduler"] == "flux2"
    assert klein["use_advanced_sampling"] is True

    moss = _cfgs("uscg-moss-image.json")
    assert len(moss) == 2
    assert moss[0]["clip_type"] == "moss_image"
    assert moss[0]["use_slg"] is True and moss[0]["slg_single_layers"] == "22"
    assert moss[1]["use_slg"] is False
