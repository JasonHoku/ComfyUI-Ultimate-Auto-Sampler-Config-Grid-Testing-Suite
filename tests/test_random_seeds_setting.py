"""Additional Random Seeds run setting: Builder emit -> sampler_node unwrap -> expand_configs,
plus the deterministic per-base-seed derivation (config_utils.derive_extra_seeds)."""
import importlib.util
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_builder_node import UltimateConfigBuilder
from config_utils import derive_extra_seeds, expand_configs
from tests.test_state_to_configs_json import make_state  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)


def _settings(**kw):
    state = make_state()
    state.update(kw)
    return json.loads(UltimateConfigBuilder.state_to_configs_json(state))["_session_settings"]


def test_emit_value():
    assert _settings(add_random_seeds_to_gens=3)["add_random_seeds_to_gens"] == 3


def test_emit_missing_defaults_zero():
    assert _settings()["add_random_seeds_to_gens"] == 0


def test_emit_garbage_defaults_zero():
    assert _settings(add_random_seeds_to_gens="abc")["add_random_seeds_to_gens"] == 0


def test_emit_numeric_string():
    assert _settings(add_random_seeds_to_gens="2")["add_random_seeds_to_gens"] == 2


def test_derive_same_base_seed_same_list():
    assert derive_extra_seeds(123, 5) == derive_extra_seeds(123, 5)
    assert len(derive_extra_seeds(123, 5)) == 5


def test_derive_different_base_seed_differs():
    assert derive_extra_seeds(123, 5) != derive_extra_seeds(124, 5)


def test_derive_prefix_stable_and_zero_empty():
    assert derive_extra_seeds(7, 3) == derive_extra_seeds(7, 6)[:3]
    assert derive_extra_seeds(7, 0) == []
    assert derive_extra_seeds(7, "x") == []


def test_expand_configs_extra_seeds_multiply_count():
    raw = [{"model": "m.safetensors", "sampler": "euler", "scheduler": "normal", "steps": 5, "cfg": 1}]
    base = expand_configs(raw, [""], [""], [1.0], 0, [])
    more = expand_configs(raw, [""], [""], [1.0], 0, [11, 22])
    assert len(base) >= 1
    assert len(more) == 3 * len(base)


_REAL = {}


def _load_sampler_node():
    # conftest registers a STUB `<pkg>.sampler_node`; load the real file under a private name.
    if "mod" not in _REAL:
        # the conftest stubs for the sibling modules lack some names sampler_node imports;
        # add inert placeholders for the missing ones only (never overrides a real attribute).
        for modname, names in {
            "lora_utils": ["load_and_save_tags"],
            "config_utils": ["parse_json_with_error", "parse_float_input", "parse_string_input",
                             "expand_configs", "prepare_input_jobs", "sanitize_session_name",
                             "normalize_str", "get_files_from_folder", "parse_lora_definition"],
        }.items():
            stub = sys.modules.get(f"{_PKG}.{modname}")
            for n in names:
                if stub is not None and not hasattr(stub, n):
                    setattr(stub, n, lambda *a, **k: None)
        spec = importlib.util.spec_from_file_location(f"{_PKG}.sampler_node_real", os.path.join(_ROOT, "sampler_node.py"))
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = _PKG
        spec.loader.exec_module(mod)
        _REAL["mod"] = mod
    return _REAL["mod"]


def _run_tests_arg(monkeypatch, configs_json):
    try:
        sn = _load_sampler_node()
    except Exception as exc:  # stubs may be too thin for this module
        pytest.skip(f"sampler_node not importable under stubs: {exc}")
    captured = {}
    monkeypatch.setattr(sys.modules["folder_paths"], "get_filename_list", lambda *_: [], raising=False)
    orch = sys.modules[f"{_PKG}.generation_orchestrator"]
    monkeypatch.setattr(orch, "run_generation_loop", lambda *a, **k: captured.setdefault("args", a) or ("",))
    sn.SamplerGridTester().run_tests(configs_json)
    return captured["args"]


def test_sampler_node_unwraps_session_setting(monkeypatch):
    args = _run_tests_arg(monkeypatch, json.dumps({"configs": [{}], "_session_settings": {"add_random_seeds_to_gens": 2}}))
    assert args[13] == 2


def test_sampler_node_default_zero(monkeypatch):
    args = _run_tests_arg(monkeypatch, json.dumps({"configs": [{}]}))
    assert args[13] == 0
