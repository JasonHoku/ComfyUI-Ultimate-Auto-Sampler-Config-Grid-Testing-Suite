"""Builder state -> `_session_settings.metrics` emission and the metrics_info route."""
import asyncio
import importlib.util
import json
import os

from config_builder_node import UltimateConfigBuilder
import config_builder_node as cbn

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("_golden_state_helpers", os.path.join(_HERE, "test_state_to_configs_json.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)

ALL_ON = {k: True for k in ("sharpness", "entropy", "colorfulness", "contrast", "luma", "saturation")}


def _settings(metrics=None, present=True):
    state = _helpers.make_state()
    if present:
        state["metrics"] = metrics
    parsed = json.loads(UltimateConfigBuilder.state_to_configs_json(state))
    return parsed.get("_session_settings", {})


def test_missing_metrics_emits_nothing():
    assert "metrics" not in _settings(present=False)


def test_disabled_emits_nothing():
    assert "metrics" not in _settings({"enabled": False, "builtins": ALL_ON, "adapters": {}})


def test_enabled_with_nothing_checked_emits_nothing():
    off = {k: False for k in ALL_ON}
    assert "metrics" not in _settings({"enabled": True, "builtins": off, "adapters": {}})
    assert "metrics" not in _settings({"enabled": True})


def test_enabled_exact_dict_in_canonical_order():
    m = {"enabled": True, "builtins": {"luma": True, "sharpness": True, "entropy": False},
         "adapters": {"aesthetic": True, "imagereward": False}}
    assert _settings(m)["metrics"] == {"enabled": True, "builtins": ["sharpness", "luma"], "adapters": ["aesthetic"]}


def test_garbage_names_filtered():
    m = {"enabled": True, "builtins": {"luma": True, "nope": True}, "adapters": {"evil": True}}
    assert _settings(m)["metrics"] == {"enabled": True, "builtins": ["luma"], "adapters": []}


def test_garbage_shapes_do_not_raise():
    assert "metrics" not in _settings("yes")
    assert _settings({"enabled": True, "builtins": ["luma", "bad"], "adapters": None})["metrics"]["builtins"] == ["luma"]


def test_metrics_is_not_a_per_config_key():
    state = _helpers.make_state()
    state["metrics"] = {"enabled": True, "builtins": ALL_ON, "adapters": {}}
    parsed = json.loads(UltimateConfigBuilder.state_to_configs_json(state))
    assert all("metrics" not in c for c in parsed["configs"])


def test_metrics_info_route_shape(monkeypatch):
    captured = {}
    monkeypatch.setattr(cbn.web, "json_response", lambda data, **kw: captured.setdefault("data", data), raising=False)
    asyncio.run(cbn.get_metrics_info(None))
    data = captured["data"]
    assert data["builtins"] == ["sharpness", "entropy", "colorfulness", "contrast", "luma", "saturation", "sobel"]
    assert {a["name"] for a in data["adapters"]} >= {"aesthetic", "imagereward"}
    assert all({"name", "label", "installed", "enabled", "pack_hint"} <= set(a) for a in data["adapters"])
