"""Pure helpers behind POST /config_tester/run and GET /config_tester/run_status."""
import importlib.util
import json
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(f"{_PKG}.{name}_real", os.path.join(_ROOT, filename))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG
    spec.loader.exec_module(mod)
    return mod


run_api = _load("run_api", "run_api.py")


# ---- sanitize ---------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("../../etc", "etc"),
    ("a b/c", "abc"),
    ("ünï", "ünï"),
    ("", ""),
    (None, ""),
    ("my-session_1", "my-session_1"),
    ("..", ""),
])
def test_sanitize(raw, expected):
    assert run_api.sanitize_session_name_strict(raw) == expected


# ---- normalize --------------------------------------------------------------
def test_normalize_accepts_all_shapes():
    cfg = [{"model": "x"}]
    assert run_api.normalize_configs_payload({"configs": cfg}) == {"configs": cfg}
    assert run_api.normalize_configs_payload(cfg) == {"configs": cfg}
    assert run_api.normalize_configs_payload(json.dumps({"configs": cfg})) == {"configs": cfg}
    assert run_api.normalize_configs_payload(json.dumps(cfg)) == {"configs": cfg}


@pytest.mark.parametrize("bad", ["not json", {}, {"configs": []}, 42, [], None, {"configs": "x"}])
def test_normalize_rejects(bad):
    with pytest.raises(ValueError):
        run_api.normalize_configs_payload(bad)


def test_normalize_preserves_other_keys():
    out = run_api.normalize_configs_payload({"configs": [{"a": 1}], "_distribution": {"enabled": True}})
    assert out["_distribution"] == {"enabled": True}


# ---- overrides --------------------------------------------------------------
def test_session_precedence():
    obj, s = run_api.apply_run_overrides({"configs": [1], "_session_settings": {"session_name": "from_settings"}},
                                         session_name="from_body")
    assert s == "from_body" and obj["_session_settings"]["session_name"] == "from_body"
    obj, s = run_api.apply_run_overrides({"configs": [1], "_session_settings": {"session_name": "from_settings"}})
    assert s == "from_settings"
    obj, s = run_api.apply_run_overrides({"configs": [1]})
    assert s == "my_session"
    assert obj["_session_settings"]["session_name"] == "my_session"  # dict created when absent


def test_session_is_sanitised_and_can_end_empty():
    _, s = run_api.apply_run_overrides({"configs": [1]}, session_name="../../x y")
    assert s == "xy"
    _, s = run_api.apply_run_overrides({"configs": [1]}, session_name="///")
    assert s == ""


def test_overwrite_only_when_given():
    obj, _ = run_api.apply_run_overrides({"configs": [1]})
    assert "overwrite_existing" not in obj["_session_settings"]
    obj, _ = run_api.apply_run_overrides({"configs": [1]}, overwrite_existing=False)
    assert obj["_session_settings"]["overwrite_existing"] is False
    obj, _ = run_api.apply_run_overrides({"configs": [1]}, overwrite_existing=True)
    assert obj["_session_settings"]["overwrite_existing"] is True


def test_nonce_changes_serialised_string():
    a, _ = run_api.apply_run_overrides({"configs": [1]}, session_name="s")
    b, _ = run_api.apply_run_overrides({"configs": [1]}, session_name="s", nonce="abc")
    assert "_run_nonce" not in a and b["_run_nonce"] == "abc"
    assert json.dumps(a) != json.dumps(b)


def test_non_dict_settings_replaced():
    obj, s = run_api.apply_run_overrides({"configs": [1], "_session_settings": "junk"}, session_name="s")
    assert obj["_session_settings"] == {"session_name": "s"}


# ---- prompt -----------------------------------------------------------------
def test_build_run_prompt_exact():
    cs = json.dumps({"configs": [{"a": 1}]}, indent=2)
    assert run_api.build_run_prompt(cs, "sess") == {
        "1": {"class_type": "UltimateSamplerGrid", "inputs": {"configs_json": cs}},
        "2": {"class_type": "UltimateGridDashboard", "inputs": {"session_name": "sess", "dashboard_html": ["1", 0]}},
    }
    assert json.loads(run_api.build_run_prompt(cs, "s")["1"]["inputs"]["configs_json"]) == {"configs": [{"a": 1}]}


# ---- saved configs ----------------------------------------------------------
def test_saved_state_uses_injected_converter():
    calls = []

    def fake(state):
        calls.append(state)
        return json.dumps({"configs": [{"model": "m"}]})

    out = run_api.saved_config_to_configs_obj({"config_arrays": [], "x": 1}, fake)
    assert out == {"configs": [{"model": "m"}]} and calls[0]["x"] == 1


def test_saved_configs_shape_passes_through():
    data = {"configs": [{"a": 1}]}
    assert run_api.saved_config_to_configs_obj(data, lambda s: pytest.fail("must not call")) is data


@pytest.mark.parametrize("junk", [{}, {"foo": 1}, [], "x", None])
def test_saved_junk_raises(junk):
    with pytest.raises(ValueError):
        run_api.saved_config_to_configs_obj(junk, lambda s: "{}")


def test_resolve_saved_config(tmp_path):
    (tmp_path / "ok.json").write_text("{}")
    assert run_api.resolve_saved_config(str(tmp_path), "ok") == str(tmp_path / "ok.json")
    assert run_api.resolve_saved_config(str(tmp_path), "ok.json") == str(tmp_path / "ok.json")
    # traversal reduces to the basename inside the dir
    assert run_api.resolve_saved_config(str(tmp_path), "../../ok") == str(tmp_path / "ok.json")
    with pytest.raises(FileNotFoundError):
        run_api.resolve_saved_config(str(tmp_path), "../missing")
    with pytest.raises(FileNotFoundError):
        run_api.resolve_saved_config(str(tmp_path), "")


# ---- status helpers ---------------------------------------------------------
def test_summarize_history_never_leaks():
    entry = {
        "prompt": [1, "pid", {"1": {"inputs": {"configs_json": "SECRET"}}}, {}, ["2"]],
        "outputs": {"2": {"dashboard_html": ["<html>HUGE</html>"]}},
        "status": {"status_str": "error", "completed": False, "messages": [
            ["execution_start", {"prompt_id": "pid"}],
            ["execution_error", {"node_id": "1", "node_type": "UltimateSamplerGrid", "exception_message": "boom",
                                 "traceback": ["x"]}],
        ]},
    }
    s = run_api.summarize_history(entry)
    assert s["status"] == "error" and s["completed"] is False
    assert s["error"] == {"node_id": "1", "node_type": "UltimateSamplerGrid", "exception_message": "boom"}
    blob = json.dumps(s)
    assert "SECRET" not in blob and "HUGE" not in blob and "traceback" not in blob


def test_summarize_history_success_and_malformed():
    ok = run_api.summarize_history({"status": {"status_str": "success", "completed": True, "messages": []}})
    assert ok == {"status": "success", "completed": True}
    assert run_api.summarize_history({})["status"] == "unknown"
    run_api.summarize_history({"status": {"messages": [None, [], "x"]}})  # must not raise


def test_find_in_queue_five_and_six_tuples():
    running = [(0, "run", {}, {}, ["2"], {})]
    pending = [(5, "late", {}, {}, ["2"]), (2, "early", {}, {}, ["2"], {}), (3, "mid", {}, {}, ["2"])]
    assert run_api.find_in_queue(running, pending, "run") == ("running", 0)
    assert run_api.find_in_queue(running, pending, "early") == ("pending", 0)
    assert run_api.find_in_queue(running, pending, "mid") == ("pending", 1)
    assert run_api.find_in_queue(running, pending, "late") == ("pending", 2)
    assert run_api.find_in_queue(running, pending, "nope") is None
    assert run_api.find_in_queue(None, None, "x") is None


def test_run_tracker_is_bounded():
    t = run_api.RunTracker(cap=3)
    for i in range(5):
        t.add(f"p{i}", f"s{i}")
    assert t.get("p0") is None and t.get("p1") is None and t.get("p4") == "s4"
