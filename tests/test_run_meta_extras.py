"""run_meta_extras (manifest meta that lets the Dashboard re-queue a session) + the orchestrator wiring."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_utils import run_meta_extras

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_deep_copy_and_json_safe():
    ss = {"add_random_seeds_to_gens": 2, "upscaling": {"enabled": True, "steps": [1, 2]}}
    out = run_meta_extras(ss, False)
    ss["upscaling"]["steps"].append(3)
    ss["add_random_seeds_to_gens"] = 9
    assert out["session_settings"]["upscaling"]["steps"] == [1, 2]
    assert out["session_settings"]["add_random_seeds_to_gens"] == 2
    json.dumps(out)


def test_non_dict_gives_empty_settings():
    assert run_meta_extras(None, False)["session_settings"] == {}
    assert run_meta_extras("x", False)["session_settings"] == {}


def test_has_optional_inputs_is_bool():
    assert run_meta_extras({}, True)["has_optional_inputs"] is True
    assert run_meta_extras({}, 0)["has_optional_inputs"] is False
    assert run_meta_extras({}, None)["has_optional_inputs"] is False


def test_unserialisable_values_are_stringified():
    out = run_meta_extras({"p": object()}, False)
    json.dumps(out)


def test_every_run_generation_loop_meta_block_records_extras():
    with open(os.path.join(_ROOT, "generation_orchestrator.py"), encoding="utf-8") as f:
        src = f.read()
    body = src[src.index("def run_generation_loop"):src.index("def _conditioning_to_serializable")]
    # the four exit paths all build meta through _standard_run_meta(..., _meta_run), which spreads the extras
    blocks = re.findall(r'existing_data\["meta"\] = _standard_run_meta\((.*?)\)\n', body, flags=re.S)
    assert len(blocks) == 4
    for b in blocks:
        assert b.rstrip().endswith("_meta_run")
    helper = src[src.index("def _standard_run_meta"):src.index("def _completion_payload")]
    assert "**(meta_run or {})" in helper
