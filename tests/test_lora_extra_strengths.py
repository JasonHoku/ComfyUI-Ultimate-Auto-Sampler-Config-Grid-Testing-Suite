"""Roadmap item 12: LoRA strength rows ("+" in the Builder) -> state key lora_extra_strengths.

Python half of the contract (config_builder_node.process_lora_array). The JS half
(normalizeExtraStrengths / loraEntryCount) is tested in tests/js/lora-extra-strengths.test.js
against the same fixture, tests/fixtures/lora_extra_strengths_cases.json.
"""
import copy
import json
import math
import os

import pytest

from config_builder_node import UltimateConfigBuilder
from config_utils import config_fingerprint, expand_configs
from tests.test_state_to_configs_json import make_state

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "lora_extra_strengths_cases.json")


def _state(array=None, **kw):
    array = dict(array or {})
    loras = array.pop("loras", ["None"])
    bypass = array.pop("lora_bypass_states", None)
    weight = array.pop("lora_weight_arrays", None)
    return make_state(loras=loras, lora_bypass_states=bypass, lora_weight_arrays=weight,
                      extra_array_fields=array or None, **kw)


def _configs(state):
    return json.loads(UltimateConfigBuilder.state_to_configs_json(state))["configs"]


def _loras(state):
    """The emitted lora list of the (single) config object, always as a list."""
    configs = _configs(state)
    assert len(configs) == 1, configs
    lora = configs[0]["lora"]
    return lora if isinstance(lora, list) else [lora]


def _rows(*pairs):
    return [{"model": m, "clip": c} for m, c in pairs]


# 1 -----------------------------------------------------------------------------
def test_locked_one_lora_two_rows():
    state = _state({"loras": ["a.safetensors:1.00:1.00"],
                    "lora_extra_strengths": {"a.safetensors": _rows((0.6, 0.6), (0.8, 0.8))}})
    assert _loras(state) == ["a.safetensors:1.00:1.00", "a.safetensors:0.60:0.60", "a.safetensors:0.80:0.80"]


# 2 -----------------------------------------------------------------------------
def test_unlocked_rows_stay_pairs():
    state = _state({"loras": ["a.safetensors:1.00:1.00"],
                    "lora_strength_lock": {"a.safetensors": False},
                    "lora_extra_strengths": {"a.safetensors": _rows((0.6, 0.4))}})
    out = _loras(state)
    assert out == ["a.safetensors:1.00:1.00", "a.safetensors:0.60:0.40"]
    assert len(out) == 2  # base + 1, not a model x clip product


# 3 -----------------------------------------------------------------------------
def test_locked_forces_clip_to_model():
    state = _state({"loras": ["a.safetensors:1.00:1.00"],
                    "lora_extra_strengths": {"a.safetensors": _rows((0.6, 0.1))}})
    assert _loras(state)[1] == "a.safetensors:0.60:0.60"
    # missing lock key == locked (the UI default)
    state2 = _state({"loras": ["a.safetensors:1.00:1.00"], "lora_strength_lock": {"other": False},
                     "lora_extra_strengths": {"a.safetensors": _rows((0.6, 0.1))}})
    assert _loras(state2)[1] == "a.safetensors:0.60:0.60"


# 4 -----------------------------------------------------------------------------
def test_two_stacked_loras_cross_product_first_slowest():
    state = _state({"loras": ["a.safetensors:1.00:1.00", "b.safetensors:0.50:0.50"],
                    "lora_extra_strengths": {"a.safetensors": _rows((0.6, 0.6)),
                                             "b.safetensors": _rows((0.8, 0.8), (0.9, 0.9))}})
    out = _loras(state)
    assert len(out) == 6
    assert out[0] == "a.safetensors:1.00:1.00 + b.safetensors:0.50:0.50"
    assert out[1] == "a.safetensors:1.00:1.00 + b.safetensors:0.80:0.80"
    assert out[3] == "a.safetensors:0.60:0.60 + b.safetensors:0.50:0.50"
    assert out[5] == "a.safetensors:0.60:0.60 + b.safetensors:0.90:0.90"


# 5 -----------------------------------------------------------------------------
def test_folder_entry_is_never_stacked_with_a_file_lora():
    state = _state({"loras": ["foo/", "a.safetensors:1.00:1.00"],
                    "lora_extra_strengths": {"foo/": _rows((0.6, 0.6)),
                                             "a.safetensors": _rows((0.7, 0.7))}})
    assert _loras(state) == ["foo/", "foo/:0.60:0.60", "a.safetensors:1.00:1.00", "a.safetensors:0.70:0.70"]


# 6 -----------------------------------------------------------------------------
def test_duplicates_of_base_and_earlier_rows_are_dropped():
    state = _state({"loras": ["a.safetensors:1.00:1.00"],
                    "lora_extra_strengths": {"a.safetensors": _rows((1.0, 1.0), (0.6, 0.6), (0.604, 0.6), (0.8, 0.8))}})
    assert _loras(state) == ["a.safetensors:1.00:1.00", "a.safetensors:0.60:0.60", "a.safetensors:0.80:0.80"]


# 7 -----------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [
    "not-a-list", None, 5, {"model": 1}, [None, "abc", {"model": "abc"}, {"model": None},
                                           {"model": float("nan")}, {"model": float("inf")},
                                           {"model": 0.5, "clip": float("-inf")}, {"model": True}, []],
])
def test_invalid_rows_do_not_raise_and_are_ignored(raw):
    state = _state({"loras": ["a.safetensors:1.00:1.00"], "lora_extra_strengths": {"a.safetensors": raw}})
    assert _loras(state) == ["a.safetensors:1.00:1.00"]


def test_invalid_map_shapes_do_not_raise():
    for bad in ("x", [1, 2], 7):
        state = _state({"loras": ["a.safetensors:1.00:1.00"], "lora_extra_strengths": bad})
        assert _loras(state) == ["a.safetensors:1.00:1.00"]
    # leniently accepted: [m, c] pairs and a row without clip
    state = _state({"loras": ["a.safetensors:1.00:1.00"], "lora_strength_lock": {"a.safetensors": False},
                    "lora_extra_strengths": {"a.safetensors": [[0.6, 0.4], {"model": 0.7}]}})
    assert _loras(state) == ["a.safetensors:1.00:1.00", "a.safetensors:0.60:0.40", "a.safetensors:0.70:0.70"]


# 8 -----------------------------------------------------------------------------
def test_cap_is_eight_strengths_in_total():
    extras = _rows(*[(round(0.1 * i, 2), round(0.1 * i, 2)) for i in range(1, 11)])  # 0.1 .. 1.0 (1.0 == base)
    state = _state({"loras": ["a.safetensors:1.00:1.00"], "lora_extra_strengths": {"a.safetensors": extras}})
    out = _loras(state)
    assert len(out) == 8
    assert out[0] == "a.safetensors:1.00:1.00"
    assert out[-1] == "a.safetensors:0.70:0.70"
    # the helper itself
    cleaned = UltimateConfigBuilder._clean_extra_strengths(extras, True, 1.0, 1.0)
    assert len(cleaned) == 7


# 9 -----------------------------------------------------------------------------
def test_bypassed_lora_extras_are_ignored():
    state = _state({"loras": ["a.safetensors:1.00:1.00", "b.safetensors:0.50:0.50"],
                    "lora_bypass_states": {"a.safetensors": True},
                    "lora_extra_strengths": {"a.safetensors": _rows((0.6, 0.6))}})
    assert _loras(state) == ["b.safetensors:0.50:0.50"]


# 10 ----------------------------------------------------------------------------
@pytest.mark.parametrize("loras,weight", [
    (["myLora:0.80:0.50"], None),
    (["a.safetensors:1.00:1.00", "b.safetensors:0.50:0.50"], None),
    (["foo/", "a.safetensors:0.70:0.70"], None),
    (["Sexy-IL-v11/:1.00:1.00"], {"Sexy-IL-v11/_model": [0, 1, 2, 5, 10]}),
    (["a.safetensors:1.00:1.00"], {"a.safetensors_model": [0.5, 0.8], "a.safetensors_clip": [0.4, 0.9]}),
])
def test_states_without_the_key_are_byte_identical(loras, weight):
    base = make_state(loras=list(loras), lora_weight_arrays=copy.deepcopy(weight))
    with_empty = make_state(loras=list(loras), lora_weight_arrays=copy.deepcopy(weight),
                            extra_array_fields={"lora_extra_strengths": {}})
    with_unrelated = make_state(loras=list(loras), lora_weight_arrays=copy.deepcopy(weight),
                                extra_array_fields={"lora_extra_strengths": {"nobody.safetensors": _rows((0.6, 0.6))}})
    expected = UltimateConfigBuilder.state_to_configs_json(base)
    assert UltimateConfigBuilder.state_to_configs_json(with_empty) == expected
    assert UltimateConfigBuilder.state_to_configs_json(with_unrelated) == expected


# 11 ----------------------------------------------------------------------------
def test_new_rows_win_over_old_weight_arrays_without_brackets():
    state = _state({"loras": ["a.safetensors:1.00:1.00"],
                    "lora_weight_arrays": {"a.safetensors_model": [0.1, 0.2, 0.3]},
                    "lora_extra_strengths": {"a.safetensors": _rows((0.6, 0.6))}})
    out = _loras(state)
    assert out == ["a.safetensors:1.00:1.00", "a.safetensors:0.60:0.60"]
    assert not any("[" in s for s in out)
    # an EMPTY new entry leaves the old arrays in charge
    state = _state({"loras": ["a.safetensors:1.00:1.00"],
                    "lora_weight_arrays": {"a.safetensors_model": [0.1, 0.2, 0.3]},
                    "lora_extra_strengths": {"a.safetensors": []}})
    assert _loras(state) == ["a.safetensors:[0.1, 0.2, 0.3]"]


# 12 ----------------------------------------------------------------------------
def test_expands_to_n_times_the_other_axes_and_keeps_base_fingerprint():
    array = {"loras": ["a.safetensors:1.00:1.00"],
             "lora_extra_strengths": {"a.safetensors": _rows((0.6, 0.6), (0.8, 0.8))}}
    with_rows = _configs(_state(array, samplers=["euler", "dpmpp_2m"]))
    without = _configs(_state({"loras": ["a.safetensors:1.00:1.00"]}, samplers=["euler", "dpmpp_2m"]))

    def expand(configs):
        return expand_configs(configs, [""], [""], [1.0], 0, [], None)

    jobs, base_jobs = expand(with_rows), expand(without)
    assert len(base_jobs) == 2
    assert len(jobs) == 2 * 3  # samplers x (base + 2 rows)
    emitted = sorted({j["lora"] for j in jobs})
    assert emitted == sorted(["a.safetensors:1.00:1.00", "a.safetensors:0.60:0.60", "a.safetensors:0.80:0.80"])
    # only the lora differs between the copies of one sampler's job
    for sampler in ("euler", "dpmpp_2m"):
        same = [{k: v for k, v in j.items() if k not in ("lora", "lora_expanded")} for j in jobs if j["sampler"] == sampler]
        assert all(s == same[0] for s in same)
    # the base job keeps its identity, so a resumed run still skips it
    fp = lambda j: config_fingerprint(j, 512, 512, seed=0)
    base_fps = {fp(j) for j in base_jobs}
    base_in_new = {fp(j) for j in jobs if j["lora"] == "a.safetensors:1.00:1.00"}
    assert base_fps == base_in_new
    assert len({fp(j) for j in jobs}) == len(jobs)


# 13 ----------------------------------------------------------------------------
def _cases():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_shared_fixture_cases(case):
    assert _loras(_state(copy.deepcopy(case["array"]))) == case["expected"]


# Existing bug (plan item 7), NOT fixed by this feature: with >= 2 stackable LoRAs the combined
# stack replaces the list, so a folder entry ("name/") in the same array is silently dropped.
@pytest.mark.xfail(reason="folder entries are dropped when 2+ stackable LoRAs combine (existing behaviour)")
def test_folder_entry_survives_alongside_a_stack():
    state = _state({"loras": ["foo/", "a.safetensors:1.00:1.00", "b.safetensors:0.50:0.50"]})
    out = _loras(state)
    assert "foo/" in out
    assert "a.safetensors:1.00:1.00 + b.safetensors:0.50:0.50" in out


def test_base_strengths_mirror_parse_lora_definition():
    f = UltimateConfigBuilder._base_strengths
    assert f("a") == (1.0, 1.0)
    assert f("a:0.5") == (0.5, 0.5)          # missing CLIP = model
    assert f("a:0.5:0.25") == (0.5, 0.25)
    assert f("a:[0.5, 0.8]") is None
    assert f("a:abc") is None
    assert all(math.isfinite(x) for x in f("a:2:3"))
