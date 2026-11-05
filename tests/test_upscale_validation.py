"""Server-side upscale config validation: shared fixture with tests/js/upscale_edit.test.js."""
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from tests.upscale_runner_loader import load_runner  # noqa: E402

with open(os.path.join(_ROOT, "tests", "fixtures", "upscale_validation_cases.json"), encoding="utf-8") as _f:
    CASES = json.load(_f)

ur = load_runner()


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_validation_cases(case):
    errors, _warnings = ur._validate_upscale(case["cfg"])
    assert (not errors) == case["expect_ok"], errors
    assert sorted({e["field"] for e in errors}) == sorted(set(case["error_fields"]))
    # the string form used by the route
    msgs = ur.validate_upscale_config(case["cfg"])
    assert (not msgs) == case["expect_ok"]


def test_error_paths_point_at_the_step():
    cfg = {"pipelines": [{"steps": [{}]}, {"steps": [{}, {"hires_denoise": "2"}]}]}
    errors, _ = ur._validate_upscale(cfg)
    assert [e["path"] for e in errors] == ["p1.s1"]
    assert ur.validate_upscale_config(cfg)[0].startswith("p1.s1:")


def test_warnings_not_errors():
    cfg = {"pipelines": [{"steps": [{"mode": "florence2_hires", "hires_denoise": "0.4,0.5",
                                     "florence2": {"text_input": "face"}}]}],
           "hires_prompt_adjust": True, "hires_prompt_text": ""}
    errors, warnings = ur._validate_upscale(cfg)
    assert errors == []
    assert {w["field"] for w in warnings} == {"hires_denoise", "hires_prompt_text"}


def test_garbage_config_does_not_raise():
    for bad in (None, [], "x", {"pipelines": "nope"}, {"pipelines": [None, 3, {"steps": "x"}]}):
        assert ur.validate_upscale_config(bad)
