"""The distributed path must keep tombstones / job order / run-API settings in manifest meta."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generation_orchestrator as go  # preloaded by tests/conftest.py
from config_utils import run_meta_extras


def _meta(existing, **kw):
    args = dict(positive_text="p", negative_text="n", ckpt_name="m.safetensors", seed=7, vae_batch_size=4,
                configs_json="[]", resolutions_json="[]", image_format="webp", job_order="minimize_reloads",
                preview_on=True, pv_steps=6, pv_scale=0.5, pv_cfg=None,
                meta_run=run_meta_extras({"job_order": "minimize_reloads"}, False))
    args.update(kw)
    return go._distributed_run_meta(existing, **args)


def test_tombstones_survive():
    m = _meta({"meta": {"rejected_preview_fps": ["a", "b"]}, "items": []})
    assert m["rejected_preview_fps"] == ["a", "b"]


def test_no_previous_meta():
    assert _meta({"items": []})["rejected_preview_fps"] == []


def test_job_order_preview_and_session_settings_kept():
    m = _meta({"meta": None})
    assert m["job_order"] == "minimize_reloads"
    assert m["preview_pass"] == {"enabled": True, "steps": 6, "scale": 0.5, "cfg_override": None}
    assert m["session_settings"] == {"job_order": "minimize_reloads"}
    assert m["has_optional_inputs"] is False
    assert m["seed"] == 7 and m["image_format"] == "webp"


def test_same_keys_as_normal_loop_helper():
    """The normal loop (_standard_run_meta, shared by its 4 exit paths) and the distributed helper write the same keys."""
    args = dict(positive_text="p", negative_text="n", ckpt_name="m", seed=1, vae_batch_size=1, configs_json="[]",
                resolutions_json="[]", image_format="webp", job_order="as_written", preview_on=False, pv_steps=8,
                pv_scale=0.5, pv_cfg=None, meta_run=run_meta_extras({}, False))
    normal = go._standard_run_meta({"items": []}, **args)
    assert set(normal) == set(_meta({"items": []}))
