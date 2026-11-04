"""check_if_job_completed(preview=...) and tombstone collection."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generation_orchestrator as go  # preloaded by tests/conftest.py
from config_utils import collect_rejected_preview_fps


def _conf(**kw):
    c = {"model": "m", "lora_expanded": "None", "sampler": "euler", "scheduler": "normal",
         "steps": 30, "cfg": 7.0, "denoise": 1.0, "seed": 5}
    c.update(kw)
    return c


def _item(**kw):
    i = {"seed": 5, "width": 1024, "height": 1024, "batch_idx": 0, "sampler": "euler",
         "scheduler": "normal", "steps": 30, "cfg": 7.0, "denoise": 1.0, "model": "m",
         "positive": "p", "negative": "n", "lora": "None"}
    i.update(kw)
    return i


def _preview_item(**kw):
    return _item(preview=True, steps=8, width=512, height=512, full_width=1024, full_height=1024,
                 full_steps=30, full_cfg=7.0, **kw)


def _call(items, conf, w=1024, h=1024, **kw):
    return go.check_if_job_completed(items, conf, conf["seed"], w, h, 0, "p", "n", **kw)


def test_full_job_matches_full_item_not_preview():
    assert _call([_item()], _conf()) == 0
    assert _call([_preview_item()], _conf()) == -1


def test_preview_job_does_not_match_full_item():
    assert _call([_item()], _conf(steps=8), preview=True) == -1


def test_preview_job_matches_preview_item_on_full_dims_and_preview_steps():
    items = [_preview_item()]
    assert _call(items, _conf(steps=8), preview=True) == 0
    assert _call(items, _conf(steps=12), preview=True) == -1          # different preview steps
    assert _call(items, _conf(steps=8), w=832, preview=True) == -1    # different FULL width
    assert _call(items, _conf(steps=8), w=512, h=512, preview=True) == -1  # snapped dims are not the key


def test_legacy_items_without_preview_key_count_as_full():
    legacy = _item()
    assert "preview" not in legacy
    assert _call([legacy], _conf()) == 0


def test_positional_distributed_style_call_still_works():
    conf = _conf()
    assert go.check_if_job_completed([_item()], conf, 5, 1024, 1024, 0, "p", "n", False) == 0
    assert go.check_if_job_completed([_item()], conf, 5, 1024, 1024, 0, "p", "n") == 0
    assert go.check_if_job_completed([_item()], conf, 5, 1024, 1024, 0, "p", "n", True) == -1  # optional inputs


def test_preview_is_keyword_only():
    import pytest
    with pytest.raises(TypeError):
        go.check_if_job_completed([_item()], _conf(), 5, 1024, 1024, 0, "p", "n", False, True)


def test_collect_rejected_preview_fps():
    rej = [{"preview": True, "preview_of": "a"}, {"preview": True, "preview_of": "b"},
           {"rejected": True, "config_fp": "full"}, {"preview": True, "preview_of": "a"}]
    assert collect_rejected_preview_fps(rej, ["z", "b"]) == ["z", "b", "a"]
    assert collect_rejected_preview_fps([], None) == []
