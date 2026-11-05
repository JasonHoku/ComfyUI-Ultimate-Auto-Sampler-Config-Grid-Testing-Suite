"""conditioning_cache v2: keeps every entry and every extra key (AND prompts, attention masks, region tags)."""
import importlib.util
import json
import os
import sys

import pytest

pytestmark = pytest.mark.requires_torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)
sys.path.insert(0, _ROOT)

torch = pytest.importorskip("torch")
_spec = importlib.util.spec_from_file_location(f"{_PKG}.conditioning_cache_real",
                                               os.path.join(_ROOT, "conditioning_cache.py"))
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

TAG = {"v": 1, "index": 1, "count": 2, "axis": "cols", "weight": 1.3}


def _cond():
    return [
        [torch.randn(1, 5, 8), {"pooled_output": torch.randn(1, 4)}],
        [torch.randn(1, 3, 8), {"pooled_output": None, "attention_mask": torch.ones(1, 3, dtype=torch.int64),
                                "uscg_region": dict(TAG)}],
        [torch.randn(1, 7, 8), {"pooled_output": torch.randn(1, 4), "uscg_region": dict(TAG, index=0)}],
    ]


def _eq(a, b):
    assert len(a) == len(b)
    for (ta, da), (tb, db) in zip(a, b):
        assert torch.equal(ta, tb) and ta.dtype == tb.dtype
        assert set(da) == set(db)
        for k in da:
            if torch.is_tensor(da[k]):
                assert torch.equal(da[k], db[k]) and da[k].dtype == db[k].dtype
            else:
                assert da[k] == db[k]


def test_round_trip_three_entries_through_disk(tmp_path):
    c = cc.ConditioningCache(str(tmp_path), "h")
    cond = _cond()
    c.set("p COUPLE a | b", cond)
    c.save()
    on_disk = json.loads((tmp_path / "conditioning_cache.json").read_text())
    assert all("conditioning_v2" in e and "conditioning" not in e for e in on_disk["entries"].values())
    c2 = cc.ConditioningCache(str(tmp_path), "h")
    got = c2.get("p COUPLE a | b")
    assert got is not None and got is not cond
    _eq(got, cond)
    assert c2.stats["hits"] == 1


def test_legacy_only_entry_is_a_miss(tmp_path):
    c = cc.ConditioningCache(str(tmp_path), "h")
    legacy = c._conditioning_to_dict([[torch.randn(1, 5, 8), {"pooled_output": torch.randn(1, 4)}]])
    c.disk_cache["entries"][c._prompt_key("old")] = {"prompt": "old", "type": "positive", "conditioning": legacy}
    c.save()
    c2 = cc.ConditioningCache(str(tmp_path), "h")
    assert c2.get("old") is None
    assert c2.stats["misses"] == 1


def test_unserializable_value_is_not_written_but_stays_in_memory(tmp_path):
    c = cc.ConditioningCache(str(tmp_path), "h")
    bad = [[torch.randn(1, 2, 3), {"pooled_output": None, "weird": object()}]]
    c.set("bad", bad)
    assert c.get("bad") is bad  # memory cache
    assert c._prompt_key("bad") not in c.disk_cache["entries"]
    c.save()
    assert cc.ConditioningCache(str(tmp_path), "h").get("bad") is None


def test_bfloat16_is_not_written(tmp_path):
    c = cc.ConditioningCache(str(tmp_path), "h")
    c.set("bf", [[torch.randn(1, 2, 3).to(torch.bfloat16), {"pooled_output": None}]])
    assert c._prompt_key("bf") not in c.disk_cache["entries"]


def test_disk_cache_disabled_keeps_memory_only(tmp_path):
    c = cc.ConditioningCache(str(tmp_path), "h", enable_disk_cache=False)
    cond = _cond()
    c.set("m", cond)
    assert c.get("m") is cond
    assert c.disk_cache["entries"] == {}
