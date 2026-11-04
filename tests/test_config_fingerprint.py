import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_utils import (config_fingerprint, item_fingerprint, build_preview_index,
                          should_skip_for_previews, build_full_fp_set)


def _conf(**kw):
    c = {"model": "m.safetensors", "model_type": "checkpoint", "lora": "None", "lora_expanded": "None",
         "vae": "Default", "text_encoders": [], "clip_type": "stable_diffusion", "clip_skip": 0,
         "attention_mode": "default", "sampler": "euler", "scheduler": "normal", "cfg": 7.0,
         "denoise": 1.0, "seed": 42, "positive": "a cat", "negative": "bad", "steps": 30}
    c.update(kw)
    return c


def fp(conf=None, w=1024, h=1024, **kw):
    return config_fingerprint(conf or _conf(), w, h, **kw)


def test_stable_across_key_order_and_numeric_spelling():
    a = _conf()
    b = dict(reversed(list(_conf().items())))
    assert fp(a) == fp(b)
    assert fp(_conf(cfg="7")) == fp(_conf(cfg=7.0))
    assert fp(_conf(positive="  a cat \n")) == fp(_conf())


def test_steps_and_preview_keys_are_ignored():
    base = fp()
    assert fp(_conf(steps=8)) == base
    assert fp(_conf(preview=True, preview_of="x", full_steps=30, full_cfg=7, full_width=1, full_height=2,
                    preview_scale=0.5, id=5, file="f", gen_index=3, batch_idx=1, favorited=True,
                    rejected=True, note="n", duration=1.2, seed_behavior="randomize")) == base


@pytest.mark.parametrize("change", [
    {"model": "other"}, {"lora_expanded": "l1:1.0"}, {"vae": "v.safetensors"},
    {"text_encoders": ["t5"]}, {"clip_type": "flux"}, {"clip_skip": 2}, {"attention_mode": "sage"},
    {"sampler": "dpmpp_2m"}, {"scheduler": "karras"}, {"cfg": 5.0}, {"denoise": 0.5},
    {"positive": "a dog"}, {"negative": "worse"}, {"flux_guidance_value": 3.5},
    {"use_slg": True, "slg_single_layers": "7"}, {"model_type": "gguf"},
])
def test_each_identity_field_changes_fp(change):
    assert fp(_conf(**change)) != fp()


def test_seed_size_and_pipeline_change_fp():
    base = fp()
    assert fp(seed=43) != base
    assert fp(w=832) != base
    assert fp(h=512) != base
    assert fp(upscale_pipeline="hires") != base
    assert fp(upscale_pipeline="hires") != fp(upscale_pipeline="other")


def test_vae_spellings_equal():
    assert fp(_conf(vae=None)) == fp(_conf(vae="")) == fp(_conf(vae="Default")) == fp(_conf(vae="None"))


def test_lora_expanded_preferred_over_lora():
    assert fp(_conf(lora="[folder]", lora_expanded="a:1")) == fp(_conf(lora="a:1", lora_expanded="a:1"))


def test_config_positive_wins_over_trigger_word_positive():
    assert fp(_conf(positive="trig, a cat", config_positive="a cat")) == fp()


def test_item_fingerprint_stored_preview_and_legacy():
    full_item = dict(_conf(), width=1024, height=1024)
    legacy = item_fingerprint(full_item)
    assert legacy == fp()  # legacy item (no config_fp) == fingerprint of the conf
    stored = dict(full_item, config_fp=fp())
    assert item_fingerprint(stored) == fp()
    preview_item = dict(_conf(steps=8, cfg=3.0), width=512, height=512, preview=True,
                        full_width=1024, full_height=1024, full_steps=30, full_cfg=7.0)
    assert item_fingerprint(preview_item) == fp()


def test_golden_value_pins_encoding():
    assert fp() == GOLDEN


GOLDEN = "fp1:dc9d12e35d698e0c"


def _p(fp_, seed=1, rejected=False, favorited=False):
    return {"preview": True, "preview_of": fp_, "seed": seed, "rejected": rejected, "favorited": favorited}


def test_preview_index_rules():
    items = [_p("a", seed=9), _p("a", seed=1, rejected=True),
             _p("b", rejected=True), _p("b", favorited=True),
             _p("c"), {"config_fp": "z", "rejected": True}]
    idx = build_preview_index(items)
    assert idx["a"] == {"rejected": True, "favorited": False, "seed": 9}   # newest preview's seed
    assert idx["b"]["favorited"] and not idx["b"]["rejected"]               # favorite beats reject
    assert idx["c"] == {"rejected": False, "favorited": False, "seed": 1}
    assert "z" not in idx                                                   # full items are not indexed


def test_preview_index_tombstones():
    idx = build_preview_index([_p("a"), _p("f", favorited=True)], tombstones=["a", "gone", "f"])
    assert idx["a"]["rejected"]
    assert idx["gone"]["rejected"]
    assert not idx["f"]["rejected"]


@pytest.mark.parametrize("state,skip_rej,only_fav,expected", [
    ("rejected", False, False, None), ("rejected", True, False, "rejected_preview"),
    ("rejected", False, True, "no_favorited_preview"), ("rejected", True, True, "no_favorited_preview"),
    ("favorited", False, False, None), ("favorited", True, False, None),
    ("favorited", False, True, None), ("favorited", True, True, None),
    ("plain", False, False, None), ("plain", True, False, None),
    ("plain", False, True, "no_favorited_preview"), ("plain", True, True, "no_favorited_preview"),
    ("absent", False, False, None), ("absent", True, False, None),
    ("absent", False, True, None), ("absent", True, True, None),
])
def test_should_skip_matrix(state, skip_rej, only_fav, expected):
    items = {"rejected": [_p("a", rejected=True)], "favorited": [_p("a", favorited=True)],
             "plain": [_p("a")], "absent": []}[state]
    idx = build_preview_index(items)
    assert should_skip_for_previews("a", idx, skip_rej, only_fav) == expected


def test_full_fp_set_ignores_previews_and_includes_base_fp():
    items = [dict(_conf(), width=1024, height=1024, config_fp="F1"),
             {"preview": True, "config_fp": "P1", "preview_of": "F2"},
             {"config_fp": "U1", "base_fp": "F3"}]
    s = build_full_fp_set(items)
    assert {"F1", "U1", "F3"} <= s
    assert "P1" not in s and "F2" not in s
