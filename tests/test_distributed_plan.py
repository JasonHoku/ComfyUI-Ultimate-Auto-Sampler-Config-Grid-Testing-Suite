"""_distributed_plan_entries: the master decides order / skips / fingerprints for distributed runs,
with the same rules as the normal generation loop."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generation_orchestrator as go  # preloaded by tests/conftest.py
from config_utils import (
    build_job_plan, order_job_plan, config_fingerprint, build_preview_index, build_full_fp_set,
    should_skip_for_previews,
)


@pytest.fixture(autouse=True)
def _prompts(monkeypatch):
    monkeypatch.setattr(go, "build_prompt_with_triggers", lambda conf, mode: (conf["positive"], ""))


def _conf(**kw):
    c = {"model": "m", "lora": "None", "lora_expanded": "None", "sampler": "euler", "scheduler": "normal",
         "steps": 30, "cfg": 7.0, "denoise": 1.0, "seed": 5, "positive": "p", "negative": "n"}
    c.update(kw)
    return c


def _job(w=1024, h=1024):
    return {"label": f"{w}x{h}", "width": w, "height": h, "batch_idx": 0}


def _plan(confs, w=1024, h=1024):
    return build_job_plan(confs, [_job(w, h)])


def _full_item(conf, **kw):
    i = {"seed": conf["seed"], "width": 1024, "height": 1024, "batch_idx": 0, "sampler": conf["sampler"],
         "scheduler": conf["scheduler"], "steps": conf["steps"], "cfg": conf["cfg"], "denoise": 1.0,
         "model": conf["model"], "positive": conf["positive"], "negative": conf["negative"], "lora": "None"}
    i.update(kw)
    return i


def _preview_item(conf, **kw):
    fp = config_fingerprint(conf, 1024, 1024, conf["seed"])
    i = _full_item(conf, preview=True, steps=8, width=512, height=512, full_width=1024, full_height=1024,
                   full_steps=conf["steps"], full_cfg=conf["cfg"], config_fp=fp, preview_of=fp)
    i.update(kw)
    return i


PV = {"steps": 8, "scale": 0.5, "cfg": None}


def _run(plan, items=(), meta=None, **kw):
    kw.setdefault("overwrite_existing", False)
    kw.setdefault("has_optional_inputs", False)
    kw.setdefault("lora_triggerwords_mode", "off")
    return go._distributed_plan_entries(plan, {"items": list(items), "meta": meta or {}}, **kw)


def test_keeps_plan_order_and_input_job_shape():
    confs = [_conf(seed=1, positive="a"), _conf(seed=2, positive="b"), _conf(seed=3, positive="c")]
    plan = order_job_plan(_plan(confs), "as_written", go.get_model_cache_key)
    plan = list(reversed(plan))  # an arbitrary already-ordered plan
    entries, counts = _run(plan)
    assert [e["config"]["positive"] for e in entries] == ["c", "b", "a"]
    assert entries[0]["input_job"] == {"label": "1024x1024", "width": 1024, "height": 1024, "batch_idx": 0}
    assert counts == {"exists": 0, "promotion": 0, "full_exists": 0}


def test_skips_existing_full_item():
    c1, c2 = _conf(seed=1), _conf(seed=2)
    entries, counts = _run(_plan([c1, c2]), [_full_item(c1)])
    assert [e["config"]["seed"] for e in entries] == [2]
    assert counts["exists"] == 1


def test_overwrite_existing_renders_everything():
    c1 = _conf(seed=1)
    entries, _ = _run(_plan([c1]), [_full_item(c1)], overwrite_existing=True)
    assert len(entries) == 1


def test_optional_inputs_disable_skip_detection():
    c1 = _conf(seed=1)
    entries, _ = _run(_plan([c1]), [_full_item(c1)], has_optional_inputs=True)
    assert len(entries) == 1


def test_preview_run_skips_existing_preview_and_uses_effective_conf():
    c1, c2 = _conf(seed=1), _conf(seed=2)
    entries, counts = _run(_plan([c1, c2]), [_preview_item(c1)], preview=PV)
    assert [e["config"]["seed"] for e in entries] == [2]
    assert entries[0]["config"]["steps"] == 8        # preview steps
    assert entries[0]["config"]["cfg"] == 7.0         # cfg kept when no override
    assert counts["exists"] == 1


def test_preview_cfg_override():
    entries, _ = _run(_plan([_conf()]), preview={"steps": 6, "scale": 0.5, "cfg": 3.5})
    assert entries[0]["config"]["steps"] == 6 and entries[0]["config"]["cfg"] == 3.5
    assert entries[0]["extra"]["stamp"]["full_cfg"] == 7.0 and entries[0]["extra"]["stamp"]["full_steps"] == 30


def test_preview_run_skips_cell_with_full_item_unless_overwrite():
    c1 = _conf(seed=1)
    entries, counts = _run(_plan([c1]), [_full_item(c1)], preview=PV)
    assert entries == [] and counts["full_exists"] == 1
    entries, _ = _run(_plan([c1]), [_full_item(c1)], preview=PV, overwrite_existing=True)
    assert len(entries) == 1


def test_full_run_skips_rejected_preview_and_tombstone():
    c1, c2, c3 = _conf(seed=1), _conf(seed=2), _conf(seed=3)
    items = [_preview_item(c1, rejected=True)]
    tomb = config_fingerprint(c2, 1024, 1024, 2)
    entries, counts = _run(_plan([c1, c2, c3]), items, meta={"rejected_preview_fps": [tomb]})
    assert [e["config"]["seed"] for e in entries] == [3]
    assert counts["promotion"] == 2


def test_overwrite_does_not_override_a_rejection():
    c1 = _conf(seed=1)
    entries, counts = _run(_plan([c1]), [_preview_item(c1, rejected=True)], overwrite_existing=True)
    assert entries == [] and counts["promotion"] == 1


def test_favorite_clears_a_reject():
    c1 = _conf(seed=1)
    items = [_preview_item(c1, rejected=True), _preview_item(c1, favorited=True)]
    entries, _ = _run(_plan([c1]), items)
    assert len(entries) == 1


def test_only_favorited_previews():
    c1, c2, c3 = _conf(seed=1), _conf(seed=2), _conf(seed=3)
    items = [_preview_item(c1, favorited=True), _preview_item(c2)]
    entries, counts = _run(_plan([c1, c2, c3]), items, only_favorited=True)
    # c3 has no preview at all: nothing was previewed to judge, so it still renders
    assert [e["config"]["seed"] for e in entries] == [1, 3]
    assert counts["promotion"] == 1
    entries, _ = _run(_plan([c1]), [_preview_item(c1)], only_favorited=True, overwrite_existing=True)
    assert entries == []


def test_skip_rejected_can_be_turned_off():
    c1 = _conf(seed=1)
    entries, _ = _run(_plan([c1]), [_preview_item(c1, rejected=True)], skip_rejected=False)
    assert len(entries) == 1


def test_stamp_matches_the_normal_loop_fields():
    c = _conf(seed=5, seed_behavior="randomize")
    entries, _ = _run(_plan([c]), preview={"steps": 8, "scale": 0.5, "cfg": None})
    fp = config_fingerprint(c, 1024, 1024, 5)  # configured seed, FULL resolution (not the rolled one)
    assert entries[0]["extra"]["stamp"] == {
        "config_fp": fp, "preview": True, "preview_of": fp, "full_steps": 30, "full_cfg": 7.0,
        "full_width": 1024, "full_height": 1024, "preview_scale": 0.5,
    }
    assert entries[0]["extra"]["render"] == {"preview_scale": 0.5}
    assert "seed" not in entries[0]["extra"]  # preview rolls its own seed on the renderer


def test_full_stamp_is_only_the_fingerprint():
    entries, _ = _run(_plan([_conf()]))
    assert set(entries[0]["extra"]["stamp"]) == {"config_fp"}
    assert "render" not in entries[0]["extra"]


def test_survivor_seed_for_randomize_full_pass():
    c = _conf(seed=5, seed_behavior="randomize")
    survivor = _preview_item(c, seed=424242, favorited=True)
    entries, _ = _run(_plan([c]), [survivor])
    assert entries[0]["extra"]["seed"] == 424242
    # no survivor -> no pinned seed (the renderer rolls one)
    entries, _ = _run(_plan([c]))
    assert "seed" not in entries[0]["extra"]


def test_ltx_gets_no_fingerprint_stamp_and_no_preview_override():
    c = _conf(model_type="ltx_video")
    entries, _ = _run(_plan([c]), preview=PV)
    assert "stamp" not in entries[0]["extra"] and "render" not in entries[0]["extra"]
    assert entries[0]["config"]["steps"] == 30


def test_optional_inputs_still_stamp_but_never_promote():
    c1 = _conf(seed=1)
    entries, _ = _run(_plan([c1]), [_preview_item(c1, rejected=True)], has_optional_inputs=True)
    assert len(entries) == 1
    assert entries[0]["extra"]["stamp"]["config_fp"] == config_fingerprint(c1, 1024, 1024, 1)


def test_per_config_resolution_is_used():
    c = _conf(resolution=[768, 512])
    entries, _ = _run(_plan([c]))
    assert (entries[0]["input_job"]["width"], entries[0]["input_job"]["height"]) == (768, 512)
    assert entries[0]["extra"]["stamp"]["config_fp"] == config_fingerprint(c, 768, 512, 5)


# -- parity with the normal loop's decision functions --------------------------------------

def _loop_keep(plan, items, meta, *, preview, overwrite, skip_rej, only_fav):
    """The keep/skip decision exactly as the normal loop composes it
    (_promotion_status -> check_if_job_completed), written independently of the helper."""
    pindex = build_preview_index(items, meta.get("rejected_preview_fps", []))
    full_fps = build_full_fp_set(items) if preview else set()
    kept = []
    for pj in plan:
        c = pj["conf"]
        fp = config_fingerprint(c, pj["w"], pj["h"], c["seed"])
        if preview:
            why = "full_exists" if fp in full_fps else None
        else:
            why = should_skip_for_previews(fp, pindex, skip_rej, only_fav)
        if why and not (why == "full_exists" and overwrite):
            continue
        eff = {**c, "steps": preview["steps"], "cfg": c["cfg"]} if preview else c
        m = go.check_if_job_completed(items, eff, c["seed"], pj["w"], pj["h"], pj["batch_idx"],
                                      c["positive"], c["negative"], preview=bool(preview))
        if m != -1 and not overwrite:
            continue
        kept.append(c["seed"])
    return kept


@pytest.mark.parametrize("preview", [None, PV])
@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("only_fav", [False, True])
def test_parity_with_normal_loop_decisions(preview, overwrite, only_fav):
    confs = [_conf(seed=s) for s in range(1, 8)]
    items = [
        _full_item(confs[0]),
        _preview_item(confs[1]),
        _preview_item(confs[2], rejected=True),
        _preview_item(confs[3], favorited=True),
        _full_item(confs[4]), _preview_item(confs[4]),
        _preview_item(confs[5], rejected=True), _preview_item(confs[5], favorited=True),
    ]
    meta = {"rejected_preview_fps": [config_fingerprint(confs[6], 1024, 1024, 7)]}
    plan = _plan(confs)
    entries, _ = _run(plan, items, meta, preview=preview, overwrite_existing=overwrite, only_favorited=only_fav)
    assert [e["config"]["seed"] for e in entries] == _loop_keep(
        plan, items, meta, preview=preview, overwrite=overwrite, skip_rej=True, only_fav=only_fav)


# -- manager API detection ----------------------------------------------------------------

class _OldManager:
    def __init__(self):
        self.called = None

    def populate_jobs(self, *a):
        self.called = ("populate_jobs", a)


class _NewManager(_OldManager):
    def populate_from_plan(self, entries, run_extras=None):
        self.called = ("populate_from_plan", entries, run_extras)


def _populate(mgr, plan, **kw):
    confs = [pj["conf"] for pj in plan]
    return go._populate_distributed_manager(mgr, plan, confs, [_job()], {"items": []}, False, False, "off", **kw)


def test_old_companion_falls_back_to_populate_jobs():
    mgr = _OldManager()
    _populate(mgr, _plan([_conf()]), job_order="minimize_reloads")
    assert mgr.called[0] == "populate_jobs"


def test_new_companion_gets_the_plan():
    mgr = _NewManager()
    _populate(mgr, _plan([_conf(seed=1), _conf(seed=2)]))
    assert mgr.called[0] == "populate_from_plan"
    assert [e["config"]["seed"] for e in mgr.called[1]] == [1, 2]
