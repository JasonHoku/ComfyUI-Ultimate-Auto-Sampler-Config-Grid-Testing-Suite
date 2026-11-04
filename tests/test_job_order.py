"""Flat job plan, as_written identity and minimize_reloads ordering."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_utils import build_job_plan, order_job_plan, count_switches


def _key(c):
    return c["model"]


def _conf(model="A", lora="None", pos="p", seed=1, **kw):
    c = {"model": model, "lora": lora, "lora_expanded": lora, "positive": pos,
         "negative": "", "seed": seed, "vae": "Default"}
    c.update(kw)
    return c


def _jobs(*resolutions):
    return [{"label": f"{w}x{h}", "width": w, "height": h, "latent": None, "batch_idx": 0}
            for w, h in resolutions]


def _nested_reference(expanded, input_jobs):
    """Verbatim copy of the pre-refactor loop shape (the source of truth for as_written)."""
    out = []
    for job_idx, job in enumerate(input_jobs):
        w, h = job["width"], job["height"]
        batch_idx = job["batch_idx"]
        for conf_idx, conf in enumerate(expanded):
            if conf.get("resolution") is not None:
                if job_idx > 0:
                    continue
                w, h = conf["resolution"]
                batch_idx = 0
            out.append((conf_idx, job_idx, w, h, batch_idx))
    return out


def _total_jobs(expanded, input_jobs):
    with_res = sum(1 for c in expanded if c.get("resolution") is not None)
    return with_res + (len(expanded) - with_res) * len(input_jobs)


def _grid():
    expanded = []
    for m in "ABC":
        for lo in ("None", "L1", "L2"):
            for p in ("p1", "p2", "p3", "p4"):
                for s in (1, 2):
                    expanded.append(_conf(m, lo, p, s))
    return expanded


RES3 = _jobs((1024, 1024), (832, 1216), (1216, 832))


def test_plan_reproduces_nested_order_without_per_config_resolution():
    expanded = _grid()
    plan = build_job_plan(expanded, RES3)
    got = [(p["conf_idx"], p["job_idx"], p["w"], p["h"], p["batch_idx"]) for p in plan]
    assert got == _nested_reference(expanded, RES3)
    assert all(p["conf"] is expanded[p["conf_idx"]] for p in plan)
    assert all(p["job"] is RES3[p["job_idx"]] for p in plan)


def test_plan_len_matches_total_jobs_formula():
    expanded = _grid()
    expanded[3]["resolution"] = [640, 640]
    expanded[10]["resolution"] = [512, 768]
    assert len(build_job_plan(expanded, RES3)) == _total_jobs(expanded, RES3)
    assert len(build_job_plan(_grid(), RES3)) == 216 == _total_jobs(_grid(), RES3)


def test_per_config_resolution_appears_once_and_does_not_leak():
    expanded = [_conf(seed=1), _conf(seed=2, resolution=[640, 640]), _conf(seed=3)]
    jobs = _jobs((1024, 1024), (832, 1216))
    plan = build_job_plan(expanded, jobs)
    assert [(p["conf_idx"], p["job_idx"], p["w"], p["h"]) for p in plan] == [
        (0, 0, 1024, 1024), (1, 0, 640, 640), (2, 0, 1024, 1024),   # F2: conf 2 must NOT inherit 640x640
        (0, 1, 832, 1216), (2, 1, 832, 1216),
    ]


def test_per_config_resolution_resets_batch_idx_without_leaking():
    jobs = [{"label": "a", "width": 64, "height": 64, "latent": None, "batch_idx": 2}]
    expanded = [_conf(resolution=[8, 8]), _conf()]
    plan = build_job_plan(expanded, jobs)
    assert [p["batch_idx"] for p in plan] == [0, 2]


def test_as_written_is_identity():
    plan = build_job_plan(_grid(), RES3)
    out = order_job_plan(plan, "as_written", _key)
    assert out is plan or out == plan
    assert [id(x) for x in out] == [id(x) for x in plan]


def test_unknown_mode_is_as_written():
    plan = build_job_plan(_grid(), RES3)
    assert order_job_plan(plan, "bogus", _key) == plan


def test_worked_example_counts():
    plan = build_job_plan(_grid(), RES3)
    before = count_switches(plan, _key)
    assert before == {"model_loads": 9, "lora_switches": 27, "vae_switches": 9, "res_changes": 2}
    after_plan = order_job_plan(plan, "minimize_reloads", _key)
    after = count_switches(after_plan, _key)
    assert after == {"model_loads": 3, "lora_switches": 9, "vae_switches": 3, "res_changes": 26}
    assert len(after_plan) == len(plan)
    assert sorted(id(x) for x in after_plan) == sorted(id(x) for x in plan)


def test_folder_lora_case_alternation():
    # 1 model, 1 resolution, 4 prompts x 2 seeds; seeds pick La / Lb.
    expanded = [_conf("A", "La" if s == 1 else "Lb", p, s)
                for p in ("p1", "p2", "p3", "p4") for s in (1, 2)]
    jobs = _jobs((1024, 1024))
    plan = build_job_plan(expanded, jobs)
    # counted INCLUDING the first patch: 8 patches (= 7 after the first) -> 2 (= 1 after the first)
    assert count_switches(plan, _key)["lora_switches"] == 8
    after = order_job_plan(plan, "minimize_reloads", _key)
    assert count_switches(after, _key)["lora_switches"] == 2


def test_minimize_reloads_is_stable_and_first_appearance_ordered():
    # Models appear C, A, B (not alphabetical); within a group written order is kept.
    expanded = [_conf("C", pos="x1"), _conf("A", pos="x2"), _conf("B", pos="x3"),
                _conf("C", pos="x4"), _conf("A", pos="x5")]
    jobs = _jobs((64, 64))
    plan = build_job_plan(expanded, jobs)
    after = order_job_plan(plan, "minimize_reloads", _key)
    assert [p["conf"]["model"] + p["conf"]["positive"] for p in after] == [
        "Cx1", "Cx4", "Ax2", "Ax5", "Bx3"]


def test_edge_cases_empty_and_single():
    assert build_job_plan([], RES3) == []
    assert order_job_plan([], "minimize_reloads", _key) == []
    assert count_switches([], _key) == {"model_loads": 0, "lora_switches": 0, "vae_switches": 0, "res_changes": 0}
    plan = build_job_plan([_conf()], _jobs((64, 64)))
    assert order_job_plan(plan, "minimize_reloads", _key) == plan
    assert count_switches(plan, _key)["model_loads"] == 1


def test_vae_none_and_default_group_together():
    expanded = [_conf("A", vae="None"), _conf("A", vae="Default"), _conf("A", vae="")]
    plan = build_job_plan(expanded, _jobs((64, 64)))
    assert count_switches(plan, _key)["vae_switches"] == 1


def test_list_valued_model_keys_are_hashable():
    expanded = [_conf("A"), _conf("B"), _conf("A")]
    plan = build_job_plan(expanded, _jobs((64, 64)))
    after = order_job_plan(plan, "minimize_reloads", lambda c: [c["model"], ["te"]])
    assert [p["conf"]["model"] for p in after] == ["A", "A", "B"]
