"""Distributed runs: the master's post-phase upscales this run's images, grouped per model/LoRA/VAE.

REGRESSION (2026-11): the post-phase called run_deferred_upscales() with 11 positional arguments against a
12+ signature and entries without width/height, so it raised on every distributed run with upscaling on.
"""
import importlib.util
import inspect
import os
import re
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import generation_orchestrator as go  # noqa: E402  (conftest preloads the real module)
from config_utils import config_fingerprint  # noqa: E402

_spec = importlib.util.spec_from_file_location("_mu_real_du", os.path.join(_ROOT, "manifest_utils.py"))
_mu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mu)

_REAL_RDU = go.run_deferred_upscales
LOCAL_KEYS = {"config", "width", "height", "seed", "batch_idx", "actual_positive_prompt",
              "actual_negative_prompt", "gen_index", "config_fp"}


def _conf(model="a.safetensors", seed=1, lora="None", vae="Default", **kw):
    c = {"model": model, "sampler": "euler", "scheduler": "normal", "steps": 8, "cfg": 5.0,
         "denoise": 1.0, "seed": seed, "lora": lora, "lora_expanded": lora, "vae": vae,
         "positive": "p", "negative": "n", "gguf_options": object()}
    c.update(kw)
    return c


def _item(i, conf, **kw):
    it = {"id": i, "file": f"/view?filename=img_{i}.webp&type=output", "seed": conf["seed"],
          "width": 768, "height": 768, "batch_idx": 0, "positive": "p", "negative": "n",
          "model": conf["model"], "sampler": conf["sampler"], "scheduler": conf["scheduler"],
          "steps": conf["steps"], "cfg": conf["cfg"], "denoise": conf["denoise"], "lora": conf["lora"],
          "vae": conf.get("vae", "Default"), "config_positive": "cp", "config_negative": "cn"}
    it.update(kw)
    return it


@pytest.fixture(autouse=True)
def _real_item_filename(monkeypatch):
    monkeypatch.setattr(go, "item_filename", _mu.item_filename)


def test_queue_only_this_runs_plain_items_oldest_first():
    c1, c2 = _conf(seed=1), _conf(seed=2)
    jobs = [(c1, {}), (c2, {})]
    items = [  # newest first
        _item(5, c2),
        _item(4, c1),
        _item(3, c1, filename="x_upscaled.webp", upscaled=True),
        _item(2, c1, file=""),  # no filename
        _item(1, c1),  # pre-run
    ]
    q = go._build_distributed_upscale_queue(items, {1, 3}, jobs)
    assert [e["base_item"]["id"] for e in q] == [4, 5]
    for e in q:
        assert set(e) == LOCAL_KEYS | {"base_item"}
    assert q[0]["config"] is c1 and q[1]["config"] is c2  # job config by identity (gguf_options sentinel survives)
    assert q[0]["config_fp"] == config_fingerprint(c1, 768, 768, 1)


def test_queue_skips_preview_video_ltx():
    c = _conf()
    items = [_item(9, c, preview=True), _item(8, c, media_type="video"), _item(7, c, model_type="ltx_video")]
    assert go._build_distributed_upscale_queue(items, set(), [(c, {})]) == []


def test_queue_prefers_seed_sibling_and_handles_randomized_seed():
    a, b = _conf(seed=10), _conf(seed=20)
    q = go._build_distributed_upscale_queue([_item(2, b), _item(1, a)], set(), [(a, {}), (b, {})])
    assert [e["config"] for e in q] == [a, b]
    # randomized: the item carries a seed that is not the conf's; it still matches its job
    r = _conf(seed=10, seed_behavior="randomize")
    it = _item(3, r, seed=987654)
    q = go._build_distributed_upscale_queue([it], set(), [(r, {})])
    assert q[0]["config"] is r and q[0]["seed"] == 987654


def test_queue_fallback_conf_strips_item_fields():
    c = _conf()
    it = _item(4, _conf(model="other.safetensors"), favorited=True, note="n", gen_index=3, duration=1.5)
    q = go._build_distributed_upscale_queue([it], set(), [(c, {})])
    cfg = q[0]["config"]
    for k in ("id", "file", "favorited", "note", "gen_index", "duration"):
        assert k not in cfg
    assert cfg["positive"] == "cp" and cfg["negative"] == "cn"
    assert cfg["lora_expanded"] == "None" and cfg["model"] == "other.safetensors"


# ---- driver -------------------------------------------------------------------------------------------------

class _Rec:
    def __init__(self, mp, vae_result="vae", skip_lora=()):
        self.calls, self.loads = [], []
        self.skip_lora, self.vae_result = set(skip_lora), vae_result
        mp.setattr(go, "load_model_by_type", self._load)
        mp.setattr(go, "load_loras", self._loras)
        mp.setattr(go, "load_vae_by_name", lambda n: f"vae:{n}")
        mp.setattr(go, "cleanup_model_references", lambda m, c, cache: (None, None))
        mp.setattr(go, "run_deferred_upscales", lambda *a, **kw: self.calls.append(a))
        mp.setattr(go, "is_remote_vae", lambda v: str(v).startswith("remote"))
        self.mm = sys.modules["comfy.model_management"]
        mp.setattr(self.mm, "processing_interrupted", lambda: False, raising=False)

    def _load(self, conf, *a, **kw):
        self.loads.append(conf["model"])
        return "m", "c", (self.vae_result if self.vae_result != "none" else None)

    def _loras(self, m, c, lora, key, inc, model_cache=None):
        return "pm", "pc", lora in self.skip_lora


def _entry(conf):
    return {"config": conf, "width": 8, "height": 8, "seed": 1, "batch_idx": 0, "actual_positive_prompt": "p",
            "actual_negative_prompt": "n", "gen_index": 0, "config_fp": "x", "base_item": {}}


def _drive(queue):
    return go._run_distributed_upscales(queue, {"pipelines": [1]}, {}, {"images": "i"}, {"items": []}, "s", "u",
                                        "ck", False, None, None, None, None, None, {}, {})


def test_driver_groups_per_model_lora_vae(monkeypatch):
    rec = _Rec(monkeypatch)
    q = [_entry(_conf("a")), _entry(_conf("b")), _entry(_conf("a")), _entry(_conf("a", lora="L"))]
    _drive(q)
    assert len(rec.calls) == 3
    assert [len(c[0]) for c in rec.calls] == [2, 1, 1]  # first-appearance order
    assert len({id(c[6]) for c in rec.calls}) == 3  # a fresh conditioning cache per group
    n = len(inspect.signature(_REAL_RDU).parameters)
    assert len(rec.calls[0]) == n - 1  # config_overrides_vae stays defaulted
    assert rec.calls[0][3] == "vae" and rec.calls[0][4] == "pm" and rec.calls[0][5] == "pc"


def test_driver_skips_missing_vae_and_incompatible_lora(monkeypatch):
    rec = _Rec(monkeypatch, vae_result="none")
    _drive([_entry(_conf("a"))])
    assert rec.calls == []
    rec = _Rec(monkeypatch, skip_lora={"L"})
    _drive([_entry(_conf("a", lora="L")), _entry(_conf("b"))])
    assert len(rec.calls) == 1 and rec.calls[0][0][0]["config"]["model"] == "b"


def test_driver_named_vae_is_loaded(monkeypatch):
    rec = _Rec(monkeypatch)
    _drive([_entry(_conf("a", vae="my.vae"))])
    assert rec.calls[0][3] == "vae:my.vae"


def test_driver_stops_on_interrupt(monkeypatch):
    rec = _Rec(monkeypatch)
    monkeypatch.setattr(rec.mm, "processing_interrupted", lambda: True, raising=False)
    _drive([_entry(_conf("a")), _entry(_conf("b"))])
    assert rec.calls == []


# ---- source guards ------------------------------------------------------------------------------------------

def _src(name):
    with open(os.path.join(_ROOT, "generation_orchestrator.py"), encoding="utf-8") as f:
        text = f.read()
    m = re.search(rf"\ndef {name}\(.*?(?=\n(?:def |class )|\Z)", text, re.S)
    assert m, name
    return m.group(0)


def test_post_phase_source():
    s = _src("_run_distributed_generation")
    assert '"meta": item_meta' not in s
    assert "_run_distributed_upscales(" in s and "_build_distributed_upscale_queue(" in s
    assert s.index("_pre_run_ids =") < min(i for i in (s.find("manager.populate_jobs("), s.find("_populate_distributed_manager(")) if i >= 0)
    assert "run_deferred_upscales(" not in s


def test_deferred_uses_base_item():
    assert 'job.get("base_item")' in _src("run_deferred_upscales")
