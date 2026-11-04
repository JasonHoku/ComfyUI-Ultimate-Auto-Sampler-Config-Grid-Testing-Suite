"""Run context, attach/finalize, never-inherit, and optional scorer adapters (fake nodes)."""
import importlib
import os
import sys
import threading

import numpy as np
import pytest
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)
im = importlib.import_module(f"{_PKG}.image_metrics")

SPEC = {"builtins": ["entropy", "luma"], "adapters": []}
OTHER = {"builtins": ["contrast"], "adapters": []}


def _pil(v=128, size=(32, 24)):
    return Image.fromarray(np.full((size[1], size[0], 3), v, np.uint8))


@pytest.fixture(autouse=True)
def _clean_stack():
    im._stack.clear()
    yield
    im._stack.clear()


# ---- run context -----------------------------------------------------------
def test_no_context_means_no_metrics():
    meta = {}
    im.attach(meta, _pil())
    assert "metrics" not in meta and im.current_spec() is None


def test_attach_pil_keys_match_spec():
    tok = im.begin_run({"enabled": True, "builtins": ["luma", "entropy"]})
    meta = {}
    im.attach(meta, _pil(128))
    im.end_run(tok)
    assert set(meta["metrics"]) == {"luma", "entropy"}
    assert meta["metrics"]["luma"] == pytest.approx(128 / 255, abs=1e-3)


def test_end_run_does_not_clobber_another_runs_context():
    a = im.begin_run(SPEC)
    b = im.begin_run(OTHER)
    assert im.current_spec() == OTHER
    im.end_run(a)                      # A finishing late must not disturb B
    assert im.current_spec() == OTHER
    im.end_run(b)
    assert im.current_spec() is None
    im.end_run(a)                      # double end is a no-op


def test_inner_run_ending_restores_outer():
    a = im.begin_run(SPEC)
    b = im.begin_run(OTHER)
    im.end_run(b)
    assert im.current_spec() == SPEC
    im.end_run(a)


def test_other_thread_run_does_not_disturb_main_run():
    main = im.begin_run(SPEC)
    seen = {}

    def other():
        t = im.begin_run({"builtins": ["saturation"], "adapters": []})
        im.end_run(t)
        seen["done"] = True

    th = threading.Thread(target=other)
    th.start()
    th.join()
    assert seen["done"] and im.current_spec() == SPEC
    im.end_run(main)


def test_explicit_spec_ignores_generation_run_state():
    """The dashboard upscale thread passes spec explicitly: a concurrent run's spec must not apply."""
    tok = im.begin_run(SPEC)
    meta = {}
    im.attach(meta, _pil(), spec=None)
    assert "metrics" not in meta                     # explicit None = off, even though a run is active
    im.attach(meta, _pil(), spec=OTHER)
    assert set(meta["metrics"]) == {"contrast"}      # exactly the explicit spec, not SPEC
    m2 = {}
    im.finalize(m2, _pil(), spec=None)
    assert "metrics" not in m2
    sink = {}
    im.sink_update(sink, object(), spec=None)        # explicit None: never consults the run
    assert sink == {}
    im.end_run(tok)


def test_end_run_prints_timing_line(capsys):
    tok = im.begin_run(SPEC)
    im.attach({}, _pil())
    im.attach({}, _pil())
    im.end_run(tok)
    out = capsys.readouterr().out
    assert "metrics: 2 images, mean" in out and "max" in out


def test_attach_swallows_errors_and_warns_once(monkeypatch, capsys):
    tok = im.begin_run(SPEC)

    def boom(*a, **k):
        raise ValueError("boom")
    monkeypatch.setattr(im, "compute_for_pil", boom)
    m1, m2 = {"x": 1}, {}
    im.attach(m1, _pil())
    im.attach(m2, _pil())
    im.end_run(tok)
    out = capsys.readouterr().out
    assert m1 == {"x": 1} and m2 == {}
    assert out.count("metrics skipped: boom") == 1


def test_sink_update_without_real_torch_never_raises():
    tok = im.begin_run(SPEC)
    sink = {}
    im.sink_update(sink, object())
    im.end_run(tok)
    assert sink == {}


def test_finalize_never_inherits_source_metrics():
    meta = {"metrics": {"luma": 0.9, "aesthetic": 9.0}}
    im.finalize(meta, _pil())                         # no run -> nothing computed, stale value dropped
    assert "metrics" not in meta
    tok = im.begin_run(SPEC)
    meta = {"metrics": {"luma": 0.9, "aesthetic": 9.0}}
    im.finalize(meta, _pil(0))
    im.end_run(tok)
    assert set(meta["metrics"]) == {"luma", "entropy"} and meta["metrics"]["luma"] == 0
    m3 = {"metrics": {"stale": 1}}
    im.finalize(m3, None, {"luma": 0.5})
    assert m3["metrics"] == {"luma": 0.5}


def test_spec_from_item():
    assert im.spec_from_item({}) is None
    assert im.spec_from_item({"metrics": {}}) is None
    assert im.spec_from_item({"metrics": {"luma": 1, "aesthetic": 2, "zzz": 3}}) == {"builtins": ["luma"], "adapters": []}


def test_upscale_builders_do_not_copy_metrics():
    from florence2_hires import build_florence2_manifest_entry
    item = {"id": 1, "model": "m", "metrics": {"luma": 0.4}}
    sr = {"image_width": 8, "image_height": 8, "duration": 1.0}
    e = build_florence2_manifest_entry(sr, item, session_name="s", pipeline_name="p", upscale_id=2,
                                       upscaled_filename="x.webp", current_index=0, hires_denoise=0.4)
    assert "metrics" not in e
    src = open(os.path.join(_ROOT, "upscale_runner.py"), encoding="utf-8").read()
    assert src.count('"upscale_model", "metrics")') == 2     # SeedVR2 + HiRes/model copies


# ---- adapters ---------------------------------------------------------------
class _Loader:
    calls = 0
    FUNCTION = "load"

    def load(self):
        _Loader.calls += 1
        return ("MODEL",)


class _Scorer:
    FUNCTION = "score"
    seen = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "model": ("MODEL",), "mode": (["fast", "slow"],),
                             "scale": ("FLOAT", {"default": 2.0})}}

    def score(self, image, model, mode, scale):
        _Scorer.seen.append((model, mode, scale))
        return (np.float32(6.12345),)


class _V3Out:
    def __init__(self, *a):
        self.args = a


def _fake_ad(**over):
    ad = {"enabled": True, "node_id": "FakeScore", "image_arg": "image",
          "loaders": [{"node_id": "FakeLoad", "kwargs": {}, "pass_as": "model", "output_index": 0}],
          "output_index": 0, "pack_hint": "fake-pack"}
    ad.update(over)
    return ad


@pytest.fixture
def fake_nodes(monkeypatch):
    nodes_mod = sys.modules["nodes"]
    mapping = {"FakeLoad": _Loader, "FakeScore": _Scorer}
    monkeypatch.setattr(nodes_mod, "NODE_CLASS_MAPPINGS", mapping, raising=False)
    monkeypatch.setattr(im, "ADAPTERS", {"fake": _fake_ad()})
    _Loader.calls = 0
    _Scorer.seen = []
    return mapping


def test_adapter_runs_loader_once_per_run_and_floats_result(fake_nodes):
    spec = im.normalize_spec({"enabled": True, "builtins": ["luma"], "adapters": ["fake"]})
    assert spec["adapters"] == ["fake"]
    tok = im.begin_run(spec)
    m1, m2 = {}, {}
    im.attach(m1, _pil())
    im.attach(m2, _pil())
    im.end_run(tok)
    assert m1["metrics"]["fake"] == pytest.approx(6.1234, abs=2e-4)
    assert "luma" in m1["metrics"] and m2["metrics"]["fake"] == m1["metrics"]["fake"]
    assert _Loader.calls == 1
    assert _Scorer.seen[0] == ("MODEL", "fast", 2.0)        # combo + widget defaults filled
    tok = im.begin_run(spec)
    im.attach({}, _pil())
    im.end_run(tok)
    assert _Loader.calls == 2                               # cache dropped at end_run


def test_attach_adapters_after_sink(fake_nodes):
    tok = im.begin_run({"builtins": [], "adapters": ["fake"]})
    meta = {"metrics": {"luma": 0.1}}
    im.attach_adapters(meta, _pil())
    im.end_run(tok)
    assert set(meta["metrics"]) == {"luma", "fake"}


def test_missing_node_reported_and_not_called(fake_nodes, capsys):
    del fake_nodes["FakeScore"]
    st = {a["name"]: a for a in im.adapter_status()}
    assert st["fake"]["installed"] is False and st["fake"]["pack_hint"] == "fake-pack"
    tok = im.begin_run({"builtins": [], "adapters": ["fake"]})
    meta = {}
    im.attach(meta, _pil())
    im.end_run(tok)
    assert "metrics" not in meta and _Loader.calls == 0
    assert "metric adapter 'fake' disabled" in capsys.readouterr().out


def test_uncovered_required_input_disables_with_warning(fake_nodes, capsys):
    class Strict(_Scorer):
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",), "model": ("MODEL",), "mystery": ("LATENT",)}}
    fake_nodes["FakeScore"] = Strict
    tok = im.begin_run({"builtins": [], "adapters": ["fake"]})
    meta = {}
    im.attach(meta, _pil())
    im.attach(meta, _pil())
    im.end_run(tok)
    out = capsys.readouterr().out
    assert "metrics" not in meta and out.count("disabled") == 1 and "mystery" in out
    assert _Loader.calls == 0


def test_raising_adapter_disabled_for_rest_of_run(fake_nodes, capsys):
    calls = []

    class Boom(_Scorer):
        def score(self, image, model, mode, scale):
            calls.append(1)
            raise RuntimeError("kaput")
    fake_nodes["FakeScore"] = Boom
    tok = im.begin_run({"builtins": ["luma"], "adapters": ["fake"]})
    m1, m2 = {}, {}
    im.attach(m1, _pil())
    im.attach(m2, _pil())
    im.end_run(tok)
    out = capsys.readouterr().out
    assert len(calls) == 1 and out.count("kaput") == 1
    assert set(m1["metrics"]) == {"luma"} and set(m2["metrics"]) == {"luma"}   # builtins unaffected


def test_v3_style_node_and_prompt_arg(monkeypatch):
    seen = {}

    class V3:
        @classmethod
        def execute(cls, image, prompt, model):
            seen["prompt"] = prompt
            return _V3Out(7.5)
    monkeypatch.setattr(sys.modules["nodes"], "NODE_CLASS_MAPPINGS", {"FakeLoad": _Loader, "FakeScore": V3}, raising=False)
    monkeypatch.setattr(im, "ADAPTERS", {"fake": _fake_ad(text_arg="prompt")})
    # a non-V3-subclass with execute() and no FUNCTION is dispatched through FUNCTION default "execute"
    V3.FUNCTION = "execute"
    tok = im.begin_run({"builtins": [], "adapters": ["fake"]})
    meta = {"positive": "a red cat"}
    im.attach(meta, _pil())
    im.end_run(tok)
    assert meta["metrics"] == {"fake": 7.5} and seen["prompt"] == "a red cat"


def test_shipped_registry_entries_are_disabled():
    assert im.ADAPTERS and all(not a["enabled"] for a in im.ADAPTERS.values())
    assert im.normalize_spec({"enabled": True, "builtins": ["luma"], "adapters": ["aesthetic"]})["adapters"] == []
