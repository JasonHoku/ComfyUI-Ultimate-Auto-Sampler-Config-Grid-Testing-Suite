"""Review-d fixes: deferred-upscale resume keyed on base_fp, legacy-companion preview upscales,
latent dimension ordering, and a paused identical sweep being resumed by Start."""
import importlib.util
import os
import re
import sys
import types

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import generation_orchestrator as go  # noqa: E402
import workflow_sweep_runner as runner  # noqa: E402

_spec = importlib.util.spec_from_file_location("_mu_real_rdb", os.path.join(_ROOT, "manifest_utils.py"))
_mu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mu)


# ---- deferred upscale resume check ---------------------------------------------------------------------------

def _job(fp, tmp_path, w=512, h=512, seed=7):
    (tmp_path / "base.webp").write_bytes(b"not an image")  # makes it past the file check, fails at decode
    base = {"id": 1, "file": "/view?filename=base.webp&type=output", "seed": seed,
            "width": w, "height": h, "positive": "p", "filename": "base.webp"}
    j = {"config": {}, "width": w, "height": h, "seed": seed, "batch_idx": 0,
         "actual_positive_prompt": "p", "actual_negative_prompt": "n", "gen_index": 0, "base_item": base}
    if fp:
        j["config_fp"] = fp
    return j


def _run(monkeypatch, capsys, tmp_path, job, items):
    monkeypatch.setattr(go, "item_filename", _mu.item_filename)
    ig = [m for n, m in sys.modules.items() if n.endswith("image_generation") and hasattr(m, "generate_image")][0]
    monkeypatch.setattr(ig, "upscale_image", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(sys.modules["comfy.model_management"], "processing_interrupted",
                        lambda: False, raising=False)
    go.run_deferred_upscales([job], {"pipelines": [{}]}, {}, None, None, None, {"positive": {}, "negative": {}},
                             {"images": str(tmp_path)}, {"items": items}, "s", "u", None)
    out = capsys.readouterr().out
    return "already upscaled" in out


def _up(base_fp=None, seed=7, w=1024, **kw):
    it = {"upscaled": True, "seed": seed, "positive": "p", "width": w, "height": w}
    if base_fp:
        it["base_fp"] = base_fp
    it.update(kw)
    return it


def test_resume_keyed_on_base_fp_not_seed(monkeypatch, capsys, tmp_path):
    # two cells share seed + prompt; only cell A has been upscaled
    items = [_up(base_fp="fpA")]
    assert _run(monkeypatch, capsys, tmp_path, _job("fpA", tmp_path), items) is True
    assert _run(monkeypatch, capsys, tmp_path, _job("fpB", tmp_path), items) is False
    # both upscaled -> both skipped
    items = [_up(base_fp="fpA"), _up(base_fp="fpB")]
    assert _run(monkeypatch, capsys, tmp_path, _job("fpA", tmp_path), items) is True
    assert _run(monkeypatch, capsys, tmp_path, _job("fpB", tmp_path), items) is True


def test_fingerprinted_job_ignores_legacy_upscaled_items(monkeypatch, capsys, tmp_path):
    assert _run(monkeypatch, capsys, tmp_path, _job("fpA", tmp_path), [_up()]) is False


def test_legacy_fallback_seed_prompt_and_base_dims(monkeypatch, capsys, tmp_path):
    j = _job(None, tmp_path)
    assert _run(monkeypatch, capsys, tmp_path, j, [_up()]) is True
    assert _run(monkeypatch, capsys, tmp_path, j, [_up(seed=8)]) is False
    # base dims recorded on the upscaled item must match the job's
    assert _run(monkeypatch, capsys, tmp_path, j, [_up(base_width=512, base_height=512)]) is True
    assert _run(monkeypatch, capsys, tmp_path, j, [_up(base_width=640, base_height=512)]) is False


# ---- source guards (heavy paths cannot run under the stubs) ---------------------------------------------------

def _text(name):
    with open(os.path.join(_ROOT, name), encoding="utf-8") as f:
        return f.read()


def test_preview_override_keeps_full_settings_for_legacy_companion():
    src = _text("generation_orchestrator.py")
    assert src.index("session_settings_full = session_settings") < src.index('"upscaling": {"enabled": False}}')
    assert "upscale_session_settings=session_settings_full" in src
    m = re.search(r"\ndef _run_distributed_generation\(.*?(?=\ndef |\Z)", src, re.S).group(0)
    legacy = m.index("session_settings = upscale_session_settings")
    assert 'hasattr(manager, "populate_from_plan")' in m[:legacy]
    assert legacy < m.index("Running upscales on master")


def test_latent_dims_recomputed_before_gen_dims():
    # the local loop needs a model + sampler, so check the statement order in the source
    src = _text("generation_orchestrator.py")
    i = src.index('optional_latent["samples"].shape[3]) * latent_ratio')
    j = src.index("gen_w, gen_h = w, h", i - 400)
    assert i < j
    assert src.count("gen_w, gen_h = w, h\n            _pv_scale") == 1


# ---- sweep: Start on an identical paused sweep resumes it ------------------------------------------------------

def test_start_sweep_resumes_identical_paused_sweep():
    sys.path.insert(0, os.path.join(_ROOT, "tests"))
    import test_workflow_sweep_runner as t
    mp = pytest.MonkeyPatch()
    import folder_paths
    import tempfile
    tmp = tempfile.mkdtemp()
    mp.setattr(folder_paths, "get_output_directory", lambda: tmp, raising=False)
    mp.setattr(runner, "manifest_utils", t._real_manifest_utils(), raising=False)
    runner.reset_registry()
    try:
        bridge = t.FakeBridge()
        env = types.SimpleNamespace(bridge=bridge)
        sw = t.make(env)
        assert runner.pause(sw.id)[0] and sw.state == "paused"
        again = runner.start_sweep([t.entry()], {"session_name": "wf_test"}, "n2", None, bridge, class_info=None)
        assert again is sw and sw.state == "running"
        # a running one is still just attached
        n = len(bridge.enqueued)
        same = runner.start_sweep([t.entry()], {"session_name": "wf_test"}, "n3", None, bridge, class_info=None)
        assert same is sw and sw.state == "running" and len(bridge.enqueued) == n
    finally:
        runner.reset_registry()
        mp.undo()
