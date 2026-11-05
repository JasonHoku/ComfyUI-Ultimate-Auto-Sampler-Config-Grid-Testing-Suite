"""Issue #16: registered attention functions were installed as the override directly.

ComfyUI calls `optimized_attention_override(original_func, q, k, v, heads, ...)`, so the
registered `(q, k, v, heads, ...)` function received the original function as q and died
with "'function' object has no attribute 'shape'".
"""
import importlib.util
import os
import sys
import types

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


@pytest.fixture
def ig(monkeypatch):
    """The real image_generation.py (conftest registers a stub under the package name)."""
    import torch
    if not hasattr(torch, "inference_mode"):
        monkeypatch.setattr(torch, "inference_mode", lambda *a, **k: (lambda f: f), raising=False)
    pkg = os.path.basename(_ROOT)
    spec = importlib.util.spec_from_file_location(f"{pkg}.image_generation_real", os.path.join(_ROOT, "image_generation.py"))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # relative imports needing real ComfyUI
        pytest.skip(f"image_generation.py not importable under the stubs: {e}")
    return mod


def test_override_drops_original_function(ig):
    calls = []

    def fake_sage(*args, **kwargs):
        calls.append((args, kwargs))
        return "out"

    override = ig.make_attention_override(fake_sage, "sage")
    orig = lambda *a, **k: "orig"
    q, k, v = object(), object(), object()
    assert override(orig, q, k, v, 8, mask=None, _inside_attn_wrapper=True) == "out"
    args, kwargs = calls[0]
    assert args == (q, k, v, 8)
    assert orig not in args
    assert kwargs == {"mask": None, "_inside_attn_wrapper": True}
    assert override.uscg_attention == "sage"


def _fake_comfy(monkeypatch, funcs):
    mod = types.ModuleType("comfy.ldm.modules.attention")
    mod.get_attention_function = lambda name, default=...: funcs.get(name, default)
    for n in ("comfy.ldm", "comfy.ldm.modules"):
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
    monkeypatch.setitem(sys.modules, "comfy.ldm.modules.attention", mod)


def test_generate_image_installs_and_restores_override(ig, monkeypatch):
    """The install/restore block of generate_image, exercised via source-level contract:
    the installed object is the wrapper, not the registered function."""
    import inspect
    src = inspect.getsource(ig.generate_image)
    assert "make_attention_override(attn_func, attention_mode)" in src
    assert 'del patched_model.model_options["transformer_options"]["optimized_attention_override"]' in src


def test_fallback_retries_once_with_default(ig, monkeypatch, capsys):
    seen = []

    def fake_generate(*a, attention_mode="default", **kw):
        seen.append(attention_mode)
        if attention_mode == "sage":
            raise AttributeError("'function' object has no attribute 'shape'")
        return {"samples": 1}, 1.5

    monkeypatch.setattr(ig, "generate_image", fake_generate)
    lat, dur, fb = ig.generate_image_attn_safe("m", attention_mode="sage")
    assert seen == ["sage", "default"]
    assert (lat, dur, fb) == ({"samples": 1}, 1.5, "sage")
    out = capsys.readouterr().out
    assert "Attention mode 'sage' failed" in out and "default attention" in out


def test_no_fallback_for_default_or_success(ig, monkeypatch):
    monkeypatch.setattr(ig, "generate_image", lambda *a, attention_mode="default", **kw: ("L", 1.0))
    assert ig.generate_image_attn_safe(attention_mode="sage") == ("L", 1.0, None)

    def boom(*a, **kw):
        raise RuntimeError("x")
    monkeypatch.setattr(ig, "generate_image", boom)
    with pytest.raises(RuntimeError):
        ig.generate_image_attn_safe(attention_mode="default")


def test_interrupt_is_not_retried(ig, monkeypatch):
    class InterruptProcessingException(Exception):
        pass
    n = []

    def boom(*a, **kw):
        n.append(1)
        raise InterruptProcessingException()
    monkeypatch.setattr(ig, "generate_image", boom)
    with pytest.raises(InterruptProcessingException):
        ig.generate_image_attn_safe(attention_mode="sage")
    assert len(n) == 1
