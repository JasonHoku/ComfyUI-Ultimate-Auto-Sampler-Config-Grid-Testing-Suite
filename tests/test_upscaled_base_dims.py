"""Source guard: every place that writes an upscaled manifest item also records base_width/base_height."""
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WRITE = re.compile(r"""("upscaled": True|\["upscaled"\] = True)""")
_SENTINEL = re.compile(r"\n\s*(?:sentinel_meta\.update\(|build_florence2_no_detection_entry\()")
_BASE = ("base_width", "base_dims_of", "base_dims=")


def _read(name):
    with open(os.path.join(_ROOT, name), encoding="utf-8") as f:
        return f.read()


def _blocks(text):
    """Text from each upscaled-True write up to the manifest insert/append that follows it."""
    out = []
    for m in _WRITE.finditer(text):
        tail = text[m.end():]
        end = re.search(r"(items\"\]\.(insert|append)\()", tail)
        out.append((m.start(), text[max(0, m.start() - 1500): m.end() + (end.end() if end else 1500)]))
    return out


def test_every_upscaled_write_records_base_dims():
    total = 0
    for name in ("generation_orchestrator.py", "upscale_runner.py"):
        text = _read(name)
        for pos, block in _blocks(text):
            total += 1
            line = text.count("\n", 0, pos) + 1
            assert any(tok in block for tok in _BASE), f"{name}:{line} writes an upscaled item without base dims"
    assert total >= 6  # 1 deferred + 3 inline + 2 dashboard (sanity: the regex still finds them)


def test_sentinels_record_base_dims():
    for name in ("generation_orchestrator.py", "upscale_runner.py"):
        text = _read(name)
        for m in _SENTINEL.finditer(text):
            assert any(tok in text[m.end(): m.end() + 900] for tok in _BASE), f"{name}: sentinel without base dims"


def test_base_dims_of_cases():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_mu_real", os.path.join(_ROOT, "manifest_utils.py"))
    mu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mu)
    f = mu.base_dims_of
    assert f({"width": 832, "height": 1216}) == {"base_width": 832, "base_height": 1216}
    assert f({"width": 1248, "height": 1824, "upscaled": True, "base_width": 832, "base_height": 1216}) == \
        {"base_width": 832, "base_height": 1216}
    assert f({"width": 1248, "height": 1824, "upscaled": True}) == {}
    assert f({}) == {}
    assert f({"width": 0, "height": 5}) == {}
