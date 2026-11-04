"""manifest_utils.item_filename: generated items carry only a /view URL, upscaled/scanned ones a filename.
The deferred-upscale phase (run_upscales_at_end) skipped every image with "no filename in manifest" before it
used this helper; the source-scan test pins that call."""
import importlib.util
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_ROOT, name))
    m = importlib.util.module_from_spec(spec)
    m.__package__ = os.path.basename(_ROOT)
    spec.loader.exec_module(m)
    return m


mu = _load("manifest_utils.py", "mu_item_filename")


def test_filename_key_wins():
    assert mu.item_filename({"filename": "img_1.webp", "file": "/view?filename=other.webp"}) == "img_1.webp"


def test_parsed_from_view_url():
    url = "/view?filename=img_179378440462348.webp&type=output&subfolder=benchmarks/S/images"
    assert mu.item_filename({"file": url}) == "img_179378440462348.webp"
    assert mu.item_filename({"file": "/view?type=output&filename=a%20b.png&subfolder=x"}) == "a b.png"


def test_unusable_items():
    assert mu.item_filename({}) == ""
    assert mu.item_filename({"file": "/view?type=output"}) == ""
    assert mu.item_filename(None) == ""


def test_deferred_upscales_use_the_helper():
    src = open(os.path.join(_ROOT, "generation_orchestrator.py"), encoding="utf-8").read()
    start = src.index("def run_deferred_upscales(")
    body = src[start:start + 6000]
    assert "item_filename(base_item)" in body
    assert not re.search(r'base_item\.get\("filename"\)', body)
