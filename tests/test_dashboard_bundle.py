"""Dashboard JS/CSS bundle routes (Roadmap 27, milestone 2): html_generator bundles + run_api.bundle_response."""
import gzip
import importlib.util
import os
import re
import shutil
import subprocess
import tempfile

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)
NODE = shutil.which("node")


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(f"{_PKG}.{name}_bundle_real", os.path.join(_ROOT, filename))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG
    spec.loader.exec_module(mod)
    return mod


hg = _load("html_generator", "html_generator.py")
run_api = _load("run_api", "run_api.py")

EXPECTED_JS = (
    "logic_state.js", "logic_utils.js", "logic_ratings.js", "logic_wfparams.js", "logic_analytics.js", "logic_dragdrop.js",
    "logic_ui.js", "logic_upscale_modal.js", "logic_virtual.js", "logic_pipeline.js", "logic_compare.js",
    "logic_analytics_panel.js", "logic_events.js", "logic_init.js", "logic_standalone.js",
)
TOKENS = ("__CSS_TAG__", "__JS_TAG__", "__CSS_CONTENT__", "__JS_CONTENT__", "__TITLE__", "__NODE_ID__",
          "__STANDALONE_FLAG__")


def _shell(title="t", node_id="1", **kw):
    return hg.get_html_template(title, {"items": []}, node_id, **kw)


def _read(name):
    with open(os.path.join(hg.RESOURCE_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("USCG_INLINE_DASHBOARD_ASSETS", raising=False)


def test_js_files_exact_order():
    assert hg.JS_FILES == EXPECTED_JS
    assert hg.JS_FILES[-1] == "logic_standalone.js"
    assert hg.CSS_FILES == ("report.css",)


def test_markers_and_each_file_once_in_order():
    text = hg.get_dashboard_bundle("js").text
    last = -1
    for f in EXPECTED_JS:
        marker = f"/* --- {f} --- */"
        assert text.count(marker) == 1
        pos = text.index(marker)
        assert pos > last
        last = pos
        assert text.count(_read(f)) == 1
    assert hg.get_dashboard_bundle("css").text == _read("report.css")


def test_bundle_is_page_independent():
    b1 = hg.get_dashboard_bundle("js")
    _shell("a", "9", standalone=True)
    _shell("b", "x", inline_manifest=True)
    assert hg.get_dashboard_bundle("js") is b1
    for kind in ("js", "css"):
        for tok in TOKENS:
            assert tok not in hg.get_dashboard_bundle(kind).text


def test_inline_and_bundled_modes_carry_the_same_code():
    page = _shell(inline_assets=True)
    js = hg.get_dashboard_bundle("js").text
    css = hg.get_dashboard_bundle("css").text
    assert "<script>\n/* INJECTED LOGIC */\n" + js + "\n</script>" in page
    assert "<style>\n/* INJECTED CSS */\n" + css + "\n</style>" in page


def test_shell_references_bundles_with_hash():
    page = _shell(standalone=True)
    jh = hg.get_dashboard_bundle("js").hash
    ch = hg.get_dashboard_bundle("css").hash
    tag = f'<script src="/config_tester/dashboard_bundle_js?v={jh}"></script>'
    assert page.count(tag) == 1
    assert f'<link rel="stylesheet" href="/config_tester/dashboard_bundle_css?v={ch}">' in page
    assert "/* --- logic_" not in page
    size = len(page.encode("utf-8"))
    print(f"measured shell size: {size} bytes")
    assert size < 100 * 1024
    assert page.index("const TARGET_NODE_ID") < page.index(tag) < page.index('id="upscale-modal-overlay"')
    for tok in TOKENS:
        assert tok not in page


def test_inline_manifest_implies_inline_assets_and_overrides():
    m = {"items": []}
    implied = hg.get_html_template("t", m, "1", inline_manifest=True)
    assert "/* INJECTED LOGIC */" in implied and "dashboard_bundle_js" not in implied
    assert "/* INJECTED LOGIC */" in hg.get_html_template("t", m, "1", inline_assets=True)
    off = hg.get_html_template("t", m, "1", inline_manifest=True, inline_assets=False)
    assert "/* INJECTED LOGIC */" not in off and "dashboard_bundle_js" in off


def test_env_forces_inline(monkeypatch):
    monkeypatch.setenv("USCG_INLINE_DASHBOARD_ASSETS", "1")
    page = _shell()
    assert "/* INJECTED LOGIC */" in page and "dashboard_bundle_js" not in page


def test_hash_follows_resource_changes(tmp_path, monkeypatch):
    dst = tmp_path / "resources"
    shutil.copytree(hg.RESOURCE_DIR, dst)
    monkeypatch.setattr(hg, "RESOURCE_DIR", str(dst))
    hg._bundle_cache.clear()
    try:
        b1 = hg.get_dashboard_bundle("js")
        assert hg.get_dashboard_bundle("js") is b1  # unchanged signature: cached object
        url1 = hg.bundle_url("js")
        init = dst / "logic_init.js"
        st = init.stat()
        os.utime(init, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))  # mtime only
        b2 = hg.get_dashboard_bundle("js")
        assert b2 is not b1 and b2.hash == b1.hash
        with open(init, "ab") as f:
            f.write(b"\n// hot reload edit\n")
        b3 = hg.get_dashboard_bundle("js")
        assert b3.hash != b1.hash
        assert hg.bundle_url("js") != url1 and b3.hash in hg.bundle_url("js")
    finally:
        hg._bundle_cache.clear()


def test_gzip_round_trip_and_deterministic():
    for kind in ("js", "css"):
        b = hg.get_dashboard_bundle(kind)
        assert gzip.decompress(b.gzip) == b.body
        assert b.gzip == gzip.compress(b.body, 9, mtime=0)
        assert b.body == b.text.encode("utf-8")
        assert len(b.gzip) < len(b.body)


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        hg.get_dashboard_bundle("../x")


def test_bundle_response_headers():
    b = hg.get_dashboard_bundle("js")
    st, body, h = run_api.bundle_response(b, b.hash, None, "gzip, br")
    assert st == 200 and body == b.gzip
    assert h["Content-Encoding"] == "gzip"
    assert h["Cache-Control"] == "public, max-age=31536000, immutable"
    assert h["ETag"] == f'W/"{b.hash}"'
    assert h["Vary"] == "Accept-Encoding" and h["X-Content-Type-Options"] == "nosniff"
    st, body, h = run_api.bundle_response(b, b.hash, None, "")
    assert st == 200 and body == b.body and "Content-Encoding" not in h
    assert h["Vary"] == "Accept-Encoding" and h["X-Content-Type-Options"] == "nosniff"
    for v in ("bogus", "", None):
        _, _, h = run_api.bundle_response(b, v, None, "gzip")
        assert h["Cache-Control"] == "no-cache"
    for inm in (f'W/"{b.hash}"', f'"{b.hash}"', f'"zzz", W/"{b.hash}"'):
        st, body, h = run_api.bundle_response(b, b.hash, inm, "gzip")
        assert st == 304 and body is None and h["ETag"] == f'W/"{b.hash}"'
    st, _, _ = run_api.bundle_response(b, b.hash, 'W/"other"', "gzip")
    assert st == 200


@pytest.mark.parametrize("v", ["../../etc/passwd", "logic_ui.js", "..\\..\\windows\\win.ini"])
def test_v_is_never_a_path(v):
    b = hg.get_dashboard_bundle("js")
    st, body, h = run_api.bundle_response(b, v, None, "")
    assert st == 200 and body == b.body and h["Cache-Control"] == "no-cache"


def test_routes_are_fixed_paths():
    with open(os.path.join(_ROOT, "__init__.py"), encoding="utf-8") as f:
        src = f.read()
    assert '"/config_tester/dashboard_bundle_js"' in src
    assert '"/config_tester/dashboard_bundle_css"' in src
    for path in re.findall(r'routes\.(?:get|post|put|delete)\(\s*"([^"]+)"', src):
        assert "{" not in path, path
    m = re.search(r"def _bundle_http\(.*?(?=\n@server)", src, re.S)
    assert m
    handlers = [m.group(0)]
    for fn in ("dashboard_bundle_js", "dashboard_bundle_css"):
        handlers.append(re.search(r"async def %s\(.*?(?=\n\n\n|\Z)" % fn, src, re.S).group(0))
    for body in handlers:
        for banned in ("match_info", "open(", "FileResponse", "enable_compression"):
            assert banned not in body, banned


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_bundle_parses_and_composes_with_globals():
    page = _shell(standalone=True)
    globals_js = re.search(r"<script>\s*// INJECTED DATA \(per page\)([\s\S]*?)</script>", page).group(1)
    js = hg.get_dashboard_bundle("js").text
    with tempfile.TemporaryDirectory() as d:
        for name, code in (("bundle.js", js), ("all.js", globals_js + "\n" + js)):
            path = os.path.join(d, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            out = subprocess.run([NODE, "--check", path], capture_output=True, timeout=60)
            assert out.returncode == 0, out.stderr.decode()
