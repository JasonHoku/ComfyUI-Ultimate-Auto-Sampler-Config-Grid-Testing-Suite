"""Standalone-dashboard flag + manifest inlining in html_generator (and the dashboard.js incremental parse)."""
import importlib.util
import os
import re
import shutil
import subprocess
import tempfile

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.basename(_ROOT)
_spec = importlib.util.spec_from_file_location(f"{_PKG}.html_generator_real", os.path.join(_ROOT, "html_generator.py"))
html_generator = importlib.util.module_from_spec(_spec)
html_generator.__package__ = _PKG
_spec.loader.exec_module(html_generator)

EVIL = "evil </script><script>alert(1)</script> and </b>"
NODE = shutil.which("node")


def _page(**kw):
    """The legacy inlined-manifest page (inline_manifest=True), for the escape/regex tests."""
    manifest = {"items": [{"id": 1, "note": EVIL}], "meta": {}}
    return html_generator.get_html_template("t", manifest, "standalone", inline_manifest=True, **kw)


def _shell(items=None, title="t", **kw):
    """The default (lazy) page: manifest data passed in must NOT end up inlined."""
    manifest = {"items": items if items is not None else [{"id": 1, "note": EVIL}], "meta": {}}
    return html_generator.get_html_template(title, manifest, "standalone", **kw)


def test_standalone_flag_and_no_placeholder_left():
    on, off = _page(standalone=True), _page()
    assert "const USCG_STANDALONE = true" in on
    assert "const USCG_STANDALONE = false" in off
    for page in (on, off):
        assert "__STANDALONE_FLAG__" not in page
        assert "logic_standalone.js" in page
        assert "<title>USCG - t</title>" in page


def test_script_close_in_note_is_escaped():
    page = _page(standalone=True)
    assert "alert(1)</script>" not in page
    assert "<\\/script>" in page


def _dashboard_js_pattern():
    """The exact incremental-refresh regex used by web/dashboard.js."""
    src = open(os.path.join(_ROOT, "web", "dashboard.js"), encoding="utf-8").read()
    m = re.search(r"const pattern = /(.*__JSON_END__.*?)/;", src)
    assert m, "pattern line not found in web/dashboard.js"
    return m.group(1)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_escaped_manifest_still_parses_with_dashboard_js_extraction():
    # Run dashboard.js's own regex + JSON.parse in node over the generated page.
    page = _page(standalone=True)
    script = (
        "const html=require('fs').readFileSync(0,'utf8');"
        "const m=html.match(new RegExp(process.argv[1]));"
        "if(!m||!m[1]){console.error('no match');process.exit(2)}"
        "const o=JSON.parse(m[1]);process.stdout.write(o.items[0].note);"
    )
    out = subprocess.run([NODE, "-e", script, _dashboard_js_pattern()], input=page.encode("utf-8"),
                         capture_output=True, timeout=30)
    assert out.returncode == 0, out.stderr.decode()
    assert out.stdout.decode() == EVIL


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("standalone", [True, False])
def test_concatenated_script_parses_as_one_script(standalone):
    page = _page(standalone=standalone)
    scripts = re.findall(r"<script>([\s\S]*?)</script>", page)
    assert scripts, "no inline script found"
    js = max(scripts, key=len)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "page.js")
        with open(path, "w", encoding="utf-8") as f:
            f.write(js)
        out = subprocess.run([NODE, "--check", path], capture_output=True, timeout=60)
    assert out.returncode == 0, out.stderr.decode()


def test_dragdrop_helpers_inlined_before_logic_ui():
    html = _page()
    assert "function uscgImageRef(" in html
    assert 'id="modal-send-loadimage-btn"' in html
    assert 'draggable="true"' in html
    # node-only export guard stays harmless inline, helpers come before logic_ui's consumers
    assert html.index("function uscgImageRef(") < html.index("function sendModalToLoadImage(")


# ---- lazy shell (default) -------------------------------------------------------------------
def test_shell_inlines_no_item_data():
    page = _shell([{"id": 1, "note": "MARKER-NOTE-XYZ", "positive": "MARKER-PROMPT"}])
    assert "MARKER-NOTE-XYZ" not in page and "MARKER-PROMPT" not in page
    assert "__uscg_lazy__" in page
    assert "/*__JSON_START__*/" in page and "/*__JSON_END__*/" in page
    assert "__JSON_DATA__" not in page


def test_shell_size_does_not_grow_with_the_session():
    big = [{"id": i, "note": "x" * 2500, "positive": "p" * 300} for i in range(1000)]
    assert len(_shell(big).encode("utf-8")) < 560 * 1024
    assert len(_shell(big)) == len(_shell([]))


def test_shell_placeholders_all_replaced():
    page = html_generator.get_html_template("my-sess", {"items": []}, "42", standalone=True)
    assert 'const TARGET_NODE_ID = "42";' in page
    assert "const USCG_STANDALONE = true" in page
    assert 'id="session-input" class="session-name-input" type="text" value="my-sess"' in page
    for token in ("__TITLE__", "__NODE_ID__", "__STANDALONE_FLAG__", "__CSS_CONTENT__", "__JS_CONTENT__"):
        assert token not in page


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_shell_parses_with_dashboard_js_extraction_as_lazy():
    script = (
        "const html=require('fs').readFileSync(0,'utf8');"
        "const m=html.match(new RegExp(process.argv[1]));"
        "if(!m||!m[1]){console.error('no match');process.exit(2)}"
        "const o=JSON.parse(m[1]);process.stdout.write(JSON.stringify([o.__uscg_lazy__,o.items.length,o.session_name]));"
    )
    out = subprocess.run([NODE, "-e", script, _dashboard_js_pattern()], input=_shell(title="abc").encode("utf-8"),
                         capture_output=True, timeout=30)
    assert out.returncode == 0, out.stderr.decode()
    assert out.stdout.decode() == '[true,0,"abc"]'


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_shell_script_parses_as_one_script():
    scripts = re.findall(r"<script>([\s\S]*?)</script>", _shell(standalone=True))
    js = max(scripts, key=len)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "page.js")
        with open(path, "w", encoding="utf-8") as f:
            f.write(js)
        out = subprocess.run([NODE, "--check", path], capture_output=True, timeout=60)
    assert out.returncode == 0, out.stderr.decode()


def test_inline_manifest_flag_keeps_the_old_behaviour():
    page = _page()
    assert "__uscg_lazy__" not in page.split("/*__JSON_START__*/")[1].split("/*__JSON_END__*/")[0]
    assert '"id": 1' in page


# ---- dashboard_node: session name extraction from the generated page ---------------------------
def _load_dashboard_node():
    import sys
    import types
    sys.modules.setdefault("folder_paths", types.SimpleNamespace(get_output_directory=lambda: "."))
    pkg = sys.modules.get(_PKG) or types.ModuleType(_PKG)
    pkg.__path__ = [_ROOT]
    sys.modules[_PKG] = pkg
    sys.modules[f"{_PKG}.html_generator"] = html_generator
    spec = importlib.util.spec_from_file_location(f"{_PKG}.dashboard_node_real", os.path.join(_ROOT, "dashboard_node.py"))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("title", ["my_session", "Sexy-IL-v2"])
def test_dashboard_node_extracts_session_name_and_sets_node_id(title):
    node = _load_dashboard_node()
    page = html_generator.get_html_template(title, {"items": []}, "gen-node")
    out = node.SamplerConfigDashboardViewer().view("ignored", "55", dashboard_html=page)["ui"]
    assert out["update_session_name"] == [title]
    assert 'const TARGET_NODE_ID = "55";' in out["html"][0]
    assert 'const TARGET_NODE_ID = "gen-node";' not in out["html"][0]


def test_dashboard_node_landing_page_has_no_session_name():
    node = _load_dashboard_node()
    page = html_generator.get_html_template("", {"items": []}, "gen-node")
    out = node.SamplerConfigDashboardViewer().view("ignored", "55", dashboard_html=page)["ui"]
    assert out["update_session_name"] == []
