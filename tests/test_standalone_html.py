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
    manifest = {"items": [{"id": 1, "note": EVIL}], "meta": {}}
    return html_generator.get_html_template("t", manifest, "standalone", **kw)


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
