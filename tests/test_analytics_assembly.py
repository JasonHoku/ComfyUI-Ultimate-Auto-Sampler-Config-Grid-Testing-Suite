"""The Stats panel files are wired into the assembled dashboard page in the right order."""
import importlib.util
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _real_html_generator():
    spec = importlib.util.spec_from_file_location("uscg_real_html_generator_an", os.path.join(_ROOT, "html_generator.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_page_has_panel_logic_in_order_and_header_button():
    page = _real_html_generator().get_html_template("ci", {"items": []}, "1")
    assert 'id="analytics-open-btn"' in page
    a = page.index("function computeAnalytics(")
    b = page.index("function openAnalyticsPanel(")
    assert a < b
    assert page.index("function pipelineMatches(") < b


def test_js_load_order():
    src = open(os.path.join(_ROOT, "html_generator.py"), encoding="utf-8").read()
    m = re.search(r"js_files\s*=\s*\[(.*?)\n\s*\]", src, re.S)
    names = re.findall(r'"(logic_[a-z_]+\.js)"', m.group(1))
    assert names.index("logic_ratings.js") < names.index("logic_analytics.js")
    assert names.index("logic_wfparams.js") < names.index("logic_analytics.js")
    assert names.index("logic_analytics.js") < names.index("logic_ui.js")
    assert names.index("logic_analytics_panel.js") > names.index("logic_compare.js")
    assert names.index("logic_analytics_panel.js") < names.index("logic_standalone.js")
    for n in names:
        assert os.path.isfile(os.path.join(_ROOT, "resources", n))
