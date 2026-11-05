"""Dashboard JS/CSS served as bundle routes (Roadmap 27, milestone 2).

The fake origin (conftest.py `Dashboard.handle`) serves the REAL bundles through `run_api.bundle_response`.
Repeat-load caching is not testable here (page.route disables the HTTP cache): the live recipe covers it.
"""
import os

import pytest

if os.environ.get("USCG_REQUIRE_E2E") == "1":
    import playwright.sync_api  # noqa: F401
else:
    pytest.importorskip("playwright.sync_api")

from playwright.sync_api import expect  # noqa: E402

pytestmark = pytest.mark.e2e

ORIGIN = "http://uscg.test"
N_ITEMS = 24


def _setup(page, d):
    page.on("pageerror", lambda e: d.pageerrors.append(str(e)))
    page.route("**/*", d.handle)
    page.set_viewport_size({"width": 1600, "height": 1200})


def test_page_loads_one_js_and_one_css_bundle(page, dashboard_kit):
    Dashboard, make_manifest = dashboard_kit["Dashboard"], dashboard_kit["make_manifest"]
    d = Dashboard(page, make_manifest())
    _setup(page, d)
    page.goto(ORIGIN + "/")
    page.wait_for_function(f"typeof processedData!=='undefined' && processedData.length==={N_ITEMS}")
    expect(page.locator("#grid .card:visible")).to_have_count(N_ITEMS)
    hg = dashboard_kit["load_real_html_generator"]()
    assert sorted(d.bundle_requests) == sorted([("js", hg.get_dashboard_bundle("js").hash),
                                                ("css", hg.get_dashboard_bundle("css").hash)])
    assert page.evaluate("typeof uscgBootLoad") == "function"
    # a value that only report.css defines (:root --accent) is applied
    assert page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()") == "#00d1b2"
    assert page.evaluate("getComputedStyle(document.body).overflow") == "hidden"
    assert d.pageerrors == []


def test_srcdoc_iframe_loads_the_bundle_from_the_parent_origin(page, dashboard_kit):
    Dashboard, make_manifest = dashboard_kit["Dashboard"], dashboard_kit["make_manifest"]
    d = Dashboard(page, make_manifest())
    _setup(page, d)
    shell = dashboard_kit["build_dashboard_page"](d.manifest)
    wrap = '<!doctype html><body><iframe id="f" style="width:1400px;height:900px"></iframe></body>'
    page.route(ORIGIN + "/wrap", lambda route: route.fulfill(status=200, content_type="text/html", body=wrap))
    page.goto(ORIGIN + "/wrap")
    page.evaluate("(h) => { document.getElementById('f').srcdoc = h; }", shell)
    page.locator("#f").wait_for()
    page.wait_for_function("document.getElementById('f').contentWindow.processedData && "
                           f"document.getElementById('f').contentWindow.processedData.length==={N_ITEMS}")
    inner = page.frames[1]
    assert inner.evaluate("USCG_STANDALONE") is False
    assert inner.evaluate("processedData.length") == N_ITEMS
    assert {k for k, _ in d.bundle_requests} == {"js", "css"}
    assert d.pageerrors == []


def test_inline_assets_page_makes_no_bundle_requests(page, dashboard_kit):
    Dashboard, make_manifest = dashboard_kit["Dashboard"], dashboard_kit["make_manifest"]
    d = Dashboard(page, make_manifest())
    _setup(page, d)
    page.route(ORIGIN + "/inline", lambda route: route.fulfill(
        status=200, content_type="text/html; charset=utf-8",
        body=dashboard_kit["build_dashboard_page"](d.manifest, inline_assets=True)))
    page.goto(ORIGIN + "/inline")
    page.wait_for_function(f"typeof processedData!=='undefined' && processedData.length==={N_ITEMS}")
    assert d.bundle_requests == []
    assert d.pageerrors == []


def test_failing_bundle_route_does_not_hang_the_page(page, dashboard_kit):
    Dashboard, make_manifest = dashboard_kit["Dashboard"], dashboard_kit["make_manifest"]
    d = Dashboard(page, make_manifest())
    d.bundle_status = 500
    _setup(page, d)
    page.goto(ORIGIN + "/")
    page.wait_for_timeout(500)
    assert page.evaluate("typeof uscgBootLoad") == "undefined"
    assert page.evaluate("document.readyState") in ("interactive", "complete")
    assert sorted(k for k, _ in d.bundle_requests) == ["css", "js"]
