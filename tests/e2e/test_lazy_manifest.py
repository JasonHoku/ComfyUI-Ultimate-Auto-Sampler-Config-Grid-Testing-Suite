"""Lazy manifest loading: the page is a shell, the SPA fetches /config_tester/manifest itself.

Fake origin (see conftest.py: `Dashboard.handle` serves the route with ETag W/"v<n>" and 304).
"""
import json
import os

import pytest

if os.environ.get("USCG_REQUIRE_E2E") == "1":
    import playwright.sync_api  # noqa: F401
else:
    pytest.importorskip("playwright.sync_api")

from playwright.sync_api import expect  # noqa: E402

pytestmark = pytest.mark.e2e

ORIGIN = "http://uscg.test"
V1 = 'W/"v1"'


def _open(page, kit, title="smoke", known=None, hold=False, goto=True):
    """A Dashboard on the fake origin, not yet waited on. Returns (d, errors, dialogs)."""
    d = kit["Dashboard"](page, kit["make_manifest"]())
    d.title = title
    d.known_sessions = known
    d.hold_manifest = hold
    errors, dialogs = [], []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("dialog", lambda dlg: (dialogs.append(dlg.message), dlg.dismiss()))
    page.route("**/*", d.handle)
    page.set_viewport_size({"width": 1600, "height": 1200})
    if goto:
        page.goto(ORIGIN + "/")
    return d, errors, dialogs


def _extra_item(d, item_id=5000, **overrides):
    """A new item that introduces no new filter value (a clone of a non-euler item with a fresh id)."""
    src = next(it for it in d.manifest["items"] if it["sampler"] != "euler")
    item = dict(src, id=item_id, **overrides)
    return item


def _wait_requests(page, d, n):
    for _ in range(80):
        if len(d.manifest_requests) >= n:
            return
        page.wait_for_timeout(100)
    raise AssertionError(f"expected {n} manifest requests, saw {d.manifest_requests}")


# ---- (a) one manifest request on boot, and it is the new route -----------------------------------

def test_boot_does_one_manifest_request_and_no_view_manifest(dashboard):
    assert dashboard.manifest_requests == [{"session": "smoke", "if_none_match": None}]
    assert dashboard.view_manifest_hits == 0
    assert dashboard.ev("fullManifest.__uscg_lazy__") is None   # replaced by the fetched manifest


# ---- (b) landing shell (empty title) does no manifest request ---------------------------------------

def test_landing_shell_does_not_fetch(page, dashboard_kit):
    d, errors, dialogs = _open(page, dashboard_kit, title="")
    expect(page.locator("#session-landing")).to_be_visible()
    page.wait_for_timeout(400)
    assert d.manifest_requests == [] and d.view_manifest_hits == 0
    assert errors == [] and dialogs == []


# ---- (c) unknown session: landing page, no alert, no errors ---------------------------------------

def test_unknown_session_shows_landing_without_alert(page, dashboard_kit):
    d, errors, dialogs = _open(page, dashboard_kit, title="ghost", known={"smoke"})
    expect(page.locator("#session-landing")).to_be_visible()
    assert [r["session"] for r in d.manifest_requests] == ["ghost"]
    assert page.evaluate("window._uscgBootPending") is False
    assert errors == [] and dialogs == []


# ---- (d) slow fetch: no landing flash, 'c' is safe, cards arrive afterwards ----------------------------

def test_slow_manifest_no_landing_flash_and_compare_key_is_safe(page, dashboard_kit):
    d, errors, dialogs = _open(page, dashboard_kit, hold=True)
    page.wait_for_function("window._uscgBootPending === true")
    page.wait_for_timeout(600)
    assert len(d.held) == 1
    assert page.evaluate("getComputedStyle(document.getElementById('session-landing')).display") == "none"
    page.keyboard.press("c")
    assert page.evaluate("document.getElementById('cmp-status').textContent") == "Loading session..."
    assert not page.evaluate("!!window._compareOpen")
    d.release_manifest()
    page.wait_for_function("typeof processedData!=='undefined' && processedData.length===24")
    expect(page.locator("#grid .card:visible")).to_have_count(24)
    page.wait_for_function("window._uscgBootPending === false")
    assert errors == [] and dialogs == []


def test_live_items_during_boot_fetch_are_kept(page, dashboard_kit):
    d, errors, _ = _open(page, dashboard_kit, hold=True)
    page.wait_for_function("window._uscgBootPending === true")
    extra = _extra_item(d, 7777)
    page.evaluate("(it) => window.postMessage({type: 'update_data', data: {new_items: [it]}}, '*')", extra)
    page.wait_for_timeout(200)
    d.release_manifest()
    page.wait_for_function("typeof processedData!=='undefined' && processedData.length===25")
    assert page.evaluate("activeData.some(x => x.id === 7777)")
    assert errors == []


# ---- (e) standalone resync: conditional, silent on 304, filter-preserving on 200 ----------------------------

def test_resync_sends_if_none_match_and_keeps_view(dashboard):
    page = dashboard.page
    page.locator(".filter-popup-btn").click()
    page.locator("#btn-sampler-euler").click()
    expect(dashboard.visible_cards).to_have_count(16)
    dashboard.ev("toggleCogMenu()")
    cog_open = dashboard.ev("getComputedStyle(document.getElementById('cog-menu-dropdown')).display")
    assert cog_open != "none"
    assert len(dashboard.manifest_requests) == 1

    page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
    _wait_requests(page, dashboard, 2)
    assert dashboard.manifest_requests[1]["if_none_match"] == V1
    page.wait_for_timeout(300)
    expect(dashboard.visible_cards).to_have_count(16)           # 304: nothing re-rendered
    assert dashboard.ev("getComputedStyle(document.getElementById('cog-menu-dropdown')).display") == cog_open

    # the manifest changes: the next resync is a 200, the new card shows and the euler filter is still on
    dashboard.manifest["items"].insert(0, _extra_item(dashboard))
    dashboard.manifest_version = 2
    page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
    page.wait_for_function("typeof processedData!=='undefined' && processedData.length===17")
    expect(dashboard.visible_cards).to_have_count(17)
    assert dashboard.ev("processedData.every(d => d.sampler !== 'euler')")
    assert dashboard.ev("getComputedStyle(document.getElementById('cog-menu-dropdown')).display") == cog_open
    assert dashboard.manifest_requests[-1]["if_none_match"] == V1

    # an item with a NEW filter value (a seed nobody had) must not be hidden by the stale filter sets
    dashboard.manifest["items"].insert(0, _extra_item(dashboard, 5002, seed=987654))
    dashboard.manifest_version = 3
    page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
    page.wait_for_function("typeof processedData!=='undefined' && processedData.some(d => d.id === 5002)")


# ---- (f) in the node iframe: reload_manifest from the parent, legacy update_data still works ----------------------

WRAP = """<!doctype html><body><iframe id="f" src="/" style="width:1400px;height:900px"></iframe></body>"""
POST_RELOAD = "document.getElementById('f').contentWindow.postMessage({type: 'reload_manifest'}, '*')"


def test_reload_manifest_message_and_legacy_update_data(page, dashboard_kit):
    d, errors, dialogs = _open(page, dashboard_kit, goto=False)
    page.route(ORIGIN + "/wrap", lambda route: route.fulfill(status=200, content_type="text/html", body=WRAP))
    page.goto(ORIGIN + "/wrap")
    fr = page.frame_locator("#f")
    fr.locator("#grid .card:visible").first.wait_for()
    inner = page.frames[1]
    inner.wait_for_function("typeof processedData!=='undefined' && processedData.length===24")
    assert inner.evaluate("USCG_STANDALONE") is False

    fr.locator(".filter-popup-btn").click()
    fr.locator("#btn-sampler-euler").click()
    inner.wait_for_function("processedData.length===16")

    # unchanged manifest: reload_manifest is a conditional request answered with 304 (nothing changes)
    page.evaluate(POST_RELOAD)
    _wait_requests(page, d, 2)
    assert d.manifest_requests[-1]["if_none_match"] == V1
    page.wait_for_timeout(200)
    assert inner.evaluate("processedData.length") == 16

    # new item on disk: reload_manifest brings it in, filter kept
    d.manifest["items"].insert(0, _extra_item(d))
    d.manifest_version = 2
    page.evaluate(POST_RELOAD)
    inner.wait_for_function("processedData.length===17")
    assert inner.evaluate("processedData.every(x => x.sampler !== 'euler')")

    # legacy full-manifest push still works
    full = json.loads(json.dumps(d.manifest))
    full["items"].insert(0, _extra_item(d, 5001))
    page.evaluate("(m) => document.getElementById('f').contentWindow.postMessage("
                  "{type: 'update_data', data: {manifest: m, meta: m.meta}}, '*')", full)
    inner.wait_for_function("processedData.length===18")

    # an (empty) lazy shell manifest must never blank the grid
    page.evaluate("document.getElementById('f').contentWindow.postMessage("
                  "{type: 'update_data', data: {manifest: {items: [], meta: {}, __uscg_lazy__: true}}}, '*')")
    page.wait_for_timeout(300)
    assert inner.evaluate("processedData.length") == 18
    assert errors == [] and dialogs == []


# ---- (g) auto-fit ran on real data: a card is inside the viewport ----------------------------------------------

def test_autofit_puts_cards_in_view_without_saved_position(dashboard):
    page = dashboard.page
    page.wait_for_timeout(300)
    inside = page.evaluate("""() => {
        const v = document.getElementById('viewport').getBoundingClientRect();
        return [...document.querySelectorAll('#grid .card')].filter(c => c.offsetParent !== null).some(c => {
            const r = c.getBoundingClientRect();
            return r.width > 0 && r.bottom > v.top && r.top < v.bottom && r.right > v.left && r.left < v.right;
        });
    }""")
    assert inside
    assert dashboard.ev("window._uscgViewportRestored") is False
