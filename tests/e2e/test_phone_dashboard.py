"""Phone-sized dashboard checks (Pixel 7 emulation) plus the desktop guard for the same features.

Uses the `phone_dashboard` factory fixture from conftest.py (fake origin, no ComfyUI).
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


def _open_lightbox(d):
    first = d.ev("processedData[0]")
    d.page.locator(f"#card-{first['id']} .revise-btn").tap()
    expect(d.page.locator("#modal")).to_be_visible()
    return first


# ---- E1: stacked phone lightbox -------------------------------------------

def test_phone_lightbox_is_stacked_and_usable(phone_dashboard):
    d = phone_dashboard()
    page = d.page
    _open_lightbox(d)
    assert page.evaluate("getComputedStyle(document.querySelector('#modal .modal-main')).flexDirection") == "column"
    m = page.evaluate("""() => ({sw: document.getElementById('modal').scrollWidth, iw: window.innerWidth,
        ih: window.innerHeight, imgH: document.getElementById('m-img').getBoundingClientRect().height,
        fs: getComputedStyle(document.getElementById('f-pos')).fontSize})""")
    assert m["sw"] <= m["iw"], m
    assert m["imgH"] <= 0.62 * m["ih"], m
    assert m["fs"] == "16px", m

    # six action buttons, each a real touch target
    widths = page.evaluate("[...document.querySelectorAll('#modal-action-bar button')].map(b => b.getBoundingClientRect().width)")
    assert len(widths) == 6 and all(w >= 44 for w in widths), widths
    labels = page.evaluate("[...document.querySelectorAll('#modal-action-bar .mab-lbl')].map(b => b.textContent)")
    assert labels == ["Prev", "Favorite", "Reject", "Compare", "Revise", "Next"]

    # Revise brings the form into view
    page.locator("#modal-action-bar button", has_text="Revise").tap()
    page.wait_for_function(
        "() => { const r = document.querySelector('#modal .controls').getBoundingClientRect();"
        " return r.top < window.innerHeight && r.bottom > 0; }")
    # the last button of the form is not hidden behind the bar
    gen = page.locator("#modal button", has_text="GENERATE NEW")
    gen.scroll_into_view_if_needed()
    covered_by = page.evaluate("""() => { const b = [...document.querySelectorAll('#modal button')]
        .find(x => x.textContent.includes('GENERATE NEW')); const r = b.getBoundingClientRect();
        const el = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
        return el === b || b.contains(el); }""")
    assert covered_by

    # Favorite toggles the label and the (re)opened lightbox shows the state
    fav = page.locator("#modal-fav-btn")
    expect(fav.locator(".mab-lbl")).to_have_text("Favorite")
    fav.tap()
    expect(fav.locator(".mab-lbl")).to_have_text("Favorited")
    expect(fav.locator(".mab-ico")).to_have_text("★")
    expect(fav).to_have_class("favorited")
    page.locator("#modal-action-bar button", has_text="Next").tap()
    page.locator("#modal-action-bar button", has_text="Prev").tap()
    expect(fav.locator(".mab-lbl")).to_have_text("Favorited")   # openM syncs the state


# ---- E2: desktop guard -----------------------------------------------------

def test_desktop_is_unchanged(dashboard):
    page = dashboard.page
    first = dashboard.ev("processedData[0]")
    page.locator(f"#card-{first['id']} .revise-btn").click()
    expect(page.locator("#modal")).to_be_visible()
    assert page.evaluate("getComputedStyle(document.querySelector('#modal .modal-main')).flexDirection") == "row"
    expect(page.locator("#modal-action-bar")).to_be_hidden()
    page.locator(".modal-close-x").click()
    # the standalone-only header buttons exist on the top-level page
    expect(page.locator("#share-link-btn")).to_be_visible()
    page.evaluate("toggleCogMenu()")
    width = page.evaluate("document.getElementById('cog-menu-dropdown').getBoundingClientRect().width")
    assert 340 <= width <= 342, width   # unchanged desktop panel (340px + border)


# ---- E3: share link --------------------------------------------------------

STUB_SHARE = "window.__shared = []; navigator.share = (a) => { window.__shared.push(a); return Promise.resolve(); };"


def test_share_button_uses_navigator_share(phone_dashboard):
    d = phone_dashboard(init_scripts=[STUB_SHARE])
    d.page.locator("#share-link-btn").tap()
    d.page.wait_for_function("window.__shared.length === 1")
    assert d.ev("window.__shared[0].url").endswith("/config_tester/dashboard?session=smoke")


def test_share_button_falls_back_to_copy(phone_dashboard):
    # plain http origin = not a secure context: the execCommand path (no share API in desktop-ish contexts)
    d = phone_dashboard(init_scripts=["delete Navigator.prototype.share; window.__copied = 0;"
                                      "const ex = document.execCommand.bind(document);"
                                      "document.execCommand = (c) => { if (c === 'copy') { window.__copied++; return true; } return ex(c); };"])
    d.page.locator("#share-link-btn").tap()
    d.page.wait_for_function("window.__copied === 1")
    expect(d.page.locator("#share-link-btn")).to_have_text("✓")


# ---- E4: session picker ----------------------------------------------------

def test_session_picker_switches_session(phone_dashboard):
    sessions = [{"name": n, "mtime": 1700000000 + i, "item_count": 5 + i, "first_image": ""}
                for i, n in enumerate(("smoke", "other", "third"))]
    d = phone_dashboard(sessions=sessions)
    page = d.page
    page.locator("#session-switch-btn").tap()
    rows = page.locator("#session-picker-list .session-picker-row")
    expect(rows).to_have_count(3)
    heights = [r.bounding_box()["height"] for r in rows.all()]
    assert all(h >= 44 for h in heights), heights
    rows.filter(has_text="other").locator("span").first.tap()
    page.wait_for_function("window._uscgLoadedSession === 'other'")
    assert page.evaluate("location.search") == "?session=other"
