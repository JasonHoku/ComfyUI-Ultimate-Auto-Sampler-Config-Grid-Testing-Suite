"""Browser smoke test of the Dashboard SPA (resources/ inlined by html_generator).

Needs `pip install -r tests/e2e/requirements.txt` and `python -m playwright install chromium`.
With USCG_REQUIRE_E2E=1 (CI) a missing playwright is a failure instead of a skip.
"""
import json
import os
import re

import pytest

if os.environ.get("USCG_REQUIRE_E2E") == "1":
    import playwright.sync_api  # noqa: F401
else:
    pytest.importorskip("playwright.sync_api")

from playwright.sync_api import expect  # noqa: E402

pytestmark = pytest.mark.e2e

ACTIVE = re.compile(r"\bactive\b")
FAVORITED = re.compile(r"\bfavorited\b")


def test_cards_render(dashboard):
    page = dashboard.page
    cards = dashboard.visible_cards
    assert cards.count() == 24
    ids = {c.get_attribute("id") for c in cards.all()}
    assert ids == {f"card-{1000 + i}" for i in range(24)}
    expect(page.locator("#session-landing")).to_be_hidden()
    expect(page.locator("#viewport")).to_be_visible()
    page.wait_for_function(
        "[...document.querySelectorAll('#grid .card')].some(c => c.style.display !== 'none'"
        " && c.querySelector('img') && c.querySelector('img').naturalWidth > 0)")
    expect(page.locator("#json-btn-good")).to_have_text("View Configs (24)")


def test_favorite_toggle_updates_state_and_gold_json_bar(dashboard):
    page = dashboard.page
    star = page.locator("#card-1000 .favorite-btn")
    with page.expect_request(lambda r: r.url.endswith("/config_tester/save_changes")) as sent:
        star.click()
    change = sent.value.post_data_json["changed_items"][0]
    assert change["id"] == 1000 and change["favorited"] is True
    expect(star).to_have_class(FAVORITED)
    expect(star).to_have_text("★")
    expect(page.locator("#json-btn-favorite")).to_have_text("View Favorited (1)")
    page.locator("#json-btn-favorite").click()
    bar = page.locator("#json-bar-favorite")
    expect(bar).to_be_visible()
    entries = json.loads(bar.inner_text())
    assert len(entries) == 1
    src = next(m for m in dashboard.manifest["items"] if m["id"] == 1000)
    assert entries[0]["sampler"] == src["sampler"]
    assert entries[0]["model"] == src["model"]
    star.click()
    expect(star).to_have_text("☆")
    expect(page.locator("#json-btn-favorite")).to_have_text("View Favorited (0)")


def test_reject_hides_card(dashboard):
    page = dashboard.page
    with page.expect_request(lambda r: r.url.endswith("/config_tester/save_changes")) as sent:
        page.locator("#card-1005 .reject-btn").click()
    change = sent.value.post_data_json["changed_items"][0]
    assert change["id"] == 1005 and change["rejected"] is True
    expect(page.locator("#card-1005")).to_be_hidden()
    expect(dashboard.visible_cards).to_have_count(23)
    assert dashboard.ev("processedData.length") == 23
    expect(page.locator("#json-btn-bad")).to_have_text("View Rejected (1)")
    page.locator(".filter-popup-btn").click()
    page.locator("#top-filter-rejected").click()
    expect(page.locator("#top-filter-rejected")).to_have_class(ACTIVE)
    expect(dashboard.visible_cards).to_have_count(24)


def test_filter_group_toggle(dashboard):
    page = dashboard.page
    page.locator(".filter-popup-btn").click()
    expect(page.locator("#filters-popup")).to_be_visible()
    expect(page.locator("#filter-sampler .filter-btn")).to_have_count(3)
    euler = page.locator("#btn-sampler-euler")
    euler.click()
    expect(euler).not_to_have_class(ACTIVE)
    expect(dashboard.visible_cards).to_have_count(16)
    assert dashboard.ev("processedData.every(d => d.sampler !== 'euler')")
    euler.click()
    expect(dashboard.visible_cards).to_have_count(24)
    unipc = page.locator("#btn-sampler-unipc")
    unipc.dblclick()
    expect(dashboard.visible_cards).to_have_count(8)
    unipc.dblclick()
    expect(dashboard.visible_cards).to_have_count(24)


def test_lightbox_opens_and_arrows_navigate(dashboard):
    page = dashboard.page
    first = dashboard.ev("processedData[0]")
    pan_before = dashboard.ev("panOffsetX")
    page.locator(f"#card-{first['id']} .revise-btn").click()
    expect(page.locator("#modal")).to_be_visible()
    expect(page.locator("#modal-nav-counter")).to_have_text("1 / 24")
    assert page.locator("#m-img").get_attribute("src").endswith(first["file"])
    page.keyboard.press("ArrowRight")
    expect(page.locator("#modal-nav-counter")).to_have_text("2 / 24")
    second = dashboard.ev("processedData[1]")
    assert page.locator("#m-img").get_attribute("src").endswith(second["file"])
    page.keyboard.press("ArrowLeft")
    expect(page.locator("#modal-nav-counter")).to_have_text("1 / 24")
    page.locator(".modal-close-x").click()
    expect(page.locator("#modal")).to_be_hidden()
    assert dashboard.ev("window._modalOpen") is False
    assert dashboard.ev("panOffsetX") == pan_before


def test_sort_popup_cycles_primary_sort(dashboard):
    page = dashboard.page
    assert dashboard.ev("processedData[0].gen_index") == 0
    expect(page.locator("#sort-primary-label")).to_contain_text("Oldest")
    steps = [("Newest", 23, "newest"), ("Fastest", 0, "fastest"), ("Oldest", 0, "oldest")]
    # picking a PRIMARY option leaves the popup open (only a secondary pick closes it)
    page.locator("#sort-main-btn").click()
    expect(page.locator("#sort-popup")).to_be_visible()
    for label, gen_index, sort_id in steps:
        expect(page.locator("#sort-list-primary .sort-option")).to_have_count(dashboard.ev("sortOptionsList.length"))
        page.locator("#sort-list-primary .sort-option").get_by_text(label, exact=True).click()
        expect(page.locator("#sort-primary-label")).to_contain_text(label)
        expect(page.locator("#sort-list-primary .sort-option.selected")).to_have_text(label)
        page.wait_for_function(
            f"currentSort === '{sort_id}' && processedData[0].gen_index === {gen_index}")
        assert dashboard.ev("localStorage.getItem('ultimate_grid_sort')") == sort_id
        top = page.locator(f"#card-{dashboard.ev('processedData[0].id')}")
        expect(top).to_have_css("left", "0px")
        expect(top).to_have_css("top", "0px")
    page.locator("#sort-main-btn").click()
    expect(page.locator("#sort-popup")).to_be_hidden()


def test_shift_digit_sets_columns(dashboard):
    page = dashboard.page
    auto = dashboard.ev("columnsCount")
    page.keyboard.press("Shift+Digit3")
    page.wait_for_function("columnsCount === 3")
    expect(page.locator("#col-count")).to_have_value("3")
    assert dashboard.ev("localStorage.getItem('ultimate_grid_cols')") == "3"
    lefts = dashboard.ev("[...document.querySelectorAll('#grid .card')]"
                         ".filter(c => c.style.display !== 'none').map(c => c.style.left)")
    assert len(set(lefts)) == 3
    page.keyboard.press("Shift+Digit0")
    expect(page.locator("#col-count")).to_have_value("")
    page.wait_for_function(f"columnsCount === {auto}")


def test_update_data_message_adds_cards(dashboard):
    page = dashboard.page
    item = dashboard.make_item(24, "sdxl_base.safetensors", "euler", "normal", 9.0, 40, "None")
    page.evaluate("item => window.postMessage({type:'update_data', data:{new_items:[item]}}, '*')",
                  item)
    page.wait_for_function("processedData.length === 25")
