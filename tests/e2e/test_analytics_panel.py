"""Browser test of the docked Settings Stats panel (resources/logic_analytics*.js inlined by html_generator).

Needs `pip install -r tests/e2e/requirements.txt` and `python -m playwright install chromium`.
With USCG_REQUIRE_E2E=1 (CI) a missing playwright is a failure instead of a skip.
"""
import os

import pytest

if os.environ.get("USCG_REQUIRE_E2E") == "1":
    import playwright.sync_api  # noqa: F401
else:
    pytest.importorskip("playwright.sync_api")

from playwright.sync_api import expect  # noqa: E402

pytestmark = pytest.mark.e2e


def _axis(page, axis):
    return page.locator(f'#analytics-panel details[data-axis="{axis}"]')


def _row_cells(row):
    return [c.strip() for c in row.locator("td").all_inner_texts()]


def test_panel_groups_and_metrics(dashboard):
    page = dashboard.page
    dashboard.ev("activeData.forEach((d, i) => { d.metrics = {sharpness: 100 + i, sobel: 0.01 * (i % 5)};"
                 " d.favorited = d.sampler === 'euler'; }); updateDataPipeline();")
    page.locator("#analytics-open-btn").click()
    expect(page.locator("#analytics-panel")).to_be_visible()
    first = _axis(page, "sampler").locator("tbody tr").first
    cells = _row_cells(first)
    assert cells[0] == "euler" and cells[1] == "8" and cells[2] == "100.0"
    head = _axis(page, "sampler").locator("thead").inner_text().lower()   # CSS upper-cases the header
    assert "sobel" in head and "sharpness" in head
    assert _axis(page, "model").locator("tbody tr").count() == 2


def test_panel_live_updates(dashboard):
    page = dashboard.page
    page.locator("#analytics-open-btn").click()
    expect(page.locator("#analytics-summary")).to_contain_text("24 items")
    page.locator("#card-1000 .favorite-btn").click()
    expect(page.locator("#analytics-summary")).to_contain_text("1 favorited")
    page.locator("#card-1005 .reject-btn").dispatch_event("click")   # the docked panel may cover the card
    dpm = _axis(page, "sampler").locator("tbody tr", has_text="dpmpp_2m")
    expect(dpm.locator("td").nth(3)).to_have_text("12.5")      # counted although the grid hides it
    expect(dashboard.visible_cards).to_have_count(23)
    item = dashboard.make_item(24, "sdxl_base.safetensors", "euler", "normal", 9.0, 40, "None")
    page.evaluate("item => window.postMessage({type:'update_data', data:{new_items:[item]}}, '*')", item)
    expect(page.locator("#analytics-summary")).to_contain_text("25 items")


def test_panel_scope_markdown_escape(dashboard):
    page = dashboard.page
    page.locator("#analytics-open-btn").click()
    dashboard.ev("filters.sampler.clear(); filters.sampler.add('euler'); updateDataPipeline();")
    expect(page.locator("#analytics-summary")).to_contain_text("8 items")
    page.select_option("#analytics-scope", "all")
    expect(page.locator("#analytics-summary")).to_contain_text("24 items")
    md = dashboard.ev("analyticsPanelMarkdown()")
    assert md.startswith("## USCG stats")
    assert "| euler |" in md
    page.locator("#analytics-copy-md").click()
    expect(page.locator("#analytics-copy-md")).to_have_text("Copied")
    page.locator("#analytics-collapse-btn").click()
    expect(page.locator("#analytics-panel .ap-body")).to_be_hidden()
    expect(page.locator("#analytics-panel .ap-head")).to_be_visible()
    page.locator("#analytics-collapse-btn").click()
    expect(page.locator("#analytics-panel .ap-body")).to_be_visible()
    page.keyboard.press("Escape")
    expect(page.locator("#analytics-panel")).to_be_hidden()


def test_panel_cog_entry_and_manifest_text_is_not_html(dashboard):
    page = dashboard.page
    dashboard.ev("activeData[0].sampler = '<img src=x onerror=window.__pwned=1>'; updateDataPipeline();")
    dashboard.ev("openAnalyticsPanel()")
    expect(page.locator("#analytics-panel")).to_be_visible()
    assert dashboard.ev("window.__pwned") is None
    assert page.locator("#analytics-panel img").count() == 0


def test_panel_fits_a_phone(phone_dashboard):
    d = phone_dashboard()
    d.ev("openAnalyticsPanel()")
    box = d.page.locator("#analytics-panel").bounding_box()
    assert box["width"] <= 412 and box["x"] >= 0
