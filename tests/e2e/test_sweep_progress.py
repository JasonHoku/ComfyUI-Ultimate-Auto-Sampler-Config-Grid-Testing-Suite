"""Dashboard ETA bar across a mixed run: standard jobs complete -> sweep next -> sweep jobs; no stale hide timer."""
import os

import pytest

if os.environ.get("USCG_REQUIRE_E2E") == "1":
    import playwright.sync_api  # noqa: F401
else:
    pytest.importorskip("playwright.sync_api")

from playwright.sync_api import expect  # noqa: E402

pytestmark = pytest.mark.e2e


def post(page, data):
    page.evaluate("d => window.postMessage({type: 'progress_update', data: d}, '*')", data)


COMPLETE = {"complete": True, "total_generated": 4, "total_elapsed": "0h 1m 2s", "avg_duration": 15.5,
            "current_job": 4, "total_jobs": 4, "progress_pct": 100}
SWEEP = {"phase": "sweep", "current_job": 1, "total_jobs": 2, "progress_pct": 50, "eta_str": "30s",
         "avg_duration": 30, "finish_time": "12:00:00"}


def test_complete_with_next_phase_keeps_the_bar_and_names_the_sweep(dashboard):
    page = dashboard.page
    post(page, dict(COMPLETE, next_phase="sweep"))
    expect(page.locator("#eta-bar")).to_be_visible()
    expect(page.locator("#eta-text")).to_contain_text("sweep next")
    expect(page.locator("#eta-text")).to_contain_text("Standard jobs complete: 4 images")
    # no hide timer was armed: even after a long wait the bar stays (fake the 30 s with a real short check)
    assert dashboard.ev("typeof _etaHideTimer") != "undefined"
    assert dashboard.ev("_etaHideTimer") is None
    post(page, SWEEP)
    expect(page.locator("#eta-text")).to_contain_text("Sweep job 1/2")


def test_plain_complete_arms_the_timer_and_a_later_event_cancels_it(dashboard):
    page = dashboard.page
    post(page, COMPLETE)
    expect(page.locator("#eta-text")).to_contain_text("Complete! 4 images")
    assert dashboard.ev("_etaHideTimer") is not None
    post(page, SWEEP)                                  # a sweep that starts < 30 s after a run
    assert dashboard.ev("_etaHideTimer") is None
    expect(page.locator("#eta-bar")).to_be_visible()
    expect(page.locator("#eta-text")).to_contain_text("Sweep job 1/2")
