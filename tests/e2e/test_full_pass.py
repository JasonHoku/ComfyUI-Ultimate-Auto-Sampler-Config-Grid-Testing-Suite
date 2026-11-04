"""Dashboard 'Full pass for survivors' via the run API (POST /config_tester/run) on the fake origin."""
import json

import pytest

SESSION = "smoke"  # matches tests/e2e/conftest.py

_CONFIGS = [{"model": "sdxl_base.safetensors", "sampler": "euler", "scheduler": "normal",
             "steps": 20, "cfg": 7.0, "lora": "None"}]
_META = {
    "configs_json": json.dumps(_CONFIGS),
    "session_settings": {"preview_pass": {"enabled": True, "steps": 8, "scale": 0.5},
                         "add_random_seeds_to_gens": 2, "session_name": SESSION},
    "has_optional_inputs": False,
}


def _run_posts(d):
    return [b for (m, p, b) in d.request_log if m == "POST" and p == "/config_tester/run"]


def _toast_text(page):
    page.wait_for_function("[...document.querySelectorAll('div')].some(e => e.style.zIndex === '99999')")
    return page.evaluate("[...document.querySelectorAll('div')].filter(e => e.style.zIndex === '99999')"
                         ".map(e => e.textContent).join('|')")


@pytest.mark.parametrize("manifest_meta", [_META], indirect=True)
def test_full_pass_uses_run_api(dashboard):
    page = dashboard.page
    page.route("**/config_tester/run", lambda route: (
        dashboard.request_log.append(("POST", "/config_tester/run", json.loads(route.request.post_data)))
        or route.fulfill(status=200, content_type="application/json", body='{"prompt_id":"abcd1234-0000"}')))
    page.evaluate("runFullPassForSurvivors()")
    toast = _toast_text(page)
    assert "Full pass queued" in toast
    posts = _run_posts(dashboard)
    assert len(posts) == 1
    body = posts[0]
    ss = body["configs_json"]["_session_settings"]
    assert ss["preview_pass"]["enabled"] is False
    assert ss["skip_rejected_previews"] is True
    assert ss["add_random_seeds_to_gens"] == 2
    assert body["overwrite_existing"] is False
    assert body["force"] is True
    assert body["session_name"] == SESSION
    assert body["configs_json"]["configs"] == _CONFIGS


def test_full_pass_without_recorded_settings_posts_nothing(dashboard):
    dashboard.page.evaluate("runFullPassForSurvivors()")
    toast = _toast_text(dashboard.page)
    assert "before the hub recorded" in toast or "no recorded configs" in toast
    assert _run_posts(dashboard) == []


@pytest.mark.parametrize("manifest_meta", [{"configs_json": json.dumps(_CONFIGS)}], indirect=True)
def test_full_pass_old_session_without_settings(dashboard):
    dashboard.page.evaluate("runFullPassForSurvivors()")
    assert "before the hub recorded" in _toast_text(dashboard.page)
    assert _run_posts(dashboard) == []
