"""Revise-modal image drag / "Send to Load Image" / standalone download (issue #22), no ComfyUI needed.

The real dashboard page is served from the fake origin (see conftest.py). Top level = standalone mode; a
tiny wrapper page with the dashboard in an iframe plays the ComfyUI parent for the postMessage bridge.
"""
import json
import os

import pytest

if os.environ.get("USCG_REQUIRE_E2E") == "1":
    import playwright.sync_api  # noqa: F401
else:
    pytest.importorskip("playwright.sync_api")

from playwright.sync_api import expect  # noqa: E402

ORIGIN = "http://uscg.test"  # same as conftest.ORIGIN (asserted in the wrapper test)

pytestmark = pytest.mark.e2e

DRAGSTART_JS = """() => {
  const dt = new DataTransfer();
  const ev = new DragEvent('dragstart', {dataTransfer: dt, bubbles: true, cancelable: true});
  const notCancelled = document.getElementById('m-img').dispatchEvent(ev);
  return {notCancelled, types: [...dt.types], files: dt.files.length,
          info: dt.getData('application/x-comfy-asset-info'), uri: dt.getData('text/uri-list'),
          plain: dt.getData('text/plain')};
}"""


def test_standalone_modal_image_is_draggable_and_fills_carriers(dashboard):
    page = dashboard.page
    first = dashboard.ev("processedData[0]")
    page.locator(f"#card-{first['id']} .revise-btn").click()
    expect(page.locator("#modal")).to_be_visible()
    assert page.locator("#m-img").get_attribute("draggable") == "true"
    r = dashboard.ev(DRAGSTART_JS)
    assert r["notCancelled"] is True
    assert {"application/x-comfy-asset-info", "text/uri-list", "text/plain"} <= set(r["types"])
    assert r["files"] == 0                      # reference mode: no File unless USCG_DRAG_UPLOAD
    info = json.loads(r["info"])
    assert info["filename"] == first["file"].split("filename=")[1].split("&")[0]
    assert info["subfolder"] == "benchmarks/smoke" and info["type"] == "output"
    assert r["uri"] == ORIGIN + first["file"] == r["plain"]


def test_standalone_button_downloads_the_image(dashboard):
    page = dashboard.page
    first = dashboard.ev("processedData[0]")
    page.locator(f"#card-{first['id']} .revise-btn").click()
    btn = page.locator("#modal-send-loadimage-btn")
    expect(btn).to_have_text("\u2B07 Download image")
    with page.expect_download() as dl:
        btn.click()
    assert dl.value.suggested_filename == first["file"].split("filename=")[1].split("&")[0]


def test_video_item_is_not_draggable_and_button_hidden(dashboard):
    page = dashboard.page
    dashboard.ev("(() => { const d = activeData.find(x => x.id === processedData[0].id); d.media_type = 'video'; })()")
    dashboard.ev("openM(processedData[0].id)")
    assert page.locator("#m-img").get_attribute("draggable") == "false"
    expect(page.locator("#modal-send-loadimage-btn")).to_be_hidden()


WRAP = """<!doctype html><body><iframe id="f" src="/" style="width:1400px;height:900px"></iframe><script>
window.got = [];
window.addEventListener('message', function (ev) {
  if (!ev.data || ev.data.type !== 'uscg_send_to_load_image') return;
  window.got.push(ev.data);
  ev.source.postMessage({type: 'uscg_send_to_load_image_result', ok: true, message: 'Loaded into "Load Image" (#1)'}, '*');
});
</script></body>"""


def test_send_button_posts_reference_to_parent_and_toasts(page, dashboard_kit):
    assert dashboard_kit["ORIGIN"] == ORIGIN
    d = dashboard_kit["Dashboard"](page, dashboard_kit["make_manifest"]())
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("**/*", d.handle)
    page.route(ORIGIN + "/wrap", lambda route: route.fulfill(status=200, content_type="text/html", body=WRAP))
    page.set_viewport_size({"width": 1500, "height": 950})
    page.goto(ORIGIN + "/wrap")
    fr = page.frame_locator("#f")
    fr.locator("#grid .card:visible").first.wait_for()
    first = page.frames[1].evaluate("processedData[0]")
    fr.locator(f"#card-{first['id']} .revise-btn").click()
    btn = fr.locator("#modal-send-loadimage-btn")
    expect(btn).to_have_text("\U0001F5BC Send to Load Image")
    btn.click()
    page.wait_for_function("window.got.length === 1")
    msg = page.evaluate("window.got[0]")
    assert msg["filename"] == first["file"].split("filename=")[1].split("&")[0]
    assert msg["subfolder"] == "benchmarks/smoke" and msg["imageType"] == "output"
    assert msg["upload"] is False and msg["node_id"] == "7"
    assert msg["url"] == ORIGIN + first["file"]
    expect(fr.locator("body")).to_contain_text('Loaded into "Load Image" (#1)')
    assert errors == []
