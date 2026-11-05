"""Roadmap item 12 in a real browser: the Builder's "+" LoRA strength rows, plus the Dashboard label.

The Builder UI modules (web/conf_builder/*.js) are loaded as ES modules from the fake origin. Only
conf-builder-main.js imports ComfyUI's frontend; conf-builder-config-management.js and the modules it
imports only import each other, so a tiny stub node can drive createLoraElement() directly. The preview
panel's POST /configbuilder/preview is answered by the REAL UltimateConfigBuilder.state_to_configs_json
running in the test process, so what the page shows is what Python would emit.
"""
import json
import os

import pytest

if os.environ.get("USCG_REQUIRE_E2E") == "1":
    import playwright.sync_api  # noqa: F401
else:
    pytest.importorskip("playwright.sync_api")

from playwright.sync_api import expect  # noqa: E402

from config_builder_node import UltimateConfigBuilder  # noqa: E402
from tests.test_state_to_configs_json import make_state  # noqa: E402

pytestmark = pytest.mark.e2e

ORIGIN = "http://uscg.test"
_MOD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "web", "conf_builder")

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>builder harness</title></head>
<body style="background:#111;color:#ccc">
<div id="host"></div>
<script type="module">
import * as mgmt from '/conf_builder/conf-builder-config-management.js';
import * as util from '/conf_builder/conf-builder-utilities.js';
const AVAIL = ["a.safetensors", "b.safetensors"];
const host = document.getElementById('host');
const node = {
  state: null,
  uiState: { lorasCollapsed: {}, lorasSectionCollapsed: {} },
  htmlContainer: host,
  saveState() { mgmt.updatePreview(this); },
  renderUI() {
    host.innerHTML = '<span id="json-preview-meta"></span><pre id="json-preview"></pre>' +
      '<div id="iter" data-cb-iter-count="0"></div><div id="lora-list"></div>';
    const arr = this.state.config_arrays[0];
    host.querySelector('#iter').textContent = '\\u23f1\\ufe0f Iterations: ' + util.getIterationCount(arr);
    const list = host.querySelector('#lora-list');
    arr.loras.forEach((l, i) => list.appendChild(mgmt.createLoraElement(this, l, 0, i, AVAIL, ['/'])));
    mgmt.updatePreview(this);
  },
};
window.__node = node;
window.__setup = (state) => { node.state = state; node.renderUI(); };
window.__ready = true;
</script></body></html>"""


class Builder:
    def __init__(self, page):
        self.page = page
        self.pageerrors = []
        self.preview_posts = 0

    def handle(self, route):
        req = route.request
        if not req.url.startswith(ORIGIN + "/"):
            return route.abort()
        path = req.url[len(ORIGIN):].partition("?")[0]
        if path == "/builder.html":
            return route.fulfill(status=200, content_type="text/html; charset=utf-8", body=PAGE)
        if path.startswith("/conf_builder/") and path.endswith(".js"):
            fp = os.path.join(_MOD_DIR, os.path.basename(path))
            if os.path.isfile(fp):
                with open(fp, "rb") as f:
                    return route.fulfill(status=200, content_type="application/javascript", body=f.read())
        if path == "/configbuilder/preview" and req.method == "POST":
            self.preview_posts += 1
            state = json.loads(req.post_data)["state"]
            return route.fulfill(status=200, content_type="application/json",
                                 body=json.dumps({"configs_json": UltimateConfigBuilder.state_to_configs_json(state)}))
        return route.fulfill(status=404, body="")

    def setup(self, loras, extra=None, **kw):
        state = make_state(loras=loras, extra_array_fields=extra, **kw)
        self.page.evaluate("s => window.__setup(s)", state)
        expect(self.page.locator("#lora-list .cb-item-card")).to_have_count(len(loras))
        return state

    def preview_loras(self, timeout_ms=6000):
        """The lora list the preview panel currently shows (waits for the debounced POST to land)."""
        last = None
        for _ in range(timeout_ms // 50):
            txt = self.page.locator("#json-preview").inner_text()
            if txt.strip():
                last = json.loads(txt)["configs"][0]["lora"]
                return last if isinstance(last, list) else [last]
            self.page.wait_for_timeout(50)
        raise AssertionError("preview never rendered")

    def wait_loras(self, expected, timeout_ms=6000):
        got = None
        for _ in range(timeout_ms // 100):
            txt = self.page.locator("#json-preview").inner_text()
            if txt.strip():
                got = json.loads(txt)["configs"][0]["lora"]
                got = got if isinstance(got, list) else [got]
                if got == expected:
                    return got
            self.page.wait_for_timeout(100)
        raise AssertionError(f"preview lora list {got!r} != {expected!r}")

    def card(self, i):
        return self.page.locator("#lora-list .cb-item-card").nth(i)

    def rows(self, i=0):
        return self.card(i).locator(".cb-lora-strength-row")

    def add(self, i=0):
        self.card(i).locator(".cb-lora-add-strength").click()

    def set_number(self, locator, value):
        locator.fill(str(value))
        locator.dispatch_event("change")

    def state(self):
        return self.page.evaluate("window.__node.state.config_arrays[0]")

    def iterations(self):
        return self.page.locator("#iter").inner_text()


@pytest.fixture
def builder(page):
    b = Builder(page)
    page.on("pageerror", lambda e: b.pageerrors.append(str(e)))
    page.route("**/*", b.handle)
    page.set_viewport_size({"width": 1200, "height": 1000})
    page.goto(ORIGIN + "/builder.html")
    page.wait_for_function("window.__ready === true", timeout=15000)
    yield b
    assert b.pageerrors == []


BASE = "a.safetensors:1.00:1.00"


def test_plus_adds_rows_with_one_slider_each_and_preview_follows(builder):
    builder.setup([BASE])
    builder.add()
    builder.add()
    expect(builder.rows()).to_have_count(2)
    for k in range(2):
        assert builder.rows().nth(k).locator("input[type=range]").count() == 1
    builder.set_number(builder.rows().nth(0).locator("input[type=number]"), 0.6)
    builder.set_number(builder.rows().nth(1).locator("input[type=number]"), 0.8)
    builder.wait_loras([BASE, "a.safetensors:0.60:0.60", "a.safetensors:0.80:0.80"])
    assert builder.state()["lora_extra_strengths"]["a.safetensors"] == [
        {"model": 0.6, "clip": 0.6}, {"model": 0.8, "clip": 0.8}]
    expect(builder.page.locator("#iter")).to_contain_text("Iterations: 3")
    # the old comma-separated box is gone
    assert builder.page.locator("text=Compare Strengths").count() == 0


def test_lock_off_gives_model_and_clip_sliders_per_row(builder):
    builder.setup([BASE], {"lora_strength_lock": {"a.safetensors": False}})
    builder.add()
    expect(builder.rows()).to_have_count(1)
    assert builder.rows().nth(0).locator("input[type=range]").count() == 2
    nums = builder.rows().nth(0).locator("input[type=number]")
    builder.set_number(nums.nth(0), 0.6)
    builder.set_number(nums.nth(1), 0.4)
    builder.wait_loras([BASE, "a.safetensors:0.60:0.40"])
    # switching the lock back on forces CLIP = model on the rows
    builder.card(0).get_by_text("More LoRA Options").click()
    builder.card(0).locator("label:has-text('Lock Model') input[type=checkbox]").check()
    builder.wait_loras([BASE, "a.safetensors:0.60:0.60"])
    expect(builder.rows().nth(0).locator("input[type=range]")).to_have_count(1)


def test_remove_row_and_key_is_deleted_when_empty(builder):
    builder.setup([BASE])
    builder.add()
    builder.add()
    expect(builder.rows()).to_have_count(2)
    builder.rows().nth(0).locator(".cb-lora-remove-strength").click()
    expect(builder.rows()).to_have_count(1)
    assert len(builder.state()["lora_extra_strengths"]["a.safetensors"]) == 1
    builder.rows().nth(0).locator(".cb-lora-remove-strength").click()
    expect(builder.rows()).to_have_count(0)
    assert "a.safetensors" not in (builder.state().get("lora_extra_strengths") or {})
    builder.wait_loras([BASE])
    expect(builder.page.locator("#iter")).to_contain_text("Iterations: 1")


def test_two_loras_with_two_strengths_each_give_four_stacked_entries(builder):
    builder.setup([BASE, "b.safetensors:0.50:0.50"])
    builder.add(0)
    builder.add(1)
    builder.set_number(builder.rows(0).nth(0).locator("input[type=number]"), 0.6)
    builder.set_number(builder.rows(1).nth(0).locator("input[type=number]"), 0.8)
    out = builder.wait_loras([
        "a.safetensors:1.00:1.00 + b.safetensors:0.50:0.50",
        "a.safetensors:1.00:1.00 + b.safetensors:0.80:0.80",
        "a.safetensors:0.60:0.60 + b.safetensors:0.50:0.50",
        "a.safetensors:0.60:0.60 + b.safetensors:0.80:0.80",
    ])
    assert len(out) == 4
    # the Iterations readout counts the stack as 4 entries (it used to count 2 per-file)
    expect(builder.page.locator("#iter")).to_contain_text("Iterations: 4")


def test_row_count_is_capped_at_eight_strengths(builder):
    builder.setup([BASE])
    for _ in range(7):
        builder.add()
    expect(builder.rows()).to_have_count(7)
    expect(builder.card(0).locator(".cb-lora-add-strength")).to_be_disabled()
    builder.card(0).locator(".cb-lora-add-strength").click(force=True)
    expect(builder.rows()).to_have_count(7)
    builder.rows().nth(6).locator(".cb-lora-remove-strength").click()
    expect(builder.card(0).locator(".cb-lora-add-strength")).to_be_enabled()


def test_duplicate_row_is_greyed_and_ignored(builder):
    builder.setup([BASE])
    builder.add()   # a new row starts equal to the base
    expect(builder.rows().nth(0)).to_have_attribute("data-duplicate", "1")
    builder.wait_loras([BASE])
    builder.set_number(builder.rows().nth(0).locator("input[type=number]"), 0.7)
    expect(builder.rows().nth(0)).to_have_attribute("data-duplicate", "0")
    builder.wait_loras([BASE, "a.safetensors:0.70:0.70"])


def test_old_compare_strengths_state_shows_rows_and_converts_on_first_edit(builder):
    builder.setup([BASE], {"lora_weight_arrays": {"a.safetensors_model": [0.5, 0.8, 1.0]}})
    # shown as rows right away (base slider = first old value), preview still the old bracket form
    expect(builder.rows()).to_have_count(2)
    expect(builder.card(0)).to_contain_text("converted from Compare Strengths on first edit")
    assert builder.state()["lora_weight_arrays"] == {"a.safetensors_model": [0.5, 0.8, 1.0]}
    builder.wait_loras(["a.safetensors:[0.5, 0.8, 1]"])
    # first edit converts
    builder.add()
    expect(builder.rows()).to_have_count(3)
    st = builder.state()
    assert st["lora_weight_arrays"] == {}
    assert st["loras"] == ["a.safetensors:0.50:0.50"]
    assert st["lora_extra_strengths"]["a.safetensors"][:2] == [{"model": 0.8, "clip": 0.8}, {"model": 1.0, "clip": 1.0}]
    out = builder.wait_loras(["a.safetensors:0.50:0.50", "a.safetensors:0.80:0.80", "a.safetensors:1.00:1.00"])
    assert not any("[" in s for s in out)


def test_extra_rows_follow_their_lora_when_cards_are_reordered(builder):
    builder.setup([BASE, "b.safetensors:0.50:0.50"])
    builder.add(0)
    builder.set_number(builder.rows(0).nth(0).locator("input[type=number]"), 0.6)
    builder.page.evaluate("""() => {
        const cards = document.querySelectorAll('#lora-list .cb-item-card');
        const dt = new DataTransfer();
        dt.setData('text/plain', '0');
        dt.setData('application/x-cb-list-id', 'lora-list');
        const r = cards[1].getBoundingClientRect();
        cards[1].dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true, cancelable: true, clientY: r.bottom - 1}));
    }""")
    expect(builder.card(1).locator(".cb-lora-strength-row")).to_have_count(1)
    assert builder.state()["loras"][0].startswith("b.safetensors")
    expect(builder.rows(0)).to_have_count(0)
    assert builder.state()["lora_extra_strengths"] == {"a.safetensors": [{"model": 0.6, "clip": 0.6}]}
    builder.wait_loras(["b.safetensors:0.50:0.50 + a.safetensors:1.00:1.00",
                        "b.safetensors:0.50:0.50 + a.safetensors:0.60:0.60"])


def test_deleting_a_card_drops_its_rows(builder):
    builder.setup([BASE, "b.safetensors:0.50:0.50"])
    builder.add(0)
    builder.card(0).locator(".cb-header-bar button.danger").click()
    expect(builder.page.locator("#lora-list .cb-item-card")).to_have_count(1)
    assert "a.safetensors" not in (builder.state().get("lora_extra_strengths") or {})


# ---- Dashboard: a strength sweep keeps its label in "unique only" mode -----------------------------

def test_dashboard_label_keeps_strength_when_one_lora_is_swept(page, dashboard_kit):
    from tests.e2e.conftest import N_ITEMS  # noqa: F401  (same fake origin / manifest builder as the dashboard fixture)
    manifest = dashboard_kit["make_manifest"]()
    for i, item in enumerate(manifest["items"]):
        item["lora"] = "x.safetensors:0.60:0.60" if i % 2 else "x.safetensors:1.00:1.00"
    d = dashboard_kit["Dashboard"](page, manifest)
    page.set_viewport_size({"width": 1600, "height": 1200})
    page.on("pageerror", lambda e: d.pageerrors.append(str(e)))
    page.route("**/*", d.handle)
    page.goto(dashboard_kit["ORIGIN"] + "/")
    page.wait_for_function(f"typeof processedData!=='undefined' && processedData.length==={N_ITEMS}")
    html = page.evaluate("""() => {
        labelMode.enabled = true; labelMode.fields.lora = true; labelMode.fields.loraUniqueOnly = true;
        computeLabelGlobalValues();
        const sweep = buildLabelOverlay(activeData.find(x => x.lora === 'x.safetensors:0.60:0.60'));
        // same LoRA, ONE strength string on every card -> still hidden as "global"
        activeData.forEach(x => { x.lora = 'x.safetensors:1.00:1.00'; });
        computeLabelGlobalValues();
        const same = buildLabelOverlay(activeData[0]);
        return {sweep, same};
    }""")
    assert "x:0.60:0.60" in html["sweep"]
    assert "x:" not in html["same"]
    assert d.pageerrors == []
