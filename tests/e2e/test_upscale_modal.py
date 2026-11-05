"""Dashboard upscale modal (Roadmap 93): inline editing, validation, presets, run once, cancel, reattach, phone.

Fake origin, no ComfyUI: the upscale / preset routes are mocked through `Dashboard.routes`.
"""
import copy
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

MODEL = "4x-UltraSharpV2_Lite.safetensors"


def _step(**kw):
    s = {"active": True, "mode": "hires_only", "repeat": 1, "upscale_models": [], "upscale_ratios": "1.5",
         "upscale_size": "2.0", "hires_denoise": "0.3", "hires_steps": 0}
    s.update(kw)
    return s


def _preset(name, *steps):
    return {"name": name, "pipelines": [{"active": True, "name": "Pipeline 1", "steps": list(steps)}],
            "hires_prompt_adjust": False, "hires_prompt_behavior": "append_end", "hires_prompt_text": ""}


class UpscaleMock:
    """Registers the routes the modal talks to and logs what it sends."""

    def __init__(self, d, presets=None):
        self.d = d
        self.presets = copy.deepcopy(presets or [])
        self.events = []            # ("GET"|"POST", "presets") in order
        self.preset_posts = []
        self.start_bodies = []
        self.cancel_bodies = []
        self.reject = False
        self.status_seq = [{"status": "running", "completed": 1, "total": 3, "outputs": 2,
                            "current_image": "a.webp", "current_step": "Pipeline 1 · step 1/1 · combo 1/2", "error": None}]
        self.after_cancel = [{"status": "cancelling", "completed": 1, "total": 3, "outputs": 2, "current_step": "", "error": None},
                             {"status": "cancelled", "completed": 1, "total": 3, "outputs": 2, "current_step": "", "error": None}]
        self.jobs = []
        d.routes["/configbuilder/upscale_presets"] = self._presets
        d.routes["/configbuilder/model_lists"] = lambda r, q, b, qs: self._json(
            r, {"upscale_models": [MODEL, "RealESRGAN_x4plus.pth"]})
        d.routes["/config_tester/upscale_images"] = self._start
        d.routes["/config_tester/upscale_status"] = self._status
        d.routes["/config_tester/cancel_upscale"] = self._cancel
        d.routes["/config_tester/upscale_jobs"] = lambda r, q, b, qs: self._json(r, {"jobs": self.jobs})

    @staticmethod
    def _json(route, obj, status=200):
        return route.fulfill(status=status, content_type="application/json", body=json.dumps(obj))

    def _presets(self, route, req, body, query):
        if req.method == "POST":
            self.events.append(("POST", "presets"))
            self.preset_posts.append(body)
            self.presets = body["presets"]
            return self._json(route, {"status": "ok"})
        self.events.append(("GET", "presets"))
        return self._json(route, {"presets": self.presets})

    def _start(self, route, req, body, query):
        self.start_bodies.append(body)
        if self.reject:
            return self._json(route, {"error": "Invalid upscale config", "errors": ["p0.s0: Denoise must be between 0 and 1"]}, 400)
        return self._json(route, {"job_id": "upscale_t1", "total_images": 3})

    def _status(self, route, req, body, query):
        cur = self.status_seq[0]
        if len(self.status_seq) > 1:
            self.status_seq.pop(0)
        return self._json(route, cur)

    def _cancel(self, route, req, body, query):
        self.cancel_bodies.append(body)
        self.status_seq = list(self.after_cancel)
        return self._json(route, {"cancelled": True, "status": "cancelling"})


def _open(d):
    d.page.evaluate("openUpscaleModal(processedData[0].id)")
    expect(d.page.locator("#upscale-start-btn")).to_be_visible()


def _fresh(d):
    # the modal keeps its working config across opens and in localStorage: start every test clean
    d.page.evaluate("try { localStorage.removeItem('uscg_upscale_modal_cfg'); } catch (e) {}"
                    " upscaleModalState.currentConfig = null; upscaleModalState.loadedPresetName = null;"
                    " upscaleModalState.jobId = null; upscaleModalState.running = false; upscaleModalState.lastStatus = null;")


def _card(d, pipe, step):
    return d.page.locator(f'#upscale-modal-content [data-pipe="{pipe}"][data-step="{step}"]')


# ---- 1: edit, validate, reorder, run once ------------------------------------------------------------------
def test_edit_reorder_and_run_once(dashboard):
    d = dashboard
    m = UpscaleMock(d, [_preset("A", _step())])
    _fresh(d)
    _open(d)
    page = d.page
    page.select_option("#upscale-preset-select", label="A")
    expect(page.locator("#upscale-preset-state")).to_have_text("Preset: A")
    expect(page.locator("#upscale-start-btn")).to_have_text("Run")
    expect(page.locator("#upscale-output-summary")).to_contain_text("1 image(s)")
    expect(page.locator("#upscale-output-summary")).to_contain_text("1 output(s) = 1 new images")

    page.get_by_text("+ Add Step").click()
    expect(page.locator("#upscale-preset-state")).to_have_text("Preset: A (modified)")
    expect(page.locator("#upscale-start-btn")).to_have_text("Run once (unsaved)")
    _card(d, 0, 1).locator('select[data-field="mode"]').select_option("model_only")
    expect(page.locator("#upscale-start-btn")).to_be_disabled()
    expect(page.locator("#upscale-validation .upm-err")).to_contain_text("upscale model")
    expect(_card(d, 0, 1).locator('[data-field="upscale_models"]')).to_have_class(re.compile("upm-invalid"))
    assert page.evaluate("document.querySelectorAll('#upscale-modal-content .upm-invalid').length") >= 1

    search = _card(d, 0, 1).get_by_placeholder("Search upscale models...")
    search.click()
    search.fill("Ultra")
    page.get_by_text(MODEL).click()
    expect(page.locator("#upscale-start-btn")).to_be_enabled()
    assert page.evaluate("document.querySelectorAll('#upscale-modal-content .upm-invalid').length") == 0

    _card(d, 0, 1).locator('[data-act="step-up"]').click()
    assert page.evaluate("upscaleModalState.currentConfig.pipelines[0].steps.map(s => s.mode)") == ["model_only", "hires_only"]
    expect(_card(d, 0, 0).locator('select[data-field="mode"]')).to_have_value("model_only")

    page.locator("#upscale-start-btn").click()
    page.wait_for_function("upscaleModalState.jobId === 'upscale_t1'")
    assert len(m.start_bodies) == 1
    body = m.start_bodies[0]
    assert body["upscale_config"]["pipelines"][0]["steps"][0]["mode"] == "model_only"
    assert body["image_ids"] == [d.ev("processedData[0].id")] and body["all_favorited"] is False
    assert m.preset_posts == []                       # run once never touches the presets file
    assert m.presets[0]["pipelines"][0]["steps"][0]["mode"] == "hires_only"


def test_server_validation_errors_are_shown(dashboard):
    d = dashboard
    m = UpscaleMock(d, [_preset("A", _step())])
    m.reject = True
    _fresh(d)
    _open(d)
    d.page.select_option("#upscale-preset-select", label="A")
    d.page.locator("#upscale-start-btn").click()
    expect(d.page.locator("#upscale-validation")).to_contain_text("Denoise must be between 0 and 1")
    expect(d.page.locator("#upscale-start-btn")).to_be_enabled()
    assert d.ev("upscaleModalState.jobId") is None


def test_bad_ratio_marks_the_field_and_blocks_run(dashboard):
    d = dashboard
    UpscaleMock(d, [_preset("A", _step())])
    _fresh(d)
    _open(d)
    d.page.select_option("#upscale-preset-select", label="A")
    field = _card(d, 0, 0).locator('input[data-field="upscale_ratios"]')
    field.fill("1.5,x")
    field.dispatch_event("change")
    expect(field).to_have_class(re.compile("upm-invalid"))
    expect(d.page.locator("#upscale-start-btn")).to_be_disabled()
    field.fill("1.5, 2")
    field.dispatch_event("change")
    expect(d.page.locator("#upscale-start-btn")).to_be_enabled()


# ---- 2: save as + update = read-modify-write ------------------------------------------------------------------
def test_update_and_save_as_read_before_write(dashboard):
    d = dashboard
    m = UpscaleMock(d, [_preset("A", _step())])
    _fresh(d)
    _open(d)
    page = d.page
    page.select_option("#upscale-preset-select", label="A")
    expect(page.locator("#upscale-preset-update")).to_be_disabled()
    page.get_by_text("+ Add Step").click()
    expect(page.locator("#upscale-preset-update")).to_be_enabled()

    # somebody else (the Builder, another tab) adds a preset meanwhile: Update must not clobber it
    m.presets.append(_preset("C", _step(hires_denoise="0.5")))
    m.events.clear()
    page.locator("#upscale-preset-update").click()
    page.wait_for_function("upscaleModalState.loadedPresetSnapshot && upscaleModalState.loadedPresetSnapshot.pipelines[0].steps.length === 2")
    assert m.events == [("GET", "presets"), ("POST", "presets")]
    assert [p["name"] for p in m.presets] == ["A", "C"]
    assert len(m.presets[0]["pipelines"][0]["steps"]) == 2
    expect(page.locator("#upscale-preset-state")).to_have_text("Preset: A")
    expect(page.locator("#upscale-start-btn")).to_have_text("Run")

    # Save as a new name keeps the others and switches the loaded preset
    page.get_by_text("+ Add Step").click()
    m.events.clear()
    page.locator("#upscale-preset-saveas").click()
    page.locator("#upscale-preset-name").fill("B")
    page.locator("#upscale-preset-saveas-ok").click()
    page.wait_for_function("upscaleModalState.loadedPresetName === 'B'")
    assert m.events == [("GET", "presets"), ("POST", "presets")]
    assert [p["name"] for p in m.presets] == ["A", "C", "B"]
    assert len(m.presets[2]["pipelines"][0]["steps"]) == 3 and len(m.presets[0]["pipelines"][0]["steps"]) == 2
    expect(page.locator("#upscale-preset-state")).to_have_text("Preset: B")

    # Revert drops edits; Delete (confirm) removes only the loaded preset
    page.get_by_text("+ Add Step").click()
    expect(page.locator("#upscale-preset-state")).to_have_text("Preset: B (modified)")
    page.locator("#upscale-preset-revert").click()
    expect(page.locator("#upscale-preset-state")).to_have_text("Preset: B")
    assert page.evaluate("upscaleModalState.currentConfig.pipelines[0].steps.length") == 3
    page.once("dialog", lambda dlg: dlg.accept())
    page.locator("#upscale-preset-delete").click()
    page.wait_for_function("upscaleModalState.loadedPresetName === null")
    assert [p["name"] for p in m.presets] == ["A", "C"]
    expect(page.locator("#upscale-preset-state")).to_have_text("Unsaved config")


# ---- 3: cancel ------------------------------------------------------------------------------------------------------
def test_cancel_flow(dashboard):
    d = dashboard
    m = UpscaleMock(d, [_preset("A", _step())])
    _fresh(d)
    _open(d)
    page = d.page
    page.select_option("#upscale-preset-select", label="A")
    expect(page.locator("#upscale-cancel-job-btn")).to_be_hidden()
    page.locator("#upscale-start-btn").click()
    expect(page.locator("#upscale-modal-progress-text")).to_contain_text("Upscaling 1/3")
    expect(page.locator("#upscale-modal-progress-text")).to_contain_text("Pipeline 1")
    expect(page.locator("#upscale-editor-fieldset")).to_have_js_property("disabled", True)
    expect(page.locator("#upscale-cancel-job-btn")).to_be_visible()
    expect(page.locator("#upscale-close-btn")).to_be_enabled()

    page.locator("#upscale-cancel-job-btn").click()
    expect(page.locator("#upscale-modal-progress-text")).to_contain_text("Cancelling")
    expect(page.locator("#upscale-modal-progress-text")).to_contain_text("Cancelled: 1/3 images done, 2 new images kept", timeout=10000)
    assert m.cancel_bodies == [{"job_id": "upscale_t1"}]
    expect(page.locator("#upscale-cancel-job-btn")).to_be_hidden()
    expect(page.locator("#upscale-close-btn")).to_be_enabled()
    expect(page.locator("#upscale-close-btn")).to_have_text("Close")
    expect(page.locator("#upscale-editor-fieldset")).to_have_js_property("disabled", False)
    expect(page.locator("#upscale-start-btn")).to_be_enabled()
    page.locator("#upscale-close-btn").click()
    expect(page.locator("#upscale-modal-overlay")).to_be_hidden()


# ---- 4: close + reopen reattaches --------------------------------------------------------------------------------------
def test_close_and_reopen_reattaches_to_the_running_job(dashboard):
    d = dashboard
    m = UpscaleMock(d, [_preset("A", _step())])
    _fresh(d)
    _open(d)
    page = d.page
    page.select_option("#upscale-preset-select", label="A")
    page.locator("#upscale-start-btn").click()
    expect(page.locator("#upscale-modal-progress-text")).to_contain_text("Upscaling 1/3")
    page.locator("#upscale-close-btn").click()
    expect(page.locator("#upscale-modal-overlay")).to_be_hidden()
    assert d.ev("upscaleModalState.pollInterval !== null && upscaleModalState.jobId") == "upscale_t1"   # still polling

    _open(d)
    expect(page.locator("#upscale-modal-progress-text")).to_contain_text("Upscaling 1/3")
    expect(page.locator("#upscale-cancel-job-btn")).to_be_visible()
    expect(page.locator("#upscale-editor-fieldset")).to_have_js_property("disabled", True)
    expect(page.locator("#upscale-start-btn")).to_be_disabled()
    # the job ends while the modal is open: the (re-rendered) elements update, not detached ones
    m.status_seq = [{"status": "complete", "completed": 3, "total": 3, "outputs": 3, "current_step": "", "error": None}]
    expect(page.locator("#upscale-modal-progress-text")).to_contain_text("Complete! 3/3 images upscaled, 3 new images.", timeout=10000)
    expect(page.locator("#upscale-editor-fieldset")).to_have_js_property("disabled", False)
    expect(page.locator("#upscale-cancel-job-btn")).to_be_hidden()


def test_reload_reattaches_through_the_jobs_route(dashboard):
    d = dashboard
    m = UpscaleMock(d, [_preset("A", _step())])
    m.jobs = [{"job_id": "upscale_t1", "status": "running", "completed": 1, "total": 3}]
    _fresh(d)
    _open(d)
    expect(d.page.locator("#upscale-modal-progress-text")).to_contain_text("Upscaling 1/3")
    expect(d.page.locator("#upscale-cancel-job-btn")).to_be_visible()


# ---- 5: phone ---------------------------------------------------------------------------------------------------------------
def test_phone_modal_fits_and_is_touchable(phone_dashboard):
    d = phone_dashboard()
    UpscaleMock(d, [_preset("A", _step(), _step(mode="model_only", upscale_models=[MODEL]))])
    _fresh(d)
    _open(d)
    page = d.page
    page.select_option("#upscale-preset-select", label="A")
    m = page.evaluate("""() => { const c = document.getElementById('upscale-modal-content');
        const btns = [...c.querySelectorAll('button')].filter(b => b.offsetParent !== null);
        const inputs = [...c.querySelectorAll('input[type=text], input[type=number], select, textarea')].filter(b => b.offsetParent !== null);
        return { sw: c.scrollWidth, cw: c.clientWidth, iw: window.innerWidth, bodySw: document.documentElement.scrollWidth,
                 w: c.getBoundingClientRect().width,
                 smallBtns: btns.filter(b => { const r = b.getBoundingClientRect(); return r.width < 43.5 || r.height < 43.5; }).map(b => b.id || b.textContent),
                 smallFonts: inputs.filter(i => getComputedStyle(i).fontSize !== '16px').map(i => i.id || i.name || i.tagName) }; }""")
    assert m["sw"] <= m["cw"] + 1, m
    assert m["bodySw"] <= m["iw"] + 1, m
    assert m["w"] <= m["iw"] + 1, m
    assert m["smallBtns"] == [], m
    assert m["smallFonts"] == [], m
    reach = page.evaluate("""() => { const b = document.getElementById('upscale-start-btn'); b.scrollIntoView({block: 'nearest'});
        const r = b.getBoundingClientRect(); const el = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
        return { visible: r.top >= 0 && r.bottom <= window.innerHeight, hit: el === b || b.contains(el) }; }""")
    assert reach == {"visible": True, "hit": True}, reach
    # reorder works with a tap
    _card(d, 0, 1).locator('[data-act="step-up"]').tap()
    assert page.evaluate("upscaleModalState.currentConfig.pipelines[0].steps.map(s => s.mode)") == ["model_only", "hires_only"]


# ---- 6: desktop layout unchanged ---------------------------------------------------------------------------------------------
def test_desktop_modal_is_still_600px(dashboard):
    d = dashboard
    UpscaleMock(d, [_preset("A", _step())])
    _fresh(d)
    _open(d)
    w = d.page.evaluate("getComputedStyle(document.getElementById('upscale-modal-content')).width")
    assert w == "600px", w          # the inline 600px content box, as before
    assert d.page.evaluate("getComputedStyle(document.getElementById('upscale-modal-content')).borderRadius") == "8px"
