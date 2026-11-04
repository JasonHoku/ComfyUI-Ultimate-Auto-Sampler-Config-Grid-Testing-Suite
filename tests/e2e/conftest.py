"""Fixtures for the Dashboard SPA browser smoke test.

The page is the REAL html_generator output (loaded under a private module name,
because tests/conftest.py registers a stub `html_generator`), served from a fake
origin through `page.route`, so no web server or ComfyUI is involved.
"""
import importlib.util
import io
import json
import os

import pytest

_NODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORIGIN = "http://uscg.test"
SESSION = "smoke"
N_ITEMS = 24


def load_real_html_generator():
    spec = importlib.util.spec_from_file_location(
        "uscg_real_html_generator", os.path.join(_NODE_ROOT, "html_generator.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_dashboard_page(manifest, node_id="7"):
    """The exact HTML string the Dashboard node shows in its iframe srcdoc."""
    return load_real_html_generator().get_html_template(SESSION, manifest, node_id)


def make_item(i, model, sampler, scheduler, cfg, steps, lora):
    prompt = f"a photo of a cat, variant {i % 2}"
    return {
        "id": 1000 + i,
        "gen_index": i,
        "file": f"/view?filename=smoke_{i:04d}.png&type=output&subfolder=benchmarks/{SESSION}",
        "favorited": False,
        "rejected": False,
        "note": "",
        "positive": prompt,
        "config_positive": prompt,
        "negative": "blurry",
        "model": model,
        "lora": lora,
        "sampler": sampler,
        "scheduler": scheduler,
        "steps": steps,
        "cfg": cfg,
        "seed": 100 + i,
        "denoise": 1.0,
        "width": 512,
        "height": 768,
        "duration": round(1 + 0.1 * i, 2),
        "batch_idx": 0,
    }


def make_manifest():
    """24 items, every config tuple unique. Newest first, like the real manifest."""
    built = []
    i = 0
    for model in ("sdxl_base.safetensors", "juggernautXL_v9.safetensors"):
        for sampler in ("euler", "dpmpp_2m", "uni_pc"):
            for scheduler in ("normal", "karras"):
                for cfg in (4.0, 7.0):
                    steps = 20 if i % 2 == 0 else 30
                    lora = "None" if i % 4 < 2 else "detail_tweaker.safetensors:0.8"
                    built.append(make_item(i, model, sampler, scheduler, cfg, steps, lora))
                    i += 1
    return {
        "session_name": SESSION,
        "meta": {"model": "", "positive": "", "negative": ""},
        "items": list(reversed(built)),
    }


_PNG_CACHE = {}


def png_bytes(name):
    if name not in _PNG_CACHE:
        from PIL import Image
        h = (sum(ord(c) * (k + 7) for k, c in enumerate(name)) * 2654435761) & 0xFFFFFF
        img = Image.new("RGB", (64, 96), ((h >> 16) & 255, (h >> 8) & 255, h & 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        _PNG_CACHE[name] = buf.getvalue()
    return _PNG_CACHE[name]


class Dashboard:
    """The page plus a request log and a few helpers."""

    def __init__(self, page, manifest):
        self.page = page
        self.manifest = manifest
        self.request_log = []      # (method, path, parsed_json_body_or_None)
        self.sessions = []         # payload of /config_tester/list_sessions
        self.comparisons = []      # votes echoed back by /config_tester/save_comparisons
        self.pageerrors = []

    make_item = staticmethod(make_item)

    def ev(self, js):
        return self.page.evaluate(js)

    @property
    def visible_cards(self):
        # Pooled cards stay in the DOM with display:none: always filter on :visible.
        return self.page.locator("#grid .card:visible")

    def handle(self, route):
        req = route.request
        if not req.url.startswith(ORIGIN + "/"):
            return route.abort()   # e.g. the GitHub stars fetch
        body = None
        if req.method == "POST":
            try:
                body = json.loads(req.post_data or "null")
            except ValueError:
                body = None
        path, _, query = req.url[len(ORIGIN):].partition("?")
        self.request_log.append((req.method, path, body))
        if path == "/":
            return route.fulfill(status=200, content_type="text/html; charset=utf-8",
                                 body=build_dashboard_page(self.manifest))
        if path == "/view":
            params = dict(kv.partition("=")[::2] for kv in query.split("&") if kv)
            name = params.get("filename", "")
            if name == "manifest.json":
                return route.fulfill(status=200, content_type="application/json",
                                     body=json.dumps(self.manifest))
            return route.fulfill(status=200, content_type="image/png", body=png_bytes(name))
        if path in ("/config_tester/save_changes", "/config_tester/save_manifest"):
            return route.fulfill(status=200, content_type="application/json",
                                 body='{"status":"ok"}')
        if path == "/config_tester/list_sessions":
            return route.fulfill(status=200, content_type="application/json", body=json.dumps(self.sessions))
        if path == "/config_tester/save_comparisons":
            body = body or {}
            drop = set(body.get("remove") or [])
            self.comparisons = [c for c in self.comparisons
                                if f"{c.get('ts')}:{c.get('a')}:{c.get('b')}" not in drop]
            self.comparisons += list(body.get("append") or [])
            return route.fulfill(status=200, content_type="application/json",
                                 body=json.dumps({"comparisons": self.comparisons,
                                                  "total": len(self.comparisons)}))
        return route.fulfill(status=200, content_type="application/json", body="{}")


@pytest.fixture
def dashboard(page):
    from playwright.sync_api import expect
    d = Dashboard(page, make_manifest())
    page.set_viewport_size({"width": 1600, "height": 1200})
    page.on("pageerror", lambda e: d.pageerrors.append(str(e)))
    page.route("**/*", d.handle)
    page.goto(ORIGIN + "/")
    page.wait_for_function(
        f"typeof processedData!=='undefined' && processedData.length==={N_ITEMS}")
    expect(page.locator("#grid .card:visible")).to_have_count(N_ITEMS)
    yield d
    assert d.pageerrors == []


@pytest.fixture
def phone_dashboard(browser, playwright):
    """Factory: the same fake-origin dashboard in a Pixel 7 context (touch, coarse pointer, 412 px wide).

    `phone_dashboard(manifest=None, sessions=None, init_scripts=(), n_items=N_ITEMS)` returns a
    Dashboard; init scripts run before the page loads. Contexts are closed at teardown.
    """
    contexts, dashboards = [], []

    def make(manifest=None, sessions=None, init_scripts=(), n_items=N_ITEMS):
        from playwright.sync_api import expect
        ctx = browser.new_context(**playwright.devices["Pixel 7"])
        contexts.append(ctx)
        for js in init_scripts:
            ctx.add_init_script(js)
        page = ctx.new_page()
        d = Dashboard(page, manifest or make_manifest())
        d.sessions = sessions or []
        dashboards.append(d)
        page.on("pageerror", lambda e: d.pageerrors.append(str(e)))
        loaded = []
        page.on("console", lambda m: loaded.append(1) if "Session loaded successfully" in m.text else None)
        page.route("**/*", d.handle)
        page.goto(ORIGIN + "/")
        page.wait_for_function(
            f"typeof processedData!=='undefined' && processedData.length==={n_items}")
        expect(page.locator("#grid .card:visible").first).to_be_visible()
        for _ in range(100):   # the startup loadSession() closes the settings panel when it finishes
            if loaded:
                break
            page.wait_for_timeout(50)
        return d

    yield make
    for d in dashboards:
        assert d.pageerrors == []
    for ctx in contexts:
        ctx.close()
