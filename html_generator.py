import os
import json
import gzip
import hashlib
import threading
from typing import NamedTuple

RESOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")

# JS load order (dependencies first!). logic_standalone.js must stay LAST.
JS_FILES = (
    "logic_state.js",    # Global variables
    "logic_utils.js",    # API & Helper functions
    "logic_ratings.js",  # Pairwise review: pure logic (normalisers, Bradley-Terry, pairing, scoreboard)
    "logic_wfparams.js",  # Workflow Sweep: wf_params helpers (pure; must load before logic_ui/logic_pipeline)
    "logic_analytics.js",  # Settings stats: computeAnalytics/analyticsToMarkdown (pure; after ratings + wfparams)
    "logic_dragdrop.js", # Revise-modal image drag/send helpers (issue #22; pure, node-testable)
    "logic_ui.js",       # DOM Creation, Modals, JSON Bars
    "logic_upscale_modal.js",  # Upscale Modal (preset management, async upscale)
    "logic_virtual.js",  # The Virtual Scroll Engine & Viewport
    "logic_pipeline.js", # Sorting & Filtering Logic
    "logic_compare.js",  # Pairwise review: compare overlay, votes, scoreboard modal
    "logic_analytics_panel.js",  # Settings stats panel (DOM; uses cmpEl/cmpCopyText/cmpRatings from logic_compare)
    "logic_events.js",   # Event Listeners (Message, Keyboard)
    "logic_init.js",     # Initialization
    "logic_standalone.js"  # Standalone-page mode (own websocket, run API); must stay LAST
)
CSS_FILES = ("report.css",)

# Fixed bundle routes (no path parameters; the ?v= query is only compared, never used for lookup).
BUNDLE_ROUTES = {"js": "/config_tester/dashboard_bundle_js", "css": "/config_tester/dashboard_bundle_css"}
BUNDLE_CONTENT_TYPES = {"js": "text/javascript", "css": "text/css"}


class Bundle(NamedTuple):
    kind: str
    sig: tuple
    text: str
    body: bytes
    gzip: bytes
    hash: str


_bundle_cache = {}
_bundle_lock = threading.Lock()


def _files_for(kind):
    if kind == "js":
        return JS_FILES
    if kind == "css":
        return CSS_FILES
    raise KeyError(kind)


def _signature(names):
    """(name, mtime_ns, size) per resource file: cheap change detection for hot reload."""
    sig = [("dir", RESOURCE_DIR, 0)]
    for n in names:
        st = os.stat(os.path.join(RESOURCE_DIR, n))
        sig.append((n, st.st_mtime_ns, st.st_size))
    return tuple(sig)


def _build_text(kind):
    names = _files_for(kind)
    if kind == "css":
        with open(os.path.join(RESOURCE_DIR, names[0]), "r", encoding="utf-8") as f:
            return f.read()
    text = ""
    for n in names:
        with open(os.path.join(RESOURCE_DIR, n), "r", encoding="utf-8") as f:
            text += f"\n/* --- {n} --- */\n"
            text += f.read() + "\n"
    return text


def get_dashboard_bundle(kind):
    """The page-independent JS or CSS bundle, rebuilt only when a resource file changed.

    Thread-safe (called from the aiohttp loop and from the execution thread). Raises KeyError for
    an unknown kind and OSError when a resource file is missing.
    """
    names = _files_for(kind)
    with _bundle_lock:
        sig = _signature(names)
        cached = _bundle_cache.get(kind)
        if cached is not None and cached.sig == sig:
            return cached
        text = _build_text(kind)
        body = text.encode("utf-8")
        bundle = Bundle(kind, sig, text, body, gzip.compress(body, 9, mtime=0),
                        hashlib.sha1(body).hexdigest()[:20])
        _bundle_cache[kind] = bundle
        return bundle


def bundle_url(kind):
    return f"{BUNDLE_ROUTES[kind]}?v={get_dashboard_bundle(kind).hash}"


def get_html_template(title, manifest_data, node_id, standalone=False, inline_manifest=False, inline_assets=None):
    """Build the dashboard page.

    By default the page is a SHELL: no item data is inlined (an empty `fullManifest` flagged
    `__uscg_lazy__`), and the SPA fetches `/config_tester/manifest?session=<title>` itself. That
    keeps the HTML (and ComfyUI's `executed` websocket message / history entry, which carries the
    whole string) ~500 KB instead of growing with the session. `manifest_data` is only used when
    `inline_manifest=True` (old behaviour; offline export / tests).

    The JS/CSS are NOT inlined by default: the page references two cached bundle routes
    (`bundle_url`), which cuts the shell to ~70 KB. `inline_assets=True` (implied by
    `inline_manifest=True`, or env USCG_INLINE_DASHBOARD_ASSETS=1) inlines the same bytes instead.
    """
    # 1. Normalize Data
    if not inline_manifest:
        manifest_data = {"items": [], "meta": {"model": "", "positive": "", "negative": ""},
                         "session_name": str(title), "__uscg_lazy__": True}
    elif isinstance(manifest_data, list):
        manifest_data = {"items": manifest_data, "meta": {"model": "", "positive": "", "negative": ""}}
    
    # Indent JSON for readability
    json_str = json.dumps(manifest_data, indent=2)
    # The manifest is inlined inside a <script> element: a note/prompt containing "</script>" would
    # end it early (page break / script injection). "<\/" is an equivalent JSON escape, so
    # JSON.parse (web/dashboard.js's incremental refresh) still reads the original text.
    json_str = json_str.replace("</", "<\\/")

    # 2. Load template + assets
    if inline_assets is None:
        inline_assets = bool(inline_manifest)
    if os.environ.get("USCG_INLINE_DASHBOARD_ASSETS") == "1":
        inline_assets = True

    try:
        with open(os.path.join(RESOURCE_DIR, "template.html"), "r", encoding="utf-8") as f:
            html_template = f.read()
        js_bundle = get_dashboard_bundle("js")
        css_bundle = get_dashboard_bundle("css")
    except (FileNotFoundError, OSError) as e:
        return f"<h1>Error: Resource files not found.</h1><p>{e}</p>"

    if inline_assets:
        css_tag = f"<style>\n/* INJECTED CSS */\n{css_bundle.text}\n</style>"
        js_tag = f"<script>\n/* INJECTED LOGIC */\n{js_bundle.text}\n</script>"
    else:
        css_tag = f'<link rel="stylesheet" href="{bundle_url("css")}">'
        js_tag = f'<script src="{bundle_url("js")}"></script>'

    # 3. Inject Data
    wrapped_json = f"/*__JSON_START__*/\nlet fullManifest = {json_str};\n/*__JSON_END__*/"

    # 4. Final Replacements (asset tags LAST so resource text is never placeholder-substituted)
    final_html = html_template         .replace("__TITLE__", str(title))         .replace("__NODE_ID__", str(node_id))         .replace("__STANDALONE_FLAG__", "true" if standalone else "false")         .replace("let fullManifest = __JSON_DATA__;", wrapped_json)         .replace("__CSS_TAG__", css_tag)         .replace("__JS_TAG__", js_tag)

    return final_html
