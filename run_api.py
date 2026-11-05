"""Pure helpers for the headless run API (POST /config_tester/run, GET /config_tester/run_status).

No aiohttp / ComfyUI imports at module level so the helpers are unit-testable on their own.
The routes themselves live in __init__.py.
"""
import json
import os
import re
from collections import OrderedDict

DEFAULT_SESSION = "my_session"  # the Generator node's own default (sampler_node.py)
_MAX_TRACKED_RUNS = 200


def sanitize_session_name_strict(name):
    """Same regex as delete_session / save_manifest / get_session_html. '' when nothing is left."""
    return re.sub(r'[^\w\-]', '', str(name or ''))


def _etag_matches(if_none_match, etag):
    """RFC 9110 weak comparison of an If-None-Match header against our ETag ('*' matches anything)."""
    if not if_none_match:
        return False
    want = etag[2:] if etag.startswith("W/") else etag
    for part in str(if_none_match).split(","):
        part = part.strip()
        if part == "*":
            return True
        if part.startswith("W/"):
            part = part[2:]
        if part == want:
            return True
    return False


def bundle_response(bundle, v, if_none_match, accept_encoding):
    """HTTP shaping for the dashboard JS/CSS bundle routes: (status, body_or_None, headers).

    `v` (the ?v= query) is only COMPARED with the bundle hash, never used to look anything up: a
    stale or bogus value still gets the current bundle, just with a no-cache policy.
    """
    etag = 'W/"%s"' % bundle.hash
    headers = {
        "ETag": etag,
        "Vary": "Accept-Encoding",
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "public, max-age=31536000, immutable" if v == bundle.hash else "no-cache",
    }
    if _etag_matches(if_none_match, etag):
        return 304, None, headers
    if "gzip" in str(accept_encoding or "").lower():
        headers["Content-Encoding"] = "gzip"
        return 200, bundle.gzip, headers
    return 200, bundle.body, headers


def load_manifest_for_fetch(benchmarks_base, raw_session, if_none_match=None):
    """Backend of GET /config_tester/manifest?session=<name> (the dashboard SPA's data fetch).

    Returns (status, body_bytes_or_None, headers). One attempt only: status 503 means a torn read
    (the file changed while we read it) or invalid JSON (save_manifest is not atomic), which the
    route retries a few times before answering. The body is the manifest re-serialised COMPACT
    (the on-disk file is indent=4, ~1.9x bigger). ETag = weak validator from mtime_ns + size.
    """
    session = sanitize_session_name_strict(raw_session)
    if not session:
        return 400, json.dumps({"error": "missing or invalid session"}).encode(), {}
    base = os.path.realpath(benchmarks_base)
    path = os.path.join(base, session, "manifest.json")
    real = os.path.realpath(path)
    if not real.startswith(base + os.sep):
        return 403, json.dumps({"error": "forbidden"}).encode(), {}
    try:
        st1 = os.stat(real)
    except OSError:
        return 404, json.dumps({"error": "not found"}).encode(), {}
    etag = 'W/"%x-%x"' % (st1.st_mtime_ns, st1.st_size)
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if _etag_matches(if_none_match, etag):
        return 304, None, headers
    try:
        with open(real, "r", encoding="utf-8") as f:
            data = json.load(f)
        st2 = os.stat(real)
    except (OSError, ValueError):
        return 503, json.dumps({"error": "manifest is being written, retry"}).encode(), {"Retry-After": "1"}
    if (st2.st_mtime_ns, st2.st_size) != (st1.st_mtime_ns, st1.st_size):
        return 503, json.dumps({"error": "manifest changed while reading, retry"}).encode(), {"Retry-After": "1"}
    if isinstance(data, dict):
        data.pop("__uscg_lazy__", None)  # heal files corrupted by a SAVE on the lazy shell
    body = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return 200, body, headers


def normalize_configs_payload(configs_json):
    """Accept a dict with a `configs` list, a bare list, or a JSON string of either.

    Returns a dict that always has a non-empty `configs` list; other top-level keys
    (`_distribution`, `_session_settings`, ...) are preserved. Raises ValueError otherwise.
    """
    obj = configs_json
    if isinstance(obj, (str, bytes)):
        try:
            obj = json.loads(obj)
        except (TypeError, ValueError) as e:
            raise ValueError(f"configs_json is not valid JSON: {e}")
    if isinstance(obj, list):
        obj = {"configs": obj}
    if not isinstance(obj, dict):
        raise ValueError("configs_json must be an object with a 'configs' list, a list of configs, or a JSON string of either")
    configs = obj.get("configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError("configs_json must contain a non-empty 'configs' list")
    return obj


def apply_run_overrides(obj, session_name=None, overwrite_existing=None, nonce=None):
    """Apply run-level overrides. Returns (obj, sanitised_session_name).

    Session precedence: explicit session_name, then _session_settings.session_name, then the
    Generator default. The sanitised name is written back to _session_settings.session_name
    because that is what the Generator actually reads. `nonce` goes into an inert top-level
    key; it changes IS_CHANGED's hash so ComfyUI's node cache cannot short-circuit the run.
    """
    settings = obj.get("_session_settings")
    if not isinstance(settings, dict):
        settings = {}
        obj["_session_settings"] = settings
    chosen = session_name or settings.get("session_name") or DEFAULT_SESSION
    session = sanitize_session_name_strict(chosen)
    settings["session_name"] = session
    if overwrite_existing is not None:
        settings["overwrite_existing"] = bool(overwrite_existing)
    if nonce is not None:
        obj["_run_nonce"] = str(nonce)
    return obj, session


def build_run_prompt(configs_str, session_name):
    """The 2-node API-format prompt: Generator -> Dashboard (which owns the output)."""
    return {
        "1": {"class_type": "UltimateSamplerGrid", "inputs": {"configs_json": configs_str}},
        "2": {"class_type": "UltimateGridDashboard",
              "inputs": {"session_name": session_name, "dashboard_html": ["1", 0]}},
    }


def resolve_saved_config(configs_dir, name):
    """Mirror of the /configbuilder/load_config path handling. Returns the file path.

    Raises FileNotFoundError (missing) or PermissionError (escapes configs_dir).
    """
    filename = str(name or "")
    if not filename.endswith(".json"):
        filename += ".json"
    filename = os.path.basename(filename)
    filepath = os.path.join(configs_dir, filename)
    real_path = os.path.realpath(filepath)
    real_base = os.path.realpath(configs_dir)
    if not (real_path == real_base or real_path.startswith(real_base + os.sep)):
        raise PermissionError("config path escapes the configs directory")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"config '{filename}' not found")
    return filepath


def saved_config_to_configs_obj(data, state_to_configs_json):
    """Saved Builder files hold the Builder STATE; convert via the single source of truth."""
    if isinstance(data, dict):
        if "config_arrays" in data:
            return json.loads(state_to_configs_json(data))
        if "configs" in data:
            return data
    raise ValueError("saved config is neither a Builder state (config_arrays) nor a configs JSON (configs)")


def summarize_history(entry):
    """Small status summary of a history entry. Never returns `outputs` or `prompt`."""
    status = (entry or {}).get("status") or {}
    out = {"status": status.get("status_str", "unknown"), "completed": bool(status.get("completed", False))}
    for msg in status.get("messages") or []:
        try:
            if msg[0] == "execution_error":
                d = msg[1] or {}
                out["error"] = {
                    "node_id": d.get("node_id"),
                    "node_type": d.get("node_type"),
                    "exception_message": d.get("exception_message"),
                }
                break
        except (IndexError, TypeError, KeyError):
            continue
    return out


def find_in_queue(running, pending, prompt_id):
    """('running', 0), ('pending', position) or None. Items are tuples (number, prompt_id, ...)."""
    for item in running or []:
        if len(item) > 1 and item[1] == prompt_id:
            return ("running", 0)
    for pos, item in enumerate(sorted(pending or [], key=lambda i: i[0])):
        if len(item) > 1 and item[1] == prompt_id:
            return ("pending", pos)
    return None


class RunTracker:
    """Bounded prompt_id -> session_name map (oldest evicted first)."""
    def __init__(self, cap=_MAX_TRACKED_RUNS):
        self._cap = cap
        self._d = OrderedDict()

    def add(self, prompt_id, session):
        self._d[prompt_id] = session
        while len(self._d) > self._cap:
            self._d.popitem(last=False)

    def get(self, prompt_id):
        return self._d.get(prompt_id)
