# Contributing

PRs are welcome and encouraged. This project is actively developed — if you have a bug fix, new feature, or quality improvement in mind, open an issue to discuss it first (for larger changes) or just submit a PR directly (for clear-cut fixes). The maintainer responds promptly.

---

## Quick orientation

Before diving in, read these files in the repo root:

| File | What it covers |
|---|---|
| `README.md` | What the project does, how the three nodes connect, and user-facing features |
| `ProjectStructure.md` | Every file's purpose, all API routes with line numbers, data shapes, communication patterns, dependency graph, and 28 development gotchas |
| `AI-CONTEXT.md` | Internal dev notes: architecture layers, critical codepaths, field naming rules, config data flow, common tasks |

---

## Dev setup

**Requirements:**
- ComfyUI 0.12.3 or newer
- Python 3.10+ (whatever version your ComfyUI install uses)

**Install:**

```
ComfyUI/custom_nodes/
└── ComfyUI-Ultimate-Auto-Sampler-Config-Grid-Testing-Suite/  ← clone here
```

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI. The node will register itself automatically.

**Python dependencies:**

```bash
pip install -r requirements.txt
```

`requirements.txt` currently pulls in `diffusers` and `piexif`. ComfyUI itself provides PyTorch, numpy, and Pillow.

---

## Running tests

```bash
# Python unit tests (ComfyUI + torch are stubbed): needs pytest, numpy, pillow
pip install -r tests/requirements-ci.txt
python -m pytest tests/ -q -ra --ignore=tests/e2e     # 137 passed, 15 skipped without torch

# With real torch (e.g. ComfyUI's python), the 15 `requires_torch` tests run too.
# USCG_REQUIRE_TORCH=1 makes them fail instead of skip if torch is missing:
USCG_REQUIRE_TORCH=1 python -m pytest tests/ -q -ra --ignore=tests/e2e   # 152 passed

# Dashboard SPA browser smoke test (Playwright, Chromium, no ComfyUI needed)
pip install -r tests/e2e/requirements.txt
python -m playwright install chromium
python -m pytest tests/e2e -q -ra
```

GitHub Actions (`.github/workflows/tests.yml`) runs four jobs on every push and PR: `pytest-stubs`, `pytest-torch` (CPU torch), `js-syntax` (`node --check` on `web/` and `resources/`, the assembled Dashboard page, and `node --test tests/js/` when that folder exists) and `e2e-dashboard`.
Tests live in `tests/` and must not contain network or process-spawning code (no `socket`, `http.server`, `subprocess`, `urllib`): `tests/` ships in the registry zip and is yara-scanned.

PRs that change Python logic should add tests where it's reasonable to do so.

---

## JS syntax check

The Builder UI and Dashboard are plain JavaScript (no build step). You can catch syntax errors quickly with Node:

```bash
cd web/conf_builder
node --check conf-builder-main.js
node --check conf-builder-config-management.js
node --check conf-builder-utilities.js
node --check conf-builder-ui-components.js
node --check conf-builder-distribution.js
```

The dashboard JS files can also be checked the same way:

```bash
cd resources
node --check logic_utils.js
node --check logic_ui.js
node --check logic_virtual.js
node --check logic_pipeline.js
node --check logic_events.js
node --check logic_init.js
node --check logic_state.js
```

---

## Architecture notes worth knowing before editing

**The Builder UI is a DOM widget on the Python node.** Its entire state is serialized to a single `lora_config` STRING widget as JSON. Python's `generate_config()` reads only that widget — all other `INPUT_TYPES` entries are vestigial.

**State → configs_json transformation:** All state-to-output transformation logic lives in `state_to_configs_json` (a `@staticmethod` on `UltimateConfigBuilder` in `config_builder_node.py`, around line 524). Both runtime (`generate_config`, line ~837) and the preview endpoint (`POST /configbuilder/preview`) call this single function. **Do not add a JS-side equivalent transformer** — that pattern existed historically (`convertStateToConfigs`) and was a recurring source of drift bugs; it was deliberately removed in favor of the single source of truth in Python.

When adding a new state field that should appear in `configs_json`:
1. Add the field to `state_to_configs_json` in `config_builder_node.py`. Make sure tests in `tests/test_state_to_configs_json.py` cover the new field.
2. Update the Builder UI in `web/conf_builder/conf-builder-config-management.js` to read/write the field via `node.state.<field>`. Also add the default in the initial state object in `conf-builder-main.js` (search for `this.state = {`, around line 103), and add a migration backfill in the async init block below it (`conf-builder-main.js`, around line 484) so existing saved workflows pick up the default.
3. The preview panel will pick up the new field automatically — it just renders whatever `state_to_configs_json` returns.

**Manifest field names — check before touching favorite/reject logic:**
- `favorited` (not `favorite`) — used in `__init__.py` routes
- `note` (not `notes`) — per-item annotation
- `file` — full `/view?...` URL, not a filename
- `rejected` — required boolean for dashboard display
- `id` — timestamp-based integer: `int(time.time() * 100000) + random.randint(0, 1000)`

**Manifest items are inserted at index 0** (newest-first order).

**ComfyUI V3 vs V1 nodes:** Use the `_call_node` and `_unwrap` helpers when calling V3 nodes outside the executor pipeline. See `ltx_video_generation.py` for the established pattern.

**Windows print safety:** Two modules (`config_builder_node.py`, `ltx_video_generation.py`) shadow `print` with `safe_print` at module level (`print = safe_print`) to avoid colorama/stdout corruption that throws `OSError [Errno 22]` on long-running sessions. If you add new print-heavy code in those files, use the existing `print = safe_print` shadow at module top. Other modules use bare `print()` directly.

**Distributed mode exists.** Workers consume the already-rendered `configs_json` over the wire. Don't assume single-machine execution in generation orchestrator code.

**ComfyUI Registry security constraints** (violations get the version flagged and hidden from Manager):
- No `import requests` — use `urllib.request` (stdlib)
- No `subprocess`, `os.system`, `eval()`, or `exec()`
- No runtime `pip install`
- No custom file-serving endpoints — use ComfyUI's `/view` endpoint for images and `WEB_DIRECTORY` for JS
- All filesystem endpoints must validate paths with `_is_path_within()` and sanitize user-supplied names

**Dashboard JS changes require HTML regeneration.** The `resources/` files are inlined into a single HTML string by `html_generator.py`. After editing dashboard JS/CSS, you must re-run the sampler node or call `POST /config_tester/get_session_html` to see the changes. Browser refresh alone is not enough.

---

## PR checklist

Before submitting:

- [ ] `python -m pytest tests/ -v` passes
- [ ] JS syntax checks pass (`node --check` on changed files)
- [ ] No regressions in existing features
- [ ] For new config fields: `state_to_configs_json` updated in Python, default state and migration backfill added in `conf-builder-main.js`, and UI read/write added in `conf-builder-config-management.js`
- [ ] For user-facing features: `README.md` updated
- [ ] For Builder UI or Dashboard changes: smoke-tested in real ComfyUI (unit tests alone are not sufficient here)
- [ ] No new `import requests`, `subprocess`, `os.system`, `eval`, or `exec` calls (Registry security)

---

## Where to discuss

- **Bugs and feature requests:** GitHub Issues
- **Code changes:** GitHub Pull Requests

The maintainer (Jason Hoku) actively monitors both.

---

## License

This project is open source. The license is declared in `pyproject.toml` (`license = {file = "LICENSE"}`). A `LICENSE` file is referenced there but not yet present in the repo root — check the GitHub repository page for the current license terms.
