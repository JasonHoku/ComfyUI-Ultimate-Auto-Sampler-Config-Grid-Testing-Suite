# AI Development Context — ComfyUI Ultimate Auto Sampler Config Grid Testing Suite

> **Paste this at the start of any new Claude conversation to provide full project context.**
> **GOLDEN RULE: DO NOT REMOVE ANY CODE. DO NOT REMOVE ANY COMMENTS. ONLY CHANGE WHAT IS NECESSARY.**

## What This Is

A ComfyUI custom node extension for automated image generation grid testing. Users configure combinations of samplers, schedulers, steps, CFG, models, LoRAs, prompts, and resolutions — the system generates all Cartesian products, saves results, and displays them in a virtualized dashboard for comparison. Supports distributed multi-machine generation, upscaling, and CivitAI metadata export.

## Project Location

```
ComfyUI/custom_nodes/ComfyUI-Ultimate-Auto-Sampler-Config-Grid-Testing-Suite/
```

Read `ProjectStructure.md` in this directory for the full reference (file structure, all API routes with method / body / purpose, data shapes, communication patterns, dependency graph, and 36 development gotchas).

## Architecture (3 Layers)

1. **Python Backend** (runs inside ComfyUI server) — nodes, orchestration, generation, API endpoints
2. **JS Frontend Bridge** (`web/`) — registers nodes in ComfyUI browser tab, served at `/extensions/`
3. **Dashboard SPA** (`resources/`) — standalone app inlined into iframe by `html_generator.py`, zero external requests

## Critical Files & Their Roles

| File | Lines | Role |
|------|-------|------|
| `__init__.py` | 1597 | API endpoints (config CRUD, presets, dashboard save/export/delete, scan, async upscale, pairwise votes, metrics jobs, standalone dashboard page, run API, workflow sweep), node mappings, path security |
| `sampler_node.py` | 401 | Main node class. Unwraps configs, extracts `_distribution` + `_session_settings`, delegates to orchestrator |
| `config_builder_node.py` | 1578 | Config builder node. `state_to_configs_json(state)` is the ONLY state-to-configs implementation (`generate_config()` and `POST /configbuilder/preview` both call it). API endpoints for model lists, trigger lookups |
| `generation_orchestrator.py` | 3693 | **Largest file.** Main generation loop over a flat job plan (`config_utils.build_job_plan`), smart skip, preview pass, upscaling pipelines, deferred upscales (`run_deferred_upscales()`), GPU cooldown, distribution, ETA tracking, Start At Job # |
| `image_generation.py` | 1321 | KSampler wrapper, VAE decode, `flush_batch_with_vae()` (saves images + updates manifest), `upscale_image()`, `create_image_metadata()`, rolling ETA |
| `batch_encoding.py` | 878 | Prompt pre-encode with combinators, `pinned_clip()` (one text-encoder load per pass), `te_batch_size` text-encoder batching |
| `civitai.py`, `distribution.py` | 40, 107 | Thin facades for the companion plugins (ComfyUI-USCG-CivitAI, ComfyUI-USCG-Distributed). No outbound network code in this package. |
| `upscale_runner.py` | 642 | Dashboard async upscaling. Background thread loads models, runs pipeline chains, updates manifests, sends progress events. |
| `config_utils.py` | 1108 | Config expansion (Cartesian products), nested prompt parsing, job preparation, job plan / ordering, `config_fingerprint()` |
| `model_loader.py` | 907 | Checkpoint/LoRA/VAE/GGUF loading, latent geometry (`get_latent_downscale_ratio`), CLIP type resolution |
| `model_cache.py` | 935 | 3-tier cache: LoRA files → incremental states → patched models. Async preloading |
| `manifest_utils.py` | 316 | Load/save/merge manifest. `save_manifest()` reloads from disk first to preserve concurrent user edits (favorites, votes, metrics) |
| `metadata_packer.py` | 561 | PNG metadata embedding for CivitAI. `pack_metadata_into_image()` accepts `workflow_data` kwarg |
| `run_api.py` | 144 | Pure helpers behind `POST /config_tester/run` and `GET /config_tester/run_status` |
| `image_metrics.py`, `metrics_runner.py` | 668, 167 | Optional per-image metrics at save time; background job that backfills them for old sessions |
| `workflow_sweep.py`, `workflow_sweep_runner.py`, `workflow_sweep_collect.py` | 969, 912, 140 | Workflow Sweep: pure expansion core, in-process queue feeder (never blocks the executor), `USCGSweepCollect` output node |
| `web/conf_builder/conf-builder-main.js` | 779 | Node registration. Default state and its migration blocks define ALL fields including upscaling/cooldown |
| `web/conf_builder/conf-builder-utilities.js` | 866 | Model-list cache (`LS_CACHE_VERSION`), `getIterationCount()`, LoRA string helpers. No config converter: the preview calls `/configbuilder/preview` |
| `web/conf_builder/conf-builder-config-management.js` | 7736 | All config builder UI rendering. Upscaling section, preview fetch |
| `web/conf_builder/conf-builder-workflow-sweep.js` | 744 | Workflow Sweep UI (load API workflow, axes, dry run, start/pause/cancel) |
| `resources/logic_utils.js` | 1073 | Dashboard helpers: `exportFavorites()`, `deleteNonFavorites()`, `loadSession()`, `scanDirectory()` |
| `resources/logic_ui.js` | 2907 | Dashboard DOM: cards, modals, `buildComfyNodesWorkflow()`, analytics |
| `resources/logic_virtual.js` | 1092 | Virtual scroller: renders only visible cards from thousands |
| `resources/logic_ratings.js`, `resources/logic_compare.js` | 538, 824 | Pairwise review: pure Bradley-Terry / pairing / scoreboard logic, and the compare overlay that uses it |
| `resources/logic_wfparams.js`, `resources/logic_dragdrop.js` | 81, small | Pure helpers (node-testable): Workflow Sweep `wf_params` keys/matching/labels; Revise-image drag payload + `[output]` path for Load Image |
| `resources/logic_standalone.js` | 170 | Standalone-page mode (own websocket, run-API GENERATE NEW); loaded last |

## One Source of Truth for Config JSON

`UltimateConfigBuilder.state_to_configs_json(state)` in `config_builder_node.py` is the only implementation that turns Builder state into configs JSON. `generate_config()` (the node's runtime output) and `POST /configbuilder/preview` (what the JS preview panel calls) both run it, so they cannot drift. There is no JS converter any more; do not add one.

When adding ANY new config field, ALL of these are needed:

- default in `conf-builder-main.js` new-state AND in its three migration blocks
- default in the new-config-array object in `conf-builder-config-management.js`
- UI control in the right render function
- emission in `config_builder_node.state_to_configs_json`
- key in `_EXPECTED_CONFIG_KEYS`
- pass-through in `config_utils.expand_configs`
- consumer in the orchestrator / `generate_image`
- README bullet and unit tests (`tests/test_state_to_configs_json.py`)

Bump `LS_CACHE_VERSION` in `conf-builder-utilities.js` only if the cached model-list shape changes.

## Config Data Flow

```
JS node.state
  → JSON.stringify() → lora_config widget value
  → Python json.loads(lora_config) in generate_config() → state_to_configs_json()
  → Output: {"configs": [...], "_distribution": {...}, "_session_settings": {...}}
  → sampler_node.py unwraps configs, extracts _distribution + _session_settings
  → generation_orchestrator.run_generation_loop() receives all as params
  → config_utils.expand_configs() creates Cartesian product job list
  → Loop: load model → encode prompts → generate → save
```

## Session Settings (Not Per-Config)

`_session_settings` is session-level, embedded alongside configs:
```json
{
  "upscaling": {
    "enabled": true,
    "save_pre_upscale": false,
    "run_upscales_at_end": false,
    "hires_prompt_adjust": false,
    "hires_prompt_behavior": "append_end",
    "hires_prompt_text": "",
    "pipelines": [{ "active": true, "name": "Pipeline 1", "steps": [{ "active": true, "mode": "hires_only", "repeat": 1, "upscale_models": [], "upscale_ratios": "1.5", "hires_denoise": "0.3", "hires_steps": 0, ... }] }]
  },
  "cooldown": { "enabled": true, "seconds": 5, "every_n": 1, "clear_vram": false },
  "start_at_job": 0
}
```
Extracted by `sampler_node.py` line 277, passed to orchestrator as `session_settings` param.

## Image/Manifest Format

Images saved as WebP at `output/benchmarks/{session}/images/img_{id}.webp`

**File URL format** (MUST match exactly for ComfyUI `/view` endpoint):
```
/view?filename={name}&type=output&subfolder=benchmarks/{session}/images
```

**Manifest entry required fields:**
```json
{
  "id": 170941234567890,        // int(time.time() * 100000) + random.randint(0, 1000)
  "file": "/view?filename=...", // Full URL as above
  "rejected": false,            // Required for dashboard display
  "favorited": false,           // User annotation
  "seed": 12345, "cfg": 7.0, "steps": 20,
  "sampler": "euler", "scheduler": "normal",
  "model": "model.safetensors", "lora": "lora:1.0:1.0",
  "positive": "prompt", "negative": "neg prompt",
  "width": 1024, "height": 1024, "duration": 45.2, "denoise": 1.0
}
```

## Key Patterns

- **Model discovery:** `folder_paths.get_filename_list("key")` → API at `/configbuilder/model_lists` → JS `getModelLists()`
- **Path security:** All endpoints use `_is_path_within()` + `re.sub(r'[^\w\-]', '', name)` sanitization
- **Dashboard updates:** `PromptServer.send_sync("ultimate_grid.update", {...})` → `dashboard.js` → `postMessage()` → iframe
- **No outbound network code in this package**: CivitAI, Remote VAE and Distribution network calls live in the companion plugins; `civitai.py`, `remote_vae.py` and `distribution.py` only detect the companion and forward to it. `network_utils.py` no longer exists.
- **No `subprocess`, `os.system`, `eval()`, `exec()`** — blocked by security scanner
- **LoRA string format:** `"name:model_str:clip_str"`, stacked with `" + "` separator
- **Prompt nesting:** Flat list = OR, nested lists = AND (Cartesian product). Recursive in `parse_prompt_input_nested()`

## Field Name Warning

`manifest_utils.py` uses key `"favorite"`. `__init__.py` routes use `"favorited"`. Always check which key the specific code path expects.

## Testing Changes

- **Python files:** Restart ComfyUI
- **`web/` JS files:** Refresh browser (Ctrl+F5)
- **`resources/` files:** Must regenerate dashboard HTML — re-run sampler or call `/config_tester/get_session_html`
- **New state fields:** Add migration in `conf-builder-main.js` `onConfigure`

## Common Tasks

**Add a new config field:** follow the checklist in "One Source of Truth for Config JSON" above (defaults + migration blocks in `conf-builder-main.js`, UI in `conf-builder-config-management.js`, `state_to_configs_json()`, `_EXPECTED_CONFIG_KEYS`, `expand_configs`, then the consumer in `generation_orchestrator.py` or the relevant backend file).

**Add a new API endpoint:**
1. Add route in `__init__.py` with `@server.PromptServer.instance.routes`
2. Add path security: sanitize session name, use `_is_path_within()`
3. Call from dashboard JS (`resources/logic_utils.js`) or config builder JS (`web/conf_builder/`)

**Add a new session setting:**
1. Add to default state in `conf-builder-main.js` under appropriate key
2. Add UI in `conf-builder-config-management.js`
3. Add to the `_session_settings` block of `state_to_configs_json()` in `config_builder_node.py`
4. Consume in `generation_orchestrator.py` after the `session_settings` param check

**Modify dashboard UI:**
1. Edit files in `resources/` (HTML in `template.html`, JS in `logic_*.js`, CSS in `report.css`)
2. Remember: changes require HTML regeneration to take effect
3. JS load order is `html_generator.js_files`: logic_state.js → logic_utils.js → logic_ratings.js → logic_wfparams.js → logic_dragdrop.js → logic_ui.js → logic_upscale_modal.js → logic_virtual.js → logic_pipeline.js → logic_compare.js → logic_events.js → logic_init.js → logic_standalone.js (last). All of them share ONE inlined script scope, so top-level names must be unique
