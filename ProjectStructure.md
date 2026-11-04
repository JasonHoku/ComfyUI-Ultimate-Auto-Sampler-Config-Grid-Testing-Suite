### **ComfyUI Registry Security Requirements**

This node is published to the ComfyUI Registry. Every version is scanned by an automated
security scanner. Versions that fail the scan receive `NodeVersionStatusFlagged` status and
are hidden from ComfyUI Manager. **All contributors must follow these rules:**

1. **No `import requests`** — Use `urllib.request` (stdlib) instead. The scanner flags
   outbound HTTP libraries as potential data exfiltration vectors.
2. **No `subprocess`, `os.system`, `eval()`, `exec()`** — These are blocked as arbitrary
   code execution risks.
3. **No runtime `pip install`** — Package installation must go through `requirements.txt`
   only, never called at runtime.
4. **No code obfuscation** — No encoded strings, multi-statement tricks, or undefined
   variable patterns.
5. **No custom file-serving endpoints** — Use ComfyUI's built-in `WEB_DIRECTORY` for JS and
   the built-in `/view` endpoint for images. Never use `web.FileResponse` or create custom
   routes that serve files from disk. For external images, create symlinks into the output
   directory and use `/view?filename=X&type=output&subfolder=Y`.
6. **Path containment on all filesystem operations** — Every endpoint that touches the
   filesystem must validate that resolved paths stay within the expected base directory
   (e.g., `output/benchmarks/`). Use `os.path.realpath()` + `startswith()` checks.
7. **Sanitize all user-supplied path components** — Session names, filenames, etc. must be
   sanitized with `re.sub(r'[^\w\-]', '', name)` or `os.path.basename()` before use.

Reference: https://blog.comfy.org/p/comfyui-2025-jan-security-update

---

### **Project File Structure**

```text
ComfyUI-Ultimate-Sampler-Grid/
├── __init__.py                  # Server Routes & Node Mappings (1597 lines)
├── sampler_node.py              # Main "Sampler Grid" Node (401 lines)
├── dashboard_node.py            # "Dashboard Viewer" Node (68 lines)
├── config_builder_node.py       # "Config Builder" Node + API endpoints + state_to_configs_json (1578 lines)
├── json_text_node.py            # "Smart JSON" Node (60 lines)
│
├── generation_orchestrator.py   # The "Conductor" (Loop Logic, job plan, preview pass, 3693 lines)
├── image_generation.py          # KSampler & VAE Wrappers (1321 lines)
├── image_metrics.py             # Optional per-image metrics (sharpness/entropy/...), run context, scorer-pack adapters (ADAPTERS registry: node ids/signatures pinned to pack shas, loader defaults + `prefer`, `scale`, `max_batch`, `vram_mb`; `adapter_status()` reports `problem`) (668 lines)
├── metrics_runner.py            # Background job: compute metrics for an existing session (CPU, manifest lock) (167 lines)
├── model_loader.py              # Checkpoint/LoRA/VAE/GGUF Loading Logic, latent geometry, CLIP type (907 lines)
├── model_cache.py               # 3-Tier Model Caching System (935 lines)
├── remote_vae.py                # Remote VAE decode worker + facade for the RemoteVAE companion (314 lines)
├── florence2_hires.py           # Florence2 Hi-Res Fix: targeted-region inpaint, soft dependency (899 lines)
├── ltx_video_generation.py      # LTX 2.3 video pipeline + _call_node helper for V3/V1 nodes (716 lines)
│
├── batch_encoding.py            # Batch CLIP Encoding, Combinators, pinned_clip, Text-Encoder Batching (878 lines)
├── conditioning_cache.py        # Disk-based Conditioning Cache (372 lines)
├── lora_utils.py                # LoRA Searching & Validation (468 lines)
├── trigger_words.py             # Trigger Word Prompt Assembly (377 lines)
├── config_utils.py              # Config Parsing, Expansion, Cartesian Products, job plan + fingerprints (1108 lines)
├── directory_scanner.py         # External Directory Session Scanner (609 lines)
├── metadata_packer.py           # PNG/WebP Metadata Embedding for Exports (561 lines)
├── manifest_utils.py            # Manifest Read/Write/Merge Helpers, comparisons (316 lines)
├── upscale_runner.py            # Dashboard async upscaling background thread (642 lines)
├── civitai.py                   # Facade for the ComfyUI-USCG-CivitAI companion, no network code here (40 lines)
├── distribution.py              # Facade for the ComfyUI-USCG-Distributed companion, no network code here (107 lines)
├── workflow_sweep.py            # Workflow Sweep core: parse API workflow, bindings -> jobs, prompt substitution, fingerprints (pure, no torch/server) (969 lines)
├── workflow_sweep_runner.py     # Workflow Sweep runner: ComfyBridge (in-process post_prompt mirror), windowed feeder thread, pause/resume/cancel, route helpers (912 lines)
├── workflow_sweep_collect.py    # USCGSweepCollect output node: saves each job's image + manifest item into the USCG session (140 lines)
├── html_generator.py            # Reads /resources/ to build the Dashboard HTML (66 lines)
├── run_api.py                   # Pure helpers for POST /config_tester/run + run_status, no server imports (144 lines)
├── conftest.py                  # Registry import stub: lets pytest import this package without ComfyUI (213 lines)
├── pytest.ini                   # --import-mode=importlib, testpaths, markers
│
├── web/                         # [ComfyUI Integration Layer]
│   ├── dashboard.js             # Registers Dashboard Node & Message Handling (489 lines)
│   ├── smart_json_text.js       # Registers JSON Node & Syntax Highlighting (239 lines)
│   ├── sampler_help_note.js     # Help note on the Sampler Grid node (48 lines)
│   └── conf_builder/            # [Config Builder UI Modules]
│       ├── conf-builder-main.js           # Main entry, widget registration, default state + migrations (779 lines)
│       ├── conf-builder-ui-components.js  # UI rendering (dropdowns, sliders, cards) (972 lines)
│       ├── conf-builder-config-management.js # Save/Load/Duplicate config logic, all Builder rendering (7736 lines)
│       ├── conf-builder-utilities.js      # Model-list cache, iteration counting, LoRA string helpers (866 lines)
│       ├── conf-builder-distribution.js   # Distribution settings UI (workers, toggles) (620 lines)
│       ├── conf-builder-workflow-sweep.js # Workflow Sweep UI (load API workflow, axes, dry run, start/pause/cancel) (744 lines)
│       └── conf-builder-readme.md         # Notes on the Builder module split (134 lines)
│
├── resources/                   # [The Dashboard SPA Core] (listed in html_generator.js_files order)
│   ├── template.html            # HTML Skeleton for the generated report (1005 lines)
│   ├── report.css               # Styling (Infinite Canvas, Cards, Modals) (2062 lines)
│   ├── logic_state.js           # State Store (Redux-style: filters, favorites) (149 lines)
│   ├── logic_utils.js           # Helpers (Export, Scan, Session loading) (1073 lines)
│   ├── logic_ratings.js         # Pairwise review pure logic: axis normalisers, Bradley-Terry, pairing, scoreboard (no DOM, node-testable) (538 lines)
│   ├── logic_wfparams.js        # Workflow Sweep wf_params helpers (keys, numeric detection, match, short labels); Quick/Logic Filter + label overlay use them (pure, node-testable)
│   ├── logic_dragdrop.js        # Revise-modal image drag/send helpers (issue #22): uscgImageRef, asset-info + uri-list + DownloadURL carriers, annotated `[output]` path (no DOM, node-testable)
│   ├── logic_ui.js              # UI Renderer (DOM creation, Modal handling) (2907 lines)
│   ├── logic_upscale_modal.js   # Dashboard upscale modal (inline config, presets, async upscale) (1163 lines)
│   ├── logic_virtual.js         # Virtual DOM / Infinite Scroller Engine (1092 lines)
│   ├── logic_pipeline.js        # Data Processing (Filter -> Sort -> Layout) (831 lines)
│   ├── logic_compare.js         # Pairwise review DOM glue: compare overlay, vote queue, scoreboard modal, rating getters (824 lines)
│   ├── logic_events.js          # Interaction Handler (Hotkeys, API calls) (101 lines)
│   ├── logic_init.js            # Bootstrapper (Load JSON, Init State) (112 lines)
│   └── logic_standalone.js      # Standalone-page mode: own websocket, run-API GENERATE NEW, openStandalone() (loaded last, 170 lines)
│
├── tests/                       # pytest suite (conftest stubs ComfyUI + torch, fixtures/ holds API-workflow samples)
├── tests/e2e/                   # Playwright smoke test of the Dashboard SPA (own requirements.txt)
├── tests/js/                    # node:test unit tests for pure Dashboard logic
├── examples/                    # Ready-to-load workflows per model family (+ README with the files each needs)
├── .github/workflows/           # tests.yml (CI: pytest stubbed + torch, JS syntax, e2e), publish.yml
├── docs/plans/                  # Design docs and implementation plans
├── pyproject.toml               # Package metadata (36 lines)
├── requirements.txt             # Python deps: diffusers, piexif
├── .gitignore
├── README.md
├── Roadmap.md
└── SamplrConfig-Basic-Workflow.json  # Example workflow for quick start
```

Distribution, CivitAI and Remote-VAE network code lives in the companion plugins (ComfyUI-USCG-Distributed, -CivitAI, -RemoteVAE); `distribution.py` and `civitai.py` are thin facades.

**Output Directory Structure (Generated at Runtime):**
```text
ComfyUI/output/
├── ultimate-configs/                  # Saved config presets (NOT under benchmarks/)
│   └── {name}.json
└── benchmarks/
    ├── loras_tags.json                # LoRA trigger word cache (shared, editable via UI)
    ├── model_hashes.json              # SHA256 hash cache for CivitAI lookups
    ├── USCG-custom-resolutions.json   # User-defined custom resolution presets
    ├── USCG-upscale-presets.json      # Upscale pipeline presets (shared Builder UI + Dashboard)
    ├── USCG-config-section-presets.json # Config array presets (full config setups)
    ├── model-data/                    # CivitAI metadata cache per model
    │   └── {model_name}/metadata.json
    └── {session_name}/
        ├── manifest.json              # Generation metadata + user annotations
        ├── images/
        │   ├── img_{id}.webp          # Generated images (named by timestamp ID)
        │   └── upscaled_{idx}_{cfg}_{combo}.webp  # Upscaled images
        └── favorites/                 # Created by export_favorites endpoint
            ├── manifest.json          # Cleaned favorites-only manifest
            └── {exported images}
```

---

### **Project Architecture Overview**

**ComfyUI-Ultimate-Sampler-Grid** is a hybrid application with three layers:

1. **The Backend (Python):** Runs inside ComfyUI. Orchestrates grid generation, manages resources (caching/loading), and serves the API.
2. **The Frontend Bridge (JS in `web/`):** Runs in the ComfyUI browser tab. Registers custom nodes and widgets. Lives at ComfyUI's `/extensions/` path.
3. **The Dashboard SPA (HTML/JS/CSS in `resources/`):** A standalone, virtualized "IDE" for viewing results. Generated by the backend as inlined HTML, runs entirely in the client's browser via iframe.

```
┌─────────────────────────────────────────────────────────────┐
│ ComfyUI Browser Tab                                         │
│  ┌──────────────────┐  ┌──────────────────────────────────┐│
│  │ Config Builder    │  │ Dashboard Node                   ││
│  │ (conf-builder-*) │  │ (dashboard.js)                   ││
│  │ Direct DOM widget │  │  ┌──────────────────────────────┐││
│  │                   │  │  │ iframe (srcdoc)              │││
│  │                   │  │  │ Dashboard SPA                │││
│  │                   │  │  │ (resources/logic_*.js)       │││
│  │                   │  │  │ postMessage() ↕              │││
│  │                   │  │  └──────────────────────────────┘││
│  └──────────────────┘  └──────────────────────────────────┘│
│           ↕ widget value              ↕ server events       │
├─────────────────────────────────────────────────────────────┤
│ ComfyUI Server (Python)                                     │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │config_builder│  │sampler_node.py │  │__init__.py     │  │
│  │_node.py      │→ │                │  │(API endpoints) │  │
│  └──────────────┘  │  ↓             │  └────────────────┘  │
│                    │generation_     │                       │
│                    │orchestrator.py │                       │
│  ┌─────────┐     │  ↓             │  ┌─────────────────┐ │
│  │ModelCache│←────│model_loader +  │  │DistributionMgr  │ │
│  └─────────┘     │image_generation│  │(master jobs)    │ │
│                    │  ↓             │  └─────────────────┘ │
│                    │html_generator │         ↕ HTTP       │
│                    └────────────────┘  ┌─────────────────┐ │
│                                        │WorkerThread     │ │
│                                        │(remote workers) │ │
│                                        └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

### **Node Class Mappings**

Defined at the end of `__init__.py`:

| Internal Name | Python Class | Purpose |
|---|---|---|
| `UltimateSamplerGrid` | `SamplerGridTester` | Main generation node — receives configs, runs Cartesian grid |
| `UltimateGridDashboard` | `SamplerConfigDashboardViewer` | Dashboard viewer — renders iframe with results |
| `UltimateConfigBuilder` | `UltimateConfigBuilder` | Visual config builder — outputs `configs_json` string |
| `SmartJSONText` | `SmartJSONTextNode` | JSON text widget with syntax highlighting |
| `USCGSweepCollect` | `USCGSweepCollect` | Workflow Sweep output node (`workflow_sweep_collect.py`): saves each job's image + manifest item into the USCG session |

`WEB_DIRECTORY = "./web"` — tells ComfyUI to serve `web/` files at `/extensions/`.

---

### **API Routes Reference**

All routes registered via `PromptServer.instance.routes`. One row per route this package registers; `tests/test_docs_consistency.py` fails when a registered route is missing here.

**Config Builder APIs** (registered in `__init__.py` and `config_builder_node.py`):

| Method | Route | File | Body / query | Purpose |
|---|---|---|---|---|
| GET | `/configbuilder/list_configs` | `__init__.py` | none | List saved config preset filenames (JSON array) |
| POST | `/configbuilder/save_config` | `__init__.py` | `{name, data}` | Save a config preset |
| POST | `/configbuilder/load_config` | `__init__.py` | `{name}` | Load a config preset |
| GET | `/configbuilder/custom_resolutions` | `__init__.py` | none | Get custom resolution presets |
| POST | `/configbuilder/custom_resolutions` | `__init__.py` | JSON body, stored as sent | Save custom resolution presets |
| GET | `/configbuilder/upscale_presets` | `__init__.py` | none | Upscale pipeline presets (`{"presets": []}` when none), shared by Builder and Dashboard |
| POST | `/configbuilder/upscale_presets` | `__init__.py` | JSON body, stored as sent (400 when empty) | Save upscale pipeline presets |
| GET | `/configbuilder/config_section_presets` | `__init__.py` | none | Config array presets (`{"presets": []}` when none) |
| POST | `/configbuilder/config_section_presets` | `__init__.py` | JSON body, stored as sent (400 when empty) | Save config array presets |
| GET | `/configbuilder/section_presets` | `__init__.py` | `?section=` one of `models`, `loras`, `prompts` (400 otherwise) | Presets for one Builder section |
| POST | `/configbuilder/section_presets` | `__init__.py` | `?section=` as above, JSON body stored as sent | Save presets for one Builder section |
| POST | `/configbuilder/preview` | `config_builder_node.py` | `{state}` (the Builder state; 400 when missing) | Runs `state_to_configs_json(state)`, returns `{configs_json}`; what the JSON Preview panel shows |
| POST | `/configbuilder/lookup_triggers` | `config_builder_node.py` | `{loras: [...]}` | Bulk LoRA trigger word lookup |
| POST | `/configbuilder/lookup_lora_metadata` | `config_builder_node.py` | `{lora_name, force_refresh?}` | Full CivitAI metadata for a LoRA (via the companion facade) |
| POST | `/configbuilder/lookup_model_metadata` | `config_builder_node.py` | `{model_name, model_type? (default checkpoint), force_refresh?}` | Full CivitAI metadata for a model |
| GET | `/configbuilder/model_lists` | `config_builder_node.py` | none | All model/sampler/scheduler/VAE/upscale lists |
| GET | `/configbuilder/model_counts` | `config_builder_node.py` | none | File counts per model category (fast, no listing) |
| GET | `/configbuilder/metrics_info` | `config_builder_node.py` | none | Built-in metric names + optional scorer-pack install status (Run Settings hint) |
| POST | `/configbuilder/refresh_models` | `config_builder_node.py` | none | Signal frontend cache clear |
| GET | `/configbuilder/get_lora_triggers` | `config_builder_node.py` | `?lora_name=` | Get triggers for a single LoRA |
| POST | `/configbuilder/save_lora_triggers` | `config_builder_node.py` | `{lora_name, triggers}` | Save edited triggers for a LoRA |

**Dashboard/Tester APIs** (registered in `__init__.py`):

| Method | Route | File | Body / query | Purpose |
|---|---|---|---|---|
| POST | `/config_tester/delete_session` | `__init__.py` | `{session_name}` | Delete a session folder |
| GET | `/config_tester/list_sessions` | `__init__.py` | none | List all sessions with metadata (name, count, thumbnail, mtime) |
| POST | `/config_tester/save_changes` | `__init__.py` | `{session_name, changed_items}` | Delta save: only the changed items |
| POST | `/config_tester/save_comparisons` | `__init__.py` | `{session_name, append: [...], remove: ["ts:a:b", ...]}` | Pairwise-review votes; only touches manifest `comparisons` |
| POST | `/config_tester/clear_comparisons` | `__init__.py` | `{session_name}` | Delete all votes of a session |
| POST | `/config_tester/save_manifest` | `__init__.py` | `{session_name, manifest}` | Full merge save (keeps server-side `meta` and `comparisons`) |
| POST | `/config_tester/get_session_html` | `__init__.py` | `{session_name, node_id}` | Generate the dashboard HTML on demand (empty manifest for an unknown session, not a 404) |
| GET | `/config_tester/dashboard` | `__init__.py` | `?session=` (optional) | Dashboard SPA as a top-level page (standalone mode, `USCG_STANDALONE`, own websocket via `resources/logic_standalone.js`) |
| POST | `/config_tester/run` | `__init__.py` | `{configs_json or config_name, session_name?, overwrite_existing?, force? (default true), front?, client_id?}` | Headless run: builds the Generator+Dashboard prompt and enqueues it in-process like `/prompt`. Helpers in `run_api.py` |
| GET | `/config_tester/run_status` | `__init__.py` | `?prompt_id=` | `pending`, `running`, `success` or `error` + session / dashboard_url / item_count |
| POST | `/config_tester/export_favorites` | `__init__.py` | `{session_name, pack_metadata?, organize_by_prompt?, organize_by_lora?, export_prompt_txt?, copy_manifest? (default true), pack_workflow?, pack_nodes_workflow?, workflow_data?, nodes_workflows?}` | Export favorited images with options (metadata, workflow, organize) |
| POST | `/config_tester/delete_non_favorites` | `__init__.py` | `{session_name}` | Delete all non-favorited images from a session |
| POST | `/config_tester/delete_rejected` | `__init__.py` | `{session_name}` | Delete all rejected images from a session (rejected preview fingerprints are remembered in `meta.rejected_preview_fps`) |
| POST | `/config_tester/scan_directory` | `__init__.py` | `{directory_path, session_name}` | Scan an external dir, create symlinks, generate a manifest |
| POST | `/config_tester/upscale_images` | `__init__.py` | `{session_name, image_ids, upscale_config: {pipelines: [...]}, all_favorited?}` | Start an async dashboard upscale job; returns `{job_id, total_images}` |
| GET | `/config_tester/upscale_status` | `__init__.py` | `?job_id=` | Progress of an upscale job |
| POST | `/config_tester/cancel_upscale` | `__init__.py` | `{job_id}` | Cancel an upscale job; returns `{cancelled}` |
| POST | `/config_tester/compute_metrics` | `__init__.py` | `{session_name, overwrite?}` | Start the background metrics job for a session; returns `{job_id, total_images}` |
| GET | `/config_tester/metrics_status` | `__init__.py` | `?job_id=` | Metrics job progress |
| POST | `/config_tester/cancel_metrics` | `__init__.py` | `{job_id}` | Cancel a metrics job; returns `{cancelled}` |
| POST | `/config_tester/sweep/parse` | `__init__.py` | `{api_json}` | Workflow Sweep: describe an API-format workflow (nodes, outputs, image sources, suggested bindings) |
| POST | `/config_tester/sweep/dry_run` | `__init__.py` | `{state or configs_json, limit?}` | Workflow Sweep: first N concrete prompts + validation of job 1; queues nothing |
| GET | `/config_tester/sweep/job_prompt` | `__init__.py` | `?session_name=&sweep_id=&job_index=` | Workflow Sweep: `{prompt, params, fingerprint, output, ...}`, the job's API prompt exactly as it ran (template + substitutions) minus the USCG collector node; ids only, no paths (400 bad id/index, 404 unknown) |
| POST | `/config_tester/sweep/start` | `__init__.py` | `{state or configs_json, node_id?, client_id?}` | Workflow Sweep: start a sweep `{state or configs_json, node_id?}` (400 invalid; 200 `attached:true` when the IDENTICAL sweep is already active; 409 `{error, sweep_id, session_name}` for a different one) |
| GET | `/config_tester/sweep/status` | `__init__.py` | `?sweep_id=` or `?session_name=` | Workflow Sweep: state, counts, ETA, last errors (404 for an unknown sweep) |
| POST | `/config_tester/sweep/pause` | `__init__.py` | `{sweep_id}` | Workflow Sweep: pause (409 when not possible) |
| POST | `/config_tester/sweep/resume` | `__init__.py` | `{sweep_id}` | Workflow Sweep: resume (409 when not possible) |
| POST | `/config_tester/sweep/cancel` | `__init__.py` | `{sweep_id}` | Workflow Sweep: cancel (409 when not possible) |

**Companion plugin routes** (registered by ComfyUI-USCG-Distributed, not by this package):

| Method | Route | Purpose |
|---|---|---|
| GET | `/uscg-distribution/claim_job` | Worker claims next pending job (200 + job JSON, 204 = no jobs, 503 = inactive) |
| POST | `/uscg-distribution/submit_result` | Worker uploads completed image (multipart: {metadata, image}) |
| GET | `/uscg-distribution/status` | Get distribution status, worker stats, job counts |
| POST | `/uscg-distribution/register_worker` | Worker registration with master |
| GET | `/uscg-distribution/download_model` | Download model file for model sync `{category, filename}` |
| POST | `/uscg-distribution/heartbeat` | Worker keepalive signal |

---

### **1. The Backend: Core Generation Pipeline**

* **`sampler_node.py`** (The Interface — 314 lines)
  * **Class:** `SamplerGridTester`
  * **Entry:** `run_tests()` (line 252) → delegates to `generation_orchestrator.run_generation_loop()`
  * **Key Logic:**
    * Extracts `_distribution` and `_session_settings` from wrapped `configs_json` (lines 269-293)
    * Disables caching when optional inputs connected (line 264)
    * `IS_CHANGED()` returns `NaN` when optionals connected (forces re-execution), deterministic hash otherwise (line 103)
  * **Required Inputs:** `ckpt_name`, `positive_text`, `negative_text`, `seed`, `denoise`, `vae_batch_size`, `configs_json`, `resolutions_json`, `session_name`, `overwrite_existing`, `flush_batch_every`, `add_random_seeds_to_gens`, `lora_triggerwords_mode`, `remote_vae_endpoint`, `save_conditioning_cache_to_file`, `enable_model_cache`
  * **Optional Inputs:** `optional_model` (MODEL), `optional_clip` (CLIP), `optional_vae` (VAE), `optional_positive` (CONDITIONING), `optional_negative` (CONDITIONING), `optional_latent` (LATENT)


* **`generation_orchestrator.py`** (The Conductor — 2281 lines, largest file)
  * **Entry:** `run_generation_loop()` — manages entire generation lifecycle
  * **Key Functions:**
    * `setup_session_directories()` (line 43) — creates `benchmarks/{session}/images/` directory structure
    * `run_generation_loop()` — main orchestration
    * `check_if_job_completed()` — smart skip (matches seed/resolution/sampler/scheduler/steps/cfg/denoise against existing manifest)
    * `_preencode_all_conditionings()` — groups configs by (model, LoRA), loads each, encodes all unique prompts, serializes
    * `_conditioning_to_serializable()` / `_serializable_to_conditioning()` — tensor ↔ serialized format for distribution
    * `_run_distributed_generation()` — manages distributed pipeline
    * `_send_distribution_status()` — sends worker status updates to dashboard
  * **Upscaling Block** (lines 1096-1204): Array-based with Cartesian expansion — see dedicated section below
  * **GPU Cooldown Block** (lines 1249-1265): Pauses between generations if enabled in session_settings
  * **Interrupt Handling** (line 1269): Catches `InterruptProcessingException`, flushes pending batch and saves manifest before re-raising
  * **Lookahead:** Triggers `model_cache.preload_lora_model()` / `preload_base_model()` for the next job
  * **ETA Tracking:** Sends `ultimate_grid.progress` events to dashboard via `PromptServer.instance.send_sync()`


* **`image_generation.py`** (The Worker — 748 lines)
  * `generate_image()` (line 16) — wraps KSampler or SamplerCustomAdvanced pipeline
  * `create_image_metadata()` (line 483) — creates metadata dict from config, dimensions, duration, seed, prompts
  * `flush_batch_with_vae()` (line 605) — decodes latents, saves images, updates manifest, syncs disk tags, sends dashboard updates
  * `flush_batch_with_remote_vae()` (line 713) — queues latents for remote VAE worker
  * `decode_latent_with_vae()` — single latent → PIL Image
  * `upscale_image()` (line 265) — handles 3 modes: hires_only, model_only, model_then_hires
  * `calculate_eta()` / `print_generation_progress()` — ETA calculation and logging
  * **Image file URL format:** `f"/view?filename={filename}&type=output&subfolder=benchmarks/{session_name}/images"`
  * **Manifest entry requirements:** Each item needs `id` (timestamp-based), `file` (full /view? URL), `rejected` (boolean)


* **`config_builder_node.py`** (The Builder Backend — 1167 lines)
  * **Class:** `UltimateConfigBuilder`
  * `generate_config()` (line 465) — reads ALL state from single `lora_config` widget (all other INPUT_TYPES widgets are vestigial)
  * **Output format** (line 694): Always `{"configs": [...]}` with optional `"_distribution"` and `"_session_settings"` keys
  * **Session settings embedding** (lines 706-715): Upscaling/cooldown data embedded as `_session_settings`
  * **API endpoints:** Trigger lookups, model metadata, model lists (lines 742-1167)
  * **Model lists endpoint** (line 1029): Returns checkpoints, diffusion_models, unet_gguf, clip_gguf, text_encoders, clip_types, vae, upscale_models, samplers, schedulers


* **`html_generator.py`** (57 lines)
  * `get_html_template(title, manifest_data, node_id)` — reads all `/resources/` files, inlines into `template.html`
  * **JS Load Order:** `logic_state.js` → `logic_utils.js` → `logic_ratings.js` → `logic_wfparams.js` → `logic_dragdrop.js` → `logic_ui.js` → `logic_upscale_modal.js` → `logic_virtual.js` → `logic_pipeline.js` → `logic_compare.js` → `logic_events.js` → `logic_init.js` → `logic_standalone.js` (all files share ONE inlined script: top-level names must be unique across them)
  * Replaces placeholders: `__CSS_CONTENT__`, `__JS_CONTENT__`, `__TITLE__`, `__NODE_ID__`
  * Zero external HTTP requests — everything inlined

---

### **2. The Backend: Resource Management**

* **`model_cache.py`** (3-Tier Cache — 900 lines)
  * **Tiers:** 1. LoRA File Cache (raw state_dicts in RAM), 2. Incremental States (partial LoRA stacks), 3. Final Patched Models
  * **Key Methods:** `get_base_model()`, `put_base_model()`, `preload_lora_model()`, `preload_base_model()`, `register_schedule()`, `set_current_step()`, `clear()`, `print_stats()`
  * **Async Preloading:** Background threads via `concurrent.futures`

* **`conditioning_cache.py`** (372 lines)
  * Caches CLIP text encodings to disk
  * Hashes `(Prompt + CLIP State + LoRA Config)` to ensure validity
  * Auto-disabled when optional inputs connected

* **`batch_encoding.py`** (378 lines)
  * `encode_prompt_with_combinators(clip, text, clip_skip)` — supports AND, CAT, AVG, BREAK combinators
  * `batch_encode_prompts()` — bulk prompt encoding with cache integration
  * `encode_prompts_batched()` — `te_batch_size` > 1: several eligible prompts per text-encoder forward (right-padded, per-prompt replay of the unmodified `_encode_single`; allowlist `TE_BATCH_FAMILIES`, self-check, OOM halving, `DEFAULT_TE_BATCH_SIZE`)
  * **Combinators:** `AND` (multi-cond with optional weights like `:1.5`), `CAT` (tensor concatenation), `AVG(weight)` (weighted average), `BREAK` (77-token segment break)

* **`model_loader.py`** (805 lines)
  * `load_checkpoint()`, `load_loras()`, `load_vae_by_name()`, `load_diffusion_model_and_clip()`, `load_loras_for_preencoding()`, `cleanup_model_references()`, `print_incompatible_loras_summary()`

* **`trigger_words.py`** (377 lines)
  * `collect_unique_prompts_with_triggers()`, `build_prompt_with_triggers()`, `get_filtered_lora_triggers()`
  * **Modes:** `"None"`, `"Append To End"`, `"Append To Start"`, `"Read From Config"`
  * Per-LoRA placement via `lora_triggerwords_append_settings`. Also applies `model_prompt_prefix` / `model_prompt_suffix`.

* **`config_utils.py`** (674 lines)
  * `expand_configs()` — Cartesian product expansion of multi-value fields
  * `parse_prompt_input_nested()` — supports arbitrarily deep nested arrays for Cartesian prompting
  * `prepare_input_jobs()` — final job list preparation
  * `parse_lora_definition()` — parses `"name:model_str:clip_str"` format
  * `sanitize_session_name()`, `get_files_from_folder()`
  * Job plan + preview helpers (pure Python, unit-tested): `build_job_plan()`, `order_job_plan()` (as_written / minimize_reloads), `count_switches()`, `config_fingerprint()` / `item_fingerprint()`, `preview_dims()`, `build_preview_index()`, `should_skip_for_previews()`, `build_full_fp_set()`, `collect_rejected_preview_fps()`

* **`manifest_utils.py`** (120 lines)
  * `load_existing_manifest()`, `save_manifest()`, `merge_manifest_user_changes()`
  * **Critical:** `save_manifest()` calls `merge_manifest_user_changes()` first — reloads from disk to preserve concurrent user favorites/rejected/notes

* **`directory_scanner.py`** (463 lines)
  * `scan_directory_for_images()` — scans external directories, parses A1111-style PNG metadata
  * Preserves user tags on re-scan by merging with existing manifest

* **`metadata_packer.py`** (561 lines)
  * `pack_metadata_into_image()` (line 181) — accepts `workflow_data` kwarg for embedding workflow
  * `extract_metadata_from_image()` — reads metadata from PNG/WebP
  * `calculate_file_hash()` — SHA256 with disk cache at `benchmarks/model_hashes.json`
  * **Workflow embedding:** Supports both full ComfyUI workflow and per-image nodes workflow
  * Contains fallback `workflowExample` (line 21) — used only when no workflow data provided

---

### **2b. The Backend: Distribution System**

The distribution code (job queue, worker thread, `/uscg-distribution/*` routes, model sync) lives in the ComfyUI-USCG-Distributed companion plugin, not in this package. `distribution.py` is a thin facade: it detects whether the companion is installed (`is_distribution_available()`) and forwards `create_manager()`, `notify_workers_to_start()`, `stop_all_workers()` and `get_master_url()` to it, raising a clear install hint when it is missing. `generation_orchestrator.py` imports only the facade. The companion calls `check_if_job_completed()` positionally, so that signature must not change (new flags are keyword-only).

---

### **3. The Frontend: ComfyUI Integration (`web/`)**

* **`dashboard.js`** (432 lines)
  * Registers `UltimateGridDashboard` node
  * Listens for `ultimate_grid.update`, `ultimate_grid.progress`, `ultimate_grid.distribution_status` (the only three `api.addEventListener("ultimate_grid.*")` calls in `web/`; the standalone page in `resources/logic_standalone.js` also handles `ultimate_grid.update_data`)
  * Forwards events to matching iframe via `postMessage()`
  * **Auto-loads on node creation** with 500ms delay via `forceLoadSession()`
  * **Widgets:** "RELOAD / SHOW SESSION" button, "DELETE SESSION" button
  * **Singleton guards:** `window.__ultimateGrid*ListenerInstalled` prevents duplicate registration

* **`conf-builder-main.js`** (608 lines)
  * Registers `UltimateConfigBuilder` node
  * `onNodeCreated` **MUST be synchronous** — async init in fire-and-forget IIFE
  * Module loading via `import()` with `Date.now()` cache-busting
  * **Default state** (lines 80-147): Defines all config fields including `upscaling` and `cooldown`
  * **Node methods:** `saveState()`, `renderUI()`, `loadSession()`, `saveConfigToBackend()`, `loadConfigFromBackend()`, `triggerAutoSave()` (2s debounce)

* **`conf-builder-utilities.js`** (958 lines)
  * `convertStateToConfigs()` was removed: the Builder preview POSTs the state to `/configbuilder/preview`, which runs Python's `state_to_configs_json` (no JS converter any more)
  * `getIterationCount()` (line 457) — calculates total Cartesian product count
  * `parseLoraString()` / `buildLoraString()` — LoRA string format parsing
  * `refreshAllConfigBuilders()`, `clearAllCaches()`
  * **Global caches:** `availableLoras`, `availableModels`, `availableVAEs`, `availableSamplers`, `availableSchedulers`, `availableSessions`, `availableConfigs`, `availableUpscaleModels`
  * **Session settings output** (lines 689-711): Attaches `_session_settings` with upscaling/cooldown data

* **`conf-builder-config-management.js`** (4975 lines — largest JS file)
  * `renderUI()`, `updatePreview()`, `renderConfigSection()`, `renderSessionSection()`
  * `createConfigArrayElement()`, `createModelElement()`, `renderUpscalingSection()` (line ~4437)
  * **Migration logic** for old single-config upscaling format to array format
  * All UI rendering for the config builder widget

* **`conf-builder-ui-components.js`** (944 lines)
  * `createSearchableSelect()`, `createSlider()`, `createInputGroup()`, `createTopBar()`, `createSidebar()`, `createSectionHeader()`, `getStyles()`

* **`conf-builder-distribution.js`** (474 lines)
  * `renderDistributionSection()`, `renderDistributionSettingsSection()`
  * Worker URL management, connection testing, status indicators

* **`smart_json_text.js`** (239 lines)
  * Registers `SmartJSONText` node with syntax highlighting and validation

---

### **4. The Frontend: Dashboard SPA (`resources/`)**

The Dashboard is a standalone single-page app rendered inside an iframe. All files are inlined by `html_generator.py`.

* **`template.html`** (662 lines)
  * HTML skeleton with settings panel (cog icon), export options, analytics
  * Export checkboxes (lines 228-261): pack-metadata, export-prompt-txt, copy-manifest, pack-workflow, pack-nodes-workflow
  * DELETE ALL NON FAVORITED ITEMS button (line 257)

* **`report.css`** (1642 lines)
  * Infinite canvas styling, card grid, modals, mobile responsive
  * Dark theme with accent colors

* **`logic_init.js`** (104 lines)
  * `initDashboard()` — entry point, detects `fullManifest` (injected by Python)
  * Lifecycle: `initLogicState()` → `applyPipeline()` → `initVirtualScroller()` → `initGlobalEvents()`

* **`logic_state.js`** (104 lines)
  * Central state store (Redux-style): `items`, `filteredItems`, `selection`, `favorites`
  * Triple JSON bars: Accepted, Favorites, Rejected
  * Saves user preferences (sort order, column count) to `localStorage`

* **`logic_pipeline.js`** (510 lines)
  * `updateDataPipeline()`, `executePipeline()`, `processNewData()`, `incrementalFilter()`
  * Filtering and sorting pipeline before virtual scroller

* **`logic_virtual.js`** (925 lines)
  * Virtual scroller rendering only ~20 visible cards out of thousands
  * `renderDOM()`, `updateVisibleItems()`, `calculateVisibleRange()`, `autoFitZoom()`, `goToImage()`

* **`logic_ui.js`** (1780 lines)
  * DOM manipulation: card HTML, Revise modal, Settings panel, Analytics modals
  * `toggleCogMenu()`, `toggleFiltersPopup()`, `initFilters()`
  * `buildComfyNodesWorkflow(d)` (reusable function) — builds workflow JSON from item data
  * `copyConfigsAsComfyNodes(id)` — calls `buildComfyNodesWorkflow()` internally
  * Analytics modal with "Copy Favorited Prompts as JSON" button

* **`logic_events.js`** (91 lines)
  * Keyboard shortcuts: Arrow Keys (Pan), Space (Scroll), +/- (Zoom), F (Fit), Escape (close modals)
  * Message handler for `progress_update` and `update_data` from parent frame

* **`logic_utils.js`** (599 lines)
  * `loadSession()`, `saveState()`, `exportFavorites()`, `scanDirectory()`, `triggerGen()`, `toggleFullscreen()`
  * `rejectItem()`, `selectJSON()`, `deleteNonFavorites()`
  * Export favorites reads all checkboxes, captures workflow via `app.graph.serialize()`, builds per-image nodes workflows

---

### **Key Data Shapes**

**Manifest `comparisons`** (top level, sibling of `items`/`meta`; written ONLY by `manifest_utils.update_manifest_comparisons`, disk always wins in every other writer via `preserve_disk_comparisons`):
```json
"comparisons": [{"a": 180000000000001, "b": 180000000000002, "winner": "a", "ts": 1762000000000, "mode": "matched"}]
```
`mode` is `matched|seed|one_axis|random`; `blind: false` is present only for votes cast unblinded. Dedupe key is `ts:a:b`.

**Config Array State** (`node.state.config_arrays[n]` in JS, parsed from `lora_config` widget in Python):
```json
{
  "name": "Config 1",
  "samplers": ["euler", "dpmpp_2m"],
  "schedulers": ["normal", "karras"],
  "steps": "20, 30",
  "cfg": "7.0",
  "models": [{"path": "model.safetensors", "type": "checkpoint"}],
  "vaes": ["None"],
  "loras": ["lora.safetensors:1.00:1.00"],
  "lora_bypass_states": {"lora_name": true},
  "model_bypass_states": {"model_path": true},
  "vae_bypass_states": {"vae_name": true},
  "te_bypass_states": {"te_name": true},
  "lora_omit_triggers": [],
  "lora_triggerwords_append_settings": {},
  "lora_strength_lock": {},
  "combine": false,
  "positive_prompt_groups": [],
  "negative_prompt": "",
  "use_custom_prompts": false,
  "model_prompt_prefix": "",
  "model_prompt_suffix": "",
  "attention_modes": ["default"],
  "resolutions": [],
  "seed_behavior": "fixed",
  "full_run_seed_behavior": "fixed",
  "full_run_seed": 0,
  "text_encoders": [],
  "clip_type": "stable_diffusion",
  "gguf_options": {},
  "model_sampling_override": "none",
  "model_sampling_shift": "1.73",
  "model_sampling_flux_max_shift": "1.15",
  "model_sampling_flux_base_shift": "0.5",
  "use_advanced_sampling": false,
  "advanced_guider": "cfg_guider",
  "advanced_scheduler": "basic",
  "use_flux_guidance": false,
  "flux_guidance_value": "3.5"
}
```

**Session Settings** (embedded as `_session_settings` in config output):
```json
{
  "upscaling": {
    "enabled": true,
    "configs": [{
      "mode": "hires_only",
      "upscale_models": ["RealESRGAN_x4plus.pth"],
      "upscale_ratios": "1.5, 2.0",
      "hires_denoise": "0.3, 0.5",
      "hires_steps": 10,
      "tiled_vae": false,
      "tile_size": 512
    }]
  },
  "cooldown": {
    "enabled": true,
    "seconds": 5,
    "every_n": 1,
    "clear_vram": false
  }
}
```

**Distribution Config** (embedded as `_distribution` in config output):
```json
{
  "enabled": true,
  "worker_urls": ["http://192.168.1.100:8188", "http://192.168.1.101:8188"],
  "claim_timeout": 600,
  "use_master_encoding": false,
  "sync_models_to_workers": false
}
```

**Top-level state fields** (outside `config_arrays`):
```json
{
  "session_name": "my_session",
  "config_name": "my_config",
  "auto_save": false,
  "include_none": false,
  "label_mode": false,
  "global_positive_groups": [],
  "global_negative": "",
  "distribution_enabled": false,
  "worker_urls": [],
  "claim_timeout": 600,
  "use_master_encoding": false,
  "sync_models_to_workers": false,
  "upscaling": { "enabled": false, "configs": [...] },
  "cooldown": { "enabled": false, "seconds": 5, "every_n": 1, "clear_vram": false }
}
```

**Manifest Item** (`manifest.json` `items[n]`):
```json
{
  "id": 170941234567890,
  "seed": 12345, "cfg": 7.0, "steps": 20,
  "sampler": "euler", "scheduler": "normal",
  "model": "model.safetensors", "lora": "lora:1.0:1.0",
  "vae": "vae.safetensors", "attention_mode": "default",
  "positive": "prompt text", "negative": "negative text",
  "config_positive": "raw prompt without triggers",
  "config_negative": "raw negative without triggers",
  "denoise": 1.0, "width": 1024, "height": 1024,
  "batch_idx": 0, "gen_index": 0, "duration": 45.2,
  "file": "/view?filename=img_170941234567890.webp&type=output&subfolder=benchmarks/session/images",
  "favorited": false, "rejected": false, "note": ""
}
```

**Upscaled Manifest Item** (additional fields):
```json
{
  "upscaled": true,
  "upscale_mode": "hires_only",
  "upscale_ratio": 1.5,
  "upscale_denoise": 0.3,
  "upscale_model": "RealESRGAN_x4plus.pth"
}
```

---

### **Communication Patterns**

**Widget Bridge (Config Builder → Python):**
```
JS node.state → JSON.stringify() → lora_config widget.value → Python json.loads(lora_config)
```
All other widget params in INPUT_TYPES (samplers, schedulers, steps, cfg) are ignored by Python.

**Config Output Wrapping (Config Builder → Sampler):**
```
config_builder_node.py generate_config()
  → {"configs": [...], "_distribution": {...}, "_session_settings": {...}}
  → json string via configs_json output
  → sampler_node.py run_tests()
  → Unwraps: configs_json = json.dumps(parsed["configs"])
  → Extracts: dist_config = parsed["_distribution"]
  → Extracts: session_settings = parsed["_session_settings"]
  → Passes all to run_generation_loop()
```

**Live Dashboard Updates (Server → Dashboard iframe):**
```
generation_orchestrator → PromptServer.send_sync("ultimate_grid.update")
  → dashboard.js api.addEventListener → postMessage({type: 'update_data'})
  → iframe logic_events.js message handler
```

**Progress Updates (Server → Dashboard iframe):**
```
generation_orchestrator → PromptServer.send_sync("ultimate_grid.progress")
  → dashboard.js → postMessage({type: 'progress_update'})
  → iframe #eta-bar update
```

**Dashboard → Parent Frame:**
```
iframe postMessage({type: 'toggle_fullscreen', node_id})
  → dashboard.js listener → toggle .dashboard-fullscreen class
```

**Distribution Job Flow (Master ↔ Worker)** (the manager and worker named here live in the ComfyUI-USCG-Distributed companion):
```
Master:
  1. generation_orchestrator → DistributionManager.populate_jobs()
  2. [Optional] _preencode_all_conditionings() → manager.set_encoded_conditionings()
  3. POST /uscg-distribution/register_worker on each worker
  4. Workers poll GET /uscg-distribution/claim_job
  5. manager._job_to_dict() attaches encoded_positive/encoded_negative if available
  6. Worker processes job → POST /uscg-distribution/submit_result (multipart: metadata + image)
  7. Master saves image + updates manifest
```

**Distribution Status Updates (Server → Dashboard iframe):**
```
generation_orchestrator → PromptServer.send_sync("ultimate_grid.distribution_status")
  → dashboard.js api.addEventListener → postMessage({type: 'distribution_status'})
  → iframe displays worker activity
```

---

### **Upscaling System (Array-Based with Cartesian Expansion)**

The upscaling system allows testing multiple upscale configurations per generated image.

**UI State** (`node.state.upscaling` in `conf-builder-main.js` line 129):
```json
{
  "enabled": false,
  "configs": [{
    "mode": "hires_only",           // "hires_only" | "model_only" | "model_then_hires"
    "upscale_models": [],           // Array of upscale model names
    "upscale_ratios": "1.5",        // CSV string of ratios
    "hires_denoise": "0.3",         // CSV string of denoise values
    "hires_steps": 0,               // 0 = use config steps
    "tiled_vae": false,
    "tile_size": 512
  }]
}
```

**Data Flow:**
1. Python `state_to_configs_json()` embeds upscaling in `_session_settings` (the JS preview gets the same output from `/configbuilder/preview`)
2. Python `config_builder_node.py` line 706-715 embeds into output JSON
3. `sampler_node.py` line 277 extracts `session_settings`
4. `generation_orchestrator.py` line 1096-1204 processes after each generation

**Cartesian Expansion** (orchestrator lines 1119-1127):
- `hires_only`: Product of `ratios × denoises`
- `model_only`: Product of `models × [1.0] × [0.0]`
- `model_then_hires`: Product of `models × ratios × denoises`

**Important Implementation Details:**
- `effective_size = up_ratio if up_ratio > 1.0 else 2.0` — prevents model_only mode from resizing back to 1x
- Skip guard: `if show_model and not up_model_name: continue` — avoids crash on empty model name
- Upscaled images use format `upscaled_{gen_idx}_{ucfg_idx}_{combo_idx}.webp`
- Base images are skipped when upscaling produces results: `if not upscale_produced:` (line 1236)
- Upscaled manifest entries include extra fields: `upscaled`, `upscale_mode`, `upscale_ratio`, `upscale_denoise`, `upscale_model`

---

### **Favorites & Workflow Packing System**

**Dashboard Export Options** (template.html lines 228-261):
1. **Pack CivitAI Metadata** — converts images to PNG with A1111-compatible metadata
2. **Export Prompt .txt** — saves positive prompt alongside each image
3. **Copy Cleaned Favorites Manifest** — only favorited items, not full manifest
4. **Pack Full ComfyUI Workflow** — embeds `app.graph.serialize()` workflow into image metadata
5. **Pack Config Data as Nodes Workflow** — per-image workflow from `buildComfyNodesWorkflow()`
6. **DELETE ALL NON FAVORITED ITEMS** — removes non-favorited images and updates manifest

**Workflow Priority** (in `__init__.py` `export_favorites` line 586-592):
- Per-image nodes workflow (from `nodes_workflows` dict keyed by file URL) takes priority
- Falls back to full ComfyUI graph workflow
- Both embedded via `pack_metadata_into_image(... workflow_data=item_workflow)`

**Key Functions:**
- `logic_utils.js` `exportFavorites()` — reads checkboxes, captures workflow, builds per-image workflows, POSTs to endpoint
- `logic_ui.js` `buildComfyNodesWorkflow(d)` — reusable function generating pure-nodes workflow from item data
- `__init__.py` `export_favorites()` (line 435) — handles all export options
- `__init__.py` `delete_non_favorites()` (line 662) — deletes non-favorited images and updates manifest

---

### **File Dependency Graph (Python)**

```
sampler_node.py
  → generation_orchestrator.py (main entry)
    → config_utils.py (expand_configs, prepare_input_jobs)
    → model_loader.py (load_checkpoint, load_loras)
    → image_generation.py (generate_image, flush_batch, upscale_image)
    → batch_encoding.py (encode_prompt_with_combinators)
    → trigger_words.py (build_prompt_with_triggers)
    → manifest_utils.py (save_manifest, load_existing_manifest)
    → model_cache.py (ModelCache)
    → conditioning_cache.py (ConditioningCache)
    → remote_vae.py (RemoteVAEDecodeWorker)
    → html_generator.py (get_html_template)
    → distribution.py (facade for the ComfyUI-USCG-Distributed companion)
    → image_metrics.py (optional per-image metrics)

config_builder_node.py
  → lora_utils.py (trigger word lookup, CivitAI API)

__init__.py
  → all node classes (sampler, dashboard, config_builder, json_text)
  → html_generator.py (on-demand HTML generation)
  → run_api.py (POST /config_tester/run helpers)
  → metrics_runner.py (background metrics job)
  → workflow_sweep.py, workflow_sweep_runner.py, workflow_sweep_collect.py (Workflow Sweep)
  → upscale_runner.py (dashboard upscale job)
  → metadata_packer.py (export favorites)
  → directory_scanner.py (scan external directories)
  → manifest_utils.py (save/load manifests — note: NOT directly imported, uses inline logic)
```

---

### **Server Event Types**

| Event Name | Sender | Data Fields | Purpose |
|---|---|---|---|
| `ultimate_grid.update` | `generation_orchestrator.py` / `image_generation.py` / `remote_vae.py` / `workflow_sweep_collect.py` | `session_name, node, manifest, meta, new_items` | New images generated |
| `ultimate_grid.update_data` | `generation_orchestrator.py`, `upscale_runner.py` | `session_name, new_items` (+ `node` from the orchestrator) | Per-item replacement: the listed items replace the cards with the same id (deferred upscales, dashboard upscales) |
| `ultimate_grid.progress` | `generation_orchestrator.py`, `upscale_runner.py`, `workflow_sweep_runner.py` | `session_name, node, progress_pct, eta_str, current_job, total_jobs, avg_duration, last_duration, finish_time` | ETA/progress bar. Optional `phase`: `upscaling` (deferred and dashboard upscales) or `sweep` (Workflow Sweep, label "Sweep job"). The sweep payload is `{node, session_name, phase: "sweep", sweep_id, state, errors, current_job, total_jobs, progress_pct, eta_str, finish_time, avg_duration, last_duration}` plus, on completion, `complete, total_elapsed, total_generated` |
| `ultimate_grid.next_run_seeds` | `generation_orchestrator.py` | `node, seeds` | Per-config seeds the NEXT run should use after a `random_after` / `increment_after` / `decrement_after` full-run seed behavior, as `{config_index: seed}` (no listener in this package) |
| `ultimate_grid.metrics_progress` | `metrics_runner.py` | `session_name, job_id, status, completed, total` | Progress of a "compute metrics for this session" job (no listener in this package; the SPA polls `/config_tester/metrics_status`) |
| `ultimate_grid.distribution_status` | `generation_orchestrator.py` | `session_name, workers, stats` | Distribution worker status |

---

### **Summary of Data Flow**

1. **User Config:** Set parameters in the **Config Builder** (visual UI) or **Sampler Node** (JSON text).
2. **Config Output:** `config_builder_node.py` wraps configs in `{"configs": [...], "_distribution": {...}, "_session_settings": {...}}`.
3. **Unwrapping:** `sampler_node.py` unwraps `"configs"` key, extracts distribution and session settings.
4. **Expansion:** `generation_orchestrator.py` calls `expand_configs()` + `prepare_input_jobs()` to expand configs into a full list of jobs (Cartesian product of samplers × schedulers × steps × cfg × models × loras × prompts × resolutions × attention_modes).
5. **Distribution (optional):** If enabled, `_run_distributed_generation()` populates the job queue, optionally pre-encodes conditionings, and notifies workers.
6. **Generation:** The backend loops through its share of jobs, using **ModelCache** and **ConditioningCache** for speed.
7. **Upscaling (optional):** After each image, if upscaling enabled, runs Cartesian product of upscale configs.
8. **Storage:** Images saved to disk as WebP; metadata appended to `manifest.json`.
9. **Compilation:** `html_generator.py` reads `manifest.json` and all `/resources/` files, baking into a single `dashboard_html` string.
10. **Display:** The **Dashboard Node** (JS) renders this HTML in an iframe. Auto-loads on node creation.
11. **Boot:** `logic_init.js` calls `initDashboard()` → `showSessionLandingIfEmpty()`. If data exists, `logic_virtual.js` renders the grid.
12. **Interaction:** Click "Favorite" → `logic_events.js` → `save_changes` API → Python updates `manifest.json` on disk.
13. **GPU Cooldown (optional):** After every N generations, pauses for configured seconds.

---

### **Critical Development Gotchas**

> **GOLDEN RULE: DO NOT REMOVE ANY CODE. DO NOT REMOVE ANY COMMENTS. ONLY CHANGE WHAT IS NECESSARY.**

1. **Single Data Path**: `state_to_configs_json(state)` in `config_builder_node.py` is the only state-to-configs implementation; the JS preview calls `/configbuilder/preview`, which runs the same function. Do not add a JS-side converter. Adding a Builder/config field means ALL of: default in `conf-builder-main.js` new-state AND in its three migration blocks; default in the new-config-array object in `conf-builder-config-management.js`; UI control in the right render function; emission in `config_builder_node.state_to_configs_json`; key in `_EXPECTED_CONFIG_KEYS`; pass-through in `config_utils.expand_configs`; consumer in the orchestrator / `generate_image`; README bullet; unit tests. Bump `LS_CACHE_VERSION` in `conf-builder-utilities.js` only if the cached model-list shape changes.

2. **Widget Bridge** — The `lora_config` widget (STRING type) holds the **entire** node state as serialized JSON. Python `generate_config()` (which wraps `state_to_configs_json`) ignores all other widget params and reads only `lora_config`. The `samplers`, `schedulers`, `steps`, `cfg` widgets in `INPUT_TYPES` are vestigial (kept for ComfyUI schema compliance only).

3. **onNodeCreated Must Be Synchronous** — In `conf-builder-main.js` line 51, `onNodeCreated` MUST remain synchronous. Async initialization runs inside a fire-and-forget `(async () => { ... })()` IIFE. Making `onNodeCreated` itself async breaks widget registration.

4. **Resources Are Inlined** — `html_generator.py` reads all `/resources/` files and bakes them into one HTML string. Zero external HTTP requests. Changes to dashboard JS/CSS require re-generating the HTML (re-run the sampler node or call `/config_tester/get_session_html`).

5. **Manifest Merge on Save** — `manifest_utils.py` `save_manifest()` calls `merge_manifest_user_changes()` which reloads from disk before saving. This preserves user favorites/rejected/notes that changed concurrently while generation was running. Also, `flush_batch_with_vae()` (image_generation.py line 657) does its own disk sync.

6. **Optional Inputs Disable Caches** — `sampler_node.py` disables conditioning cache when `optional_model`/`optional_clip`/`optional_positive`/`optional_negative` are connected. `IS_CHANGED()` returns `NaN` (forces re-execution). `check_if_job_completed()` skips model/lora/prompt matching.

7. **Folder Expansion Syntax** — `"folder/"` = expand to individual entries (one combination each). `"folder/*"` = stack ALL files in folder together as a single combined entry. Weight array syntax `"lora:[0.5, 0.8]:1.0"` creates grid search over strengths. No trailing slash = single file match.

8. **LoRA String Format** — `"name:model_str:clip_str"`, stacked with `" + "` separator. `parseLoraString()` in JS, `parse_lora_definition()` in Python. Default strengths: 1.0 for both if colon-separated parts are missing.

9. **Cache Busting** — `conf-builder-main.js` uses `Date.now()` timestamp on `import()` calls for module loading. Also intercepts `window.fetch` to detect `/object_info` calls and trigger cache refresh. Static JS served via ComfyUI's `/extensions/` path.

10. **Config Presets Path** — Saved to `output/ultimate-configs/` (defined in `__init__.py` line 34), NOT `output/benchmarks/configs/`. Session data goes under `output/benchmarks/{session_name}/`.

11. **Field Name Inconsistency: `favorite` vs `favorited`** — `manifest_utils.py` `merge_manifest_user_changes()` uses the key `"favorite"`. But `__init__.py` routes like `save_manifest` and `export_favorites` check `"favorited"`. When touching favorite/unfavorite logic, check which key name the specific code path uses.

12. **Prompt Expansion is Recursive** — `parse_prompt_input_nested()` in `config_utils.py` supports arbitrarily deep nested arrays. Flat list = OR (options). List containing sub-lists = AND (Cartesian product). This is used by both the node text inputs and per-config prompt groups from the builder UI.

13. **Config Builder Output is ALWAYS Wrapped** — `config_builder_node.py` always outputs `{"configs": [...]}` (with optional `"_distribution"` and `"_session_settings"` keys). The `sampler_node.py` MUST unwrap the `"configs"` key before passing to `expand_configs()`. Both the `_distribution` present AND absent cases must be handled (see lines 269-293 of `sampler_node.py`).

14. **Dashboard Auto-Load on Creation** — `web/dashboard.js` calls `forceLoadSession()` with 500ms delay when a dashboard node is created. The `get_session_html` endpoint in `__init__.py` returns a full HTML template with empty manifest for non-existent sessions (NOT a 404). This ensures the landing page shows immediately.

15. **`logic_pipeline.js` Legacy Init Guard** — Lines 489-505 have a `DOMContentLoaded` handler that calls `loadSession()` ONLY when `fullManifest.items.length > 0`. Without this guard, `loadSession()` triggers an alert popup for non-existent sessions. The proper initialization path is `logic_init.js` → `init()` → `showSessionLandingIfEmpty()`.

16. **Distribution Default Values Must Stay in Sync** — `claim_timeout` default (600) appears in: `distribution_manager.py` (constructor), `config_builder_node.py` (fallback), `conf-builder-distribution.js` (slider default), `conf-builder-main.js` (default state). All four must match. Same for `use_master_encoding` default (false).

17. **Conditioning Serialization Uses np.copy()** — `_serializable_to_conditioning()` in `generation_orchestrator.py` must call `np.frombuffer(...).copy()` before `torch.from_numpy()` because `np.frombuffer()` returns read-only arrays incompatible with GPU tensor operations.

18. **Multi-Entry Conditioning (AND combinator)** — The AND combinator creates multiple `[tensor, {pooled_output}]` entries in a conditioning list. Serialization in `_conditioning_to_serializable()` must iterate ALL entries, not just `[0]`. Deserialization must reconstruct the full list.

19. **State Migration for New Fields** — When adding new fields to the Config Builder default state in `conf-builder-main.js` (around line 80), also add a migration check (around line 363 in the `onConfigure` handler) to backfill the default for existing saved workflows. Without migration, old workflows will have `undefined` for the new field.

20. **ComfyUI Registry Security** — No `import requests`, no `subprocess`/`os.system`/`eval()`/`exec()`, no runtime `pip install`, no custom file-serving endpoints, no code obfuscation. Use `urllib.request` for HTTP. Use ComfyUI's `/view` endpoint for serving files. Sanitize all user-supplied paths. See top of this file for full rules.

21. **Image File URL Format** — Must be `/view?filename={name}&type=output&subfolder=benchmarks/{session}/images` for ComfyUI's built-in view endpoint. Both `flush_batch_with_vae()` (image_generation.py line 649) and upscaling (orchestrator line 1191) must use this exact format. Missing `type` or `subfolder` params causes 404s.

22. **Manifest Entry Required Fields** — Each item needs `id` (timestamp-based integer), `file` (full /view? URL), `rejected` (boolean) for dashboard display. Missing `id` causes holes in the grid. The `id` format is `int(time.time() * 100000) + random.randint(0, 1000)`.

23. **Base Image Skip on Upscale** — When upscaling is enabled and produces results, the base (non-upscaled) image must NOT be saved to the manifest. Controlled by `upscale_produced` flag and `if not upscale_produced:` guard (orchestrator line 1236).

24. **Session Settings Architecture** — `_session_settings` is a special key embedded in the JSON alongside configs. It's NOT per-config — it's session-level. Contains `upscaling` and `cooldown` settings. Extracted by `sampler_node.py` (line 277) and passed to `run_generation_loop()` as `session_settings` param. The orchestrator receives it as a function parameter, not from configs.

25. **Model Discovery Pattern** — `folder_paths.get_filename_list("key")` returns available models. API endpoint `/configbuilder/model_lists` returns all lists. JS side: `conf-builder-utilities.js` `getModelLists()` fetches and stores in global vars. `conf-builder-main.js` passes to renderers as `modelLists` object.

26. **Bypass States Are UI-Only** — `vae_bypass_states` and `te_bypass_states` control filtering in `state_to_configs_json()` but should NOT appear in config output. They're internal UI state persisted only by `node.saveState()`. `lora_bypass_states` and `model_bypass_states` DO appear in output for smart-skip matching.

27. **Manifest Save from Dashboard** — Two save mechanisms exist:
    - `save_changes` (line 234 in `__init__.py`): Delta save — only changed items by ID. Fast. Used for individual star/reject.
    - `save_manifest` (line 308): Full merge save — preserves server-side additions. Used for bulk operations.

28. **Path Security** — All filesystem endpoints use `_is_path_within()` (line 27 in `__init__.py`) to validate paths stay within `benchmarks/` base directory. Session names sanitized with `re.sub(r'[^\w\-]', '', name)`.

29. **Image Metrics Run Context** - `image_metrics.begin_run(spec)` (wrapped around `run_generation_loop` in `sampler_node.py`) returns a token and `end_run(token)` closes only that context, so a second run or the dashboard upscale thread can never clobber it. Generation-side hooks read the active context (`decode_latent_with_vae(..., metrics_sink=)`, `flush_batch_with_vae`, orchestrator upscale inserts via `image_metrics.finalize`); the dashboard upscale thread (`upscale_runner.py`) must NOT read generation-run state, so it passes an explicit spec derived from the source item (`spec_from_item`). Every place that copies a source item into an upscaled entry excludes `"metrics"` (never inherit); the new entry is re-measured. `metrics` is session-level (`_session_settings.metrics`), not a per-config key.

30. **Workflow Sweep never blocks the executor.** `USCGSweepCollect` and the Generator in workflow mode only start/record work; the runner feeds sweep prompts through `PromptServer.instance.prompt_queue` from its own thread (an in-process mirror of `server.py post_prompt`, loop-bound calls via `ComfyBridge.run_on_loop`, never from the loop thread, no lock held across it). A node that waited on a prompt it queued would deadlock ComfyUI's single executor thread. The expansion/substitution core (`workflow_sweep.py`) is pure; only `ComfyBridge` touches ComfyUI, so tests use a fake bridge.

31. **pytest package-name trap**: pytest runs with `--import-mode=importlib` and a root `pytest.ini`. A parent folder named `integration`, `metrics` etc. (no hyphen) is imported as a package and the stub conftest breaks. Worktree folder names need a hyphen; never add `tests/__init__.py`.

32. **Manifest writers preserve disk state**: every writer (generation saves, the remote VAE worker, the `save_manifest` route) keeps the on-disk `comparisons` (`manifest_utils.preserve_disk_comparisons`) and the per-item `metrics` written by `metrics_runner` (an item's on-disk `metrics` survives when the in-memory copy has none, in `manifest_utils.merge_manifest_user_changes`). Only `update_manifest_comparisons` writes votes; only `metrics_runner` rewrites `metrics` in place.

33. **Flat job plan + `config_fingerprint`**: `run_generation_loop` walks a flat `job_plan` (`config_utils.build_job_plan`, one entry per job, binding `conf, conf_idx, job_idx, job, w, h, batch_idx`) instead of nested resolution x config loops; `order_job_plan` reorders it for `job_order: minimize_reloads` (stable sort on first-appearance ranks). `check_if_job_completed(..., preview=False)` keeps its positional signature (the distributed companion calls it positionally); the `preview` flag is keyword-only and previews/full items never match each other. `config_fingerprint()` is the identity of a grid cell (stored as `config_fp` on every new item; preview items add `preview`, `preview_of`, `full_*`; upscaled items add `base_fp`). Bump `FP_VERSION` (currently `"fp1"`, and keep the old reader) if you ever change its payload; a golden-value test pins it. Preview/promotion/job-order features are ignored by the distributed path.

34. **`pinned_clip`**: every loop that encodes more than one prompt runs inside `batch_encoding.pinned_clip(clip, sample_texts)` (the orchestrator's pre-encode groups, the per-job encode and the deferred-upscale re-encode). Nesting is safe.

35. **Dynamic-VRAM re-prepare**: on ComfyUI 0.4x (comfy_aimdo) `load_models_gpu` re-runs `ModelPatcherDynamic.load()` on every encode; the repeated "prepared for dynamic VRAM loading" log line is the symptom. TE batching (`te_batch_size`) self-checks batch 1 against a single encode and halves on OOM; non-batchable prompts keep the per-prompt path.

36. **`html_generator.js_files` order is load-bearing** (one shared script scope): `logic_ratings.js` (pure) before `logic_ui.js`, `logic_compare.js` after `logic_pipeline.js`, `logic_standalone.js` LAST (top-level-page overrides). CI assembles the page and syntax-checks it.

---

### **Testing Changes**

- **Python backend files:** Restart ComfyUI
- **`web/` JS files:** Refresh browser (Ctrl+F5) — served via `/extensions/` with cache-busting
- **`resources/` JS/CSS/HTML files:** Dashboard HTML must be regenerated — re-run the sampler node or call `/config_tester/get_session_html`
- **Config Builder state fields:** Also update migration checks in `conf-builder-main.js` `onConfigure`
- **New config fields:** `state_to_configs_json()` in Python is the only converter (the JS preview calls `/configbuilder/preview`); then follow the checklist in gotcha 1

---

### **Key Functions Quick Reference**

**Python (by file):**

| File | Key Functions | Line Numbers |
|---|---|---|
| `sampler_node.py` | `run_tests()`, `find_existing_match()`, `get_latent_channels()`, `IS_CHANGED()` | 252, 201, 169, 103 |
| `config_builder_node.py` | `state_to_configs_json()`, `generate_config()`, `preview_endpoint()`, `process_lora_array()`, `get_available_sessions()`, `expand_lora_folders()` | - |
| `generation_orchestrator.py` | `run_generation_loop()`, `check_if_job_completed()`, `setup_session_directories()`, `_preencode_all_conditionings()`, `_conditioning_to_serializable()`, `_run_distributed_generation()` | -, -, 43, -, -, - |
| `image_generation.py` | `generate_image()`, `create_image_metadata()`, `flush_batch_with_vae()`, `upscale_image()`, `decode_latent_with_vae()`, `calculate_eta()` | 16, 483, 605, 265, -, 540 |
| `batch_encoding.py` | `encode_prompt_with_combinators()`, `batch_encode_prompts()` | |
| `config_utils.py` | `expand_configs()`, `parse_prompt_input_nested()`, `prepare_input_jobs()`, `parse_lora_definition()`, `sanitize_session_name()` | |
| `model_loader.py` | `load_checkpoint()`, `load_loras()`, `load_vae_by_name()`, `load_diffusion_model_and_clip()` | |
| `model_cache.py` | `ModelCache.get_base_model()`, `.put_base_model()`, `.preload_lora_model()`, `.register_schedule()` | |
| `trigger_words.py` | `collect_unique_prompts_with_triggers()`, `build_prompt_with_triggers()`, `get_filtered_lora_triggers()` | |
| `manifest_utils.py` | `load_existing_manifest()`, `save_manifest()`, `merge_manifest_user_changes()` | |
| `metadata_packer.py` | `pack_metadata_into_image()`, `extract_metadata_from_image()`, `calculate_file_hash()` | 181, -, - |
| `directory_scanner.py` | `scan_directory_for_images()`, `parse_a1111_parameters()` | |
| `distribution.py` | `is_distribution_available()`, `create_manager()`, `notify_workers_to_start()`, `stop_all_workers()`, `get_master_url()` (facade; the manager and worker live in the companion) | |
| `run_api.py` | `build_run_prompt()`, `normalize_configs_payload()`, `resolve_saved_config()`, `summarize_history()`, `RunTracker` | |
| `image_metrics.py` | `compute_metrics_numpy()`, `compute_metrics_torch()`, `begin_run()`, `end_run()`, `finalize()`, `adapter_status()` | |
| `metrics_runner.py` | `start_metrics_job()`, `get_metrics_status()`, `cancel_metrics_job()` | |
| `workflow_sweep.py` | `parse_api_workflow()`, `suggest_bindings()`, `parse_for_ui()` (pure; no torch or server imports) | |
| `workflow_sweep_runner.py` | `dry_run()`, `api_start()`, `start_sweep()`, `get_status()`, `pause()`, `resume()`, `cancel()`, `ComfyBridge` | |

**JavaScript (by file):**

| File | Key Functions |
|---|---|
| `conf-builder-main.js` | `ensureModulesLoaded()`, `node.saveState()`, `node.renderUI()`, `node.loadSession()`, `node.triggerAutoSave()` |
| `conf-builder-utilities.js` | `getIterationCount()`, `parseLoraString()`, `buildLoraString()`, `refreshAllConfigBuilders()`, `getModelLists()` |
| `conf-builder-config-management.js` | `renderUI()`, `updatePreview()`, `createConfigArrayElement()`, `createModelElement()`, `renderUpscalingSection()` |
| `conf-builder-ui-components.js` | `createSearchableSelect()`, `createSlider()`, `getStyles()` |
| `conf-builder-distribution.js` | `renderDistributionSection()`, `renderDistributionSettingsSection()` |
| `dashboard.js` | `forceLoadSession()`, event listeners for update/progress/distribution_status |
| `logic_ui.js` | `toggleCogMenu()`, `toggleFiltersPopup()`, `initFilters()`, `buildComfyNodesWorkflow()`, `copyConfigsAsComfyNodes()` |
| `logic_utils.js` | `loadSession()`, `exportFavorites()`, `scanDirectory()`, `triggerGen()`, `toggleFullscreen()`, `deleteNonFavorites()` |
| `logic_virtual.js` | `renderDOM()`, `updateVisibleItems()`, `calculateVisibleRange()`, `autoFitZoom()`, `goToImage()` |
| `logic_pipeline.js` | `updateDataPipeline()`, `executePipeline()`, `processNewData()`, `incrementalFilter()` |
| `logic_wfparams.js` | `wfParamKeys()`, `wfParamValue/Number()`, `wfParamMatches()`, `wfShortLabel()`, `wfDisplay()` (pure; tests in `tests/js/wfparams.test.js`) |
| `logic_ratings.js` | `computeBradleyTerry()`, `buildPairSource()`, `nextPair()`, `axisDiff()`, `computeScoreboard()` (pure; tests in `tests/js/ratings.test.js`, run `node --test tests/js/*.test.js`) |
| `logic_dragdrop.js` | `uscgImageRef()`, `uscgImageAssetInfo()`, `uscgAnnotatedPath()`, `uscgFillDragData()`, `USCG_DRAG_UPLOAD` (pure; tests in `tests/js/dragdrop.test.js`). Used by `logic_ui.js` (`dragstart` on `#m-img`, `sendModalToLoadImage()`). Parent bridge: iframe posts `uscg_send_to_load_image` {url, filename, subfolder, imageType, upload, node_id}, `web/dashboard.js` fills the selected/only/new LoadImage and answers `uscg_send_to_load_image_result` {ok, message} |
| `logic_compare.js` | `openCompareMode()`, `cmpVoteSide()`, `cmpUndo()`, `openScoreboard()`, `getItemRating()`, `compareRatings()`, `onComparisonsReplaced()` |
| `logic_events.js` | Message handler for `progress_update`, `update_data`, `distribution_status` |
