# ComfyUI Ultimate Auto Sampler Config Grid Testing Suite

<br>
Want to support development of this project? Buy me a coffee on Ko-fi:
<br><br>
<a href="https://ko-fi.com/jasonhoku" target="_blank">
  <img src="https://storage.ko-fi.com/cdn/brandasset/kofi_button_blue.png" alt="Support Me on Ko-fi" height="41">
</a>
<br><br>

**A professional-grade benchmarking and "IDE-like" testing suite for ComfyUI.**

Stop guessing which Sampler, Scheduler, Prompt, Denoise, Model, Lora or CFG value works best. This custom node suite allows you to generate massive Cartesian product grids, view them in an interactive infinite-canvas dashboard, and refine your settings with a "Revise & Generate" workflow without ever leaving the interface.

<img width="1840" height="895" alt="image" src="https://github.com/user-attachments/assets/5f8ac634-1ae8-4b51-a9d7-1b1752d1bda4" />

---

## ⚡ Quick Start — The 3 Nodes You Need

This suite is **three separate nodes** that must all be added and wired together. Don't stop after adding just one — here's how they connect:

```
┌──────────────────────────┐     configs_json    ┌──────────────────────────┐     dashboard_html    ┌──────────────────────────┐
│                          │ ──────────────────▶ │                          │ ────────────────────▶ │                          │
│  Ultimate Config Builder │                     │  Ultimate Sampler Grid   │                       │  Ultimate Grid Dashboard │
│      (Visual UI)         │                     │       (Generator)        │                       │         (Viewer)         │
│                          │                     │                          │                       │                          │
└──────────────────────────┘                     └──────────────────────────┘                       └──────────────────────────┘
   Point-and-click GUI for                          The engine — loads models,                         Renders the infinite-canvas
   building your test grid.                          runs every sampler × CFG ×                         dashboard with your result
   (Optional — you can also                          step × LoRA combination.                           grid for browsing, rating,
   write JSON directly.)                             Also accepts Checkpoint, CLIP,                     and revising.
                                                     and VAE inputs.
```

Node names in the "Add Node" menu, under `sampling/testing`:

| # | Node Name                      | Role                                                |
|---|--------------------------------|-----------------------------------------------------|
| 1 | **Ultimate Config Builder**    | Visual GUI for building configs *(optional)*        |
| 2 | **Ultimate Sampler Grid**      | The generator — **required**                        |
| 3 | **Ultimate Grid Dashboard**    | The result viewer — **required**                    |

### 🚀 Fastest way to try it

**Drag [`SamplrConfig-Basic-Workflow.json`](SamplrConfig-Basic-Workflow.json) into your ComfyUI canvas** — it includes all three nodes pre-wired with a working example. Swap in your checkpoint and hit Run.

> **Troubleshooting:** If the Config Builder loads as a plain node with no sliders/UI, right-click it and choose *Fix this node*, or delete and re-add it. Make sure you're on the latest version via Comfy Manager (1.9.2 or newer) and have restarted ComfyUI after updating.

### Manual wiring

If you'd rather build the graph yourself:

1. Add **Ultimate Sampler Grid** and connect your Checkpoint / CLIP / VAE to its inputs.
2. Add **Ultimate Grid Dashboard** and connect the Generator's `dashboard_html` output to the Viewer's input.
3. *(Optional)* Add **Ultimate Config Builder** and wire its `configs_json` output into the Generator's `configs_json` input. Without it, you can type a JSON config directly into the Generator's `configs_json` widget.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
  - [Powerful Grid Generation](#-powerful-grid-generation)
  - [Interactive Dashboard (The "IDE")](#-interactive-dashboard-the-ide)
  - [The "Revise & Generate" Workflow](#-the-revise--generate-workflow)
  - [Curation & JSON Export](#-curation--json-export)
  - [Config Builder Node (Visual UI For Creating Runs)](#-config-builder-node-visual-ui)
    - [Basic Setup](#basic-setup)
    - [Understanding the Interface](#understanding-the-interface)
    - [Session Management](#session-management)
    - [Config Management (Save/Load Presets)](#config-management-saveload-presets)
    - [Config Arrays](#config-arrays)
    - [Fast CivitAI Info Lookup](#fast-civitai-info-lookup)
- [Installation](#-installation)
- [Getting Started](#getting-started)
  - [The Nodes](#1-the-nodes)
  - [Generator Node Parameters](#2-generator-node-parameters)
  - [The JSON Configuration](#3-the-json-configuration)
  - [Hybrid Inputs (Optional)](#4-hybrid-inputs-optional)

    - [Basic Parameters](#basic-parameters)
    - [Iteration Count Display](#iteration-count-display)
    - [Config Array Controls](#config-array-controls)
  - [Models Section](#models-section)
  - [LoRAs Section](#loras-section)
  - [LoRA Trigger Words](#lora-trigger-words)
    - [Trigger Word Lookup Modal](#trigger-word-lookup-modal)
    - [Trigger Word Placement Control](#trigger-word-placement-control)
  - [LoRA Bypass States](#lora-bypass-states)
  - [Strength Locking](#strength-locking)
  - [Folder Expansion (Models & LoRAs)](#folder-expansion-models--loras)
  - [LoRA Stacking](#lora-stacking)
  - [VAEs Section](#vaes-section)
  - [Attention Modes](#attention-modes)
  - [Model-Specific Prompts](#model-specific-prompts)
  - [Recursive Cartesian Prompt Builder](#recursive-cartesian-prompt-builder)
- [Dashboard Interface](#-dashboard-interface)
  - [Header Bar](#header-bar)
  - [Toolbar](#toolbar)
  - [Navigation & Controls](#navigation--controls)
  - [Card Overlays](#card-overlays)
  - [JSON Bars](#json-bars-bottom---horizontal-layout)
  - [Revision Modal](#revision-modal)
- [Example Workflows](#-example-workflows)
  - [Quick Quality Test](#quick-quality-test-40-images)
  - [Multi-Model Comparison](#multi-model-comparison-9-images)
  - [LoRA Stack Testing](#lora-stack-testing-24-images)
  - [Random LoRA Selection](#random-lora-selection)
  - [Full Model Folder Test](#full-model-folder-test)
  - [CLIP Skip for Anime Models](#clip-skip-for-anime-models)
  - [LoRA Trigger Word Filtering](#lora-trigger-word-filtering)
  - [Model Folder Expansion](#model-folder-expansion)
  - [LoRA Strength Sweep](#lora-strength-sweep)
  - [Quality Enhancement Stack](#quality-enhancement-stack)
- [Preset Configs](#-preset-configs)
- [Performance Tips](#-performance-tips)
- [Tips & Best Practices](#-tips--best-practices)
- [Troubleshooting](#-troubleshooting)
  - [Generation Issues](#generation-issues)
  - [Dashboard Issues](#dashboard-issues)
  - [LoRA & Trigger Word Issues](#lora--trigger-word-issues)
  - [Model & CLIP Issues](#model--clip-issues)
  - [Config Builder UI Issues](#config-builder-ui-issues)
  - [Remote VAE Issues](#remote-vae-issues)
  - [Interrupt/Cancel Issues](#interruptcancel-issues)
  - [Browser Compatibility](#browser-compatibility)
- [JSON Output Format Reference](#json-output-format-reference)
- [API Integration (Config Builder)](#api-integration-config-builder)
- [File Locations](#file-locations)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Distributed Processing](#-distributed-processing)
  - [Setting Up Distribution](#setting-up-distribution)
  - [Master Text Encoding](#master-text-encoding)
  - [Model Sync](#model-sync)
  - [Distribution Troubleshooting](#distribution-troubleshooting)
- [Upscaling & Post-Processing](#-upscaling--post-processing)
- [Favorites Export & Workflow Packing](#-favorites-export--workflow-packing)
- [Notes for AI Development](#-notes-for-ai-development)
- [Changelog](#-changelog)
- [License](#-license)
- [Credits](#-credits)

---

## 🌟 Key Features

### 🚀 Powerful Grid Generation

<img width="951" height="601" alt="image" src="https://github.com/user-attachments/assets/2e2c3253-7ab3-46e2-9b60-440de5077ee3" />


* **Cartesian Product Engine:** Automatically generates every permutation of your input settings. Test unlimited Samplers, Schedulers, CFG scales, Sizes, Prompts, LoRA combinations all in one go.
* **Visual Config Builder:** A point-and-click GUI for building complex sampler configurations — no JSON editing required. Searchable dropdowns, drag sliders, folder expansion, LoRA stacking, and trigger word management all built in.
* **Non-Standard Model Support:** Full support for SD3, Flux, Z-Image, and other non-standard architectures with automatic latent channel detection.
* **Multi-Model Support:** Test multiple checkpoints in a single run by passing an array of model names or folder paths.
* **Model Folder Expansion:** Use `"model": "FolderName/"` to test all checkpoints in a folder automatically — perfect for comparing different versions or architectures.
* **CLIP Skip Support:** Control which CLIP layer to use for text encoding with the `clip_skip` parameter — essential for anime models (typically -2) vs realistic models (typically 0).
* **Intelligent CLIP Encoding:** Automatically detects when multiple models are used and handles CLIP encoding correctly per-model to ensure accurate results.
* **Batch Encoding on Model Switch:** When switching between models, all prompts are batch-encoded at once rather than per-generation, resulting in 3-6x faster encoding for multi-model workflows.
* **Multi-LoRA Stacking:** Layer multiple LoRAs with custom strengths using the `+` separator. Supports folder expansion for testing entire LoRA directories.
* **LoRA Trigger Word Filtering:** Use `lora_omit_triggers` to exclude specific trigger words from auto-appended LoRA triggers, giving you fine control over prompts.
* **Random LoRA Selection:** Randomly select LoRAs from folders with `[count,strength]` or `[count,strength,random]` syntax. Supports both reproducible (seed-based) and truly random selection modes.
* **Auto LoRA Trigger Words:** Automatically fetches and appends LoRA trigger words from CivitAI API using SHA256 hash lookup. Results are cached locally for offline use.
* **Multi-Seed Generation:** Add extra random variations per config with the `add_random_seeds_to_gens` parameter — perfect for evaluating consistency.
* **Smart Caching:** Intelligently skips model and LoRA reloading when consecutive runs share the same resources, making generation instant for parameter tweaks.
* **Stop & Resume:** Intelligent skip detection — if you stop a generation mid-run, resuming will skip already-generated images and continue where you left off.
* **Advanced Skip Logic:** Uses conditioning tensor hashing to detect prompt changes even when using pre-encoded conditioning from CLIP nodes.
* **LoRA Compatibility Detection:** Automatically detects and skips incompatible LoRAs with detailed error reporting, preventing log spam.
* **Graceful Interruption:** Cancel button now stops ALL remaining jobs (not just current generation) and saves all completed work including pending remote VAE decodes.
* **VAE Batching:** Includes a `vae_batch_size` input to batch decode images, significantly speeding up large grid runs.
* **Live Dashboard Updates:** Configure `flush_batch_every` to update the dashboard incrementally (e.g., every 4 images) instead of waiting for the entire batch to complete.
* **Remote VAE Support:** Offload VAE decoding to remote servers (HuggingFace endpoints or local) for 20-30% faster generation and lower VRAM usage.
* **VAE Selection & Iteration:** Test multiple VAEs per config — add individual VAEs from a searchable dropdown or expand entire VAE folders. Each VAE becomes a dimension in the Cartesian grid.
* **Attention Mode Testing:** Grid-test different attention implementations (`default`, `sageattn`, `xformers`, etc.) as a combinatorial dimension. Find the optimal attention backend for your hardware.
* **CLIP Encoding Combinators:** Advanced prompt syntax with `AND`, `CAT`, `AVG(weight)`, and `BREAK` keywords for multi-segment conditioning. `AND` creates separate conditioning entries, `CAT` concatenates token contexts, `AVG(0.5)` blends prompts, and `BREAK` forces new 77-token chunks.
* **Model-Specific Prompts:** Per-config prompt prefix and suffix fields for model-appropriate quality tags (e.g., prepend `"masterpiece, best quality"` for anime models, append `"4k, photorealistic"` for realistic models).
* **Recursive Cartesian Prompt Builder:** Visual chip-based prompt editor with `{option1|option2|option3}` syntax that expands into all combinations. Nest multiple groups for exponential prompt variation.
* **Model & LoRA On/Off Switches:** Toggle individual models and LoRAs on/off without removing them from the config. Bypassed entries are omitted from both the preview JSON and the generated config output.
* **Drag-and-Drop Reorder:** Rearrange models, LoRAs, text encoders, and VAEs by dragging them into your preferred order.
* **Per-Config Resolutions:** Override the sampler node's global resolution with per-config resolution lists. Each resolution becomes a Cartesian dimension.
* **Redesigned Resolution Picker:** Nested dropdown with categorized presets (SD 1.5, SDXL, Flux, etc.) and a custom resolution editor for saving your own presets.
* **Seed Behavior Settings:** Per-config seed control — `fixed` uses the node seed, `randomize` generates a new seed per image, `full_run_seed` overrides the node seed entirely.
* **Configurable Label Mode:** Toggle overlay labels on dashboard cards showing model, sampler, scheduler, steps, CFG, and other parameters directly on each image.
* **GPU Cooldown:** Automatic pause between generations to prevent thermal throttling. Configurable interval, duration, and optional VRAM clearing.
* **Image Metrics (optional):** Run Settings checkbox that stores sharpness, entropy, colorfulness, contrast, luma and saturation for every saved image in the manifest (see [Image Metrics](#image-metrics-optional)).

### 🌐 Distributed Processing (Multi-Machine)
* **Multi-Worker Distribution:** Split large grid runs across multiple ComfyUI instances. A master node coordinates job distribution while workers process images in parallel and upload results back.
* **One-Click Setup:** Enable distribution in the Config Builder's Distribution Settings section. Add worker URLs (e.g., `http://192.168.1.100:8188`) and toggle "Enable Distribution."
* **Worker Connection Testing:** Test connectivity to each worker with a single click — status dots show green (reachable), yellow (testing), or red (error).
* **Master Text Encoding:** Optional "Use Master Text Encoding" toggle — the master pre-encodes ALL unique prompts (including LoRA-specific combinations) and sends the encoded conditionings to workers. Workers skip CLIP encoding entirely, saving significant time on prompt-heavy grids.
* **Model Sync:** Optional "Sync Models to Workers" toggle — automatically transfers checkpoint and LoRA files from the master to workers that don't have them.
* **Smart Job Queue:** Thread-safe job distribution with automatic re-queuing on worker timeout (configurable, default 600s) and max 3 retries before abandoning a job.
* **Per-Worker Logging:** Final generation summary shows each worker's completed/failed job counts for full visibility into the distributed run.
* **Live Distribution Status:** Dashboard receives real-time distribution status updates showing worker activity and progress.
* **Backward Compatible:** Workers gracefully fall back to local CLIP encoding if master-encoded conditionings aren't available.

### 🔍 Upscaling & Post-Processing
* **Array-Based Upscaling:** Define multiple upscale configurations, each with its own mode, models, ratios, and denoise values. All combinations are expanded via Cartesian product — test 3 upscale models × 2 ratios × 2 denoise values = 12 upscaled variants per source image.
* **Five Upscaling Modes:** `hires_only` (latent upscale + re-denoise), `model_only` (upscale model like RealESRGAN), `model_then_hires` (upscale model followed by hires denoise pass), `seedvr2` (SeedVR2 diffusion-based upscaler), and `florence2_hires` (Florence2-driven targeted hi-res fix — see below).
* **Florence2 Hi-Res Fix:** Targeted region inpaint using [kijai/ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2)'s referring-expression segmentation. Type what you want fixed (`face`, `hands`, `eyes`, `clothing`) as a free-text input; Florence2 detects the polygon, USCG crops to that region, runs a megapixel-scaled hi-res inpaint pass with your chosen denoise/sampler/steps, and pastes the result back with feathered blending. Pick model/LoRA from each image's original manifest entry or use the Builder's currently-loaded one. Grid-test face fixes the same way you grid-test base configs — combine with other upscale steps in one pipeline for a base-hi-res → face-fix chain. Requires the `ComfyUI-Florence2` custom node (auto-detected at job start with a clear install hint if missing).
* **Upscale Model Discovery:** Automatically detects installed upscale models from ComfyUI's `upscale_models` folder with a searchable dropdown in the Config Builder.
* **Tiled VAE Support:** Enable tiled VAE decoding for upscaled images to handle large resolutions without running out of VRAM.
* **Only Final Output Saved:** When upscaling is enabled, only the upscaled version is saved — the intermediate base image is automatically skipped, keeping your session clean.

### 📦 Favorites Export & Workflow Packing
* **Pack Full ComfyUI Workflow:** Embed the entire current ComfyUI graph into exported favorite images as PNG metadata. Drag the exported image back into ComfyUI to restore the full workflow.
* **Pack Config Data as Nodes Workflow:** Generate a per-image pure-nodes workflow from each image's specific config (sampler, scheduler, steps, CFG, model, LoRA, prompt). Each exported image gets its own workflow matching exactly how it was generated.
* **Cleaned Favorites Manifest:** Export only favorited items in the manifest — non-favorited entries are excluded from the copied manifest.json.
* **Delete All Non-Favorited Items:** One-click cleanup to permanently delete all non-favorited images from a session and update the manifest. Includes a confirmation dialog to prevent accidents.
* **Export Positive Prompt as .txt:** Save the positive prompt alongside each exported image as a plain text file — useful for training datasets.
* **Organize by Prompt or LoRA:** Optionally sort exported favorites into subfolders grouped by unique prompt or LoRA combination.

### 🎨 Interactive Dashboard (The "IDE")
* **Infinite Canvas with Pan/Zoom:** Google Maps-style navigation with mouse drag, mousewheel zoom, and keyboard shortcuts.
* **Virtual Scrolling:** Ultra-optimized rendering handles thousands of images smoothly by only loading visible items — scroll through 5000+ images without lag.
* **Mobile Touch Support:** Full pinch-to-zoom and pan gestures on mobile devices with optimized touch controls.
* **Fullscreen Mode:** Click the fullscreen button (⛶) to expand the dashboard to fill your entire screen.
* **Favorites System:** Star your best images with a ⭐ button — favorites are collected in a separate gold JSON bar for easy export.
* **Smart Filtering:** Toggle visibility by Model, Sampler, Scheduler, Denoise, or LoRA type. **Shift+Click** to isolate a single filter for quick A/B testing.
* **Intelligent Sorting:** Instantly sort your grid by **Oldest**, **Newest**, or **Fastest** (generation time). Your preference is saved to localStorage.
* **Go to Image #:** Jump directly to any image number with the "Go to #" input field in the header.
* **Auto-Load Sessions:** Dashboard automatically loads when generation starts — no manual session name entry needed.
* **Session Management:** Save and load previous testing sessions directly from the UI.
* **Keyboard Navigation:** `Space` to scroll rows, `Arrow Keys` to pan, `+/-` to zoom, `0` to reset, `F` to auto-fit.
* **Real-Time ETA Progress Bar:** Shows estimated time remaining during generation with a progress bar in the dashboard header.
* **Unified Settings Panel:** Gear icon (⚙️) opens a consolidated settings panel with Session management, Grid Display controls, Export Favorites, and Scan External Directory — replacing the old separate SESSION button.
* **Scan External Directory:** Load image sessions from any folder (including other ComfyUI instances) into the dashboard for viewing and filtering. Reads PNG metadata for full config display.
* **Export Favorites:** Copy all favorited images to a dedicated export folder with optional CivitAI metadata packing for sharing.

### ⚡ The "Revise & Generate" Workflow
* **One-Click Revision:** Click "REVISE" on any image to open a detail view.
* **Complete Metadata View:** Shows model, seed, prompts (with trigger words), and all generation parameters.
* **Instant Tweak:** Adjust CFG, Steps, or Sampler for *just that specific image*.
* **Generate New:** A "GENERATE NEW" button queues the new variation immediately without needing to disconnect wires or change the main batch.
* **Similarity Reel:** The revision modal shows a side-scrolling reel of all other images that share the same seed, allowing for perfect A/B comparison.
* **Multiple Prompts:** Use an array to run multiple prompts in one run without re-running your entire workflow: `["picture of a forest", "mountains at night", "masterpiece, painting of dog"]`

### 🧹 Curation & JSON Export
* **Rejection System:** Click the red **"✕"** on bad generations to hide them.
* **Triple JSON Bars (Horizontal Layout):**
    * **Green Bar (Left):** Automatically groups all *accepted* configs into a clean, optimized JSON array ready for copy-pasting.
    * **Gold Bar (Center):** Contains all *favorited* configs — your best-performing settings.
    * **Red Bar (Right):** Collects all *rejected* configs so you know exactly what settings to avoid.

---

## Optional Companion Plugins

Some advanced features live in separate companion plugins:

- **[ComfyUI-USCG-RemoteVAE](https://github.com/JasonHoku/ComfyUI-USCG-RemoteVAE)**
  — Offload VAE decoding to HuggingFace Remote VAE endpoints
  (saves VRAM for large latents). Install via Comfy Manager
  (search "USCG Remote VAE") or `git clone` into `custom_nodes/`.

- **[ComfyUI-USCG-Distributed](https://github.com/JasonHoku/ComfyUI-USCG-Distributed)**
  — Distribute grid generation across multiple ComfyUI machines on
  your LAN (master + worker model). Install via Comfy Manager
  (search "USCG Distributed") or `git clone` into `custom_nodes/`.

- **[ComfyUI-USCG-CivitAI](https://github.com/JasonHoku/ComfyUI-USCG-CivitAI)**
  — Auto-detect LoRA trigger words and look up model/LoRA metadata from
  CivitAI (by file hash). Install via Comfy Manager (search "USCG CivitAI")
  or `git clone` into `custom_nodes/`.

---

## 📦 Installation

1. Navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```

2. Clone this repository:
    ```bash
    git clone https://github.com/YOUR_USERNAME/ComfyUI-Ultimate-Auto-Sampler-Config-Grid-Testing-Suite.git
    ```

3. Restart ComfyUI.

The Config Builder node is included automatically — no separate installation is required.

**Requirements:**
- ComfyUI (latest version recommended)

**File Structure:**
```
ComfyUI/custom_nodes/ComfyUI-Ultimate-Auto-Sampler/
├── __init__.py                      # Server routes, API endpoints, node mappings
├── sampler_node.py                  # Main "Sampler Grid" node
├── dashboard_node.py                # "Dashboard Viewer" node
├── config_builder_node.py           # "Config Builder" node backend
├── json_text_node.py                # "Smart JSON" node
│
├── generation_orchestrator.py       # Main orchestration layer & generation loop
├── image_generation.py              # KSampler & VAE wrappers
├── model_loader.py                  # Checkpoint/LoRA/VAE loading and patching
├── model_cache.py                   # 3-tier model caching system
├── remote_vae.py                    # Remote VAE offloading
│
├── batch_encoding.py                # CLIP batch encoding with combinator support
├── conditioning_cache.py            # Disk-based conditioning cache
├── lora_utils.py                    # LoRA searching & validation
├── trigger_words.py                 # CivitAI trigger word fetching & prompt assembly
├── config_utils.py                  # Config parsing, expansion & Cartesian products
├── directory_scanner.py             # External directory session scanner
├── metadata_packer.py               # PNG/WebP metadata embedding for exports
├── manifest_utils.py                # Manifest read/write/merge helpers
├── html_generator.py                # Reads /resources/ to build dashboard HTML
│
├── distribution_manager.py          # Distributed job queue coordinator (master)
├── distribution_worker.py           # Worker thread for remote processing
├── distribution_routes.py           # Distribution API endpoints
│
├── web/                             # [ComfyUI Integration Layer]
│   ├── dashboard.js                 # Dashboard node registration & messaging
│   ├── smart_json_text.js           # JSON node & syntax highlighting
│   └── conf_builder/                # [Config Builder UI Modules]
│       ├── conf-builder-main.js           # Entry point, widget registration
│       ├── conf-builder-ui-components.js  # Reusable UI (dropdowns, sliders, cards)
│       ├── conf-builder-config-management.js # Save/Load/state rendering logic
│       ├── conf-builder-utilities.js      # Data fetching, caching, calculations
│       └── conf-builder-distribution.js   # Distribution settings UI
│
└── resources/                       # [Dashboard SPA Core]
    ├── template.html                # HTML skeleton for dashboard
    ├── report.css                   # Styling (infinite canvas, cards, modals)
    ├── logic_state.js               # Global state variables
    ├── logic_utils.js               # API & helper functions
    ├── logic_ui.js                  # UI renderer (DOM creation, modals)
    ├── logic_virtual.js             # Virtual scroll engine & viewport
    ├── logic_pipeline.js            # Sorting & filtering pipeline
    ├── logic_events.js              # Event listeners (messages, keyboard)
    └── logic_init.js                # Initialization & startup
```

---

## Getting Started

### 1. The Nodes
This suite consists of three nodes found under the `sampling/testing` category:

1. **Ultimate Sampler Grid (Generator):** The engine. It handles model loading, grid generation, and saving.
2. **Ultimate Grid Dashboard (Viewer):** The display. It renders the HTML output.
3. **Ultimate Config Builder:** The visual GUI for building configs without writing JSON.

**Basic Setup:**
* Add the **Generator** node.
* Connect your Checkpoint, CLIP, and VAE (optional, see "Hybrid Inputs" below).
* Add the **Viewer** node.
* Connect the `dashboard_html` output from the Generator to the input of the Viewer.
* *(Optional)* Add the **Config Builder** node and connect its `configs_json` output to the Generator's `configs_json` input.

### 2. Generator Node Parameters

#### Core Settings
* **`ckpt_name`**: Default checkpoint to use (can be overridden by `model` in JSON or optional input).
* **`positive_text`** / **`negative_text`**: Prompts. Supports arrays: `["prompt 1", "prompt 2"]` or JSON arrays.
* **`seed`**: Base seed. Each config uses this seed unless overridden.
* **`denoise`**: Denoise strength(s). Supports single value, array, or comma-separated: `"1.0"` or `"0.8, 0.9, 1.0"`.

#### Performance Settings
* **`vae_batch_size`**: How many images to decode at once.
  - `4` (default): Good for 8-12GB VRAM
  - `12-24`: For 24GB+ VRAM  
  - `-1`: Decode all images at once (fastest, but risky)
  
* **`flush_batch_every`**: Update dashboard every X images (0 = use VAE batch size).
  - `0`: Wait until VAE batch is full
  - `4`: Update dashboard every 4 images (recommended for live monitoring)

#### Advanced Settings
* **`overwrite_existing`**: 
  - `False` (default): Skip already-generated images (resume mode)
  - `True`: Re-generate everything
  
* **`add_random_seeds_to_gens`**: Generate X extra random variations per config.
  - `0` (default): Only use base seed
  - `3`: Generate 3 additional random seed variations per config
  - Random seeds are deterministic per base seed — changing base seed generates new random variations

* **`lookup_and_append_lora_triggerwords`**: Automatically fetch and append LoRA trigger words.
  - `False` (default): Use prompts as-is
  - `True`: Calculate SHA256 hash of each LoRA, query CivitAI API for trigger words, cache results locally, and prepend to prompts
  - Cache stored in `loras_tags.json` for offline use

* **`session_name`**: Folder name where results are saved (`ComfyUI/output/benchmarks/{session_name}/`).

### 3. The JSON Configuration
The `configs_json` widget determines your grid. It accepts an array of objects. All fields support single values or arrays. You can write JSON by hand or use the [Config Builder Node](#-config-builder-node-visual-ui) to build it visually.

**Basic Example:**
```json
[
  {
    "sampler": ["euler", "dpmpp_2m"],
    "scheduler": ["normal", "karras"],
    "steps": [20, 30],
    "cfg": [7.0, 8.0]
  }
]
```
*Generates 16 images (2 samplers × 2 schedulers × 2 steps × 2 cfg)*

**Advanced Features:**

#### Multi-Model Testing
```json
[
  {
    "sampler": "euler",
    "scheduler": "normal", 
    "steps": 20,
    "cfg": 7.0,
    "model": [
      "sd_xl_base_1.0.safetensors",
      "juggernautXL_v9.safetensors",
      "ponyDiffusionV6XL_v6.safetensors"
    ]
  }
]
```
*Tests 3 different models with the same settings*

#### Folder Expansion
```json
[
  {
    "sampler": "euler",
    "scheduler": "normal",
    "steps": 20, 
    "cfg": 7.0,
    "model": "sdxl_models/"
  }
]
```
*Trailing `/` tests ALL models in the folder.*

#### Multi-LoRA with Stacking
```json
[
  {
    "sampler": "euler",
    "scheduler": "normal",
    "steps": 20,
    "cfg": 7.0,
    "lora": [
      "None",
      "style_lora.safetensors:0.8:1.0",
      "style_lora.safetensors:0.5:1.0 + detail_lora.safetensors:1.0:1.0",
      "loras_folder/"
    ]
  }
]
```
*LoRA format: `filename:strength_model:strength_clip`*  
*Stack with `+` separator: `lora1:0.8:1.0 + lora2:1.0:1.0`*

#### Resolution Grid
```json
// In resolutions_json input:
[
  [1024, 1024],
  [1024, 1536], 
  [1536, 1024]
]
```

#### Multiple Prompts
```json
// In positive_text input (as JSON array):
[
  "a beautiful landscape, mountains, sunset",
  "cyberpunk city at night, neon lights",
  "portrait of a warrior, detailed armor"
]
```

### 4. Hybrid Inputs (Optional)
The Generator node features built-in widgets for Model Selection and Prompts, but also has **Optional Inputs** for flexibility:
* **Standalone Mode:** Use the dropdown menu to select a checkpoint and type prompts into the text boxes.
* **Hybrid Mode:** Connect a `MODEL`, `CLIP`, `VAE`, or `CONDITIONING` wire. The node will automatically ignore the internal widget and use the connected input instead.
* **Non-Standard Models:** For SD3, Flux, Z-Image, and other architectures:
  - Connect `optional_model`, `optional_clip`, and `optional_vae` from your model loader
  - Connect `optional_positive` and `optional_negative` for pre-encoded conditioning
  - The node automatically detects latent channel count (4 for SD1.5/SDXL, 16 for SD3/Flux/Z-Image)
  - Skip detection uses conditioning tensor hashing to properly detect prompt changes
* **Latent Input:** Connect a `LATENT` to use img2img or upscaling workflows.
  - For SD3/Flux: Use `EmptySD3LatentImage` instead of `EmptyLatentImage`
  - Latent dimensions are automatically preserved

---

## 🔧 Config Builder Node (Visual UI)

<img width="1408" height="867" alt="image" src="https://github.com/user-attachments/assets/7eb54f26-d098-4c80-8bb8-37e85a1ef1dc" />

The Config Builder node replaces manual JSON editing with an intuitive visual interface. Instead of writing configuration arrays by hand, you can point and click to select models and LoRAs, drag sliders to adjust strengths, expand entire folders, stack multiple LoRAs, manage trigger words, and preview JSON output in real-time.

All configuration data is automatically synchronized with the Ultimate Sampler Grid node.

### Basic Setup

1. **Add the Node:** Right-click on the canvas → `Add Node` → `sampling/testing` → `Ultimate Config Builder`
2. **Connect to Sampler:**
   - Connect the `configs_json` output to the `configs_json` input of the `Ultimate Sampler Grid (Generator)` node
   - Connect the `session_name` output to the `session_name` input of the generator (optional)
3. **Start Building:** The Config Builder UI will appear in the node with one config array ready to customize.

### Understanding the Interface

```
┌─────────────────────────────────────────────────┐
│ 📁 Session Management                          │
│ (Session name, Load session, Refresh button)   │
├─────────────────────────────────────────────────┤
│ 💾 Config Management                           │
│ (Save/Load JSON presets, Auto-save toggle)     │
├─────────────────────────────────────────────────┤
│ ⚙️ Config Arrays                               │
│ ┌─────────────────────────────────────────┐   │
│ │ Config 1                                 │   │
│ │ - Samplers, Schedulers, Steps, CFG      │   │
│ │ - Models Section (expandable)            │   │
│ │ - LoRAs Section (expandable)             │   │
│ │ - Omit Triggers Section                  │   │
│ └─────────────────────────────────────────┘   │
│ ┌─────────────────────────────────────────┐   │
│ │ Config 2...                              │   │
│ └─────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│ 📋 JSON Preview                                │
│ (Real-time preview of generated config)        │
└─────────────────────────────────────────────────┘
```

### Session Management

Located at the top of the interface, this section controls the output session name for your tests.

* **Session Name:** Sets the folder name where results are saved (`ComfyUI/output/benchmarks/{session_name}/`). Enter a descriptive name like `"sdxl_quality_test"` or `"character_lora_comparison"`. Changes automatically update the connected Sampler Grid node.
* **Load Session:** Load a previously saved session's metadata from a searchable dropdown.
* **Refresh Models/LoRAs:** Reload the list of available models and LoRAs from disk. Use when you've added new files or the dropdowns are showing stale data.

### Config Management (Save/Load Presets)

Save and load configuration presets as JSON files for reuse across projects.

* **Config Name (Filename):** Name for the JSON preset file. Stored at `ComfyUI/output/benchmarks/configs/{config_name}.json`.
* **Load Saved Config:** Searchable dropdown showing all available `.json` config files. Replaces current config arrays with the loaded preset.
* **Save Config Now:** Manual save button to persist current configuration to disk. Creates a reusable JSON file containing all config arrays, models, LoRAs, trigger settings, and strength locks.
* **Auto-Save (2s):** Toggle to automatically save the config file every 2 seconds. Recommended for long editing sessions.

### Config Arrays

Config Arrays are the building blocks of your testing grid. Each array defines a set of parameters that will be multiplied into all possible combinations.

### Fast CivitAI Info Lookup

<img width="1178" height="505" alt="image" src="https://github.com/user-attachments/assets/68146aae-c785-4e4b-bc5e-8a1962a14270" />

<img width="1072" height="506" alt="image" src="https://github.com/user-attachments/assets/a11178e2-0745-4e56-a9b4-5225f968b051" />

<img width="1074" height="271" alt="image" src="https://github.com/user-attachments/assets/7849d711-f939-47b5-be20-c44eb664fd19" />

#### Basic Parameters

* **Config Name:** Label for this config group (purely organizational, not used in generation).
* **Samplers:** Comma-separated list of samplers to test (e.g., `"euler, dpmpp_2m, dpmpp_2m_sde"`).
* **Schedulers:** Comma-separated list of schedulers (e.g., `"normal, karras, exponential"`).
* **Steps:** Comma-separated list of step counts (e.g., `"20, 30, 40"`).
* **CFG:** Comma-separated list of CFG scale values (e.g., `"6.0, 7.0, 8.0"`).
* **Samplers Dropdown:** Searchable dropdown auto-populated with all available samplers from the connected sampler node. Select multiple to add them as a comma-separated list.
* **Schedulers Dropdown:** Searchable dropdown auto-populated with all available schedulers. Works the same as the samplers dropdown.
* **Attention Mode:** Dropdown selector for attention implementations (`default`, `sageattn`, `xformers`, etc.). Select multiple to test as a grid dimension.

#### Iteration Count Display

```
⏱️ Iterations: 144
```

Shows the total number of images this config array will generate:
**Formula:** Models × LoRAs × Samplers × Schedulers × Steps × CFG  
Updates in real-time as you add/remove items.

#### Config Array Controls

* **➕ Add Model** — Adds a new model slot to the Models section.
* **➕ Add LoRA** — Adds a new LoRA slot to the LoRAs section.
* **📋 Duplicate Config** — Creates an exact copy of this config array (useful for variations).
* **🗑️ Delete Config** — Removes this entire config array.
* **▼ Collapse/Expand** — Toggle visibility of Models and LoRAs sections.
* **➕ Add VAE** — Adds a new VAE slot to the VAEs section for testing multiple VAE decoders.

### Models Section

Each model card includes:

* **Model Name (Searchable):** Dropdown with fuzzy search through all available checkpoints. Supports folder structure display (e.g., `"subfolder/model.safetensors"`).
* **Type Selection:**
  - **Single File (default):** Test just this specific model.
  - **Folder (Separate):** Expand folder — each model becomes a separate test. Multiplies iteration count by number of models in folder.
  - **Folder (Combined):** *Reserved for future use.*
* **🗑️ Remove Model:** Deletes this model from the config (at least one model is always required).
* **🔘 On/Off (Bypass) Switch:** Toggle a model on or off without removing it. Bypassed models are grayed out and excluded from the generated config output.
* **🔍 CivitAI Model Lookup:** Opens a metadata panel showing model info fetched from CivitAI (name, base model, tags, creator, description). Includes a direct link to the model page.

### LoRAs Section

Each LoRA card includes:

* **LoRA Name (Searchable):** Dropdown with fuzzy search. Supports special syntax — end with `/` for folder expansion, end with `/*` for combined folder stack.
* **Type Selection:**
  - **Single File:** Load this specific LoRA.
  - **Folder (Separate):** Test each LoRA in the folder individually with the same strength settings.
  - **Folder (Combined):** Stack ALL LoRAs in the folder into a single load.
* **Model Strength Slider:** Range 0.0–2.0 (default 1.0). Controls how strongly the LoRA affects the model. Number input accepts values beyond slider range.
* **CLIP Strength Slider:** Range 0.0–2.0 (default 1.0). Controls how strongly the LoRA affects text encoding. Independent from model strength.
* **🔗 Lock Strengths:** Toggle to link model and CLIP strengths so both sliders move together.
* **🔍 Lookup Trigger Words:** Opens the Trigger Word Lookup Modal (fetches from CivitAI API).
* **🚫 Bypass Trigger Fetch:** Toggle to prevent automatic trigger word lookup for this LoRA.
* **🗑️ Remove LoRA:** Deletes this LoRA from the config (at least one is always required).
* **🔘 On/Off (Bypass) Switch:** Toggle a LoRA on or off without removing it. Bypassed LoRAs are grayed out and excluded from both the preview JSON and the Python node's `config_json` output.

### LoRA Trigger Words

The Config Builder integrates with CivitAI's API to automatically fetch and manage trigger words for your LoRAs.

**How It Works:**
1. When you select a LoRA, the node calculates its SHA256 hash.
2. The hash is sent to CivitAI to retrieve metadata.
3. Results are cached in `ComfyUI/output/benchmarks/loras_tags.json`.
4. Cached data is reused on subsequent loads (offline-capable).

#### Trigger Word Lookup Modal

Click the **🔍 Lookup Trigger Words** button to open the modal:

```
╔════════════════════════════════════════════╗
║ LoRA Trigger Word Manager                 ║
║                                            ║
║ LoRA: character_style_v2.safetensors      ║
║ Model: Character Style LoRA v2             ║
║ Base Model: SDXL 1.0                       ║
║ Creator: ArtistName                        ║
║                                            ║
║ Tags: character, anime, style              ║
║                                            ║
║ Trigger Words:                             ║
║ ☑️ character_name                          ║
║ ☑️ red_hair                                ║
║ ☑️ blue_eyes                               ║
║ ☐ masterpiece  (excluded)                 ║
║ ☐ best quality (excluded)                 ║
║                                            ║
║ [✅ Save Selections]  [❌ Cancel]          ║
╚════════════════════════════════════════════╝
```

* **Checkboxes** to select which trigger words to exclude.
* **Metadata display** showing model info before deciding.
* **Persist selections** — saved exclusions apply to all future generations.
* **CivitAI link** to open the model page in a new tab.

Excluded triggers are stored in the config array's `lora_omit_triggers` array and automatically filtered during generation.

#### Trigger Word Placement Control

Control exactly where trigger words are inserted in your prompts. For each LoRA with trigger words, you can choose:

* **None (Default):** Trigger words are **prepended** to the beginning of the positive prompt. Best for character LoRAs and style LoRAs that need early influence.
* **Positive Start:** Explicitly prepend to positive prompt (same as None/default).
* **Positive End:** **Append** trigger words to the end of the positive prompt. Best for quality tags and detail LoRAs.
* **Negative Start:** **Prepend** trigger words to the negative prompt. Best for negative embeddings and "anti-style" LoRAs.
* **Negative End:** **Append** trigger words to the negative prompt. Best for quality-control negative tags.

Set placement by opening the trigger word modal, selecting placement from the dropdown at the bottom, and saving. The placement is stored in `lora_triggerwords_append_settings`.

### LoRA Bypass States

Bypass states allow you to mark specific LoRAs to skip automatic trigger word fetching.

**When to use:** LoRAs without triggers, offline work, custom workflows where you're manually adding triggers, or speed optimization for known LoRAs.

Toggle the **🚫 Bypass Trigger Fetch** checkbox on a LoRA card. The card shows a ⚠️ indicator when bypassed. Bypass state is saved in the `lora_bypass_states` object.

### Strength Locking

Click the **🔗 Lock Strengths** button on a LoRA card to link model and CLIP strengths. When locked, both sliders move together maintaining the ratio. For example, if locked at 1.0/0.8 (ratio 1.25), changing model to 1.2 sets CLIP to 0.96.

Useful for maintaining tested ratios and quickly scaling both strengths up/down together.

### Folder Expansion (Models & LoRAs)

Folder expansion allows you to test entire directories of models or LoRAs without manually adding each file.

**Model Folder Expansion:** Select a folder path ending with `/` and set type to "Folder (Separate)." All `.safetensors` files in that folder are tested individually with the same settings.

**LoRA Folder Expansion — Separate:** Path ending with `/`, type "Folder (Separate)." Each LoRA in the folder is tested individually. All use the same strength settings, and trigger words are fetched for each automatically.

**LoRA Folder Expansion — Combined:** Path ending with `/*`, type "Folder (Combined)." ALL LoRAs in the folder are stacked into a single load. Creates one config with multiple LoRAs stacked at the same strength. Trigger words from all LoRAs are combined. Great for quality enhancement stacks, multi-concept combinations, or testing "everything at once" scenarios.

### LoRA Stacking

Combine multiple LoRAs into a single generation by adding multiple LoRA slots to a config array.

**Manual Stacking:** Click **➕ Add LoRA** multiple times and configure each independently with different files, strengths, and trigger settings. All selected LoRAs are applied simultaneously.

**Generated config format:**
```json
{
  "lora": "character_lora.safetensors:1.0:1.0 + style_lora.safetensors:0.8:0.8 + detail_lora.safetensors:0.5:0.5"
}
```

**Mixed Expansion + Manual:** You can combine folder expansion with manual stacking — for example, a combined quality folder plus a specific character LoRA stacked together.

### VAEs Section

Each VAE card includes:

* **VAE Name (Searchable):** Dropdown with fuzzy search through all available VAE models. Supports folder structure display.
* **Type Selection:**
  - **Single File:** Test just this specific VAE.
  - **Folder (Separate):** Expand folder — each VAE becomes a separate iteration, multiplying the total grid size.
* **🗑️ Remove VAE:** Deletes this VAE from the config.

VAEs are treated as a Cartesian dimension just like samplers or models. If you add 3 VAEs, your total iteration count is multiplied by 3.

### Attention Modes

Select which attention backend implementations to test. Available options depend on your system (e.g., `default`, `sageattn`, `xformers`). Like other parameters, you can select multiple modes and they become a grid dimension.

### Skip Layer Guidance (DiT models)

**Extra Model & Sampling Options → Skip Layer Guidance** wraps ComfyUI's `SkipLayerGuidanceDiTSimple` node: between Start % and End % the negative (uncond) pass skips the listed transformer blocks, which sharpens structure on DiT models (MOSS-Image, Flux.2, SD3, ...). Fields: Double Layers, Single Layers (comma-separated block indices), Start %, End %. The defaults are the official MOSS-Image workflow's values (single block `22`, 0–50 %); ComfyUI's generic default is `7, 8, 9`. Every image in the config array gets the patch, so compare with/without by duplicating the array.

JSON: `"use_slg": true, "slg_single_layers": "22", "slg_double_layers": "", "slg_start_percent": 0.0, "slg_end_percent": 0.5`

### MOSS-Image, Flux.2, Krea 2 (16 px latent models)

Set **Model Type = Diffusion Model**, pick the diffusion file (e.g. `moss_image_sft_int8_convrot.safetensors`), the text encoder (`qwen3.5_9b_w4a8_gptq.safetensors`), **CLIP Type = `moss_image`**, and the Flux.2 VAE (`flux2-vae.safetensors`) in the VAEs section.

* The CLIP Type list is read from your installed ComfyUI's `CLIPLoader`, so new model families appear without a plugin update. If you leave it at `stable_diffusion` for a diffusion model, the generator picks the CLIP type that matches the detected model family and logs it (MOSS-Image needs `moss_image` to get its chat template; with `stable_diffusion` ComfyUI builds a generic Qwen3.5 encoder).
* The empty latent follows the model's latent format: 16 px per latent cell for the Flux.2 family, 8 for SD / SDXL / Flux.1. Use sizes that are multiples of 16; a 832x1216 job renders at 832x1216. Earlier releases always used 8, which made 16x models render at double size and run ~10x slower than the stock workflows.
* Pre-encoding loads the text encoder once per pass instead of once per prompt (ComfyUI's dynamic VRAM loader re-prepares a model on every encode, which is the repeated `Model ... prepared for dynamic VRAM loading` log line). What remains is the encoder itself: the Qwen3.5-9B encoder costs roughly 0.8 s per prompt on a 16 GB card, the same as the stock workflow, so a 60-prompt grid spends about a minute encoding before the first image. SDXL's CLIPs take about 0.15 s per prompt, Krea 2 and Qwen-Image about 0.3 s. The pre-encode summary line reports the measured ms/prompt.

### Model-Specific Prompts

Per-config prompt prefix and suffix fields that are prepended/appended to all prompts generated by that config array:

* **Prompt Prefix:** Text prepended to the positive prompt (e.g., `"masterpiece, best quality, "` for anime models).
* **Prompt Suffix:** Text appended to the positive prompt (e.g., `", 4k, photorealistic"` for realistic models).

These fields are especially useful when testing multiple models that require different quality tags or style descriptors.

### Recursive Cartesian Prompt Builder

A visual chip-based prompt editor that supports variation groups using `{option1|option2|option3}` syntax:

```
a {photo|painting|sketch} of a {cat|dog} in a {forest|city}
```

This expands into all combinations (3 × 2 × 2 = 12 prompts):
- "a photo of a cat in a forest"
- "a photo of a cat in a city"
- "a painting of a dog in a forest"
- ... etc.

The editor provides a visual chip interface for building and editing these variation groups. Each group is displayed as a clickable chip that can be expanded to add/remove options. Multiple groups can be nested for exponential prompt variation.

---

### Image Metrics (Optional)

Run Settings has a **Compute Image Metrics** checkbox (default OFF). When it is on, every saved still image (base generations and upscales; videos are skipped) gets a `metrics` object in `manifest.json`, computed from the decoded image at save time (about 1-5 ms per image on the GPU, nothing when the box is unchecked):

| Metric | What it is | Range |
|---|---|---|
| `sharpness` | Variance of the 3x3 Laplacian of luma on the 0-255 scale (OpenCV-style "variance of Laplacian") | 0 = flat, large = lots of edges |
| `entropy` | Shannon entropy of the 256-bin luma histogram | 0-8 bits |
| `colorfulness` | Hasler and Suesstrunk (2003) colorfulness | ~0 grey, >100 very colorful |
| `contrast` | Standard deviation of luma (RMS contrast) | 0-0.5 |
| `luma` | Mean brightness (Rec.601 luma) | 0-1 |
| `saturation` | Mean HSV saturation | 0-1 |

Notes:
* Metrics are measured on an "analysis image": the long side is block-averaged down to at most 1024 px (by the smallest integer factor), so a 2x upscale stays comparable with its 1024 base image and the cost is capped. The constant is `ANALYSIS_MAX_SIDE` in `image_metrics.py`; changing it would make old sessions incomparable.
* They measure image **signal, not taste**: noise raises sharpness and entropy, and a dark moody image is not a worse one. Use them to rank and filter, then look.
* Upscales are measured again (a derived image never inherits its source's numbers). Dashboard-triggered upscales measure only when the source image had metrics.
* Items produced by remote Distributed workers inside the companion plugin get no metrics yet; master-side saves and upscales do.
* Optional scorer packs (aesthetic / reward models) plug in through `nodes.NODE_CLASS_MAPPINGS` without any import, and their scores are stored next to the built-ins. The shipped entries stay disabled until their node ids are verified against the real packs, so only the built-in metrics are available today. The Builder lists each pack with an installed / not installed hint (text only, nothing is downloaded).
* The setting is session-level (like GPU Cooldown), so it is NOT a per-config field: it lives in `_session_settings.metrics`, not in `_EXPECTED_CONFIG_KEYS` or `expand_configs`.

## 🖥️ Dashboard Interface

### Header Bar
* **Model/Prompt Info:** Shows current model and prompt metadata.
* **Go to Image #:** Jump directly to any image by entering its number (shown in bottom-left of cards).
* **Column Count:** Set fixed grid columns or leave at 0 for auto-sizing.
* **Zoom Controls:** `⊙` (reset), `−` (zoom out), `+` (zoom in).

### Toolbar
* **⚙️ Settings Panel (Gear Icon):** Opens a unified settings panel with four sections:
  - **Session:** Name input + SAVE/LOAD buttons. Dashboard auto-loads when connected to sampler and generation starts.
  - **Grid Display:** Go to Image #, Column Count, Reset Zoom controls.
  - **Export Favorites:** Export starred images to a dedicated folder with optional CivitAI metadata packing.
  - **Scan External Directory:** Load image sessions from any folder path into the dashboard.

* **ETA Progress Bar:** During generation, a real-time progress bar appears in the header showing estimated time remaining and completion percentage.

* **Filter Groups:** Click colored buttons to toggle visibility:
  - **Model** (Purple): Filter by checkpoint
  - **Sampler** (Cyan): Filter by sampler type
  - **Scheduler** (Blue): Filter by scheduler
  - **Denoise** (Red): Filter by denoise value
  - **LoRA** (Orange): Filter by LoRA configs
  - **Shift+Click:** Isolate single filter (deselect all others)

* **Sort Button:** Cycles between **Sort: Oldest** (default), **Sort: Newest**, and **Sort: Fastest** (by generation time). Preference is saved to localStorage.

* **Fullscreen Button (⛶):** Expand dashboard to fill entire screen.

### Navigation & Controls

**Mouse:**
- Left-click or middle-click drag to pan
- Scroll wheel to zoom in/out
- Right-click on canvas to focus for keyboard controls

**Touch (Mobile/Tablet):**
- Single finger drag to pan
- Two finger pinch/spread to zoom
- Tap card to reveal buttons

**Keyboard:**
- `Space` — Scroll down one row
- `Shift+Space` — Scroll up one row
- `↑↓←→` — Pan in any direction
- `+/-` — Zoom in/out
- `0` — Reset zoom to 1:1
- `F` — Auto-fit first row to viewport width

### Card Overlays
* **Bottom Left:** Index number (#1, #2, etc.) — used for "Go to #" feature.
* **Bottom Right:** Generation time in seconds.
* **Top Left (on hover):** Red ✕ button to reject/hide image.
* **Top Right (on hover):** Gold ⭐ button to favorite image, and green "REVISE" button to open studio view.

### JSON Bars (Bottom — Horizontal Layout)
* **Green Bar (Left — Accepted):** Contains optimized JSON of all currently visible images. Click to select all, then copy-paste back into the `configs_json` widget to refine your batch.
* **Gold Bar (Center — Favorites):** Contains configs of all images you starred with ⭐. Your best-performing settings in one place.
* **Red Bar (Right — Rejected):** Contains the configs of images you deleted with the ✕ button. Know what to avoid.

### Revision Modal

Clicking **REVISE** on a card opens the studio view:
1. **Left:** Full-resolution preview.
2. **Top Right — Read-Only Info:** Model used, seed number, positive prompt (with trigger words if applicable), negative prompt.
3. **Bottom Right — Adjustable Parameters:** Sampler, Scheduler, Steps, CFG, Denoise, LoRA.
4. **Bottom:** "Related Variants" reel showing other images with the same seed.
5. **GENERATE NEW:** Queues the specific config you just edited.

---

## 🎯 Example Workflows

### Quick Quality Test (40 images)
```json
[
  {
    "sampler": ["dpmpp_2m", "euler"],
    "scheduler": ["karras", "normal"],
    "steps": [20, 30],
    "cfg": [6.0, 7.0, 8.0],
    "denoise": "1.0"
  }
]
```

### Multi-Model Comparison (9 images)
```json
[
  {
    "sampler": "euler",
    "scheduler": "normal",
    "steps": 25,
    "cfg": 7.0,
    "model": [
      "model_v1.safetensors",
      "model_v2.safetensors", 
      "model_v3.safetensors"
    ]
  }
]
```
Set `add_random_seeds_to_gens: 2` to get 3 variations per model (27 total images).

### LoRA Stack Testing (24 images)
```json
[
  {
    "sampler": "euler",
    "scheduler": "normal",
    "steps": 25,
    "cfg": 7.0,
    "lora": [
      "None",
      "style.safetensors:0.6:1.0",
      "style.safetensors:0.8:1.0",
      "style.safetensors:1.0:1.0",
      "style.safetensors:0.8:1.0 + detail.safetensors:0.5:1.0",
      "style.safetensors:1.0:1.0 + detail.safetensors:1.0:1.0"
    ]
  }
]
```

### Random LoRA Selection

Randomly select LoRAs from a folder for variety and experimentation:

**Basic:** `"lora": "XL/Styles/[3,0.85]"` — Selects 3 random LoRAs at strength 0.85 (seed-based, reproducible).

**Truly Random:** `"lora": "XL/Styles/[3,0.85,random]"` — Add `random` keyword for non-reproducible selection.

**Dual Strength:** `"lora": "XL/Characters/[2,0.8,0.6]"` — Model strength 0.8, CLIP strength 0.6.

**Combined with Regular LoRAs:**
```json
"lora": "XL/base.safetensors:1.0 + XL/Styles/[2,0.7] + XL/Details/[1,0.5,random]"
```
Loads `base.safetensors` always, picks 2 reproducible style LoRAs, and picks 1 truly random detail LoRA.

### Full Model Folder Test
```json
[
  {
    "sampler": "dpmpp_2m",
    "scheduler": "karras",
    "steps": 25,
    "cfg": 7.0,
    "model": "realistic_models/"
  }
]
```

### CLIP Skip for Anime Models
```json
[
  {
    "sampler": "euler",
    "scheduler": "normal",
    "steps": 28,
    "cfg": 7.0,
    "model": "anime_model.safetensors",
    "clip_skip": -2
  }
]
```
- `clip_skip: 0` (default) — Use last layer (best for realistic models)
- `clip_skip: -2` — Skip 2 layers (best for anime/illustration models)

Test multiple values: `"clip_skip": [0, -1, -2, -3]`

### LoRA Trigger Word Filtering
```json
[
  {
    "sampler": "euler",
    "steps": 28,
    "cfg": 7.0,
    "lora": "style2.safetensors:0.8:0.6",
    "lora_omit_triggers": ["style", "makeup", "jewelry"]
  }
]
```
Filters apply to ALL LoRAs in the stack.

### Model Folder Expansion
```json
[
  {
    "model": ["SD1.5/", "SDXL/", "Flux/"],
    "sampler": "euler",
    "steps": 28
  }
]
```
Tests all models from all three folders with the same settings.

### LoRA Strength Sweep

Find the optimal strength for a new LoRA using the Config Builder: add the same LoRA 6 times with strengths from 0.4 to 1.5, keep everything else minimal (1 sampler, 1 scheduler), and compare the 6 images side by side.

### Quality Enhancement Stack

Create a consistent quality stack using the Config Builder: add 3 LoRAs manually (e.g., detail_enhancer at 1.0, color_boost at 0.6, sharpness at 0.5), use the trigger word modal to set all placements to **Positive End**, exclude generic tags like "masterpiece" and "best quality", then save as a reusable preset.

---

## 📋 Preset Configs

### 🏆 Group 1: The "Gold Standards" (Reliable Realism)

*Tests the 5 most reliable industry-standard combinations.*  
5 samplers × 2 schedulers × 2 step settings × 2 cfgs = **40 images**

```json
[
  {
    "sampler": ["dpmpp_2m", "dpmpp_2m_sde", "euler", "uni_pc", "heun"],
    "scheduler": ["karras", "normal"],
    "steps": [25, 30],
    "cfg": [6.0, 7.0],
    "lora": "None"
  }
]
```

### 🎨 Group 2: Artistic & Painterly

*Tests creative/soft combinations best for illustration and anime.*  
2 samplers × 3 schedulers × 3 step settings × 2 cfgs = **36 images**

```json
[
  {
    "sampler": ["euler", "dpmpp_2m"],
    "scheduler": ["simple", "beta", "normal"],
    "steps": [20, 25, 30],
    "cfg": [1.0, 4.5],
    "lora": "None"
  }
]
```

### ⚡ Group 3: Speed / Turbo / LCM

*Tests ultra-fast configs. (Note: Ensure you are using a Turbo/LCM capable model or LoRA).*  
4 samplers × 3 schedulers × 4 step settings × 2 cfgs = **96 images**

```json
[
  {
    "sampler": ["lcm", "euler", "dpmpp_sde", "euler_ancestral"],
    "scheduler": ["simple", "sgm_uniform", "karras"],
    "steps": [4, 5, 6, 8],
    "cfg": [1.0, 1.5],
    "lora": "None"
  }
]
```

### 🦾 Group 4: Flux & SD3 Specials

*Tests configs specifically tuned for newer Rectified Flow models like Flux and SD3.*  
2 samplers × 3 schedulers × 3 step settings × 2 cfgs = **36 images**

```json
[
  {
    "sampler": ["euler", "dpmpp_2m"],
    "scheduler": ["simple", "beta", "normal"],
    "steps": [20, 25, 30],
    "cfg": [1.0, 4.5],
    "lora": "None"
  }
]
```

### 🧪 Group 5: Experimental & Unique

*Tests niche combinations for discovering unique textures.*  
6 samplers × 4 schedulers × 5 step settings × 4 cfgs = **480 images**

```json
[
  {
    "sampler": ["dpmpp_3m_sde", "ddim", "ipndm", "heunpp2", "dpm_2_ancestral", "euler"],
    "scheduler": ["exponential", "normal", "karras", "beta"],
    "steps": [25, 30, 35, 40, 50],
    "cfg": [4.5, 6.0, 7.0, 8.0],
    "lora": "None"
  }
]
```

---

## 🔧 Performance Tips

### For Large Batches (1000+ images)
1. Set `flush_batch_every: 10-20` to see progress updates without overwhelming the browser.
2. Use `vae_batch_size: 8-12` (balance between speed and VRAM).
3. Enable `overwrite_existing: False` so you can stop/resume safely.

### For Multi-Model Testing
1. Use identical LoRA/prompt settings across models for fair comparison.
2. Use `add_random_seeds_to_gens: 2-3` to evaluate model consistency.

### For Memory-Constrained Systems
1. Lower `vae_batch_size` to 1-2.

---

## 💡 Tips & Best Practices

### Organization
- Use descriptive config names: `"SDXL_Quality_Test_v2"` instead of `"config1"`.
- Group related tests in separate config arrays.
- Save presets early using auto-save to prevent data loss.
- Version your configs with numbers in saved config names.

### Config Builder Workflow Efficiency
- **Duplicate instead of recreate:** Use the Duplicate Config button for variations.
- **Load templates:** Keep a library of saved configs for common test scenarios.
- **Leverage the trigger word modal:** Bulk-exclude unwanted triggers once — they persist.
- **Lock strengths for ratios:** When you find good ratios (e.g., 1.0/0.8), lock them.
- **Collapse unused sections:** Minimize Models/LoRAs sections when not actively editing.

### Trigger Word Management
- Exclude generic terms like "masterpiece", "best quality", "highres" — they often appear but may be redundant.
- Use placement strategically: character/style triggers → Positive Start, quality/detail triggers → Positive End, anti-style triggers → Negative Start.
- Bypass technical LoRAs — technique LoRAs (e.g., "detail tweaker") rarely have meaningful triggers.

### Testing Strategy
- Start small: 1 model × 1 LoRA × 2 samplers = 2 images.
- Expand gradually after confirming the basics work.
- Use separate config arrays — don't mix unrelated tests in one array.
- Monitor iteration counts — the ⏱️ display updates in real-time. Watch for unexpected explosions.
- Save before large grids — always save your config before running 500+ image generations.

---

## ⚠️ Troubleshooting

Please check the [GitHub Issues list](https://github.com/YOUR_USERNAME/ComfyUI-Ultimate-Auto-Sampler-Config-Grid-Testing-Suite/issues) and post any issues there after searching. (Pro-tip: delete `state:open` from the search to see closed issues too.)

### Generation Issues
* **"Session not found":** Ensure the `session_name` matches a folder inside `ComfyUI/output/benchmarks/`.
* **OOM Errors:** Lower the `vae_batch_size` to 1 or 2.
* **Images not resuming:** Make sure `overwrite_existing: False`. Check console for skip messages.
* **Random seeds different each run:** Intentional — random seeds are tied to the base seed.
* **"mat1 and mat2 shapes cannot be multiplied":** Model architecture mismatch. For SD3/Flux/Z-Image, ensure you connect ALL optional inputs. Check LoRA compatibility.
* **Dashboard not auto-loading:** Ensure the dashboard node is connected to the sampler's `dashboard_html` output. Try the "RELOAD / SHOW SESSION" button.

### Dashboard Issues
* **Cards not appearing:** Click inside the viewport area first to give it focus.
* **Can't scroll/pan:** Right-click on the canvas area to focus it, or left-click drag.
* **Slow performance:** Virtual scrolling handles 5000+ images. If slow, close other tabs, reduce browser zoom to 100%, or clear localStorage.
* **Images not loading:** Scroll slower to give the lazy loader time to fetch.

### LoRA & Trigger Word Issues
* **Trigger words not appearing:** Enable `lookup_and_append_lora_triggerwords` in the sampler node. Ensure internet connection for first-time lookup. Check `loras_tags.json` for cached results.
* **"INCOMPATIBLE LORA DETECTED":** Normal — the node automatically skips incompatible LoRAs. Ensure your LoRAs match your model architecture.
* **Random LoRA selection not working:** Ensure folder path includes `/` before the brackets (e.g., `"XL/Styles/[3,0.85]"`). Check console for debug messages.
* **Trigger words not filtered with `lora_omit_triggers`:** Ensure `lookup_and_append_lora_triggerwords` is enabled. Filtering is case-insensitive and handles trailing commas.

### Model & CLIP Issues
* **Images look wrong with multiple models:** System handles this automatically. You should see `⚠️ Multiple models detected - pre-encoding DISABLED` in console.
* **CLIP skip not affecting results:** Verify the model supports it. Use 0 for realistic, -2 for anime. SDXL may be less sensitive than SD1.5.
* **"No files found in folder":** Ensure trailing `/` in path. Folder names are case-sensitive on Linux/Mac.

### Config Builder UI Issues
* **Dropdowns not showing models/LoRAs:** Click the **🔄 Refresh Models/LoRAs** button.
* **UI not appearing or showing blank:** Check browser console (F12) for JavaScript errors. Ensure the node is "Ultimate Config Builder." Refresh the page.
* **Changes not saving:** Ensure auto-save is enabled OR manually click "Save Config Now." Check that the config name field is filled. Verify write permissions to `ComfyUI/output/benchmarks/configs/`.
* **"No triggers found" for a known LoRA:** LoRA may not be on CivitAI or hash mismatch. Manually add triggers or enable Bypass Trigger Fetch.
* **Trigger words not updating after exclusions:** Open the modal again and verify checkboxes. Click "Save Selections" (not just closing the modal). Check JSON preview for `lora_omit_triggers`.
* **API timeout or slow lookups:** First lookup requires hashing + API call (10-30 seconds). Subsequent lookups use cache (instant). Bypass trigger fetch for offline work.
* **Iteration count unexpectedly high:** Check for folder expansion in Models/LoRAs sections and accidental commas in CSV fields.
* **Config not loading from saved file:** Verify file exists in `ComfyUI/output/benchmarks/configs/`. Check JSON syntax. Try manually editing and re-saving.
* **LoRA stacking not working:** Check JSON preview — LoRAs should join with ` + `. Verify each has format `"name:model_str:clip_str"`. Don't mix "None" with actual LoRAs.
* **Sampler Grid not receiving configs:** Verify the `configs_json` wire is connected. Check JSON preview. Try disconnecting and reconnecting. Queue a prompt to force update.

### Remote VAE Issues
* **Not working:** Check endpoint is set correctly. Verify server is running. Look for `🌐 Remote VAE worker started` in console.
* **Hangs on "Waiting for remote VAE":** Server may have crashed. Check server logs. Restart and try again.

### Interrupt/Cancel Issues
* **Cancel doesn't stop generation:** Ensure latest version with interrupt handling. Should see `🛑 INTERRUPTED - Stopping all jobs` in console.
* **Work lost after canceling:** Manifest is saved on interrupt. Check `ComfyUI/output/benchmarks/{session_name}/manifest.json`.

### Browser Compatibility
* **Chrome/Edge:** Full support ✅
* **Firefox:** Full support ✅  
* **Safari:** Mostly works, some keyboard shortcuts may conflict
* **Mobile:** Full touch support ✅ (iOS Safari, Android Chrome tested)

---

## JSON Output Format Reference

The Config Builder generates JSON in the exact format required by the Ultimate Sampler Grid:

```json
[
  {
    "sampler": ["euler", "dpmpp_2m"],
    "scheduler": "normal",
    "steps": 25,
    "cfg": [6.0, 7.0, 8.0],
    "lora": "style.safetensors:0.8:0.8 + detail.safetensors:0.5:0.5",
    "model": "base_xl.safetensors",
    "vae": ["vae_v1.safetensors", "vae_v2.safetensors"],
    "attention_mode": ["default", "sageattn"],
    "prompt_prefix": "masterpiece, best quality, ",
    "prompt_suffix": ", 4k, detailed",
    "lora_omit_triggers": ["masterpiece", "best quality"],
    "lora_triggerwords_append_settings": {
      "style.safetensors": "positive_start",
      "detail.safetensors": "positive_end"
    },
    "lora_bypass_states": {
      "technique_lora.safetensors": true
    },
    "model_bypass_states": {
      "old_model.safetensors": true
    },
    "lora_strength_lock": {
      "style.safetensors": true
    }
  }
]
```

**Field Types:**
- **Single value:** `"euler"` or `7.0`
- **Array:** `["euler", "dpmpp_2m"]` or `[6.0, 7.0, 8.0]`
- **LoRA string:** `"name:model_str:clip_str"` — stack with ` + `
- **Folder:** `"path/"` (separate) or `"path/*"` (combined)
- **VAE:** Single VAE name or array of VAE names (each becomes a grid dimension)
- **Attention mode:** Single mode string or array of modes to test
- **Prompt prefix/suffix:** Strings prepended/appended to all prompts in this config
- **Bypass states:** Object mapping model/LoRA names to `true` (bypassed) or `false` (active)

---

## API Integration (Config Builder)

For advanced users, the Config Builder exposes several backend endpoints:

### `/configbuilder/lookup_triggers` (POST)
Bulk lookup trigger words for multiple LoRAs.

**Request:**
```json
{
  "loras": ["lora1.safetensors", "lora2.safetensors"]
}
```

**Response:**
```json
{
  "triggers": {
    "lora1.safetensors": ["trigger1", "trigger2"],
    "lora2.safetensors": ["trigger3"]
  }
}
```

### `/configbuilder/lookup_lora_metadata` (POST)
Fetch complete metadata for a specific LoRA from CivitAI.

**Request:**
```json
{
  "lora_name": "character_lora.safetensors"
}
```

**Response:**
```json
{
  "metadata": {
    "name": "Character LoRA v2",
    "model_name": "Character Series",
    "trained_words": ["character", "outfit"],
    "base_model": "SDXL 1.0",
    "description": "...",
    "tags": ["character", "anime"],
    "url": "https://civitai.com/models/12345",
    "hash": "abc123...",
    "creator": "ArtistName"
  }
}
```

### `/configbuilder/refresh_models` (POST)
Signal to clear frontend caches (called by ComfyUI when refreshing node definitions).

### `/configbuilder/lookup_model_metadata` (POST)
Fetch metadata for a checkpoint model from CivitAI (same as LoRA metadata but for models).

**Request:**
```json
{
  "model_name": "realistic_vision_v5.safetensors"
}
```

**Response:**
```json
{
  "metadata": {
    "name": "Realistic Vision V5.1",
    "model_name": "Realistic Vision",
    "base_model": "SD 1.5",
    "description": "...",
    "tags": ["realistic", "photorealistic"],
    "url": "https://civitai.com/models/12345",
    "hash": "abc123...",
    "creator": "SG161222"
  }
}
```

### `/configbuilder/scan_directory` (POST)
Scan an external directory for image sessions and load them into the dashboard.

**Request:**
```json
{
  "path": "D:/other_comfyui/output/benchmarks/my_session"
}
```

**Response:**
```json
{
  "success": true,
  "images_found": 42,
  "manifest": { ... }
}
```

### Distribution API (registered via `distribution_routes.py`)

These endpoints are used internally by the distribution system:

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/uscg-distribution/claim_job` | Worker claims next pending job (200 + job JSON, 204 = no jobs, 503 = inactive) |
| POST | `/uscg-distribution/submit_result` | Worker uploads completed image (multipart: metadata + image) |
| GET | `/uscg-distribution/status` | Get distribution status and worker stats |
| POST | `/uscg-distribution/register_worker` | Worker registration with master |
| GET | `/uscg-distribution/download_model` | Download model file for model sync |
| POST | `/uscg-distribution/heartbeat` | Worker keepalive signal |

---

## File Locations

### Generated Files

**Config Presets:**
```
ComfyUI/output/benchmarks/configs/
├── my_quality_test.json
├── model_comparison.json
└── lora_strength_sweep.json
```

**Session Output:**
```
ComfyUI/output/benchmarks/{session_name}/
├── manifest.json
├── image_001.png
├── image_002.png
└── ...
```

**LoRA Trigger Cache:**
```
ComfyUI/output/benchmarks/loras_tags.json
```

**LoRA Metadata:**
```
ComfyUI/output/benchmarks/model-data/
├── lora_name/
│   └── metadata.json
```

---

## Keyboard Shortcuts

### Dashboard
| Shortcut | Action |
|----------|--------|
| `Space` | Pan down one row |
| `Shift+Space` | Pan up one row |
| `↑↓←→` | Pan in any direction |
| `+/-` | Zoom in/out |
| `0` | Reset zoom to 1:1 |
| `F` | Auto-fit first row to viewport width |
| `Escape` | Close active modal/popup (priority: Revision > Filters > Settings > Fullscreen) |

### Config Builder
| Shortcut | Action |
|----------|--------|
| `Ctrl+F / Cmd+F` | Search within searchable dropdowns (when focused) |
| `Enter` | Confirm selection in dropdown |
| `Escape` | Close dropdown without selecting |
| `Tab` | Navigate between input fields |

---

## 🌐 Distributed Processing

Distribute large grid runs across multiple ComfyUI instances for parallel image generation.

### Setting Up Distribution

1. **Ensure Workers Are Running:** Each worker must be a separate ComfyUI instance with this node suite installed and running.
2. **Open Config Builder:** In the Config Builder node, scroll to the **Distribution Settings** section.
3. **Add Worker URLs:** Enter the full URL of each worker (e.g., `http://192.168.1.100:8188`). Click **+ Add Worker** for additional workers.
4. **Test Connections:** Click the test button next to each worker URL. A green dot means the worker is reachable.
5. **Enable Distribution:** Toggle the distribution switch ON.
6. **Run Generation:** Queue the workflow. The master node will coordinate job distribution automatically.

**How It Works:**
```
Master (your machine):
  1. Expands configs into individual jobs
  2. [Optional] Pre-encodes all prompts (Master Text Encoding)
  3. Notifies workers to start polling
  4. Processes its own jobs while workers claim theirs
  5. Workers upload completed images back to master
  6. Master saves everything to the session manifest
```

### Master Text Encoding

When enabled, the master pre-encodes ALL unique positive/negative prompts for every (model, LoRA) combination before distributing jobs. Workers receive the encoded conditionings in their job claim response and skip CLIP encoding entirely.

**Benefits:**
- Workers don't need to load text encoders at all
- Eliminates duplicate CLIP encoding across workers
- Significant time savings for prompt-heavy grids

**Enable:** Toggle "Use Master Text Encoding" in Distribution Settings.

**Technical Notes:**
- Encoded conditionings are serialized as base64-encoded NumPy arrays (~640KB per job)
- Multi-entry conditionings (AND combinator) are fully supported
- Workers fall back to local encoding if pre-encoded data is unavailable (backward compatible)

### Model Sync

When enabled, the master automatically transfers checkpoint and LoRA files to workers that don't have them.

**Enable:** Toggle "Sync Models to Workers" in Distribution Settings.

### Distribution Troubleshooting

* **Workers not connecting:** Verify the worker URL is correct (include port). Check firewalls. Both master and worker must have this node suite installed.
* **Jobs timing out:** Increase the "Claim Timeout" slider (default 600s). First-time model loads on workers can take several minutes.
* **Duplicate images:** If a worker times out but actually completes the job, duplicates may appear. Increase claim timeout to prevent this.
* **Workers not using pre-encoded prompts:** Ensure "Use Master Text Encoding" is enabled. Check worker console for `"Using master pre-encoded conditionings"` log message.
* **Model sync failing:** Ensure both master and worker have sufficient disk space. Check worker console for download errors.

---

## 🤖 Notes for AI Development

This section provides critical context for AI assistants (Claude, GPT, etc.) working on this codebase.

### Critical Constraints

**DO NOT REMOVE ANY CODE. DO NOT REMOVE ANY COMMENTS. ONLY CHANGE WHAT IS NECESSARY.** This is the #1 rule for all code changes. The codebase has many interconnected parts and removing code often breaks things in unexpected ways.

### Architecture Mental Model

This is a **hybrid application** with three layers:
1. **Python Backend** — Runs inside ComfyUI's server process. Handles generation, caching, file I/O, and API endpoints.
2. **JS Frontend Bridge** (`web/` directory) — Runs in the ComfyUI browser tab. Registers custom nodes, manages widget state, and forwards server events.
3. **Dashboard SPA** (`resources/` directory) — Runs **inside an iframe** in the dashboard node. A standalone app with its own state management, virtual scrolling, and event handling.

### Key Gotchas for AI Development

1. **Config Builder output is ALWAYS wrapped** — `config_builder_node.py` outputs `{"configs": [...], "_distribution": {...}}`. The `sampler_node.py` MUST unwrap the `"configs"` key before passing to `expand_configs()`. If you add new top-level keys to the config output, update the extraction logic in `sampler_node.py` lines 269-286.

2. **Two independent config-to-JSON implementations** — `convertStateToConfigs()` in `conf-builder-utilities.js` (JS preview) and `generate_config()` in `config_builder_node.py` (Python output) both transform `node.state` into config JSON. **Both must stay in sync** when adding new config fields.

3. **Dashboard runs in an iframe** — The dashboard HTML is set via `iframe.srcdoc`. All JS/CSS is inlined (no external requests). Communication between the parent ComfyUI page and the iframe uses `postMessage()`. The `dashboard.js` file bridges server events to iframe messages.

4. **Dashboard auto-loads on node creation** — `web/dashboard.js` calls `forceLoadSession()` with a 500ms delay when a dashboard node is created. The `get_session_html` API endpoint returns a full HTML template with empty manifest for non-existent sessions (NOT a 404).

5. **`logic_pipeline.js` has a legacy init block** — Lines 489-505 have a `DOMContentLoaded` handler that calls `loadSession()` only when `fullManifest.items.length > 0`. The proper initialization happens in `logic_init.js` which calls `init()`. Don't add a second unconditional `loadSession()` call — it causes alert popups for empty sessions.

6. **Manifest merge on save** — `save_manifest()` always reloads from disk first to preserve user favorites/rejected/notes. This is intentional for concurrent access (generation + user interaction). Don't bypass this.

7. **`onNodeCreated` must be synchronous** — In `conf-builder-main.js`, async work runs inside fire-and-forget `(async () => { ... })()`. Making `onNodeCreated` async breaks ComfyUI widget registration.

8. **Optional inputs disable caches** — When `optional_model`/`optional_clip`/`optional_positive`/`optional_negative` are connected to the sampler node, conditioning cache is disabled and `IS_CHANGED()` returns `NaN` (forces re-execution every time).

9. **Distribution conditionings are serialized with np.copy()** — `np.frombuffer()` returns read-only arrays. Must call `.copy()` before `torch.from_numpy()` or GPU tensor operations will fail.

10. **Default state migration** — When adding new fields to the Config Builder state (in `conf-builder-main.js`), also add a migration check in the `onNodeCreated` function (around line 363) to backfill the default value for existing saved workflows.

11. **claim_timeout default is 600** — This value appears in `distribution_manager.py`, `config_builder_node.py`, `conf-builder-distribution.js`, and `conf-builder-main.js`. Keep all four in sync.

12. **ComfyUI Registry security** — No `import requests`, no `subprocess`/`eval`/`exec`, no custom file-serving endpoints. Use `urllib.request` for HTTP and ComfyUI's `/view` endpoint for serving files. See `ProjectStructure.md` for full rules.

### File Dependencies (Import Graph)

```
sampler_node.py
  → generation_orchestrator.py (main entry point)
    → config_utils.py (expand_configs, prepare_input_jobs)
    → model_loader.py (load_checkpoint, load_loras)
    → image_generation.py (generate_image, flush_batch)
    → batch_encoding.py (encode_prompt_with_combinators)
    → trigger_words.py (build_prompt_with_triggers)
    → manifest_utils.py (save_manifest, load_existing_manifest)
    → model_cache.py (ModelCache)
    → conditioning_cache.py (ConditioningCache)
    → remote_vae.py (RemoteVAEDecodeWorker)
    → html_generator.py (get_html_template)
    → distribution_manager.py (DistributionManager)
    → distribution_worker.py (WorkerThread)

config_builder_node.py
  → lora_utils.py (trigger word lookup)

__init__.py
  → all node classes
  → distribution_routes.py (distribution API endpoints)
  → html_generator.py (on-demand HTML generation)
  → metadata_packer.py (export favorites)
  → directory_scanner.py (scan external directories)
  → manifest_utils.py (save/load manifests)
```

### Server Event Types

| Event Name | Sender | Data | Purpose |
|---|---|---|---|
| `ultimate_grid.update` | `generation_orchestrator.py` | `{session_name, node, manifest, meta, new_items}` | New images generated |
| `ultimate_grid.progress` | `generation_orchestrator.py` | `{session_name, progress_pct, eta_str, ...}` | ETA/progress updates |
| `ultimate_grid.distribution_status` | `generation_orchestrator.py` | `{session_name, workers, stats}` | Distribution status |

### Testing Changes

After modifying backend Python files, restart ComfyUI. After modifying `web/` JS files, refresh the browser (Ctrl+F5). After modifying `resources/` JS/CSS/HTML files, the dashboard HTML must be regenerated — either re-run the sampler node or call the `/config_tester/get_session_html` endpoint.

---

## 📝 Changelog

### Update 3/9/26 — Upscaling, Workflow Packing & Config Builder Enhancements
* 🔍 **Array-Based Upscaling System:** Define multiple upscale configs with Cartesian expansion across models, ratios, and denoise values. Five modes: hires_only, model_only, model_then_hires, seedvr2, and florence2_hires (targeted region inpaint via Florence2 segmentation — needs ComfyUI-Florence2).
* 🖼️ **Upscale Model Discovery:** Auto-detects installed upscale models (RealESRGAN, etc.) from ComfyUI's folder system with searchable dropdown.
* 📦 **Pack Full Workflow into Exports:** Embed the entire ComfyUI graph into exported favorite images as PNG metadata — drag back into ComfyUI to restore.
* 🔧 **Pack Config as Nodes Workflow:** Generate per-image pure-nodes workflows matching each image's exact generation parameters.
* 📋 **Cleaned Favorites Manifest:** Export only favorited items in the manifest copy (previously exported everything).
* 🗑️ **Delete All Non-Favorited Items:** One-click cleanup button with confirmation dialog to remove all non-starred images from a session.
* 📄 **Export Prompt .txt Files:** Save positive prompt as plain text alongside each exported image.
* ❄️ **GPU Cooldown:** Configurable automatic pauses between generations to prevent thermal throttling, with optional VRAM clearing.
* 🌱 **Seed Behavior Settings:** Per-config seed control — fixed, randomize, or full run seed override.
* 🎯 **Per-Config Resolutions:** Override global resolution per config array. Each resolution becomes a Cartesian dimension.
* 🎨 **Redesigned Resolution Picker:** Nested dropdown with categorized presets (SD 1.5, SDXL, Flux, etc.) and custom resolution editor.
* 🏷️ **Configurable Label Overlays:** Toggle parameter labels (model, sampler, steps, CFG) directly on dashboard cards.
* ↕️ **Drag-and-Drop Reorder:** Rearrange models, LoRAs, text encoders, and VAEs by dragging.
* 🔧 **Dynamic VRAM Compatibility:** Fixes for `force_patch_weights` and `torch.inference_mode` to work with ComfyUI's Dynamic VRAM mode.
* 🐛 **Distribution Worker Fixes:** Workers now receive all 8 advanced sampling parameters (model_sampling_override, advanced_guider, flux_guidance, etc.) — previously produced different outputs from master.
* 🐛 **Custom Prompts Fix:** Fixed session load, checkbox toggle, and count display in per-config custom prompts section.
* 🐛 **Optional Input Skip Fix:** Disabled job skip/resume when optional inputs are connected (external changes can't be reliably detected).

### Update 3/6/26 — Distributed Processing & Dashboard Improvements
* 🌐 **Multi-Worker Distribution:** Split grid runs across multiple ComfyUI instances with automatic job coordination, claiming, timeout handling, and result upload.
* 🧠 **Master Text Encoding:** Master pre-encodes all prompts and sends conditionings to workers, eliminating duplicate CLIP encoding.
* 📦 **Model Sync:** Optionally transfer checkpoint and LoRA files from master to workers automatically.
* 📊 **Per-Worker Logging:** Final summary shows each worker's completed/failed job counts.
* ⚙️ **Distribution Settings UI:** Config Builder section with worker URL management, connection testing, and toggles for text encoding and model sync.
* 🏠 **Session Landing Page:** Dashboard now auto-loads and shows a session picker on startup instead of a blank black box.
* 🔄 **Dashboard Auto-Load:** Dashboard iframe automatically loads session HTML on node creation with 500ms delay.
* 🛠️ **Non-Existent Session Handling:** `get_session_html` endpoint returns a full template with empty manifest instead of 404.
* ⏱️ **Claim Timeout Update:** Default claim timeout increased from 300s to 600s across all files.
* 🤖 **AI Development Notes:** Added comprehensive development notes to README and ProjectStructure for AI-assisted development.

### Update 2/16/26 — Config Builder & Dashboard Overhaul
* 🎛️ **VAE Selection & Iteration:** Test multiple VAEs per config with searchable dropdown and folder expansion. VAEs are a full Cartesian dimension.
* 🧠 **Attention Mode Testing:** Grid-test different attention implementations (`default`, `sageattn`, `xformers`) as a combinatorial dimension.
* 🔗 **CLIP Encoding Combinators:** `AND`, `CAT`, `AVG(weight)`, and `BREAK` keywords for multi-segment conditioning and prompt blending.
* 📝 **Model-Specific Prompts:** Per-config prompt prefix and suffix fields for model-appropriate quality tags.
* 🔄 **Recursive Cartesian Prompt Builder:** Visual chip-based editor with `{option1|option2}` syntax expanding into all combinations.
* 🔘 **Model & LoRA On/Off Switches:** Toggle entries on/off without removing them. Bypassed entries excluded from preview and output.
* 🔍 **CivitAI Model Metadata Lookup:** Fetch model info from CivitAI directly in the Config Builder (same as existing LoRA lookup).
* 📋 **Sampler/Scheduler Dropdowns:** Searchable dropdowns auto-populated from the connected sampler node, replacing free-text CSV entry.
* ⚙️ **Unified Settings Panel:** SESSION button merged into a gear icon settings panel with Session, Grid Display, Export Favorites, and Scan Directory sections.
* 📊 **Real-Time ETA Progress Bar:** Shows estimated time remaining during generation in the dashboard header.
* 📂 **Scan External Directory:** Load image sessions from any folder into the dashboard for viewing and filtering.
* 📦 **Export Favorites:** Copy starred images to a dedicated folder with optional CivitAI metadata packing.
* 🛑 **Cancel During Text Encoding:** Properly cancels the entire run when interrupted during the text encoding step.
* 🔄 **Connected Node Change Detection:** Detects when upstream nodes have changed for smarter job resuming.
* ⌨️ **Close Modal Hotkeys:** Escape key closes modals in priority order (Revision > Filters > Settings > Fullscreen).
* 🧹 **Dashboard Topbar Consolidation:** Streamlined header with condensed controls.

### Update 2/5/26 — Code Refactoring & Performance Improvements
* 🏗️ **Major Code Refactoring:** Reorganized codebase into 6 modular files for better maintainability (`trigger_words.py`, `batch_encoding.py`, `manifest_utils.py`, `model_loader.py`, `image_generation.py`, `generation_orchestrator.py`).
* 🎯 **CLIP Skip Support:** Control CLIP layer usage with `clip_skip` parameter. Essential for anime models (-2) vs realistic models (0). Supports arrays for testing multiple values.
* 🧠 **Intelligent CLIP Encoding:** Multi-model workflows now handle CLIP correctly with automatic detection and per-model encoding.
* ⚡ **Batch Encoding on Model Switch:** 3-6x faster encoding for multi-model workflows.
* 🗑️ **LoRA Trigger Word Filtering:** New `lora_omit_triggers` parameter to exclude specific trigger words.
* 📁 **Model Folder Expansion:** Use `"model": "FolderName/"` to test all checkpoints in a folder.
* 🛑 **Graceful Interruption:** Cancel now stops ALL jobs, flushes pending batches, waits for remote VAE, and saves manifest.
* 🌐 **Remote VAE Fixes:** Fixed initialization, job queuing, and interrupt handling.
* 🔧 **str_model/str_clip Removal:** Deprecated config fields removed. Strengths now specified in LoRA string.

### Update 2/5/26 — Random LoRA Selection
* 🎲 **Random LoRA Selection:** `[count,strength]` syntax for seed-based selection, `[count,strength,random]` for truly random, dual strength support, cross-platform paths, and full trigger word integration.

### Update 1/14/26 — Major Feature Update
* 🎯 **Non-Standard Model Support:** SD3, Flux, Z-Image with automatic latent channel detection.
* ⭐ **Favorites System:** Star images with dedicated gold JSON bar.
* 🎨 **Horizontal JSON Bars:** Redesigned three-bar layout.
* 🔍 **LoRA Auto Trigger Words:** CivitAI API integration with local caching.
* 🚫 **LoRA Compatibility Detection:** Automatic skip with clear error messages.
* ⌨️ **Shift+Click Filter Isolation.**
* 🎯 **Go to Image #.**
* 🚀 **Dashboard Auto-Load.**
* 📱 **Mobile Touch Support.**
* 🔐 **Conditioning Change Detection.**
* 📋 **Enhanced Revise Modal.**

### Update 1/11/26 — Major Overhaul
* ✨ **Virtual Scrolling:** Handles 5000+ images.
* 🖼️ **Fullscreen Mode.**
* 🔄 **Multi-Model Support** with folder expansion.
* 🎨 **Multi-LoRA Stacking** with `+` separator.
* 🎲 **Multi-Seed Generation.**
* ⏸️ **Stop & Resume.**
* ⌨️ **Keyboard Navigation.**
* 📊 **Live Updates** with `flush_batch_every`.
* 💾 **Persistent Settings.**
* 🎯 **Auto-Fit Zoom.**
* ⚡ **Massive performance refactoring.**

---

## 📜 License

MIT License. Feel free to use, modify, and distribute.

---

## 🙏 Credits

Created for the ComfyUI community. Special thanks to all contributors and testers who helped refine this tool.

**Special Features Powered By:**
- CivitAI API for trigger word lookups
- ComfyUI's object_info system for model/LoRA discovery

**Star this repo if you find it useful!** ⭐

**Support development:** [Buy me a coffee on Ko-fi](https://ko-fi.com/jasonhoku) ☕

**Version:** 3.1
**Last Updated:** March 9, 2026
