

99% of this project's near 10k lines of code was written by either Gemini or Claude. Sorry if that upsets you, I've been a programmer for 15 years and it's simply a much faster means of developing.

With that said, here is a guide on:

Easyily Get AI To Add Features To This Project:

Gemini Pro & Claude Sonnet 4.5 both work great but they sometimes make mistakes. The key is to really get a very clear prompt built out.

Step 1.
Send the AI the ProjectStructure.md and README.md files and your task and ask it to tell you which files need to be edited to get your task completed.

Step 2. 
Start a new chat and attach the files it mentioned, along with the following prompt or something similar with your task request prompt in the middle:

Prompt Fill-in:

Help me update my ComfyUI Custom Node. When updating files, DO NOT REMOVE ANY CODE. DO NOT REMOVE ANY COMMENTS. ONLY CHANGE WHAT IS NECESSARY.

PUT YOUR TASK HERE

Check ProjectStructure.md to get an idea of file structure and contents, check README.md  to get an idea of the project as a whole.

step 3
?????

step 4
Profit!!!

But seriously, after updating your code see if it works. If it didn't send the error message if there isnt one or your symptoms. If its still not working try a different AI or try breaking down your task in smaller pieces.


---

!!!!
Check Roadmap.md for some tasks and do them. If you think you can do the ones marked as New to-do items, needs more info/explaining/numbering those are good.

There is info on the project in README.md and info on the file structure and notes on each files functions in the ProjectStructure.md 
!!!!
ProjectStructure.md has notes to assist AI in developing this project
!!!!


# Run this after lots of updates & before starting a new convo  

## Please update ProjectStructure.md to be better structured and fixed up for having all relevant AI Notes in it that will help and assist AI to develop this project further. Add your findings, file system notes, important considerations and more to it and make sure its up to date and full of helpful info.



#  **ComfyUI Ultimate Sampler Grid – Development Roadmap** 


## ~~Feature label mode labels in dashboard when showing lora labels should also show the model str and clip strength like how they're written in the config_json, as lora:1.0:1.0 etc~~ DONE

## ~~Feature: Checkbox for "Run Upscales At End Of Session Instead Of After Each Gen"~~ DONE — deferred upscale queue with step-major model batching, resume support, and dashboard progress

## ~~Bug: The duration/time taken to generate metric doesn't work properly if upscale is used~~ DONE — duration now includes all upscale time

## ~~Feature: Slider in the Dashboard, in the settings cog modal, at the top of Label Model section between the description and the enable button, that increases/decreases the size of the labels.~~ DONE

## ~~Add Steps, CFG & Upscale Model/Method to Dashboard Filters list~~ DONE

## ~~Bug: If we omit entries from the Builder UI's config_json output the manifest.json files lose that data and entries that are toggled off will be lost when loading the session from the manifest file.~~ DONE — all fields now always included with defaults

## Feature idea, Config Section Presets, Saveable, Nameable & Loadable or Importable/Mergeable presets for each section, Prompts, Models, Text Encoders, Vaes, Loras, Trigger Omit, Upscalers. Would make mixing and matching and setting up new configs easier & more customizable. (Upscale presets DONE — other sections still TODO)


## ~~The Dashboard Session Load Option in the cog settings topbar modal needs sort by most recent and session search feature like how the Builder UI Session Load has.~~ DONE

## Feature: Add builder UI option to pack pure nodes like REVISE copy as comfy nodes into every image always.

## Issue: Distributed Worker bug with save models to worker, not working

## ~~Small bug, centering zoom doesn't perfectly fit the images into the canvas, I think it doesnt account for the topbar and bottom smartjson bar sizes.~~ DONE

## ~~Vae settings in the Builder UI passed to sampler gen orchestrator through the config_json need to override the hf_remote setting in the sampler node input selection.~~ DONE — verified by tests/test_roadmap_verify.py: a non-Default Builder VAE (local file or `remote:URL`) always decodes with that VAE and beats the sampler node's remote_vae_endpoint; Default falls back to the node endpoint. Fixed: a config VAE of "None"/"" on a GGUF/diffusion model skipped the "no VAE source" pre-check (now `normalize_config_vae` is used everywhere).

## ~~Issue: If running a job in one tab and reviewing images from a past session with the Dashboard in another tab, the Dashboard receives updates across all tabs and force-loads the Dashboard to the currently running session.~~ DONE — auto-load only when no session loaded yet

## ~~The batch encoding step lost its ability to stop everything during its run loop.~~ DONE — verified (tests/test_roadmap_verify.py): `batch_encode_with_cache` checks the interrupt before every prompt (also across batch boundaries) and raises InterruptProcessingException with no further prompts encoded.

## Feature: Drag and drop re-organzie loras, models, text encoders, vaes. This would make changing their run order and lora_tag append order much easier to control. (Partly verified: the reorder maths is now a pure `moveItem`/`reorderTarget` with node tests, the new order is proven to reach configs_json (model list order, LoRA stack/tag order); fixed a doubled handle glyph on text-encoder rows and drops from non-handle drags. STILL needs a real-browser pass: dragover/dragleave flicker over child elements, no auto-scroll while dragging in a long list, no touch support (HTML5 DnD), and the full-UI rebuild after each drop.)

## ~~Jobs don't generate prompts in the order they're written in the array, it's mildly inconvenient. Lets fix that and make it run in order from top to bottom of the array.~~ DONE

## ~~Custom Job Resume / Skipping - Start At Job # Option~~ DONE — Run Settings section in Builder UI

## Batch Encoding could use with implementing the smart, look ahead, caching system we built for lora swapping.

## bug fix, the Builder UI is very slow to update when first starting, adding models, adding loras, a few other things, we need to check it and optimize it to load and respond faster. (Needs more improvement. Static review done; section-preset GETs are now cached. Remaining hotspots: (1) first start blocks on the multi-MB `/object_info` fetch for LoRAs/checkpoints (conf-builder-utilities.js `_getObjectInfo`), (2) every add/remove/reorder rebuilds the WHOLE UI via `renderUI` (`innerHTML = getStyles() + ...`, conf-builder-config-management.js ~7494) although a partial `rerenderConfigSection` exists, (3) `saveState` pretty-prints the whole state on every keystroke (conf-builder-main.js ~267).)

## NEW: Dashboard Upscale Feature — Upscale single images or all favorites directly from the dashboard with preset management. Phase 1 DONE. Phase 2 (inline pipeline editing, cancel) TODO.




#### ~~**12. Arrays in LoRA Weights**~~ DONE — the ＋ button in the Builder's LoRA card adds strength rows (one slider, or Model + CLIP sliders with the lock off; max 8 per LoRA); each row is one full copy of the config's jobs, stacked LoRAs get every combination. State key `lora_extra_strengths`; the old Compare Strengths box is converted on first edit.

Builder UI could have a + button right next to model strength text input (on its right side) to make it easy to add a compare different lora strengths. If activated it should add an additonal slider, or two sliders if 🔒 Lock Model & CLIP Strength Together is deactivated, for each time they click +. If a lora has a multiple strength array, the preview json and py node json_output should duplicate the full config for that run for each entry with the only difference being each lora strength. If multiple loras have multiple arrays it needs to output a cartesian of an each for each of all lora strenth combos. 



## ~~Bug Fix: Batch Encoding Runs Before Job Skip/Continue/Resume check.~~ DONE — verified (tests/test_roadmap_verify.py): `configs_needing_work()` (extracted from the loop) keeps only configs with at least one still-todo plan entry, so finished jobs' prompts are not encoded; all-done encodes nothing. STILL TODO (optional input tracking): with optional model/clip/vae/conditioning/latent inputs connected, `check_if_job_completed` returns -1 and skip is disabled entirely, because the fingerprint knows nothing about upstream nodes. It would need a content hash of each connected input (model/clip/vae weights id, conditioning tensors, latent tensor) stored in the manifest items and compared in the skip check.


#### **7. CivitAI Download Integration** (low priority)
A button in the builder UI to pack short sha256 into config with an explanation that it can be used to share or move an Ultimate Sampler Config Tester workflow and allow for downloading all models and loras in the workflow from civitAI with a few simple easy clicks. lora_utils has calculate civit model has function in it. dropdown configurable options for where to store each file type.

#### ~~**9. Tag/Token-Based Omit Logic**~~ DONE — verified (tests/test_roadmap_verify.py): word-boundary token matching ("blue" omits "blue eyes" not "blueberry"), case/whitespace/trailing-comma normalised. Fixed two bugs: terms starting/ending in punctuation such as "(masterpiece)" never matched, and a blank omit term omitted EVERY trigger.

#### ~~**10. Validation Warning (Omit vs Lookup) - Warn user if omits are added but lookup is off**~~ DONE — the Builder warning now fires exactly when omits exist and Run Settings "LoRA Triggerwords Mode" is None (it used to key on per-LoRA append settings, a false positive in every other mode) and updates live when the mode changes; the run also prints a console warning (`omit_without_lookup_warning`). Tests: tests/test_roadmap_verify.py, tests/js/roadmap_verify.test.js.


#### **11. Model-Specific Prompts** (Needs improvement)

#### **13. Real-Time ETA** (Improved — now uses rolling 10-job window for more responsive estimates. Distributed runs: DONE, combined-throughput ETA over all active machines, pushed to the dashboard as `phase: "distributed"` progress events)

#### **14. Cache Trigger Word Placement** (low priority)
* **Problem:** `trigger_words.py` logic runs every loop iteration.
* **Target Files:** `trigger_words.py`


#### ~~**20. Hotkeys Reference List**~~ DONE — updated with R, Shift+0-9, Ctrl+S, Escape, Double-click


#### **22. Import Configs (Merge)** (low priority)

* **Problem:** Can only load full sessions, not merge snippets.


#### **23. Pseudo-JSON Nodes (Recursion)** (low priority) - STARTED: v1 is the **Workflow Sweep** (Builder: Add Workflow Sweep): any API-format workflow runs as-is with chosen inputs swept (see README, 'Workflow Sweep'). Still open: inserting nodes into the USCG pipeline itself (video outputs and mixed workflow + standard runs shipped 11/4/26). (v1 shipped 11/4/26)
* **Problem:** Advanced. Running a raw JSON workflow as a sub-node. Inserting any node into any part of the genereation would make this tool have an immense customizability increase.
More info needed on how this could work, would like to see a visual interface for it in the builder UI eventaully. (big job, very low priority)


#### **24. Combinatorial Randomization** - More Randomization tools - generate x configs from y possibilities and z prompts - (very low priority)

* **Problem:**  Feature. Combinatorial generation logic. Combine random prompts with random loras, fun!


---

## Strategic Roadmap (from R&D competitive analysis, 2026-04)

#### **25. Image Quality Metrics** (unique differentiator — no competitor has this)
Compute objective quality metrics during generation and store in manifest:
- **Laplacian variance** (sharpness/blur detection)
- **Histogram entropy** (color diversity/richness)
- **Sobel edge quality** (detail preservation)
Add as sortable/filterable fields in Dashboard. Enables data-driven parameter selection instead of purely visual comparison. Prior art: SamplerSchedulerMetricsTester (proves the concept, only 9 stars but validates demand).

DONE (11/4/26): Sobel edge quality (sobel), backfill fills missing keys. Built-in metrics (sharpness, entropy, colorfulness, contrast, luma, saturation, sobel), Dashboard sort/filter/labels, backfill job for old sessions. Still open: scorer packs.

#### **26. Analytics Panel in Dashboard** (transforms tool from "look at pictures" to "data-driven decisions")
* **Milestone 1:** Aggregate stats bar — avg generation time by sampler/scheduler, favorites rate by model/LoRA. DONE (11/4/26) Milestone 1: Stats panel (computeAnalytics in resources/logic_analytics.js is the data source for M2).
* **Milestone 2:** Canvas-based charts — bar charts, scatter plots (steps vs duration, CFG vs quality metrics)
* **Milestone 3:** Export — downloadable reports with charts + top images
Pure client-side using canvas API from existing `activeData`. Prior art: W&B dashboards.

STARTED (11/4/26): the Pairwise Review scoreboard (per-axis win % + ratings) covers part of Milestone 1.

#### **27. Static Dashboard** (eliminate HTML regeneration requirement)
* **Milestone 1:** REST manifest fetch — Dashboard loads manifest via `fetch('/config_tester/get_manifest?session=name')` instead of inline `__JSON_DATA__`
* **Milestone 2:** Remove html_generator dependency — serve Dashboard as static HTML from `WEB_DIRECTORY`
* **Milestone 3:** Hot-reload in dev — JS changes visible on browser refresh without re-running sampler
Biggest architectural leverage: eliminates the #1 developer friction point.

STARTED (11/4/26): `/config_tester/dashboard` serves the SPA as a standalone page with its own websocket.
DONE (11/4/26) Milestone 1: the page is a lazy shell (`html_generator`, `__uscg_lazy__`); the SPA fetches `GET /config_tester/manifest?session=` (compact JSON, gzip, ETag/304; the route is named `manifest`, not `get_manifest`). The node's `executed` message no longer carries the manifest (`web/dashboard.js` sends `reload_manifest`). Milestone 3 comes free for F5 / RELOAD (the page is rebuilt from `resources/` on every request); a node already showing a page keeps its old JS until RELOAD.
DONE (11/4/26) Milestone 2 via the cheap alternative (static `web/` serving is blocked: ComfyUI's frontend imports every `web/**/*.js` as an extension module): fixed path-less routes `GET /config_tester/dashboard_bundle_js?v=<hash>` / `..._css?v=<hash>` serve the concatenated `resources/` files (`html_generator.get_dashboard_bundle`, rebuilt on `(mtime, size)` change so Milestone 3 hot reload still works) with a weak ETag, `Cache-Control: public, max-age=31536000, immutable`, precomputed gzip; the shell references them with `<script src>` / `<link>` and shrank from ~527 KB to ~70 KB, repeat loads cost only the shell's 304. The node iframe uses them too (every run's `executed` message and history entry is ~450 KB lighter). `inline_assets=True` / env `USCG_INLINE_DASHBOARD_ASSETS=1` keeps a self-contained page (`inline_manifest=True` implies it). Tests: `tests/test_dashboard_bundle.py`, `tests/e2e/test_dashboard_bundle.py`.

#### **28. Onboarding Wizard & Tutorial**
* **Milestone 1:** First-run wizard in Builder UI — walks through: create session → pick model → add LoRA → run → view dashboard (5 interactive steps)
* **Milestone 2:** Video tutorial (3-minute YouTube walkthrough)
README is 2000+ words; most users won't read it. Efficiency Nodes grew via community tutorials.


---

## Open follow-ups (11/4/26)

#### **29. Text-Encoder Batching: live check, then flip the default**
Verify `te_batch_size` on the 5060 Ti with the real MOSS-Image, Krea 2 and Qwen-Image encoders, then decide whether the default of 1 (off) should change.

#### **30. Distributed companion parity**
**Done (wave 3, `feat/dist-parity`):** `job_order`, the preview pass and image metrics now work in distributed mode (master-owned plan, `populate_from_plan`, `dist_worker` on every item). DONE (feat/dist-open): rejects made mid-run are re-checked at claim time, claim affinity, Start At Job in distributed mode, LTX rendered on the master (never on workers). Open: live two-machine verification of the whole set.

#### **73. Distributed Model Sync hardening**
DONE (feat/dist-open, both repos): the master now takes over jobs still claimed after the wait timeout instead of discarding the workers' images (`wait_timeout`, default `max(600, claim_timeout)`); the companion de-duplicates concurrent downloads of one file, resolves model folders with fallbacks, normalises backslash names, refuses `..` paths and treats a VAE of `None` as the default. Open: confirm with a real slow-download run which defect the original failure was.

#### **31. Scorer-pack adapters**
Verify the node ids of the optional aesthetic / reward scorer packs against the real packs, then enable the shipped adapter entries.

#### **32. Workflow Sweep: video outputs and mixed runs**
Allow video outputs from a sweep and sweep + standard configs in one Generator run (both are v1 limits).
DONE (11/4/26): video outputs (SaveVideo / CreateVideo VIDEO sockets, VHS_VideoCombine frames via a spliced CreateVideo, `USCGSweepCollectVideo`). DONE (11/4/26, `feat/sweep-mixed`): mixed sweep + standard runs in one Generator run (standard configs first, then the sweep, one session, global Start At Job numbering); SaveWEBM / SaveAnimatedWEBP / SaveAnimatedPNG as frame sources (same CreateVideo splice, fps clamped to 1-120). Still open: several outputs per job.
