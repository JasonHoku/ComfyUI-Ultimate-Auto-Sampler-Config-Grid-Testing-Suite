/**
 * Config Builder Main Entry Point
 * Loads modules from /config_builder/js/ route (served with no-cache headers)
 */

import { app } from "../../../scripts/app.js";

// Cache-busting timestamp (regenerated on every page load)
const CACHE_BUST = Date.now();

// Module storage
let utilities, uiComponents, configManagement, distributionModule;

// Singleton promise to ensure we only trigger the load once
let moduleLoadPromise = null;

function ensureModulesLoaded() {
    if (moduleLoadPromise) return moduleLoadPromise;

    console.log('[ConfigBuilder] Loading modules with cache-bust:', CACHE_BUST);
    moduleLoadPromise = (async () => {
        const utilitiesModule = await import(`./conf-builder-utilities.js?v=${CACHE_BUST}`);
        const uiComponentsModule = await import(`./conf-builder-ui-components.js?v=${CACHE_BUST}`);
        const configManagementModule = await import(`./conf-builder-config-management.js?v=${CACHE_BUST}`);
        const distributionMod = await import(`./conf-builder-distribution.js?v=${CACHE_BUST}`);

        utilities = utilitiesModule;
        uiComponents = uiComponentsModule;
        configManagement = configManagementModule;
        distributionModule = distributionMod;

        return { utilities, uiComponents, configManagement, distributionModule };
    })();
    return moduleLoadPromise;
}

// --- NODE REGISTRATION ---

app.registerExtension({
    name: "UltimateConfigBuilder.CompleteHTML",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "UltimateConfigBuilder") {
            
            // Trigger module load immediately when we see our node definition
            ensureModulesLoaded();

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            // FIX: Removed 'async' keyword. This function must remain synchronous!
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);

                const configWidget = this.widgets?.find(w => w.name === "lora_config");
                if (!configWidget) return result;

                // 1. Synchronous Setup (CRITICAL: Must happen immediately)
                // The "converted-widget" trick collapses the LiteGraph slot, but on the
                // modern Vue node renderer a per-widget DOM element may still mount and
                // peek out under the HTML overlay. As a belt-and-suspenders, also hide
                // the actual DOM element when it appears (poll briefly until it does).
                this.widgets.forEach(w => {
                    w.type = "converted-widget";
                    w.computeSize = () => [0, -4];
                });
                // Capture references to the python-side widgets we want to hide
                // BEFORE we add the DOM widget below (otherwise we'd hide the UI).
                const _widgetsToHide = this.widgets.slice();
                const _hideWidgetEls = () => {
                    _widgetsToHide.forEach(w => {
                        const el = w.element || w.inputEl;
                        if (el && el.style && el.style.display !== "none") {
                            el.style.display = "none";
                        }
                    });
                };
                _hideWidgetEls();
                let _hideAttempts = 0;
                const _hideRetry = () => {
                    _hideWidgetEls();
                    if (++_hideAttempts < 40) setTimeout(_hideRetry, 50);
                };
                setTimeout(_hideRetry, 50);
                
                // Assign the widget reference immediately so onConfigure can find it
                this.configWidget = configWidget;

                // Setup default state immediately
                this.uiState = {
                    modelsSectionCollapsed: {},
                    lorasSectionCollapsed: {},
                    vaesSectionCollapsed: {},
                    modelsCollapsed: {},
                    lorasCollapsed: {},
                    vaesCollapsed: {},
                    promptsSectionCollapsed: {},
                    globalPromptsSectionCollapsed: false,
                    extraOptionsSectionCollapsed: {},
                    promptRawMode: {}  // Track JSON vs Visual mode per prompt editor
                };
                
                // Initialize default state structure
                this.state = {
                    session_name: "my_test_session",
                    config_name: "default_config",
                    auto_save: false,
                    include_none: false,
                    label_mode: false,
                    global_positive_groups: [],
                    global_negative: "",
                    distribution_enabled: false,
                    worker_urls: [],
                    claim_timeout: 600,
                    use_master_encoding: false,
                    config_arrays: [{
                        name: "Config 1",
                        kind: "standard",
                        samplers: ["euler", "dpmpp_2m"],
                        schedulers: ["normal", "karras"],
                        steps: "20, 30",
                        cfg: "7.0",
                        models: ["None"],
                        vaes: ["None"],
                        text_encoders: [],
                        clip_type: "stable_diffusion",
                        gguf_options: {},
                        loras: ["None"],
                        lora_omit_triggers: [],
                        lora_triggerwords_append_settings: {},
                        lora_bypass_states: {},
                        lora_strength_lock: {},
                        model_bypass_states: {},
                        vae_bypass_states: {},
                        te_bypass_states: {},
                        combine: false,
                        positive_prompt_groups: [],
                        negative_prompt: "",
                        use_custom_prompts: false,
                        model_prompt_prefix: "",
                        model_prompt_suffix: "",
                        attention_modes: ["default"],
                        // Extra Model & Sampling Options
                        model_sampling_override: "none",
                        model_sampling_shift: "1.73",
                        model_sampling_flux_max_shift: "1.15",
                        model_sampling_flux_base_shift: "0.5",
                        use_advanced_sampling: false,
                        advanced_guider: "cfg_guider",
                        advanced_scheduler: "basic",
                        use_flux_guidance: false,
                        flux_guidance_value: "3.5",
                        // Kohya Deep Shrink (PatchModelAddDownscale)
                        use_deep_shrink: false,
                        deep_shrink_block_number: 3,
                        deep_shrink_downscale_factor: 2.0,
                        deep_shrink_start_percent: 0.0,
                        deep_shrink_end_percent: 0.35,
                        deep_shrink_downscale_after_skip: true,
                        deep_shrink_downscale_method: "bicubic",
                        deep_shrink_upscale_method: "bicubic",
                        // Skip Layer Guidance (SkipLayerGuidanceDiTSimple)
                        use_slg: false,
                        slg_double_layers: "",
                        slg_single_layers: "22",
                        slg_start_percent: 0.0,
                        slg_end_percent: 0.5
                    }],
                    upscaling: {
                        enabled: false,
                        save_pre_upscale: false,
                        run_upscales_at_end: false,
                        hires_prompt_adjust: false,
                        hires_prompt_behavior: "append_end",
                        hires_prompt_text: "",
                        pipelines: [{
                            active: true,
                            name: "Pipeline 1",
                            steps: [{
                                active: true,
                                mode: "hires_only",
                                repeat: 1,
                                upscale_models: [],
                                upscale_ratios: "1.5",
                                upscale_size: "2.0",
                                hires_denoise: "0.3",
                                hires_steps: 0,
                                tiled_vae: false,
                                tile_size: 512,
                                tile_overlap: 64,
                                temporal_size: 512,
                                temporal_overlap: 64,
                                resize_method: "bilinear",
                                hires_tiled_sampling: false,
                                hires_tile_width: 512,
                                hires_tile_height: 512,
                                hires_mask_blur: 8,
                                hires_tile_padding: 32,
                                hires_force_uniform_tiles: false
                            }]
                        }]
                    },
                    cooldown: {
                        enabled: false,
                        seconds: 5,
                        every_n: 1,
                        clear_vram: false
                    },
                    image_format: "webp",
                    metrics: { enabled: false, builtins: { sharpness: true, entropy: true, colorfulness: true, contrast: true, luma: true, saturation: true }, adapters: {} },
                    overwrite_existing: false,
                    flush_batch_every: 1,
                    lora_triggerwords_mode: "None",
                    save_conditioning_cache_to_file: false,
                    enable_model_cache: false,
                    vae_batch_size: 1,
                    add_random_seeds_to_gens: 0,
                    te_batch_size: 1,
                    job_order: "as_written",
                    preview_pass: { enabled: false, steps: 8, scale: 0.5, cfg_override: null },
                    skip_rejected_previews: true,
                    only_favorited_previews: false
                };

                // Create HTML container immediately
                this.htmlContainer = document.createElement("div");
                this.htmlContainer.style.cssText = `width: 100%; height: 100%; background: #1a1a1a; display: flex; flex-direction: column;`;
                this.addDOMWidget("config_ui", "div", this.htmlContainer, { serialize: false, hideOnZoom: false });

                // Default node size — the Builder UI's HTML overlay needs real
                // estate to be useful out of the box. ComfyUI's default is
                // ~200x100 which leaves the UI invisible. Bump to a generous
                // default; users can still resize. Only applied on FRESH
                // create (not on workflow load — saved size is preserved).
                if (!this._uscgSizeInitialized) {
                    this._uscgSizeInitialized = true;
                    this.size = [
                        Math.max(this.size?.[0] || 0, 720),
                        Math.max(this.size?.[1] || 0, 900),
                    ];
                }

                // 2. Define methods (Synchronously attached)
                this.triggerAutoSave = function () {
                    if (this.state.auto_save && this.state.config_name) {
                        if (this.autoSaveTimer) clearTimeout(this.autoSaveTimer);
                        this.autoSaveTimer = setTimeout(() => {
                            this.saveConfigToBackend();
                        }, 2000);
                    }
                };

                this.saveState = function () {
                    // Guard: Ensure modules are loaded before trying to update preview
                    if (!configWidget || !configManagement) return;
                    
                    configWidget.value = JSON.stringify(this.state, null, 2);
                    this.updatePreview();
                    this.triggerAutoSave();
                };

                this.updatePreview = function () {
                    if (configManagement) {
                        configManagement.updatePreview(this);
                    }
                };

                this.saveConfigToBackend = async function () {
                    const name = this.state.config_name;
                    if (!name) return;
                    try {
                        await fetch("/configbuilder/save_config", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ name: name, data: this.state })
                        });
                        if (utilities) { utilities.clearConfigsCache(); await utilities.getAvailableConfigs(); }
                    } catch (e) {
                        console.error("Save Failed", e);
                    }
                };

                this.loadConfigFromBackend = async function (filename) {
                    try {
                        const resp = await fetch("/configbuilder/load_config", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ name: filename })
                        });
                        if (resp.ok) {
                            const data = await resp.json();
                            this.state = data;
                            if (!this.state.config_name) this.state.config_name = filename.replace(".json", "");
                            if (this.state.auto_save === undefined) this.state.auto_save = false;

                            // Migration: ensure config_arrays have all required fields
                            if (this.state.config_arrays) {
                                this.state.config_arrays.forEach(arr => {
                                    if (!arr.attention_modes) arr.attention_modes = ["default"];
                                    if (!arr.resolutions) arr.resolutions = [];
                                    if (arr.model_sampling_override === undefined) arr.model_sampling_override = "none";
                                    if (arr.model_sampling_shift === undefined) arr.model_sampling_shift = "1.73";
                                    if (arr.model_sampling_flux_max_shift === undefined) arr.model_sampling_flux_max_shift = "1.15";
                                    if (arr.model_sampling_flux_base_shift === undefined) arr.model_sampling_flux_base_shift = "0.5";
                                    if (arr.use_advanced_sampling === undefined) arr.use_advanced_sampling = false;
                                    if (arr.advanced_guider === undefined) arr.advanced_guider = "cfg_guider";
                                    if (arr.advanced_scheduler === undefined) arr.advanced_scheduler = "basic";
                                    if (arr.use_flux_guidance === undefined) arr.use_flux_guidance = false;
                                    if (arr.flux_guidance_value === undefined) arr.flux_guidance_value = "3.5";
                                    if (arr.use_deep_shrink === undefined) arr.use_deep_shrink = false;
                                    if (arr.deep_shrink_block_number === undefined) arr.deep_shrink_block_number = 3;
                                    if (arr.deep_shrink_downscale_factor === undefined) arr.deep_shrink_downscale_factor = 2.0;
                                    if (arr.deep_shrink_start_percent === undefined) arr.deep_shrink_start_percent = 0.0;
                                    if (arr.deep_shrink_end_percent === undefined) arr.deep_shrink_end_percent = 0.35;
                                    if (arr.deep_shrink_downscale_after_skip === undefined) arr.deep_shrink_downscale_after_skip = true;
                                    if (arr.deep_shrink_downscale_method === undefined) arr.deep_shrink_downscale_method = "bicubic";
                                    if (arr.deep_shrink_upscale_method === undefined) arr.deep_shrink_upscale_method = "bicubic";
                                    if (arr.use_slg === undefined) arr.use_slg = false;
                                    if (arr.slg_double_layers === undefined) arr.slg_double_layers = "";
                                    if (arr.slg_single_layers === undefined) arr.slg_single_layers = "22";
                                    if (arr.slg_start_percent === undefined) arr.slg_start_percent = 0.0;
                                    if (arr.slg_end_percent === undefined) arr.slg_end_percent = 0.5;
                                    if (arr.kind === undefined) arr.kind = "standard";
                                });
                            }

                            // Migration: ensure top-level Run Settings fields exist
                            if (this.state.overwrite_existing === undefined) this.state.overwrite_existing = false;
                            if (this.state.flush_batch_every === undefined) this.state.flush_batch_every = 1;
                            if (this.state.lora_triggerwords_mode === undefined) this.state.lora_triggerwords_mode = "None";
                            if (this.state.save_conditioning_cache_to_file === undefined) this.state.save_conditioning_cache_to_file = false;
                            if (this.state.enable_model_cache === undefined) this.state.enable_model_cache = false;
                            if (this.state.vae_batch_size === undefined) this.state.vae_batch_size = 1;
                            if (this.state.add_random_seeds_to_gens === undefined) this.state.add_random_seeds_to_gens = 0;
                            if (this.state.te_batch_size === undefined) this.state.te_batch_size = 1;
                            if (this.state.job_order === undefined) this.state.job_order = "as_written";
                            if (this.state.preview_pass === undefined) this.state.preview_pass = { enabled: false, steps: 8, scale: 0.5, cfg_override: null };
                            if (this.state.skip_rejected_previews === undefined) this.state.skip_rejected_previews = true;
                            if (this.state.only_favorited_previews === undefined) this.state.only_favorited_previews = false;
                            // Migration: optional image metrics (session-level, default OFF)
                            if (!this.state.metrics || typeof this.state.metrics !== "object") this.state.metrics = { enabled: false, builtins: {}, adapters: {} };
                            if (!this.state.metrics.builtins) this.state.metrics.builtins = {};
                            if (!this.state.metrics.adapters) this.state.metrics.adapters = {};
                            ["sharpness", "entropy", "colorfulness", "contrast", "luma", "saturation"].forEach(k => {
                                if (this.state.metrics.builtins[k] === undefined) this.state.metrics.builtins[k] = true;
                            });

                            this.saveState();
                            this.renderUI();
                        }
                    } catch (e) {
                        console.error("Load failed", e);
                    }
                };
                
                this.renderUI = async function () {
                    // Guard against running before modules are ready
                    if (!utilities || !configManagement) return;

                    const [availableLoras, loraFolders, availableSessions, availableConfigs] = await Promise.all([
                        utilities.getAvailableLoras(),
                        utilities.getLoraFolders(),
                        utilities.getAvailableSessions(),
                        utilities.getAvailableConfigs(),
                        utilities.getModelLists()
                    ]);

                    // Build modelLists object for config management
                    const modelLists = {
                        checkpoints: await utilities.getAvailableModels(),
                        checkpointFolders: await utilities.getModelFolders(),
                        diffusionModels: utilities.getAvailableDiffusionModels(),
                        diffusionFolders: utilities.getDiffusionModelFolders(),
                        ggufModels: utilities.getAvailableGGUFModels(),
                        ggufFolders: utilities.getGGUFModelFolders(),
                        textEncoders: utilities.getAvailableTextEncoders(),
                        textEncoderFolders: utilities.getTextEncoderFolders(),
                        clipTypes: utilities.getClipTypes(),
                        dualClipTypes: utilities.getDualClipTypes(),
                        vaeModels: utilities.getAvailableVAEs(),
                        vaeFolders: utilities.getVAEFolders(),
                        samplers: utilities.getAvailableSamplers(),
                        schedulers: utilities.getAvailableSchedulers(),
                        upscaleModels: utilities.getAvailableUpscaleModels(),
                        upscaleModelFolders: utilities.getUpscaleModelFolders(),
                        latentUpscaleModels: utilities.getAvailableLatentUpscaleModels ? utilities.getAvailableLatentUpscaleModels() : [],
                        latentUpscaleModelFolders: utilities.getLatentUpscaleModelFolders ? utilities.getLatentUpscaleModelFolders() : ["/"]
                    };

                    await configManagement.renderUI(
                        this,
                        availableLoras,
                        modelLists,
                        loraFolders,
                        availableSessions,
                        availableConfigs,
                        utilities.refreshAllConfigBuilders
                    );
                };

                this.loadSession = async function (sessionName) {
                    if (!utilities) return; // Guard
                    console.log(`[ConfigBuilder] Loading session: ${sessionName}`);

                    const loraFolders = await utilities.getLoraFolders();
                    const modelFolders = await utilities.getModelFolders();

                    try {
                        const manifestUrl = `/view?filename=manifest.json&type=output&subfolder=benchmarks/${sessionName}&t=${Date.now()}`;
                        const resp = await fetch(manifestUrl);
                        if (!resp.ok) return;
                        const manifest = await resp.json();
                        const meta = manifest.meta || {};

                        if (meta.configs_json) {
                            try {
                                let configs = JSON.parse(meta.configs_json);
                                // Handle new format: {configs: [...], _distribution: {...}}
                                if (configs && !Array.isArray(configs) && configs.configs) {
                                    configs = configs.configs;
                                }
                                let loadedArrays = utilities.convertConfigsToConfigArrays(configs);
                                const normalize = (str) => str.replace(/\\/g, "/");

                                loadedArrays.forEach(arr => {
                                    arr.loras = arr.loras.map(loraStr => {
                                        const parsed = utilities.parseLoraString(loraStr);
                                        const normName = normalize(parsed.name);
                                        if (normName.endsWith("/") || normName.endsWith("/*")) {
                                            return utilities.buildLoraString(normName, parsed.model_str, parsed.clip_str);
                                        }
                                        const potentialFolder = normName + "/";
                                        if (loraFolders && loraFolders.includes(potentialFolder)) {
                                            return utilities.buildLoraString(potentialFolder, parsed.model_str, parsed.clip_str);
                                        }
                                        if (loraFolders && loraFolders.includes(normName)) {
                                            return utilities.buildLoraString(normName, parsed.model_str, parsed.clip_str);
                                        }
                                        return loraStr;
                                    });

                                    // Handle both string and object-format models
                                    arr.models = arr.models.map(modelEntry => {
                                        if (typeof modelEntry === 'object' && modelEntry !== null) {
                                            // Object format: {path, type} — normalize path
                                            return {
                                                ...modelEntry,
                                                path: normalize(modelEntry.path || "")
                                            };
                                        }
                                        // String format (checkpoint) — normalize as before
                                        const normModel = normalize(modelEntry);
                                        if (normModel.endsWith("/")) return normModel;
                                        if (modelFolders && modelFolders.includes(normModel + "/")) return normModel + "/";
                                        return normModel;
                                    });

                                    // Ensure new fields exist
                                    if (!arr.text_encoders) arr.text_encoders = [];
                                    if (!arr.clip_type) arr.clip_type = "stable_diffusion";
                                    if (!arr.gguf_options) arr.gguf_options = {};
                                    if (!arr.vaes) arr.vaes = ["None"];

                                    // Ensure extra model & sampling options exist
                                    if (arr.model_sampling_override === undefined) arr.model_sampling_override = "none";
                                    if (arr.model_sampling_shift === undefined) arr.model_sampling_shift = "1.73";
                                    if (arr.model_sampling_flux_max_shift === undefined) arr.model_sampling_flux_max_shift = "1.15";
                                    if (arr.model_sampling_flux_base_shift === undefined) arr.model_sampling_flux_base_shift = "0.5";
                                    if (arr.use_advanced_sampling === undefined) arr.use_advanced_sampling = false;
                                    if (arr.advanced_guider === undefined) arr.advanced_guider = "cfg_guider";
                                    if (arr.advanced_scheduler === undefined) arr.advanced_scheduler = "basic";
                                    if (arr.use_flux_guidance === undefined) arr.use_flux_guidance = false;
                                    if (arr.flux_guidance_value === undefined) arr.flux_guidance_value = "3.5";
                                    if (arr.use_deep_shrink === undefined) arr.use_deep_shrink = false;
                                    if (arr.deep_shrink_block_number === undefined) arr.deep_shrink_block_number = 3;
                                    if (arr.deep_shrink_downscale_factor === undefined) arr.deep_shrink_downscale_factor = 2.0;
                                    if (arr.deep_shrink_start_percent === undefined) arr.deep_shrink_start_percent = 0.0;
                                    if (arr.deep_shrink_end_percent === undefined) arr.deep_shrink_end_percent = 0.35;
                                    if (arr.deep_shrink_downscale_after_skip === undefined) arr.deep_shrink_downscale_after_skip = true;
                                    if (arr.deep_shrink_downscale_method === undefined) arr.deep_shrink_downscale_method = "bicubic";
                                    if (arr.deep_shrink_upscale_method === undefined) arr.deep_shrink_upscale_method = "bicubic";
                                    if (arr.use_slg === undefined) arr.use_slg = false;
                                    if (arr.slg_double_layers === undefined) arr.slg_double_layers = "";
                                    if (arr.slg_single_layers === undefined) arr.slg_single_layers = "22";
                                    if (arr.slg_start_percent === undefined) arr.slg_start_percent = 0.0;
                                    if (arr.slg_end_percent === undefined) arr.slg_end_percent = 0.5;
                                    if (arr.kind === undefined) arr.kind = "standard";
                                });

                                this.state.config_arrays = loadedArrays;

                                // Restore global prompts from loaded configs
                                // When configs have _prompt_source: "global", extract those prompts
                                // as global prompts rather than per-config custom prompts
                                const firstGlobalConfig = configs.find(c => c._prompt_source === "global");
                                if (firstGlobalConfig) {
                                    if (firstGlobalConfig.positive) {
                                        if (Array.isArray(firstGlobalConfig.positive)) {
                                            this.state.global_positive_groups = firstGlobalConfig.positive;
                                        } else if (typeof firstGlobalConfig.positive === 'string' && firstGlobalConfig.positive.trim()) {
                                            this.state.global_positive_groups = [[firstGlobalConfig.positive]];
                                        }
                                    }
                                    if (firstGlobalConfig.negative) {
                                        this.state.global_negative = typeof firstGlobalConfig.negative === 'string'
                                            ? firstGlobalConfig.negative : "";
                                    }
                                } else {
                                    // No global prompts in loaded session
                                    this.state.global_positive_groups = [];
                                    this.state.global_negative = "";
                                }

                            } catch (e) {
                                console.error("[ConfigBuilder] Error parsing configs_json:", e);
                            }
                        }

                        this.state.session_name = sessionName;
                        this.saveState();
                        this.renderUI();
                    } catch (e) {
                        console.error("[ConfigBuilder] Error loading session:", e);
                    }
                };

                this.migrateOldFormat = function (oldState) {
                    if (!utilities) return this.state; // Fallback
                    const arrays = oldState.lora_config?.arrays || [];
                    return {
                        session_name: oldState.session_name || "my_test_session",
                        config_name: "default_config",
                        auto_save: false,
                        include_none: oldState.include_none !== undefined ? oldState.include_none : false,
                        global_positive_groups: [],
                        global_negative: "",
                        config_arrays: arrays.map(arr => ({
                            name: arr.name,
                            samplers: Array.isArray(oldState.samplers) ? oldState.samplers : (oldState.samplers || "euler").split(",").map(s => s.trim()).filter(s => s),
                            schedulers: Array.isArray(oldState.schedulers) ? oldState.schedulers : (oldState.schedulers || "normal").split(",").map(s => s.trim()).filter(s => s),
                            steps: oldState.steps || "20",
                            cfg: oldState.cfg || "7.0",
                            models: oldState.model ? [utilities.normalizePath(oldState.model)] : ["None"],
                            vaes: ["None"],
                            text_encoders: [],
                            clip_type: "stable_diffusion",
                            gguf_options: {},
                            loras: arr.loras ? arr.loras.map(l => utilities.normalizePath(l)) : ["None"],
                            lora_omit_triggers: [],
                            lora_triggerwords_append_settings: {},
                            lora_bypass_states: {},
                            lora_strength_lock: {},
                            model_bypass_states: {},
                            combine: arr.combine || false,
                            positive_prompt_groups: [],
                            negative_prompt: "",
                            use_custom_prompts: false,
                            model_prompt_prefix: "",
                            model_prompt_suffix: "",
                            attention_modes: ["default"]
                        }))
                    };
                };

                // 3. Asynchronous Initialization (Fire and Forget)
                // This allows the node to be "Created" immediately, while data loads in background.
                (async () => {
                    await ensureModulesLoaded();
                    
                    // Register this node for refresh tracking
                    utilities.getActiveConfigBuilderNodes().add(this);

                    // Process existing state now that utilities are loaded
                    try {
                        const existing = JSON.parse(configWidget.value);
                        if (existing.config_arrays) {
                            this.state = existing;
                            if (!this.state.config_name) this.state.config_name = "default_config";
                            if (this.state.auto_save === undefined) this.state.auto_save = false;
                            if (this.state.label_mode === undefined) this.state.label_mode = false;

                            // Migration: ensure global prompt fields exist
                            if (!this.state.global_positive_groups) this.state.global_positive_groups = [];
                            if (this.state.global_negative === undefined) this.state.global_negative = "";

                            // Migration: ensure distribution fields exist
                            if (this.state.distribution_enabled === undefined) this.state.distribution_enabled = false;
                            if (!this.state.worker_urls) this.state.worker_urls = [];
                            if (this.state.claim_timeout === undefined) this.state.claim_timeout = 600;
                            if (this.state.use_master_encoding === undefined) this.state.use_master_encoding = false;

                            // Migration logic requiring utilities
                            this.state.config_arrays.forEach(arr => {
                                if (arr.model && !arr.models) {
                                    arr.models = [arr.model];
                                    delete arr.model;
                                }
                                if (!arr.models) arr.models = ["None"];
                                // Normalize models — handle both string and object formats
                                arr.models = arr.models.map(m => {
                                    if (typeof m === 'object' && m !== null) {
                                        return { ...m, path: utilities.normalizePath(m.path || "") };
                                    }
                                    return utilities.normalizePath(m);
                                });
                                arr.loras = arr.loras ? arr.loras.map(l => {
                                    const p = utilities.parseLoraString(l);
                                    return utilities.buildLoraString(p.name, p.model_str, p.clip_str);
                                }) : ["None"];

                                // Ensure keys exist
                                if (!arr.lora_omit_triggers) arr.lora_omit_triggers = [];
                                if (!arr.lora_triggerwords_append_settings) arr.lora_triggerwords_append_settings = {};
                                if (!arr.lora_bypass_states) arr.lora_bypass_states = {};
                                if (!arr.lora_strength_lock) arr.lora_strength_lock = {};
                                if (!arr.model_bypass_states) arr.model_bypass_states = {};

                                // Migration: ensure prompt fields exist
                                if (!arr.positive_prompt_groups) arr.positive_prompt_groups = [];
                                if (arr.negative_prompt === undefined) arr.negative_prompt = "";
                                if (arr.use_custom_prompts === undefined) arr.use_custom_prompts = false;

                                // Migration: ensure new model type fields exist
                                if (!arr.text_encoders) arr.text_encoders = [];
                                if (!arr.clip_type) arr.clip_type = "stable_diffusion";
                                if (!arr.gguf_options) arr.gguf_options = {};

                                // Migration: ensure VAE field exists
                                if (!arr.vaes) arr.vaes = ["None"];

                                // Migration: ensure model prompt prefix/suffix fields exist
                                if (arr.model_prompt_prefix === undefined) arr.model_prompt_prefix = "";
                                if (arr.model_prompt_suffix === undefined) arr.model_prompt_suffix = "";

                                // Migration: ensure attention modes field exists
                                if (!arr.attention_modes) arr.attention_modes = ["default"];

                                // Migration: ensure resolutions field exists
                                if (!arr.resolutions) arr.resolutions = [];

                                // Migration: ensure extra model & sampling options exist
                                if (arr.model_sampling_override === undefined) arr.model_sampling_override = "none";
                                if (arr.model_sampling_shift === undefined) arr.model_sampling_shift = "1.73";
                                if (arr.model_sampling_flux_max_shift === undefined) arr.model_sampling_flux_max_shift = "1.15";
                                if (arr.model_sampling_flux_base_shift === undefined) arr.model_sampling_flux_base_shift = "0.5";
                                if (arr.use_advanced_sampling === undefined) arr.use_advanced_sampling = false;
                                if (arr.advanced_guider === undefined) arr.advanced_guider = "cfg_guider";
                                if (arr.advanced_scheduler === undefined) arr.advanced_scheduler = "basic";
                                if (arr.use_flux_guidance === undefined) arr.use_flux_guidance = false;
                                if (arr.flux_guidance_value === undefined) arr.flux_guidance_value = "3.5";
                                if (arr.use_deep_shrink === undefined) arr.use_deep_shrink = false;
                                if (arr.deep_shrink_block_number === undefined) arr.deep_shrink_block_number = 3;
                                if (arr.deep_shrink_downscale_factor === undefined) arr.deep_shrink_downscale_factor = 2.0;
                                if (arr.deep_shrink_start_percent === undefined) arr.deep_shrink_start_percent = 0.0;
                                if (arr.deep_shrink_end_percent === undefined) arr.deep_shrink_end_percent = 0.35;
                                if (arr.deep_shrink_downscale_after_skip === undefined) arr.deep_shrink_downscale_after_skip = true;
                                if (arr.deep_shrink_downscale_method === undefined) arr.deep_shrink_downscale_method = "bicubic";
                                if (arr.deep_shrink_upscale_method === undefined) arr.deep_shrink_upscale_method = "bicubic";
                                if (arr.use_slg === undefined) arr.use_slg = false;
                                if (arr.slg_double_layers === undefined) arr.slg_double_layers = "";
                                if (arr.slg_single_layers === undefined) arr.slg_single_layers = "22";
                                if (arr.slg_start_percent === undefined) arr.slg_start_percent = 0.0;
                                if (arr.slg_end_percent === undefined) arr.slg_end_percent = 0.5;
                                if (arr.kind === undefined) arr.kind = "standard";
                            });
                        } else if (existing.lora_config) {
                            this.state = this.migrateOldFormat(existing);
                        }
                    } catch (e) { }

                    // Initial Data Fetch — localStorage cache provides instant data,
                    // these calls refresh from server in background
                    await Promise.all([
                        utilities.getAvailableLoras(),
                        utilities.getLoraFolders(),
                        utilities.getModelLists(),
                        utilities.getAvailableSessions(),
                        utilities.getAvailableConfigs()
                    ]);

                    // Finally Render
                    this.renderUI();

                    // Start background polling for model changes (every 30s)
                    const self = this;
                    utilities.startModelCountPolling(async (newCounts) => {
                        console.log("[ConfigBuilder] 🔍 Model changes detected, refreshing...");
                        utilities.clearAllCaches();
                        await Promise.all([
                            utilities.getAvailableLoras(),
                            utilities.getLoraFolders(),
                            utilities.getModelLists()
                        ]);
                        self.renderUI();
                    });
                })();

                return result;
            };

            // Add cleanup when node is removed
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function () {
                if (utilities) {
                    utilities.getActiveConfigBuilderNodes().delete(this);
                    // Stop polling if no more active nodes
                    if (utilities.getActiveConfigBuilderNodes().size === 0) {
                        utilities.stopModelCountPolling();
                    }
                }
                if (onRemoved) {
                    onRemoved.apply(this, arguments);
                }
            };
        }
    },

    // Listen for API calls that indicate node definitions were updated
    async setup() {
        console.log("[ConfigBuilder] Setting up auto-refresh listener");

        // Ensure modules are loaded at app setup time
        await ensureModulesLoaded();

        let isRefreshing = false;

        // Hook into the app's refreshComboInNodes function if it exists
        if (app.refreshComboInNodes) {
            const originalRefresh = app.refreshComboInNodes;
            app.refreshComboInNodes = async function () {
                console.log("[ConfigBuilder] 🔄 Detected refreshComboInNodes call");

                const result = await originalRefresh.apply(this, arguments);

                if (!isRefreshing && utilities) {
                    isRefreshing = true;
                    setTimeout(async () => {
                        console.log("[ConfigBuilder] Clearing caches and refreshing nodes");
                        await utilities.refreshAllConfigBuilders();
                        isRefreshing = false;
                    }, 1000);
                }

                return result;
            };
        }

        // Also monitor fetch calls to /object_info as a backup
        const originalFetch = window.fetch;
        let lastObjectInfoTime = 0;
        window.fetch = async function (...args) {
            const options = args[1];
            if (options && options.headers && options.headers["X-Config-Builder-Internal"]) {
                return originalFetch.apply(this, args);
            }

            const result = await originalFetch.apply(this, args);

            if (args[0] && typeof args[0] === 'string' && args[0].includes('/object_info')) {
                const now = Date.now();
                if (now - lastObjectInfoTime > 2000 && !isRefreshing && utilities) {
                    lastObjectInfoTime = now;
                    console.log("[ConfigBuilder] 🔄 Detected EXTERNAL /object_info fetch");

                    isRefreshing = true;
                    setTimeout(async () => {
                        console.log("[ConfigBuilder] Clearing caches and refreshing nodes");
                        await utilities.refreshAllConfigBuilders();
                        isRefreshing = false;
                    }, 1000);
                }
            }

            return result;
        };

        console.log("[ConfigBuilder] Auto-refresh listener installed");
    }
});

console.log('[ConfigBuilder] ✅ Main entry point loaded');