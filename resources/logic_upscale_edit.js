// ============================================================================
// UPSCALE EDIT — pure helpers for the dashboard upscale modal (no DOM, node-testable).
// Step defaults (moved here from logic_upscale_modal.js; the modal keeps calling the same global
// names), config normalisation, validation (mirrored by upscale_runner.validate_upscale_config
// through tests/fixtures/upscale_validation_cases.json), reordering, equality, preset shape and the
// output counter. Loaded right BEFORE logic_upscale_modal.js; all files share one inlined script scope.
// ============================================================================

// Default Florence2 Hi-Res Fix options — mirrors createDefaultFlorence2Options()
// in conf-builder-config-management.js. Keep these two in sync if you change either.
function _defaultFlorence2Options() {
    return {
        model: "microsoft/Florence-2-base",
        text_input: "face",
        output_mask_select: "",
        max_new_tokens: 1024,
        florence2_input_mp: 0.5,
        target_megapixels: 1.0,
        crop_padding: 64,
        min_crop_resolution: 0,       // No floor — Florence2's polygon decides
        max_crop_resolution: 99999,   // No ceiling — paste-back natural cap is the image itself
        grow_expand: 32,
        feather_left: 128,
        feather_top: 128,
        feather_right: 128,
        feather_bottom: 128,
        model_source: "from_manifest",
        on_no_detection: "skip"
    };
}

// Default SeedVR2 options — matches SeedVR2 node defaults
function _defaultSeedVR2Options() {
    return {
        dit_model: "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        resolution: 1080,
        max_resolution: 0,
        seed: 42,
        color_correction: "lab",
        batch_size: 1,
        input_noise_scale: 0.0,
        latent_noise_scale: 0.0,
        blocks_to_swap: 0,
        attention_mode: "sdpa",
        offload_device: "cpu",
        cache_model: false,
        encode_tiled: false,
        encode_tile_size: 1024,
        encode_tile_overlap: 128,
        decode_tiled: false,
        decode_tile_size: 1024,
        decode_tile_overlap: 128,
        vae_offload_device: "none",
        vae_cache_model: false
    };
}

// Default upscale step config — ALL fields matching the Builder UI's createDefaultStep()
function _defaultUpscaleStep() {
    return {
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
        hires_force_uniform_tiles: false,
        seedvr2: _defaultSeedVR2Options(),
        florence2: _defaultFlorence2Options()
    };
}

// Ensure a step has all required fields (migration/defaults) — mirrors Builder's ensureStepFields()
function _ensureUpscaleStepFields(step) {
    if (step.active === undefined) step.active = true;
    if (!step.mode) step.mode = "hires_only";
    if (!step.repeat || step.repeat < 1) step.repeat = 1;
    if (!step.upscale_models) step.upscale_models = [];
    if (!step.upscale_ratios) step.upscale_ratios = "1.5";
    if (!step.upscale_size) step.upscale_size = "2.0";
    if (!step.hires_denoise) step.hires_denoise = "0.3";
    if (step.hires_steps === undefined) step.hires_steps = 0;
    if (step.tiled_vae === undefined) step.tiled_vae = false;
    if (!step.tile_size) step.tile_size = 512;
    if (!step.tile_overlap) step.tile_overlap = 64;
    if (!step.temporal_size) step.temporal_size = 512;
    if (!step.temporal_overlap) step.temporal_overlap = 64;
    if (!step.resize_method) step.resize_method = "bilinear";
    if (step.hires_tiled_sampling === undefined) step.hires_tiled_sampling = false;
    if (!step.hires_tile_width) step.hires_tile_width = 512;
    if (!step.hires_tile_height) step.hires_tile_height = 512;
    if (step.hires_mask_blur === undefined) step.hires_mask_blur = 8;
    if (!step.hires_tile_padding) step.hires_tile_padding = 32;
    if (step.hires_force_uniform_tiles === undefined) step.hires_force_uniform_tiles = false;
    if (!step.seedvr2) step.seedvr2 = _defaultSeedVR2Options();
    if (!step.florence2) step.florence2 = _defaultFlorence2Options();
}

// Deep-cloned config with defaults for every pipeline/step. Never invents a pipeline or a step.
function normalizeUpscaleConfig(cfg) {
    var src = (cfg && typeof cfg === 'object') ? cfg : {};
    var out = {
        pipelines: [],
        hires_prompt_adjust: !!src.hires_prompt_adjust,
        hires_prompt_behavior: src.hires_prompt_behavior || 'append_end',
        hires_prompt_text: src.hires_prompt_text || ''
    };
    (Array.isArray(src.pipelines) ? src.pipelines : []).forEach(function(p) {
        if (!p || typeof p !== 'object') return;
        var pipe = JSON.parse(JSON.stringify(p));
        if (pipe.active === undefined) pipe.active = true;
        if (!pipe.name) pipe.name = 'Pipeline';
        pipe.steps = Array.isArray(pipe.steps) ? pipe.steps.filter(function(s) { return s && typeof s === 'object'; }) : [];
        pipe.steps.forEach(_ensureUpscaleStepFields);
        out.pipelines.push(pipe);
    });
    return out;
}

// ---- validation -----------------------------------------------------------------------------
var _UPSCALE_MODES = ['hires_only', 'model_only', 'model_then_hires', 'seedvr2', 'florence2_hires'];
var _UPSCALE_NUM_RE = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

// Strict number: finite number or a plain decimal string; anything else -> null
function _upNum(v) {
    if (typeof v === 'number') return isFinite(v) ? v : null;
    if (typeof v === 'string') {
        var s = v.trim();
        return _UPSCALE_NUM_RE.test(s) ? parseFloat(s) : null;
    }
    return null;
}

// '1.5, 2' -> {parts, vals, bad}; empty entries are skipped like the runner does
function _upNumList(v) {
    var parts = String(v).split(',').map(function(p) { return p.trim(); }).filter(function(p) { return p; });
    var vals = [], bad = false;
    parts.forEach(function(p) { var n = _upNum(p); if (n === null) bad = true; else vals.push(n); });
    return { parts: parts, vals: vals, bad: bad };
}

function _upIsInt(n) { return n !== null && Math.floor(n) === n; }

// -> {ok, errors:[{path, field, msg}], warnings:[...]}. path is "p0.s1" (pipeline/step index) or "".
function validateUpscaleConfig(cfg, opts) {
    var available = (opts && opts.availableModels) || null;
    var errors = [], warnings = [];
    function err(path, field, msg) { errors.push({ path: path, field: field, msg: msg }); }
    function warn(path, field, msg) { warnings.push({ path: path, field: field, msg: msg }); }

    if (!cfg || typeof cfg !== 'object' || !Array.isArray(cfg.pipelines)) {
        err('', 'pipelines', 'Config has no pipelines');
        return { ok: false, errors: errors, warnings: warnings };
    }
    var c = normalizeUpscaleConfig(cfg);
    // indexes refer to the ORIGINAL arrays (normalize drops only non-object entries)
    var anyActive = false;
    c.pipelines.forEach(function(pipe, pi) {
        if (pipe.active === false) return;
        pipe.steps.forEach(function(step, si) {
            if (step.active === false) return;
            anyActive = true;
            var path = 'p' + pi + '.s' + si;
            var mode = step.mode;
            if (_UPSCALE_MODES.indexOf(mode) < 0) { err(path, 'mode', "Unknown mode '" + mode + "'"); return; }
            var hires = mode === 'hires_only' || mode === 'model_then_hires';
            var f2 = mode === 'florence2_hires';
            var model = mode === 'model_only' || mode === 'model_then_hires';
            var n, nl;

            var rn = _upNum(step.repeat);
            if (rn === null || !_upIsInt(rn) || rn < 1 || rn > 20) err(path, 'repeat', 'Repeat must be a whole number from 1 to 20');

            if (hires) {
                nl = _upNumList(step.upscale_ratios);
                if (!nl.parts.length) err(path, 'upscale_ratios', 'Enter at least one ratio, e.g. 1.5, 2');
                else if (nl.bad) err(path, 'upscale_ratios', 'Ratios must be numbers separated by commas');
                else if (nl.vals.some(function(v) { return v <= 0 || v > 8; })) err(path, 'upscale_ratios', 'Each ratio must be above 0 and at most 8');
            }
            if (hires || f2) {
                nl = _upNumList(step.hires_denoise);
                if (!nl.parts.length) err(path, 'hires_denoise', 'Enter at least one denoise value, e.g. 0.3');
                else if (nl.bad) err(path, 'hires_denoise', 'Denoise must be numbers separated by commas');
                else if (nl.vals.some(function(v) { return v < 0 || v > 1; })) err(path, 'hires_denoise', 'Each denoise value must be between 0 and 1');
                else if (f2 && nl.vals.length > 1) warn(path, 'hires_denoise', 'Florence2 uses only the first denoise value');
                var hn = _upNum(step.hires_steps);
                if (hn === null || !_upIsInt(hn) || hn < 0 || hn > 150) err(path, 'hires_steps', 'HiRes steps must be a whole number from 0 to 150');
            }
            if (hires && step.hires_tiled_sampling) {
                [['hires_tile_width', 'Tile width'], ['hires_tile_height', 'Tile height']].forEach(function(kl) {
                    n = _upNum(step[kl[0]]);
                    if (n === null || n < 64 || n % 8 !== 0) err(path, kl[0], kl[1] + ' must be a multiple of 8 and at least 64');
                });
            }
            if (hires && step.tiled_vae) {
                n = _upNum(step.tile_size);
                if (n === null || n < 128) err(path, 'tile_size', 'VAE tile size must be at least 128');
            }
            if (model) {
                var ms = Array.isArray(step.upscale_models) ? step.upscale_models.filter(function(m) { return m; }) : [];
                if (!ms.length) err(path, 'upscale_models', 'Select at least one upscale model');
                else if (available) {
                    ms.forEach(function(m) {
                        if (available.indexOf(m) < 0) warn(path, 'upscale_models', "Model '" + m + "' is not installed on this machine");
                    });
                }
            }
            if (mode === 'model_only') {
                n = _upNum(step.upscale_size);
                if (n === null || n <= 0) err(path, 'upscale_size', 'Size multiplier must be a number above 0');
            }
            if (mode === 'seedvr2') {
                var sv = (step.seedvr2 && typeof step.seedvr2 === 'object') ? step.seedvr2 : {};
                if (!String(sv.dit_model || '').trim()) err(path, 'dit_model', 'Choose a SeedVR2 DiT model');
                if (sv.resolution !== undefined && sv.resolution !== null) {
                    n = _upNum(sv.resolution);
                    if (n === null || n < 16) err(path, 'resolution', 'SeedVR2 resolution must be at least 16');
                }
            }
            if (f2) {
                var fl = (step.florence2 && typeof step.florence2 === 'object') ? step.florence2 : {};
                if (!String(fl.text_input || '').trim()) err(path, 'text_input', 'Say what Florence2 should detect (e.g. face)');
            }
        });
    });
    if (!anyActive) err('', 'pipelines', 'Turn on at least one pipeline that has an active step');
    if (c.hires_prompt_adjust && !String(c.hires_prompt_text || '').trim()) {
        warn('', 'hires_prompt_text', 'Prompt adjustment is on but the HiRes prompt is empty');
    }
    // UI-only hint (not mirrored server side): the runner saves ONE result per pipeline, so several
    // ratio/denoise/model values in the LAST step feed each other and only the last combination is kept.
    c.pipelines.forEach(function(pipe, pi) {
        if (pipe.active === false) return;
        var last = _lastActiveStep(pipe);
        if (last && _stepCombos(last.step) > 1) {
            warn('p' + pi + '.s' + last.index, 'combos', 'Several values in the final step: only the last combination is saved');
        }
    });
    return { ok: errors.length === 0, errors: errors, warnings: warnings };
}

// ---- editing helpers --------------------------------------------------------------------------
// Move arr[from] to index `to` in place (steps or pipelines); out-of-range is a no-op. Returns arr.
function moveItem(arr, from, to) {
    if (!Array.isArray(arr)) return arr;
    if (from === to || from < 0 || to < 0 || from >= arr.length || to >= arr.length) return arr;
    var it = arr.splice(from, 1)[0];
    arr.splice(to, 0, it);
    return arr;
}

function _stableStringify(v) {
    if (v === null || typeof v !== 'object') return JSON.stringify(v);
    if (Array.isArray(v)) return '[' + v.map(_stableStringify).join(',') + ']';
    return '{' + Object.keys(v).sort().map(function(k) { return JSON.stringify(k) + ':' + _stableStringify(v[k]); }).join(',') + '}';
}

// Equal after normalisation, independent of key order
function configsEqual(a, b) {
    return _stableStringify(normalizeUpscaleConfig(a)) === _stableStringify(normalizeUpscaleConfig(b));
}

// The shape stored in /configbuilder/upscale_presets (same as the Builder's presets)
function presetFromConfig(name, cfg) {
    var n = normalizeUpscaleConfig(cfg);
    return {
        name: name,
        pipelines: n.pipelines,
        hires_prompt_adjust: n.hires_prompt_adjust,
        hires_prompt_behavior: n.hires_prompt_behavior,
        hires_prompt_text: n.hires_prompt_text
    };
}

// ---- output counting (mirrors upscale_runner._upscale_one_item) ------------------------------------
function _lastActiveStep(pipe) {
    var last = null;
    (pipe.steps || []).forEach(function(s, i) { if (s && s.active !== false) last = { step: s, index: i }; });
    return last;
}

// Number of ratio x denoise x model combinations a single step runs
function _stepCombos(step) {
    var mode = step.mode || 'hires_only';
    if (mode === 'seedvr2' || mode === 'florence2_hires') return 1;
    var ratios = Math.max(1, _upNumList(step.upscale_ratios || '1.5').parts.length);
    var denoises = Math.max(1, _upNumList(step.hires_denoise || '0.3').parts.length);
    var models = Math.max(1, (step.upscale_models || []).filter(function(m) { return m; }).length);
    if (mode === 'hires_only') return ratios * denoises;
    if (mode === 'model_only') return models;
    return models * ratios * denoises;
}

// {perImage, perPipeline:[{index, name, outputs, combos}]}. The runner writes one manifest entry per
// active pipeline (the final step's last combination); `combos` is how many combinations that step runs.
// Chained steps multiply the work, not the outputs.
function countUpscaleOutputs(cfg) {
    var c = normalizeUpscaleConfig(cfg);
    var per = [];
    c.pipelines.forEach(function(pipe, i) {
        if (pipe.active === false) return;
        var last = _lastActiveStep(pipe);
        if (!last) return;
        per.push({ index: i, name: pipe.name, outputs: 1, combos: _stepCombos(last.step) });
    });
    return { perImage: per.reduce(function(a, p) { return a + p.outputs; }, 0), perPipeline: per };
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        _defaultFlorence2Options: _defaultFlorence2Options, _defaultSeedVR2Options: _defaultSeedVR2Options,
        _defaultUpscaleStep: _defaultUpscaleStep, _ensureUpscaleStepFields: _ensureUpscaleStepFields,
        normalizeUpscaleConfig: normalizeUpscaleConfig, validateUpscaleConfig: validateUpscaleConfig,
        moveItem: moveItem, configsEqual: configsEqual, presetFromConfig: presetFromConfig,
        countUpscaleOutputs: countUpscaleOutputs
    };
}
