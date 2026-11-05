// ============================================================================
// UPSCALE MODAL — Dashboard image upscaling with inline config + presets
// ============================================================================

// State for the upscale modal
var upscaleModalState = {
    open: false,
    imageId: null,
    imageData: null,
    presets: [],
    currentConfig: null,
    loadedPresetName: null,       // preset the working config was loaded from / last saved as
    loadedPresetSnapshot: null,   // normalized copy of that preset (what "modified" is measured against)
    validation: null,             // last validateUpscaleConfig() result
    serverErrors: [],             // validation errors the server returned for the last Run
    running: false,
    lastStatus: null,
    jobId: null,
    pollInterval: null,
    availableModels: [],  // Upscale model list from server
};

// Step defaults, normalisation, validation, moveItem, configsEqual, presetFromConfig and
// countUpscaleOutputs live in logic_upscale_edit.js (loaded right before this file).

// Default config
function _defaultUpscaleConfig() {
    return {
        pipelines: [{ active: true, name: "Pipeline 1", steps: [_defaultUpscaleStep()] }],
        hires_prompt_adjust: false, hires_prompt_behavior: "append_end", hires_prompt_text: "",
    };
}

/**
 * Open the upscale modal for a specific image
 */
function openUpscaleModal(imageId) {
    var item = activeData ? activeData.find(function(d) { return d.id === imageId; }) : null;
    if (!item) return;

    upscaleModalState.open = true;
    upscaleModalState.imageId = imageId;
    upscaleModalState.imageData = item;
    // NOTE: jobId / pollInterval are kept: a job started earlier keeps running while the modal is closed,
    // and reopening shows it again (the poll never depended on the modal being open).

    // Initialize with the last used config (survives a page reload), else the default
    if (!upscaleModalState.currentConfig) {
        _restoreUpscaleCfg();
        if (!upscaleModalState.currentConfig) upscaleModalState.currentConfig = _defaultUpscaleConfig();
    }

    // Fetch presets, model list and any running job for this session in parallel, then render
    Promise.all([
        fetchUpscalePresets(),
        fetchUpscaleModels(),
        _reattachUpscaleJob()
    ]).then(function() {
        _syncLoadedPreset();
        renderUpscaleModal();
    });
}

/**
 * Close the upscale modal. A running job is NOT stopped (use Cancel for that) and keeps being polled.
 */
function closeUpscaleModal() {
    upscaleModalState.open = false;
    var modal = document.getElementById('upscale-modal-overlay');
    if (modal) modal.style.display = 'none';
}

/**
 * Persist the working config + which preset it came from (per browser; try/catch: storage can be blocked)
 */
function _persistUpscaleCfg() {
    try {
        localStorage.setItem('uscg_upscale_modal_cfg', JSON.stringify({
            config: upscaleModalState.currentConfig,
            loadedPresetName: upscaleModalState.loadedPresetName
        }));
    } catch (e) { /* storage blocked: the modal still works */ }
}

function _restoreUpscaleCfg() {
    try {
        var raw = localStorage.getItem('uscg_upscale_modal_cfg');
        if (!raw) return;
        var saved = JSON.parse(raw);
        if (saved && saved.config && Array.isArray(saved.config.pipelines)) {
            upscaleModalState.currentConfig = normalizeUpscaleConfig(saved.config);
            upscaleModalState.loadedPresetName = saved.loadedPresetName || null;
        }
    } catch (e) { /* ignore a corrupt entry */ }
}

/**
 * After presets are fetched: the loaded preset's snapshot is whatever the server has NOW
 * (a preset that was deleted elsewhere turns the config into an unsaved one).
 */
function _syncLoadedPreset() {
    var name = upscaleModalState.loadedPresetName;
    if (!name) { upscaleModalState.loadedPresetSnapshot = null; return; }
    var p = null;
    upscaleModalState.presets.forEach(function(x) { if (x.name === name) p = x; });
    if (p) {
        upscaleModalState.loadedPresetSnapshot = normalizeUpscaleConfig(p);
    } else {
        upscaleModalState.loadedPresetName = null;
        upscaleModalState.loadedPresetSnapshot = null;
    }
}

/**
 * Fetch upscale presets from server
 */
async function fetchUpscalePresets() {
    try {
        var resp = await fetch('/configbuilder/upscale_presets');
        if (resp.ok) {
            var data = await resp.json();
            upscaleModalState.presets = data.presets || [];
        }
    } catch (e) { console.error('Failed to fetch upscale presets:', e); }
}

/**
 * Fetch available upscale models from server
 */
async function fetchUpscaleModels() {
    try {
        var resp = await fetch('/configbuilder/model_lists');
        if (resp.ok) {
            var data = await resp.json();
            upscaleModalState.availableModels = data.upscale_models || [];
        }
    } catch (e) { console.error('Failed to fetch upscale models:', e); }
}

/**
 * Current session name (same source the old Start button used)
 */
function _upscaleSessionName() {
    var el = document.getElementById('session-input');
    return (el && el.value) || 'default';
}

/**
 * After a reload there is no jobId in memory: ask the server whether a job is still running for this session
 */
async function _reattachUpscaleJob() {
    if (upscaleModalState.jobId) return;
    try {
        var resp = await fetch('/config_tester/upscale_jobs?session_name=' + encodeURIComponent(_upscaleSessionName()));
        if (!resp.ok) return;
        var data = await resp.json();
        var jobs = (data && data.jobs) || [];
        for (var i = 0; i < jobs.length; i++) {
            if (jobs[i].status === 'running' || jobs[i].status === 'cancelling') {
                upscaleModalState.jobId = jobs[i].job_id;
                upscaleModalState.running = true;
                _startUpscalePoll(jobs[i].job_id);
                return;
            }
        }
    } catch (e) { /* older server without the route: nothing to reattach */ }
}

/**
 * Read-modify-write on the shared presets file: the POST replaces the WHOLE file, so always re-read first
 * (another tab or the Builder may have changed it since this modal loaded its copy).
 */
async function _updateUpscalePresets(mutator) {
    var resp = await fetch('/configbuilder/upscale_presets');
    if (!resp.ok) throw new Error('could not read the presets file (HTTP ' + resp.status + ')');
    var data = await resp.json();
    var list = mutator((data && data.presets) || []);
    var post = await fetch('/configbuilder/upscale_presets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ presets: list })
    });
    if (!post.ok) throw new Error('could not save the presets file (HTTP ' + post.status + ')');
    upscaleModalState.presets = list;
    return list;
}

/**
 * Save the current config under `name` (replaces a preset with that name, else appends) and make it the loaded preset
 */
async function saveUpscalePreset(name) {
    name = (name || '').trim();
    if (!name || !upscaleModalState.currentConfig) return;
    var preset = presetFromConfig(name, upscaleModalState.currentConfig);
    try {
        await _updateUpscalePresets(function(list) {
            var idx = -1;
            list.forEach(function(p, i) { if (p.name === name) idx = i; });
            if (idx >= 0) list[idx] = preset; else list.push(preset);
            return list;
        });
    } catch (e) { alert('Failed to save preset: ' + e.message); return; }
    upscaleModalState.loadedPresetName = name;
    upscaleModalState.loadedPresetSnapshot = normalizeUpscaleConfig(preset);
    _persistUpscaleCfg();
    renderUpscaleModal();
}

/**
 * Delete a preset by name. The working config stays as it is (it becomes "Unsaved config").
 */
async function deleteUpscalePreset(name) {
    try {
        await _updateUpscalePresets(function(list) { return list.filter(function(p) { return p.name !== name; }); });
    } catch (e) { alert('Failed to delete preset: ' + e.message); return; }
    if (upscaleModalState.loadedPresetName === name) {
        upscaleModalState.loadedPresetName = null;
        upscaleModalState.loadedPresetSnapshot = null;
    }
    _persistUpscaleCfg();
    renderUpscaleModal();
}

/**
 * Load a preset (by index into upscaleModalState.presets) into the working config
 */
function loadUpscalePreset(index) {
    var preset = upscaleModalState.presets[index];
    if (!preset) return;
    var cfg = normalizeUpscaleConfig(preset);
    cfg.pipelines.forEach(function(p) {
        if (!p.steps || p.steps.length === 0) p.steps = [_defaultUpscaleStep()];
    });
    upscaleModalState.currentConfig = cfg;
    upscaleModalState.loadedPresetName = preset.name;
    upscaleModalState.loadedPresetSnapshot = normalizeUpscaleConfig(preset);
    _persistUpscaleCfg();
    renderUpscaleModal();
}

/**
 * Throw away the edits: back to the loaded preset
 */
function revertUpscaleConfig() {
    if (!upscaleModalState.loadedPresetSnapshot) return;
    upscaleModalState.currentConfig = JSON.parse(JSON.stringify(upscaleModalState.loadedPresetSnapshot));
    _persistUpscaleCfg();
    renderUpscaleModal();
}

/**
 * Helper: create a labeled select
 */
function _makeSelect(label, options, currentVal, onChange, key) {
    var row = document.createElement('div');
    row.style.cssText = 'display: flex; align-items: center; gap: 6px; margin-bottom: 6px;';
    var lbl = document.createElement('span');
    lbl.textContent = label;
    lbl.style.cssText = 'font-size: 11px; color: #999; min-width: 70px;';
    var sel = document.createElement('select');
    sel.style.cssText = 'flex: 1; background: #1a1a1a; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 3px 6px; font-size: 11px;';
    options.forEach(function(o) {
        var opt = document.createElement('option');
        opt.value = typeof o === 'object' ? o.value : o;
        opt.textContent = typeof o === 'object' ? o.label : o;
        if (opt.value === String(currentVal)) opt.selected = true;
        sel.appendChild(opt);
    });
    if (key) sel.setAttribute('data-field', key);
    sel.onchange = function() { onChange(sel.value); };
    row.appendChild(lbl);
    row.appendChild(sel);
    return row;
}

/**
 * Helper: create a labeled text input
 */
function _makeInput(label, currentVal, placeholder, onChange, key) {
    var row = document.createElement('div');
    row.style.cssText = 'display: flex; align-items: center; gap: 6px; margin-bottom: 6px;';
    var lbl = document.createElement('span');
    lbl.textContent = label;
    lbl.style.cssText = 'font-size: 11px; color: #999; min-width: 70px;';
    var inp = document.createElement('input');
    inp.type = 'text';
    inp.value = currentVal || '';
    inp.placeholder = placeholder || '';
    inp.style.cssText = 'flex: 1; background: #1a1a1a; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 3px 6px; font-size: 11px;';
    if (key) inp.setAttribute('data-field', key);
    inp.onchange = function() { onChange(inp.value); };
    row.appendChild(lbl);
    row.appendChild(inp);
    return row;
}

/**
 * Helper: create a labeled number input
 */
function _makeNumber(label, currentVal, min, max, step, onChange, key) {
    var row = document.createElement('div');
    row.style.cssText = 'display: flex; align-items: center; gap: 6px; margin-bottom: 6px;';
    var lbl = document.createElement('span');
    lbl.textContent = label;
    lbl.style.cssText = 'font-size: 11px; color: #999; min-width: 70px;';
    var inp = document.createElement('input');
    inp.type = 'number';
    inp.value = currentVal;
    inp.min = min;
    inp.max = max;
    inp.step = step;
    inp.style.cssText = 'flex: 1; background: #1a1a1a; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 3px 6px; font-size: 11px; max-width: 80px;';
    if (key) inp.setAttribute('data-field', key);
    inp.onchange = function() { var v = parseFloat(inp.value); onChange(isNaN(v) ? min : Math.max(min, Math.min(max, v))); };
    row.appendChild(lbl);
    row.appendChild(inp);
    return row;
}

/**
 * Helper: create a labeled checkbox
 */
function _makeCheckbox(label, checked, onChange) {
    var row = document.createElement('div');
    row.style.cssText = 'display: flex; align-items: center; gap: 6px; margin-bottom: 6px;';
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = checked;
    cb.onchange = function() { onChange(cb.checked); };
    var lbl = document.createElement('span');
    lbl.textContent = label;
    lbl.style.cssText = 'font-size: 11px; color: #ccc; cursor: pointer;';
    lbl.onclick = function() { cb.checked = !cb.checked; onChange(cb.checked); };
    row.appendChild(cb);
    row.appendChild(lbl);
    return row;
}

// ============================================================================
// PIPELINE EDITOR — Full pipeline/step UI, reusable component
// Mirrors the Builder UI's renderPipelines() in conf-builder-config-management.js
// ============================================================================

/**
 * Small square icon button (reorder arrows). Class + data-act let tests and CSS find it; the phone
 * stylesheet raises it to a 44px touch target.
 */
function _upmIconBtn(text, act, title, disabled, onclick) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'upm-icon-btn';
    b.setAttribute('data-act', act);
    b.textContent = text;
    b.title = title;
    b.disabled = !!disabled;
    b.style.cssText = 'background: #333; color: #cc99ff; border: 1px solid #665599; border-radius: 3px; padding: 2px 7px; font-size: 11px; cursor: pointer;' + (disabled ? ' opacity: 0.35;' : '');
    b.onclick = function(e) { e.stopPropagation(); onclick(); };
    return b;
}

/**
 * Render the full pipeline editor into a container.
 * @param {HTMLElement} container — DOM element to render into (will be cleared)
 * @param {Object} config — { pipelines: [...], hires_prompt_adjust, hires_prompt_behavior, hires_prompt_text }
 * @param {Array} modelList — array of upscale model filenames
 * @param {Object} callbacks — { onUpdate(), onRender() }
 *   onUpdate — called when any config value changes (caller can persist)
 *   onRender — called when DOM needs full re-render (add/remove pipeline/step, mode change)
 */
function renderPipelineEditor(container, config, modelList, callbacks) {
    container.textContent = '';

    // Ensure all pipelines/steps have required fields
    if (!config.pipelines || config.pipelines.length === 0) {
        config.pipelines = [{ active: true, name: "Pipeline 1", steps: [_defaultUpscaleStep()] }];
    }
    config.pipelines.forEach(function(p) {
        if (p.active === undefined) p.active = true;
        if (!p.name) p.name = "Pipeline";
        if (!p.steps || p.steps.length === 0) p.steps = [_defaultUpscaleStep()];
        p.steps.forEach(_ensureUpscaleStepFields);
    });

    // --- HiRes Prompt Adjustment Section ---
    var hiresDiv = document.createElement('div');
    hiresDiv.style.cssText = 'margin-bottom: 8px; padding: 6px 8px; background: #1e1e1e; border-radius: 4px;';
    var hiresPromptRow = _makeCheckbox('Adjust Prompt During HiRes Fix', config.hires_prompt_adjust || false, function(v) {
        config.hires_prompt_adjust = v;
        hiresOpts.style.display = v ? 'block' : 'none';
        callbacks.onUpdate();
    });
    hiresDiv.appendChild(hiresPromptRow);

    var hiresOpts = document.createElement('div');
    hiresOpts.style.cssText = 'margin-top: 4px; display: ' + (config.hires_prompt_adjust ? 'block' : 'none') + ';';
    hiresOpts.appendChild(_makeSelect('Behavior:', [
        { value: 'prepend', label: 'Append To Front' },
        { value: 'append_end', label: 'Append To End' },
        { value: 'replace', label: 'Replace Prompt' },
    ], config.hires_prompt_behavior || 'append_end', function(v) {
        config.hires_prompt_behavior = v;
        callbacks.onUpdate();
    }));
    // Prompt text textarea
    var promptRow = document.createElement('div');
    promptRow.style.cssText = 'margin-bottom: 6px;';
    var promptLbl = document.createElement('span');
    promptLbl.textContent = 'HiRes Prompt:';
    promptLbl.style.cssText = 'font-size: 11px; color: #999; display: block; margin-bottom: 2px;';
    var promptTa = document.createElement('textarea');
    promptTa.value = config.hires_prompt_text || '';
    promptTa.placeholder = 'Enter prompt adjustment text...';
    promptTa.style.cssText = 'width: 100%; min-height: 40px; background: #1a1a1a; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 4px 6px; font-size: 11px; resize: vertical; box-sizing: border-box;';
    promptTa.onchange = function() { config.hires_prompt_text = promptTa.value; callbacks.onUpdate(); };
    promptRow.appendChild(promptLbl);
    promptRow.appendChild(promptTa);
    hiresOpts.appendChild(promptRow);
    hiresDiv.appendChild(hiresOpts);
    container.appendChild(hiresDiv);

    // --- Pipelines ---
    var pipelinesContainer = document.createElement('div');

    function reRender() {
        renderPipelineEditor(container, config, modelList, callbacks);
        if (callbacks.onRender) callbacks.onRender();   // marks / validation must be re-applied to the NEW DOM
    }

    config.pipelines.forEach(function(pipeline, pipeIdx) {
        var pipeCard = document.createElement('div');
        pipeCard.style.cssText = 'border: 2px solid #665599; border-radius: 8px; margin-bottom: 8px; background: #1a1a2e;' + (pipeline.active === false ? ' opacity: 0.5;' : '');

        // Pipeline header (collapsible)
        var pipeHeader = document.createElement('div');
        pipeHeader.style.cssText = 'display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; cursor: pointer; background: #252540; border-radius: 6px 6px 0 0;';

        var pipeLeft = document.createElement('div');
        pipeLeft.style.cssText = 'display: flex; align-items: center; gap: 6px;';

        var pipeActiveCb = document.createElement('input');
        pipeActiveCb.type = 'checkbox';
        pipeActiveCb.checked = pipeline.active !== false;
        pipeActiveCb.title = pipeline.active !== false ? 'Pipeline is ON' : 'Pipeline is OFF';
        pipeActiveCb.onclick = function(e) { e.stopPropagation(); };
        pipeActiveCb.onchange = function() {
            pipeline.active = pipeActiveCb.checked;
            pipeCard.style.opacity = pipeline.active !== false ? '1' : '0.5';
            callbacks.onUpdate();
        };
        pipeLeft.appendChild(pipeActiveCb);

        var pipeNameInput = document.createElement('input');
        pipeNameInput.type = 'text';
        pipeNameInput.value = pipeline.name || ('Pipeline ' + (pipeIdx + 1));
        pipeNameInput.style.cssText = 'font-weight: bold; color: #cc99ff; font-size: 12px; background: transparent; border: 1px solid transparent; padding: 2px 4px; width: 130px;';
        pipeNameInput.onclick = function(e) { e.stopPropagation(); };
        pipeNameInput.onfocus = function() { pipeNameInput.style.borderColor = '#665599'; };
        pipeNameInput.onblur = function() { pipeNameInput.style.borderColor = 'transparent'; };
        pipeNameInput.onchange = function() { pipeline.name = pipeNameInput.value; callbacks.onUpdate(); };
        pipeLeft.appendChild(pipeNameInput);

        var pipeStepCount = document.createElement('span');
        pipeStepCount.style.cssText = 'font-size: 10px; color: #888;';
        pipeStepCount.textContent = '(' + pipeline.steps.length + ' step' + (pipeline.steps.length !== 1 ? 's' : '') + ')';
        pipeLeft.appendChild(pipeStepCount);

        pipeHeader.appendChild(pipeLeft);

        var pipeRight = document.createElement('div');
        pipeRight.style.cssText = 'display: flex; align-items: center; gap: 6px;';

        if (config.pipelines.length > 1) {
            pipeRight.appendChild(_upmIconBtn('▲', 'pipe-up', 'Move pipeline up', pipeIdx === 0, function() {
                moveItem(config.pipelines, pipeIdx, pipeIdx - 1);
                callbacks.onUpdate();
                reRender();
            }));
            pipeRight.appendChild(_upmIconBtn('▼', 'pipe-down', 'Move pipeline down', pipeIdx === config.pipelines.length - 1, function() {
                moveItem(config.pipelines, pipeIdx, pipeIdx + 1);
                callbacks.onUpdate();
                reRender();
            }));
            var pipeDelBtn = document.createElement('button');
            pipeDelBtn.textContent = '\u2715';
            pipeDelBtn.style.cssText = 'background: #cc3333; color: #fff; border: none; border-radius: 3px; padding: 2px 8px; font-size: 12px; cursor: pointer;';
            pipeDelBtn.onclick = function(e) {
                e.stopPropagation();
                config.pipelines.splice(pipeIdx, 1);
                callbacks.onUpdate();
                reRender();
            };
            pipeRight.appendChild(pipeDelBtn);
        }

        var collapseIcon = document.createElement('span');
        collapseIcon.style.cssText = 'color: #888; font-size: 12px; user-select: none;';
        collapseIcon.textContent = '\u25BC';
        pipeRight.appendChild(collapseIcon);

        pipeHeader.appendChild(pipeRight);
        pipeCard.appendChild(pipeHeader);

        // Pipeline body (collapsible)
        var pipeBody = document.createElement('div');
        pipeBody.style.cssText = 'padding: 6px 10px;';

        pipeHeader.onclick = function() {
            var isCollapsed = pipeBody.style.display === 'none';
            pipeBody.style.display = isCollapsed ? 'block' : 'none';
            collapseIcon.textContent = isCollapsed ? '\u25BC' : '\u25B6';
        };

        // Render steps
        pipeline.steps.forEach(function(ucfg, stepIdx) {
            _renderStep(pipeBody, config, pipeline, ucfg, stepIdx, modelList, callbacks, reRender);
        });

        // Add Step button
        var addStepBtn = document.createElement('button');
        addStepBtn.textContent = '+ Add Step';
        addStepBtn.style.cssText = 'background: #333; color: #cc99ff; border: 1px solid #665599; border-radius: 4px; padding: 3px 10px; font-size: 11px; cursor: pointer; margin-top: 4px;';
        addStepBtn.onclick = function() {
            pipeline.steps.push(_defaultUpscaleStep());
            callbacks.onUpdate();
            reRender();
        };
        pipeBody.appendChild(addStepBtn);

        pipeCard.appendChild(pipeBody);
        pipelinesContainer.appendChild(pipeCard);
    });

    // Add Pipeline button
    var addPipeBtn = document.createElement('button');
    addPipeBtn.textContent = '+ Add Pipeline';
    addPipeBtn.style.cssText = 'background: #333; color: #cc99ff; border: 1px solid #cc99ff; border-radius: 4px; padding: 4px 12px; font-size: 11px; cursor: pointer; margin-top: 4px;';
    addPipeBtn.onclick = function() {
        config.pipelines.push({
            active: true,
            name: 'Pipeline ' + (config.pipelines.length + 1),
            steps: [_defaultUpscaleStep()]
        });
        callbacks.onUpdate();
        reRender();
    };
    pipelinesContainer.appendChild(addPipeBtn);

    container.appendChild(pipelinesContainer);
}

/**
 * Render a single step card inside a pipeline body.
 * Internal helper for renderPipelineEditor.
 */
function _renderStep(pipeBody, config, pipeline, ucfg, stepIdx, modelList, callbacks, reRender) {
    var card = document.createElement('div');
    card.setAttribute('data-pipe', String(config.pipelines.indexOf(pipeline)));
    card.setAttribute('data-step', String(stepIdx));
    card.style.cssText = 'border: 1px solid #444; border-radius: 6px; margin-bottom: 6px; padding: 8px;' + (ucfg.active === false ? ' opacity: 0.5;' : '');

    // Card header
    var cardHeader = document.createElement('div');
    cardHeader.style.cssText = 'display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;';

    var cardLeft = document.createElement('div');
    cardLeft.style.cssText = 'display: flex; align-items: center; gap: 6px;';

    var activeCb = document.createElement('input');
    activeCb.type = 'checkbox';
    activeCb.checked = ucfg.active !== false;
    activeCb.onchange = function() {
        ucfg.active = activeCb.checked;
        card.style.opacity = ucfg.active !== false ? '1' : '0.5';
        callbacks.onUpdate();
    };
    cardLeft.appendChild(activeCb);

    var cardTitle = document.createElement('span');
    cardTitle.style.cssText = 'font-weight: bold; color: #cc99ff; font-size: 12px;';
    cardTitle.textContent = 'Step ' + (stepIdx + 1);
    cardLeft.appendChild(cardTitle);

    // Repeat field inline
    var repeatLbl = document.createElement('span');
    repeatLbl.style.cssText = 'font-size: 10px; color: #888; margin-left: 6px;';
    repeatLbl.textContent = 'Repeat:';
    cardLeft.appendChild(repeatLbl);

    var repeatInput = document.createElement('input');
    repeatInput.type = 'number';
    repeatInput.value = ucfg.repeat || 1;
    repeatInput.min = 1; repeatInput.max = 20; repeatInput.step = 1;
    repeatInput.setAttribute('data-field', 'repeat');
    repeatInput.style.cssText = 'width: 44px; padding: 1px 3px; font-size: 10px; background: #1a1a1a; color: #ccc; border: 1px solid #444; border-radius: 3px;';
    repeatInput.onchange = function() { ucfg.repeat = Math.max(1, Math.min(20, parseInt(repeatInput.value, 10) || 1)); callbacks.onUpdate(); reRender(); };
    cardLeft.appendChild(repeatInput);

    if (ucfg.repeat > 1) {
        var repeatNote = document.createElement('span');
        repeatNote.style.cssText = 'font-size: 9px; color: #cc99ff;';
        repeatNote.textContent = '(\u00D7' + ucfg.repeat + ' feedback loop)';
        cardLeft.appendChild(repeatNote);
    }

    cardHeader.appendChild(cardLeft);

    if (pipeline.steps.length > 1) {
        var cardRight = document.createElement('div');
        cardRight.style.cssText = 'display: flex; align-items: center; gap: 4px;';
        cardRight.appendChild(_upmIconBtn('▲', 'step-up', 'Move step up', stepIdx === 0, function() {
            moveItem(pipeline.steps, stepIdx, stepIdx - 1);
            callbacks.onUpdate();
            reRender();
        }));
        cardRight.appendChild(_upmIconBtn('▼', 'step-down', 'Move step down', stepIdx === pipeline.steps.length - 1, function() {
            moveItem(pipeline.steps, stepIdx, stepIdx + 1);
            callbacks.onUpdate();
            reRender();
        }));
        var delBtn = document.createElement('button');
        delBtn.type = 'button';
        delBtn.setAttribute('data-act', 'step-del');
        delBtn.textContent = '✕';
        delBtn.style.cssText = 'background: #cc3333; color: #fff; border: none; border-radius: 3px; padding: 2px 8px; font-size: 11px; cursor: pointer;';
        delBtn.onclick = function() {
            pipeline.steps.splice(stepIdx, 1);
            callbacks.onUpdate();
            reRender();
        };
        cardRight.appendChild(delBtn);
        cardHeader.appendChild(cardRight);
    }
    card.appendChild(cardHeader);

    // Grid of settings
    var grid = document.createElement('div');
    grid.style.cssText = 'display: flex; flex-wrap: wrap; gap: 4px;';

    // Mode select
    grid.appendChild(_makeSelect('Mode:', [
        { value: 'hires_only', label: 'HiRes Fix Only' },
        { value: 'model_only', label: 'Model Upscale Only' },
        { value: 'model_then_hires', label: 'Model + HiRes Fix' },
        { value: 'seedvr2', label: 'SeedVR2 Upscale' },
        { value: 'florence2_hires', label: 'Florence2 Hi-Res Fix' },
    ], ucfg.mode, function(v) {
        ucfg.mode = v;
        callbacks.onUpdate();
        reRender();
    }, 'mode'));

    // Resize method
    grid.appendChild(_makeSelect('Resize:', [
        'nearest-exact', 'bilinear', 'area', 'bicubic', 'lanczos'
    ], ucfg.resize_method || 'bilinear', function(v) {
        ucfg.resize_method = v;
        callbacks.onUpdate();
    }));

    var showHires = ucfg.mode === 'hires_only' || ucfg.mode === 'model_then_hires' || ucfg.mode === 'florence2_hires';
    var showModel = ucfg.mode === 'model_only' || ucfg.mode === 'model_then_hires';
    var showSeedVR2 = ucfg.mode === 'seedvr2';
    var showFlorence2 = ucfg.mode === 'florence2_hires';

    // --- HiRes fields ---
    if (showHires) {
        // Upscale Ratios is irrelevant for florence2_hires (it inpaints a region, not
        // the whole image; target_megapixels controls the internal pass size instead).
        if (!showFlorence2) {
            grid.appendChild(_makeInput('Ratios:', ucfg.upscale_ratios || '1.5', '1.2, 1.5, 2.0', function(v) {
                ucfg.upscale_ratios = v; callbacks.onUpdate();
            }, 'upscale_ratios'));
        }
        grid.appendChild(_makeInput('Denoise:', ucfg.hires_denoise || '0.3', '0.2, 0.3, 0.5', function(v) {
            ucfg.hires_denoise = v; callbacks.onUpdate();
        }, 'hires_denoise'));
        grid.appendChild(_makeNumber('HiRes Steps:', ucfg.hires_steps || 0, 0, 150, 1, function(v) {
            ucfg.hires_steps = v; callbacks.onUpdate();
        }, 'hires_steps'));

        // Tiled sampling — hidden for florence2_hires (crops are too small to need tiling).
        if (!showFlorence2) {
            grid.appendChild(_makeCheckbox('HiRes Tiled Sampling', ucfg.hires_tiled_sampling || false, function(v) {
                ucfg.hires_tiled_sampling = v; callbacks.onUpdate(); reRender();
            }));
        }

        if (ucfg.hires_tiled_sampling && !showFlorence2) {
            grid.appendChild(_makeNumber('Tile Width:', ucfg.hires_tile_width || 512, 128, 2048, 64, function(v) {
                ucfg.hires_tile_width = v; callbacks.onUpdate();
            }, 'hires_tile_width'));
            grid.appendChild(_makeNumber('Tile Height:', ucfg.hires_tile_height || 512, 128, 2048, 64, function(v) {
                ucfg.hires_tile_height = v; callbacks.onUpdate();
            }, 'hires_tile_height'));
            grid.appendChild(_makeNumber('Mask Blur:', ucfg.hires_mask_blur || 8, 0, 64, 1, function(v) {
                ucfg.hires_mask_blur = v; callbacks.onUpdate();
            }));
            grid.appendChild(_makeNumber('Tile Padding:', ucfg.hires_tile_padding || 32, 0, 256, 8, function(v) {
                ucfg.hires_tile_padding = v; callbacks.onUpdate();
            }));
            grid.appendChild(_makeCheckbox('Force Uniform Tiles', ucfg.hires_force_uniform_tiles !== false, function(v) {
                ucfg.hires_force_uniform_tiles = v; callbacks.onUpdate();
            }));
        }

        // Tiled VAE — hidden for florence2_hires (crops decode quickly without tiling).
        if (!showFlorence2) {
            grid.appendChild(_makeCheckbox('Tiled VAE Decode', ucfg.tiled_vae || false, function(v) {
                ucfg.tiled_vae = v; callbacks.onUpdate(); reRender();
            }));
        }

        if (ucfg.tiled_vae && !showFlorence2) {
            grid.appendChild(_makeNumber('Tile Size:', ucfg.tile_size || 512, 128, 1024, 64, function(v) {
                ucfg.tile_size = v; callbacks.onUpdate();
            }, 'tile_size'));
            grid.appendChild(_makeNumber('Tile Overlap:', ucfg.tile_overlap || 64, 0, 512, 8, function(v) {
                ucfg.tile_overlap = v; callbacks.onUpdate();
            }));
            grid.appendChild(_makeNumber('Temporal Size:', ucfg.temporal_size || 512, 128, 1024, 64, function(v) {
                ucfg.temporal_size = v; callbacks.onUpdate();
            }));
            grid.appendChild(_makeNumber('Temporal Overlap:', ucfg.temporal_overlap || 64, 0, 512, 8, function(v) {
                ucfg.temporal_overlap = v; callbacks.onUpdate();
            }));
        }
    }

    // --- Model fields ---
    if (showModel) {
        // Output size multiplier — only for model_only mode
        if (ucfg.mode === 'model_only') {
            grid.appendChild(_makeInput('Size Mult:', ucfg.upscale_size || '2.0', '2.0', function(v) {
                ucfg.upscale_size = v; callbacks.onUpdate();
            }, 'upscale_size'));
        }

        // Model multi-select: chips + searchable dropdown
        var modelsWrap = document.createElement('div');
        modelsWrap.style.cssText = 'width: 100%; margin-bottom: 6px;';

        modelsWrap.setAttribute('data-field', 'upscale_models');
        var noModels = !ucfg.upscale_models || ucfg.upscale_models.length === 0;
        if (noModels) {
            modelsWrap.style.border = '2px solid #ff3333';
            modelsWrap.style.borderRadius = '4px';
            modelsWrap.style.padding = '4px';
        }

        var modelsLbl = document.createElement('div');
        modelsLbl.style.cssText = 'font-size: 11px; color: ' + (noModels ? '#ff3333' : '#999') + '; margin-bottom: 3px;';
        modelsLbl.textContent = 'Upscale Models (' + (ucfg.upscale_models || []).length + ')' + (noModels ? ' \u2014 Select a model!' : '');
        modelsWrap.appendChild(modelsLbl);

        // Model chips
        (ucfg.upscale_models || []).forEach(function(modelName, mIdx) {
            var chip = document.createElement('div');
            chip.style.cssText = 'display: inline-flex; align-items: center; gap: 3px; background: #333; border-radius: 4px; padding: 2px 6px; margin: 2px; font-size: 10px; color: #ccc;';
            chip.textContent = modelName.replace(/\\/g, '/').split('/').pop();
            var removeBtn = document.createElement('span');
            removeBtn.textContent = '\u2715';
            removeBtn.style.cssText = 'cursor: pointer; color: #cc3333; margin-left: 3px; font-size: 11px;';
            removeBtn.onclick = function() {
                ucfg.upscale_models.splice(mIdx, 1);
                callbacks.onUpdate();
                reRender();
            };
            chip.appendChild(removeBtn);
            modelsWrap.appendChild(chip);
        });

        // Searchable model dropdown
        var modelSearchWrap = document.createElement('div');
        modelSearchWrap.style.cssText = 'position: relative; margin-top: 3px;';

        var modelSearchInput = document.createElement('input');
        modelSearchInput.type = 'text';
        modelSearchInput.placeholder = 'Search upscale models...';
        modelSearchInput.style.cssText = 'width: 100%; background: #1a1a1a; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 3px 6px; font-size: 10px; box-sizing: border-box;';

        var modelDropdown = document.createElement('div');
        modelDropdown.style.cssText = 'display: none; position: absolute; top: 100%; left: 0; right: 0; max-height: 150px; overflow-y: auto; background: #222; border: 1px solid #555; border-radius: 4px; z-index: 100;';

        function populateModelDropdown(query) {
            modelDropdown.textContent = '';
            var q = (query || '').toLowerCase();
            var existing = ucfg.upscale_models || [];
            var filtered = (modelList || []).filter(function(m) {
                if (existing.indexOf(m) >= 0) return false;
                return !q || m.toLowerCase().indexOf(q) >= 0;
            });
            if (filtered.length === 0) {
                modelDropdown.style.display = 'none';
                return;
            }
            modelDropdown.style.display = 'block';
            filtered.forEach(function(m) {
                var opt = document.createElement('div');
                opt.style.cssText = 'padding: 3px 8px; font-size: 10px; color: #ccc; cursor: pointer;';
                opt.textContent = m.replace(/\\/g, '/').split('/').pop();
                opt.title = m;
                opt.onmouseenter = function() { opt.style.background = '#444'; };
                opt.onmouseleave = function() { opt.style.background = 'transparent'; };
                opt.onclick = function() {
                    if (!ucfg.upscale_models) ucfg.upscale_models = [];
                    if (ucfg.upscale_models.indexOf(m) < 0) {
                        ucfg.upscale_models.push(m);
                        callbacks.onUpdate();
                        reRender();
                    }
                };
                modelDropdown.appendChild(opt);
            });
        }

        modelSearchInput.onfocus = function() { populateModelDropdown(modelSearchInput.value); };
        modelSearchInput.oninput = function() { populateModelDropdown(modelSearchInput.value); };
        modelSearchInput.onblur = function() {
            // Delay hide so click on dropdown item registers
            setTimeout(function() { modelDropdown.style.display = 'none'; }, 200);
        };

        modelSearchWrap.appendChild(modelSearchInput);
        modelSearchWrap.appendChild(modelDropdown);
        modelsWrap.appendChild(modelSearchWrap);
        grid.appendChild(modelsWrap);
    }

    // --- SeedVR2 fields ---
    if (showSeedVR2) {
        if (!ucfg.seedvr2) ucfg.seedvr2 = _defaultSeedVR2Options();
        var sv = ucfg.seedvr2;

        var ditModels = [
            'seedvr2_ema_3b_fp8_e4m3fn.safetensors', 'seedvr2_ema_3b_fp16.safetensors',
            'seedvr2_ema_3b-Q4_K_M.gguf', 'seedvr2_ema_3b-Q8_0.gguf',
            'seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors', 'seedvr2_ema_7b_fp16.safetensors',
            'seedvr2_ema_7b-Q4_K_M.gguf', 'seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors',
            'seedvr2_ema_7b_sharp_fp16.safetensors', 'seedvr2_ema_7b_sharp-Q4_K_M.gguf'
        ];
        grid.appendChild(_makeSelect('DiT Model:', ditModels.map(function(m) {
            return { value: m, label: m.replace('.safetensors', '').replace('.gguf', ' (GGUF)') };
        }), sv.dit_model, function(v) { sv.dit_model = v; callbacks.onUpdate(); }));

        grid.appendChild(_makeNumber('Resolution:', sv.resolution || 1080, 16, 16384, 2, function(v) { sv.resolution = v; callbacks.onUpdate(); }, 'resolution'));
        grid.appendChild(_makeNumber('Max Res (0=none):', sv.max_resolution || 0, 0, 16384, 2, function(v) { sv.max_resolution = v; callbacks.onUpdate(); }));
        grid.appendChild(_makeNumber('Seed:', sv.seed || 42, 0, 4294967295, 1, function(v) { sv.seed = v; callbacks.onUpdate(); }));

        grid.appendChild(_makeSelect('Color Correct:', ['lab', 'wavelet', 'wavelet_adaptive', 'hsv', 'adain', 'none'],
            sv.color_correction || 'lab', function(v) { sv.color_correction = v; callbacks.onUpdate(); }));

        grid.appendChild(_makeNumber('Input Noise:', sv.input_noise_scale || 0, 0, 1, 0.001, function(v) { sv.input_noise_scale = v; callbacks.onUpdate(); }));
        grid.appendChild(_makeNumber('Latent Noise:', sv.latent_noise_scale || 0, 0, 1, 0.001, function(v) { sv.latent_noise_scale = v; callbacks.onUpdate(); }));
        grid.appendChild(_makeNumber('Blocks to Swap:', sv.blocks_to_swap || 0, 0, 36, 1, function(v) { sv.blocks_to_swap = v; callbacks.onUpdate(); }));

        grid.appendChild(_makeSelect('Attention:', ['sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', 'sageattn_3'],
            sv.attention_mode || 'sdpa', function(v) { sv.attention_mode = v; callbacks.onUpdate(); }));

        grid.appendChild(_makeSelect('DiT Offload:', ['none', 'cpu'], sv.offload_device || 'cpu', function(v) { sv.offload_device = v; callbacks.onUpdate(); }));
        grid.appendChild(_makeCheckbox('Encode Tiled', sv.encode_tiled || false, function(v) { sv.encode_tiled = v; callbacks.onUpdate(); reRender(); }));
        if (sv.encode_tiled) {
            grid.appendChild(_makeNumber('Enc Tile Size:', sv.encode_tile_size || 1024, 64, 4096, 32, function(v) { sv.encode_tile_size = v; callbacks.onUpdate(); }));
            grid.appendChild(_makeNumber('Enc Overlap:', sv.encode_tile_overlap || 128, 0, 1024, 32, function(v) { sv.encode_tile_overlap = v; callbacks.onUpdate(); }));
        }
        grid.appendChild(_makeCheckbox('Decode Tiled', sv.decode_tiled || false, function(v) { sv.decode_tiled = v; callbacks.onUpdate(); reRender(); }));
        if (sv.decode_tiled) {
            grid.appendChild(_makeNumber('Dec Tile Size:', sv.decode_tile_size || 1024, 64, 4096, 32, function(v) { sv.decode_tile_size = v; callbacks.onUpdate(); }));
            grid.appendChild(_makeNumber('Dec Overlap:', sv.decode_tile_overlap || 128, 0, 1024, 32, function(v) { sv.decode_tile_overlap = v; callbacks.onUpdate(); }));
        }
    }

    // --- Florence2 Hi-Res Fix fields ---
    if (showFlorence2) {
        if (!ucfg.florence2) ucfg.florence2 = _defaultFlorence2Options();
        var f2 = ucfg.florence2;

        // Group A: Florence2 detection
        var f2Models = [
            'microsoft/Florence-2-base',
            'microsoft/Florence-2-base-ft',
            'microsoft/Florence-2-large',
            'microsoft/Florence-2-large-ft'
        ];
        grid.appendChild(_makeSelect('Florence2 Model:', f2Models, f2.model || 'microsoft/Florence-2-base', function(v) {
            f2.model = v; callbacks.onUpdate();
        }));

        grid.appendChild(_makeInput('Detect What?', f2.text_input || 'face', 'face, head, hands, eyes...', function(v) {
            f2.text_input = v; callbacks.onUpdate();
        }, 'text_input'));

        grid.appendChild(_makeInput('Mask Select:', f2.output_mask_select || '', '(empty = all)', function(v) {
            f2.output_mask_select = v; callbacks.onUpdate();
        }));

        grid.appendChild(_makeNumber('Max New Tokens:', f2.max_new_tokens != null ? f2.max_new_tokens : 1024, 64, 2048, 32, function(v) {
            f2.max_new_tokens = v; callbacks.onUpdate();
        }));

        grid.appendChild(_makeNumber('Florence2 Input MP:', f2.florence2_input_mp != null ? f2.florence2_input_mp : 0.5, 0, 4.0, 0.01, function(v) {
            f2.florence2_input_mp = v; callbacks.onUpdate();
        }));

        // Group B: Crop & resize
        grid.appendChild(_makeNumber('Hi-Res Target MP:', f2.target_megapixels != null ? f2.target_megapixels : 1.0, 0.25, 4.0, 0.25, function(v) {
            f2.target_megapixels = v; callbacks.onUpdate();
        }));

        grid.appendChild(_makeNumber('Crop Padding (px):', f2.crop_padding != null ? f2.crop_padding : 64, 0, 256, 8, function(v) {
            f2.crop_padding = v; callbacks.onUpdate();
        }));

        // Group C: Mask shaping
        grid.appendChild(_makeNumber('Grow Mask (px):', f2.grow_expand != null ? f2.grow_expand : 32, -64, 256, 1, function(v) {
            f2.grow_expand = v; callbacks.onUpdate();
        }));

        ['left', 'top', 'right', 'bottom'].forEach(function(side) {
            var key = 'feather_' + side;
            var label = 'Feather ' + side.charAt(0).toUpperCase() + side.slice(1) + ' (px):';
            grid.appendChild(_makeNumber(label, f2[key] != null ? f2[key] : 128, 0, 256, 1, function(v) {
                f2[key] = v; callbacks.onUpdate();
            }));
        });

        // Group E: Model/LoRA source
        grid.appendChild(_makeSelect('Model/LoRA Source:', [
            { value: 'from_manifest', label: 'From manifest (per-image)' },
            { value: 'from_builder', label: 'From this Builder config' }
        ], f2.model_source || 'from_manifest', function(v) {
            f2.model_source = v; callbacks.onUpdate();
        }));

        // Group F: No-detection (single-option for forward-compat)
        grid.appendChild(_makeSelect('If No Detection:', [
            { value: 'skip', label: 'Skip + log' }
        ], f2.on_no_detection || 'skip', function(v) {
            f2.on_no_detection = v; callbacks.onUpdate();
        }));
    }

    card.appendChild(grid);

    // Iteration count display
    var countDisplay = document.createElement('div');
    countDisplay.style.cssText = 'color: #00cc00; font-family: monospace; font-size: 10px; margin-top: 4px;';
    var ratios = (ucfg.upscale_ratios || '1.5').split(',').map(function(s) { return s.trim(); }).filter(function(s) { return s; }).length;
    var denoises = (ucfg.hires_denoise || '0.3').split(',').map(function(s) { return s.trim(); }).filter(function(s) { return s; }).length;
    var models = Math.max(1, (ucfg.upscale_models || []).length);
    var combos = 1;
    if (showSeedVR2) combos = 1;
    else if (showFlorence2) combos = 1; // 1 output per input (uses single hires_denoise + target_megapixels)
    else if (showHires) combos *= ratios * denoises;
    if (showModel) combos *= models;
    var repeatCount = ucfg.repeat || 1;
    countDisplay.textContent = combos + ' upscale combo(s) per image' + (repeatCount > 1 ? ' \u00D7 ' + repeatCount + ' repeats' : '');
    card.appendChild(countDisplay);

    pipeBody.appendChild(card);
}


// ============================================================================
// MODAL RENDERING — Uses renderPipelineEditor for full pipeline UI
// ============================================================================

/**
 * Small DOM helper: <button> with id/text/style/handler
 */
function _upmButton(id, text, css, onclick) {
    var b = document.createElement('button');
    if (id) b.id = id;
    b.type = 'button';
    b.textContent = text;
    b.style.cssText = css;
    if (onclick) b.onclick = onclick;
    return b;
}

var _UPM_BTN = 'border: none; border-radius: 3px; padding: 3px 8px; font-size: 11px; cursor: pointer;';

/**
 * Number of images the current scope will process
 */
function _upscaleScopeCount() {
    var scopeEl = document.querySelector('input[name="upscale-scope"]:checked');
    var favs = activeData ? activeData.filter(function(d) { return d.favorited; }).length : 0;
    return (scopeEl && scopeEl.value === 'favorites') ? favs : 1;
}

/**
 * Recompute everything that depends on the working config: validation marks + list, preset badge and
 * buttons, Run label, output summary. Safe to call at any time (finds elements by id every call).
 */
function _refreshUpscaleModalState() {
    var modal = document.getElementById('upscale-modal-content');
    if (!modal) return;
    var st = upscaleModalState;
    var cfg = st.currentConfig;
    if (!cfg) return;
    _persistUpscaleCfg();

    // --- validation ---
    var res = validateUpscaleConfig(cfg, { availableModels: st.availableModels });
    st.validation = res;
    Array.prototype.forEach.call(modal.querySelectorAll('.upm-invalid'), function(el) { el.classList.remove('upm-invalid'); });
    res.errors.forEach(function(e) {
        var m = /^p(\d+)\.s(\d+)$/.exec(e.path);
        if (!m) return;
        var card = modal.querySelector('[data-pipe="' + m[1] + '"][data-step="' + m[2] + '"]');
        var el = card ? card.querySelector('[data-field="' + e.field + '"]') : null;
        if (el) el.classList.add('upm-invalid');
    });
    var list = document.getElementById('upscale-validation');
    if (list) {
        list.textContent = '';
        var serverErrs = st.serverErrors || [];
        var rows = [];
        res.errors.forEach(function(e) { rows.push({ cls: 'upm-err', text: (e.path ? e.path + ': ' : '') + e.msg }); });
        res.warnings.forEach(function(w) { rows.push({ cls: 'upm-warn', text: (w.path ? w.path + ': ' : '') + w.msg }); });
        serverErrs.forEach(function(t) { rows.push({ cls: 'upm-err', text: 'Server: ' + t }); });
        rows.forEach(function(r) {
            var d = document.createElement('div');
            d.className = r.cls;
            d.textContent = r.text;
            list.appendChild(d);
        });
        list.style.display = rows.length ? 'block' : 'none';
    }

    // --- preset badge / buttons ---
    var loaded = st.loadedPresetName;
    var modified = !!(loaded && st.loadedPresetSnapshot && !configsEqual(cfg, st.loadedPresetSnapshot));
    var badge = document.getElementById('upscale-preset-state');
    if (badge) {
        badge.textContent = loaded ? ('Preset: ' + loaded + (modified ? ' (modified)' : '')) : 'Unsaved config';
        badge.style.color = modified || !loaded ? '#ffcc55' : '#8fdc8f';
    }
    var running = !!st.running;
    var upd = document.getElementById('upscale-preset-update');
    if (upd) upd.disabled = running || !(loaded && modified);
    var rev = document.getElementById('upscale-preset-revert');
    if (rev) rev.disabled = running || !(loaded && modified);
    var del = document.getElementById('upscale-preset-delete');
    if (del) del.disabled = running || !loaded;

    // --- Run button ---
    var startBtn = document.getElementById('upscale-start-btn');
    if (startBtn && !running) {
        startBtn.textContent = (loaded && !modified) ? 'Run' : 'Run once (unsaved)';
        startBtn.disabled = !res.ok;
        startBtn.style.opacity = res.ok ? '1' : '0.5';
        startBtn.title = res.ok ? '' : 'Fix the highlighted fields first';
    }

    // --- output summary ---
    var sum = document.getElementById('upscale-output-summary');
    if (sum) {
        var n = _upscaleScopeCount();
        var counts = countUpscaleOutputs(cfg);
        sum.textContent = n + ' image(s) × ' + counts.perImage + ' output(s) = ' + (n * counts.perImage) + ' new images';
    }
}

/**
 * Lock/unlock the editor and flip the action buttons for a running job
 */
function _applyUpscaleRunningState() {
    var running = !!upscaleModalState.running;
    var fs = document.getElementById('upscale-editor-fieldset');
    if (fs) fs.disabled = running;
    var cancelBtn = document.getElementById('upscale-cancel-job-btn');
    if (cancelBtn) cancelBtn.style.display = running ? '' : 'none';
    var startBtn = document.getElementById('upscale-start-btn');
    if (startBtn && running) {
        startBtn.disabled = true;
        startBtn.style.opacity = '0.5';
        startBtn.textContent = 'Upscaling...';
    }
    var progressDiv = document.getElementById('upscale-modal-progress');
    if (progressDiv && (running || upscaleModalState.lastStatus)) progressDiv.style.display = 'block';
    _refreshUpscaleModalState();
}

/**
 * Status line for a job status object
 */
function _upscaleStatusText(status) {
    var s = status.status;
    var base = (status.completed || 0) + '/' + (status.total || 0);
    if (s === 'running') {
        return 'Upscaling ' + base + (status.current_step ? ' · ' + status.current_step : '') + '...';
    }
    if (s === 'cancelling') return 'Cancelling after the current sampler step…';
    if (s === 'cancelled') return 'Cancelled: ' + base + ' images done, ' + (status.outputs || 0) + ' new images kept';
    if (s === 'complete') return 'Complete! ' + base + ' images upscaled, ' + (status.outputs || 0) + ' new images.';
    if (s === 'not_found') return 'This job is no longer known to the server (it was restarted or the record expired).';
    return 'Error: ' + (status.error || 'Unknown error');
}

/**
 * Poll one job by id until it ends. Never holds DOM references: every tick looks the elements up by id,
 * so a re-render (preset load, reopen) cannot leave it talking to detached nodes.
 */
function _startUpscalePoll(jobId) {
    if (upscaleModalState.pollInterval) clearInterval(upscaleModalState.pollInterval);
    upscaleModalState.jobId = jobId;
    upscaleModalState.running = true;
    var tick = async function() {
        try {
            var resp = await fetch('/config_tester/upscale_status?job_id=' + encodeURIComponent(jobId));
            var status = await resp.json();
            if (upscaleModalState.jobId !== jobId) return;      // superseded
            upscaleModalState.lastStatus = status;
            var fill = document.getElementById('upscale-modal-progress-fill');
            var text = document.getElementById('upscale-modal-progress-text');
            var progressDiv = document.getElementById('upscale-modal-progress');
            if (progressDiv) progressDiv.style.display = 'block';
            if (fill && status.total > 0) fill.style.width = Math.round((status.completed / status.total) * 100) + '%';
            if (text) text.textContent = _upscaleStatusText(status);
            var terminal = ['complete', 'cancelled', 'error', 'not_found'].indexOf(status.status) >= 0;
            var cancelBtn = document.getElementById('upscale-cancel-job-btn');
            if (cancelBtn && status.status === 'cancelling') { cancelBtn.disabled = true; cancelBtn.textContent = 'Cancelling...'; }
            if (terminal) {
                clearInterval(upscaleModalState.pollInterval);
                upscaleModalState.pollInterval = null;
                upscaleModalState.jobId = null;
                upscaleModalState.running = false;
                if (cancelBtn) { cancelBtn.disabled = false; cancelBtn.textContent = 'Cancel job'; }
                _applyUpscaleRunningState();
            }
        } catch (e) { /* ignore a failed poll; the next tick retries */ }
    };
    upscaleModalState.pollInterval = setInterval(tick, 1500);
    tick();
}

/**
 * Render the upscale modal
 */
function renderUpscaleModal() {
    var overlay = document.getElementById('upscale-modal-overlay');
    if (!overlay) return;
    overlay.style.display = 'flex';

    var st = upscaleModalState;
    var favCount = activeData ? activeData.filter(function(d) { return d.favorited; }).length : 0;
    var cfg = st.currentConfig;

    var modal = document.getElementById('upscale-modal-content');
    if (!modal) return;
    modal.textContent = '';

    // Header
    var header = document.createElement('div');
    header.style.cssText = 'display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;';
    var title = document.createElement('h3');
    title.style.cssText = 'margin: 0; color: #fff; font-size: 16px;';
    title.textContent = 'Upscale Image';
    var closeX = _upmButton('upscale-header-close', '✕', 'background: none; border: none; color: #999; font-size: 18px; cursor: pointer;', closeUpscaleModal);
    closeX.title = 'Close (a running job keeps going)';
    header.appendChild(title);
    header.appendChild(closeX);
    modal.appendChild(header);

    // Everything the user can edit lives in one fieldset so a running job locks it in one go
    var fieldset = document.createElement('fieldset');
    fieldset.id = 'upscale-editor-fieldset';
    fieldset.style.cssText = 'border: none; margin: 0; padding: 0; min-width: 0;';
    modal.appendChild(fieldset);

    // Scope toggle
    var scopeDiv = document.createElement('div');
    scopeDiv.style.cssText = 'margin-bottom: 10px; padding: 8px; background: #252525; border-radius: 4px;';
    var radioThis = document.createElement('input');
    radioThis.type = 'radio'; radioThis.name = 'upscale-scope'; radioThis.value = 'single';
    radioThis.checked = true; radioThis.id = 'upscale-scope-single';
    var labelThis = document.createElement('label');
    labelThis.htmlFor = 'upscale-scope-single';
    labelThis.style.cssText = 'color: #ccc; font-size: 12px; cursor: pointer; margin-right: 16px;';
    labelThis.textContent = ' Upscale This Image';
    var radioAll = document.createElement('input');
    radioAll.type = 'radio'; radioAll.name = 'upscale-scope'; radioAll.value = 'favorites';
    radioAll.id = 'upscale-scope-favorites';
    var labelAll = document.createElement('label');
    labelAll.htmlFor = 'upscale-scope-favorites';
    labelAll.style.cssText = 'color: #ccc; font-size: 12px; cursor: pointer;';
    labelAll.textContent = ' Upscale All Favorited (' + favCount + ' images)';
    radioThis.onchange = radioAll.onchange = _refreshUpscaleModalState;
    scopeDiv.appendChild(radioThis); scopeDiv.appendChild(labelThis);
    scopeDiv.appendChild(radioAll); scopeDiv.appendChild(labelAll);
    fieldset.appendChild(scopeDiv);

    // === PRESET BAR ===
    var presetDiv = document.createElement('div');
    presetDiv.id = 'upscale-preset-bar';
    presetDiv.style.cssText = 'margin-bottom: 10px; padding: 8px; background: #1e1e1e; border-radius: 4px; border: 1px solid #333;';
    var presetRow = document.createElement('div');
    presetRow.style.cssText = 'display: flex; gap: 4px; align-items: center; flex-wrap: wrap;';
    var presetSelect = document.createElement('select');
    presetSelect.id = 'upscale-preset-select';
    presetSelect.style.cssText = 'flex: 1; min-width: 120px; background: #1a1a1a; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 3px 6px; font-size: 11px;';
    var defaultOpt = document.createElement('option');
    defaultOpt.value = ''; defaultOpt.textContent = '-- Presets --';
    presetSelect.appendChild(defaultOpt);
    st.presets.forEach(function(p, i) {
        var opt = document.createElement('option');
        opt.value = String(i); opt.textContent = p.name;
        if (p.name === st.loadedPresetName) opt.selected = true;
        presetSelect.appendChild(opt);
    });
    presetSelect.onchange = function() {
        var idx = parseInt(presetSelect.value, 10);
        if (!isNaN(idx)) loadUpscalePreset(idx);
    };
    var badge = document.createElement('span');
    badge.id = 'upscale-preset-state';
    badge.style.cssText = 'font-size: 11px; font-weight: bold; white-space: nowrap;';
    presetRow.appendChild(presetSelect);
    presetRow.appendChild(badge);
    presetDiv.appendChild(presetRow);

    var presetBtns = document.createElement('div');
    presetBtns.style.cssText = 'display: flex; gap: 4px; margin-top: 6px; flex-wrap: wrap;';
    var updBtn = _upmButton('upscale-preset-update', 'Update preset', _UPM_BTN + 'background: var(--accent); color: #000; font-weight: bold;', function() {
        if (st.loadedPresetName) saveUpscalePreset(st.loadedPresetName);
    });
    var saveAsBtn = _upmButton('upscale-preset-saveas', 'Save as…', _UPM_BTN + 'background: #555; color: #fff;', function() {
        var row = document.getElementById('upscale-saveas-row');
        if (row) {
            row.style.display = row.style.display === 'none' ? 'flex' : 'none';
            var inp = document.getElementById('upscale-preset-name');
            if (inp && row.style.display !== 'none') inp.focus();
        }
    });
    var revBtn = _upmButton('upscale-preset-revert', 'Revert', _UPM_BTN + 'background: #555; color: #fff;', revertUpscaleConfig);
    var delBtn = _upmButton('upscale-preset-delete', 'Delete', _UPM_BTN + 'background: var(--danger); color: #fff;', function() {
        if (st.loadedPresetName && confirm('Delete preset "' + st.loadedPresetName + '"?')) deleteUpscalePreset(st.loadedPresetName);
    });
    presetBtns.appendChild(updBtn); presetBtns.appendChild(saveAsBtn);
    presetBtns.appendChild(revBtn); presetBtns.appendChild(delBtn);
    presetDiv.appendChild(presetBtns);

    var saveAsRow = document.createElement('div');
    saveAsRow.id = 'upscale-saveas-row';
    saveAsRow.style.cssText = 'display: none; gap: 4px; margin-top: 6px; align-items: center;';
    var nameInp = document.createElement('input');
    nameInp.type = 'text'; nameInp.id = 'upscale-preset-name'; nameInp.placeholder = 'New preset name';
    nameInp.style.cssText = 'flex: 1; background: #1a1a1a; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 3px 6px; font-size: 11px;';
    var doSave = function() {
        var n = nameInp.value.trim();
        if (!n) return;
        var exists = st.presets.some(function(p) { return p.name === n; });
        if (exists && n !== st.loadedPresetName && !confirm('A preset named "' + n + '" exists. Replace it?')) return;
        saveUpscalePreset(n);
    };
    nameInp.onkeydown = function(e) { if (e.key === 'Enter') doSave(); };
    var okBtn = _upmButton('upscale-preset-saveas-ok', 'OK', _UPM_BTN + 'background: var(--accent); color: #000; font-weight: bold;', doSave);
    saveAsRow.appendChild(nameInp); saveAsRow.appendChild(okBtn);
    presetDiv.appendChild(saveAsRow);
    fieldset.appendChild(presetDiv);

    // === FULL PIPELINE EDITOR ===
    var configDiv = document.createElement('div');
    configDiv.id = 'upscale-config-box';
    configDiv.style.cssText = 'margin-bottom: 10px; padding: 8px; background: #252525; border-radius: 4px; max-height: 55vh; overflow-y: auto;';

    var configTitle = document.createElement('div');
    configTitle.style.cssText = 'font-size: 12px; color: #fff; font-weight: bold; margin-bottom: 8px;';
    configTitle.textContent = 'Upscale Pipeline Configuration';
    configDiv.appendChild(configTitle);

    // Ensure config exists
    if (!cfg) { cfg = _defaultUpscaleConfig(); st.currentConfig = cfg; }

    // Pipeline editor container (separate from configDiv title so reRender only clears this)
    var editorContainer = document.createElement('div');
    editorContainer.id = 'upscale-editor';
    configDiv.appendChild(editorContainer);

    var editorCallbacks = {
        onUpdate: function() { st.serverErrors = []; _refreshUpscaleModalState(); },
        onRender: function() { _refreshUpscaleModalState(); }
    };
    renderPipelineEditor(editorContainer, cfg, st.availableModels, editorCallbacks);
    fieldset.appendChild(configDiv);

    // Validation list + what this run will produce
    var valDiv = document.createElement('div');
    valDiv.id = 'upscale-validation';
    valDiv.style.display = 'none';
    modal.appendChild(valDiv);

    var summary = document.createElement('div');
    summary.id = 'upscale-output-summary';
    summary.style.cssText = 'font-size: 12px; color: #00cc00; font-family: monospace; margin-bottom: 8px;';
    modal.appendChild(summary);

    var note = document.createElement('div');
    note.style.cssText = 'font-size: 10px; color: #777; margin-bottom: 8px;';
    var packOn = fullManifest && fullManifest.meta && fullManifest.meta.session_settings && fullManifest.meta.session_settings.pack_nodes_metadata;
    note.textContent = 'Saved with the session’s image format. ComfyUI workflow embedding follows the session setting' + (packOn ? ' (on).' : ' (off).');
    modal.appendChild(note);

    // Progress area (hidden until upscale starts)
    var progressDiv = document.createElement('div');
    progressDiv.id = 'upscale-modal-progress';
    progressDiv.style.cssText = 'margin-bottom: 10px; display: none;';
    var progressBar = document.createElement('div');
    progressBar.style.cssText = 'height: 6px; background: #333; border-radius: 3px; overflow: hidden;';
    var progressFill = document.createElement('div');
    progressFill.id = 'upscale-modal-progress-fill';
    progressFill.style.cssText = 'height: 100%; background: var(--accent); width: 0%; transition: width 0.3s;';
    progressBar.appendChild(progressFill);
    progressDiv.appendChild(progressBar);
    var progressText = document.createElement('div');
    progressText.id = 'upscale-modal-progress-text';
    progressText.style.cssText = 'font-size: 11px; color: #999; margin-top: 4px;';
    progressDiv.appendChild(progressText);
    modal.appendChild(progressDiv);
    if (st.lastStatus) {
        progressText.textContent = _upscaleStatusText(st.lastStatus);
        if (st.lastStatus.total > 0) progressFill.style.width = Math.round((st.lastStatus.completed / st.lastStatus.total) * 100) + '%';
    }

    // Action buttons: Cancel job (only while running), Close, Run
    var actionsDiv = document.createElement('div');
    actionsDiv.id = 'upscale-actions';
    actionsDiv.style.cssText = 'display: flex; gap: 8px; justify-content: flex-end; flex-wrap: wrap;';
    var cancelJobBtn = _upmButton('upscale-cancel-job-btn', 'Cancel job',
        'background: #aa3333; color: #fff; border: none; border-radius: 4px; padding: 6px 16px; font-size: 12px; cursor: pointer; display: none;',
        function() {
            if (!st.jobId) return;
            cancelJobBtn.disabled = true;
            cancelJobBtn.textContent = 'Cancelling...';
            fetch('/config_tester/cancel_upscale', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_id: st.jobId })
            }).catch(function() { /* the poll shows the truth */ });
        });
    var closeBtn = _upmButton('upscale-close-btn', 'Close',
        'background: #444; color: #fff; border: none; border-radius: 4px; padding: 6px 16px; font-size: 12px; cursor: pointer;',
        closeUpscaleModal);
    var startBtn = _upmButton('upscale-start-btn', 'Run',
        'background: var(--accent); color: #000; border: none; border-radius: 4px; padding: 6px 16px; font-size: 12px; cursor: pointer; font-weight: bold;',
        startUpscaleFromModal);
    actionsDiv.appendChild(cancelJobBtn);
    actionsDiv.appendChild(closeBtn);
    actionsDiv.appendChild(startBtn);
    modal.appendChild(actionsDiv);

    _applyUpscaleRunningState();
    _refreshUpscaleModalState();
}

/**
 * Start the upscale job from the modal (sends the working config as it is; nothing is saved to a preset)
 */
async function startUpscaleFromModal() {
    var st = upscaleModalState;
    var cfg = st.currentConfig;
    if (!cfg || st.running) return;

    var check = validateUpscaleConfig(cfg, { availableModels: st.availableModels });
    if (!check.ok) { _refreshUpscaleModalState(); return; }

    var scopeEl = document.querySelector('input[name="upscale-scope"]:checked');
    var scope = scopeEl ? scopeEl.value : 'single';
    var allFavorited = scope === 'favorites';
    var imageIds = allFavorited ? [] : [st.imageId];

    var startBtn = document.getElementById('upscale-start-btn');
    if (startBtn) { startBtn.disabled = true; startBtn.style.opacity = '0.5'; startBtn.textContent = 'Starting...'; }

    try {
        var resp = await fetch('/config_tester/upscale_images', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_name: _upscaleSessionName(),
                image_ids: imageIds,
                upscale_config: cfg,
                all_favorited: allFavorited,
            })
        });
        var result = await resp.json();
        if (!resp.ok || result.error) {
            // 400 {errors:[...]} = the server's own validation; 409 = a job already runs for this session
            st.serverErrors = result.errors && result.errors.length ? result.errors : [result.error || ('HTTP ' + resp.status)];
            _refreshUpscaleModalState();
            return;
        }
        st.serverErrors = [];
        st.lastStatus = null;
        var text = document.getElementById('upscale-modal-progress-text');
        if (text) text.textContent = 'Starting...';
        var fill = document.getElementById('upscale-modal-progress-fill');
        if (fill) fill.style.width = '0%';
        _startUpscalePoll(result.job_id);
        _applyUpscaleRunningState();
    } catch (e) {
        st.serverErrors = ['Failed to start upscale: ' + e];
        _refreshUpscaleModalState();
    }
}
