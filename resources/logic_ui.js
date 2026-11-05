/**
 * UI COMPONENTS - COMPACT LAYOUT WITH FILTERS POPUP
 * Fixed: Index tag z-index, time tag position
 * Added: Favorites feature, Filters popup, Prompt/Size/Seed filters
 */

// --- LABEL MODE FUNCTIONS ---

/**
 * Compute which values are global (same on every card) across activeData.
 * Called once when label mode is toggled on or when data changes while labels are active.
 */
function computeLabelGlobalValues() {
    if (!activeData || activeData.length === 0) {
        labelGlobalValues = null;
        return;
    }

    const fields = ['model', 'lora', 'sampler', 'scheduler', 'cfg', 'steps', 'seed', 'denoise', 'positive',
                     'upscale', 'upscaleMode', 'upscaleModel', 'upscaleRatio', 'upscaleDenoise',
                     'upscaleResizeMethod', 'upscaleHiresSteps', 'upscaleTiling',
                     'hiresPromptBehavior', 'hiresPromptText'];
    const valueSets = {};
    fields.forEach(f => valueSets[f] = new Set());

    for (const item of activeData) {
        valueSets.model.add(item.model || meta.model || "Default");
        valueSets.lora.add(item.lora || "None");
        valueSets.sampler.add(item.sampler || "");
        valueSets.scheduler.add(item.scheduler || "");
        valueSets.cfg.add(String(item.cfg));
        valueSets.steps.add(String(item.steps));
        valueSets.seed.add(String(item.seed));
        valueSets.denoise.add(String(item.denoise));
        valueSets.positive.add(item.positive || meta.positive || "");
        // Upscale composite (for the "all" toggle)
        const upParts = [];
        if (item.upscale_mode) upParts.push(item.upscale_mode);
        if (item.upscale_model) upParts.push(item.upscale_model);
        if (item.upscale_ratio) upParts.push('x' + item.upscale_ratio);
        if (item.upscale_denoise !== undefined && item.upscale_denoise !== null) upParts.push('dn:' + item.upscale_denoise);
        valueSets.upscale.add(upParts.length > 0 ? upParts.join(' ') : '');
        // Individual upscale fields (stringify arrays for set comparison)
        const toStr = (v) => v === undefined || v === null ? '' : (Array.isArray(v) ? JSON.stringify(v) : String(v));
        valueSets.upscaleMode.add(toStr(item.upscale_mode));
        valueSets.upscaleModel.add(toStr(item.upscale_model));
        valueSets.upscaleRatio.add(toStr(item.upscale_ratio));
        valueSets.upscaleDenoise.add(toStr(item.upscale_denoise));
        valueSets.upscaleResizeMethod.add(toStr(item.upscale_resize_method));
        valueSets.upscaleHiresSteps.add(toStr(item.upscale_hires_steps));
        // Tiling composite: combine tiled_vae + tiled_sampling info
        const tileParts = [];
        if (item.upscale_tiled_vae) tileParts.push('VAE:' + toStr(item.upscale_tile_size));
        if (item.upscale_tiled_sampling) tileParts.push('Sample:' + toStr(item.upscale_tile_w) + 'x' + toStr(item.upscale_tile_h));
        valueSets.upscaleTiling.add(tileParts.length > 0 ? tileParts.join(' ') : '');
        valueSets.hiresPromptBehavior.add(toStr(item.hires_prompt_behavior));
        valueSets.hiresPromptText.add(toStr(item.hires_prompt_text));
    }

    // A value is "global" if only 1 unique value exists for that field
    labelGlobalValues = {};
    fields.forEach(f => {
        labelGlobalValues[f] = valueSets[f].size === 1;
    });

    // For lora unique-only: also track which individual loras appear on every card.
    // A name is only "global" when it is on every card AND with a single strength string:
    // a strength sweep of one LoRA (x:1.00:1.00 vs x:0.60:0.60) must keep its tag.
    if (labelMode.fields.loraUniqueOnly) {
        const loraCountMap = {};
        const loraPartMap = {};
        let totalCards = activeData.length;
        for (const item of activeData) {
            const loraStr = item.lora || "None";
            if (loraStr === "None") continue;
            const parts = loraStr.split(" + ");
            const seen = new Set();
            for (const part of parts) {
                const name = part.split(":")[0];
                if (!seen.has(name)) {
                    seen.add(name);
                    loraCountMap[name] = (loraCountMap[name] || 0) + 1;
                }
                if (!loraPartMap[name]) loraPartMap[name] = new Set();
                loraPartMap[name].add(part.trim());
            }
        }
        labelGlobalValues._loraGlobalNames = new Set();
        for (const [name, count] of Object.entries(loraCountMap)) {
            if (count === totalCards && loraPartMap[name].size === 1) {
                labelGlobalValues._loraGlobalNames.add(name);
            }
        }
    }
}

/**
 * Build label overlay HTML for a single card data item.
 */
function buildLabelOverlay(d) {
    if (!labelMode.enabled) return '';

    const uniqueOnly = labelMode.fields.loraUniqueOnly;
    const tags = [];

    if (labelMode.fields.model) {
        const val = d.model || meta.model || "Default";
        if (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.model) {
            const short = val.replace(/\\/g, '/').split('/').pop().replace(/\.[^.]+$/, '');
            tags.push(`<span class="label-tag label-model" title="${val}">${short}</span>`);
        }
    }

    if (labelMode.fields.lora) {
        const loraStr = d.lora || "None";
        if (loraStr !== "None") {
            const parts = loraStr.split(" + ");
            for (const part of parts) {
                const components = part.split(":");
                const name = components[0];
                const shortName = name.replace(/\\/g, '/').split('/').pop().replace(/\.[^.]+$/, '');
                // Skip loras that are on every card when unique-only is on
                if (uniqueOnly && labelGlobalValues && labelGlobalValues._loraGlobalNames && labelGlobalValues._loraGlobalNames.has(name)) continue;
                // Show model_str:clip_str like in config_json (e.g. "loraName:1.0:0.8")
                const strength = components.length >= 3 ? `:${components[1]}:${components[2]}` :
                                 components.length === 2 ? `:${components[1]}` : '';
                tags.push(`<span class="label-tag label-lora" title="${part}">${shortName}${strength}</span>`);
            }
        }
    }

    if (labelMode.fields.prompt) {
        const val = d.positive || meta.positive || "";
        if (val && (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.positive)) {
            const short = val.length > 40 ? val.substring(0, 38) + '...' : val;
            tags.push(`<span class="label-tag label-prompt" title="${val}">${short}</span>`);
        }
    }

    if (labelMode.fields.sampler) {
        const val = d.sampler || "";
        if (val && (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.sampler)) {
            tags.push(`<span class="label-tag label-sampler">${val}</span>`);
        }
    }

    if (labelMode.fields.scheduler) {
        const val = d.scheduler || "";
        if (val && (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.scheduler)) {
            tags.push(`<span class="label-tag label-scheduler">${val}</span>`);
        }
    }

    if (labelMode.fields.cfg) {
        const val = d.cfg;
        if (val !== undefined && (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.cfg)) {
            tags.push(`<span class="label-tag label-cfg">CFG:${val}</span>`);
        }
    }

    if (labelMode.fields.steps) {
        const val = d.steps;
        if (val !== undefined && (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.steps)) {
            tags.push(`<span class="label-tag label-steps">Steps:${val}</span>`);
        }
    }

    if (labelMode.fields.seed) {
        const val = d.seed;
        if (val !== undefined && (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.seed)) {
            tags.push(`<span class="label-tag label-seed">Seed:${val}</span>`);
        }
    }

    if (labelMode.fields.denoise) {
        const val = d.denoise;
        if (val !== undefined && (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.denoise)) {
            tags.push(`<span class="label-tag label-denoise">Dn:${val}</span>`);
        }
    }

    // --- Upscale labels (each on its own line via class) ---
    const anyUpscaleField = labelMode.fields.upscale || labelMode.fields.upscaleMode || labelMode.fields.upscaleModel ||
        labelMode.fields.upscaleRatio || labelMode.fields.upscaleDenoise || labelMode.fields.upscaleResizeMethod ||
        labelMode.fields.upscaleHiresSteps || labelMode.fields.upscaleTiling;

    if (anyUpscaleField && d.upscaled) {
        // Helper: format a value that may be scalar or array (stacking mode) into a display string
        const fmtVal = (v, prefix, transform) => {
            if (v === undefined || v === null) return null;
            const fn = transform || (x => String(x));
            if (Array.isArray(v)) return prefix + v.map(fn).join(', ');
            return prefix + fn(v);
        };
        const shortModelName = (m) => m ? m.replace(/\\/g, '/').split('/').pop().replace(/\.[^.]+$/, '') : '';
        const isStacked = d.upscale_stacked;

        // "Upscale (all)" — combined summary tag
        if (labelMode.fields.upscale) {
            const parts = [];
            if (d.upscale_mode) {
                const modes = Array.isArray(d.upscale_mode) ? d.upscale_mode.map(m => m.replace(/_/g, ' ')).join(', ') : d.upscale_mode.replace(/_/g, ' ');
                parts.push(modes);
            }
            if (d.upscale_model) {
                const models = Array.isArray(d.upscale_model) ? d.upscale_model.map(shortModelName).join(', ') : shortModelName(d.upscale_model);
                parts.push(models);
            }
            if (d.upscale_ratio) {
                const ratios = Array.isArray(d.upscale_ratio) ? d.upscale_ratio.map(r => 'x' + r).join(', ') : 'x' + d.upscale_ratio;
                parts.push(ratios);
            }
            const label = parts.length > 0 ? (isStacked ? '⛓ ' : '') + parts.join(' | ') : 'Upscaled';
            if (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.upscale) {
                tags.push(`<span class="label-tag label-upscale" title="${label}">${label}</span>`);
            }
        }
        // Individual upscale fields — each on its own line
        if (labelMode.fields.upscaleMode && d.upscale_mode) {
            if (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.upscaleMode) {
                const val = fmtVal(d.upscale_mode, 'Mode: ', m => m.replace(/_/g, ' '));
                tags.push(`<span class="label-tag label-upscale">${val}</span>`);
            }
        }
        if (labelMode.fields.upscaleModel && d.upscale_model) {
            if (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.upscaleModel) {
                const raw = d.upscale_model;
                const val = fmtVal(raw, 'Model: ', shortModelName);
                const full = Array.isArray(raw) ? raw.join(', ') : raw;
                tags.push(`<span class="label-tag label-upscale" title="${full}">${val}</span>`);
            }
        }
        if (labelMode.fields.upscaleRatio && d.upscale_ratio !== undefined) {
            if (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.upscaleRatio) {
                const val = fmtVal(d.upscale_ratio, 'Ratio: ', r => 'x' + r);
                tags.push(`<span class="label-tag label-upscale">${val}</span>`);
            }
        }
        if (labelMode.fields.upscaleDenoise && d.upscale_denoise !== undefined) {
            if (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.upscaleDenoise) {
                const val = fmtVal(d.upscale_denoise, 'Denoise: ');
                tags.push(`<span class="label-tag label-upscale">${val}</span>`);
            }
        }
        if (labelMode.fields.upscaleResizeMethod && d.upscale_resize_method) {
            if (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.upscaleResizeMethod) {
                const val = fmtVal(d.upscale_resize_method, 'Resize: ');
                tags.push(`<span class="label-tag label-upscale">${val}</span>`);
            }
        }
        if (labelMode.fields.upscaleHiresSteps && d.upscale_hires_steps) {
            if (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.upscaleHiresSteps) {
                const val = fmtVal(d.upscale_hires_steps, 'HiRes Steps: ');
                tags.push(`<span class="label-tag label-upscale">${val}</span>`);
            }
        }
        if (labelMode.fields.upscaleTiling) {
            const tileParts = [];
            if (d.upscale_tiled_vae) {
                const ts = d.upscale_tile_size;
                tileParts.push('VAE Tile: ' + (Array.isArray(ts) ? ts.join(', ') : (ts || 512)));
            }
            if (d.upscale_tiled_sampling) {
                const tw = d.upscale_tile_w;
                const th = d.upscale_tile_h;
                if (Array.isArray(tw)) {
                    tileParts.push('Sample Tile: ' + tw.map((w, i) => w + 'x' + (th[i] || w)).join(', '));
                } else {
                    tileParts.push('Sample Tile: ' + (tw || 512) + 'x' + (th || 512));
                }
            }
            if (tileParts.length > 0) {
                if (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.upscaleTiling) {
                    for (const tp of tileParts) {
                        tags.push(`<span class="label-tag label-upscale">${tp}</span>`);
                    }
                }
            }
        }
        if (labelMode.fields.hiresPromptBehavior && d.hires_prompt_behavior) {
            if (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.hiresPromptBehavior) {
                const val = fmtVal(d.hires_prompt_behavior, 'HiRes Prompt: ', b => b.replace(/_/g, ' '));
                tags.push(`<span class="label-tag label-hiresPrompt">${val}</span>`);
            }
        }
        if (labelMode.fields.hiresPromptText && d.hires_prompt_text) {
            if (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.hiresPromptText) {
                const text = d.hires_prompt_text;
                const short = text.length > 30 ? text.substring(0, 28) + '...' : text;
                tags.push(`<span class="label-tag label-hiresPrompt" title="${text}">${short}</span>`);
            }
        }
    } else if (anyUpscaleField && !d.upscaled) {
        // Show "No Upscale" only when there's a mix of upscaled and non-upscaled images
        if (!uniqueOnly || !labelGlobalValues || !labelGlobalValues.upscale) {
            const hasAnyUpscaled = activeData && activeData.some(i => i.upscaled);
            if (hasAnyUpscaled) {
                tags.push(`<span class="label-tag label-upscale" style="opacity:0.5;">No Upscale</span>`);
            }
        }
    }

    if (labelMode.fields.rating && typeof getItemRating === 'function') {
        const rt = getItemRating(d.id);
        if (rt && rt.votes > 0) {
            tags.push(`<span class="label-tag label-rating" title="Pairwise rating (Bradley-Terry) and number of votes">★${Math.round(rt.rating)} · ${rt.votes}</span>`);
        }
    }

    if (labelMode.fields.metrics && d.metrics && typeof d.metrics === 'object') {
        const sig3 = (v) => {
            const n = Number(v);
            if (!isFinite(n)) return null;
            return String(parseFloat(n.toPrecision(3)));
        };
        const mkeys = (typeof getMetricKeys === 'function') ? getMetricKeys() : Object.keys(d.metrics);
        for (const k of mkeys) {
            if (!(k in d.metrics)) continue;
            const v = sig3(d.metrics[k]);
            if (v === null) continue;
            const short = METRIC_SHORT[k] || metricLabel(k).slice(0, 3);
            tags.push(`<span class="label-tag label-metric" title="${metricLabel(k)}: ${d.metrics[k]}">${short}:${v}</span>`);
        }
    }

    if (labelMode.fields.wfParams && d.wf_params && typeof d.wf_params === 'object' && typeof wfParamKeys === 'function') {
        const wfKeys = wfParamKeys(activeData);
        const esc = (t) => String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        for (const row of wfKeys) {
            if (row.distinct <= 1) continue;          // constant axes are noise
            const v = wfParamValue(d, row.key);
            if (v === null) continue;
            tags.push(`<span class="label-tag label-wf" title="${esc(row.key)}: ${esc(v)}">${esc(wfShortLabel(row.key, wfKeys))}:${esc(wfDisplay(v))}</span>`);
        }
    }

    if (tags.length === 0) return '';
    return `<div class="label-overlay">${tags.join('')}</div>`;
}

/**
 * Toggle label mode on/off
 */
function toggleLabelMode(enabled) {
    labelMode.enabled = enabled;
    const fieldsContainer = document.getElementById('label-fields-container');
    if (fieldsContainer) fieldsContainer.style.display = enabled ? 'block' : 'none';

    if (enabled) {
        computeLabelGlobalValues();
    }
    saveLabelPreferences();
    refreshLabelOverlays();
}

/**
 * Update label field selections from checkboxes
 */
function updateLabelFields() {
    const fields = ['model', 'lora', 'prompt', 'sampler', 'scheduler', 'cfg', 'steps', 'seed', 'denoise', 'upscale',
                     'upscaleMode', 'upscaleModel', 'upscaleRatio', 'upscaleDenoise', 'upscaleResizeMethod', 'upscaleHiresSteps', 'upscaleTiling',
                     'hiresPromptBehavior', 'hiresPromptText', 'rating', 'metrics', 'wfParams'];
    fields.forEach(f => {
        const cb = document.getElementById(`label-field-${f}`);
        if (cb) labelMode.fields[f] = cb.checked;
    });
    const uniqueCb = document.getElementById('label-unique-only');
    if (uniqueCb) labelMode.fields.loraUniqueOnly = uniqueCb.checked;

    if (labelMode.enabled) {
        computeLabelGlobalValues();
    }
    saveLabelPreferences();
    refreshLabelOverlays();
}

/**
 * Update label font size from slider
 */
function updateLabelSize(size) {
    labelMode.labelSize = parseInt(size);
    const display = document.getElementById('label-size-value');
    if (display) display.textContent = size + 'px';
    // Apply size to all existing label tags via CSS custom property
    document.documentElement.style.setProperty('--label-font-size', size + 'px');
    saveLabelPreferences();
}

/**
 * Save label preferences to localStorage
 */
function saveLabelPreferences() {
    try {
        localStorage.setItem('ultimate_grid_labels', JSON.stringify(labelMode));
    } catch (e) { /* ignore */ }
}

/**
 * Load label preferences from localStorage and sync UI checkboxes
 */
function loadLabelPreferences() {
    try {
        const saved = localStorage.getItem('ultimate_grid_labels');
        if (saved) {
            const parsed = JSON.parse(saved);
            labelMode.enabled = parsed.enabled || false;
            if (parsed.labelSize) labelMode.labelSize = parsed.labelSize;
            if (parsed.fields) {
                Object.assign(labelMode.fields, parsed.fields);
            }
        }
    } catch (e) { /* ignore */ }

    // Sync UI checkboxes
    const toggle = document.getElementById('label-mode-toggle');
    if (toggle) toggle.checked = labelMode.enabled;

    // Sync label size slider
    const sizeSlider = document.getElementById('label-size-slider');
    if (sizeSlider) sizeSlider.value = labelMode.labelSize;
    const sizeDisplay = document.getElementById('label-size-value');
    if (sizeDisplay) sizeDisplay.textContent = labelMode.labelSize + 'px';
    document.documentElement.style.setProperty('--label-font-size', labelMode.labelSize + 'px');

    const fieldsContainer = document.getElementById('label-fields-container');
    if (fieldsContainer) fieldsContainer.style.display = labelMode.enabled ? 'block' : 'none';

    const fields = ['model', 'lora', 'prompt', 'sampler', 'scheduler', 'cfg', 'steps', 'seed', 'denoise', 'upscale',
                     'upscaleMode', 'upscaleModel', 'upscaleRatio', 'upscaleDenoise', 'upscaleResizeMethod', 'upscaleHiresSteps', 'upscaleTiling',
                     'hiresPromptBehavior', 'hiresPromptText', 'rating', 'metrics', 'wfParams'];
    fields.forEach(f => {
        const cb = document.getElementById(`label-field-${f}`);
        if (cb) cb.checked = labelMode.fields[f] || false;
    });

    const uniqueCb = document.getElementById('label-unique-only');
    if (uniqueCb) uniqueCb.checked = labelMode.fields.loraUniqueOnly !== false;

    if (labelMode.enabled) {
        computeLabelGlobalValues();
    }
}

/**
 * Refresh label overlays on all visible cards (add/remove/update)
 */
function refreshLabelOverlays() {
    for (const [id, card] of nodeMap) {
        const d = card._dataItem;
        if (!d) continue;

        // Remove existing label overlay
        const existing = card.querySelector('.label-overlay');
        if (existing) existing.remove();

        // Add new one if enabled
        if (labelMode.enabled) {
            const html = buildLabelOverlay(d);
            if (html) {
                const wrapper = card.querySelector('.img-wrapper');
                if (wrapper) {
                    wrapper.insertAdjacentHTML('beforeend', html);
                }
            }
        }
    }
}

// Cache for filter buttons
let filterButtonCache = {};

// Toggle Filters Popup
function toggleFiltersPopup() {
    const popup = document.getElementById('filters-popup');
    const overlay = document.getElementById('filters-overlay');

    if (popup.style.display === 'none' || !popup.style.display) {
        popup.style.display = 'block';
        overlay.style.display = 'block';
        document.body.style.overflow = 'hidden'; // Prevent background scroll
    } else {
        popup.style.display = 'none';
        overlay.style.display = 'none';
        document.body.style.overflow = '';
    }
}


// Toggle Favorites Filter
function toggleFavoritesFilter() {
    topFilters.showFavorites = !topFilters.showFavorites;
    
    const btn = document.getElementById('top-filter-favorites');
    if (btn) {
        if (topFilters.showFavorites) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    }
    
    updateDataPipeline();
    console.log('[Top Filter] Favorites:', topFilters.showFavorites ? 'ON' : 'OFF');
}

// Toggle Non-Favorited Filter
function toggleNonFavoritedFilter() {
    topFilters.showNonFavorited = !topFilters.showNonFavorited;
    
    const btn = document.getElementById('top-filter-nonfavorited');
    if (btn) {
        if (topFilters.showNonFavorited) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    }
    
    updateDataPipeline();
    console.log('[Top Filter] Non-Favorited:', topFilters.showNonFavorited ? 'ON' : 'OFF');
}

// Toggle Rejected Filter
function toggleRejectedFilter() {
    topFilters.showRejected = !topFilters.showRejected;
    
    const btn = document.getElementById('top-filter-rejected');
    if (btn) {
        if (topFilters.showRejected) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    }
    
    updateDataPipeline();
    console.log('[Top Filter] Rejected:', topFilters.showRejected ? 'ON' : 'OFF');
}
// Settings Panel Toggle (unified cog + session)
function toggleCogMenu() {
    const panel = document.getElementById('cog-menu-dropdown');
    const overlay = document.getElementById('cog-menu-overlay');
    if (!panel) return;

    if (panel.style.display === 'none' || !panel.style.display) {
        panel.style.display = 'block';
        if (overlay) overlay.style.display = 'block';
        document.body.style.overflow = 'hidden';
    } else {
        panel.style.display = 'none';
        if (overlay) overlay.style.display = 'none';
        document.body.style.overflow = '';
    }
}

// Close popup on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        // Close Revise modal first (highest priority)
        const reviseModal = document.getElementById('modal');
        if (reviseModal && reviseModal.style.display !== 'none' && reviseModal.style.display !== '') {
            closeM();
            return;
        }

        // Close pairwise-review scoreboard
        const scoreboardModal = document.getElementById('scoreboard-modal');
        if (scoreboardModal) {
            scoreboardModal.remove();
            return;
        }

        // Close analytics results modal
        const analyticsModal = document.getElementById('analytics-results-modal');
        if (analyticsModal) {
            analyticsModal.remove();
            return;
        }

        // Close settings panel
        const cogPanel = document.getElementById('cog-menu-dropdown');
        if (cogPanel && cogPanel.style.display !== 'none') {
            toggleCogMenu();
            return;
        }

        // Close filters popup
        const filtersPopup = document.getElementById('filters-popup');
        if (filtersPopup && filtersPopup.style.display !== 'none') {
            toggleFiltersPopup();
            return;
        }

        // Close the docked Stats panel
        if (typeof closeAnalyticsPanel === 'function' && closeAnalyticsPanel()) return;
    }
});

// Truncate long text with tooltip
function truncateText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// Initialize Filter Buttons (with caching)
function initFilters() {
    if (!activeData || activeData.length === 0) return;

    // Ensure all filter Sets exist (add new ones if missing)
    const filterKeys = ['model', 'sampler', 'scheduler', 'denoise', 'lora', 'positive', 'negative', 'size', 'seed', 'steps', 'cfg', 'upscaleMethod', 'mediaType', 'passType'];
    filterKeys.forEach(key => {
        if (!filters.hasOwnProperty(key) || !(filters[key] instanceof Set)) {
            filters[key] = new Set();
        }
    });

    filterKeys.forEach(key => {
        const unique = [...new Set(activeData.map(d => {
            if (key === 'model') return d.model || meta.model || "Default";
            if (key === 'positive') {
                const st = document.getElementById('toggle-triggers')?.checked !== false;
                return st ? (d.positive || meta.positive || "") : (d.config_positive || d.positive || meta.positive || "");
            }
            if (key === 'negative') {
                const st = document.getElementById('toggle-triggers')?.checked !== false;
                return st ? (d.negative || meta.negative || "") : (d.config_negative || d.negative || meta.negative || "");
            }
            if (key === 'size') return `${d.width}x${d.height}`;
            if (key === 'steps') return String(d.steps);
            if (key === 'cfg') return String(d.cfg);
            if (key === 'upscaleMethod') {
                if (!d.upscaled) return 'No Upscale';
                const mode = d.upscale_mode || '';
                const model = d.upscale_model;
                const shortModel = model ? String(model).replace(/\\/g, '/').split('/').pop().replace(/\.[^.]+$/, '') : '';
                return shortModel ? `${mode} + ${shortModel}` : mode || 'Upscaled';
            }
            if (key === 'mediaType') return d.media_type || 'image';
            if (key === 'passType') return d.preview ? 'Preview' : 'Full';
            return d[key];
        }))].sort();

        const container = document.getElementById('filter-' + key);
        if (!container) return;

        // The Pass row only matters once the session holds preview images
        if (key === 'passType') {
            const passRow = container.closest('.filter-row');
            if (passRow) passRow.style.display = unique.length > 1 ? '' : 'none';
        }

        const cacheKey = unique.join(',');
        if (filterButtonCache[key] === cacheKey) {
            return;
        }

        filterButtonCache[key] = cacheKey;
        container.innerHTML = '';

        unique.forEach(val => {
            const safeVal = String(val).replace(/[^a-zA-Z0-9]/g, '');
            const btnId = `btn-${key}-${safeVal}`;

            let b = document.createElement('button');
            b.id = btnId;
            b.className = `filter-btn active ${key}`;

            let label = val;
            let fullText = val;

            // Handle special formatting for different filter types
            if (key === 'lora') {
                if (val === "None") {
                    label = "None";
                } else if (val.includes(" + ")) {
                    label = "Stack";
                    fullText = val.replace(/ \+ /g, '\n');
                } else {
                    let clean = val.replace(/\\/g, '/').split('/').pop().split(':')[0];
                    // Keep the strengths so the buttons of a strength sweep differ (detail:0.60:0.60)
                    const strengths = val.includes(':') ? val.slice(val.indexOf(':')) : '';
                    label = truncateText(clean, 12) + strengths;
                    fullText = val;
                }
            } else if (key === 'model') {
                let clean = val.replace(/\\/g, '/').split('/').pop();
                label = truncateText(clean, 12);
                fullText = val;
            } else if (key === 'positive' || key === 'negative') {
                // Truncate prompts to 30 characters for button display
                label = truncateText(val, 30);
                fullText = val;
            } else if (key === 'seed') {
                // Display seeds in a more readable format
                label = String(val);
                fullText = val;
            }

            b.innerText = label;
            b.title = fullText;

            // Isolate filter: deselect all others, or re-select all if already isolated
            const isolateFilter = () => {
                // Check if this is the only active filter
                const isOnlyActive = filters[key].size === 1 && filters[key].has(val);

                if (isOnlyActive) {
                    // If it's the only one active, select all instead
                    unique.forEach(v => filters[key].add(v));
                    const allButtons = container.querySelectorAll('.filter-btn');
                    allButtons.forEach(btn => btn.classList.add('active'));
                } else {
                    // Clear all filters of this type, add only this one
                    filters[key].clear();
                    filters[key].add(val);
                    const allButtons = container.querySelectorAll('.filter-btn');
                    allButtons.forEach(btn => btn.classList.remove('active'));
                    b.classList.add('active');
                }

                updateDataPipeline();
            };

            b.onclick = (e) => {
                // Shift-click: Isolate this filter (deselect all others of this type)
                if (e.shiftKey) {
                    e.preventDefault();
                    isolateFilter();
                } else {
                    // Normal click: Toggle this filter
                    if (filters[key].has(val)) {
                        filters[key].delete(val);
                        b.classList.remove('active');
                    } else {
                        filters[key].add(val);
                        b.classList.add('active');
                    }

                    updateDataPipeline();
                }
            };

            // Double-click: Isolate this filter (same as shift-click)
            b.ondblclick = (e) => {
                e.preventDefault();
                isolateFilter();
            };

            filters[key].add(val);
            container.appendChild(b);
        });
    });
}

// Reset all button-filters to "all selected" and update the pipeline.
// Also clears searchFilters and logicFilters so the full dataset is visible.
function resetAllFilters() {
    if (!activeData || activeData.length === 0) return;

    const filterKeys = ['model', 'sampler', 'scheduler', 'denoise', 'lora', 'positive', 'negative', 'size', 'seed', 'steps', 'cfg', 'upscaleMethod', 'mediaType', 'passType'];

    // Re-populate every filter Set with all unique values from activeData
    filterKeys.forEach(key => {
        if (!filters[key]) filters[key] = new Set();
        filters[key].clear();
        activeData.forEach(d => {
            let val;
            if (key === 'model') val = d.model || meta.model || "Default";
            else if (key === 'positive') val = d.positive || meta.positive || "";
            else if (key === 'negative') val = d.negative || meta.negative || "";
            else if (key === 'size') val = `${d.width}x${d.height}`;
            else if (key === 'steps') val = String(d.steps);
            else if (key === 'cfg') val = String(d.cfg);
            else if (key === 'upscaleMethod') {
                if (!d.upscaled) { val = 'No Upscale'; }
                else {
                    const mode = d.upscale_mode || '';
                    const model = d.upscale_model;
                    const shortModel = model ? String(model).replace(/\\/g, '/').split('/').pop().replace(/\.[^.]+$/, '') : '';
                    val = shortModel ? `${mode} + ${shortModel}` : mode || 'Upscaled';
                }
            } else if (key === 'mediaType') val = d.media_type || 'image';
            else if (key === 'passType') val = d.preview ? 'Preview' : 'Full';
            else val = d[key];
            if (val !== undefined && val !== null) filters[key].add(val);
        });
    });

    // Mark all filter buttons as active
    const filterButtonContainers = document.querySelectorAll('.filter-buttons-container');
    filterButtonContainers.forEach(container => {
        container.querySelectorAll('.filter-btn').forEach(btn => btn.classList.add('active'));
    });

    // Clear search filters
    if (typeof searchFilters !== 'undefined') {
        searchFilters = [];
        if (typeof renderSearchFilters === 'function') renderSearchFilters();
    }

    // Clear logic filters
    if (typeof logicFilters !== 'undefined') {
        logicFilters = [];
        if (typeof renderLogicFilters === 'function') renderLogicFilters();
    }

    // Clear quick filters
    if (typeof quickFilters !== 'undefined') {
        quickFilters.length = 0;
        if (typeof renderQuickFilters === 'function') renderQuickFilters();
    }

    // Clear the filter button cache so initFilters rebuilds on next call
    if (typeof filterButtonCache !== 'undefined') {
        for (const key in filterButtonCache) delete filterButtonCache[key];
    }

    updateDataPipeline();
    console.log('[Reset Filters] All filters reset to default (show all)');
}

// Toggle trigger word visibility in prompts
function toggleTriggerWords(showTriggers) {
    // Rebuild filters with new prompt mode
    initFilters();
    updateDataPipeline();
    // If modal is open, refresh prompt display
    if (window.currentModalId) {
        openM(window.currentModalId);
    }
}


let pendingSaveItems = new Map();  // CHANGED: Set -> Map
let saveTimer = null;
const SAVE_BATCH_DELAY = 100;

/**
 * OPTIMIZED: Schedule a save that only sends changed items
 */
function scheduleBatchedSave() {
    if (saveTimer) {
        clearTimeout(saveTimer);
    }

    saveTimer = setTimeout(async () => {
        if (pendingSaveItems.size === 0) return;

        console.log(`[Save] 💾 Sending ${pendingSaveItems.size} changed items to server`);

        try {
            const sessionName = document.getElementById('session-input')?.value || "default";

            // Convert Map to array of items
            const changedItems = Array.from(pendingSaveItems.values());

            // CHANGED: Use new endpoint that only accepts changed items
            const response = await fetch('/config_tester/save_changes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_name: sessionName,
                    changed_items: changedItems  // Only changed items!
                })
            });

            if (!response.ok) {
                throw new Error(`Save failed: ${response.status}`);
            }

            console.log(`[Save] ✅ Successfully saved ${pendingSaveItems.size} changes`);
            pendingSaveItems.clear();

        } catch (error) {
            console.error('[Save] ❌ Batch save failed:', error);
            // Don't clear pendingSaveItems - will retry on next change
        }
    }, SAVE_BATCH_DELAY);
}
function markItemChanged(item) {
    if (!item || !item.id) return;

    // Store the full item object, not just the ID
    pendingSaveItems.set(item.id, item);
    scheduleBatchedSave();
}
// Lightweight JSON update (separate debounce from save)
let jsonUpdateTimer = null;
const JSON_UPDATE_DELAY = 500; // 500ms debounce

function scheduleJSONUpdate() {
    if (jsonUpdateTimer) {
        clearTimeout(jsonUpdateTimer);
    }

    jsonUpdateTimer = setTimeout(() => {
        updateJSONs(processedData);
    }, JSON_UPDATE_DELAY);
}

// Toggle Favorite

async function toggleFavorite(element) {
    // [OPTIMIZATION] Find parent card and grab data directly (O(1) access)
    // No getElementById, no array.find, no iteration.
    const card = element.closest('.card');
    if (!card || !card._dataItem) return;

    const item = card._dataItem;

    // Update Data
    item.favorited = !item.favorited;

    // Update UI - Scope querySelector to just this card for extra speed
    const favBtn = card.querySelector('.favorite-btn');

    if (favBtn) {
        favBtn.classList.toggle('favorited', item.favorited);
        favBtn.innerText = item.favorited ? '★' : '☆';

        // Simple animation
        favBtn.style.transform = 'scale(1.2)';
        setTimeout(() => favBtn.style.transform = 'scale(1)', 120);
    }

    markItemChanged(item);
    scheduleJSONUpdate();
}


/**
 * Show error alert popup with details
 */
function showSaveErrorAlert(title, message, technicalDetails = '') {
    // Remove any existing error alert
    const existingAlert = document.getElementById('save-error-alert');
    if (existingAlert) {
        existingAlert.remove();
    }

    // Create overlay
    const overlay = document.createElement('div');
    overlay.id = 'save-error-alert';
    overlay.className = 'error-alert-overlay';

    // Create popup
    const popup = document.createElement('div');
    popup.className = 'error-alert-popup';

    // Build details section if we have technical info
    const detailsHtml = technicalDetails ? `
        <details class="error-alert-details">
            <summary>Technical Details</summary>
            <pre>${technicalDetails}</pre>
        </details>
    ` : '';

    popup.innerHTML = `
        <div class="error-alert-header">
            <span class="error-alert-icon">⚠️</span>
            <h3>${title}</h3>
        </div>
        <p class="error-alert-message">${message}</p>
        ${detailsHtml}
        <div class="error-alert-actions">
            <button class="error-alert-button error-alert-button-primary" onclick="this.closest('.error-alert-overlay').remove()">
                OK
            </button>
        </div>
    `;

    overlay.appendChild(popup);
    document.body.appendChild(overlay);

    // Close on overlay click
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
            overlay.remove();
        }
    });

    // Close on Escape key
    const escHandler = (e) => {
        if (e.key === 'Escape') {
            overlay.remove();
            document.removeEventListener('keydown', escHandler);
        }
    };
    document.addEventListener('keydown', escHandler);

    // Auto-close after 12 seconds
    setTimeout(() => {
        if (overlay.parentNode) {
            overlay.style.opacity = '0';
            setTimeout(() => overlay.remove(), 200);
        }
    }, 12000);
}


function copyConfigsAsComfyNodes(id) {
    // console.log(param)


    const d = activeData.find(x => x.id === id);
    if (!d) {
        alert('Configuration data not found!');
        return;
    }

    try {
        const workflow = buildComfyNodesWorkflow(d);

        // Copy to clipboard logic (Standard)
        const jsonString = JSON.stringify(workflow, null, 2);

        const copyToClipboard = async (text) => {
            try {
                await navigator.clipboard.writeText(text);
                return true;
            } catch {
                const textarea = document.createElement('textarea');
                textarea.value = text;
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                document.body.appendChild(textarea);
                textarea.select();
                const success = document.execCommand('copy');
                document.body.removeChild(textarea);
                return success;
            }
        };

        copyToClipboard(jsonString).then((success) => {
            if (success) {
                alert('✅ ComfyUI workflow copied to clipboard!\n\nYou can now paste this into ComfyUI.');
                console.log('[ComfyUI] Workflow copied:', workflow);
            }
        }).catch(err => {
            console.error('[ComfyUI] Failed to copy:', err);
            alert('❌ Failed to copy to clipboard. JSON logged to console.');
            console.log('ComfyUI Workflow JSON:', jsonString);
        });

    } catch (error) {
        console.error('[ComfyUI] Error generating workflow:', error);
        alert('❌ Error generating workflow: ' + error.message);
    }
}


// Create card - FIXED UI LAYOUT WITH FAVORITE BUTTON
function createCard(d) {
    const totalIndex = idToIndexMap.get(d.id) || 0;
    const card = document.createElement('div');
    card.className = 'card';
    card.id = `card-${d.id}`;
    card.dataset.id = d.id;
    card._dataItem = d;
    // Check if favorited
    const isFavorited = d.favorited || false;
    const favClass = isFavorited ? 'favorited' : '';
    const favIcon = isFavorited ? '★' : '☆';

    // Calculate LoRA display. Keep the row to a SINGLE line regardless of
    // how many LoRAs are stacked — when cards are taller than the virtual
    // scroller's row stride, they overflow into adjacent grid rows and
    // break the 1-9 quick-favorite hotkey's row clustering. Full list is
    // still available via tooltip (title attribute).
    let loraLine = "";
    if (d.lora === "None") {
        loraLine = `<div class="stat"><b>LoRA:</b> <span style="opacity:0.3">-</span></div>`;
    } else if (d.lora.includes(" + ")) {
        const parts = d.lora.split(" + ");
        const count = parts.length;
        const firstName = String(parts[0]).replace(/\\/g, '/').split('/').pop().split(':')[0];
        const truncatedFirst = firstName.length > 14 ? firstName.substring(0, 12) + '…' : firstName;
        const fullList = d.lora.replace(/ \+ /g, '\n');
        loraLine = `<div class="stat" title="${fullList}"><b>LoRA:</b> <span style="color:var(--accent-lora)">${truncatedFirst} +${count - 1} more</span></div>`;
    } else {
        const rawName = String(d.lora);
        let fileName = rawName.replace(/\\/g, '/').split('/').pop().split(':')[0];
        if (fileName.length > 20) fileName = fileName.substring(0, 18) + '…';
        loraLine = `<div class="stat" title="${d.lora}"><b>LoRA:</b> <span>${fileName}</span></div>`;
    }

    let promptInfo = "";
    // Always show prompt info if available
    if (d.positive || meta.positive) {
        const promptText = d.positive || meta.positive || "";
        const shortPrompt = truncateText(promptText, 30);
        promptInfo = `<div class="stat" title="${promptText}"><b>Pos:</b> ${shortPrompt}</div>`;
    }

    const modelName = d.model || meta.model || "Default";
    const shortModel = modelName.replace(/\\/g, '/').split('/').pop();
    const finalModel = shortModel.length > 25 ? shortModel.substring(0, 22) + "..." : shortModel;

    // Calculate aspect ratio
    const aspectRatio = (d.width && d.height) ? (d.height / d.width) : 1;
    const paddingBottom = (aspectRatio * 100).toFixed(2);

    // Detect video vs image and prepare variants
    const isVideo = d.media_type === 'video';
    const mediaElement = isVideo
        ? `<video ondblclick="toggleFavorite(this)" data-src="${d.file}" muted loop playsinline preload="metadata" draggable="false"></video>`
        : `<img ondblclick="toggleFavorite(this)" data-src="${d.file}" alt="Image ${d.id}" draggable="false">`;
    // Workflow Sweep items: the Revise modal builds a standard config and would lose the workflow template
    // (they get a JOB button instead: openM shows the read-only Workflow job panel)
    const reviseBtn = (isVideo && d.model_type !== 'workflow') ? '' : `<button class="revise-btn" onclick="event.stopPropagation(); openM(${d.id})">${d.model_type === 'workflow' ? 'JOB' : 'REVISE'}</button>`;
    const upscaleBtn = isVideo ? '' : `<button class="upscale-btn" onclick="openUpscaleModal(${d.id})" title="Upscale this image">\u2B06</button>`;
    const videoBadge = isVideo ? '<div class="video-badge">\u25B6 VIDEO</div>' : '';
    const previewBadge = d.preview
        ? `<div class="preview-badge" title="Preview of ${d.full_width}x${d.full_height} @ ${d.full_steps} steps">PREVIEW \u00B7 ${d.steps} st</div>`
        : '';

    // FIXED LAYOUT: Star top-right, Revise below it, time bottom-right, index bottom-left
    card.innerHTML = `
        <div class="img-wrapper" style="padding-bottom: ${paddingBottom}%;">
            ${mediaElement}
            <button class="reject-btn" onclick="rejectItem(this)">✕</button>
            <button class="favorite-btn ${favClass}" onclick="toggleFavorite(this)">${favIcon}</button>
            ${reviseBtn}
            ${upscaleBtn}
            ${videoBadge}
            ${previewBadge}
            <div class="time-tag">${d.duration}s</div>
            <div class="index-tag">#${totalIndex}</div>
            ${buildLabelOverlay(d)}
        </div>
        <div class="info">
            <div class="stat" title="${modelName}"><b>Model:</b> <span>${finalModel}</span></div>

            ${loraLine}
            <div class="stat"><b>Smp:</b> <span>${d.sampler} / ${d.scheduler}</span></div>
            <div class="stat">
                <b>Cfg:</b> ${d.cfg} &nbsp; <b>Stp:</b> ${d.steps} &nbsp; <b>Dn:</b> <span style="color:var(--accent-denoise)">${d.denoise}</span>
            </div>
            
            ${promptInfo}
            <div class="stat"><b>Size:</b> ${d.width}x${d.height} &nbsp; <b>Seed:</b> ${d.seed}</div>
        </div>`;

    // For videos: autoplay is handled by the virtual scroll lifecycle in logic_virtual.js.
    // Double-click on video still toggles favorite via the ondblclick attribute.
    // loadedmetadata updates per-card aspect ratio as a safety net for unprobed videos.
    if (isVideo) {
        const videoEl = card.querySelector('video');
        if (videoEl) {
            videoEl.addEventListener('loadedmetadata', function() {
                if (videoEl.videoWidth && videoEl.videoHeight) {
                    d.width = videoEl.videoWidth;
                    d.height = videoEl.videoHeight;
                    const ar = videoEl.videoHeight / videoEl.videoWidth;
                    const wrapper = card.querySelector('.img-wrapper');
                    if (wrapper) wrapper.style.paddingBottom = (ar * 100).toFixed(2) + '%';
                }
            });
        }
    }

    return card;
}

// Workflow Sweep items show a read-only "Workflow job" panel instead of the revise form.
function _setModalWorkflowMode(d) {
    const isWf = d.model_type === 'workflow';
    const rev = document.getElementById('m-revise-controls');
    const pnl = document.getElementById('m-workflow-panel');
    if (!rev || !pnl) return;
    rev.style.display = isWf ? 'none' : 'flex';
    pnl.style.display = isWf ? 'flex' : 'none';
    if (!isWf) return;
    const esc = (t) => String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    const metaEl = document.getElementById('m-wf-meta');
    if (metaEl) {
        const bits = [d.wf_source || 'workflow'];
        if (d.wf_output && d.wf_output.node_id !== undefined) bits.push('output ' + d.wf_output.node_id + (d.wf_output.output_index ? ':' + d.wf_output.output_index : ''));
        if (d.wf_sweep_id) bits.push('sweep ' + d.wf_sweep_id);
        if (d.wf_job_index !== undefined && d.wf_job_index !== null) bits.push('job #' + (Number(d.wf_job_index) + 1));
        metaEl.textContent = bits.join(' \u00B7 ');
    }
    const tbl = document.getElementById('m-wf-params');
    if (tbl) {
        const p = (d.wf_params && typeof d.wf_params === 'object') ? d.wf_params : {};
        const keys = Object.keys(p).sort();
        tbl.innerHTML = keys.length
            ? keys.map(k => `<tr><td style="padding:3px 8px 3px 0; color:#888; vertical-align:top;">${esc(k)}</td><td style="padding:3px 0; color:#fff; word-break:break-all;">${esc(p[k])}</td></tr>`).join('')
            : '<tr><td style="color:#666;">No varied parameters recorded.</td></tr>';
    }
    const canFetch = !!d.wf_sweep_id && d.wf_job_index !== undefined && d.wf_job_index !== null;
    ['m-wf-download', 'm-wf-copy'].forEach(bid => {
        const b = document.getElementById(bid);
        if (!b) return;
        b.disabled = !canFetch;
        b.title = canFetch ? '' : 'This item was not produced by a recorded sweep (no sweep id).';
        b.style.opacity = canFetch ? '' : '0.5';
    });
}

// ---- Revise-modal image: drag onto a Load Image node / canvas, or "Send to Load Image" (issue #22) ----
// The drag carries ComfyUI's own asset-info JSON + a same-origin URL (what its Assets panel sends), so a node
// drop references the file in output/ and a canvas drop creates a Load Image node. Helpers: logic_dragdrop.js.
function _uscgModalRef() {
    const d = activeData.find(x => x.id === window.currentModalId);
    return d ? uscgImageRef(d) : null;
}

async function _uscgPrefetchDragFile(ref, id) {
    if (!USCG_DRAG_UPLOAD || !ref) return;
    if (window._uscgDragFile && window._uscgDragFile.id === id) return;
    try {
        const blob = await (await fetch(ref.absUrl)).blob();
        if (window.currentModalId === id) window._uscgDragFile = { id: id, file: new File([blob], ref.filename, { type: blob.type || ref.mime }) };
    } catch (e) { /* no File: the URL carriers still work */ }
}

function _uscgInstallModalDrag() {
    if (window._uscgDragInstalled) return;
    const img = document.getElementById('m-img');
    if (!img) return;
    window._uscgDragInstalled = true;
    img.addEventListener('dragstart', function (e) {
        const ref = _uscgModalRef();
        if (!ref || !e.dataTransfer) { e.preventDefault(); return; }
        const f = window._uscgDragFile && window._uscgDragFile.id === window.currentModalId ? window._uscgDragFile.file : null;
        uscgFillDragData(e.dataTransfer, ref, USCG_DRAG_UPLOAD ? f : null);
    });
}

function _uscgSyncModalDrag(d) {
    _uscgInstallModalDrag();
    const ref = uscgImageRef(d);
    const img = document.getElementById('m-img');
    if (img) img.draggable = !!ref;
    const btn = document.getElementById('modal-send-loadimage-btn');
    if (btn) {
        btn.style.display = ref ? '' : 'none';
        btn.textContent = (typeof USCG_STANDALONE !== 'undefined' && USCG_STANDALONE)
            ? '\u2B07 Download image' : '\uD83D\uDDBC Send to Load Image';
    }
    window._uscgDragFile = null;
    _uscgPrefetchDragFile(ref, d.id);
}

function _uscgAskParentLoadImage(ref, upload) {
    return new Promise(function (resolve) {
        if (!window.parent || window.parent === window) { resolve(null); return; }
        var done = false;
        var onMsg = function (ev) {
            if (!ev.data || ev.data.type !== 'uscg_send_to_load_image_result') return;
            done = true;
            window.removeEventListener('message', onMsg);
            resolve({ ok: !!ev.data.ok, message: ev.data.message || '' });
        };
        window.addEventListener('message', onMsg);
        window.parent.postMessage({
            type: 'uscg_send_to_load_image', url: ref.absUrl, filename: ref.filename, subfolder: ref.subfolder,
            imageType: ref.type, upload: !!upload, node_id: TARGET_NODE_ID
        }, '*');
        setTimeout(function () {
            if (done) return;
            window.removeEventListener('message', onMsg);
            resolve(null);
        }, 8000);
    });
}

async function sendModalToLoadImage() {
    const ref = _uscgModalRef();
    if (!ref) { _uscgToast('This item is not an image.', false); return; }
    if (typeof USCG_STANDALONE !== 'undefined' && USCG_STANDALONE) {
        const a = document.createElement('a');
        a.href = ref.absUrl; a.download = ref.filename;
        document.body.appendChild(a); a.click(); a.remove();
        return;
    }
    const res = await _uscgAskParentLoadImage(ref, USCG_DRAG_UPLOAD);
    if (res) { _uscgToast(res.message || (res.ok ? 'Sent to Load Image.' : 'Could not send.'), res.ok); return; }
    _uscgToast('The ComfyUI tab did not answer. Drag the image onto a Load Image node instead.', false);
}

// Open Revision Modal
function openM(id) {
    window.currentModalId = id;
    const d = activeData.find(x => x.id === id);
    if (!d) return;
    // a video item opens its poster (middle frame) in the modal; the card itself plays the video
    document.getElementById('m-img').src = (d.media_type === 'video' && d.poster) ? d.poster : d.file;
    _setModalWorkflowMode(d);
    _uscgSyncModalDrag(d);
    const previewNote = document.getElementById('m-preview-note');
    if (previewNote) {
        if (d.preview) {
            previewNote.textContent = `Preview of ${d.full_width}x${d.full_height} @ ${d.full_steps} steps (this image: ${d.width}x${d.height} @ ${d.steps} steps)`;
            previewNote.style.display = 'block';
        } else {
            previewNote.style.display = 'none';
        }
    }
    console.log(d)
    // Populate read-only info fields
    const modelEl = document.getElementById('f-model');
    const seedEl = document.getElementById('f-seed');
    const posEl = document.getElementById('f-pos');
    const negEl = document.getElementById('f-neg');

    if (modelEl) modelEl.value = d.model || meta.model || "Default";
    if (seedEl) seedEl.value = d.seed || 0;
    // Check trigger word toggle
    const showTriggers = document.getElementById('toggle-triggers')?.checked !== false;
    if (posEl) posEl.value = showTriggers
        ? (d.positive || meta.positive || "")
        : (d.config_positive || d.positive || meta.positive || "");
    if (negEl) negEl.value = showTriggers
        ? (d.negative || meta.negative || "")
        : (d.config_negative || d.negative || meta.negative || "");

    // Populate editable parameter fields
    const map = {
        'smp': d.sampler,
        'sch': d.scheduler,
        'stp': d.steps,
        'cfg': d.cfg,
        'den': d.denoise,
        'lor': d.lora
    };

    for (let k in map) {
        const el = document.getElementById('f-' + k);
        if (el) el.value = map[k];
    }

    // Populate related variants reel
    const r = document.getElementById('reel');
    r.innerHTML = '';

    activeData.forEach(x => {
        if (x.rejected) return;
        if (x.seed === d.seed) {
            const i = document.createElement('img');
            i.src = (x.media_type === 'video' && x.poster) ? x.poster : x.file;
            i.onclick = () => openM(x.id);
            if (x.id === id) i.style.borderColor = "var(--accent)";
            r.appendChild(i);
        }
    });

    document.getElementById('modal').style.display = 'flex';

    // Mark modal as open so grid keyboard shortcuts are suppressed
    window._modalOpen = true;

    // Attach modal keyboard handler (capture phase so it fires before grid handler)
    if (window._modalKeyHandler) {
        document.removeEventListener('keydown', window._modalKeyHandler, true);
    }
    window._modalKeyHandler = function(e) {
        if (!window._modalOpen) return;

        // Don't intercept typing in form inputs
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        const currentId = window.currentModalId;
        if (currentId == null || !processedData || processedData.length === 0) return;

        // Navigate to next/prev item in the current filtered/sorted set (processedData)
        const currentIndex = processedData.findIndex(x => x.id === currentId);

        const goNext = () => {
            if (currentIndex < processedData.length - 1) {
                openM(processedData[currentIndex + 1].id);
            }
        };
        const goPrev = () => {
            if (currentIndex > 0) {
                openM(processedData[currentIndex - 1].id);
            }
        };

        switch (e.key) {
            case 'ArrowRight':
                e.preventDefault();
                e.stopPropagation();
                goNext();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                e.stopPropagation();
                goPrev();
                break;
            case ' ':
                e.preventDefault();
                e.stopPropagation();
                if (e.shiftKey) { goPrev(); } else { goNext(); }
                break;
            case 'Escape':
                e.preventDefault();
                e.stopPropagation();
                closeM();
                break;
            default:
                // 1-9 keys (no shift/ctrl/alt): quick favorite current modal item
                if (!e.shiftKey && !e.ctrlKey && !e.altKey && !e.metaKey &&
                    e.code >= 'Digit1' && e.code <= 'Digit9') {
                    e.preventDefault();
                    e.stopPropagation();
                    toggleModalFavorite();
                }
                break;
        }
    };
    document.addEventListener('keydown', window._modalKeyHandler, true);

    // Update navigation counter
    _updateModalCounter();
    _syncModalFavBtn(d);
}

// Favorite the item shown in the lightbox (1-9 keys and the phone action bar).
function toggleModalFavorite() {
    const item = activeData ? activeData.find(x => x.id === window.currentModalId) : null;
    if (!item) return;
    item.favorited = !item.favorited;
    if (typeof markItemChanged === 'function') markItemChanged(item);
    if (typeof scheduleJSONUpdate === 'function') scheduleJSONUpdate();
    // Keep the rendered card's star in sync when the card is in the DOM
    const card = (typeof nodeMap !== 'undefined' && nodeMap) ? nodeMap.get(item.id) : null;
    const favBtn = card ? card.querySelector('.favorite-btn') : null;
    if (favBtn) {
        favBtn.classList.toggle('favorited', item.favorited);
        favBtn.innerText = item.favorited ? '★' : '☆';
    }
    // Visual feedback on the modal image
    const mImg = document.getElementById('m-img');
    if (mImg) {
        const color = item.favorited ? '#00cc44' : '#cc4444';
        mImg.style.transition = 'box-shadow 0.3s';
        mImg.style.boxShadow = `0 0 30px ${color}`;
        setTimeout(() => { mImg.style.boxShadow = ''; }, 400);
    }
    // Update the navigation counter tooltip if displayed
    const counterEl = document.getElementById('modal-nav-counter');
    if (counterEl) {
        counterEl.title = item.favorited ? 'Favorited' : 'Not favorited';
    }
    _syncModalFavBtn(item);
}

// Phone action bar Favorite button: icon + label live in spans (never write innerText on the button).
function _syncModalFavBtn(item) {
    const btn = document.getElementById('modal-fav-btn');
    if (!btn || !item) return;
    const ico = btn.querySelector('.mab-ico'), lbl = btn.querySelector('.mab-lbl');
    if (ico) ico.textContent = item.favorited ? '★' : '☆';
    if (lbl) lbl.textContent = item.favorited ? 'Favorited' : 'Favorite';
    btn.classList.toggle('favorited', !!item.favorited);
}

// Lightbox prev/next by delta through the filtered/sorted set (phone action bar).
function modalNav(delta) {
    if (!processedData) return;
    const idx = processedData.findIndex(x => x.id === window.currentModalId);
    const target = processedData[idx + delta];
    if (idx !== -1 && target) openM(target.id);
}

// Reject the lightbox item, then move on to the neighbour (or close when it was the last one).
function rejectModalItem() {
    const item = activeData ? activeData.find(x => x.id === window.currentModalId) : null;
    if (!item) return;
    item.rejected = true;
    if (typeof markItemChanged === 'function') markItemChanged(item);
    const idx = processedData ? processedData.findIndex(x => x.id === item.id) : -1;
    const next = idx !== -1 ? (processedData[idx + 1] || processedData[idx - 1]) : null;
    // Re-run the pipeline after the neighbour is chosen: it drops the rejected item from the grid.
    setTimeout(() => {
        if (typeof updateDataPipeline === 'function') updateDataPipeline();
        if (typeof scheduleJSONUpdate === 'function') scheduleJSONUpdate();
    }, 100);
    if (next && next.id !== item.id) openM(next.id); else closeM();
}

// Update the modal nav counter display
function _updateModalCounter() {
    const counterEl = document.getElementById('modal-nav-counter');
    if (!counterEl || !processedData) return;
    const idx = processedData.findIndex(x => x.id === window.currentModalId);
    if (idx === -1) { counterEl.textContent = ''; return; }
    counterEl.textContent = `${idx + 1} / ${processedData.length}`;
}

function closeM() {
    document.getElementById('modal').style.display = 'none';
    window._modalOpen = false;
    if (window._modalKeyHandler) {
        document.removeEventListener('keydown', window._modalKeyHandler, true);
        window._modalKeyHandler = null;
    }
}

// THROTTLED JSON Updates
let jsonUpdateTimeout = null;

function updateJSONs(visible) {
    if (jsonUpdateTimeout) {
        clearTimeout(jsonUpdateTimeout);
    }

    jsonUpdateTimeout = setTimeout(() => {
        // 🚀 OPTIMIZED: Single pass through data instead of 3 separate filters
        const good = [];
        const favorited = [];
        const rejected = [];

        for (const item of activeData) {
            if (item.rejected) {
                rejected.push(item);
            } else {
                good.push(item);
                if (item.favorited) {
                    favorited.push(item);
                }
            }
        }

        // Disabled automatic JSON generation - now on-demand via buttons
        // Store the datasets for on-demand generation
        window.cachedJSONData = { good, favorited, rejected };

        // Calculate unique config counts for button labels
        const goodUniqueCount = countUniqueConfigs(good);
        const favUniqueCount = countUniqueConfigs(favorited);
        const rejUniqueCount = countUniqueConfigs(rejected);

        // Update button labels with unique counts
        updateJSONButtonLabels(goodUniqueCount, favUniqueCount, rejUniqueCount);
        if (typeof analyticsPanelOnDataChanged === 'function') analyticsPanelOnDataChanged();
    }, 100); // Reduced from 300ms since we have separate debounce above
}

// OPTIMIZED JSON generation
const jsonElCache = new Map();

// Fallback for browsers that don't support requestIdleCallback
const runIdle = window.requestIdleCallback || (cb => setTimeout(cb, 1));
const cancelIdle = window.cancelIdleCallback || clearTimeout;

function generateSmartJSON(dataset, targetId) {
    const el = document.getElementById(targetId);
    if (!el) return;

    if (dataset.length === 0) {
        el.innerText = "[]";
        return;
    }

    // Extract unique configurations
    const configMap = new Map();

    for (const d of dataset) {
        const config = {
            sampler: d.sampler,
            scheduler: d.scheduler,
            steps: d.steps,
            cfg: d.cfg,
            denoise: d.denoise,
            lora: d.lora,
            model: d.model || "Default"
        };

        // Create a unique key for this configuration
        const key = JSON.stringify(config);

        // Only add if we haven't seen this exact config before
        if (!configMap.has(key)) {
            configMap.set(key, config);
        }
    }

    // Convert map to array
    const uniqueConfigs = Array.from(configMap.values());

    // Limit output
    const limit = Math.min(uniqueConfigs.length, 100);
    const limited = uniqueConfigs.slice(0, limit);

    let jsonText = JSON.stringify(limited, null, 2);

    if (uniqueConfigs.length > 100) {
        jsonText += `\n\n// ... and ${uniqueConfigs.length - 100} more unique configs`;
    }

    el.innerText = jsonText;
}

// Helper function to count unique configs in a dataset
function countUniqueConfigs(dataset) {
    const configSet = new Set();

    for (const d of dataset) {
        const key = JSON.stringify({
            sampler: d.sampler,
            scheduler: d.scheduler,
            steps: d.steps,
            cfg: d.cfg,
            denoise: d.denoise,
            lora: d.lora,
            model: d.model || "Default"
        });
        configSet.add(key);
    }

    return configSet.size;
}

// Add button label update function
function updateJSONButtonLabels(goodCount, favCount, rejCount) {
    const goodBtn = document.getElementById('json-btn-good');
    const favBtn = document.getElementById('json-btn-favorite');
    const rejBtn = document.getElementById('json-btn-bad');

    if (goodBtn) goodBtn.innerText = `View Configs (${goodCount})`;
    if (favBtn) favBtn.innerText = `View Favorited (${favCount})`;
    if (rejBtn) rejBtn.innerText = `View Rejected (${rejCount})`;
}

// On-demand JSON generation functions
function viewGoodJSON() {
    if (!window.cachedJSONData) return;
    const bar = document.getElementById('json-bar-good');
    if (!bar) return;

    // Toggle visibility
    if (bar.style.display === 'block') {
        bar.style.display = 'none';
        return;
    }

    const { good } = window.cachedJSONData;
    generateSmartJSON(good, 'json-bar-good');

    bar.style.display = 'block';
    bar.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function viewFavoritedJSON() {
    if (!window.cachedJSONData) return;
    const bar = document.getElementById('json-bar-favorite');
    if (!bar) return;

    // Toggle visibility
    if (bar.style.display === 'block') {
        bar.style.display = 'none';
        return;
    }

    const { favorited } = window.cachedJSONData;
    generateSmartJSON(favorited, 'json-bar-favorite');

    bar.style.display = 'block';
    bar.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function viewRejectedJSON() {
    if (!window.cachedJSONData) return;
    const bar = document.getElementById('json-bar-bad');
    if (!bar) return;

    // Toggle visibility
    if (bar.style.display === 'block') {
        bar.style.display = 'none';
        return;
    }

    const { rejected } = window.cachedJSONData;
    generateSmartJSON(rejected, 'json-bar-bad');

    bar.style.display = 'block';
    bar.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
/**
 * SEARCH FILTER FUNCTIONS
 */

// Add a search filter
function addSearchFilter() {
    const typeSelect = document.getElementById('search-filter-type');
    const inputField = document.getElementById('search-filter-input');

    if (!typeSelect || !inputField) return;

    const filterType = typeSelect.value;
    const searchTerm = inputField.value.trim();

    // Validation
    if (!filterType) {
        alert('Please select a filter type');
        return;
    }

    if (!searchTerm) {
        alert('Please enter a search term');
        return;
    }

    // Check if this exact filter already exists
    const exists = searchFilters.some(f => f.type === filterType && f.term === searchTerm);
    if (exists) {
        alert('This search filter already exists');
        return;
    }

    // Add to search filters array
    searchFilters.push({
        type: filterType,
        term: searchTerm
    });

    // Clear inputs
    typeSelect.value = '';
    inputField.value = '';

    // Update UI
    renderSearchFilters();

    // Trigger data pipeline update
    updateDataPipeline();

    console.log(`[Search Filter] Added: ${filterType} contains "${searchTerm}"`);
}

// Remove a search filter
function removeSearchFilter(index) {
    if (index >= 0 && index < searchFilters.length) {
        const removed = searchFilters.splice(index, 1)[0];
        console.log(`[Search Filter] Removed: ${removed.type} contains "${removed.term}"`);

        // Update UI
        renderSearchFilters();

        // Trigger data pipeline update
        updateDataPipeline();
    }
}

// Clear all search filters
function clearAllSearchFilters() {
    if (searchFilters.length === 0) return;

    searchFilters = [];
    renderSearchFilters();
    updateDataPipeline();

    console.log('[Search Filter] All search filters cleared');
}

// Render search filter tags
function renderSearchFilters() {
    const container = document.getElementById('active-search-filters');
    if (!container) return;

    container.innerHTML = '';

    if (searchFilters.length === 0) {
        container.innerHTML = '<div style="color: #555; font-size: 10px; padding: 4px 0;">No active search filters</div>';
        return;
    }

    searchFilters.forEach((filter, index) => {
        const tag = document.createElement('div');
        tag.className = 'search-filter-tag';

        // Format display name for filter type
        const typeNames = {
            'model': 'Model',
            'positive': 'Positive',
            'negative': 'Negative',
            'sampler': 'Sampler',
            'scheduler': 'Scheduler',
            'lora': 'LoRA'
        };

        tag.innerHTML = `
            <span class="filter-type ${filter.type}">${typeNames[filter.type] || filter.type}</span>
            <span class="filter-term" title="${filter.term}">${filter.term}</span>
            <button class="remove-btn" onclick="removeSearchFilter(${index})" title="Remove filter">✕</button>
        `;

        container.appendChild(tag);
    });

    // Add clear all button if there are multiple filters
    if (searchFilters.length > 1) {
        const clearBtn = document.createElement('button');
        clearBtn.className = 'search-filter-add-btn';
        clearBtn.style.background = '#ff3860';
        clearBtn.style.padding = '4px 10px';
        clearBtn.innerText = 'CLEAR ALL';
        clearBtn.onclick = clearAllSearchFilters;
        container.appendChild(clearBtn);
    }
}

// Check if an item matches search filters
function matchesSearchFilters(item) {
    if (searchFilters.length === 0) return true;

    // Item must match ALL search filters (AND logic)
    return searchFilters.every(filter => {
        let value = '';

        switch (filter.type) {
            case 'model':
                value = item.model || meta.model || "Default";
                break;
            case 'positive':
                value = item.positive || meta.positive || "";
                break;
            case 'negative':
                value = item.negative || meta.negative || "";
                break;
            case 'sampler':
                value = item.sampler || "";
                break;
            case 'scheduler':
                value = item.scheduler || "";
                break;
            case 'lora':
                value = item.lora || "";
                break;
            default:
                return false;
        }

        // Case-insensitive search
        return value.toLowerCase().includes(filter.term.toLowerCase());
    });
}


// ============================================================
// LOGIC FILTER FUNCTIONS (numeric comparison filters)
// ============================================================

// Add a logic filter
function addLogicFilter() {
    const fieldSelect = document.getElementById('logic-filter-field');
    const opSelect = document.getElementById('logic-filter-op');
    const valueInput = document.getElementById('logic-filter-value');

    if (!fieldSelect || !opSelect || !valueInput) return;

    const field = fieldSelect.value;
    const op = opSelect.value;
    const rawValue = valueInput.value.trim();

    if (!field) { alert('Please select a field'); return; }
    if (rawValue === '') { alert('Please enter a value'); return; }

    const value = parseFloat(rawValue);
    if (isNaN(value)) { alert('Value must be a number'); return; }

    // Add to logicFilters array
    logicFilters.push({ field, op, value });

    // Clear inputs
    fieldSelect.value = '';
    valueInput.value = '';

    renderLogicFilters();
    updateDataPipeline();

    console.log(`[Logic Filter] Added: ${field} ${op} ${value}`);
}

// --- Workflow params in the Logic Filter (numeric wf_params keys only; group is absent when no item has wf_params) ---
var _wfUiSig = null;

function populateWfParamFilterUI() {
    if (typeof wfParamKeys !== 'function') return;
    const keys = wfParamKeys(activeData).filter(k => k.numeric);
    const sig = keys.map(k => k.key).join(',');
    if (sig === _wfUiSig) return;
    _wfUiSig = sig;
    const sel = document.getElementById('logic-filter-field');
    if (!sel) return;
    const prev = sel.value;
    const old = document.getElementById('logic-filter-wf-group');
    if (old) old.remove();
    if (keys.length) {
        const group = document.createElement('optgroup');
        group.id = 'logic-filter-wf-group';
        group.label = 'Workflow params';
        const all = keys.map(k => k.key);
        keys.forEach(k => {
            const o = document.createElement('option');
            o.value = 'wf:' + k.key;
            o.textContent = wfShortLabel(k.key, all);
            o.title = k.key;
            group.appendChild(o);
        });
        sel.appendChild(group);
    }
    sel.value = prev;
}

// --- Metric fields in the Logic Filter (populated from the data; hidden when no item has metrics) ---
var _metricUiSig = null;

function populateMetricFilterUI() {
    if (typeof getMetricKeys !== 'function') return;
    const keys = getMetricKeys();
    const sig = keys.join(',');
    if (sig === _metricUiSig) return;
    _metricUiSig = sig;

    const sel = document.getElementById('logic-filter-field');
    if (sel) {
        const prev = sel.value;
        const old = document.getElementById('logic-filter-metric-group');
        if (old) old.remove();
        if (keys.length) {
            const group = document.createElement('optgroup');
            group.id = 'logic-filter-metric-group';
            group.label = 'Metrics';
            keys.forEach(k => {
                const o = document.createElement('option');
                o.value = 'metric:' + k;
                o.textContent = metricLabel(k);
                group.appendChild(o);
            });
            sel.appendChild(group);
        }
        sel.value = prev;
    }

    const row = document.getElementById('metric-range-row');
    const rsel = document.getElementById('metric-range-field');
    if (row && rsel) {
        const prev = rsel.value;
        rsel.innerHTML = '';
        keys.forEach(k => {
            const o = document.createElement('option');
            o.value = k;
            o.textContent = metricLabel(k);
            rsel.appendChild(o);
        });
        if (keys.includes(prev)) rsel.value = prev;
        row.style.display = keys.length ? 'flex' : 'none';
        updateMetricRangeHints();
    }
}

// Placeholders show the observed range of the selected metric
function updateMetricRangeHints() {
    const rsel = document.getElementById('metric-range-field');
    const minI = document.getElementById('metric-range-min');
    const maxI = document.getElementById('metric-range-max');
    if (!rsel || !minI || !maxI || !rsel.value) return;
    let lo = Infinity, hi = -Infinity;
    for (const it of activeData) {
        const v = it && it.metrics ? it.metrics[rsel.value] : undefined;
        if (typeof v === 'number' && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    minI.placeholder = isFinite(lo) ? 'min ' + parseFloat(lo.toPrecision(3)) : 'min';
    maxI.placeholder = isFinite(hi) ? 'max ' + parseFloat(hi.toPrecision(3)) : 'max';
}

// "Metric range" ADD: pushes up to two logic filters (>= min, <= max)
function addMetricRange() {
    const rsel = document.getElementById('metric-range-field');
    const minI = document.getElementById('metric-range-min');
    const maxI = document.getElementById('metric-range-max');
    if (!rsel || !minI || !maxI || !rsel.value) return;
    const field = 'metric:' + rsel.value;
    const lo = minI.value.trim() === '' ? null : parseFloat(minI.value);
    const hi = maxI.value.trim() === '' ? null : parseFloat(maxI.value);
    if ((lo === null || isNaN(lo)) && (hi === null || isNaN(hi))) { alert('Enter a min and/or max'); return; }
    if (lo !== null && !isNaN(lo)) logicFilters.push({ field, op: '>=', value: lo });
    if (hi !== null && !isNaN(hi)) logicFilters.push({ field, op: '<=', value: hi });
    minI.value = '';
    maxI.value = '';
    renderLogicFilters();
    updateDataPipeline();
}

// Remove a logic filter by index
function removeLogicFilter(index) {
    if (index >= 0 && index < logicFilters.length) {
        const removed = logicFilters.splice(index, 1)[0];
        console.log(`[Logic Filter] Removed: ${removed.field} ${removed.op} ${removed.value}`);
        renderLogicFilters();
        updateDataPipeline();
    }
}

// Clear all logic filters
function clearAllLogicFilters() {
    if (logicFilters.length === 0) return;
    logicFilters = [];
    renderLogicFilters();
    updateDataPipeline();
    console.log('[Logic Filter] All logic filters cleared');
}

// Render logic filter chips
function renderLogicFilters() {
    const container = document.getElementById('active-logic-filters');
    if (!container) return;

    container.innerHTML = '';

    if (logicFilters.length === 0) {
        container.innerHTML = '<div style="color: #555; font-size: 10px; padding: 4px 0;">No active logic filters</div>';
        return;
    }

    const fieldLabels = {
        'lora_strength': 'LoRA Str',
        'cfg': 'CFG',
        'steps': 'Steps',
        'denoise': 'Denoise',
        'width': 'Width',
        'height': 'Height',
        'seed': 'Seed'
    };

    logicFilters.forEach((filter, index) => {
        const tag = document.createElement('div');
        tag.className = 'search-filter-tag';

        const label = fieldLabels[filter.field] ||
            (typeof filter.field === 'string' && filter.field.startsWith('metric:') ? metricLabel(filter.field.slice(7)) :
             (typeof filter.field === 'string' && filter.field.startsWith('wf:') && typeof wfShortLabel === 'function'
                ? wfShortLabel(filter.field.slice(3), wfParamKeys(activeData)) : filter.field));
        tag.innerHTML = `
            <span class="filter-type" style="background: #1a4a2a; color: #4ecf7a;">${label}</span>
            <span class="filter-term">${filter.op} ${filter.value}</span>
            <button class="remove-btn" onclick="removeLogicFilter(${index})" title="Remove filter">✕</button>
        `;

        container.appendChild(tag);
    });

    // Add clear all button if there are multiple filters
    if (logicFilters.length > 1) {
        const clearBtn = document.createElement('button');
        clearBtn.className = 'search-filter-add-btn';
        clearBtn.style.background = '#ff3860';
        clearBtn.style.padding = '4px 10px';
        clearBtn.innerText = 'CLEAR ALL';
        clearBtn.onclick = clearAllLogicFilters;
        container.appendChild(clearBtn);
    }
}

// ============================================================
// QUICK FILTER FUNCTIONS (faceted chip filtering)
// ============================================================

// Human-readable labels for each filter type
const _qfTypeLabels = {
    sampler: 'Sampler',
    scheduler: 'Scheduler',
    lora: 'LoRA',
    lora_strength: 'LoRA Strength',
    model: 'Model',
    denoise: 'Denoise',
    size: 'Size',
    seed: 'Seed',
    steps: 'Steps',
    cfg: 'CFG',
    upscaleMethod: 'Upscale',
    mediaType: 'Media',
    passType: 'Pass',
    positive: 'Positive',
    negative: 'Negative'
};

// All types that Quick Filter supports (type-menu order)
const _qfAllTypes = ['lora', 'lora_strength', 'cfg', 'steps', 'sampler', 'scheduler', 'denoise', 'size', 'model', 'seed', 'upscaleMethod', 'mediaType', 'passType', 'positive', 'negative'];

// Types shown in the expanded facet panel (omit prompt types for brevity)
const _qfFacetableTypes = ['sampler', 'scheduler', 'lora', 'lora_strength', 'model', 'denoise', 'cfg', 'steps', 'size', 'seed', 'upscaleMethod', 'mediaType', 'passType'];

// Workflow Sweep: one extra chip type per VARIED wf_params key ("wf:<key>"); the lists are dynamic.
function _qfTypes(base) {
    if (typeof wfParamKeys !== 'function') return base;
    return base.concat(wfParamKeys(typeof activeData !== 'undefined' ? activeData : []).filter(k => k.distinct > 1).map(k => 'wf:' + k.key));
}

function _qfLabel(type) {
    if (_qfTypeLabels[type]) return _qfTypeLabels[type];
    if (typeof type === 'string' && type.startsWith('wf:') && typeof wfShortLabel === 'function') {
        return 'WF ' + wfShortLabel(type.slice(3), wfParamKeys(typeof activeData !== 'undefined' ? activeData : []));
    }
    return type;
}

/**
 * Extract a facet value (or array of values) from an item for a given type.
 * Returns null if the field is missing.
 */
function _qfExtractItemValue(item, type) {
    if (typeof type === 'string' && type.startsWith('wf:')) {
        const raw = wfParamValue(item, type.slice(3));
        return raw === null ? null : String(raw);
    }
    if (type === 'lora') {
        const lora = String(item.lora || 'None');
        if (lora === 'None') return ['None'];
        return lora.split(' + ').map(p => p.split(':')[0].trim());
    }
    if (type === 'lora_strength') {
        const lora = String(item.lora || 'None');
        if (lora === 'None') return null;
        const firstPart = lora.split(' + ')[0];
        const parts = firstPart.split(':');
        if (parts.length < 2) return null;
        const v = parseFloat(parts[1]);
        return isNaN(v) ? null : v;
    }
    if (type === 'size') {
        return `${item.width || '?'}x${item.height || '?'}`;
    }
    if (type === 'model') {
        return item.model || meta && meta.model || 'Default';
    }
    if (type === 'steps') return String(item.steps);
    if (type === 'cfg') return String(item.cfg);
    if (type === 'positive') return item.positive || (meta && meta.positive) || '';
    if (type === 'negative') return item.negative || (meta && meta.negative) || '';
    if (type === 'upscaleMethod') {
        if (!item.upscaled) return 'No Upscale';
        const mode = item.upscale_mode || '';
        const model = item.upscale_model;
        const shortModel = model ? String(model).replace(/\\/g, '/').split('/').pop().replace(/\.[^.]+$/, '') : '';
        return shortModel ? `${mode} + ${shortModel}` : mode || 'Upscaled';
    }
    if (type === 'mediaType') return item.media_type || 'image';
    if (type === 'passType') return item.preview ? 'Preview' : 'Full';
    const v = item[type];
    if (v === undefined || v === null) return null;
    return String(v);
}

/**
 * Get all unique values in activeData for a given type.
 * Returns a sorted array of strings.
 */
function _qfUniqueValuesForType(type) {
    const seen = new Set();
    const data = (typeof activeData !== 'undefined' ? activeData : []);
    for (const item of data) {
        const val = _qfExtractItemValue(item, type);
        if (val === null || val === undefined) continue;
        if (Array.isArray(val)) {
            val.forEach(v => seen.add(String(v)));
        } else {
            seen.add(String(val));
        }
    }
    return [...seen].sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }));
}

/**
 * Compute facets for an expanded chip:
 * - subset: items matching this chip's primary (type, value)
 * - for each other type, collect unique values in that subset
 */
function _computeChipFacets(chip) {
    const data = (typeof activeData !== 'undefined' ? activeData : []);
    const subset = data.filter(item => _quickFilterMatches(item, chip));
    const facets = {};
    const facetTypes = _qfTypes(_qfFacetableTypes);
    for (const type of facetTypes) {
        if (type === chip.type) continue;
        facets[type] = new Set();
    }
    for (const item of subset) {
        for (const type of facetTypes) {
            if (type === chip.type) continue;
            const val = _qfExtractItemValue(item, type);
            if (val === null || val === undefined) continue;
            if (Array.isArray(val)) {
                val.forEach(v => facets[type].add(String(v)));
            } else {
                facets[type].add(String(val));
            }
        }
    }
    return { count: subset.length, facets };
}

// Module-level reference so the outside-click handler can be removed by name.
let _qfMenusOutsideHandler = null;

function _qfAttachOutsideHandler() {
    if (_qfMenusOutsideHandler) return; // already attached — don't double-register
    _qfMenusOutsideHandler = (e) => {
        const tm = document.getElementById('quick-filter-type-menu');
        const vm = document.getElementById('quick-filter-value-menu');
        const btn = document.getElementById('quick-filter-add-btn');
        // Keep menus open when the click is inside any controlled element.
        if (tm && tm.contains(e.target)) return;
        if (vm && vm.contains(e.target)) return;
        if (btn && btn.contains(e.target)) return;
        _closeQuickFilterMenus();
    };
    // Defer one tick so the click that opened the menu doesn't immediately close it.
    setTimeout(() => {
        if (_qfMenusOutsideHandler) {
            // Capture phase bypasses any e.stopPropagation() inside the menus.
            document.addEventListener('click', _qfMenusOutsideHandler, true);
        }
    }, 0);
}

function _qfDetachOutsideHandler() {
    if (_qfMenusOutsideHandler) {
        document.removeEventListener('click', _qfMenusOutsideHandler, true);
        _qfMenusOutsideHandler = null;
    }
}

/**
 * Close all open quick-filter menus by clicking outside.
 */
function _closeQuickFilterMenus() {
    const tm = document.getElementById('quick-filter-type-menu');
    const vm = document.getElementById('quick-filter-value-menu');
    if (tm) tm.style.display = 'none';
    if (vm) vm.style.display = 'none';
    _qfDetachOutsideHandler();
}

/**
 * Open the type-picker dropdown from the "+ Add Quick Filter" button.
 */
function _openQuickFilterTypeMenu(btnEl) {
    _closeQuickFilterMenus();
    const tm = document.getElementById('quick-filter-type-menu');
    if (!tm) return;
    tm.innerHTML = '';

    const data = (typeof activeData !== 'undefined' ? activeData : []);
    if (data.length === 0) {
        tm.innerHTML = '<div style="padding:8px 12px; color:#666; font-size:11px;">No data loaded</div>';
        tm.style.display = 'block';
        return;
    }

    for (const type of _qfTypes(_qfAllTypes)) {
        const label = _qfLabel(type);
        const div = document.createElement('div');
        div.style.cssText = 'padding:7px 14px; cursor:pointer; font-size:12px; color:#ccc; white-space:nowrap;';
        div.textContent = label;
        div.onmouseover = () => div.style.background = '#2a2a2a';
        div.onmouseout = () => div.style.background = '';
        div.onclick = (e) => {
            e.stopPropagation();
            tm.style.display = 'none';
            _openQuickFilterValueMenu(type);
        };
        tm.appendChild(div);
    }

    tm.style.display = 'block';
    _qfAttachOutsideHandler();
}

/**
 * Open the value-picker dropdown for a given type.
 */
function _openQuickFilterValueMenu(type) {
    _closeQuickFilterMenus();
    const vm = document.getElementById('quick-filter-value-menu');
    if (!vm) return;
    vm.innerHTML = '';

    const values = _qfUniqueValuesForType(type);
    const label = _qfLabel(type);

    if (values.length === 0) {
        vm.innerHTML = `<div style="padding:8px 12px; color:#666; font-size:11px;">No values for ${label}</div>`;
    } else {
        const header = document.createElement('div');
        header.style.cssText = 'padding:5px 12px; font-size:10px; color:#555; border-bottom:1px solid #2a2a2a; text-transform:uppercase; letter-spacing:0.05em;';
        header.textContent = label;
        vm.appendChild(header);

        for (const val of values) {
            const div = document.createElement('div');
            div.style.cssText = 'padding:6px 14px; cursor:pointer; font-size:12px; color:#ccc; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:240px;';
            div.title = val;
            div.textContent = val;
            div.onmouseover = () => div.style.background = '#2a2a2a';
            div.onmouseout = () => div.style.background = '';
            div.onclick = (e) => {
                e.stopPropagation();
                vm.style.display = 'none';
                addQuickFilter(type, val);
            };
            vm.appendChild(div);
        }
    }

    vm.style.display = 'block';
    _qfAttachOutsideHandler();
}

/**
 * Add a quick-filter chip. Prevents exact duplicates.
 */
function addQuickFilter(type, value) {
    _closeQuickFilterMenus();
    // Prevent duplicates
    const exists = quickFilters.some(f => f.type === type && String(f.value) === String(value));
    if (exists) {
        console.log(`[QuickFilter] Duplicate chip skipped: ${type}=${value}`);
        return;
    }
    quickFilters.push({ type, value: String(value), _expanded: false });
    renderQuickFilters();
    updateDataPipeline();
    console.log(`[QuickFilter] Added chip: ${type}=${value} (total: ${quickFilters.length})`);
}

/**
 * Remove a quick-filter chip by index.
 */
function removeQuickFilter(idx) {
    if (idx >= 0 && idx < quickFilters.length) {
        const removed = quickFilters.splice(idx, 1)[0];
        console.log(`[QuickFilter] Removed chip: ${removed.type}=${removed.value}`);
        renderQuickFilters();
        updateDataPipeline();
    }
}

/**
 * Toggle the expand/collapse state of a chip.
 */
function toggleQuickFilterChip(idx) {
    if (idx >= 0 && idx < quickFilters.length) {
        quickFilters[idx]._expanded = !quickFilters[idx]._expanded;
        renderQuickFilters();
    }
}

/**
 * Clear all quick-filter chips.
 */
function clearAllQuickFilters() {
    if (quickFilters.length === 0) return;
    quickFilters.length = 0;
    renderQuickFilters();
    updateDataPipeline();
    console.log('[QuickFilter] All chips cleared');
}

/**
 * Render quick-filter chips into #quick-filter-chips.
 */
function renderQuickFilters() {
    const container = document.getElementById('quick-filter-chips');
    if (!container) return;
    container.innerHTML = '';

    if (quickFilters.length === 0) {
        container.innerHTML = '<div style="color:#555; font-size:10px; padding:4px 0;">No active quick filters</div>';
        return;
    }

    quickFilters.forEach((chip, idx) => {
        const label = _qfLabel(chip.type);

        const chipEl = document.createElement('div');
        chipEl.style.cssText = 'border:1px solid #1e4a6e; border-radius:4px; margin-bottom:6px; overflow:hidden; background:#0d1f2d;';

        // --- Header row ---
        const header = document.createElement('div');
        header.style.cssText = 'display:flex; align-items:center; padding:4px 6px; gap:6px; cursor:pointer; user-select:none;';

        const typeBadge = document.createElement('span');
        typeBadge.style.cssText = 'background:#1a3a5c; color:#5ab4f5; font-size:10px; padding:1px 6px; border-radius:3px; font-weight:600; letter-spacing:0.04em; flex-shrink:0;';
        typeBadge.textContent = label.toUpperCase();

        const valueSpan = document.createElement('span');
        valueSpan.style.cssText = 'flex:1; font-size:11px; color:#d0d8e0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;';
        valueSpan.title = chip.value;
        valueSpan.textContent = chip.value;

        const toggleBtn = document.createElement('button');
        toggleBtn.style.cssText = 'background:none; border:none; color:#5ab4f5; cursor:pointer; font-size:11px; padding:0 4px; flex-shrink:0;';
        toggleBtn.title = chip._expanded ? 'Collapse' : 'Expand facets';
        toggleBtn.textContent = chip._expanded ? '▼' : '▶';
        toggleBtn.onclick = (e) => { e.stopPropagation(); toggleQuickFilterChip(idx); };

        const removeBtn = document.createElement('button');
        removeBtn.style.cssText = 'background:none; border:none; color:#666; cursor:pointer; font-size:12px; padding:0 2px; flex-shrink:0; line-height:1;';
        removeBtn.title = 'Remove filter';
        removeBtn.textContent = '✕';
        removeBtn.onclick = (e) => { e.stopPropagation(); removeQuickFilter(idx); };

        header.onclick = () => toggleQuickFilterChip(idx);
        header.appendChild(typeBadge);
        header.appendChild(valueSpan);
        header.appendChild(toggleBtn);
        header.appendChild(removeBtn);
        chipEl.appendChild(header);

        // --- Expanded body ---
        if (chip._expanded) {
            const body = document.createElement('div');
            body.style.cssText = 'border-top:1px solid #1e4a6e; padding:8px; background:#0a151f;';

            const { count, facets } = _computeChipFacets(chip);

            const countEl = document.createElement('div');
            countEl.style.cssText = 'font-size:10px; color:#5ab4f5; margin-bottom:6px; font-weight:600;';
            countEl.textContent = `Items matching: ${count}`;
            body.appendChild(countEl);

            let anyFacets = false;
            for (const type of _qfTypes(_qfFacetableTypes)) {
                if (type === chip.type) continue;
                const vals = facets[type];
                if (!vals || vals.size === 0) continue;

                const sortedVals = [...vals].sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }));

                anyFacets = true;
                const row = document.createElement('div');
                row.style.cssText = 'margin-bottom:5px;';

                const rowLabel = document.createElement('span');
                rowLabel.style.cssText = 'font-size:9px; color:#555; text-transform:uppercase; letter-spacing:0.05em; margin-right:6px;';
                rowLabel.textContent = _qfLabel(type) + ':';
                row.appendChild(rowLabel);

                for (const val of sortedVals) {
                    // Check if this value is already an active chip
                    const alreadyActive = quickFilters.some(f => f.type === type && f.value === val);
                    const btn = document.createElement('button');
                    btn.style.cssText = `display:inline-block; margin:1px 2px; padding:2px 7px; font-size:10px; border-radius:3px; cursor:pointer; border:1px solid ${alreadyActive ? '#2a6e44' : '#1e4a6e'}; background:${alreadyActive ? '#1a4a2a' : 'transparent'}; color:${alreadyActive ? '#4ecf7a' : '#8ab4d4'}; transition:background 0.1s;`;
                    btn.title = alreadyActive ? `Already filtering ${type}=${val}` : `Add filter: ${type}=${val}`;
                    btn.textContent = val.length > 30 ? val.slice(0, 28) + '…' : val;
                    if (!alreadyActive) {
                        btn.onclick = (e) => { e.stopPropagation(); addQuickFilter(type, val); };
                        btn.onmouseover = () => btn.style.background = '#1a3a5c';
                        btn.onmouseout = () => btn.style.background = 'transparent';
                    }
                    row.appendChild(btn);
                }
                body.appendChild(row);
            }

            if (!anyFacets) {
                const empty = document.createElement('div');
                empty.style.cssText = 'font-size:10px; color:#444; font-style:italic;';
                empty.textContent = 'No other facets available in this subset.';
                body.appendChild(empty);
            }

            chipEl.appendChild(body);
        }

        container.appendChild(chipEl);
    });

    // Clear-all button when multiple chips are active
    if (quickFilters.length > 1) {
        const clearBtn = document.createElement('button');
        clearBtn.className = 'search-filter-add-btn';
        clearBtn.style.cssText = 'background:#ff3860; padding:4px 10px; margin-top:4px;';
        clearBtn.textContent = 'CLEAR ALL';
        clearBtn.onclick = clearAllQuickFilters;
        container.appendChild(clearBtn);
    }
}

// ============================================================
// MANIFEST ANALYTICS
// ============================================================

/**
 * Dispatch to the correct analysis function based on type.
 * type: 'lora' | 'model' | 'prompt' | 'tags'
 */
function runManifestAnalysis(type) {
    const statusEl = document.getElementById('analytics-status');

    // Guard: fullManifest must exist
    if (!fullManifest || !Array.isArray(fullManifest.items) || fullManifest.items.length === 0) {
        if (statusEl) {
            statusEl.innerText = 'No manifest data available.';
            statusEl.style.color = '#ff3860';
        }
        return;
    }

    const items = fullManifest.items;
    let results = [];
    let title = '';

    if (type === 'lora') {
        title = 'LoRA Usage Stats';
        results = analyzeFieldStats(items, 'lora');
    } else if (type === 'model') {
        title = 'Model Usage Stats';
        results = analyzeFieldStats(items, 'model');
    } else if (type === 'prompt') {
        title = 'Full Prompt Stats';
        results = analyzePromptStats(items);
    } else if (type === 'tags') {
        title = 'Prompt Tag Stats';
        results = analyzeTagStats(items);
    }

    if (results.length === 0) {
        if (statusEl) {
            statusEl.innerText = 'No data found (check favorited items).';
            statusEl.style.color = '#ffaa00';
        }
        return;
    }

    if (statusEl) {
        statusEl.innerText = results.length + ' unique entries found.';
        statusEl.style.color = '#4caf50';
        setTimeout(() => { if (statusEl) statusEl.innerText = ''; }, 4000);
    }

    showAnalyticsModal(title, results, type);
}

/**
 * Generic counter for lora and model fields.
 * Splits field value by " + " to handle combined entries.
 * Tracks both favorited and total occurrences per entry.
 * @param {Array} items - all manifest items
 * @param {string} field - 'lora' or 'model'
 * @returns {Array} sorted [{favs, total, name}] descending by favs, then by percentage
 */
function analyzeFieldStats(items, field) {
    const favs = new Map();
    const totals = new Map();

    for (const item of items) {
        const raw = item[field];
        if (!raw || raw === 'None') continue;

        const entries = String(raw).split(' + ').map(s => s.trim()).filter(Boolean);
        for (const entry of entries) {
            totals.set(entry, (totals.get(entry) || 0) + 1);
            if (item.favorited) {
                favs.set(entry, (favs.get(entry) || 0) + 1);
            }
        }
    }

    return Array.from(totals.entries())
        .map(([name, total]) => ({ favs: favs.get(name) || 0, total, name }))
        .sort((a, b) =>
            b.favs - a.favs ||
            (b.total ? b.favs / b.total : 0) - (a.total ? a.favs / a.total : 0) ||
            a.name.localeCompare(b.name)
        );
}

/**
 * Full Prompt Stats: for each unique positive prompt, track favs and total.
 * @param {Array} items - all manifest items
 * @returns {Array} sorted [{favs, total, name}] descending
 */
function analyzePromptStats(items) {
    const favs = new Map();
    const totals = new Map();

    for (const item of items) {
        const prompt = item.positive;
        if (!prompt) continue;
        totals.set(prompt, (totals.get(prompt) || 0) + 1);
        if (item.favorited) {
            favs.set(prompt, (favs.get(prompt) || 0) + 1);
        }
    }

    return Array.from(totals.entries())
        .map(([name, total]) => ({ favs: favs.get(name) || 0, total, name }))
        .sort((a, b) =>
            b.favs - a.favs ||
            (b.total ? b.favs / b.total : 0) - (a.total ? a.favs / a.total : 0) ||
            a.name.localeCompare(b.name)
        );
}

/**
 * Tag Stats: splits item.positive by comma, trims, tracks favs + total per tag.
 * @param {Array} items - all manifest items
 * @returns {Array} sorted [{favs, total, name}] descending
 */
function analyzeTagStats(items) {
    const favs = new Map();
    const totals = new Map();

    for (const item of items) {
        const prompt = item.positive;
        if (!prompt) continue;

        const tags = prompt.split(',').map(t => t.trim()).filter(Boolean);
        for (const tag of tags) {
            totals.set(tag, (totals.get(tag) || 0) + 1);
            if (item.favorited) {
                favs.set(tag, (favs.get(tag) || 0) + 1);
            }
        }
    }

    return Array.from(totals.entries())
        .map(([name, total]) => ({ favs: favs.get(name) || 0, total, name }))
        .sort((a, b) =>
            b.favs - a.favs ||
            (b.total ? b.favs / b.total : 0) - (a.total ? a.favs / a.total : 0) ||
            a.name.localeCompare(b.name)
        );
}

/**
 * Escape HTML entities for safe innerHTML insertion.
 */
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/**
 * Show the analytics results modal.
 * Follows the same dynamic overlay pattern as showSaveErrorAlert.
 * @param {string} title - modal title
 * @param {Array} results - [{count, name}]
 * @param {string} type - 'lora'|'model'|'prompt'|'tags' (for accent color)
 */
function showAnalyticsModal(title, results, type) {
    // Deduplication: remove existing analytics modal
    const existing = document.getElementById('analytics-results-modal');
    if (existing) existing.remove();

    // Accent color per type
    const accentMap = {
        lora:   '#8b5cf6',
        model:  '#6366f1',
        prompt: '#00d1b2',
        tags:   '#d0873e'
    };
    const accent = accentMap[type] || '#00d1b2';

    // Build table rows HTML (limit to 500 rows max for performance)
    const displayResults = results.slice(0, 500);
    const totalItems = results.length;
    const truncated = results.length > 500;

    const rowsHtml = displayResults.map((r, i) => {
        const rank = i + 1;
        const pct = r.total > 0 ? Math.round((r.favs / r.total) * 100) : 0;
        const countText = r.favs + ' / ' + r.total + ' = ' + pct + '%';
        return '<tr class="analytics-row">' +
            '<td class="analytics-rank">' + rank + '</td>' +
            '<td class="analytics-count" style="color:' + accent + ';">' + countText + '</td>' +
            '<td class="analytics-name">' + escapeHtml(r.name) + '</td>' +
        '</tr>';
    }).join('');

    const truncatedNote = truncated
        ? '<div style="text-align:center; font-size:10px; color:#666; padding:8px 0;">' +
              'Showing top 500 of ' + totalItems + ' entries.' +
          '</div>'
        : '';

    // Count summary
    const favCount = results.filter(r => r.favs > 0).length;
    const totalFavs = results.reduce((s, r) => s + r.favs, 0);
    const totalOccurrences = results.reduce((s, r) => s + r.total, 0);
    const summaryHtml =
        '<div class="analytics-summary">' +
            '<span>' + totalItems + ' unique entries</span>' +
            '<span style="color:#666;">|</span>' +
            '<span>' + totalFavs + ' favorites / ' + totalOccurrences + ' total</span>' +
            '<span style="color:#666;">|</span>' +
            '<span>' + favCount + ' entries with favorites</span>' +
        '</div>';

    // For prompt type: build "Copy Favorited Prompts as JSON" button
    const copyFavBtnHtml = (type === 'prompt')
        ? '<button id="analytics-copy-fav-btn" style="' +
              'background:#00d1b2; color:#000; border:none; border-radius:5px;' +
              'padding:6px 12px; font-size:11px; font-weight:bold; cursor:pointer;' +
              'margin-right:8px; white-space:nowrap;' +
          '" title="Copy all favorited prompts as a JSON array">📋 Copy Favorited Prompts as JSON</button>'
        : '';

    // Create overlay
    const overlay = document.createElement('div');
    overlay.id = 'analytics-results-modal';
    overlay.className = 'analytics-modal-overlay';

    overlay.innerHTML =
        '<div class="analytics-modal-popup">' +
            '<div class="analytics-modal-header" style="border-bottom-color:' + accent + ';">' +
                '<span class="analytics-modal-title" style="color:' + accent + ';">' + escapeHtml(title) + '</span>' +
                '<div style="display:flex; align-items:center; gap:4px;">' +
                    copyFavBtnHtml +
                    '<button class="close-popup-btn" onclick="document.getElementById(\'analytics-results-modal\').remove()">&#10005;</button>' +
                '</div>' +
            '</div>' +
            summaryHtml +
            '<div class="analytics-modal-body">' +
                '<table class="analytics-table">' +
                    '<thead>' +
                        '<tr>' +
                            '<th class="analytics-th">#</th>' +
                            '<th class="analytics-th" style="color:' + accent + ';">Favs / Total = %</th>' +
                            '<th class="analytics-th">Name</th>' +
                        '</tr>' +
                    '</thead>' +
                    '<tbody>' +
                        rowsHtml +
                    '</tbody>' +
                '</table>' +
                truncatedNote +
            '</div>' +
        '</div>';

    document.body.appendChild(overlay);

    // Wire up "Copy Favorited Prompts as JSON" button (prompt type only)
    if (type === 'prompt') {
        const copyFavBtn = document.getElementById('analytics-copy-fav-btn');
        if (copyFavBtn) {
            copyFavBtn.addEventListener('click', async () => {
                // All results with favs > 0 are favorited prompts; include all (not just displayResults slice)
                const favoritedPrompts = results.filter(r => r.favs > 0).map(r => r.name);
                const jsonStr = JSON.stringify(favoritedPrompts, null, 2);
                try {
                    await navigator.clipboard.writeText(jsonStr);
                } catch {
                    // Fallback for browsers that block async clipboard
                    const ta = document.createElement('textarea');
                    ta.value = jsonStr;
                    ta.style.position = 'fixed';
                    ta.style.opacity = '0';
                    document.body.appendChild(ta);
                    ta.select();
                    document.execCommand('copy');
                    document.body.removeChild(ta);
                }
                const orig = copyFavBtn.textContent;
                copyFavBtn.textContent = '✅ Copied ' + favoritedPrompts.length + ' prompts!';
                copyFavBtn.style.background = '#00aa88';
                setTimeout(() => {
                    copyFavBtn.textContent = orig;
                    copyFavBtn.style.background = '#00d1b2';
                }, 2000);
            });
        }
    }

    // Click-outside-to-close
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) overlay.remove();
    });
}


// ============================================================
// IMAGE METRICS FOR AN EXISTING SESSION (background job, CPU only)
// ============================================================
var _sessionMetricsJob = { id: null, timer: null };

async function startSessionMetrics() {
    const btn = document.getElementById('compute-metrics-btn');
    const out = document.getElementById('compute-metrics-progress');
    if (_sessionMetricsJob.id) {
        // Second click while running = cancel
        try {
            await fetch('/config_tester/cancel_metrics', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_id: _sessionMetricsJob.id })
            });
        } catch (e) { /* ignore */ }
        if (out) out.textContent = 'Cancelling...';
        return;
    }
    const sessionName = document.getElementById('session-input')?.value || '';
    if (!sessionName) { alert('No session selected'); return; }
    if (out) { out.style.display = 'block'; out.textContent = 'Starting...'; }
    try {
        const resp = await fetch('/config_tester/compute_metrics', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_name: sessionName, fill_missing: true })
        });
        const result = await resp.json();
        if (result.error) { if (out) out.textContent = result.error; return; }
        _sessionMetricsJob.id = result.job_id;
        if (btn) btn.textContent = 'CANCEL (click again)';
        _sessionMetricsJob.timer = setInterval(async () => {
            try {
                const st = await (await fetch('/config_tester/metrics_status?job_id=' + _sessionMetricsJob.id)).json();
                if (out) out.textContent = 'Measuring ' + st.completed + '/' + st.total + '...';
                if (st.status !== 'running') {
                    clearInterval(_sessionMetricsJob.timer);
                    _sessionMetricsJob = { id: null, timer: null };
                    if (btn) btn.textContent = 'COMPUTE METRICS FOR THIS SESSION';
                    if (out) out.textContent = st.status === 'error' ? ('Error: ' + (st.error || 'unknown'))
                        : (st.status === 'cancelled' ? 'Cancelled at ' : 'Done: ') + st.completed + '/' + st.total
                          + (st.failed ? ' (' + st.failed + ' skipped)' : '') + (st.status === 'complete' ? '. Reloading...' : '');
                    if (st.status === 'complete' && typeof loadSession === 'function') loadSession();
                }
            } catch (e) { /* ignore poll errors */ }
        }, 1500);
    } catch (e) {
        if (out) out.textContent = 'Failed to start: ' + e;
    }
}
