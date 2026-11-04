/**
 * OPTIMIZED DATA PIPELINE & SORT LOGIC
 * Fixes: Clean code (no window. prefixes), restores refreshIndices, enables Dual Sort
 */

// Threshold for switching to async pipeline processing (used by multiple functions)
const ASYNC_PIPELINE_THRESHOLD = 500;

// RESTORED: Helper to update index map
// Maps each item's ID to its sequential position (1-based) when sorted by ID.
// This provides a stable "original order" number for each card regardless of
// current sort/filter, used for "Go To Card #" and the #N tag on cards.
function refreshIndices() {
    if (!activeData) return;
    const sorted = activeData.slice().sort((a, b) => (a.gen_index ?? a.id) - (b.gen_index ?? b.id));
    idToIndexMap = new Map(sorted.map((item, index) => [item.id, index + 1]));
}

// ============================================================
// LOGIC FILTER HELPERS (numeric field extraction + comparison)
// ============================================================

/**
 * Extract a numeric field value from a manifest item.
 * Returns null if the field is missing or not parseable as a number.
 */
function _extractFieldValue(item, field) {
    if (field === 'lora_strength') {
        // Parse first lora's model strength from "name:0.5:0.7" format
        const lora = item.lora;
        if (!lora || lora === 'None') return null;
        const firstPart = String(lora).split(' + ')[0];
        const parts = firstPart.split(':');
        if (parts.length < 2) return null;
        const v = parseFloat(parts[1]);
        return isNaN(v) ? null : v;
    }
    // Simple top-level numeric fields
    const v = item[field];
    if (typeof v === 'number') return v;
    if (typeof v === 'string') {
        const n = parseFloat(v);
        return isNaN(n) ? null : n;
    }
    return null;
}

/**
 * Evaluate a comparison: itemValue op filterValue.
 * Returns true if the item passes the comparison, false otherwise.
 */
function _evalOp(itemValue, op, filterValue) {
    const a = parseFloat(itemValue);
    const b = parseFloat(filterValue);
    if (isNaN(a) || isNaN(b)) return false;
    switch (op) {
        case '>':  return a > b;
        case '<':  return a < b;
        case '>=': return a >= b;
        case '<=': return a <= b;
        case '=':  return Math.abs(a - b) < 1e-6;
        case '!=': return Math.abs(a - b) >= 1e-6;
        default:   return true;
    }
}

/**
 * Check if an item passes all active logic filters (AND logic).
 * Returns true if no logic filters are active or all pass.
 */
function matchesLogicFilters(item) {
    if (typeof logicFilters === 'undefined' || logicFilters.length === 0) return true;
    for (const f of logicFilters) {
        const itemValue = _extractFieldValue(item, f.field);
        // If field is missing/null on the item, skip this filter (don't exclude)
        if (itemValue === null || itemValue === undefined) continue;
        if (!_evalOp(itemValue, f.op, f.value)) return false;
    }
    return true;
}

// ============================================================
// QUICK FILTER HELPERS (faceted chip filtering)
// ============================================================

/**
 * Test whether a single item matches a single quick-filter chip.
 * - lora: matches if the chip value is one of the lora names in the item's stack.
 * - model: exact equality on the full model string.
 * - numeric types (cfg/steps/denoise/seed): float tolerance 1e-6.
 * - everything else: String exact match.
 */
function _quickFilterMatches(item, f) {
    if (f.type === 'lora') {
        const lora = String(item.lora || 'None');
        if (lora === 'None') return f.value === 'None';
        const names = lora.split(' + ').map(p => p.split(':')[0].trim());
        return names.includes(f.value);
    }
    if (f.type === 'model') {
        return String(item.model || '') === String(f.value);
    }
    if (f.type === 'lora_strength') {
        const lora = String(item.lora || 'None');
        if (lora === 'None') return false;
        const firstPart = lora.split(' + ')[0];
        const parts = firstPart.split(':');
        if (parts.length < 2) return false;
        const itemStrength = parseFloat(parts[1]);
        const filterStrength = parseFloat(f.value);
        if (isNaN(itemStrength) || isNaN(filterStrength)) return false;
        return Math.abs(itemStrength - filterStrength) < 1e-6;
    }
    if (['cfg', 'steps', 'denoise', 'seed'].includes(f.type)) {
        const a = parseFloat(item[f.type]);
        const b = parseFloat(f.value);
        if (isNaN(a) || isNaN(b)) return String(item[f.type]) === String(f.value);
        return Math.abs(a - b) < 1e-6;
    }
    if (f.type === 'size') {
        const itemSize = `${item.width || '?'}x${item.height || '?'}`;
        return itemSize === String(f.value);
    }
    return String(item[f.type]) === String(f.value);
}

/**
 * Check if an item passes ALL active quick-filter chips (AND logic).
 * Returns true when no chips are active, or every chip matches.
 */
function matchesQuickFilters(item) {
    if (typeof quickFilters === 'undefined' || quickFilters.length === 0) return true;
    for (const f of quickFilters) {
        if (!_quickFilterMatches(item, f)) return false;
    }
    return true;
}

// Generate cache key from current filters
function getFilterKey() {
    const parts = [
        currentSort,
        currentSecondarySort,
        topFilters.showFavorites ? '1' : '0',
        topFilters.showNonFavorited ? '1' : '0',
        topFilters.showRejected ? '1' : '0',
        [...filters.sampler].sort().join(','),
        [...filters.scheduler].sort().join(','),
        [...filters.denoise].sort().join(','),
        [...filters.lora].sort().join(','),
        [...filters.model].sort().join(','),
        [...filters.positive].sort().join(','),
        [...filters.negative].sort().join(','),
        [...filters.size].sort().join(','),
        [...filters.seed].sort().join(','),
        [...filters.steps].sort().join(','),
        [...filters.cfg].sort().join(','),
        [...filters.upscaleMethod].sort().join(','),
        [...filters.mediaType].sort().join(','),
        [...(filters.passType || [])].sort().join(',') + '#' + supersededFps.size,
        searchFilters.map(f => `${f.type}:${f.term}`).join('|'),
        (typeof logicFilters !== 'undefined' ? logicFilters.map(f => `${f.field}${f.op}${f.value}`).join('|') : ''),
        (typeof quickFilters !== 'undefined' ? quickFilters.map(f => `${f.type}:${f.value}`).join('|') : '')
    ];
    return parts.join('|');
}

// --- 3. SORT UI LOGIC ---

function initSortUI() {
    console.log('[initSortUI] Called');
    
    const pList = document.getElementById('sort-list-primary');
    const sList = document.getElementById('sort-list-secondary');
    
    console.log('[initSortUI] Elements found:', {
        pList: !!pList,
        sList: !!sList,
        currentSort,
        currentSecondarySort
    });
    
    // Safety check: if HTML elements are missing, stop
    if(!pList || !sList) {
        console.error('[initSortUI] Missing elements! Cannot initialize sort UI');
        return;
    }

    // Helper to create options
    const renderOptions = (container, isSecondary) => {
        console.log(`[initSortUI] Rendering ${isSecondary ? 'secondary' : 'primary'} options`);
        container.innerHTML = ''; // Clear existing
        
        // Add "None" option for secondary only
        if (isSecondary) {
            const div = document.createElement('div');
            div.className = `sort-option ${currentSecondarySort === 'none' ? 'secondary-selected' : ''}`;
            div.textContent = 'None';
            div.onclick = () => setSecondarySort('none');
            container.appendChild(div);
        }

        sortOptionsList.forEach(opt => {
            const div = document.createElement('div');
            div.className = 'sort-option';
            div.textContent = opt.label;
            
            // Interaction Logic
            if (!isSecondary) {
                // Primary Column
                if (currentSort === opt.id) div.classList.add('selected');
                div.onclick = () => setPrimarySort(opt.id);
            } else {
                // Secondary Column
                if (currentSecondarySort === opt.id) div.classList.add('secondary-selected');
                
                // Disable if same as primary
                if (currentSort === opt.id) {
                    div.style.opacity = '0.3';
                    div.style.pointerEvents = 'none';
                }
                div.onclick = () => setSecondarySort(opt.id);
            }
            container.appendChild(div);
        });
        
        console.log(`[initSortUI] Rendered ${container.children.length} options for ${isSecondary ? 'secondary' : 'primary'}`);
    };

    renderOptions(pList, false);
    renderOptions(sList, true);
    
    // Update the main button label
    const primLabel = sortOptionsList.find(x => x.id === currentSort)?.label || 'Oldest';
    const secLabel = sortOptionsList.find(x => x.id === currentSecondarySort)?.label;
    
    const btnLbl = document.getElementById('sort-primary-label');
    if(btnLbl) {
        const secHtml = (secLabel && currentSecondarySort !== 'none') ? '+ ' + secLabel : '';
        btnLbl.innerHTML = `${primLabel} <span style="color:#666; font-weight:400; font-size:9px;">${secHtml}</span>`;
        console.log('[initSortUI] Updated button label:', btnLbl.innerHTML);
    }
}

function toggleSortPopup() {
    console.log('[toggleSortPopup] Called');
    
    const popup = document.getElementById('sort-popup');
    const btn = document.getElementById('sort-main-btn');
    
    if (!popup) {
        console.error('[toggleSortPopup] Popup element not found!');
        return;
    }
    
    const isHidden = popup.style.display === 'none' || !popup.style.display;
    console.log('[toggleSortPopup] Current state:', isHidden ? 'hidden' : 'visible');
    
    if (isHidden) {
        console.log('[toggleSortPopup] Opening popup');
        initSortUI(); 
        popup.style.display = 'flex';
        if(btn) btn.classList.add('active');
    } else {
        console.log('[toggleSortPopup] Closing popup');
        closeSortPopup();
    }
}

function closeSortPopup() {
    const popup = document.getElementById('sort-popup');
    const btn = document.getElementById('sort-main-btn');
    if (popup) popup.style.display = 'none';
    if (btn) btn.classList.remove('active');
}

function setPrimarySort(id) {
    console.log('[setPrimarySort]', id);
    currentSort = id;
    localStorage.setItem('ultimate_grid_sort', currentSort);
    initSortUI(); 
    updateDataPipeline();
}

function setSecondarySort(id) {
    console.log('[setSecondarySort]', id);
    currentSecondarySort = id;
    localStorage.setItem('ultimate_grid_sort_sec', currentSecondarySort);
    closeSortPopup(); 
    initSortUI();
    updateDataPipeline();
}

// Unified Comparison Helper
function compareItems(a, b, sortKey) {
    switch (sortKey) {
        case 'newest': return (b.gen_index ?? b.id) - (a.gen_index ?? a.id);
        case 'oldest': return (a.gen_index ?? a.id) - (b.gen_index ?? b.id);
        case 'fastest': return a.duration - b.duration;
        case 'favorited': return (b.favorited ? 1 : 0) - (a.favorited ? 1 : 0);
        case 'cfg': return a.cfg - b.cfg;
        case 'denoise': return a.denoise - b.denoise;
        case 'seed': return a.seed - b.seed;
        case 'model': 
            return (a.model || meta.model || "Default").toLowerCase().localeCompare((b.model || meta.model || "Default").toLowerCase());
        case 'prompt': 
            return (a.positive || meta.positive || "").toLowerCase().localeCompare((b.positive || meta.positive || "").toLowerCase());
        case 'lora': 
            return (a.lora || "None").toLowerCase().localeCompare((b.lora || "None").toLowerCase());
        case 'sampler': 
            return (a.sampler || "").toLowerCase().localeCompare((b.sampler || "").toLowerCase());
        case 'size':
            return (a.width * a.height) - (b.width * b.height);
        case 'rating':
            return (typeof compareRatings === 'function') ? compareRatings(a, b) : 0;
        default: return 0;
    }
}

// Master Sort Function
function runMultiSort(list) {
    list.sort((a, b) => {
        // 1. Primary Sort
        let res = compareItems(a, b, currentSort);
        
        // 2. Secondary Sort (if Primary is tied and Secondary is active)
        if (res === 0 && currentSecondarySort !== 'none' && currentSecondarySort !== currentSort) {
            res = compareItems(a, b, currentSecondarySort);
        }
        
        // 3. Fallback to gen_index/ID (Stable sort, backwards-compatible)
        if (res === 0) {
            return (a.gen_index ?? a.id) - (b.gen_index ?? b.id);
        }
        return res;
    });
}

// --- 4. PIPELINE & FILTER LOGIC ---

// Preview pass support -------------------------------------------------------
// A preview is "superseded" once a full (non-preview) item exists for the same cell fingerprint.
// Superseded previews are hidden by default; isolating the Pass filter on "Preview" (without "Full") shows them.
function isSupersededPreview(d) {
    if (!d.preview || !d.preview_of || !supersededFps.has(d.preview_of)) return false;
    const pf = filters.passType;
    if (pf && pf.has('Preview') && !pf.has('Full')) return false;
    return true;
}

// newItems omitted: rebuild the whole set from activeData. With newItems: add only their fingerprints and
// return true when that superseded a preview that is currently displayed (caller then re-runs the pipeline).
function refreshSupersededFps(newItems) {
    if (!newItems) {
        const s = new Set();
        for (let i = 0; i < activeData.length; i++) {
            const d = activeData[i];
            if (d.preview) continue;
            if (d.config_fp) s.add(d.config_fp);
            if (d.base_fp) s.add(d.base_fp);
        }
        supersededFps = s;
        return false;
    }
    let grew = false;
    for (let i = 0; i < newItems.length; i++) {
        const d = newItems[i];
        if (d.preview) continue;
        if (d.config_fp && !supersededFps.has(d.config_fp)) { supersededFps.add(d.config_fp); grew = true; }
        if (d.base_fp && !supersededFps.has(d.base_fp)) { supersededFps.add(d.base_fp); grew = true; }
    }
    if (!grew) return false;
    return processedData.some(p => p.preview && supersededFps.has(p.preview_of));
}

// Shows "N previews | M awaiting a full pass" + the full-pass button when the session holds previews.
function updatePreviewBar() {
    const bar = document.getElementById('preview-bar');
    if (!bar) return;
    let previews = 0, survivors = 0;
    for (let i = 0; i < activeData.length; i++) {
        const d = activeData[i];
        if (!d.preview) continue;
        previews++;
        if (!d.rejected && !supersededFps.has(d.preview_of)) survivors++;
    }
    if (previews === 0) { bar.style.display = 'none'; return; }
    bar.style.display = 'flex';
    const txt = document.getElementById('preview-bar-text');
    if (txt) txt.textContent = `${previews} preview image${previews === 1 ? '' : 's'} | ${survivors} not rejected and awaiting a full pass`;
    const btn = document.getElementById('full-pass-btn');
    if (btn) {
        btn.textContent = `\u25B6 Full pass for survivors (${survivors})`;
        btn.disabled = survivors === 0;
    }
}

function incrementalFilter(newItems) {
    const hasButtonFilters = filters.sampler.size > 0 ||
        filters.scheduler.size > 0 ||
        filters.denoise.size > 0 ||
        filters.lora.size > 0 ||
        filters.model.size > 0 ||
        filters.positive.size > 0 ||
        filters.negative.size > 0 ||
        filters.size.size > 0 ||
        filters.seed.size > 0 ||
        filters.steps.size > 0 ||
        filters.cfg.size > 0 ||
        filters.upscaleMethod.size > 0 ||
        filters.mediaType.size > 0 ||
        (filters.passType && filters.passType.size > 0);

    return newItems.filter(d => {
        // Apply top-level filters (Favorites/Non-Favorited/Rejected visibility)
        if (d.favorited && !topFilters.showFavorites) return false;
        if (!d.favorited && !d.rejected && !topFilters.showNonFavorited) return false;
        if (d.rejected && !topFilters.showRejected) return false;
        if (d.preview && isSupersededPreview(d)) return false;

        // Apply button filters
        if (hasButtonFilters) {
            if (filters.sampler.size > 0 && !filters.sampler.has(d.sampler)) return false;
            if (filters.scheduler.size > 0 && !filters.scheduler.has(d.scheduler)) return false;
            if (filters.denoise.size > 0 && !filters.denoise.has(d.denoise)) return false;
            if (filters.lora.size > 0 && !filters.lora.has(d.lora)) return false;
            if (filters.model.size > 0 && !filters.model.has(d.model || meta.model || "Default")) return false;
            if (filters.positive.size > 0) {
                const st = document.getElementById('toggle-triggers')?.checked !== false;
                const posValue = st ? (d.positive || meta.positive || "") : (d.config_positive || d.positive || meta.positive || "");
                if (!filters.positive.has(posValue)) return false;
            }
            if (filters.negative.size > 0) {
                const st = document.getElementById('toggle-triggers')?.checked !== false;
                const negValue = st ? (d.negative || meta.negative || "") : (d.config_negative || d.negative || meta.negative || "");
                if (!filters.negative.has(negValue)) return false;
            }
            if (filters.size.size > 0) {
                const sizeStr = `${d.width}x${d.height}`;
                if (!filters.size.has(sizeStr)) return false;
            }
            if (filters.seed.size > 0 && !filters.seed.has(d.seed)) return false;
            if (filters.steps.size > 0 && !filters.steps.has(String(d.steps))) return false;
            if (filters.cfg.size > 0 && !filters.cfg.has(String(d.cfg))) return false;
            if (filters.upscaleMethod.size > 0) {
                let upVal;
                if (!d.upscaled) { upVal = 'No Upscale'; }
                else {
                    const mode = d.upscale_mode || '';
                    const model = d.upscale_model;
                    const shortModel = model ? String(model).replace(/\\/g, '/').split('/').pop().replace(/\.[^.]+$/, '') : '';
                    upVal = shortModel ? `${mode} + ${shortModel}` : mode || 'Upscaled';
                }
                if (!filters.upscaleMethod.has(upVal)) return false;
            }
            if (filters.mediaType.size > 0) {
                const mt = d.media_type || 'image';
                if (!filters.mediaType.has(mt)) return false;
            }
            if (filters.passType && filters.passType.size > 0) {
                if (!filters.passType.has(d.preview ? 'Preview' : 'Full')) return false;
            }
        }

        // Apply search filters
        if (typeof matchesSearchFilters === 'function') {
            if (!matchesSearchFilters(d)) return false;
        }

        // Apply logic filters (numeric comparisons)
        if (!matchesLogicFilters(d)) return false;

        // Apply quick filters (faceted chip AND logic)
        if (!matchesQuickFilters(d)) return false;

        return true;
    });
}

function updateDataPipeline() {
    if (isPipelinePending) return;
    isPipelinePending = true;

    requestAnimationFrame(() => {
        executePipeline();
        isPipelinePending = false;
    });
}

// Abort controller for canceling in-progress async pipelines
let pipelineAbortId = 0;

function executePipeline() {
    const startTime = performance.now();
    refreshSupersededFps();
    const currentFilterKey = getFilterKey();
    const filtersChanged = currentFilterKey !== lastFilterKey;

    // Always filter from activeData to prevent data loss from self-filtering
    console.log(`[Pipeline] ${filtersChanged ? 'Filters changed, full reprocess' : 'Re-applying filters'}`);
    if (filtersChanged) {
        lastFilterKey = currentFilterKey;
    }

    // Check for button filters once to avoid repeated Set.size lookups
    const hasButtonFilters = filters.sampler.size > 0 ||
        filters.scheduler.size > 0 ||
        filters.denoise.size > 0 ||
        filters.lora.size > 0 ||
        filters.model.size > 0 ||
        filters.positive.size > 0 ||
        filters.negative.size > 0 ||
        filters.size.size > 0 ||
        filters.seed.size > 0 ||
        filters.steps.size > 0 ||
        filters.cfg.size > 0 ||
        filters.upscaleMethod.size > 0 ||
        filters.mediaType.size > 0 ||
        (filters.passType && filters.passType.size > 0);

    const hasSearchFilters = typeof matchesSearchFilters === 'function' && searchFilters.length > 0;

    processedData = activeData.filter(d => {
        // Apply top-level filters (Favorites/Non-Favorited/Rejected visibility)
        if (d.favorited && !topFilters.showFavorites) return false;
        if (!d.favorited && !d.rejected && !topFilters.showNonFavorited) return false;
        if (d.rejected && !topFilters.showRejected) return false;
        if (d.preview && isSupersededPreview(d)) return false;

        // Apply button filters (skip entirely if none are active)
        if (hasButtonFilters) {
            if (filters.sampler.size > 0 && !filters.sampler.has(d.sampler)) return false;
            if (filters.scheduler.size > 0 && !filters.scheduler.has(d.scheduler)) return false;
            if (filters.denoise.size > 0 && !filters.denoise.has(d.denoise)) return false;
            if (filters.lora.size > 0 && !filters.lora.has(d.lora)) return false;
            if (filters.model.size > 0 && !filters.model.has(d.model || meta.model || "Default")) return false;
            if (filters.positive.size > 0) {
                const st = document.getElementById('toggle-triggers')?.checked !== false;
                const posValue = st ? (d.positive || meta.positive || "") : (d.config_positive || d.positive || meta.positive || "");
                if (!filters.positive.has(posValue)) return false;
            }
            if (filters.negative.size > 0) {
                const st = document.getElementById('toggle-triggers')?.checked !== false;
                const negValue = st ? (d.negative || meta.negative || "") : (d.config_negative || d.negative || meta.negative || "");
                if (!filters.negative.has(negValue)) return false;
            }
            if (filters.size.size > 0 && !filters.size.has(`${d.width}x${d.height}`)) return false;
            if (filters.seed.size > 0 && !filters.seed.has(d.seed)) return false;
            if (filters.steps.size > 0 && !filters.steps.has(String(d.steps))) return false;
            if (filters.cfg.size > 0 && !filters.cfg.has(String(d.cfg))) return false;
            if (filters.upscaleMethod.size > 0) {
                let upVal;
                if (!d.upscaled) { upVal = 'No Upscale'; }
                else {
                    const mode = d.upscale_mode || '';
                    const model = d.upscale_model;
                    const shortModel = model ? String(model).replace(/\\/g, '/').split('/').pop().replace(/\.[^.]+$/, '') : '';
                    upVal = shortModel ? `${mode} + ${shortModel}` : mode || 'Upscaled';
                }
                if (!filters.upscaleMethod.has(upVal)) return false;
            }
            if (filters.mediaType.size > 0) {
                const mt = d.media_type || 'image';
                if (!filters.mediaType.has(mt)) return false;
            }
            if (filters.passType && filters.passType.size > 0) {
                if (!filters.passType.has(d.preview ? 'Preview' : 'Full')) return false;
            }
        }

        if (hasSearchFilters && !matchesSearchFilters(d)) return false;

        // Apply logic filters (numeric comparisons)
        if (!matchesLogicFilters(d)) return false;

        // Apply quick filters (faceted chip AND logic)
        if (!matchesQuickFilters(d)) return false;

        return true;
    });

    // Apply sorting
    // Fast path: for oldest/newest with no secondary sort, activeData is stored newest-first
    // (items are inserted/unshifted at position 0), so processedData after filter() is also newest-first.
    // For 'oldest' we reverse; for 'newest' it's already correct.
    if ((currentSort === 'oldest' || currentSort === 'newest') && currentSecondarySort === 'none') {
        if (currentSort === 'oldest') {
            processedData.reverse();
        }
    } else {
        runMultiSort(processedData);
    }

    const elapsed = (performance.now() - startTime).toFixed(1);
    console.log(`[Pipeline] Processed ${processedData.length} items in ${elapsed}ms`);
    updateJSONs(processedData);
    updatePreviewBar();

    if (filtersChanged) {
        if(typeof renderDOM === 'function') renderDOM();
    } else {
        if (typeof updateVisibleItems === 'function') updateVisibleItems();
    }
}

// OPTIMIZED: Process new data incrementally
// NOTE: Caller (logic_events.js) already adds newItems to fullManifest.items
// which IS activeData (same array reference from init). So we must NOT push
// to activeData again here to avoid duplicate entries.
function processNewData(newItems) {
    if (!newItems || newItems.length === 0) return;

    // IMPORTANT: Update indices when new data arrives
    // (activeData already contains the new items via fullManifest.items reference)
    refreshIndices();

    // A full image landing for a cell supersedes its preview: if any preview on screen just became
    // superseded, re-run the whole pipeline (it rebuilds the superseded set and hides them).
    if (refreshSupersededFps(newItems)) {
        updateFiltersForNewData(newItems);
        updateDataPipeline();
        return;
    }
    updatePreviewBar();

    // Recompute label global values when data changes (new items may change what's unique)
    // Then refresh ALL existing card labels since what's "unique" may have changed
    if (typeof computeLabelGlobalValues === 'function' && labelMode && labelMode.enabled) {
        computeLabelGlobalValues();
        if (typeof refreshLabelOverlays === 'function') {
            refreshLabelOverlays();
        }
    }

    // Update filter options
    updateFiltersForNewData(newItems);

    // Process pipeline
    const filtered = incrementalFilter(newItems);
    if (filtered.length === 0) {
        // All new items were filtered out, just update JSONs
        updateJSONs(processedData);
        return;
    }

    let prependedToTop = false;

    // Logic for Prepend/Append based on Sort Mode
    if (currentSort === 'newest') {
        // Prepend to beginning
        processedData.unshift(...filtered);
        // Only consider it "prepended" (stable scroll) if we aren't sub-sorting
        if(currentSecondarySort === 'none') prependedToTop = true;
    } else if (currentSort === 'oldest') {
        // Append to end
        processedData.push(...filtered);
    } else {
        // For other sort modes with large datasets, use binary insertion instead of full re-sort
        if (processedData.length > ASYNC_PIPELINE_THRESHOLD && filtered.length < 50) {
            // Binary insert each new item into already-sorted array (much faster than re-sorting all)
            for (const item of filtered) {
                binaryInsert(processedData, item);
            }
        } else {
            // Small dataset or large batch of new items: just append and re-sort
            processedData.push(...filtered);
            runMultiSort(processedData);
        }
    }

    updateJSONs(processedData);

    if (prependedToTop && typeof forceVisibleRangeUpdate === 'function') {
        forceVisibleRangeUpdate(filtered.length);
    } else if (typeof updateVisibleItems === 'function') {
        updateVisibleItems();
    }
}

// Binary insert a single item into an already-sorted array
function binaryInsert(arr, item) {
    let low = 0;
    let high = arr.length;

    while (low < high) {
        const mid = (low + high) >>> 1;
        let cmp = compareItems(arr[mid], item, currentSort);
        if (cmp === 0 && currentSecondarySort !== 'none' && currentSecondarySort !== currentSort) {
            cmp = compareItems(arr[mid], item, currentSecondarySort);
        }
        if (cmp === 0) {
            cmp = (arr[mid].gen_index ?? arr[mid].id) - (item.gen_index ?? item.id); // Stable sort fallback
        }
        if (cmp <= 0) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    arr.splice(low, 0, item);
}

// Load sort order from localStorage
function loadSortPreference() {
    const savedSort = localStorage.getItem('ultimate_grid_sort');
    const savedSec = localStorage.getItem('ultimate_grid_sort_sec');
    
    const validSortOptions = sortOptionsList.map(o => o.id);

    if (savedSort && validSortOptions.includes(savedSort)) {
        currentSort = savedSort;
    }
    
    if (savedSec) {
        currentSecondarySort = savedSec;
    }

    console.log(`[loadSortPreference] Loaded sort: ${currentSort}, Secondary: ${currentSecondarySort}`);
    
    // Initialize the UI after a brief delay to ensure DOM is ready
    setTimeout(() => {
        console.log('[loadSortPreference] Initializing Sort UI after delay');
        initSortUI();
    }, 100);
}

function updateFiltersForNewData(newItems) {
    // Track which filter categories actually got new unique values
    let changed = false;
    const filterKeys = ['model', 'sampler', 'scheduler', 'denoise', 'lora', 'positive', 'negative', 'size', 'seed', 'passType'];

    // Build a set of new values for each key in a single pass over newItems
    for (let i = 0; i < newItems.length; i++) {
        const d = newItems[i];
        for (let k = 0; k < filterKeys.length; k++) {
            const key = filterKeys[k];
            let val;
            if (key === 'model') val = d.model || meta.model || "Default";
            else if (key === 'positive') val = d.positive || meta.positive || "";
            else if (key === 'negative') val = d.negative || meta.negative || "";
            else if (key === 'size') val = `${d.width}x${d.height}`;
            else if (key === 'passType') val = d.preview ? 'Preview' : 'Full';
            else val = d[key];

            if (val !== undefined && val !== null && !(filters[key] && filters[key].has(val))) {
                changed = true;
                // Don't break - we need to continue checking all items/keys
                // so that initFilters gets a complete picture
            }
        }
        if (changed) break; // Once we know we need to rebuild, no need to check more items
    }

    if (changed && typeof initFilters === 'function') {
        initFilters(); // Re-render filter buttons
    }
}

// --- 5. INITIALIZATION ---
// Ensure init fires after DOM is ready
// Note: init() in logic_init.js handles the full initialization including
// showSessionLandingIfEmpty() for sessions with no data. Only call loadSession()
// here when we have embedded data to reload from disk.
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        if (fullManifest && fullManifest.items && fullManifest.items.length > 0) {
            console.log('[Init] DOMContentLoaded - calling loadSession');
            loadSession();
        }
    });
} else {
    if (fullManifest && fullManifest.items && fullManifest.items.length > 0) {
        console.log('[Init] DOM already loaded - calling loadSession');
        loadSession();
    }
}