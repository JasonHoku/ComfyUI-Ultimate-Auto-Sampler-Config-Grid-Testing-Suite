/**
 * EVENT LISTENERS - INCREMENTAL UPDATES
 * Only processes NEW data, doesn't refilter everything
 */

// Intercept Ctrl+S inside the dashboard iframe to prevent browser "Save As" dialog
// and forward the save intent to ComfyUI's parent window
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        e.stopPropagation();
        // Standalone page: nothing to forward to (edits already auto-save), just keep the browser's Save dialog away.
        if (!USCG_STANDALONE) window.parent.postMessage({ type: 'forward_ctrl_s' }, '*');
    }
});

// Track last update to prevent duplicates
let lastUpdateId = 0;

/**
 * Replace the whole manifest with `manifest` WITHOUT touching filters, viewport or the grid layout
 * (the non-destructive path used by update_data{manifest}, reload_manifest and the standalone resync).
 * A lazy shell (`__uscg_lazy__`: empty on purpose) is ignored so it can never blank the grid.
 */
function applyManifestReplace(manifest, newMeta) {
    if (!manifest || manifest.__uscg_lazy__) return;
    const knownIds = new Set((activeData || []).map(function (it) { return it.id; }));
    fullManifest = manifest;
    activeData = fullManifest.items || [];
    meta = newMeta || manifest.meta || {};
    if (typeof onComparisonsReplaced === 'function') onComparisonsReplaced();

    // Recompute label global values with new data
    if (typeof computeLabelGlobalValues === 'function' && labelMode && labelMode.enabled) {
        computeLabelGlobalValues();
    }

    refreshIndices();

    // Items we have not seen before may carry filter values (seed, sampler, ...) the filter sets do not
    // know yet: register them exactly like the live path does, or the pipeline would hide those cards.
    const fresh = activeData.filter(function (it) { return !knownIds.has(it.id); });
    if (fresh.length && typeof updateFiltersForNewData === 'function') updateFiltersForNewData(fresh);

    // Full reprocess needed
    updateDataPipeline();
}

var _etaHideTimer = null;

/** The ETA bar text for one progress event (pure). */
function formatProgressText(p) {
    if (p.complete) {
        if (p.next_phase === 'sweep') return `Standard jobs complete: ${p.total_generated} images in ${p.total_elapsed}. Workflow sweep next…`;
        return `Complete! ${p.total_generated} images in ${p.total_elapsed} (${p.avg_duration}s/job)`;
    }
    const phaseLabel = p.phase === 'upscaling' ? 'Upscaling' : (p.phase === 'sweep' ? 'Sweep job' : 'Job');
    const finishPart = p.finish_time ? ` | ~${p.finish_time}` : '';
    return `${phaseLabel} ${p.current_job}/${p.total_jobs} (${p.progress_pct}%) | ETA: ${p.eta_str}${finishPart} | ${p.avg_duration}s/job${p.errors ? ` | ${p.errors} error${p.errors === 1 ? '' : 's'}` : ''}${(p.state === 'paused' || p.state === 'cancelled') ? ` | ${p.state}` : ''}`;
}

// WebSocket Listener - OPTIMIZED
window.addEventListener('message', (event) => {
    // Handle real-time ETA/progress updates
    if (event.data && event.data.type === 'progress_update') {
        const p = event.data.data;
        if (!p) return;
        const etaBar = document.getElementById('eta-bar');
        if (etaBar) {
            etaBar.style.display = 'flex';
            const etaText = document.getElementById('eta-text');
            const etaProgress = document.getElementById('eta-progress');
            if (!p.complete || p.next_phase) { clearTimeout(_etaHideTimer); _etaHideTimer = null; }
            if (etaText) etaText.textContent = formatProgressText(p);
            if (etaProgress) etaProgress.style.width = p.complete ? '100%' : `${p.progress_pct}%`;
            if (p.complete && !p.next_phase) {
                // Generation complete: auto-hide after 30 seconds (any later progress event cancels it)
                clearTimeout(_etaHideTimer);
                _etaHideTimer = setTimeout(() => { if (etaBar) etaBar.style.display = 'none'; }, 30000);
            }
        }
        return;
    }

    // The Generator finished (or the node was refreshed): re-fetch the manifest. If-None-Match makes an
    // unchanged manifest cost one 304; a changed one replaces the data without resetting the view.
    if (event.data && event.data.type === 'reload_manifest') {
        const sessInput = document.getElementById('session-input');
        const sess = sessInput ? sessInput.value : '';
        if (!sess) return;
        fetchManifest(sess, { ifChanged: true }).then(function (res) {
            if (res.status === 200 && res.data) applyManifestReplace(res.data, res.data.meta);
        }).catch(function (e) { console.warn('[Events] reload_manifest failed:', e); });
        return;
    }

    if (event.data && event.data.type === 'update_data') {
        const payload = event.data.data;
        if (!payload) return;

        // 1. Handle NEW ITEMS (incremental)
        if (payload.new_items && payload.new_items.length > 0) {
            console.log(`[Events] 📥 Received ${payload.new_items.length} new items`);

            if (!fullManifest.items) fullManifest.items = [];

            // Add to fullManifest data source only (processNewData will add to activeData)
            fullManifest.items.unshift(...payload.new_items);

            // CRITICAL: Process only new items, don't refilter everything
            // processNewData handles: adding to activeData, refreshIndices, filter updates, and rendering
            if (typeof processNewData === 'function') {
                processNewData(payload.new_items);
            } else {
                // Fallback: sync activeData and do full reprocess
                activeData = fullManifest.items;
                if (typeof computeLabelGlobalValues === 'function' && labelMode && labelMode.enabled) {
                    computeLabelGlobalValues();
                }
                refreshIndices();
                updateFiltersForNewData(payload.new_items);
                updateDataPipeline();
            }

            lastUpdateId++;
            return;
        } 
        
        // 2. Handle FULL MANIFEST reload (rare)
        if (payload.manifest) {
            if (payload.manifest.__uscg_lazy__) return;  // lazy shell: no data, keep what we have
            console.log('[Events] 📚 Full manifest reload');
            applyManifestReplace(payload.manifest, payload.meta);
        }
    }
});