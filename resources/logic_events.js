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
            if (p.complete) {
                // Generation complete
                if (etaText) etaText.textContent = `Complete! ${p.total_generated} images in ${p.total_elapsed} (${p.avg_duration}s/job)`;
                if (etaProgress) etaProgress.style.width = '100%';
                // Auto-hide after 30 seconds
                setTimeout(() => { if (etaBar) etaBar.style.display = 'none'; }, 30000);
            } else {
                if (etaText) {
                    const phaseLabel = p.phase === 'upscaling' ? 'Upscaling' : (p.phase === 'sweep' ? 'Sweep job' : 'Job');
                    const finishPart = p.finish_time ? ` | ~${p.finish_time}` : '';
                    etaText.textContent = `${phaseLabel} ${p.current_job}/${p.total_jobs} (${p.progress_pct}%) | ETA: ${p.eta_str}${finishPart} | ${p.avg_duration}s/job${p.errors ? ` | ${p.errors} error${p.errors === 1 ? '' : 's'}` : ''}${(p.state === 'paused' || p.state === 'cancelled') ? ` | ${p.state}` : ''}`;
                }
                if (etaProgress) {
                    etaProgress.style.width = `${p.progress_pct}%`;
                }
            }
        }
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
            console.log('[Events] 📚 Full manifest reload');
            fullManifest = payload.manifest;
            activeData = fullManifest.items || [];
            meta = payload.meta || {};
            if (typeof onComparisonsReplaced === 'function') onComparisonsReplaced();

            // Recompute label global values with new data
            if (typeof computeLabelGlobalValues === 'function' && labelMode && labelMode.enabled) {
                computeLabelGlobalValues();
            }

            refreshIndices();

            // Full reprocess needed
            updateDataPipeline();
        }
    }
});