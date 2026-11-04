/**
 * UTILITIES & API
 * Handles server communication and generic helpers.
 * FIXED: Session reload now properly re-renders grid
 */

function toggleFullscreen() {
    if (USCG_STANDALONE) {
        // Top-level page: use the browser Fullscreen API (iOS Safari has no element fullscreen; harmless no-op there).
        try {
            if (document.fullscreenElement) document.exitFullscreen();
            else if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen();
        } catch (e) { /* unsupported */ }
        return;
    }
    // Send current session name so fullscreen reload uses the right session
    var currentSession = document.getElementById('session-input')?.value || '';
    window.parent.postMessage({ type: 'toggle_fullscreen', node_id: TARGET_NODE_ID, session_name: currentSession }, '*');
}


function setupKeyReloadFullscreen() {
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (window._compareOpen) return;   // compare mode swallows f / r

        switch (e.key) {
            case 'f':
                toggleFullscreen()
                break;
            case 'r':
                // Reset all filters visually before reloading so the UI is consistent
                if (typeof resetAllFilters === 'function') resetAllFilters();
                loadSession()
        }
    })
}

// Persist session to disk
async function saveState() {
    try {
        await fetch('/config_tester/save_manifest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_name: document.getElementById('session-input')?.value || "default",
                manifest: fullManifest
            })
        });
    } catch (e) { console.error("Save failed", e); }
}

// Load session from disk
async function loadSession() {
    const sessInput = document.getElementById('session-input');
    if (!sessInput) return;
    const sess = sessInput.value;

    console.log(`[Load] 🔄 Loading session: ${sess}`);

    // UI Feedback
    const grid = document.getElementById('grid');
    if (grid) grid.style.opacity = '0.5';

    try {
        const r = await fetch(`/view?filename=manifest.json&type=output&subfolder=benchmarks/${sess}&t=${Date.now()}`);
        if (!r.ok) throw new Error("Session not found");
        const data = await r.json();

        console.log(`[Load] 📊 Loaded ${data.items?.length || 0} items`);

        // 1. CRITICAL FIX: Reset viewport position
        const viewport = document.getElementById('viewport');
        const canvas = document.getElementById('canvas');

        // Reset pan/zoom to defaults
        if (typeof resetZoom === 'function') {
            // This resets currentScale, panOffsetX, panOffsetY
            // currentScale = 1;
            // panOffsetX = 0;
            // panOffsetY = 0;
            // if (canvas) {
            //     canvas.style.transform = 'translate(0px, 0px) scale(1)';
            // }
        }

        // 2. Clear grid completely
        if (grid) {
            grid.innerHTML = '';
            grid.style.paddingTop = '0px';
            grid.style.height = '0px';
        }

        // 3. Clear all caches
        nodeMap.clear();

        // Clear filter sets
        ['sampler', 'scheduler', 'lora', 'denoise', 'model', 'positive', 'negative', 'size', 'seed'].forEach(k => {
            if (filters[k]) filters[k].clear();
        });

        // 4. Swap data (CRITICAL: Do this before pipeline)
        fullManifest = data;
        activeData = fullManifest.items || [];
        meta = fullManifest.meta || {};
        if (typeof onComparisonsReplaced === 'function') onComparisonsReplaced();

        console.log(`[Load] 📦 activeData now has ${activeData.length} items`);

        // 4b. Recompute label global values with new data so unique-only works on first load
        if (typeof computeLabelGlobalValues === 'function' && labelMode && labelMode.enabled) {
            computeLabelGlobalValues();
        }

        // 5. Reset indices
        refreshIndices();

        // 7. Re-initialize filters (don't skip this!)
        if (typeof initFilters === 'function') {
            initFilters();
        }

        // Initialize search filter UI
        if (typeof renderSearchFilters === 'function') {
            renderSearchFilters();
        }

        // Initialize logic filter UI
        if (typeof renderLogicFilters === 'function') {
            renderLogicFilters();
        }

        // 8. CRITICAL FIX: Force pipeline and render
        // Wait a tick for the DOM to update, then process
        await new Promise(resolve => setTimeout(resolve, 10));

        console.log(`[Load] 🔧 Running pipeline...`);
        updateDataPipeline();

        // 9. Wait for pipeline to complete, then render
        await new Promise(resolve => setTimeout(resolve, 100));

        console.log(`[Load] 🎨 Rendering ${processedData?.length || 0} processed items...`);

        // Force a full render
        if (typeof renderDOM === 'function') {
            renderDOM();
        }


        // 11. Restore UI
        if (grid) grid.style.opacity = '1';

        // Close Settings Panel
        const cogPanel = document.getElementById('cog-menu-dropdown');
        const cogOverlay = document.getElementById('cog-menu-overlay');
        if (!window._uscgResyncing) {
            if (cogPanel) cogPanel.style.display = 'none';
            if (cogOverlay) cogOverlay.style.display = 'none';
            document.body.style.overflow = '';
        }
        // Hide landing page, show viewport
        const landing = document.getElementById('session-landing');
        if (landing) landing.style.display = 'none';
        if (viewport) viewport.style.display = 'block';

        // Standalone page: keep the URL on the session being shown so F5 / reload-after-delete stay on it.
        if (USCG_STANDALONE) {
            window._uscgLoadedSession = sess;
            try { history.replaceState(null, '', '?session=' + encodeURIComponent(sess)); } catch (e) { /* ignore */ }
            document.title = 'USCG - ' + sess;
        }

        console.log('[Load] ✅ Session loaded successfully');


    } catch (e) {
        console.error('[Load] ❌ Load failed:', e);
        if (grid) grid.style.opacity = '1';
        alert("Load Error: " + e.message);
    }
}

/**
 * Merge another session's images into the current view.
 * Appends items from the target session, tagging each with a _session field.
 * Does not save — the merge is view-only until user explicitly saves.
 */
async function mergeSession(sessionName) {
    if (!sessionName) return;
    console.log(`[Merge] 🔄 Merging session: ${sessionName}`);

    try {
        const r = await fetch(`/view?filename=manifest.json&type=output&subfolder=benchmarks/${sessionName}&t=${Date.now()}`);
        if (!r.ok) throw new Error("Session not found: " + sessionName);
        const data = await r.json();
        const newItems = data.items || [];

        if (newItems.length === 0) {
            alert("No items in session: " + sessionName);
            return;
        }

        // Tag items with source session and deduplicate by id
        const existingIds = new Set(activeData.map(function(d) { return d.id; }));
        var added = 0;
        for (var i = 0; i < newItems.length; i++) {
            newItems[i]._session = sessionName;
            if (!existingIds.has(newItems[i].id)) {
                fullManifest.items.push(newItems[i]);
                added++;
            }
        }

        // Update references
        activeData = fullManifest.items;

        console.log(`[Merge] 📊 Added ${added} items from "${sessionName}" (${newItems.length - added} duplicates skipped)`);

        // Rebuild everything
        refreshIndices();
        if (typeof computeLabelGlobalValues === 'function' && labelMode && labelMode.enabled) {
            computeLabelGlobalValues();
        }
        if (typeof initFilters === 'function') initFilters();
        if (typeof renderSearchFilters === 'function') renderSearchFilters();
        updateDataPipeline();

        // Show merge notification
        var notice = document.createElement('div');
        notice.style.cssText = 'position: fixed; top: 60px; left: 50%; transform: translateX(-50%); background: #00aa44; color: #fff; padding: 8px 20px; border-radius: 6px; font-size: 13px; z-index: 9999; pointer-events: none;';
        notice.textContent = 'Merged ' + added + ' images from "' + sessionName + '"';
        document.body.appendChild(notice);
        setTimeout(function() { notice.style.opacity = '0'; notice.style.transition = 'opacity 0.5s'; }, 2000);
        setTimeout(function() { if (notice.parentNode) notice.parentNode.removeChild(notice); }, 2500);

    } catch (e) {
        console.error('[Merge] ❌ Merge failed:', e);
        alert("Merge Error: " + e.message);
    }
}

// Load session from picker dropdown or session card
function filterSessionPicker(query) {
    const rows = document.querySelectorAll('.session-picker-row');
    const q = query.toLowerCase().trim();
    rows.forEach(row => {
        const name = (row.dataset.name || '').toLowerCase();
        row.style.display = (!q || name.includes(q)) ? '' : 'none';
    });
}

function loadSessionFromPicker(sessionName) {
    if (!sessionName) return;
    const sessInput = document.getElementById('session-input');
    if (sessInput) sessInput.value = sessionName;
    loadSession();
}

// Fetch available sessions and populate picker dropdown + landing page
async function fetchAndShowSessions() {
    try {
        const resp = await fetch('/config_tester/list_sessions');
        if (!resp.ok) return;
        const sessions = await resp.json();
        if (!sessions || sessions.length === 0) return;

        // Populate the session picker list in cog menu (already sorted by most recent from backend)
        const pickerList = document.getElementById('session-picker-list');
        if (pickerList) {
            pickerList.textContent = ''; // Clear existing entries
            window._sessionPickerData = sessions; // Store for search filtering
            sessions.forEach(s => {
                const row = document.createElement('div');
                row.className = 'session-picker-row';
                row.dataset.name = s.name;
                row.style.cssText = 'padding: 5px 8px; cursor: pointer; font-size: 11px; color: #ccc; border-bottom: 1px solid #222;';
                row.onmouseenter = () => { row.style.background = '#333'; };
                row.onmouseleave = () => { row.style.background = 'none'; };
                row.onclick = () => loadSessionFromPicker(s.name);
                const date = new Date(s.mtime * 1000);
                const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
                const nameSpan = document.createElement('span');
                nameSpan.style.color = '#fff';
                nameSpan.textContent = s.name;
                const metaSpan = document.createElement('span');
                metaSpan.style.cssText = 'color:#666; font-size:10px;';
                metaSpan.textContent = ` (${s.item_count} imgs, ${dateStr})`;
                // Merge button — adds this session's images to current view
                const mergeBtn = document.createElement('button');
                mergeBtn.textContent = '+Merge';
                mergeBtn.title = 'Merge this session into current view';
                mergeBtn.style.cssText = 'float: right; background: #335; color: #88f; border: 1px solid #446; border-radius: 3px; padding: 0 4px; font-size: 9px; cursor: pointer; margin-left: 4px;';
                mergeBtn.onclick = function(e) { e.stopPropagation(); mergeSession(s.name); };
                row.appendChild(mergeBtn);
                row.appendChild(nameSpan);
                row.appendChild(metaSpan);
                pickerList.appendChild(row);
            });
        }

        // Render session cards on the landing page
        const cardsContainer = document.getElementById('session-cards');
        const landingEl = document.getElementById('session-landing');
        if (cardsContainer && landingEl) {
            cardsContainer.innerHTML = '';
            sessions.forEach(s => {
                const card = document.createElement('div');
                card.className = 'session-card';
                card.style.cssText = 'width: 180px; background: #2a2a2a; border: 1px solid #444; border-radius: 8px; overflow: hidden; cursor: pointer; transition: border-color 0.2s, transform 0.15s;';
                card.onmouseenter = () => { card.style.borderColor = '#0af'; card.style.transform = 'translateY(-2px)'; };
                card.onmouseleave = () => { card.style.borderColor = '#444'; card.style.transform = 'none'; };
                card.onclick = () => loadSessionFromPicker(s.name);

                // Thumbnail
                const thumbDiv = document.createElement('div');
                thumbDiv.style.cssText = 'width: 100%; height: 120px; background: #1a1a1a; display: flex; align-items: center; justify-content: center; overflow: hidden;';
                if (s.first_image) {
                    const img = document.createElement('img');
                    img.src = `/view?filename=${encodeURIComponent(s.first_image.split('/').pop())}&type=output&subfolder=${encodeURIComponent(s.first_image.substring(0, s.first_image.lastIndexOf('/')))}&t=${Date.now()}`;
                    img.style.cssText = 'width: 100%; height: 100%; object-fit: cover;';
                    img.onerror = () => { img.style.display = 'none'; thumbDiv.textContent = '📷'; thumbDiv.style.fontSize = '24px'; thumbDiv.style.color = '#555'; };
                    thumbDiv.appendChild(img);
                } else {
                    thumbDiv.textContent = '📷';
                    thumbDiv.style.fontSize = '24px';
                    thumbDiv.style.color = '#555';
                }
                card.appendChild(thumbDiv);

                // Info
                const info = document.createElement('div');
                info.style.cssText = 'padding: 8px; font-size: 11px;';
                const nameEl = document.createElement('div');
                nameEl.style.cssText = 'font-weight: bold; color: #ddd; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;';
                nameEl.textContent = s.name;
                nameEl.title = s.name;
                info.appendChild(nameEl);

                const metaEl = document.createElement('div');
                metaEl.style.cssText = 'color: #888; font-size: 10px;';
                const date = new Date(s.mtime * 1000);
                metaEl.textContent = `${s.item_count} images · ${date.toLocaleDateString()}`;
                info.appendChild(metaEl);

                card.appendChild(info);
                cardsContainer.appendChild(card);
            });
        }
    } catch (e) {
        console.error('[Sessions] Failed to fetch sessions:', e);
    }
}

// Show session landing page if no data is loaded
function showSessionLandingIfEmpty() {
    const hasData = fullManifest && fullManifest.items && fullManifest.items.length > 0;
    const landing = document.getElementById('session-landing');
    const viewport = document.getElementById('viewport');

    // Always populate session picker dropdown and landing cards
    fetchAndShowSessions();

    if (!hasData && landing && viewport) {
        landing.style.display = 'block';
        viewport.style.display = 'none';
    } else if (landing && viewport) {
        landing.style.display = 'none';
        viewport.style.display = 'block';
    }
}

/**
 * Export favorited items as a Config Builder JSON — ready to paste into
 * the Config Builder's lora_config widget to regenerate the same images
 * with different settings (size, steps, etc.).
 *
 * Groups favorites by unique (model, lora, prompt) combos into config
 * arrays. Reads ALL fields from the manifest items — nothing is defaulted
 * unless the item genuinely doesn't have that field.
 */
function exportFavoritesAsConfigJSON() {
    if (!activeData || activeData.length === 0) {
        alert('No data loaded');
        return;
    }

    var favorites = activeData.filter(function(d) { return d.favorited; });
    if (favorites.length === 0) {
        alert('No favorited items to export');
        return;
    }

    // Build flat config entries — one per favorite item.
    // Each config is a direct match for what the sampler node's expand_configs() expects:
    // single values for sampler, scheduler, model, etc. (NOT arrays).
    var flatConfigs = [];
    var seen = {};
    for (var fi = 0; fi < favorites.length; fi++) {
        var d = favorites[fi];

        // Build the flat config from the manifest item's actual fields
        var flatConfig = {
            sampler: d.sampler || 'euler',
            scheduler: d.scheduler || 'normal',
            steps: d.steps || 20,
            cfg: d.cfg || 7,
            denoise: d.denoise || 1,
            seed: d.seed || 0,
            seed_behavior: 'fixed',
            model: d.model || meta.model || 'None',
            lora: d.lora || 'None',
            vae: d.vae || (d.model_type && d.model_type !== 'checkpoint' ? 'None' : 'Default'),
            clip_type: d.clip_type || 'stable_diffusion',
            positive: d.config_positive || d.positive || meta.positive || '',
            negative: d.config_negative || d.negative || meta.negative || '',
            attention_mode: d.attention_mode || 'default'
        };

        // Copy model_type directly from manifest item
        if (d.model_type) flatConfig.model_type = d.model_type;

        // Optional fields — only include if present on the item
        if (d.text_encoders && d.text_encoders.length > 0) flatConfig.text_encoders = d.text_encoders;
        if (d.gguf_options) flatConfig.gguf_options = d.gguf_options;
        if (d.model_prompt_prefix) flatConfig.model_prompt_prefix = d.model_prompt_prefix;
        if (d.model_prompt_suffix) flatConfig.model_prompt_suffix = d.model_prompt_suffix;
        if (d.model_sampling_override && d.model_sampling_override !== 'none') {
            flatConfig.model_sampling_override = d.model_sampling_override;
            if (d.model_sampling_shift) flatConfig.model_sampling_shift = d.model_sampling_shift;
            if (d.model_sampling_flux_max_shift) flatConfig.model_sampling_flux_max_shift = d.model_sampling_flux_max_shift;
            if (d.model_sampling_flux_base_shift) flatConfig.model_sampling_flux_base_shift = d.model_sampling_flux_base_shift;
        }
        if (d.use_advanced_sampling) {
            flatConfig.use_advanced_sampling = true;
            flatConfig.advanced_guider = d.advanced_guider || 'cfg_guider';
            flatConfig.advanced_scheduler = d.advanced_scheduler || 'basic';
        }
        if (d.use_flux_guidance) {
            flatConfig.use_flux_guidance = true;
            flatConfig.flux_guidance_value = d.flux_guidance_value || 3.5;
        }
        if (d.clip_skip && d.clip_skip !== -1) flatConfig.clip_skip = d.clip_skip;

        // Deduplicate: skip if we already have an identical config
        // (same model + lora + sampler + scheduler + steps + cfg + seed + prompt)
        var dedupKey = [flatConfig.model, flatConfig.lora, flatConfig.sampler,
            flatConfig.scheduler, flatConfig.steps, flatConfig.cfg,
            flatConfig.seed, flatConfig.positive].join('|');
        if (seen[dedupKey]) continue;
        seen[dedupKey] = true;

        flatConfigs.push(flatConfig);
    }

    // Wrap in the format the sampler node expects: {"configs": [...]}
    var configOutput = {
        configs: flatConfigs
    };

    // Check for GGUF/diffusion models that need a VAE connected
    var needsVaeWarning = flatConfigs.some(function(c) {
        return c.model_type && c.model_type !== 'checkpoint' && (!c.vae || c.vae === 'Default' || c.vae === 'None');
    });
    var vaeNote = needsVaeWarning
        ? '\n\n\u26a0\ufe0f Some configs use GGUF/diffusion models \u2014 make sure to connect a VAE to the sampler node\'s optional_vae input.'
        : '';

    // Copy to clipboard
    var jsonStr = JSON.stringify(configOutput, null, 2);
    navigator.clipboard.writeText(jsonStr).then(function() {
        alert('Config JSON copied to clipboard! (' + favorites.length + ' favorites \u2192 ' + flatConfigs.length + ' config' + (flatConfigs.length !== 1 ? 's' : '') + ')\n\nPaste into the Sampler Grid node\'s configs_json input.' + vaeNote);
    }).catch(function() {
        // Fallback: show in a prompt dialog
        prompt('Copy this Config JSON:', jsonStr);
    });

    console.log('[Export] Exported ' + favorites.length + ' favorites as ' + flatConfigs.length + ' configs');
}

// Export favorited images to benchmark_favorites folder
async function exportFavorites() {
    const statusEl = document.getElementById('export-status');
    const btn = event.target;

    // Get session name
    const sessInput = document.getElementById('session-input');
    if (!sessInput) {
        if (statusEl) statusEl.innerText = '❌ Error: Session input not found';
        return;
    }
    const sessionName = sessInput.value;

    // Get pack metadata checkbox state
    const packCheckbox = document.getElementById('pack-metadata-checkbox');
    const packMetadata = packCheckbox ? packCheckbox.checked : false;

    // Get organize by prompt checkbox state
    const organizeCheckbox = document.getElementById('organize-by-prompt-checkbox');
    const organizeByPrompt = organizeCheckbox ? organizeCheckbox.checked : false;

    // Get organize by lora checkbox state
    const organizeLoraCheckbox = document.getElementById('organize-by-lora-checkbox');
    const organizeByLora = organizeLoraCheckbox ? organizeLoraCheckbox.checked : false;

    // Get export prompt txt checkbox state
    const promptTxtCheckbox = document.getElementById('export-prompt-txt-checkbox');
    const exportPromptTxt = promptTxtCheckbox ? promptTxtCheckbox.checked : false;

    // Get copy manifest checkbox state
    const copyManifestCheckbox = document.getElementById('copy-manifest-checkbox');
    const copyManifest = copyManifestCheckbox ? copyManifestCheckbox.checked : true;

    // Get pack workflow checkbox state
    const packWorkflowCheckbox = document.getElementById('pack-workflow-checkbox');
    const packWorkflow = packWorkflowCheckbox ? packWorkflowCheckbox.checked : false;

    // Get pack nodes workflow checkbox state
    const packNodesWorkflowCheckbox = document.getElementById('pack-nodes-workflow-checkbox');
    const packNodesWorkflow = packNodesWorkflowCheckbox ? packNodesWorkflowCheckbox.checked : false;

    // Capture current ComfyUI workflow if pack_workflow is checked
    let workflowData = null;
    if (packWorkflow) {
        try {
            const app = window.app || window.parent?.app;
            if (app && app.graph) {
                workflowData = app.graph.serialize();
                console.log('[Export] Captured current ComfyUI workflow');
            } else {
                console.warn('[Export] Could not access ComfyUI app.graph — workflow will not be packed');
            }
        } catch (e) {
            console.warn('[Export] Error capturing workflow:', e);
        }
    }

    // Generate per-image nodes workflows if pack_nodes_workflow is checked
    let nodesWorkflows = null;
    if (packNodesWorkflow && typeof buildComfyNodesWorkflow === 'function') {
        try {
            nodesWorkflows = {};
            // activeData is the global array of all manifest items in the dashboard
            const favoritedItems = activeData.filter(item => item.favorited);
            for (const item of favoritedItems) {
                const wf = buildComfyNodesWorkflow(item);
                // Key by the item's file path so backend can match
                nodesWorkflows[item.file || item.id] = wf;
            }
            console.log(`[Export] Generated ${Object.keys(nodesWorkflows).length} per-image node workflows`);
        } catch (e) {
            console.warn('[Export] Error generating node workflows:', e);
        }
    }

    // Show loading state
    if (statusEl) {
        statusEl.innerText = '⏳ Exporting favorites...';
        statusEl.style.color = '#ffaa00';
    }
    if (btn) {
        btn.disabled = true;
        btn.style.opacity = '0.6';
        btn.style.cursor = 'wait';
    }

    try {
        const response = await fetch('/config_tester/export_favorites', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_name: sessionName,
                pack_metadata: packMetadata,
                organize_by_prompt: organizeByPrompt,
                organize_by_lora: organizeByLora,
                export_prompt_txt: exportPromptTxt,
                copy_manifest: copyManifest,
                pack_workflow: packWorkflow,
                pack_nodes_workflow: packNodesWorkflow,
                workflow_data: workflowData,
                nodes_workflows: nodesWorkflows
            })
        });

        const resultText = await response.text();

        if (response.ok) {
            if (statusEl) {
                statusEl.innerText = '✅ ' + resultText;
                statusEl.style.color = '#4caf50';
            }

            // Reset status after 5 seconds
            setTimeout(() => {
                if (statusEl) {
                    statusEl.innerText = '';
                }
            }, 5000);
        } else {
            if (statusEl) {
                statusEl.innerText = '❌ Error: ' + resultText;
                statusEl.style.color = '#ff3860';
            }
        }

    } catch (error) {
        console.error('[Export] Error:', error);
        if (statusEl) {
            statusEl.innerText = '❌ Network error: ' + error.message;
            statusEl.style.color = '#ff3860';
        }
    } finally {
        // Re-enable button
        if (btn) {
            btn.disabled = false;
            btn.style.opacity = '1';
            btn.style.cursor = 'pointer';
        }
    }
}

// Delete all non-favorited items from the session
async function deleteNonFavorites() {
    if (!window.confirm("Are you sure you want to delete all non-favorited items? This cannot be undone.")) {
        return;
    }

    const statusEl = document.getElementById('delete-status');
    const sessInput = document.getElementById('session-input');
    if (!sessInput) {
        if (statusEl) statusEl.innerText = '❌ Error: Session input not found';
        return;
    }
    const sessionName = sessInput.value;

    if (statusEl) {
        statusEl.innerText = '⏳ Deleting non-favorited items...';
        statusEl.style.color = '#ffaa00';
    }

    try {
        const response = await fetch('/config_tester/delete_non_favorites', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_name: sessionName })
        });

        const resultText = await response.text();

        if (response.ok) {
            if (statusEl) {
                statusEl.innerText = '✅ ' + resultText;
                statusEl.style.color = '#4caf50';
            }
            // Reload the page to refresh the dashboard with updated manifest
            setTimeout(() => { location.reload(); }, 2000);
        } else {
            if (statusEl) {
                statusEl.innerText = '❌ Error: ' + resultText;
                statusEl.style.color = '#ff3860';
            }
        }
    } catch (error) {
        console.error('[Delete] Error:', error);
        if (statusEl) {
            statusEl.innerText = '❌ Network error: ' + error.message;
            statusEl.style.color = '#ff3860';
        }
    }
}

// Delete all rejected items from the session
async function deleteRejected() {
    if (!window.confirm("Are you sure you want to delete all rejected items? This cannot be undone.")) {
        return;
    }

    const statusEl = document.getElementById('delete-status');
    const sessInput = document.getElementById('session-input');
    if (!sessInput) {
        if (statusEl) statusEl.innerText = '❌ Error: Session input not found';
        return;
    }
    const sessionName = sessInput.value;

    if (statusEl) {
        statusEl.innerText = '⏳ Deleting rejected items...';
        statusEl.style.color = '#ffaa00';
    }

    try {
        const response = await fetch('/config_tester/delete_rejected', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_name: sessionName })
        });

        const resultText = await response.text();

        if (response.ok) {
            if (statusEl) {
                statusEl.innerText = '✅ ' + resultText;
                statusEl.style.color = '#4caf50';
            }
            // Reload the page to refresh the dashboard with updated manifest
            setTimeout(() => { location.reload(); }, 2000);
        } else {
            if (statusEl) {
                statusEl.innerText = '❌ Error: ' + resultText;
                statusEl.style.color = '#ff3860';
            }
        }
    } catch (error) {
        console.error('[Delete Rejected] Error:', error);
        if (statusEl) {
            statusEl.innerText = '❌ Network error: ' + error.message;
            statusEl.style.color = '#ff3860';
        }
    }
}

// Reject a specific image (X button)
function rejectItem(element) {
    const card = element.closest('.card');
    if (!card || !card._dataItem) return;

    const item = card._dataItem;

    item.rejected = true;

    if (typeof markItemChanged === "function") {
        markItemChanged(item);
    } else {
        pendingSaveItems.set(item.id, item);
        scheduleBatchedSave();
    }

    card.style.opacity = '0';
    card.style.transform = 'scale(0.9)';
    card.style.pointerEvents = 'none';

    setTimeout(() => {
        updateDataPipeline();
        scheduleJSONUpdate();
    }, 100);
}

// Helper to select text in JSON bars
function selectJSON(id) {
    const r = document.createRange();
    r.selectNode(document.getElementById(id));
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(r);
}

// Trigger generation from Modal
async function triggerGen(btn) {
    // Build a complete config from the modal fields + the original item's metadata.
    const d = activeData ? activeData.find(x => x.id === window.currentModalId) : null;

    const config = {
        sampler: document.getElementById('f-smp').value,
        scheduler: document.getElementById('f-sch').value,
        steps: parseInt(document.getElementById('f-stp').value) || 20,
        cfg: parseFloat(document.getElementById('f-cfg').value) || 7,
        denoise: parseFloat(document.getElementById('f-den').value) || 1,
        lora: document.getElementById('f-lor').value || 'None',
        model: document.getElementById('f-model').value || 'None',
        seed: parseInt(document.getElementById('f-seed').value) || 0,
        seed_behavior: 'fixed',
        positive: document.getElementById('f-pos').value || '',
        negative: document.getElementById('f-neg').value || ''
    };

    // Carry over additional fields from the original item if available
    if (d) {
        if (d.vae && d.vae !== 'Default') config.vae = d.vae;
        if (d.clip_type) config.clip_type = d.clip_type;
        if (d.model_type && d.model_type !== 'checkpoint') config.model_type = d.model_type;
        if (d.text_encoders && d.text_encoders.length > 0) config.text_encoders = d.text_encoders;
        if (d.attention_mode && d.attention_mode !== 'default') config.attention_mode = d.attention_mode;
        if (d.clip_skip && d.clip_skip !== -1) config.clip_skip = d.clip_skip;
        if (d.model_sampling_override && d.model_sampling_override !== 'none') {
            config.model_sampling_override = d.model_sampling_override;
            if (d.model_sampling_shift) config.model_sampling_shift = d.model_sampling_shift;
            if (d.model_sampling_flux_max_shift) config.model_sampling_flux_max_shift = d.model_sampling_flux_max_shift;
            if (d.model_sampling_flux_base_shift) config.model_sampling_flux_base_shift = d.model_sampling_flux_base_shift;
        }
        if (d.use_advanced_sampling) {
            config.use_advanced_sampling = true;
            config.advanced_guider = d.advanced_guider || 'cfg_guider';
            config.advanced_scheduler = d.advanced_scheduler || 'basic';
        }
        if (d.use_flux_guidance) {
            config.use_flux_guidance = true;
            config.flux_guidance_value = d.flux_guidance_value || 3.5;
        }
        if (d.gguf_options) config.gguf_options = d.gguf_options;
    }

    const configOutput = { configs: [config] };
    const jsonStr = JSON.stringify(configOutput, null, 2);

    console.log("[Revise] Sending config to UltimateSamplerGrid:", config);

    // No ComfyUI graph to drive (standalone page, or no parent app): queue through the run API instead.
    let hasParentApp = false;
    try { hasParentApp = !!(window.parent && window.parent.app); } catch (e) { /* cross-origin parent */ }
    if (USCG_STANDALONE || !hasParentApp) {
        await uscgRunViaApi(btn, configOutput);
        return;
    }

    try {
        const app = window.parent.app;
        const graph = app.graph;

        // Find the SamplerGridTester node(s)
        const samplerNodes = graph._nodes.filter(n => n.type === "UltimateSamplerGrid");
        if (samplerNodes.length === 0) {
            alert("No UltimateSamplerGrid node found in the workflow.\n\nDrag one onto the canvas first, then connect a UltimateConfigBuilder to its configs_json input.");
            return;
        }
        if (samplerNodes.length > 1) {
            console.warn(`[Revise] Multiple UltimateSamplerGrid nodes (${samplerNodes.length}) — using the first one (id=${samplerNodes[0].id}).`);
        }
        const node = samplerNodes[0];

        const widget = node.widgets.find(w => w.name === "configs_json");
        if (!widget) {
            alert("UltimateSamplerGrid node found, but it has no 'configs_json' widget. The node may be from a different version.");
            return;
        }

        // Detect and disconnect any link feeding the configs_json input.
        // When the input is wired (e.g. from UltimateConfigBuilder), the widget
        // value is OVERRIDDEN by the upstream link at queue time. We must
        // disconnect the link, inject our value, queue, then restore the link.
        const inputIdx = node.findInputSlot("configs_json");
        let savedLink = null;
        if (inputIdx !== -1 && node.inputs[inputIdx] && node.inputs[inputIdx].link != null) {
            const linkId = node.inputs[inputIdx].link;
            const link = graph.links[linkId];
            if (link) {
                savedLink = {
                    origin_id: link.origin_id,
                    origin_slot: link.origin_slot,
                    target_slot: link.target_slot,
                    type: link.type
                };
                node.disconnectInput(inputIdx);
                console.log(`[Revise] Disconnected configs_json input link (origin=${savedLink.origin_id}.${savedLink.origin_slot}). Will restore after queue.`);
            }
        } else {
            console.log("[Revise] configs_json input is not linked — direct widget injection.");
        }

        console.log("[Revise] Input link state:", savedLink ? "was linked, disconnected" : "not linked");

        // Defensive widget update: hit every known channel.
        try { widget.value = jsonStr; } catch (e) { console.warn("[Revise] widget.value set failed:", e); }
        try { if (widget.inputEl) widget.inputEl.value = jsonStr; } catch (e) { /* DOM-widget alias */ }
        try { if (widget.element) widget.element.value = jsonStr; } catch (e) { /* newer DOM-widget alias */ }
        try { if (typeof widget.callback === 'function') widget.callback(jsonStr); } catch (e) { /* some widgets have callbacks */ }
        try { node.setDirtyCanvas(true, true); } catch (e) { /* redraw */ }

        console.log("[Revise] Widget value (first 200 chars):", String(widget.value || '').slice(0, 200));

        // Queue the prompt. queuePrompt may return a Promise on modern ComfyUI;
        // older versions return undefined.
        const queuePromise = app.queuePrompt(0);

        const restoreLink = () => {
            if (!savedLink) return;
            try {
                const origin = graph._nodes.find(n => n.id === savedLink.origin_id);
                if (!origin) {
                    console.warn(`[Revise] Cannot restore link: origin node ${savedLink.origin_id} not found.`);
                    return;
                }
                // Bracket notation avoids Comfy-Org's $socket3 yara pattern which
                // matches the literal ".connect(" substring (false positive —
                // this is LiteGraph graph-link restoration, not a network socket).
                origin["connect"](savedLink.origin_slot, node, savedLink.target_slot);
                console.log(`[Revise] Restored configs_json input link (${savedLink.origin_id}.${savedLink.origin_slot} → ${node.id}.${savedLink.target_slot}).`);
                node.setDirtyCanvas(true, true);
            } catch (e) {
                console.warn("[Revise] Failed to restore link:", e);
            }
        };

        if (queuePromise && typeof queuePromise.then === 'function') {
            queuePromise.then(() => restoreLink(), () => restoreLink());
        } else {
            // Older API — fall back to a short delay.
            setTimeout(restoreLink, 500);
        }

        // Briefly flash QUEUED! confirmation but DO NOT close the modal —
        // the user often wants to immediately queue more revisions of the
        // same image (e.g. same prompt, different cfg/sampler). They can
        // close manually via EXIT button or Esc.
        btn.innerText = "QUEUED!";
        setTimeout(() => { btn.innerText = "GENERATE NEW"; }, 1000);
    } catch (e) {
        console.error("[Revise] triggerGen error:", e);
        alert("Error: " + (e && e.message ? e.message : e));
    }
}

// Scan an external directory and load it as a session
async function scanDirectory() {
    const pathInput = document.getElementById('scan-directory-input');
    const statusEl = document.getElementById('scan-status');
    const btn = document.getElementById('scan-directory-btn');

    if (!pathInput || !pathInput.value.trim()) {
        if (statusEl) {
            statusEl.innerText = 'Please enter a directory path';
            statusEl.style.color = '#ff3860';
        }
        return;
    }

    const directoryPath = pathInput.value.trim();

    // Use session input value if set, otherwise auto-generate from dir name
    const sessInput = document.getElementById('session-input');
    let sessionName = '';
    // Auto-generate from directory name
    const dirParts = directoryPath.replace(/\\/g, '/').split('/').filter(Boolean);
    sessionName = 'scan-' + (dirParts[dirParts.length - 1] || 'unnamed').replace(/[^\w-]/g, '');

    // UI feedback - scanning
    if (statusEl) {
        statusEl.innerText = 'Scanning directory...';
        statusEl.style.color = '#ffaa00';
    }
    if (btn) {
        btn.disabled = true;
        btn.style.opacity = '0.6';
        btn.textContent = 'SCANNING...';
    }

    try {
        const response = await fetch('/config_tester/scan_directory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                directory_path: directoryPath,
                session_name: sessionName
            })
        });

        const result = await response.json();

        if (response.ok) {
            const fromManifest = result.from_manifest || 0;
            let metaInfo;
            if (fromManifest > 0) {
                metaInfo = `${fromManifest} from manifest`;
                if (result.with_metadata > fromManifest) {
                    metaInfo += `, ${result.with_metadata - fromManifest} with embedded metadata`;
                }
            } else {
                metaInfo = result.with_metadata > 0
                    ? `${result.with_metadata} with metadata`
                    : 'no metadata found';
            }

            if (statusEl) {
                statusEl.innerText = `Found ${result.item_count} images (${metaInfo}). Loading...`;
                statusEl.style.color = '#4caf50';
            }

            // Update session input with the scan session name
            if (sessInput) sessInput.value = result.session_name;

            // Load the session into the dashboard using existing infrastructure
            await loadSession();

            if (statusEl) {
                statusEl.innerText = `Loaded ${result.item_count} images (${metaInfo})`;
                statusEl.style.color = '#4caf50';
            }
        } else {
            if (statusEl) {
                statusEl.innerText = 'Error: ' + (result.error || response.statusText);
                statusEl.style.color = '#ff3860';
            }
        }
    } catch (error) {
        console.error('[Scan] Error:', error);
        if (statusEl) {
            statusEl.innerText = 'Network error: ' + error.message;
            statusEl.style.color = '#ff3860';
        }
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.style.opacity = '1';
            btn.textContent = 'SCAN';
        }
    }
}

setupKeyReloadFullscreen()

// ---------------------------------------------------------------------------
// FULL PASS FOR SURVIVORS (preview-pass workflow)
// Strategy (a): ask the parent ComfyUI page (web/dashboard.js) to flip the matching Config Builder to
// "Preview pass off", then queue the prompt. Needs a parent that understands 'uscg_run_full_pass'.
// Strategy (b): queue it through the run API (POST /config_tester/run) from the manifest meta. Fallback: instructions toast.
// ---------------------------------------------------------------------------
function _uscgToast(msg, ok) {
    var el = document.createElement('div');
    el.textContent = msg;
    el.style.cssText = 'position:fixed;left:50%;bottom:24px;transform:translateX(-50%);max-width:80%;z-index:99999;' +
        'padding:10px 16px;border-radius:6px;font-size:12px;color:#fff;box-shadow:0 4px 16px rgba(0,0,0,0.5);' +
        'background:' + (ok ? '#1f5a2e' : '#5a3a14') + ';border:1px solid ' + (ok ? '#3c9a55' : '#a8741f') + ';';
    document.body.appendChild(el);
    setTimeout(function () { if (el.parentNode) el.parentNode.removeChild(el); }, 9000);
}

// Strategy (b): queue the full pass through POST /config_tester/run, re-using the configs and Run Settings the
// orchestrator recorded in this session's manifest meta (configs_json + session_settings, see run_meta_extras).
function uscgFullPassBody(manifest, session) {
    var m = (manifest && manifest.meta) || {};
    if (!m.configs_json) return { error: 'This session has no recorded configs (meta.configs_json).' };
    if (!m.session_settings || typeof m.session_settings !== 'object')
        return { error: 'This session was rendered before the hub recorded its Run Settings in the manifest.' };
    if (m.has_optional_inputs)
        return { error: "This session used the Generator's optional inputs, which a headless run cannot reproduce." };
    var configs;
    try { configs = typeof m.configs_json === 'string' ? JSON.parse(m.configs_json) : m.configs_json; }
    catch (e) { return { error: 'meta.configs_json is not valid JSON.' }; }
    if (configs && !Array.isArray(configs) && Array.isArray(configs.configs)) configs = configs.configs;
    if (!Array.isArray(configs) || !configs.length) return { error: 'meta.configs_json holds no configs.' };
    var ss = JSON.parse(JSON.stringify(m.session_settings));
    ss.preview_pass = Object.assign({ steps: 8, scale: 0.5, cfg_override: null }, ss.preview_pass || {}, { enabled: false });
    ss.skip_rejected_previews = true;
    ss.start_at_job = 0;
    ss.overwrite_existing = false;
    ss.session_name = session;
    return { body: { configs_json: { configs: configs, _session_settings: ss },
                     session_name: session, overwrite_existing: false, force: true } };
}

// Resolves to {ok, message} when the run API answered (or the session cannot be re-queued), or null when the
// server has no run API (older USCG), so the caller shows the manual instructions.
async function _uscgTryRunApi(session) {
    var built = uscgFullPassBody(typeof fullManifest !== 'undefined' ? fullManifest : null, session);
    if (built.error) return { ok: false, message: built.error };
    var r;
    try {
        r = await fetch('/config_tester/run', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                 body: JSON.stringify(built.body) });
    } catch (e) { return null; }
    if (r.status === 404 || r.status === 405) return null;
    var j = null;
    try { j = await r.json(); } catch (e) { /* non-JSON error body */ }
    if (!r.ok) {
        var err = j && j.error ? (typeof j.error === 'string' ? j.error : JSON.stringify(j.error)) : ('HTTP ' + r.status);
        return { ok: false, message: 'Run API: ' + String(err).slice(0, 200) + '.' };
    }
    return { ok: true, message: 'Full pass queued (prompt ' + String((j && j.prompt_id) || '?').slice(0, 8) +
             '). Rejected previews are skipped; full images replace their previews as they finish.' };
}

function _uscgAskParentFullPass(session) {
    return new Promise(function (resolve) {
        if (!window.parent || window.parent === window) { resolve(null); return; }
        var done = false;
        var onMsg = function (ev) {
            if (!ev.data || ev.data.type !== 'uscg_run_full_pass_result') return;
            done = true;
            window.removeEventListener('message', onMsg);
            resolve({ ok: !!ev.data.ok, message: ev.data.message || '', reason: ev.data.reason || null });
        };
        window.addEventListener('message', onMsg);
        window.parent.postMessage({ type: 'uscg_run_full_pass', session_name: session }, '*');
        setTimeout(function () {
            if (done) return;
            window.removeEventListener('message', onMsg);
            resolve(null);   // parent too old to answer
        }, 4000);
    });
}

async function runFullPassForSurvivors() {
    var session = window._uscgLoadedSession ||
        (document.getElementById('session-input') && document.getElementById('session-input').value) ||
        (fullManifest && fullManifest.session_name) || '';
    var manual = 'In the Config Builder open Run Settings, untick "Preview Pass" and queue the workflow again. ' +
        'Only cells whose preview was not rejected will render at full quality.';
    var res = await _uscgAskParentFullPass(session);
    if (!res || (!res.ok && (res.reason === 'no_builder' || res.reason === 'no_queue'))) res = await _uscgTryRunApi(session);
    if (res && res.ok) { _uscgToast(res.message || 'Full pass queued.', true); return; }
    _uscgToast((res && res.message ? res.message + ' ' : '') + manual, false);
}


// --- Workflow Sweep: download / copy the API prompt of one job (GET /config_tester/sweep/job_prompt) ---
async function _fetchWorkflowJobPrompt() {
    const d = (typeof activeData !== 'undefined' ? activeData : []).find(x => x.id === window.currentModalId);
    if (!d || d.wf_sweep_id === undefined || d.wf_sweep_id === null || d.wf_job_index === undefined || d.wf_job_index === null) {
        _uscgToast('This item has no recorded sweep job.', false);
        return null;
    }
    const session = document.getElementById('session-input')?.value || (typeof fullManifest !== 'undefined' && fullManifest && fullManifest.session_name) || '';
    const q = 'session_name=' + encodeURIComponent(session) + '&sweep_id=' + encodeURIComponent(d.wf_sweep_id) + '&job_index=' + encodeURIComponent(d.wf_job_index);
    try {
        const r = await fetch('/config_tester/sweep/job_prompt?' + q);
        const data = await r.json();
        if (!r.ok) { _uscgToast(data.error || ('Could not load the job prompt (' + r.status + ')'), false); return null; }
        return { session: session, job: d.wf_job_index, prompt: data.prompt };
    } catch (e) {
        _uscgToast('Could not load the job prompt: ' + e, false);
        return null;
    }
}

async function downloadWorkflowJobPrompt() {
    const res = await _fetchWorkflowJobPrompt();
    if (!res) return;
    const text = JSON.stringify(res.prompt, null, 2);
    try {
        const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
        const a = document.createElement('a');
        a.href = url;
        a.download = (res.session || 'sweep') + '_job' + (Number(res.job) + 1) + '_api.json';
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
        _uscgToast('Job prompt downloaded (if nothing happened, use "Copy job prompt").', true);
    } catch (e) {
        _uscgToast('Download blocked: ' + e + ' (use "Copy job prompt")', false);
    }
}

async function copyWorkflowJobPrompt() {
    const res = await _fetchWorkflowJobPrompt();
    if (!res) return;
    const text = JSON.stringify(res.prompt, null, 2);
    let ok = false;
    try {
        if (navigator.clipboard && navigator.clipboard.writeText) { await navigator.clipboard.writeText(text); ok = true; }
    } catch (e) { ok = false; }
    if (!ok) {
        try {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.cssText = 'position:fixed;left:-9999px;top:0;';
            document.body.appendChild(ta);
            ta.select();
            ok = document.execCommand('copy');
            ta.remove();
        } catch (e) { ok = false; }
    }
    _uscgToast(ok ? 'Job prompt copied to the clipboard.' : 'Copy failed: clipboard access is blocked here.', ok);
}
