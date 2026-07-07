/**
 * Config Builder Configuration Management Module
 * Handles state management, UI rendering, and config operations
 */

import {
    normalizePath,
    getShortName,
    parseLoraString,
    buildLoraString,
    getIterationCount,

    countPromptCombinations,
    expandPromptPreview,
    getAvailableVAEs,
    getVAEFolders,
    getAvailableTextEncoders,
    getAvailableUpscaleModels,
    getAvailableLatentUpscaleModels,
    getAvailableSamplers
} from './conf-builder-utilities.js';

import {
    createSearchableSelect,
    createSlider,
    createInputGroup,
    getStyles,
    createTopBar,
    createSidebar,
    createSectionHeader
} from './conf-builder-ui-components.js';

import { renderDistributionSection, renderDistributionSettingsSection } from './conf-builder-distribution.js';

// 3-level hierarchy: Model Type > Orientation > Sizes
const RESOLUTION_PRESETS = {
    "SD 1.5": {
        "Square":    ["512x512", "640x640", "768x768"],
        "Portrait":  ["384x640", "384x704", "448x576", "448x640", "512x768", "512x896", "576x1024"],
        "Landscape": ["640x384", "704x384", "576x448", "640x448", "768x512", "896x512", "1024x576"]
    },
    "SDXL": {
        "Square":    ["1024x1024", "896x896", "768x768"],
        "Portrait":  ["640x1536", "768x1344", "832x1216", "896x1152", "960x1088"],
        "Landscape": ["1536x640", "1344x768", "1216x832", "1152x896", "1088x960"]
    },
    "Illustrious / Pony": {
        "Square":    ["1024x1024", "896x896"],
        "Portrait":  ["832x1216", "768x1344", "640x1536", "896x1152"],
        "Landscape": ["1216x832", "1344x768", "1536x640", "1152x896"]
    },
    "Flux": {
        "Square":    ["1024x1024", "768x768", "512x512"],
        "Portrait":  ["768x1344", "832x1216", "896x1152", "640x1536"],
        "Landscape": ["1344x768", "1216x832", "1152x896", "1536x640"]
    },
    "Wan / Video": {
        "Square":    ["480x480", "512x512", "832x832"],
        "Portrait":  ["480x832", "480x720", "576x1024", "720x1280"],
        "Landscape": ["832x480", "720x480", "1024x576", "1280x720"]
    },
    "HiRes / Upscale": {
        "Square":    ["1536x1536", "2048x2048", "2560x2560"],
        "Portrait":  ["1024x1536", "1152x1728", "1536x2048"],
        "Landscape": ["1536x1024", "1728x1152", "2048x1536"]
    }
};

// =============================================================================
// REMOTE VAE COMPANION DETECTION
//
// The HuggingFace Remote VAE feature lives in a separate optional plugin:
// ComfyUI-USCG-RemoteVAE. We detect its presence by fetching /uscg-remote-vae/health
// (200 means installed, 404 means missing) and load the endpoint list from
// /uscg-remote-vae/endpoints.
//
// Both promises are cached per Builder load. The "Re-check after install"
// button in the install card clears them.
// =============================================================================

let _remoteVaeStatusPromise = null;
function isRemoteVaeAvailable() {
    if (_remoteVaeStatusPromise === null) {
        _remoteVaeStatusPromise = fetch('/uscg-remote-vae/health')
            .then(r => r.ok)
            .catch(() => false);
    }
    return _remoteVaeStatusPromise;
}

let _remoteVaeEndpointsPromise = null;
function loadRemoteVaeEndpoints() {
    if (_remoteVaeEndpointsPromise === null) {
        _remoteVaeEndpointsPromise = fetch('/uscg-remote-vae/endpoints')
            .then(r => r.ok ? r.json() : [])
            .catch(() => []);
    }
    return _remoteVaeEndpointsPromise;
}

function _resetRemoteVaeCaches() {
    _remoteVaeStatusPromise = null;
    _remoteVaeEndpointsPromise = null;
}

// =============================================================================
// CIVITAI COMPANION DETECTION
//
// CivitAI metadata lookup lives in ComfyUI-USCG-CivitAI companion plugin.
// 200 from /uscg-civitai/health → installed. 404 → missing.
//
// Cached per Builder load.
// =============================================================================

let _civitaiStatusPromise = null;
function isCivitaiAvailable() {
    if (_civitaiStatusPromise === null) {
        _civitaiStatusPromise = fetch('/uscg-civitai/health')
            .then(r => r.ok)
            .catch(() => false);
    }
    return _civitaiStatusPromise;
}

function _renderCivitaiCompanionNotice() {
    /** Eye-catching dark-red notice shown when CivitAI companion is missing.
     *  DOM methods only — no innerHTML (security hook trips on it). */
    const notice = document.createElement('div');
    notice.className = 'cb-civitai-notice';
    notice.style.cssText = [
        "border-left: 4px solid #ee4444",
        "background: rgba(180, 40, 40, 0.18)",
        "border-radius: 4px",
        "padding: 8px 12px",
        "margin: 6px 0",
        "font-size: 11px",
        "font-weight: 500",
        "color: #ffdddd",
        "line-height: 1.4",
    ].join(';');
    notice.appendChild(document.createTextNode("ℹ️ Plugin feature — install "));
    const link = document.createElement('a');
    link.href = "https://github.com/JasonHoku/ComfyUI-USCG-CivitAI";
    link.target = "_blank";
    link.rel = "noopener";
    link.style.cssText = "color: #ffaaaa; text-decoration: underline; font-weight: 600;";
    link.textContent = "ComfyUI-USCG-CivitAI";
    notice.appendChild(link);
    notice.appendChild(document.createTextNode(" for auto-trigger detection + metadata lookup."));
    return notice;
}

// Debounced renderUI to batch rapid state changes and avoid redundant full rebuilds
// Pass optional arrayIdx for partial re-render of just one config section
let _renderUITimer = null;
function debouncedRenderUI(node, arrayIdx) {
    if (_renderUITimer) clearTimeout(_renderUITimer);
    _renderUITimer = setTimeout(() => {
        _renderUITimer = null;
        if (arrayIdx !== undefined && arrayIdx !== null) {
            rerenderConfigSection(node, arrayIdx);
        } else {
            node.renderUI();
        }
    }, 150);
}

/**
 * Partial re-render: rebuild only a specific config array section instead of the entire UI.
 * Much faster than full renderUI() for structural changes within a single config
 * (add/remove model, add/remove lora, etc.).
 * Falls back to full renderUI if the target section isn't found in the DOM.
 */
function rerenderConfigSection(node, arrayIdx) {
    const target = document.getElementById(`cb-config-${arrayIdx}`);
    if (!target) {
        // Section not in DOM (collapsed or first render) — fall back to full render
        debouncedRenderUI(node);
        return;
    }

    // Import model lists from cache (already loaded)
    import('./conf-builder-utilities.js').then(async (utilities) => {
        const modelLists = {
            checkpoints: utilities.getAvailableModels ? await utilities.getAvailableModels() : [],
            diffusionModels: utilities.getAvailableDiffusionModels(),
            ggufModels: utilities.getAvailableGGUFModels(),
            textEncoders: utilities.getAvailableTextEncoders(),
            textEncoderFolders: utilities.getTextEncoderFolders(),
            vae: utilities.getAvailableVAEs(),
            vaeFolders: utilities.getVAEFolders(),
            vaeModels: utilities.getAvailableVAEs(),  // Alias used by LTX picker
            upscaleModels: utilities.getAvailableUpscaleModels ? utilities.getAvailableUpscaleModels() : [],
            latentUpscaleModels: utilities.getAvailableLatentUpscaleModels ? utilities.getAvailableLatentUpscaleModels() : [],
            latentUpscaleModelFolders: utilities.getLatentUpscaleModelFolders ? utilities.getLatentUpscaleModelFolders() : ["/"],
            samplers: utilities.getAvailableSamplers(),
            schedulers: utilities.getAvailableSchedulers(),
            clipTypes: utilities.getClipTypes(),
            dualClipTypes: utilities.getDualClipTypes(),
        };
        const availableLoras = await utilities.getAvailableLoras();
        const loraFolders = await utilities.getLoraFolders();
        const configArray = node.state.config_arrays[arrayIdx];
        if (!configArray) return;

        // Save scroll position of the section
        const scrollParent = target.closest('.cb-main-content');
        const savedScroll = scrollParent ? scrollParent.scrollTop : 0;

        // Clear and rebuild just this config section
        target.textContent = '';
        const freshElement = createConfigArrayElement(node, configArray, arrayIdx, modelLists);
        // Copy children from fresh element into existing target
        while (freshElement.firstChild) target.appendChild(freshElement.firstChild);
        // Copy styles
        target.style.borderLeft = freshElement.style.borderLeft;

        renderConfigPromptsSection(node, target, configArray, arrayIdx);
        renderModelsSection(node, target, configArray, arrayIdx, modelLists);
        renderVAEsSection(node, target, configArray, arrayIdx, modelLists);
        renderLorasSection(node, target, configArray, arrayIdx, availableLoras, loraFolders);

        // Update preview
        if (typeof updatePreview === 'function') updatePreview(node);

        // Restore scroll
        if (scrollParent) scrollParent.scrollTop = savedScroll;
    });
}

// --- DRAG-AND-DROP REORDER HELPER ---
// Makes item cards within a list reorderable by dragging their header bar.
// stateArray: the backing array in node.state (e.g. config_arrays[arrayIdx].loras)
// arrayIdx: which config array this belongs to
// itemIdx: the index of this item in the array
// node: the config builder node (for saveState/renderUI)
function setupDragReorder(cardDiv, headerBar, stateArrayGetter, itemIdx, node) {
    headerBar.draggable = true;
    headerBar.style.cursor = "grab";

    // Drag handle icon prepended to header
    const handle = document.createElement("span");
    handle.textContent = "⠿";
    handle.style.cssText = "color: #666; font-size: 14px; margin-right: 6px; cursor: grab; user-select: none;";
    handle.className = "cb-drag-handle";
    headerBar.insertBefore(handle, headerBar.firstChild);

    headerBar.addEventListener("dragstart", (e) => {
        e.stopPropagation();
        cardDiv.classList.add("cb-dragging");
        e.dataTransfer.effectAllowed = "move";
        e.dataTransfer.setData("text/plain", String(itemIdx));
        // Store which list this came from so we don't mix lists
        e.dataTransfer.setData("application/x-cb-list-id", cardDiv.parentElement?.id || "");
    });

    headerBar.addEventListener("dragend", (e) => {
        cardDiv.classList.remove("cb-dragging");
        // Clean up all drop indicators
        document.querySelectorAll(".cb-drop-above, .cb-drop-below").forEach(el => {
            el.classList.remove("cb-drop-above", "cb-drop-below");
        });
    });

    cardDiv.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.stopPropagation();
        e.dataTransfer.dropEffect = "move";
        // Show drop indicator based on mouse position relative to card center
        const rect = cardDiv.getBoundingClientRect();
        const midY = rect.top + rect.height / 2;
        cardDiv.classList.remove("cb-drop-above", "cb-drop-below");
        if (e.clientY < midY) {
            cardDiv.classList.add("cb-drop-above");
        } else {
            cardDiv.classList.add("cb-drop-below");
        }
    });

    cardDiv.addEventListener("dragleave", (e) => {
        cardDiv.classList.remove("cb-drop-above", "cb-drop-below");
    });

    cardDiv.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        cardDiv.classList.remove("cb-drop-above", "cb-drop-below");

        // Verify same list
        const sourceListId = e.dataTransfer.getData("application/x-cb-list-id");
        if (sourceListId && sourceListId !== (cardDiv.parentElement?.id || "")) return;

        const fromIdx = parseInt(e.dataTransfer.getData("text/plain"));
        if (isNaN(fromIdx) || fromIdx === itemIdx) return;

        // Determine insert position based on mouse position
        const rect = cardDiv.getBoundingClientRect();
        const midY = rect.top + rect.height / 2;
        let toIdx = e.clientY < midY ? itemIdx : itemIdx + 1;
        // Adjust for removal shifting indices
        if (fromIdx < toIdx) toIdx--;

        if (fromIdx === toIdx) return;

        const arr = stateArrayGetter();
        if (!arr || fromIdx >= arr.length) return;

        // Splice: remove from old position, insert at new
        const [item] = arr.splice(fromIdx, 1);
        arr.splice(toIdx, 0, item);

        node.saveState();
        debouncedRenderUI(node);
    });
}

// --- SESSION SECTION RENDERER ---

export function renderSessionSection(node, container, availableSessions, refreshAllConfigBuilders) {
    const section = document.createElement("div");
    section.className = "cb-section";
    section.innerHTML = '<div class="cb-section-title">📁 Session Management</div>';
    const grid = document.createElement("div");
    grid.className = "cb-flex-grid";

    const nameInput = document.createElement("input");
    nameInput.className = "cb-input";
    nameInput.value = node.state.session_name;
    nameInput.onchange = () => { node.state.session_name = nameInput.value; node.saveState(); };
    grid.appendChild(createInputGroup("Session Name", nameInput));

    const loadSearchable = createSearchableSelect(
        availableSessions,
        "",
        async (value) => {
            if (value && value !== "None") {
                node.state.auto_save = false;
                await node.loadSession(value);
            }
        },
        "Search sessions..."
    );
    grid.appendChild(createInputGroup("Load Session", loadSearchable));

    const refreshBtn = document.createElement("button");
    refreshBtn.className = "cb-button primary";
    refreshBtn.textContent = "🔄 Refresh Models/LoRAs";
    refreshBtn.style.width = "100%";
    refreshBtn.title = "Clear cache and reload model/LoRA lists from disk";
    refreshBtn.onclick = async () => {
        refreshBtn.disabled = true;
        refreshBtn.textContent = "🔄 Refreshing...";
        await refreshAllConfigBuilders();
        refreshBtn.disabled = false;
        refreshBtn.textContent = "✅ Refreshed!";
        setTimeout(() => { refreshBtn.textContent = "🔄 Refresh Models/LoRAs"; }, 2000);
    };
    grid.appendChild(createInputGroup("Reload Lists", refreshBtn));

    section.appendChild(grid);
    container.appendChild(section);
}

// --- CONFIG SECTION RENDERER ---

export function renderConfigSection(node, container, availableConfigs) {
    const section = document.createElement("div");
    section.className = "cb-section";
    section.innerHTML = '<div class="cb-section-title">💾 Config Management</div>';
    const grid = document.createElement("div");
    grid.className = "cb-flex-grid";

    const nameInput = document.createElement("input");
    nameInput.className = "cb-input";
    nameInput.value = node.state.config_name;
    nameInput.placeholder = "my_config";
    nameInput.onchange = () => {
        node.state.config_name = nameInput.value;
        node.saveState();
    };
    grid.appendChild(createInputGroup("Config Name (Filename)", nameInput));

    const loadSearchable = createSearchableSelect(
        availableConfigs,
        "",
        async (value) => {
            if (value && value !== "None") {
                await node.loadConfigFromBackend(value);
            }
        },
        "Load Config..."
    );
    grid.appendChild(createInputGroup("Load Saved Config", loadSearchable));

    const buttonsDiv = document.createElement("div");
    buttonsDiv.style.cssText = "display: flex; flex-direction: column; gap: 5px; height: 100%; justify-content: flex-end;";

    const saveBtn = document.createElement("button");
    saveBtn.className = "cb-button primary";
    saveBtn.textContent = "💾 Save Config Now";
    saveBtn.style.width = "100%";
    saveBtn.onclick = async () => {
        await node.saveConfigToBackend();
        const { getAvailableConfigs, clearConfigsCache } = await import('./conf-builder-utilities.js');
        clearConfigsCache();  // Invalidate so getAvailableConfigs fetches fresh list
        await getAvailableConfigs();
        node.renderUI();
    };
    buttonsDiv.appendChild(saveBtn);

    const autoSaveLabel = document.createElement("label");
    autoSaveLabel.className = "cb-toggle";
    autoSaveLabel.style.fontSize = "12px";

    const autoSaveCheckbox = document.createElement("input");
    autoSaveCheckbox.type = "checkbox";
    autoSaveCheckbox.checked = node.state.auto_save;
    autoSaveCheckbox.onchange = () => {
        node.state.auto_save = autoSaveCheckbox.checked;
        node.saveState();
    };

    autoSaveLabel.appendChild(autoSaveCheckbox);
    autoSaveLabel.appendChild(document.createTextNode(" Auto-Save (2s)"));
    buttonsDiv.appendChild(autoSaveLabel);

    grid.appendChild(createInputGroup("Actions", buttonsDiv));
    section.appendChild(grid);
    container.appendChild(section);
}

// --- CHIP LIST BUILDER (Reusable dropdown + chips component) ---

/**
 * Creates a dropdown + chip list UI for selecting items from a predefined list.
 * Used for samplers, schedulers, and similar list-based selections.
 *
 * @param {object} params
 * @param {string} params.label - Display label (e.g. "Samplers")
 * @param {string} params.stateKey - Key in configArray (e.g. "samplers")
 * @param {string[]|Function} params.options - Available options array or getter function returning options
 * @param {object} params.node - Node reference for saving state
 * @param {number} params.arrayIdx - Config array index
 * @param {object} params.configArray - The config array object
 * @param {string} params.accentColor - Accent color for styling (default: "#0088ff")
 * @param {string} params.placeholder - Empty state placeholder text
 * @returns {HTMLElement} The complete chip list builder element
 */
function createChipListBuilder({ label, stateKey, options, node, arrayIdx, configArray, accentColor = "#0088ff", placeholder = "None selected" }) {
    const container = document.createElement("div");
    container.style.cssText = "width: 100%; margin-bottom: 2px;";

    // Migrate from comma-separated string to array if needed
    if (typeof configArray[stateKey] === "string") {
        const arr = configArray[stateKey].split(",").map(s => s.trim()).filter(s => s);
        configArray[stateKey] = arr;
        node.state.config_arrays[arrayIdx][stateKey] = arr;
    }
    if (!Array.isArray(configArray[stateKey])) {
        configArray[stateKey] = [];
        node.state.config_arrays[arrayIdx][stateKey] = [];
    }

    // Chips container
    const chipsContainer = document.createElement("div");
    chipsContainer.style.cssText = "display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; min-height: 24px;";

    const renderChips = () => {
        chipsContainer.innerHTML = "";
        const items = configArray[stateKey];
        if (!items || items.length === 0) {
            const ph = document.createElement("div");
            ph.textContent = placeholder;
            ph.style.cssText = "color: #666; font-style: italic; padding: 2px 4px; font-size: 11px;";
            chipsContainer.appendChild(ph);
            return;
        }
        items.forEach((item, idx) => {
            const chip = document.createElement("div");
            chip.style.cssText = `display: flex; align-items: center; background: #444; color: #fff; border-radius: 12px; padding: 2px 8px; font-size: 11px;`;
            const text = document.createElement("span");
            text.textContent = item;
            chip.appendChild(text);
            const closeBtn = document.createElement("span");
            closeBtn.textContent = "×";
            closeBtn.style.cssText = "margin-left: 6px; cursor: pointer; color: #ff8888; font-weight: bold;";
            closeBtn.onclick = () => {
                node.state.config_arrays[arrayIdx][stateKey].splice(idx, 1);
                node.saveState();
                renderChips();
            };
            chip.appendChild(closeBtn);
            chipsContainer.appendChild(chip);
        });
    };
    renderChips();
    container.appendChild(chipsContainer);

    // Input row: dropdown + Add + Remove All
    const inputRow = document.createElement("div");
    inputRow.style.cssText = "display: flex; gap: 4px; align-items: center;";

    const select = document.createElement("select");
    select.className = "cb-select";
    select.style.cssText = "flex: 1; font-size: 11px; padding: 3px 4px;";

    const populateSelect = () => {
        select.innerHTML = "";
        const currentItems = configArray[stateKey] || [];
        const resolvedOptions = typeof options === "function" ? options() : options;
        const available = (resolvedOptions || []).filter(o => !currentItems.includes(o));
        if (available.length === 0) {
            const opt = document.createElement("option");
            opt.textContent = "(all added)";
            opt.disabled = true;
            select.appendChild(opt);
        } else {
            available.forEach(name => {
                const opt = document.createElement("option");
                opt.value = name;
                opt.textContent = name;
                select.appendChild(opt);
            });
        }
    };
    populateSelect();

    // Add item immediately when clicking a dropdown option
    select.onchange = () => {
        const val = select.value;
        if (val && !configArray[stateKey].includes(val)) {
            node.state.config_arrays[arrayIdx][stateKey].push(val);
            node.saveState();
            renderChips();
            populateSelect();
        }
    };

    const addBtn = document.createElement("button");
    addBtn.className = "cb-button primary";
    addBtn.textContent = "+";
    addBtn.title = "Add selected";
    addBtn.style.cssText = "padding: 3px 8px; font-size: 11px; min-width: 28px;";
    addBtn.onclick = () => {
        const val = select.value;
        if (val && !configArray[stateKey].includes(val)) {
            node.state.config_arrays[arrayIdx][stateKey].push(val);
            node.saveState();
            renderChips();
            populateSelect();
        }
    };

    const removeAllBtn = document.createElement("button");
    removeAllBtn.className = "cb-button";
    removeAllBtn.textContent = "Clear";
    removeAllBtn.title = "Remove all";
    removeAllBtn.style.cssText = "padding: 3px 8px; font-size: 11px; min-width: 40px; color: #ff8888;";
    removeAllBtn.onclick = () => {
        node.state.config_arrays[arrayIdx][stateKey] = [];
        configArray[stateKey] = [];
        node.saveState();
        renderChips();
        populateSelect();
    };

    inputRow.appendChild(select);
    inputRow.appendChild(addBtn);
    inputRow.appendChild(removeAllBtn);
    container.appendChild(inputRow);

    return container;
}

// ===== CUSTOM RESOLUTIONS CACHE =====
let _customResolutionsCache = null;
let _customResolutionsLoaded = false;

async function loadCustomResolutions() {
    if (_customResolutionsLoaded && _customResolutionsCache) return _customResolutionsCache;
    try {
        const resp = await fetch("/configbuilder/custom_resolutions");
        if (resp.ok) {
            _customResolutionsCache = await resp.json();
        } else {
            _customResolutionsCache = { categories: [] };
        }
    } catch {
        _customResolutionsCache = { categories: [] };
    }
    _customResolutionsLoaded = true;
    return _customResolutionsCache;
}

async function saveCustomResolutions(data) {
    _customResolutionsCache = data;
    try {
        await fetch("/configbuilder/custom_resolutions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });
    } catch (e) {
        console.error("[ConfigBuilder] Error saving custom resolutions:", e);
    }
}

// ===== CUSTOM RESOLUTIONS EDITOR MODAL =====
function openCustomResolutionsEditor(onSaved) {
    // Load current data
    loadCustomResolutions().then(data => {
        // Deep clone so edits don't mutate cache until save
        const editData = JSON.parse(JSON.stringify(data));

        // Create overlay
        const overlay = document.createElement("div");
        overlay.style.cssText = "position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 100000; display: flex; align-items: center; justify-content: center;";

        const modal = document.createElement("div");
        modal.style.cssText = "background: #2a2a2a; border: 1px solid #555; border-radius: 8px; width: 560px; max-height: 80vh; display: flex; flex-direction: column; box-shadow: 0 8px 32px rgba(0,0,0,0.5);";

        // Header
        const header = document.createElement("div");
        header.style.cssText = "padding: 14px 18px; border-bottom: 1px solid #444; display: flex; justify-content: space-between; align-items: center;";
        const title = document.createElement("div");
        title.style.cssText = "font-size: 15px; font-weight: bold; color: #ffcc44;";
        title.textContent = "Custom Resolutions Editor";
        const closeBtn = document.createElement("button");
        closeBtn.textContent = "×";
        closeBtn.style.cssText = "background: none; border: none; color: #999; font-size: 20px; cursor: pointer; padding: 0 4px;";
        closeBtn.onclick = () => overlay.remove();
        header.appendChild(title);
        header.appendChild(closeBtn);
        modal.appendChild(header);

        // Body (scrollable)
        const body = document.createElement("div");
        body.style.cssText = "padding: 14px 18px; overflow-y: auto; flex: 1;";

        const renderCategories = () => {
            body.innerHTML = "";

            if (editData.categories.length === 0) {
                const empty = document.createElement("div");
                empty.style.cssText = "color: #666; font-style: italic; padding: 20px; text-align: center;";
                empty.textContent = "No custom categories yet. Add one below!";
                body.appendChild(empty);
            }

            editData.categories.forEach((cat, catIdx) => {
                const catDiv = document.createElement("div");
                catDiv.style.cssText = "margin-bottom: 14px; background: #333; border-radius: 6px; padding: 10px; border: 1px solid #444;";

                // Category header
                const catHeader = document.createElement("div");
                catHeader.style.cssText = "display: flex; align-items: center; gap: 8px; margin-bottom: 8px;";

                const catInput = document.createElement("input");
                catInput.type = "text";
                catInput.value = cat.name || "";
                catInput.placeholder = "Category name";
                catInput.style.cssText = "flex: 1; background: #222; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 4px 8px; font-size: 13px;";
                catInput.onchange = () => { editData.categories[catIdx].name = catInput.value; };

                const deleteCatBtn = document.createElement("button");
                deleteCatBtn.textContent = "Delete";
                deleteCatBtn.style.cssText = "background: #442222; color: #ff6666; border: 1px solid #663333; border-radius: 4px; padding: 3px 10px; font-size: 11px; cursor: pointer;";
                deleteCatBtn.onclick = () => { editData.categories.splice(catIdx, 1); renderCategories(); };

                catHeader.appendChild(catInput);
                catHeader.appendChild(deleteCatBtn);
                catDiv.appendChild(catHeader);

                // Resolution items
                (cat.items || []).forEach((res, resIdx) => {
                    const resRow = document.createElement("div");
                    resRow.style.cssText = "display: flex; align-items: center; gap: 6px; margin-bottom: 4px; margin-left: 12px;";

                    const resLabel = document.createElement("span");
                    resLabel.style.cssText = "color: #ccc; font-size: 12px; min-width: 90px;";
                    resLabel.textContent = res.replace("x", " × ");

                    const removeResBtn = document.createElement("button");
                    removeResBtn.textContent = "×";
                    removeResBtn.style.cssText = "background: none; border: none; color: #ff6666; cursor: pointer; font-size: 14px; font-weight: bold; padding: 0 4px;";
                    removeResBtn.onclick = () => { editData.categories[catIdx].items.splice(resIdx, 1); renderCategories(); };

                    resRow.appendChild(resLabel);
                    resRow.appendChild(removeResBtn);
                    catDiv.appendChild(resRow);
                });

                // Add resolution row
                const addResRow = document.createElement("div");
                addResRow.style.cssText = "display: flex; align-items: center; gap: 4px; margin-left: 12px; margin-top: 6px;";

                const addW = document.createElement("input");
                addW.type = "number"; addW.placeholder = "W"; addW.min = 64; addW.max = 8192; addW.step = 8;
                addW.style.cssText = "width: 70px; background: #222; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 3px 6px; font-size: 11px;";

                const xLbl = document.createElement("span");
                xLbl.textContent = "×"; xLbl.style.cssText = "color: #888; font-size: 11px;";

                const addH = document.createElement("input");
                addH.type = "number"; addH.placeholder = "H"; addH.min = 64; addH.max = 8192; addH.step = 8;
                addH.style.cssText = "width: 70px; background: #222; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 3px 6px; font-size: 11px;";

                const addResBtn = document.createElement("button");
                addResBtn.textContent = "+ Add";
                addResBtn.style.cssText = "background: #335522; color: #88cc44; border: 1px solid #447733; border-radius: 4px; padding: 3px 10px; font-size: 11px; cursor: pointer;";
                addResBtn.onclick = () => {
                    const w = parseInt(addW.value), h = parseInt(addH.value);
                    if (w > 0 && h > 0) {
                        if (!editData.categories[catIdx].items) editData.categories[catIdx].items = [];
                        const res = `${w}x${h}`;
                        if (!editData.categories[catIdx].items.includes(res)) {
                            editData.categories[catIdx].items.push(res);
                            renderCategories();
                        }
                    }
                };

                addResRow.appendChild(addW);
                addResRow.appendChild(xLbl);
                addResRow.appendChild(addH);
                addResRow.appendChild(addResBtn);
                catDiv.appendChild(addResRow);

                body.appendChild(catDiv);
            });
        };
        renderCategories();
        modal.appendChild(body);

        // Footer
        const footer = document.createElement("div");
        footer.style.cssText = "padding: 12px 18px; border-top: 1px solid #444; display: flex; justify-content: space-between; align-items: center;";

        const addCatBtn = document.createElement("button");
        addCatBtn.textContent = "+ Add Category";
        addCatBtn.style.cssText = "background: #224455; color: #44aadd; border: 1px solid #336677; border-radius: 4px; padding: 6px 14px; font-size: 12px; cursor: pointer;";
        addCatBtn.onclick = () => { editData.categories.push({ name: "New Category", items: [] }); renderCategories(); };

        const saveBtn = document.createElement("button");
        saveBtn.textContent = "Save";
        saveBtn.style.cssText = "background: #335522; color: #88ff44; border: 1px solid #447733; border-radius: 4px; padding: 6px 20px; font-size: 13px; font-weight: bold; cursor: pointer;";
        saveBtn.onclick = async () => {
            await saveCustomResolutions(editData);
            overlay.remove();
            if (onSaved) onSaved();
        };

        footer.appendChild(addCatBtn);
        footer.appendChild(saveBtn);
        modal.appendChild(footer);

        overlay.appendChild(modal);
        overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
        document.body.appendChild(overlay);
    });
}

// ===== NESTED RESOLUTION DROPDOWN =====
function createResolutionDropdown({ onSelect, currentItems, onEditCustom }) {
    const wrapper = document.createElement("div");
    wrapper.style.cssText = "position: relative; display: inline-block;";

    // Trigger button — large, visible
    const triggerBtn = document.createElement("button");
    triggerBtn.className = "cb-button";
    triggerBtn.style.cssText = "padding: 4px 12px; font-size: 12px; display: flex; align-items: center; gap: 6px; cursor: pointer; min-width: 140px; justify-content: space-between; background: #333; border: 1px solid #555; color: #ccc; border-radius: 4px;";
    triggerBtn.innerHTML = '<span>Add Resolution</span><span style="font-size: 10px;">&#9660;</span>';
    wrapper.appendChild(triggerBtn);

    // Dropdown panel — no overflow clipping so submenus can extend outside
    const dropdown = document.createElement("div");
    dropdown.style.cssText = "display: none; position: absolute; top: 100%; left: 0; background: #2a2a2a; border: 1px solid #555; border-radius: 6px; min-width: 200px; z-index: 99999; box-shadow: 0 6px 24px rgba(0,0,0,0.6); overflow: visible;";
    wrapper.appendChild(dropdown);

    let isOpen = false;

    const closeDropdown = () => {
        dropdown.style.display = "none";
        isOpen = false;
    };

    const buildMenu = async () => {
        dropdown.innerHTML = "";
        const alreadyAdded = currentItems() || [];
        const customData = await loadCustomResolutions();

        // --- Custom category (always first) ---
        const customCats = customData.categories || [];
        if (customCats.length > 0) {
            const customEntry = createMenuCategory("Custom", null);
            customCats.forEach(cat => {
                const available = (cat.items || []).filter(r => !alreadyAdded.includes(r));
                if (available.length === 0) return;
                const subEntry = createSubMenu(cat.name || "Unnamed", available, onSelect, closeDropdown);
                customEntry._submenu.appendChild(subEntry);
            });
            dropdown.appendChild(customEntry);
            // Separator
            const sep = document.createElement("div");
            sep.style.cssText = "height: 1px; background: #444; margin: 4px 0;";
            dropdown.appendChild(sep);
        }

        // "Create New Custom" button
        const createBtn = document.createElement("div");
        createBtn.style.cssText = "padding: 6px 14px; color: #44aadd; cursor: pointer; font-size: 12px; display: flex; align-items: center; gap: 6px;";
        createBtn.innerHTML = '<span style="font-size: 14px;">&#9998;</span> Edit Custom Resolutions...';
        createBtn.onmouseenter = () => { createBtn.style.background = "#333"; };
        createBtn.onmouseleave = () => { createBtn.style.background = ""; };
        createBtn.onclick = () => {
            closeDropdown();
            onEditCustom();
        };
        dropdown.appendChild(createBtn);

        // Separator
        const sep2 = document.createElement("div");
        sep2.style.cssText = "height: 1px; background: #444; margin: 4px 0;";
        dropdown.appendChild(sep2);

        // --- Built-in presets (3-level: Model > Orientation > Sizes) ---
        for (const [modelType, orientations] of Object.entries(RESOLUTION_PRESETS)) {
            const modelEntry = createMenuCategory(modelType, null);

            for (const [orientation, sizes] of Object.entries(orientations)) {
                const available = sizes.filter(r => !alreadyAdded.includes(r));
                if (available.length === 0) continue;
                const orientEntry = createSubMenu(orientation, available, onSelect, closeDropdown);
                modelEntry._submenu.appendChild(orientEntry);
            }

            dropdown.appendChild(modelEntry);
        }
    };

    // Create a top-level category with a submenu
    function createMenuCategory(label, _unused) {
        const item = document.createElement("div");
        item.style.cssText = "position: relative;";

        const row = document.createElement("div");
        row.style.cssText = "padding: 6px 14px; color: #eee; cursor: pointer; font-size: 12px; display: flex; justify-content: space-between; align-items: center; white-space: nowrap;";
        row.innerHTML = `<span>${label}</span><span style="font-size: 9px; color: #888; margin-left: 12px;">&#9654;</span>`;
        item.appendChild(row);

        const submenu = document.createElement("div");
        submenu.style.cssText = "display: none; position: absolute; left: 100%; top: 0; background: #2a2a2a; border: 1px solid #555; border-radius: 6px; min-width: 180px; z-index: 100000; box-shadow: 0 4px 16px rgba(0,0,0,0.5); overflow: visible;";
        item.appendChild(submenu);
        item._submenu = submenu;

        row.onmouseenter = () => {
            // Close sibling submenus
            const siblings = item.parentElement?.children || [];
            for (const s of siblings) {
                const sm = s.querySelector("div:nth-child(2)");
                if (sm && sm !== submenu) sm.style.display = "none";
                if (s.querySelector("div:first-child")) s.querySelector("div:first-child").style.background = "";
            }
            submenu.style.display = "block";
            row.style.background = "#444";
        };
        item.onmouseleave = () => {
            submenu.style.display = "none";
            row.style.background = "";
        };

        return item;
    }

    // Create a sub-category (orientation) with resolution items
    function createSubMenu(label, sizes, onSelect, closeDropdown) {
        const item = document.createElement("div");
        item.style.cssText = "position: relative;";

        const row = document.createElement("div");
        row.style.cssText = "padding: 5px 12px; color: #ccc; cursor: pointer; font-size: 12px; display: flex; justify-content: space-between; align-items: center; white-space: nowrap;";
        row.innerHTML = `<span>${label}</span><span style="font-size: 9px; color: #888; margin-left: 12px;">&#9654;</span>`;
        item.appendChild(row);

        const submenu = document.createElement("div");
        submenu.style.cssText = "display: none; position: absolute; left: 100%; top: 0; background: #2a2a2a; border: 1px solid #555; border-radius: 6px; min-width: 140px; z-index: 100001; box-shadow: 0 4px 16px rgba(0,0,0,0.5); overflow: visible;";

        sizes.forEach(res => {
            const resItem = document.createElement("div");
            resItem.style.cssText = "padding: 5px 14px; color: #ffcc44; cursor: pointer; font-size: 12px; white-space: nowrap;";
            resItem.textContent = res.replace("x", " × ");
            resItem.onmouseenter = () => { resItem.style.background = "#554400"; };
            resItem.onmouseleave = () => { resItem.style.background = ""; };
            resItem.onclick = (e) => {
                e.stopPropagation();
                onSelect(res);
                closeDropdown();
            };
            submenu.appendChild(resItem);
        });

        item.appendChild(submenu);

        row.onmouseenter = () => {
            const siblings = item.parentElement?.children || [];
            for (const s of siblings) {
                const sm = s.querySelector("div:nth-child(2)");
                if (sm && sm !== submenu) sm.style.display = "none";
                if (s.querySelector("div:first-child")) s.querySelector("div:first-child").style.background = "";
            }
            submenu.style.display = "block";
            row.style.background = "#3a3a3a";
        };
        item.onmouseleave = () => {
            submenu.style.display = "none";
            row.style.background = "";
        };

        return item;
    }

    triggerBtn.onclick = async (e) => {
        e.stopPropagation();
        if (isOpen) {
            closeDropdown();
        } else {
            await buildMenu();
            dropdown.style.display = "block";
            isOpen = true;
        }
    };

    // Close on outside click
    const outsideHandler = (e) => {
        if (isOpen && !wrapper.contains(e.target)) {
            closeDropdown();
        }
    };
    document.addEventListener("mousedown", outsideHandler);
    // Cleanup when element is removed
    const observer = new MutationObserver(() => {
        if (!document.body.contains(wrapper)) {
            document.removeEventListener("mousedown", outsideHandler);
            observer.disconnect();
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });

    return wrapper;
}

// ===== RESOLUTION BUILDER (main component) =====
function createResolutionBuilder({ node, arrayIdx, configArray }) {
    const container = document.createElement("div");
    container.style.cssText = "width: 100%; margin-bottom: 2px;";

    // Ensure array exists
    if (!Array.isArray(configArray.resolutions)) {
        configArray.resolutions = [];
        node.state.config_arrays[arrayIdx].resolutions = [];
    }

    // ===== CHIPS DISPLAY =====
    const chipsContainer = document.createElement("div");
    chipsContainer.style.cssText = "display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; min-height: 24px;";

    const renderChips = () => {
        chipsContainer.innerHTML = "";
        const items = configArray.resolutions;

        if (!items || items.length === 0) {
            const ph = document.createElement("div");
            ph.textContent = "Using sampler node resolutions";
            ph.style.cssText = "color: #666; font-style: italic; padding: 2px 4px; font-size: 11px;";
            chipsContainer.appendChild(ph);
            return;
        }

        items.forEach((item, idx) => {
            const chip = document.createElement("div");
            chip.style.cssText = "display: flex; align-items: center; background: #554400; color: #ffcc44; border-radius: 12px; padding: 2px 8px; font-size: 11px; border: 1px solid #886600;";

            const text = document.createElement("span");
            text.textContent = item;
            chip.appendChild(text);

            const closeBtn = document.createElement("span");
            closeBtn.textContent = "×";
            closeBtn.style.cssText = "margin-left: 6px; cursor: pointer; color: #ff8888; font-weight: bold;";
            closeBtn.onclick = () => {
                node.state.config_arrays[arrayIdx].resolutions.splice(idx, 1);
                node.saveState();
                renderChips();
            };
            chip.appendChild(closeBtn);
            chipsContainer.appendChild(chip);
        });
    };
    renderChips();
    container.appendChild(chipsContainer);

    // ===== DROPDOWN ROW =====
    const dropdownRow = document.createElement("div");
    dropdownRow.style.cssText = "display: flex; gap: 4px; align-items: center; margin-bottom: 6px;";

    // Nested dropdown
    const resDropdown = createResolutionDropdown({
        onSelect: (res) => {
            if (!configArray.resolutions.includes(res)) {
                node.state.config_arrays[arrayIdx].resolutions.push(res);
                node.saveState();
                renderChips();
            }
        },
        currentItems: () => configArray.resolutions || [],
        onEditCustom: () => {
            openCustomResolutionsEditor(() => {
                _customResolutionsLoaded = false;
            });
        }
    });
    dropdownRow.appendChild(resDropdown);

    // Clear all button (next to dropdown)
    const clearBtn = document.createElement("button");
    clearBtn.className = "cb-button";
    clearBtn.textContent = "Clear";
    clearBtn.title = "Remove all resolutions";
    clearBtn.style.cssText = "padding: 4px 10px; font-size: 11px; min-width: 40px; color: #ff8888;";
    clearBtn.onclick = () => {
        node.state.config_arrays[arrayIdx].resolutions = [];
        configArray.resolutions = [];
        node.saveState();
        renderChips();
    };
    dropdownRow.appendChild(clearBtn);
    container.appendChild(dropdownRow);

    // ===== CUSTOM W x H ROW =====
    const customRow = document.createElement("div");
    customRow.style.cssText = "display: flex; gap: 4px; align-items: center;";

    const customW = document.createElement("input");
    customW.type = "number";
    customW.className = "cb-input";
    customW.placeholder = "W";
    customW.style.cssText = "width: 55px; font-size: 11px; padding: 3px 4px;";
    customW.min = 64; customW.max = 8192; customW.step = 8;

    const xLabel = document.createElement("span");
    xLabel.textContent = "×";
    xLabel.style.cssText = "color: #888; font-size: 11px;";

    const customH = document.createElement("input");
    customH.type = "number";
    customH.className = "cb-input";
    customH.placeholder = "H";
    customH.style.cssText = "width: 55px; font-size: 11px; padding: 3px 4px;";
    customH.min = 64; customH.max = 8192; customH.step = 8;

    const addCustomBtn = document.createElement("button");
    addCustomBtn.className = "cb-button primary";
    addCustomBtn.textContent = "+";
    addCustomBtn.title = "Add custom resolution";
    addCustomBtn.style.cssText = "padding: 3px 8px; font-size: 11px; min-width: 28px;";
    addCustomBtn.onclick = () => {
        const w = parseInt(customW.value);
        const h = parseInt(customH.value);
        if (w > 0 && h > 0) {
            const res = `${w}x${h}`;
            if (!configArray.resolutions.includes(res)) {
                node.state.config_arrays[arrayIdx].resolutions.push(res);
                node.saveState();
                renderChips();
            }
            customW.value = "";
            customH.value = "";
        }
    };

    customRow.appendChild(customW);
    customRow.appendChild(xLabel);
    customRow.appendChild(customH);
    customRow.appendChild(addCustomBtn);
    container.appendChild(customRow);

    return container;
}


// --- CONFIG ARRAY ELEMENT CREATOR ---

// Color palette for config array headers — cycles for easy visual distinction
const CONFIG_COLORS = ["#0088ff", "#ff6600", "#00cc66", "#cc44cc", "#ffaa00", "#44cccc", "#ff4466", "#88aa00"];

export function createConfigArrayElement(node, configArray, arrayIdx, modelLists) {
    const accentColor = CONFIG_COLORS[arrayIdx % CONFIG_COLORS.length];
    const div = document.createElement("div");
    div.className = "cb-array";
    div.id = `cb-config-${arrayIdx}`;
    div.style.borderLeft = `4px solid ${accentColor}`;

    // Prominent config header banner
    const configBanner = document.createElement("div");
    configBanner.style.cssText = `display: flex; align-items: center; gap: 10px; padding: 8px 12px; margin: -12px -12px 12px -12px; background: linear-gradient(90deg, ${accentColor}22, transparent); border-bottom: 2px solid ${accentColor}44; border-radius: 4px 4px 0 0;`;

    const configIndex = document.createElement("span");
    configIndex.style.cssText = `background: ${accentColor}; color: #000; font-weight: 900; font-size: 13px; padding: 2px 8px; border-radius: 4px; font-family: monospace; min-width: 20px; text-align: center;`;
    configIndex.textContent = `${arrayIdx + 1}`;
    configBanner.appendChild(configIndex);

    const configTitle = document.createElement("span");
    configTitle.style.cssText = `color: ${accentColor}; font-weight: 800; font-size: 14px; flex: 1;`;
    configTitle.textContent = configArray.name || `Config ${arrayIdx + 1}`;
    configBanner.appendChild(configTitle);

    div.appendChild(configBanner);

    const settingsGrid = document.createElement("div");
    settingsGrid.className = "cb-flex-grid";

    // Helper for inputs to reduce code duplication
    const addInput = (label, key) => {
        const input = document.createElement("input");
        input.className = "cb-input";
        input.value = configArray[key];
        input.onchange = () => { node.state.config_arrays[arrayIdx][key] = input.value; node.saveState(); };
        settingsGrid.appendChild(createInputGroup(label, input));
    };

    addInput("Config Name", "name");

    // Samplers - dropdown + chips list builder
    const samplersBuilder = createChipListBuilder({
        label: "Samplers", stateKey: "samplers", options: (modelLists && modelLists.samplers) || [],
        node, arrayIdx, configArray, accentColor: "#0088ff", placeholder: "No samplers selected"
    });
    settingsGrid.appendChild(createInputGroup("Samplers", samplersBuilder));

    // Schedulers - dropdown + chips list builder
    if (!isLTXConfigArray(configArray)) {
        const schedulersBuilder = createChipListBuilder({
            label: "Schedulers", stateKey: "schedulers", options: (modelLists && modelLists.schedulers) || [],
            node, arrayIdx, configArray, accentColor: "#00aa66", placeholder: "No schedulers selected"
        });
        settingsGrid.appendChild(createInputGroup("Schedulers", schedulersBuilder));
    }

    // Attention Mode - dropdown + chips list builder
    const ATTENTION_MODES = ["default", "xformers", "pytorch", "flash", "sage", "sage3", "sub_quad", "split"];
    if (!configArray.attention_modes || configArray.attention_modes.length === 0) {
        configArray.attention_modes = ["default"];
    }
    const attentionBuilder = createChipListBuilder({
        label: "Attention", stateKey: "attention_modes", options: ATTENTION_MODES,
        node, arrayIdx, configArray, accentColor: "#cc44cc", placeholder: "default"
    });
    settingsGrid.appendChild(createInputGroup("Attention", attentionBuilder));

    // Resolutions - categorized preset dropdown + custom input + chips
    const resolutionBuilder = createResolutionBuilder({ node, arrayIdx, configArray });
    settingsGrid.appendChild(createInputGroup("Resolutions", resolutionBuilder));

    if (!isLTXConfigArray(configArray)) {
        addInput("Steps", "steps");
    }
    addInput("CFG", "cfg");

    // Seed Behavior Select
    const seedSelect = document.createElement("select");
    seedSelect.className = "cb-select";
    seedSelect.innerHTML = `
        <option value="fixed" ${(configArray.seed_behavior || "fixed") === "fixed" ? 'selected' : ''}>Fixed</option>
        <option value="randomize" ${configArray.seed_behavior === "randomize" ? 'selected' : ''}>Randomize every gen</option>
    `;
    seedSelect.onchange = () => {
        node.state.config_arrays[arrayIdx].seed_behavior = seedSelect.value;
        node.saveState();
    };
    settingsGrid.appendChild(createInputGroup("Seed Behavior (Per Gen)", seedSelect));

    // Full Run Seed Behavior — applied before/after the entire grid test session
    const fullRunSeedSelect = document.createElement("select");
    fullRunSeedSelect.className = "cb-select";
    fullRunSeedSelect.innerHTML = `
        <option value="fixed" ${(configArray.full_run_seed_behavior || "fixed") === "fixed" ? 'selected' : ''}>Fixed</option>
        <option value="random_before" ${configArray.full_run_seed_behavior === "random_before" ? 'selected' : ''}>Random Before Entire Run</option>
        <option value="random_after" ${configArray.full_run_seed_behavior === "random_after" ? 'selected' : ''}>Random After Entire Run</option>
        <option value="increment_after" ${configArray.full_run_seed_behavior === "increment_after" ? 'selected' : ''}>Increment After Entire Run</option>
        <option value="decrement_after" ${configArray.full_run_seed_behavior === "decrement_after" ? 'selected' : ''}>Decrement After Entire Run</option>
    `;
    fullRunSeedSelect.onchange = () => {
        node.state.config_arrays[arrayIdx].full_run_seed_behavior = fullRunSeedSelect.value;
        node.saveState();
    };
    settingsGrid.appendChild(createInputGroup("Full Run Seed Behavior", fullRunSeedSelect));

    // Full Run Seed — overrides node seed when > 0
    const fullRunSeedInput = document.createElement("input");
    fullRunSeedInput.type = "number";
    fullRunSeedInput.className = "cb-input";
    fullRunSeedInput.value = configArray.full_run_seed || 0;
    fullRunSeedInput.min = 0;
    fullRunSeedInput.step = 1;
    fullRunSeedInput.placeholder = "0 = use node seed";
    fullRunSeedInput.onchange = () => {
        node.state.config_arrays[arrayIdx].full_run_seed = parseInt(fullRunSeedInput.value) || 0;
        node.saveState();
    };
    settingsGrid.appendChild(createInputGroup("Seed Number", fullRunSeedInput));

    div.appendChild(settingsGrid);

    const controlsBar = document.createElement("div");
    controlsBar.className = "cb-controls-bar";

    const iterationCount = getIterationCount(configArray);
    const countDisplay = document.createElement("div");
    countDisplay.style.cssText = "color: #00cc00; font-family: monospace; font-size: 13px; font-weight: bold; display: flex; align-items: center;";
    countDisplay.innerHTML = `⏱️ Iterations: ${iterationCount}`;
    controlsBar.appendChild(countDisplay);

    const spacer = document.createElement("div");
    spacer.style.flex = "1";
    controlsBar.appendChild(spacer);

    const addModelBtn = document.createElement("button");
    addModelBtn.className = "cb-button";
    addModelBtn.style.borderLeft = "4px solid #cc6600";
    addModelBtn.textContent = `➕ Add Model`;
    addModelBtn.onclick = () => {
        if (!node.state.config_arrays[arrayIdx].models) node.state.config_arrays[arrayIdx].models = [];
        node.state.config_arrays[arrayIdx].models.push("None");
        node.saveState();
        debouncedRenderUI(node);
    };
    controlsBar.appendChild(addModelBtn);

    const addVaeBtn = document.createElement("button");
    addVaeBtn.className = "cb-button";
    addVaeBtn.style.borderLeft = "4px solid #9900cc";
    addVaeBtn.textContent = `➕ Add VAE`;
    addVaeBtn.onclick = () => {
        if (!node.state.config_arrays[arrayIdx].vaes) node.state.config_arrays[arrayIdx].vaes = [];
        node.state.config_arrays[arrayIdx].vaes.push("None");
        node.saveState();
        debouncedRenderUI(node);
    };
    controlsBar.appendChild(addVaeBtn);

    const addLoraBtn = document.createElement("button");
    addLoraBtn.className = "cb-button";
    addLoraBtn.style.borderLeft = "4px solid #0066cc";
    addLoraBtn.textContent = `➕ Add LoRA`;
    addLoraBtn.onclick = () => {
        if (!node.state.config_arrays[arrayIdx].loras) node.state.config_arrays[arrayIdx].loras = [];
        node.state.config_arrays[arrayIdx].loras.push("None");
        node.saveState();
        debouncedRenderUI(node);
    };
    controlsBar.appendChild(addLoraBtn);

    const duplicateBtn = document.createElement("button");
    duplicateBtn.className = "cb-button primary";
    duplicateBtn.textContent = "📋 Duplicate";
    duplicateBtn.onclick = () => {
        const clone = JSON.parse(JSON.stringify(configArray));
        clone.name = `${clone.name} (Copy)`;
        node.state.config_arrays.splice(arrayIdx + 1, 0, clone);
        node.saveState();
        node.renderUI();
    };
    controlsBar.appendChild(duplicateBtn);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "cb-button danger";
    deleteBtn.textContent = "🗑️ Delete";
    deleteBtn.onclick = () => {
        if (node.state.config_arrays.length > 1) {
            node.state.config_arrays.splice(arrayIdx, 1);
            node.saveState();
            node.renderUI();
        }
    };
    controlsBar.appendChild(deleteBtn);

    div.appendChild(controlsBar);

    // Label Mode overlay — shows config name at bottom center of card
    if (node.state.label_mode && configArray.name) {
        const labelOverlay = document.createElement("div");
        labelOverlay.className = "cb-config-label-overlay";
        labelOverlay.textContent = configArray.name;
        div.appendChild(labelOverlay);
    }

    return div;
}

// --- MODEL ELEMENT CREATOR ---

export function createModelElement(node, modelEntry, arrayIdx, modelIdx, modelLists) {
    // Handle both string format (legacy) and object format {path, type}
    const modelPath = typeof modelEntry === 'string' ? modelEntry : (modelEntry?.path || "None");
    const modelType = typeof modelEntry === 'string' ? 'checkpoint' : (modelEntry?.type || 'checkpoint');
    const isFolder = modelPath.endsWith("/");

    const div = document.createElement("div");
    div.className = "cb-item-card model-card";
    const uid = `${arrayIdx}_${modelIdx}`;

    // Initialize model bypass states if they don't exist
    if (!node.state.config_arrays[arrayIdx].model_bypass_states) {
        node.state.config_arrays[arrayIdx].model_bypass_states = {};
    }

    // Get bypass state
    const isBypassed = node.state.config_arrays[arrayIdx].model_bypass_states[modelPath] || false;

    // Initial State
    const isCollapsed = node.uiState.modelsCollapsed[uid] || false;

    // Header with bypass toggle
    const header = document.createElement("div");
    header.className = "cb-header-bar";

    const leftGroup = document.createElement("div");
    leftGroup.className = "cb-header-left";

    // Bypass Checkbox (in header, before toggle arrow)
    if (modelPath !== "None") {
        const bypassLabel = document.createElement("label");
        bypassLabel.style.cssText = "display: flex; align-items: center; gap: 4px; cursor: pointer; margin-right: 8px;";
        bypassLabel.title = "Bypass (disable) this Model";

        const bypassCheck = document.createElement("input");
        bypassCheck.type = "checkbox";
        bypassCheck.checked = !isBypassed; // Inverted: checked = enabled
        bypassCheck.style.cssText = "cursor: pointer;";
        bypassCheck.onclick = (e) => {
            e.stopPropagation(); // Prevent header collapse
            const newBypassState = !bypassCheck.checked;
            node.state.config_arrays[arrayIdx].model_bypass_states[modelPath] = newBypassState;
            node.saveState();
            updatePreview(node);
            // Visual feedback
            div.style.opacity = newBypassState ? "0.5" : "1.0";
            div.style.filter = newBypassState ? "grayscale(0.7)" : "none";
        };

        bypassLabel.appendChild(bypassCheck);
        const bypassText = document.createElement("span");
        bypassText.textContent = "On";
        bypassText.style.cssText = "font-size: 11px; color: #cc8844;";
        bypassLabel.appendChild(bypassText);
        leftGroup.appendChild(bypassLabel);
    }

    const toggleArrow = document.createElement("span");
    toggleArrow.textContent = isCollapsed ? "▶" : "▼";
    toggleArrow.style.color = "#aaa";
    toggleArrow.style.fontSize = "10px";
    toggleArrow.style.width = "12px";
    leftGroup.appendChild(toggleArrow);

    const label = document.createElement("span");
    label.textContent = `Model #${modelIdx + 1}`;
    label.style.color = "#aaa";
    label.style.whiteSpace = "nowrap";
    leftGroup.appendChild(label);

    // Show model type badge
    if (modelType !== "checkpoint") {
        const typeBadge = document.createElement("span");
        typeBadge.textContent = modelType === "gguf" ? "GGUF" : modelType === "ltx_video" ? "LTX" : "DM";
        typeBadge.style.cssText = "background: #553300; color: #ffaa44; padding: 1px 5px; border-radius: 3px; font-size: 9px; font-weight: bold;";
        leftGroup.appendChild(typeBadge);
    }

    const nameSpan = document.createElement("span");
    nameSpan.className = "cb-header-name";
    nameSpan.textContent = getShortName(modelPath);
    leftGroup.appendChild(nameSpan);

    header.appendChild(leftGroup);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "cb-button danger";
    deleteBtn.style.padding = "2px 6px";
    deleteBtn.style.fontSize = "10px";
    deleteBtn.textContent = "✖";
    deleteBtn.onclick = () => {
        node.state.config_arrays[arrayIdx].models.splice(modelIdx, 1);
        node.saveState();
        debouncedRenderUI(node);
    };
    header.appendChild(deleteBtn);
    div.appendChild(header);

    // Drag-and-drop reorder
    setupDragReorder(div, header, () => node.state.config_arrays[arrayIdx].models, modelIdx, node);

    // Apply bypass visual state
    if (isBypassed) {
        div.style.opacity = "0.5";
        div.style.filter = "grayscale(0.7)";
    }

    // Content Container
    const contentDiv = document.createElement("div");
    contentDiv.style.display = isCollapsed ? "none" : "flex";
    contentDiv.style.flexDirection = "column";
    contentDiv.style.gap = "6px";
    contentDiv.style.width = "100%";

    header.onclick = (e) => {
        if (e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT' || e.target.tagName === 'LABEL') return;
        const isNowCollapsed = contentDiv.style.display !== "none";
        contentDiv.style.display = isNowCollapsed ? "none" : "flex";
        toggleArrow.textContent = isNowCollapsed ? "▶" : "▼";
        node.uiState.modelsCollapsed[uid] = isNowCollapsed;
    };

    // Model Type Select (Checkpoint / Diffusion Model / GGUF)
    const modelTypeSelect = document.createElement("select");
    modelTypeSelect.className = "cb-select";
    modelTypeSelect.innerHTML = `
        <option value="checkpoint" ${modelType === 'checkpoint' ? 'selected' : ''}>Checkpoint</option>
        <option value="diffusion_model" ${modelType === 'diffusion_model' ? 'selected' : ''}>Diffusion Model</option>
        <option value="gguf" ${modelType === 'gguf' ? 'selected' : ''}>GGUF</option>
    `;
    const ltxVideoOpt = document.createElement("option");
    ltxVideoOpt.value = "ltx_video";
    ltxVideoOpt.textContent = "LTX Video";
    if (modelType === "ltx_video") ltxVideoOpt.selected = true;
    modelTypeSelect.appendChild(ltxVideoOpt);
    modelTypeSelect.onchange = () => {
        const newType = modelTypeSelect.value;
        node.state.config_arrays[arrayIdx].models[modelIdx] = newType === "checkpoint"
            ? "None"
            : { path: "None", type: newType };
        node.saveState();
        debouncedRenderUI(node);
    };
    contentDiv.appendChild(modelTypeSelect);

    // File/Folder Type Select
    const typeSelect = document.createElement("select");
    typeSelect.className = "cb-select";
    const fileLabel = modelType === "gguf" ? "GGUF File"
        : modelType === "diffusion_model" ? "Diffusion File"
        : modelType === "ltx_video" ? "LTX Diffusion File"
        : "Checkpoint File";
    typeSelect.innerHTML = `
        <option value="file" ${!isFolder ? 'selected' : ''}>${fileLabel}</option>
        <option value="folder" ${isFolder ? 'selected' : ''}>Folder</option>
    `;
    typeSelect.onchange = () => {
        const newVal = typeSelect.value === "folder" ? "/" : "None";
        if (modelType === "checkpoint") {
            node.state.config_arrays[arrayIdx].models[modelIdx] = newVal;
        } else {
            node.state.config_arrays[arrayIdx].models[modelIdx] = { path: newVal, type: modelType };
        }
        node.saveState();
        debouncedRenderUI(node);
    };
    contentDiv.appendChild(typeSelect);

    // Pick the correct file list and folder list based on model type
    let fileOptions, folderOptions;
    if (modelType === 'gguf') {
        fileOptions = modelLists.ggufModels || [];
        folderOptions = modelLists.ggufFolders || ["/"];
    } else if (modelType === 'diffusion_model' || modelType === 'ltx_video') {
        // LTX 2.3 ships as diffusion-only safetensors (separate VAEs); same source folder as diffusion_model
        fileOptions = modelLists.diffusionModels || [];
        folderOptions = modelLists.diffusionFolders || ["/"];
    } else {
        fileOptions = modelLists.checkpoints || [];
        folderOptions = modelLists.checkpointFolders || ["/"];
    }

    // Searchable Select
    const options = isFolder ? folderOptions : fileOptions;
    const currentVal = modelPath;
    const optionsList = (options && options.includes(currentVal)) || currentVal === "None" || currentVal === "/"
        ? options || ["None"]
        : [currentVal, ...(options || ["None"])];

    const nameSearchable = createSearchableSelect(
        optionsList,
        currentVal,
        (value) => {
            const normalized = normalizePath(value);
            if (modelType === "checkpoint") {
                node.state.config_arrays[arrayIdx].models[modelIdx] = normalized;
            } else {
                node.state.config_arrays[arrayIdx].models[modelIdx] = { path: normalized, type: modelType };
            }
            node.saveState();
            debouncedRenderUI(node);
        },
        isFolder ? "Search folders..." : "Search models..."
    );
    contentDiv.appendChild(nameSearchable);

    // Folder expand button
    if (isFolder && modelPath !== "None" && modelPath !== "/") {
        const expandBtn = document.createElement("button");
        expandBtn.className = "cb-button";
        expandBtn.style.borderLeft = "3px solid #cc6600";
        expandBtn.style.width = "100%";
        expandBtn.style.fontSize = "11px";
        expandBtn.style.marginTop = "4px";
        expandBtn.textContent = "📂 Add all individually";
        expandBtn.onclick = () => {
            const normalize = (str) => str.replace(/\\/g, "/");
            const folderPrefix = normalize(modelPath);
            const matchingModels = fileOptions ? fileOptions.filter(m => normalize(m).startsWith(folderPrefix)) : [];
            if (matchingModels.length > 0) {
                const expanded = matchingModels.map(m => modelType === "checkpoint" ? m : { path: m, type: modelType });
                node.state.config_arrays[arrayIdx].models.splice(modelIdx, 1, ...expanded);
                node.saveState();
                debouncedRenderUI(node);
            } else {
                alert(`No models found in folder: ${folderPrefix}`);
            }
        };
        contentDiv.appendChild(expandBtn);
    }

    // --- COLLAPSIBLE "MORE MODEL OPTIONS" SECTION ---
    if (modelPath !== "None" && !isFolder) {
        const moreOptionsUid = `${uid}-moreoptions`;
        const isMoreOptionsCollapsed = node.uiState.modelsCollapsed[moreOptionsUid] !== false; // Default collapsed

        const moreOptionsSection = document.createElement("div");
        moreOptionsSection.style.cssText = `background: #252525; border-radius: 4px; padding: 8px; margin-top: 6px; border-left: 3px solid #cc6600;`;

        const moreOptionsHeader = document.createElement("div");
        moreOptionsHeader.style.cssText = "display: flex; justify-content: space-between; align-items: center; cursor: pointer; user-select: none;";

        const moreOptionsTitle = document.createElement("div");
        moreOptionsTitle.textContent = "⚙️ More Model Options";
        moreOptionsTitle.style.cssText = "font-size: 11px; font-weight: bold; color: #cc6600;";

        const moreOptionsArrow = document.createElement("span");
        moreOptionsArrow.textContent = isMoreOptionsCollapsed ? "▶" : "▼";
        moreOptionsArrow.style.cssText = "font-size: 10px; color: #cc6600;";

        moreOptionsHeader.appendChild(moreOptionsTitle);
        moreOptionsHeader.appendChild(moreOptionsArrow);
        moreOptionsSection.appendChild(moreOptionsHeader);

        const moreOptionsContent = document.createElement("div");
        moreOptionsContent.style.display = isMoreOptionsCollapsed ? "none" : "flex";
        moreOptionsContent.style.flexDirection = "column";
        moreOptionsContent.style.gap = "8px";
        moreOptionsContent.style.marginTop = "8px";

        // Toggle handler
        moreOptionsHeader.onclick = () => {
            const isNowCollapsed = moreOptionsContent.style.display !== "none";
            moreOptionsContent.style.display = isNowCollapsed ? "none" : "flex";
            moreOptionsArrow.textContent = isNowCollapsed ? "▶" : "▼";
            node.uiState.modelsCollapsed[moreOptionsUid] = isNowCollapsed;
        };

        // Model Metadata Lookup Button
        const metadataBtn = document.createElement("button");
        metadataBtn.className = "cb-button";
        metadataBtn.style.cssText = `width: 100%; background: linear-gradient(135deg, #cc8833, #cc6600); border-left: 4px solid #ffaa44; margin-top: 4px;`;
        metadataBtn.textContent = "🔍 Lookup Model Metadata from CivitAI";
        metadataBtn.onclick = async () => await showModelMetadataModal(node, arrayIdx, modelPath, modelType);
        moreOptionsContent.appendChild(metadataBtn);

        // Show inline notice when CivitAI companion is missing (async)
        isCivitaiAvailable().then(available => {
            if (!available) moreOptionsContent.appendChild(_renderCivitaiCompanionNotice());
        });

        moreOptionsSection.appendChild(moreOptionsContent);
        contentDiv.appendChild(moreOptionsSection);
    }

    div.appendChild(contentDiv);
    return div;
}

// --- LORA ELEMENT CREATOR ---


export function createLoraElement(node, loraStr, arrayIdx, loraIdx, availableLoras, loraFolders) {
    const div = document.createElement("div");
    div.className = "cb-item-card";
    const parsed = parseLoraString(loraStr);
    const isCombined = parsed.name.endsWith("/*");
    const cleanName = parsed.name.replace(/\*$/, "");
    const isFolder = parsed.name.endsWith("/") || isCombined;
    const uid = `${arrayIdx}_${loraIdx}`;

    const isCollapsed = node.uiState.lorasCollapsed[uid] || false;
    let currentModelStr = parsed.model_str;
    let currentClipStr = parsed.clip_str;

    // Initialize state objects if they don't exist
    if (!node.state.config_arrays[arrayIdx].lora_bypass_states) {
        node.state.config_arrays[arrayIdx].lora_bypass_states = {};
    }
    if (!node.state.config_arrays[arrayIdx].lora_strength_lock) {
        node.state.config_arrays[arrayIdx].lora_strength_lock = {};
    }

    // Get bypass state
    const isBypassed = node.state.config_arrays[arrayIdx].lora_bypass_states[parsed.name] || false;

    // Get strength lock state (default to true - locked by default)
    const isStrengthLocked = node.state.config_arrays[arrayIdx].lora_strength_lock[parsed.name] !== false;

    // Header with bypass toggle
    const header = document.createElement("div");
    header.className = "cb-header-bar";

    const leftGroup = document.createElement("div");
    leftGroup.className = "cb-header-left";

    // Bypass Checkbox (in header, before toggle arrow)
    const bypassLabel = document.createElement("label");
    bypassLabel.style.cssText = "display: flex; align-items: center; gap: 4px; cursor: pointer; margin-right: 8px;";
    bypassLabel.title = "Bypass (disable) this LoRA";

    const bypassCheck = document.createElement("input");
    bypassCheck.type = "checkbox";
    bypassCheck.checked = !isBypassed; // Inverted: checked = enabled
    bypassCheck.style.cssText = "cursor: pointer;";
    bypassCheck.onclick = (e) => {
        e.stopPropagation(); // Prevent header collapse
        const newBypassState = !bypassCheck.checked;
        node.state.config_arrays[arrayIdx].lora_bypass_states[parsed.name] = newBypassState;
        node.saveState();
        updatePreview(node);
        // Visual feedback
        div.style.opacity = newBypassState ? "0.5" : "1.0";
        div.style.filter = newBypassState ? "grayscale(0.7)" : "none";
    };

    bypassLabel.appendChild(bypassCheck);
    const bypassText = document.createElement("span");
    bypassText.textContent = "On";
    bypassText.style.cssText = "font-size: 11px; color: #0cc;";
    bypassLabel.appendChild(bypassText);
    leftGroup.appendChild(bypassLabel);

    const toggleArrow = document.createElement("span");
    toggleArrow.textContent = isCollapsed ? "▶" : "▼";
    toggleArrow.style.cssText = "margin-right: 6px;";
    leftGroup.appendChild(toggleArrow);

    const label = document.createElement("span");
    label.style.color = "#0066cc";
    label.style.fontSize = "10px";
    label.style.marginRight = "6px";
    label.textContent = "LoRA:";
    leftGroup.appendChild(label);

    const nameSpan = document.createElement("span");
    nameSpan.className = "cb-header-name";
    nameSpan.textContent = getShortName(parsed.name);
    leftGroup.appendChild(nameSpan);

    header.appendChild(leftGroup);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "cb-button danger";
    deleteBtn.style.padding = "2px 6px";
    deleteBtn.style.fontSize = "10px";
    deleteBtn.textContent = "✖";
    deleteBtn.onclick = () => {
        node.state.config_arrays[arrayIdx].loras.splice(loraIdx, 1);
        node.saveState();
        debouncedRenderUI(node);
    };
    header.appendChild(deleteBtn);
    div.appendChild(header);

    // Drag-and-drop reorder
    setupDragReorder(div, header, () => node.state.config_arrays[arrayIdx].loras, loraIdx, node);

    // Apply bypass visual state
    if (isBypassed) {
        div.style.opacity = "0.5";
        div.style.filter = "grayscale(0.7)";
    }

    // Content
    const contentDiv = document.createElement("div");
    contentDiv.style.display = isCollapsed ? "none" : "flex";
    contentDiv.style.flexDirection = "column";
    contentDiv.style.gap = "6px";
    contentDiv.style.width = "100%";

    // --- TOGGLE LOGIC (INSTANT) ---
    header.onclick = (e) => {
        if (e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT') return;

        const isNowCollapsed = contentDiv.style.display !== "none";
        contentDiv.style.display = isNowCollapsed ? "none" : "flex";
        toggleArrow.textContent = isNowCollapsed ? "▶" : "▼";

        node.uiState.lorasCollapsed[uid] = isNowCollapsed;
    };

    const typeSelect = document.createElement("select");
    typeSelect.className = "cb-select";
    typeSelect.innerHTML = `
        <option value="file" ${!isFolder ? 'selected' : ''}>LoRA File</option>
        <option value="folder" ${isFolder && !isCombined ? 'selected' : ''}>Folder (Individual)</option>
        <option value="combined" ${isCombined ? 'selected' : ''}>Folder (Combined Stack)</option>
    `;
    typeSelect.onchange = () => {
        if (typeSelect.value === "folder") node.state.config_arrays[arrayIdx].loras[loraIdx] = buildLoraString("/", currentModelStr, currentClipStr);
        else if (typeSelect.value === "combined") node.state.config_arrays[arrayIdx].loras[loraIdx] = buildLoraString("/*", currentModelStr, currentClipStr);
        else node.state.config_arrays[arrayIdx].loras[loraIdx] = buildLoraString("None", currentModelStr, currentClipStr);
        node.saveState();
        debouncedRenderUI(node);
    };
    contentDiv.appendChild(typeSelect);

    const options = isFolder ? loraFolders : availableLoras;
    const currentVal = isCombined ? cleanName : parsed.name;
    const optionsList = (options && options.includes(currentVal)) || currentVal === "None" || currentVal === "/" || currentVal === ""
        ? options || ["None"]
        : [currentVal, ...(options || ["None"])];

    const nameSearchable = createSearchableSelect(
        optionsList,
        currentVal,
        (selectedName) => {
            const finalName = isCombined && !selectedName.endsWith("*") && selectedName !== "None"
                ? normalizePath(selectedName) + "*"
                : normalizePath(selectedName);
            node.state.config_arrays[arrayIdx].loras[loraIdx] = buildLoraString(finalName, currentModelStr, currentClipStr);
            node.saveState();
            debouncedRenderUI(node);
        },
        isFolder ? "Search folders..." : "Search LoRAs..."
    );
    contentDiv.appendChild(nameSearchable);

    // Check if this LoRA has weight arrays stored
    if (!node.state.config_arrays[arrayIdx].lora_weight_arrays) {
        node.state.config_arrays[arrayIdx].lora_weight_arrays = {};
    }
    const weightArrays = node.state.config_arrays[arrayIdx].lora_weight_arrays;
    // "Active" = array exists (ANY length). Length-1 arrays pass through Python
    // unchanged (only length>1 gets bracket-rewritten), so persisting [1] is safe
    // and lets us remember the "user activated compare mode" UI state across reloads.
    const isCompareActive = weightArrays[parsed.name + "_model"] !== undefined;

    const modelSlider = createSlider("Model Strength", currentModelStr, 0, 2, 0.05, (val) => {
        currentModelStr = val;
        const currentName = isCombined ? cleanName + "*" : parsed.name;

        // If strength is locked, update both sliders
        if (isStrengthLocked) {
            currentClipStr = val;
        }

        node.state.config_arrays[arrayIdx].loras[loraIdx] = buildLoraString(currentName, currentModelStr, currentClipStr);
        node.saveState();

        // Update CLIP slider if locked
        if (isStrengthLocked && clipSliderContainer) {
            const clipSliderInput = clipSliderContainer.querySelector('input[type="range"]');
            const clipNumberInput = clipSliderContainer.querySelector('input[type="number"]');
            if (clipSliderInput) clipSliderInput.value = val;
            if (clipNumberInput) clipNumberInput.value = val;
        }
    });
    contentDiv.appendChild(modelSlider);

    // === Compare Strengths section (locked-semantics) ===
    // ONE button toggles a SINGLE labeled input. We only persist
    // weightArrays[name + "_model"] — Python emits "name:[a, b, c]" (no clip
    // part) which the orchestrator expands as N configs each with
    // model=clip=value. This avoids the Cartesian explosion that
    // "name:[m]:[c]" (two arrays) would trigger.
    //
    // When Strength Lock is ON (default): label says "Strengths:" because
    // the array applies to both model and clip equally.
    // When Strength Lock is OFF: label says "Model Strengths:" because the
    // clip is fixed at the scalar slider value (set independently below).
    const compareBtnRow = document.createElement("div");
    compareBtnRow.style.cssText = "margin: -4px 0 4px 0; display: flex; align-items: center; gap: 4px;";

    const compareBtn = document.createElement("button");
    compareBtn.className = "cb-btn";
    compareBtn.textContent = isCompareActive ? "✕ Compare Strengths" : "+ Compare Strengths";
    compareBtn.style.cssText = "font-size: 9px; padding: 1px 6px; color: " + (isCompareActive ? "#f80" : "#0af") + ";";
    compareBtn.title = "Test this LoRA at multiple strength values in one run.\n\nClick to activate, then enter comma-separated values (e.g. 0.5, 0.8, 1.0). Each value generates one config — model and clip strength both equal that value (locked semantics).\n\nWith Strength Lock OFF, the scalar Clip slider below sets a fixed clip strength while model varies.\n\nClick again to deactivate.";
    compareBtnRow.appendChild(compareBtn);
    contentDiv.appendChild(compareBtnRow);

    // Single labeled input
    const compareSection = document.createElement("div");
    compareSection.style.cssText = "display: " + (isCompareActive ? "flex" : "none") + "; flex-direction: column; gap: 3px; margin: 0 0 6px 12px; padding: 4px 6px; border-left: 2px solid #0af; background: rgba(0,170,255,0.04);";

    const strengthRow = document.createElement("div");
    strengthRow.style.cssText = "display: flex; align-items: center; gap: 6px;";
    const strengthLabel = document.createElement("span");
    strengthLabel.textContent = isStrengthLocked ? "Strengths:" : "Model Strengths:";
    strengthLabel.style.cssText = "font-size: 10px; color: #ccc; white-space: nowrap; min-width: 100px;";
    const strengthInput = document.createElement("input");
    strengthInput.type = "text";
    strengthInput.placeholder = "1, 0.8, 0.5";
    strengthInput.value = isCompareActive
        ? (weightArrays[parsed.name + "_model"] || [1]).join(", ")
        : "1";
    strengthInput.style.cssText = "flex: 1; background: #1a1a1a; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 2px 6px; font-size: 10px;";
    strengthInput.onchange = () => {
        const vals = strengthInput.value.split(",").map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
        if (vals.length > 0) {
            weightArrays[parsed.name + "_model"] = vals;
        }
        // Always remove _clip — Cartesian path is intentionally dropped.
        delete weightArrays[parsed.name + "_clip"];
        node.saveState();
        updatePreview(node);
    };
    strengthRow.appendChild(strengthLabel);
    strengthRow.appendChild(strengthInput);
    compareSection.appendChild(strengthRow);
    contentDiv.appendChild(compareSection);

    compareBtn.onclick = () => {
        const currentlyActive = weightArrays[parsed.name + "_model"] !== undefined;
        if (currentlyActive) {
            // Deactivate — remove arrays and hide section
            delete weightArrays[parsed.name + "_model"];
            delete weightArrays[parsed.name + "_clip"];
            compareSection.style.display = "none";
            compareBtn.textContent = "+ Compare Strengths";
            compareBtn.style.color = "#0af";
        } else {
            // Activate — seed model array with [1], never write _clip
            weightArrays[parsed.name + "_model"] = [1];
            delete weightArrays[parsed.name + "_clip"];
            compareSection.style.display = "flex";
            compareBtn.textContent = "✕ Compare Strengths";
            compareBtn.style.color = "#f80";
            strengthInput.value = "1";
            // Focus so the user can type immediately
            strengthInput.focus();
            strengthInput.select();
        }
        node.saveState();
        updatePreview(node);
    };

    // CLIP Slider - conditionally visible based on lock state
    let clipSliderContainer = null;
    if (!isStrengthLocked) {
        clipSliderContainer = createSlider("CLIP Strength", currentClipStr, 0, 2, 0.05, (val) => {
            currentClipStr = val;
            const currentName = isCombined ? cleanName + "*" : parsed.name;
            node.state.config_arrays[arrayIdx].loras[loraIdx] = buildLoraString(currentName, currentModelStr, currentClipStr);
            node.saveState();
        });
        contentDiv.appendChild(clipSliderContainer);
    }

    // --- COLLAPSIBLE "MORE LORA OPTIONS" SECTION ---
    if (parsed.name !== "None") {
        const moreOptionsUid = `${uid}-moreoptions`;
        const isMoreOptionsCollapsed = node.uiState.lorasCollapsed[moreOptionsUid] !== false; // Default collapsed

        const moreOptionsSection = document.createElement("div");
        moreOptionsSection.style.cssText = `background: #252525; border-radius: 4px; padding: 8px; margin-top: 6px; border-left: 3px solid #9966cc;`;

        const moreOptionsHeader = document.createElement("div");
        moreOptionsHeader.style.cssText = "display: flex; justify-content: space-between; align-items: center; cursor: pointer; user-select: none;";

        const moreOptionsTitle = document.createElement("div");
        moreOptionsTitle.textContent = "⚙️ More LoRA Options";
        moreOptionsTitle.style.cssText = "font-size: 11px; font-weight: bold; color: #9966cc;";

        const moreOptionsArrow = document.createElement("span");
        moreOptionsArrow.textContent = isMoreOptionsCollapsed ? "▶" : "▼";
        moreOptionsArrow.style.cssText = "font-size: 10px; color: #9966cc;";

        moreOptionsHeader.appendChild(moreOptionsTitle);
        moreOptionsHeader.appendChild(moreOptionsArrow);
        moreOptionsSection.appendChild(moreOptionsHeader);

        const moreOptionsContent = document.createElement("div");
        moreOptionsContent.style.display = isMoreOptionsCollapsed ? "none" : "flex";
        moreOptionsContent.style.flexDirection = "column";
        moreOptionsContent.style.gap = "8px";
        moreOptionsContent.style.marginTop = "8px";

        // Toggle handler
        moreOptionsHeader.onclick = () => {
            const isNowCollapsed = moreOptionsContent.style.display !== "none";
            moreOptionsContent.style.display = isNowCollapsed ? "none" : "flex";
            moreOptionsArrow.textContent = isNowCollapsed ? "▶" : "▼";
            node.uiState.lorasCollapsed[moreOptionsUid] = isNowCollapsed;
        };

        // 1. Strength Lock Checkbox
        const strengthLockLabel = document.createElement("label");
        strengthLockLabel.style.cssText = "display: flex; align-items: center; gap: 6px; font-size: 12px; cursor: pointer;";

        const strengthLockCheck = document.createElement("input");
        strengthLockCheck.type = "checkbox";
        strengthLockCheck.checked = isStrengthLocked;
        strengthLockCheck.onchange = () => {
            node.state.config_arrays[arrayIdx].lora_strength_lock[parsed.name] = strengthLockCheck.checked;

            // If locking, sync CLIP to Model strength
            if (strengthLockCheck.checked) {
                currentClipStr = currentModelStr;
                const currentName = isCombined ? cleanName + "*" : parsed.name;
                node.state.config_arrays[arrayIdx].loras[loraIdx] = buildLoraString(currentName, currentModelStr, currentClipStr);
            }

            node.saveState();
            debouncedRenderUI(node); // Re-render to show/hide CLIP slider
        };

        strengthLockLabel.appendChild(strengthLockCheck);
        strengthLockLabel.appendChild(document.createTextNode("🔒 Lock Model & CLIP Strength Together"));
        moreOptionsContent.appendChild(strengthLockLabel);

        // 2. Auto Append Trigger Words Section
        const triggerSubSection = document.createElement("div");
        triggerSubSection.style.cssText = `background: #2a2a2a; border-radius: 4px; padding: 8px; border-left: 3px solid #00aa88;`;

        const triggerTitle = document.createElement("div");
        triggerTitle.textContent = "🏷️ Auto Append LoRA Trigger Words To:";
        triggerTitle.style.cssText = "font-size: 11px; font-weight: bold; color: #00aa88; margin-bottom: 6px;";
        triggerSubSection.appendChild(triggerTitle);

        if (!node.state.config_arrays[arrayIdx].lora_triggerwords_append_settings) {
            node.state.config_arrays[arrayIdx].lora_triggerwords_append_settings = {};
        }
        const currentPlacement = node.state.config_arrays[arrayIdx].lora_triggerwords_append_settings[parsed.name] || "none";

        const checkboxContainer = document.createElement("div");
        checkboxContainer.style.cssText = "display: flex; gap: 12px; align-items: center; flex-wrap: wrap;";

        const createCheck = (lbl, val) => {
            const label = document.createElement("label");
            label.style.cssText = "display: flex; align-items: center; gap: 4px; font-size: 12px; cursor: pointer;";
            const check = document.createElement("input");
            check.type = "checkbox";
            check.checked = currentPlacement === val;
            check.onchange = async () => {
                // Special handling for "dont_append"
                if (val === "dont_append" && check.checked) {
                    // Fetch triggers and add them to omit list
                    const triggers = await fetchLoraTriggersForOmit(node, arrayIdx, parsed.name);
                    if (triggers && triggers.length > 0) {
                        if (!node.state.config_arrays[arrayIdx].lora_omit_triggers) {
                            node.state.config_arrays[arrayIdx].lora_omit_triggers = [];
                        }
                        triggers.forEach(trigger => {
                            if (!node.state.config_arrays[arrayIdx].lora_omit_triggers.includes(trigger)) {
                                node.state.config_arrays[arrayIdx].lora_omit_triggers.push(trigger);
                            }
                        });
                    }
                }

                node.state.config_arrays[arrayIdx].lora_triggerwords_append_settings[parsed.name] = check.checked ? val : "none";
                node.saveState();

                // Manually uncheck the others
                if (check.checked) {
                    const other = checkboxContainer.querySelectorAll('input');
                    other.forEach(i => { if (i !== check) i.checked = false; });
                }
            };
            label.appendChild(check);
            label.appendChild(document.createTextNode(lbl));
            return label;
        };

        checkboxContainer.appendChild(createCheck("Start", "start"));
        checkboxContainer.appendChild(createCheck("End", "end"));
        checkboxContainer.appendChild(createCheck("Don't Append", "dont_append"));

        triggerSubSection.appendChild(checkboxContainer);

        // Info text for Don't Append
        const dontAppendInfo = document.createElement("div");
        dontAppendInfo.style.cssText = "font-size: 10px; color: #888; font-style: italic; margin-top: 4px;";
        dontAppendInfo.textContent = "ℹ️ 'Don't Append' adds all trigger words to the omit list";
        triggerSubSection.appendChild(dontAppendInfo);

        // Show inline notice when CivitAI companion is missing (async)
        isCivitaiAvailable().then(available => {
            if (!available) triggerSubSection.appendChild(_renderCivitaiCompanionNotice());
        });

        moreOptionsContent.appendChild(triggerSubSection);

        // 3. LoRA Metadata Lookup Button
        const metadataBtn = document.createElement("button");
        metadataBtn.className = "cb-button";
        metadataBtn.style.cssText = `width: 100%; background: linear-gradient(135deg, #cc6699, #9966cc); border-left: 4px solid #ff66cc; margin-top: 4px;`;
        metadataBtn.textContent = "🔍 Lookup LoRA Metadata from CivitAI";
        metadataBtn.onclick = async () => await showLoraMetadataModal(node, arrayIdx, parsed.name);
        moreOptionsContent.appendChild(metadataBtn);

        // Show inline notice when CivitAI companion is missing (async)
        isCivitaiAvailable().then(available => {
            if (!available) moreOptionsContent.appendChild(_renderCivitaiCompanionNotice());
        });

        // 4. Edit Trigger Words Button
        const editTriggersBtn = document.createElement("button");
        editTriggersBtn.className = "cb-button";
        editTriggersBtn.style.cssText = `width: 100%; background: linear-gradient(135deg, #336633, #446644); border-left: 4px solid #66cc66; margin-top: 4px;`;
        editTriggersBtn.textContent = "✏️ Edit Trigger Words";
        editTriggersBtn.onclick = async () => await showEditTriggersModal(node, arrayIdx, parsed.name);
        moreOptionsContent.appendChild(editTriggersBtn);

        moreOptionsSection.appendChild(moreOptionsContent);
        contentDiv.appendChild(moreOptionsSection);
    }

    // Folder expand button (outside More Options)
    if (isFolder && parsed.name !== "None") {
        const expandBtn = document.createElement("button");
        expandBtn.className = "cb-button";
        expandBtn.style.cssText = "width: 100%; border-left: 3px solid #0066cc; font-size: 11px; margin-top: 4px;";
        expandBtn.textContent = "📂 Add all individually";
        expandBtn.onclick = () => {
            const normalize = (str) => str.replace(/\\/g, "/");
            let matchingLoras;
            if (cleanName === "/" || cleanName === "") matchingLoras = availableLoras || [];
            else {
                const folderPrefix = normalize(cleanName);
                matchingLoras = availableLoras ? availableLoras.filter(l => normalize(l).startsWith(folderPrefix)) : [];
            }
            if (matchingLoras.length > 0) {
                const withStrengths = matchingLoras.map(l => buildLoraString(l, parsed.model_str, parsed.clip_str));
                node.state.config_arrays[arrayIdx].loras.splice(loraIdx, 1, ...withStrengths);
                node.saveState();
                debouncedRenderUI(node);
            } else {
                alert(`No LoRAs found in folder: ${cleanName}`);
            }
        };
        contentDiv.appendChild(expandBtn);
    }

    div.appendChild(contentDiv);
    return div;
}

// Helper function to fetch triggers for "Don't Append" option
async function fetchLoraTriggersForOmit(node, arrayIdx, loraName) {
    try {
        const resp = await fetch("/configbuilder/lookup_triggers", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ loras: [loraName] })
        });

        if (resp.ok) {
            const data = await resp.json();
            return data.triggers[loraName] || [];
        }
    } catch (e) {
        console.error("[ConfigBuilder] Error fetching triggers for omit:", e);
    }
    return [];
}

// New modal for LoRA metadata lookup
async function showLoraMetadataModal(node, arrayIdx, loraName, forceRefresh = false) {
    const overlay = document.createElement("div");
    overlay.style.cssText = `position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.85); display: flex; align-items: center; justify-content: center; z-index: 10000;`;

    const modal = document.createElement("div");
    // Red X close button in top right
    modal.classList.add("cb-modal-popup");
    const closeX = document.createElement("button");
    closeX.textContent = "✖";
    closeX.style.cssText = "position: absolute; top: 10px; right: 10px; background: #cc3333; color: white; border: none; border-radius: 4px; width: 28px; height: 28px; font-size: 16px; cursor: pointer; display: flex; align-items: center; justify-content: center; z-index: 1;";
    closeX.onmouseover = () => closeX.style.background = "#dd4444";
    closeX.onmouseout = () => closeX.style.background = "#cc3333";
    // Shared close function to clean up modal and event listener
    const closeModal = () => {
        if (overlay.parentNode) document.body.removeChild(overlay);
        document.removeEventListener('keydown', escHandler);
    };

    closeX.onclick = closeModal;
    modal.appendChild(closeX);

    // Close on Escape key
    const escHandler = (e) => {
        if (e.key === 'Escape') closeModal();
    };
    document.addEventListener('keydown', escHandler);

    const title = document.createElement("h3");
    title.textContent = "🔍 LoRA Metadata Lookup";
    title.style.cssText = "margin: 0 0 15px 0; color: #9966cc;";
    modal.appendChild(title);

    // CivitAI companion notice — shows when companion missing
    const _modalNoticeSlot = document.createElement('div');
    modal.appendChild(_modalNoticeSlot);
    isCivitaiAvailable().then(available => {
        if (!available) _modalNoticeSlot.replaceWith(_renderCivitaiCompanionNotice());
    });

    const status = document.createElement("div");
    status.textContent = `🔄 Fetching metadata for: ${loraName.split('/').pop()} (This could take a few seconds the fist time)`;
    status.style.cssText = "margin-bottom: 15px; color: #aaa;";
    modal.appendChild(status);

    const content = document.createElement("div");
    modal.appendChild(content);

    const closeBtn = document.createElement("button");
    closeBtn.className = "cb-button";
    closeBtn.textContent = "Close";
    closeBtn.style.marginTop = "15px";
    closeBtn.onclick = closeModal;

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    // Fetch metadata
    try {
        const resp = await fetch("/configbuilder/lookup_lora_metadata", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lora_name: loraName, force_refresh: forceRefresh })
        });

        if (!resp.ok) {
            const errorData = await resp.json();
            status.textContent = "❌ Error: " + (errorData.error || "Failed to fetch metadata");
            status.style.color = "#ff6666";
            modal.appendChild(closeBtn);
            return;
        }

        const data = await resp.json();
        const metadata = data.metadata;

        status.textContent = "✅ Metadata loaded successfully!";
        status.style.color = "#66ff66";

        // Display metadata
        content.innerHTML = "";

        // Tag diff banner (shown when force_refresh detects different tags)
        if (data.tags_changed) {
            const tagDiffBanner = document.createElement("div");
            tagDiffBanner.style.cssText = "background: #1a3300; border: 2px solid #44cc00; border-radius: 6px; padding: 10px 14px; margin-bottom: 15px;";

            const tagDiffTitle = document.createElement("div");
            tagDiffTitle.style.cssText = "font-size: 14px; font-weight: bold; color: #44cc00; margin-bottom: 8px;";
            tagDiffTitle.textContent = "🔄 Trigger Words Changed";
            tagDiffBanner.appendChild(tagDiffTitle);

            const oldTagsDiv = document.createElement("div");
            oldTagsDiv.style.cssText = "font-size: 11px; color: #cc6666; margin-bottom: 4px;";
            oldTagsDiv.textContent = "Old: " + (data.old_tags.length > 0 ? data.old_tags.join(", ") : "(none)");
            tagDiffBanner.appendChild(oldTagsDiv);

            const newTagsDiv = document.createElement("div");
            newTagsDiv.style.cssText = "font-size: 11px; color: #66cc66; margin-bottom: 8px;";
            newTagsDiv.textContent = "New: " + data.new_tags.join(", ");
            tagDiffBanner.appendChild(newTagsDiv);

            const tagBtnRow = document.createElement("div");
            tagBtnRow.style.cssText = "display: flex; gap: 8px;";

            const updateTagsBtn = document.createElement("button");
            updateTagsBtn.className = "cb-button primary";
            updateTagsBtn.style.cssText = "flex: 1; background: #336600; border: 1px solid #44cc00; color: #88ff44;";
            updateTagsBtn.textContent = "✅ Update Tags";
            updateTagsBtn.onclick = async () => {
                try {
                    const saveResp = await fetch("/configbuilder/save_lora_triggers", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ lora_name: loraName, triggers: data.new_tags })
                    });
                    if (saveResp.ok) {
                        updateTagsBtn.textContent = "✅ Updated!";
                        updateTagsBtn.disabled = true;
                        keepTagsBtn.disabled = true;
                    }
                } catch (e) {
                    updateTagsBtn.textContent = "❌ Error";
                }
            };

            const keepTagsBtn = document.createElement("button");
            keepTagsBtn.className = "cb-button";
            keepTagsBtn.style.cssText = "flex: 1;";
            keepTagsBtn.textContent = "Keep Current";
            keepTagsBtn.onclick = () => {
                tagDiffBanner.style.display = "none";
            };

            tagBtnRow.appendChild(updateTagsBtn);
            tagBtnRow.appendChild(keepTagsBtn);
            tagDiffBanner.appendChild(tagBtnRow);
            content.appendChild(tagDiffBanner);
        }

        // Cache warning banner (shown prominently when data is from disk cache)
        if (data.cached) {
            const cacheBanner = document.createElement("div");
            cacheBanner.style.cssText = "background: #553300; border: 2px solid #ffaa00; border-radius: 6px; padding: 10px 14px; margin-bottom: 15px; text-align: center;";
            cacheBanner.innerHTML = `
                <div style="font-size: 16px; font-weight: bold; color: #ffaa00;">⚠️ READ FROM DISK CACHE</div>
                <div style="font-size: 12px; color: #ccaa66; margin-top: 4px;">Last looked up on: <strong>${data.cache_date || 'Unknown'}</strong></div>
            `;
            const refetchBtn = document.createElement("button");
            refetchBtn.className = "cb-button";
            refetchBtn.style.cssText = "margin-top: 8px; background: #664400; border: 1px solid #ffaa00; color: #ffcc44; font-size: 12px; padding: 6px 16px;";
            refetchBtn.textContent = "🔄 Re-fetch from CivitAI Now";
            refetchBtn.onclick = async () => {
                refetchBtn.disabled = true;
                refetchBtn.textContent = "🔄 Fetching...";
                closeModal();
                await showLoraMetadataModal(node, arrayIdx, loraName, true);
            };
            cacheBanner.appendChild(refetchBtn);
            content.appendChild(cacheBanner);
        }

        // Model name and creator
        const headerSection = document.createElement("div");
        headerSection.style.cssText = "margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #444;";
        headerSection.innerHTML = `
            <div style="font-size: 18px; font-weight: bold; color: #ff66cc; margin-bottom: 5px;">${metadata.model_name}</div>
            <div style="font-size: 14px; color: #aaa;">Version: ${metadata.name}</div>
            <div style="font-size: 12px; color: #888;">Creator: ${metadata.creator}</div>
            <div style="font-size: 11px; color: #666; margin-top: 5px;">
                Hash: <code style="background: #1a1a1a; padding: 2px 6px; border-radius: 3px;">${metadata.short_hash}</code>
            </div>
        `;
        content.appendChild(headerSection);

        // Base Model and Tags
        const infoSection = document.createElement("div");
        infoSection.style.cssText = "margin-bottom: 15px;";
        infoSection.innerHTML = `
            <div style="margin-bottom: 8px;"><strong style="color: #9966cc;">Base Model:</strong> ${metadata.base_model}</div>
            <div style="margin-bottom: 8px;">
                <strong style="color: #9966cc;">Tags:</strong>
                <div style="margin-top: 4px; display: flex; flex-wrap: wrap; gap: 4px;">
                    ${metadata.tags.slice(0, 10).map(tag => `<span style="background: #444; padding: 2px 8px; border-radius: 10px; font-size: 11px;">${tag}</span>`).join('')}
                </div>
            </div>
        `;
        content.appendChild(infoSection);

        // Trigger Words
        if (metadata.trained_words && metadata.trained_words.length > 0) {
            const triggerSection = document.createElement("div");
            triggerSection.style.cssText = "margin-bottom: 15px; background: #252525; padding: 10px; border-radius: 4px; border-left: 3px solid #00aa88;";
            triggerSection.innerHTML = `
                <div style="font-weight: bold; color: #00aa88; margin-bottom: 6px;">🏷️ Trigger Words:</div>
                <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                    ${metadata.trained_words.map(word => `<span style="background: #333; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-family: monospace;">${word}</span>`).join('')}
                </div>
            `;
            content.appendChild(triggerSection);
        }

        // Images
        if (metadata.images && metadata.images.length > 0) {
            const imagesSection = document.createElement("div");
            imagesSection.style.cssText = "margin-bottom: 15px;";
            imagesSection.innerHTML = `<div style="font-weight: bold; color: #9966cc; margin-bottom: 8px;">📸 Example Images:</div>`;

            const imageGrid = document.createElement("div");
            imageGrid.style.cssText = "display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px;";

            metadata.images.forEach(img => {
                const imgContainer = document.createElement("div");
                imgContainer.style.cssText = "position: relative; aspect-ratio: 1; overflow: hidden; border-radius: 4px; border: 1px solid #444; cursor: pointer;";
                imgContainer.title = "Click to open full size";

                const imgElem = document.createElement("img");
                imgElem.src = img.url;
                imgElem.style.cssText = "width: 100%; height: 100%; object-fit: cover;";
                imgElem.onclick = () => window.open(img.url, '_blank');

                imgContainer.appendChild(imgElem);
                imageGrid.appendChild(imgContainer);
            });

            imagesSection.appendChild(imageGrid);
            content.appendChild(imagesSection);
        }

        // Links
        const linksSection = document.createElement("div");
        linksSection.style.cssText = "margin-top: 15px; padding-top: 10px; border-top: 2px solid #444;";

        const civitaiLink = document.createElement("a");
        civitaiLink.href = metadata.url;
        civitaiLink.target = "_blank";
        civitaiLink.textContent = "🌐 View on CivitAI";
        civitaiLink.style.cssText = "display: inline-block; background: #0066cc; color: white; padding: 8px 16px; border-radius: 4px; text-decoration: none; margin-right: 10px;";

        const savedPath = document.createElement("div");
        savedPath.style.cssText = "margin-top: 8px; font-size: 11px; color: #666;";
        savedPath.textContent = `💾 Metadata saved to: ${data.saved_to}`;

        linksSection.appendChild(civitaiLink);
        linksSection.appendChild(savedPath);
        content.appendChild(linksSection);

        // --- Full JSON Response Section ---
        const jsonSection = document.createElement("div");
        jsonSection.style.cssText = "margin-top: 15px; padding-top: 10px; border-top: 2px solid #444;";

        // Toggle button to show/hide full JSON
        const jsonToggleBtn = document.createElement("button");
        jsonToggleBtn.className = "cb-button";
        jsonToggleBtn.style.cssText = "margin-bottom: 8px; background: #333; border-left: 3px solid #9966cc;";
        jsonToggleBtn.textContent = "📋 Show Full JSON Response";
        const jsonPre = document.createElement("pre");
        jsonPre.style.cssText = "display: none; background: #111; color: #aaffaa; padding: 10px; border-radius: 4px; font-size: 11px; max-height: 400px; overflow: auto; white-space: pre-wrap; word-break: break-all; border: 1px solid #333;";
        jsonPre.textContent = JSON.stringify(metadata, null, 2);
        jsonToggleBtn.onclick = () => {
            const isHidden = jsonPre.style.display === "none";
            jsonPre.style.display = isHidden ? "block" : "none";
            jsonToggleBtn.textContent = isHidden ? "📋 Hide Full JSON Response" : "📋 Show Full JSON Response";
        };
        jsonSection.appendChild(jsonToggleBtn);
        jsonSection.appendChild(jsonPre);

        // Save Full JSON button
        const saveJsonBtn = document.createElement("button");
        saveJsonBtn.className = "cb-button";
        saveJsonBtn.style.cssText = "background: #333; border-left: 3px solid #00aa88; margin-top: 4px;";
        saveJsonBtn.textContent = "💾 Save Full JSON to File";
        saveJsonBtn.onclick = () => {
            const blob = new Blob([JSON.stringify(metadata, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `${loraName.split('/').pop().replace(/\.\w+$/, '')}_metadata.json`;
            a.click();
            URL.revokeObjectURL(url);
            saveJsonBtn.textContent = "✅ Saved!";
            setTimeout(() => { saveJsonBtn.textContent = "💾 Save Full JSON to File"; }, 2000);
        };
        jsonSection.appendChild(saveJsonBtn);

        content.appendChild(jsonSection);

    } catch (e) {
        status.textContent = "❌ Error: " + e.message;
        status.style.color = "#ff6666";
        console.error("[ConfigBuilder] Metadata lookup error:", e);
    }

    modal.appendChild(closeBtn);
}

// Modal for Model/Checkpoint metadata lookup from CivitAI
async function showModelMetadataModal(node, arrayIdx, modelName, modelType, forceRefresh = false) {
    const overlay = document.createElement("div");
    overlay.style.cssText = `position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.85); display: flex; align-items: center; justify-content: center; z-index: 10000;`;

    const modal = document.createElement("div");
    modal.classList.add("cb-modal-popup");

    // Red X close button
    const closeX = document.createElement("button");
    closeX.textContent = "✖";
    closeX.style.cssText = "position: absolute; top: 10px; right: 10px; background: #cc3333; color: white; border: none; border-radius: 4px; width: 28px; height: 28px; font-size: 16px; cursor: pointer; display: flex; align-items: center; justify-content: center; z-index: 1;";
    closeX.onmouseover = () => closeX.style.background = "#dd4444";
    closeX.onmouseout = () => closeX.style.background = "#cc3333";

    const closeModal = () => {
        if (overlay.parentNode) document.body.removeChild(overlay);
        document.removeEventListener('keydown', escHandler);
    };

    closeX.onclick = closeModal;
    modal.appendChild(closeX);

    // Close on Escape key
    const escHandler = (e) => {
        if (e.key === 'Escape') closeModal();
    };
    document.addEventListener('keydown', escHandler);

    const title = document.createElement("h3");
    title.textContent = "🔍 Model Metadata Lookup";
    title.style.cssText = "margin: 0 0 15px 0; color: #cc6600;";
    modal.appendChild(title);

    const status = document.createElement("div");
    status.textContent = `🔄 Fetching metadata for: ${modelName.split('/').pop()} (This could take a few seconds the fist time)`;
    status.style.cssText = "margin-bottom: 15px; color: #aaa;";
    modal.appendChild(status);

    const content = document.createElement("div");
    modal.appendChild(content);

    const closeBtn = document.createElement("button");
    closeBtn.className = "cb-button";
    closeBtn.textContent = "Close";
    closeBtn.style.marginTop = "15px";
    closeBtn.onclick = closeModal;

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    // Fetch metadata
    try {
        const resp = await fetch("/configbuilder/lookup_model_metadata", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_name: modelName, model_type: modelType, force_refresh: forceRefresh })
        });

        if (!resp.ok) {
            const errorData = await resp.json();
            status.textContent = "❌ Error: " + (errorData.error || "Failed to fetch metadata");
            status.style.color = "#ff6666";

            // If we got a hash back even though CivitAI didn't have it, show it
            if (errorData.short_hash) {
                const hashInfo = document.createElement("div");
                hashInfo.style.cssText = "margin-top: 8px; font-size: 11px; color: #888;";
                hashInfo.textContent = `Hash: ${errorData.short_hash}`;
                content.appendChild(hashInfo);
            }

            modal.appendChild(closeBtn);
            return;
        }

        const data = await resp.json();
        const metadata = data.metadata;

        status.textContent = "✅ Metadata loaded successfully!";
        status.style.color = "#66ff66";

        // Display metadata
        content.innerHTML = "";

        // Cache warning banner (shown prominently when data is from disk cache)
        if (data.cached) {
            const cacheBanner = document.createElement("div");
            cacheBanner.style.cssText = "background: #553300; border: 2px solid #ffaa00; border-radius: 6px; padding: 10px 14px; margin-bottom: 15px; text-align: center;";
            cacheBanner.innerHTML = `
                <div style="font-size: 16px; font-weight: bold; color: #ffaa00;">⚠️ READ FROM DISK CACHE</div>
                <div style="font-size: 12px; color: #ccaa66; margin-top: 4px;">Last looked up on: <strong>${data.cache_date || 'Unknown'}</strong></div>
            `;
            const refetchBtn = document.createElement("button");
            refetchBtn.className = "cb-button";
            refetchBtn.style.cssText = "margin-top: 8px; background: #664400; border: 1px solid #ffaa00; color: #ffcc44; font-size: 12px; padding: 6px 16px;";
            refetchBtn.textContent = "🔄 Re-fetch from CivitAI Now";
            refetchBtn.onclick = async () => {
                refetchBtn.disabled = true;
                refetchBtn.textContent = "🔄 Fetching...";
                closeModal();
                await showModelMetadataModal(node, arrayIdx, modelName, modelType, true);
            };
            cacheBanner.appendChild(refetchBtn);
            content.appendChild(cacheBanner);
        }

        // Model name and creator
        const headerSection = document.createElement("div");
        headerSection.style.cssText = "margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #444;";
        headerSection.innerHTML = `
            <div style="font-size: 18px; font-weight: bold; color: #ffaa44; margin-bottom: 5px;">${metadata.model_name}</div>
            <div style="font-size: 14px; color: #aaa;">Version: ${metadata.name}</div>
            <div style="font-size: 12px; color: #888;">Creator: ${metadata.creator}</div>
            <div style="font-size: 11px; color: #666; margin-top: 5px;">
                Hash: <code style="background: #1a1a1a; padding: 2px 6px; border-radius: 3px;">${metadata.short_hash}</code>
            </div>
        `;
        content.appendChild(headerSection);

        // Base Model and Tags
        const infoSection = document.createElement("div");
        infoSection.style.cssText = "margin-bottom: 15px;";
        infoSection.innerHTML = `
            <div style="margin-bottom: 8px;"><strong style="color: #cc6600;">Base Model:</strong> ${metadata.base_model}</div>
            <div style="margin-bottom: 8px;">
                <strong style="color: #cc6600;">Tags:</strong>
                <div style="margin-top: 4px; display: flex; flex-wrap: wrap; gap: 4px;">
                    ${metadata.tags.slice(0, 10).map(tag => `<span style="background: #444; padding: 2px 8px; border-radius: 10px; font-size: 11px;">${tag}</span>`).join('')}
                </div>
            </div>
        `;
        content.appendChild(infoSection);

        // Trigger Words (some models have them too)
        if (metadata.trained_words && metadata.trained_words.length > 0) {
            const triggerSection = document.createElement("div");
            triggerSection.style.cssText = "margin-bottom: 15px; background: #252525; padding: 10px; border-radius: 4px; border-left: 3px solid #00aa88;";
            triggerSection.innerHTML = `
                <div style="font-weight: bold; color: #00aa88; margin-bottom: 6px;">🏷️ Trigger Words:</div>
                <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                    ${metadata.trained_words.map(word => `<span style="background: #333; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-family: monospace;">${word}</span>`).join('')}
                </div>
            `;
            content.appendChild(triggerSection);
        }

        // Images
        if (metadata.images && metadata.images.length > 0) {
            const imagesSection = document.createElement("div");
            imagesSection.style.cssText = "margin-bottom: 15px;";
            imagesSection.innerHTML = `<div style="font-weight: bold; color: #cc6600; margin-bottom: 8px;">📸 Example Images:</div>`;

            const imageGrid = document.createElement("div");
            imageGrid.style.cssText = "display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px;";

            metadata.images.forEach(img => {
                const imgContainer = document.createElement("div");
                imgContainer.style.cssText = "position: relative; aspect-ratio: 1; overflow: hidden; border-radius: 4px; border: 1px solid #444; cursor: pointer;";
                imgContainer.title = "Click to open full size";

                const imgElem = document.createElement("img");
                imgElem.src = img.url;
                imgElem.style.cssText = "width: 100%; height: 100%; object-fit: cover;";
                imgElem.onclick = () => window.open(img.url, '_blank');

                imgContainer.appendChild(imgElem);
                imageGrid.appendChild(imgContainer);
            });

            imagesSection.appendChild(imageGrid);
            content.appendChild(imagesSection);
        }

        // Links
        const linksSection = document.createElement("div");
        linksSection.style.cssText = "margin-top: 15px; padding-top: 10px; border-top: 2px solid #444;";

        const civitaiLink = document.createElement("a");
        civitaiLink.href = metadata.url;
        civitaiLink.target = "_blank";
        civitaiLink.textContent = "🌐 View on CivitAI";
        civitaiLink.style.cssText = "display: inline-block; background: #0066cc; color: white; padding: 8px 16px; border-radius: 4px; text-decoration: none; margin-right: 10px;";

        const savedPath = document.createElement("div");
        savedPath.style.cssText = "margin-top: 8px; font-size: 11px; color: #666;";
        savedPath.textContent = `💾 Metadata saved to: ${data.saved_to}`;

        linksSection.appendChild(civitaiLink);
        linksSection.appendChild(savedPath);
        content.appendChild(linksSection);

        // --- Full JSON Response Section ---
        const jsonSection = document.createElement("div");
        jsonSection.style.cssText = "margin-top: 15px; padding-top: 10px; border-top: 2px solid #444;";

        // Toggle button to show/hide full JSON
        const jsonToggleBtn = document.createElement("button");
        jsonToggleBtn.className = "cb-button";
        jsonToggleBtn.style.cssText = "margin-bottom: 8px; background: #333; border-left: 3px solid #cc6600;";
        jsonToggleBtn.textContent = "📋 Show Full JSON Response";
        const jsonPre = document.createElement("pre");
        jsonPre.style.cssText = "display: none; background: #111; color: #aaffaa; padding: 10px; border-radius: 4px; font-size: 11px; max-height: 400px; overflow: auto; white-space: pre-wrap; word-break: break-all; border: 1px solid #333;";
        jsonPre.textContent = JSON.stringify(metadata, null, 2);
        jsonToggleBtn.onclick = () => {
            const isHidden = jsonPre.style.display === "none";
            jsonPre.style.display = isHidden ? "block" : "none";
            jsonToggleBtn.textContent = isHidden ? "📋 Hide Full JSON Response" : "📋 Show Full JSON Response";
        };
        jsonSection.appendChild(jsonToggleBtn);
        jsonSection.appendChild(jsonPre);

        // Save Full JSON button
        const saveJsonBtn = document.createElement("button");
        saveJsonBtn.className = "cb-button";
        saveJsonBtn.style.cssText = "background: #333; border-left: 3px solid #00aa88; margin-top: 4px;";
        saveJsonBtn.textContent = "💾 Save Full JSON to File";
        saveJsonBtn.onclick = () => {
            const blob = new Blob([JSON.stringify(metadata, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `${modelName.split('/').pop().replace(/\.\w+$/, '')}_metadata.json`;
            a.click();
            URL.revokeObjectURL(url);
            saveJsonBtn.textContent = "✅ Saved!";
            setTimeout(() => { saveJsonBtn.textContent = "💾 Save Full JSON to File"; }, 2000);
        };
        jsonSection.appendChild(saveJsonBtn);

        content.appendChild(jsonSection);

    } catch (e) {
        status.textContent = "❌ Error: " + e.message;
        status.style.color = "#ff6666";
        console.error("[ConfigBuilder] Model metadata lookup error:", e);
    }

    modal.appendChild(closeBtn);
}

// ============================================================
// Trigger Word Editor Modal
// ============================================================
async function showEditTriggersModal(node, arrayIdx, loraName) {
    // Build modal overlay (same pattern as metadata modals)
    const overlay = document.createElement("div");
    overlay.style.cssText = "position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.85); z-index: 10000; display: flex; align-items: center; justify-content: center;";

    const modal = document.createElement("div");
    modal.style.cssText = "background: #1a1a1a; border: 2px solid #66cc66; border-radius: 12px; padding: 25px; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto; position: relative;";

    const closeModal = () => { if (document.body.contains(overlay)) document.body.removeChild(overlay); };
    overlay.onclick = (e) => { if (e.target === overlay) closeModal(); };
    document.addEventListener("keydown", function escHandler(e) {
        if (e.key === "Escape") { closeModal(); document.removeEventListener("keydown", escHandler); }
    });

    // Close X button
    const closeX = document.createElement("button");
    closeX.textContent = "✕";
    closeX.style.cssText = "position: absolute; top: 10px; right: 15px; background: none; border: none; color: #ff4444; font-size: 20px; cursor: pointer;";
    closeX.onclick = closeModal;
    modal.appendChild(closeX);

    // Title
    const title = document.createElement("h3");
    title.textContent = "✏️ Edit Trigger Words";
    title.style.cssText = "margin: 0 0 5px 0; color: #66cc66;";
    modal.appendChild(title);

    const subtitle = document.createElement("div");
    subtitle.style.cssText = "font-size: 12px; color: #888; margin-bottom: 15px;";
    subtitle.textContent = loraName.split('/').pop();
    modal.appendChild(subtitle);

    const status = document.createElement("div");
    status.textContent = "🔄 Loading trigger words...";
    status.style.cssText = "margin-bottom: 15px; color: #aaa; font-size: 12px;";
    modal.appendChild(status);

    const chipsContainer = document.createElement("div");
    chipsContainer.style.cssText = "display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 15px; min-height: 30px; padding: 8px; background: #111; border: 1px solid #333; border-radius: 6px;";
    modal.appendChild(chipsContainer);

    // Input row
    const inputRow = document.createElement("div");
    inputRow.style.cssText = "display: flex; gap: 6px; margin-bottom: 15px;";
    const input = document.createElement("input");
    input.className = "cb-input";
    input.type = "text";
    input.placeholder = "Add new trigger word...";
    input.style.cssText = "flex: 1; padding: 6px 10px; font-size: 12px;";
    const addWordBtn = document.createElement("button");
    addWordBtn.className = "cb-button primary";
    addWordBtn.textContent = "+ Add";
    addWordBtn.style.cssText = "padding: 6px 12px; font-size: 12px;";
    inputRow.appendChild(input);
    inputRow.appendChild(addWordBtn);
    modal.appendChild(inputRow);

    // Button row
    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display: flex; gap: 8px; justify-content: flex-end; margin-top: 10px;";

    const saveBtn = document.createElement("button");
    saveBtn.className = "cb-button";
    saveBtn.style.cssText = "background: #336633; border: 1px solid #66cc66; color: #88ff88; padding: 8px 20px;";
    saveBtn.textContent = "💾 Save";

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "cb-button";
    cancelBtn.style.cssText = "background: #333; color: #aaa; padding: 8px 20px;";
    cancelBtn.textContent = "Cancel";
    cancelBtn.onclick = closeModal;

    btnRow.appendChild(cancelBtn);
    btnRow.appendChild(saveBtn);
    modal.appendChild(btnRow);

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    // State
    let currentTriggers = [];

    function renderTriggerChips() {
        chipsContainer.innerHTML = "";
        if (currentTriggers.length === 0) {
            const empty = document.createElement("span");
            empty.style.cssText = "color: #666; font-size: 11px; font-style: italic;";
            empty.textContent = "No trigger words";
            chipsContainer.appendChild(empty);
            return;
        }
        currentTriggers.forEach((trigger, idx) => {
            const chip = document.createElement("span");
            chip.style.cssText = "background: #224422; border: 1px solid #448844; color: #aaffaa; padding: 3px 8px; border-radius: 12px; font-size: 11px; display: flex; align-items: center; gap: 4px;";
            chip.textContent = trigger;
            const removeBtn = document.createElement("span");
            removeBtn.textContent = "×";
            removeBtn.style.cssText = "cursor: pointer; color: #ff6666; font-weight: bold; margin-left: 2px;";
            removeBtn.onclick = () => {
                currentTriggers.splice(idx, 1);
                renderTriggerChips();
            };
            chip.appendChild(removeBtn);
            chipsContainer.appendChild(chip);
        });
    }

    function addTriggerWord() {
        const word = input.value.trim();
        if (word && !currentTriggers.includes(word)) {
            currentTriggers.push(word);
            renderTriggerChips();
            input.value = "";
        }
        input.focus();
    }

    addWordBtn.onclick = addTriggerWord;
    input.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); addTriggerWord(); } };

    saveBtn.onclick = async () => {
        saveBtn.disabled = true;
        saveBtn.textContent = "💾 Saving...";
        try {
            const resp = await fetch("/configbuilder/save_lora_triggers", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ lora_name: loraName, triggers: currentTriggers })
            });
            if (resp.ok) {
                status.textContent = "✅ Trigger words saved!";
                status.style.color = "#66ff66";
                setTimeout(closeModal, 800);
            } else {
                const err = await resp.json();
                status.textContent = "❌ Error: " + (err.error || "Save failed");
                status.style.color = "#ff6666";
                saveBtn.disabled = false;
                saveBtn.textContent = "💾 Save";
            }
        } catch (e) {
            status.textContent = "❌ Error: " + e.message;
            status.style.color = "#ff6666";
            saveBtn.disabled = false;
            saveBtn.textContent = "💾 Save";
        }
    };

    // Fetch current triggers
    try {
        const resp = await fetch(`/configbuilder/get_lora_triggers?lora_name=${encodeURIComponent(loraName)}`);
        if (resp.ok) {
            const data = await resp.json();
            currentTriggers = data.triggers || [];
            status.textContent = `Loaded ${currentTriggers.length} trigger word(s)`;
            status.style.color = "#88ff88";
        } else {
            status.textContent = "⚠️ No triggers found (you can add new ones)";
            status.style.color = "#ffaa00";
        }
    } catch (e) {
        status.textContent = "⚠️ Could not load triggers: " + e.message;
        status.style.color = "#ffaa00";
    }
    renderTriggerChips();
    input.focus();
}


// --- SECTION PRESET UI HELPER ---
// Creates a compact preset dropdown row (Load/Save/Delete) for any section
function createSectionPresetRow(sectionName, getData, applyData, node) {
    const row = document.createElement("div");
    row.style.cssText = "display: flex; gap: 3px; align-items: center; margin-left: auto;";

    const sel = document.createElement("select");
    sel.style.cssText = "background: #1a1a1a; color: #ccc; border: 1px solid #444; border-radius: 3px; padding: 2px 4px; font-size: 9px; max-width: 100px;";
    const defOpt = document.createElement("option");
    defOpt.value = ""; defOpt.textContent = "Presets";
    sel.appendChild(defOpt);
    sel._presets = [];

    // Fetch presets
    fetch(`/configbuilder/section_presets?section=${sectionName}`).then(r => r.json()).then(data => {
        (data.presets || []).forEach((p, i) => {
            const opt = document.createElement("option");
            opt.value = i; opt.textContent = p.name;
            sel.appendChild(opt);
        });
        sel._presets = data.presets || [];
    }).catch(() => {});

    const loadBtn = document.createElement("button");
    loadBtn.className = "cb-button";
    loadBtn.style.cssText = "font-size: 8px; padding: 1px 4px;";
    loadBtn.textContent = "Load";
    loadBtn.onclick = () => {
        const idx = parseInt(sel.value);
        if (isNaN(idx) || !sel._presets[idx]) return;
        applyData(JSON.parse(JSON.stringify(sel._presets[idx].data)));
        node.saveState();
        debouncedRenderUI(node);
    };

    const saveBtn = document.createElement("button");
    saveBtn.className = "cb-button";
    saveBtn.style.cssText = "font-size: 8px; padding: 1px 4px;";
    saveBtn.textContent = "Save";
    saveBtn.onclick = () => {
        const name = prompt(`Save ${sectionName} preset as:`);
        if (!name) return;
        const presets = sel._presets || [];
        const newPreset = { name, data: JSON.parse(JSON.stringify(getData())) };
        const existingIdx = presets.findIndex(p => p.name === name);
        if (existingIdx >= 0) presets[existingIdx] = newPreset;
        else presets.push(newPreset);
        fetch(`/configbuilder/section_presets?section=${sectionName}`, {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ presets })
        }).then(() => {
            sel._presets = presets;
            // Refresh dropdown
            while (sel.options.length > 1) sel.remove(1);
            presets.forEach((p, i) => {
                const opt = document.createElement("option");
                opt.value = i; opt.textContent = p.name;
                sel.appendChild(opt);
            });
        });
    };

    const delBtn = document.createElement("button");
    delBtn.className = "cb-button";
    delBtn.style.cssText = "font-size: 8px; padding: 1px 4px; color: #ff6666;";
    delBtn.textContent = "\u2715";
    delBtn.onclick = () => {
        const idx = parseInt(sel.value);
        if (isNaN(idx) || !sel._presets[idx]) return;
        if (!confirm(`Delete "${sel._presets[idx].name}"?`)) return;
        sel._presets.splice(idx, 1);
        fetch(`/configbuilder/section_presets?section=${sectionName}`, {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ presets: sel._presets })
        });
        sel.remove(idx + 1);
        sel.value = "";
    };

    row.appendChild(sel);
    row.appendChild(loadBtn);
    row.appendChild(saveBtn);
    row.appendChild(delBtn);
    return row;
}

// --- RENDER MODELS AND LORAS SECTIONS ---

export function renderModelsSection(node, div, configArray, arrayIdx, modelLists) {
    if (!configArray.models || configArray.models.length === 0) configArray.models = ["None"];

    const isSectionCollapsed = node.uiState.modelsSectionCollapsed[arrayIdx] || false;

    const modelGrid = document.createElement("div");
    modelGrid.className = "cb-list-grid";
    modelGrid.id = `cb-config-${arrayIdx}-models`;

    const modelHeader = document.createElement("div");
    modelHeader.className = "cb-section-toggle";
    modelHeader.style.cssText = "padding: 8px; background: #3a3a3a; border-radius: 4px; margin-bottom: 8px; font-weight: bold; color: #cc6600;";

    // Count total models (handle both string and object format)
    let totalModels = 0;
    configArray.models.forEach(m => {
        const mPath = typeof m === 'string' ? m : (m?.path || "None");
        const mType = typeof m === 'string' ? 'checkpoint' : (m?.type || 'checkpoint');
        let list;
        if (mType === 'gguf') list = modelLists.ggufModels;
        else if (mType === 'diffusion_model') list = modelLists.diffusionModels;
        else list = modelLists.checkpoints;

        if (mPath === "None") totalModels++;
        else if (mPath.endsWith("/")) {
            const norm = normalizePath(mPath);
            if (norm === "/") totalModels += list ? list.length : 1;
            else totalModels += list ? list.filter(am => normalizePath(am).startsWith(norm)).length : 1;
        } else totalModels++;
    });

    const titleSpan = document.createElement("span");
    titleSpan.textContent = `Models (${configArray.models.length} Entries, Totaling ${totalModels} Models)`;
    modelHeader.appendChild(titleSpan);

    // Models section presets
    const modelsPresetRow = createSectionPresetRow("models",
        () => ({ models: configArray.models, text_encoders: configArray.text_encoders || [], clip_type: configArray.clip_type || "stable_diffusion" }),
        (data) => {
            node.state.config_arrays[arrayIdx].models = data.models || ["None"];
            if (data.text_encoders) node.state.config_arrays[arrayIdx].text_encoders = data.text_encoders;
            if (data.clip_type) node.state.config_arrays[arrayIdx].clip_type = data.clip_type;
        },
        node
    );
    modelsPresetRow.onclick = (e) => e.stopPropagation(); // Don't toggle collapse on preset clicks
    modelHeader.appendChild(modelsPresetRow);

    const arrowSpan = document.createElement("span");
    arrowSpan.textContent = isSectionCollapsed ? "▶" : "▼";
    modelHeader.appendChild(arrowSpan);

    modelGrid.appendChild(modelHeader);

    const contentContainer = document.createElement("div");
    contentContainer.style.display = isSectionCollapsed ? "none" : "contents";

    modelHeader.onclick = () => {
        const isNowCollapsed = contentContainer.style.display === "none";
        if (isNowCollapsed) {
            contentContainer.style.display = "contents";
            arrowSpan.textContent = "▼";
            node.uiState.modelsSectionCollapsed[arrayIdx] = false;
        } else {
            contentContainer.style.display = "none";
            arrowSpan.textContent = "▶";
            node.uiState.modelsSectionCollapsed[arrayIdx] = true;
        }
    };

    configArray.models.forEach((model, modelIdx) => {
        contentContainer.appendChild(createModelElement(node, model, arrayIdx, modelIdx, modelLists));
    });

    const addRow = document.createElement("div");
    addRow.style.width = "100%";
    addRow.style.padding = "4px 0";
    const addBtn = document.createElement("button");
    addBtn.className = "cb-button";
    addBtn.style.cssText = "width: 100%; border: 1px dashed #555; background: rgba(0,0,0,0.2); color: #aaa;";
    addBtn.textContent = "➕ Add New Model";
    addBtn.onmouseover = () => addBtn.style.background = "rgba(255,255,255,0.1)";
    addBtn.onmouseout = () => addBtn.style.background = "rgba(0,0,0,0.2)";
    addBtn.onclick = () => {
        node.state.config_arrays[arrayIdx].models.push("None");
        node.saveState();
        debouncedRenderUI(node);
    };
    addRow.appendChild(addBtn);
    contentContainer.appendChild(addRow);

    // --- TEXT ENCODERS SECTION (shown when any model is non-checkpoint) ---
    const hasNonCheckpoint = configArray.models.some(m =>
        typeof m === 'object' && m !== null && m.type && m.type !== 'checkpoint'
    );

    if (hasNonCheckpoint) {
        renderTextEncodersSection(node, contentContainer, configArray, arrayIdx, modelLists);
    }

    // --- GGUF OPTIONS SECTION (shown when any model is GGUF) ---
    const hasGGUF = configArray.models.some(m =>
        typeof m === 'object' && m !== null && m.type === 'gguf'
    );

    if (hasGGUF) {
        renderGGUFOptionsSection(node, contentContainer, configArray, arrayIdx);
    }

    modelGrid.appendChild(contentContainer);
    div.appendChild(modelGrid);

    // Extra Model & Sampling Options (sub-section of Models)
    renderExtraModelSamplingSection(node, div, configArray, arrayIdx);
}

// --- TEXT ENCODERS SECTION ---
function renderTextEncodersSection(node, container, configArray, arrayIdx, modelLists) {
    const section = document.createElement("div");
    section.id = `cb-config-${arrayIdx}-te`;
    section.style.cssText = "width: 100%; border-top: 1px solid #444; margin-top: 8px; padding-top: 8px;";

    const header = document.createElement("div");
    header.className = "cb-section-toggle";
    header.style.cssText = "padding: 8px; background: #3a3a3a; border-radius: 4px; margin-bottom: 8px; font-weight: bold; color: #66bbff;";
    header.textContent = "Text Encoders (CLIP)";
    section.appendChild(header);

    // Clip Type selector
    const clipRow = document.createElement("div");
    clipRow.style.cssText = "display: flex; gap: 6px; align-items: center; margin-bottom: 6px;";
    const clipLabel = document.createElement("label");
    clipLabel.textContent = "CLIP Type:";
    clipLabel.style.cssText = "color: #aaa; font-size: 11px; white-space: nowrap;";
    clipRow.appendChild(clipLabel);

    const clipSelect = document.createElement("select");
    clipSelect.className = "cb-select";
    // Combine single and dual clip types
    const allClipTypes = [...new Set([...(modelLists.clipTypes || []), ...(modelLists.dualClipTypes || [])])];
    allClipTypes.forEach(ct => {
        const opt = document.createElement("option");
        opt.value = ct;
        opt.textContent = ct;
        if (ct === (configArray.clip_type || "stable_diffusion")) opt.selected = true;
        clipSelect.appendChild(opt);
    });
    clipSelect.onchange = () => {
        node.state.config_arrays[arrayIdx].clip_type = clipSelect.value;
        node.saveState();
    };
    clipRow.appendChild(clipSelect);
    section.appendChild(clipRow);

    // Initialize text encoder bypass states if they don't exist
    if (!node.state.config_arrays[arrayIdx].te_bypass_states) {
        node.state.config_arrays[arrayIdx].te_bypass_states = {};
    }

    // Text encoder entries
    const teList = configArray.text_encoders || [];
    teList.forEach((te, teIdx) => {
        const tePath = te || "None";

        // Get bypass state
        const isTeBypassed = node.state.config_arrays[arrayIdx].te_bypass_states[tePath] || false;

        const teRow = document.createElement("div");
        teRow.className = "cb-te-row";
        teRow.style.cssText = "display: flex; gap: 4px; align-items: center; margin-bottom: 4px;";

        // Drag handle for text encoder reorder
        const teDragHandle = document.createElement("span");
        teDragHandle.textContent = "⠿";
        teDragHandle.className = "cb-drag-handle";
        teDragHandle.draggable = true;
        teDragHandle.style.cssText = "color: #666; font-size: 14px; cursor: grab; user-select: none; padding: 0 4px;";
        teRow.appendChild(teDragHandle);
        // Reuse drag reorder with the handle as the draggable trigger and the row as the card
        setupDragReorder(teRow, teDragHandle, () => node.state.config_arrays[arrayIdx].text_encoders, teIdx, node);

        // Apply bypass visual state
        if (isTeBypassed) {
            teRow.style.opacity = "0.5";
            teRow.style.filter = "grayscale(0.7)";
        }

        // Bypass Checkbox - same pattern as model bypass
        if (tePath !== "None") {
            const teBypassLabel = document.createElement("label");
            teBypassLabel.style.cssText = "display: flex; align-items: center; gap: 4px; cursor: pointer; margin-right: 4px;";
            teBypassLabel.title = "Bypass (disable) this Text Encoder";

            const teBypassCheck = document.createElement("input");
            teBypassCheck.type = "checkbox";
            teBypassCheck.checked = !isTeBypassed; // Inverted: checked = enabled
            teBypassCheck.style.cssText = "cursor: pointer;";
            teBypassCheck.onclick = (e) => {
                e.stopPropagation();
                const newBypassState = !teBypassCheck.checked;
                node.state.config_arrays[arrayIdx].te_bypass_states[tePath] = newBypassState;
                node.saveState();
                updatePreview(node);
                // Visual feedback
                teRow.style.opacity = newBypassState ? "0.5" : "1.0";
                teRow.style.filter = newBypassState ? "grayscale(0.7)" : "none";
            };

            const teBypassText = document.createElement("span");
            teBypassText.textContent = "On";
            teBypassText.style.cssText = "font-size: 11px; color: #44aaff;";

            teBypassLabel.appendChild(teBypassCheck);
            teBypassLabel.appendChild(teBypassText);
            teRow.appendChild(teBypassLabel);
        }

        const teOptions = (modelLists.textEncoders && modelLists.textEncoders.length > 0)
            ? modelLists.textEncoders
            : ["None"];
        const teSearchable = createSearchableSelect(
            teOptions,
            te || "None",
            (value) => {
                node.state.config_arrays[arrayIdx].text_encoders[teIdx] = normalizePath(value);
                node.saveState();
            },
            "Search text encoders..."
        );
        teSearchable.style.flex = "1";
        teRow.appendChild(teSearchable);

        const delBtn = document.createElement("button");
        delBtn.className = "cb-button danger";
        delBtn.style.cssText = "padding: 2px 6px; font-size: 10px;";
        delBtn.textContent = "✖";
        delBtn.onclick = () => {
            node.state.config_arrays[arrayIdx].text_encoders.splice(teIdx, 1);
            node.saveState();
            debouncedRenderUI(node);
        };
        teRow.appendChild(delBtn);
        section.appendChild(teRow);
    });

    // Add text encoder button
    const addTeBtn = document.createElement("button");
    addTeBtn.className = "cb-button";
    addTeBtn.style.cssText = "width: 100%; border: 1px dashed #446; background: rgba(0,50,100,0.2); color: #88bbff; font-size: 11px;";
    addTeBtn.textContent = "+ Add Text Encoder";
    addTeBtn.onclick = () => {
        if (!node.state.config_arrays[arrayIdx].text_encoders) {
            node.state.config_arrays[arrayIdx].text_encoders = [];
        }
        node.state.config_arrays[arrayIdx].text_encoders.push("None");
        node.saveState();
        debouncedRenderUI(node);
    };
    section.appendChild(addTeBtn);

    container.appendChild(section);
}

// --- GGUF OPTIONS SECTION ---
function renderGGUFOptionsSection(node, container, configArray, arrayIdx) {
    const section = document.createElement("details");
    section.style.cssText = "width: 100%; border-top: 1px solid #444; margin-top: 8px; padding-top: 8px;";

    const summary = document.createElement("summary");
    summary.textContent = "GGUF Options";
    summary.style.cssText = "cursor: pointer; color: #aaa; font-size: 11px; font-weight: bold;";
    section.appendChild(summary);

    const opts = configArray.gguf_options || {};
    const grid = document.createElement("div");
    grid.style.cssText = "display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 6px;";

    // Helper for select rows
    const addSelectRow = (label, options, currentVal, onChange) => {
        const lbl = document.createElement("label");
        lbl.textContent = label;
        lbl.style.cssText = "color: #aaa; font-size: 11px; display: flex; align-items: center;";
        grid.appendChild(lbl);

        const sel = document.createElement("select");
        sel.className = "cb-select";
        options.forEach(o => {
            const opt = document.createElement("option");
            opt.value = o;
            opt.textContent = o;
            if (o === currentVal) opt.selected = true;
            sel.appendChild(opt);
        });
        sel.onchange = () => onChange(sel.value);
        grid.appendChild(sel);
    };

    const dtypeOptions = ["default", "target", "float32", "float16", "bfloat16"];

    addSelectRow("Dequant dtype:", dtypeOptions, opts.dequant_dtype || "default", (val) => {
        if (!node.state.config_arrays[arrayIdx].gguf_options) node.state.config_arrays[arrayIdx].gguf_options = {};
        node.state.config_arrays[arrayIdx].gguf_options.dequant_dtype = val;
        node.saveState();
    });

    addSelectRow("Patch dtype:", dtypeOptions, opts.patch_dtype || "default", (val) => {
        if (!node.state.config_arrays[arrayIdx].gguf_options) node.state.config_arrays[arrayIdx].gguf_options = {};
        node.state.config_arrays[arrayIdx].gguf_options.patch_dtype = val;
        node.saveState();
    });

    // Patch on device checkbox
    const podLabel = document.createElement("label");
    podLabel.textContent = "Patch on device:";
    podLabel.style.cssText = "color: #aaa; font-size: 11px; display: flex; align-items: center;";
    grid.appendChild(podLabel);

    const podCheck = document.createElement("input");
    podCheck.type = "checkbox";
    podCheck.checked = opts.patch_on_device || false;
    podCheck.onchange = () => {
        if (!node.state.config_arrays[arrayIdx].gguf_options) node.state.config_arrays[arrayIdx].gguf_options = {};
        node.state.config_arrays[arrayIdx].gguf_options.patch_on_device = podCheck.checked;
        node.saveState();
    };
    grid.appendChild(podCheck);

    section.appendChild(grid);
    container.appendChild(section);
}

// --- VAE SECTION RENDERER ---

export function renderVAEsSection(node, div, configArray, arrayIdx, modelLists) {
    if (!configArray.vaes || configArray.vaes.length === 0) return; // Don't show section if no VAEs added

    const vaeList = modelLists?.vaeModels || getAvailableVAEs();
    const vFolders = modelLists?.vaeFolders || getVAEFolders();

    const isSectionCollapsed = node.uiState.vaesSectionCollapsed?.[arrayIdx] || false;

    const vaeGrid = document.createElement("div");
    vaeGrid.className = "cb-list-grid";
    vaeGrid.id = `cb-config-${arrayIdx}-vae`;

    const vaeHeader = document.createElement("div");
    vaeHeader.className = "cb-section-toggle";
    vaeHeader.style.cssText = "padding: 8px; background: #444; border-radius: 4px; margin-bottom: 8px; font-weight: bold; color: #cc66ff;";

    const activeVaes = configArray.vaes.filter(v => v && v !== "None").length;
    const titleSpan = document.createElement("span");
    titleSpan.textContent = `VAEs (${configArray.vaes.length} Entries, ${activeVaes} Active)`;
    vaeHeader.appendChild(titleSpan);

    const arrowSpan = document.createElement("span");
    arrowSpan.textContent = isSectionCollapsed ? "▶" : "▼";
    vaeHeader.appendChild(arrowSpan);

    vaeGrid.appendChild(vaeHeader);

    const contentContainer = document.createElement("div");
    contentContainer.style.display = isSectionCollapsed ? "none" : "contents";

    vaeHeader.onclick = () => {
        const isNowCollapsed = contentContainer.style.display === "none";
        if (isNowCollapsed) {
            contentContainer.style.display = "contents";
            arrowSpan.textContent = "▼";
            if (!node.uiState.vaesSectionCollapsed) node.uiState.vaesSectionCollapsed = {};
            node.uiState.vaesSectionCollapsed[arrayIdx] = false;
        } else {
            contentContainer.style.display = "none";
            arrowSpan.textContent = "▶";
            if (!node.uiState.vaesSectionCollapsed) node.uiState.vaesSectionCollapsed = {};
            node.uiState.vaesSectionCollapsed[arrayIdx] = true;
        }
    };

    configArray.vaes.forEach((vae, vaeIdx) => {
        contentContainer.appendChild(createVAEElement(node, vae, arrayIdx, vaeIdx, vaeList, vFolders));
    });

    const addRow = document.createElement("div");
    addRow.style.width = "100%";
    addRow.style.padding = "4px 0";
    const addBtn = document.createElement("button");
    addBtn.className = "cb-button";
    addBtn.style.cssText = "width: 100%; border: 1px dashed #555; background: rgba(0,0,0,0.2); color: #aaa;";
    addBtn.textContent = "➕ Add New VAE";
    addBtn.onmouseover = () => addBtn.style.background = "rgba(255,255,255,0.1)";
    addBtn.onmouseout = () => addBtn.style.background = "rgba(0,0,0,0.2)";
    addBtn.onclick = () => {
        node.state.config_arrays[arrayIdx].vaes.push("None");
        node.saveState();
        debouncedRenderUI(node);
    };
    addRow.appendChild(addBtn);
    contentContainer.appendChild(addRow);

    vaeGrid.appendChild(contentContainer);
    div.appendChild(vaeGrid);
}

// --- EXTRA MODEL & SAMPLING OPTIONS SECTION ---

export function renderExtraModelSamplingSection(node, div, configArray, arrayIdx) {
    // Default collapsed per design
    const isSectionCollapsed = node.uiState.extraOptionsSectionCollapsed?.[arrayIdx] !== false;

    const sectionGrid = document.createElement("div");
    sectionGrid.className = "cb-list-grid";

    const sectionHeader = document.createElement("div");
    sectionHeader.className = "cb-section-toggle";
    sectionHeader.style.cssText = "padding: 8px; background: #3a3a3a; border-radius: 4px; margin-bottom: 8px; font-weight: bold; color: #ffaa00;";

    const titleSpan = document.createElement("span");
    titleSpan.textContent = "Extra Model & Sampling Options";
    sectionHeader.appendChild(titleSpan);

    const arrowSpan = document.createElement("span");
    arrowSpan.textContent = isSectionCollapsed ? "▶" : "▼";
    sectionHeader.appendChild(arrowSpan);

    sectionGrid.appendChild(sectionHeader);

    const contentContainer = document.createElement("div");
    contentContainer.style.display = isSectionCollapsed ? "none" : "contents";

    sectionHeader.onclick = () => {
        const isNowCollapsed = contentContainer.style.display === "none";
        if (isNowCollapsed) {
            contentContainer.style.display = "contents";
            arrowSpan.textContent = "▼";
            if (!node.uiState.extraOptionsSectionCollapsed) node.uiState.extraOptionsSectionCollapsed = {};
            node.uiState.extraOptionsSectionCollapsed[arrayIdx] = false;
        } else {
            contentContainer.style.display = "none";
            arrowSpan.textContent = "▶";
            if (!node.uiState.extraOptionsSectionCollapsed) node.uiState.extraOptionsSectionCollapsed = {};
            node.uiState.extraOptionsSectionCollapsed[arrayIdx] = true;
        }
    };

    const innerWrapper = document.createElement("div");
    innerWrapper.style.cssText = "padding: 4px 8px; display: flex; flex-direction: column; gap: 12px;";

    // ========== SUB-GROUP 1: Model Sampling Override ==========
    const group1 = document.createElement("div");
    group1.style.cssText = "border-left: 3px solid #ffaa00; padding-left: 8px;";

    const group1Header = document.createElement("div");
    group1Header.style.cssText = "font-size: 11px; font-weight: bold; margin-bottom: 4px; color: #ffaa00;";
    group1Header.textContent = "MODEL SAMPLING OVERRIDE";
    group1.appendChild(group1Header);

    const group1Info = document.createElement("div");
    group1Info.style.cssText = "font-size: 10px; color: #888; font-style: italic; margin-bottom: 6px;";
    group1Info.textContent = "Patches the model's internal noise schedule for specific model families";
    group1.appendChild(group1Info);

    // Dropdown: None / AuraFlow / Flux / SD3
    const overrideRow = document.createElement("div");
    overrideRow.style.cssText = "display: flex; gap: 6px; align-items: center; margin-bottom: 6px;";
    const overrideLabel = document.createElement("label");
    overrideLabel.textContent = "Override:";
    overrideLabel.style.cssText = "color: #aaa; font-size: 11px; white-space: nowrap; min-width: 55px;";
    overrideRow.appendChild(overrideLabel);

    const overrideSelect = document.createElement("select");
    overrideSelect.className = "cb-select";
    overrideSelect.style.cssText = "flex: 1; max-width: 200px;";
    [["none", "None"], ["aura_flow", "AuraFlow (Qwen Image)"], ["flux", "Flux"], ["flux2", "Flux2"], ["sd3", "SD3"]].forEach(([val, label]) => {
        const opt = document.createElement("option");
        opt.value = val;
        opt.textContent = label;
        if (configArray.model_sampling_override === val) opt.selected = true;
        overrideSelect.appendChild(opt);
    });
    overrideRow.appendChild(overrideSelect);
    group1.appendChild(overrideRow);

    // Conditional params container
    const paramsContainer = document.createElement("div");
    paramsContainer.style.cssText = "margin-top: 4px;";

    function updateParamsVisibility() {
        paramsContainer.innerHTML = "";
        const override = configArray.model_sampling_override;

        if (override === "aura_flow" || override === "sd3") {
            // Single shift param
            const row = document.createElement("div");
            row.style.cssText = "display: flex; gap: 6px; align-items: center; margin-bottom: 4px;";
            const lbl = document.createElement("label");
            lbl.textContent = "Shift:";
            lbl.style.cssText = "color: #aaa; font-size: 11px; white-space: nowrap; min-width: 55px;";
            row.appendChild(lbl);
            const inp = document.createElement("input");
            inp.type = "text";
            inp.className = "cb-input";
            inp.style.cssText = "width: 120px;";
            inp.value = configArray.model_sampling_shift || (override === "sd3" ? "3.0" : "1.73");
            inp.placeholder = override === "sd3" ? "3.0" : "1.73";
            inp.title = "Comma-separated values for grid testing";
            inp.onchange = () => {
                configArray.model_sampling_shift = inp.value;
                node.saveState();
            };
            row.appendChild(inp);
            const hint = document.createElement("span");
            hint.style.cssText = "font-size: 10px; color: #666;";
            hint.textContent = override === "sd3" ? "(default: 3.0, multiplier: 1000)" : "(default: 1.73, multiplier: 1.0)";
            row.appendChild(hint);
            paramsContainer.appendChild(row);
        } else if (override === "flux") {
            if (!isLTXConfigArray(configArray)) {
                // Max shift
                const row1 = document.createElement("div");
                row1.style.cssText = "display: flex; gap: 6px; align-items: center; margin-bottom: 4px;";
                const lbl1 = document.createElement("label");
                lbl1.textContent = "Max Shift:";
                lbl1.style.cssText = "color: #aaa; font-size: 11px; white-space: nowrap; min-width: 55px;";
                row1.appendChild(lbl1);
                const inp1 = document.createElement("input");
                inp1.type = "text";
                inp1.className = "cb-input";
                inp1.style.cssText = "width: 120px;";
                inp1.value = configArray.model_sampling_flux_max_shift || "1.15";
                inp1.placeholder = "1.15";
                inp1.title = "Comma-separated values for grid testing";
                inp1.onchange = () => {
                    configArray.model_sampling_flux_max_shift = inp1.value;
                    node.saveState();
                };
                row1.appendChild(inp1);
                paramsContainer.appendChild(row1);

                // Base shift
                const row2 = document.createElement("div");
                row2.style.cssText = "display: flex; gap: 6px; align-items: center; margin-bottom: 4px;";
                const lbl2 = document.createElement("label");
                lbl2.textContent = "Base Shift:";
                lbl2.style.cssText = "color: #aaa; font-size: 11px; white-space: nowrap; min-width: 55px;";
                row2.appendChild(lbl2);
                const inp2 = document.createElement("input");
                inp2.type = "text";
                inp2.className = "cb-input";
                inp2.style.cssText = "width: 120px;";
                inp2.value = configArray.model_sampling_flux_base_shift || "0.5";
                inp2.placeholder = "0.5";
                inp2.title = "Comma-separated values for grid testing";
                inp2.onchange = () => {
                    configArray.model_sampling_flux_base_shift = inp2.value;
                    node.saveState();
                };
                row2.appendChild(inp2);
                paramsContainer.appendChild(row2);

                const hint = document.createElement("div");
                hint.style.cssText = "font-size: 10px; color: #666;";
                hint.textContent = "Dynamic shift computed from image dimensions";
                paramsContainer.appendChild(hint);
            }
        }
    }

    overrideSelect.onchange = () => {
        configArray.model_sampling_override = overrideSelect.value;
        updateParamsVisibility();
        node.saveState();
    };
    updateParamsVisibility();
    group1.appendChild(paramsContainer);
    innerWrapper.appendChild(group1);

    // ========== SUB-GROUP 2: Advanced Sampling Pipeline ==========
    const group2 = document.createElement("div");
    group2.style.cssText = "border-left: 3px solid #ffaa00; padding-left: 8px;";

    const group2Header = document.createElement("div");
    group2Header.style.cssText = "font-size: 11px; font-weight: bold; margin-bottom: 4px; color: #ffaa00;";
    group2Header.textContent = "ADVANCED SAMPLING PIPELINE";
    group2.appendChild(group2Header);

    const group2Info = document.createElement("div");
    group2Info.style.cssText = "font-size: 10px; color: #888; font-style: italic; margin-bottom: 6px;";
    group2Info.textContent = "Replaces KSampler with explicit SamplerCustomAdvanced flow (Noise + Guider + Sampler + Sigmas)";
    group2.appendChild(group2Info);

    // Toggle checkbox
    const toggleRow = document.createElement("div");
    toggleRow.style.cssText = "display: flex; gap: 6px; align-items: center; margin-bottom: 6px;";
    const toggleLabel = document.createElement("label");
    toggleLabel.style.cssText = "display: flex; align-items: center; gap: 6px; cursor: pointer;";
    const toggleCheck = document.createElement("input");
    toggleCheck.type = "checkbox";
    toggleCheck.checked = configArray.use_advanced_sampling || false;
    toggleCheck.style.cssText = "cursor: pointer;";
    toggleLabel.appendChild(toggleCheck);
    const toggleText = document.createElement("span");
    toggleText.textContent = "Enable Advanced Sampling";
    toggleText.style.cssText = "color: #ccc; font-size: 11px;";
    toggleLabel.appendChild(toggleText);
    toggleRow.appendChild(toggleLabel);
    group2.appendChild(toggleRow);

    // Sub-options container (shown when enabled)
    const advancedOpts = document.createElement("div");
    advancedOpts.style.cssText = "margin-left: 12px;";
    advancedOpts.style.display = configArray.use_advanced_sampling ? "block" : "none";

    // Guider dropdown
    const guiderRow = document.createElement("div");
    guiderRow.style.cssText = "display: flex; gap: 6px; align-items: center; margin-bottom: 4px;";
    const guiderLabel = document.createElement("label");
    guiderLabel.textContent = "Guider:";
    guiderLabel.style.cssText = "color: #aaa; font-size: 11px; white-space: nowrap; min-width: 55px;";
    guiderRow.appendChild(guiderLabel);
    const guiderSelect = document.createElement("select");
    guiderSelect.className = "cb-select";
    guiderSelect.style.cssText = "flex: 1; max-width: 200px;";
    [["cfg_guider", "CFG Guider"], ["basic_guider", "Basic Guider (no CFG)"]].forEach(([val, label]) => {
        const opt = document.createElement("option");
        opt.value = val;
        opt.textContent = label;
        if (configArray.advanced_guider === val) opt.selected = true;
        guiderSelect.appendChild(opt);
    });
    guiderSelect.onchange = () => {
        configArray.advanced_guider = guiderSelect.value;
        node.saveState();
    };
    guiderRow.appendChild(guiderSelect);
    advancedOpts.appendChild(guiderRow);

    // Scheduler dropdown
    const schedulerRow = document.createElement("div");
    schedulerRow.style.cssText = "display: flex; gap: 6px; align-items: center; margin-bottom: 4px;";
    const schedulerLabel = document.createElement("label");
    schedulerLabel.textContent = "Scheduler:";
    schedulerLabel.style.cssText = "color: #aaa; font-size: 11px; white-space: nowrap; min-width: 55px;";
    schedulerRow.appendChild(schedulerLabel);
    const schedulerSelect = document.createElement("select");
    schedulerSelect.className = "cb-select";
    schedulerSelect.style.cssText = "flex: 1; max-width: 200px;";
    [["basic", "Basic Scheduler"], ["flux2", "Flux2 Scheduler"]].forEach(([val, label]) => {
        const opt = document.createElement("option");
        opt.value = val;
        opt.textContent = label;
        if (configArray.advanced_scheduler === val) opt.selected = true;
        schedulerSelect.appendChild(opt);
    });
    schedulerSelect.onchange = () => {
        configArray.advanced_scheduler = schedulerSelect.value;
        node.saveState();
    };
    schedulerRow.appendChild(schedulerSelect);
    advancedOpts.appendChild(schedulerRow);

    const advancedHint = document.createElement("div");
    advancedHint.style.cssText = "font-size: 10px; color: #666;";
    advancedHint.textContent = "When ON, creates ON/OFF grid variants. Uses existing seed, sampler, cfg, and steps from config.";
    advancedOpts.appendChild(advancedHint);

    toggleCheck.onchange = () => {
        configArray.use_advanced_sampling = toggleCheck.checked;
        advancedOpts.style.display = toggleCheck.checked ? "block" : "none";
        node.saveState();
    };

    group2.appendChild(advancedOpts);
    innerWrapper.appendChild(group2);

    // ========== SUB-GROUP 3: Flux Guidance ==========
    const group3 = document.createElement("div");
    group3.style.cssText = "border-left: 3px solid #ffaa00; padding-left: 8px;";

    const group3Header = document.createElement("div");
    group3Header.style.cssText = "font-size: 11px; font-weight: bold; margin-bottom: 4px; color: #ffaa00;";
    group3Header.textContent = "FLUX GUIDANCE";
    group3.appendChild(group3Header);

    const group3Info = document.createElement("div");
    group3Info.style.cssText = "font-size: 10px; color: #888; font-style: italic; margin-bottom: 6px;";
    group3Info.textContent = "Modifies positive conditioning with a guidance value (used by Flux models)";
    group3.appendChild(group3Info);

    // Toggle checkbox
    const fluxToggleRow = document.createElement("div");
    fluxToggleRow.style.cssText = "display: flex; gap: 6px; align-items: center; margin-bottom: 6px;";
    const fluxToggleLabel = document.createElement("label");
    fluxToggleLabel.style.cssText = "display: flex; align-items: center; gap: 6px; cursor: pointer;";
    const fluxToggleCheck = document.createElement("input");
    fluxToggleCheck.type = "checkbox";
    fluxToggleCheck.checked = configArray.use_flux_guidance || false;
    fluxToggleCheck.style.cssText = "cursor: pointer;";
    fluxToggleLabel.appendChild(fluxToggleCheck);
    const fluxToggleText = document.createElement("span");
    fluxToggleText.textContent = "Enable Flux Guidance";
    fluxToggleText.style.cssText = "color: #ccc; font-size: 11px;";
    fluxToggleLabel.appendChild(fluxToggleText);
    fluxToggleRow.appendChild(fluxToggleLabel);
    group3.appendChild(fluxToggleRow);

    // Guidance value input (shown when enabled)
    const fluxOpts = document.createElement("div");
    fluxOpts.style.cssText = "margin-left: 12px;";
    fluxOpts.style.display = configArray.use_flux_guidance ? "block" : "none";

    const guidanceRow = document.createElement("div");
    guidanceRow.style.cssText = "display: flex; gap: 6px; align-items: center; margin-bottom: 4px;";
    const guidanceLabel = document.createElement("label");
    guidanceLabel.textContent = "Guidance:";
    guidanceLabel.style.cssText = "color: #aaa; font-size: 11px; white-space: nowrap; min-width: 55px;";
    guidanceRow.appendChild(guidanceLabel);
    const guidanceInput = document.createElement("input");
    guidanceInput.type = "text";
    guidanceInput.className = "cb-input";
    guidanceInput.style.cssText = "width: 120px;";
    guidanceInput.value = configArray.flux_guidance_value || "3.5";
    guidanceInput.placeholder = "3.5";
    guidanceInput.title = "Comma-separated values for grid testing (range 0-100)";
    guidanceInput.onchange = () => {
        configArray.flux_guidance_value = guidanceInput.value;
        node.saveState();
    };
    guidanceRow.appendChild(guidanceInput);
    const guidanceHint = document.createElement("span");
    guidanceHint.style.cssText = "font-size: 10px; color: #666;";
    guidanceHint.textContent = "(default: 3.5, range: 0-100)";
    guidanceRow.appendChild(guidanceHint);
    fluxOpts.appendChild(guidanceRow);

    fluxToggleCheck.onchange = () => {
        configArray.use_flux_guidance = fluxToggleCheck.checked;
        fluxOpts.style.display = fluxToggleCheck.checked ? "block" : "none";
        node.saveState();
    };

    group3.appendChild(fluxOpts);
    if (!isLTXConfigArray(configArray)) {
        innerWrapper.appendChild(group3);
    }

    // ========== SUB-GROUP 4: Kohya Deep Shrink (PatchModelAddDownscale) ==========
    const group4 = document.createElement("div");
    group4.style.cssText = "border-left: 3px solid #ffaa00; padding-left: 8px;";

    const group4Header = document.createElement("div");
    group4Header.style.cssText = "font-size: 11px; font-weight: bold; margin-bottom: 4px; color: #ffaa00;";
    group4Header.textContent = "KOHYA DEEP SHRINK (PatchModelAddDownscale)";
    group4.appendChild(group4Header);

    const group4Info = document.createElement("div");
    group4Info.style.cssText = "font-size: 10px; color: #888; font-style: italic; margin-bottom: 6px;";
    group4Info.textContent = "Patches the UNet to downscale features at a specific block during early diffusion. Lets you generate at higher target resolutions than the model was trained for.";
    group4.appendChild(group4Info);

    // Toggle checkbox
    const dsToggleRow = document.createElement("div");
    dsToggleRow.style.cssText = "display: flex; gap: 6px; align-items: center; margin-bottom: 6px;";
    const dsToggleLabel = document.createElement("label");
    dsToggleLabel.style.cssText = "display: flex; align-items: center; gap: 6px; cursor: pointer;";
    const dsToggleCheck = document.createElement("input");
    dsToggleCheck.type = "checkbox";
    dsToggleCheck.checked = configArray.use_deep_shrink || false;
    dsToggleCheck.style.cssText = "cursor: pointer;";
    dsToggleLabel.appendChild(dsToggleCheck);
    const dsToggleText = document.createElement("span");
    dsToggleText.textContent = "Enable Deep Shrink";
    dsToggleText.style.cssText = "color: #ccc; font-size: 11px;";
    dsToggleLabel.appendChild(dsToggleText);
    dsToggleRow.appendChild(dsToggleLabel);
    group4.appendChild(dsToggleRow);

    // Sub-options container (shown when enabled)
    const dsOpts = document.createElement("div");
    dsOpts.style.cssText = "margin-left: 12px;";
    dsOpts.style.display = configArray.use_deep_shrink ? "block" : "none";

    // Helper: render one labeled input row
    function _dsRow(labelText, inputEl, hintText) {
        const row = document.createElement("div");
        row.style.cssText = "display: flex; gap: 6px; align-items: center; margin-bottom: 4px;";
        const lbl = document.createElement("label");
        lbl.textContent = labelText;
        lbl.style.cssText = "color: #aaa; font-size: 11px; white-space: nowrap; min-width: 130px;";
        row.appendChild(lbl);
        row.appendChild(inputEl);
        if (hintText) {
            const hint = document.createElement("span");
            hint.style.cssText = "font-size: 10px; color: #666;";
            hint.textContent = hintText;
            row.appendChild(hint);
        }
        return row;
    }

    // block_number
    const blockInput = document.createElement("input");
    blockInput.type = "number"; blockInput.className = "cb-input";
    blockInput.style.cssText = "width: 80px;"; blockInput.min = "1"; blockInput.max = "32"; blockInput.step = "1";
    blockInput.value = configArray.deep_shrink_block_number ?? 3;
    blockInput.onchange = () => { configArray.deep_shrink_block_number = parseInt(blockInput.value) || 3; node.saveState(); };
    dsOpts.appendChild(_dsRow("Block Number:", blockInput, "(default: 3)"));

    // downscale_factor
    const factorInput = document.createElement("input");
    factorInput.type = "number"; factorInput.className = "cb-input";
    factorInput.style.cssText = "width: 80px;"; factorInput.min = "0.1"; factorInput.max = "9.0"; factorInput.step = "0.1";
    factorInput.value = configArray.deep_shrink_downscale_factor ?? 2.0;
    factorInput.onchange = () => { configArray.deep_shrink_downscale_factor = parseFloat(factorInput.value) || 2.0; node.saveState(); };
    dsOpts.appendChild(_dsRow("Downscale Factor:", factorInput, "(default: 2.0)"));

    // start_percent
    const startInput = document.createElement("input");
    startInput.type = "number"; startInput.className = "cb-input";
    startInput.style.cssText = "width: 80px;"; startInput.min = "0.0"; startInput.max = "1.0"; startInput.step = "0.05";
    startInput.value = configArray.deep_shrink_start_percent ?? 0.0;
    startInput.onchange = () => { configArray.deep_shrink_start_percent = parseFloat(startInput.value); if (isNaN(configArray.deep_shrink_start_percent)) configArray.deep_shrink_start_percent = 0.0; node.saveState(); };
    dsOpts.appendChild(_dsRow("Start %:", startInput, "(default: 0.0 = beginning)"));

    // end_percent
    const endInput = document.createElement("input");
    endInput.type = "number"; endInput.className = "cb-input";
    endInput.style.cssText = "width: 80px;"; endInput.min = "0.0"; endInput.max = "1.0"; endInput.step = "0.05";
    endInput.value = configArray.deep_shrink_end_percent ?? 0.35;
    endInput.onchange = () => { configArray.deep_shrink_end_percent = parseFloat(endInput.value); if (isNaN(configArray.deep_shrink_end_percent)) configArray.deep_shrink_end_percent = 0.35; node.saveState(); };
    dsOpts.appendChild(_dsRow("End %:", endInput, "(default: 0.35)"));

    // downscale_after_skip
    const afterSkipRow = document.createElement("div");
    afterSkipRow.style.cssText = "display: flex; gap: 6px; align-items: center; margin-bottom: 4px;";
    const afterSkipLbl = document.createElement("label");
    afterSkipLbl.style.cssText = "display: flex; align-items: center; gap: 6px; cursor: pointer;";
    const afterSkipCheck = document.createElement("input");
    afterSkipCheck.type = "checkbox";
    afterSkipCheck.checked = configArray.deep_shrink_downscale_after_skip !== false; // default true
    afterSkipCheck.onchange = () => { configArray.deep_shrink_downscale_after_skip = afterSkipCheck.checked; node.saveState(); };
    afterSkipLbl.appendChild(afterSkipCheck);
    const afterSkipText = document.createElement("span");
    afterSkipText.textContent = "Downscale After Skip";
    afterSkipText.style.cssText = "color: #ccc; font-size: 11px;";
    afterSkipLbl.appendChild(afterSkipText);
    afterSkipRow.appendChild(afterSkipLbl);
    const afterSkipHint = document.createElement("span");
    afterSkipHint.style.cssText = "font-size: 10px; color: #666;";
    afterSkipHint.textContent = "(default: ON)";
    afterSkipRow.appendChild(afterSkipHint);
    dsOpts.appendChild(afterSkipRow);

    // downscale_method
    const downMethods = ["bicubic", "nearest-exact", "bilinear", "area", "bislerp"];
    const downSelect = document.createElement("select");
    downSelect.className = "cb-select";
    downSelect.style.cssText = "width: 140px;";
    downMethods.forEach(m => {
        const opt = document.createElement("option"); opt.value = m; opt.textContent = m;
        if ((configArray.deep_shrink_downscale_method || "bicubic") === m) opt.selected = true;
        downSelect.appendChild(opt);
    });
    downSelect.onchange = () => { configArray.deep_shrink_downscale_method = downSelect.value; node.saveState(); };
    dsOpts.appendChild(_dsRow("Downscale Method:", downSelect, ""));

    // upscale_method
    const upSelect = document.createElement("select");
    upSelect.className = "cb-select";
    upSelect.style.cssText = "width: 140px;";
    downMethods.forEach(m => {
        const opt = document.createElement("option"); opt.value = m; opt.textContent = m;
        if ((configArray.deep_shrink_upscale_method || "bicubic") === m) opt.selected = true;
        upSelect.appendChild(opt);
    });
    upSelect.onchange = () => { configArray.deep_shrink_upscale_method = upSelect.value; node.saveState(); };
    dsOpts.appendChild(_dsRow("Upscale Method:", upSelect, ""));

    dsToggleCheck.onchange = () => {
        configArray.use_deep_shrink = dsToggleCheck.checked;
        dsOpts.style.display = dsToggleCheck.checked ? "block" : "none";
        node.saveState();
    };

    group4.appendChild(dsOpts);
    if (!isLTXConfigArray(configArray)) {
        innerWrapper.appendChild(group4);
    }

    contentContainer.appendChild(innerWrapper);
    sectionGrid.appendChild(contentContainer);
    div.appendChild(sectionGrid);
}

// --- VAE ELEMENT CREATOR HELPERS ---

function _renderRemoteVaeInstallCard(node) {
    const card = document.createElement('div');
    card.className = 'cb-remote-vae-install-card';
    card.style.cssText = [
        "border: 1px solid #b07530",
        "background: rgba(176, 117, 48, 0.08)",
        "border-radius: 4px",
        "padding: 8px 10px",
        "margin: 4px 0",
        "font-size: 11px",
        "color: #ddd",
        "line-height: 1.4",
    ].join(';');

    const header = document.createElement('div');
    header.style.fontWeight = "600";
    header.style.marginBottom = "6px";
    header.style.color = "#f0a050";
    header.textContent = "⚠ Remote VAE requires the companion plugin";
    card.appendChild(header);

    const intro = document.createElement('div');
    intro.textContent = "Install via:";
    card.appendChild(intro);

    const list = document.createElement('ul');
    list.style.margin = "4px 0 4px 16px";
    list.style.padding = "0";

    const liManager = document.createElement('li');
    liManager.appendChild(document.createTextNode("Comfy Manager — search "));
    const bold = document.createElement('b');
    bold.textContent = "USCG Remote VAE";
    liManager.appendChild(bold);
    list.appendChild(liManager);

    const liManual = document.createElement('li');
    liManual.appendChild(document.createTextNode("Manual: "));
    const codeClone = document.createElement('code');
    codeClone.style.fontSize = "10px";
    codeClone.textContent = "git clone";
    liManual.appendChild(codeClone);
    liManual.appendChild(document.createTextNode(" the repo into "));
    const codeDir = document.createElement('code');
    codeDir.style.fontSize = "10px";
    codeDir.textContent = "custom_nodes/";
    liManual.appendChild(codeDir);
    list.appendChild(liManual);

    card.appendChild(list);

    const restart = document.createElement('div');
    restart.style.marginTop = "6px";
    restart.textContent = "Then restart ComfyUI.";
    card.appendChild(restart);

    const buttonRow = document.createElement('div');
    buttonRow.style.marginTop = "8px";
    buttonRow.style.display = "flex";
    buttonRow.style.gap = "8px";

    const ghLink = document.createElement('a');
    ghLink.href = "https://github.com/JasonHoku/ComfyUI-USCG-RemoteVAE";
    ghLink.target = "_blank";
    ghLink.rel = "noopener";
    ghLink.style.cssText = "background:#444;color:#ddd;padding:4px 8px;border-radius:3px;text-decoration:none;font-size:11px;";
    ghLink.textContent = "Open on GitHub ↗";
    buttonRow.appendChild(ghLink);

    const recheckBtn = document.createElement('button');
    recheckBtn.type = "button";
    recheckBtn.className = "cb-button";
    recheckBtn.style.fontSize = "11px";
    recheckBtn.style.padding = "4px 8px";
    recheckBtn.textContent = "Re-check after install";
    recheckBtn.addEventListener('click', () => {
        _resetRemoteVaeCaches();
        debouncedRenderUI(node);
    });
    buttonRow.appendChild(recheckBtn);

    card.appendChild(buttonRow);

    return card;
}

function _renderRemoteVaeControls(node, arrayIdx, vaeIdx, vaeName, endpoints) {
    // endpoints: list of {name, url} from companion plugin's /uscg-remote-vae/endpoints
    const wrap = document.createElement('div');

    const presetSelect = document.createElement('select');
    presetSelect.className = 'cb-select';
    presetSelect.style.marginBottom = '4px';

    const currentUrl = vaeName.replace(/^remote:/, '');

    // "Custom URL" sentinel option first.
    const customOpt = document.createElement('option');
    customOpt.value = '';
    customOpt.textContent = 'Custom URL';
    if (!endpoints.some(e => e.url === currentUrl)) customOpt.selected = true;
    presetSelect.appendChild(customOpt);

    endpoints.forEach(({ name, url }) => {
        const opt = document.createElement('option');
        opt.value = url;
        opt.textContent = name;
        if (currentUrl === url) opt.selected = true;
        presetSelect.appendChild(opt);
    });

    const urlInput = document.createElement('input');
    urlInput.className = 'cb-input';
    urlInput.type = 'text';
    urlInput.placeholder = 'https://your-endpoint.huggingface.cloud/';
    urlInput.value = currentUrl;
    urlInput.style.fontFamily = 'monospace';
    urlInput.style.fontSize = '12px';

    urlInput.onchange = () => {
        const url = urlInput.value.trim();
        node.state.config_arrays[arrayIdx].vaes[vaeIdx] = url ? `remote:${url}` : 'remote:';
        node.saveState();
        const match = endpoints.find(e => e.url === url);
        presetSelect.value = match ? match.url : '';
    };
    urlInput.onblur = urlInput.onchange;

    presetSelect.onchange = () => {
        const selectedUrl = presetSelect.value;
        if (selectedUrl) {
            urlInput.value = selectedUrl;
            node.state.config_arrays[arrayIdx].vaes[vaeIdx] = `remote:${selectedUrl}`;
            node.saveState();
        } else {
            urlInput.value = '';
            urlInput.focus();
        }
    };

    wrap.appendChild(presetSelect);
    wrap.appendChild(urlInput);

    const helper = document.createElement('div');
    helper.style.cssText = 'font-size: 9px; color: #666; padding: 2px 4px;';
    helper.textContent = 'Select a preset or enter an allowlisted endpoint URL';
    wrap.appendChild(helper);

    return wrap;
}

// --- VAE ELEMENT CREATOR ---

function createVAEElement(node, vaeName, arrayIdx, vaeIdx, vaeList, vFolders) {
    const isFolder = vaeName && vaeName.endsWith("/");
    const isRemote = vaeName && vaeName.startsWith("remote:");

    const div = document.createElement("div");
    div.className = "cb-item-card";
    div.style.borderLeft = isRemote ? "3px solid #00b894" : "3px solid #9900cc";
    const uid = `vae_${arrayIdx}_${vaeIdx}`;

    const isCollapsed = node.uiState.vaesCollapsed?.[uid] || false;

    // Initialize VAE bypass states if they don't exist
    if (!node.state.config_arrays[arrayIdx].vae_bypass_states) {
        node.state.config_arrays[arrayIdx].vae_bypass_states = {};
    }

    // Get bypass state
    const isBypassed = node.state.config_arrays[arrayIdx].vae_bypass_states[vaeName] || false;

    // Header
    const header = document.createElement("div");
    header.className = "cb-header-bar";

    const leftGroup = document.createElement("div");
    leftGroup.className = "cb-header-left";

    // Bypass Checkbox (in header, before toggle arrow) - same pattern as model bypass
    if (vaeName && vaeName !== "None") {
        const bypassLabel = document.createElement("label");
        bypassLabel.style.cssText = "display: flex; align-items: center; gap: 4px; cursor: pointer; margin-right: 8px;";
        bypassLabel.title = "Bypass (disable) this VAE";

        const bypassCheck = document.createElement("input");
        bypassCheck.type = "checkbox";
        bypassCheck.checked = !isBypassed; // Inverted: checked = enabled
        bypassCheck.style.cssText = "cursor: pointer;";
        bypassCheck.onclick = (e) => {
            e.stopPropagation(); // Prevent header collapse
            const newBypassState = !bypassCheck.checked;
            node.state.config_arrays[arrayIdx].vae_bypass_states[vaeName] = newBypassState;
            node.saveState();
            updatePreview(node);
            // Visual feedback
            div.style.opacity = newBypassState ? "0.5" : "1.0";
            div.style.filter = newBypassState ? "grayscale(0.7)" : "none";
        };

        bypassLabel.appendChild(bypassCheck);
        const bypassText = document.createElement("span");
        bypassText.textContent = "On";
        bypassText.style.cssText = "font-size: 11px; color: #9900cc;";
        bypassLabel.appendChild(bypassText);
        leftGroup.appendChild(bypassLabel);
    }

    const toggleArrow = document.createElement("span");
    toggleArrow.textContent = isCollapsed ? "▶" : "▼";
    toggleArrow.style.color = "#aaa";
    toggleArrow.style.fontSize = "10px";
    toggleArrow.style.width = "12px";
    leftGroup.appendChild(toggleArrow);

    const label = document.createElement("span");
    label.textContent = `VAE #${vaeIdx + 1}`;
    label.style.color = isRemote ? "#00b894" : "#9900cc";
    label.style.fontSize = "10px";
    label.style.marginRight = "6px";
    leftGroup.appendChild(label);

    const nameSpan = document.createElement("span");
    nameSpan.className = "cb-header-name";
    // Remote URLs would be mangled by getShortName() which splits on "/",
    // so display them directly instead
    if (isRemote) {
        const remoteUrl = vaeName.replace(/^remote:/, "");
        nameSpan.textContent = remoteUrl || "Remote URL (empty)";
    } else {
        nameSpan.textContent = getShortName(vaeName || "None");
    }
    leftGroup.appendChild(nameSpan);

    header.appendChild(leftGroup);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "cb-button danger";
    deleteBtn.style.padding = "2px 6px";
    deleteBtn.style.fontSize = "10px";
    deleteBtn.textContent = "✖";
    deleteBtn.onclick = () => {
        node.state.config_arrays[arrayIdx].vaes.splice(vaeIdx, 1);
        if (node.state.config_arrays[arrayIdx].vaes.length === 0) {
            node.state.config_arrays[arrayIdx].vaes = ["None"];
        }
        node.saveState();
        debouncedRenderUI(node);
    };
    header.appendChild(deleteBtn);
    div.appendChild(header);

    // Drag-and-drop reorder
    setupDragReorder(div, header, () => node.state.config_arrays[arrayIdx].vaes, vaeIdx, node);

    // Apply bypass visual state
    if (isBypassed) {
        div.style.opacity = "0.5";
        div.style.filter = "grayscale(0.7)";
    }

    // Content container
    const contentDiv = document.createElement("div");
    contentDiv.style.display = isCollapsed ? "none" : "flex";
    contentDiv.style.flexDirection = "column";
    contentDiv.style.gap = "6px";
    contentDiv.style.width = "100%";

    header.onclick = (e) => {
        if (e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT' || e.target.tagName === 'LABEL') return;
        const isNowCollapsed = contentDiv.style.display !== "none";
        contentDiv.style.display = isNowCollapsed ? "none" : "flex";
        toggleArrow.textContent = isNowCollapsed ? "▶" : "▼";
        if (!node.uiState.vaesCollapsed) node.uiState.vaesCollapsed = {};
        node.uiState.vaesCollapsed[uid] = isNowCollapsed;
    };

    // File/Folder/Remote Type Select
    const typeSelect = document.createElement("select");
    typeSelect.className = "cb-select";
    typeSelect.innerHTML = `
        <option value="file" ${(!isFolder && !isRemote) ? 'selected' : ''}>VAE File</option>
        <option value="folder" ${isFolder ? 'selected' : ''}>Folder</option>
        <option value="remote" ${isRemote ? 'selected' : ''}>Remote URL</option>
    `;
    typeSelect.onchange = () => {
        let newVal;
        if (typeSelect.value === "folder") {
            newVal = "/";
        } else if (typeSelect.value === "remote") {
            newVal = "remote:";
        } else {
            newVal = "None";
        }
        node.state.config_arrays[arrayIdx].vaes[vaeIdx] = newVal;
        node.saveState();
        debouncedRenderUI(node);
    };
    contentDiv.appendChild(typeSelect);

    if (isRemote) {
        // Async render: drop a placeholder, then swap based on companion availability.
        const placeholder = document.createElement('div');
        placeholder.style.fontSize = "9px";
        placeholder.style.color = "#666";
        placeholder.style.padding = "2px 4px";
        placeholder.textContent = "Checking for Remote VAE companion plugin...";
        contentDiv.appendChild(placeholder);

        isRemoteVaeAvailable().then(available => {
            if (!available) {
                placeholder.replaceWith(_renderRemoteVaeInstallCard(node));
            } else {
                loadRemoteVaeEndpoints().then(endpoints => {
                    placeholder.replaceWith(
                        _renderRemoteVaeControls(node, arrayIdx, vaeIdx, vaeName, endpoints)
                    );
                });
            }
        });
    } else {
        // Searchable Select for VAE file or folder
        const options = isFolder ? vFolders : vaeList;
        const currentVal = vaeName || "None";
        const optionsList = (options && options.includes(currentVal)) || currentVal === "None" || currentVal === "/"
            ? options || ["None"]
            : [currentVal, ...(options || ["None"])];

        const nameSearchable = createSearchableSelect(
            optionsList,
            currentVal,
            (value) => {
                node.state.config_arrays[arrayIdx].vaes[vaeIdx] = normalizePath(value);
                node.saveState();
                debouncedRenderUI(node);
            },
            isFolder ? "Search folders..." : "Search VAEs..."
        );
        contentDiv.appendChild(nameSearchable);

        // Folder expand button
        if (isFolder && vaeName !== "None" && vaeName !== "/") {
            const expandBtn = document.createElement("button");
            expandBtn.className = "cb-button";
            expandBtn.style.cssText = "width: 100%; border-left: 3px solid #9900cc; font-size: 11px; margin-top: 4px;";
            expandBtn.textContent = "📂 Add all individually";
            expandBtn.onclick = () => {
                const normalize = (str) => str.replace(/\\/g, "/");
                const folderPrefix = normalize(vaeName);
                const matchingVAEs = vaeList ? vaeList.filter(v => normalize(v).startsWith(folderPrefix)) : [];
                if (matchingVAEs.length > 0) {
                    node.state.config_arrays[arrayIdx].vaes.splice(vaeIdx, 1, ...matchingVAEs);
                    node.saveState();
                    debouncedRenderUI(node);
                } else {
                    alert(`No VAEs found in folder: ${folderPrefix}`);
                }
            };
            contentDiv.appendChild(expandBtn);
        }
    }

    div.appendChild(contentDiv);
    return div;
}

// --- LTX VIDEO SETTINGS SECTION ---

// Derive model_type from a configArray's models[] (matches the Python-side
// state_to_configs_json convention in config_builder_node.py). Returns "checkpoint" by default.
function getConfigArrayModelType(configArray) {
    let modelType = "checkpoint";
    for (const m of (configArray?.models || [])) {
        if (typeof m === 'object' && m !== null && m.type) {
            modelType = m.type;
        }
    }
    return modelType;
}

function isLTXConfigArray(configArray) {
    return getConfigArrayModelType(configArray) === "ltx_video";
}

function renderLTXVideoShape(node, configArray) {
    const ltx = configArray.ltx_video;
    const div = document.createElement("div");
    div.className = "cb-subsection";

    const titleDiv = document.createElement("div");
    titleDiv.className = "cb-subsection-title";
    titleDiv.textContent = "── Video Shape ──";
    div.appendChild(titleDiv);

    // Frame readout — declared up front so the slider onChange handlers can reference it
    const readout = document.createElement("div");
    readout.className = "cb-readout";
    readout.style.cssText = "font-size: 11px; color: #888; padding: 4px 8px;";

    const updateFrameReadout = () => {
        const f = (ltx.duration_seconds * ltx.frame_rate) + 1;
        readout.textContent = "→ Will generate " + f + " frames per video";
    };

    // Duration slider (1–60 sec, step 1)
    div.appendChild(createSlider("Duration (sec)", ltx.duration_seconds, 1, 60, 1, (v) => {
        ltx.duration_seconds = v;
        node.saveState();
        updateFrameReadout();
    }));

    // Frame rate slider (8–60 fps, step 1)
    div.appendChild(createSlider("Frame Rate (fps)", ltx.frame_rate, 8, 60, 1, (v) => {
        ltx.frame_rate = v;
        node.saveState();
        updateFrameReadout();
    }));

    div.appendChild(readout);
    updateFrameReadout();

    return div;
}

function renderLTXModelFiles(node, configArray, modelLists) {
    const ltx = configArray.ltx_video;
    const div = document.createElement("div");
    div.className = "cb-subsection";

    const titleDiv = document.createElement("div");
    titleDiv.className = "cb-subsection-title";
    titleDiv.textContent = "── Model Files ──";
    div.appendChild(titleDiv);

    // Pull from modelLists (same source the existing image-gen pickers use, so the
    // refresh-models button updates everything). Fall back to direct getters if
    // modelLists wasn't provided (defensive — older call sites).
    const textEncoders = (modelLists && modelLists.textEncoders) || getAvailableTextEncoders() || [];
    const vaeList = (modelLists && (modelLists.vaeModels || modelLists.vae)) || getAvailableVAEs() || [];
    const latentUpscaleModels = (modelLists && modelLists.latentUpscaleModels) || getAvailableLatentUpscaleModels() || [];

    const fields = [
        {label: "Dual CLIP 1 (gemma):", optionsList: textEncoders,
            placeholder: "Search text encoders...",
            getter: () => (ltx.clip_models && ltx.clip_models[0]) || "",
            setter: (v) => {
                if (!ltx.clip_models) ltx.clip_models = ["", ""];
                ltx.clip_models[0] = v;
                node.saveState();
            }},
        {label: "Dual CLIP 2 (projection):", optionsList: textEncoders,
            placeholder: "Search text encoders...",
            getter: () => (ltx.clip_models && ltx.clip_models[1]) || "",
            setter: (v) => {
                if (!ltx.clip_models) ltx.clip_models = ["", ""];
                ltx.clip_models[1] = v;
                node.saveState();
            }},
        {label: "Video VAE:", optionsList: vaeList,
            placeholder: "Search VAEs...",
            getter: () => ltx.vae_video || "",
            setter: (v) => { ltx.vae_video = v; node.saveState(); }},
        {label: "Audio VAE:", optionsList: vaeList,
            placeholder: "Search VAEs...",
            getter: () => ltx.vae_audio || "",
            setter: (v) => { ltx.vae_audio = v; node.saveState(); }},
        {label: "Latent Upscaler:", optionsList: latentUpscaleModels,
            placeholder: "Search latent upscalers...",
            getter: () => ltx.latent_upscaler || "",
            setter: (v) => { ltx.latent_upscaler = v; node.saveState(); }},
    ];

    for (const f of fields) {
        const row = document.createElement("div");
        row.className = "cb-row";

        const lbl = document.createElement("label");
        lbl.textContent = f.label;
        lbl.className = "cb-label";
        row.appendChild(lbl);

        const currentVal = f.getter();
        // Ensure the current value is included in the list even if not present in the cached options
        const optionsList = (f.optionsList && f.optionsList.includes(currentVal)) || !currentVal
            ? (f.optionsList || [])
            : [currentVal, ...(f.optionsList || [])];

        const dropdown = createSearchableSelect(
            optionsList,
            currentVal,
            f.setter,
            f.placeholder
        );
        row.appendChild(dropdown);
        div.appendChild(row);
    }

    return div;
}

function renderLTXStage(node, configArray, stageNum) {
    const ltx = configArray.ltx_video;
    const samplerKey = "sampler_stage" + stageNum;
    const sigmasKey = "sigmas_stage" + stageNum;
    const titleText = stageNum === 1
        ? "── Stage 1: Initial Sampling ──"
        : "── Stage 2: Spatial Upscale + Refinement ──";

    const div = document.createElement("div");
    div.className = "cb-subsection";

    const titleDiv = document.createElement("div");
    titleDiv.className = "cb-subsection-title";
    titleDiv.textContent = titleText;
    div.appendChild(titleDiv);

    // Sampler dropdown row
    const samplerRow = document.createElement("div");
    samplerRow.className = "cb-row";

    const samplerLabel = document.createElement("label");
    samplerLabel.className = "cb-label";
    samplerLabel.textContent = "Sampler:";
    samplerRow.appendChild(samplerLabel);

    // Use getAvailableSamplers() (imported from conf-builder-utilities.js) to get the live
    // sampler list — same source as the main image-gen sampler dropdown.
    const currentSampler = ltx[samplerKey] || "";
    const samplerList = getAvailableSamplers();
    // Ensure current value is in the list even if not yet cached
    const samplerOptions = (samplerList.includes(currentSampler) || !currentSampler)
        ? samplerList
        : [currentSampler, ...samplerList];
    const samplerDropdown = createSearchableSelect(
        samplerOptions,
        currentSampler,
        (v) => { ltx[samplerKey] = v; node.saveState(); },
        "Select sampler..."
    );
    samplerRow.appendChild(samplerDropdown);
    div.appendChild(samplerRow);

    // Sigmas text field with Compare button
    const sigmasRow = document.createElement("div");
    sigmasRow.className = "cb-row";

    const sigmasLabel = document.createElement("label");
    sigmasLabel.className = "cb-label";
    sigmasLabel.textContent = "Sigmas:";
    sigmasRow.appendChild(sigmasLabel);

    const sigmasInput = document.createElement("input");
    sigmasInput.type = "text";
    sigmasInput.className = "cb-input";
    sigmasInput.style.cssText = "flex: 1; font-family: monospace; font-size: 10px;";
    sigmasInput.value = Array.isArray(ltx[sigmasKey]) ? ltx[sigmasKey].join("; ") : (ltx[sigmasKey] || "");
    sigmasInput.title =
        "Comma-separated float schedule, e.g. '0.85, 0.7250, 0.4219, 0.0'.\n\n" +
        "Click '+ Compare' to sweep multiple sigma presets - separate them with semicolons:\n" +
        "  '1.0,0.5,0.0; 0.9,0.4,0.0' produces 2 runs per config.";
    sigmasInput.onchange = () => {
        const v = sigmasInput.value;
        if (v.includes(";")) {
            ltx[sigmasKey] = v.split(";").map(s => s.trim()).filter(Boolean);
        } else {
            ltx[sigmasKey] = v;
        }
        node.saveState();
    };
    sigmasRow.appendChild(sigmasInput);

    const compareBtn = document.createElement("button");
    compareBtn.className = "cb-btn";
    compareBtn.textContent = "+ Compare Sigmas";
    compareBtn.title =
        "Test multiple sigma presets for stage " + stageNum + " in one run.\n\n" +
        "Separate presets with semicolons. Each preset is a comma-list of floats.\n" +
        "Example: '1.0, 0.5, 0.0; 0.9, 0.4, 0.0' → 2 runs per config.";
    compareBtn.onclick = () => {
        if (!sigmasInput.value.includes(";")) {
            sigmasInput.value = sigmasInput.value + "; ";
            sigmasInput.focus();
            sigmasInput.setSelectionRange(sigmasInput.value.length, sigmasInput.value.length);
        }
    };
    sigmasRow.appendChild(compareBtn);
    div.appendChild(sigmasRow);

    return div;
}

function renderLTXImageInput(node, configArray) {
    const ltx = configArray.ltx_video;

    const div = document.createElement("div");
    div.className = "cb-subsection";

    const titleDiv = document.createElement("div");
    titleDiv.className = "cb-subsection-title";
    titleDiv.textContent = "── Image-to-Video Input ──";
    div.appendChild(titleDiv);

    const hint = document.createElement("div");
    hint.style.cssText = "font-size: 10px; color: #999; padding: 2px 8px 6px;";
    hint.textContent = "(leave blank for text-to-video)";
    div.appendChild(hint);

    // Image picker — text input that accepts: single path, folder ending in /, or JSON array
    const imgRow = document.createElement("div");
    imgRow.className = "cb-row";
    imgRow.style.cssText = "display: flex; align-items: center; gap: 6px; padding: 4px 8px;";

    const imgLabel = document.createElement("label");
    imgLabel.className = "cb-label";
    imgLabel.textContent = "Image:";
    imgLabel.style.minWidth = "80px";
    imgRow.appendChild(imgLabel);

    const imgInput = document.createElement("input");
    imgInput.type = "text";
    imgInput.className = "cb-input";
    imgInput.style.flex = "1";
    imgInput.placeholder = "path/to/img.png  |  MyImages/  |  [\"a.png\", \"b.png\"]";
    imgInput.value = ltx.input_image == null
        ? ""
        : (Array.isArray(ltx.input_image) ? JSON.stringify(ltx.input_image) : ltx.input_image);
    imgInput.title =
        "Image-to-video input. Leave blank for text-to-video.\n\n" +
        "Accepts:\n" +
        "  • Single file path (e.g. C:/imgs/cat.png)\n" +
        "  • Folder ending in / (expands to all .png/.jpg/.jpeg/.webp inside)\n" +
        "  • JSON array of paths (e.g. [\"a.png\", \"b.png\"])\n\n" +
        "Folders/arrays sweep — one config per image.";
    imgInput.onchange = () => {
        const v = imgInput.value.trim();
        if (!v) {
            ltx.input_image = null;
        } else if (v.startsWith("[")) {
            try { ltx.input_image = JSON.parse(v); }
            catch { ltx.input_image = v; }
        } else {
            ltx.input_image = v;
        }
        node.saveState();
    };

    // Drag-and-drop + clipboard paste support
    const _uploadImageBlob = async (blob, filenameHint) => {
        const formData = new FormData();
        const file = blob instanceof File
            ? blob
            : new File([blob], filenameHint || ("clipboard_" + Date.now() + ".png"), { type: blob.type || "image/png" });
        formData.append("image", file);
        try {
            const resp = await fetch("/upload/image", { method: "POST", body: formData });
            if (!resp.ok) {
                console.error("[LTX] /upload/image failed:", resp.status, await resp.text());
                return null;
            }
            const data = await resp.json();
            return data && data.name ? data.name : null;
        } catch (e) {
            console.error("[LTX] /upload/image error:", e);
            return null;
        }
    };

    // Append uploaded image(s) to whatever is already in the input. If the field
    // is empty, the new value(s) become the value. If non-empty, existing value
    // gets merged into a JSON array with the new image(s) appended. Folders
    // (strings ending in /) get appended too — user can clean up manually if
    // they want pure folder-expansion mode back.
    const _setImageValue = (val) => {
        if (!val) return;
        const incoming = Array.isArray(val) ? val.slice() : [val];

        // Read the current input value and parse what's already there
        const current = imgInput.value.trim();
        let existing = [];
        if (current) {
            if (current.startsWith("[")) {
                try {
                    const parsed = JSON.parse(current);
                    existing = Array.isArray(parsed) ? parsed : [current];
                } catch {
                    existing = [current];
                }
            } else {
                existing = [current];
            }
        }

        const merged = [...existing, ...incoming];
        if (merged.length === 1) {
            ltx.input_image = merged[0];
            imgInput.value = merged[0];
        } else {
            ltx.input_image = merged;
            imgInput.value = JSON.stringify(merged);
        }
        node.saveState();
    };

    // Drag visual feedback — apply to the imgRow so the highlight is visible across label+input
    const _origBorder = imgInput.style.border;
    const _highlightOn = () => {
        imgInput.style.border = "2px dashed #0af";
        imgInput.style.background = "rgba(0, 170, 255, 0.05)";
    };
    const _highlightOff = () => {
        imgInput.style.border = _origBorder;
        imgInput.style.background = "";
    };

    // Drop zone — listen on both the row and the input
    for (const target of [imgRow, imgInput]) {
        target.addEventListener("dragover", (e) => {
            // Only highlight if dragged item contains files
            if (e.dataTransfer && e.dataTransfer.types && e.dataTransfer.types.indexOf("Files") !== -1) {
                e.preventDefault();
                e.stopPropagation();
                _highlightOn();
            }
        });
        target.addEventListener("dragleave", (e) => {
            // Only un-highlight when leaving the actual zone (not entering child element)
            if (e.target === target) _highlightOff();
        });
        target.addEventListener("drop", async (e) => {
            if (!e.dataTransfer || !e.dataTransfer.files || e.dataTransfer.files.length === 0) return;
            e.preventDefault();
            e.stopPropagation();
            _highlightOff();
            const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith("image/"));
            if (files.length === 0) return;
            imgInput.placeholder = "Uploading " + files.length + " image" + (files.length > 1 ? "s" : "") + "...";
            const uploaded = [];
            for (const f of files) {
                const name = await _uploadImageBlob(f, f.name);
                if (name) uploaded.push(name);
            }
            imgInput.placeholder = "path/to/img.png  |  MyImages/  |  [\"a.png\", \"b.png\"]";
            if (uploaded.length > 0) _setImageValue(uploaded);
        });
    }

    // Clipboard paste — when focused in the input
    imgInput.addEventListener("paste", async (e) => {
        if (!e.clipboardData) return;
        // Look for image/* item
        const items = Array.from(e.clipboardData.items || []);
        const imgItem = items.find(it => it.type && it.type.startsWith("image/"));
        if (!imgItem) return;  // Not an image paste — let default text paste happen
        e.preventDefault();
        const blob = imgItem.getAsFile();
        if (!blob) return;
        imgInput.placeholder = "Uploading pasted image...";
        const ext = (blob.type.split("/")[1] || "png").split(";")[0];
        const name = await _uploadImageBlob(blob, "clipboard_" + Date.now() + "." + ext);
        imgInput.placeholder = "path/to/img.png  |  MyImages/  |  [\"a.png\", \"b.png\"]";
        if (name) _setImageValue(name);
    });

    imgRow.appendChild(imgInput);
    div.appendChild(imgRow);

    // Stage 1 strength slider (0-1, step 0.05)
    div.appendChild(createSlider(
        "Stage 1 Strength",
        ltx.image_strength_stage1 != null ? ltx.image_strength_stage1 : 0.8,
        0, 1, 0.05,
        (v) => { ltx.image_strength_stage1 = v; node.saveState(); }
    ));

    // Stage 2 strength slider (0-1, step 0.05)
    div.appendChild(createSlider(
        "Stage 2 Strength",
        ltx.image_strength_stage2 != null ? ltx.image_strength_stage2 : 1.0,
        0, 1, 0.05,
        (v) => { ltx.image_strength_stage2 = v; node.saveState(); }
    ));

    // Image compression slider (0-100, step 1)
    div.appendChild(createSlider(
        "Image Compression",
        ltx.img_compression != null ? ltx.img_compression : 18,
        0, 100, 1,
        (v) => { ltx.img_compression = v; node.saveState(); }
    ));

    return div;
}


function renderLTXAudio(node, configArray) {
    const div = document.createElement("div");
    div.className = "cb-subsection";

    const titleDiv = document.createElement("div");
    titleDiv.className = "cb-subsection-title";
    titleDiv.textContent = "── Audio ──";
    div.appendChild(titleDiv);

    const note = document.createElement("div");
    note.style.cssText = "font-size: 10px; color: #666; padding: 4px 8px;";

    const strong = document.createElement("strong");
    strong.textContent = "On";
    note.append("Audio: ", strong, " (Phase A: always-on. Toggle added in Phase C.)");

    div.appendChild(note);
    return div;
}

function renderLTXSection(node, configArray, arrayIdx, modelLists) {
    if (!isLTXConfigArray(configArray)) return null;

    // Initialize defaults on first render (state lives per-configArray, not on node.state)
    if (!configArray.ltx_video) {
        configArray.ltx_video = {
            clip_models: ["", ""],
            vae_video: "",
            vae_audio: "",
            latent_upscaler: "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            duration_seconds: 5,
            frame_rate: 25,
            sampler_stage1: "euler_ancestral_cfg_pp",
            sigmas_stage1: "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
            sampler_stage2: "euler_cfg_pp",
            sigmas_stage2: "0.85, 0.7250, 0.4219, 0.0",
            image_strength_stage1: 0.8,
            image_strength_stage2: 1.0,
            img_compression: 18,
            input_image: null,
            audio_mode: "on",
        };
        node.saveState();
    }

    const section = document.createElement("div");
    section.className = "cb-section";
    section.id = "ltx-section-" + arrayIdx;

    const header = document.createElement("div");
    header.className = "cb-section-header";

    const title = document.createElement("span");
    title.textContent = "⚙️ LTX Video Settings";
    header.appendChild(title);

    const helpBtn = document.createElement("button");
    helpBtn.className = "cb-help-btn";
    helpBtn.textContent = "?";
    helpBtn.title = "LTX 2.3 two-stage pipeline help";
    header.appendChild(helpBtn);

    section.appendChild(header);

    // Sub-sections — defined in subsequent tasks (A13-A16). For now, stub them
    // as no-op functions so this skeleton renders without errors. The real
    // implementations will replace these stubs.
    if (typeof renderLTXModelFiles === "function") section.appendChild(renderLTXModelFiles(node, configArray, modelLists));
    if (typeof renderLTXVideoShape === "function") section.appendChild(renderLTXVideoShape(node, configArray));
    if (typeof renderLTXStage === "function") {
        section.appendChild(renderLTXStage(node, configArray, 1));
        section.appendChild(renderLTXStage(node, configArray, 2));
    }
    if (typeof renderLTXImageInput === "function") section.appendChild(renderLTXImageInput(node, configArray));
    if (typeof renderLTXAudio === "function") section.appendChild(renderLTXAudio(node, configArray));

    return section;
}

export function renderLorasSection(node, div, configArray, arrayIdx, availableLoras, loraFolders) {
    if (!configArray.loras || configArray.loras.length === 0) configArray.loras = ["None"];

    const isSectionCollapsed = node.uiState.lorasSectionCollapsed[arrayIdx] || false;

    const loraGrid = document.createElement("div");
    loraGrid.className = "cb-list-grid";
    loraGrid.id = `cb-config-${arrayIdx}-loras`;
    // FIX: Removed flexDirection column. Let Grid/Flex handle it.
    loraGrid.style.width = "100%";

    const loraHeader = document.createElement("div");
    loraHeader.className = "cb-section-toggle";
    loraHeader.style.cssText = "padding: 8px; background: #3a3a3a; border-radius: 4px; margin-bottom: 8px; font-weight: bold; color: #4499ff;";

    const totalEntries = configArray.loras.length;
    let totalLoras = 0;
    configArray.loras.forEach(l => {
        const parsed = parseLoraString(l);
        if (parsed.name.endsWith("/*") || parsed.name.endsWith("/")) {
            const folderName = parsed.name.replace(/\*$/, "");
            if (folderName === "/") totalLoras += availableLoras ? availableLoras.length : 0;
            else {
                const prefix = normalizePath(folderName);
                totalLoras += availableLoras ? availableLoras.filter(al => normalizePath(al).startsWith(prefix)).length : 0;
            }
        } else if (parsed.name !== "None") totalLoras++;
    });

    const titleSpan = document.createElement("span");
    titleSpan.textContent = `LoRAs (${totalEntries} Entries, Totaling ${totalLoras} LoRAs)`;
    loraHeader.appendChild(titleSpan);

    // LoRAs section presets
    const lorasPresetRow = createSectionPresetRow("loras",
        () => ({ loras: configArray.loras, lora_weight_arrays: configArray.lora_weight_arrays || {}, lora_strength_lock: configArray.lora_strength_lock || {} }),
        (data) => {
            node.state.config_arrays[arrayIdx].loras = data.loras || ["None"];
            if (data.lora_weight_arrays) node.state.config_arrays[arrayIdx].lora_weight_arrays = data.lora_weight_arrays;
            if (data.lora_strength_lock) node.state.config_arrays[arrayIdx].lora_strength_lock = data.lora_strength_lock;
        },
        node
    );
    lorasPresetRow.onclick = (e) => e.stopPropagation(); // Don't toggle collapse on preset clicks
    loraHeader.appendChild(lorasPresetRow);

    const arrowSpan = document.createElement("span");
    arrowSpan.textContent = isSectionCollapsed ? "▶" : "▼";
    loraHeader.appendChild(arrowSpan);

    loraGrid.appendChild(loraHeader);

    // CONTENT CONTAINER
    const contentContainer = document.createElement("div");
    // Use 'contents' to allow grid/flex wrapping of children to work properly with parent
    contentContainer.style.display = isSectionCollapsed ? "none" : "contents";

    // --- HEADER CLICK (INSTANT) ---
    loraHeader.onclick = () => {
        const isNowCollapsed = contentContainer.style.display === "none";
        if (isNowCollapsed) {
            contentContainer.style.display = "contents";
            arrowSpan.textContent = "▼";
            node.uiState.lorasSectionCollapsed[arrayIdx] = false;
        } else {
            contentContainer.style.display = "none";
            arrowSpan.textContent = "▶";
            node.uiState.lorasSectionCollapsed[arrayIdx] = true;
        }
    };

    configArray.loras.forEach((lora, loraIdx) => {
        contentContainer.appendChild(createLoraElement(node, lora, arrayIdx, loraIdx, availableLoras, loraFolders));
    });

    const addRow = document.createElement("div");
    addRow.style.width = "100%";
    addRow.style.padding = "4px 0";
    const addBtn = document.createElement("button");
    addBtn.className = "cb-button";
    addBtn.style.cssText = "width: 100%; border: 1px dashed #555; background: rgba(0,0,0,0.2); color: #aaa;";
    addBtn.textContent = "➕ Add New LoRA";
    addBtn.onmouseover = () => addBtn.style.background = "rgba(255,255,255,0.1)";
    addBtn.onmouseout = () => addBtn.style.background = "rgba(0,0,0,0.2)";
    addBtn.onclick = () => {
        node.state.config_arrays[arrayIdx].loras.push("None");
        node.saveState();
        debouncedRenderUI(node);
    };
    addRow.appendChild(addBtn);
    contentContainer.appendChild(addRow);

    loraGrid.appendChild(contentContainer);
    div.appendChild(loraGrid);

    // OMIT TRIGGERS (Outside the flex grid loop to stay at bottom)
    const omitContainer = document.createElement("div");
    omitContainer.style.display = isSectionCollapsed ? "none" : "block"; // Separate container for omit, basic block

    // Hack: Attach header click listener to this too? 
    // Easier way: The header click updates TWO containers?
    // Or simpler: put omitContainer inside contentContainer? 
    // contentContainer is display: contents, so omitContainer becomes a flex item. 
    // It should be full width.
    omitContainer.style.width = "100%";
    omitContainer.style.flexBasis = "100%"; // Force new line in flex wrap

    renderOmitTriggersSection(node, omitContainer, configArray, arrayIdx);
    contentContainer.appendChild(omitContainer);
}

function renderOmitTriggersSection(node, div, configArray, arrayIdx) {
    const omitSection = document.createElement("div");
    omitSection.style.cssText = `
        width: 100%; background: #252525; border-radius: 4px; padding: 10px; margin-top: 10px; border-left: 3px solid #cc6600;
    `;

    const omitTitle = document.createElement("div");
    omitTitle.textContent = "🚫 Omit Trigger Words";
    omitTitle.style.cssText = "font-weight: bold; margin-bottom: 8px; color: #cc6600; font-size: 12px;";
    omitSection.appendChild(omitTitle);

    if (!configArray.lora_omit_triggers) configArray.lora_omit_triggers = [];

    const chipsContainer = document.createElement("div");
    chipsContainer.style.cssText = `display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; min-height: 30px;`;

    // Warning element for when omits are set but no trigger append settings exist
    const warningEl = document.createElement("div");
    warningEl.style.cssText = "display: none; background: #3d2e00; border: 1px solid #cc6600; border-radius: 4px; padding: 6px 10px; margin-bottom: 8px; font-size: 11px; color: #ffaa00;";
    warningEl.textContent = "⚠️ Omit list is active, but no LoRAs have trigger word append settings configured. Make sure the Sampler node's \"lora_triggerwords_mode\" is not set to \"None\", or omits will have no effect.";
    omitSection.appendChild(warningEl);

    const updateOmitWarning = () => {
        const hasOmits = configArray.lora_omit_triggers && configArray.lora_omit_triggers.length > 0;
        const appendSettings = configArray.lora_triggerwords_append_settings || {};
        const hasAppendSettings = Object.values(appendSettings).some(v => v === "start" || v === "end");
        // Show warning if omits exist but no per-lora append settings are configured
        warningEl.style.display = (hasOmits && !hasAppendSettings) ? "block" : "none";
    };

    const renderChips = () => {
        chipsContainer.innerHTML = "";
        updateOmitWarning();
        if (configArray.lora_omit_triggers.length === 0) {
            const placeholder = document.createElement("div");
            placeholder.textContent = "No triggers omitted";
            placeholder.style.cssText = "color: #666; font-style: italic; padding: 4px;";
            chipsContainer.appendChild(placeholder);
            return;
        }
        configArray.lora_omit_triggers.forEach((trigger, tIdx) => {
            const chip = document.createElement("div");
            chip.style.cssText = `display: flex; align-items: center; background: #444; color: #fff; border-radius: 12px; padding: 2px 8px; font-size: 11px;`;
            const text = document.createElement("span");
            text.textContent = trigger;
            chip.appendChild(text);
            const closeBtn = document.createElement("span");
            closeBtn.textContent = "×";
            closeBtn.style.cssText = "margin-left: 6px; cursor: pointer; color: #ff8888; font-weight: bold;";
            closeBtn.onclick = () => {
                node.state.config_arrays[arrayIdx].lora_omit_triggers.splice(tIdx, 1);
                node.saveState();
                renderChips();
            };
            chip.appendChild(closeBtn);
            chipsContainer.appendChild(chip);
        });
    };
    renderChips();
    omitSection.appendChild(chipsContainer);

    const inputRow = document.createElement("div");
    inputRow.style.cssText = "display: flex; gap: 8px; margin-bottom: 8px;";
    const triggerInput = document.createElement("input");
    triggerInput.className = "cb-input";
    triggerInput.placeholder = "Enter trigger to omit...";
    triggerInput.style.flex = "1";
    triggerInput.onkeydown = (e) => { if (e.key === "Enter" && triggerInput.value.trim()) addTrigger(); };

    const addTriggerBtn = document.createElement("button");
    addTriggerBtn.className = "cb-button primary";
    addTriggerBtn.textContent = "Add";
    addTriggerBtn.style.padding = "4px 12px";

    const addTrigger = () => {
        const val = triggerInput.value.trim();
        if (val && !configArray.lora_omit_triggers.includes(val)) {
            node.state.config_arrays[arrayIdx].lora_omit_triggers.push(val);
            node.saveState();
            renderChips();
            triggerInput.value = "";
        }
    };
    addTriggerBtn.onclick = addTrigger;

    const removeAllBtn = document.createElement("button");
    removeAllBtn.className = "cb-button";
    removeAllBtn.textContent = "Clear";
    removeAllBtn.title = "Remove all omitted triggers";
    removeAllBtn.style.cssText = "padding: 4px 10px; font-size: 11px; min-width: 40px; color: #ff8888;";
    removeAllBtn.onclick = () => {
        node.state.config_arrays[arrayIdx].lora_omit_triggers = [];
        configArray.lora_omit_triggers = [];
        node.saveState();
        renderChips();
    };

    inputRow.appendChild(triggerInput);
    inputRow.appendChild(addTriggerBtn);
    inputRow.appendChild(removeAllBtn);
    omitSection.appendChild(inputRow);

    const lookupBtn = document.createElement("button");
    lookupBtn.className = "cb-button";
    lookupBtn.style.cssText = `width: 100%; background: linear-gradient(135deg, #0066cc, #0088ff); border-left: 4px solid #00aaff; margin-top: 4px;`;
    lookupBtn.textContent = "🔎 Lookup Current LoRA Triggerwords For Review";
    // NOTE: This assumes showTriggerLookupModal is exported/available. 
    // It is defined in this same file below (or above if moved). 
    // Ensure showTriggerLookupModal is imported or defined in scope.
    // In this module it is defined in the same file.
    lookupBtn.onclick = async () => await showTriggerLookupModal(node, arrayIdx);
    omitSection.appendChild(lookupBtn);

    div.appendChild(omitSection);
}

// --- TRIGGER LOOKUP MODAL ---
export async function showTriggerLookupModal(node, arrayIdx) {
    const configArray = node.state.config_arrays[arrayIdx];
    const overlay = document.createElement("div");
    overlay.style.cssText = `position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.8); display: flex; align-items: center; justify-content: center; z-index: 10000;`;
    const modal = document.createElement("div");
    modal.style.cssText = `position: relative; background: #2a2a2a; border: 2px solid #0066cc; border-radius: 8px; padding: 20px; max-width: 600px; max-height: 80vh; overflow-y: auto; color: white;`;

    // Shared close function to clean up modal and event listener
    const closeModal = () => {
        if (overlay.parentNode) document.body.removeChild(overlay);
        document.removeEventListener('keydown', escHandler);
    };

    // X close button in top right
    const closeX = document.createElement("button");
    closeX.textContent = "\u2716";
    closeX.style.cssText = "position: absolute; top: 10px; right: 10px; background: #cc3333; color: white; border: none; border-radius: 4px; width: 28px; height: 28px; font-size: 16px; cursor: pointer; display: flex; align-items: center; justify-content: center; z-index: 1;";
    closeX.onmouseover = () => closeX.style.background = "#dd4444";
    closeX.onmouseout = () => closeX.style.background = "#cc3333";
    closeX.onclick = closeModal;
    modal.appendChild(closeX);

    // Close on Escape key
    const escHandler = (e) => {
        if (e.key === 'Escape') closeModal();
    };
    document.addEventListener('keydown', escHandler);

    const title = document.createElement("h3");
    title.textContent = "🔎 LoRA Trigger Words Lookup";
    title.style.cssText = "margin: 0 0 15px 0; color: #0066cc;";
    modal.appendChild(title);

    // CivitAI companion notice — shows when companion missing
    const _twLookupNoticeSlot = document.createElement('div');
    modal.appendChild(_twLookupNoticeSlot);
    isCivitaiAvailable().then(available => {
        if (!available) _twLookupNoticeSlot.replaceWith(_renderCivitaiCompanionNotice());
    });

    const status = document.createElement("div");
    status.textContent = "🔄 Fetching trigger words from CivitAI...";
    status.style.cssText = "margin-bottom: 15px; color: #aaa;";
    modal.appendChild(status);
    const content = document.createElement("div");
    modal.appendChild(content);
    const buttonBar = document.createElement("div");
    buttonBar.style.cssText = "display: flex; gap: 10px; margin-top: 15px; justify-content: flex-end;";
    const addAllBtn = document.createElement("button");
    addAllBtn.className = "cb-button primary";
    addAllBtn.textContent = "➕ Add All Selected to Omit List";
    addAllBtn.disabled = true;
    const closeBtn = document.createElement("button");
    closeBtn.className = "cb-button";
    closeBtn.textContent = "Close";
    closeBtn.onclick = closeModal;
    buttonBar.appendChild(addAllBtn);
    buttonBar.appendChild(closeBtn);
    modal.appendChild(buttonBar);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    try {
        const loras = configArray.loras.filter(l => l && l !== "None");
        const response = await fetch("/configbuilder/lookup_triggers", {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ loras })
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        const triggers = data.triggers || {};
        status.textContent = `✅ Found triggers for ${Object.keys(triggers).length} LoRAs`;
        const selectedTriggers = new Set();
        // Build a normalized set of omitted triggers for visual comparison
        const omitSet = new Set((configArray.lora_omit_triggers || []).map(t => t.toLowerCase().trim()));

        Object.entries(triggers).forEach(([loraName, triggerList]) => {
            const loraSection = document.createElement("div");
            loraSection.style.cssText = `background: #333; border-left: 3px solid #0066cc; padding: 10px; margin-bottom: 10px; border-radius: 4px;`;
            const loraTitle = document.createElement("div");
            loraTitle.textContent = getShortName(loraName.replace('.safetensors', ''));
            loraTitle.style.cssText = "font-weight: bold; margin-bottom: 8px; color: #0066cc;";
            loraSection.appendChild(loraTitle);
            if (!triggerList || triggerList.length === 0) {
                const noTriggers = document.createElement("div");
                noTriggers.textContent = "No triggers found";
                noTriggers.style.cssText = "color: #888; font-style: italic;";
                loraSection.appendChild(noTriggers);
            } else {
                triggerList.forEach(trigger => {
                    const isOmitted = omitSet.has(trigger.toLowerCase().trim());
                    const triggerRow = document.createElement("label");
                    triggerRow.style.cssText = `display: flex; align-items: center; gap: 8px; padding: 4px; cursor: pointer; border-radius: 3px;`;
                    triggerRow.onmouseover = () => triggerRow.style.background = "#444";
                    triggerRow.onmouseout = () => triggerRow.style.background = "transparent";
                    const checkbox = document.createElement("input");
                    checkbox.type = "checkbox";
                    checkbox.checked = false;
                    checkbox.onchange = () => {
                        if (checkbox.checked) selectedTriggers.add(trigger); else selectedTriggers.delete(trigger);
                        addAllBtn.disabled = selectedTriggers.size === 0;
                    };
                    const triggerText = document.createElement("span");
                    triggerText.textContent = trigger;
                    if (isOmitted) {
                        triggerText.style.cssText = "color: #888; text-decoration: line-through; opacity: 0.5;";
                        // Add omitted label
                        const omitLabel = document.createElement("span");
                        omitLabel.textContent = "(omitted)";
                        omitLabel.style.cssText = "color: #cc6600; font-size: 10px; margin-left: 4px;";
                        triggerRow.appendChild(checkbox);
                        triggerRow.appendChild(triggerText);
                        triggerRow.appendChild(omitLabel);
                    } else {
                        triggerText.style.cssText = "color: #ddd;";
                        triggerRow.appendChild(checkbox);
                        triggerRow.appendChild(triggerText);
                    }
                    loraSection.appendChild(triggerRow);
                });
            }
            content.appendChild(loraSection);
        });
        addAllBtn.onclick = () => {
            const existing = new Set(configArray.lora_omit_triggers);
            selectedTriggers.forEach(t => existing.add(t));
            node.state.config_arrays[arrayIdx].lora_omit_triggers = Array.from(existing);
            node.saveState();
            debouncedRenderUI(node);
            document.body.removeChild(overlay);
        };
    } catch (error) {
        status.textContent = `❌ Error: ${error.message}`;
        console.error("[ConfigBuilder] Trigger lookup error:", error);
    }
}

// =============================================================================
// --- PROMPT BUILDER COMPONENTS ---
// =============================================================================

/**
 * Creates a reusable prompt group editor.
 * Supports visual chip/tag mode and raw JSON mode.
 * Shows live preview of Cartesian product combinations.
 *
 * @param {Array} groups - Nested array of prompt groups: [["a", "b"], ["c", "d"]]
 * @param {Function} onChange - Callback when groups change: (newGroups) => void
 * @param {string} label - Label for this editor (e.g. "Positive Prompt" or "Negative Prompt")
 * @param {string} borderColor - CSS color for left border accent
 * @returns {HTMLElement}
 */
function createPromptGroupEditor(groups, onChange, label, borderColor = "#0066cc", { rawModeKey = null, uiState = null } = {}) {
    const container = document.createElement("div");
    container.style.cssText = `display: flex; flex-direction: column; gap: 8px; width: 100%;`;

    // Track current groups locally so mode switches use latest data
    let currentGroupsState = groups;

    // Track mode state — persist in uiState if key provided, otherwise local
    let isRawMode = (rawModeKey && uiState && uiState.promptRawMode) ? (uiState.promptRawMode[rawModeKey] || false) : false;

    // --- HEADER ---
    const header = document.createElement("div");
    header.style.cssText = "display: flex; justify-content: space-between; align-items: center;";

    const titleEl = document.createElement("div");
    titleEl.style.cssText = `font-size: 12px; font-weight: bold; color: ${borderColor};`;
    titleEl.textContent = label;
    header.appendChild(titleEl);

    const modeToggleContainer = document.createElement("div");
    modeToggleContainer.style.cssText = "display: flex; gap: 4px;";

    const visualBtn = document.createElement("button");
    visualBtn.className = isRawMode ? "cb-prompt-mode-toggle" : "cb-prompt-mode-toggle active";
    visualBtn.textContent = "Visual";

    const rawBtn = document.createElement("button");
    rawBtn.className = isRawMode ? "cb-prompt-mode-toggle active" : "cb-prompt-mode-toggle";
    rawBtn.textContent = "JSON";

    modeToggleContainer.appendChild(visualBtn);
    modeToggleContainer.appendChild(rawBtn);
    header.appendChild(modeToggleContainer);
    container.appendChild(header);

    // --- VISUAL MODE CONTAINER ---
    const visualContainer = document.createElement("div");
    visualContainer.style.cssText = `display: ${isRawMode ? "none" : "flex"}; flex-direction: column; gap: 6px;`;

    // --- RAW MODE CONTAINER ---
    const rawContainer = document.createElement("div");
    rawContainer.style.display = isRawMode ? "block" : "none";

    const rawTextarea = document.createElement("textarea");
    rawTextarea.className = "cb-prompt-raw-editor";
    rawTextarea.placeholder = '[["variation1", "variation2"], ["subject1", "subject2"]]\nRecursive: ["fixed text", ["optA", "optB"], ["sub", ["nested1", "nested2"]]]';
    rawTextarea.value = groups.length > 0 ? JSON.stringify(groups, null, 2) : "";

    rawTextarea.onchange = () => {
        try {
            const parsed = JSON.parse(rawTextarea.value);
            if (Array.isArray(parsed)) {
                // Accept recursive structures directly - the backend and preview
                // functions handle arbitrary nesting depth
                currentGroupsState = parsed;
                onChange(parsed);
                rawTextarea.style.borderColor = "#00aa44";
                setTimeout(() => { rawTextarea.style.borderColor = "#3a3a3a"; }, 1000);
                renderVisualGroups(parsed);
                renderPreview(parsed);
            }
        } catch (e) {
            rawTextarea.style.borderColor = "#cc3333";
        }
    };
    rawContainer.appendChild(rawTextarea);

    // --- MODE TOGGLE LOGIC ---
    visualBtn.onclick = () => {
        isRawMode = false;
        visualContainer.style.display = "flex";
        rawContainer.style.display = "none";
        visualBtn.classList.add("active");
        rawBtn.classList.remove("active");
        if (rawModeKey && uiState && uiState.promptRawMode) uiState.promptRawMode[rawModeKey] = false;
    };
    rawBtn.onclick = () => {
        isRawMode = true;
        visualContainer.style.display = "none";
        rawContainer.style.display = "block";
        rawBtn.classList.add("active");
        visualBtn.classList.remove("active");
        if (rawModeKey && uiState && uiState.promptRawMode) uiState.promptRawMode[rawModeKey] = true;
        // Sync raw editor with current groups (use tracked state, not stale parameter)
        rawTextarea.value = currentGroupsState.length > 0 ? JSON.stringify(currentGroupsState, null, 2) : "";
    };

    container.appendChild(visualContainer);
    container.appendChild(rawContainer);

    // --- PREVIEW SECTION ---
    const previewContainer = document.createElement("div");
    container.appendChild(previewContainer);

    function renderPreview(currentGroups) {
        previewContainer.innerHTML = "";
        if (!currentGroups || !Array.isArray(currentGroups) || currentGroups.length === 0) return;

        // Pass the full structure to count/preview — they handle recursive nesting
        const count = countPromptCombinations(currentGroups);
        const previews = expandPromptPreview(currentGroups, 20);

        const previewDiv = document.createElement("div");
        previewDiv.className = "cb-prompt-preview";

        const countLabel = document.createElement("div");
        countLabel.style.cssText = "font-weight: bold; margin-bottom: 4px; color: #00cc88;";
        countLabel.textContent = `${count} combination${count !== 1 ? 's' : ''}${count > 20 ? ' (showing first 20)' : ''}:`;
        previewDiv.appendChild(countLabel);

        previews.forEach(preview => {
            const item = document.createElement("div");
            item.className = "cb-prompt-preview-item";
            item.textContent = preview;
            previewDiv.appendChild(item);
        });

        previewContainer.appendChild(previewDiv);
    }

    // --- VISUAL GROUP RENDERING ---
    function renderVisualGroups(currentGroups) {
        visualContainer.innerHTML = "";

        // Detect if structure uses deep recursive nesting (anything beyond 2 levels)
        const hasDeepNesting = (currentGroups || []).some(item => {
            if (!Array.isArray(item)) return false;
            return item.some(sub => Array.isArray(sub));
        });

        if (hasDeepNesting) {
            // Show read-only notice for recursive structures - edit in JSON mode
            const notice = document.createElement("div");
            notice.style.cssText = "padding: 8px; background: #2a2a3a; border: 1px solid #5544aa; border-radius: 4px; margin-bottom: 8px;";
            notice.innerHTML = `
                <div style="font-size: 11px; color: #aa88ff; font-weight: bold; margin-bottom: 4px;">🔀 Recursive Cartesian Structure Detected</div>
                <div style="font-size: 10px; color: #888;">This prompt uses nested arrays for recursive Cartesian products. Use the <b>JSON</b> editor to modify it.
                <br><br><b>Format:</b> <code style="color: #aaa;">["fixed text", ["option A", "option B"], ["sub1", ["nested1", "nested2"]]]</code>
                <br><b>Rules:</b> Strings = literal text, flat lists = options (OR), lists containing lists = sequence (AND, Cartesian product)</div>
            `;
            visualContainer.appendChild(notice);

            // Show flattened read-only preview of the structure
            (currentGroups || []).forEach((item, idx) => {
                const itemDiv = document.createElement("div");
                itemDiv.style.cssText = "padding: 4px 8px; font-size: 11px; color: #aaa; font-family: monospace; background: #1a1a2a; border-radius: 3px; margin-bottom: 2px;";
                if (Array.isArray(item)) {
                    itemDiv.textContent = `[${idx}] ${JSON.stringify(item)}`;
                    itemDiv.style.color = "#88aaff";
                } else {
                    itemDiv.textContent = `[${idx}] "${item}"`;
                    itemDiv.style.color = "#aaffaa";
                }
                visualContainer.appendChild(itemDiv);
            });
            return;
        }

        // Standard 2-level visual editor for flat groups
        (currentGroups || []).forEach((group, groupIdx) => {
            // Normalize: if group is a string, wrap it for display
            const groupArr = Array.isArray(group) ? group : [String(group)];

            const groupDiv = document.createElement("div");
            groupDiv.className = "cb-prompt-group";

            // Group header
            const groupHeader = document.createElement("div");
            groupHeader.className = "cb-prompt-group-header";

            const groupLabel = document.createElement("span");
            groupLabel.textContent = `Group ${groupIdx + 1} (${groupArr.length} variation${groupArr.length !== 1 ? 's' : ''})`;
            groupHeader.appendChild(groupLabel);

            const groupDeleteBtn = document.createElement("button");
            groupDeleteBtn.className = "cb-button danger";
            groupDeleteBtn.style.cssText = "padding: 1px 6px; font-size: 10px;";
            groupDeleteBtn.textContent = "✖";
            groupDeleteBtn.title = "Remove this group";
            groupDeleteBtn.onclick = () => {
                const newGroups = [...currentGroups];
                newGroups.splice(groupIdx, 1);
                currentGroupsState = newGroups;
                onChange(newGroups);
                renderVisualGroups(newGroups);
                renderPreview(newGroups);
                rawTextarea.value = newGroups.length > 0 ? JSON.stringify(newGroups, null, 2) : "";
            };
            groupHeader.appendChild(groupDeleteBtn);
            groupDiv.appendChild(groupHeader);

            // Chips container
            const chipsDiv = document.createElement("div");
            chipsDiv.className = "cb-prompt-chips";

            groupArr.forEach((variation, varIdx) => {
                const chip = document.createElement("span");
                chip.className = "cb-prompt-chip";

                const chipText = document.createElement("span");
                chipText.textContent = String(variation);
                chipText.title = String(variation);
                chip.appendChild(chipText);

                const chipClose = document.createElement("span");
                chipClose.className = "chip-close";
                chipClose.textContent = "×";
                chipClose.onclick = () => {
                    const newGroups = currentGroups.map(g => Array.isArray(g) ? [...g] : [String(g)]);
                    newGroups[groupIdx].splice(varIdx, 1);
                    // Remove group if empty
                    if (newGroups[groupIdx].length === 0) {
                        newGroups.splice(groupIdx, 1);
                    }
                    currentGroupsState = newGroups;
                    onChange(newGroups);
                    renderVisualGroups(newGroups);
                    renderPreview(newGroups);
                    rawTextarea.value = newGroups.length > 0 ? JSON.stringify(newGroups, null, 2) : "";
                };
                chip.appendChild(chipClose);
                chipsDiv.appendChild(chip);
            });

            // Add variation input
            const addVarBtn = document.createElement("button");
            addVarBtn.className = "cb-prompt-add-variation";
            addVarBtn.textContent = "+ variation";
            addVarBtn.onclick = () => {
                // Replace button with inline input
                const inputContainer = document.createElement("div");
                inputContainer.style.cssText = "display: inline-flex; gap: 2px;";

                const input = document.createElement("input");
                input.className = "cb-input";
                input.style.cssText = "width: 180px; padding: 3px 6px; font-size: 12px;";
                input.placeholder = "Enter prompt text...";

                const confirmBtn = document.createElement("button");
                confirmBtn.className = "cb-button primary";
                confirmBtn.style.cssText = "padding: 2px 8px; font-size: 11px;";
                confirmBtn.textContent = "Add";

                const addVariation = () => {
                    const val = input.value.trim();
                    if (val) {
                        const newGroups = currentGroups.map(g => Array.isArray(g) ? [...g] : [String(g)]);
                        newGroups[groupIdx].push(val);
                        currentGroupsState = newGroups;
                        onChange(newGroups);
                        renderVisualGroups(newGroups);
                        renderPreview(newGroups);
                        rawTextarea.value = newGroups.length > 0 ? JSON.stringify(newGroups, null, 2) : "";
                    }
                };

                input.onkeydown = (e) => {
                    if (e.key === "Enter") addVariation();
                    if (e.key === "Escape") {
                        inputContainer.replaceWith(addVarBtn);
                    }
                };
                confirmBtn.onclick = addVariation;

                inputContainer.appendChild(input);
                inputContainer.appendChild(confirmBtn);
                addVarBtn.replaceWith(inputContainer);
                input.focus();
            };
            chipsDiv.appendChild(addVarBtn);

            groupDiv.appendChild(chipsDiv);
            visualContainer.appendChild(groupDiv);
        });

        // Add Group button
        const addGroupBtn = document.createElement("button");
        addGroupBtn.className = "cb-prompt-add-group-btn";
        addGroupBtn.textContent = "➕ Add Prompt Group";
        addGroupBtn.onclick = () => {
            // Create input for the first variation of the new group
            const inputContainer = document.createElement("div");
            inputContainer.style.cssText = "display: flex; gap: 4px; margin-top: 4px;";

            const input = document.createElement("input");
            input.className = "cb-input";
            input.style.cssText = "flex: 1; padding: 6px 8px; font-size: 12px;";
            input.placeholder = "Enter first variation for new group...";

            const confirmBtn = document.createElement("button");
            confirmBtn.className = "cb-button primary";
            confirmBtn.style.padding = "4px 12px";
            confirmBtn.textContent = "Add Group";

            const addGroup = () => {
                const val = input.value.trim();
                if (val) {
                    const newGroups = [...(currentGroups || []), [val]];
                    currentGroupsState = newGroups;
                    onChange(newGroups);
                    renderVisualGroups(newGroups);
                    renderPreview(newGroups);
                    rawTextarea.value = newGroups.length > 0 ? JSON.stringify(newGroups, null, 2) : "";
                }
            };

            input.onkeydown = (e) => {
                if (e.key === "Enter") addGroup();
                if (e.key === "Escape") {
                    inputContainer.replaceWith(addGroupBtn);
                }
            };
            confirmBtn.onclick = addGroup;

            inputContainer.appendChild(input);
            inputContainer.appendChild(confirmBtn);
            addGroupBtn.replaceWith(inputContainer);
            input.focus();
        };
        visualContainer.appendChild(addGroupBtn);
    }

    // Initial render
    renderVisualGroups(groups);
    renderPreview(groups);

    return container;
}

// --- GLOBAL PROMPTS SECTION ---

export function renderGlobalPromptsSection(node, container) {
    const isCollapsed = node.uiState.globalPromptsSectionCollapsed || false;

    const section = document.createElement("div");
    section.className = "cb-section full-width";
    section.id = "cb-sec-prompts";

    // Header (collapsible) — uniform section header
    const header = createSectionHeader("📝", "Global Prompts (Override Node Inputs)", "prompts", {
        collapsible: true,
        collapsed: isCollapsed,
        onToggle: () => {
            const isNowCollapsed = contentDiv.style.display !== "none";
            contentDiv.style.display = isNowCollapsed ? "none" : "flex";
            // Update arrow in header
            const arrow = header.querySelector("span:last-child");
            if (arrow) arrow.textContent = isNowCollapsed ? "▶" : "▼";
            node.uiState.globalPromptsSectionCollapsed = isNowCollapsed;
        }
    });

    section.appendChild(header);

    // Content container
    const contentDiv = document.createElement("div");
    contentDiv.style.display = isCollapsed ? "none" : "flex";
    contentDiv.style.flexDirection = "column";
    contentDiv.style.gap = "12px";

    // Info text
    const info = document.createElement("div");
    info.style.cssText = "font-size: 11px; color: #999; font-style: italic;";
    info.textContent = "These prompts override the sampler node's positive/negative text inputs for ALL config arrays. Per-config prompts (below) can further override these.";
    contentDiv.appendChild(info);

    // Positive Prompt Editor
    const positiveSection = document.createElement("div");
    positiveSection.className = "cb-prompt-section global";

    const positiveEditor = createPromptGroupEditor(
        node.state.global_positive_groups || [],
        (newGroups) => {
            node.state.global_positive_groups = newGroups;
            node.saveState();
        },
        "✅ Positive Prompt Groups",
        "#00aa44",
        { rawModeKey: "global_positive", uiState: node.uiState }
    );
    positiveSection.appendChild(positiveEditor);
    // Scrollable container for positive prompts (can get very long)
    positiveSection.style.maxHeight = "1000px";
    positiveSection.style.overflowY = "auto";
    contentDiv.appendChild(positiveSection);

    // Negative Prompt (simple text area, not nested groups)
    const negativeSection = document.createElement("div");
    negativeSection.className = "cb-prompt-section global";

    const negLabel = document.createElement("div");
    negLabel.style.cssText = "font-size: 12px; font-weight: bold; color: #cc4444; margin-bottom: 6px;";
    negLabel.textContent = "❌ Negative Prompt";
    negativeSection.appendChild(negLabel);

    const negInput = document.createElement("textarea");
    negInput.className = "cb-prompt-raw-editor";
    negInput.style.minHeight = "50px";
    negInput.placeholder = "bad quality, worst quality, lowres";
    negInput.value = node.state.global_negative || "";
    negInput.onchange = () => {
        node.state.global_negative = negInput.value;
        node.saveState();
    };
    negativeSection.appendChild(negInput);
    contentDiv.appendChild(negativeSection);

    section.appendChild(contentDiv);
    container.appendChild(section);
}

// --- PER-CONFIG PROMPTS SECTION ---

export function renderConfigPromptsSection(node, div, configArray, arrayIdx) {
    const uid = `prompts_${arrayIdx}`;
    const isCollapsed = node.uiState.promptsSectionCollapsed[uid] !== false; // Default collapsed

    const section = document.createElement("div");
    section.style.cssText = "width: 100%; margin-top: 10px; padding-top: 10px; border-top: 1px dashed #444;";

    // Header
    const header = document.createElement("div");
    header.className = "cb-section-toggle";
    header.style.cssText = "padding: 8px; background: #3a3a3a; border-radius: 4px; margin-bottom: 8px; font-weight: bold; color: #9966cc;";

    const promptCount = configArray.use_custom_prompts && configArray.positive_prompt_groups
        ? countPromptCombinations(configArray.positive_prompt_groups)
        : 0;

    const titleSpan = document.createElement("span");
    titleSpan.textContent = `Prompts${promptCount > 0 ? ` (${promptCount} combinations)` : ''}`;
    header.appendChild(titleSpan);

    const arrowSpan = document.createElement("span");
    arrowSpan.textContent = isCollapsed ? "▶" : "▼";
    header.appendChild(arrowSpan);

    section.appendChild(header);

    // Content
    const contentDiv = document.createElement("div");
    contentDiv.style.display = isCollapsed ? "none" : "block";

    header.onclick = () => {
        const isNowCollapsed = contentDiv.style.display !== "none";
        contentDiv.style.display = isNowCollapsed ? "none" : "block";
        arrowSpan.textContent = isNowCollapsed ? "▶" : "▼";
        node.uiState.promptsSectionCollapsed[uid] = isNowCollapsed;
    };

    // Toggle: Use Custom Prompts
    const toggleLabel = document.createElement("label");
    toggleLabel.style.cssText = "display: flex; align-items: center; gap: 8px; cursor: pointer; margin-bottom: 10px; font-size: 13px;";

    const toggleCheck = document.createElement("input");
    toggleCheck.type = "checkbox";
    toggleCheck.checked = configArray.use_custom_prompts || false;
    toggleCheck.onchange = () => {
        node.state.config_arrays[arrayIdx].use_custom_prompts = toggleCheck.checked;
        node.saveState();
        debouncedRenderUI(node);
    };
    toggleLabel.appendChild(toggleCheck);
    toggleLabel.appendChild(document.createTextNode("Use Custom Prompts (Override Global/Node)"));
    contentDiv.appendChild(toggleLabel);

    // Helper to update the prompt count in the section header
    const updatePromptCountDisplay = () => {
        const groups = node.state.config_arrays[arrayIdx].positive_prompt_groups;
        const count = node.state.config_arrays[arrayIdx].use_custom_prompts && groups
            ? countPromptCombinations(groups) : 0;
        titleSpan.textContent = `Prompts${count > 0 ? ` (${count} combinations)` : ''}`;
    };

    if (configArray.use_custom_prompts) {
        // Positive Prompt Editor
        const positiveSection = document.createElement("div");
        positiveSection.className = "cb-prompt-section per-config";

        const positiveEditor = createPromptGroupEditor(
            configArray.positive_prompt_groups || [],
            (newGroups) => {
                node.state.config_arrays[arrayIdx].positive_prompt_groups = newGroups;
                node.saveState();
                updatePromptCountDisplay();
            },
            "✅ Positive Prompt Groups",
            "#9966cc",
            { rawModeKey: `config_${arrayIdx}_positive`, uiState: node.uiState }
        );
        positiveSection.appendChild(positiveEditor);
        // Scrollable container for positive prompts (can get very long)
        positiveSection.style.maxHeight = "1000px";
        positiveSection.style.overflowY = "auto";
        contentDiv.appendChild(positiveSection);

        // Negative Prompt (simple text)
        const negativeSection = document.createElement("div");
        negativeSection.className = "cb-prompt-section per-config";
        negativeSection.style.marginTop = "8px";

        const negLabel = document.createElement("div");
        negLabel.style.cssText = "font-size: 12px; font-weight: bold; color: #cc4444; margin-bottom: 6px;";
        negLabel.textContent = "❌ Negative Prompt";
        negativeSection.appendChild(negLabel);

        const negInput = document.createElement("textarea");
        negInput.className = "cb-prompt-raw-editor";
        negInput.style.minHeight = "50px";
        negInput.placeholder = "bad quality, worst quality, lowres";
        negInput.value = configArray.negative_prompt || "";
        negInput.onchange = () => {
            node.state.config_arrays[arrayIdx].negative_prompt = negInput.value;
            node.saveState();
        };
        negativeSection.appendChild(negInput);
        contentDiv.appendChild(negativeSection);
    } else {
        const infoText = document.createElement("div");
        infoText.style.cssText = "font-size: 11px; color: #666; font-style: italic; padding: 4px;";
        infoText.textContent = "Using global prompts (or node inputs if no global prompts defined).";
        contentDiv.appendChild(infoText);
    }

    // --- Model Prompt Prefix / Suffix (always shown, applies to all prompts for this config) ---
    const affixSection = document.createElement("div");
    affixSection.style.cssText = "width: 100%; margin-top: 10px; padding-top: 8px; border-top: 1px solid #3a3a3a;";

    const affixLabel = document.createElement("div");
    affixLabel.style.cssText = "font-size: 11px; font-weight: bold; color: #88aa00; margin-bottom: 6px;";
    affixLabel.textContent = "🏷️ Model Quality Tags (Prefix / Suffix)";
    affixSection.appendChild(affixLabel);

    const affixHint = document.createElement("div");
    affixHint.style.cssText = "font-size: 10px; color: #666; margin-bottom: 6px; font-style: italic;";
    affixHint.textContent = "Added before/after ALL prompts for this config. Great for model-specific quality tags (e.g. 'score_9, score_8_up' for Pony, 'masterpiece, best quality' for SD1.5).";
    affixSection.appendChild(affixHint);

    // Prefix input
    const prefixRow = document.createElement("div");
    prefixRow.style.cssText = "display: flex; align-items: center; gap: 6px; margin-bottom: 4px;";
    const prefixLabel = document.createElement("span");
    prefixLabel.style.cssText = "font-size: 11px; color: #aaa; min-width: 45px;";
    prefixLabel.textContent = "Prefix:";
    prefixRow.appendChild(prefixLabel);

    const prefixInput = document.createElement("input");
    prefixInput.type = "text";
    prefixInput.className = "cb-input";
    prefixInput.style.cssText = "flex: 1; font-size: 11px;";
    prefixInput.placeholder = "e.g. score_9, score_8_up, score_7_up";
    prefixInput.value = configArray.model_prompt_prefix || "";
    prefixInput.onchange = () => {
        node.state.config_arrays[arrayIdx].model_prompt_prefix = prefixInput.value;
        node.saveState();
    };
    prefixRow.appendChild(prefixInput);
    affixSection.appendChild(prefixRow);

    // Suffix input
    const suffixRow = document.createElement("div");
    suffixRow.style.cssText = "display: flex; align-items: center; gap: 6px;";
    const suffixLabel = document.createElement("span");
    suffixLabel.style.cssText = "font-size: 11px; color: #aaa; min-width: 45px;";
    suffixLabel.textContent = "Suffix:";
    suffixRow.appendChild(suffixLabel);

    const suffixInput = document.createElement("input");
    suffixInput.type = "text";
    suffixInput.className = "cb-input";
    suffixInput.style.cssText = "flex: 1; font-size: 11px;";
    suffixInput.placeholder = "e.g. highly detailed, 8k resolution";
    suffixInput.value = configArray.model_prompt_suffix || "";
    suffixInput.onchange = () => {
        node.state.config_arrays[arrayIdx].model_prompt_suffix = suffixInput.value;
        node.saveState();
    };
    suffixRow.appendChild(suffixInput);
    affixSection.appendChild(suffixRow);

    contentDiv.appendChild(affixSection);

    section.appendChild(contentDiv);
    div.appendChild(section);
}

// ============================================================================
// UPSCALING SETTINGS (Session-level, pipeline-based with sequential steps)
// ============================================================================

// Default Florence2 Hi-Res Fix options
function createDefaultFlorence2Options() {
    return {
        model: "microsoft/Florence-2-base",
        text_input: "face",
        output_mask_select: "",
        max_new_tokens: 1024,
        florence2_input_mp: 0.5,
        target_megapixels: 1.0,
        crop_padding: 64,
        min_crop_resolution: 0,        // No floor — Florence2's polygon decides
        max_crop_resolution: 99999,    // No ceiling — paste-back natural cap is the image itself
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
function createDefaultSeedVR2Options() {
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

// Helper: create a default upscale step with all fields
function createDefaultStep() {
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
        seedvr2: createDefaultSeedVR2Options(),
        florence2: createDefaultFlorence2Options()
    };
}

// Helper: ensure a step has all required fields (migration/defaults)
function ensureStepFields(step) {
    if (step.active === undefined) step.active = true;
    if (!step.repeat || step.repeat < 1) step.repeat = 1;
    if (!step.upscale_size) step.upscale_size = "2.0";
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
    if (!step.seedvr2) step.seedvr2 = createDefaultSeedVR2Options();
    if (!step.florence2) step.florence2 = createDefaultFlorence2Options();
}

export function renderUpscalingSection(node, container, modelLists) {
    if (!node.state.upscaling) {
        node.state.upscaling = {
            enabled: false,
            save_pre_upscale: false,
            run_upscales_at_end: false,
            hires_prompt_adjust: false,
            hires_prompt_behavior: "append_end",
            hires_prompt_text: "",
            pipelines: [{
                active: true,
                name: "Pipeline 1",
                steps: [createDefaultStep()]
            }]
        };
    }
    // Migration: convert old single-config format (no configs array, has mode directly)
    if (node.state.upscaling.mode && !node.state.upscaling.configs && !node.state.upscaling.pipelines) {
        const old = node.state.upscaling;
        const step = createDefaultStep();
        step.mode = old.mode || "hires_only";
        step.upscale_models = old.upscale_model ? [old.upscale_model] : [];
        step.upscale_ratios = String(old.upscale_ratio || "1.5");
        step.upscale_size = String(old.upscale_size || "2.0");
        step.hires_denoise = String(old.hires_denoise || "0.3");
        step.hires_steps = old.hires_steps || 0;
        step.tiled_vae = old.tiled_vae || false;
        step.tile_size = old.tile_size || 512;
        node.state.upscaling = {
            enabled: old.enabled || false,
            pipelines: [{
                active: true,
                name: "Pipeline 1",
                steps: [step]
            }]
        };
    }
    // Migration: convert old configs[] + run_mode format to pipelines
    if (node.state.upscaling.configs && !node.state.upscaling.pipelines) {
        const old = node.state.upscaling;
        const configs = old.configs || [];
        const runMode = old.run_mode || "comparison";
        let pipelines;
        if (runMode === "stacking") {
            // Stacking: all configs become steps in one pipeline
            pipelines = [{
                active: true,
                name: "Pipeline 1",
                steps: configs.map(c => {
                    const step = { ...createDefaultStep(), ...c, repeat: 1 };
                    return step;
                })
            }];
        } else {
            // Comparison: each config becomes its own pipeline with one step
            pipelines = configs.map((c, i) => ({
                active: c.active !== false,
                name: `Pipeline ${i + 1}`,
                steps: [{ ...createDefaultStep(), ...c, repeat: 1, active: true }]
            }));
        }
        node.state.upscaling = {
            enabled: old.enabled || false,
            pipelines: pipelines
        };
        // Clean up old fields from migrated data
        delete node.state.upscaling.run_mode;
        delete node.state.upscaling.configs;
    }
    // Migration: ensure all steps have required fields
    if (node.state.upscaling.pipelines) {
        node.state.upscaling.pipelines.forEach(p => {
            if (p.active === undefined) p.active = true;
            if (!p.name) p.name = "Pipeline";
            if (!p.steps || p.steps.length === 0) p.steps = [createDefaultStep()];
            p.steps.forEach(ensureStepFields);
        });
    }
    // Migration: add new global upscale settings
    if (node.state.upscaling.save_pre_upscale === undefined) node.state.upscaling.save_pre_upscale = false;
    if (node.state.upscaling.run_upscales_at_end === undefined) node.state.upscaling.run_upscales_at_end = false;
    if (node.state.upscaling.hires_prompt_adjust === undefined) node.state.upscaling.hires_prompt_adjust = false;
    if (!node.state.upscaling.hires_prompt_behavior) node.state.upscaling.hires_prompt_behavior = "append_end";
    if (node.state.upscaling.hires_prompt_text === undefined) node.state.upscaling.hires_prompt_text = "";
    const ups = node.state.upscaling;

    const section = document.createElement("div");
    section.className = "cb-section";

    // Header with enable toggle
    const header = document.createElement("div");
    header.className = "cb-section-header";
    header.style.cssText = "display: flex; align-items: center; gap: 8px; padding: 8px 12px; cursor: pointer;";

    const enableCb = document.createElement("input");
    enableCb.type = "checkbox";
    enableCb.checked = ups.enabled;
    enableCb.onclick = (e) => e.stopPropagation();
    enableCb.onchange = () => {
        ups.enabled = enableCb.checked;
        body.style.display = enableCb.checked ? "block" : "none";
        node.saveState();
    };

    const title = document.createElement("span");
    title.textContent = "🔍 Upscaling Settings";
    title.style.cssText = "font-weight: bold; color: #cc99ff; font-size: 14px;";

    // Tooltip (?) icon
    const tooltip = document.createElement("span");
    tooltip.textContent = "?";
    tooltip.style.cssText = "display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; border-radius: 50%; background: #555; color: #fff; font-size: 11px; font-weight: bold; cursor: help; margin-left: 4px; position: relative;";
    tooltip.title = "Upscale models are stored in ComfyUI/models/upscale_models.\n\nModel + Hires mode will take whatever size output the upscale model gives and resize it to the starting image size × Upscale Ratio setting before running HiRes Fix.\n\nPipelines are independent — each starts from the original base image.\nSteps within a pipeline chain sequentially.\nRepeat on a step runs it N times back-to-back (feedback loop).";

    header.appendChild(enableCb);
    header.appendChild(title);
    header.appendChild(tooltip);
    section.appendChild(header);

    // Body
    const body = document.createElement("div");
    body.style.display = ups.enabled ? "block" : "none";
    body.style.padding = "8px 12px";

    // Global upscale settings (above pipelines)
    const globalSettings = document.createElement("div");
    globalSettings.style.cssText = "margin-bottom: 8px; padding: 6px 8px; background: #252525; border-radius: 4px; border: 1px solid #444;";

    // Upscale Presets (load/save pipeline configurations)
    const presetRow = document.createElement("div");
    presetRow.style.cssText = "display: flex; gap: 4px; align-items: center; margin-bottom: 8px;";
    const presetLabel = document.createElement("span");
    presetLabel.textContent = "Preset:";
    presetLabel.style.cssText = "font-size: 11px; color: #999; white-space: nowrap;";
    const presetSelect = document.createElement("select");
    presetSelect.style.cssText = "flex: 1; background: #1a1a1a; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 3px 6px; font-size: 11px;";
    const presetDefaultOpt = document.createElement("option");
    presetDefaultOpt.value = ""; presetDefaultOpt.textContent = "-- Presets --";
    presetSelect.appendChild(presetDefaultOpt);

    // Initialize presets array immediately (fetch will populate it)
    presetSelect._presets = [];

    // Fetch presets async
    fetch("/configbuilder/upscale_presets").then(r => r.json()).then(data => {
        const presets = data.presets || [];
        presets.forEach((p, i) => {
            const opt = document.createElement("option");
            opt.value = i; opt.textContent = p.name;
            presetSelect.appendChild(opt);
        });
        presetSelect._presets = presets;
        console.log("[ConfigBuilder] Loaded " + presets.length + " upscale presets");
    }).catch(e => { console.error("[ConfigBuilder] Failed to load upscale presets:", e); });

    const presetLoadBtn = document.createElement("button");
    presetLoadBtn.textContent = "Load";
    presetLoadBtn.className = "cb-btn";
    presetLoadBtn.style.cssText = "font-size: 10px; padding: 2px 6px;";
    presetLoadBtn.onclick = () => {
        const idx = parseInt(presetSelect.value);
        if (isNaN(idx) || !presetSelect._presets) return;
        const preset = presetSelect._presets[idx];
        if (!preset) return;
        // Apply preset to current upscaling state
        ups.pipelines = JSON.parse(JSON.stringify(preset.pipelines));
        if (preset.hires_prompt_adjust !== undefined) ups.hires_prompt_adjust = preset.hires_prompt_adjust;
        if (preset.hires_prompt_behavior) ups.hires_prompt_behavior = preset.hires_prompt_behavior;
        if (preset.hires_prompt_text !== undefined) ups.hires_prompt_text = preset.hires_prompt_text;
        node.setDirtyCanvas(true);
        // Re-render to reflect the loaded config
        renderUpscalingSection(node, container, modelLists);
    };

    const presetSaveBtn = document.createElement("button");
    presetSaveBtn.textContent = "Save";
    presetSaveBtn.className = "cb-btn";
    presetSaveBtn.style.cssText = "font-size: 10px; padding: 2px 6px;";
    presetSaveBtn.onclick = () => {
        const name = prompt("Preset name:");
        if (!name) return;
        const presets = presetSelect._presets || [];
        const newPreset = {
            name: name,
            pipelines: JSON.parse(JSON.stringify(ups.pipelines)),
            hires_prompt_adjust: ups.hires_prompt_adjust || false,
            hires_prompt_behavior: ups.hires_prompt_behavior || "append_end",
            hires_prompt_text: ups.hires_prompt_text || "",
        };
        // Replace existing preset with same name, or add new
        const existingIdx = presets.findIndex(p => p.name === name);
        if (existingIdx >= 0) {
            presets[existingIdx] = newPreset;
        } else {
            presets.push(newPreset);
        }
        console.log("[ConfigBuilder] Saving upscale preset:", name, "total:", presets.length);
        fetch("/configbuilder/upscale_presets", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({presets: presets})
        }).then(r => {
            console.log("[ConfigBuilder] Preset save response:", r.status);
            presetSelect._presets = presets;
            // Re-render to reflect updated presets without duplicates
            renderUpscalingSection(node, container, modelLists);
        }).catch(e => { console.error("[ConfigBuilder] Preset save failed:", e); });
    };

    presetRow.appendChild(presetLabel);
    presetRow.appendChild(presetSelect);
    presetRow.appendChild(presetLoadBtn);
    presetRow.appendChild(presetSaveBtn);
    globalSettings.appendChild(presetRow);

    // Save Pre-Upscaled Output checkbox
    const preUpscaleLabel = document.createElement("label");
    preUpscaleLabel.style.cssText = "display: flex; align-items: center; gap: 6px; font-size: 12px; color: #ccc; cursor: pointer; margin-bottom: 4px;";
    const preUpscaleCb = document.createElement("input");
    preUpscaleCb.type = "checkbox";
    preUpscaleCb.checked = ups.save_pre_upscale || false;
    preUpscaleCb.onchange = () => { ups.save_pre_upscale = preUpscaleCb.checked; node.saveState(); };
    preUpscaleLabel.appendChild(preUpscaleCb);
    preUpscaleLabel.appendChild(document.createTextNode("Also Save & Display Pre-Upscaled Output"));
    globalSettings.appendChild(preUpscaleLabel);

    // Run Upscales At End Of Session checkbox
    const endUpscaleDiv = document.createElement("div");
    endUpscaleDiv.style.cssText = "margin-bottom: 4px;";
    const endUpscaleLabel = document.createElement("label");
    endUpscaleLabel.style.cssText = "display: flex; align-items: center; gap: 6px; font-size: 12px; color: #ccc; cursor: pointer;";
    const endUpscaleCb = document.createElement("input");
    endUpscaleCb.type = "checkbox";
    endUpscaleCb.checked = ups.run_upscales_at_end || false;
    endUpscaleCb.onchange = () => {
        ups.run_upscales_at_end = endUpscaleCb.checked;
        endUpscaleWarn.style.display = endUpscaleCb.checked ? "block" : "none";
        node.saveState();
    };
    endUpscaleLabel.appendChild(endUpscaleCb);
    endUpscaleLabel.appendChild(document.createTextNode("Run Upscales At End Of Session Instead Of After Each Gen"));
    endUpscaleDiv.appendChild(endUpscaleLabel);
    const endUpscaleDesc = document.createElement("div");
    endUpscaleDesc.style.cssText = "font-size: 9px; color: #666; margin: 2px 0 4px 24px;";
    endUpscaleDesc.textContent = "May help speed on VRAM-constrained devices. Groups upscales by model to minimize swaps.";
    endUpscaleDiv.appendChild(endUpscaleDesc);
    const endUpscaleWarn = document.createElement("div");
    endUpscaleWarn.style.cssText = "font-size: 9px; color: #f80; margin: 2px 0 4px 24px; display: " + (endUpscaleCb.checked ? "block" : "none") + ";";
    endUpscaleWarn.textContent = "WARNING: If \"Also Save & Display Pre-Upscaled Output\" is NOT checked, images won't show up in the Dashboard until the entire run is complete.";
    endUpscaleDiv.appendChild(endUpscaleWarn);
    globalSettings.appendChild(endUpscaleDiv);

    // Adjust Prompt During HiRes Fix checkbox
    const hiresPromptLabel = document.createElement("label");
    hiresPromptLabel.style.cssText = "display: flex; align-items: center; gap: 6px; font-size: 12px; color: #ccc; cursor: pointer;";
    const hiresPromptCb = document.createElement("input");
    hiresPromptCb.type = "checkbox";
    hiresPromptCb.checked = ups.hires_prompt_adjust || false;
    hiresPromptCb.onchange = () => {
        ups.hires_prompt_adjust = hiresPromptCb.checked;
        hiresPromptOptions.style.display = ups.hires_prompt_adjust ? "block" : "none";
        node.saveState();
    };
    hiresPromptLabel.appendChild(hiresPromptCb);
    hiresPromptLabel.appendChild(document.createTextNode("Adjust Prompt During HiRes Fix"));
    globalSettings.appendChild(hiresPromptLabel);

    // HiRes prompt options (shown when checkbox is enabled)
    const hiresPromptOptions = document.createElement("div");
    hiresPromptOptions.style.cssText = "margin-top: 6px; padding: 6px 8px; background: #1e1e1e; border-radius: 4px; display: " + (ups.hires_prompt_adjust ? "block" : "none") + ";";

    // Prompt Adjust Behavior dropdown
    const behaviorSelect = document.createElement("select");
    behaviorSelect.className = "cb-select";
    behaviorSelect.style.cssText = "margin-bottom: 6px; width: 100%;";
    behaviorSelect.innerHTML = `
        <option value="prepend" ${ups.hires_prompt_behavior === "prepend" ? 'selected' : ''}>Append To Front</option>
        <option value="append_end" ${ups.hires_prompt_behavior === "append_end" ? 'selected' : ''}>Append To End</option>
        <option value="replace" ${ups.hires_prompt_behavior === "replace" ? 'selected' : ''}>Replace Prompt</option>
    `;
    behaviorSelect.onchange = () => { ups.hires_prompt_behavior = behaviorSelect.value; node.saveState(); };
    hiresPromptOptions.appendChild(createInputGroup("Prompt Adjust Behavior", behaviorSelect));

    // Prompt text input
    const promptTextInput = document.createElement("textarea");
    promptTextInput.className = "cb-input";
    promptTextInput.style.cssText = "width: 100%; min-height: 60px; resize: vertical; font-size: 12px;";
    promptTextInput.value = ups.hires_prompt_text || "";
    promptTextInput.placeholder = "Enter prompt adjustment text...";
    promptTextInput.onchange = () => { ups.hires_prompt_text = promptTextInput.value; node.saveState(); };
    hiresPromptOptions.appendChild(createInputGroup("HiRes Prompt Text", promptTextInput));

    globalSettings.appendChild(hiresPromptOptions);
    body.appendChild(globalSettings);

    // Pipelines container
    const pipelinesContainer = document.createElement("div");
    body.appendChild(pipelinesContainer);

    function renderPipelines() {
        pipelinesContainer.innerHTML = "";

        ups.pipelines.forEach((pipeline, pipeIdx) => {
            const pipeCard = document.createElement("div");
            pipeCard.style.cssText = "border: 2px solid #665599; border-radius: 8px; margin-bottom: 10px; background: #1a1a2e;" + (pipeline.active === false ? " opacity: 0.5;" : "");

            // Pipeline header (collapsible)
            const pipeHeader = document.createElement("div");
            pipeHeader.style.cssText = "display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; cursor: pointer; background: #252540; border-radius: 6px 6px 0 0;";

            const pipeLeft = document.createElement("div");
            pipeLeft.style.cssText = "display: flex; align-items: center; gap: 8px;";

            const pipeActiveCb = document.createElement("input");
            pipeActiveCb.type = "checkbox";
            pipeActiveCb.checked = pipeline.active !== false;
            pipeActiveCb.title = pipeline.active !== false ? "Pipeline is ON — included in generation" : "Pipeline is OFF — excluded from generation";
            pipeActiveCb.onclick = (e) => e.stopPropagation();
            pipeActiveCb.onchange = () => {
                pipeline.active = pipeActiveCb.checked;
                pipeCard.style.opacity = pipeline.active !== false ? "1" : "0.5";
                node.saveState();
            };
            pipeLeft.appendChild(pipeActiveCb);

            const pipeNameInput = document.createElement("input");
            pipeNameInput.type = "text";
            pipeNameInput.className = "cb-input";
            pipeNameInput.value = pipeline.name || `Pipeline ${pipeIdx + 1}`;
            pipeNameInput.style.cssText = "font-weight: bold; color: #cc99ff; font-size: 13px; background: transparent; border: 1px solid transparent; padding: 2px 6px; width: 180px;";
            pipeNameInput.onclick = (e) => e.stopPropagation();
            pipeNameInput.onfocus = () => { pipeNameInput.style.borderColor = "#665599"; };
            pipeNameInput.onblur = () => { pipeNameInput.style.borderColor = "transparent"; };
            pipeNameInput.onchange = () => { pipeline.name = pipeNameInput.value; node.saveState(); };
            pipeLeft.appendChild(pipeNameInput);

            const pipeStepCount = document.createElement("span");
            pipeStepCount.style.cssText = "font-size: 11px; color: #888;";
            pipeStepCount.textContent = `(${pipeline.steps.length} step${pipeline.steps.length !== 1 ? "s" : ""})`;
            pipeLeft.appendChild(pipeStepCount);

            pipeHeader.appendChild(pipeLeft);

            const pipeRight = document.createElement("div");
            pipeRight.style.cssText = "display: flex; align-items: center; gap: 6px;";

            if (ups.pipelines.length > 1) {
                const pipeDelBtn = document.createElement("button");
                pipeDelBtn.className = "cb-button";
                pipeDelBtn.textContent = "✕";
                pipeDelBtn.style.cssText = "background: #cc3333; padding: 2px 8px; font-size: 12px;";
                pipeDelBtn.onclick = (e) => {
                    e.stopPropagation();
                    ups.pipelines.splice(pipeIdx, 1);
                    node.saveState();
                    renderPipelines();
                };
                pipeRight.appendChild(pipeDelBtn);
            }

            const collapseIcon = document.createElement("span");
            collapseIcon.style.cssText = "color: #888; font-size: 14px; user-select: none;";
            collapseIcon.textContent = "▼";
            pipeRight.appendChild(collapseIcon);

            pipeHeader.appendChild(pipeRight);
            pipeCard.appendChild(pipeHeader);

            // Pipeline body (collapsible content)
            const pipeBody = document.createElement("div");
            pipeBody.style.cssText = "padding: 8px 12px;";

            // Toggle collapse on header click
            pipeHeader.onclick = () => {
                const isCollapsed = pipeBody.style.display === "none";
                pipeBody.style.display = isCollapsed ? "block" : "none";
                collapseIcon.textContent = isCollapsed ? "▼" : "▶";
            };

            // Render steps within this pipeline
            pipeline.steps.forEach((ucfg, stepIdx) => {
                const card = document.createElement("div");
                card.style.cssText = "border: 1px solid #444; border-radius: 6px; margin-bottom: 8px; padding: 8px;" + (ucfg.active === false ? " opacity: 0.5;" : "");

                // Card header with active toggle and delete button
                const cardHeader = document.createElement("div");
                cardHeader.style.cssText = "display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;";

                const cardLeft = document.createElement("div");
                cardLeft.style.cssText = "display: flex; align-items: center; gap: 6px;";

                const activeCb = document.createElement("input");
                activeCb.type = "checkbox";
                activeCb.checked = ucfg.active !== false;
                activeCb.title = ucfg.active !== false ? "Step is ON — included in generation" : "Step is OFF — excluded from generation";
                activeCb.onchange = () => {
                    ucfg.active = activeCb.checked;
                    card.style.opacity = ucfg.active !== false ? "1" : "0.5";
                    node.saveState();
                };
                cardLeft.appendChild(activeCb);

                const cardTitle = document.createElement("span");
                cardTitle.style.cssText = "font-weight: bold; color: #cc99ff; font-size: 13px;";
                cardTitle.textContent = `Step ${stepIdx + 1}`;
                cardLeft.appendChild(cardTitle);

                // Repeat field (inline in header)
                const repeatLabel = document.createElement("span");
                repeatLabel.style.cssText = "font-size: 11px; color: #888; margin-left: 8px;";
                repeatLabel.textContent = "Repeat:";
                cardLeft.appendChild(repeatLabel);

                const repeatInput = document.createElement("input");
                repeatInput.type = "number";
                repeatInput.className = "cb-input";
                repeatInput.value = ucfg.repeat || 1;
                repeatInput.min = 1;
                repeatInput.max = 20;
                repeatInput.step = 1;
                repeatInput.style.cssText = "width: 50px; padding: 1px 4px; font-size: 11px;";
                repeatInput.onchange = () => { ucfg.repeat = Math.max(1, parseInt(repeatInput.value) || 1); node.saveState(); renderPipelines(); };
                cardLeft.appendChild(repeatInput);

                if (ucfg.repeat > 1) {
                    const repeatNote = document.createElement("span");
                    repeatNote.style.cssText = "font-size: 10px; color: #cc99ff;";
                    repeatNote.textContent = `(×${ucfg.repeat} feedback loop)`;
                    cardLeft.appendChild(repeatNote);
                }

                cardHeader.appendChild(cardLeft);

                if (pipeline.steps.length > 1) {
                    const delBtn = document.createElement("button");
                    delBtn.className = "cb-button";
                    delBtn.textContent = "✕";
                    delBtn.style.cssText = "background: #cc3333; padding: 2px 8px; font-size: 12px;";
                    delBtn.onclick = () => {
                        pipeline.steps.splice(stepIdx, 1);
                        node.saveState();
                        renderPipelines();
                    };
                    cardHeader.appendChild(delBtn);
                }
                card.appendChild(cardHeader);

                const grid = document.createElement("div");
                grid.className = "cb-flex-grid";

                // Mode select
                const modeSelect = document.createElement("select");
                modeSelect.className = "cb-select";
                modeSelect.innerHTML = `
                    <option value="hires_only" ${ucfg.mode === "hires_only" ? 'selected' : ''}>HiRes Fix Only</option>
                    <option value="model_only" ${ucfg.mode === "model_only" ? 'selected' : ''}>Model Upscale Only</option>
                    <option value="model_then_hires" ${ucfg.mode === "model_then_hires" ? 'selected' : ''}>Model Upscale \u2192 HiRes Fix</option>
                    <option value="seedvr2" ${ucfg.mode === "seedvr2" ? 'selected' : ''}>SeedVR2 Upscale</option>
                    <option value="florence2_hires" ${ucfg.mode === "florence2_hires" ? 'selected' : ''}>Florence2 Hi-Res Fix</option>
                `;
                modeSelect.onchange = () => {
                    ucfg.mode = modeSelect.value;
                    node.saveState();
                    renderPipelines();
                };
                grid.appendChild(createInputGroup("Mode", modeSelect));

                // Resize method select (applies to all modes)
                const resizeSelect = document.createElement("select");
                resizeSelect.className = "cb-select";
                const resizeMethods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"];
                resizeSelect.innerHTML = resizeMethods.map(m =>
                    `<option value="${m}" ${(ucfg.resize_method || "bilinear") === m ? 'selected' : ''}>${m.charAt(0).toUpperCase() + m.slice(1)}</option>`
                ).join("");
                resizeSelect.onchange = () => { ucfg.resize_method = resizeSelect.value; node.saveState(); };
                grid.appendChild(createInputGroup("Resize Method", resizeSelect));

                const showHires = ucfg.mode === "hires_only" || ucfg.mode === "model_then_hires" || ucfg.mode === "florence2_hires";
                const showModel = ucfg.mode === "model_only" || ucfg.mode === "model_then_hires";
                const showSeedVR2 = ucfg.mode === "seedvr2";
                const showFlorence2 = ucfg.mode === "florence2_hires";

                // --- HiRes fields (multi-value: ratios, denoise) ---
                if (showHires) {
                    // Upscale Ratios is irrelevant for florence2_hires (it inpaints a region,
                    // not the whole image; Target MP controls the internal pass size instead).
                    if (!showFlorence2) {
                        const ratiosInput = document.createElement("input");
                        ratiosInput.type = "text";
                        ratiosInput.className = "cb-input";
                        ratiosInput.value = ucfg.upscale_ratios || "1.5";
                        ratiosInput.placeholder = "1.2, 1.5, 2.0";
                        ratiosInput.onchange = () => { ucfg.upscale_ratios = ratiosInput.value; node.saveState(); };
                        grid.appendChild(createInputGroup("Upscale Ratios", ratiosInput));
                    }

                    // NOTE: renderUpscalingSection is a session-level section (no configArray param).
                    // Cannot use isLTXConfigArray() here — left as node.state.model_type guard intentionally.
                    if (node.state.model_type !== "ltx_video") {
                        const denoiseInput = document.createElement("input");
                        denoiseInput.type = "text";
                        denoiseInput.className = "cb-input";
                        denoiseInput.value = ucfg.hires_denoise || "0.3";
                        denoiseInput.placeholder = "0.2, 0.3, 0.5";
                        denoiseInput.onchange = () => { ucfg.hires_denoise = denoiseInput.value; node.saveState(); };
                        grid.appendChild(createInputGroup("HiRes Denoise", denoiseInput));
                    }

                    const stepsInput = document.createElement("input");
                    stepsInput.type = "number";
                    stepsInput.className = "cb-input";
                    stepsInput.value = ucfg.hires_steps || 0;
                    stepsInput.min = 0;
                    stepsInput.max = 150;
                    stepsInput.step = 1;
                    stepsInput.onchange = () => { ucfg.hires_steps = parseInt(stepsInput.value); node.saveState(); };
                    grid.appendChild(createInputGroup("HiRes Steps (0=same)", stepsInput));

                    // --- HiRes Tiled Sampling (tile the latent for KSampler to prevent OOM on large upscales) ---
                    // Hidden for florence2_hires — tiled sampling doesn't apply to its
                    // crop-then-paste pipeline (crops are too small to need tiling).
                    if (!showFlorence2) {
                        const tiledSamplingCb = document.createElement("input");
                        tiledSamplingCb.type = "checkbox";
                        tiledSamplingCb.checked = ucfg.hires_tiled_sampling || false;
                        tiledSamplingCb.onchange = () => { ucfg.hires_tiled_sampling = tiledSamplingCb.checked; node.saveState(); renderPipelines(); };
                        grid.appendChild(createInputGroup("HiRes Tiled Sampling", tiledSamplingCb));
                    }

                    if (ucfg.hires_tiled_sampling && !showFlorence2) {
                        const htWidthInput = document.createElement("input");
                        htWidthInput.type = "number";
                        htWidthInput.className = "cb-input";
                        htWidthInput.value = ucfg.hires_tile_width || 512;
                        htWidthInput.min = 128;
                        htWidthInput.max = 2048;
                        htWidthInput.step = 64;
                        htWidthInput.onchange = () => { ucfg.hires_tile_width = parseInt(htWidthInput.value); node.saveState(); };
                        grid.appendChild(createInputGroup("Tile Width", htWidthInput));

                        const htHeightInput = document.createElement("input");
                        htHeightInput.type = "number";
                        htHeightInput.className = "cb-input";
                        htHeightInput.value = ucfg.hires_tile_height || 512;
                        htHeightInput.min = 128;
                        htHeightInput.max = 2048;
                        htHeightInput.step = 64;
                        htHeightInput.onchange = () => { ucfg.hires_tile_height = parseInt(htHeightInput.value); node.saveState(); };
                        grid.appendChild(createInputGroup("Tile Height", htHeightInput));

                        const htBlurInput = document.createElement("input");
                        htBlurInput.type = "number";
                        htBlurInput.className = "cb-input";
                        htBlurInput.value = ucfg.hires_mask_blur || 8;
                        htBlurInput.min = 0;
                        htBlurInput.max = 64;
                        htBlurInput.step = 1;
                        htBlurInput.onchange = () => { ucfg.hires_mask_blur = parseInt(htBlurInput.value); node.saveState(); };
                        grid.appendChild(createInputGroup("Mask Blur", htBlurInput));

                        const htPaddingInput = document.createElement("input");
                        htPaddingInput.type = "number";
                        htPaddingInput.className = "cb-input";
                        htPaddingInput.value = ucfg.hires_tile_padding || 32;
                        htPaddingInput.min = 0;
                        htPaddingInput.max = 256;
                        htPaddingInput.step = 8;
                        htPaddingInput.onchange = () => { ucfg.hires_tile_padding = parseInt(htPaddingInput.value); node.saveState(); };
                        grid.appendChild(createInputGroup("Tile Padding", htPaddingInput));

                        const htUniformCb = document.createElement("input");
                        htUniformCb.type = "checkbox";
                        htUniformCb.checked = ucfg.hires_force_uniform_tiles !== false;
                        htUniformCb.onchange = () => { ucfg.hires_force_uniform_tiles = htUniformCb.checked; node.saveState(); };
                        grid.appendChild(createInputGroup("Force Uniform Tiles", htUniformCb));
                    }

                    // --- Tiled VAE Decode (separate from tiled sampling) ---
                    const tiledCb = document.createElement("input");
                    tiledCb.type = "checkbox";
                    tiledCb.checked = ucfg.tiled_vae || false;
                    tiledCb.onchange = () => { ucfg.tiled_vae = tiledCb.checked; node.saveState(); renderPipelines(); };
                    grid.appendChild(createInputGroup("Tiled VAE Decode", tiledCb));

                    if (ucfg.tiled_vae) {
                        const tileSizeInput = document.createElement("input");
                        tileSizeInput.type = "number";
                        tileSizeInput.className = "cb-input";
                        tileSizeInput.value = ucfg.tile_size || 512;
                        tileSizeInput.min = 128;
                        tileSizeInput.max = 1024;
                        tileSizeInput.step = 64;
                        tileSizeInput.onchange = () => { ucfg.tile_size = parseInt(tileSizeInput.value); node.saveState(); };
                        grid.appendChild(createInputGroup("Tile Size", tileSizeInput));

                        const tileOverlapInput = document.createElement("input");
                        tileOverlapInput.type = "number";
                        tileOverlapInput.className = "cb-input";
                        tileOverlapInput.value = ucfg.tile_overlap || 64;
                        tileOverlapInput.min = 0;
                        tileOverlapInput.max = 512;
                        tileOverlapInput.step = 8;
                        tileOverlapInput.onchange = () => { ucfg.tile_overlap = parseInt(tileOverlapInput.value); node.saveState(); };
                        grid.appendChild(createInputGroup("Tile Overlap", tileOverlapInput));

                        const temporalSizeInput = document.createElement("input");
                        temporalSizeInput.type = "number";
                        temporalSizeInput.className = "cb-input";
                        temporalSizeInput.value = ucfg.temporal_size || 512;
                        temporalSizeInput.min = 128;
                        temporalSizeInput.max = 1024;
                        temporalSizeInput.step = 64;
                        temporalSizeInput.onchange = () => { ucfg.temporal_size = parseInt(temporalSizeInput.value); node.saveState(); };
                        grid.appendChild(createInputGroup("Temporal Size", temporalSizeInput));

                        const temporalOverlapInput = document.createElement("input");
                        temporalOverlapInput.type = "number";
                        temporalOverlapInput.className = "cb-input";
                        temporalOverlapInput.value = ucfg.temporal_overlap || 64;
                        temporalOverlapInput.min = 0;
                        temporalOverlapInput.max = 512;
                        temporalOverlapInput.step = 8;
                        temporalOverlapInput.onchange = () => { ucfg.temporal_overlap = parseInt(temporalOverlapInput.value); node.saveState(); };
                        grid.appendChild(createInputGroup("Temporal Overlap", temporalOverlapInput));
                    }
                }

                // --- Model fields (multi-add searchable) ---
                if (showModel) {
                    // Model upscale multiplier — only for model_only mode
                    // In model_then_hires, the Upscale Ratios field already controls final size
                    if (ucfg.mode === "model_only") {
                        const sizeInput = document.createElement("input");
                        sizeInput.type = "text";
                        sizeInput.className = "cb-input";
                        sizeInput.value = ucfg.upscale_size || "2.0";
                        sizeInput.placeholder = "2.0";
                        sizeInput.style.width = "60px";
                        sizeInput.title = "Resizes the model's output to (original size × this value). Upscale models have a fixed native scale (e.g. 4x). This setting lets you control the final output size — set to 2.0 to get 2x output even from a 4x model.";
                        sizeInput.onchange = () => { ucfg.upscale_size = sizeInput.value; node.saveState(); };

                        const sizeGroup = createInputGroup("Output Size Multiplier", sizeInput);
                        // Add tooltip icon after label
                        const sizeTooltip = document.createElement("span");
                        sizeTooltip.textContent = "?";
                        sizeTooltip.style.cssText = "display: inline-flex; align-items: center; justify-content: center; width: 14px; height: 14px; border-radius: 50%; background: #555; color: #fff; font-size: 10px; font-weight: bold; cursor: help; margin-left: 4px; flex-shrink: 0;";
                        sizeTooltip.title = "Upscale models output at a fixed native scale (e.g. 4x for most ESRGAN models). This setting resizes that output to (original size × multiplier). Example: A 512px image with a 4x model outputs 2048px. Setting this to 2.0 resizes it down to 1024px.";
                        const sizeLabel = sizeGroup.querySelector("label");
                        if (sizeLabel) { sizeLabel.style.display = "inline-flex"; sizeLabel.style.alignItems = "center"; sizeLabel.appendChild(sizeTooltip); }
                        grid.appendChild(sizeGroup);
                    }

                    // Upscale models list with add/remove
                    const modelsContainer = document.createElement("div");
                    const noModels = !ucfg.upscale_models || ucfg.upscale_models.length === 0;

                    // Red outline when no model is selected
                    if (noModels) {
                        modelsContainer.style.cssText = "border: 2px solid #ff3333; border-radius: 4px; padding: 4px;";
                    }

                    const modelsLabel = document.createElement("div");
                    modelsLabel.style.cssText = "font-size: 12px; color: " + (noModels ? "#ff3333" : "#aaa") + "; margin-bottom: 4px;";
                    modelsLabel.textContent = `Upscale Models (${(ucfg.upscale_models || []).length})` + (noModels ? " — Select a model!" : "");
                    modelsContainer.appendChild(modelsLabel);

                    // Existing model chips
                    (ucfg.upscale_models || []).forEach((modelName, mIdx) => {
                        const chip = document.createElement("div");
                        chip.style.cssText = "display: inline-flex; align-items: center; gap: 4px; background: #333; border-radius: 4px; padding: 2px 8px; margin: 2px; font-size: 12px;";
                        chip.textContent = modelName;
                        const removeBtn = document.createElement("span");
                        removeBtn.textContent = "✕";
                        removeBtn.style.cssText = "cursor: pointer; color: #cc3333; margin-left: 4px;";
                        removeBtn.onclick = () => {
                            ucfg.upscale_models.splice(mIdx, 1);
                            node.saveState();
                            renderPipelines();
                        };
                        chip.appendChild(removeBtn);
                        modelsContainer.appendChild(chip);
                    });

                    // Add model searchable select
                    const addSelect = createSearchableSelect(
                        modelLists?.upscaleModels || [],
                        "",
                        (value) => {
                            if (value && !ucfg.upscale_models.includes(value)) {
                                ucfg.upscale_models.push(value);
                                node.saveState();
                                renderPipelines();
                            }
                        },
                        "Search upscale models..."
                    );
                    modelsContainer.appendChild(addSelect);
                    grid.appendChild(modelsContainer);
                }

                // --- SeedVR2 fields ---
                if (showSeedVR2) {
                    if (!ucfg.seedvr2) ucfg.seedvr2 = createDefaultSeedVR2Options();
                    const sv = ucfg.seedvr2;

                    // DiT Model dropdown
                    const ditModels = [
                        "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                        "seedvr2_ema_3b_fp16.safetensors",
                        "seedvr2_ema_3b-Q4_K_M.gguf",
                        "seedvr2_ema_3b-Q8_0.gguf",
                        "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
                        "seedvr2_ema_7b_fp16.safetensors",
                        "seedvr2_ema_7b-Q4_K_M.gguf",
                        "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
                        "seedvr2_ema_7b_sharp_fp16.safetensors",
                        "seedvr2_ema_7b_sharp-Q4_K_M.gguf"
                    ];
                    const ditSelect = document.createElement("select");
                    ditSelect.className = "cb-select";
                    ditModels.forEach(m => {
                        const opt = document.createElement("option");
                        opt.value = m; opt.textContent = m.replace(".safetensors", "").replace(".gguf", " (GGUF)");
                        if (sv.dit_model === m) opt.selected = true;
                        ditSelect.appendChild(opt);
                    });
                    ditSelect.onchange = () => { sv.dit_model = ditSelect.value; node.saveState(); };
                    grid.appendChild(createInputGroup("DiT Model", ditSelect));

                    // Resolution
                    const resInput = document.createElement("input");
                    resInput.type = "number"; resInput.className = "cb-input";
                    resInput.value = sv.resolution || 1080; resInput.min = 16; resInput.max = 16384; resInput.step = 2;
                    resInput.onchange = () => { sv.resolution = parseInt(resInput.value) || 1080; node.saveState(); };
                    grid.appendChild(createInputGroup("Target Resolution", resInput));

                    // Max Resolution
                    const maxResInput = document.createElement("input");
                    maxResInput.type = "number"; maxResInput.className = "cb-input";
                    maxResInput.value = sv.max_resolution || 0; maxResInput.min = 0; maxResInput.max = 16384; maxResInput.step = 2;
                    maxResInput.title = "Maximum any-edge resolution (0 = no limit)";
                    maxResInput.onchange = () => { sv.max_resolution = parseInt(maxResInput.value) || 0; node.saveState(); };
                    grid.appendChild(createInputGroup("Max Resolution (0=none)", maxResInput));

                    // Seed
                    const seedInput = document.createElement("input");
                    seedInput.type = "number"; seedInput.className = "cb-input";
                    seedInput.value = sv.seed || 42; seedInput.min = 0;
                    seedInput.onchange = () => { sv.seed = parseInt(seedInput.value) || 42; node.saveState(); };
                    grid.appendChild(createInputGroup("Seed", seedInput));

                    // Color Correction
                    const ccSelect = document.createElement("select");
                    ccSelect.className = "cb-select";
                    ["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"].forEach(cc => {
                        const opt = document.createElement("option");
                        opt.value = cc; opt.textContent = cc;
                        if (sv.color_correction === cc) opt.selected = true;
                        ccSelect.appendChild(opt);
                    });
                    ccSelect.onchange = () => { sv.color_correction = ccSelect.value; node.saveState(); };
                    grid.appendChild(createInputGroup("Color Correction", ccSelect));

                    // Noise scales
                    const inNoiseInput = document.createElement("input");
                    inNoiseInput.type = "number"; inNoiseInput.className = "cb-input";
                    inNoiseInput.value = sv.input_noise_scale || 0; inNoiseInput.min = 0; inNoiseInput.max = 1; inNoiseInput.step = 0.001;
                    inNoiseInput.onchange = () => { sv.input_noise_scale = parseFloat(inNoiseInput.value) || 0; node.saveState(); };
                    grid.appendChild(createInputGroup("Input Noise Scale", inNoiseInput));

                    const latNoiseInput = document.createElement("input");
                    latNoiseInput.type = "number"; latNoiseInput.className = "cb-input";
                    latNoiseInput.value = sv.latent_noise_scale || 0; latNoiseInput.min = 0; latNoiseInput.max = 1; latNoiseInput.step = 0.001;
                    latNoiseInput.onchange = () => { sv.latent_noise_scale = parseFloat(latNoiseInput.value) || 0; node.saveState(); };
                    grid.appendChild(createInputGroup("Latent Noise Scale", latNoiseInput));

                    // Batch Size
                    const batchInput = document.createElement("input");
                    batchInput.type = "number"; batchInput.className = "cb-input";
                    batchInput.value = sv.batch_size || 1; batchInput.min = 1; batchInput.max = 16384; batchInput.step = 4;
                    batchInput.onchange = () => { sv.batch_size = parseInt(batchInput.value) || 1; node.saveState(); };
                    grid.appendChild(createInputGroup("Batch Size", batchInput));

                    // --- VRAM Optimization ---
                    const vramLabel = document.createElement("div");
                    vramLabel.style.cssText = "width: 100%; font-size: 11px; color: #cc99ff; font-weight: bold; margin-top: 4px; border-top: 1px solid #444; padding-top: 4px;";
                    vramLabel.textContent = "VRAM Optimization";
                    grid.appendChild(vramLabel);

                    // Blocks to Swap
                    const blocksInput = document.createElement("input");
                    blocksInput.type = "number"; blocksInput.className = "cb-input";
                    blocksInput.value = sv.blocks_to_swap || 0; blocksInput.min = 0; blocksInput.max = 36; blocksInput.step = 1;
                    blocksInput.title = "Number of transformer blocks to offload to CPU (saves VRAM, slower)";
                    blocksInput.onchange = () => { sv.blocks_to_swap = parseInt(blocksInput.value) || 0; node.saveState(); };
                    grid.appendChild(createInputGroup("Blocks to Swap", blocksInput));

                    // Attention Mode
                    const attSelect = document.createElement("select");
                    attSelect.className = "cb-select";
                    ["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"].forEach(a => {
                        const opt = document.createElement("option");
                        opt.value = a; opt.textContent = a;
                        if (sv.attention_mode === a) opt.selected = true;
                        attSelect.appendChild(opt);
                    });
                    attSelect.onchange = () => { sv.attention_mode = attSelect.value; node.saveState(); };
                    grid.appendChild(createInputGroup("Attention Mode", attSelect));

                    // DiT Offload Device
                    const offloadSelect = document.createElement("select");
                    offloadSelect.className = "cb-select";
                    ["none", "cpu"].forEach(d => {
                        const opt = document.createElement("option");
                        opt.value = d; opt.textContent = d;
                        if (sv.offload_device === d) opt.selected = true;
                        offloadSelect.appendChild(opt);
                    });
                    offloadSelect.onchange = () => { sv.offload_device = offloadSelect.value; node.saveState(); };
                    grid.appendChild(createInputGroup("DiT Offload Device", offloadSelect));

                    // Cache DiT Model
                    const cacheCb = document.createElement("input");
                    cacheCb.type = "checkbox"; cacheCb.checked = sv.cache_model || false;
                    cacheCb.onchange = () => { sv.cache_model = cacheCb.checked; node.saveState(); };
                    grid.appendChild(createInputGroup("Cache DiT Model", cacheCb));

                    // --- VAE Tiling ---
                    const vaeLabel = document.createElement("div");
                    vaeLabel.style.cssText = "width: 100%; font-size: 11px; color: #cc99ff; font-weight: bold; margin-top: 4px; border-top: 1px solid #444; padding-top: 4px;";
                    vaeLabel.textContent = "VAE Tiling";
                    grid.appendChild(vaeLabel);

                    // Encode Tiled
                    const encTiledCb = document.createElement("input");
                    encTiledCb.type = "checkbox"; encTiledCb.checked = sv.encode_tiled || false;
                    encTiledCb.onchange = () => { sv.encode_tiled = encTiledCb.checked; node.saveState(); renderPipelines(); };
                    grid.appendChild(createInputGroup("Encode Tiled", encTiledCb));

                    if (sv.encode_tiled) {
                        const encTsInput = document.createElement("input");
                        encTsInput.type = "number"; encTsInput.className = "cb-input";
                        encTsInput.value = sv.encode_tile_size || 1024; encTsInput.min = 64; encTsInput.step = 32;
                        encTsInput.onchange = () => { sv.encode_tile_size = parseInt(encTsInput.value) || 1024; node.saveState(); };
                        grid.appendChild(createInputGroup("Encode Tile Size", encTsInput));

                        const encToInput = document.createElement("input");
                        encToInput.type = "number"; encToInput.className = "cb-input";
                        encToInput.value = sv.encode_tile_overlap || 128; encToInput.min = 0; encToInput.step = 32;
                        encToInput.onchange = () => { sv.encode_tile_overlap = parseInt(encToInput.value) || 128; node.saveState(); };
                        grid.appendChild(createInputGroup("Encode Tile Overlap", encToInput));
                    }

                    // Decode Tiled
                    const decTiledCb = document.createElement("input");
                    decTiledCb.type = "checkbox"; decTiledCb.checked = sv.decode_tiled || false;
                    decTiledCb.onchange = () => { sv.decode_tiled = decTiledCb.checked; node.saveState(); renderPipelines(); };
                    grid.appendChild(createInputGroup("Decode Tiled", decTiledCb));

                    if (sv.decode_tiled) {
                        const decTsInput = document.createElement("input");
                        decTsInput.type = "number"; decTsInput.className = "cb-input";
                        decTsInput.value = sv.decode_tile_size || 1024; decTsInput.min = 64; decTsInput.step = 32;
                        decTsInput.onchange = () => { sv.decode_tile_size = parseInt(decTsInput.value) || 1024; node.saveState(); };
                        grid.appendChild(createInputGroup("Decode Tile Size", decTsInput));

                        const decToInput = document.createElement("input");
                        decToInput.type = "number"; decToInput.className = "cb-input";
                        decToInput.value = sv.decode_tile_overlap || 128; decToInput.min = 0; decToInput.step = 32;
                        decToInput.onchange = () => { sv.decode_tile_overlap = parseInt(decToInput.value) || 128; node.saveState(); };
                        grid.appendChild(createInputGroup("Decode Tile Overlap", decToInput));
                    }

                    // VAE Offload Device
                    const vaeOffloadSelect = document.createElement("select");
                    vaeOffloadSelect.className = "cb-select";
                    ["none", "cpu"].forEach(d => {
                        const opt = document.createElement("option");
                        opt.value = d; opt.textContent = d;
                        if (sv.vae_offload_device === d) opt.selected = true;
                        vaeOffloadSelect.appendChild(opt);
                    });
                    vaeOffloadSelect.onchange = () => { sv.vae_offload_device = vaeOffloadSelect.value; node.saveState(); };
                    grid.appendChild(createInputGroup("VAE Offload Device", vaeOffloadSelect));

                    // Cache VAE Model
                    const vaeCacheCb = document.createElement("input");
                    vaeCacheCb.type = "checkbox"; vaeCacheCb.checked = sv.vae_cache_model || false;
                    vaeCacheCb.onchange = () => { sv.vae_cache_model = vaeCacheCb.checked; node.saveState(); };
                    grid.appendChild(createInputGroup("Cache VAE Model", vaeCacheCb));
                }

                // --- Florence2 Hi-Res Fix sub-panel ---
                if (showFlorence2) {
                    if (!ucfg.florence2) ucfg.florence2 = createDefaultFlorence2Options();
                    const f2 = ucfg.florence2;

                    // --- Group A: Florence2 detection ---
                    const f2ModelSelect = document.createElement("select");
                    f2ModelSelect.className = "cb-select";
                    f2ModelSelect.title = "Florence2 model for segmentation. base/large work for referring_expression_segmentation; ft variants slightly worse.";
                    [
                        "microsoft/Florence-2-base",
                        "microsoft/Florence-2-base-ft",
                        "microsoft/Florence-2-large",
                        "microsoft/Florence-2-large-ft",
                    ].forEach(name => {
                        const opt = document.createElement("option");
                        opt.value = name; opt.textContent = name;
                        if (f2.model === name) opt.selected = true;
                        f2ModelSelect.appendChild(opt);
                    });
                    f2ModelSelect.onchange = () => { f2.model = f2ModelSelect.value; node.saveState(); };
                    grid.appendChild(createInputGroup("Florence2 Model", f2ModelSelect));

                    const f2TextInput = document.createElement("input");
                    f2TextInput.type = "text"; f2TextInput.className = "cb-input";
                    f2TextInput.value = f2.text_input || "face";
                    f2TextInput.placeholder = "face, head, hands, eyes...";
                    f2TextInput.title = "What to detect. Try: face, head, hands, eyes, person, clothing.";
                    f2TextInput.onchange = () => { f2.text_input = f2TextInput.value; node.saveState(); };
                    grid.appendChild(createInputGroup("Detect What?", f2TextInput));

                    const f2MaskSelInput = document.createElement("input");
                    f2MaskSelInput.type = "text"; f2MaskSelInput.className = "cb-input";
                    f2MaskSelInput.value = f2.output_mask_select || "";
                    f2MaskSelInput.placeholder = '(empty = all)';
                    f2MaskSelInput.title = "Empty = all detections. '0' = primary/largest. '0,2' = pick specific indices.";
                    f2MaskSelInput.onchange = () => { f2.output_mask_select = f2MaskSelInput.value; node.saveState(); };
                    grid.appendChild(createInputGroup("Output Mask Select", f2MaskSelInput));

                    const f2MaxTokInput = document.createElement("input");
                    f2MaxTokInput.type = "number"; f2MaxTokInput.className = "cb-input";
                    f2MaxTokInput.value = f2.max_new_tokens ?? 1024;
                    f2MaxTokInput.min = 64; f2MaxTokInput.max = 2048; f2MaxTokInput.step = 32;
                    f2MaxTokInput.title = "Florence2 beam-search budget. Default 1024 (kijai stock). Model emits EOS when the polygon is complete (typically ~50-100 tokens for a face) — the cap is just a ceiling. Lowering too far causes truncated polygons (mask doesn't cover the full detected region). Only reduce if you're VRAM-constrained AND comfortable with potentially partial polygons.";
                    f2MaxTokInput.onchange = () => { f2.max_new_tokens = parseInt(f2MaxTokInput.value, 10); node.saveState(); };
                    grid.appendChild(createInputGroup("Max New Tokens", f2MaxTokInput));

                    const f2InMpInput = document.createElement("input");
                    f2InMpInput.type = "number"; f2InMpInput.className = "cb-input";
                    f2InMpInput.value = f2.florence2_input_mp != null ? f2.florence2_input_mp : 0.5;
                    f2InMpInput.min = 0; f2InMpInput.max = 4.0; f2InMpInput.step = 0.01;
                    f2InMpInput.title = "Resize source image to this MP BEFORE Florence2 detection. Florence2's vision encoder activations scale with input dims — 0.5 MP cuts encoder VRAM ~4x with negligible detection-accuracy loss. The mask gets scaled back up to source resolution after detection, so the crop/inpaint/paste-back still runs at full quality. Set 0 to disable. Only takes effect when source > this MP.";
                    f2InMpInput.onchange = () => { f2.florence2_input_mp = parseFloat(f2InMpInput.value); node.saveState(); };
                    grid.appendChild(createInputGroup("Florence2 Input MP", f2InMpInput));

                    // --- Group B: Crop & resize ---
                    const f2MpInput = document.createElement("input");
                    f2MpInput.type = "number"; f2MpInput.className = "cb-input";
                    f2MpInput.value = f2.target_megapixels ?? 1.0;
                    f2MpInput.min = 0.25; f2MpInput.max = 4.0; f2MpInput.step = 0.25;
                    f2MpInput.title = "Resize cropped region to this many megapixels before the hi-res pass. 1.0 ≈ 1024×1024.";
                    f2MpInput.onchange = () => { f2.target_megapixels = parseFloat(f2MpInput.value); node.saveState(); };
                    grid.appendChild(createInputGroup("Hi-Res Target (MP)", f2MpInput));

                    const f2PadInput = document.createElement("input");
                    f2PadInput.type = "number"; f2PadInput.className = "cb-input";
                    f2PadInput.value = f2.crop_padding ?? 64;
                    f2PadInput.min = 0; f2PadInput.max = 256; f2PadInput.step = 8;
                    f2PadInput.title = "Padding around detected polygon. Higher = more blending context.";
                    f2PadInput.onchange = () => { f2.crop_padding = parseInt(f2PadInput.value, 10); node.saveState(); };
                    grid.appendChild(createInputGroup("Crop Padding (px)", f2PadInput));

                    // Min/Max Crop Resolution removed from UI 2026-05-18 — they were
                    // forcing square-ish crops and could push the crop window away from
                    // Florence2's detected region when bbox was near an image edge.
                    // Florence2's polygon + grow_expand + crop_padding now fully determine
                    // the crop region. The python helper still accepts the params; defaults
                    // are unconstrained (min=0, max=very large).

                    // --- Group C: Mask shaping ---
                    const f2GrowInput = document.createElement("input");
                    f2GrowInput.type = "number"; f2GrowInput.className = "cb-input";
                    f2GrowInput.value = f2.grow_expand ?? 32;
                    f2GrowInput.min = -64; f2GrowInput.max = 256; f2GrowInput.step = 1;
                    f2GrowInput.title = "GrowMask expand. Negative shrinks. Adds pixels to polygon edges.";
                    f2GrowInput.onchange = () => { f2.grow_expand = parseInt(f2GrowInput.value, 10); node.saveState(); };
                    grid.appendChild(createInputGroup("Grow Mask (px)", f2GrowInput));

                    ["left", "top", "right", "bottom"].forEach(side => {
                        const inp = document.createElement("input");
                        inp.type = "number"; inp.className = "cb-input";
                        const key = `feather_${side}`;
                        inp.value = f2[key] ?? 128;
                        inp.min = 0; inp.max = 256; inp.step = 1;
                        inp.title = `FeatherMask ${side}-side alpha falloff in pixels.`;
                        inp.onchange = () => { f2[key] = parseInt(inp.value, 10); node.saveState(); };
                        grid.appendChild(createInputGroup(`Feather ${side.charAt(0).toUpperCase() + side.slice(1)} (px)`, inp));
                    });

                    // --- Group E: Model/LoRA source ---
                    const f2SourceSelect = document.createElement("select");
                    f2SourceSelect.className = "cb-select";
                    f2SourceSelect.title = "'From manifest' uses each image's original model/LoRA. 'From this Builder config' uses the same model the upscale session loaded.";
                    [
                        ["from_manifest", "From manifest (per-image)"],
                        ["from_builder", "From this Builder config"],
                    ].forEach(([val, lbl]) => {
                        const opt = document.createElement("option");
                        opt.value = val; opt.textContent = lbl;
                        if (f2.model_source === val) opt.selected = true;
                        f2SourceSelect.appendChild(opt);
                    });
                    f2SourceSelect.onchange = () => { f2.model_source = f2SourceSelect.value; node.saveState(); };
                    grid.appendChild(createInputGroup("Model/LoRA Source", f2SourceSelect));

                    // --- Group F: No-detection ---
                    const f2NoDetSelect = document.createElement("select");
                    f2NoDetSelect.className = "cb-select";
                    f2NoDetSelect.title = "What to do when Florence2 finds nothing in the image.";
                    const optSkip = document.createElement("option");
                    optSkip.value = "skip"; optSkip.textContent = "Skip + log";
                    if ((f2.on_no_detection || "skip") === "skip") optSkip.selected = true;
                    f2NoDetSelect.appendChild(optSkip);
                    f2NoDetSelect.onchange = () => { f2.on_no_detection = f2NoDetSelect.value; node.saveState(); };
                    grid.appendChild(createInputGroup("If No Detection", f2NoDetSelect));
                }

                card.appendChild(grid);

                // Iteration count for this step
                const countDisplay = document.createElement("div");
                countDisplay.style.cssText = "color: #00cc00; font-family: monospace; font-size: 11px; margin-top: 4px;";
                const ratios = (ucfg.upscale_ratios || "1.5").split(",").map(s => s.trim()).filter(s => s).length;
                const denoises = (ucfg.hires_denoise || "0.3").split(",").map(s => s.trim()).filter(s => s).length;
                const models = Math.max(1, (ucfg.upscale_models || []).length);
                let combos = 1;
                if (showSeedVR2) combos = 1; // SeedVR2 = 1 output per input
                else if (showFlorence2) combos = 1; // Florence2 = 1 output per input (uses hires_denoise as a single value, not array)
                else if (showHires) combos *= ratios * denoises;
                if (showModel) combos *= models;
                const repeatCount = ucfg.repeat || 1;
                countDisplay.textContent = `⏱️ ${combos} upscale combination(s) per base image` + (repeatCount > 1 ? ` × ${repeatCount} repeats` : "");
                card.appendChild(countDisplay);

                pipeBody.appendChild(card);
            });

            // Add Step button inside pipeline
            const addStepBtn = document.createElement("button");
            addStepBtn.className = "cb-button";
            addStepBtn.textContent = "➕ Add Step";
            addStepBtn.style.cssText = "margin-top: 4px; border-left: 4px solid #9966cc;";
            addStepBtn.onclick = () => {
                pipeline.steps.push(createDefaultStep());
                node.saveState();
                renderPipelines();
            };
            pipeBody.appendChild(addStepBtn);

            pipeCard.appendChild(pipeBody);
            pipelinesContainer.appendChild(pipeCard);
        });

        // Add Pipeline button
        const addPipeBtn = document.createElement("button");
        addPipeBtn.className = "cb-button";
        addPipeBtn.textContent = "➕ Add Pipeline";
        addPipeBtn.style.cssText = "margin-top: 4px; border-left: 4px solid #cc99ff;";
        addPipeBtn.onclick = () => {
            ups.pipelines.push({
                active: true,
                name: `Pipeline ${ups.pipelines.length + 1}`,
                steps: [createDefaultStep()]
            });
            node.saveState();
            renderPipelines();
        };
        pipelinesContainer.appendChild(addPipeBtn);
    }

    renderPipelines();
    section.appendChild(body);
    container.appendChild(section);
}

// ============================================================================
// GPU COOLDOWN SETTINGS (Session-level, applies to all configs)
// ============================================================================
export function renderCooldownSection(node, container) {
    if (!node.state.cooldown) {
        node.state.cooldown = {
            enabled: false,
            seconds: 5,
            every_n: 1,
            clear_vram: false
        };
    }
    const cd = node.state.cooldown;

    const section = document.createElement("div");
    section.className = "cb-section";

    // Header with enable toggle
    const header = document.createElement("div");
    header.className = "cb-section-header";
    header.style.cssText = "display: flex; align-items: center; gap: 8px; padding: 8px 12px; cursor: pointer;";

    const enableCb = document.createElement("input");
    enableCb.type = "checkbox";
    enableCb.checked = cd.enabled;
    enableCb.onclick = (e) => e.stopPropagation();
    enableCb.onchange = () => {
        cd.enabled = enableCb.checked;
        body.style.display = enableCb.checked ? "block" : "none";
        node.saveState();
    };

    const title = document.createElement("span");
    title.textContent = "❄️ GPU Cooldown Breaks";
    title.style.cssText = "font-weight: bold; color: #66ccff; font-size: 14px;";

    header.appendChild(enableCb);
    header.appendChild(title);
    section.appendChild(header);

    // Body
    const body = document.createElement("div");
    body.style.display = cd.enabled ? "block" : "none";
    body.style.padding = "8px 12px";

    const grid = document.createElement("div");
    grid.className = "cb-flex-grid";

    const secondsInput = document.createElement("input");
    secondsInput.type = "number";
    secondsInput.className = "cb-input";
    secondsInput.value = cd.seconds;
    secondsInput.min = 1;
    secondsInput.max = 300;
    secondsInput.step = 1;
    secondsInput.onchange = () => { cd.seconds = parseInt(secondsInput.value); node.saveState(); };
    grid.appendChild(createInputGroup("Cooldown Seconds", secondsInput));

    const everyNInput = document.createElement("input");
    everyNInput.type = "number";
    everyNInput.className = "cb-input";
    everyNInput.value = cd.every_n;
    everyNInput.min = 1;
    everyNInput.max = 100;
    everyNInput.step = 1;
    everyNInput.onchange = () => { cd.every_n = parseInt(everyNInput.value); node.saveState(); };
    grid.appendChild(createInputGroup("Every N Generations", everyNInput));

    const vramCb = document.createElement("input");
    vramCb.type = "checkbox";
    vramCb.checked = cd.clear_vram;
    vramCb.onchange = () => {
        cd.clear_vram = vramCb.checked;
        node.saveState();
    };
    grid.appendChild(createInputGroup("Clear VRAM During Cooldown", vramCb));

    body.appendChild(grid);
    section.appendChild(body);
    container.appendChild(section);
}

// --- MAIN RENDER UI FUNCTION ---

// ============================================================================
// RUN SETTINGS (Start At Job #, etc.)
// ============================================================================
export function renderRunSettingsSection(node, container) {
    if (node.state.start_at_job === undefined) node.state.start_at_job = 0;
    if (node.state.image_format === undefined) node.state.image_format = "webp";
    if (node.state.overwrite_existing === undefined) node.state.overwrite_existing = false;
    if (node.state.flush_batch_every === undefined) node.state.flush_batch_every = 1;
    if (node.state.lora_triggerwords_mode === undefined) node.state.lora_triggerwords_mode = "None";
    if (node.state.save_conditioning_cache_to_file === undefined) node.state.save_conditioning_cache_to_file = false;
    if (node.state.enable_model_cache === undefined) node.state.enable_model_cache = false;
    if (node.state.vae_batch_size === undefined) node.state.vae_batch_size = 1;
    if (node.state.add_random_seeds_to_gens === undefined) node.state.add_random_seeds_to_gens = 0;

    const section = document.createElement("div");
    section.className = "cb-section full-width";
    section.id = "cb-sec-runsettings";

    const header = document.createElement("div");
    header.className = "cb-section-header";
    header.textContent = "▸ Run Settings";
    header.style.cursor = "pointer";

    const content = document.createElement("div");
    content.className = "cb-section-content";
    content.style.display = "none";

    header.onclick = () => {
        const isOpen = content.style.display !== "none";
        content.style.display = isOpen ? "none" : "block";
        header.textContent = (isOpen ? "▸" : "▾") + " Run Settings";
    };

    // Start At Job # input
    const jobDiv = document.createElement("div");
    jobDiv.style.cssText = "margin-bottom: 8px; display: flex; align-items: center; gap: 8px;";
    const jobLabel = document.createElement("label");
    jobLabel.textContent = "Start At Job #:";
    jobLabel.style.cssText = "font-size: 12px; color: #ccc; white-space: nowrap;";
    const jobInput = document.createElement("input");
    jobInput.type = "number";
    jobInput.min = "0";
    jobInput.value = node.state.start_at_job || 0;
    jobInput.style.cssText = "width: 70px; background: #1a1a1a; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 4px 6px; font-size: 12px;";
    jobInput.onchange = () => {
        node.state.start_at_job = parseInt(jobInput.value) || 0;
        node.saveState();
    };
    const jobDesc = document.createElement("div");
    jobDesc.style.cssText = "font-size: 9px; color: #666;";
    jobDesc.textContent = "Skip to this job number (0 = start from beginning). Useful for resuming at a specific point.";
    jobDiv.appendChild(jobLabel);
    jobDiv.appendChild(jobInput);
    content.appendChild(jobDiv);
    content.appendChild(jobDesc);

    // Image Format dropdown
    const fmtDiv = document.createElement("div");
    fmtDiv.style.cssText = "margin-top: 10px; display: flex; align-items: center; gap: 8px;";
    const fmtLabel = document.createElement("label");
    fmtLabel.textContent = "Image Save Format:";
    fmtLabel.style.cssText = "font-size: 12px; color: #ccc; white-space: nowrap;";
    const fmtSelect = document.createElement("select");
    fmtSelect.style.cssText = "background: #1a1a1a; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 4px 6px; font-size: 12px;";
    [["webp", "WebP (default, smallest)"], ["png", "PNG (lossless)"], ["jpg", "JPG (small, lossy)"]].forEach(([val, label]) => {
        const opt = document.createElement("option");
        opt.value = val;
        opt.textContent = label;
        if (val === (node.state.image_format || "webp")) opt.selected = true;
        fmtSelect.appendChild(opt);
    });
    fmtSelect.onchange = () => {
        node.state.image_format = fmtSelect.value;
        node.saveState();
    };
    const fmtDesc = document.createElement("div");
    fmtDesc.style.cssText = "font-size: 9px; color: #666; margin-top: 2px;";
    fmtDesc.textContent = "File format for saved images. PNG is lossless. JPG saves at quality 95.";
    fmtDiv.appendChild(fmtLabel);
    fmtDiv.appendChild(fmtSelect);
    content.appendChild(fmtDiv);
    content.appendChild(fmtDesc);

    // Helper: renders a labeled row with a "?" tooltip icon, binding to node.state[stateKey].
    function _addRunSetting(content, opts) {
        // opts: { stateKey, label, tooltip, kind: "bool"|"int"|"select", options?, min?, max?, defaultValue }
        const row = document.createElement("div");
        row.style.cssText = "margin-top: 10px; display: flex; align-items: center; gap: 8px;";

        const label = document.createElement("label");
        label.textContent = opts.label + ":";
        label.style.cssText = "font-size: 12px; color: #ccc; white-space: nowrap;";

        const help = document.createElement("span");
        help.textContent = "?";
        help.title = opts.tooltip;
        help.style.cssText = "display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; border-radius: 50%; background: #2a2a2a; color: #8af; font-size: 10px; font-weight: bold; cursor: help; user-select: none;";

        let input;
        if (opts.kind === "bool") {
            input = document.createElement("input");
            input.type = "checkbox";
            input.checked = !!node.state[opts.stateKey];
            input.style.cssText = "transform: scale(1.2); cursor: pointer;";
            input.onchange = () => { node.state[opts.stateKey] = input.checked; node.saveState(); };
        } else if (opts.kind === "int") {
            input = document.createElement("input");
            input.type = "number";
            if (opts.min !== undefined) input.min = opts.min;
            if (opts.max !== undefined) input.max = opts.max;
            input.value = node.state[opts.stateKey] !== undefined ? node.state[opts.stateKey] : opts.defaultValue;
            input.style.cssText = "width: 70px; background: #1a1a1a; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 4px 6px; font-size: 12px;";
            input.onchange = () => {
                const v = parseInt(input.value);
                node.state[opts.stateKey] = isNaN(v) ? opts.defaultValue : v;
                node.saveState();
            };
        } else if (opts.kind === "select") {
            input = document.createElement("select");
            input.style.cssText = "background: #1a1a1a; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 4px 6px; font-size: 12px;";
            (opts.options || []).forEach(o => {
                const opt = document.createElement("option");
                opt.value = o;
                opt.textContent = o;
                if (o === (node.state[opts.stateKey] || opts.defaultValue)) opt.selected = true;
                input.appendChild(opt);
            });
            input.onchange = () => { node.state[opts.stateKey] = input.value; node.saveState(); };
        }

        row.appendChild(label);
        row.appendChild(help);
        row.appendChild(input);
        content.appendChild(row);
    }

    _addRunSetting(content, { stateKey: "overwrite_existing", label: "Overwrite Existing", kind: "bool",
        tooltip: "True = Re-run everything. False = Skip already generated images (Resume)." });
    _addRunSetting(content, { stateKey: "flush_batch_every", label: "Flush Batch Every", kind: "int", min: 0, max: 64, defaultValue: 1,
        tooltip: "Update dashboard every X images. 0 = Use VAE Batch Size." });
    _addRunSetting(content, { stateKey: "lora_triggerwords_mode", label: "LoRA Triggerwords Mode", kind: "select",
        options: ["None", "Append To End", "Append To Start", "Read From Config"], defaultValue: "None",
        tooltip: "None = Don't fetch/append trigger words. Append To End = Add triggers at end of prompt (default behavior). Append To Start = Add triggers at start of prompt. Read From Config = Use lora_triggerwords_append_settings in config JSON to specify per-lora placement." });

    // CivitAI companion notice — shows when mode is not "None" AND companion missing
    const _twModeNoticeSlot = document.createElement('div');
    content.appendChild(_twModeNoticeSlot);
    const _twModeRow = content.children[content.children.length - 2];  // the row just appended by _addRunSetting
    const _twModeSelect = _twModeRow ? _twModeRow.querySelector('select') : null;
    const _updateTwModeNotice = () => {
        _twModeNoticeSlot.replaceChildren();
        const currentVal = (node.state.lora_triggerwords_mode || "None");
        if (currentVal === "None") return;
        isCivitaiAvailable().then(available => {
            // Re-check current value at notice-render time (user may have changed back)
            if ((node.state.lora_triggerwords_mode || "None") === "None") return;
            if (!available) _twModeNoticeSlot.appendChild(_renderCivitaiCompanionNotice());
        });
    };
    if (_twModeSelect) {
        _twModeSelect.addEventListener('change', _updateTwModeNotice);
    }
    _updateTwModeNotice();  // initial check on render

    _addRunSetting(content, { stateKey: "save_conditioning_cache_to_file", label: "Save Conditioning Cache To File", kind: "bool",
        tooltip: "Save CLIP conditioning cache to disk. Useful when experimenting with the same prompts/models — skips text encoding on resume. WARNING: Can create very large files in output/benchmarks. Automatically disabled when optional inputs (model/clip/conditioning) are connected." });
    _addRunSetting(content, { stateKey: "enable_model_cache", label: "Enable Model Cache", kind: "bool",
        tooltip: "Experimental: intelligent model/LoRA caching with async background preloading. Speeds up generation when switching LoRAs frequently. Loads cached LoRAs from RAM instead of disk. Disable to reduce RAM/VRAM usage." });
    _addRunSetting(content, { stateKey: "vae_batch_size", label: "VAE Batch Size", kind: "int", min: -1, max: 64, defaultValue: 1,
        tooltip: "How many images to encode/decode per VAE pass. Lower = less VRAM. -1 = process all at once. Default: 4." });
    _addRunSetting(content, { stateKey: "add_random_seeds_to_gens", label: "Additional Random Seeds", kind: "int", min: 0, max: 100, defaultValue: 0,
        tooltip: "Generate this many extra images per config using additional random seeds. 0 = disabled." });

    section.appendChild(header);
    section.appendChild(content);
    container.appendChild(section);
}

// Persisted preview panel height per node (keyed by node.id, so each node
// remembers its own ratio across renderUI() rebuilds within a session).
const _previewHeights = new Map();

export function renderPreviewSection(container, node) {
    const panel = document.createElement("div");
    panel.className = "cb-preview-panel";
    panel.id = "cb-sec-preview";

    // Restore prior height if user dragged it during this session
    const savedH = node && _previewHeights.get(node.id);
    if (savedH) panel.style.flexBasis = savedH + "px";

    // Drag handle (top edge) — drag UP to grow the panel, DOWN to shrink it
    const handle = document.createElement("div");
    handle.className = "cb-preview-resize-handle";
    handle.title = "Drag to resize preview panel";
    panel.appendChild(handle);

    const header = document.createElement("div");
    header.className = "cb-preview-panel-header";
    const title = document.createElement("span");
    title.className = "cb-preview-panel-title";
    title.textContent = "📄 Output Preview (configs_json)";
    const meta = document.createElement("span");
    meta.className = "cb-preview-panel-meta";
    meta.id = "json-preview-meta";
    header.appendChild(title);
    header.appendChild(meta);
    panel.appendChild(header);

    const pre = document.createElement("pre");
    pre.className = "cb-preview-panel-content";
    pre.id = "json-preview";
    pre.textContent = "Loading preview…";
    panel.appendChild(pre);

    container.appendChild(panel);

    // Wire resize handle
    handle.addEventListener("mousedown", (e) => {
        e.preventDefault();
        const startY = e.clientY;
        const startH = panel.getBoundingClientRect().height;
        handle.classList.add("dragging");
        const onMove = (ev) => {
            const dy = startY - ev.clientY; // up = larger
            const newH = Math.max(120, Math.min(2000, startH + dy));
            panel.style.flexBasis = newH + "px";
            if (node) _previewHeights.set(node.id, newH);
        };
        const onUp = () => {
            handle.classList.remove("dragging");
            document.removeEventListener("mousemove", onMove);
            document.removeEventListener("mouseup", onUp);
        };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
    });
}

function _setPreviewStale(node, errMsg) {
    const meta = node.htmlContainer?.querySelector("#json-preview-meta");
    if (meta) {
        meta.textContent = "⚠ stale: " + (errMsg || "").slice(0, 60);
        meta.style.color = "#c33";
    }
}

function _setPreviewMeta(node, data) {
    const meta = node.htmlContainer?.querySelector("#json-preview-meta");
    if (!meta) return;
    let n = 0;
    try {
        const parsed = JSON.parse(data.configs_json);
        n = (parsed.configs || []).length;
    } catch (_) { /* ignore — server returned non-JSON; stale handler covers it */ }
    meta.textContent = `${n} config${n === 1 ? '' : 's'}`;
    meta.style.color = "";
}

export function updatePreview(node) {
    const preview = node.htmlContainer?.querySelector("#json-preview");
    if (!preview) return;

    if (node._previewTimer) clearTimeout(node._previewTimer);
    node._previewSeq = (node._previewSeq || 0) + 1;
    const seq = node._previewSeq;

    node._previewTimer = setTimeout(async () => {
        try {
            const resp = await fetch("/configbuilder/preview", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ state: node.state }),
            });
            if (seq !== node._previewSeq) return; // newer request superseded us
            if (!resp.ok) {
                const errText = await resp.text();
                throw new Error(errText || `HTTP ${resp.status}`);
            }
            const data = await resp.json();
            preview.textContent = data.configs_json;
            _setPreviewMeta(node, data);
        } catch (e) {
            if (seq !== node._previewSeq) return;
            _setPreviewStale(node, e.message || String(e));
        }
    }, 800);
}

export async function renderUI(node, availableLoras, modelLists, loraFolders, availableSessions, availableConfigs, refreshAllConfigBuilders) {
    // Save scroll from the main content area
    const scrollContainer = node.htmlContainer.querySelector(".cb-main-content");
    const savedScrollTop = scrollContainer ? scrollContainer.scrollTop : 0;

    node.htmlContainer.innerHTML = getStyles() + '<div class="cb-container" id="cb-root"></div>';

    const root = node.htmlContainer.querySelector("#cb-root");

    // === TOP BAR (sticky) ===
    const topBar = createTopBar(node, {
        availableSessions,
        availableConfigs,
        refreshAllConfigBuilders,
        onLoadSession: async (value) => {
            node.state.auto_save = false;
            await node.loadSession(value);
        },
        onLoadConfig: async (value) => {
            await node.loadConfigFromBackend(value);
        },
        onSaveConfig: async () => {
            await node.saveConfigToBackend();
            const { getAvailableConfigs, clearConfigsCache } = await import('./conf-builder-utilities.js');
            clearConfigsCache();
            await getAvailableConfigs();
            node.renderUI();
        }
    });
    root.appendChild(topBar);

    // === LAYOUT WRAPPER (sidebar + main content) ===
    const layoutWrapper = document.createElement("div");
    layoutWrapper.className = "cb-layout-wrapper";

    // Create main content area first (sidebar needs reference for scroll spy)
    const mainContent = document.createElement("div");
    mainContent.className = "cb-main-content";

    // Create sidebar
    const sidebar = createSidebar(node, mainContent, refreshAllConfigBuilders);
    layoutWrapper.appendChild(sidebar);
    layoutWrapper.appendChild(mainContent);
    root.appendChild(layoutWrapper);

    // === MAIN CONTENT SECTIONS ===

    // Run Settings Section — session-level generator overrides. Rendered at
    // the very top so users see core run-behavior knobs (resume/overwrite,
    // batching, caching) before anything else.
    renderRunSettingsSection(node, mainContent);

    // Distribution Settings Section (always shown with On/Off toggle)
    renderDistributionSettingsSection(node, mainContent);

    // Global Prompts Section
    renderGlobalPromptsSection(node, mainContent);

    // Config Arrays Section
    const configSection = document.createElement("div");
    configSection.className = "cb-section full-width";
    configSection.id = "cb-sec-configs";

    // Uniform section header for Config Arrays
    const configHeader = createSectionHeader("⚙️", "Config Arrays", "configs");
    configSection.appendChild(configHeader);

    const headerBar = document.createElement("div");
    headerBar.style.cssText = "margin-bottom: 10px; display: flex; align-items: center; gap: 12px;";
    const addConfigBtn = document.createElement("button");
    addConfigBtn.className = "cb-button primary";
    addConfigBtn.textContent = "➕ Add Config Array";
    addConfigBtn.onclick = () => {
        node.state.config_arrays.push({
            name: `Config ${node.state.config_arrays.length + 1}`,
            samplers: ["euler"],
            schedulers: ["normal"],
            steps: "20",
            cfg: "7.0",
            seed_behavior: "fixed",
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
            resolutions: [],
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
            deep_shrink_upscale_method: "bicubic"
        });
        node.saveState();
        node.renderUI();
    };
    headerBar.appendChild(addConfigBtn);

    // Config Array Presets (save/load entire config arrays)
    const configPresetSelect = document.createElement("select");
    configPresetSelect.style.cssText = "background: #1a1a1a; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 3px 6px; font-size: 10px; max-width: 140px;";
    const configPresetDefault = document.createElement("option");
    configPresetDefault.value = ""; configPresetDefault.textContent = "-- Config Presets --";
    configPresetSelect.appendChild(configPresetDefault);

    fetch("/configbuilder/config_section_presets").then(r => r.json()).then(data => {
        (data.presets || []).forEach((p, i) => {
            const opt = document.createElement("option");
            opt.value = i; opt.textContent = p.name;
            configPresetSelect.appendChild(opt);
        });
        configPresetSelect._presets = data.presets || [];
    }).catch(() => { configPresetSelect._presets = []; });

    const configPresetLoadBtn = document.createElement("button");
    configPresetLoadBtn.className = "cb-btn";
    configPresetLoadBtn.textContent = "Load";
    configPresetLoadBtn.style.cssText = "font-size: 10px; padding: 2px 6px;";
    configPresetLoadBtn.onclick = () => {
        const idx = parseInt(configPresetSelect.value);
        if (isNaN(idx) || !configPresetSelect._presets) return;
        const preset = configPresetSelect._presets[idx];
        if (!preset || !preset.config_arrays) return;
        node.state.config_arrays = JSON.parse(JSON.stringify(preset.config_arrays));
        if (preset.global_positive_groups) node.state.global_positive_groups = JSON.parse(JSON.stringify(preset.global_positive_groups));
        if (preset.global_negative !== undefined) node.state.global_negative = preset.global_negative;
        node.saveState();
        node.renderUI();
    };

    const configPresetSaveBtn = document.createElement("button");
    configPresetSaveBtn.className = "cb-btn";
    configPresetSaveBtn.textContent = "Save";
    configPresetSaveBtn.style.cssText = "font-size: 10px; padding: 2px 6px;";
    configPresetSaveBtn.onclick = () => {
        const name = prompt("Config preset name:");
        if (!name) return;
        const presets = configPresetSelect._presets || [];
        presets.push({
            name: name,
            config_arrays: JSON.parse(JSON.stringify(node.state.config_arrays)),
            global_positive_groups: JSON.parse(JSON.stringify(node.state.global_positive_groups || [])),
            global_negative: node.state.global_negative || "",
        });
        fetch("/configbuilder/config_section_presets", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({presets: presets})
        }).then(() => {
            configPresetSelect._presets = presets;
            const opt = document.createElement("option");
            opt.value = presets.length - 1; opt.textContent = name;
            configPresetSelect.appendChild(opt);
            configPresetSelect.value = presets.length - 1;
        });
    };

    const configPresetDelBtn = document.createElement("button");
    configPresetDelBtn.className = "cb-btn";
    configPresetDelBtn.textContent = "\uD83D\uDDD1"; // 🗑
    configPresetDelBtn.style.cssText = "font-size: 10px; padding: 2px 4px; background: var(--danger); color: #fff;";
    configPresetDelBtn.onclick = () => {
        const idx = parseInt(configPresetSelect.value);
        if (isNaN(idx) || !configPresetSelect._presets || idx < 0 || idx >= configPresetSelect._presets.length) return;
        if (!confirm('Delete preset "' + configPresetSelect._presets[idx].name + '"?')) return;
        configPresetSelect._presets.splice(idx, 1);
        fetch("/configbuilder/config_section_presets", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({presets: configPresetSelect._presets})
        });
        configPresetSelect.value = "";
        // Remove the option
        for (let i = configPresetSelect.options.length - 1; i >= 0; i--) {
            if (configPresetSelect.options[i].value == idx) configPresetSelect.remove(i);
        }
    };

    headerBar.appendChild(configPresetSelect);
    headerBar.appendChild(configPresetLoadBtn);
    headerBar.appendChild(configPresetSaveBtn);
    headerBar.appendChild(configPresetDelBtn);

    configSection.appendChild(headerBar);

    // Quick-jump navigation bar (only show when there are 2+ configs)
    if (node.state.config_arrays.length > 1) {
        const navBar = document.createElement("div");
        navBar.style.cssText = "display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; padding: 8px; background: #2a2a2a; border-radius: 4px; position: sticky; top: 0; z-index: 50;";

        const navLabel = document.createElement("span");
        navLabel.style.cssText = "color: #999; font-size: 11px; font-weight: bold; display: flex; align-items: center; margin-right: 4px;";
        navLabel.textContent = "JUMP TO:";
        navBar.appendChild(navLabel);

        node.state.config_arrays.forEach((ca, idx) => {
            const color = CONFIG_COLORS[idx % CONFIG_COLORS.length];
            const btn = document.createElement("button");
            btn.className = "cb-button";
            btn.style.cssText = `padding: 3px 10px; font-size: 11px; font-weight: bold; border-left: 3px solid ${color}; color: ${color};`;
            btn.textContent = `${idx + 1}. ${ca.name || 'Config ' + (idx + 1)}`;
            btn.onclick = () => {
                const target = configSection.querySelector(`#cb-config-${idx}`);
                if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
            };
            navBar.appendChild(btn);
        });
        configSection.appendChild(navBar);
    }

    const arraysContainer = document.createElement("div");
    arraysContainer.className = "cb-arrays-container";

    node.state.config_arrays.forEach((configArray, arrayIdx) => {
        const arrayElement = createConfigArrayElement(node, configArray, arrayIdx, modelLists);
        renderConfigPromptsSection(node, arrayElement, configArray, arrayIdx);
        renderModelsSection(node, arrayElement, configArray, arrayIdx, modelLists);
        // LTX Video Settings — only renders when this configArray's models[] contain an ltx_video model
        const ltxSection = renderLTXSection(node, configArray, arrayIdx, modelLists);
        if (ltxSection) arrayElement.appendChild(ltxSection);
        renderVAEsSection(node, arrayElement, configArray, arrayIdx, modelLists);
        renderLorasSection(node, arrayElement, configArray, arrayIdx, availableLoras, loraFolders);
        arraysContainer.appendChild(arrayElement);
    });

    configSection.appendChild(arraysContainer);
    mainContent.appendChild(configSection);

    // Session-level settings (applies to all configs)
    renderUpscalingSection(node, mainContent, modelLists);
    renderCooldownSection(node, mainContent);

    // Output Preview — rendered as a SIBLING of layoutWrapper (not inside
    // mainContent), so it sits as a fixed/resizable panel at the bottom of
    // the node, always visible regardless of scroll position.
    renderPreviewSection(root, node);
    updatePreview(node);

    // Restore scroll position on main content
    const newScrollContainer = node.htmlContainer.querySelector(".cb-main-content");
    if (newScrollContainer) {
        newScrollContainer.scrollTop = savedScrollTop;
    }
}