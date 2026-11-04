/**
 * Config Builder Workflow Sweep Module
 *
 * UI for a config array of kind "workflow": load an API-format ComfyUI workflow (Export (API)),
 * pick the image socket to collect, turn literal inputs into sweep axes ("bindings"), dry-run the
 * generated prompts and start / pause / resume / cancel the sweep. Python side: workflow_sweep.py
 * (expansion), workflow_sweep_runner.py (queue feeder), routes /config_tester/sweep/*.
 *
 * ES module, imported by conf-builder-config-management.js like conf-builder-distribution.js.
 */

import { createSectionHeader } from './conf-builder-ui-components.js';
import { countPromptCombinations } from './conf-builder-utilities.js';

const JSON_HEADERS = { "Content-Type": "application/json" };

// input name -> manifest field (mirrors ROLE_HINTS in workflow_sweep.py)
const FIELD_HINTS = {
    ckpt_name: "model", unet_name: "model", seed: "seed", noise_seed: "seed", steps: "steps", cfg: "cfg",
    sampler_name: "sampler", scheduler: "scheduler", denoise: "denoise", vae_name: "vae",
};
const AXIS_FIELDS = ["model", "sampler", "scheduler", "steps", "cfg", "seed", "denoise", "vae"];
const ACTIVE_STATES = ["preparing", "running", "paused"];

// ---------------------------------------------------------------------------
// state helpers
// ---------------------------------------------------------------------------

export function defaultWorkflowState() {
    return { source_name: "", api_prompt: null, output: {}, run_other_outputs: false, bindings: [] };
}

export function ensureWorkflowState(configArray) {
    if (!configArray.workflow || typeof configArray.workflow !== "object") configArray.workflow = defaultWorkflowState();
    const wf = configArray.workflow;
    if (!Array.isArray(wf.bindings)) wf.bindings = [];
    if (!wf.output || typeof wf.output !== "object") wf.output = {};
    if (wf.run_other_outputs === undefined) wf.run_other_outputs = false;
    if (wf.api_prompt === undefined) wf.api_prompt = null;
    if (wf.source_name === undefined) wf.source_name = "";
    return wf;
}

/** A fresh config array of kind "workflow" (the Add Workflow Sweep button pushes this). */
export function newWorkflowArray(index) {
    return {
        name: `Workflow ${index}`,
        kind: "workflow",
        workflow: defaultWorkflowState(),
        positive_prompt_groups: [],
        negative_prompt: "",
        use_custom_prompts: false,
    };
}

/**
 * Inverse of config_builder_node._workflow_array_to_config: a saved workflow config entry (manifest
 * meta.configs_json) back into a Builder array of kind "workflow". Never throws on odd input.
 */
export function workflowConfigToArray(cfg, idx = 0) {
    const arr = newWorkflowArray(idx + 1);
    cfg = (cfg && typeof cfg === "object") ? cfg : {};
    if (cfg.name) arr.name = String(cfg.name);
    arr.workflow = {
        source_name: String(cfg.workflow_source || ""),
        api_prompt: (cfg.workflow && typeof cfg.workflow === "object" && Object.keys(cfg.workflow).length) ? cfg.workflow : null,
        output: (cfg.workflow_output && typeof cfg.workflow_output === "object") ? cfg.workflow_output : {},
        run_other_outputs: !!cfg.workflow_run_other_outputs,
        bindings: Array.isArray(cfg.workflow_bindings) ? cfg.workflow_bindings : [],
    };
    if (cfg._prompt_source === "custom" && Array.isArray(cfg.positive)) {
        arr.positive_prompt_groups = cfg.positive;
        arr.negative_prompt = typeof cfg.negative === "string" ? cfg.negative : "";
        arr.use_custom_prompts = true;
    }
    ensureWorkflowState(arr);
    return arr;
}

function el(tag, props = {}, css = "") {
    const e = document.createElement(tag);
    Object.assign(e, props);
    if (css) e.style.cssText = css;
    return e;
}

function nextBindingId(wf) {
    const ids = new Set(wf.bindings.map(b => b.id));
    let n = 1;
    while (ids.has(`b${n}`)) n++;
    return `b${n}`;
}

function isLink(v) {
    return Array.isArray(v) && v.length === 2 && (typeof v[0] === "string" || typeof v[0] === "number") && Number.isInteger(v[1]);
}

function nodeTitle(prompt, id) {
    const n = prompt && prompt[id];
    return (n && ((n._meta && n._meta.title) || n.class_type)) || String(id);
}

/** Node ids upstream of (and including) the output socket's node. */
function ancestorsOf(prompt, outId) {
    const seen = new Set();
    if (!prompt || prompt[outId] === undefined) return seen;
    const queue = [String(outId)];
    seen.add(String(outId));
    while (queue.length) {
        const cur = queue.shift();
        const inputs = (prompt[cur] && prompt[cur].inputs) || {};
        for (const v of Object.values(inputs)) {
            if (isLink(v)) {
                const up = String(v[0]);
                if (prompt[up] !== undefined && !seen.has(up)) { seen.add(up); queue.push(up); }
            }
        }
    }
    return seen;
}

function targetKey(t) { return `${t.node_id}::${t.input}`; }

function boundTargets(wf) {
    const map = new Map();
    for (const b of wf.bindings) {
        const ts = b.kind === "resolution" ? [b.width, b.height] : (b.targets || []);
        ts.filter(Boolean).forEach(t => map.set(targetKey(t), b.id));
        (b.slots || []).forEach(s => ["lora_name", "strength_model", "strength_clip"].forEach(i => map.set(`${s.node_id}::${i}`, b.id)));
    }
    return map;
}

function templateValue(wf, t) {
    const n = wf.api_prompt && wf.api_prompt[t.node_id];
    return n && n.inputs ? n.inputs[t.input] : undefined;
}

/** Parse a typed chip value: numbers stay numbers when the template literal is numeric. */
function typedValue(raw, template) {
    const text = String(raw).trim();
    if (typeof template === "number") {
        const n = Number(text);
        return Number.isFinite(n) ? n : text;
    }
    if (typeof template === "boolean") return text.toLowerCase() === "true";
    return text;
}

function bindingLabel(wf, b) {
    const p = wf.api_prompt || {};
    const t = (x) => x ? `${nodeTitle(p, x.node_id)}.${x.input} #${x.node_id}` : "?";
    if (b.kind === "lora") return `LoRA slots: ${(b.slots || []).map(s => `${nodeTitle(p, s.node_id)} #${s.node_id}`).join(", ")}`;
    if (b.kind === "resolution") return `Resolution: ${t(b.width)} x ${t(b.height)}`;
    const label = b.kind === "positive" ? "Positive prompt" : b.kind === "negative" ? "Negative prompt" : "Values";
    return `${label}: ${(b.targets || []).map(t).join(", ")}`;
}

function axisSize(configArray, b, globalPositive) {
    if (b.kind === "positive") {
        const groups = (configArray.use_custom_prompts && (configArray.positive_prompt_groups || []).length)
            ? configArray.positive_prompt_groups : (globalPositive || []);
        return groups.length ? countPromptCombinations(groups) : 0;
    }
    if (b.kind === "negative") return 1;   // one negative string (pairing with the positives is the Dry run's job)
    return (b.values || []).length;
}

// ---------------------------------------------------------------------------
// server calls
// ---------------------------------------------------------------------------

const parseCache = new WeakMap();

function ensureParse(wf, force = false) {
    const key = wf.api_prompt;
    if (!key) return Promise.resolve(null);
    if (!force && parseCache.has(key)) return parseCache.get(key);
    const p = fetch("/config_tester/sweep/parse", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify({ api_json: key }) })
        .then(async (r) => {
            const d = await r.json();
            if (!r.ok) throw new Error(d.error || (d.errors && d.errors[0]) || `HTTP ${r.status}`);
            return d;
        });
    parseCache.set(key, p);
    p.catch(() => parseCache.delete(key));
    return p;
}

async function post(path, body) {
    const r = await fetch(path, { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) });
    let data = {};
    try { data = await r.json(); } catch (e) { /* non-JSON error body */ }
    return { ok: r.ok, status: r.status, data };
}

/** Builder state with only the workflow arrays (a sweep cannot be mixed with standard configs). */
function workflowOnlyState(node) {
    const state = JSON.parse(JSON.stringify(node.state));
    state.config_arrays = state.config_arrays.filter(a => a.kind === "workflow");
    return state;
}

// ---------------------------------------------------------------------------
// array header (banner + name + duplicate / delete)
// ---------------------------------------------------------------------------

export function createWorkflowArrayElement(node, configArray, arrayIdx, accentColor = "#00aaff") {
    const div = el("div", { className: "cb-array", id: `cb-config-${arrayIdx}` }, `border-left: 4px solid ${accentColor};`);
    const banner = el("div", {}, `display: flex; align-items: center; gap: 10px; padding: 8px 12px; margin: -12px -12px 12px -12px; background: linear-gradient(90deg, ${accentColor}22, transparent); border-bottom: 2px solid ${accentColor}44; border-radius: 4px 4px 0 0;`);
    banner.appendChild(el("span", { textContent: `${arrayIdx + 1}` }, `background: ${accentColor}; color: #000; font-weight: 900; font-size: 13px; padding: 2px 8px; border-radius: 4px; font-family: monospace; min-width: 20px; text-align: center;`));
    banner.appendChild(el("span", { textContent: "Workflow Sweep" }, `color: ${accentColor}; font-weight: 800; font-size: 14px;`));

    const nameInput = el("input", { className: "cb-input", value: configArray.name || `Workflow ${arrayIdx + 1}`, title: "Config name" }, "flex: 1; max-width: 280px;");
    nameInput.onchange = () => { configArray.name = nameInput.value; node.saveState(); };
    banner.appendChild(nameInput);
    banner.appendChild(el("span", {}, "flex: 1;"));

    const dup = el("button", { className: "cb-button primary", textContent: "📋 Duplicate" });
    dup.onclick = () => {
        const clone = JSON.parse(JSON.stringify(configArray));
        clone.name = `${clone.name} (Copy)`;
        node.state.config_arrays.splice(arrayIdx + 1, 0, clone);
        node.saveState();
        node.renderUI();
    };
    banner.appendChild(dup);
    const del = el("button", { className: "cb-button danger", textContent: "🗑️ Delete" });
    del.onclick = () => {
        if (node.state.config_arrays.length > 1) {
            node.state.config_arrays.splice(arrayIdx, 1);
            node.saveState();
            node.renderUI();
        }
    };
    banner.appendChild(del);
    div.appendChild(banner);
    return div;
}

// ---------------------------------------------------------------------------
// main section
// ---------------------------------------------------------------------------

export function renderWorkflowSweepSection(node, arrayElement, configArray, arrayIdx) {
    const wf = ensureWorkflowState(configArray);
    const section = el("div", { className: "cb-section full-width" }, "margin-top: 8px;");
    section.appendChild(createSectionHeader("🧪", "Workflow Sweep", "configs"));
    const body = el("div", {}, "padding: 8px 4px; display: flex; flex-direction: column; gap: 10px;");
    section.appendChild(body);
    arrayElement.appendChild(section);

    let info = null;           // result of /sweep/parse for the current api_prompt
    let parseError = "";
    let loading = false;
    let nodeFilter = "";
    let onlyUpstream = true;
    let sweepId = null;        // sweep being tracked for this Builder
    let pollTimer = null;
    let lastStatus = null;

    const save = () => { node.saveState(); };
    const rerender = () => render();

    // ---- source -------------------------------------------------------
    function renderSource(container) {
        const box = el("div", {}, "display: flex; flex-direction: column; gap: 6px;");
        box.appendChild(el("div", {
            textContent: "Load an API-format workflow (ComfyUI: Workflow > Export (API)). Pick the literal inputs to sweep; the rest of the workflow runs exactly as exported.",
        }, "font-size: 11px; color: #999;"));
        const row = el("div", {}, "display: flex; gap: 6px; align-items: center; flex-wrap: wrap;");
        const fileInput = el("input", { type: "file", accept: ".json,application/json" }, "display: none;");
        const loadFile = el("button", { className: "cb-button", textContent: "📂 Load file" });
        loadFile.onclick = () => fileInput.click();
        fileInput.onchange = () => {
            const f = fileInput.files && fileInput.files[0];
            if (!f) return;
            const reader = new FileReader();
            reader.onload = () => loadWorkflowText(String(reader.result || ""), f.name);
            reader.readAsText(f);
        };
        const pasteBtn = el("button", { className: "cb-button", textContent: "📋 Use pasted JSON" });
        row.append(loadFile, fileInput, pasteBtn);
        if (wf.api_prompt) {
            row.appendChild(el("span", { textContent: `Loaded: ${wf.source_name || "(pasted)"} - ${Object.keys(wf.api_prompt).length} nodes` }, "color: #6c6; font-size: 11px;"));
            const clear = el("button", { className: "cb-button danger", textContent: "Remove workflow" });
            clear.onclick = () => {
                if (wf.bindings.length && !confirm("Remove the workflow and its bindings?")) return;
                configArray.workflow = defaultWorkflowState();
                info = null; parseError = "";
                save(); rerender();
            };
            row.appendChild(clear);
        }
        box.appendChild(row);
        const ta = el("textarea", { className: "cb-input", placeholder: "...or paste the API JSON here", rows: 3 }, "width: 100%; font-family: monospace; font-size: 11px; resize: vertical;");
        pasteBtn.onclick = () => loadWorkflowText(ta.value, "pasted workflow");
        box.appendChild(ta);
        if (parseError) box.appendChild(el("div", { textContent: `⚠️ ${parseError}` }, "color: #ff8888; font-size: 12px;"));
        container.appendChild(box);
    }

    async function loadWorkflowText(text, name) {
        parseError = "";
        let obj;
        try { obj = JSON.parse(text); } catch (e) { parseError = `Not valid JSON: ${e.message}`; return rerender(); }
        if (obj && Array.isArray(obj.nodes) && Array.isArray(obj.links)) {
            parseError = "This is an editor-format workflow. Use Workflow > Export (API) in ComfyUI and load that file.";
            return rerender();
        }
        if (obj && obj.prompt && typeof obj.prompt === "object" && !Object.values(obj).some(v => v && v.class_type)) obj = obj.prompt;
        const probe = { api_prompt: obj };
        loading = true; rerender();
        try {
            const parsed = await ensureParse(probe, true);
            configArray.workflow = defaultWorkflowState();
            Object.assign(configArray.workflow, { api_prompt: obj, source_name: name || "" });
            parseCache.set(configArray.workflow.api_prompt, Promise.resolve(parsed));
            info = parsed;
            const best = (parsed.image_sources || [])[0];
            if (best) configArray.workflow.output = { node_id: best.node_id, output_index: best.output_index, picked_from: best.picked_from };
            save();
        } catch (e) {
            parseError = String(e.message || e);
        }
        loading = false;
        rerender();
    }

    // ---- output picker ------------------------------------------------
    function renderOutput(container) {
        const row = el("div", {}, "display: flex; gap: 8px; align-items: center; flex-wrap: wrap;");
        row.appendChild(el("label", { className: "cb-label", textContent: "Image to collect" }, "min-width: 110px;"));
        const sel = el("select", { className: "cb-select" }, "flex: 1; min-width: 220px;");
        const sources = (info && info.image_sources) || [];
        const cur = wf.output && wf.output.node_id !== undefined ? `${wf.output.node_id}:${wf.output.output_index || 0}` : "";
        if (!sources.length) sel.appendChild(el("option", { value: "", textContent: "(no IMAGE output found)" }));
        sources.forEach(s => {
            const label = `${nodeTitle(wf.api_prompt, s.node_id)} #${s.node_id} output ${s.output_index}` + (s.picked_from ? ` (feeds ${nodeTitle(wf.api_prompt, s.picked_from)} #${s.picked_from})` : "");
            sel.appendChild(el("option", { value: `${s.node_id}:${s.output_index}`, textContent: label, selected: `${s.node_id}:${s.output_index}` === cur }));
        });
        sel.onchange = () => {
            const [n, i] = sel.value.split(":");
            if (n === undefined || n === "") return;
            wf.output = { node_id: n, output_index: parseInt(i, 10) || 0 };
            save(); rerender();
        };
        row.appendChild(sel);
        const cb = el("input", { type: "checkbox", checked: !!wf.run_other_outputs });
        cb.onchange = () => { wf.run_other_outputs = cb.checked; save(); };
        const lab = el("label", {}, "font-size: 11px; color: #aaa; display: flex; gap: 4px; align-items: center;");
        lab.append(cb, document.createTextNode("Also run the workflow's own outputs (SaveImage etc.)"));
        row.appendChild(lab);
        container.appendChild(row);
    }

    // ---- suggestions --------------------------------------------------
    function addBinding(b) {
        b.id = nextBindingId(wf);
        wf.bindings.push(b);
        save(); rerender();
    }

    function addLiteral(nodeId, input, override) {
        const current = templateValue(wf, { node_id: nodeId, input });
        const field = FIELD_HINTS[input] || null;
        const b = { kind: "literal", field, targets: [{ node_id: nodeId, input }], values: (current !== undefined && !isLink(current)) ? [current] : [] };
        if (override) b.override_link = true;
        addBinding(b);
    }

    function renderSuggestions(container) {
        const used = boundTargets(wf);
        const sugg = ((info && info.suggestions) || []).filter(s => {
            if (s.kind === "lora") return !wf.bindings.some(b => b.kind === "lora");
            if (s.kind === "resolution") return !wf.bindings.some(b => b.kind === "resolution");
            return !(s.targets || []).some(t => used.has(targetKey(t)));
        });
        if (!sugg.length) return;
        const box = el("div", {}, "display: flex; flex-wrap: wrap; gap: 4px; align-items: center;");
        box.appendChild(el("span", { textContent: "Suggested:" }, "font-size: 11px; color: #999; margin-right: 4px;"));
        const seenFieldNode = new Set();
        sugg.forEach(s => {
            const key = `${s.kind}|${s.field}|${(s.targets || []).map(targetKey).join(",")}`;
            if (seenFieldNode.has(key)) return;
            seenFieldNode.add(key);
            const label = s.kind === "positive" ? "Positive prompt" : s.kind === "negative" ? "Negative prompt"
                : s.kind === "lora" ? "LoRA" : s.kind === "resolution" ? "Resolution" : (s.field || "value");
            const btn = el("button", { className: "cb-button", textContent: `＋ ${label}`, title: s.reason || "" }, "padding: 2px 8px; font-size: 11px;");
            btn.onclick = () => {
                if (s.kind === "positive" || s.kind === "negative") addBinding({ kind: s.kind, targets: s.targets });
                else if (s.kind === "lora") addBinding({ kind: "lora", slots: s.slots, values: [] });
                else if (s.kind === "resolution") addBinding({ kind: "resolution", width: s.width, height: s.height, values: ["1024x1024"] });
                else addLiteral(s.targets[0].node_id, s.targets[0].input, false);
            };
            box.appendChild(btn);
        });
        container.appendChild(box);
    }

    // ---- node table ---------------------------------------------------
    function renderNodeTable(container) {
        if (!info || !wf.api_prompt) return;
        const used = boundTargets(wf);
        const prompt = wf.api_prompt;
        const upstream = wf.output && wf.output.node_id !== undefined ? ancestorsOf(prompt, wf.output.node_id) : null;

        const head = el("div", {}, "display: flex; gap: 8px; align-items: center; flex-wrap: wrap;");
        head.appendChild(el("strong", { textContent: "Workflow nodes" }, "font-size: 12px;"));
        const filter = el("input", { className: "cb-input", placeholder: "filter by title, class or input...", value: nodeFilter }, "flex: 1; min-width: 160px;");
        filter.oninput = () => { nodeFilter = filter.value; renderRows(); };
        const upCb = el("input", { type: "checkbox", checked: onlyUpstream });
        upCb.onchange = () => { onlyUpstream = upCb.checked; renderRows(); };
        const upLab = el("label", {}, "font-size: 11px; color: #aaa; display: flex; gap: 4px; align-items: center;");
        upLab.append(upCb, document.createTextNode("only nodes feeding the chosen image"));
        head.append(filter, upLab);
        container.appendChild(head);

        const list = el("div", {}, "max-height: 340px; overflow-y: auto; border: 1px solid #333; border-radius: 4px;");
        container.appendChild(list);

        function renderRows() {
            list.textContent = "";
            const f = nodeFilter.trim().toLowerCase();
            let shown = 0;
            (info.nodes || []).forEach(n => {
                if (onlyUpstream && upstream && !upstream.has(String(n.id))) return;
                const rows = (n.inputs || []).filter(i => {
                    if (!f) return true;
                    return `${n.title} ${n.class_type} ${n.id} ${i.name}`.toLowerCase().includes(f);
                });
                if (!rows.length) return;
                shown++;
                const card = el("details", {}, "border-bottom: 1px solid #2a2a2a; padding: 2px 6px;");
                if (f) card.open = true;
                const sum = el("summary", {}, "cursor: pointer; font-size: 12px; padding: 2px 0;");
                sum.textContent = `#${n.id}  ${n.title}` + (n.title !== n.class_type ? `  (${n.class_type})` : "") + (n.missing_class ? "  ⚠️ not installed" : "");
                card.appendChild(sum);
                rows.forEach(inp => {
                    const r = el("div", {}, "display: flex; gap: 8px; align-items: center; padding: 1px 0 1px 14px; font-size: 11px;");
                    r.appendChild(el("span", { textContent: inp.name }, "min-width: 150px; color: #9cf; font-family: monospace;"));
                    const val = inp.is_link ? `🔗 #${inp.link[0]}:${inp.link[1]}` : String(inp.value);
                    r.appendChild(el("span", { textContent: val.length > 70 ? val.slice(0, 70) + "..." : val, title: val }, `flex: 1; color: ${inp.is_link ? "#888" : "#ddd"}; overflow: hidden; white-space: nowrap; text-overflow: ellipsis;`));
                    const boundId = used.get(`${n.id}::${inp.name}`);
                    if (boundId) {
                        r.appendChild(el("span", { textContent: `✓ ${boundId}` }, "color: #6c6; font-size: 11px;"));
                    } else {
                        const add = el("button", { className: "cb-button", textContent: inp.is_link ? "＋ override link" : "＋ sweep" }, "padding: 0 6px; font-size: 10px;");
                        add.onclick = () => addLiteral(String(n.id), inp.name, !!inp.is_link);
                        r.appendChild(add);
                    }
                    card.appendChild(r);
                });
                list.appendChild(card);
            });
            if (!shown) list.appendChild(el("div", { textContent: "No matching nodes." }, "padding: 8px; color: #777; font-size: 11px;"));
        }
        renderRows();
    }

    // ---- bindings -----------------------------------------------------
    function comboOptionsFor(b) {
        if (!info || !(b.targets || []).length) return null;
        const t = b.targets[0];
        const n = (info.nodes || []).find(x => String(x.id) === String(t.node_id));
        const i = n && (n.inputs || []).find(x => x.name === t.input);
        return i && Array.isArray(i.options) && i.options.length ? i.options : null;
    }

    function renderValuesEditor(card, b) {
        if (b.kind === "positive" || b.kind === "negative") {
            card.appendChild(el("div", { textContent: `Values come from this config's Prompts section (${b.kind}).` }, "font-size: 11px; color: #999;"));
            return;
        }
        const chips = el("div", {}, "display: flex; flex-wrap: wrap; gap: 4px; min-height: 22px;");
        const renderChips = () => {
            chips.textContent = "";
            if (!(b.values || []).length) chips.appendChild(el("span", { textContent: "no values yet" }, "color: #c88; font-style: italic; font-size: 11px;"));
            (b.values || []).forEach((v, idx) => {
                const chip = el("span", {}, "display: inline-flex; align-items: center; background: #444; color: #fff; border-radius: 12px; padding: 2px 8px; font-size: 11px;");
                chip.appendChild(el("span", { textContent: String(v) }));
                const x = el("span", { textContent: "×" }, "margin-left: 6px; cursor: pointer; color: #ff8888; font-weight: bold;");
                x.onclick = () => { b.values.splice(idx, 1); save(); renderChips(); refreshFooter(); };
                chip.appendChild(x);
                chips.appendChild(chip);
            });
        };
        renderChips();
        card.appendChild(chips);

        const first = (b.targets || [])[0];
        const tmpl = first ? templateValue(wf, first) : undefined;
        const add = (raw) => {
            const parts = b.kind === "literal" && typeof tmpl !== "string" ? String(raw).split(",") : [String(raw)];
            parts.map(p => p.trim()).filter(p => p !== "").forEach(p => {
                b.values = b.values || [];
                const v = b.kind === "literal" ? typedValue(p, tmpl) : p;
                if (!b.values.some(x => x === v)) b.values.push(v);
            });
            save(); renderChips(); refreshFooter();
        };
        const row = el("div", {}, "display: flex; gap: 4px; align-items: center; flex-wrap: wrap;");
        const opts = b.kind === "literal" ? comboOptionsFor(b) : (b.kind === "lora" ? loraOptions(b) : null);
        if (opts) {
            const sel = el("select", { className: "cb-select" }, "flex: 1; min-width: 200px; font-size: 11px;");
            opts.forEach(o => sel.appendChild(el("option", { value: String(o), textContent: String(o) })));
            const addBtn = el("button", { className: "cb-button", textContent: "Add" });
            addBtn.onclick = () => add(sel.value);
            const allBtn = el("button", { className: "cb-button", textContent: "Add all" });
            allBtn.onclick = () => { opts.forEach(o => { if (!b.values.includes(o)) b.values.push(o); }); save(); renderChips(); refreshFooter(); };
            row.append(sel, addBtn, allBtn);
            if (b.kind === "lora") {
                const hint = el("span", { textContent: "strengths: name:0.8 or name:[0.5,1.0]" }, "font-size: 10px; color: #888;");
                row.appendChild(hint);
            }
        }
        const free = el("input", {
            className: "cb-input",
            placeholder: b.kind === "resolution" ? "1024x1024, then Enter" : b.kind === "lora" ? "lora.safetensors:0.8  or None" : (opts ? "or type a value" : "value(s), comma separated"),
        }, "flex: 1; min-width: 160px; font-size: 11px;");
        free.onkeydown = (e) => { if (e.key === "Enter" && free.value.trim()) { add(free.value); free.value = ""; } };
        const freeAdd = el("button", { className: "cb-button", textContent: "＋" });
        freeAdd.onclick = () => { if (free.value.trim()) { add(free.value); free.value = ""; } };
        row.append(free, freeAdd);
        card.appendChild(row);
    }

    function loraOptions(b) {
        if (!info || !(b.slots || []).length) return null;
        const n = (info.nodes || []).find(x => String(x.id) === String(b.slots[0].node_id));
        const i = n && (n.inputs || []).find(x => x.name === "lora_name");
        const opts = i && Array.isArray(i.options) ? i.options.slice() : [];
        return opts.length ? ["None", ...opts] : null;
    }

    function renderBindings(container) {
        const box = el("div", {}, "display: flex; flex-direction: column; gap: 6px;");
        box.appendChild(el("strong", { textContent: `Sweep axes (${wf.bindings.length})` }, "font-size: 12px;"));
        if (!wf.bindings.length) {
            box.appendChild(el("div", { textContent: "No axes yet: use the suggestions above or '＋ sweep' in the node list. With no axes the workflow runs once as exported." }, "font-size: 11px; color: #999;"));
        }
        wf.bindings.forEach((b, bi) => {
            const card = el("div", {}, "border: 1px solid #3a3a3a; border-left: 3px solid #0aa; border-radius: 4px; padding: 6px 8px; display: flex; flex-direction: column; gap: 4px;");
            const head = el("div", {}, "display: flex; gap: 8px; align-items: center;");
            head.appendChild(el("span", { textContent: b.id }, "background: #0aa; color: #000; font-weight: bold; font-size: 10px; padding: 0 5px; border-radius: 3px;"));
            head.appendChild(el("span", { textContent: bindingLabel(wf, b) }, "flex: 1; font-size: 11px; color: #ddd;"));
            const rm = el("button", { className: "cb-button danger", textContent: "✕" }, "padding: 0 6px;");
            rm.onclick = () => { wf.bindings.splice(bi, 1); save(); rerender(); };
            head.appendChild(rm);
            card.appendChild(head);

            if (b.kind === "literal") {
                const meta = el("div", {}, "display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 11px; color: #aaa;");
                meta.appendChild(document.createTextNode("Dashboard field:"));
                const fsel = el("select", { className: "cb-select" }, "font-size: 11px;");
                fsel.appendChild(el("option", { value: "", textContent: "(none)" }));
                AXIS_FIELDS.forEach(f => fsel.appendChild(el("option", { value: f, textContent: f, selected: b.field === f })));
                fsel.onchange = () => { b.field = fsel.value || null; save(); };
                meta.appendChild(fsel);
                const joinSel = el("select", { className: "cb-select" }, "font-size: 11px; max-width: 260px;");
                joinSel.appendChild(el("option", { value: "", textContent: "＋ also write the same value to..." }));
                const usedNow = boundTargets(wf);
                (info && info.nodes || []).forEach(n => (n.inputs || []).forEach(i => {
                    if (i.is_link || usedNow.has(`${n.id}::${i.name}`)) return;
                    joinSel.appendChild(el("option", { value: `${n.id}::${i.name}`, textContent: `#${n.id} ${n.title} . ${i.name}` }));
                }));
                joinSel.onchange = () => {
                    if (!joinSel.value) return;
                    const [nid, inp] = joinSel.value.split("::");
                    b.targets.push({ node_id: nid, input: inp });
                    save(); rerender();
                };
                meta.appendChild(joinSel);
                if (b.targets.length > 1) {
                    const un = el("button", { className: "cb-button", textContent: "remove last target" }, "font-size: 10px; padding: 0 6px;");
                    un.onclick = () => { b.targets.pop(); save(); rerender(); };
                    meta.appendChild(un);
                }
                if (b.override_link) meta.appendChild(el("span", { textContent: "⚠️ replaces a link" }, "color: #fa4;"));
                card.appendChild(meta);
            }
            renderValuesEditor(card, b);
            box.appendChild(card);
        });
        container.appendChild(box);
    }

    // ---- footer: counts + sweep controls ------------------------------
    let footerCount = null;
    function refreshFooter() {
        if (!footerCount) return;
        let total = 1;
        const parts = [];
        wf.bindings.forEach(b => {
            const n = axisSize(configArray, b, node.state.global_positive_groups);
            total *= n;
            parts.push(String(n));
        });
        footerCount.textContent = wf.bindings.length
            ? `Jobs: ${parts.join(" x ")} = ${total}` + (wf.bindings.some(b => b.kind === "positive" || b.kind === "negative") ? " (estimate; Dry run gives the exact count)" : "")
            : "Jobs: 1 (no axes)";
        footerCount.style.color = total > 0 ? "#00cc00" : "#ff8888";
    }

    function renderControls(container) {
        const box = el("div", {}, "display: flex; flex-direction: column; gap: 6px; border-top: 1px solid #333; padding-top: 8px;");
        footerCount = el("div", {}, "font-family: monospace; font-size: 13px; font-weight: bold;");
        box.appendChild(footerCount);
        refreshFooter();

        const row = el("div", {}, "display: flex; gap: 6px; flex-wrap: wrap; align-items: center;");
        const state = lastStatus ? lastStatus.state : null;
        const active = ACTIVE_STATES.includes(state);
        const mk = (text, cls, handler, disabled) => {
            const b = el("button", { className: `cb-button ${cls || ""}`, textContent: text, disabled: !!disabled });
            b.onclick = handler;
            row.appendChild(b);
            return b;
        };
        const ready = !!(wf.api_prompt && wf.output && wf.output.node_id !== undefined);
        mk("🔍 Dry run", "", () => dryRun(), !ready || loading);
        mk("▶ Start sweep", "primary", () => startSweep(), !ready || active || loading);
        mk("⏸ Pause", "", () => control("pause"), state !== "running");
        mk("⏯ Resume", "", () => control("resume"), state !== "paused");
        mk("⏹ Cancel", "danger", () => control("cancel"), !active);
        box.appendChild(row);
        box.appendChild(el("div", { textContent: "Start runs every Workflow Sweep config of this Builder. Or queue the Generator node: it starts the same sweep. Jobs go through ComfyUI's own queue, so you can keep using ComfyUI." }, "font-size: 10px; color: #888;"));

        statusLine = el("div", {}, "font-size: 12px; color: #ccc; white-space: pre-wrap;");
        box.appendChild(statusLine);
        paintStatus();
        container.appendChild(box);
    }

    let statusLine = null;
    let attachedNote = "";
    function paintStatus(extra) {
        if (!statusLine) return;
        const s = lastStatus;
        let text = extra || "";
        if (s) {
            text = `Sweep ${s.sweep_id}: ${s.state}` + (s.reason ? ` (${s.reason})` : "")
                + ` | ${s.done}/${s.to_run} done | ${s.errors} error(s) | ${s.in_flight} in flight`
                + (s.skipped_existing ? ` | ${s.skipped_existing} already in the session` : "")
                + (s.eta_str ? ` | ETA ${s.eta_str}` : "")
                + (s.last_errors && s.last_errors.length ? `\nLast error (job ${s.last_errors[s.last_errors.length - 1].index + 1}): ${s.last_errors[s.last_errors.length - 1].error}` : "");
            (s.warnings || []).forEach(w => { text += `\n⚠️ ${w}`; });
        }
        statusLine.textContent = attachedNote && s ? `${attachedNote}
${text}` : text;
    }

    async function fetchStatus() {
        try {
            const q = sweepId ? `sweep_id=${encodeURIComponent(sweepId)}` : `session_name=${encodeURIComponent((node.state.session_name || "").replace(/[^\w-]/g, "") || "default_session")}`;
            const r = await fetch(`/config_tester/sweep/status?${q}`);
            if (!r.ok) return null;
            return await r.json();
        } catch (e) { return null; }
    }

    function startPolling() {
        if (pollTimer) return;
        pollTimer = setInterval(async () => {
            if (!section.isConnected) { clearInterval(pollTimer); pollTimer = null; return; }
            const s = await fetchStatus();
            if (!s) return;
            const before = lastStatus && lastStatus.state;
            lastStatus = s;
            paintStatus();
            if (before !== s.state) rerender();           // button enable states follow the sweep state
            if (!ACTIVE_STATES.includes(s.state)) { clearInterval(pollTimer); pollTimer = null; }
        }, 2000);
    }

    async function startSweep() {
        attachedNote = "";
        paintStatus("Starting...");
        const { ok, status, data } = await post("/config_tester/sweep/start", { state: workflowOnlyState(node) });
        if (!ok) {
            paintStatus(status === 409 ? `${data.error}` : `Could not start: ${data.error || status}`);
            if (status === 409 && data.sweep_id) { sweepId = data.sweep_id; lastStatus = await fetchStatus(); rerender(); startPolling(); }
            return;
        }
        sweepId = data.sweep_id;
        attachedNote = data.attached ? `Already running: attached to sweep ${data.sweep_id}` : "";
        lastStatus = await fetchStatus();
        rerender();
        startPolling();
    }

    async function control(action) {
        if (!sweepId) return;
        const { ok, data } = await post(`/config_tester/sweep/${action}`, { sweep_id: sweepId });
        lastStatus = (await fetchStatus()) || lastStatus;
        if (!ok) paintStatus(`${action} failed: ${data.state || data.error || ""}`);
        rerender();
        if (lastStatus && ACTIVE_STATES.includes(lastStatus.state)) startPolling();
    }

    async function dryRun() {
        paintStatus("Building prompts...");
        const { ok, data } = await post("/config_tester/sweep/dry_run", { state: workflowOnlyState(node), limit: 3 });
        if (!ok) { paintStatus(`Dry run failed: ${data.error || "unknown error"}`); return; }
        paintStatus(`Dry run: ${data.total} job(s), ${data.to_run} to run, ${data.skipped_existing} already in the session.`);
        showDryRunModal(data);
    }

    function showDryRunModal(data) {
        const overlay = el("div", {}, "position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 100000; display: flex; align-items: center; justify-content: center;");
        const modal = el("div", {}, "background: #1e1e1e; color: #ddd; border: 1px solid #555; border-radius: 6px; width: min(900px, 94vw); max-height: 88vh; overflow-y: auto; padding: 14px; font-size: 12px;");
        overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
        modal.appendChild(el("h3", { textContent: "Workflow Sweep: dry run" }, "margin: 0 0 8px 0;"));
        modal.appendChild(el("div", { textContent: `${data.total} job(s) total, ${data.to_run} to run, ${data.skipped_existing} already in the session.` }));
        (data.axes || []).forEach(a => modal.appendChild(el("div", { textContent: `axis ${a.order + 1}: ${a.label} x ${a.count}` }, "color: #9cf;")));
        (data.warnings || []).forEach(w => modal.appendChild(el("div", { textContent: `⚠️ ${w}` }, "color: #fa4;")));
        const v = data.validation || {};
        modal.appendChild(el("div", { textContent: v.ok ? "✅ Job 1 passes ComfyUI's validation." : `❌ Job 1 fails validation: ${v.error}` }, `margin: 6px 0; color: ${v.ok ? "#6c6" : "#f88"};`));
        (data.jobs || []).forEach(j => {
            const d = el("details", {}, "border: 1px solid #333; border-radius: 4px; margin: 4px 0; padding: 4px 8px;");
            d.appendChild(el("summary", { textContent: `Job ${j.index + 1}${j.skip ? " (skipped: already in session)" : ""}  ${JSON.stringify(j.params || {})}` }, "cursor: pointer;"));
            d.appendChild(el("div", { textContent: `fingerprint ${j.fingerprint}; partial_execution_targets ${JSON.stringify(j.partial_execution_targets)}` }, "color: #888; font-size: 11px;"));
            const dl = el("button", { className: "cb-button", textContent: "⬇ Download job (API JSON)" }, "margin: 4px 0;");
            dl.onclick = () => {
                const blob = new Blob([JSON.stringify(j.prompt, null, 2)], { type: "application/json" });
                const url = URL.createObjectURL(blob);
                const a = el("a", { href: url, download: `sweep_job_${j.index + 1}.json` });
                document.body.appendChild(a); a.click(); a.remove();
                setTimeout(() => URL.revokeObjectURL(url), 1000);
            };
            d.appendChild(dl);
            d.appendChild(el("pre", { textContent: JSON.stringify(j.prompt, null, 2) }, "max-height: 260px; overflow: auto; background: #111; padding: 6px; font-size: 10px;"));
            modal.appendChild(d);
        });
        const close = el("button", { className: "cb-button primary", textContent: "Close" }, "margin-top: 8px;");
        close.onclick = () => overlay.remove();
        modal.appendChild(close);
        overlay.appendChild(modal);
        document.body.appendChild(overlay);
    }

    // ---- render -------------------------------------------------------
    function render() {
        body.textContent = "";
        footerCount = null;
        statusLine = null;
        renderSource(body);
        if (loading) body.appendChild(el("div", { textContent: "Analyzing workflow..." }, "color: #999;"));
        if (wf.api_prompt && info) {
            renderOutput(body);
            renderSuggestions(body);
            renderBindings(body);
            renderNodeTable(body);
        } else if (wf.api_prompt && !loading && !parseError) {
            body.appendChild(el("div", { textContent: "Analyzing workflow..." }, "color: #999;"));
        }
        renderControls(body);
    }

    render();
    if (wf.api_prompt) {
        ensureParse(wf).then(p => { info = p; parseError = ""; render(); })
            .catch(e => { parseError = String(e.message || e); render(); });
    }
    // pick up a sweep already running for this session (e.g. started from the Generator node)
    fetchStatus().then(s => {
        if (s && s.sweep_id) {
            sweepId = s.sweep_id; lastStatus = s; paintStatus();
            if (ACTIVE_STATES.includes(s.state)) { startPolling(); rerender(); }
        }
    });
}
