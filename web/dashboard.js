import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "UltimateGrid.Dashboard",

    setup(app) {
        // 1. INJECT CSS (once only)
        if (!document.getElementById('ultimate-grid-css')) {
            const style = document.createElement('style');
            style.id = 'ultimate-grid-css';
            style.innerHTML = `
                .dashboard-fullscreen {
                    position: fixed !important;
                    top: 0; left: 0; width: 100vw !important; height: 100vh !important;
                    z-index: 99999 !important;
                    background: #0b0b0b;
                    border: none !important;
                }
            `;
            document.head.appendChild(style);
        }

        // 2. LIVE UPDATE LISTENER (SINGLETON - only register once)
        if (!window.__ultimateGridUpdateListenerInstalled) {
            api.addEventListener("ultimate_grid.update", (event) => {
                const payload = event.detail;
                const { session_name, node } = payload;

                console.log(`[UltimateGrid] Update received for: ${session_name}`);

                // Find all dashboard nodes
                const dashboardNodes = app.graph._nodes.filter(n => n.type === "UltimateGridDashboard");

                dashboardNodes.forEach(dashboardNode => {
                    const sessionWidget = dashboardNode.widgets.find(w => w.name === "session_name");

                    if (!sessionWidget) {
                        console.warn(`[UltimateGrid] Dashboard ${dashboardNode.id} missing session_name widget`);
                        return;
                    }

                    const currentSessionName = sessionWidget.value;

                    // CASE 1: Session matches and dashboard is loaded → send live update
                    if (currentSessionName === session_name &&
                        dashboardNode.loaded_session === session_name &&
                        dashboardNode.iframe?.contentWindow) {

                        console.log(`[UltimateGrid] 📤 Live update → Dashboard ${dashboardNode.id}`);

                        try {
                            dashboardNode.iframe.contentWindow.postMessage({
                                type: 'update_data',
                                data: payload
                            }, '*');
                        } catch (e) {
                            console.error('[UltimateGrid] Failed to send message:', e);
                        }
                    }

                    // CASE 2: Session matches but not loaded yet → auto-load ONLY if no session is loaded
                    // Don't force-load if the user is manually browsing a different session in another tab
                    else if (currentSessionName === session_name &&
                        !dashboardNode.loaded_session) {

                        console.log(`[UltimateGrid] 🔄 Auto-loading session: ${session_name}`);

                        if (dashboardNode.forceLoadSession) {
                            dashboardNode.forceLoadSession(session_name);
                        }
                    }

                    // CASE 3: Dashboard connected to sampler but different session → update
                    else if (currentSessionName !== session_name) {
                        const isConnected = dashboardNode.inputs?.some(input => {
                            const link = app.graph.links[input.link];
                            return link && link.origin_id === parseInt(node);
                        });

                        if (isConnected) {
                            console.log(`[UltimateGrid] 🔗 Connected dashboard → Update to: ${session_name}`);

                            sessionWidget.value = session_name;

                            if (dashboardNode.forceLoadSession) {
                                dashboardNode.forceLoadSession(session_name);
                            }
                        }
                    }
                });
            });

            window.__ultimateGridUpdateListenerInstalled = true;
            console.log('[UltimateGrid] ✅ Update listener installed');
        }

        // PROGRESS/ETA LISTENER (SINGLETON - forward ETA to dashboard iframe)
        if (!window.__ultimateGridProgressListenerInstalled) {
            api.addEventListener("ultimate_grid.progress", (event) => {
                const payload = event.detail;
                const { session_name } = payload;

                const dashboardNodes = app.graph._nodes.filter(n => n.type === "UltimateGridDashboard");
                dashboardNodes.forEach(dashboardNode => {
                    const sessionWidget = dashboardNode.widgets.find(w => w.name === "session_name");
                    if (!sessionWidget) return;

                    if (sessionWidget.value === session_name &&
                        dashboardNode.loaded_session === session_name &&
                        dashboardNode.iframe?.contentWindow) {
                        try {
                            dashboardNode.iframe.contentWindow.postMessage({
                                type: 'progress_update',
                                data: payload
                            }, '*');
                        } catch (e) { /* ignore */ }
                    }
                });
            });
            window.__ultimateGridProgressListenerInstalled = true;
        }

        // DISTRIBUTION STATUS LISTENER (SINGLETON - forward distribution status to dashboard iframe)
        if (!window.__ultimateGridDistributionStatusListenerInstalled) {
            api.addEventListener("ultimate_grid.distribution_status", (event) => {
                const payload = event.detail;
                const { session_name } = payload;

                const dashboardNodes = app.graph._nodes.filter(n => n.type === "UltimateGridDashboard");
                dashboardNodes.forEach(dashboardNode => {
                    const sessionWidget = dashboardNode.widgets.find(w => w.name === "session_name");
                    if (!sessionWidget) return;

                    if (sessionWidget.value === session_name &&
                        dashboardNode.loaded_session === session_name &&
                        dashboardNode.iframe?.contentWindow) {
                        try {
                            dashboardNode.iframe.contentWindow.postMessage({
                                type: 'distribution_status',
                                data: payload
                            }, '*');
                        } catch (e) { /* ignore */ }
                    }
                });
            });
            window.__ultimateGridDistributionStatusListenerInstalled = true;
        }

        // 3. FULLSCREEN + CTRL+S LISTENER (SINGLETON - only register once)
        if (!window.__ultimateGridFullscreenListenerInstalled) {
            window.addEventListener("message", (event) => {
                // Forward Ctrl+S from dashboard iframe to ComfyUI's save workflow
                if (event.data?.type === 'forward_ctrl_s') {
                    document.dispatchEvent(new KeyboardEvent('keydown', {
                        key: 's', code: 'KeyS', ctrlKey: true, bubbles: true, cancelable: true
                    }));
                    return;
                }

                // Full pass for survivors (preview-pass workflow): flip the matching Config Builder to
                // "Preview pass off" and queue the prompt. Feature-detected; the dashboard falls back to an
                // instructions toast when this does not answer (or app.queuePrompt is unavailable).
                // Revise-modal "Send to Load Image" (issue #22): fill the selected / only / a new Load Image node.
                if (event.data?.type === 'uscg_send_to_load_image') {
                    const reply = (ok, message) => {
                        try { event.source?.postMessage({ type: 'uscg_send_to_load_image_result', ok, message }, '*'); } catch (e) { /* ignore */ }
                    };
                    (async () => {
                        try {
                            const d = event.data;
                            const u = new URL(String(d.url || ''), location.href);
                            if (u.origin !== location.origin || u.pathname !== '/view') {
                                reply(false, "Refused: the image URL is not this ComfyUI server's /view.");
                                return;
                            }
                            const filename = String(d.filename || u.searchParams.get('filename') || '');
                            const subfolder = String(d.subfolder || '');
                            const imgType = String(d.imageType || 'output');
                            if (!filename) { reply(false, 'No filename.'); return; }
                            const all = (app.graph?._nodes || []);
                            const loads = all.filter(n => n.type === 'LoadImage');
                            const sel = Object.values(app.canvas?.selected_nodes || {}).filter(n => n.type === 'LoadImage');
                            let node = sel[0] || (loads.length === 1 ? loads[0] : null);
                            if (!node) {
                                node = LiteGraph.createNode('LoadImage');
                                if (!node) { reply(false, 'Could not create a Load Image node.'); return; }
                                const dash = all.find(n => String(n.id) === String(d.node_id));
                                node.pos = dash ? [dash.pos[0] + dash.size[0] + 40, dash.pos[1]] : [60, 60];
                                app.graph.add(node);
                            }
                            const w = node.widgets?.find(x => x.name === 'image');
                            if (!w) { reply(false, 'The Load Image node has no image widget.'); return; }
                            let value = (subfolder ? subfolder + '/' : '') + filename + ' [' + imgType + ']';
                            if (d.upload) {
                                const blob = await (await fetch(u.href)).blob();
                                const fd = new FormData();
                                fd.append('image', new File([blob], filename, { type: blob.type }));
                                fd.append('type', 'input');
                                const r = await api.fetchApi('/upload/image', { method: 'POST', body: fd });
                                if (!r.ok) { reply(false, 'Upload failed (HTTP ' + r.status + ').'); return; }
                                const j = await r.json();
                                value = j.subfolder ? j.subfolder + '/' + j.name : j.name;
                            }
                            const old = w.value;
                            if (w.options?.values && !w.options.values.includes(value)) w.options.values.push(value);
                            w.value = value;
                            w.callback?.(value);
                            node.onWidgetChanged?.(w.name, value, old, w);
                            app.graph.setDirtyCanvas(true, true);
                            reply(true, `Loaded into "${node.title || 'Load Image'}" (#${node.id})`);
                        } catch (e) {
                            reply(false, 'Could not send the image: ' + (e?.message || e));
                        }
                    })();
                    return;
                }

                if (event.data?.type === 'uscg_run_full_pass') {
                    const reply = (ok, message, reason) => {
                        try { event.source?.postMessage({ type: 'uscg_run_full_pass_result', ok, message, reason }, '*'); } catch (e) { /* ignore */ }
                    };
                    try {
                        const san = (s) => String(s || '').replace(/[^\w\-]/g, '');
                        const wanted = san(event.data.session_name);
                        const builders = (app.graph?._nodes || []).filter(n =>
                            n.type === "UltimateConfigBuilder" && n.state && san(n.state.session_name) === wanted);
                        if (builders.length !== 1) {
                            reply(false, builders.length === 0
                                ? `No Config Builder in this workflow uses session "${wanted}".`
                                : `${builders.length} Config Builders use session "${wanted}".`,
                                builders.length === 0 ? 'no_builder' : undefined);
                            return;
                        }
                        if (typeof app.queuePrompt !== 'function') {
                            reply(false, 'This ComfyUI frontend does not expose app.queuePrompt.', 'no_queue');
                            return;
                        }
                        const b = builders[0];
                        if (!b.state.preview_pass || typeof b.state.preview_pass !== 'object') {
                            b.state.preview_pass = { enabled: false, steps: 8, scale: 0.5, cfg_override: null };
                        }
                        b.state.preview_pass.enabled = false;
                        b.state.skip_rejected_previews = true;
                        b.saveState?.();
                        b.renderUI?.();
                        Promise.resolve(app.queuePrompt(0, 1)).then(
                            () => reply(true, 'Queued the full pass (Preview pass switched off in the Config Builder).'),
                            (err) => reply(false, 'Queueing failed: ' + (err?.message || err)));
                    } catch (e) {
                        reply(false, 'Could not queue the full pass: ' + (e?.message || e));
                    }
                    return;
                }

                if (event.data?.type === 'toggle_fullscreen') {
                    const nodeId = parseInt(event.data.node_id);
                    const graphNode = app.graph.getNodeById(nodeId);

                    if (!graphNode?.iframe) {
                        console.warn('[UltimateGrid] Fullscreen toggle failed: node or iframe not found');
                        return;
                    }

                    const el = graphNode.iframe;

                    // Save viewport position before fullscreen toggle
                    // so it can be restored after the iframe reloads
                    try {
                        if (el.contentWindow?.saveViewportPosition) {
                            el.contentWindow.saveViewportPosition();
                        }
                    } catch (e) { /* cross-origin or not loaded yet */ }

                    el.addEventListener('load', () => {
                        console.log('[UltimateGrid] Iframe reloaded after fullscreen toggle, restoring session...');

                        // Get the session name — prefer the one from the iframe (user may have
                        // loaded a different session via the picker), fall back to node widget
                        const sessionName = event.data.session_name
                            || graphNode.widgets?.find(w => w.name === "session_name")?.value
                            || null;

                        if (sessionName && graphNode.forceLoadSession) {
                            graphNode.forceLoadSession(sessionName);
                        }
                    }, { once: true });
                    // Exit fullscreen
                    if (el.classList.contains('dashboard-fullscreen')) {
                        el.classList.remove('dashboard-fullscreen');

                        // Restore to original position
                        if (graphNode.fs_placeholder?.parentNode) {
                            graphNode.fs_placeholder.parentNode.insertBefore(el, graphNode.fs_placeholder);
                            graphNode.fs_placeholder.remove();
                        }

                        graphNode.fs_placeholder = null;
                        el.style.width = (graphNode.size[0] - 20) + "px";
                        el.style.height = "100%";

                        console.log('[UltimateGrid] ↙️ Exited fullscreen');
                    }
                    // Enter fullscreen
                    else {
                        // Create placeholder
                        const placeholder = document.createElement("div");
                        placeholder.style.display = "none";
                        graphNode.fs_placeholder = placeholder;

                        if (el.parentNode) {
                            el.parentNode.insertBefore(placeholder, el);
                        }

                        // Move to body
                        document.body.appendChild(el);
                        el.classList.add('dashboard-fullscreen');

                        console.log('[UltimateGrid] ↗️ Entered fullscreen');
                    }
                }
            });

            window.__ultimateGridFullscreenListenerInstalled = true;
            console.log('[UltimateGrid] ✅ Fullscreen listener installed');
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "UltimateGridDashboard") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                this.resizable = true;
                const node = this;
                node.loaded_session = null;
                node.is_fetching = false;

                // Helper: Get current session name
                const getSessionName = () => {
                    const widget = node.widgets.find(w => w.name === "session_name");
                    return widget ? widget.value : null;
                };

                /**
                 * Force load session - FIXED (no 404 call!)
                 */
                this.forceLoadSession = async (sessionName) => {
                    if (node.is_fetching) return;
                    node.is_fetching = true;
                    try {
                        const response = await fetch("/config_tester/get_session_html", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ session_name: sessionName, node_id: node.id })
                        });
                        if (response.ok) {
                            const html = await response.text();
                            if (node.iframe) {
                                node.iframe.srcdoc = html;
                                node.loaded_session = sessionName;
                            }
                        }
                    } catch (e) { console.error(e); }
                    finally { node.is_fetching = false; }
                };;

                // Button: Reload/Show Session
                this.addWidget("button", "RELOAD / SHOW SESSION", null, () => {
                    const sessionName = getSessionName();
                    if (sessionName) {
                        this.forceLoadSession(sessionName);
                    } else {
                        alert("No session name set.");
                    }
                });

                // Button: Delete Session
                this.addWidget("button", "DELETE SESSION (Files)", null, async () => {
                    const sessionName = getSessionName();

                    if (!sessionName) {
                        alert("No session name set.");
                        return;
                    }

                    if (!confirm(`Are you sure you want to delete session "${sessionName}"?\n\nThis will delete all images and data.`)) {
                        return;
                    }

                    try {
                        const response = await fetch("/config_tester/delete_session", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ session_name: sessionName })
                        });

                        if (response.ok) {
                            node.iframe.srcdoc = "<h3 style='color:#888; text-align:center; padding:20px;'>Session Deleted.</h3>";
                            node.loaded_session = null;
                            alert(`Session "${sessionName}" deleted successfully.`);
                        } else {
                            const errorText = await response.text();
                            alert(`Error deleting session: ${errorText}`);
                        }
                    } catch (error) {
                        console.error('[UltimateGrid] Delete failed:', error);
                        alert(`Connection error: ${error.message}`);
                    }
                });

                /**
                 * onExecuted - Handle dashboard HTML from server
                 */
                this.onExecuted = function (message) {
                    if (message?.html) {
                        const currentSession = getSessionName();

                        // Optimization: If session already loaded, try incremental update
                        if (node.loaded_session === currentSession) {
                            const newHtml = message.html[0];

                            // Try to extract just the manifest data
                            const pattern = /\/\*__JSON_START__\*\/\s*let fullManifest = (\{[\s\S]*?\});\s*\/\*__JSON_END__\*\//;
                            const match = newHtml.match(pattern);

                            if (match && match[1]) {
                                try {
                                    const newManifest = JSON.parse(match[1]);

                                    // Send incremental update instead of full reload
                                    if (node.iframe?.contentWindow) {
                                        node.iframe.contentWindow.postMessage({
                                            type: 'update_data',
                                            data: {
                                                manifest: newManifest,
                                                meta: newManifest.meta || {}
                                            }
                                        }, '*');

                                        console.log('[UltimateGrid] 📊 Incremental update sent');
                                        return; // Skip full HTML reload
                                    }
                                } catch (error) {
                                    console.warn('[UltimateGrid] Incremental update failed, doing full reload:', error);
                                }
                            }
                        }

                        // Full reload (when incremental fails or first load)
                        node.iframe.srcdoc = message.html[0];
                        node.loaded_session = currentSession;
                        console.log('[UltimateGrid] 🔄 Full reload');
                    }

                    // Update session name widget if server sent new name
                    if (message?.update_session_name && message.update_session_name[0]) {
                        const widget = node.widgets.find(w => w.name === "session_name");
                        if (widget) {
                            widget.value = message.update_session_name[0];
                            node.setDirtyCanvas(true, true);
                        }
                    }
                };

                /**
                 * Create iframe widget
                 */
                const iframe = document.createElement("iframe");
                Object.assign(iframe.style, {
                    border: "none",
                    background: "#0b0b0b",
                    width: "100%",
                    height: "100%",
                    display: "block",
                    borderRadius: "0 0 4px 4px"
                });

                this.iframe = iframe;
                this.addDOMWidget("dashboard_viewer", "iframe", iframe);

                // Handle resize
                this.onResize = function (size) {
                    node.setDirtyCanvas(true, true);
                };

                // Set default size
                this.setSize([900, 750]);

                // Auto-load: show landing page for new nodes, or saved session
                // for workflows loaded from disk. onConfigure fires synchronously
                // before the setTimeout, so we can use a flag to distinguish.
                node._configured_from_workflow = false;
                setTimeout(() => {
                    if (node._configured_from_workflow) {
                        // Workflow loaded — show the saved session
                        const savedSession = getSessionName();
                        if (savedSession) {
                            this.forceLoadSession(savedSession);
                        } else {
                            this.forceLoadSession("");
                        }
                    } else {
                        // Fresh node — show landing page with session cards
                        this.forceLoadSession("");
                    }
                }, 500);

                // onConfigure fires when loading a saved workflow (before setTimeout)
                const origOnConfigure = this.onConfigure;
                this.onConfigure = function (config) {
                    if (origOnConfigure) origOnConfigure.apply(this, arguments);
                    node._configured_from_workflow = true;
                };

                // Cleanup on node removal
                const originalOnRemoved = this.onRemoved;
                this.onRemoved = function () {
                    // Clean up iframe
                    if (node.iframe) {
                        node.iframe.remove();
                        node.iframe = null;
                    }

                    // Clean up placeholder
                    if (node.fs_placeholder) {
                        node.fs_placeholder.remove();
                        node.fs_placeholder = null;
                    }

                    console.log('[UltimateGrid] 🧹 Node cleanup complete');

                    if (originalOnRemoved) {
                        originalOnRemoved.apply(this, arguments);
                    }
                };

                return r;
            };
        }
    },
});

console.log('[UltimateGrid] ✅ Dashboard extension loaded');