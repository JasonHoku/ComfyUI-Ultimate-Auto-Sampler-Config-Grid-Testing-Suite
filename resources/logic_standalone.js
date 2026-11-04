/**
 * STANDALONE MODE (dashboard opened as a top-level page: /config_tester/dashboard)
 * Everything that talks to the ComfyUI tab (postMessage bridge in web/dashboard.js) is replaced here:
 *  - live updates: own ComfyUI websocket, re-dispatched as the SAME message shapes the bridge sends,
 *    so logic_events.js renders them with no duplicated logic
 *  - "GENERATE NEW" in the lightbox queues through POST /config_tester/run
 *  - the node iframe's header button opens this page via openStandalone()
 * Loaded LAST (after logic_init.js). All behaviour is inside `if (USCG_STANDALONE)`; the node iframe is untouched.
 */

// Header button (node iframe only): open this dashboard as its own page.
function openStandalone() {
    var sess = (document.getElementById('session-input') || {}).value || '';
    var base = document.baseURI;
    if (!base || base.indexOf('http') !== 0) {
        try { base = window.parent.location.href; } catch (e) { base = ''; }
    }
    var url;
    try {
        url = new URL('/config_tester/dashboard' + (sess ? '?session=' + encodeURIComponent(sess) : ''), base).href;
    } catch (e) {
        alert('Could not work out the dashboard URL: ' + e.message);
        return;
    }
    // No `noopener` feature: it makes window.open() always return null, hiding blocked popups.
    var w = window.open(url, '_blank');
    if (w) {
        try { w.opener = null; } catch (e) { /* cross-origin guard */ }
    } else {
        prompt('Open this URL in your browser', url);
    }
}

// URL of the session this page shows (landing page -> landing URL), for sharing.
function uscgSessionUrl() {
    var sess = window._uscgLoadedSession || '';
    return new URL('/config_tester/dashboard' + (sess ? '?session=' + encodeURIComponent(sess) : ''), location.href).href;
}

// Share (phone share sheet) or copy the session link. Plain http over a Tailscale IP is not a secure
// context, so the execCommand path is the common one there; its boolean result is checked.
async function uscgShareLink(btn) {
    var url = uscgSessionUrl();
    var done = function () {
        if (!btn) return;
        var old = btn.innerHTML;
        btn.textContent = '✓';
        setTimeout(function () { btn.innerHTML = old; }, 1200);
    };
    if (navigator.share) {
        try {
            await navigator.share({ title: document.title, url: url });
            done();
            return;
        } catch (e) {
            if (e && e.name === 'AbortError') return;   // the user closed the share sheet
        }
    }
    if (navigator.clipboard && window.isSecureContext) {
        try { await navigator.clipboard.writeText(url); done(); return; } catch (e) { /* fall through */ }
    }
    var ok = false;
    try {
        var ta = document.createElement('textarea');
        ta.value = url;
        ta.setAttribute('readonly', '');
        ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0;font-size:16px;';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        ok = document.execCommand('copy');
        document.body.removeChild(ta);
    } catch (e) { ok = false; }
    if (ok) done(); else prompt('Copy this link', url);
}

// Phone header 📂 button: open the settings panel at the session list.
function uscgOpenSessionPicker() {
    var panel = document.getElementById('cog-menu-dropdown');
    if (panel && (panel.style.display === 'none' || !panel.style.display) && typeof toggleCogMenu === 'function') toggleCogMenu();
    var list = document.getElementById('session-picker-list');
    if (list && list.scrollIntoView) list.scrollIntoView({ block: 'nearest' });
    // only focus the search box with a real keyboard (a phone would pop the on-screen one)
    if (window.matchMedia && matchMedia('(pointer: fine)').matches) {
        var s = document.getElementById('session-search');
        if (s) s.focus();
    }
}

// Lightbox "GENERATE NEW" without a ComfyUI graph: queue via the run API into the session being viewed.
async function uscgRunViaApi(btn, configOutput) {
    var sess = window._uscgLoadedSession || (document.getElementById('session-input') || {}).value || '';
    var original = btn ? btn.innerText : '';
    var flash = function (text, ms) {
        if (!btn) return;
        btn.innerText = text;
        setTimeout(function () { btn.innerText = original || 'GENERATE NEW'; }, ms);
    };
    try {
        var body = { configs_json: configOutput, force: true };
        if (sess) body.session_name = sess;
        var r = await fetch('/config_tester/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        var j = null;
        try { j = await r.json(); } catch (e) { /* non-JSON error body */ }
        if (!r.ok) {
            var msg = j && j.error ? (typeof j.error === 'string' ? j.error : (j.error.message || JSON.stringify(j.error))) : ('HTTP ' + r.status);
            console.error('[Revise] /config_tester/run failed:', j);
            flash('ERROR: ' + String(msg).slice(0, 60), 3000);
            return;
        }
        console.log('[Revise] queued via /config_tester/run:', j.prompt_id);
        flash('QUEUED!', 1000);
    } catch (e) {
        console.error('[Revise] run API error:', e);
        flash('ERROR: ' + (e && e.message ? e.message : e), 3000);
    }
}

if (USCG_STANDALONE) {
    (function () {
        document.body.classList.add('uscg-standalone');

        var openBtn = document.getElementById('open-standalone-btn');
        if (openBtn) openBtn.style.display = 'none';
        ['session-switch-btn', 'share-link-btn'].forEach(function (id) {
            var b = document.getElementById(id);
            if (b) b.style.display = '';
        });

        var sendBtn = document.getElementById('modal-send-loadimage-btn');
        if (sendBtn) {
            sendBtn.textContent = '\u2B07 Download image';
            sendBtn.title = 'Save this image (dragging it to the desktop also works)';
        }

        var pack = document.getElementById('pack-workflow-checkbox');
        if (pack) {
            pack.checked = false;
            pack.disabled = true;
            pack.title = 'Needs the ComfyUI tab (open the dashboard from the node to pack the workflow)';
        }

        // The session this page currently shows ('' on the landing page).
        var titleEl = document.getElementById('session-input');
        window._uscgLoadedSession = (titleEl && titleEl.value) ? titleEl.value : '';
        if (window._uscgLoadedSession) document.title = 'USCG - ' + window._uscgLoadedSession;

        function knownIds() {
            var s = new Set();
            var items = (typeof fullManifest !== 'undefined' && fullManifest && fullManifest.items) || [];
            for (var i = 0; i < items.length; i++) s.add(items[i].id);
            return s;
        }

        function dispatch(type, data) {
            window.postMessage({ type: type, data: data }, location.origin);
        }

        function onEvent(name, data) {
            if (!data) return;
            var isUpdate = (name === 'ultimate_grid.update' || name === 'ultimate_grid.update_data');
            var isProgress = (name === 'ultimate_grid.progress');
            if (!isUpdate && !isProgress) return;

            // Landing page: follow a newly started session (mirrors the node bridge's "no session loaded" case).
            if (!window._uscgLoadedSession) {
                if (name === 'ultimate_grid.update' && data.session_name) {
                    loadSessionFromPicker(data.session_name);
                }
                return;
            }
            if (data.session_name !== window._uscgLoadedSession) return;

            if (isProgress) { dispatch('progress_update', data); return; }

            if (data.new_items && data.new_items.length) {
                // processNewData does not dedupe: a resume reload plus a late event would double the cards.
                var have = knownIds();
                var fresh = data.new_items.filter(function (it) { return !have.has(it.id); });
                if (!fresh.length) return;
                var copy = {};
                for (var k in data) copy[k] = data[k];
                copy.new_items = fresh;
                dispatch('update_data', copy);
            } else if (data.manifest) {
                dispatch('update_data', data);
            }
        }

        // ---- websocket with backoff + resync on resume -------------------------------------
        var ws = null, backoff = 1000, dropped = false, resyncTimer = null;
        var clientId = 'uscg-dash-' + Math.random().toString(16).slice(2, 10) + Date.now().toString(16);

        function resync() {
            if (!window._uscgLoadedSession) return;
            if (resyncTimer) return;  // debounce: a reconnect and a visibility change must not both reload
            resyncTimer = setTimeout(function () {
                resyncTimer = null;
                if (navigator.onLine === false) return;
                // a background resync must not close the settings panel / session picker the user has open
                window._uscgResyncing = true;
                Promise.resolve(loadSession()).finally(function () { window._uscgResyncing = false; });
            }, 2000);
        }

        function connect() {
            var url = (location.protocol === 'https:' ? 'wss' : 'ws') + '://' + location.host + '/ws?clientId=' + clientId;
            try { ws = new WebSocket(url); } catch (e) { schedule(); return; }
            ws.onopen = function () {
                backoff = 1000;
                if (dropped) { dropped = false; resync(); }
            };
            ws.onmessage = function (ev) {
                if (typeof ev.data !== 'string') return;  // binary preview frames
                var msg;
                try { msg = JSON.parse(ev.data); } catch (e) { return; }
                if (msg && msg.type) onEvent(msg.type, msg.data);
            };
            ws.onclose = function () { dropped = true; schedule(); };
            ws.onerror = function () { try { ws.close(); } catch (e) { /* already closed */ } };
        }

        function schedule() {
            setTimeout(connect, backoff);
            backoff = Math.min(backoff * 2, 30000);
        }

        document.addEventListener('visibilitychange', function () {
            if (document.visibilityState === 'visible') resync();
        });

        connect();
    })();
}
