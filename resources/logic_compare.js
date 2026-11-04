/**
 * PAIRWISE REVIEW - DOM + STATE GLUE
 * Compare overlay, vote/undo/reveal flow, persistence queue, scoreboard modal,
 * settings bindings, and the rating getters used by the grid (sort + label badge).
 * Pure logic lives in logic_ratings.js (no DOM there). Everything here is prefixed
 * cmp / _cmp because all logic_*.js files share one inlined <script> scope.
 */

var _cmpSettings = { mode: 'matched', axis: '', reveal: 1000, blind: true, restrict: true };

var _cmp = {
    open: false,
    src: null,
    srcKey: '',
    state: { votesById: new Map(), votedPairs: new Set(), skipped: new Set(), allowRepeats: false },
    current: null,       // {a, b (items), repeating, axis}
    next: null,          // prefetched pair
    revealing: false,
    revealTimer: null,
    unblind: false,
    sessionVotes: 0,
    undoStack: [],       // votes cast in this compare session (objects living in comparisons)
    lastTs: 0,
    pairsEstimate: 0
};

// Persistence queue
var _cmpPendingAppend = [];
var _cmpPendingRemove = [];
var _cmpInflightKeys = new Set();
var _cmpFlushTimer = null;
var _cmpInFlight = false;
var _cmpRetryMs = 0;
var _cmpSinceFlush = 0;
var _cmpSess = '';
var _cmpLastServer = [];     // last list known to be on disk for _cmpSess
var _cmpVersion = 0;         // bumped on any change of fullManifest.comparisons
var _cmpRatingCache = { version: -1, n: -1, byId: new Map() };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function cmpSessionName() {
    var el = document.getElementById('session-input');
    return (el && el.value) || 'default';
}

function cmpList() {
    if (!fullManifest) return [];
    if (!Array.isArray(fullManifest.comparisons)) fullManifest.comparisons = [];
    return fullManifest.comparisons;
}

function cmpSetStatus(msg, isError) {
    var el = document.getElementById('cmp-status');
    if (!el) return;
    el.textContent = msg || '';
    el.style.color = isError ? '#ff5555' : '#666';
}

function cmpBump() {
    _cmpVersion++;
}

function cmpItemsMap() {
    var m = new Map();
    var items = (fullManifest && fullManifest.items) || [];
    for (var i = 0; i < items.length; i++) m.set(items[i].id, items[i]);
    return m;
}

// ---------------------------------------------------------------------------
// Ratings (used by the Rating sort and the Rating label badge)
// ---------------------------------------------------------------------------
function cmpRatings() {
    var n = fullManifest && fullManifest.items ? fullManifest.items.length : 0;
    if (_cmpRatingCache.version !== _cmpVersion || _cmpRatingCache.n !== n) {
        var ids = new Set();
        var items = (fullManifest && fullManifest.items) || [];
        for (var i = 0; i < items.length; i++) ids.add(items[i].id);
        _cmpRatingCache = { version: _cmpVersion, n: n, byId: computeBradleyTerry(cmpList(), ids) };
    }
    return _cmpRatingCache.byId;
}

function getItemRating(id) {
    return cmpRatings().get(id) || null;
}

/** Sort comparator: rated items first (descending), unrated last, never NaN. */
function compareRatings(a, b) {
    var ra = getItemRating(a.id), rb = getItemRating(b.id);
    if (!ra && !rb) return 0;
    if (!ra) return 1;
    if (!rb) return -1;
    var d = rb.rating - ra.rating;
    return isFinite(d) ? d : 0;
}

// ---------------------------------------------------------------------------
// Settings (cog panel)
// ---------------------------------------------------------------------------
function cmpLoadSettings() {
    try {
        var raw = localStorage.getItem('ultimate_grid_compare');
        if (raw) {
            var p = JSON.parse(raw);
            if (['matched', 'seed', 'one_axis', 'random'].indexOf(p.mode) >= 0) _cmpSettings.mode = p.mode;
            if (typeof p.axis === 'string' && (p.axis === '' || CMP_AXES.indexOf(p.axis) >= 0)) _cmpSettings.axis = p.axis;
            if ([0, 1000, 2000].indexOf(+p.reveal) >= 0) _cmpSettings.reveal = +p.reveal;
            if (typeof p.blind === 'boolean') _cmpSettings.blind = p.blind;
            if (typeof p.restrict === 'boolean') _cmpSettings.restrict = p.restrict;
        }
    } catch (e) { /* ignore */ }
}

function cmpSaveSettings() {
    try { localStorage.setItem('ultimate_grid_compare', JSON.stringify(_cmpSettings)); } catch (e) { /* ignore */ }
}

function cmpSyncSettingsUI() {
    var set = function (id, fn) { var el = document.getElementById(id); if (el) fn(el); };
    set('cmp-mode', function (el) { el.value = _cmpSettings.mode; });
    set('cmp-axis', function (el) { el.value = _cmpSettings.axis; });
    set('cmp-reveal', function (el) { el.value = String(_cmpSettings.reveal); });
    set('cmp-blind', function (el) { el.checked = _cmpSettings.blind; });
    set('cmp-restrict', function (el) { el.checked = _cmpSettings.restrict; });
}

function cmpSettingsChanged() {
    var v = function (id) { var el = document.getElementById(id); return el; };
    var mode = v('cmp-mode'), axis = v('cmp-axis'), reveal = v('cmp-reveal'), blind = v('cmp-blind'), restrict = v('cmp-restrict');
    if (mode) _cmpSettings.mode = mode.value;
    if (axis) _cmpSettings.axis = axis.value;
    if (reveal) _cmpSettings.reveal = +reveal.value;
    if (blind) _cmpSettings.blind = blind.checked;
    if (restrict) _cmpSettings.restrict = restrict.checked;
    cmpSaveSettings();
    if (_cmp.open) {
        cmpEnsureSource(true);
        cmpShowNext(true);
    }
}

function initCompareUI() {
    cmpLoadSettings();
    cmpSyncSettingsUI();
    _cmpSess = cmpSessionName();
    window.addEventListener('pagehide', function () { cmpFlushNow(true); });
    document.addEventListener('visibilitychange', function () {
        if (document.visibilityState === 'hidden') cmpFlushNow(true);
    });
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------
function cmpScheduleFlush() {
    if (_cmpFlushTimer) clearTimeout(_cmpFlushTimer);
    _cmpFlushTimer = setTimeout(function () { _cmpFlushTimer = null; cmpFlushNow(false); }, 1500);
}

function cmpApplyPending(list) {
    var out = mergeComparisons(list, _cmpPendingAppend);
    if (_cmpPendingRemove.length) {
        var rm = new Set(_cmpPendingRemove);
        out = out.filter(function (c) { return !rm.has(cmpVoteKey(c)); });
    }
    return out;
}

async function cmpFlushNow(keepalive) {
    if (_cmpFlushTimer) { clearTimeout(_cmpFlushTimer); _cmpFlushTimer = null; }
    if (_cmpInFlight) return;
    if (!_cmpPendingAppend.length && !_cmpPendingRemove.length) return;
    var sentA = _cmpPendingAppend.slice();
    var sentR = _cmpPendingRemove.slice();
    var session = _cmpSess || cmpSessionName();
    _cmpInFlight = true;
    _cmpInflightKeys = new Set(sentA.map(cmpVoteKey));
    try {
        var res = await fetch('/config_tester/save_comparisons', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            keepalive: !!keepalive,
            body: JSON.stringify({ session_name: session, append: sentA, remove: sentR })
        });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        var data = await res.json();
        var sentASet = new Set(sentA);
        _cmpPendingAppend = _cmpPendingAppend.filter(function (c) { return !sentASet.has(c); });
        var sentRSet = new Set(sentR);
        _cmpPendingRemove = _cmpPendingRemove.filter(function (k) { return !sentRSet.has(k); });
        _cmpSinceFlush = 0;
        _cmpRetryMs = 0;
        if (session === cmpSessionName()) {
            _cmpLastServer = Array.isArray(data.comparisons) ? data.comparisons : _cmpLastServer;
            if (fullManifest) fullManifest.comparisons = cmpApplyPending(_cmpLastServer);
            cmpBump();
            cmpRefreshAfterSync();
        }
        cmpSetStatus('Votes saved (' + (data.total !== undefined ? data.total : cmpList().length) + ' total)', false);
    } catch (e) {
        _cmpRetryMs = Math.min(_cmpRetryMs ? _cmpRetryMs * 2 : 2000, 30000);
        cmpSetStatus('Saving votes failed (' + (e && e.message || e) + '), retrying in ' + Math.round(_cmpRetryMs / 1000) + 's', true);
        if (_cmpFlushTimer) clearTimeout(_cmpFlushTimer);
        _cmpFlushTimer = setTimeout(function () { _cmpFlushTimer = null; cmpFlushNow(false); }, _cmpRetryMs);
    } finally {
        _cmpInFlight = false;
        _cmpInflightKeys = new Set();
        // votes cast while the request was running
        if ((_cmpPendingAppend.length || _cmpPendingRemove.length) && !_cmpFlushTimer) cmpScheduleFlush();
    }
}

/** Called after the manifest was replaced (loadSession / full-manifest push). */
function onComparisonsReplaced() {
    var now = cmpSessionName();
    if (_cmpSess && _cmpSess !== now) {
        // Session switched: send what is still queued to the OLD session (best effort), then reset
        cmpFlushNow(true);
        _cmpPendingAppend = [];
        _cmpPendingRemove = [];
        _cmpLastServer = [];
        if (_cmp.open) cmpCloseCompare(true);
    }
    _cmpSess = now;
    if (fullManifest) {
        if (Array.isArray(fullManifest.comparisons)) _cmpLastServer = fullManifest.comparisons;
        else fullManifest.comparisons = _cmpLastServer.slice();   // e.g. a generation push without votes
        fullManifest.comparisons = cmpApplyPending(fullManifest.comparisons);
    }
    cmpBump();
    cmpRefreshAfterSync();
}

function cmpRefreshAfterSync() {
    if (_cmp.open) {
        cmpRebuildVoteState();
        cmpUpdateProgress();
    }
}

function cmpRecordVote(c) {
    cmpList().push(c);
    _cmpPendingAppend.push(c);
    _cmpSess = cmpSessionName();
    cmpBump();
    _cmpSinceFlush++;
    if (_cmpSinceFlush >= 20) cmpFlushNow(false); else cmpScheduleFlush();
}

function cmpRetractVote(c) {
    var key = cmpVoteKey(c);
    var list = cmpList();
    for (var i = list.length - 1; i >= 0; i--) {
        if (list[i] === c || cmpVoteKey(list[i]) === key) { list.splice(i, 1); break; }
    }
    var idx = _cmpPendingAppend.indexOf(c);
    if (idx >= 0 && !_cmpInflightKeys.has(key)) _cmpPendingAppend.splice(idx, 1);
    else {
        if (idx >= 0) _cmpPendingAppend.splice(idx, 1);
        _cmpPendingRemove.push(key);   // already sent (or in flight): ask the server to drop it
    }
    cmpBump();
    cmpScheduleFlush();
}

async function clearComparisons() {
    var n = cmpList().length;
    if (!n) { cmpSetStatus('No votes to clear', false); return; }
    if (!window.confirm('Delete all ' + n + ' votes saved in this session? This cannot be undone.')) return;
    try {
        var res = await fetch('/config_tester/clear_comparisons', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_name: cmpSessionName() })
        });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        _cmpPendingAppend = [];
        _cmpPendingRemove = [];
        _cmpLastServer = [];
        if (fullManifest) fullManifest.comparisons = [];
        _cmp.undoStack = [];
        _cmp.state.skipped.clear();
        cmpBump();
        cmpRefreshAfterSync();
        cmpAfterRatingsChanged();
        cmpSetStatus('Cleared ' + n + ' votes', false);
    } catch (e) {
        cmpSetStatus('Clearing votes failed: ' + (e && e.message || e), true);
    }
}

function cmpAfterRatingsChanged() {
    if (typeof refreshLabelOverlays === 'function') refreshLabelOverlays();
    if ((currentSort === 'rating' || currentSecondarySort === 'rating') && typeof updateDataPipeline === 'function') {
        updateDataPipeline();
    }
}

// ---------------------------------------------------------------------------
// Pair source / queue
// ---------------------------------------------------------------------------
function cmpBuildPool() {
    var out = [];
    var src = processedData || [];
    for (var i = 0; i < src.length; i++) {
        var d = src[i];
        if (d && d.file && !d.rejected && !d.florence2_no_detection) out.push(d);   // a reject is a decision already made
    }
    return out;
}

function cmpRebuildVoteState() {
    var ids = new Set();
    var items = (fullManifest && fullManifest.items) || [];
    for (var i = 0; i < items.length; i++) ids.add(items[i].id);
    var votesById = new Map(), votedPairs = new Set();
    var list = cmpList();
    for (var j = 0; j < list.length; j++) {
        var c = list[j];
        if (!c || !ids.has(c.a) || !ids.has(c.b) || c.a === c.b) continue;
        votedPairs.add(cmpPairKey(c.a, c.b));
        votesById.set(c.a, (votesById.get(c.a) || 0) + 1);
        votesById.set(c.b, (votesById.get(c.b) || 0) + 1);
    }
    _cmp.state.votesById = votesById;
    _cmp.state.votedPairs = votedPairs;
    cmpComputeRemaining();
}

function cmpComputeRemaining() {
    if (!_cmp.src) { _cmp.pairsEstimate = 0; return; }
    var inPool = new Set();
    _cmp.src.buckets.forEach(function (b) { b.items.forEach(function (e) { inPool.add(e.id); }); });
    var voted = 0;
    _cmp.state.votedPairs.forEach(function (k) {
        var p = k.split('|');
        if (inPool.has(+p[0]) && inPool.has(+p[1])) voted++;
    });
    _cmp.pairsEstimate = Math.max(0, _cmp.src.estimate - voted);
}

function cmpEnsureSource(force) {
    var key = '';
    try { key = getFilterKey(); } catch (e) { key = ''; }
    key += '|' + activeData.length + '|' + _cmpSettings.mode + '|' + _cmpSettings.axis;
    if (!force && _cmp.src && _cmp.srcKey === key) return;
    _cmp.srcKey = key;
    _cmp.src = buildPairSource(cmpBuildPool(), _cmpSettings.mode, _cmpSettings.axis, { meta: meta });
    _cmp.next = null;
    cmpRebuildVoteState();
}

function cmpPickPair() {
    var p = _cmp.next;
    _cmp.next = null;
    if (p) {
        var k = cmpPairKey(p.a.id, p.b.id);
        var stale = _cmp.state.skipped.has(k) || (!p.repeating && _cmp.state.votedPairs.has(k));
        if (!stale) return p;
    }
    return nextPair(_cmp.src, _cmp.state, Math.random);
}

function cmpPrefetch() {
    _cmp.next = nextPair(_cmp.src, _cmp.state, Math.random);
    if (_cmp.next) {
        [_cmp.next.a, _cmp.next.b].forEach(function (d) {
            if (d.media_type !== 'video') { var im = new Image(); im.src = d.file; }
        });
    }
}

// ---------------------------------------------------------------------------
// Overlay rendering
// ---------------------------------------------------------------------------
function cmpModeLabel() {
    var m = _cmpSettings.mode;
    if (m === 'matched') return 'Matched pairs';
    if (m === 'seed') return 'Seed pairs';
    if (m === 'random') return 'Random pairs';
    return 'One axis' + (_cmpSettings.axis ? ': ' + _cmpSettings.axis : '');
}

function cmpUpdateProgress() {
    var el = document.getElementById('cmp-progress');
    if (!el) return;
    var total = 0;
    var list = cmpList();
    for (var i = 0; i < list.length; i++) total++;
    var poolN = _cmp.src ? _cmp.src.poolSize : 0;
    var rep = document.getElementById('cmp-repeat-badge');
    if (rep) rep.style.display = (_cmp.current && _cmp.current.repeating) ? '' : 'none';
    el.textContent = 'Session ' + _cmp.sessionVotes + ' · Total ' + total + ' · Pool ' + poolN.toLocaleString() +
        ' items · ~' + Math.round(_cmp.pairsEstimate).toLocaleString() + ' unvoted pairs';
    var ml = document.getElementById('cmp-mode-label');
    if (ml) ml.textContent = cmpModeLabel();
    var ub = document.getElementById('cmp-unblind-btn');
    if (ub) ub.classList.toggle('active', _cmp.unblind);
}

function cmpFillMedia(el, d, asVideo) {
    el.textContent = '';
    var m;
    if (asVideo) {
        m = document.createElement('video');
        m.src = d.file;
        m.autoplay = true; m.muted = true; m.loop = true; m.playsInline = true;
        m.setAttribute('muted', ''); m.setAttribute('playsinline', '');
        var p = m.play && m.play(); if (p && p.catch) p.catch(function () { });
    } else {
        m = document.createElement('img');
        m.src = d.file;
        m.draggable = false;
    }
    el.appendChild(m);
}

function cmpRenderChips(show) {
    var ca = document.getElementById('cmp-chips-a'), cb = document.getElementById('cmp-chips-b');
    if (ca) ca.textContent = '';
    if (cb) cb.textContent = '';
    if (!show || !_cmp.current) return;
    var A = _cmp.current.a, B = _cmp.current.b;
    var df = axisDiff(A, B, { meta: meta });
    var sa = itemSig(A, meta), sb = itemSig(B, meta);
    var add = function (host, axis, sig) {
        var s = document.createElement('span');
        s.className = 'cmp-chip';
        s.textContent = axis + ': ' + sig.labels[axis];
        s.title = axis + ': ' + sig.axes[axis];
        host.appendChild(s);
    };
    df.axes.forEach(function (ax) { add(ca, ax, sa); add(cb, ax, sb); });
    if (!df.axes.length || !df.ctxEqual) {
        [ca, cb].forEach(function (h) {
            var s = document.createElement('span');
            s.className = 'cmp-chip';
            s.textContent = df.axes.length ? 'other settings differ' : (df.ctxEqual ? 'identical settings' : 'other settings differ');
            h.appendChild(s);
        });
    }
}

function cmpRenderPair() {
    var pair = _cmp.current;
    var empty = document.getElementById('cmp-empty');
    var panes = document.getElementById('cmp-panes');
    var bar = document.getElementById('cmp-bar');
    if (!pair) {
        if (empty) { empty.style.display = 'flex'; }
        if (panes) panes.style.visibility = 'hidden';
        if (bar) bar.style.visibility = 'hidden';
        cmpUpdateProgress();
        return;
    }
    if (empty) empty.style.display = 'none';
    if (panes) panes.style.visibility = '';
    if (bar) bar.style.visibility = '';
    var video = pair.a.media_type === 'video' && pair.b.media_type === 'video';
    cmpFillMedia(document.getElementById('cmp-media-a'), pair.a, video);
    cmpFillMedia(document.getElementById('cmp-media-b'), pair.b, video);
    ['a', 'b'].forEach(function (s) {
        var p = document.getElementById('cmp-pane-' + s);
        if (p) p.classList.remove('cmp-winner', 'cmp-tie');
    });
    cmpRenderChips(_cmp.unblind || !_cmpSettings.blind);
    cmpUpdateProgress();
}

function cmpEmptyMessage() {
    var el = document.getElementById('cmp-empty');
    if (!el) return;
    var pool = _cmp.src ? _cmp.src.poolSize : 0;
    var msg = 'No comparable pairs for this mode with the current filters.';
    if (_cmp.src && _cmp.src.buckets.length && _cmp.state.skipped.size) {
        msg = 'Every remaining pair was skipped in this compare session.';
    } else {
        var pairs = cmpBuildPool();
        var tries = [['matched', ''], ['seed', ''], ['one_axis', ''], ['random', '']];
        var best = null;
        tries.forEach(function (t) {
            if (t[0] === _cmpSettings.mode && t[1] === _cmpSettings.axis) return;
            var s = buildPairSource(pairs, t[0], t[1], { meta: meta });
            if (s.estimate > 0 && (!best || s.estimate > best.n)) best = { mode: t[0], n: s.estimate };
        });
        var names = { matched: 'Matched', seed: 'Seed pairs', one_axis: 'One axis', random: 'Random' };
        if (best) msg += ' Try the "' + names[best.mode] + '" pairing mode (about ' + Math.round(best.n).toLocaleString() + ' pairs) in Settings > Pairwise Review.';
        else msg += ' The pool has ' + pool + ' item(s); widen the filters or generate more images.';
    }
    el.textContent = msg;
}

function cmpShowNext(rebuilt) {
    if (!_cmp.open) return;
    _cmp.revealing = false;
    if (_cmp.revealTimer) { clearTimeout(_cmp.revealTimer); _cmp.revealTimer = null; }
    _cmp.current = _cmp.src ? cmpPickPair() : null;
    if (!_cmp.current) cmpEmptyMessage();
    cmpRenderPair();
    if (_cmp.current) cmpPrefetch();
}

function cmpShowPair(pair) {
    _cmp.revealing = false;
    if (_cmp.revealTimer) { clearTimeout(_cmp.revealTimer); _cmp.revealTimer = null; }
    _cmp.current = pair;
    var empty = document.getElementById('cmp-empty');
    if (empty) empty.style.display = 'none';
    cmpRenderPair();
    _cmp.next = null;
    cmpPrefetch();
}

// ---------------------------------------------------------------------------
// Vote / skip / undo
// ---------------------------------------------------------------------------
function cmpAdvance() {
    cmpShowNext(false);
}

function cmpVoteSide(side) {
    if (!_cmp.open) return;
    if (_cmp.revealing) { cmpAdvance(); return; }     // tap during reveal: advance, no vote
    var pair = _cmp.current;
    if (!pair || (side !== 'a' && side !== 'b' && side !== 'tie')) return;
    var ts = Math.max(Date.now(), _cmp.lastTs + 1);
    _cmp.lastTs = ts;
    var c = { a: pair.a.id, b: pair.b.id, winner: side, ts: ts, mode: _cmpSettings.mode };
    if (!(_cmpSettings.blind && !_cmp.unblind)) c.blind = false;
    cmpRecordVote(c);
    _cmp.undoStack.push(c);
    _cmp.sessionVotes++;
    var pk = cmpPairKey(c.a, c.b);
    _cmp.state.votedPairs.add(pk);
    _cmp.state.votesById.set(c.a, (_cmp.state.votesById.get(c.a) || 0) + 1);
    _cmp.state.votesById.set(c.b, (_cmp.state.votesById.get(c.b) || 0) + 1);
    cmpComputeRemaining();
    cmpUpdateProgress();
    var ms = _cmpSettings.reveal;
    if (ms > 0) {
        _cmp.revealing = true;
        var pa = document.getElementById('cmp-pane-a'), pb = document.getElementById('cmp-pane-b');
        if (side === 'tie') { pa.classList.add('cmp-tie'); pb.classList.add('cmp-tie'); }
        else (side === 'a' ? pa : pb).classList.add('cmp-winner');
        cmpRenderChips(true);
        _cmp.revealTimer = setTimeout(cmpAdvance, ms);
    } else {
        cmpAdvance();
    }
}

function cmpSkip() {
    if (!_cmp.open) return;
    if (_cmp.revealing) { cmpAdvance(); return; }
    var pair = _cmp.current;
    if (!pair) return;
    _cmp.state.skipped.add(cmpPairKey(pair.a.id, pair.b.id));   // memory only, never persisted
    cmpAdvance();
}

function cmpUndo() {
    if (!_cmp.open || !_cmp.undoStack.length) return;
    var c = _cmp.undoStack.pop();
    cmpRetractVote(c);
    _cmp.sessionVotes = Math.max(0, _cmp.sessionVotes - 1);
    cmpRebuildVoteState();
    var items = cmpItemsMap();
    var A = items.get(c.a), B = items.get(c.b);
    if (A && B) cmpShowPair({ a: A, b: B, repeating: false, axis: null });
    else cmpShowNext(false);
}

function cmpToggleUnblind() {
    _cmp.unblind = !_cmp.unblind;
    if (_cmp.current && !_cmp.revealing) cmpRenderChips(_cmp.unblind || !_cmpSettings.blind);
    cmpUpdateProgress();
}

// ---------------------------------------------------------------------------
// Open / close + keys
// ---------------------------------------------------------------------------
function cmpKeyHandler(e) {
    if (!window._compareOpen) return;
    var t = e.target && e.target.tagName;
    if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT') return;
    var chord = e.ctrlKey || e.metaKey || e.altKey;
    var key = e.key;
    var isUndoChord = (e.ctrlKey || e.metaKey) && !e.altKey && (key === 'z' || key === 'Z');
    if (chord && !isUndoChord) return;            // Ctrl+S etc. keep reaching their own handlers

    var sbOpen = !!document.getElementById('scoreboard-modal');
    var stop = function () { e.preventDefault(); e.stopPropagation(); };

    if (key === 'Escape') {
        stop();
        var sb = document.getElementById('scoreboard-modal');
        if (sb) sb.remove(); else closeCompareMode();
        return;
    }
    if (sbOpen) return;                           // scoreboard on top: only Escape is ours
    if (e.repeat) { stop(); return; }

    var k = key.length === 1 ? key.toLowerCase() : key;
    if (k === 'z' || k === 'Backspace') { stop(); cmpUndo(); return; }
    if (_cmp.revealing) {
        // any other key during the reveal advances and does not vote
        if (key === 'Tab' || key.indexOf('F') === 0 && key.length > 1) return;
        stop(); cmpAdvance(); return;
    }
    switch (k) {
        case 'a': case 'ArrowLeft': stop(); cmpVoteSide('a'); return;
        case 'd': case 'ArrowRight': stop(); cmpVoteSide('b'); return;
        case 's': case 'ArrowDown': stop(); cmpVoteSide('tie'); return;
        case 'w': case 'ArrowUp': stop(); cmpSkip(); return;
        case 'i': stop(); cmpToggleUnblind(); return;
        case 'b': stop(); openScoreboard(); return;
    }
    // swallow everything else (digits would favourite grid cards, f/r would fullscreen/reload)
    if (key.length === 1 || key.indexOf('Arrow') === 0 || key === 'Enter') stop();
}

function openCompareMode() {
    if (_cmp.open) return;
    if (!fullManifest || !(fullManifest.items || []).length) {
        cmpSetStatus('Load a session with images first', true);
        return;
    }
    cmpLoadSettings();
    cmpSyncSettingsUI();
    // close the settings panel if it is open (the overlay is above it anyway)
    var cog = document.getElementById('cog-menu-dropdown');
    if (cog && cog.style.display !== 'none' && cog.style.display !== '' && typeof toggleCogMenu === 'function') toggleCogMenu();
    _cmpSess = cmpSessionName();
    if (!Array.isArray(fullManifest.comparisons)) fullManifest.comparisons = cmpApplyPending(_cmpLastServer.slice());
    _cmp.open = true;
    _cmp.unblind = false;
    _cmp.sessionVotes = 0;
    _cmp.undoStack = [];
    _cmp.state.skipped = new Set();
    _cmp.state.allowRepeats = false;
    _cmp.lastTs = 0;
    window._compareOpen = true;
    window._modalOpen = true;
    document.getElementById('compare-overlay').style.display = 'flex';
    document.removeEventListener('keydown', window._cmpKeyHandler, true);
    window._cmpKeyHandler = cmpKeyHandler;
    document.addEventListener('keydown', window._cmpKeyHandler, true);
    cmpEnsureSource(true);
    cmpShowNext(true);
}

function cmpCloseCompare(silent) {
    if (!_cmp.open) return;
    _cmp.open = false;
    window._compareOpen = false;
    var m = document.getElementById('modal');
    window._modalOpen = !!(m && m.style.display === 'flex');
    if (_cmp.revealTimer) { clearTimeout(_cmp.revealTimer); _cmp.revealTimer = null; }
    _cmp.revealing = false;
    document.removeEventListener('keydown', window._cmpKeyHandler, true);
    window._cmpKeyHandler = null;
    var ov = document.getElementById('compare-overlay');
    if (ov) ov.style.display = 'none';
    ['cmp-media-a', 'cmp-media-b'].forEach(function (id) { var el = document.getElementById(id); if (el) el.textContent = ''; });
    var sb = document.getElementById('scoreboard-modal');
    if (sb && !silent) sb.remove();
    cmpFlushNow(false);
    if (!silent) cmpAfterRatingsChanged();
}

function closeCompareMode() {
    cmpCloseCompare(false);
}

// ---------------------------------------------------------------------------
// Scoreboard modal
// ---------------------------------------------------------------------------
function cmpCopyText(text, btn) {
    var done = function () {
        if (!btn) return;
        var o = btn.textContent;
        btn.textContent = 'Copied';
        setTimeout(function () { btn.textContent = o; }, 1500);
    };
    var fallback = function () {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch (e) { /* ignore */ }
        document.body.removeChild(ta);
        done();
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, fallback);
    } else fallback();
}

function cmpCompareMore(axis) {
    var sb = document.getElementById('scoreboard-modal');
    if (sb) sb.remove();
    if (axis === 'seed') { _cmpSettings.mode = 'seed'; _cmpSettings.axis = ''; }
    else { _cmpSettings.mode = 'one_axis'; _cmpSettings.axis = axis; }
    cmpSaveSettings();
    cmpSyncSettingsUI();
    if (_cmp.open) { cmpEnsureSource(true); cmpShowNext(true); }
    else openCompareMode();
}

function cmpEl(tag, cls, text) {
    var el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined) el.textContent = text;
    return el;
}

function cmpPct(x) { return (x * 100).toFixed(0) + '%'; }

function openScoreboard() {
    var existing = document.getElementById('scoreboard-modal');
    if (existing) existing.remove();
    var items = cmpItemsMap();
    var restrict = null;
    if (_cmpSettings.restrict) {
        restrict = new Set();
        (processedData || []).forEach(function (d) { restrict.add(d.id); });
    }
    var ratings = cmpRatings();
    var sb = computeScoreboard(cmpList(), items, ratings, { restrictIds: restrict, minN: 3, meta: meta });

    var overlay = cmpEl('div', 'analytics-modal-overlay');
    overlay.id = 'scoreboard-modal';
    var popup = cmpEl('div', 'analytics-modal-popup');
    var head = cmpEl('div', 'analytics-modal-header');
    head.style.borderBottomColor = '#e0c040';
    var title = cmpEl('span', 'analytics-modal-title', 'Pairwise scoreboard');
    title.style.color = '#e0c040';
    var closeBtn = cmpEl('button', 'close-popup-btn');
    closeBtn.innerHTML = '&#10005;';
    closeBtn.addEventListener('click', function () { overlay.remove(); });
    head.appendChild(title); head.appendChild(closeBtn);
    popup.appendChild(head);

    var summary = cmpEl('div', 'analytics-summary');
    summary.appendChild(cmpEl('span', '', sb.total + ' votes' + (restrict ? ' inside the current filter' : '')));
    summary.appendChild(cmpEl('span', '', '|'));
    summary.appendChild(cmpEl('span', '', 'win % counts ties as half; best = highest 95% Wilson lower bound with n >= 3'));
    summary.title = 'Pairs are chosen with a preference for low-vote items and respect your filters, so win % is not a controlled effect size.';
    popup.appendChild(summary);

    var body = cmpEl('div', 'analytics-modal-body');
    var order = CMP_AXES.concat(['seed']);
    order.forEach(function (ax) {
        var blk = sb.axes[ax];
        var det = cmpEl('details', 'sb-axis');
        det.open = blk.rows.length > 0;
        var sm = cmpEl('summary', '', ax === 'seed' ? 'seed (noise floor)' : ax);
        sm.appendChild(cmpEl('span', 'sb-meta', blk.matchedVotes + ' matched votes'));
        det.appendChild(sm);
        var anyEnough = blk.rows.some(function (r) { return r.enough; });
        if (!anyEnough) {
            det.appendChild(cmpEl('div', 'sb-empty', 'not enough data (' + blk.matchedVotes + ' matched votes)'));
        }
        if (blk.rows.length) {
            var table = cmpEl('table', 'analytics-table');
            var thead = cmpEl('thead');
            var hr = cmpEl('tr');
            ['value', 'n', 'W-L-T', 'win %', '95% CI', 'mean rating', ''].forEach(function (h) { hr.appendChild(cmpEl('th', 'analytics-th', h)); });
            thead.appendChild(hr); table.appendChild(thead);
            var tb = cmpEl('tbody');
            blk.rows.forEach(function (r) {
                var tr = cmpEl('tr', 'analytics-row' + (r.enough ? '' : ' sb-row-dim') + (r.value === blk.best ? ' sb-row-best' : ''));
                var tdv = cmpEl('td', 'sb-val', r.label); tdv.title = r.value;
                tr.appendChild(tdv);
                tr.appendChild(cmpEl('td', 'sb-num', String(r.n)));
                tr.appendChild(cmpEl('td', 'sb-num', r.wins + '-' + r.losses + '-' + r.ties));
                tr.appendChild(cmpEl('td', 'sb-num', r.enough ? cmpPct(r.winPct) : 'not enough data'));
                tr.appendChild(cmpEl('td', 'sb-num', r.enough ? cmpPct(r.wilsonLow) + '–' + cmpPct(r.wilsonHigh) : ''));
                tr.appendChild(cmpEl('td', 'sb-num', r.meanRating === null ? '' : String(Math.round(r.meanRating))));
                tr.appendChild(cmpEl('td', '', ''));
                tb.appendChild(tr);
            });
            table.appendChild(tb);
            det.appendChild(table);
        }
        var more = cmpEl('div', 'sb-link', ax === 'seed' ? 'compare more seeds' : 'compare more');
        more.addEventListener('click', function () { cmpCompareMore(ax); });
        det.appendChild(more);
        body.appendChild(det);
    });
    popup.appendChild(body);

    var foot = cmpEl('div', 'sb-footer');
    var seedVotes = sb.axes.seed.matchedVotes;
    var txt = sb.attributable + ' of ' + sb.total + ' votes attributable to exactly one axis';
    if (seedVotes) txt += ' (+ ' + seedVotes + ' seed-only votes in the noise-floor block)';
    foot.appendChild(cmpEl('div', '', txt));
    if (sb.topExtraKeys.length) {
        foot.appendChild(cmpEl('div', '', 'Other differences in the remaining votes: ' +
            sb.topExtraKeys.map(function (k) { return k.key + ' (' + k.count + ')'; }).join(', ')));
    }
    popup.appendChild(foot);

    var actions = cmpEl('div', 'sb-actions');
    var bj = cmpEl('button', '', 'Copy JSON'), bc = cmpEl('button', '', 'Copy CSV'), bx = cmpEl('button', '', 'Close');
    bj.addEventListener('click', function () { cmpCopyText(JSON.stringify(sb, null, 2), bj); });
    bc.addEventListener('click', function () { cmpCopyText(scoreboardToCSV(sb), bc); });
    bx.addEventListener('click', function () { overlay.remove(); });
    actions.appendChild(bj); actions.appendChild(bc); actions.appendChild(bx);
    popup.appendChild(actions);

    overlay.appendChild(popup);
    overlay.addEventListener('click', function (e) { if (e.target === overlay) overlay.remove(); });
    document.body.appendChild(overlay);

    // close the settings panel so the modal is not stacked over it
    var cog = document.getElementById('cog-menu-dropdown');
    if (cog && cog.style.display !== 'none' && cog.style.display !== '' && typeof toggleCogMenu === 'function') toggleCogMenu();
}
