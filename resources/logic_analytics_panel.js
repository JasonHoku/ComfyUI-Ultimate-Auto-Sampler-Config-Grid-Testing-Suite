/**
 * SETTINGS STATS PANEL (DOM)
 * A docked, non-modal panel (header bar button / cog entry) that shows, per setting axis, how often each value
 * was favorited / rejected, its mean duration, mean metrics and mean pairwise rating. The numbers come from
 * computeAnalytics() (logic_analytics.js). Loaded after logic_compare.js: uses cmpEl, cmpCopyText, cmpRatings.
 * Every node is built with textContent (manifest strings never reach innerHTML).
 *
 * Refresh hooks: analyticsPanelOnDataChanged() is called from updateJSONs() (every pipeline run / favorite /
 * reject / new item) and cmpAfterRatingsChanged() (votes); it coalesces renders to once per 250 ms.
 */

var _anPanel = { open: false, collapsed: false, dirty: false, raf: 0, lastRender: 0, scope: 'filter', sortCol: 'favRate', sortDir: -1, openAxes: null };
var AN_STORE_KEY = 'ultimate_grid_stats';
var AN_ROW_CAP = 50;
var AN_RENDER_GAP_MS = 250;
var AN_DEFAULT_OPEN = ['model', 'lora', 'sampler', 'scheduler', 'steps', 'cfg'];

function anLoadSettings() {
    try {
        var s = JSON.parse(localStorage.getItem(AN_STORE_KEY) || 'null');
        if (!s || typeof s !== 'object') return;
        if (s.scope === 'all' || s.scope === 'filter') _anPanel.scope = s.scope;
        if (typeof s.collapsed === 'boolean') _anPanel.collapsed = s.collapsed;
        if (typeof s.sortCol === 'string' && s.sortCol.length < 80) _anPanel.sortCol = s.sortCol;
        if (s.sortDir === 1 || s.sortDir === -1) _anPanel.sortDir = s.sortDir;
    } catch (e) { /* storage unavailable */ }
}

function anSaveSettings() {
    try {
        localStorage.setItem(AN_STORE_KEY, JSON.stringify({
            scope: _anPanel.scope, collapsed: _anPanel.collapsed, sortCol: _anPanel.sortCol, sortDir: _anPanel.sortDir
        }));
    } catch (e) { /* storage unavailable */ }
}

/** Items the panel counts. 'filter' = the content filters, ignoring the favorites/non-fav/rejected visibility toggles. */
function anScopeItems() {
    var src = activeData || [];
    if (_anPanel.scope === 'all') {
        return src.filter(function (d) { return !(d.preview && isSupersededPreview(d)); });
    }
    var ctx = pipelineFilterCtx();
    return src.filter(function (d) { return pipelineMatches(d, ctx, true); });
}

function anCompute() {
    return computeAnalytics(anScopeItems(), {
        meta: meta,
        ratings: typeof cmpRatings === 'function' ? cmpRatings() : null
    });
}

function anSessionTitle() {
    var inp = document.getElementById('session-input');
    return (inp && inp.value) || (meta && meta.session_name) || 'session';
}

function anScopeLabel() {
    return _anPanel.scope === 'all' ? 'whole session' : 'current filter';
}

/** Markdown for the current scope and sort (Copy Markdown button; also used by the e2e test). */
function analyticsPanelMarkdown() {
    return analyticsToMarkdown(anCompute(), {
        title: anSessionTitle(), scopeLabel: anScopeLabel(), sortCol: _anPanel.sortCol, sortDir: _anPanel.sortDir
    });
}

function anPanelEl() { return document.getElementById('analytics-panel'); }

function anCreatePanel() {
    var panel = cmpEl('div', 'ap-panel');
    panel.id = 'analytics-panel';

    var head = cmpEl('div', 'ap-head');
    head.appendChild(cmpEl('span', 'ap-title', 'STATS'));

    var scope = document.createElement('select');
    scope.id = 'analytics-scope';
    scope.title = 'Which items are counted';
    [['filter', 'Current filter (all favs/rejects counted)'], ['all', 'Whole session']].forEach(function (o) {
        var opt = document.createElement('option');
        opt.value = o[0];
        opt.textContent = o[1];
        scope.appendChild(opt);
    });
    scope.value = _anPanel.scope;
    scope.addEventListener('change', function () {
        _anPanel.scope = scope.value === 'all' ? 'all' : 'filter';
        anSaveSettings();
        renderAnalyticsPanel();
    });
    head.appendChild(scope);

    var copy = cmpEl('button', '', 'Copy Markdown');
    copy.id = 'analytics-copy-md';
    copy.title = 'Copy every table as Markdown';
    copy.addEventListener('click', function () { cmpCopyText(analyticsPanelMarkdown(), copy); });
    head.appendChild(copy);

    var col = cmpEl('button', '', _anPanel.collapsed ? '▸' : '▾');
    col.id = 'analytics-collapse-btn';
    col.title = 'Collapse / expand';
    col.addEventListener('click', function () {
        _anPanel.collapsed = !_anPanel.collapsed;
        anSaveSettings();
        col.textContent = _anPanel.collapsed ? '▸' : '▾';
        panel.classList.toggle('ap-collapsed', _anPanel.collapsed);
        if (!_anPanel.collapsed && _anPanel.dirty) renderAnalyticsPanel();
    });
    head.appendChild(col);

    var close = cmpEl('button', '', '✕');
    close.id = 'analytics-close-btn';
    close.title = 'Close (Esc)';
    close.addEventListener('click', function () { closeAnalyticsPanel(); });
    head.appendChild(close);
    panel.appendChild(head);

    var summary = cmpEl('div', 'analytics-summary');
    summary.id = 'analytics-summary';
    panel.appendChild(summary);
    panel.appendChild(cmpEl('div', 'ap-body'));
    panel.appendChild(cmpEl('div', 'ap-foot'));
    panel.classList.toggle('ap-collapsed', _anPanel.collapsed);
    document.body.appendChild(panel);
    return panel;
}

function openAnalyticsPanel() {
    if (!_anPanel.loaded) { anLoadSettings(); _anPanel.loaded = true; }
    var panel = anPanelEl();
    if (!panel) panel = anCreatePanel();
    panel.style.display = 'flex';
    _anPanel.open = true;
    var cog = document.getElementById('cog-menu-dropdown');
    if (cog && cog.style.display !== 'none' && cog.style.display !== '' && typeof toggleCogMenu === 'function') toggleCogMenu();
    renderAnalyticsPanel();
}

function toggleAnalyticsPanel() {
    if (_anPanel.open) closeAnalyticsPanel(); else openAnalyticsPanel();
}

/** Returns true when it closed something (Escape priority chain). */
function closeAnalyticsPanel() {
    if (!_anPanel.open) return false;
    var panel = anPanelEl();
    if (panel) panel.style.display = 'none';
    _anPanel.open = false;
    if (_anPanel.raf) { clearTimeout(_anPanel.raf); _anPanel.raf = 0; }
    return true;
}

function anSortArrow(col) {
    return _anPanel.sortCol === col ? (_anPanel.sortDir < 0 ? ' ▼' : ' ▲') : '';
}

function anSetSort(col) {
    if (_anPanel.sortCol === col) _anPanel.sortDir = -_anPanel.sortDir;
    else { _anPanel.sortCol = col; _anPanel.sortDir = col === 'label' ? 1 : -1; }
    anSaveSettings();
    renderAnalyticsPanel();
}

function anHeaderCell(text, col, numeric) {
    var th = cmpEl('th', 'analytics-th ap-sortable', text + anSortArrow(col));
    th.dataset.col = col;
    if (numeric) th.style.textAlign = 'right';
    th.addEventListener('click', function () { anSetSort(col); });
    return th;
}

function anAxisTable(ax, res) {
    var wrap = cmpEl('div', 'ap-table-wrap');
    var table = cmpEl('table', 'analytics-table');
    var thead = document.createElement('thead');
    var hr = document.createElement('tr');
    hr.appendChild(anHeaderCell('Value', 'label', false));
    hr.appendChild(anHeaderCell('n', 'n', true));
    hr.appendChild(anHeaderCell('Fav %', 'favRate', true));
    hr.appendChild(anHeaderCell('Rej %', 'rejRate', true));
    hr.appendChild(anHeaderCell('Avg s', 'durMean', true));
    res.metricKeys.forEach(function (k) { hr.appendChild(anHeaderCell(metricLabel(k), 'metric:' + k, true)); });
    if (res.hasRatings) hr.appendChild(anHeaderCell('Rating', 'rating', true));
    thead.appendChild(hr);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    var rows = sortAnalyticsRows(ax.rows, _anPanel.sortCol, _anPanel.sortDir);
    var bestKey = (ax.rows[0] && ax.rows[0].enough && ax.rows[0].favRate > 0) ? ax.rows[0].key : null;
    rows.slice(0, AN_ROW_CAP).forEach(function (r) {
        var tr = cmpEl('tr', 'analytics-row');
        if (!r.enough) tr.classList.add('sb-row-dim');
        if (bestKey !== null && r.key === bestKey) tr.classList.add('sb-row-best');
        var v = cmpEl('td', 'sb-val', r.label);
        v.title = r.key;
        tr.appendChild(v);
        var cells = [String(r.n), anFmt(r.favRate, 'pct', true), anFmt(r.rejRate, 'pct', true), anFmt(r.durMean, 'dur', true)];
        res.metricKeys.forEach(function (k) { cells.push(r.metrics[k] ? anFmt(r.metrics[k].mean, 'metric', true) : '–'); });
        if (res.hasRatings) cells.push(r.ratingMean === null ? '–' : anFmt(r.ratingMean, 'rating', true) + ' (' + r.ratingN + ')');
        cells.forEach(function (c) { tr.appendChild(cmpEl('td', 'sb-num', c)); });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    if (rows.length > AN_ROW_CAP) {
        var more = cmpEl('div', 'ap-more', '+' + (rows.length - AN_ROW_CAP) + ' more (included in Copy Markdown)');
        var holder = document.createElement('div');
        holder.appendChild(wrap);
        holder.appendChild(more);
        return holder;
    }
    return wrap;
}

function anIsAxisOpen(ax) {
    var st = _anPanel.openAxes;
    if (st && Object.prototype.hasOwnProperty.call(st, ax.id)) return !!st[ax.id];
    return ax.kind === 'wf' || AN_DEFAULT_OPEN.indexOf(ax.id) >= 0;
}

function renderAnalyticsPanel() {
    var panel = anPanelEl();
    if (!panel || !_anPanel.open) return;
    _anPanel.dirty = false;
    _anPanel.lastRender = Date.now();
    var res = anCompute();

    var body = panel.querySelector('.ap-body');
    var summary = document.getElementById('analytics-summary');
    var foot = panel.querySelector('.ap-foot');
    var scrollTop = body.scrollTop;
    // Remember which axes the user opened or closed (survives the rebuild)
    body.querySelectorAll('details[data-axis]').forEach(function (d) {
        if (!_anPanel.openAxes) _anPanel.openAxes = {};
        _anPanel.openAxes[d.dataset.axis] = d.open;
    });

    var scopeSel = document.getElementById('analytics-scope');
    if (scopeSel && scopeSel.value !== _anPanel.scope) scopeSel.value = _anPanel.scope;

    // Summary line
    var o = res.overall;
    var metricsOn = 0;
    res.metricKeys.forEach(function (k) { if (o.metrics[k] && o.metrics[k].n > metricsOn) metricsOn = o.metrics[k].n; });
    var favPct = res.total ? (res.favorited / res.total * 100).toFixed(0) : '0';
    var bits = [res.total + ' items', res.favorited + ' favorited (' + favPct + '%)', res.rejected + ' rejected'];
    if (o.durMean !== null) bits.push('avg ' + o.durMean.toFixed(2) + ' s');
    bits.push('metrics on ' + metricsOn);
    bits.push('ratings on ' + o.ratingN);
    summary.textContent = bits.join(' · ');

    // Axis tables
    while (body.firstChild) body.removeChild(body.firstChild);
    var constants = [];
    res.axes.forEach(function (ax) {
        if (ax.constant) { if (ax.rows.length) constants.push(ax.label + ' = ' + ax.rows[0].label); return; }
        var det = cmpEl('details', 'sb-axis');
        det.dataset.axis = ax.id;
        det.open = anIsAxisOpen(ax);
        var sum = document.createElement('summary');
        sum.textContent = ax.label;
        sum.appendChild(cmpEl('span', 'sb-meta', ' · ' + ax.distinct + ' values'));
        det.appendChild(sum);
        det.appendChild(anAxisTable(ax, res));
        det.addEventListener('toggle', function () {
            if (!_anPanel.openAxes) _anPanel.openAxes = {};
            _anPanel.openAxes[ax.id] = det.open;
        });
        body.appendChild(det);
    });
    if (!res.axes.length || res.total === 0) body.appendChild(cmpEl('div', 'ap-more', 'No items in this scope.'));
    body.scrollTop = scrollTop;

    // Footer
    while (foot.firstChild) foot.removeChild(foot.firstChild);
    if (constants.length) foot.appendChild(cmpEl('div', '', 'Same for every item: ' + constants.join(', ')));
    if (res.skippedAxes.length) {
        foot.appendChild(cmpEl('div', '', 'Skipped (too many distinct values): ' +
            res.skippedAxes.map(function (s) { return s.id.slice(3) + ' (' + s.distinct + ')'; }).join(', ')));
    }
    if (_anPanel.scope === 'filter' && typeof topFilters !== 'undefined' && !topFilters.showRejected) {
        foot.appendChild(cmpEl('div', '', 'Rejected items are counted even though the grid hides them.'));
    }
    foot.appendChild(cmpEl('div', '', 'Rows with n < ' + AN_MIN_N + ' are dimmed and sorted last.'));
}

/** Hook target (updateJSONs, cmpAfterRatingsChanged): coalesced live refresh. */
function analyticsPanelOnDataChanged() {
    if (!_anPanel.open) return;
    if (_anPanel.collapsed) { _anPanel.dirty = true; return; }
    if (_anPanel.raf) return;
    var wait = Math.max(0, AN_RENDER_GAP_MS - (Date.now() - _anPanel.lastRender));
    var run = function () {
        _anPanel.raf = 0;
        if (_anPanel.open && !_anPanel.collapsed) renderAnalyticsPanel();
    };
    _anPanel.raf = setTimeout(function () {
        if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run); else run();
    }, wait);
}
