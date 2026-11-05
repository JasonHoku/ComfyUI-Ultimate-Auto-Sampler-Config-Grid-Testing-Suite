/**
 * SETTINGS STATS (ANALYTICS) - PURE LOGIC (no DOM access)
 * Groups the items by every setting axis (model, LoRA, sampler, ... plus workflow-sweep params) and
 * reduces each group to favorite/reject rate, mean duration, mean metrics and mean pairwise rating.
 * The dashboard inlines every logic_*.js into ONE <script>, so top-level names here are prefixed
 * (AN_ / an). Node tests load this file through the module.exports guard at the bottom
 * (tests/js/analytics.test.js). The result shape is the data source for the later chart milestones.
 *
 * Needs normalizeAxis/fmtNum (logic_ratings.js) and wfShortLabel (logic_wfparams.js).
 */

var AN_AXES = ['model', 'lora', 'sampler', 'scheduler', 'steps', 'cfg', 'denoise', 'resolution', 'attention_mode', 'upscale', 'pass', 'prompt'];
var AN_AXIS_LABELS = {
    model: 'Model', lora: 'LoRA', sampler: 'Sampler', scheduler: 'Scheduler', steps: 'Steps', cfg: 'CFG',
    denoise: 'Denoise', resolution: 'Resolution', attention_mode: 'Attention', upscale: 'Upscale', pass: 'Pass', prompt: 'Prompt'
};
var AN_MAX_WF_DISTINCT = 64;   // wf_params axes with more values than this are skipped (seeds, ...)
var AN_MIN_N = 3;
var AN_DEPS = (typeof module !== 'undefined' && module.exports && typeof require === 'function')
    ? { normalizeAxis: require('./logic_ratings.js').normalizeAxis, fmtNum: require('./logic_ratings.js').fmtNum, wfShortLabel: require('./logic_wfparams.js').wfShortLabel }
    : null;

function anDep(name) { return AN_DEPS ? AN_DEPS[name] : (typeof window !== 'undefined' ? window[name] : undefined); }

/** {key, label} of item d on one axis. */
function anAxisValue(axis, d, meta) {
    if (axis === 'attention_mode') {
        var v = d.attention_mode;
        var s = Array.isArray(v) ? v.join('+') : String(v || 'default');
        return { key: s, label: s };
    }
    if (axis === 'pass') {
        var p = d.preview ? 'Preview' : 'Full';
        return { key: p, label: p };
    }
    return anDep('normalizeAxis')(axis, d, meta);
}

function anNewAcc(key, label) {
    return { key: key, label: label, n: 0, fav: 0, rej: 0, durSum: 0, durN: 0, mSum: {}, mN: {}, rSum: 0, rN: 0 };
}

function anAccAdd(acc, fav, rej, dur, metrics, mkeys, rating) {
    acc.n++;
    if (fav) acc.fav++;
    if (rej) acc.rej++;
    if (dur !== null) { acc.durSum += dur; acc.durN++; }
    for (var i = 0; i < mkeys.length; i++) {
        var k = mkeys[i];
        acc.mSum[k] = (acc.mSum[k] || 0) + metrics[k];
        acc.mN[k] = (acc.mN[k] || 0) + 1;
    }
    if (rating !== null) { acc.rSum += rating; acc.rN++; }
}

function anFinishRow(acc, metricKeys, minN) {
    var metrics = {};
    for (var i = 0; i < metricKeys.length; i++) {
        var k = metricKeys[i];
        metrics[k] = acc.mN[k] ? { mean: acc.mSum[k] / acc.mN[k], n: acc.mN[k] } : null;
    }
    return {
        key: acc.key, label: acc.label, n: acc.n, fav: acc.fav, rej: acc.rej,
        favRate: acc.n ? acc.fav / acc.n : 0, rejRate: acc.n ? acc.rej / acc.n : 0,
        durMean: acc.durN ? acc.durSum / acc.durN : null, durN: acc.durN,
        metrics: metrics, ratingMean: acc.rN ? acc.rSum / acc.rN : null, ratingN: acc.rN,
        enough: acc.n >= minN
    };
}

function anDefaultMetricOrder() {
    return (typeof METRIC_BUILTIN_ORDER !== 'undefined' && Array.isArray(METRIC_BUILTIN_ORDER)) ? METRIC_BUILTIN_ORDER : [];
}

/**
 * One pass over `items`. opts = { meta, ratings: Map|null, metricOrder, minN, axes }.
 * Returns { total, favorited, rejected, metricKeys, hasRatings, overall, axes:[{id,label,kind,n,distinct,constant,rows}], skippedAxes }.
 */
function computeAnalytics(items, opts) {
    opts = opts || {};
    var meta = opts.meta;
    var ratings = opts.ratings || null;
    var order = opts.metricOrder || anDefaultMetricOrder();
    var minN = opts.minN === undefined ? AN_MIN_N : opts.minN;
    var axisIds = opts.axes || AN_AXES;
    var fmtNum = anDep('fmtNum');
    var list = items || [];

    var axisMaps = new Map();           // axis id -> Map(key -> acc)
    for (var a = 0; a < axisIds.length; a++) axisMaps.set(axisIds[a], new Map());
    var wfMaps = new Map();             // 'wf:<key>' -> Map(key -> acc)
    var overall = anNewAcc('(all)', '(all)');
    var metricSeen = new Set();
    var total = 0, favorited = 0, rejected = 0, hasRatings = false;

    for (var i = 0; i < list.length; i++) {
        var d = list[i];
        if (!d) continue;
        var fav = !!d.favorited, rej = !!d.rejected;
        var dn = +d.duration;
        var dur = (isFinite(dn) && dn > 0) ? dn : null;
        var metrics = d.metrics && typeof d.metrics === 'object' ? d.metrics : null;
        var mkeys = [];
        if (metrics) {
            for (var mk in metrics) {
                var mv = metrics[mk];
                if (typeof mv === 'number' && isFinite(mv)) { mkeys.push(mk); metricSeen.add(mk); }
            }
        }
        var rating = null;
        if (ratings) {
            var rr = ratings.get(d.id);
            if (rr && rr.votes > 0 && isFinite(+rr.rating)) { rating = +rr.rating; hasRatings = true; }
        }
        total++;
        if (fav) favorited++;
        if (rej) rejected++;
        anAccAdd(overall, fav, rej, dur, metrics, mkeys, rating);

        for (var ai = 0; ai < axisIds.length; ai++) {
            var id = axisIds[ai];
            var av = anAxisValue(id, d, meta);
            var m = axisMaps.get(id);
            var acc = m.get(av.key);
            if (!acc) { acc = anNewAcc(av.key, av.label); m.set(av.key, acc); }
            anAccAdd(acc, fav, rej, dur, metrics, mkeys, rating);
        }

        var wf = d.wf_params;
        if (wf && typeof wf === 'object') {
            for (var wk in wf) {
                var wv = wf[wk];
                if (wv === null || wv === undefined) continue;
                var wid = 'wf:' + wk;
                var wm = wfMaps.get(wid);
                if (!wm) { wm = new Map(); wfMaps.set(wid, wm); }
                var vk = typeof wv === 'number' ? fmtNum(wv) : String(wv);
                var wacc = wm.get(vk);
                if (!wacc) { wacc = anNewAcc(vk, vk); wm.set(vk, wacc); }
                anAccAdd(wacc, fav, rej, dur, metrics, mkeys, rating);
            }
        }
    }

    var metricKeys = order.filter(function (k) { return metricSeen.has(k); })
        .concat(Array.from(metricSeen).filter(function (k) { return order.indexOf(k) < 0; }).sort());

    function build(id, label, kind, map) {
        var rows = [];
        map.forEach(function (acc) { rows.push(anFinishRow(acc, metricKeys, minN)); });
        return { id: id, label: label, kind: kind, n: total, distinct: rows.length, constant: rows.length < 2, rows: sortAnalyticsRows(rows, 'favRate', -1) };
    }

    var axes = [];
    for (var b = 0; b < axisIds.length; b++) {
        var bm = axisMaps.get(axisIds[b]);
        if (bm.size) axes.push(build(axisIds[b], AN_AXIS_LABELS[axisIds[b]] || axisIds[b], 'builtin', bm));
    }
    var wfIds = Array.from(wfMaps.keys()).sort();
    var wfKeys = wfIds.map(function (x) { return x.slice(3); });
    var skippedAxes = [];
    var shortLabel = anDep('wfShortLabel');
    for (var w = 0; w < wfIds.length; w++) {
        var wmap = wfMaps.get(wfIds[w]);
        if (wmap.size > AN_MAX_WF_DISTINCT) { skippedAxes.push({ id: wfIds[w], distinct: wmap.size }); continue; }
        var shortName = shortLabel ? shortLabel(wfKeys[w], wfKeys) : wfKeys[w];
        axes.push(build(wfIds[w], 'Workflow: ' + shortName, 'wf', wmap));
    }

    return {
        total: total, favorited: favorited, rejected: rejected, metricKeys: metricKeys, hasRatings: hasRatings,
        overall: anFinishRow(overall, metricKeys, minN), axes: axes, skippedAxes: skippedAxes
    };
}

function anSortValue(row, col) {
    if (col === 'label') return row.label;
    if (col === 'n') return row.n;
    if (col === 'favRate') return row.favRate;
    if (col === 'rejRate') return row.rejRate;
    if (col === 'durMean') return row.durMean;
    if (col === 'rating') return row.ratingMean;
    if (typeof col === 'string' && col.indexOf('metric:') === 0) {
        var m = row.metrics && row.metrics[col.slice(7)];
        return m ? m.mean : null;
    }
    return row.n;
}

/** New sorted array. Rows with enough samples always come first; nulls last; ties: n desc then label asc. */
function sortAnalyticsRows(rows, col, dir) {
    var sign = dir < 0 ? -1 : 1;
    function lab(a, b) { return String(a.label).localeCompare(String(b.label), undefined, { numeric: true }); }
    return rows.slice().sort(function (a, b) {
        if (a.enough !== b.enough) return a.enough ? -1 : 1;
        var va = anSortValue(a, col), vb = anSortValue(b, col);
        var na = va === null || va === undefined || (typeof va === 'number' && !isFinite(va));
        var nb = vb === null || vb === undefined || (typeof vb === 'number' && !isFinite(vb));
        if (na !== nb) return na ? 1 : -1;
        if (!na) {
            var c = typeof va === 'string' ? String(va).localeCompare(String(vb), undefined, { numeric: true }) : (va < vb ? -1 : (va > vb ? 1 : 0));
            if (c) return c * sign;
        }
        if (b.n !== a.n) return b.n - a.n;
        return lab(a, b);
    });
}

/** Cell text. kind: pct | dur | metric | rating. null -> blank (markdown, default) or the en dash (panel, dash=true). */
function anFmt(v, kind, dash) {
    if (v === null || v === undefined || (typeof v === 'number' && !isFinite(v))) return dash ? '\u2013' : '';
    switch (kind) {
        case 'pct': return (v * 100).toFixed(1);
        case 'dur': return v.toFixed(2);
        case 'metric': return String(parseFloat(v.toPrecision(3)));
        case 'rating': return String(Math.round(v));
    }
    return String(v);
}

function anMetricLabel(k) {
    if (typeof metricLabel === 'function') return metricLabel(k);
    return String(k).replace(/[_-]+/g, ' ').replace(/\b\w/g, function (c) { return c.toUpperCase(); });
}

function anEsc(s) { return String(s).replace(/\|/g, '\\|').replace(/[\r\n]+/g, ' '); }

/** Markdown report of a computeAnalytics result. opts = { title, scopeLabel, sortCol, sortDir }. */
function analyticsToMarkdown(res, opts) {
    opts = opts || {};
    var col = opts.sortCol || 'favRate';
    var dir = opts.sortDir === undefined ? -1 : opts.sortDir;
    var head = '## USCG stats: ' + anEsc(opts.title || 'session') + ' (' + (opts.scopeLabel ? anEsc(opts.scopeLabel) + ', ' : '') + res.total + ' items)';
    var out = [head,
        "_Fav % / Rej % = share of the group's items; Avg s = mean duration; metric columns = mean over items that have it; rows with n < " + AN_MIN_N + ' are marked *._'];
    var constants = [];
    for (var a = 0; a < res.axes.length; a++) {
        var ax = res.axes[a];
        if (ax.constant) { if (ax.rows.length) constants.push(ax.label + ' = ' + anEsc(ax.rows[0].label)); continue; }
        var cols = ['Value', 'n', 'Fav %', 'Rej %', 'Avg s'].concat(res.metricKeys.map(anMetricLabel));
        var seps = ['---', '---:', '---:', '---:', '---:'].concat(res.metricKeys.map(function () { return '---:'; }));
        if (res.hasRatings) { cols.push('Rating'); seps.push('---:'); }
        out.push('', '### ' + anEsc(ax.label), '| ' + cols.join(' | ') + ' |', '|' + seps.join('|') + '|');
        var rows = sortAnalyticsRows(ax.rows, col, dir);
        for (var r = 0; r < rows.length; r++) {
            var row = rows[r];
            var cells = [anEsc(row.label), String(row.n) + (row.enough ? '' : '*'), anFmt(row.favRate, 'pct'), anFmt(row.rejRate, 'pct'), anFmt(row.durMean, 'dur')];
            for (var k = 0; k < res.metricKeys.length; k++) {
                var mm = row.metrics[res.metricKeys[k]];
                cells.push(mm ? anFmt(mm.mean, 'metric') : '');
            }
            if (res.hasRatings) cells.push(row.ratingMean === null ? '' : anFmt(row.ratingMean, 'rating') + ' (' + row.ratingN + ')');
            out.push('| ' + cells.join(' | ') + ' |');
        }
    }
    if (constants.length) out.push('', 'Same for every item: ' + constants.join(', '));
    if (res.skippedAxes && res.skippedAxes.length) {
        out.push('', 'Skipped (too many distinct values): ' + res.skippedAxes.map(function (s) { return anEsc(s.id) + ' (' + s.distinct + ')'; }).join(', '));
    }
    return out.join('\n') + '\n';
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        AN_AXES: AN_AXES, AN_MAX_WF_DISTINCT: AN_MAX_WF_DISTINCT, AN_MIN_N: AN_MIN_N,
        computeAnalytics: computeAnalytics, sortAnalyticsRows: sortAnalyticsRows,
        analyticsToMarkdown: analyticsToMarkdown, anAxisValue: anAxisValue, anFmt: anFmt
    };
}
