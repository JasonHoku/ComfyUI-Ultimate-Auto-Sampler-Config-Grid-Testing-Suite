/**
 * PAIRWISE REVIEW - PURE LOGIC (no DOM access)
 * Axis normalisation, attribution, Bradley-Terry ratings, pair selection, scoreboard.
 * The dashboard inlines every logic_*.js into ONE <script>, so top-level names here are
 * prefixed (CMP_ / cmp / rt) or unique. Node tests load this file through the
 * module.exports guard at the bottom (tests/js/ratings.test.js).
 *
 * Vote shape (manifest top-level "comparisons"):
 *   { a, b, winner: 'a'|'b'|'tie', ts, mode, blind? }   a = left/top item id
 */

var CMP_AXES = ['model', 'lora', 'sampler', 'scheduler', 'cfg', 'steps', 'denoise', 'resolution', 'prompt', 'upscale'];
// 'seed' is a diagnostic axis (noise floor): it participates in axisDiff but not in CMP_AXES
var CMP_DIFF_AXES = CMP_AXES.concat(['seed']);

// Item keys that never make two items "different settings" (deny list on purpose: unknown
// keys such as vae / clip_type / use_slg are treated as confounders and block attribution).
var CMP_EXCLUDED_KEYS = new Set([
    'id', 'gen_index', 'file', 'filename', 'duration', 'batch_idx', 'favorited', 'rejected',
    'note', 'notes', 'positive', 'config_positive', 'negative', 'config_negative', 'seed',
    'width', 'height', 'base_width', 'base_height', 'model', 'lora', 'lora_expanded',
    'sampler', 'scheduler', 'cfg', 'steps', 'denoise', 'upscaled', 'source_file',
    'image_path', 'preview_path', 'video_path', 'hash', 'hires_denoise',
    // per-image outcomes / bookkeeping, not settings
    'config_fp', 'base_fp', 'metrics', 'preview_of', 'source_relpath',
    'florence2_bbox', 'florence2_detection_count', 'florence2_no_detection'
]);
var CMP_EXCLUDED_RE = /^upscale_|^hires_prompt_|_hash$|^worker|^_/;

// ---------------------------------------------------------------------------
// Normalisers
// ---------------------------------------------------------------------------
function fmtNum(v) {
    var ok = (typeof v === 'number') || (typeof v === 'string' && v.trim() !== '');
    if (ok && isFinite(+v)) return String(+(+v).toFixed(6));
    return String(v === undefined || v === null ? '?' : v);
}

function cmpBase(s) {
    return String(s).replace(/\\/g, '/').split('/').pop().replace(/\.[^.]+$/, '');
}

function cmpProduct(r) {
    var arr = Array.isArray(r) ? r : [r];
    var p = 1, any = false;
    for (var i = 0; i < arr.length; i++) {
        var n = +arr[i];
        if (isFinite(n) && n > 0 && arr[i] !== null && arr[i] !== '') { p *= n; any = true; }
    }
    return any ? p : 0;
}

function normModel(d, m) {
    var raw = String(d.model || (m && m.model) || 'Default');
    return { key: raw.replace(/\\/g, '/'), label: cmpBase(raw) };
}

function normLora(d) {
    var raw = String(d.lora_expanded || d.lora || 'None');
    if (raw === 'None' || raw === '') return { key: 'None', label: 'None' };
    var parts = raw.split(' + ').map(function (p) {
        var c = p.split(':');
        var name = c[0].replace(/\\/g, '/');
        var ms = fmtNum(c[1] !== undefined && c[1] !== '' ? c[1] : 1);
        var cs = fmtNum(c[2] !== undefined && c[2] !== '' ? c[2] : ms);
        return { name: name, ms: ms, cs: cs };
    });
    parts.sort(function (x, y) {
        var a = x.name + ':' + x.ms + ':' + x.cs, b = y.name + ':' + y.ms + ':' + y.cs;
        return a < b ? -1 : a > b ? 1 : 0;
    });
    return {
        key: parts.map(function (p) { return p.name + ':' + p.ms + ':' + p.cs; }).join(' + '),
        label: parts.map(function (p) {
            return cmpBase(p.name) + '@' + p.ms + (p.cs !== p.ms ? '/' + p.cs : '');
        }).join(' + ')
    };
}

function normPrompt(d, m) {
    var raw = d.config_positive !== undefined && d.config_positive !== null ? d.config_positive
        : (d.positive !== undefined && d.positive !== null ? d.positive
            : ((m && m.positive) || ''));
    var s = String(raw).replace(/\s+/g, ' ').trim();
    return { key: s, label: s.length > 60 ? s.slice(0, 60) + '…' : s };
}

function normResolution(d) {
    var k;
    if (d.base_width && d.base_height) {
        k = d.base_width + 'x' + d.base_height;
    } else if (!d.upscaled) {
        k = d.width + 'x' + d.height;
    } else {
        // legacy manifests (before base_width/base_height): estimate from upscale_ratio
        var r = cmpProduct(d.upscale_ratio);
        if (r > 0 && d.width > 0 && d.height > 0) {
            k = (Math.round(d.width / r / 8) * 8) + 'x' + (Math.round(d.height / r / 8) * 8);
        } else {
            k = 'AR ' + (d.height > 0 ? (d.width / d.height).toFixed(2) : '?');
        }
    }
    return { key: k, label: k };
}

function normUpscale(d) {
    if (!d.upscaled) return { key: 'none', label: 'none' };
    var join = function (v) { return Array.isArray(v) ? v.join('>') : v; };
    var mode = d.upscale_mode !== undefined && d.upscale_mode !== null ? String(join(d.upscale_mode)) : '';
    var models = d.upscale_model !== undefined && d.upscale_model !== null && d.upscale_model !== ''
        ? (Array.isArray(d.upscale_model) ? d.upscale_model : [d.upscale_model]).map(cmpBase).join('>') : '';
    var ratio = d.upscale_ratio !== undefined && d.upscale_ratio !== null
        ? (Array.isArray(d.upscale_ratio) ? d.upscale_ratio.map(fmtNum).join('>') : fmtNum(d.upscale_ratio)) : '';
    var parts = [];
    if (mode) parts.push(mode);
    if (models) parts.push(models);
    if (ratio) parts.push('x' + ratio);
    if (d.upscale_denoise !== undefined && d.upscale_denoise !== null) parts.push('d' + fmtNum(d.upscale_denoise));
    if (d.hires_prompt_behavior) parts.push('hp:' + d.hires_prompt_behavior);
    if (!parts.length) parts.push('upscaled');
    var label = [mode, models, ratio ? 'x' + ratio : ''].filter(Boolean).join(' ') || 'upscaled';
    return { key: parts.join('|'), label: label };
}

function normalizeAxis(axis, d, m) {
    switch (axis) {
        case 'model': return normModel(d, m);
        case 'lora': return normLora(d);
        case 'sampler': { var s = String(d.sampler == null ? '?' : d.sampler); return { key: s, label: s }; }
        case 'scheduler': { var c = String(d.scheduler == null ? '?' : d.scheduler); return { key: c, label: c }; }
        case 'cfg': { var f = fmtNum(d.cfg); return { key: f, label: f }; }
        case 'steps': { var t = fmtNum(d.steps); return { key: t, label: t }; }
        case 'denoise': { var n = fmtNum(d.denoise); return { key: n, label: n }; }
        case 'resolution': return normResolution(d);
        case 'prompt': return normPrompt(d, m);
        case 'upscale': return normUpscale(d);
        case 'seed': { var e = fmtNum(d.seed); return { key: e, label: e }; }
    }
    return { key: '?', label: '?' };
}

function cmpStable(v) {
    if (v === null || typeof v !== 'object') return JSON.stringify(v === undefined ? null : v);
    if (Array.isArray(v)) return '[' + v.map(cmpStable).join(',') + ']';
    return '{' + Object.keys(v).sort().map(function (k) { return JSON.stringify(k) + ':' + cmpStable(v[k]); }).join(',') + '}';
}

function cmpCtxEntries(d) {
    var out = {};
    var ks = Object.keys(d);
    for (var i = 0; i < ks.length; i++) {
        var k = ks[i];
        if (CMP_EXCLUDED_KEYS.has(k) || CMP_EXCLUDED_RE.test(k)) continue;
        out[k] = d[k];
    }
    out.negative = d.config_negative !== undefined && d.config_negative !== null ? d.config_negative : d.negative;
    return out;
}

function cmpMediaType(d) { return d.media_type === 'video' ? 'video' : 'image'; }

var _cmpSigCache = new WeakMap();

/** {axes:{axis:key}, labels:{axis:label}, ctx} cached per item object (invalidated on w/h/meta change). */
function itemSig(d, m) {
    var hit = _cmpSigCache.get(d);
    if (hit && hit.w === d.width && hit.h === d.height && hit.m === m) return hit;
    var axes = {}, labels = {};
    for (var i = 0; i < CMP_DIFF_AXES.length; i++) {
        var ax = CMP_DIFF_AXES[i];
        var r = normalizeAxis(ax, d, m);
        axes[ax] = r.key; labels[ax] = r.label;
    }
    var sig = { axes: axes, labels: labels, ctx: cmpStable(cmpCtxEntries(d)), w: d.width, h: d.height, m: m };
    _cmpSigCache.set(d, sig);
    return sig;
}

/** Which axes differ between two items. ctxKeys only when opts.withCtxKeys (diagnostics). */
function axisDiff(A, B, opts) {
    opts = opts || {};
    var m = opts.meta;
    var sa = itemSig(A, m), sb = itemSig(B, m);
    var axes = [];
    for (var i = 0; i < CMP_DIFF_AXES.length; i++) {
        var ax = CMP_DIFF_AXES[i];
        if (sa.axes[ax] !== sb.axes[ax]) axes.push(ax);
    }
    var ctxEqual = sa.ctx === sb.ctx;
    var ctxKeys = [];
    if (!ctxEqual && opts.withCtxKeys) {
        var ea = cmpCtxEntries(A), eb = cmpCtxEntries(B);
        var all = new Set(Object.keys(ea).concat(Object.keys(eb)));
        all.forEach(function (k) {
            if (cmpStable(ea[k]) !== cmpStable(eb[k])) ctxKeys.push(k);
        });
        ctxKeys.sort();
    }
    return { axes: axes, ctxEqual: ctxEqual, ctxKeys: ctxKeys };
}

// ---------------------------------------------------------------------------
// Comparison list helpers
// ---------------------------------------------------------------------------
function cmpVoteKey(c) { return c.ts + ':' + c.a + ':' + c.b; }

function mergeComparisons(existing, incoming) {
    var out = [], seen = new Set();
    var all = (existing || []).concat(incoming || []);
    for (var i = 0; i < all.length; i++) {
        var c = all[i];
        if (!c) continue;
        var k = cmpVoteKey(c);
        if (seen.has(k)) continue;
        seen.add(k);
        out.push(c);
    }
    return out;
}

function cmpPairKey(a, b) { return a < b ? a + '|' + b : b + '|' + a; }

function cmpValidVote(c, ids) {
    return c && c.a !== c.b && (c.winner === 'a' || c.winner === 'b' || c.winner === 'tie') &&
        ids.has(c.a) && ids.has(c.b);
}

// ---------------------------------------------------------------------------
// Bradley-Terry (MM, Hunter 2004) with a ghost-player Laplace prior
// ---------------------------------------------------------------------------
function computeBradleyTerry(comparisons, presentIds, opts) {
    opts = opts || {};
    var prior = opts.prior !== undefined ? opts.prior : 1;
    var maxIter = opts.maxIter || 200;
    var tol = opts.tol || 1e-6;
    var ids = presentIds instanceof Set ? presentIds : new Set(presentIds || []);
    var rec = new Map();      // id -> {w,l,t,W,nb:Map}
    var get = function (id) {
        var r = rec.get(id);
        if (!r) { r = { w: 0, l: 0, t: 0, W: prior, nb: new Map() }; rec.set(id, r); }
        return r;
    };
    for (var i = 0; i < (comparisons || []).length; i++) {
        var c = comparisons[i];
        if (!cmpValidVote(c, ids)) continue;
        var ra = get(c.a), rb = get(c.b);
        ra.nb.set(c.b, (ra.nb.get(c.b) || 0) + 1);
        rb.nb.set(c.a, (rb.nb.get(c.a) || 0) + 1);
        if (c.winner === 'a') { ra.w++; rb.l++; ra.W += 1; }
        else if (c.winner === 'b') { rb.w++; ra.l++; rb.W += 1; }
        else { ra.t++; rb.t++; ra.W += 0.5; rb.W += 0.5; }
    }
    var out = new Map();
    if (!rec.size) return out;
    var p = new Map();
    rec.forEach(function (_, id) { p.set(id, 1); });
    var nG = 2 * prior;
    for (var it = 0; it < maxIter; it++) {
        var next = new Map(), delta = 0;
        rec.forEach(function (r, id) {
            var pi = p.get(id);
            var denom = nG / (pi + 1);
            r.nb.forEach(function (n, j) { denom += n / (pi + p.get(j)); });
            var pn = r.W / denom;
            next.set(id, pn);
            var dl = Math.abs(Math.log(pn) - Math.log(pi));
            if (dl > delta) delta = dl;
        });
        p = next;
        if (delta < tol) break;
    }
    rec.forEach(function (r, id) {
        var pi = p.get(id);
        out.set(id, { rating: 1500 + 400 * Math.log10(pi), votes: r.w + r.l + r.t, w: r.w, l: r.l, t: r.t, p: pi });
    });
    return out;
}

// ---------------------------------------------------------------------------
// Pair source + selection
// ---------------------------------------------------------------------------
var CMP_MODES = ['matched', 'seed', 'one_axis', 'random'];

/**
 * pool: array of item objects. mode: matched | seed | one_axis | random. axisFocus: axis or ''.
 * opts.meta: manifest meta for fallbacks. Returns {mode, axisFocus, buckets, estimate}.
 */
function buildPairSource(pool, mode, axisFocus, opts) {
    opts = opts || {};
    var m = opts.meta;
    var intern = new Map();
    var iid = function (s) {
        var v = intern.get(s);
        if (v === undefined) { v = intern.size; intern.set(s, v); }
        return v;
    };
    var entries = [];
    for (var i = 0; i < pool.length; i++) {
        var d = pool[i];
        if (!d) continue;
        var sg = itemSig(d, m);
        var vec = new Array(CMP_AXES.length);
        for (var a = 0; a < CMP_AXES.length; a++) vec[a] = iid(CMP_AXES[a] + '=' + sg.axes[CMP_AXES[a]]);
        entries.push({
            d: d, id: d.id, vec: vec, seed: iid('seed=' + sg.axes.seed), ctx: iid('ctx=' + sg.ctx),
            mt: cmpMediaType(d)
        });
    }
    var promptIdx = CMP_AXES.indexOf('prompt');
    var buckets = [];
    var addBuckets = function (keyFn, distinctFn, validFn, axisIdx) {
        var map = new Map();
        for (var i = 0; i < entries.length; i++) {
            var e = entries[i];
            var k = keyFn(e);
            var arr = map.get(k);
            if (!arr) { arr = []; map.set(k, arr); }
            arr.push(e);
        }
        map.forEach(function (items) {
            if (items.length < 2) return;
            var first = distinctFn(items[0]), ok = false;
            for (var j = 1; j < items.length; j++) if (distinctFn(items[j]) !== first) { ok = true; break; }
            if (!ok) return;
            buckets.push({ items: items, weight: Math.min(items.length, 50), valid: validFn, axis: axisIdx === undefined ? null : CMP_AXES[axisIdx] });
        });
    };
    var configSig = function (e) {
        var parts = [];
        for (var a = 0; a < e.vec.length; a++) if (a !== promptIdx) parts.push(e.vec[a]);
        parts.push(e.ctx);
        return parts.join(',');
    };
    if (mode === 'matched') {
        addBuckets(function (e) { return e.vec[promptIdx] + '|' + e.seed + '|' + e.mt; }, configSig,
            function (x, y) { return configSig(x) !== configSig(y); });
    } else if (mode === 'seed') {
        addBuckets(function (e) { return e.vec.join(',') + '|' + e.ctx + '|' + e.mt; },
            function (e) { return e.seed; }, function (x, y) { return x.seed !== y.seed; });
    } else if (mode === 'one_axis') {
        var list = axisFocus && CMP_AXES.indexOf(axisFocus) >= 0 ? [CMP_AXES.indexOf(axisFocus)] : CMP_AXES.map(function (_, i) { return i; });
        list.forEach(function (ai) {
            addBuckets(function (e) {
                    var rest = [];
                    for (var a = 0; a < e.vec.length; a++) if (a !== ai) rest.push(e.vec[a]);
                    return ai + '|' + e.ctx + '|' + rest.join(',') + '|' + e.seed + '|' + e.mt;
                },
                function (e) { return e.vec[ai]; }, function (x, y) { return x.vec[ai] !== y.vec[ai]; }, ai);
        });
    } else {
        addBuckets(function (e) { return e.mt; }, function (e) { return e.id; }, function () { return true; });
    }
    var estimate = 0;
    for (var b = 0; b < buckets.length; b++) { var k = buckets[b].items.length; estimate += k * (k - 1) / 2; }
    return { mode: mode, axisFocus: axisFocus || '', buckets: buckets, estimate: estimate, poolSize: entries.length };
}

function cmpPickOne(items, rng) { return items[Math.floor(rng() * items.length)]; }

/**
 * state: {votesById: Map id->count, votedPairs: Set(pairKey), skipped: Set(pairKey), allowRepeats}
 * Returns {a, b (item objects, random left/right), repeating, axis} or null.
 */
function nextPair(src, state, rng) {
    rng = rng || Math.random;
    if (!src || !src.buckets.length) return null;
    var votes = function (id) { return (state.votesById && state.votesById.get(id)) || 0; };
    var cum = [], total = 0;
    for (var i = 0; i < src.buckets.length; i++) { total += src.buckets[i].weight; cum.push(total); }
    var pickBucket = function () {
        var r = rng() * total, lo = 0, hi = cum.length - 1;
        while (lo < hi) { var mid = (lo + hi) >> 1; if (cum[mid] > r) hi = mid; else lo = mid + 1; }
        return src.buckets[lo];
    };
    var attempt = function (allowRepeats) {
        var best = null;
        for (var s = 0; s < 64; s++) {
            var bk = pickBucket();
            var anchor = null;
            for (var t = 0; t < 4; t++) {
                var c = cmpPickOne(bk.items, rng);
                if (!anchor || votes(c.id) < votes(anchor.id)) anchor = c;
            }
            var partner = null;
            for (var u = 0; u < 8; u++) {
                var y = cmpPickOne(bk.items, rng);
                if (y === anchor || y.id === anchor.id || !bk.valid(anchor, y)) continue;
                var pk = cmpPairKey(anchor.id, y.id);
                if (state.skipped && state.skipped.has(pk)) continue;
                if (!allowRepeats && state.votedPairs && state.votedPairs.has(pk)) continue;
                if (!partner || votes(y.id) < votes(partner.id)) partner = y;
            }
            if (!partner) continue;
            var score = votes(anchor.id) + votes(partner.id) + rng() * 0.5;
            if (!best || score < best.score) best = { score: score, x: anchor, y: partner, axis: bk.axis };
        }
        return best;
    };
    var repeating = false;
    var best = attempt(!!state.allowRepeats);
    if (!best && !state.allowRepeats) { best = attempt(true); repeating = !!best; }
    else if (best && state.allowRepeats) repeating = true;
    if (!best) return null;
    var swap = rng() < 0.5;
    return { a: swap ? best.y.d : best.x.d, b: swap ? best.x.d : best.y.d, repeating: repeating, axis: best.axis };
}

// ---------------------------------------------------------------------------
// Scoreboard
// ---------------------------------------------------------------------------
function wilsonInterval(s, n, z) {
    z = z || 1.96;
    if (!n) return { low: 0, high: 0 };
    var p = s / n, z2 = z * z;
    var den = 1 + z2 / n;
    var centre = (p + z2 / (2 * n)) / den;
    var half = (z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / den;
    return { low: Math.max(0, centre - half), high: Math.min(1, centre + half) };
}

function cmpItemLookup(itemsById) {
    return itemsById instanceof Map ? function (id) { return itemsById.get(id); } : function (id) { return itemsById[id]; };
}

function computeScoreboard(comparisons, itemsById, ratings, opts) {
    opts = opts || {};
    var minN = opts.minN !== undefined ? opts.minN : 3;
    var m = opts.meta;
    var restrict = opts.restrictIds || null;
    var lookup = cmpItemLookup(itemsById);
    var axesOut = {};
    var tallies = {};
    var order = CMP_AXES.concat(['seed']);
    order.forEach(function (ax) { tallies[ax] = new Map(); });
    var bump = function (ax, key, label, s, res) {
        var t = tallies[ax].get(key);
        if (!t) { t = { key: key, label: label, n: 0, s: 0, w: 0, l: 0, t: 0 }; tallies[ax].set(key, t); }
        t.n++; t.s += s;
        if (res === 'w') t.w++; else if (res === 'l') t.l++; else t.t++;
    };
    var total = 0, attributable = 0, nonAttr = 0;
    var extra = new Map();
    var matchedVotes = {};
    for (var i = 0; i < (comparisons || []).length; i++) {
        var c = comparisons[i];
        if (!c || c.a === c.b) continue;
        if (c.winner !== 'a' && c.winner !== 'b' && c.winner !== 'tie') continue;
        var A = lookup(c.a), B = lookup(c.b);
        if (!A || !B) continue;
        if (restrict && !(restrict.has(c.a) && restrict.has(c.b))) continue;
        total++;
        var df = axisDiff(A, B, { meta: m });
        var single = df.ctxEqual && (df.axes.length === 1);
        if (single) {
            var ax = df.axes[0];
            var sa = itemSig(A, m), sb = itemSig(B, m);
            var va = c.winner === 'a' ? 1 : c.winner === 'tie' ? 0.5 : 0;
            bump(ax, sa.axes[ax], sa.labels[ax], va, c.winner === 'a' ? 'w' : c.winner === 'b' ? 'l' : 't');
            bump(ax, sb.axes[ax], sb.labels[ax], 1 - va, c.winner === 'b' ? 'w' : c.winner === 'a' ? 'l' : 't');
            matchedVotes[ax] = (matchedVotes[ax] || 0) + 1;
            if (ax !== 'seed') attributable++;
        } else {
            nonAttr++;
            var dd = axisDiff(A, B, { meta: m, withCtxKeys: true });
            var keys = dd.ctxKeys.slice();
            if (dd.axes.length > 1) keys = keys.concat(dd.axes);
            if (!keys.length) keys.push('(identical settings)');
            keys.forEach(function (k) { extra.set(k, (extra.get(k) || 0) + 1); });
        }
    }
    // mean rating per axis value over rated items
    var means = {};
    order.forEach(function (ax) { means[ax] = new Map(); });
    if (ratings && ratings.size) {
        var consider = function (d) {
            var r = ratings.get(d.id);
            if (!r) return;
            var sg = itemSig(d, m);
            order.forEach(function (ax) {
                var k = sg.axes[ax];
                var e = means[ax].get(k);
                if (!e) { e = { sum: 0, n: 0 }; means[ax].set(k, e); }
                e.sum += r.rating; e.n++;
            });
        };
        if (itemsById instanceof Map) itemsById.forEach(function (d, id) { if (!restrict || restrict.has(id)) consider(d); });
        else Object.keys(itemsById).forEach(function (id) { var d = itemsById[id]; if (!restrict || restrict.has(d.id)) consider(d); });
    }
    order.forEach(function (ax) {
        var rows = [];
        tallies[ax].forEach(function (t) {
            var w = wilsonInterval(t.s, t.n);
            var mr = means[ax].get(t.key);
            rows.push({
                value: t.key, label: t.label, n: t.n, wins: t.w, losses: t.l, ties: t.t,
                winPct: t.n ? t.s / t.n : 0, wilsonLow: w.low, wilsonHigh: w.high,
                meanRating: mr && mr.n ? mr.sum / mr.n : null, enough: t.n >= minN
            });
        });
        rows.sort(function (x, y) { return (y.winPct - x.winPct) || (y.n - x.n); });
        var best = null;
        rows.forEach(function (r) {
            if (!r.enough) return;
            if (!best || r.wilsonLow > best.wilsonLow || (r.wilsonLow === best.wilsonLow && r.winPct > best.winPct)) best = r;
        });
        if (best) best.best = true;
        axesOut[ax] = { rows: rows, best: best ? best.value : null, matchedVotes: matchedVotes[ax] || 0 };
    });
    var topExtraKeys = Array.from(extra.entries()).map(function (e) { return { key: e[0], count: e[1] }; })
        .sort(function (x, y) { return y.count - x.count || (x.key < y.key ? -1 : 1); }).slice(0, 5);
    return { axes: axesOut, attributable: attributable, total: total, nonAttributable: nonAttr, topExtraKeys: topExtraKeys };
}

function cmpCsvCell(v) {
    var s = v === null || v === undefined ? '' : String(v);
    return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

function scoreboardToCSV(sb) {
    var lines = ['axis,value,n,wins,losses,ties,win_pct,wilson_low,wilson_high,mean_rating,best'];
    Object.keys(sb.axes).forEach(function (ax) {
        sb.axes[ax].rows.forEach(function (r) {
            lines.push([
                ax, r.value, r.n, r.wins, r.losses, r.ties,
                (r.winPct * 100).toFixed(1), (r.wilsonLow * 100).toFixed(1), (r.wilsonHigh * 100).toFixed(1),
                r.meanRating === null ? '' : r.meanRating.toFixed(1), r.value === sb.axes[ax].best ? 'true' : 'false'
            ].map(cmpCsvCell).join(','));
        });
    });
    return lines.join('\n');
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CMP_AXES: CMP_AXES, CMP_DIFF_AXES: CMP_DIFF_AXES, CMP_MODES: CMP_MODES,
        CMP_EXCLUDED_KEYS: CMP_EXCLUDED_KEYS, CMP_EXCLUDED_RE: CMP_EXCLUDED_RE,
        fmtNum: fmtNum, normalizeAxis: normalizeAxis, itemSig: itemSig, axisDiff: axisDiff,
        mergeComparisons: mergeComparisons, cmpVoteKey: cmpVoteKey, cmpPairKey: cmpPairKey,
        computeBradleyTerry: computeBradleyTerry, buildPairSource: buildPairSource, nextPair: nextPair,
        wilsonInterval: wilsonInterval, computeScoreboard: computeScoreboard, scoreboardToCSV: scoreboardToCSV
    };
}
