'use strict';
const test = require('node:test');
const assert = require('node:assert');
const R = require('../../resources/logic_ratings.js');

function rngSeed(seed) {
    let a = seed >>> 0;
    return function () {
        a |= 0; a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
let nextId = 1;
function item(o) {
    return Object.assign({
        id: nextId++, file: '/view?x', model: 'm.safetensors', lora: 'None', sampler: 'euler', scheduler: 'simple',
        cfg: 7, steps: 20, denoise: 1, seed: 1, width: 1024, height: 1024,
        config_positive: 'a cat', positive: 'a cat', negative: '', duration: 1, favorited: false
    }, o);
}
const idOf = (x) => (typeof x === 'object' ? x.id : x);
const vote = (a, b, winner, ts) => ({ a: idOf(a), b: idOf(b), winner, ts: ts || 1, mode: 'matched' });

// ---- normalizers ----
test('fmtNum', () => {
    assert.strictEqual(R.fmtNum(1.0), '1');
    assert.strictEqual(R.fmtNum('7.50'), '7.5');
    assert.strictEqual(R.fmtNum(undefined), '?');
    assert.strictEqual(R.fmtNum('abc'), 'abc');
});
test('lora keys are order-insensitive and legacy-equal', () => {
    const k = (l) => R.normalizeAxis('lora', { lora: l }, {}).key;
    assert.strictEqual(k('a\\b.safetensors:1.0:1.0 + c:0.5'), k('c:0.5:0.5 + a/b.safetensors:1:1'));
    assert.strictEqual(k('x'), k('x:1:1'));
    assert.strictEqual(k('None'), k(''));
    assert.strictEqual(k(undefined), 'None');
    assert.strictEqual(R.normalizeAxis('lora', { lora: 'sub/foo.safetensors:0.8:0.5' }, {}).label, 'foo@0.8/0.5');
});
test('lora prefers lora_expanded', () => {
    assert.strictEqual(R.normalizeAxis('lora', { lora: 'a', lora_expanded: 'b:1:1' }, {}).key, 'b:1:1');
});
test('prompt axis uses config_positive (trigger words do not matter)', () => {
    const a = item({ positive: 'trigger, a cat', config_positive: 'a  cat ' });
    const b = item({ positive: 'a cat', config_positive: 'a cat' });
    assert.strictEqual(R.normalizeAxis('prompt', a, {}).key, R.normalizeAxis('prompt', b, {}).key);
    assert.strictEqual(R.normalizeAxis('prompt', { config_positive: 'x'.repeat(100) }, {}).label.length, 61);
});
test('model falls back to meta and strips path', () => {
    assert.strictEqual(R.normalizeAxis('model', {}, { model: 'dir\\foo.safetensors' }).label, 'foo');
    assert.strictEqual(R.normalizeAxis('model', {}, {}).key, 'Default');
});
test('resolution of an upscaled item maps back to base size', () => {
    const d = { width: 1536, height: 2304, upscaled: true, upscale_ratio: 1.5 };
    assert.strictEqual(R.normalizeAxis('resolution', d, {}).key, '1024x1536');
    const arr = { width: 2304, height: 2304, upscaled: true, upscale_ratio: [1.5, 1.5] };
    assert.strictEqual(R.normalizeAxis('resolution', arr, {}).key, '1024x1024');
    assert.strictEqual(R.normalizeAxis('resolution', { width: 100, height: 50, upscaled: true }, {}).key, 'AR 2.00');
    assert.strictEqual(R.normalizeAxis('resolution', { width: 1, height: 1, base_width: 512, base_height: 768, upscaled: true }, {}).key, '512x768');
    assert.strictEqual(R.normalizeAxis('resolution', { width: 640, height: 480 }, {}).key, '640x480');
});
test('upscale key is stable for array modes', () => {
    const d = { upscaled: true, upscale_mode: ['model_only', 'hires'], upscale_model: ['dir/4x.pth', 'x/y.pth'], upscale_ratio: [2, 1.5], upscale_denoise: 0.35 };
    const k1 = R.normalizeAxis('upscale', d, {}).key;
    assert.strictEqual(k1, R.normalizeAxis('upscale', JSON.parse(JSON.stringify(d)), {}).key);
    assert.ok(k1.includes('model_only>hires') && k1.includes('x2>1.5') && k1.includes('d0.35'));
    assert.strictEqual(R.normalizeAxis('upscale', {}, {}).key, 'none');
});

// ---- axisDiff ----
test('axisDiff single, double, ctx and excluded keys', () => {
    const a = item({});
    assert.deepStrictEqual(R.axisDiff(a, item({ sampler: 'dpmpp' })).axes, ['sampler']);
    assert.deepStrictEqual(R.axisDiff(a, item({ sampler: 'dpmpp', cfg: 5 })).axes, ['sampler', 'cfg'].sort((x, y) => R.CMP_DIFF_AXES.indexOf(x) - R.CMP_DIFF_AXES.indexOf(y)));
    const v = R.axisDiff(a, item({ vae: 'x.vae' }), { withCtxKeys: true });
    assert.strictEqual(v.ctxEqual, false);
    assert.deepStrictEqual(v.ctxKeys, ['vae']);
    const e = R.axisDiff(a, item({ conditioning_pos_hash: 'zz', duration: 99, favorited: true, gen_index: 5, upscale_x: 1, worker_id: 'w' }));
    assert.deepStrictEqual(e.axes, []);
    assert.strictEqual(e.ctxEqual, true);
    assert.deepStrictEqual(R.axisDiff(a, item({ seed: 2 })).axes, ['seed']);
});

test('dist_worker (which machine rendered it) never makes two items different settings', () => {
    const a = item({ dist_worker: 'master' });
    const b = item({ dist_worker: 'w_1a2b3c4d' });
    assert.ok(R.CMP_EXCLUDED_KEYS.has('dist_worker'));
    assert.deepStrictEqual(R.axisDiff(a, b).axes, []);
    assert.strictEqual(R.axisDiff(a, b, { withCtxKeys: true }).ctxEqual, true);
});

// ---- Bradley-Terry ----
test('BT: A beats B 3-1', () => {
    const A = item({}), B = item({});
    const cs = [vote(A, B, 'a', 1), vote(A, B, 'a', 2), vote(B, A, 'b', 3), vote(A, B, 'b', 4)];
    const r = R.computeBradleyTerry(cs, new Set([A.id, B.id]));
    assert.ok(r.get(A.id).rating > r.get(B.id).rating);
    assert.deepStrictEqual([r.get(A.id).w, r.get(A.id).l, r.get(A.id).t, r.get(A.id).votes], [3, 1, 0, 4]);
    assert.deepStrictEqual([r.get(B.id).w, r.get(B.id).l], [1, 3]);
});
test('BT: unbeaten item is finite', () => {
    const A = item({}), B = item({});
    const cs = [1, 2, 3, 4, 5].map((t) => vote(A, B, 'a', t));
    const r = R.computeBradleyTerry(cs, new Set([A.id, B.id]));
    assert.ok(Number.isFinite(r.get(A.id).rating) && Number.isFinite(r.get(B.id).rating));
    assert.ok(r.get(A.id).rating > 1500 && r.get(B.id).rating < 1500);
});
test('BT: all ties give equal 1500', () => {
    const [A, B, C] = [item({}), item({}), item({})];
    const cs = [vote(A, B, 'tie', 1), vote(B, C, 'tie', 2), vote(B, C, 'tie', 3)];
    const r = R.computeBradleyTerry(cs, new Set([A.id, B.id, C.id]));
    for (const id of [A.id, B.id, C.id]) assert.ok(Math.abs(r.get(id).rating - 1500) < 1e-6);
});
test('BT: chain order and order invariance', () => {
    const [A, B, C] = [item({}), item({}), item({})];
    const cs = [vote(A, B, 'a', 1), vote(B, C, 'a', 2)];
    const ids = new Set([A.id, B.id, C.id]);
    const r = R.computeBradleyTerry(cs, ids);
    assert.ok(r.get(A.id).rating > r.get(B.id).rating && r.get(B.id).rating > r.get(C.id).rating);
    const r2 = R.computeBradleyTerry(cs.slice().reverse(), ids);
    for (const id of ids) assert.ok(Math.abs(r.get(id).rating - r2.get(id).rating) < 1e-9);
});
test('BT: empty input, dangling ids, convergence', () => {
    assert.strictEqual(R.computeBradleyTerry([], new Set([1])).size, 0);
    const [A, B] = [item({}), item({})];
    const r = R.computeBradleyTerry([vote(A, 99999999, 'a', 1), vote(A, B, 'a', 2), vote(A, A, 'a', 3)], new Set([A.id, B.id]));
    assert.strictEqual(r.get(A.id).votes, 1);
    const items = Array.from({ length: 30 }, () => item({}));
    const rng = rngSeed(7);
    const cs = [];
    for (let i = 0; i < 600; i++) {
        const x = items[Math.floor(rng() * 30)], y = items[Math.floor(rng() * 30)];
        if (x === y) continue;
        cs.push(vote(x, y, (x.id % 7) > (y.id % 7) ? 'a' : 'b', i + 1));
    }
    const big = R.computeBradleyTerry(cs, new Set(items.map((d) => d.id)), { maxIter: 200 });
    for (const v of big.values()) assert.ok(Number.isFinite(v.rating));
});

// ---- pairing ----
const freshState = () => ({ votesById: new Map(), votedPairs: new Set(), skipped: new Set(), allowRepeats: false });
function drawMany(src, n, rng, state) {
    const out = [];
    for (let i = 0; i < n; i++) { const p = R.nextPair(src, state || freshState(), rng); if (p) out.push(p); }
    return out;
}
function grid() {
    const items = [];
    for (const p of ['cat', 'dog']) for (const s of [1, 2]) for (const sm of ['euler', 'dpm', 'heun'])
        items.push(item({ config_positive: p, positive: p, seed: s, sampler: sm }));
    return items;
}
test('matched pairs share prompt+seed and differ in config', () => {
    const src = R.buildPairSource(grid(), 'matched', '');
    const ps = drawMany(src, 200, rngSeed(1));
    assert.ok(ps.length > 0);
    for (const p of ps) {
        assert.strictEqual(p.a.config_positive, p.b.config_positive);
        assert.strictEqual(p.a.seed, p.b.seed);
        assert.notStrictEqual(p.a.sampler, p.b.sampler);
    }
});
test('one_axis pairs differ in exactly the focus axis', () => {
    const items = grid();
    items.push(item({ config_positive: 'cat', positive: 'cat', seed: 1, cfg: 9 }));
    const src = R.buildPairSource(items, 'one_axis', 'sampler');
    const ps = drawMany(src, 100, rngSeed(2));
    assert.ok(ps.length > 0);
    for (const p of ps) assert.deepStrictEqual(R.axisDiff(p.a, p.b).axes, ['sampler']);
    const src2 = R.buildPairSource(items, 'one_axis', '');
    for (const p of drawMany(src2, 100, rngSeed(3))) assert.strictEqual(R.axisDiff(p.a, p.b).axes.length, 1);
});
test('seed mode pairs differ only in seed', () => {
    const src = R.buildPairSource(grid(), 'seed', '');
    const ps = drawMany(src, 100, rngSeed(4));
    assert.ok(ps.length);
    for (const p of ps) assert.deepStrictEqual(R.axisDiff(p.a, p.b).axes, ['seed']);
});
test('no valid bucket gives null; media types never mix', () => {
    const src = R.buildPairSource([item({}), item({ seed: 5 })], 'matched', '');
    assert.strictEqual(src.buckets.length, 0);
    assert.strictEqual(R.nextPair(src, freshState(), rngSeed(1)), null);
    const items = [item({ sampler: 'a' }), item({ sampler: 'b', media_type: 'video' }), item({ sampler: 'c' }), item({ sampler: 'd', media_type: 'video' })];
    const rs = R.buildPairSource(items, 'random', '');
    const ps = drawMany(rs, 100, rngSeed(5));
    assert.ok(ps.length > 0);
    for (const p of ps) assert.strictEqual(p.a.media_type === 'video', p.b.media_type === 'video');
});
test('voted pairs are skipped until exhausted, then repeats', () => {
    const items = [item({ sampler: 'a' }), item({ sampler: 'b' }), item({ sampler: 'c' })];
    const src = R.buildPairSource(items, 'matched', '');
    const state = freshState();
    const rng = rngSeed(11);
    const seen = new Set();
    for (let i = 0; i < 3; i++) {
        const p = R.nextPair(src, state, rng);
        assert.ok(p && !p.repeating);
        const k = R.cmpPairKey(p.a.id, p.b.id);
        assert.ok(!state.votedPairs.has(k));
        state.votedPairs.add(k); seen.add(k);
        state.votesById.set(p.a.id, (state.votesById.get(p.a.id) || 0) + 1);
        state.votesById.set(p.b.id, (state.votesById.get(p.b.id) || 0) + 1);
    }
    assert.strictEqual(seen.size, 3);
    const p = R.nextPair(src, state, rng);
    assert.ok(p && p.repeating === true);
});
test('skipped pairs are excluded even with repeats', () => {
    const items = [item({ sampler: 'a' }), item({ sampler: 'b' })];
    const src = R.buildPairSource(items, 'matched', '');
    const st = freshState();
    st.skipped.add(R.cmpPairKey(items[0].id, items[1].id));
    st.allowRepeats = true;
    assert.strictEqual(R.nextPair(src, st, rngSeed(1)), null);
});
test('low-vote items are preferred', () => {
    const items = Array.from({ length: 20 }, (_, i) => item({ sampler: 's' + i }));
    const src = R.buildPairSource(items, 'matched', '');
    const st = freshState();
    items.slice(0, 10).forEach((d) => st.votesById.set(d.id, 50));
    const low = new Set(items.slice(10).map((d) => d.id));
    let lowCount = 0, tot = 0;
    for (const p of drawMany(src, 200, rngSeed(9), st)) {
        tot += 2; lowCount += (low.has(p.a.id) ? 1 : 0) + (low.has(p.b.id) ? 1 : 0);
    }
    assert.ok(lowCount / tot > 0.8, 'low-vote share ' + lowCount / tot);
});

// ---- scoreboard ----
test('scoreboard: 3-sampler dataset, ties half, best row, n<3', () => {
    const [E, D, H] = ['euler', 'dpm', 'heun'].map((s) => item({ sampler: s }));
    const byId = new Map([E, D, H].map((d) => [d.id, d]));
    const cs = [];
    let ts = 1;
    for (let i = 0; i < 4; i++) cs.push(vote(E, D, 'a', ts++));
    cs.push(vote(D, E, 'tie', ts++));
    cs.push(vote(H, D, 'a', ts++));
    const ids = new Set(byId.keys());
    const sb = R.computeScoreboard(cs, byId, R.computeBradleyTerry(cs, ids), {});
    const rows = Object.fromEntries(sb.axes.sampler.rows.map((r) => [r.value, r]));
    assert.strictEqual(rows.euler.n, 5);
    assert.strictEqual(rows.euler.wins, 4); assert.strictEqual(rows.euler.ties, 1);
    assert.ok(Math.abs(rows.euler.winPct - 4.5 / 5) < 1e-9);
    assert.strictEqual(rows.dpm.n, 6);
    assert.ok(Math.abs(rows.dpm.winPct - 0.5 / 6) < 1e-9);
    assert.strictEqual(rows.heun.n, 1); assert.strictEqual(rows.heun.enough, false);
    assert.strictEqual(sb.axes.sampler.best, 'euler');
    assert.strictEqual(sb.attributable, 6); assert.strictEqual(sb.total, 6);
    assert.ok(rows.euler.meanRating > rows.dpm.meanRating);
    assert.ok(rows.euler.wilsonLow < rows.euler.winPct && rows.euler.wilsonHigh > rows.euler.winPct);
});
test('scoreboard: non-attributable counted, seed is noise floor, restrictIds', () => {
    const a = item({}), b = item({ sampler: 'x', cfg: 5 }), c = item({ vae: 'v1' }), d = item({ seed: 9 });
    const byId = new Map([a, b, c, d].map((x) => [x.id, x]));
    const cs = [vote(a, b, 'a', 1), vote(a, c, 'a', 2), vote(a, c, 'b', 3), vote(a, d, 'tie', 4)];
    const sb = R.computeScoreboard(cs, byId, new Map(), {});
    assert.strictEqual(sb.total, 4); assert.strictEqual(sb.attributable, 0); assert.strictEqual(sb.nonAttributable, 3);
    assert.strictEqual(sb.topExtraKeys[0].key, 'vae'); assert.strictEqual(sb.topExtraKeys[0].count, 2);
    assert.ok(sb.topExtraKeys.some((k) => k.key === 'cfg') && sb.topExtraKeys.some((k) => k.key === 'sampler'));
    assert.strictEqual(sb.axes.seed.rows.length, 2);
    const only = R.computeScoreboard(cs, byId, new Map(), { restrictIds: new Set([a.id, b.id]) });
    assert.strictEqual(only.total, 1);
    assert.strictEqual(R.computeScoreboard([vote(a, 424242, 'a', 9)], byId, new Map(), {}).total, 0);
});
test('scoreboard CSV quoting + wilson', () => {
    const a = item({ config_positive: 'a, "quoted" line' }), b = item({ config_positive: 'plain' });
    const byId = new Map([a, b].map((x) => [x.id, x]));
    const cs = [vote(a, b, 'a', 1), vote(a, b, 'a', 2), vote(b, a, 'b', 3)];
    const csv = R.scoreboardToCSV(R.computeScoreboard(cs, byId, new Map(), {}));
    assert.ok(csv.startsWith('axis,value,n,wins,losses,ties,win_pct,wilson_low,wilson_high,mean_rating,best\n'));
    assert.ok(csv.includes('"a, ""quoted"" line"'));
    const w = R.wilsonInterval(5, 10);
    assert.ok(w.low < 0.5 && w.high > 0.5);
    assert.deepStrictEqual(R.wilsonInterval(0, 0), { low: 0, high: 0 });
});
test('mergeComparisons dedupes by ts:a:b', () => {
    const m = R.mergeComparisons([{ a: 1, b: 2, ts: 5 }], [{ a: 1, b: 2, ts: 5 }, { a: 1, b: 2, ts: 6 }, { a: 2, b: 1, ts: 5 }]);
    assert.strictEqual(m.length, 3);
});

// ---- wave 2 polish ----
test('exact base dims win over a conflicting ratio guess; also on a non-upscaled item', () => {
    const d = { width: 1248, height: 1824, upscaled: true, upscale_ratio: 1.5, base_width: 832, base_height: 1216 };
    assert.strictEqual(R.normalizeAxis('resolution', d, {}).key, '832x1216');
    assert.strictEqual(R.normalizeAxis('resolution', { width: 100, height: 100, base_width: 64, base_height: 96 }, {}).key, '64x96');
});
test('per-image fields never make two items different settings (F2)', () => {
    const a = item({ config_fp: 'a', base_fp: 'x', metrics: { s: 1 }, preview_of: 5, source_relpath: 'p/a.png' });
    const b = item({ cfg: 5, config_fp: 'b', base_fp: 'y', metrics: { s: 2 }, preview_of: 6, source_relpath: 'p/b.png',
        florence2_bbox: [1, 2, 3, 4], florence2_detection_count: 2 });
    const df = R.axisDiff(a, b, { withCtxKeys: true });
    assert.strictEqual(df.ctxEqual, true, JSON.stringify(df.ctxKeys));
    assert.deepStrictEqual(df.axes, ['cfg']);
});

test('cmpSwipeSide decisions', () => {
    assert.strictEqual(R.cmpSwipeSide(-80, 5, 200, 412), 'a');
    assert.strictEqual(R.cmpSwipeSide(80, 5, 200, 412), 'b');
    assert.strictEqual(R.cmpSwipeSide(40, 0, 200, 412), null);      // too short
    assert.strictEqual(R.cmpSwipeSide(80, 70, 200, 412), null);     // diagonal
    assert.strictEqual(R.cmpSwipeSide(-80, 0, 1500, 412), null);    // slow and short
    assert.strictEqual(R.cmpSwipeSide(-130, 0, 1500, 412), 'a');    // slow but long
    assert.strictEqual(R.cmpSwipeSide(-200, 0, 200, 1600), null);   // width 1600 needs 240 px
    assert.strictEqual(R.cmpSwipeSide(-250, 0, 200, 1600), 'a');
});
