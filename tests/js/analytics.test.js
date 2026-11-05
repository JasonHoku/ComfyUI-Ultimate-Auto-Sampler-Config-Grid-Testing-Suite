'use strict';
const test = require('node:test');
const assert = require('node:assert');
const A = require('../../resources/logic_analytics.js');

let nextId = 1;
function item(o) {
    return Object.assign({
        id: nextId++, file: '/view?x', model: 'm.safetensors', lora: 'None', sampler: 'euler', scheduler: 'simple',
        cfg: 7, steps: 20, denoise: 1, seed: 1, width: 1024, height: 1024,
        config_positive: 'a cat', positive: 'a cat', negative: '', duration: 1, favorited: false
    }, o);
}
function many(n, o) { const out = []; for (let i = 0; i < n; i++) out.push(item(typeof o === 'function' ? o(i) : o)); return out; }
const axis = (res, id) => res.axes.find(a => a.id === id);
const row = (res, id, key) => axis(res, id).rows.find(r => r.key === key);

test('grouping by sampler with fav and reject rates', () => {
    const items = [
        ...many(3, i => ({ sampler: 'euler', favorited: i < 2 })),
        ...many(3, i => ({ sampler: 'dpmpp', rejected: i === 0 }))
    ];
    const res = A.computeAnalytics(items, {});
    const s = axis(res, 'sampler');
    assert.deepStrictEqual(s.rows.map(r => r.key), ['euler', 'dpmpp']);
    assert.ok(Math.abs(s.rows[0].favRate - 2 / 3) < 1e-9);
    assert.ok(Math.abs(s.rows[1].rejRate - 1 / 3) < 1e-9);
    assert.strictEqual(res.overall.n, 6);
    assert.strictEqual(res.total, 6);
    assert.strictEqual(res.favorited, 2);
    assert.strictEqual(res.rejected, 1);
});

test('duration mean skips missing, zero and NaN', () => {
    const items = [item({ duration: 2 }), item({ duration: 4 }), item({ duration: 0 }), item({ duration: NaN }), item({ duration: undefined })];
    const res = A.computeAnalytics(items, {});
    assert.strictEqual(res.overall.durMean, 3);
    assert.strictEqual(res.overall.durN, 2);
    const none = A.computeAnalytics([item({ duration: 0 })], {});
    assert.strictEqual(none.overall.durMean, null);
});

test('metric means count only items that have the metric; key order', () => {
    const items = [
        item({ metrics: { sobel: 0.2, sharpness: 10, aesthetic: 5 } }),
        item({ metrics: { sharpness: 30 } }),
        item({})
    ];
    const res = A.computeAnalytics(items, { metricOrder: ['sharpness', 'sobel'] });
    assert.deepStrictEqual(res.metricKeys, ['sharpness', 'sobel', 'aesthetic']);
    assert.deepStrictEqual(res.overall.metrics.sharpness, { mean: 20, n: 2 });
    assert.deepStrictEqual(res.overall.metrics.sobel, { mean: 0.2, n: 1 });
});

test('ratings map feeds ratingMean/ratingN; hasRatings false without votes', () => {
    const a = item({}), b = item({}), c = item({});
    const ratings = new Map([[a.id, { rating: 1500, votes: 4 }], [b.id, { rating: 1600, votes: 2 }]]);
    const res = A.computeAnalytics([a, b, c], { ratings });
    assert.strictEqual(res.hasRatings, true);
    assert.strictEqual(res.overall.ratingMean, 1550);
    assert.strictEqual(res.overall.ratingN, 2);
    assert.strictEqual(A.computeAnalytics([a], { ratings: new Map() }).hasRatings, false);
    assert.strictEqual(A.computeAnalytics([a], { ratings: null }).hasRatings, false);
    assert.strictEqual(A.computeAnalytics([a], { ratings: new Map([[a.id, { rating: 1500, votes: 0 }]]) }).hasRatings, false);
});

test('normalizeAxis reuse: lora stack order, model path separators, upscale base resolution', () => {
    const items = [
        item({ lora: 'b.safetensors:1 + a.safetensors:0.8' }),
        item({ lora: 'a.safetensors:0.8 + b.safetensors:1' }),
        item({ model: 'sub\\m.safetensors' }),
        item({ model: 'sub/m.safetensors' }),
        item({ width: 2048, height: 2048, base_width: 1024, base_height: 1024, upscaled: true })
    ];
    const res = A.computeAnalytics(items, {});
    const lora = axis(res, 'lora');
    assert.strictEqual(lora.rows.find(r => r.n === 2 && r.key !== 'None').n, 2);
    assert.strictEqual(axis(res, 'model').rows.find(r => r.key === 'sub/m.safetensors').n, 2);
    const reso = axis(res, 'resolution');
    assert.strictEqual(reso.rows.length, 1);
    assert.strictEqual(reso.rows[0].n, 5);
});

test('attention_mode and pass axes', () => {
    const items = [item({}), item({ attention_mode: 'sage' }), item({ preview: true })];
    const res = A.computeAnalytics(items, {});
    assert.deepStrictEqual(axis(res, 'attention_mode').rows.map(r => r.key).sort(), ['default', 'sage']);
    assert.strictEqual(row(res, 'attention_mode', 'default').n, 2);
    assert.deepStrictEqual(axis(res, 'pass').rows.map(r => r.key).sort(), ['Full', 'Preview']);
});

test('workflow param axes, constants and the distinct cap', () => {
    const items = many(70, i => ({ wf_params: { 'KSampler.cfg': i % 2 ? 4 : 5, 'Node.fixed': 'x', 'Node.seed': i } }));
    const res = A.computeAnalytics(items, {});
    const cfg = axis(res, 'wf:KSampler.cfg');
    assert.ok(cfg && cfg.kind === 'wf' && cfg.rows.length === 2 && !cfg.constant);
    assert.strictEqual(axis(res, 'wf:Node.fixed').constant, true);
    assert.strictEqual(axis(res, 'wf:Node.seed'), undefined);
    assert.deepStrictEqual(res.skippedAxes, [{ id: 'wf:Node.seed', distinct: 70 }]);
});

test('constant builtin axes are flagged', () => {
    const res = A.computeAnalytics(many(4, i => ({ sampler: i % 2 ? 'a' : 'b' })), {});
    assert.strictEqual(axis(res, 'model').constant, true);
    assert.strictEqual(axis(res, 'sampler').constant, false);
});

test('sortAnalyticsRows: enough first, metric desc nulls last, label asc', () => {
    const mk = (label, n, favRate, sharp) => ({ key: label, label, n, favRate, rejRate: 0, durMean: null, ratingMean: null, enough: n >= 3,
        metrics: { sharpness: sharp === null ? null : { mean: sharp, n } } });
    const rows = [mk('small', 1, 1, 999), mk('big', 4, 0.75, 10), mk('mid', 3, 0.5, 30), mk('nomet', 3, 0.9, null)];
    assert.deepStrictEqual(A.sortAnalyticsRows(rows, 'favRate', -1).map(r => r.label), ['nomet', 'big', 'mid', 'small']);
    assert.deepStrictEqual(A.sortAnalyticsRows(rows, 'metric:sharpness', -1).map(r => r.label), ['mid', 'big', 'nomet', 'small']);
    assert.deepStrictEqual(A.sortAnalyticsRows(rows, 'metric:sharpness', 1).map(r => r.label), ['big', 'mid', 'nomet', 'small']);
    assert.deepStrictEqual(A.sortAnalyticsRows(rows, 'label', 1).map(r => r.label), ['big', 'mid', 'nomet', 'small']);
    assert.strictEqual(rows[0].label, 'small');   // input not mutated
});

test('analyticsToMarkdown: exact output for a small input', () => {
    const items = [
        item({ sampler: 'euler', favorited: true, duration: 2, metrics: { sharpness: 100.123 } }),
        item({ sampler: 'euler', duration: 4, metrics: { sharpness: 200 } }),
        item({ sampler: 'dpm|pp', rejected: true, duration: 3 })
    ];
    const res = A.computeAnalytics(items, { metricOrder: ['sharpness'] });
    const md = A.analyticsToMarkdown(res, { title: 'demo', scopeLabel: 'whole session' });
    const expected = [
        '## USCG stats: demo (whole session, 3 items)',
        "_Fav % / Rej % = share of the group's items; Avg s = mean duration; metric columns = mean over items that have it; rows with n < 3 are marked *._",
        '',
        '### Sampler',
        '| Value | n | Fav % | Rej % | Avg s | Sharpness |',
        '|---|---:|---:|---:|---:|---:|',
        '| euler | 2* | 50.0 | 0.0 | 3.00 | 150 |',
        '| dpm\\|pp | 1* | 0.0 | 100.0 | 3.00 |  |',
        '',
        'Same for every item: Model = m, LoRA = None, Scheduler = simple, Steps = 20, CFG = 7, Denoise = 1, Resolution = 1024x1024, Attention = default, Upscale = none, Pass = Full, Prompt = a cat',
        ''
    ].join('\n');
    assert.strictEqual(md, expected);
    assert.ok(!/Rating/.test(md));
    const withR = A.computeAnalytics(items, { ratings: new Map([[items[0].id, { rating: 1523.4, votes: 5 }]]) });
    const md2 = A.analyticsToMarkdown(withR, {});
    assert.ok(md2.includes('| Rating |'));
    assert.ok(md2.includes('1523 (1)'));
});

test('one pass: favorited read once per item, and large input is fast', () => {
    const reads = new Map();
    const items = many(200, {}).map(d => new Proxy(d, {
        get(t, k) { if (k === 'favorited') reads.set(t.id, (reads.get(t.id) || 0) + 1); return t[k]; }
    }));
    A.computeAnalytics(items, {});
    assert.strictEqual(reads.size, 200);
    for (const v of reads.values()) assert.strictEqual(v, 1);

    const big = many(5000, i => ({
        sampler: ['euler', 'dpmpp', 'lms'][i % 3], cfg: 3 + (i % 5),
        metrics: { sharpness: i, entropy: 5, colorfulness: 20, contrast: 0.2, luma: 0.5, saturation: 0.4, sobel: 0.05 },
        wf_params: { 'A.x': i % 7, 'B.y': i % 4 }
    }));
    const t0 = Date.now();
    const res = A.computeAnalytics(big, {});
    assert.ok(Date.now() - t0 < 500, 'took ' + (Date.now() - t0) + ' ms');
    assert.strictEqual(res.total, 5000);
});

test('empty input does not throw', () => {
    for (const input of [[], null, undefined]) {
        const res = A.computeAnalytics(input, {});
        assert.strictEqual(res.total, 0);
        assert.deepStrictEqual(res.axes, []);
        assert.strictEqual(res.overall.durMean, null);
        assert.ok(A.analyticsToMarkdown(res, {}).startsWith('## USCG stats'));
    }
});
