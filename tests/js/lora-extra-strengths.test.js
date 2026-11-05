// LoRA strength rows (Roadmap 12). The helpers live in an ES module, so this test slices the pure
// functions out of the source and evaluates them (same trick as workflow-config-to-array.test.js).
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '../..');
const src = fs.readFileSync(path.join(root, 'web/conf_builder/conf-builder-utilities.js'), 'utf8');

function grab(name) {
    const re = new RegExp('(?:export )?function ' + name + '\\(');
    const m = re.exec(src);
    assert.ok(m, 'function ' + name + ' not found');
    let i = src.indexOf('{', m.index), depth = 0, j = i;
    for (; j < src.length; j++) {
        if (src[j] === '{') depth++;
        else if (src[j] === '}') { depth--; if (depth === 0) break; }
    }
    return src.slice(m.index, j + 1).replace(/^export /, '');
}
function grabConst(name) {
    const re = new RegExp('(?:export )?const ' + name + ' = ');
    const m = re.exec(src);
    assert.ok(m, 'const ' + name + ' not found');
    // a one-line const, or an arrow function that ends at the first "\n};"
    const eol = src.indexOf('\n', m.index);
    const oneLine = !src.slice(m.index, eol).trimEnd().endsWith('{');
    const stop = oneLine ? eol : src.indexOf('\n};', m.index) + 3;
    return src.slice(m.index, stop).replace(/^export /, '');
}

const code = [
    grabConst('MAX_LORA_STRENGTHS'), grabConst('_num2'),
    ...['normalizePath', 'parseLoraString', 'normalizeExtraStrengths', 'legacyWeightArraysToExtras',
        '_baseOfLoraString', 'loraEntryCount'].map(grab),
].join('\n');
const L = new Function(code + '\nreturn { normalizeExtraStrengths, legacyWeightArraysToExtras, loraEntryCount, MAX_LORA_STRENGTHS };')();

const rows = (...pairs) => pairs.map(([model, clip]) => ({ model, clip }));

test('normalizeExtraStrengths: rounds to 2 decimals, forces clip when locked', () => {
    const r = L.normalizeExtraStrengths(rows([0.604, 0.1], [0.8, 0.4]), true, 1, 1);
    assert.deepStrictEqual(r.effective, rows([0.6, 0.6], [0.8, 0.8]));
    const u = L.normalizeExtraStrengths(rows([0.604, 0.1], [0.8, 0.4]), false, 1, 1);
    assert.deepStrictEqual(u.effective, rows([0.6, 0.1], [0.8, 0.4]));
    assert.deepStrictEqual(L.normalizeExtraStrengths([{ model: 0.7 }], false, 1, 1).effective, rows([0.7, 0.7]));
});

test('normalizeExtraStrengths: drops non-finite / junk rows, tolerates non-lists', () => {
    const bad = [null, 'abc', { model: 'abc' }, { model: null }, { model: NaN }, { model: Infinity }, { model: 0.5, clip: 'x' }, { model: true }, { model: 0.5 }];
    assert.deepStrictEqual(L.normalizeExtraStrengths(bad, true, 1, 1).effective, rows([0.5, 0.5]));
    for (const v of [null, undefined, 'x', 5, {}]) assert.deepStrictEqual(L.normalizeExtraStrengths(v, true, 1, 1), { rows: [], effective: [] });
});

test('normalizeExtraStrengths: duplicates of the base or of an earlier row are flagged, not effective', () => {
    const r = L.normalizeExtraStrengths(rows([1, 1], [0.6, 0.6], [0.604, 0.6], [0.8, 0.8]), true, 1, 1);
    assert.deepStrictEqual(r.rows.map(x => x.duplicate), [true, false, true, false]);
    assert.deepStrictEqual(r.effective, rows([0.6, 0.6], [0.8, 0.8]));
    // unknown base (bracket string): nothing is a base duplicate
    assert.strictEqual(L.normalizeExtraStrengths(rows([1, 1]), true, null, null).effective.length, 1);
});

test('normalizeExtraStrengths: at most 8 strengths in total (7 extra rows)', () => {
    const many = rows(...Array.from({ length: 10 }, (_, i) => [(i + 1) / 10, (i + 1) / 10]));
    assert.strictEqual(L.normalizeExtraStrengths(many, true, 5, 5).rows.length, 7);
    assert.strictEqual(L.MAX_LORA_STRENGTHS, 8);
});

test('legacyWeightArraysToExtras: model list -> CLIP = model, first value becomes the base', () => {
    const r = L.legacyWeightArraysToExtras({ 'a_model': [0.5, 0.8, 1] }, 'a', 1, 1);
    assert.deepStrictEqual(r.base, { model: 0.5, clip: 0.5 });
    assert.deepStrictEqual(r.extras, rows([0.8, 0.8], [1, 1]));
});

test('legacyWeightArraysToExtras: model + clip lists -> every pair, capped at 8; clip-only; nothing to convert', () => {
    const r = L.legacyWeightArraysToExtras({ 'a_model': [0.5, 0.8], 'a_clip': [0.4, 0.9] }, 'a', 1, 1);
    assert.deepStrictEqual([r.base, ...r.extras], rows([0.5, 0.4], [0.5, 0.9], [0.8, 0.4], [0.8, 0.9]));
    const big = L.legacyWeightArraysToExtras({ 'a_model': [1, 2, 3, 4, 5], 'a_clip': [1, 2, 3] }, 'a', 1, 1);
    assert.strictEqual(1 + big.extras.length, 8);
    const c = L.legacyWeightArraysToExtras({ 'a_clip': [0.4, 0.9] }, 'a', 0.7, 0.7);
    assert.deepStrictEqual([c.base, ...c.extras], rows([0.7, 0.4], [0.7, 0.9]));
    assert.strictEqual(L.legacyWeightArraysToExtras({ 'a_model': [1] }, 'a', 1, 1), null);
    assert.strictEqual(L.legacyWeightArraysToExtras({}, 'a', 1, 1), null);
    assert.strictEqual(L.legacyWeightArraysToExtras(null, 'a', 1, 1), null);
    assert.strictEqual(L.legacyWeightArraysToExtras({ 'b_model': [1, 2] }, 'a', 1, 1), null);
});

test('loraEntryCount equals the number of emitted lora strings for every shared fixture case', () => {
    const cases = JSON.parse(fs.readFileSync(path.join(root, 'tests/fixtures/lora_extra_strengths_cases.json'), 'utf8')).cases;
    assert.ok(cases.length >= 8);
    for (const c of cases) {
        assert.strictEqual(L.loraEntryCount(c.array, c.available_loras || null), c.expected.length, c.name);
    }
});

test('loraEntryCount: stacked LoRAs are ONE entry (no per-file overcount), legacy arrays and folders count', () => {
    assert.strictEqual(L.loraEntryCount({ loras: ['a:1.00:1.00', 'b:0.50:0.50'] }, null), 1);
    assert.strictEqual(L.loraEntryCount({ loras: ['None'] }, null), 1);
    assert.strictEqual(L.loraEntryCount({ loras: [] }, null), 1);
    assert.strictEqual(L.loraEntryCount({ loras: ['a:1:1'], lora_weight_arrays: { 'a_model': [1, 2, 3] } }, null), 3);
    assert.strictEqual(L.loraEntryCount({ loras: ['a:1:1'], lora_weight_arrays: { 'a_model': [1, 2], 'a_clip': [1, 2, 3] } }, null), 6);
    const avail = ['foo/x.safetensors', 'foo/y.safetensors', 'z.safetensors'];
    assert.strictEqual(L.loraEntryCount({ loras: ['foo/'] }, avail), 2);
    assert.strictEqual(L.loraEntryCount({ loras: ['foo/', 'z.safetensors:1:1'] }, avail), 3);
    assert.strictEqual(L.loraEntryCount({ loras: ['foo/'], lora_extra_strengths: { 'foo/': rows([0.6, 0.6]) } }, avail), 4);
    assert.strictEqual(L.loraEntryCount({ loras: ['a:1:1'], lora_bypass_states: { a: true } }, null), 1);
});
