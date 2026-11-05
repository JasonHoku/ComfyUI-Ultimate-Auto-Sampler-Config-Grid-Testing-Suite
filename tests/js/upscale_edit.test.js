'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const E = require('../../resources/logic_upscale_edit.js');

const cases = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'fixtures', 'upscale_validation_cases.json'), 'utf8'));

cases.forEach(function (c) {
    test('validateUpscaleConfig: ' + c.name, () => {
        const r = E.validateUpscaleConfig(c.cfg);
        assert.strictEqual(r.ok, c.expect_ok, JSON.stringify(r.errors));
        const fields = Array.from(new Set(r.errors.map(e => e.field))).sort();
        assert.deepStrictEqual(fields, c.error_fields.slice().sort());
    });
});

test('validate: error paths name the pipeline and step', () => {
    const r = E.validateUpscaleConfig({ pipelines: [{ steps: [{}] }, { steps: [{}, { hires_denoise: '2' }] }] });
    assert.deepStrictEqual(r.errors.map(e => e.path), ['p1.s1']);
});

test('validate: missing model is a warning only when the model list is known', () => {
    const cfg = { pipelines: [{ steps: [{ mode: 'model_only', upscale_models: ['gone.pth'] }] }] };
    assert.strictEqual(E.validateUpscaleConfig(cfg).warnings.length, 0);
    const r = E.validateUpscaleConfig(cfg, { availableModels: ['other.pth'] });
    assert.strictEqual(r.ok, true);
    assert.ok(r.warnings.some(w => w.field === 'upscale_models'));
    assert.strictEqual(E.validateUpscaleConfig(cfg, { availableModels: ['gone.pth'] }).warnings.length, 0);
});

test('validate: florence2 with several denoise values and empty hires prompt are warnings', () => {
    const r = E.validateUpscaleConfig({
        hires_prompt_adjust: true, hires_prompt_text: ' ',
        pipelines: [{ steps: [{ mode: 'florence2_hires', hires_denoise: '0.4,0.5', florence2: { text_input: 'face' } }] }]
    });
    assert.strictEqual(r.ok, true);
    assert.deepStrictEqual(r.warnings.map(w => w.field).sort(), ['hires_denoise', 'hires_prompt_text']);
});

test('moveItem moves in place, clamps nothing, ignores bad indexes', () => {
    const a = ['a', 'b', 'c'];
    assert.strictEqual(E.moveItem(a, 2, 0), a);
    assert.deepStrictEqual(a, ['c', 'a', 'b']);
    E.moveItem(a, 0, 1);
    assert.deepStrictEqual(a, ['a', 'c', 'b']);
    E.moveItem(a, 1, 1); E.moveItem(a, -1, 0); E.moveItem(a, 0, 3); E.moveItem(a, 5, 0);
    assert.deepStrictEqual(a, ['a', 'c', 'b']);
    assert.strictEqual(E.moveItem(null, 0, 1), null);
});

test('configsEqual ignores key order and fills defaults', () => {
    const a = { pipelines: [{ name: 'P', active: true, steps: [{ mode: 'hires_only', upscale_ratios: '1.5', hires_denoise: '0.3' }] }] };
    const b = { hires_prompt_text: '', pipelines: [{ steps: [{ hires_denoise: '0.3', mode: 'hires_only' }], name: 'P' }] };
    assert.ok(E.configsEqual(a, b));
    const c = JSON.parse(JSON.stringify(b));
    c.pipelines[0].steps[0].hires_denoise = '0.4';
    assert.ok(!E.configsEqual(a, c));
    const d = JSON.parse(JSON.stringify(b));
    d.pipelines[0].steps.push({});
    assert.ok(!E.configsEqual(a, d));
});

test('configsEqual notices a reorder', () => {
    const a = { pipelines: [{ name: 'P1', steps: [{}] }, { name: 'P2', steps: [{}] }] };
    const b = JSON.parse(JSON.stringify(a));
    E.moveItem(b.pipelines, 1, 0);
    assert.ok(!E.configsEqual(a, b));
    E.moveItem(b.pipelines, 1, 0);
    assert.ok(E.configsEqual(a, b));
});

test('normalizeUpscaleConfig deep clones and fills every step default', () => {
    const src = { pipelines: [{ steps: [{ mode: 'seedvr2' }] }] };
    const n = E.normalizeUpscaleConfig(src);
    assert.notStrictEqual(n.pipelines[0], src.pipelines[0]);
    assert.strictEqual(src.pipelines[0].steps[0].repeat, undefined);          // input untouched
    const st = n.pipelines[0].steps[0];
    const def = E._defaultUpscaleStep();
    Object.keys(def).forEach(k => assert.ok(k in st, 'missing default ' + k));
    assert.strictEqual(st.mode, 'seedvr2');
    assert.strictEqual(n.pipelines[0].active, true);
    assert.strictEqual(n.hires_prompt_behavior, 'append_end');
    assert.deepStrictEqual(E.normalizeUpscaleConfig(null).pipelines, []);
});

test('presetFromConfig has the Builder preset shape', () => {
    const p = E.presetFromConfig('Mine', { pipelines: [{ name: 'P', steps: [{}] }], hires_prompt_adjust: true, hires_prompt_text: 'x' });
    assert.deepStrictEqual(Object.keys(p).sort(), ['hires_prompt_adjust', 'hires_prompt_behavior', 'hires_prompt_text', 'name', 'pipelines']);
    assert.strictEqual(p.name, 'Mine');
    assert.strictEqual(p.hires_prompt_adjust, true);
    assert.strictEqual(p.pipelines[0].steps[0].mode, 'hires_only');
});

test('countUpscaleOutputs: one output per active pipeline, combos from the last step', () => {
    const cfg = { pipelines: [
        { name: 'P1', steps: [{ mode: 'model_only', upscale_models: ['a', 'b', 'c'] }, { mode: 'model_then_hires', upscale_models: ['m'], upscale_ratios: '1.5, 2', hires_denoise: '0.3,0.4' }] },
        { name: 'P2', steps: [{ mode: 'hires_only' }] },
        { name: 'off', active: false, steps: [{}] },
        { name: 'sv', steps: [{ mode: 'seedvr2' }] },
        { name: 'allOff', steps: [{ active: false }] }
    ] };
    const r = E.countUpscaleOutputs(cfg);
    assert.strictEqual(r.perImage, 3);
    assert.deepStrictEqual(r.perPipeline.map(p => p.name), ['P1', 'P2', 'sv']);
    assert.deepStrictEqual(r.perPipeline.map(p => p.combos), [4, 1, 1]);        // chained steps add work, not outputs
    assert.strictEqual(E.countUpscaleOutputs({ pipelines: [] }).perImage, 0);
});

test('validate: several values in the final step warns that only the last combination is kept', () => {
    const r = E.validateUpscaleConfig({ pipelines: [{ steps: [{ upscale_ratios: '1.5,2' }] }] });
    assert.strictEqual(r.ok, true);
    assert.ok(r.warnings.some(w => w.field === 'combos' && w.path === 'p0.s0'));
    assert.ok(!E.validateUpscaleConfig({ pipelines: [{ steps: [{}] }] }).warnings.some(w => w.field === 'combos'));
});

test('step defaults match the Builder createDefaultStep keys', () => {
    const src = fs.readFileSync(path.join(__dirname, '..', '..', 'web', 'conf_builder', 'conf-builder-config-management.js'), 'utf8');
    const m = src.match(/function createDefaultStep\(\)\s*\{\s*return\s*\{([\s\S]*?)\n\s*\};/);
    assert.ok(m, 'createDefaultStep not found');
    const builderKeys = Array.from(m[1].matchAll(/^\s{8}([a-z0-9_]+):/gm)).map(x => x[1]).sort();
    const modalKeys = Object.keys(E._defaultUpscaleStep()).sort();
    assert.deepStrictEqual(modalKeys, builderKeys);
});
