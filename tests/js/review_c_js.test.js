'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const R = require('../../resources/logic_ratings.js');

const base = { file: '/view?x', model: 'm', lora: 'None', sampler: 'euler', scheduler: 'simple', cfg: 7, steps: 20,
    denoise: 1, seed: 1, width: 1024, height: 1024, config_positive: 'a cat', negative: '' };

test('CMP_EXCLUDED_RE drops every wf_* key', () => {
    for (const k of ['wf_subs', 'wf_params', 'wf_source', 'wf_output', 'wf_output_media', 'wf_frames_from']) {
        assert.ok(R.CMP_EXCLUDED_RE.test(k), k);
    }
    assert.ok(!R.CMP_EXCLUDED_RE.test('sampler'));
});

test('sweep items differing only in wf_* keys have equal compare signatures', () => {
    const a = Object.assign({ id: 1, model_type: 'workflow' }, base, { wf_subs: { x: 1 }, wf_params: { 'K.cfg': 4 }, wf_source: 'a.json', wf_output: '3:0' });
    const b = Object.assign({ id: 2, model_type: 'workflow' }, base, { wf_subs: { x: 2 }, wf_params: { 'K.cfg': 5 }, wf_source: 'b.json', wf_output: '9:0' });
    assert.strictEqual(JSON.stringify(R.itemSig(a)), JSON.stringify(R.itemSig(b)));
});

test('web workflow-sweep render re-reads configArray.workflow (no stale const wf)', () => {
    const src = fs.readFileSync(path.join(__dirname, '../../web/conf_builder/conf-builder-workflow-sweep.js'), 'utf8');
    assert.match(src, /let wf = ensureWorkflowState\(configArray\)/);
    assert.match(src, /function render\(\) \{\s*wf = ensureWorkflowState\(configArray\)/);
});

test('logic_utils export/generate skip model_type workflow (source check; functions need a DOM)', () => {
    const src = fs.readFileSync(path.join(__dirname, '../../resources/logic_utils.js'), 'utf8');
    assert.strictEqual((src.match(/model_type === 'workflow'/g) || []).length, 2);
});
