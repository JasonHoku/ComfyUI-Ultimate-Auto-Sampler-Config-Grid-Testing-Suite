const test = require('node:test');
const assert = require('node:assert');
const wf = require('../../resources/logic_wfparams.js');

const items = [
    { id: 1, wf_params: { 'KSampler.cfg': 4, 'KSampler.sampler_name': 'euler', 'KSampler.steps': '20', 'A.seed': 1, 'B.seed': 2 } },
    { id: 2, wf_params: { 'KSampler.cfg': 5.5, 'KSampler.sampler_name': 'dpmpp', 'KSampler.steps': 30, 'A.seed': 1, 'B.seed': 2 } },
    { id: 3 },
    { id: 4, wf_params: { 'KSampler.cfg': 5.5, 'KSampler.sampler_name': 'euler' } },
];

test('wfParamKeys: sorted, numeric detection with mixed values, distinct counts', () => {
    const keys = wf.wfParamKeys(items);
    assert.deepStrictEqual(keys.map(k => k.key), ['A.seed', 'B.seed', 'KSampler.cfg', 'KSampler.sampler_name', 'KSampler.steps']);
    const by = Object.fromEntries(keys.map(k => [k.key, k]));
    assert.strictEqual(by['KSampler.cfg'].numeric, true);
    assert.strictEqual(by['KSampler.cfg'].distinct, 2);
    assert.strictEqual(by['KSampler.sampler_name'].numeric, false);
    assert.strictEqual(by['KSampler.steps'].numeric, true);   // '20' and 30 both parse
    assert.strictEqual(by['A.seed'].distinct, 1);
    assert.deepStrictEqual(wf.wfParamKeys(null), []);
    assert.deepStrictEqual(wf.wfParamKeys([{ id: 1 }]), []);
});

test('wfParamKeys: one non-numeric value makes the key non-numeric; cache refreshes on length change', () => {
    const arr = [{ wf_params: { k: 1 } }];
    assert.strictEqual(wf.wfParamKeys(arr)[0].numeric, true);
    assert.strictEqual(wf.wfParamKeys(arr), wf.wfParamKeys(arr));      // cached
    arr.push({ wf_params: { k: 'abc' } });
    assert.strictEqual(wf.wfParamKeys(arr)[0].numeric, false);
});

test('wfParamValue / wfParamNumber', () => {
    assert.strictEqual(wf.wfParamValue(items[0], 'KSampler.cfg'), 4);
    assert.strictEqual(wf.wfParamValue(items[2], 'KSampler.cfg'), null);
    assert.strictEqual(wf.wfParamValue(items[3], 'nope'), null);
    assert.strictEqual(wf.wfParamNumber(items[0], 'KSampler.steps'), 20);
    assert.strictEqual(wf.wfParamNumber(items[0], 'KSampler.sampler_name'), null);
    assert.strictEqual(wf.wfParamNumber(items[2], 'KSampler.cfg'), null);
});

test('wfParamMatches: tolerance for numbers, string equality otherwise', () => {
    assert.ok(wf.wfParamMatches(items[1], 'KSampler.cfg', '5.5'));
    assert.ok(wf.wfParamMatches({ wf_params: { c: 0.1 + 0.2 } }, 'c', 0.3));
    assert.ok(!wf.wfParamMatches(items[1], 'KSampler.cfg', 5.6));
    assert.ok(wf.wfParamMatches(items[0], 'KSampler.sampler_name', 'euler'));
    assert.ok(!wf.wfParamMatches(items[0], 'KSampler.sampler_name', 'Euler'));
    assert.ok(!wf.wfParamMatches(items[2], 'KSampler.cfg', 4));
});

test('wfShortLabel: last segment unless it collides', () => {
    const keys = wf.wfParamKeys(items);
    assert.strictEqual(wf.wfShortLabel('KSampler.cfg', keys), 'cfg');
    assert.strictEqual(wf.wfShortLabel('A.seed', keys), 'A.seed');
    assert.strictEqual(wf.wfShortLabel('B.seed', ['A.seed', 'B.seed']), 'B.seed');
    assert.strictEqual(wf.wfShortLabel('plain', ['plain']), 'plain');
});

test('wfDisplay: 4 significant digits, strings cut at 24', () => {
    assert.strictEqual(wf.wfDisplay(4), '4');
    assert.strictEqual(wf.wfDisplay(0.123456), '0.1235');
    assert.strictEqual(wf.wfDisplay(1234567), '1235000');
    assert.strictEqual(wf.wfDisplay('euler'), 'euler');
    assert.strictEqual(wf.wfDisplay('x'.repeat(40)).length, 24);
});
