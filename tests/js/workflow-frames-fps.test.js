// framesFps: the fps shown in the Builder's "Video" option label for VHS and the core frame sinks.
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

function grab(src, name) {
    const m = new RegExp('(?:export )?(?:async )?function ' + name + '[(]').exec(src);
    assert.ok(m, name + ' not found');
    let i = src.indexOf('{', m.index), depth = 0, j = i;
    for (; j < src.length; j++) {
        if (src[j] === '{') depth++;
        else if (src[j] === '}') { depth--; if (depth === 0) break; }
    }
    return src.slice(m.index, j + 1).replace(/^export /, '');
}

const sweep = fs.readFileSync(path.join(__dirname, '../../web/conf_builder/conf-builder-workflow-sweep.js'), 'utf8');
const { framesFps } = new Function(grab(sweep, 'framesFps') + '\nreturn { framesFps };')();

test('VHS frame_rate and core sink fps are read', () => {
    assert.strictEqual(framesFps({ 2: { inputs: { frame_rate: 8 } } }, '2'), 8);
    assert.strictEqual(framesFps({ 2: { inputs: { fps: 12.5 } } }, '2'), 12.5);
    assert.strictEqual(framesFps({ 2: { inputs: { frame_rate: 16, fps: 4 } } }, '2'), 16);
});

test('links, strings and missing nodes give undefined', () => {
    assert.strictEqual(framesFps({ 2: { inputs: { fps: ['3', 0] } } }, '2'), undefined);
    assert.strictEqual(framesFps({ 2: { inputs: { fps: '12' } } }, '2'), undefined);
    assert.strictEqual(framesFps({}, '2'), undefined);
    assert.strictEqual(framesFps(null, '2'), undefined);
});
