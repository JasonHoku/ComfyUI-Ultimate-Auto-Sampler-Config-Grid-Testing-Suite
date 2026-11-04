// outputKey / outputFromKey are pure helpers inside an ES module that imports ComfyUI's frontend, so this test
// slices them out of the source (same trick as workflow-config-to-array.test.js).
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '../../web/conf_builder/conf-builder-workflow-sweep.js'), 'utf8');

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

const { outputKey, outputFromKey } = new Function(['outputKey', 'outputFromKey'].map(grab).join('\n') +
    '\nreturn { outputKey, outputFromKey };')();

test('image output round-trips and keeps the stored shape (no media keys)', () => {
    const o = { node_id: '8', output_index: 0 };
    assert.strictEqual(outputKey(o), '8:0:image:');
    assert.deepStrictEqual(outputFromKey(outputKey(o)), { node_id: '8', output_index: 0 });
    // a picked_from helper key never leaks into the stored output
    assert.deepStrictEqual(outputFromKey(outputKey({ node_id: '8', output_index: 1, picked_from: '9' })),
        { node_id: '8', output_index: 1 });
});

test('video (VIDEO socket) output round-trips', () => {
    const o = { node_id: '2', output_index: 0, media: 'video' };
    assert.strictEqual(outputKey(o), '2:0:video:');
    assert.deepStrictEqual(outputFromKey(outputKey(o)), o);
});

test('video frames (VHS) output round-trips with frames_from', () => {
    const o = { node_id: '1', output_index: 0, media: 'video', frames_from: '2' };
    assert.strictEqual(outputKey(o), '1:0:video:2');
    assert.deepStrictEqual(outputFromKey(outputKey(o)), o);
});

test('image and video keys of the same socket differ, and bad keys give null', () => {
    assert.notStrictEqual(outputKey({ node_id: '2', output_index: 0 }), outputKey({ node_id: '2', output_index: 0, media: 'video' }));
    assert.strictEqual(outputFromKey(''), null);
    assert.strictEqual(outputFromKey(':0:image:'), null);
});
