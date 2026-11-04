// Review-d (C): outputFromKey must survive subgraph node ids ("57:35"); saveState must not persist the lazy shell.
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
const { outputKey, outputFromKey } = new Function(grab(sweep, 'outputKey') + '\n' + grab(sweep, 'outputFromKey') +
    '\nreturn { outputKey, outputFromKey };')();

test('subgraph node id round-trips', () => {
    const img = { node_id: '57:35', output_index: 1 };
    assert.deepStrictEqual(outputFromKey(outputKey(img)), img);
    const vid = { node_id: '57:35', output_index: 0, media: 'video', frames_from: '12:3' };
    assert.deepStrictEqual(outputFromKey(outputKey(vid)), vid);
    assert.deepStrictEqual(outputFromKey(outputKey({ node_id: '8', output_index: 0 })), { node_id: '8', output_index: 0 });
});

const utils = fs.readFileSync(path.join(__dirname, '../../resources/logic_utils.js'), 'utf8');
function runSave(manifest) {
    const calls = [];
    const f = new Function('fullManifest', 'fetch', 'document', 'console',
        grab(utils, 'saveState') + '\nreturn saveState();');
    return f(manifest, async (u, o) => { calls.push(JSON.parse(o.body)); },
        { getElementById: () => ({ value: 's1' }) }, { log() {}, error() {} }).then(() => calls);
}

test('saveState does nothing on the lazy shell', async () => {
    assert.deepStrictEqual(await runSave({ items: [], __uscg_lazy__: true, session_name: 's1' }), []);
});

test('saveState sends a copy without the lazy flag', async () => {
    const m = { items: [{ id: 1 }], meta: {} };
    const calls = await runSave(m);
    assert.strictEqual(calls.length, 1);
    assert.strictEqual(calls[0].manifest.items.length, 1);
    assert.ok(!('__uscg_lazy__' in calls[0].manifest));
});
