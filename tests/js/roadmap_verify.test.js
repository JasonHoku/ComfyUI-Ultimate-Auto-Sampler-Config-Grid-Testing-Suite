// Pure helpers from the Builder's ES module (which imports ComfyUI's frontend): sliced out of the source
// and evaluated, same trick as workflow-config-to-array.test.js.
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, '../../web/conf_builder/conf-builder-config-management.js'), 'utf8');

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

const { reorderTarget, moveItem, omitWarningVisible } = new Function(
    grab('reorderTarget') + '\n' + grab('moveItem') + '\n' + grab('omitWarningVisible') +
    '\nreturn { reorderTarget, moveItem, omitWarningVisible };')();

const run = (list, from, onto, after) => { const l = list.slice(); moveItem(l, from, onto, after); return l.join(''); };

test('moveItem: drop above / below a card, forward and backward', () => {
    assert.strictEqual(run(['a', 'b', 'c', 'd'], 0, 2, false), 'bacd'); // a above c
    assert.strictEqual(run(['a', 'b', 'c', 'd'], 0, 2, true), 'bcad');  // a below c
    assert.strictEqual(run(['a', 'b', 'c', 'd'], 3, 1, false), 'adbc'); // d above b
    assert.strictEqual(run(['a', 'b', 'c', 'd'], 3, 1, true), 'abdc');  // d below b
    assert.strictEqual(run(['a', 'b', 'c', 'd'], 1, 3, true), 'acdb');  // to the very end
    assert.strictEqual(run(['a', 'b', 'c', 'd'], 2, 0, false), 'cabd'); // to the very start
});

test('moveItem: dropping next to itself or on its neighbour edge is a no-op', () => {
    const l = ['a', 'b', 'c'];
    assert.strictEqual(moveItem(l, 1, 1, false), false);
    assert.strictEqual(moveItem(l, 1, 1, true), false);
    assert.strictEqual(moveItem(l, 1, 2, false), false);   // "above c" == where b already is
    assert.strictEqual(moveItem(l, 1, 0, true), false);    // "below a" == where b already is
    assert.strictEqual(l.join(''), 'abc');
});

test('moveItem: out-of-range / bad indices never corrupt the list', () => {
    const l = ['a', 'b'];
    assert.strictEqual(moveItem(l, 5, 0, false), false);
    assert.strictEqual(moveItem(l, -1, 0, false), false);
    assert.strictEqual(moveItem(l, NaN, 0, false), false);
    assert.strictEqual(l.length, 2);
});

test('moveItem keeps every element exactly once for every from/onto/after combination', () => {
    const base = ['a', 'b', 'c', 'd', 'e'];
    for (let from = 0; from < 5; from++) for (let onto = 0; onto < 5; onto++) for (const after of [false, true]) {
        const l = base.slice();
        moveItem(l, from, onto, after);
        assert.strictEqual(l.slice().sort().join(''), 'abcde');
        if (from !== onto) {
            // the dragged item ends up adjacent to the card it was dropped on, on the requested side
            const di = l.indexOf(base[from]), ti = l.indexOf(base[onto]);
            assert.strictEqual(di - ti, after ? 1 : -1);
        }
    }
});

test('reorderTarget: single-element list has nowhere to go', () => {
    assert.strictEqual(reorderTarget(0, 0, true, 1), -1);
});

test('omitWarningVisible: shown only when omits exist and trigger lookup mode is None', () => {
    const arr = { lora_omit_triggers: ['blue'] };
    assert.strictEqual(omitWarningVisible(arr, 'None'), true);
    assert.strictEqual(omitWarningVisible(arr, undefined), true);          // fresh state defaults to None
    assert.strictEqual(omitWarningVisible(arr, 'Append To End'), false);
    assert.strictEqual(omitWarningVisible(arr, 'Append To Start'), false);
    assert.strictEqual(omitWarningVisible(arr, 'Read From Config'), false); // unspecified LoRAs default to "end"
    assert.strictEqual(omitWarningVisible({ lora_omit_triggers: [] }, 'None'), false);
    assert.strictEqual(omitWarningVisible({}, 'None'), false);
    assert.strictEqual(omitWarningVisible(null, 'None'), false);
});

test('source: the warning listens for Run Settings mode changes and the text-encoder handle is not doubled', () => {
    assert.match(src, /cb-lora-mode-changed/);
    assert.match(src, /!headerBar\.classList\.contains\("cb-drag-handle"\)/);
});
