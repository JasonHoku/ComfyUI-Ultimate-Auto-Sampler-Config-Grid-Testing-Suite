'use strict';
// The Python port (comfy_nodes_workflow.build_nodes_workflow) is pinned to this JS builder through
// tests/fixtures/nodes_workflow_golden.json; this test proves the golden file is what the JS emits.
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const { buildComfyNodesWorkflow } = require('../../resources/logic_comfynodes.js');
const fx = (n) => JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'fixtures', n), 'utf8'));
const items = fx('nodes_workflow_items.json');
const golden = fx('nodes_workflow_golden.json');

test('fixture and golden have the same number of cases', () => {
    assert.strictEqual(items.length, golden.length);
    assert.ok(items.length >= 7);
});

items.forEach((item, i) => {
    test('buildComfyNodesWorkflow matches the golden fixture #' + i, () => {
        const wf = JSON.parse(JSON.stringify(buildComfyNodesWorkflow(item)));
        delete wf.id;
        assert.deepStrictEqual(wf, golden[i]);
    });
});
