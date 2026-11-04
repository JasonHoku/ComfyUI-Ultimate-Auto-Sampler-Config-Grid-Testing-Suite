// workflowConfigToArray lives in an ES module that imports ComfyUI's frontend, so this test slices the
// pure helper functions out of the source and evaluates them (same trick as other tests that avoid the DOM).
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

const code = ['defaultWorkflowState', 'ensureWorkflowState', 'newWorkflowArray', 'workflowConfigToArray'].map(grab).join('\n');
const { workflowConfigToArray } = new Function(code + '\nreturn { workflowConfigToArray };')();

const cfg = {
    model_type: 'workflow', name: 'My sweep', workflow: { '3': { class_type: 'KSampler', inputs: { seed: 1 } } },
    workflow_source: 'wf.json', workflow_output: { node_id: '9', output_index: 0 }, workflow_run_other_outputs: true,
    workflow_bindings: [{ id: 'b1', kind: 'literal', targets: [{ node_id: '3', input: 'seed' }], values: [1, 2] }],
    positive: [['a cat']], negative: 'bad', _prompt_source: 'custom',
};

test('restores workflow, bindings, output and custom prompts', () => {
    const a = workflowConfigToArray(cfg, 0);
    assert.strictEqual(a.kind, 'workflow');
    assert.strictEqual(a.name, 'My sweep');
    assert.deepStrictEqual(a.workflow.api_prompt, cfg.workflow);
    assert.strictEqual(a.workflow.source_name, 'wf.json');
    assert.deepStrictEqual(a.workflow.output, { node_id: '9', output_index: 0 });
    assert.strictEqual(a.workflow.run_other_outputs, true);
    assert.deepStrictEqual(a.workflow.bindings, cfg.workflow_bindings);
    assert.strictEqual(a.use_custom_prompts, true);
    assert.deepStrictEqual(a.positive_prompt_groups, [['a cat']]);
    assert.strictEqual(a.negative_prompt, 'bad');
});

test('global prompts are not copied into the array; odd input never throws', () => {
    const a = workflowConfigToArray({ ...cfg, _prompt_source: 'global' }, 2);
    assert.strictEqual(a.use_custom_prompts, false);
    assert.deepStrictEqual(a.positive_prompt_groups, []);
    const b = workflowConfigToArray(null, 1);
    assert.strictEqual(b.kind, 'workflow');
    assert.strictEqual(b.name, 'Workflow 2');
    assert.strictEqual(b.workflow.api_prompt, null);
    assert.deepStrictEqual(b.workflow.bindings, []);
});
