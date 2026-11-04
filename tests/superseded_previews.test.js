// Run: node tests/superseded_previews.test.js  (dashboard superseded-preview logic, no browser needed)
const fs = require('fs');
const src = fs.readFileSync(require('path').join(__dirname, '..', 'resources', 'logic_pipeline.js'), 'utf8');
function grab(name) {
  const i = src.indexOf('function ' + name + '(');
  let depth = 0, j = src.indexOf('{', i);
  for (let k = j; k < src.length; k++) {
    if (src[k] === '{') depth++;
    if (src[k] === '}') { depth--; if (depth === 0) return src.slice(i, k + 1); }
  }
}
global.document = { getElementById: () => null };
let code = 'var supersededFps = new Set(); var activeData = []; var processedData = []; var filters = { passType: new Set(["Preview","Full"]) };\n' +
  grab('isSupersededPreview') + '\n' + grab('refreshSupersededFps') + '\n' + grab('updatePreviewBar') + '\n' +
  'return {isSupersededPreview, refreshSupersededFps, state: () => ({supersededFps, filters, setActive: (a, p) => { activeData = a; processedData = p; }})};';
const api = new Function(code)();
const assert = require('assert');
const prev = { id: 1, preview: true, preview_of: 'fp:a' };
const prev2 = { id: 2, preview: true, preview_of: 'fp:b' };
const full = { id: 3, config_fp: 'fp:a' };
const up = { id: 4, config_fp: 'fp:x', base_fp: 'fp:b' };
const st = api.state();
st.setActive([prev, prev2], [prev, prev2]);
assert.strictEqual(api.refreshSupersededFps(), false);
assert.strictEqual(api.isSupersededPreview(prev), false);
// incremental: full item lands for fp:a while preview is displayed
assert.strictEqual(api.refreshSupersededFps([full]), true);
assert.strictEqual(api.isSupersededPreview(prev), true);
assert.strictEqual(api.isSupersededPreview(prev2), false);
// upscaled item supersedes through base_fp
assert.strictEqual(api.refreshSupersededFps([up]), true);
assert.strictEqual(api.isSupersededPreview(prev2), true);
// no growth -> false
assert.strictEqual(api.refreshSupersededFps([full]), false);
// isolating Preview-only shows superseded previews
st.filters.passType.delete('Full');
assert.strictEqual(api.isSupersededPreview(prev), false);
st.filters.passType.add('Full');
// full rebuild
st.setActive([prev, prev2, full, up], []);
api.refreshSupersededFps();
assert.strictEqual(api.state().supersededFps.size, 3);
console.log('superseded logic ok');
