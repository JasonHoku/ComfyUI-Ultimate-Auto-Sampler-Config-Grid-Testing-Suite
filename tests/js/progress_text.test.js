// Run: node --test tests/js/progress_text.test.js  (ETA bar label for progress events, no browser needed)
const fs = require('fs');
const path = require('path');
const test = require('node:test');
const assert = require('assert');

const src = fs.readFileSync(path.join(__dirname, '..', '..', 'resources', 'logic_events.js'), 'utf8');
function grab(name) {
  const i = src.indexOf('function ' + name + '(');
  assert.ok(i >= 0, name + ' missing');
  let depth = 0, j = src.indexOf('{', i);
  for (let k = j; k < src.length; k++) {
    if (src[k] === '{') depth++;
    if (src[k] === '}') { depth--; if (depth === 0) return src.slice(i, k + 1); }
  }
}
const formatProgressText = new Function(grab('formatProgressText') + '\nreturn formatProgressText;')();

const base = { current_job: 3, total_jobs: 6, progress_pct: 50, eta_str: '2m 5s', finish_time: '12:00:00', avg_duration: 10.5 };

test('distributed label shows the machine count', () => {
  const t = formatProgressText({ ...base, phase: 'distributed', workers: 2 });
  assert.strictEqual(t, 'Job 3/6 (50%) | ETA: 2m 5s | ~12:00:00 | 10.5s/job | 2 machine(s)');
});

test('distributed without workers omits the machine part', () => {
  const t = formatProgressText({ ...base, phase: 'distributed', workers: 0 });
  assert.strictEqual(t, 'Job 3/6 (50%) | ETA: 2m 5s | ~12:00:00 | 10.5s/job');
});

test('upscaling label is unchanged', () => {
  const t = formatProgressText({ ...base, phase: 'upscaling', workers: 3 });
  assert.strictEqual(t, 'Upscaling 3/6 (50%) | ETA: 2m 5s | ~12:00:00 | 10.5s/job');
});

test('sweep label keeps errors and paused state', () => {
  assert.strictEqual(
    formatProgressText({ ...base, phase: 'sweep', errors: 1, state: 'paused', finish_time: '' }),
    'Sweep job 3/6 (50%) | ETA: 2m 5s | 10.5s/job | 1 error | paused');
  assert.strictEqual(
    formatProgressText({ ...base, phase: 'sweep', errors: 2, state: 'cancelled' }),
    'Sweep job 3/6 (50%) | ETA: 2m 5s | ~12:00:00 | 10.5s/job | 2 errors | cancelled');
});

test('plain run (no phase) uses the Job label', () => {
  assert.strictEqual(formatProgressText(base), 'Job 3/6 (50%) | ETA: 2m 5s | ~12:00:00 | 10.5s/job');
});
