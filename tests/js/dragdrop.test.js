'use strict';
const test = require('node:test');
const assert = require('node:assert');
const D = require('../../resources/logic_dragdrop.js');

const B = 'http://127.0.0.1:8188/';
const mk = (file, extra) => Object.assign({ id: 1, file: file }, extra || {});

test('uscgImageRef parses a /view URL (webp, nested subfolder)', () => {
    const r = D.uscgImageRef(mk('/view?filename=img_1.webp&type=output&subfolder=benchmarks/Sess/images'), B);
    assert.strictEqual(r.filename, 'img_1.webp');
    assert.strictEqual(r.subfolder, 'benchmarks/Sess/images');
    assert.strictEqual(r.type, 'output');
    assert.strictEqual(r.mime, 'image/webp');
    assert.strictEqual(r.absUrl, B + 'view?filename=img_1.webp&type=output&subfolder=benchmarks/Sess/images');
    assert.strictEqual(D.uscgAnnotatedPath(r), 'benchmarks/Sess/images/img_1.webp [output]');
});

test('uscgImageRef: png/jpg mimes, url-encoded names, default type, empty subfolder', () => {
    assert.strictEqual(D.uscgImageRef(mk('/view?filename=a.PNG'), B).mime, 'image/png');
    assert.strictEqual(D.uscgImageRef(mk('/view?filename=a.jpeg'), B).mime, 'image/jpeg');
    const r = D.uscgImageRef(mk('/view?filename=my%20pic%20%281%29.jpg&type=temp&subfolder='), B);
    assert.strictEqual(r.filename, 'my pic (1).jpg');
    assert.strictEqual(r.type, 'temp');
    assert.strictEqual(D.uscgAnnotatedPath(r), 'my pic (1).jpg [temp]');
    assert.strictEqual(D.uscgImageRef(mk('/view?filename=a.png'), B).type, 'output');
});

test('uscgImageRef returns null for videos, workflow cards, bad input', () => {
    assert.strictEqual(D.uscgImageRef(mk('/view?filename=a.mp4', { media_type: 'video' }), B), null);
    assert.strictEqual(D.uscgImageRef(mk('/view?filename=a.webp', { model_type: 'workflow' }), B), null);
    assert.strictEqual(D.uscgImageRef(mk('/view?filename=a.mp4'), B), null);
    assert.strictEqual(D.uscgImageRef(mk('/view?type=output'), B), null);
    assert.strictEqual(D.uscgImageRef(null, B), null);
    assert.strictEqual(D.uscgImageRef({}, B), null);
});

function fakeDt() {
    const data = {}, added = [];
    return { data, added, setData: (k, v) => { data[k] = v; }, items: { add: (f) => added.push(f) } };
}

test('uscgFillDragData sets the four carriers and no File by default', () => {
    const ref = D.uscgImageRef(mk('/view?filename=img_1.webp&type=output&subfolder=b/s'), B);
    const dt = fakeDt();
    D.uscgFillDragData(dt, ref, null);
    assert.deepStrictEqual(JSON.parse(dt.data['application/x-comfy-asset-info']),
        { filename: 'img_1.webp', subfolder: 'b/s', type: 'output', display_name: 'img_1.webp' });
    assert.strictEqual(dt.data['text/uri-list'], ref.absUrl);
    assert.strictEqual(dt.data['text/plain'], ref.absUrl);
    assert.strictEqual(dt.data['DownloadURL'], 'image/webp:img_1.webp:' + ref.absUrl);
    assert.strictEqual(dt.added.length, 0);
    assert.strictEqual(dt.effectAllowed, 'copy');
});

test('uscgFillDragData attaches a File only when one is passed; survives items.add throwing', () => {
    const ref = D.uscgImageRef(mk('/view?filename=a.png'), B);
    const dt = fakeDt();
    D.uscgFillDragData(dt, ref, { name: 'a.png' });
    assert.strictEqual(dt.added.length, 1);
    const bad = { setData() {}, items: { add() { throw new Error('protected'); } } };
    assert.doesNotThrow(() => D.uscgFillDragData(bad, ref, {}));
});

test('upload mode is off by default', () => { assert.strictEqual(D.USCG_DRAG_UPLOAD, false); });
