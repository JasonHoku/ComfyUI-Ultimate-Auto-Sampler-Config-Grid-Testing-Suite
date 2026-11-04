/**
 * Drag-and-drop / "Send to Load Image" helpers for the Revise modal image (issue #22).
 * Pure functions only (no DOM): loaded as a plain script inlined into the dashboard HTML and, via the
 * guarded export at the bottom, by node (tests/js/dragdrop.test.js). Top-level names are shared with every
 * other logic_*.js (ONE inlined script), so everything here is prefixed `uscg`/`USCG_`.
 */

// false = drag/send REFERENCE the image in the benchmark folder (`subfolder/file [output]`, no copy in input/).
// true  = additionally attach the image bytes as a File (drag) / upload to input/ (button).
const USCG_DRAG_UPLOAD = false;

const USCG_DRAG_MIME = { webp: 'image/webp', png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg' };

// Parse a manifest item's `file` (a /view?filename=..&type=..&subfolder=.. URL) into what the drop
// handlers need. Returns null for items that are not draggable images (videos, workflow cards, odd URLs).
function uscgImageRef(d, baseHref) {
    if (!d || !d.file || d.media_type === 'video' || d.model_type === 'workflow') return null;
    var u;
    try { u = new URL(d.file, baseHref || (typeof document !== 'undefined' ? document.baseURI : 'http://localhost/')); }
    catch (e) { return null; }
    var filename = u.searchParams.get('filename');
    if (!filename) return null;
    var m = /\.([A-Za-z0-9]+)$/.exec(filename);
    var mime = m ? USCG_DRAG_MIME[m[1].toLowerCase()] : null;
    if (!mime) return null;
    return {
        absUrl: u.href,
        filename: filename,
        subfolder: u.searchParams.get('subfolder') || '',
        type: u.searchParams.get('type') || 'output',
        mime: mime
    };
}

// The JSON ComfyUI's own Assets panel puts under `application/x-comfy-asset-info` (documented fields only).
function uscgImageAssetInfo(ref) {
    return { filename: ref.filename, subfolder: ref.subfolder, type: ref.type, display_name: ref.filename };
}

// Widget value LoadImage accepts for a file that lives in output/ (annotated path, no copy).
function uscgAnnotatedPath(ref) {
    return (ref.subfolder ? ref.subfolder + '/' : '') + ref.filename + ' [' + ref.type + ']';
}

// Fill a DataTransfer during `dragstart`. `file` (a File) is attached only when given.
function uscgFillDragData(dt, ref, file) {
    dt.setData('application/x-comfy-asset-info', JSON.stringify(uscgImageAssetInfo(ref)));
    dt.setData('text/uri-list', ref.absUrl);
    dt.setData('text/plain', ref.absUrl);
    dt.setData('DownloadURL', ref.mime + ':' + ref.filename + ':' + ref.absUrl);
    if (file && dt.items && typeof dt.items.add === 'function') {
        try { dt.items.add(file); } catch (e) { /* protected mode / unsupported: URL carriers still work */ }
    }
    try { dt.effectAllowed = 'copy'; } catch (e) { /* ignore */ }
    return dt;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        USCG_DRAG_UPLOAD: USCG_DRAG_UPLOAD, uscgImageRef: uscgImageRef, uscgImageAssetInfo: uscgImageAssetInfo,
        uscgAnnotatedPath: uscgAnnotatedPath, uscgFillDragData: uscgFillDragData
    };
}
