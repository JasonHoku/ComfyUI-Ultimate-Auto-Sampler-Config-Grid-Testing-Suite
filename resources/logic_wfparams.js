/**
 * WORKFLOW SWEEP PARAMS - PURE LOGIC (no DOM access)
 * Helpers for item.wf_params ({"KSampler.cfg": 4.0, ...}; values are numbers OR strings), used by the
 * Quick Filter, the Logic Filter ("wf:<key>" fields) and the label overlay.
 * The dashboard inlines every logic_*.js into ONE <script>, so top-level names are prefixed wf*.
 * Node tests load this file through the module.exports guard at the bottom (tests/js/wfparams.test.js).
 */

var _wfKeysCache = { arr: null, len: -1, keys: [] };

function _wfIsNumeric(v) {
    if (typeof v === 'number') return isFinite(v);
    if (typeof v === 'string') return v.trim() !== '' && isFinite(Number(v));
    return false;
}

/** [{key, numeric, distinct}] sorted by key. `numeric` = EVERY present value is a finite number / numeric string. Cached per (array, length). */
function wfParamKeys(items) {
    if (!items) return [];
    if (_wfKeysCache.arr === items && _wfKeysCache.len === items.length) return _wfKeysCache.keys;
    const info = {};
    for (let i = 0; i < items.length; i++) {
        const p = items[i] && items[i].wf_params;
        if (!p || typeof p !== 'object') continue;
        for (const k in p) {
            const v = p[k];
            if (v === null || v === undefined) continue;
            const e = info[k] || (info[k] = { numeric: true, vals: new Set() });
            if (!_wfIsNumeric(v)) e.numeric = false;
            e.vals.add(typeof v + ':' + String(v));
        }
    }
    const keys = Object.keys(info).sort().map(k => ({ key: k, numeric: info[k].numeric, distinct: info[k].vals.size }));
    _wfKeysCache = { arr: items, len: items.length, keys: keys };
    return keys;
}

/** Raw value or null. */
function wfParamValue(item, key) {
    const p = item && item.wf_params;
    if (!p || typeof p !== 'object') return null;
    const v = p[key];
    return (v === undefined || v === null) ? null : v;
}

/** Finite number or null. */
function wfParamNumber(item, key) {
    const v = wfParamValue(item, key);
    if (v === null || !_wfIsNumeric(v)) return null;
    return Number(v);
}

/** Numeric tolerance 1e-6 when both sides are numeric, String equality otherwise. */
function wfParamMatches(item, key, value) {
    const v = wfParamValue(item, key);
    if (v === null) return false;
    if (_wfIsNumeric(v) && _wfIsNumeric(value)) return Math.abs(Number(v) - Number(value)) < 1e-6;
    return String(v) === String(value);
}

/** Text after the last '.', unless that short name collides with another key (then the full key). `keys` = strings or wfParamKeys() rows. */
function wfShortLabel(key, keys) {
    const names = (keys || []).map(k => (k && typeof k === 'object') ? k.key : k);
    const short = String(key).slice(String(key).lastIndexOf('.') + 1) || String(key);
    const clash = names.some(n => n !== key && String(n).slice(String(n).lastIndexOf('.') + 1) === short);
    return clash ? String(key) : short;
}

/** Numbers: 4 significant digits; strings cut to 24 chars. */
function wfDisplay(v) {
    if (typeof v === 'number') return String(isFinite(v) ? parseFloat(v.toPrecision(4)) : v);
    const s = String(v);
    return s.length > 24 ? s.slice(0, 23) + '…' : s;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        wfParamKeys: wfParamKeys, wfParamValue: wfParamValue, wfParamNumber: wfParamNumber,
        wfParamMatches: wfParamMatches, wfShortLabel: wfShortLabel, wfDisplay: wfDisplay
    };
}
