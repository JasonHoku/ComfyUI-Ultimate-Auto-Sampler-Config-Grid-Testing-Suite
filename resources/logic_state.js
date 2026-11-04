/**
 * GLOBAL STATE MANAGEMENT
 * Holds the raw data, layout metrics, and viewport coordinates.
 */

// --- DATA CONTAINERS ---



// --- 2. HELPERS (Restored) ---
var currentSort = currentSort || 'oldest';
var currentSecondarySort = currentSecondarySort || 'none';
var processedData = processedData || [];
var activeData = activeData || [];
var searchFilters = searchFilters || [];
var isPipelinePending = false;
var lastFilterKey = null;
var idToIndexMap = new Map();
var meta = meta || {}; 

// Filter State Containers
var filters = filters || {
    model: new Set(),
    sampler: new Set(),
    scheduler: new Set(),
    denoise: new Set(),
    lora: new Set(),
    positive: new Set(),
    negative: new Set(),
    size: new Set(),
    seed: new Set(),
    steps: new Set(),
    cfg: new Set(),
    upscaleMethod: new Set(),
    mediaType: new Set(),
    passType: new Set()   // 'Preview' | 'Full'
};

// config_fp / base_fp of every non-preview item: a preview whose preview_of is in here has been superseded
// by a full render and is hidden by default (see isSupersededPreview in logic_pipeline.js).
var supersededFps = supersededFps || new Set();

// Top-level filter toggles (true = show, false = hide)
var topFilters = topFilters || {
    showFavorites: true,    // Default ON
    showNonFavorited: true, // Default ON
    showRejected: false     // Default OFF
};

const sortOptionsList = [
    { id: 'oldest', label: 'Oldest' },
    { id: 'newest', label: 'Newest' },
    { id: 'fastest', label: 'Fastest' },
    { id: 'favorited', label: 'Favorited' },
    { id: 'model', label: 'Model' },
    { id: 'prompt', label: 'Prompt' },
    { id: 'cfg', label: 'CFG' },
    { id: 'denoise', label: 'Denoise' },
    { id: 'lora', label: 'LoRA' },
    { id: 'sampler', label: 'Sampler' },
    { id: 'seed', label: 'Seed' },
    { id: 'size', label: 'Size' },
    { id: 'rating', label: 'Rating' }
];

// --- SEARCH FILTER STATE ---
// Array of search filter objects: { type: 'model', term: 'mountain' }

// --- LOGIC FILTER STATE ---
// Array of numeric comparison filter objects: { field: 'cfg', op: '>', value: 7 }
// Multiple filters are applied with AND logic. OR is a future extension.
var logicFilters = logicFilters || [];

// --- QUICK FILTER STATE ---
// Array of faceted filter chips: { type: 'lora', value: 'DMD2.safetensors', _expanded: false }
// Multiple chips are ANDed together. LoRA chips match any lora in a stack.
var quickFilters = quickFilters || [];


// --- VIRTUALIZATION & PIPELINE ---
let visibleSlice = { start: -1, end: -1 };

const nodeMap = new Map();
// --- LAYOUT METRICS ---
let metrics = { 
    cardWidth: 240,
    cardHeight: 350,
    colCount: 4,
    totalHeight: 0,
    ready: false 
};

// --- VIEWPORT COORDINATES (Legacy - kept for compatibility) ---
let px = 0;
let py = 0;
let s = 1;

// --- RENDER THROTTLING ---
let isRenderPending = false; 
let lastRenderTime = 0;
const RENDER_THROTTLE_MS = 16;

// --- DEBOUNCE TIMER (Prevents rapid re-renders) ---
let updateDebounceTimer = null;

// --- LABEL MODE STATE ---
// Which fields to show as overlays on cards (persisted to localStorage)
var labelMode = labelMode || {
    enabled: false,
    labelSize: 9,
    fields: {
        model: false,
        lora: false,
        loraUniqueOnly: true,   // When true, skip loras that appear on every card
        prompt: false,
        sampler: false,
        scheduler: false,
        cfg: false,
        steps: false,
        seed: false,
        denoise: false,
        upscale: false,
        upscaleMode: false,
        upscaleModel: false,
        upscaleRatio: false,
        upscaleDenoise: false,
        upscaleResizeMethod: false,
        upscaleHiresSteps: false,
        upscaleTiling: false,
        hiresPromptBehavior: false,
        hiresPromptText: false,
        rating: false           // Pairwise-review rating badge
    }
};

// Cache of values that are the same across ALL items (computed once per dataset)
var labelGlobalValues = labelGlobalValues || null;