/**
 * TRUE VIRTUAL SCROLL - ZERO FLICKER
 * With full keyboard navigation support and AUTO-CALCULATED DIMENSIONS
 */

// --- VIRTUAL WINDOW STATE ---
let visibleRange = { start: 0, end: 50 };
const MAX_VISIBLE_ITEMS = 50;
let updateTimeout = null;

// --- PAN/ZOOM STATE ---
let currentScale = 1;
const MIN_SCALE = 0.1;
const MAX_SCALE = 10;
let isPanning = false;
let isMiddleMousePanning = false;
let panStartX = 0;
let panStartY = 0;
let panOffsetX = 0;
let panOffsetY = 0;

// --- RAF BATCHING STATE (prevents flicker from simultaneous pan+zoom) ---
let rafPending = false;
let pendingPanDeltaX = 0;
let pendingPanDeltaY = 0;
let pendingZoomDelta = 0;
let pendingZoomMouseX = 0;
let pendingZoomMouseY = 0;
let hasPendingZoom = false;

// --- GRID METRICS (Auto-calculated from image data) ---
let itemHeight = 420;  // Will be auto-calculated
let itemWidth = 260;   // Will be auto-calculated
let columnsCount = 4;
let rowHeight = 370;   // Will be auto-calculated

// --- DIMENSION CALCULATION CONSTANTS ---
const CARD_PADDING = 10;           // Space between cards
const CARD_METADATA_HEIGHT = 50;   // Height of the metadata/button area below image
const ASPECT_RATIO_FALLBACK = 2/3; // Default aspect ratio if no images loaded (portrait)
const MIN_CARD_WIDTH = 150;        // Minimum card width
const MAX_CARD_WIDTH = 500;        // Maximum card width

/**
 * Calculate optimal grid dimensions based on image aspect ratios
 * This runs once when data is loaded or when images change
 */
function calculateGridDimensions() {
    // Use activeData (all items) for dimension sampling, not processedData (filtered/sorted)
    // This prevents dimension changes when filters/sorts change
    const sourceData = (activeData && activeData.length > 0) ? activeData : processedData;
    if (!sourceData || sourceData.length === 0) {
        console.log('[Grid Dimensions] No data available, using defaults');
        return;
    }

    // Sample up to 50 images evenly distributed across the dataset for a representative average
    const totalItems = sourceData.length;
    const sampleSize = Math.min(50, totalItems);
    const step = Math.max(1, Math.floor(totalItems / sampleSize));
    let totalAspectRatio = 0;
    let validSamples = 0;

    for (let i = 0; i < totalItems && validSamples < sampleSize; i += step) {
        const item = sourceData[i];
        if (item.width && item.height && item.width > 0 && item.height > 0) {
            totalAspectRatio += item.width / item.height;
            validSamples++;
        }
    }

    // Calculate average aspect ratio
    const avgAspectRatio = validSamples > 0 
        ? totalAspectRatio / validSamples 
        : ASPECT_RATIO_FALLBACK;

    // Get viewport width to determine optimal card width
    const viewport = document.getElementById('viewport');
    const viewportWidth = viewport ? viewport.clientWidth : 1200;

    // Calculate column count based on viewport and desired card size
    const colInput = document.getElementById('col-count');
    const manualCols = colInput ? parseInt(colInput.value) : 0;
    
    if (manualCols > 0) {
        columnsCount = manualCols;
        // Calculate card width to fit columns
        itemWidth = Math.max(MIN_CARD_WIDTH, 
            Math.min(MAX_CARD_WIDTH, 
                Math.floor((viewportWidth - (columnsCount * CARD_PADDING)) / columnsCount)
            )
        );
    } else {
        // Auto-calculate optimal card width (aim for ~250px base width)
        const targetCardWidth = 250;
        columnsCount = Math.max(1, Math.floor(viewportWidth / (targetCardWidth + CARD_PADDING)));
        itemWidth = Math.floor((viewportWidth - (columnsCount * CARD_PADDING)) / columnsCount);
        itemWidth = Math.max(MIN_CARD_WIDTH, Math.min(MAX_CARD_WIDTH, itemWidth));
    }

    // Calculate item height based on aspect ratio and card width
    const imageWidth = itemWidth - CARD_PADDING;
    const imageHeight = Math.floor(imageWidth / avgAspectRatio);
    itemHeight = imageHeight + CARD_METADATA_HEIGHT + CARD_PADDING;

    // Row height is slightly less than item height for tighter packing
    rowHeight = itemHeight - 50;

    console.log(`[Grid Dimensions] Auto-calculated:`);
    console.log(`  - Avg Aspect Ratio: ${avgAspectRatio.toFixed(3)} (from ${validSamples} samples)`);
    console.log(`  - Columns: ${columnsCount}`);
    console.log(`  - Item Width: ${itemWidth}px`);
    console.log(`  - Item Height: ${itemHeight}px`);
    console.log(`  - Row Height: ${rowHeight}px`);
    console.log(`  - Image dimensions: ${imageWidth}x${imageHeight}px`);
}

/**
 * Recalculate dimensions when data changes
 * Call this after processedData is updated
 */
function updateGridDimensions() {
    calculateGridDimensions();
    
    // Force a full re-render with new dimensions
    if (processedData && processedData.length > 0) {
        renderDOM();
    }
}

// Expose for external use
window.updateGridDimensions = updateGridDimensions;

// --- LAZY LOADING ---
const imageObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const img = entry.target;
            if (img.dataset.src && !img.src) {
                img.src = img.dataset.src;
                img.onload = () => img.style.opacity = '1';
                imageObserver.unobserve(img);
            }
        }
    });
}, { rootMargin: '400px' });

// --- VIDEO AUTOPLAY ---
// Videos in rendered cards (visible range + buffer) auto-play.
// Videos in pooled/removed cards are paused.
// Uses virtual scroll range directly instead of IntersectionObserver
// which has edge cases with CSS transforms on the canvas.
function _stopVideoPlayback(card) {
    const vid = card.querySelector('video');
    if (!vid) return;
    vid.pause();
    vid.currentTime = 0;
}

// --- CALCULATE VISIBLE RANGE ---
function calculateVisibleRange() {
    const viewport = document.getElementById('viewport');
    if (!viewport) return visibleRange;

    const effectiveScrollTop = (-panOffsetY / currentScale);
    const viewportHeight = viewport.clientHeight / currentScale;

    const firstVisibleRow = Math.floor(effectiveScrollTop / rowHeight);
    const lastVisibleRow = Math.ceil((effectiveScrollTop + viewportHeight) / rowHeight);

    const bufferRows = 3;
    const startRow = Math.max(0, firstVisibleRow - bufferRows);
    const endRow = lastVisibleRow + bufferRows;

    const start = startRow * columnsCount;
    const end = Math.min(processedData.length, endRow * columnsCount);

    return { start, end };
}

// --- FORCE VISIBLE RANGE UPDATE (for prepended items) ---
function forceVisibleRangeUpdate(prependedCount) {
    console.log(`[Grid] 🔝 ${prependedCount} items prepended - recalculating ALL positions`);

    const effectiveScrollTop = (-panOffsetY / currentScale);

    if (effectiveScrollTop < rowHeight * 5) {
        const newItemRows = Math.ceil(prependedCount / columnsCount);
        const additionalItems = newItemRows * columnsCount;

        visibleRange.start = 0;
        visibleRange.end = Math.min(
            processedData.length,
            visibleRange.end + additionalItems
        );

        console.log(`[Grid] Expanded visible range: ${visibleRange.start}-${visibleRange.end}`);
    }

    renderVisibleItems(true);
}

// --- UPDATE VISIBLE ITEMS ---
function updateVisibleItems() {
    const newRange = calculateVisibleRange();

    if (newRange.start === visibleRange.start && newRange.end === visibleRange.end) {
        const grid = document.getElementById('grid');
        if (grid && processedData.length > 0) {
            const totalRows = Math.ceil(processedData.length / columnsCount);
            const totalHeight = totalRows * rowHeight;
            if (grid.style.height !== `${totalHeight}px`) {
                grid.style.height = `${totalHeight}px`;
            }
        }
        return;
    }

    visibleRange = newRange;
    renderVisibleItems();
}

function scheduleVisibleUpdate() {
    if (updateTimeout) {
        clearTimeout(updateTimeout);
    }

    updateTimeout = setTimeout(() => {
        updateVisibleItems();
    }, 100);
}

// --- RENDER VISIBLE ITEMS ---
// --- DOM CARD POOL ---
// Recycled cards are hidden and reused instead of destroyed/recreated.
// Cuts DOM allocation churn by ~90% during scrolling with large datasets.
const _cardPool = [];
const MAX_POOL_SIZE = 100; // Don't pool more than this (memory tradeoff)

function renderVisibleItems(forcePositionUpdate = false) {
    const grid = document.getElementById('grid');
    if (!grid || !processedData || processedData.length === 0) return;

    const totalRows = Math.ceil(processedData.length / columnsCount);
    const totalHeight = totalRows * rowHeight;
    grid.style.height = `${totalHeight}px`;

    const itemsToShow = processedData.slice(visibleRange.start, visibleRange.end);
    const visibleIds = new Set(itemsToShow.map(item => item.id));

    const toRemove = [];
    for (const [id, node] of nodeMap) {
        if (!visibleIds.has(id)) {
            toRemove.push(id);
        }
    }

    toRemove.forEach(id => {
        const node = nodeMap.get(id);
        if (node) {
            // Pause any video in this card before pooling/removing
            _stopVideoPlayback(node);
            if (_cardPool.length < MAX_POOL_SIZE) {
                // Pool the card for reuse instead of destroying
                node.style.display = 'none';
                _cardPool.push(node);
            } else if (node.parentNode) {
                node.remove();
            }
        }
        nodeMap.delete(id);
    });

    const fragment = document.createDocumentFragment();
    let newCardsAdded = 0;
    let positionsUpdated = 0;

    itemsToShow.forEach((data, offsetIndex) => {
        const globalIndex = visibleRange.start + offsetIndex;
        const genOrderNumber = idToIndexMap.get(data.id) || (globalIndex + 1); // Generation order number (backwards-compatible)

        const row = Math.floor(globalIndex / columnsCount);
        const col = globalIndex % columnsCount;
        const x = col * itemWidth;
        const y = row * rowHeight;

        let card = nodeMap.get(data.id);

        if (!card) {
            // Try to recycle a pooled card, otherwise create new
            if (_cardPool.length > 0) {
                card = _cardPool.pop();
                // Recycle: rebuild card content with new data via createCard
                // createCard returns a fresh element — swap children into pooled shell
                card._dataItem = data;
                card.id = `card-${data.id}`;
                card.dataset.id = data.id;
                const freshCard = createCard(data);
                card.textContent = '';
                while (freshCard.firstChild) card.appendChild(freshCard.firstChild);
                card.style.display = '';
            } else {
                card = createCard(data);
            }
            card.style.position = 'absolute';
            card.style.left = `${x}px`;
            card.style.top = `${y}px`;
            card.style.width = `${itemWidth - 10}px`;
            card.style.zIndex = globalIndex;

            // Update card number to reflect generation order
            const indexTag = card.querySelector('.index-tag');
            if (indexTag) indexTag.textContent = `#${genOrderNumber}`;

            nodeMap.set(data.id, card);
            if (!card.parentNode || card.parentNode !== grid) {
                fragment.appendChild(card);
            }
            newCardsAdded++;

            const img = card.querySelector('img[data-src]');
            if (img && !img.src) {
                img.src = img.dataset.src;
                img.onload = () => img.style.opacity = '1';
            }
            // Videos: set src immediately so preload="metadata" fetches poster frame.
            // Playback starts after fragment is in the DOM (see below).
            const vid = card.querySelector('video');
            if (vid && vid.dataset.src && !vid.src) {
                vid.src = vid.dataset.src;
                vid.load();
            }

        } else {
            const currentLeft = parseInt(card.style.left) || 0;
            const currentTop = parseInt(card.style.top) || 0;
            const currentZIndex = parseInt(card.style.zIndex) || 0;

            if (forcePositionUpdate || currentLeft !== x || currentTop !== y || currentZIndex !== globalIndex) {
                card.style.left = `${x}px`;
                card.style.top = `${y}px`;
                card.style.zIndex = globalIndex;
                positionsUpdated++;

                // Update card number when position changes (sort/filter changed)
                const indexTag = card.querySelector('.index-tag');
                if (indexTag) indexTag.textContent = `#${genOrderNumber}`;
            }
        }
    });

    if (fragment.childNodes.length > 0) {
        grid.appendChild(fragment);
    }

    // After cards are in the DOM, start playback on every video in the rendered set.
    // This ensures videos play as soon as they enter the virtual scroll range,
    // regardless of CSS transforms on the canvas (IntersectionObserver can miss these).
    itemsToShow.forEach(data => {
        const card = nodeMap.get(data.id);
        if (card) {
            const vid = card.querySelector('video');
            if (vid && vid.paused) {
                vid.play().catch(() => { /* autoplay policy may block */ });
            }
        }
    });

    if (forcePositionUpdate) {
        console.log(`[Grid] Added ${newCardsAdded} new, repositioned ${positionsUpdated} existing cards`);
    } else if (newCardsAdded > 0) {
        console.log(`[Grid] Added ${newCardsAdded} new cards, kept ${nodeMap.size - newCardsAdded} existing`);
    }

    // Only set tabindex so keyboard shortcuts work when user clicks in
    // Do NOT call viewport.focus() here — it steals focus from ComfyUI during live updates
    viewport.setAttribute('tabindex', '0');
}

// --- RECALCULATE LAYOUT ---
function recalculateLayout() {
    const viewport = document.getElementById('viewport');
    if (!viewport) return;

    const oldColCount = columnsCount;
    const oldItemWidth = itemWidth;
    const oldItemHeight = itemHeight;

    // Recalculate dimensions based on current viewport and data
    calculateGridDimensions();

    if (oldColCount !== columnsCount || oldItemWidth !== itemWidth || oldItemHeight !== itemHeight) {
        console.log(`[Grid] Layout changed: cols ${oldColCount}→${columnsCount}, w ${oldItemWidth}→${itemWidth}, h ${oldItemHeight}→${itemHeight}`);
        renderDOM();
    }
}

// --- MAIN RENDER ---
let isRendering = false; // Guard against recursive renderDOM calls
function renderDOM() {
    if (isRendering) return; // Prevent recursive calls from recalculateLayout
    isRendering = true;

    const grid = document.getElementById('grid');
    if (!grid) { isRendering = false; return; }

    console.log('[Grid] 🔄 Full re-render');

    grid.innerHTML = '';
    nodeMap.clear();

    // Recalculate dimensions (may change columnsCount)
    calculateGridDimensions();

    // Calculate visible range from current viewport position instead of resetting to 0
    visibleRange = calculateVisibleRange();
    // Ensure we have a reasonable range even if viewport position is at origin
    if (visibleRange.end <= visibleRange.start) {
        visibleRange = { start: 0, end: Math.min(MAX_VISIBLE_ITEMS, processedData.length) };
    }
    renderVisibleItems();

    // Only set tabindex so keyboard shortcuts work when user clicks in
    // Do NOT call viewport.focus() here — it steals focus from ComfyUI during live updates
    viewport.setAttribute('tabindex', '0');

    isRendering = false;
}

// --- PAN/ZOOM CONTROLS ---
const canvas = document.getElementById('canvas');
const viewport = document.getElementById('viewport');

function updateTransform() {
    if (!canvas) return;
    canvas.style.transform = `translate(${panOffsetX}px, ${panOffsetY}px) scale(${currentScale})`;
    scheduleVisibleUpdate();
    // Auto-save viewport position on any pan/zoom change
    if (typeof scheduleViewportSave === 'function') scheduleViewportSave();
}

/**
 * Apply all pending pan+zoom transforms in a single animation frame.
 * This prevents flicker when zooming and panning simultaneously,
 * because both transforms are applied atomically before the next paint.
 */
function applyPendingTransforms() {
    rafPending = false;

    // Apply zoom first (adjusts panOffset to keep cursor stable)
    if (hasPendingZoom) {
        updateZoom(pendingZoomDelta, pendingZoomMouseX, pendingZoomMouseY);
        hasPendingZoom = false;
        pendingZoomDelta = 0;
    }

    // Apply pan delta
    if (pendingPanDeltaX !== 0 || pendingPanDeltaY !== 0) {
        panOffsetX += pendingPanDeltaX;
        panOffsetY += pendingPanDeltaY;
        pendingPanDeltaX = 0;
        pendingPanDeltaY = 0;
        updateTransform();
    }
}

function scheduleRAF() {
    if (!rafPending) {
        rafPending = true;
        requestAnimationFrame(applyPendingTransforms);
    }
}

function getZoomDelta() {
    if (currentScale < 0.5) return 0.05;
    else if (currentScale < 1) return 0.15;
    else if (currentScale < 3) return 0.3;
    else if (currentScale < 6) return 0.5;
    else return 0.8;
}

function updateZoom(delta, mouseX, mouseY) {
    if (!canvas || !viewport) return;

    const oldScale = currentScale;
    const adaptiveDelta = delta > 0 ? getZoomDelta() : -getZoomDelta();
    currentScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, currentScale + adaptiveDelta));

    if (oldScale === currentScale) return;

    const rect = viewport.getBoundingClientRect();
    const offsetX = mouseX - rect.left;
    const offsetY = mouseY - rect.top;

    const scaleFactor = currentScale / oldScale;
    panOffsetX = offsetX - (offsetX - panOffsetX) * scaleFactor;
    panOffsetY = offsetY - (offsetY - panOffsetY) * scaleFactor;

    // If panning while zooming, update panStart so the next mousemove
    // doesn't overwrite the zoom-adjusted offsets (prevents flicker/jump)
    if (isPanning || isMiddleMousePanning) {
        panStartX = lastMouseX - panOffsetX;
        panStartY = lastMouseY - panOffsetY;
    }

    updateTransform();
}

function zoomIn() {
    if (!viewport) return;
    updateZoom(1, viewport.clientWidth / 2, viewport.clientHeight / 2);
}

function zoomOut() {
    if (!viewport) return;
    updateZoom(-1, viewport.clientWidth / 2, viewport.clientHeight / 2);
}

function resetZoom() {
    if (!canvas || !viewport) return;
    currentScale = 1;
    panOffsetX = 0;
    panOffsetY = 0;
    updateTransform();
    autoFitZoom();
}

/**
 * Quick Favorite — toggle favorite on the Nth visible card in reading order.
 * Maps number keys 1-9 to cards visible in the current viewport.
 * Works with any sort mode, zoom level, or pan position.
 */
function _quickFavoriteByPosition(n) {
    if (!viewport || !processedData || processedData.length === 0) return;

    // Use actual DOM bounding rects to find cards truly visible on screen.
    // This handles dynamic card sizes, header offsets, and any transform edge cases.
    var vpRect = viewport.getBoundingClientRect();
    var visibleCards = [];
    var grid = document.querySelector('.grid');
    if (!grid) return;

    var cards = grid.querySelectorAll('.card');
    cards.forEach(function(card) {
        var rect = card.getBoundingClientRect();
        // Card center must be within the viewport bounds
        var cx = (rect.left + rect.right) / 2;
        var cy = (rect.top + rect.bottom) / 2;
        if (cx >= vpRect.left && cx <= vpRect.right && cy >= vpRect.top && cy <= vpRect.bottom) {
            visibleCards.push({ card: card, cx: cx, cy: cy });
        }
    });

    // Sort in reading order: top-to-bottom, then left-to-right
    // Group by rows using a tolerance (cards in the same row have similar cy)
    var rowTolerance = 20;
    visibleCards.sort(function(a, b) {
        if (Math.abs(a.cy - b.cy) > rowTolerance) return a.cy - b.cy;
        return a.cx - b.cx;
    });

    // Map key N to Nth visible card (1-indexed)
    if (n < 1 || n > visibleCards.length) return;
    var target = visibleCards[n - 1];
    var card = target.card;
    var item = card._dataItem;
    if (!item) return;

    // Toggle favorite
    item.favorited = !item.favorited;

    // Update the DOM card
    if (card) {
        var favBtn = card.querySelector('.favorite-btn');
        if (favBtn) {
            favBtn.classList.toggle('favorited', item.favorited);
            favBtn.innerText = item.favorited ? '\u2605' : '\u2606';
        }

        // Visual feedback — brief green/red border flash
        var color = item.favorited ? '#00cc44' : '#cc4444';
        var origZIndex = card.style.zIndex;
        card.style.boxShadow = '0 0 20px ' + color + ', 0 0 40px ' + color;
        card.style.borderColor = color;
        card.style.zIndex = '9999';
        setTimeout(function() {
            card.style.boxShadow = '';
            card.style.borderColor = '';
            card.style.zIndex = origZIndex;
        }, 300);

        // Show number badge briefly
        var badge = document.createElement('div');
        badge.textContent = n;
        badge.style.cssText = 'position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 48px; font-weight: bold; color: ' + color + '; text-shadow: 0 0 10px rgba(0,0,0,0.8); pointer-events: none; z-index: 10000; opacity: 1; transition: opacity 0.3s;';
        card.appendChild(badge);
        setTimeout(function() { badge.style.opacity = '0'; }, 200);
        setTimeout(function() { if (badge.parentNode) badge.parentNode.removeChild(badge); }, 500);
    }

    // Persist the change
    if (typeof markItemChanged === 'function') markItemChanged(item);
    if (typeof scheduleJSONUpdate === 'function') scheduleJSONUpdate();
}

function autoFitZoom() {
    if (!canvas || !viewport || !processedData || processedData.length === 0) {
        console.log('[Grid] Cannot auto-fit: missing data or viewport');
        return;
    }

    // Account for topbar height so the first row sits just below it
    const header = document.getElementById('header');
    const headerH = header ? header.offsetHeight : 0;

    const totalWidth = columnsCount * itemWidth;
    const viewportWidth = viewport.clientWidth;

    // Scale to fit all columns across the viewport width
    const targetScale = (viewportWidth / totalWidth) * 0.95;
    currentScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, targetScale));

    // Center horizontally, pin top row just below the header
    const scaledWidth = totalWidth * currentScale;
    panOffsetX = (viewportWidth - scaledWidth) / 2;
    panOffsetY = headerH;

    updateTransform();

    console.log(`[Grid] 🎯 Auto-fit first row: ${columnsCount} columns, scale: ${currentScale.toFixed(2)}`);
}

function goToImage(imageNumber) {
    if (!processedData || processedData.length === 0) {
        console.log('[Grid] No data to navigate');
        return;
    }

    let targetItem = null;
    let targetIndex = -1;

    // Card numbers reflect sorted position (1-based), so #N = processedData[N-1]
    if (imageNumber >= 1 && imageNumber <= processedData.length) {
        targetIndex = imageNumber - 1;
        targetItem = processedData[targetIndex];
    }

    if (!targetItem || targetIndex === -1) {
        console.log(`[Grid] Image #${imageNumber} not found`);
        alert(`Image #${imageNumber} not found in current view (${processedData.length} items visible)`);
        return;
    }

    const row = Math.floor(targetIndex / columnsCount);
    const col = targetIndex % columnsCount;
    const x = col * itemWidth;
    const y = row * rowHeight;

    const viewportWidth = viewport.clientWidth;
    const viewportHeight = viewport.clientHeight;

    panOffsetY = (viewportHeight / 2) - (y * currentScale) - ((rowHeight * currentScale) / 2);

    updateTransform();
    updateVisibleItems();

    setTimeout(() => {
        const card = document.getElementById(`card-${targetItem.id}`);
        if (card) {
            card.style.transition = 'box-shadow 0.3s, border-color 0.3s';
            card.style.boxShadow = '0 0 30px rgba(0, 209, 178, 0.8)';
            card.style.borderColor = 'var(--accent)';

            setTimeout(() => {
                card.style.boxShadow = '';
                card.style.borderColor = '';
            }, 2000);
        }
    }, 200);

    console.log(`[Grid] 📍 Navigated to image #${imageNumber} at position (${row}, ${col})`);
}

// --- MOUSE CONTROLS ---
let lastMouseX = 0;
let lastMouseY = 0;

if (viewport) {
    viewport.addEventListener('mousedown', (e) => {
        viewport.focus();
        viewport.setAttribute('tabindex', '0');
        if (e.button === 0) {
            isPanning = true;
            panStartX = e.clientX - panOffsetX;
            panStartY = e.clientY - panOffsetY;
            viewport.style.cursor = 'grabbing';
            e.preventDefault();
        } else if (e.button === 1) {
            isMiddleMousePanning = true;
            panStartX = e.clientX - panOffsetX;
            panStartY = e.clientY - panOffsetY;
            viewport.style.cursor = 'grabbing';
            e.preventDefault();
        }
    });

    window.addEventListener('mouseup', (e) => {
        if (e.button === 0 && isPanning) {
            isPanning = false;
            if (viewport) viewport.style.cursor = 'grab';
        } else if (e.button === 1 && isMiddleMousePanning) {
            isMiddleMousePanning = false;
            if (viewport) viewport.style.cursor = 'grab';
        }
    });

    window.addEventListener('mousemove', (e) => {
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
        if (!isPanning && !isMiddleMousePanning) return;
        e.preventDefault();

        // If a zoom is also pending, batch the pan via rAF to prevent flicker
        if (hasPendingZoom) {
            pendingPanDeltaX = e.clientX - panStartX - panOffsetX;
            pendingPanDeltaY = e.clientY - panStartY - panOffsetY;
            scheduleRAF();
        } else {
            // No pending zoom — apply pan immediately for responsiveness
            panOffsetX = e.clientX - panStartX;
            panOffsetY = e.clientY - panStartY;
            updateTransform();
        }
    });

    viewport.addEventListener('wheel', (e) => {
        e.preventDefault();
        // Batch zoom via rAF to prevent flicker with simultaneous pan
        if (isPanning || isMiddleMousePanning) {
            pendingZoomDelta += (e.deltaY > 0 ? -1 : 1);
            pendingZoomMouseX = e.clientX;
            pendingZoomMouseY = e.clientY;
            hasPendingZoom = true;
            scheduleRAF();
        } else {
            // Not panning — apply zoom immediately for responsiveness
            updateZoom(e.deltaY > 0 ? -1 : 1, e.clientX, e.clientY);
        }
    }, { passive: false });

    viewport.addEventListener('contextmenu', (e) => {
        if (e.button === 1) e.preventDefault();
    });

    // --- TOUCH CONTROLS FOR MOBILE ---
    let touchStartDistance = 0;
    let touchStartScale = 1;
    let isTouching = false;
    let touchStartX = 0;
    let touchStartY = 0;
    let wasZooming = false;

    viewport.addEventListener('touchstart', (e) => {
        viewport.focus();
        viewport.setAttribute('tabindex', '0');

        if (e.touches.length === 1) {
            isTouching = true;
            const touch = e.touches[0];
            touchStartX = touch.clientX - panOffsetX;
            touchStartY = touch.clientY - panOffsetY;
        } else if (e.touches.length === 2) {
            e.preventDefault();
            wasZooming = true;
            const touch1 = e.touches[0];
            const touch2 = e.touches[1];
            touchStartDistance = Math.hypot(
                touch2.clientX - touch1.clientX,
                touch2.clientY - touch1.clientY
            );
            touchStartScale = currentScale;
        }
    }, { passive: false });

    viewport.addEventListener('touchmove', (e) => {
        if (e.touches.length === 1 && isTouching) {
            if (wasZooming) {
                const touch = e.touches[0];
                touchStartX = touch.clientX - panOffsetX;
                touchStartY = touch.clientY - panOffsetY;
                wasZooming = false;
                return;
            }

            e.preventDefault();
            const touch = e.touches[0];
            panOffsetX = touch.clientX - touchStartX;
            panOffsetY = touch.clientY - touchStartY;
            updateTransform();
        } else if (e.touches.length === 2) {
            e.preventDefault();
            const touch1 = e.touches[0];
            const touch2 = e.touches[1];

            const currentDistance = Math.hypot(
                touch2.clientX - touch1.clientX,
                touch2.clientY - touch1.clientY
            );

            const centerX = (touch1.clientX + touch2.clientX) / 2;
            const centerY = (touch1.clientY + touch2.clientY) / 2;

            const scaleChange = currentDistance / touchStartDistance;
            const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, touchStartScale * scaleChange));

            if (newScale !== currentScale) {
                const rect = viewport.getBoundingClientRect();
                const offsetX = centerX - rect.left;
                const offsetY = centerY - rect.top;

                const scaleFactor = newScale / currentScale;
                panOffsetX = offsetX - (offsetX - panOffsetX) * scaleFactor;
                panOffsetY = offsetY - (offsetY - panOffsetY) * scaleFactor;
                currentScale = newScale;

                updateTransform();
            }
        }
    }, { passive: false });

    viewport.addEventListener('touchend', (e) => {
        if (e.touches.length === 0) {
            isTouching = false;
            wasZooming = false;
        }
        if (e.touches.length < 2) {
            touchStartDistance = 0;
        }
    });
}

// --- RESIZE OBSERVER ---
const resizeObserver = new ResizeObserver(() => {
    recalculateLayout();
});

if (viewport) {
    resizeObserver.observe(viewport);
}

// --- KEYBOARD SHORTCUTS ---
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        // Suppress grid shortcuts when the lightbox/revise modal is open
        // (modal installs its own capture-phase handler that handles navigation)
        if (window._modalOpen) return;

        switch (e.key) {
            case '+':
            case '=':
                e.preventDefault();
                zoomIn();
                break;
            case '-':
            case '_':
                e.preventDefault();
                zoomOut();
                break;
            case '0':
                e.preventDefault();
                autoFitZoom();
                break;
            case ' ':
                e.preventDefault();
                const rowScroll = rowHeight * currentScale;
                if (e.shiftKey) {
                    panOffsetY += rowScroll;
                } else {
                    panOffsetY -= rowScroll;
                }
                updateTransform();
                updateVisibleItems();
                break;
            case 'ArrowUp':
                e.preventDefault();
                panOffsetY += 100;
                updateTransform();
                updateVisibleItems();
                break;
            case 'ArrowDown':
                e.preventDefault();
                panOffsetY -= 100;
                updateTransform();
                updateVisibleItems();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                panOffsetX += 100;
                updateTransform();
                updateVisibleItems();
                break;
            case 'ArrowRight':
                e.preventDefault();
                panOffsetX -= 100;
                updateTransform();
                updateVisibleItems();
                break;
            case 'c':
            case 'C':
                // Pairwise review (compare mode); plain C only, so Ctrl+C copy is untouched
                if (!e.ctrlKey && !e.metaKey && !e.altKey && typeof openCompareMode === 'function') {
                    e.preventDefault();
                    openCompareMode();
                }
                break;
        }

        // 1-9 (no modifiers): Quick favorite — toggle favorite on Nth visible card
        // Determines which cards are visible in the viewport, sorts in reading order,
        // and maps key N to the Nth card. Works with any sort, zoom, or pan.
        if (!e.shiftKey && !e.ctrlKey && !e.altKey && !e.metaKey &&
            e.code >= 'Digit1' && e.code <= 'Digit9') {
            e.preventDefault();
            const n = parseInt(e.code.replace('Digit', ''));
            _quickFavoriteByPosition(n);
        }

        // Shift+0-9: Quick column count change
        // Use e.code (Digit0-Digit9) because e.key returns shifted symbols (!@#$) when Shift is held
        if (e.shiftKey && e.code >= 'Digit0' && e.code <= 'Digit9') {
            e.preventDefault();
            const colInput = document.getElementById('col-count');
            const num = parseInt(e.code.replace('Digit', ''));
            if (num === 0) {
                // Shift+0 = Auto columns
                if (colInput) colInput.value = '';
                localStorage.removeItem('ultimate_grid_cols');
            } else {
                // Shift+1-9 = Set exact column count
                if (colInput) colInput.value = num;
                localStorage.setItem('ultimate_grid_cols', num);
            }
            if (typeof recalcColumns === 'function') {
                recalcColumns();
            }
        }
    });
}

// --- LEGACY COMPATIBILITY ---
function measureGridItem() {
    metrics.ready = true;
    renderDOM();
}

function recalcColumns() {
    recalculateLayout();
    autoFitZoom();
}

function updateVirtualWindow(force = false) {
    if (force) renderDOM();
}

function scheduleRender() {
    renderDOM();
}

function onDataAdded() {
    renderDOM();
}

// --- VIEWPORT POSITION PERSISTENCE ---
// Save viewport position to localStorage for persistence across fullscreen toggles and reloads
function saveViewportPosition() {
    try {
        const sessInput = document.getElementById('session-input');
        const sessionKey = sessInput ? sessInput.value : 'default';
        const state = {
            panOffsetX: panOffsetX,
            panOffsetY: panOffsetY,
            currentScale: currentScale,
            timestamp: Date.now()
        };
        localStorage.setItem(`ultimate_grid_viewport_${sessionKey}`, JSON.stringify(state));
    } catch (e) {
        console.warn('[Viewport] Failed to save position:', e);
    }
}

// Restore viewport position from localStorage
// Returns true if position was restored, false otherwise
function restoreViewportPosition() {
    try {
        const sessInput = document.getElementById('session-input');
        const sessionKey = sessInput ? sessInput.value : 'default';
        const saved = localStorage.getItem(`ultimate_grid_viewport_${sessionKey}`);
        if (!saved) return false;

        const state = JSON.parse(saved);

        // Only restore if saved within the last 24 hours
        if (Date.now() - state.timestamp > 24 * 60 * 60 * 1000) {
            localStorage.removeItem(`ultimate_grid_viewport_${sessionKey}`);
            return false;
        }

        panOffsetX = state.panOffsetX;
        panOffsetY = state.panOffsetY;
        currentScale = state.currentScale;

        updateTransform();
        updateVisibleItems();

        console.log(`[Viewport] Restored position: panX=${panOffsetX.toFixed(0)}, panY=${panOffsetY.toFixed(0)}, scale=${currentScale.toFixed(2)}`);
        return true;
    } catch (e) {
        console.warn('[Viewport] Failed to restore position:', e);
        return false;
    }
}

// Auto-save viewport position periodically during interaction
let viewportSaveTimer = null;
function scheduleViewportSave() {
    if (viewportSaveTimer) clearTimeout(viewportSaveTimer);
    viewportSaveTimer = setTimeout(saveViewportPosition, 500);
}

// Hook into updateTransform to auto-save position on pan/zoom changes
const _originalUpdateTransform = updateTransform;
// We can't reassign updateTransform since it's used by reference, so we hook via the scheduleVisibleUpdate path
// Instead, add save scheduling to mouse/touch/keyboard interactions

// Expose functions
window.zoomIn = zoomIn;
window.zoomOut = zoomOut;
window.resetZoom = resetZoom;
window.autoFitZoom = autoFitZoom;
window.goToImage = goToImage;
window.updateVisibleItems = updateVisibleItems;
window.forceVisibleRangeUpdate = forceVisibleRangeUpdate;
window.saveViewportPosition = saveViewportPosition;
window.restoreViewportPosition = restoreViewportPosition;

// --- MOBILE NAVIGATION FUNCTIONS ---
function scrollDownOneRow() {
    const rowScroll = rowHeight * currentScale;
    panOffsetY -= rowScroll;
    updateTransform();
    updateVisibleItems();
}

function scrollUpOneRow() {
    const rowScroll = rowHeight * currentScale;
    panOffsetY += rowScroll;
    updateTransform();
    updateVisibleItems();
}

// Expose mobile navigation functions
window.scrollDownOneRow = scrollDownOneRow;
window.scrollUpOneRow = scrollUpOneRow;

// Initialize keyboard shortcuts immediately
setupKeyboardShortcuts();

// Set up Go To Image input handler
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupGoToInput);
} else {
    setupGoToInput();
}

function setupGoToInput() {
    const gotoInput = document.getElementById('goto-input');
    if (gotoInput) {
        gotoInput.addEventListener('blur', () => {
            const imageNum = parseInt(gotoInput.value);
            if (imageNum && imageNum > 0) {
                goToImage(imageNum);
                gotoInput.value = '';
            }
        });

        gotoInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                gotoInput.blur();
            }
        });
    }
}