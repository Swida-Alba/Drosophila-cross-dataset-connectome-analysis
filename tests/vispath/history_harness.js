// History (undo/redo) logic harness: extracts the history-related functions
// from a generated vispath network HTML and runs them against headless
// Cytoscape with minimal DOM stubs, so the undo/redo semantics can be
// regression-tested without a browser.
// Usage: node history_test.js <node-modules-dir> <path-to-network.html>
const cytoscape = require(process.argv[2] + '/node_modules/cytoscape');
const fs = require('fs');

const htmlPath = process.argv[3] || '/tmp/vispath-test/network_test.html';
const html = fs.readFileSync(htmlPath, 'utf8');

// Extract a top-level function declaration with balanced braces.
function extractFunction(name, source) {
    const marker = 'function ' + name + '(';
    const start = source.indexOf(marker);
    if (start === -1) throw new Error('function not found: ' + name);
    const open = source.indexOf('{', start);
    let depth = 0;
    for (let i = open; i < source.length; i++) {
        if (source[i] === '{') depth++;
        else if (source[i] === '}') {
            depth--;
            if (depth === 0) return source.slice(start, i + 1);
        }
    }
    throw new Error('unbalanced braces: ' + name);
}

const FUNCTIONS = [
    'updateHoverInfo',
    'captureStyleBypass',
    'captureState', 'restoreState', 'syncToggleButtons',
    'pushStateHistory', 'pushHistory', 'registerDragHistory',
    'undo', 'redo',
    'updateUndoRedoButtons', 'updateHistoryList', 'jumpToHistory',
    'restoreGlobalStyles', 'updateNodeSize', 'updateNodeShape', 'updateEdgeWidth',
    'updateFontSize', 'updateArrowSize', 'updateEdgeLabelFontSize', 'updateMetric', 'updateEdgeWidths',
    'syncSelectedGeometryInputs', 'updateAlignButtons', 'alignSelectedNodes',
    'syncTransformInputs', 'syncGapDisplays', 'measureAxisGap', 'isVisibleElement', 'metricEdgeValue',
    'visibleNodeCentroid', 'gapAxesSwapped',
    'updateEdgeMetricLabels',
    'updateSelectionChip', 'historyIcon',
    'updateIgnoredEdgesPlaceholder',
    'parseEdgeFilterInput', 'updateIgnoredEdges',
    'parseEdgeFilterExpressions', 'parseEdgeSingleExpression',
    'evaluateEdgeCondition', 'shouldIgnoreEdge', 'applyEdgeFilter',
    'isEdgeInCurrentGraph', 'isDeadEndNodeIn', 'recomputeDeadEnds',
    'reapplyDeadEndHiding', 'reapplyOrphanHiding',
];

// NOTE: sources come from the project's own generated HTML (trusted,
// locally-produced artifact); new Function only executes that code against
// the headless core + stubs.
function buildScope(cy) {
    const fnSources = FUNCTIONS.map(f => extractFunction(f, html)).join('\n');

    const prelude = `
        let undoStack = [];
        let redoStack = [];
        const HISTORY_LIMIT = 50;
        let lastFilterHistoryValue = '';
        let pendingDragState = null;
        let selectedElement = null;
        let labelPosition = 'center';
        let hemisphereMirrorEnabled = false;
        let selfLoopsHidden = false;
        let orphansHidden = false;
        let deadEndsHidden = false;
        // Global style controls mirrored by captureState/restoreState (the
        // real page declares these at script top level, next to the DOM).
        let currentMetric = 'weight';
        let globalNodeSize = 40;
        let globalEdgeWidth = 3;
        let globalFontSize = 12;
        let globalEdgeLabelFontSize = 9;
        let globalArrowSize = 9;
        let globalEdgeWidthScale = 'log_e';
        let globalNodeShape = 'circle';
        let pendingNudge = null;
        function flushPendingNudge() {}
        let reciprocalOffset = 5;
        let restoringHistoryState = false;
        // Layout transform trackers mirrored by captureState's globalStyles
        // (absolute inter-node gaps in px + rotation base).
        let lastGapX = null;
        let lastGapY = null;
        let baselineGapX = null;
        let baselineGapY = null;
        let lastRotationDeg = 0;
        // Operation feedback is a toast now; updateHoverInfo delegates to it.
        function showToast(message, type, action) {}
        let ignoredEdges = new Set();
        let ignoredEdgeExpressions = [];
        let edgeFilterGroups = [];
        function refreshEdgeStyles() {}
        const els = {};
        function makeEl(id) {
            const el = {
                id: id, value: '', textContent: '', style: {},
                options: [{ textContent: '' }], selectedIndex: 0,
                appendChild: function (o) { el.options.push(o); }
            };
            Object.defineProperty(el, 'innerHTML', {
                set: function (v) { el.options = []; },
                get: function () { return ''; }
            });
            els[id] = el;
            return el;
        }
        const document = {
            getElementById: function (id) { return els[id] || makeEl(id); },
            createElement: function (tag) {
                return { tag: tag, textContent: '', disabled: false, value: '', options: [] };
            }
        };
    `;

    const src = prelude + fnSources + `
        return {
            undo, redo, pushHistory, pushStateHistory, captureState, restoreState,
            registerDragHistory, alignSelectedNodes, captureStyleBypass,
            updateAlignButtons,
            getEl: (id) => els[id] || makeEl(id),
            jumpToHistory, updateIgnoredEdges, syncToggleButtons,
            getUndoStack: () => undoStack, getRedoStack: () => redoStack,
            getFilterInput: () => els['ignoreEdgesInput'] || makeEl('ignoreEdgesInput'),
            getHistorySelect: () => els['historyList'] || makeEl('historyList'),
            getDeadEndBtn: () => els['hideDeadEndsBtn'] || makeEl('hideDeadEndsBtn'),
            setDeadEndsHidden: (v) => { deadEndsHidden = v; },
            setOrphansHidden: (v) => { orphansHidden = v; },
            setSelfLoopsHidden: (v) => { selfLoopsHidden = v; },
            getDeadEndsHidden: () => deadEndsHidden,
            getOrphansHidden: () => orphansHidden,
            getSelfLoopsHidden: () => selfLoopsHidden,
        };
    `;

    const api = new Function('cy', src)(cy);
    return api;
}

function buildGraph(nodes, edges) {
    const elements = [];
    for (const [id, ntype] of Object.entries(nodes)) {
        elements.push({ data: { id: id, node_type: ntype, label: id } });
    }
    let i = 0;
    for (const [s, t, w] of edges) {
        elements.push({ group: 'edges', data: { id: 'e' + (i++), source: s, target: t, weight: w || 1, original_weight: w || 1, label: s + '>' + t } });
    }
    // styleEnabled: true so per-element style bypasses (color/size
    // overrides) are actually applied — the default headless core ignores
    // them, which would make the style-bypass snapshot tests meaningless.
    return cytoscape({ headless: true, styleEnabled: true, elements: elements });
}

let failures = 0;
function check(name, got, expected) {
    const g = JSON.stringify(got);
    const e = JSON.stringify(expected);
    const pass = g === e;
    console.log((pass ? 'PASS' : 'FAIL') + ' | ' + name + ' | got=' + g + (pass ? '' : ' expected=' + e));
    if (!pass) failures++;
}

// ===== Test A: snapshot deep-copies data()/position() =====
{
    const cy = buildGraph({ A: 'intermediate', B: 'intermediate' }, [['A', 'B', 5]]);
    const api = buildScope(cy);
    const A = cy.getElementById('A');
    A.position({ x: 10, y: 20 });
    const snap = api.captureState();
    // mutate everything after capture
    A.position({ x: 999, y: 888 });
    A.data('label', 'MUTATED');
    A.addClass('hidden');
    cy.getElementById('B').data('node_type', 'target');
    const snapA = snap.nodes.find(n => n.data.id === 'A');
    const snapB = snap.nodes.find(n => n.data.id === 'B');
    check('snapshot position is detached copy', [snapA.position.x, snapA.position.y], [10, 20]);
    check('snapshot data is detached copy', snapA.data.label, 'A');
    check('snapshot other node data detached', snapB.data.node_type, 'intermediate');
    // classes() may be a string (browser) or array (headless) — both are
    // accepted by cy.add; the important part is that post-capture class
    // mutations do not leak into the snapshot.
    check('snapshot classes not leaked', snapA.classes.includes('hidden'), false);
}

// ===== Test B: undo/redo restores positions and classes =====
// NOTE: restoreState removes and re-adds elements, so element references
// must be re-fetched (cy.getElementById) after undo/redo.
{
    const cy = buildGraph({ A: 'intermediate', B: 'intermediate' }, [['A', 'B', 5]]);
    const api = buildScope(cy);
    cy.getElementById('A').position({ x: 10, y: 20 });
    api.pushHistory('Move A');
    cy.getElementById('A').position({ x: 300, y: 400 });
    check('moved', (() => { const p = cy.getElementById('A').position(); return [p.x, p.y]; })(), [300, 400]);
    api.undo();
    check('undo restores position', (() => { const p = cy.getElementById('A').position(); return [p.x, p.y]; })(), [10, 20]);
    api.redo();
    check('redo reapplies position', (() => { const p = cy.getElementById('A').position(); return [p.x, p.y]; })(), [300, 400]);
}

// ===== Test C: undo restores DATA mutations (deep-copy fix) =====
{
    const cy = buildGraph({ A: 'intermediate' }, []);
    const api = buildScope(cy);
    api.pushHistory('Edit node');
    cy.getElementById('A').data('label', 'NEW LABEL');
    cy.getElementById('A').data('node_type', 'target');
    api.undo();
    check('undo restores label', cy.getElementById('A').data('label'), 'A');
    check('undo restores node_type', cy.getElementById('A').data('node_type'), 'intermediate');
    api.redo();
    check('redo reapplies label', cy.getElementById('A').data('label'), 'NEW LABEL');
}

// ===== Test D: filter ops recorded and undone =====
{
    const cy = buildGraph({ A: 'intermediate', B: 'intermediate' }, [['A', 'B', 5], ['B', 'A', 9]]);
    const api = buildScope(cy);
    const input = api.getFilterInput();
    input.value = '<8';
    api.updateIgnoredEdges();
    check('filter applied', cy.edges('.filtered').length, 1);
    check('filter recorded', api.getUndoStack().length, 1);
    check('filter label', api.getUndoStack()[0].label, 'Edge filter');
    api.undo();
    check('undo filter: input cleared', input.value, '');
    check('undo filter: classes cleared', cy.edges('.filtered').length, 0);
    api.redo();
    check('redo filter: input back', input.value, '<8');
    check('redo filter: classes back', cy.edges('.filtered').length, 1);
}

// ===== Test E: visibility flags + button labels restored =====
{
    const cy = buildGraph({ A: 'intermediate', B: 'intermediate' }, [['A', 'B', 5]]);
    const api = buildScope(cy);
    // simulate dead ends ON state
    cy.getElementById('A').addClass('deadend-hidden');
    api.setDeadEndsHidden(true);
    api.syncToggleButtons();
    const btn = api.getDeadEndBtn();
    check('button shows ON state', btn.textContent, '👁️ Show Dead Ends');
    api.pushHistory('Toggle dead-ends');
    // user toggles OFF
    cy.elements().removeClass('deadend-hidden');
    api.setDeadEndsHidden(false);
    api.syncToggleButtons();
    check('button shows OFF state', btn.textContent, '💀 Hide Dead Ends');
    api.undo();
    check('undo restores flag', api.getDeadEndsHidden(), true);
    // recompute (via restoreState->applyEdgeFilter) hides BOTH A and B here:
    // A is out-only non-source and B is in-only non-target in the A->B graph
    check('undo restores classes', cy.nodes('.deadend-hidden').length, 2);
    check('undo restores button', btn.textContent, '👁️ Show Dead Ends');
    api.redo();
    check('redo flag off', api.getDeadEndsHidden(), false);
    check('redo button', btn.textContent, '💀 Hide Dead Ends');
}

// ===== Test F: jumpToHistory backward and forward (fixed off-by-one) =====
{
    const cy = buildGraph({ A: 'intermediate' }, []);
    const api = buildScope(cy);
    const posOf = () => { const p = cy.getElementById('A').position(); return [p.x, p.y]; };
    cy.getElementById('A').position({ x: 0, y: 0 });
    api.pushHistory('op1');
    cy.getElementById('A').position({ x: 1, y: 0 });
    api.pushHistory('op2');
    cy.getElementById('A').position({ x: 2, y: 0 });
    api.pushHistory('op3');
    cy.getElementById('A').position({ x: 3, y: 0 });
    check('three ops recorded', api.getUndoStack().length, 3);
    // dropdown combined list: [S0 (op1), S1 (op2), S2 (op3), ▶current=S3];
    // jumping to index i lands on the state AFTER op i (index 0 = initial)
    api.jumpToHistory(0);
    check('jump back: all undone', api.getUndoStack().length, 0);
    check('jump back: position', posOf(), [0, 0]);
    api.jumpToHistory(2);
    check('jump forward: state after op3', api.getUndoStack().length, 2);
    check('jump forward: position', posOf(), [2, 0]);
    check('jump forward: labels', api.getUndoStack().map(e => e.label), ['op1', 'op2']);
    check('jump forward: op3 redoable', api.getRedoStack().length, 1);
    api.jumpToHistory(1);
    check('jump middle: position', posOf(), [1, 0]);
    check('jump middle: redo has op3', api.getRedoStack().length, 2);
    api.jumpToHistory(3);
    check('jump to current marker: no-op', posOf(), [3, 0]);
}

// ===== Test G: new operation clears redo =====
{
    const cy = buildGraph({ A: 'intermediate' }, []);
    const api = buildScope(cy);
    api.pushHistory('x');
    api.pushHistory('y');
    api.undo();
    check('redo populated', api.getRedoStack().length, 1);
    api.pushHistory('z');
    check('new op clears redo', api.getRedoStack().length, 0);
    check('labels', api.getUndoStack().map(e => e.label), ['x', 'z']);
}

// ===== Test H: history limit bound =====
{
    const cy = buildGraph({ A: 'intermediate' }, []);
    const api = buildScope(cy);
    for (let i = 0; i < 60; i++) api.pushHistory('op' + i);
    check('history bounded at 50', api.getUndoStack().length, 50);
    check('oldest dropped', api.getUndoStack()[0].label, 'op10');
}

// ===== Test I: node drag (grab/dragfree) is committed to history =====
// Regression: the stash was wired to a node-level 'dragstart' event that
// Cytoscape.js never emits (dragstart exists only for core pan gestures),
// so node relocations were never recorded and undo could not restore them.
// The stash now happens on 'grab'; emitting the real event sequence
// grab -> position change -> dragfree must commit a 'Move nodes' entry,
// while grab + release without moving must NOT.
{
    const cy = buildGraph({ A: 'intermediate', B: 'intermediate' }, [['A', 'B', 5]]);
    const api = buildScope(cy);
    api.registerDragHistory();
    const A = cy.getElementById('A');
    A.position({ x: 10, y: 20 });
    // grab -> move -> dragfree: recorded
    A.emit('grab');
    A.position({ x: 300, y: 400 });
    A.emit('dragfree');
    check('drag recorded', api.getUndoStack().length, 1);
    check('drag label', api.getUndoStack()[0].label, 'Move nodes');
    api.undo();
    check('undo restores pre-drag position', (() => { const p = cy.getElementById('A').position(); return [p.x, p.y]; })(), [10, 20]);
    api.redo();
    check('redo reapplies dragged position', (() => { const p = cy.getElementById('A').position(); return [p.x, p.y]; })(), [300, 400]);
    // grab -> release without moving: NOT recorded
    cy.getElementById('B').emit('grab');
    cy.getElementById('B').emit('dragfree');
    check('click-without-drag not recorded', api.getUndoStack().length, 1);
}

// ===== Test J: per-element style overrides round-trip through undo/redo =====
// Snapshots capture json().style (explicit bypasses only) and restoreState
// re-applies them, so individual size/color edits survive undo/redo.
{
    const cy = buildGraph({ A: 'intermediate' }, []);
    const api = buildScope(cy);
    cy.getElementById('A').style({ 'width': '60px', 'height': '60px' });
    api.pushHistory('Resize element');
    cy.getElementById('A').style({ 'width': '20px', 'height': '20px' });
    check('resized', cy.getElementById('A').numericStyle('width'), 20);
    api.undo();
    check('undo restores size bypass', cy.getElementById('A').numericStyle('width'), 60);
    api.redo();
    check('redo reapplies size bypass', cy.getElementById('A').numericStyle('width'), 20);
}

// ===== Test K: alignment of selected nodes =====
// alignSelectedNodes('h') sets every selected node to the mean Y (X
// untouched), records one history entry, and undo restores the spread.
{
    const cy = buildGraph({ A: 'intermediate', B: 'intermediate', C: 'intermediate' }, []);
    const api = buildScope(cy);
    cy.getElementById('A').position({ x: 0, y: 10 });
    cy.getElementById('B').position({ x: 50, y: 30 });
    cy.getElementById('C').position({ x: 100, y: 50 });
    cy.getElementById('A').select();
    cy.getElementById('B').select();
    cy.getElementById('C').select();
    api.alignSelectedNodes('h');
    check('aligned to mean Y', cy.nodes().map(n => n.position().y), [30, 30, 30]);
    check('X untouched', cy.nodes().map(n => n.position().x), [0, 50, 100]);
    check('align recorded', api.getUndoStack()[0].label, 'Align nodes');
    api.undo();
    check('undo restores Y spread', cy.nodes().map(n => n.position().y), [10, 30, 50]);
    // vertical alignment on the mean X
    api.alignSelectedNodes('v');
    check('aligned to mean X', cy.nodes().map(n => n.position().x), [50, 50, 50]);
    check('Y untouched', cy.nodes().map(n => n.position().y), [10, 30, 50]);
    // geometry modifiers + confirm button hide when nothing is selected
    cy.$(':selected').unselect();
    api.updateAlignButtons();
    check('geometry apply hidden with no selection', api.getEl('applyGeometryBtn').style.display, 'none');
    check('node geometry group hidden', api.getEl('geomNodeGroup').style.display, 'none');
    check('edge geometry group hidden', api.getEl('geomEdgeGroup').style.display, 'none');
    cy.getElementById('A').select();
    api.updateAlignButtons();
    check('geometry apply shown with selection', api.getEl('applyGeometryBtn').style.display, 'block');
}

// ===== Test L: computed (non-bypass) style entries are NOT snapshotted =====
// Regression: Cytoscape's default :active rule writes overlay-color /
// overlay-opacity into _private.style while a node is grabbed, and those
// entries linger after the drag; capturing them turned the transient drag
// shading into a permanent bypass that survived undo AND deselection.
{
    const cy = buildGraph({ A: 'intermediate' }, []);
    const api = buildScope(cy);
    const A = cy.getElementById('A');
    A.style({ 'width': '60px' });  // real bypass
    // simulate the lingering computed :active entries (no bypass flag);
    // inspect captureStyleBypass directly — feeding such entries back into
    // the style engine would crash it, so no engine call after poisoning
    A._private.style['overlay-opacity'] = { name: 'overlay-opacity', value: 0.25 };
    A._private.style['overlay-color'] = { name: 'overlay-color', value: [0, 0, 0] };
    const snap = api.captureStyleBypass(A);
    check('real bypass captured', snap.width, 60);
    check('computed overlay-opacity skipped', 'overlay-opacity' in snap, false);
    check('computed overlay-color skipped', 'overlay-color' in snap, false);
}

console.log(failures === 0 ? 'ALL HISTORY TESTS PASSED' : failures + ' HISTORY TEST(S) FAILED');
// process.exitCode alone would hang: styleEnabled cytoscape cores keep
// background timers alive, so exit explicitly with the proper code.
process.exit(failures === 0 ? 0 : 1);
