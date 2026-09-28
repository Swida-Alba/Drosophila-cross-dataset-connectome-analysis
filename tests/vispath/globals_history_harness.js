// Test harness: global style adjustments (node size, edge width, font,
// arrow, scaling method, metric) must be recorded in the operation history,
// undo/redo must restore them exactly, and self-loop curvature must adapt
// to the actual node size and edge width.
// Usage: node globals_history_harness.js <node-modules-dir> <path-to-network.html>
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
    'pushStateHistory', 'pushHistory', 'undo', 'redo',
    'updateUndoRedoButtons', 'updateHistoryList',
    'restoreGlobalStyles', 'updateNodeSize', 'updateNodeShape', 'updateEdgeWidth',
    'updateFontSize', 'updateArrowSize', 'updateEdgeLabelFontSize', 'updateMetric', 'updateEdgeWidths',
    'syncTransformInputs', 'syncGapDisplays', 'measureAxisGap', 'isVisibleElement', 'metricEdgeValue',
    'visibleNodeCentroid', 'gapAxesSwapped',
    'updateEdgeMetricLabels',
    'applyEdgeFilter', 'updateIgnoredEdgesPlaceholder',
    'shouldIgnoreEdge', 'evaluateEdgeCondition',
    'reapplyDeadEndHiding', 'reapplyOrphanHiding',
    'updateSelectionChip', 'historyIcon',
    'refreshEdgeStyles', 'clearEdgeEndpointOverrides', 'applyStraightEdgeStyle',
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
        let currentMetric = 'weight';
        let globalNodeSize = 40;
        let globalEdgeWidth = 3;
        let globalFontSize = 12;
        let globalEdgeLabelFontSize = 9;
        let globalArrowSize = 9;
        let globalEdgeWidthScale = 'log_e';
        let globalNodeShape = 'circle';
        let reciprocalMode = 'straight';
        function applyReciprocalMode(m) { reciprocalMode = m; }
        function syncReciprocalControls() {}
        let pendingNudge = null;
        function flushPendingNudge() {}
        let pendingStyle = null;
        function queueStyleHistory(label) {}
        function flushPendingStyle() {}
        let reciprocalOffset = 5;
        let restoringHistoryState = false;
        // Edge-filter groups (referenced by applyEdgeFilter / shouldIgnoreEdge)
        let edgeFilterGroups = [];
        // Layout transform trackers mirrored by captureState's globalStyles
        // (absolute inter-node gaps in px + rotation base).
        let lastGapX = null;
        let lastGapY = null;
        let baselineGapX = null;
        let baselineGapY = null;
        let lastRotationDeg = 0;
        // Operation feedback is a toast now; updateHoverInfo delegates to it.
        function showToast(message, type, action) {}
        let straightReciprocalEdgesEnabled = true;
        // Visibility/label flags referenced by captureState/restoreState
        let selfLoopsHidden = false;
        let orphansHidden = false;
        let deadEndsHidden = false;
        let hemisphereMirrorEnabled = false;
        let labelPosition = 'center';
        function updateHoverInfo() {}
        // restoreState re-applies the edge filter; the filter semantics are
        // covered by deadend_harness.js, so stub them here.
        function parseEdgeFilterInput() {}
        function applyEdgeFilter() {}
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
            undo, redo, pushHistory, captureState, restoreState,
            updateNodeSize, updateNodeShape, updateEdgeWidth, updateFontSize, updateArrowSize,
            updateMetric, updateEdgeWidths, refreshEdgeStyles,
            getEl: (id) => els[id] || makeEl(id),
            getUndoStack: () => undoStack, getRedoStack: () => redoStack,
        };
    `;

    const api = new Function('cy', src)(cy);
    return api;
}

function buildGraph() {
    // A source->target edge plus one self-loop on L (the loop is what
    // refreshEdgeStyles must keep outside the node).
    const elements = [
        { data: { id: 'S', node_type: 'source', label: 'S' } },
        { data: { id: 'T', node_type: 'target', label: 'T' } },
        { data: { id: 'L', node_type: 'intermediate', label: 'L' } },
        { group: 'edges', data: { id: 'e0', source: 'S', target: 'T', weight: 10, original_weight: 10, scaled_width: 2 } },
        { group: 'edges', data: { id: 'e1', source: 'L', target: 'L', weight: 5, original_weight: 5, scaled_width: 1 } },
    ];
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

// ===== Test A: node-size slider adjustments are recorded and undoable =====
{
    const cy = buildGraph();
    const api = buildScope(cy);
    api.getEl('nodeSizeSlider').value = '40';
    api.getEl('nodeSizeValue');
    api.updateNodeSize(60);   // user drags the Node Size slider 40 -> 60
    check('node size applied to stylesheet', cy.getElementById('S').numericStyle('width'), 60);
    check('node size recorded', api.getUndoStack().length, 1);
    check('node size label', api.getUndoStack()[0].label, 'Adjust node size');
    api.undo();
    check('undo restores node size', cy.getElementById('S').numericStyle('width'), 40);
    api.redo();
    check('redo reapplies node size', cy.getElementById('S').numericStyle('width'), 60);
    // snapshots carry the global style state
    const snap = api.captureState();
    check('snapshot carries globalStyles.nodeSize', snap.globalStyles.nodeSize, 60);
    check('snapshot carries globalStyles.metric', snap.globalStyles.metric, 'weight');
}

// ===== Test B: edge width / scaling method / metric all recorded =====
{
    const cy = buildGraph();
    const api = buildScope(cy);
    api.getEl('edgeWidthScale').value = 'log_e';
    api.getEl('metricSelect').value = 'weight';
    api.getEl('edgeWidthSlider').value = '3';
    api.updateEdgeWidth(8);
    check('edge width recorded', api.getUndoStack()[0].label, 'Adjust edge width');
    api.getEl('edgeWidthScale').value = 'linear';
    api.updateEdgeWidths();
    check('scale change recorded', api.getUndoStack()[1].label, 'Change edge width scale');
    api.getEl('metricSelect').value = 'ratio';
    api.updateMetric();
    check('metric change recorded', api.getUndoStack()[2].label, 'Change metric');
    // undo in reverse order restores each control exactly
    api.undo();
    check('undo restores metric', api.getEl('metricSelect').value, 'weight');
    api.undo();
    check('undo restores scale', api.getEl('edgeWidthScale').value, 'log_e');
    api.undo();
    check('undo restores edge width', api.getEl('edgeWidthSlider').value, 3);
    const snap = api.captureState();
    check('snapshot metric after undo', snap.globalStyles.metric, 'weight');
    check('snapshot edgeWidth after undo', snap.globalStyles.edgeWidth, 3);
    api.redo();
    api.redo();
    api.redo();
    check('redo all: metric', api.getEl('metricSelect').value, 'ratio');
    check('redo all: scale', api.getEl('edgeWidthScale').value, 'linear');
    check('redo all: width', api.getEl('edgeWidthSlider').value, 8);
}

// ===== Test C: self-loop geometry = the 3/4-circle fit, scaled by node =====
// Cytoscape renders every self-loop as a cubic bezier whose control points
// sit at 1.4 x control-point-step-size from the node center (distances are
// ignored for loops). With the default 90deg sweep, endpoints land on the
// node circle exactly 90deg apart (top -> left) with tangents through the
// node center. The step size must be (3.0 * nodeRadius) / 1.4 — the closest
// single-cubic approximation of the ideal 3/4 circle (same radius as the
// node) — and must follow the ACTUAL rendered node size.
{
    const cy = buildGraph();
    const api = buildScope(cy);
    api.getEl('nodeSizeSlider').value = '40';
    const loop = cy.getElementById('e1');
    const step = () => loop.numericStyle('control-point-step-size');
    const nodeWidth = () => cy.getElementById('L').numericStyle('width');
    const expectedStep = () => (3.0 * nodeWidth() / 2) / 1.4;

    // refresh first, exactly like the page does after any size/width change
    api.refreshEdgeStyles(false);
    check('step size = 3r/1.4 at default size', Math.round(step() * 1e6) / 1e6, Math.round(expectedStep() * 1e6) / 1e6);
    check('loop-direction set', loop.style('loop-direction'), '-45deg');
    check('loop-sweep set', loop.style('loop-sweep'), '-90deg');

    // replicate Cytoscape's findLoopPoints/findEndpoints geometry: control
    // points at 1.4*step along the out/in angles; endpoints on the node
    // circle along those rays (top and left, exactly 90deg apart)
    const L = 1.4 * step();
    const loopDir = -45 * Math.PI / 180, loopSwp = -90 * Math.PI / 180;
    const outAngle = loopDir - Math.PI / 2 - loopSwp / 2;
    const inAngle = loopDir - Math.PI / 2 + loopSwp / 2;
    const c1 = { x: Math.cos(outAngle) * L, y: Math.sin(outAngle) * L };
    const c2 = { x: Math.cos(inAngle) * L, y: Math.sin(inAngle) * L };
    const r = nodeWidth() / 2;
    check('control points at 3r from center', [Math.round(c1.x), Math.round(Math.abs(c1.y)), Math.round(Math.abs(c2.x)), Math.round(c2.y)], [0, 3 * r, 3 * r, 0]);
    const start = { x: Math.cos(outAngle) * r, y: Math.sin(outAngle) * r };
    const end = { x: Math.cos(inAngle) * r, y: Math.sin(inAngle) * r };
    const sep = Math.acos(Math.min(1, Math.max(-1, (start.x * end.x + start.y * end.y) / (r * r)))) * 180 / Math.PI;
    check('endpoints on node circle (start top, end left)', [Math.round(start.x), Math.round(Math.abs(start.y)), Math.round(Math.abs(end.x)), Math.round(end.y)], [0, r, r, 0]);
    check('endpoints exactly 90deg apart', Math.round(sep), 90);

    // node enlarged via per-element bypass (geometry editor): loop scales
    cy.getElementById('L').style({ 'width': '120px', 'height': '120px' });
    api.refreshEdgeStyles(false);
    check('step size follows per-element node size', Math.round(step() * 1e6) / 1e6, Math.round((3.0 * 120 / 2) / 1.4 * 1e6) / 1e6);

    // edge width must NOT affect the loop (the 3/4 circle is the centerline)
    loop.style('width', '12px');
    api.refreshEdgeStyles(false);
    check('step size independent of edge width', Math.round(step() * 1e6) / 1e6, Math.round((3.0 * 120 / 2) / 1.4 * 1e6) / 1e6);

    // global node-size slider also rescales the loop (updateNodeSize
    // refreshes edge styles)
    cy.getElementById('L').removeStyle('width');
    cy.getElementById('L').removeStyle('height');
    api.updateNodeSize(80);
    check('step size follows global slider', Math.round(step() * 1e6) / 1e6, Math.round((3.0 * 80 / 2) / 1.4 * 1e6) / 1e6);
}

console.log(failures === 0 ? 'ALL GLOBAL-STYLE TESTS PASSED' : failures + ' GLOBAL-STYLE TEST(S) FAILED');
// styleEnabled cytoscape cores keep background timers alive; exit explicitly.
process.exit(failures === 0 ? 0 : 1);
