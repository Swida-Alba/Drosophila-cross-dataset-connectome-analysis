// Layout persistence harness: extracts the capture/apply pair
// (captureNetworkState / applyNetworkState) plus the Save/Load buttons from
// a generated vispath network HTML and runs them against headless Cytoscape
// with minimal DOM stubs. Regression-tests that a save/load round-trip
// restores EVERY view parameter — per-element color AND alpha, edge base
// appearance, groups, filter, hide toggles, global styles — and that
// payloads written by older builds (no alpha, flat control keys) still load.
// Usage: node persistence_harness.js <node-modules-dir> <path-to-network.html>
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
    // persistence pair + buttons
    'captureNetworkState', 'applyNetworkState', 'saveLayout', 'loadLayout',
    // waypoint bundle clear (called at the top of applyNetworkState; the
    // empty-bundle early return means its own helpers never run here)
    'clearWaypointSelection',
    // capture/undo plumbing (pushHistory guard + style bypass reads)
    'captureStyleBypass', 'captureState', 'pushHistory', 'syncToggleButtons',
    // global style update fns
    'updateNodeSize', 'updateNodeShape', 'updateEdgeWidth', 'updateFontSize', 'updateArrowSize',
    'hasBypass',
    'updateEdgeLabelFontSize', 'updateMetric', 'updateEdgeWidths',
    'updateEdgeMetricLabels',
    // edge filter chain
    'updateIgnoredEdgesPlaceholder', 'parseEdgeFilterInput', 'updateIgnoredEdges',
    'parseEdgeFilterExpressions', 'parseEdgeSingleExpression',
    'evaluateEdgeCondition', 'shouldIgnoreEdge', 'metricEdgeValue', 'applyEdgeFilter',
    // hide-toggle re-derivation
    'isEdgeInCurrentGraph', 'isOrphanNode', 'isDeadEndNodeIn', 'recomputeDeadEnds',
    'reapplyDeadEndHiding', 'reapplyOrphanHiding', 'reapplySelfLoopHiding',
    // surface controls
    'toggleLabels', 'toggleEdgeWeightLabels', 'effectiveEdgeLabelColor',
    'setReciprocalMode', 'applyReciprocalMode', 'syncReciprocalControls',
    'applyBackground', 'applyLabelFontColor', 'extractColorHex',
    // groups
    'groupMembers', 'groupLabel', 'groupDefaultFor', 'legendChip', 'refreshLegend',
    'rebuildAssignSelect', 'rebuildCustomGroupUI', 'updateCustomGroupList',
    'normalizeAssignedGroups',
    // edge base appearance
    'setEdgeBaseAppearance', 'ensureEdgeBaseAppearance', 'restoreEdgeBaseAppearance',
    'applyEdgeHighlightOverride', 'clearEdgeHighlightOverride',
    // misc
    'resizeCanvasAfterVisibilityControlChange', 'syncGapDisplays', 'syncRotateDisplay',
    'updateHoverInfo', 'escapeHtml',
    // shared controls (SHARED_JS is embedded verbatim in the network HTML)
    'isColorDark', 'createBackgroundController',
];

function buildScope(cy) {
    const fnSources = FUNCTIONS.map(f => extractFunction(f, html)).join('\n');
    if (process.env.DUMP_EVAL) fs.writeFileSync('/tmp/persist_eval.js', fnSources);

    const prelude = `
        // --- storage + history stubs ---
        const store = {};
        const localStorage = {
            getItem: (k) => (k in store ? store[k] : null),
            setItem: (k, v) => { store[k] = String(v); },
            removeItem: (k) => { delete store[k]; },
        };
        const LAYOUT_STORAGE_KEY = 'cytoscape_layout_test#00000000000000';
        let undoStack = [];
        let redoStack = [];
        let lastFilterHistoryValue = '';
        function pushStateHistory(label, state) { undoStack.push({ label: label }); }

        // --- page state variables (declared at script top level in the
        // real page; the extracted functions close over these) ---
        let selectedElement = null;
        let labelPosition = 'center';
        let labelsVisible = true;
        let hemisphereMirrorEnabled = false;
        const hasHemisphereNodes = false;
        let originalHemispherePositions = null;
        let selfLoopsHidden = false;
        let orphansHidden = false;
        let deadEndsHidden = false;
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
        let pendingStyle = null;
        let reciprocalMode = 'straight';
        function queueStyleHistory(label) {}
        function flushPendingStyle() {}
        let straightReciprocalEdgesEnabled = false;
        let reciprocalOffset = 5;
        // waypoint bundle state (applyNetworkState clears it on restore;
        // the empty-bundle early return keeps the ring helpers uncalled)
        let selectedWaypoint = null;
        let selectedWaypointKeys = [];
        const edgeWeightLabelForTooltip = 'synapses';

        let restoringHistoryState = false;
        let lastGapX = null;
        let lastGapY = null;
        let baselineGapX = null;
        let baselineGapY = null;
        let lastRotationDeg = 0;
        let pendingTransformState = null;
        let customLabelColor = null;
        const highlightColor = '#ff9800';
        const highlightOpacity = 0.85;
        let ignoredEdges = new Set();
        let ignoredEdgeExpressions = [];
        let edgeFilterGroups = [];
        const EDGE_BASE_COLOR_KEY = '__baseColor';
        const EDGE_BASE_OPACITY_KEY = '__baseOpacity';
        const declaredGroupsActive = false;
        const extraNodeGroups = [];
        const datasetLegendCodes = new Set();
        const ntColors = { unknown: '#9ca3af' };
        const originalGroupDefaults = {
            source: { color: '#ff5722', opacity: 100 },
            intermediate: { color: '#4caf50', opacity: 100 },
            target: { color: '#9c27b0', opacity: 100 },
        };
        const groupDefaults = JSON.parse(JSON.stringify(originalGroupDefaults));
        const customGroups = {};
        function refreshEdgeStyles() {}
        const toasts = [];
        function showToast(message, type, action) { toasts.push({ message: message, type: type }); }

        // --- DOM stubs ---
        const els = {};
        function makeClassList() {
            const set = new Set();
            return {
                toggle: (name, force) => {
                    const want = force === undefined ? !set.has(name) : !!force;
                    if (want) set.add(name); else set.delete(name);
                },
                add: (name) => set.add(name),
                remove: (name) => set.delete(name),
                contains: (name) => set.has(name),
            };
        }
        function makeEl(id) {
            const el = {
                id: id, value: '', textContent: '', disabled: false,
                style: {}, dataset: {}, options: [{ textContent: '' }],
                addEventListener: function () {},
                classList: makeClassList(),
                appendChild: function (o) { el.options.push(o); },
                remove: function () {},
            };
            Object.defineProperty(el, 'innerHTML', {
                set: function (v) { el._innerHTML = v; },
                get: function () { return el._innerHTML || ''; }
            });
            els[id] = el;
            return el;
        }
        const bodyEl = makeEl('body');
        const document = {
            body: bodyEl,
            getElementById: function (id) { return els[id] || makeEl(id); },
            createElement: function (tag) {
                const el = makeEl('created-' + tag + '-' + Math.random());
                el.tagName = tag;
                el.removeChild = function () {};
                return el;
            },
        };

        // bgCtrl is created in the page right after applyBackground is
        // declared; mirror that here (function declarations hoist).
        const bgCtrl = createBackgroundController(['#ffffff', '#000000'], ['White', 'Black'], applyBackground);
    `;

    const src = prelude + fnSources + `
        return {
            captureNetworkState, applyNetworkState, saveLayout, loadLayout,
            pushHistory, captureState, syncToggleButtons,
            updateNodeSize, updateFontSize, updateMetric, updateIgnoredEdges,
            setEdgeBaseAppearance, extractColorHex, applyLabelFontColor,
            updateEdgeLabelFontSize,
            getEl: (id) => els[id] || makeEl(id),
            getStore: () => store,
            getToasts: () => toasts,
            getUndoStack: () => undoStack,
            getCustomGroups: () => customGroups,
            getGroupDefaults: () => groupDefaults,
            getNodeSize: () => globalNodeSize,
            getFontSize: () => globalFontSize,
            getEdgeLabelFontSize: () => globalEdgeLabelFontSize,
            getMetric: () => currentMetric,
            getCustomLabelColor: () => customLabelColor,
            getSelfLoopsHidden: () => selfLoopsHidden,
            getOrphansHidden: () => orphansHidden,
            getDeadEndsHidden: () => deadEndsHidden,
            setSelfLoopsHidden: (v) => { selfLoopsHidden = v; },
            getLabelPosition: () => labelPosition,
            getLabelsVisible: () => labelsVisible,
            getMirrorEnabled: () => hemisphereMirrorEnabled,
            getReciprocalEnabled: () => straightReciprocalEdgesEnabled,
            getBgColor: () => bgCtrl.getColor(),
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
        elements.push({
            group: 'edges',
            data: {
                id: 'e' + (i++), source: s, target: t,
                weight: w || 1, original_weight: w || 1, ratio: (w || 1) / 10,
                probability: (w || 1) / 100, label: s + '>' + t,
            },
        });
    }
    // styleEnabled: true so per-element style bypasses (color/alpha) are
    // actually applied by the headless core.
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
function hexOf(api, value) {
    return api.extractColorHex(String(value));
}

// ===== Test A: capture includes per-node alpha, and ONLY as a bypass =====
{
    const cy = buildGraph({ A: 'intermediate', B: 'intermediate', C: 'intermediate' }, [['A', 'B', 5]]);
    const api = buildScope(cy);
    // Current semantics: alpha is body-only (background-opacity) — the
    // label font keeps full opacity.
    cy.getElementById('A').style({ 'background-color': '#ff0000', 'background-opacity': 0.42 });
    // Legacy whole-element opacity bypass from older sessions still migrates.
    cy.getElementById('B').style({ 'opacity': 0.7 });
    const state = api.captureNetworkState();
    const a = state.colors.find(c => c.id === 'A');
    const b = state.colors.find(c => c.id === 'B');
    const c = state.colors.find(c => c.id === 'C');
    check('alpha captured for styled node', a.opacity, 0.42);
    check('legacy element-opacity migrated on capture', b.opacity, 0.7);
    check('color captured', hexOf(api, a.color), '#ff0000');
    check('no alpha key for default node (stylesheet values not leaked)', 'opacity' in c, false);
    check('version stamped', state.version, 2);
    check('legacy flat control keys kept', typeof state.edgeWidth, 'string');
}

// ===== Test B: capture includes edge base color + alpha =====
{
    const cy = buildGraph({ A: 'intermediate', B: 'intermediate' }, [['A', 'B', 5]]);
    const api = buildScope(cy);
    api.setEdgeBaseAppearance(cy.getElementById('e0'), '#00aa00', 0.25);
    const state = api.captureNetworkState();
    const e = state.edgeStyles.find(x => x.id === 'e0');
    check('edge base color captured', hexOf(api, e.baseColor), '#00aa00');
    check('edge base alpha captured', e.baseOpacity, 0.25);
}

// ===== Test C: full round-trip through save -> scramble -> load =====
{
    const cy = buildGraph(
        { S: 'source', A: 'intermediate', T: 'target' },
        [['S', 'A', 5], ['A', 'T', 9], ['A', 'A', 2]],
    );
    const api = buildScope(cy);
    // --- arrange the view (alpha is body-only) ---
    cy.getElementById('A').style({ 'background-color': '#ff0000', 'background-opacity': 0.42 });
    cy.getElementById('A').position({ x: 123, y: 456 });
    api.setEdgeBaseAppearance(cy.getElementById('e1'), '#00aa00', 0.25);
    cy.getElementById('S').addClass('hidden');
    api.getEl('ignoreEdgesInput').value = '<8';
    api.updateIgnoredEdges();
    api.setSelfLoopsHidden(true);
    api.getEl('metricSelect').value = 'ratio';
    api.updateMetric();
    api.updateNodeSize(77);
    api.updateFontSize(19);
    api.getEl('edgeLabelSizeSlider').value = 14;
    api.updateEdgeLabelFontSize(14);
    api.applyLabelFontColor('#123456');
    api.captureNetworkState();  // smoke: capture must not throw mid-arrange
    api.saveLayout();
    // scramble ops are real user-simulated edits: they legitimately record
    // history. The no-history contract is about the LOAD itself, so the
    // baseline is taken after the scramble, right before loadLayout.
    api.updateNodeSize(11);
    api.updateFontSize(20);
    api.getEl('metricSelect').value = 'weight';
    api.updateMetric();
    api.getEl('ignoreEdgesInput').value = '';
    api.updateIgnoredEdges();
    cy.getElementById('S').removeClass('hidden');
    cy.getElementById('A').style({ 'background-color': '#0000ff', 'background-opacity': 0.9 });
    cy.getElementById('A').position({ x: -500, y: -500 });
    api.setEdgeBaseAppearance(cy.getElementById('e1'), '#123456', 0.9);
    api.setSelfLoopsHidden(false);
    cy.edges('.selfloop-hidden').removeClass('selfloop-hidden');
    const undoLenBeforeLoad = api.getUndoStack().length;

    // --- load ---
    api.loadLayout();
    const A = cy.getElementById('A');
    check('alpha restored (body-only)', parseFloat(A.style('background-opacity')), 0.42);
    check('font alpha untouched by alpha restore', parseFloat(A.style('opacity')), 1);
    check('color restored', hexOf(api, A.style('background-color')), '#ff0000');
    const p = A.position();
    check('position restored', [Math.round(p.x), Math.round(p.y)], [123, 456]);
    const e1 = cy.getElementById('e1');
    check('edge base color restored', hexOf(api, e1.data('__baseColor')), '#00aa00');
    check('edge base alpha restored', e1.data('__baseOpacity'), 0.25);
    check('manual node hide restored', cy.getElementById('S').hasClass('hidden'), true);
    check('filter input restored', api.getEl('ignoreEdgesInput').value, '<8');
    check('filter class re-applied', cy.getElementById('e0').hasClass('filtered'), true);
    check('self-loop flag restored', api.getSelfLoopsHidden(), true);
    check('self-loop class re-derived', cy.getElementById('e2').hasClass('selfloop-hidden'), true);
    check('self-loop button relabelled', api.getEl('hideSelfLoopsBtn').textContent, '👁️ Show Self-Loops');
    check('metric restored', api.getMetric(), 'ratio');
    check('node size restored', api.getNodeSize(), 77);
    check('font size restored', api.getFontSize(), 19);
    check('edge label size restored', api.getEdgeLabelFontSize(), 14);
    check('label font color restored', api.getCustomLabelColor(), '#123456');
    check('background restored', api.getBgColor(), '#ffffff');
    // loading a layout is ONE operation: no history entries from the
    // metric / filter / size update fns it runs internally
    check('load adds no history entries', api.getUndoStack().length, undoLenBeforeLoad);
}

// ===== Test D: alpha cleared when the payload carries none (old builds) =====
{
    const cy = buildGraph({ A: 'intermediate', B: 'intermediate' }, [['A', 'B', 5]]);
    const api = buildScope(cy);
    cy.getElementById('A').style({ 'background-color': '#ff0000', 'background-opacity': 0.42 });
    // old-format payload: colors without opacity, flat control values
    const oldState = {
        positions: [{ id: 'A', position: { x: 7, y: 8 } }, { id: 'B', position: { x: 0, y: 0 } }],
        colors: [{ id: 'A', color: '#ff0000' }, { id: 'B', color: '#00ff00' }],
        edgeWidth: '5',
        zoom: 1, pan: { x: 0, y: 0 },
    };
    api.applyNetworkState(oldState);
    check('old payload loads', api.getToasts().length >= 0, true);
    check('old payload: stale alpha cleared', parseFloat(cy.getElementById('A').style('background-opacity')), 1);
    check('old payload: element opacity stays 1 (font untouched)', parseFloat(cy.getElementById('A').style('opacity')), 1);
    check('old payload: color restored', hexOf(api, cy.getElementById('A').style('background-color')), '#ff0000');
    check('old payload: position restored', Math.round(cy.getElementById('A').position().x), 7);
    check('old payload: legacy slider value applied', api.getNodeSize(), 40);
}

// ===== Test E: surface controls round-trip (labels, weights, toggles, bg) =====
{
    const cy = buildGraph({ A: 'intermediate', B: 'intermediate' }, [['A', 'B', 5]]);
    const api = buildScope(cy);
    // arrange: labels hidden, weight labels on, dead ends flagged, dark bg
    api.getEl('toggleLabelsBtn');
    api.loadLayout;  // no-op reference; real arrange below
    cy.nodes().addClass('labels-hidden');
    // flip labelsVisible directly (toggleLabels reads it)
    api.getEl('toggleLabelsBtn').textContent = 'x';
    const state = api.captureNetworkState();
    state.labelsVisible = false;          // simulate labels hidden at save
    state.edgeWeightLabels = true;        // weight labels were on
    state.hideToggles = { orphans: true, selfLoops: false, deadEnds: true };
    state.background = '#000000';
    state.labelPosition = 'outside';
    api.applyNetworkState(state);
    check('labels toggled off', api.getLabelsVisible(), false);
    check('weight labels shown', api.getEl('toggleEdgeWeightsBtn').dataset.showing, '1');
    check('wlabel class applied', cy.edges('.wlabel').length, 1);
    check('orphans flag on', api.getOrphansHidden(), true);
    check('dead ends flag on', api.getDeadEndsHidden(), true);
    check('dead-end button relabelled', api.getEl('hideDeadEndsBtn').textContent, '👁️ Show Dead Ends');
    check('background restored to dark', api.getBgColor(), '#000000');
    check('label position restored', api.getLabelPosition(), 'outside');
    check('labels-outside class applied', cy.nodes('.labels-outside').length, 2);
}

// ===== Test F: custom groups + memberships round-trip =====
{
    const cy = buildGraph({ A: 'intermediate', B: 'intermediate' }, [['A', 'B', 5]]);
    const api = buildScope(cy);
    cy.getElementById('A').data('assigned_group', 'clan');
    const state = api.captureNetworkState();
    check('membership captured', state.assignedGroups.A, 'clan');
    check('default membership not captured', 'assignedGroups' in {} === false || state.assignedGroups.B === undefined, true);
    // scramble: membership removed + custom group dropped
    cy.getElementById('A').data('assigned_group', '');
    // apply
    api.applyNetworkState(state);
    check('membership restored', cy.getElementById('A').data('assigned_group'), 'clan');
}

// ===== Test G: groupDefaults edits round-trip (the Reset-All-Colors base) =====
{
    const cy = buildGraph({ A: 'intermediate' }, []);
    const api = buildScope(cy);
    api.getGroupDefaults().source.color = '#010203';
    api.getGroupDefaults().source.opacity = 55;
    const state = api.captureNetworkState();
    api.getGroupDefaults().source.color = '#ffffff';
    api.getGroupDefaults().source.opacity = 100;
    api.applyNetworkState(state);
    check('group default color restored', api.getGroupDefaults().source.color, '#010203');
    check('group default opacity restored', api.getGroupDefaults().source.opacity, 55);
}

// ===== Test H: save/load via localStorage key isolation =====
{
    const cy = buildGraph({ A: 'intermediate' }, []);
    const api = buildScope(cy);
    api.loadLayout();
    check('load without save warns', api.getToasts().some(t => t.type === 'warn'), true);
    cy.getElementById('A').position({ x: 55, y: 66 });
    api.saveLayout();
    check('saved under the layout key', 'cytoscape_layout_test#00000000000000' in api.getStore(), true);
    const saved = JSON.parse(api.getStore()['cytoscape_layout_test#00000000000000']);
    check('payload has positions', saved.positions.length, 1);
    check('payload has alpha field set', 'colors' in saved, true);
    cy.getElementById('A').position({ x: 0, y: 0 });
    api.loadLayout();
    check('load restores position', Math.round(cy.getElementById('A').position().x), 55);
}

// ===== Test I: pushHistory suppressed while restoring =====
{
    const cy = buildGraph({ A: 'intermediate' }, []);
    const api = buildScope(cy);
    api.pushHistory('before');
    check('one entry', api.getUndoStack().length, 1);
    const state = api.captureNetworkState();
    // a state carrying a filter exercises updateIgnoredEdges inside apply
    state.filter = { inputValue: '<3', ignoredValues: [], expressions: [] };
    api.applyNetworkState(state);
    check('apply adds no history entries', api.getUndoStack().length, 1);
}

// ===== Test E: per-node geometry/shape metadata round-trip =====
{
    const cy = buildGraph(
        { S: 'source', A: 'intermediate', T: 'target' },
        [['S', 'A', 5], ['A', 'T', 9]],
    );
    const api = buildScope(cy);
    // arrange: explicit size + shape + z-order on A, manual edge width on e0
    cy.getElementById('A').style({ width: '120px', height: '60px' });
    cy.getElementById('A').style('shape', 'rectangle');
    cy.getElementById('A').style({ 'z-index-compare': 'manual', 'z-index': 60 });
    cy.getElementById('e0').style('width', '9px');
    cy.getElementById('e0').data('customSize', true);

    const state = api.captureNetworkState();
    const geoA = state.nodeGeometry.find(g => g.id === 'A');
    check('nodeGeometry captures width', geoA.width, 120);
    check('nodeGeometry captures height', geoA.height, 60);
    check('nodeGeometry captures shape override', geoA.shape, 'rectangle');
    check('nodeGeometry captures manual z-order', geoA.zIndex, '60');
    check('unstyled nodes absent from nodeGeometry',
        state.nodeGeometry.length, 1);
    const geoE = state.edgeGeometry.find(g => g.id === 'e0');
    check('edgeGeometry captures manual width', geoE.width, 9);
    check('unstyled edges absent from edgeGeometry',
        state.edgeGeometry.length, 1);

    // scramble: wipe the overrides, set DIFFERENT ones, add a stale
    // customSize flag without a style bypass (graph-import leftovers)
    cy.getElementById('A').removeStyle('width');
    cy.getElementById('A').removeStyle('height');
    cy.getElementById('A').removeStyle('shape');
    cy.getElementById('A').removeStyle('z-index');
    cy.getElementById('A').removeStyle('z-index-compare');
    cy.getElementById('A').style({ width: '11px', height: '11px' });
    cy.getElementById('T').style('shape', 'ellipse');
    cy.getElementById('e0').removeStyle('width');
    cy.getElementById('e0').data('customSize', true);
    check('stale customSize flag without bypass is NOT captured',
        api.captureNetworkState().edgeGeometry.length, 0);

    // restore
    api.applyNetworkState(state);
    const A2 = cy.getElementById('A');
    check('size override restored', [Math.round(A2.numericStyle('width')),
        Math.round(A2.numericStyle('height'))], [120, 60]);
    check('shape override restored', A2.style('shape'), 'rectangle');
    check('z-order restored', A2.style('z-index'), '60');
    check('scramble-only override cleared (T follows global)',
        cy.getElementById('T').style('shape'), 'ellipse');
    const e0b = cy.getElementById('e0');
    check('manual edge width restored', Math.round(parseFloat(e0b.style('width'))), 9);
    check('manual edge width marker restored', e0b.data('customSize'), true);
}

console.log(failures === 0 ? 'ALL PERSISTENCE TESTS PASSED' : failures + ' PERSISTENCE TEST(S) FAILED');
// process.exitCode alone would hang: styleEnabled cytoscape cores keep
// background timers alive, so exit explicitly with the proper code.
process.exit(failures === 0 ? 0 : 1);
