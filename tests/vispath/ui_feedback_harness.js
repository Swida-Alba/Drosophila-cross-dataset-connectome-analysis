// UI-feedback harness: extracts the REAL pure helpers introduced by the
// UI redesign from a generated vispath network HTML and runs them against
// headless Cytoscape — the toast queue, the dialog controller promise
// state machine, and the node-search matcher.
// Usage: node ui_feedback_harness.js <node-modules-dir> <path-to-network.html>
const cytoscape = require(process.argv[2] + '/node_modules/cytoscape');
const fs = require('fs');

const htmlPath = process.argv[3] || '/tmp/vispath-test/network_test.html';
const html = (() => {
    // Offline-embedded documents carry the vendored libraries inline;
    // common function names (e.g. dagre's `undo`) would shadow the
    // page's own during extraction.  Keep only the app's script block —
    // the one holding the elements JSON.
    const whole = fs.readFileSync(htmlPath, 'utf8');
    const blocks = whole.match(/<script[^>]*>[\s\S]*?<\/script>/g) || [];
    const own = blocks.filter(b => b.includes('const elements = {'));
    if (own.length !== 1)
        throw new Error('app script block not found in ' + htmlPath);
    return own[0];
})();

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
    'createToastQueue', 'createDialogController', 'matchNodes',
    'isVisibleElement', 'cycleSearchMatch',
];

// NOTE: sources come from the project's own generated HTML (trusted,
// locally-produced artifact); new Function only executes that code against
// the headless core + stubs.
function buildScope(cy) {
    const fnSources = FUNCTIONS.map(f => extractFunction(f, html)).join('\n');
    const src = fnSources + `
        return { createToastQueue, createDialogController, matchNodes };
    `;
    return new Function('cy', src)(cy);
}

// Scope for the search stepper: cycleSearchMatch runs against stubbed
// state so the test records WHICH match index gets applied.
function buildSearchScope() {
    const fnSource = extractFunction('cycleSearchMatch', html);
    const src = `
        let searchMatches = [];
        let searchIndex = -1;
        const applied = [];
        const counts = [];
        function updateSearchCount() {
            counts.push(searchMatches.length === 0 ? '' : (searchIndex + 1) + '/' + searchMatches.length);
        }
        function applySearchMatch() { applied.push(searchIndex); }
        ${fnSource}
        return {
            setMatches: (m) => { searchMatches = m; searchIndex = m.length > 0 ? 0 : -1; },
            cycle: (d) => cycleSearchMatch(d),
            getIndex: () => searchIndex,
            getApplied: () => applied,
            getCounts: () => counts,
        };
    `;
    return new Function('cy', src)();
}

// A(0,0), B(100,0), C(0,50) with labels; B additionally matches "bee".
function buildGraph() {
    const cy = cytoscape({
        headless: true,
        styleEnabled: true,
        elements: [
            { data: { id: 'N_A', node_type: 'source', label: 'Apple' } },
            { data: { id: 'N_B', node_type: 'intermediate', label: 'bee' } },
            { data: { id: 'N_C', node_type: 'target', label: 'Cherry' } },
            { group: 'edges', data: { id: 'e1', source: 'N_A', target: 'N_B', weight: 5 } },
        ],
    });
    return cy;
}

let failures = 0;
function check(name, got, expected) {
    const g = JSON.stringify(got);
    const e = JSON.stringify(expected);
    const pass = g === e;
    console.log((pass ? 'PASS' : 'FAIL') + ' | ' + name + ' | got=' + g + (pass ? '' : ' expected=' + e));
    if (!pass) failures++;
}

// Scope for the directional box-selection geometry: the REAL frame tests
// and selection applier run against a positioned headless graph.
function buildBoxScope(cy) {
    const names = ['isVisibleElement', 'boxNormalized', 'pointInRect',
        'segmentIntersectsRect', 'nodeRect', 'nodeFullyInRect', 'nodeTouchesRect',
        'edgeSamplePoints', 'edgeFullyInRect', 'edgeTouchesRect',
        'applyDirectionalBoxSelection'];
    const src = names.map(f => extractFunction(f, html)).join('\n') + `
        return { apply: applyDirectionalBoxSelection, seg: segmentIntersectsRect };
    `;
    return new Function('cy', src)(cy);
}

// A(0,0), B(100,0), C(0,50); e1 A→B horizontal, e2 A→C vertical,
// e3 B→C diagonal. Default nodes are 30x30 (±15 around the center).
function buildBoxGraph() {
    const cy = cytoscape({
        headless: true,
        styleEnabled: true,
        elements: [
            { data: { id: 'A', node_type: 'source', label: 'Apple' } },
            { data: { id: 'B', node_type: 'intermediate', label: 'bee' } },
            { data: { id: 'C', node_type: 'target', label: 'Cherry' } },
            { group: 'edges', data: { id: 'e1', source: 'A', target: 'B', weight: 5 } },
            { group: 'edges', data: { id: 'e2', source: 'A', target: 'C', weight: 5 } },
            { group: 'edges', data: { id: 'e3', source: 'B', target: 'C', weight: 5 } },
        ],
    });
    const coords = { A: [0, 0], B: [100, 0], C: [0, 50] };
    for (const [id, [x, y]] of Object.entries(coords)) {
        cy.getElementById(id).position({ x: x, y: y });
    }
    return cy;
}

function selectedIds(cy) {
    return cy.elements(':selected').map(e => e.id()).sort();
}

// Promise-resolution assertions below need an async context.
(async function main() {

// ===== Test A: toast queue — ordering, cap, expiry =====
{
    const cy = buildGraph();
    const api = buildScope(cy);
    const q = api.createToastQueue({ maxVisible: 3, ttl: 3000, errorTtl: 6000 });
    let now = 0;
    q.push('one', 'info', null, now);
    q.push('two', 'success', null, now);
    q.push('three', 'warn', null, now);
    check('three toasts visible', q.active().length, 3);
    q.push('four', 'info', null, now);
    check('cap keeps newest 3', q.active().map(i => i.message), ['two', 'three', 'four']);
    q.push('error', 'error', null, now);
    // at t=3500 the plain toasts expired, the error (6000) survives
    const after = q.expire(3500);
    check('error outlives plain toasts', after.map(i => i.message), ['error']);
    check('expire before ttl keeps all', q.expire(4000).length, 1);
    check('expire after ttl empties', q.expire(9500).length, 0);
}

// ===== Test B: dialog controller — open/set/confirm and cancel =====
{
    const cy = buildGraph();
    const api = buildScope(cy);
    const ctl = api.createDialogController();
    const spec = {
        title: 'Edit node',
        fields: [
            { key: 'label', label: 'Label', type: 'text', value: 'A' },
            { key: 'type', label: 'Type', type: 'select', value: 'intermediate', options: ['source', 'intermediate', 'target'] },
        ],
    };
    let resolved = null, rejected = null;
    const promise = ctl.open(spec)
        .then(v => { resolved = v; })
        .catch(e => { rejected = e; });
    check('dialog is active', ctl.isActive(), true);
    check('second open rejected', ctl.open(spec), null);
    ctl.set('label', 'A2');
    ctl.set('type', 'target');
    const values = ctl.confirm();
    check('confirm returns values', [values.label, values.type], ['A2', 'target']);
    await promise;  // microtask: the .then must run before we assert it
    check('promise resolved with values', resolved, { label: 'A2', type: 'target' });
    check('dialog inactive after confirm', ctl.isActive(), false);
    check('confirm without dialog is null', ctl.confirm(), null);
}

{
    const cy = buildGraph();
    const api = buildScope(cy);
    const ctl = api.createDialogController();
    let rejected = null;
    const promise = ctl.open({ title: 'x', fields: [] }).catch(e => { rejected = e; });
    ctl.cancel();
    await promise;
    check('cancel rejects with cancelled', rejected, 'cancelled');
    check('cancel without dialog is null', ctl.cancel(), null);
    check('dialog inactive after cancel', ctl.isActive(), false);
}

// ===== Test C: node search matcher (min 2 chars, id + label, CI) =====
{
    const cy = buildGraph();
    const api = buildScope(cy);
    check('short query matches nothing', api.matchNodes('a').length, 0);
    check('empty query matches nothing', api.matchNodes('').length, 0);
    check('id match is case-insensitive', api.matchNodes('n_b').length, 1);
    check('label match works', api.matchNodes('cherry').length, 1);
    check('no match returns empty', api.matchNodes('zebra').length, 0);
    check('two-char prefix matches label', api.matchNodes('ap').length, 1);
}

// ===== Test D: search stepping APPLIES each match (select + center) =====
// (the reported bug: navigation only bumped the 1/N counter — the shared
// cycleSearchMatch helper now drives the ▲/▼ buttons AND the ↑/↓ keys and
// applies every step)
{
    const api = buildSearchScope();
    api.setMatches(['n1', 'n2', 'n3']);
    api.cycle(1);   // 1/3 -> 2/3
    check('next applies match 2', api.getApplied(), [1]);
    api.cycle(1);   // -> 3/3
    api.cycle(1);   // wraps -> 1/3
    check('wrap-around cycle order', api.getApplied(), [1, 2, 0]);
    api.cycle(-1);  // wraps back -> 3/3
    check('previous wraps to the last match', api.getApplied(), [1, 2, 0, 2]);
    check('counter tracked every step', api.getCounts(), ['2/3', '3/3', '1/3', '3/3']);
    // no matches: a no-op, never applies
    const before = api.getApplied().slice();
    api.setMatches([]);
    api.cycle(1);
    api.cycle(-1);
    check('empty matches never apply', api.getApplied(), before);
}

// ===== Test E: the canvas drag-mode switch drives cytoscape =====
// (Select mode turns user panning OFF with box selection pinned ON —
// cytoscape maps a non-pannable canvas to box-select-on-drag)
{
    const cy = buildGraph();
    const els = {};
    const document = {
        getElementById: (id) => (els[id] = els[id] || {
            toggles: [],
            classList: { toggle(c, on) { this._t = this._t || []; this._t.push([c, !!on]); } },
            setAttribute() {}
        })
    };
    const hover = [];
    function updateHoverInfo(m) { hover.push(m); }
    const src = `
        let canvasDragMode = 'pan';
        ${extractFunction('setCanvasDragMode', html)}
        return { set: (m) => setCanvasDragMode(m), mode: () => canvasDragMode };
    `;
    const api = new Function('cy', 'document', 'updateHoverInfo', src)(cy, document, updateHoverInfo);
    check('pan is the default mode', api.mode(), 'pan');
    check('pan mode keeps panning on', cy.userPanningEnabled(), true);
    api.set('select');
    check('select mode disables panning', cy.userPanningEnabled(), false);
    check('box selection pinned on', cy.boxSelectionEnabled(), true);
    check('mode tracker at select', api.mode(), 'select');
    api.set('pan');
    check('pan mode restores panning', cy.userPanningEnabled(), true);
    check('feedback shown for both switches', hover.length, 2);
    api.set('nonsense');
    check('invalid mode is a no-op', cy.userPanningEnabled(), true);
    check('invalid mode adds no feedback', hover.length, 2);
}

// ===== Test F: directional box selection (L→R full / R→L touch) =====
// The reported asymmetry: nodes select on touch while edges need full
// containment. Now the DIRECTION decides — L→R frames are strict for
// everything, R→L (dashed) frames select on touch for everything. The
// gesture is also a combined select/deselect: a frame over ONLY
// already-selected elements removes them, ⇧+drag per-element toggles,
// and an empty frame is a no-op.
{
    // the box (80,-20)-(140,20) swallows B entirely and CROSSES e1 near x=100
    const rect = [80, -20, 140, 20];

    // L→R = full: B in; e1 is REJECTED even though the frame crosses it
    {
        const cy = buildBoxGraph();
        const api = buildBoxScope(cy);
        api.apply(rect[0], rect[1], rect[2], rect[3], false, null);
        check('L→R full selects only the contained node', selectedIds(cy), ['B']);
    }
    // R→L = touch (same frame, reversed drag): e1 and e3 join B
    {
        const cy = buildBoxGraph();
        const api = buildBoxScope(cy);
        api.apply(rect[2], rect[3], rect[0], rect[1], false, null);
        check('R→L touch adds the crossed edges', selectedIds(cy), ['B', 'e1', 'e3']);
    }
    // a frame that only PARTLY overlaps A: full rejects it, touch selects it
    {
        const cy = buildBoxGraph();
        const api = buildBoxScope(cy);
        api.apply(10, -20, 140, 20, false, null);
        check('L→R full skips the half-covered node', selectedIds(cy), ['B']);
        api.apply(140, 20, 10, -20, false, null);
        check('R→L touch selects the half-covered node',
            selectedIds(cy), ['A', 'B', 'e1', 'e3']);
    }
    // ⇧ (additive) unions with the pre-gesture selection instead of
    // replacing it
    {
        const cy = buildBoxGraph();
        const api = buildBoxScope(cy);
        cy.getElementById('A').select();
        const base = cy.elements().filter(e => e.selected());
        api.apply(rect[0], rect[1], rect[2], rect[3], true, base);
        check('additive keeps the pre-gesture selection',
            selectedIds(cy), ['A', 'B']);
    }
    // Liang–Barsky sanity: a segment crossing the frame with BOTH endpoints
    // outside still counts as touching
    {
        const cy = buildBoxGraph();
        const api = buildBoxScope(cy);
        check('crossing segment touches', api.seg(50, -30, 50, 30,
            { x1: 40, y1: -10, x2: 60, y2: 10 }), true);
        check('non-crossing segment does not', api.seg(50, 12, 50, 30,
            { x1: 40, y1: -10, x2: 60, y2: 10 }), false);
    }
    // ===== combined select/deselect =====
    // a frame over ONLY already-selected elements DESELECTS them and keeps
    // the rest of the selection
    {
        const cy = buildBoxGraph();
        const api = buildBoxScope(cy);
        cy.getElementById('A').select();
        cy.getElementById('B').select();
        const base = cy.elements().filter(e => e.selected());
        api.apply(rect[0], rect[1], rect[2], rect[3], false, base);
        check('frame over selected objects deselects them',
            selectedIds(cy), ['A']);
    }
    // ⇧ over a MIX flips each covered element: A drops out, B and e1 join
    {
        const cy = buildBoxGraph();
        const api = buildBoxScope(cy);
        cy.getElementById('A').select();
        const base = cy.elements().filter(e => e.selected());
        api.apply(-20, -20, 140, 20, true, base);
        check('⇧ toggle flips covered elements both ways',
            selectedIds(cy), ['B', 'e1']);
    }
    // a frame that catches nothing leaves the selection alone
    {
        const cy = buildBoxGraph();
        const api = buildBoxScope(cy);
        cy.getElementById('A').select();
        const base = cy.elements().filter(e => e.selected());
        api.apply(200, 200, 210, 210, false, base);
        check('empty frame is a no-op', selectedIds(cy), ['A']);
    }
}

console.log(failures === 0 ? 'ALL UI-FEEDBACK TESTS PASSED' : failures + ' UI-FEEDBACK TEST(S) FAILED');
// process.exitCode alone would hang: styleEnabled cytoscape cores keep
// background timers alive, so exit explicitly with the proper code.
process.exit(failures === 0 ? 0 : 1);

})();
