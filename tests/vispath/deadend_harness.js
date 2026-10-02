// Test harness: extracts isDeadEndNodeIn / recomputeDeadEnds from a
// generated vispath network HTML and runs them against headless Cytoscape.
// Usage: node deadend_test.js <path-to-network.html>
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

// NOTE: the extracted sources come from the project's own generated HTML
// artifact (written by vispath.py in this same repo), i.e. a trusted,
// locally-produced input — never user-supplied. new Function is used only
// to execute those extracted functions against a headless Cytoscape core.
function loadFunction(name, cy) {
    const fnSrc = extractFunction(name, html);
    return new Function('cy', fnSrc + '\nreturn ' + name + ';')(cy);
}

// Extract the Cytoscape stylesheet array from the generated HTML (the
// `style: [...]` block inside the cytoscape() call) using a bracket-
// balanced scanner. Quotes are honored so brackets inside strings do not
// terminate the array early.
function extractStyleArray(source) {
    const marker = 'style: [';
    const start = source.indexOf(marker);
    if (start === -1) throw new Error('stylesheet array not found');
    let depth = 0;
    let inStr = null;
    for (let i = start + marker.length - 1; i < source.length; i++) {
        const c = source[i];
        if (inStr) {
            if (c === '\\') { i++; continue; }
            if (c === inStr) inStr = null;
            continue;
        }
        if (c === '\'' || c === '"') { inStr = c; continue; }
        if (c === '[') depth++;
        else if (c === ']') {
            depth--;
            if (depth === 0) return source.slice(start + marker.length - 1, i + 1);
        }
    }
    throw new Error('unbalanced stylesheet array');
}

// isOrphanNode calls isEdgeInCurrentGraph, so both sources are compiled
// into the same function scope.
function loadOrphanHelper(cy) {
    const src = extractFunction('isEdgeInCurrentGraph', html) + '\n' + extractFunction('isOrphanNode', html);
    return new Function('cy', src + '\nreturn isOrphanNode;')(cy);
}

const isDeadEndNodeIn = loadFunction('isDeadEndNodeIn', null);

// recomputeDeadEnds calls isDeadEndNodeIn, so both sources are compiled
// into the same function scope.
function makeRecompute(cy) {
    const src = extractFunction('isDeadEndNodeIn', html) + '\n' + extractFunction('recomputeDeadEnds', html);
    return new Function('cy', src + '\nreturn recomputeDeadEnds;')(cy);
}

// The stylesheet is embedded as a JS array literal; cytoscape accepts the
// parsed array (a raw string would be parsed as the CSS-ish string format).
const styleArray = new Function('return ' + extractStyleArray(html))();

function buildGraph(nodes, edges) {
    const elements = [];
    for (const [id, ntype] of Object.entries(nodes)) {
        elements.push({ data: { id: id, node_type: ntype } });
    }
    let i = 0;
    for (const [s, t, w] of edges) {
        elements.push({ group: 'edges', data: { id: 'e' + (i++), source: s, target: t, weight: w || 1, original_weight: w || 1 } });
    }
    return cytoscape({ headless: true, styleEnabled: true, elements: elements, style: styleArray });
}

function hiddenDeadEnds(cy) {
    return cy.nodes('.deadend-hidden').map(n => n.id()).sort();
}

let failures = 0;
function check(name, got, expected) {
    const g = JSON.stringify(got);
    const e = JSON.stringify([].concat(expected).sort());
    const pass = g === e;
    console.log((pass ? 'PASS' : 'FAIL') + ' | ' + name + ' | hidden=' + g + (pass ? '' : ' expected=' + e));
    if (!pass) failures++;
}

// --- Scenario 1: dead-end branch off a source-anchored chain ---
// S(source) -> A -> B (B is an in-only non-target). B is a dead end; hiding
// B exposes A as an in-only non-target dead end. Once A is hidden, S's only
// edge is dead-end-hidden, so S becomes an orphan - and orphans are dead
// ends - so S is hidden too.
{
    const cy = buildGraph({ S: 'source', A: 'intermediate', B: 'intermediate' }, [['S', 'A'], ['A', 'B']]);
    makeRecompute(cy)();
    check('source-anchored chain propagates (incl. orphaned source)', hiddenDeadEnds(cy), ['A', 'B', 'S']);
}

// --- Scenario 2: pure intermediate chain A->B->C (order independence) ---
// A is out-only non-source, C is in-only non-target: BOTH must be hidden
// regardless of node iteration order. B (in+out) then has only dead-end
// edges -> orphan -> dead end, so all three are hidden.
{
    const cy = buildGraph({ A: 'intermediate', B: 'intermediate', C: 'intermediate' }, [['A', 'B'], ['B', 'C']]);
    makeRecompute(cy)();
    check('chain hides all (ends + orphaned middle)', hiddenDeadEnds(cy), ['A', 'B', 'C']);
}

// --- Scenario 3: mutual dead ends X->D ---
// X out-only non-source, D in-only non-target: both hidden.
{
    const cy = buildGraph({ X: 'intermediate', D: 'intermediate' }, [['X', 'D']]);
    makeRecompute(cy)();
    check('mutual dead ends both hidden', hiddenDeadEnds(cy), ['X', 'D']);
}

// --- Scenario 4: complete source->target chain is NOT hidden ---
{
    const cy = buildGraph({ S: 'source', A: 'intermediate', B: 'intermediate', T: 'target' }, [['S', 'A'], ['A', 'B'], ['B', 'T']]);
    makeRecompute(cy)();
    check('source-target chain untouched', hiddenDeadEnds(cy), []);
}

// --- Scenario 5: current graph determined by the hide-edges filter ---
// Filter hides S->A (weight 10) and B->T (weight 30) via the 'filtered'
// class -> current graph is A->B. A becomes out-only non-source, B becomes
// in-only non-target -> both dead ends. S and T then have zero visible
// edges -> orphans -> dead ends too.
{
    const cy = buildGraph({ S: 'source', A: 'intermediate', B: 'intermediate', T: 'target' }, [['S', 'A', 10], ['A', 'B', 20], ['B', 'T', 30]]);
    const byWeight = {};
    cy.edges().forEach(e => { byWeight[e.data('weight')] = e.id(); });
    cy.getElementById(byWeight[10]).addClass('filtered');
    cy.getElementById(byWeight[30]).addClass('filtered');
    makeRecompute(cy)();
    check('filter defines current graph (orphaned ends hidden)', hiddenDeadEnds(cy), ['A', 'B', 'S', 'T']);
}

// --- Scenario 6: self-loop edges never count toward degrees ---
{
    const cy = buildGraph({
        A: 'intermediate', B: 'intermediate', C: 'intermediate'
    }, [['A', 'B'], ['B', 'A'], ['B', 'C']]);
    cy.add({ group: 'edges', data: { id: 'loopC', source: 'C', target: 'C', weight: 1 } });
    cy.getElementById('loopC').addClass('selfloop-hidden');
    // C in-only non-target -> dead end; A/B mutually connected -> not dead ends
    makeRecompute(cy)();
    check('self-loop edges excluded', hiddenDeadEnds(cy), ['C']);
}

// --- Scenario 7: manually hidden edges do not count ---
{
    const cy = buildGraph({
        S: 'source', A: 'intermediate', B: 'intermediate', C: 'intermediate'
    }, [['S', 'A'], ['A', 'B'], ['B', 'C'], ['A', 'C']]);
    // Hide B->C manually: current graph S->A, A->B, A->C. B and C are
    // in-only non-target -> dead ends; hiding them propagates to A, whose
    // only remaining visible edge is the incoming S->A (in-only non-target).
    // S then has only dead-end edges -> orphan -> dead end.
    cy.edges().forEach(e => {
        if (e.source().id() === 'B' && e.target().id() === 'C') e.addClass('hidden');
    });
    makeRecompute(cy)();
    check('manually hidden edges excluded', hiddenDeadEnds(cy), ['A', 'B', 'C', 'S']);
}

// --- Scenario 8: propagation through a long chain to fixpoint ---
// S -> N1 -> ... -> N6 (all intermediate). N6 (in-only) is a dead end first;
// hiding propagates backward: N5, N4, ... all become in-only non-target, and
// finally S is left with only dead-end edges -> orphan -> dead end.
{
    const nodes = { S: 'source' };
    const edges = [];
    for (let i = 1; i <= 6; i++) {
        nodes['N' + i] = 'intermediate';
        edges.push(i === 1 ? ['S', 'N1'] : ['N' + (i - 1), 'N' + i]);
    }
    const cy = buildGraph(nodes, edges);
    makeRecompute(cy)();
    check('long chain propagates to fixpoint (incl. orphaned source)', hiddenDeadEnds(cy), ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'S']);
}

// --- Scenario 10: dead-end classes must actually HIDE elements ---
// The stylesheet must contain node/edge deadend-hidden rules with
// display:none, and a hidden dead end must compute to display:none
// (regression: classes were assigned and counted, but no style hid them).
// Graph: a full S->A->B->T chain (survives), an orphan O (no edges) and a
// self-loop-only node L - both orphans, both dead ends.
{
    const cy = buildGraph({ S: 'source', A: 'intermediate', B: 'intermediate', T: 'target', O: 'intermediate', L: 'intermediate' },
        [['S', 'A'], ['A', 'B'], ['B', 'T']]);
    cy.add({ group: 'edges', data: { id: 'loopL', source: 'L', target: 'L', weight: 1 } });
    makeRecompute(cy)();
    const displayOf = id => [cy.getElementById(id).numericStyle('display')];
    check('dead-end node computes display:none (orphan O)', displayOf('O'), ['none']);
    check('dead-end node computes display:none (self-loop-only L)', displayOf('L'), ['none']);
    check('chain node stays visible', displayOf('A'), ['element']);
    check('source stays visible', displayOf('S'), ['element']);
    check('target stays visible', displayOf('T'), ['element']);
    const edgeDisplays = {};
    cy.edges().forEach(e => { edgeDisplays[e.data('id')] = [e.numericStyle('display')]; });
    check('self-loop edge hidden (loopL)', edgeDisplays.loopL, ['none']);
    check('chain edge stays visible (e1)', edgeDisplays.e1, ['element']);
}

// --- Scenario 11: orphans are dead ends ---
// Nodes with no edges at all are orphans, and orphans are a kind of dead
// end, so dead-end hiding removes them (both O and the edge-less source S).
{
    const cy = buildGraph({ S: 'source', O: 'intermediate' }, []);
    makeRecompute(cy)();
    check('orphans are dead ends', hiddenDeadEnds(cy), ['O', 'S']);
}

// --- Scenario 12: self-loop-only nodes are orphans (and dead ends) ---
// A node whose only edge is a self-loop has no connections in the current
// graph: isOrphanNode must be true and dead-end hiding must remove it.
// A node with a self-loop PLUS a real edge is not an orphan.
{
    const cy = buildGraph({ A: 'intermediate', L: 'intermediate' }, [['A', 'L']]);
    cy.add({ group: 'edges', data: { id: 'loopL', source: 'L', target: 'L', weight: 1 } });
    const isOrphanNode = loadOrphanHelper(cy);
    check('node with self-loop + real edge is not an orphan', [isOrphanNode(cy.getElementById('L'))], [false]);
    check('node with real edge is not an orphan', [isOrphanNode(cy.getElementById('A'))], [false]);

    const cy2 = buildGraph({ L: 'intermediate' }, []);
    cy2.add({ group: 'edges', data: { id: 'loopL', source: 'L', target: 'L', weight: 1 } });
    const isOrphanNode2 = loadOrphanHelper(cy2);
    check('self-loop-only node is an orphan', [isOrphanNode2(cy2.getElementById('L'))], [true]);
    makeRecompute(cy2)();
    check('self-loop-only node hidden as dead end', hiddenDeadEnds(cy2), ['L']);
}

// --- Scenario 13: orphan detection follows the current graph ---
// A node whose edges are all filter-hidden or dead-end-hidden is an orphan;
// self-loop edges never count toward connectivity.
{
    const cy = buildGraph({ S: 'source', A: 'intermediate', B: 'intermediate' }, [['S', 'A'], ['A', 'B']]);
    const isOrphanNode = loadOrphanHelper(cy);
    const all = cy.edges();
    all.forEach(e => e.addClass('filtered'));
    check('all-edges-filtered node is an orphan', [isOrphanNode(cy.getElementById('A'))], [true]);
    check('source with all edges filtered is an orphan', [isOrphanNode(cy.getElementById('S'))], [true]);
    all.forEach(e => e.removeClass('filtered'));
    check('restored edges make node non-orphan', [isOrphanNode(cy.getElementById('A'))], [false]);
}

// --- Scenario 9: re-run is idempotent and re-derives classes ---
// Running recompute twice yields the same set; manually removing a dead-end
// class then re-running restores it (fresh set, no stale-class reliance).
{
    const cy = buildGraph({ A: 'intermediate', B: 'intermediate' }, [['A', 'B']]);
    const recompute = makeRecompute(cy);
    recompute();
    const first = hiddenDeadEnds(cy);
    recompute();
    check('recompute idempotent', hiddenDeadEnds(cy), first);
    cy.getElementById('A').removeClass('deadend-hidden');
    cy.getElementById('e0').removeClass('deadend-hidden');
    recompute();
    check('stale classes re-derived', hiddenDeadEnds(cy), ['A', 'B']);
}

console.log(failures === 0 ? 'ALL DEAD-END TESTS PASSED' : failures + ' DEAD-END TEST(S) FAILED');
// Force exit: the style-enabled Cytoscape cores keep the event loop alive
// even after all tests complete, which would hang the pytest subprocess.
process.exit(failures === 0 ? 0 : 1);
