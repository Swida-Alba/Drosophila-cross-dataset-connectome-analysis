// Edit-mode edge-drawing harness: executes the generated page's REAL
// handleNodeClickForEdge against headless Cytoscape.
//
// Regression for a silent blocker: the duplicate-pair guard combined its
// two bidirectional queries with `.concat(`.  A Cytoscape collection has no
// concat method, so the call fell through to Array.prototype.concat, which
// treats each collection as ONE item — the result is a plain array of length
// 2 even when both queries match nothing.  `bidirCovers.length > 0` then
// always won, so every second click answered "Edge already exists" and no
// new edge was ever created.  A source-text check cannot see this (the
// emitted code looks right); only running the function can.
//
// Usage: node edge_add_harness.js <node-prefix> <path-to-network.html>
const cytoscape = require(process.argv[2] + '/node_modules/cytoscape');
const fs = require('fs');

const htmlPath = process.argv[3];
const html = (() => {
    // Offline-embedded documents carry the vendored libraries inline;
    // keep only the app's script block (the one holding the elements JSON).
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

// Extract the embedded `const elements = { nodes: ..., edges: ... }`
// literal (quote-aware balanced scan) and evaluate it.
function extractElementsObject(source) {
    const marker = 'const elements = ';
    const start = source.indexOf(marker);
    if (start === -1) throw new Error('elements object not found');
    let i = start + marker.length;
    while (source[i] !== '{') i++;
    let depth = 0, inStr = null;
    for (let j = i; j < source.length; j++) {
        const c = source[j];
        if (inStr) {
            if (c === '\\') { j++; continue; }
            if (c === inStr) inStr = null;
            continue;
        }
        if (c === '"' || c === '\'') { inStr = c; continue; }
        if (c === '{') depth++;
        else if (c === '}') {
            depth--;
            if (depth === 0) return new Function('return ' + source.slice(i, j + 1))();
        }
    }
    throw new Error('unbalanced elements object');
}

const elements = extractElementsObject(html);
const cy = cytoscape({ headless: true, elements: elements });

// Evaluate the page's own handler with the page's collaborators stubbed at
// the same seams (hover text, history bookkeeping, base appearance).
const api = (() => {
    const src = [
        'let edgeDrawMode = false;',
        'let sourceNodeForEdge = null;',
        'let nextEdgeId = 1000;',
        'const defaultEdgeColor = "#646464";',
        'const defaultEdgeOpacity = 1;',
        'const hover = [];',
        'let historyCalls = 0;',
        'function updateHoverInfo(t) { hover.push(String(t)); }',
        'function pushHistory() { historyCalls++; }',
        'function setEdgeBaseAppearance(e, c, o) { e.data("test_base_color", c); e.data("test_base_opacity", o); }',
        extractFunction('handleNodeClickForEdge', html),
        'return {',
        '  tap(id) { handleNodeClickForEdge(cy.getElementById(id)); },',
        '  get drawing() { return edgeDrawMode; },',
        '  get source() { return sourceNodeForEdge && sourceNodeForEdge.id(); },',
        '  get hover() { return hover; },',
        '  get historyCalls() { return historyCalls; },',
        '};',
    ].join('\n');
    return new Function('cy', src)(cy);
})();

let failures = 0;
function check(label, ok, detail) {
    console.log((ok ? 'PASS' : 'FAIL') + ' - ' + label + (ok || !detail ? '' : ': ' + detail));
    if (!ok) failures++;
}

function pair(src, tgt) {
    return cy.edges('[source = "' + src + '"][target = "' + tgt + '"]');
}
function lastHover() { return api.hover[api.hover.length - 1] || ''; }

// 1. Draw a brand-new pair: it must create exactly one edge.
api.tap('Y');
check('first click arms edge drawing', api.drawing === true && api.source === 'Y',
    JSON.stringify({ drawing: api.drawing, source: api.source }));
const before = cy.edges().length;
api.tap('D');
check('second click creates the edge', pair('Y', 'D').length === 1 && cy.edges().length === before + 1,
    JSON.stringify({ created: pair('Y', 'D').length, total: cy.edges().length }));
check('the new edge carries default data', (() => {
    const e = pair('Y', 'D');
    return e.length === 1 && e.data('weight') === 1 && e.data('original_weight') === 1
        && e.data('scaled_width') === 3 && e.data('test_base_color') === '#646464'
        && e.data('test_base_opacity') === 1;
})(), pair('Y', 'D').json && JSON.stringify(pair('Y', 'D').data()));
check('history records the addition once', api.historyCalls === 1, String(api.historyCalls));
check('hover confirms the creation', /Edge created: Y → D/.test(lastHover()), lastHover());
check('drawing state resets after use', api.drawing === false && api.source === null,
    JSON.stringify({ drawing: api.drawing, source: api.source }));

// 2. Same direction again: blocked as a duplicate, no second edge.
const afterCreate = cy.edges().length;
api.tap('Y'); api.tap('D');
check('duplicate forward pair is blocked', pair('Y', 'D').length === 1 && cy.edges().length === afterCreate,
    JSON.stringify({ edges: pair('Y', 'D').length, total: cy.edges().length }));
check('duplicate reports already-exists', /already exists: Y → D/.test(lastHover()), lastHover());

// 3. Reverse of a plain edge: blocked with the reverse-specific message.
api.tap('D'); api.tap('Y');
check('reverse of a plain edge is blocked', pair('D', 'Y').length === 0 && pair('Y', 'D').length === 1,
    JSON.stringify({ reverse: pair('D', 'Y').length, total: cy.edges().length }));
check('reverse reports reverse-exists', /Reverse edge exists: Y → D/.test(lastHover()), lastHover());

// 4. A merged bidirectional element covers the pair BOTH ways — the
//    union() path.  This is the exact branch that used to always fire.
pair('Y', 'D').data('bidirectional', 1);
const beforeBidir = cy.edges().length;
api.tap('D'); api.tap('Y');
check('bidirectional element blocks the reverse draw',
    pair('D', 'Y').length === 0 && cy.edges().length === beforeBidir,
    JSON.stringify({ reverse: pair('D', 'Y').length, total: cy.edges().length }));
check('bidirectional block reports already-exists', /already exists: D → Y/.test(lastHover()), lastHover());

// 5. Self-loop: refused, and drawing mode ends.
api.tap('A'); api.tap('A');
check('self-loop is refused', /self-loop/i.test(lastHover()) && api.drawing === false, lastHover());

// 6. A restored portable page can already own e1000+ ids. Reference the
//    drawn edge as e1000 (as if it had been saved and reopened), then draw
//    another pair: the allocator must skip the taken id instead of letting
//    cy.add throw "Can not create second element with ID e1000".
{
    const data = pair('Y', 'D').data();
    pair('Y', 'D').remove();
    cy.add({ group: 'edges', data: {
        id: 'e1000', source: 'Y', target: 'D', weight: data.weight,
        original_weight: data.original_weight, scaled_width: data.scaled_width,
        bidirectional: 1
    } });
}
let threw = null;
try { api.tap('X'); api.tap('D'); } catch (err) { threw = String(err.message || err); }
check('drawing after a restored e1000 id does not throw', threw === null, threw);
const drawn = pair('X', 'D');
check('drawing still works after blocked attempts',
    drawn.length === 1 && drawn.id() !== 'e1000', drawn.id());

console.log(failures ? failures + ' EDGE-ADD CHECKS FAILED' : 'ALL EDGE-ADD TESTS PASSED');
process.exit(failures ? 1 : 0);