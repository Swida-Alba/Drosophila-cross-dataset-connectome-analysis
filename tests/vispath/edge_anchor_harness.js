// Test harness: reciprocal-edge anchoring on DIAGONAL pairs, executed
// against the generated page's own geometry functions plus headless
// Cytoscape.  Regression for the diagonal bug where reciprocal arrows
// detached from the nodes and the two halves crossed in an X mid-gap:
//   1. rimDistance must use the exact polar rim for ellipse nodes (the
//      old ray-box formula overshoots a diagonal ellipse by up to 41%:
//      100px circle on a 3-4-5 diagonal -> 62.5 vs the true 50);
//   2. the computed endpoint anchors must sit ON each node's rim and ON
//      the perpendicular offset line;
//   3. cytoscape must parse the explicit px endpoints / zero
//      distance-from-node writes back exactly as the page intended (the
//      RENDERED seat itself is proven by the browser gate on a real
//      generated page — the endpoint pipeline only runs inside a real
//      canvas draw pass, which headless Node cannot drive).
// Usage: node edge_anchor_harness.js <node-prefix> <path-to-network.html>
// (The npm cytoscape install supplies a headless instance to verify that
// cytoscape parses/consumes the style writes; the RENDERED seat is proven
// by the browser gate on the real page, since the endpoint pipeline only
// runs inside a real canvas draw pass.)
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
    const marker = 'const ' + name + ' = ';
    const start = source.indexOf(marker);
    if (start === -1) throw new Error('function not found: ' + name);
    const open = source.indexOf('{', start);
    let depth = 0;
    for (let i = open; i < source.length; i++) {
        if (source[i] === '{') depth++;
        else if (source[i] === '}') {
            depth--;
            if (depth === 0) return source.slice(start, i + 1) + ';';
        }
    }
    throw new Error('unbalanced braces: ' + name);
}

// Evaluate the page's own geometry helpers with a stubbed `document`
// (they only read the node-size slider fallback from it).
const geometry = (() => {
    const src = ['nodeHalfSizes', 'rimDistance', 'rimPointOnOffsetLine']
        .map(n => extractFunction(n, html)).join('\n')
        + '\nmodule.exports = { nodeHalfSizes, rimDistance, rimPointOnOffsetLine };';
    const mod = { exports: {} };
    new Function('module', 'exports', 'document', src)(
        mod, mod.exports, { getElementById: () => ({ value: '40' }) });
    return mod.exports;
})();

let failures = 0;
function check(label, ok, detail) {
    if (ok) { console.log('  ok  ' + label); return; }
    failures++;
    console.error('  FAIL ' + label + (detail !== undefined ? '  -> ' + detail : ''));
}
function near(a, b, tol) { return Math.abs(a - b) <= tol; }

// Mock node standing in for a cytoscape node for the pure-geometry stage.
function mockNode(w, h, shape) {
    return {
        numericStyle: (prop) => (prop === 'width' ? w : prop === 'height' ? h : undefined),
        style: (prop) => (prop === 'shape' ? shape : undefined),
    };
}

console.log('[1] rimDistance — ellipse polar rim vs the old box overshoot');
{
    const circ100 = mockNode(100, 100, 'ellipse');
    // 3-4-5 diagonal: the old box formula returned 62.5, the true rim is 50
    check('100px circle, dir (120,90) rim = 50', near(geometry.rimDistance(circ100, 120, 90), 50, 1e-6),
        geometry.rimDistance(circ100, 120, 90));
    // exact 45°: old formula 28.28 on a 40px node, true rim 20
    const circ40 = mockNode(40, 40, 'ellipse');
    check('40px circle, dir (1,1) rim = 20', near(geometry.rimDistance(circ40, 1, 1), 20, 1e-6),
        geometry.rimDistance(circ40, 1, 1));
    // axis-aligned diagonals must stay exact (regression guard)
    check('40px circle, horizontal rim = 20', near(geometry.rimDistance(circ40, 99, 0), 20, 1e-6));
    check('40px circle, vertical rim = 20', near(geometry.rimDistance(circ40, 0, 99), 20, 1e-6));
    // tall ellipse: rim along x = hw, along y = hh
    const tall = mockNode(60, 100, 'ellipse');
    check('60x100 ellipse, horizontal rim = 30', near(geometry.rimDistance(tall, 5, 0), 30, 1e-6));
    check('60x100 ellipse, vertical rim = 50', near(geometry.rimDistance(tall, 0, 5), 50, 1e-6));
    // non-ellipse shapes keep the box intersection (62.5 on the 3-4-5 diag)
    const rect = mockNode(100, 100, 'round-rectangle');
    check('100px round-rectangle, dir (120,90) rim = 62.5',
        near(geometry.rimDistance(rect, 120, 90), 62.5, 1e-6),
        geometry.rimDistance(rect, 120, 90));
}

console.log('[2] rimPointOnOffsetLine — anchors on the rim AND on the offset line');
{
    const circ100 = mockNode(100, 100, 'ellipse');
    // canonical direction (120,90), offset 5 along perp (-0.6, 0.8) -> (-3, 4)
    const anchor = geometry.rimPointOnOffsetLine(circ100, 120, 90, -3, 4);
    const dist = Math.hypot(anchor[0], anchor[1]);
    check('anchor sits on the 50px rim', near(dist, 50, 0.5), dist);
    // on the offset line through (-3,4) with direction (0.8,0.6):
    // the cross-product distance from the line must vanish
    const lineDist = Math.abs((anchor[0] - (-3)) * 0.6 - (anchor[1] - 4) * 0.8);
    check('anchor lies on the offset line', lineDist < 0.5, lineDist);
    // zero offset collapses onto the center line at the exact rim point
    const onLine = geometry.rimPointOnOffsetLine(circ100, 120, 90, 0, 0);
    check('offset 0 -> anchor at (40, 30)',
        near(onLine[0], 40, 0.5) && near(onLine[1], 30, 0.5), onLine);
    // horizontal pairs keep working (regression guard)
    const horiz = geometry.rimPointOnOffsetLine(circ100, 300, 0, 0, 5);
    check('horizontal pair anchor rim-dist = 50', near(Math.hypot(horiz[0], horiz[1]), 50, 0.5),
        Math.hypot(horiz[0], horiz[1]));
}

console.log("[3] headless cytoscape — the page's style writes parse and consume cleanly");
{
    const cy = cytoscape({
        headless: true,
        styleEnabled: true,
        elements: [
            { data: { id: 'a', label: 'aMe4' }, position: { x: 0, y: 0 } },
            { data: { id: 'b', label: 'cM09' }, position: { x: 120, y: 90 } },
            { data: { id: 'ab', source: 'a', target: 'b', pair_id: 'a<->b', pair_canonical: 1 } },
            { data: { id: 'ba', source: 'b', target: 'a', pair_id: 'a<->b', pair_canonical: 0 } },
        ],
        style: [
            { selector: 'node', style: { width: 100, height: 100, shape: 'ellipse' } },
            { selector: 'edge', style: { 'curve-style': 'straight', 'target-arrow-shape': 'triangle', 'arrow-scale': 1.5 } },
        ],
    });

    // Reproduce the page's straight branch for both halves: canonical
    // frame, one shared perpendicular offset, explicit rim endpoints.
    const offset = 5;
    const canonicalSign = 'a' < 'b' ? 1 : -1;
    const canonicalDx = 120, canonicalDy = 90;
    const canonicalDistance = Math.hypot(canonicalDx, canonicalDy);
    const perpX = -canonicalDy / canonicalDistance;
    const perpY = canonicalDx / canonicalDistance;
    const offX = perpX * offset * canonicalSign;
    const offY = perpY * offset * canonicalSign;

    [['ab', 'a', 'b'], ['ba', 'b', 'a']].forEach(([id, src, tgt]) => {
        const e = cy.getElementById(id);
        const srcNode = cy.getElementById(src);
        const tgtNode = cy.getElementById(tgt);
        const srcPos = srcNode.position();
        const tgtPos = tgtNode.position();
        const srcRim = geometry.rimPointOnOffsetLine(
            srcNode, tgtPos.x - srcPos.x, tgtPos.y - srcPos.y, offX, offY);
        const tgtRim = geometry.rimPointOnOffsetLine(
            tgtNode, srcPos.x - tgtPos.x, srcPos.y - tgtPos.y, offX, offY);
        e.style({
            'edge-distances': 'node-position',
            'source-endpoint': srcRim[0].toFixed(2) + ' ' + srcRim[1].toFixed(2),
            'target-endpoint': tgtRim[0].toFixed(2) + ' ' + tgtRim[1].toFixed(2),
            'source-distance-from-node': 0,
            'target-distance-from-node': 0,
        });
        // cytoscape must parse the manual endpoints back as px offset pairs
        const se = e.pstyle('source-endpoint');
        const te = e.pstyle('target-endpoint');
        check(id + ' source-endpoint parses as px pair',
            se && se.units && se.units[0] === 'px' && se.units[1] === 'px'
            && near(se.pfValue[0], srcRim[0], 0.01) && near(se.pfValue[1], srcRim[1], 0.01),
            JSON.stringify({ value: se && se.value, pfValue: se && se.pfValue }));
        check(id + ' target-endpoint parses as px pair',
            te && te.units && te.units[0] === 'px' && te.units[1] === 'px'
            && near(te.pfValue[0], tgtRim[0], 0.01) && near(te.pfValue[1], tgtRim[1], 0.01),
            JSON.stringify({ value: te && te.value, pfValue: te && te.pfValue }));
        check(id + ' distance-from-node pinned to 0',
            e.pstyle('source-distance-from-node').pfValue === 0
            && e.pstyle('target-distance-from-node').pfValue === 0);
    });

    cy.destroy();
}

if (failures > 0) {
    console.error('EDGE-ANCHOR HARNESS: ' + failures + ' FAILURE(S)');
    process.exit(1);
}
console.log('ALL EDGE-ANCHOR TESTS PASSED');
