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

// Extract a top-level const arrow/value declaration with balanced braces
// or brackets (ANCHOR8_DIRS is a bare array literal).
function extractFunction(name, source) {
    const marker = 'const ' + name + ' = ';
    const start = source.indexOf(marker);
    if (start === -1) throw new Error('function not found: ' + name);
    let open = -1, openCh = null, closeCh = null;
    for (let i = start + marker.length; i < source.length; i++) {
        const c = source[i];
        if (c === '{' || c === '[') {
            open = i; openCh = c; closeCh = c === '{' ? '}' : ']'; break;
        }
    }
    let depth = 0;
    for (let i = open; i < source.length; i++) {
        if (source[i] === openCh) depth++;
        else if (source[i] === closeCh) {
            depth--;
            if (depth === 0) return source.slice(start, i + 1) + ';';
        }
    }
    throw new Error('unbalanced braces: ' + name);
}

// Evaluate the page's own geometry helpers with a stubbed `document`
// (they only read the node-size slider fallback from it).
const geometry = (() => {
    const src = ['nodeHalfSizes', 'rimDistance', 'rimPointOnOffsetLine',
                 'ANCHOR8_DIRS', 'snapAnchorIndex', 'anchorPoint', 'writeEdgeAnchors']
        .map(n => extractFunction(n, html)).join('\n')
        + '\nmodule.exports = { nodeHalfSizes, rimDistance, rimPointOnOffsetLine, ANCHOR8_DIRS, snapAnchorIndex, anchorPoint, writeEdgeAnchors };';
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

console.log('[4] snapAnchorIndex — 45° sectors, mirror symmetry, degenerate north');
{
    const E = 0, SE = 1, S = 2, SW = 3, W = 4, NW = 5, N = 6, NE = 7;
    check('east sector center', geometry.snapAnchorIndex(100, 0) === E,
        geometry.snapAnchorIndex(100, 0));
    // 22.5° is the sector boundary: 22° stays east, 23° falls to SE
    check('22° -> E', geometry.snapAnchorIndex(100, 40.4) === E,
        geometry.snapAnchorIndex(100, 40.4));
    check('23° -> SE', geometry.snapAnchorIndex(100, 42.4) === SE,
        geometry.snapAnchorIndex(100, 42.4));
    check('68° -> SE', geometry.snapAnchorIndex(42.4, 100) === SE,
        geometry.snapAnchorIndex(42.4, 100));
    check('68.2° -> S', geometry.snapAnchorIndex(40, 100) === S,
        geometry.snapAnchorIndex(40, 100));
    check('west across the ±π seam', geometry.snapAnchorIndex(-100, 1.3) === W,
        geometry.snapAnchorIndex(-100, 1.3));
    check('north', geometry.snapAnchorIndex(0, -100) === N,
        geometry.snapAnchorIndex(0, -100));
    // every anchor's opposite is +4 (mod 8) — the mirror rule behind the
    // auto snap (source = snap(d), target = snap(-d))
    let mirrorOK = true;
    for (let i = 0; i < 8; i++) {
        const [ux, uy] = geometry.ANCHOR8_DIRS[i];
        if (geometry.snapAnchorIndex(-ux, -uy) !== (i + 4) % 8) mirrorOK = false;
    }
    check('mirror rule snap(-d) = snap(d)+4 for all 8', mirrorOK);
    check('degenerate (0,0) -> N', geometry.snapAnchorIndex(0, 0) === N,
        geometry.snapAnchorIndex(0, 0));
    check('non-finite -> N', geometry.snapAnchorIndex(NaN, 1) === N,
        geometry.snapAnchorIndex(NaN, 1));
}

console.log('[5] anchorPoint — exact box corners, exact mids, ellipse 45° rim');
{
    const rect = mockNode(100, 60, 'round-rectangle');   // hw=50, hh=30
    let p = geometry.anchorPoint(rect, 1);   // SE
    check('box SE corner = (50, 30)', near(p[0], 50, 1e-6) && near(p[1], 30, 1e-6), p);
    p = geometry.anchorPoint(rect, 4);   // W
    check('box W mid = (-50, 0)', near(p[0], -50, 1e-6) && near(p[1], 0, 1e-6), p);
    p = geometry.anchorPoint(rect, 6);   // N
    check('box N mid = (0, -30)', near(p[0], 0, 1e-6) && near(p[1], -30, 1e-6), p);

    const circ = mockNode(100, 100, 'ellipse');   // r=50
    p = geometry.anchorPoint(circ, 1);   // 45° rim point
    check('circle SE anchor = (35.36, 35.36)',
        near(p[0], 50 / Math.SQRT2, 1e-6) && near(p[1], 50 / Math.SQRT2, 1e-6), p);
    p = geometry.anchorPoint(circ, 0);
    check('circle E mid = (50, 0)', near(p[0], 50, 1e-6) && near(p[1], 0, 1e-6), p);

    // aspect ellipse: the diagonal anchor rides the exact polar rim
    const tall = mockNode(60, 100, 'ellipse');   // hw=30, hh=50
    p = geometry.anchorPoint(tall, 3);   // SW
    check('ellipse SW anchor on the 45° diagonal',
        near(Math.abs(p[0]), Math.abs(p[1]), 1e-9) && p[0] < 0 && p[1] > 0, p);
    check('ellipse SW anchor on the rim',
        near((p[0] / 30) * (p[0] / 30) + (p[1] / 50) * (p[1] / 50), 1, 1e-6),
        (p[0] / 30) * (p[0] / 30) + (p[1] / 50) * (p[1] / 50));
    // out-of-range and negative indices wrap mod 8
    check('index wrap: 9 -> 1 (SE)', near(geometry.anchorPoint(rect, 9)[0], 50, 1e-6) &&
        near(geometry.anchorPoint(rect, 9)[1], 30, 1e-6), geometry.anchorPoint(rect, 9));
    check('index wrap: -4 -> 4 (W)', near(geometry.anchorPoint(rect, -4)[0], -50, 1e-6),
        geometry.anchorPoint(rect, -4));
}

console.log("[6] writeEdgeAnchors — auto snap mirrors, per-end override wins, parse-back");
{
    const cy2 = cytoscape({
        headless: true,
        styleEnabled: true,
        elements: [
            { data: { id: 'a', label: 'a' }, position: { x: 0, y: 0 } },
            { data: { id: 'b', label: 'b' }, position: { x: 120, y: 90 } },
            { data: { id: 'ab', source: 'a', target: 'b', weight: 3 } },
        ],
        style: [
            { selector: 'node', style: { width: 100, height: 100, shape: 'ellipse' } },
            { selector: 'edge', style: { 'curve-style': 'straight' } },
        ],
    });
    const e = cy2.getElementById('ab');
    // headless cytoscape ignores the element-definition `position` key —
    // set the seats explicitly (stage [3] hardcodes its deltas, so it
    // never noticed).
    cy2.getElementById('b').position({ x: 120, y: 90 });
    geometry.writeEdgeAnchors(e);
    // direction (120, 90) is 36.9° -> source snaps SE; the target end
    // mirrors to NW (216° sector)
    let se = e.pstyle('source-endpoint');
    let te = e.pstyle('target-endpoint');
    check('auto source = SE 45° point',
        near(se.pfValue[0], 50 / Math.SQRT2, 0.01) && near(se.pfValue[1], 50 / Math.SQRT2, 0.01),
        JSON.stringify({ v: se && se.pfValue }));
    check('auto target = NW mirrored point',
        near(te.pfValue[0], -50 / Math.SQRT2, 0.01) && near(te.pfValue[1], -50 / Math.SQRT2, 0.01),
        JSON.stringify({ v: te && te.pfValue }));
    check('distance-from-node pinned to 0',
        e.pstyle('source-distance-from-node').pfValue === 0 &&
        e.pstyle('target-distance-from-node').pfValue === 0);
    // a dragged override pins THAT end only; the other stays auto
    e.data('anchorSrc', 0);   // E mid on the source node
    geometry.writeEdgeAnchors(e);
    se = e.pstyle('source-endpoint');
    te = e.pstyle('target-endpoint');
    check('override pins source to E mid (50, 0)',
        near(se.pfValue[0], 50, 0.01) && near(se.pfValue[1], 0, 0.01),
        JSON.stringify({ v: se && se.pfValue }));
    check('un-overridden target stays auto NW',
        near(te.pfValue[0], -50 / Math.SQRT2, 0.01) && near(te.pfValue[1], -50 / Math.SQRT2, 0.01),
        JSON.stringify({ v: te && te.pfValue }));
    cy2.destroy();
}

console.log('[7] renderEdgeWaypoints — bend renders exactly at the stored waypoint');
{
    // renderEdgeWaypoints is a function DECLARATION (not a const arrow)
    const rwSrc = (() => {
        const marker = 'function renderEdgeWaypoints(';
        const start = html.indexOf(marker);
        if (start === -1) throw new Error('renderEdgeWaypoints not found');
        const open = html.indexOf('{', start);
        let depth = 0;
        for (let i = open; i < html.length; i++) {
            if (html[i] === '{') depth++;
            else if (html[i] === '}') { depth--; if (depth === 0) return html.slice(start, i + 1); }
        }
        throw new Error('unbalanced braces: renderEdgeWaypoints');
    })();
    const renderEdgeWaypoints = new Function(rwSrc + '; return renderEdgeWaypoints;')();

    const cy3 = cytoscape({
        headless: true,
        styleEnabled: true,
        elements: [
            { data: { id: 'a' }, position: { x: 0, y: 0 } },
            { data: { id: 'b' }, position: { x: 0, y: 0 } },
            { data: { id: 'ab', source: 'a', target: 'b', weight: 3 } },
        ],
        style: [
            { selector: 'node', style: { width: 100, height: 100, shape: 'ellipse' } },
            { selector: 'edge', style: { 'curve-style': 'straight' } },
        ],
    });
    cy3.getElementById('b').position({ x: 120, y: 90 });
    const e = cy3.getElementById('ab');
    // deliberately near one end of the line (t≈0.18) — the old basis's
    // worst drift zone; explicit anchor endpoints must not matter
    const wp = { x: 24, y: 12 };
    e.data('waypoints', [wp]);
    e.data('hasWaypoints', true);
    geometry.writeEdgeAnchors(e);
    renderEdgeWaypoints(e);
    check('edge-distances basis = node-position',
        e.pstyle('edge-distances').value === 'node-position',
        e.pstyle('edge-distances').value);
    const w = e.pstyle('segment-weights').pfValue[0];
    const d = e.pstyle('segment-distances').pfValue[0];
    // reconstruct the bend exactly the way cytoscape's findSegmentsPoints
    // does for edge-distances:'node-position' (midptPts = node positions,
    // normal = (-dy,dx)/len)
    const s = cy3.getElementById('a').position(), t = cy3.getElementById('b').position();
    const dx = t.x - s.x, dy = t.y - s.y, len = Math.hypot(dx, dy);
    const rx = s.x + w * dx + d * (-dy / len);
    const ry = s.y + w * dy + d * (dx / len);
    check('bend round-trips to the stored waypoint', near(rx, wp.x, 1e-6) && near(ry, wp.y, 1e-6),
        { rx: rx, ry: ry, wp: wp });
    // the old default basis ('intersection') spans the rim-to-rim chord,
    // re-parameterizing the same weight onto a shorter line: the bend sat
    // |rim·(2w−1)| off the stored point — at t≈0.18 on 50px rims, ~32px
    const rimA = geometry.rimDistance(mockNode(100, 100, 'ellipse'), dx, dy);
    const rimB = geometry.rimDistance(mockNode(100, 100, 'ellipse'), -dx, -dy);
    const oldPosAlong = rimA + w * (len - rimA - rimB);
    const drift = Math.abs(w * len - oldPosAlong);
    check('old rim-span basis would drift >5px at this weight', drift > 5, drift);
    cy3.destroy();
}

if (failures > 0) {
    console.error('EDGE-ANCHOR HARNESS: ' + failures + ' FAILURE(S)');
    process.exit(1);
}
console.log('ALL EDGE-ANCHOR TESTS PASSED');
