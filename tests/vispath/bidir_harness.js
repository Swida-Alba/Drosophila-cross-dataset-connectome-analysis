// Test harness: merged bidirectional edges + square-node vocabulary in a
// generated vispath network HTML.  Verifies against headless Cytoscape:
//   1. the stylesheet carries the edge[bidirectional = 1] source-arrow
//      selector and the node shape passthrough (when active);
//   2. the dead-end filter counts a merged edge on BOTH endpoints (the
//      directional counting would otherwise flag both ends as dead ends);
//   3. the edge-list CSV export expands a merged edge into two directional
//      rows sharing a bidirectional_pair id, each with its own weight.
// Usage: node bidir_harness.js <node-prefix> <path-to-network.html>
const cytoscape = require(process.argv[2] + '/node_modules/cytoscape');
const fs = require('fs');

const htmlPath = process.argv[3] || '/tmp/vispath-test/bidir_test.html';
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
            if (depth === 0) return source.slice(i, j + 1);
        }
    }
    throw new Error('unbalanced elements object');
}

// Extract the Cytoscape stylesheet array (bracket-balanced, quote-aware).
function extractStyleArray(source) {
    const marker = 'style: [';
    const start = source.indexOf(marker);
    if (start === -1) throw new Error('stylesheet array not found');
    let depth = 0, inStr = null;
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

// Minimal RFC-4180 CSV parser (quotes, doubled quotes, newlines in fields).
function parseCSV(text) {
    const rows = [];
    let row = [], field = '', inQuotes = false;
    for (let i = 0; i < text.length; i++) {
        const c = text[i];
        if (inQuotes) {
            if (c === '"') {
                if (text[i + 1] === '"') { field += '"'; i++; }
                else inQuotes = false;
            } else field += c;
        } else {
            if (c === '"') inQuotes = true;
            else if (c === ',') { row.push(field); field = ''; }
            else if (c === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
            else if (c !== '\r') field += c;
        }
    }
    if (field.length > 0 || row.length > 0) { row.push(field); rows.push(row); }
    return rows;
}

// NOTE: the extracted sources come from the project's own generated HTML
// artifact (written by vispath.py in this same repo), i.e. a trusted,
// locally-produced input — never user-supplied. new Function is used only
// to execute those extracted functions against a headless Cytoscape core.
function makeExporter(cy, customGroups) {
    const preamble = "const EDGE_BASE_COLOR_KEY = '__baseColor';\n" +
                     "const EDGE_BASE_OPACITY_KEY = '__baseOpacity';\n";
    const src = preamble +
        extractFunction('setEdgeBaseAppearance', html) + '\n' +
        extractFunction('initializeEdgeBaseStyles', html) + '\n' +
        extractFunction('extractColorHex', html) + '\n' +
        extractFunction('csvEscapeField', html) + '\n' +
        extractFunction('getNtGroupCSV', html) + '\n' +
        extractFunction('groupMembers', html) + '\n' +
        extractFunction('buildEdgeListCSV', html) + '\n';
    return new Function('cy', 'customGroups', src +
        '\nreturn { init: initializeEdgeBaseStyles, build: buildEdgeListCSV };')(cy, customGroups);
}

function makeDeadEndFn(cy) {
    const src = extractFunction('isDeadEndNodeIn', html) + '\n';
    return new Function('cy', src + '\nreturn isDeadEndNodeIn;')(cy);
}

// Runtime mode switcher: applyReciprocalMode extracted from the page with
// its page-global collaborators (edge-weight label, label refresh, filter
// re-apply) stubbed.  `reciprocalMode` is the page-global it mutates.
function makeModeFn(cy) {
    const src = "let reciprocalMode = 'straight';\n" +
        "const weightLabel = 'synapses';\n" +
        "const updateEdgeMetricLabels = () => {};\n" +
        "const applyEdgeFilter = () => {};\n" +
        "const document = { getElementById: () => null };\n" +
        extractFunction('applyReciprocalMode', html) + '\n';
    return new Function('cy', src + '\n' +
        'return { apply: applyReciprocalMode, mode: () => reciprocalMode };')(cy);
}

const elements = new Function('return (' + extractElementsObject(html) + ')')();
const styleArray = new Function('return ' + extractStyleArray(html))();

function buildGraph() {
    return cytoscape({
        headless: true,
        styleEnabled: true,
        elements: JSON.parse(JSON.stringify(elements)),
        style: styleArray
    });
}

let failures = 0;
function check(name, pass, detail) {
    console.log((pass ? 'PASS' : 'FAIL') + ' | ' + name + (pass ? '' : ' | ' + (detail || '')));
    if (!pass) failures++;
}

// --- Scenario 1: the generated stylesheet carries the bidirectional
// selector with BOTH arrowheads, and rendered nodes resolve the shape.
// (Skipped on plain merge-off generations: no selector is expected.) ---
{
    const hasBidir = elements.edges.some(e => e.data && e.data.bidirectional);
    if (hasBidir) {
        const bidirSel = styleArray.find(s => s && s.selector === 'edge[bidirectional = 1]');
        check('stylesheet has edge[bidirectional = 1] selector', !!bidirSel, 'selector missing');
        if (bidirSel) {
            check('bidirectional selector adds source-arrow-shape triangle',
                bidirSel.style['source-arrow-shape'] === 'triangle',
                JSON.stringify(bidirSel.style));
            check('bidirectional selector colors the source arrow from data',
                bidirSel.style['source-arrow-color'] === 'data(color)',
                JSON.stringify(bidirSel.style));
        }
    } else {
        console.log('SKIP | bidirectional scenarios (plain merge-off generation)');
    }
}

// --- Scenario 2: dead-end counting treats a merged edge as BOTH in and
// out for both endpoints (the directional reading flags both ends). ---
{
    const cy = buildGraph();
    // reproduce the page-load behavior: merge pairs whose initial mode is
    // merged (a no-op for straight-initial generations)
    makeModeFn(cy).apply('merged');
    const bidirEdges = cy.edges('[bidirectional = 1]');
    check('graph contains merged bidirectional edges when data has them',
        elements.edges.some(e => e.data && e.data.bidirectional)
            ? bidirEdges.length > 0 : true,
        'count=' + bidirEdges.length);
    if (bidirEdges.length > 0) {
        const isDeadEnd = makeDeadEndFn(cy);
        const e = bidirEdges[0];
        const u = e.source();
        const v = e.target();
        // Force both endpoints to intermediate so the role exemptions do
        // not mask a counting regression (sources/targets are exempt).
        u.data('node_type', 'intermediate');
        v.data('node_type', 'intermediate');
        const uDead = isDeadEnd(u, new Set());
        const vDead = isDeadEnd(v, new Set());
        check('merged edge: source endpoint not a dead end', !uDead, 'flagged');
        check('merged edge: target endpoint not a dead end', !vDead, 'flagged');
    }
}

// --- Scenario 3: CSV export expands a merged pair into two directional
// rows sharing the pair id; plain edges keep an empty pair cell. ---
{
    const cy = buildGraph();
    makeModeFn(cy).apply('merged');
    const exporter = makeExporter(cy, {});
    exporter.init();
    const rows = parseCSV(exporter.build());
    const header = rows[0];
    const pairCol = header.indexOf('bidirectional_pair');
    check('CSV header carries bidirectional_pair', pairCol === header.length - 1,
        'header=' + JSON.stringify(header));

    const bidirEdges = cy.edges('[bidirectional = 1]');
    if (bidirEdges.length > 0 && pairCol >= 0) {
        const e = bidirEdges[0];
        const uLabel = e.source().data('label') || e.source().id();
        const vLabel = e.target().data('label') || e.target().id();
        const expectedPair = uLabel + '<->' + vLabel;
        const fwdW = Number(e.data('weight_forward'));
        const revW = Number(e.data('weight_reverse'));
        const fwdRow = rows.find(r => r[0] === uLabel && r[1] === vLabel && r[pairCol] === expectedPair);
        const revRow = rows.find(r => r[0] === vLabel && r[1] === uLabel && r[pairCol] === expectedPair);
        check('forward-direction row exported with pair id', !!fwdRow,
            'missing ' + uLabel + '->' + vLabel + ' @' + expectedPair);
        check('reverse-direction row exported with pair id', !!revRow,
            'missing ' + vLabel + '->' + uLabel + ' @' + expectedPair);
        if (fwdRow) check('forward row carries its own weight', parseFloat(fwdRow[2]) === fwdW,
            fwdRow[2] + ' != ' + fwdW);
        if (revRow) check('reverse row carries its own weight', parseFloat(revRow[2]) === revW,
            revRow[2] + ' != ' + revW);
    }
    // Row count: hidden pair halves skipped, merged pairs two rows.
    const expectedRows = 1 + cy.edges().toArray().filter(e => !e.hasClass('pair-hidden'))
        .reduce((n, e) => n + (e.data('bidirectional') ? 2 : 1), 0);
    check('CSV row count matches plain+expanded edges', rows.length === expectedRows,
        'rows=' + rows.length + ' expected=' + expectedRows);

    // MODE SWITCH: straight un-hides the reverse halves, clears the
    // bidirectional flags, and both halves export with the shared pair id
    makeModeFn(cy).apply('straight');
    const rows2 = parseCSV(exporter.build());
    if (elements.edges.some(e => e.data && e.data.pair_id)) {
        const pairRows = rows2.filter(r => r[pairCol] && r[pairCol].includes('<->'));
        check('straight mode: both halves export with the shared pair id',
            pairRows.length === elements.edges.filter(e => e.data && e.data.pair_id).length,
            'pairRows=' + pairRows.length);
        check('straight mode: no bidirectional flags remain',
            cy.edges('[bidirectional = 1]').length === 0,
            'flags=' + cy.edges('[bidirectional = 1]').length);
        check('straight mode: hidden halves unhidden',
            cy.edges('.pair-hidden').length === 0,
            'hidden=' + cy.edges('.pair-hidden').length);
    }
}

// --- Scenario 4: node shape passthrough resolves on the rendered nodes
// (only present when the generation emitted shape keys). ---
{
    const cy = buildGraph();
    const hasShapeData = cy.nodes().filter(n => n.data('shape') !== undefined).length > 0;
    if (hasShapeData) {
        const shapeOk = cy.nodes().every(n => {
            const s = n.data('shape');
            return s === 'round-rectangle' || s === 'rectangle' || s === 'ellipse';
        });
        check('every node shape key uses a Cytoscape shape name', shapeOk,
            'unexpected shape values');
        const nodeSel = styleArray.find(s => s && s.selector === 'node');
        check('node selector uses the shape passthrough',
            nodeSel && nodeSel.style['shape'] === 'data(shape)',
            nodeSel ? JSON.stringify(nodeSel.style) : 'node selector missing');
    } else {
        console.log('SKIP | shape keys not emitted (circle generation)');
    }
}

console.log(failures === 0 ? 'ALL BIDIRECTIONAL-EDGE TESTS PASSED' : failures + ' BIDIRECTIONAL-EDGE TEST(S) FAILED');
// Force exit: style-enabled Cytoscape cores keep the event loop alive.
process.exit(failures === 0 ? 0 : 1);
