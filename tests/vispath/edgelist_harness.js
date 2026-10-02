// Test harness: extracts the edge-list CSV export functions from a
// generated vispath network HTML and runs them against headless Cytoscape.
// Verifies that the exported CSV matches the embedded input elements
// (source/target/weight/color/NT/grouping) and that in-HTML edits are
// reflected in the export.
// Usage: node edgelist_harness.js <node-prefix> <path-to-network.html>
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

// Extract the Cytoscape stylesheet array (same approach as the dead-end
// harness: bracket-balanced scan honoring string quotes).
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
        '\nreturn { init: initializeEdgeBaseStyles, build: buildEdgeListCSV, ntGroup: getNtGroupCSV };')(cy, customGroups);
}

// Independent oracle for the NT grouping (mirrors vispath's NT_GROUPS).
const NT_GROUP_ORACLE = {
    excitatory: ['acetylcholine', 'ach', 'glutamate', 'glut'],
    inhibitory: ['gaba'],
    modulatory: ['dopamine', 'da', 'serotonin', 'ser', '5-ht', 'octopamine', 'oct'],
    unknown: ['unknown', 'none', '']
};
function expectedNtGroup(nt) {
    if (nt === null || nt === undefined || nt === '') return nt === '' ? 'unknown' : 'unknown';
    const lower = String(nt).trim().toLowerCase();
    for (const group of Object.keys(NT_GROUP_ORACLE)) {
        if (NT_GROUP_ORACLE[group].indexOf(lower) !== -1) return group;
    }
    return 'unknown';
}

const elements = new Function('return (' + extractElementsObject(html) + ')')();
const styleArray = new Function('return ' + extractStyleArray(html))();

function buildGraph() {
    // Deep-copy: scenarios mutate their graph instance.
    return cytoscape({
        headless: true,
        styleEnabled: true,
        elements: JSON.parse(JSON.stringify(elements)),
        style: styleArray
    });
}

let failures = 0;
function check(name, pass, detail) {
    console.log((pass ? 'PASS' : 'FAIL') + ' | ' + name + (pass ? '' : ' | ' + detail));
    if (!pass) failures++;
}

// --- Scenario 1: untouched export matches the embedded input elements ---
const nodeInfo = {};
elements.nodes.forEach(n => { nodeInfo[n.data.id] = n.data; });

const cy1 = buildGraph();
const exporter1 = makeExporter(cy1, {});
exporter1.init();  // same seeding the page performs on load
const csvText = exporter1.build();
const rows = parseCSV(csvText);

// Dump for the pytest-side input/export comparison against conn_df.
console.log('EXPORTED_CSV_JSON::' + JSON.stringify(rows));

const HEADER = ['source', 'target', 'weight', 'color', 'nt_type', 'nt_group',
                'source_group', 'target_group', 'custom_groups', 'ratio', 'probability',
                'edge info', 'source info', 'target info', 'bidirectional_pair'];
check('CSV header matches the documented columns',
    JSON.stringify(rows[0]) === JSON.stringify(HEADER),
    'got ' + JSON.stringify(rows[0]));
check('row count equals embedded edge count',
    rows.length - 1 === elements.edges.length,
    'rows=' + (rows.length - 1) + ' edges=' + elements.edges.length);

let rowsMatchInput = true;
let rowDetail = '';
elements.edges.forEach((embedded, i) => {
    const row = rows[i + 1];
    const d = embedded.data;
    const srcInfo = nodeInfo[d.source] || {};
    const tgtInfo = nodeInfo[d.target] || {};
    const expectedWeight = (d.original_weight !== undefined && d.original_weight !== null)
        ? d.original_weight : d.weight;
    const expectedNt = d.nt_type || '';
    const expected = {
        source: srcInfo.label || d.source,
        target: tgtInfo.label || d.target,
        weight: expectedWeight,
        nt_type: expectedNt,
        nt_group: expectedNt ? expectedNtGroup(expectedNt) : '',
        source_group: srcInfo.node_type || 'intermediate',
        target_group: tgtInfo.node_type || 'intermediate',
        custom_groups: '',
        ratio: d.ratio ? d.ratio : '',
        probability: d.probability ? d.probability : ''
    };
    const problems = [];
    if (row[0] !== String(expected.source)) problems.push('source ' + row[0] + ' != ' + expected.source);
    if (row[1] !== String(expected.target)) problems.push('target ' + row[1] + ' != ' + expected.target);
    if (parseFloat(row[2]) !== parseFloat(expected.weight)) problems.push('weight ' + row[2] + ' != ' + expected.weight);
    if (!/^#[0-9a-f]{6}$/i.test(row[3])) problems.push('color not hex: ' + row[3]);
    if (row[4] !== expected.nt_type) problems.push('nt_type ' + row[4] + ' != ' + expected.nt_type);
    if (row[5] !== expected.nt_group) problems.push('nt_group ' + row[5] + ' != ' + expected.nt_group);
    if (row[6] !== expected.source_group) problems.push('source_group ' + row[6] + ' != ' + expected.source_group);
    if (row[7] !== expected.target_group) problems.push('target_group ' + row[7] + ' != ' + expected.target_group);
    if (row[8] !== '') problems.push('custom_groups not empty: ' + row[8]);
    if (expected.ratio === '' ? row[9] !== '' : Math.abs(parseFloat(row[9]) - parseFloat(expected.ratio)) > 1e-9) {
        problems.push('ratio ' + row[9] + ' != ' + expected.ratio);
    }
    if (expected.probability === '' ? row[10] !== '' : Math.abs(parseFloat(row[10]) - parseFloat(expected.probability)) > 1e-9) {
        problems.push('probability ' + row[10] + ' != ' + expected.probability);
    }
    // hover-info cells: {key:val; ...} JSON-like entries
    if (!/^(\{.*\})?$/.test(row[11])) problems.push('edge info not {key:val; ...}: ' + row[11]);
    if (!/^(\{.*\})?$/.test(row[12])) problems.push('source info not {key:val; ...}: ' + row[12]);
    if (!/^(\{.*\})?$/.test(row[13])) problems.push('target info not {key:val; ...}: ' + row[13]);
    if (row[11] !== '' && !/\{weight:[^;}]+/.test(row[11])) problems.push('edge info missing weight: ' + row[11]);
    // trailing bidirectional_pair cell: plain directed edges carry no pair
    if (row[14] !== '') problems.push('bidirectional_pair not empty for plain edge: ' + row[14]);
    if (problems.length > 0) {
        rowsMatchInput = false;
        rowDetail += 'edge ' + d.source + '->' + d.target + ': ' + problems.join('; ') + ' | ';
    }
});
check('every exported row matches the embedded input', rowsMatchInput, rowDetail);

// --- Scenario 2: NT group oracle table ---
{
    const cy = buildGraph();
    const exporter = makeExporter(cy, {});
    const cases = [
        ['acetylcholine', 'excitatory'], ['ACH', 'excitatory'],
        ['glutamate', 'inhibitory'], ['GLUT', 'inhibitory'], ['glut', 'inhibitory'],
        ['gaba', 'inhibitory'], ['GABA', 'inhibitory'],
        ['dopamine', 'modulatory'], ['serotonin', 'modulatory'], ['5-HT', 'modulatory'],
        ['octopamine', 'modulatory'], ['DA', 'modulatory'],
        ['unknown', 'unknown'], ['someNewNT', 'unknown'], [null, 'unknown']
    ];
    let ok = true, detail = '';
    for (const [input, want] of cases) {
        const got = exporter.ntGroup(input);
        if (got !== want) { ok = false; detail += input + '->' + got + ' (want ' + want + '); '; }
    }
    check('NT grouping mirrors the Python get_nt_group mapping', ok, detail);
}

// --- Scenario 3: in-HTML edits are reflected in the export ---
if (elements.edges.length > 0) {
    const cy = buildGraph();
    const customGroups = {};
    const exporter = makeExporter(cy, customGroups);
    exporter.init();

    const edges = cy.edges();
    const e0 = edges[0];
    const eLast = edges[edges.length - 1];
    const srcNode = e0.source();

    // Edit a weight (signed) the way editEdgeProperties does.
    e0.data('weight', 7.5);
    e0.data('original_weight', -7.5);
    e0.data('is_negative', 1);
    // Recolor the edge the way the color controls do (base appearance).
    e0.data('__baseColor', '#123456');
    // Relabel a node with comma + quotes to exercise CSV escaping, and
    // re-group another node the way the group-assignment flow does (single
    // assigned_group membership; node_type stays immutable).
    srcNode.data('label', 'S,R "renamed"');
    if (edges.length > 1) {
        eLast.source().data('assigned_group', 'target');
    }
    // Put the last edge's TARGET node into a custom group (membership is a
    // live node field; the CSV custom_groups column derives from endpoints).
    customGroups['MyGroup'] = { label: 'MyGroup', color: '#000000', opacity: 100 };
    eLast.target().data('assigned_group', 'MyGroup');

    const editedRows = parseCSV(exporter.build());
    const r0 = editedRows[1];
    check('edited signed weight is exported', parseFloat(r0[2]) === -7.5, 'got ' + r0[2]);
    check('edited color is exported', r0[3] === '#123456', 'got ' + r0[3]);
    check('relabeled node round-trips through CSV quoting',
        r0[0] === 'S,R "renamed"', 'got ' + r0[0]);
    const rLast = editedRows[editedRows.length - 1];
    check('re-grouped endpoint node is exported', rLast[6] === 'target', 'got ' + rLast[6]);
    check('custom group membership is exported', rLast[8] === 'MyGroup', 'got ' + rLast[8]);

    // Deleting an edge removes it from the export.
    const before = editedRows.length - 1;
    e0.remove();
    const afterRows = parseCSV(exporter.build());
    check('deleted edge disappears from the export',
        afterRows.length - 1 === before - 1,
        'before=' + before + ' after=' + (afterRows.length - 1));
}

// --- Scenario 4: an empty canvas exports only the header ---
{
    const cy = cytoscape({ headless: true, styleEnabled: true, elements: [], style: styleArray });
    const exporter = makeExporter(cy, {});
    const emptyRows = parseCSV(exporter.build());
    check('empty canvas exports header only',
        emptyRows.length === 1 && JSON.stringify(emptyRows[0]) === JSON.stringify(HEADER),
        JSON.stringify(emptyRows));
}

console.log(failures === 0 ? 'ALL EDGE-LIST EXPORT TESTS PASSED' : failures + ' EDGE-LIST EXPORT TEST(S) FAILED');
// Force exit: style-enabled Cytoscape cores keep the event loop alive.
process.exit(failures === 0 ? 0 : 1);
