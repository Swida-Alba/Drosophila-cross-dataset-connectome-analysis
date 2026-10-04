// Group color harness: extracts the REAL groupMembers / applyGroupColor /
// setEdgeBaseAppearance from a generated vispath network HTML and runs them
// against headless Cytoscape with minimal DOM stubs.
//
// Regression: applyGroupColor used to skip every currently-SELECTED member
// (`if (!node.selected())`), so the button looked dead after Assign /
// "Select group" (which leave the group selected) or recolored only the one
// unselected member. Membership also used a string-concatenated selector
// that threw on group names containing quotes.
// Usage: node group_color_harness.js <node-modules-dir> <path-to-network.html>
const cytoscape = require(process.argv[2] + '/node_modules/cytoscape');
const fs = require('fs');

const htmlPath = process.argv[3] || '/tmp/vispath-test/network_test.html';
const html = (() => {
    // Offline-embedded documents carry the vendored libraries inline;
    // keep only the app's script block — the one holding the elements JSON.
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

const fnSources = [
    'groupMembers', 'applyGroupColor', 'setEdgeBaseAppearance',
].map(f => extractFunction(f, html)).join('\n');

// NOTE: sources come from the project's own generated HTML (trusted,
// locally-produced artifact); new Function only executes that code against
// the headless core + stubs.
function buildScope(cy, domValues) {
    const prelude = `
        let selectedElement = null;
        const groupDefaults = {};
        const customGroups = {};
        const extraNodeGroups = [];
        const EDGE_BASE_COLOR_KEY = '__baseColor';
        const EDGE_BASE_OPACITY_KEY = '__baseOpacity';
        function pushHistory() {}
        function refreshLegend() {}
        const els = {};
        function makeEl(id) {
            const el = { id: id, value: domValues[id] !== undefined ? domValues[id] : '' };
            els[id] = el;
            return el;
        }
        const document = {
            getElementById: function (id) { return els[id] || makeEl(id); },
        };
    `;
    const src = prelude + fnSources + `
        return {
            applyGroupColor,
            getGroupDefaults: () => groupDefaults,
            getCustomGroups: () => customGroups,
            setGroupInput: (id, v) => { (els[id] || makeEl(id)).value = v; },
        };
    `;
    return new Function('cy', 'domValues', src)(cy, domValues);
}

function buildGraph() {
    const elements = [
        // three source members — the "Apply to Group" bug target
        { data: { id: 's1', label: 's1', node_type: 'source', assigned_group: 'source' } },
        { data: { id: 's2', label: 's2', node_type: 'source', assigned_group: 'source' } },
        { data: { id: 's3', label: 's3', node_type: 'source', assigned_group: 'source' } },
        { data: { id: 't1', label: 't1', node_type: 'target', assigned_group: 'target' } },
        // a member of a custom group whose name contains a QUOTE — the old
        // selector string threw a syntax error on exactly this
        { data: { id: 'c1', label: 'c1', node_type: 'intermediate', assigned_group: 'we"ird' } },
        { data: { id: 'u1', label: 'u1', node_type: 'intermediate', assigned_group: '' } },
        { group: 'edges', data: { id: 'eh', source: 's1', target: 't1', weight: 1, is_negative: 0 }, classes: 'highlighted' },
        { group: 'edges', data: { id: 'en', source: 's2', target: 't1', weight: 1, is_negative: 0 } },
    ];
    // styleEnabled: true so per-element style bypasses (the group recolor)
    // are actually applied and readable back.
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

// ===== Test A: selected members are recolored too (the core regression) =====
// headless Cytoscape reads styles back as 'rgb(r,g,b)' / numeric strings —
// normalize before comparing.
const hexOf = c => {
    const m = /^rgb\((\d+),\s*(\d+),\s*(\d+)\)$/.exec(String(c));
    if (!m) return c;
    return '#' + m.slice(1).map(v => (+v).toString(16).padStart(2, '0')).join('');
};
{
    const cy = buildGraph();
    const api = buildScope(cy, { groupSelector: 'source', groupColor: '#ff0000', groupOpacity: '40' });
    cy.getElementById('s1').select();
    cy.getElementById('s2').select();  // 2 of 3 members selected — old code skipped exactly these
    api.applyGroupColor();
    const bg = id => hexOf(cy.getElementById(id).style('background-color'));
    const op = id => Number(cy.getElementById(id).style('background-opacity'));
    check('selected s1 recolored', bg('s1'), '#ff0000');
    check('selected s2 recolored', bg('s2'), '#ff0000');
    check('unselected s3 recolored', bg('s3'), '#ff0000');
    check('opacity applied to all', [op('s1'), op('s2'), op('s3')], [0.4, 0.4, 0.4]);
    check('other group untouched', bg('t1') !== '#ff0000', true);
    check('group default recorded', api.getGroupDefaults().source, { color: '#ff0000', opacity: 40 });
    check('customized flag cleared', cy.getElementById('s1').data('customColor'), false);
}

// ===== Test B: edge groups recolor selected edges but keep highlights =====
{
    const cy = buildGraph();
    const api = buildScope(cy, { groupSelector: 'positive_edges', groupColor: '#00ff00', groupOpacity: '100' });
    cy.getElementById('en').select();  // selected plain edge — old code skipped it
    api.applyGroupColor();
    check('selected plain edge re-based', cy.getElementById('en').data('__baseColor'), '#00ff00');
    check('highlighted edge protected', cy.getElementById('eh').data('__baseColor'), undefined);
    check('edge default recorded', api.getGroupDefaults().positive_edges, { color: '#00ff00', opacity: 100 });
}

// ===== Test C: All Nodes recolors every member group, selection included =====
{
    const cy = buildGraph();
    const api = buildScope(cy, { groupSelector: 'all_nodes', groupColor: '#123456', groupOpacity: '80' });
    api.getCustomGroups()['we"ird'] = { label: 'we"ird', color: '#000000', opacity: 100 };
    cy.getElementById('s1').select();
    cy.getElementById('c1').select();
    api.applyGroupColor();
    const bg = id => hexOf(cy.getElementById(id).style('background-color'));
    check('source member recolored', bg('s1'), '#123456');
    check('target member recolored', bg('t1'), '#123456');
    check('unassigned recolored', bg('u1'), '#123456');
    check('quoted-name custom member recolored', bg('c1'), '#123456');
    check('no selector crash on quote name', true, true);
}

// ===== Test D: custom group (quoted name) recolors its members =====
{
    const cy = buildGraph();
    const api = buildScope(cy, { groupSelector: 'custom_we"ird', groupColor: '#0000ff', groupOpacity: '100' });
    api.getCustomGroups()['we"ird'] = { label: 'we"ird', color: '#000000', opacity: 100 };
    cy.getElementById('c1').select();
    api.applyGroupColor();
    check('quoted-name member recolored', hexOf(cy.getElementById('c1').style('background-color')), '#0000ff');
    check('custom group look recorded', api.getCustomGroups()['we"ird'].color, '#0000ff');
}

console.log(failures === 0 ? 'ALL GROUP-COLOR TESTS PASSED' : failures + ' GROUP-COLOR TEST(S) FAILED');
// process.exitCode alone would hang: styleEnabled cytoscape cores keep
// background timers alive, so exit explicitly with the proper code.
process.exit(failures === 0 ? 0 : 1);
