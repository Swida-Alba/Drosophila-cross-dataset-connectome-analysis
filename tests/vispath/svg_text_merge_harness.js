// SVG text-merge harness: extracts the REAL sanitizeSvgForOffice from a
// generated vispath network HTML and runs it under jsdom's DOMParser /
// XMLSerializer against fixtures that mirror cytoscape-svg@0.4.0's output
// shape (one sibling <text> per fillText call inside the label's own
// translate <g>). Asserts multi-line labels re-merge into one <text> with
// one <tspan x= y=> per line while every other text is left untouched.
// Usage: node svg_text_merge_harness.js <node-modules-dir> <path-to-network.html>
const fs = require('fs');

const htmlPath = process.argv[3];
const html = (() => {
    // Offline-embedded documents carry the vendored libraries inline;
    // keep only the app's shared-controls block (the one holding the
    // sanitizer).
    const whole = fs.readFileSync(htmlPath, 'utf8');
    const blocks = whole.match(/<script[^>]*>[\s\S]*?<\/script>/g) || [];
    const own = blocks.filter(b => b.includes('function sanitizeSvgForOffice'));
    if (own.length !== 1)
        throw new Error('shared-controls script block not found in ' + htmlPath);
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

const fnSource = extractFunction('sanitizeSvgForOffice', html);

// NOTE: the source comes from the project's own generated HTML (trusted,
// locally-produced artifact); new Function only executes that code.
function buildSanitizer(DOMParser, XMLSerializer) {
    return new Function('DOMParser', 'XMLSerializer',
        fnSource + '\nreturn sanitizeSvgForOffice;')(DOMParser, XMLSerializer);
}

// jsdom provides the DOMParser/XMLSerializer globals the sanitizer uses.
const dom = new (require(process.argv[2] + '/node_modules/jsdom').JSDOM)('');
const sanitizeSvgForOffice = buildSanitizer(dom.window.DOMParser, dom.window.XMLSerializer);

let failures = 0;
function check(name, got, expected) {
    const g = JSON.stringify(got);
    const e = JSON.stringify(expected);
    const pass = g === e;
    console.log((pass ? 'PASS' : 'FAIL') + ' | ' + name + ' | got=' + g + (pass ? '' : ' expected=' + e));
    if (!pass) failures++;
}

// Exact attribute set cytoscape-svg's __applyText emits for a fillText
// (plus the fill from the style state); one element per label line.
function labelText(line, x, y, overrides) {
    const attrs = Object.assign({
        'font-family': 'Helvetica',
        'font-size': '12px',
        'font-style': 'normal',
        'font-weight': 'normal',
        'text-decoration': 'none',
        'text-anchor': 'middle',
        'dominant-baseline': 'central',
        'fill': '#000000',
    }, overrides || {});
    const attrText = Object.keys(attrs).map(a => a + '="' + attrs[a] + '"').join(' ');
    return '<text ' + attrText + ' x="' + x + '" y="' + y + '">' + line + '</text>';
}

function parse(svgText) {
    return new dom.window.DOMParser().parseFromString(svgText, 'image/svg+xml');
}

function textInfo(t) {
    return {
        x: t.getAttribute('x'),
        y: t.getAttribute('y'),
        fill: t.getAttribute('fill'),
        fontSize: t.getAttribute('font-size'),
        anchor: t.getAttribute('text-anchor'),
        baseline: t.getAttribute('dominant-baseline'),
        tspans: Array.prototype.map.call(t.children, c => ({
            tag: c.tagName, x: c.getAttribute('x'), y: c.getAttribute('y'),
            text: c.textContent,
        })),
    };
}

// ===== A: two-line label merges into one <text> with two <tspan>s =====
{
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="800">' +
        '<g transform="translate(497.3, 352)">' +
        labelText('Dorsal lateral', 0, -8) + labelText('207', 0, 8) +
        '</g></svg>';
    const doc = parse(sanitizeSvgForOffice(svg));
    const texts = doc.querySelectorAll('text');
    check('A one text survives', texts.length, 1);
    const info = textInfo(texts[0]);
    check('A outer text drops x', info.x, null);
    check('A outer text drops y', info.y, null);
    check('A outer keeps fill/font/anchor', [info.fill, info.fontSize, info.anchor, info.baseline],
        ['#000000', '12px', 'middle', 'central']);
    check('A two tspans in reading order', info.tspans, [
        { tag: 'tspan', x: '0', y: '-8', text: 'Dorsal lateral' },
        { tag: 'tspan', x: '0', y: '8', text: '207' },
    ]);
    const g = texts[0].parentNode;
    // the global-grouping pass re-emits the label transform as an
    // equivalent matrix when hoisting the text into #texts
    check('A label transform preserved', g.getAttribute('transform'), 'matrix(1 0 0 1 497.3 352)');
}

// ===== B: single-line label untouched (x/y kept, no tspans) =====
{
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="800">' +
        '<g transform="translate(50, 60)">' + labelText('lonely', 0, 0) + '</g></svg>';
    const doc = parse(sanitizeSvgForOffice(svg));
    const texts = doc.querySelectorAll('text');
    check('B one text survives', texts.length, 1);
    check('B untouched', textInfo(texts[0]), {
        x: '0', y: '0', fill: '#000000', fontSize: '12px', anchor: 'middle',
        baseline: 'central', tspans: [],
    });
}

// ===== C: same-style texts in DIFFERENT parent groups not merged =====
{
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="800">' +
        '<g transform="translate(10, 20)">' + labelText('207', 0, 8) + '</g>' +
        '<g transform="translate(30, 40)">' + labelText('207', 0, 8) + '</g></svg>';
    const doc = parse(sanitizeSvgForOffice(svg));
    check('C two texts survive', doc.querySelectorAll('text').length, 2);
    check('C no tspans', doc.querySelectorAll('tspan').length, 0);
}

// ===== D: x differs beyond tolerance -> untouched =====
{
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="800">' +
        '<g transform="translate(10, 20)">' +
        labelText('left line', 0, -8) + labelText('shifted line', 10, 8) +
        '</g></svg>';
    const doc = parse(sanitizeSvgForOffice(svg));
    check('D two texts survive', doc.querySelectorAll('text').length, 2);
    check('D no tspans', doc.querySelectorAll('tspan').length, 0);
}

// ===== E: y delta out of the line-height band -> untouched =====
{
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="800">' +
        '<g transform="translate(10, 20)">' +
        labelText('far below', 0, -8) + labelText('other label', 0, 108) +
        '</g></svg>';
    const doc = parse(sanitizeSvgForOffice(svg));
    check('E far-apart texts survive', doc.querySelectorAll('text').length, 2);
    // identical y (a duplicated draw) is also outside the band
    const svg2 = '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="800">' +
        '<g transform="translate(10, 20)">' +
        labelText('twin one', 0, 8) + labelText('twin two', 0, 8) +
        '</g></svg>';
    const doc2 = parse(sanitizeSvgForOffice(svg2));
    check('E same-y texts survive', doc2.querySelectorAll('text').length, 2);
}

// ===== F: different fill -> untouched =====
{
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="800">' +
        '<g transform="translate(10, 20)">' +
        labelText('dark line', 0, -8, { fill: '#000000' }) +
        labelText('207', 0, 8, { fill: '#FF0000' }) +
        '</g></svg>';
    const doc = parse(sanitizeSvgForOffice(svg));
    check('F two texts survive', doc.querySelectorAll('text').length, 2);
    check('F no tspans', doc.querySelectorAll('tspan').length, 0);
}

// ===== G: non-text sibling between the lines breaks the run =====
{
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="800">' +
        '<g transform="translate(10, 20)">' +
        labelText('line one', 0, -8) +
        '<rect x="0" y="0" width="4" height="4" fill="#FFFFFF"/>' +
        labelText('line two', 0, 8) +
        '</g></svg>';
    const doc = parse(sanitizeSvgForOffice(svg));
    check('G two texts survive', doc.querySelectorAll('text').length, 2);
    check('G no tspans', doc.querySelectorAll('tspan').length, 0);
}

// ===== H: three-line label -> three tspans in order =====
{
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="800">' +
        '<g transform="translate(9, 8)">' +
        labelText('alpha', 0, -16) + labelText('beta', 0, 0) + labelText('gamma', 0, 16) +
        '</g></svg>';
    const doc = parse(sanitizeSvgForOffice(svg));
    const texts = doc.querySelectorAll('text');
    check('H one text survives', texts.length, 1);
    check('H three tspans in order', textInfo(texts[0]).tspans, [
        { tag: 'tspan', x: '0', y: '-16', text: 'alpha' },
        { tag: 'tspan', x: '0', y: '0', text: 'beta' },
        { tag: 'tspan', x: '0', y: '16', text: 'gamma' },
    ]);
}

// ===== I: full-document smoke — #texts holds one text per label =====
{
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="1000" height="800">' +
        '<defs></defs>' +
        '<g><rect x="0" y="0" width="1000" height="800" fill="#FFFFFF"/></g>' +
        '<g><path d="M 10 10 L 200 200 L 190 205 Z" fill="#CCCCCC" stroke="none"/></g>' +
        '<g transform="translate(100, 100)">' +
        labelText('BANC mapped', 0, -8) + labelText('207', 0, 8) +
        '</g>' +
        '<g transform="translate(300, 300)">' + labelText('flat', 0, 0) + '</g>' +
        '</svg>';
    const out = sanitizeSvgForOffice(svg);
    const doc = parse(out);
    check('I viewBox added', doc.documentElement.getAttribute('viewBox'), '0 0 1000 800');
    const textsG = doc.querySelector('#texts');
    check('I #texts exists', textsG !== null, true);
    const texts = textsG.querySelectorAll('text');
    check('I one text per label', texts.length, 2);
    const twoLine = Array.prototype.find.call(texts, t => t.querySelectorAll('tspan').length > 0);
    check('I two-line label merged', textInfo(twoLine).tspans, [
        { tag: 'tspan', x: '0', y: '-8', text: 'BANC mapped' },
        { tag: 'tspan', x: '0', y: '8', text: '207' },
    ]);
    const flat = Array.prototype.find.call(texts, t => t.querySelectorAll('tspan').length === 0);
    check('I single-line label keeps x/y', [flat.getAttribute('x'), flat.getAttribute('y')], ['0', '0']);
    const shapesG = doc.querySelector('#shapes');
    check('I #shapes keeps geometry', shapesG.querySelectorAll('rect, path').length, 2);
}

console.log(failures === 0 ? 'ALL SVG TEXT-MERGE TESTS PASSED' : failures + ' SVG TEXT-MERGE TEST(S) FAILED');
process.exit(failures === 0 ? 0 : 1);
