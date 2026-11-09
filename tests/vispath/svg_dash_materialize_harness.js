// SVG dash-materialize harness: extracts the REAL sanitizeSvgForOffice from a
// generated vispath network HTML and runs it under jsdom's DOMParser /
// XMLSerializer against dashed/dotted shaft fixtures. Asserts the export
// converts stroke-dasharray shafts into real fill geometry (closed outline
// polygons / dots) with the canvas width-scaled density (dash=3w, gap=2w), so
// Office's editable conversion cannot re-map the dash density.
// Usage: node svg_dash_materialize_harness.js <node-modules-dir> <path-to-network.html>
const fs = require('fs');

const htmlPath = process.argv[3];
const html = (() => {
    const whole = fs.readFileSync(htmlPath, 'utf8');
    const blocks = whole.match(/<script[^>]*>[\s\S]*?<\/script>/g) || [];
    const own = blocks.filter(b => b.includes('function sanitizeSvgForOffice'));
    if (own.length !== 1)
        throw new Error('shared-controls script block not found in ' + htmlPath);
    return own[0];
})();

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
function buildSanitizer(DOMParser, XMLSerializer) {
    return new Function('DOMParser', 'XMLSerializer',
        fnSource + '\nreturn sanitizeSvgForOffice;')(DOMParser, XMLSerializer);
}
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
function parse(svgText) {
    return new dom.window.DOMParser().parseFromString(svgText, 'image/svg+xml');
}
function shaft(width, dasharray, d, color) {
    return '<path fill="none" stroke="' + (color || '#ffa500') + '" stroke-width="' + width +
        '" stroke-opacity="0.8" stroke-dasharray="' + dasharray + '" d="' + d + '"/>';
}
function dashPolys(doc, color) {
    return Array.prototype.filter.call(doc.querySelectorAll('path'), function (p) {
        return p.getAttribute('fill') === (color || '#ffa500') &&
            p.getAttribute('stroke') === 'none' &&
            /Z\s*$/.test(p.getAttribute('d') || '') &&
            ((p.getAttribute('d') || '').match(/-?\d+(?:\.\d+)?/g) || []).length === 8;
    });
}

// ===== A: width-4 dashed shaft -> 10 fill polygons (dash 12 / gap 8) =====
{
    const svg = '<svg xmlns="http://www.w3.org/2000/svg">' + shaft(4, '12,8', 'M 0 0 L 200 0') + '</svg>';
    const out = sanitizeSvgForOffice(svg);
    const doc = parse(out);
    check('A no stroke-dasharray survives', out.includes('stroke-dasharray'), false);
    check('A dash count = 200/(3w+2w) = 10', dashPolys(doc).length, 10);
}
// ===== B: width-8 dashed shaft -> 5 fill polygons (dash 24 / gap 16) =====
{
    const svg = '<svg xmlns="http://www.w3.org/2000/svg">' + shaft(8, '24,16', 'M 0 0 L 200 0') + '</svg>';
    const out = sanitizeSvgForOffice(svg);
    const doc = parse(out);
    check('B no stroke-dasharray survives', out.includes('stroke-dasharray'), false);
    check('B dash count scales with width = 5', dashPolys(doc).length, 5);
}
// ===== C: dotted shaft -> round dots (circles), no dasharray =====
{
    const svg = '<svg xmlns="http://www.w3.org/2000/svg">' + shaft(4, '0.01,6', 'M 0 0 L 200 0') + '</svg>';
    const out = sanitizeSvgForOffice(svg);
    const doc = parse(out);
    check('C no stroke-dasharray survives', out.includes('stroke-dasharray'), false);
    check('C dots emitted as circles', doc.querySelectorAll('circle').length > 0, true);
}
// ===== D: solid shaft untouched by the dash step (no dasharray introduced) =====
{
    const svg = '<svg xmlns="http://www.w3.org/2000/svg">' +
        '<path fill="none" stroke="#ffa500" stroke-width="4" d="M 0 0 L 200 0"/></svg>';
    const out = sanitizeSvgForOffice(svg);
    check('D solid stays dasharray-free', out.includes('stroke-dasharray'), false);
}

if (failures) {
    console.error(failures + ' dash-materialize checks failed');
    process.exit(1);
}
console.log('ALL SVG DASH-MATERIALIZE TESTS PASSED');