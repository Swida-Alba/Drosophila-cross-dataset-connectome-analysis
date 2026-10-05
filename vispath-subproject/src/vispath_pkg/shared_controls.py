"""
Shared HTML/JS control backend for the vispath visualizations.

The network, Sankey, and heatmap HTML generators embed the same control
surface: PNG/SVG export, background toggle (White/Dark/Custom), status
display, and label toggling. This module is the single source of truth
for those shared blocks so the three templates cannot drift from each
other.

Everything here is a plain Python string (never an f-string), so the
blocks can be embedded into the templates' f-strings via a normal
placeholder (``{shared_controls.SHARED_JS}``) without any brace
escaping.
"""


def js_escape(value):
    """Escape a Python string for safe embedding inside a JS single-quoted
    string literal (and, transitively, inside HTML).

    Escapes backslash, quotes, newlines, and the HTML-significant
    characters so a data-derived title/filename can never break out of
    the string, the surrounding ``<script>`` block, or the HTML itself.
    """
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\\'")
    text = text.replace('"', '\\"')
    text = text.replace("\n", "\\n").replace("\r", "\\r")
    text = text.replace("<", "\\u003c").replace(">", "\\u003e")
    text = text.replace("&", "\\u0026")
    return text


def html_escape(value):
    """Escape a Python string for safe embedding as HTML text content."""
    if value is None:
        return ""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def json_safe(value, default=None):
    """json.dumps a value for embedding inside an inline ``<script>`` block.

    ``json.dumps`` escapes quotes and backslashes but NOT ``<``, ``>``, or
    ``&``, so a data label containing ``</script>`` would terminate the
    inline script and allow arbitrary HTML/JS injection. Replace those
    characters (plus the JS line separators U+2028/U+2029) with their
    ``\\uXXXX`` escapes, which JSON.parse decodes back identically.
    """
    import json as _json

    text = _json.dumps(value, default=default)
    return (
        text.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


# =====================================================================
# Shared JavaScript library
# =====================================================================
# Plain string on purpose: all braces are literal JS. Embedded into the
# templates via ``{SHARED_JS}`` inside their f-strings.
SHARED_JS = r"""
/* =====================================================================
   Shared vispath controls (vispath_pkg.shared_controls)
   Single source of truth for export / background / status helpers.
   ===================================================================== */

function isColorDark(color) {
    // Convert hex to RGB and calculate luminance
    let r, g, b;
    if (color.startsWith('#')) {
        const hex = color.slice(1);
        r = parseInt(hex.substr(0, 2), 16);
        g = parseInt(hex.substr(2, 2), 16);
        b = parseInt(hex.substr(4, 2), 16);
    } else {
        return false;
    }
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    return luminance < 0.5;
}

/* Escape user-provided labels before they are interpolated into HTML
   (innerHTML / Plotly hovertemplate). Data labels are untrusted. */
function escapeHtml(value) {
    if (value === null || value === undefined) { return ''; }
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/* Parse an export-scale input (number or range input). NaN-safe:
   falls back to `fallback`, and large values go through a confirm. */
function getExportScale(inputId, fallback, maxSafe) {
    const el = document.getElementById(inputId);
    let scale = el ? parseFloat(el.value) : NaN;
    if (isNaN(scale) || scale < 1) { scale = fallback; }
    if (scale > maxSafe) {
        const proceed = confirm(
            'Exporting at ' + scale + 'x may fail in your browser (very large image).\n\n' +
            'Click OK to attempt the requested ' + scale + 'x export, or Cancel to export at a safer ' + maxSafe + 'x.'
        );
        if (!proceed) { scale = maxSafe; }
    }
    return scale;
}

/* Download helpers: the anchor is appended to the DOM before the click
   (required by Safari) and the object URL is revoked after a delay so
   the download has time to start. */
function downloadDataUrl(dataUrl, filename) {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
}

/* Unified status display: type-aware color, XSS-safe textContent,
   auto-clears after 3 s. Types: success / info / warning / error. */
function showStatusInContainer(containerId, message, type) {
    const el = document.getElementById(containerId);
    if (!el) { return; }
    el.textContent = message;
    const colors = { success: '#2e7d32', info: '#666', warning: '#e65100', error: '#c62828' };
    el.style.color = colors[type] || colors.info;
    el.classList.add('status-message');
    el.classList.remove('status-success', 'status-info', 'status-warning', 'status-error');
    el.classList.add('status-' + (type || 'info'));
    setTimeout(function () { el.textContent = ''; }, 3000);
}

/* Background controller (presets, e.g. White / Dark). Applies through the
   per-visualization applyFn and remembers the current color so exports
   reproduce the visible background (PPT-safe). The toggle button cycles
   the presets and previews each with its own fill (contrast-aware text);
   the custom color box beside it is a separate always-visible control
   that applies immediately and never relabels the button. */
function createBackgroundController(colors, labels, applyFn) {
    let mode = 0;
    let currentColor = colors[0];
    function paintButton() {
        const btn = document.getElementById('bgToggleBtn');
        if (!btn) { return; }
        btn.style.background = currentColor;
        btn.style.color = isColorDark(currentColor) ? '#e5e7eb' : '#1f2937';
        btn.style.border = '1px solid #9ca3af';
    }
    paintButton();
    return {
        getColor: function () { return currentColor; },
        toggle: function (labelPrefix) {
            mode = (mode + 1) % colors.length;
            const btn = document.getElementById('bgToggleBtn');
            if (btn) { btn.textContent = (labelPrefix || '') + labels[mode]; }
            currentColor = colors[mode];
            paintButton();
            applyFn(colors[mode]);
        },
        applyCustom: function () {
            const picker = document.getElementById('customBgColor');
            if (!picker) { return; }
            currentColor = picker.value;
            applyFn(currentColor);
        },
        /* Restore a specific background color (layout persistence): preset
           colors re-select their mode so the next toggle continues from
           them; anything else is treated like a custom color (mode left
           alone, same contract as applyCustom). */
        apply: function (color) {
            if (color === undefined || color === null || color === '') { return; }
            const idx = colors.indexOf(color);
            if (idx !== -1) {
                mode = idx;
                const btn = document.getElementById('bgToggleBtn');
                if (btn) { btn.textContent = labels[mode]; }
            }
            currentColor = color;
            paintButton();
            applyFn(color);
        },
        reset: function (labelPrefix) {
            mode = 0;
            const btn = document.getElementById('bgToggleBtn');
            if (btn) { btn.textContent = (labelPrefix || '') + labels[0]; }
            currentColor = colors[0];
            paintButton();
            applyFn(colors[0]);
        }
    };
}

/* Plotly export backend (Sankey + heatmap).
   NOTE: width/height are pre-multiplied by scale here; do NOT also pass
   `scale` to Plotly.toImage - it multiplies the dimensions again,
   producing scale²-sized images. */
function exportPlotlyToImage(gd, format, filename, scale, width, height, onSuccess) {
    try {
        const opts = { format: format };
        if (scale > 1) {
            opts.width = Math.round(width * scale);
            opts.height = Math.round(height * scale);
        } else {
            opts.width = width;
            opts.height = height;
        }
        Plotly.toImage(gd, opts).then(function (dataUrl) {
            downloadDataUrl(dataUrl, filename);
            console.log(format.toUpperCase() + ' exported (' + (scale > 1 ? scale + 'x' : 'native size') + ').');
            if (typeof onSuccess === 'function') { onSuccess(); }
        }).catch(function (error) {
            console.error(format.toUpperCase() + ' export failed:', error);
            alert(format.toUpperCase() + ' export failed. Try lowering the scale (<=4). See console for details.');
        });
    } catch (err) {
        console.error(format.toUpperCase() + ' export failed (synchronous error):', err);
        alert(format.toUpperCase() + ' export failed. Try lowering the scale (<=4). See console for details.');
    }
}

/* Office-safe rewrite of the cytoscape-svg output. PowerPoint's
   "Convert to Shape" (and the Word/Outlook SVG importers) are the most
   conservative consumers the exported file ever meets, so the SVG is
   normalized into the subset they handle faithfully:

   1. Arrowhead underlays: cytoscape paints every translucent arrowhead
      over an opaque WHITE twin triangle (a canvas paint trick so the
      edge line cannot show through the translucent arrow). Once
      converted to shapes those twins are real geometry: one invisible
      white dart per arrow that surfaces as a white chunk cutting the
      edge on any non-white slide or after the shapes are recolored.
      Each (opaque backing, translucent top) twin pair with identical
      path data is collapsed into ONE opaque arrow whose fill is the
      exact composite — pixel-identical, no helper shapes.
   2. rgb() functional colors become #rrggbb; path numbers are
      re-emitted in fixed-point, rounded to 3 decimals (exponent forms
      and 15-digit floats are where lenient canvas consumers and strict
      shape importers diverge).
   3. paint-order is dropped (Office ignores it; document order already
      matches the canvas paint order) and the serializer's empty
      trailing restore <g> is removed.
   4. Multi-line labels: cytoscape draws a wrapped label as one fillText
      per line and the serializer emits one sibling <text> per call —
      PowerPoint converts each <text> into its own text box. Consecutive
      same-style lines (same anchored x, one line height apart in y) are
      re-merged into a single <text> holding one <tspan x= y=> per line,
      so Office imports one editable, movable text box per label.
   5. Edge fusion + stroke-to-geometry: every edge's shaft and arrowheads
      are wrapped in one group, the shaft's stroke becomes a filled
      outline polygon (never degenerate — the stroke width inflates both
      axes, which <line> boxes cannot do for near-vertical/horizontal
      shafts and PowerPoint then renders the stroke displaced from the
      correctly-positioned frame), and each arrowhead stays a simple
      single-subpath triangle (multi-subpath merged paths misplace their
      subpath origins in the converted frame). No <line>, no multi-subpath
      paths, nothing for the converter to misplace.
   6. A viewBox is added when missing so scaling stays unambiguous.
   7. Two global groups close it out: all geometry under id="shapes" and
      all node labels under id="texts" (PowerPoint converts each <g> to a
      group, so both sets stay selectable as units).
   Labels stay real <text> (they convert into editable text boxes). */
function sanitizeSvgForOffice(svgText) {
    let doc;
    try {
        doc = new DOMParser().parseFromString(svgText, 'image/svg+xml');
    } catch (err) { return svgText; }
    if (!doc || !doc.documentElement || doc.querySelector('parsererror')) { return svgText; }
    const root = doc.documentElement;

    function parseColor(value) {
        if (!value) { return null; }
        let m = value.match(/^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$/);
        if (m) { return [+m[1], +m[2], +m[3]]; }
        m = value.match(/^#([0-9a-f]{6})$/i);
        if (m) { return [parseInt(m[1].slice(0, 2), 16), parseInt(m[1].slice(2, 4), 16), parseInt(m[1].slice(4, 6), 16)]; }
        return null;
    }
    function toHex(rgb) {
        return '#' + rgb.map(function (c) {
            return Math.max(0, Math.min(255, Math.round(c))).toString(16).padStart(2, '0');
        }).join('');
    }

    // 1. Collapse (opaque backing, translucent top) twin pairs: the top
    //    fill becomes the exact composite over the backing color, the
    //    backing is removed. Only same-`d` adjacent pairs qualify, so a
    //    legitimately white-filled shape is never touched.
    const paths = Array.prototype.slice.call(root.querySelectorAll('path'));
    const removed = [];
    for (let i = 0; i < paths.length - 1; i++) {
        const backing = paths[i], top = paths[i + 1];
        if (!backing.parentNode || backing.getAttribute('d') !== top.getAttribute('d')) { continue; }
        const backFill = parseColor(backing.getAttribute('fill'));
        const topFill = parseColor(top.getAttribute('fill'));
        if (!backFill || !topFill) { continue; }
        const backOp = backing.getAttribute('fill-opacity') === null ? 1 : parseFloat(backing.getAttribute('fill-opacity'));
        const topOp = top.getAttribute('fill-opacity') === null ? 1 : parseFloat(top.getAttribute('fill-opacity'));
        if (!isFinite(backOp) || !isFinite(topOp) || backOp !== 1 || topOp >= 1) { continue; }
        const blended = topFill.map(function (c, k) { return backFill[k] * (1 - topOp) + c * topOp; });
        top.setAttribute('fill', toHex(blended));
        top.setAttribute('fill-opacity', '1');
        removed.push(backing);
    }
    removed.forEach(function (el) { el.parentNode.removeChild(el); });

    // 2. rgb() -> #rrggbb on paint attributes.
    ['fill', 'stroke', 'stop-color', 'flood-color'].forEach(function (attr) {
        root.querySelectorAll('[' + attr + ']').forEach(function (el) {
            const rgb = parseColor(el.getAttribute(attr));
            if (rgb) { el.setAttribute(attr, toHex(rgb)); }
        });
    });

    // 3. Fixed-point path data (3 decimals, no exponent forms).
    const numRe = /-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g;
    root.querySelectorAll('path').forEach(function (el) {
        const d = el.getAttribute('d');
        if (!d) { return; }
        el.setAttribute('d', d.replace(numRe, function (n) {
            const v = parseFloat(n);
            return (Math.round(v * 1000) / 1000).toString();
        }));
    });

    // 4. paint-order carries no information Office honors.
    root.querySelectorAll('[paint-order]').forEach(function (el) { el.removeAttribute('paint-order'); });

    // 5. Multi-line labels: cytoscape draws a wrapped label as one
    //    fillText per line and cytoscape-svg emits one sibling <text> per
    //    call, which PowerPoint converts into one text box PER LINE. The
    //    lines of one label are consecutive siblings inside the label's
    //    own translate <g>, share every style attribute and their
    //    anchored x, and sit one line height (≈ the font size) apart in
    //    y — exactly the merge criteria below. Each cluster becomes ONE
    //    <text> (first element's style, no x/y on the outer element)
    //    holding one <tspan x= y=> per line in reading order; Office
    //    imports that as a single editable text box. Everything else
    //    (single-line labels, other styles, non-consecutive elements) is
    //    left untouched.
    function mergeMultilineLabels() {
        function styleKey(el) {
            const parts = [];
            for (let i = 0; i < el.attributes.length; i++) {
                const a = el.attributes[i];
                if (a.name !== 'x' && a.name !== 'y') { parts.push(a.name + '=' + a.value); }
            }
            parts.sort();
            return parts.join('|');
        }
        function fontSizeOf(el) {
            const m = /^([\d.]+)(px)?$/.exec(el.getAttribute('font-size') || '');
            return m ? parseFloat(m[1]) : NaN;
        }
        const byParent = new Map();
        root.querySelectorAll('text').forEach(function (t) {
            const p = t.parentNode;
            if (!p) { return; }
            if (!byParent.has(p)) { byParent.set(p, []); }
            byParent.get(p).push(t);
        });
        byParent.forEach(function (texts) {
            let i = 0;
            while (i < texts.length) {
                const first = texts[i];
                const fs = fontSizeOf(first);
                const key = styleKey(first);
                const x0 = parseFloat(first.getAttribute('x'));
                const cluster = [first];
                let cursor = first;
                let yPrev = parseFloat(first.getAttribute('y'));
                for (;;) {
                    const next = cursor.nextElementSibling;
                    if (!next || next.tagName !== 'text') { break; }
                    if (styleKey(next) !== key) { break; }
                    if (!isFinite(x0) || Math.abs(parseFloat(next.getAttribute('x')) - x0) > 0.5) { break; }
                    const dy = parseFloat(next.getAttribute('y')) - yPrev;
                    if (!isFinite(fs) || !(dy >= 0.4 * fs && dy <= 3 * fs)) { break; }
                    cluster.push(next);
                    yPrev = parseFloat(next.getAttribute('y'));
                    cursor = next;
                }
                if (cluster.length >= 2) {
                    const merged = doc.createElementNS('http://www.w3.org/2000/svg', 'text');
                    for (let k = 0; k < first.attributes.length; k++) {
                        const a = first.attributes[k];
                        if (a.name !== 'x' && a.name !== 'y') { merged.setAttribute(a.name, a.value); }
                    }
                    cluster.forEach(function (line) {
                        const tspan = doc.createElementNS('http://www.w3.org/2000/svg', 'tspan');
                        tspan.setAttribute('x', line.getAttribute('x'));
                        tspan.setAttribute('y', line.getAttribute('y'));
                        while (line.firstChild) { tspan.appendChild(line.firstChild); }
                        merged.appendChild(tspan);
                    });
                    first.parentNode.insertBefore(merged, first);
                    cluster.forEach(function (line) { line.parentNode.removeChild(line); });
                    i += cluster.length;
                } else {
                    i += 1;
                }
            }
        });
    }
    mergeMultilineLabels();

    // 6. Empty groups (the serializer's trailing restore artifact).
    root.querySelectorAll('g').forEach(function (g) {
        if (!g.firstElementChild) { g.parentNode.removeChild(g); }
    });

    // 7. Edge fusion + stroke-to-geometry. PowerPoint's converter turns
    //    every path into its own shape and has two failure modes this
    //    export must not feed it: <line> elements whose extent is
    //    degenerate (a near-vertical shaft's box width rounds to ~0;
    //    PowerPoint clamps the box and then renders the stroke displaced
    //    from the shape's correct position), and multi-subpath merged
    //    paths (the rewritten subpath origins mismatch — content offset
    //    inside a correctly-positioned frame). Every edge therefore
    //    becomes ONE group of simple single-subpath closed polygons: the
    //    shaft's stroke is converted to a filled outline polygon
    //    (composited opaque over the background; never degenerate, the
    //    stroke width inflates both axes), and each arrowhead stays its
    //    own triangle. Rendering is unchanged.
    const allPaths = Array.prototype.slice.call(root.querySelectorAll('path'));
    const bgRect = root.querySelector('rect');
    const bgFill = (bgRect && bgRect.getAttribute('fill')) || '#ffffff';
    function vpIsArrowTri(p) {
        const d = p.getAttribute('d') || '';
        if (!/Z\s*$/.test(d)) { return false; }
        if (p.getAttribute('fill') === 'none' || p.getAttribute('fill') === null) { return false; }
        if (p.getAttribute('stroke') !== 'none') { return false; }
        return (d.match(/-?\d+(?:\.\d+)?/g) || []).length === 6;   // M + 2x L = 3 points
    }
    // Parse a stroked open shaft into a sample polyline. Handles the two
    // forms cytoscape-svg emits: "M x y L x y" and "M x y L|Q ... " paths
    // (quadratics are flattened). Returns null for anything else (dashed
    // shafts stay stroked — dashes cannot become fill geometry).
    function vpShaftSamples(p) {
        if (p.getAttribute('fill') !== 'none') { return null; }
        if (p.getAttribute('stroke-dasharray')) { return null; }
        const d = p.getAttribute('d') || '';
        const nums = (d.match(/-?\d+(?:\.\d+)?/g) || []).map(Number);
        if (nums.length < 4 || nums.length % 2) { return null; }
        const cmds = d.match(/[MLQ]/g) || [];
        if (cmds[0] !== 'M') { return null; }
        const pts = [{ x: nums[0], y: nums[1] }];
        let ci = 1, ni = 2, ok = true;
        for (let c = 1; c < cmds.length && ok; c++) {
            if (cmds[c] === 'L') {
                if (ni + 1 >= nums.length + 1) { ok = false; break; }
                pts.push({ x: nums[ni], y: nums[ni + 1] }); ni += 2;
            } else if (cmds[c] === 'Q') {
                const cx = nums[ni], cy2 = nums[ni + 1], ex = nums[ni + 2], ey = nums[ni + 3];
                ni += 4;
                const p0 = pts[pts.length - 1];
                for (let s = 1; s <= 12; s++) {
                    const u = s / 12, v = 1 - u;
                    pts.push({ x: v * v * p0.x + 2 * u * v * cx + u * u * ex,
                               y: v * v * p0.y + 2 * u * v * cy2 + u * u * ey });
                }
            } else { ok = false; }
            void ci;
        }
        if (!ok || pts.length < 2) { return null; }
        const w = parseFloat(p.getAttribute('stroke-width'));
        return { pts: pts, width: isFinite(w) && w > 0 ? w : 1,
                 color: p.getAttribute('stroke'),
                 opacity: p.getAttribute('stroke-opacity') === null ? 1 : parseFloat(p.getAttribute('stroke-opacity')) };
    }
    function vpR3(v) { return (Math.round(v * 1000) / 1000).toString(); }
    // Closed outline polygon of the sampled shaft, inflated by w/2 on
    // both sides (interior vertices use averaged segment normals).
    function vpOutlinePoly(info) {
        const pts = info.pts, hw = info.width / 2;
        const left = [], right = [];
        for (let i = 0; i < pts.length; i++) {
            const prev = pts[Math.max(0, i - 1)], next = pts[Math.min(pts.length - 1, i + 1)];
            let sx = next.x - prev.x, sy = next.y - prev.y;
            const sl = Math.hypot(sx, sy) || 1;
            sx /= sl; sy /= sl;
            left.push({ x: pts[i].x + (-sy) * hw, y: pts[i].y + sx * hw });
            right.push({ x: pts[i].x - (-sy) * hw, y: pts[i].y - sx * hw });
        }
        const ring = left.concat(right.reverse());
        return 'M' + ring.map(function (q) { return vpR3(q.x) + ' ' + vpR3(q.y); }).join(' L ') + ' Z';
    }
    let vi = 0;
    while (vi < allPaths.length) {
        const p = allPaths[vi];
        const isShaft = p.parentNode && p.getAttribute('fill') === 'none';
        const shaft = isShaft ? vpShaftSamples(p) : null;
        const arrows = [];
        let vj = vi + 1;
        while (vj < allPaths.length && arrows.length < 2 && vpIsArrowTri(allPaths[vj])) {
            arrows.push(allPaths[vj]); vj++;
        }
        if (shaft && arrows.length > 0) {
            const g = doc.createElementNS('http://www.w3.org/2000/svg', 'g');
            g.setAttribute('data-vp-edge', '1');   // the global-grouping pass keeps this group indivisible
            p.parentNode.insertBefore(g, p);
            // shaft: stroke converted to a filled outline (base = background)
            const base = parseColor(shaft.color) || parseColor(bgFill) || [0, 0, 0];
            const bgc = parseColor(bgFill) || [255, 255, 255];
            const op = isFinite(shaft.opacity) ? shaft.opacity : 1;
            const blended = base.map(function (c, k) { return bgc[k] * (1 - op) + c * op; });
            const outline = doc.createElementNS('http://www.w3.org/2000/svg', 'path');
            outline.setAttribute('fill', toHex(blended));
            outline.setAttribute('fill-opacity', '1');
            outline.setAttribute('stroke', 'none');
            outline.setAttribute('d', vpOutlinePoly(shaft));
            g.appendChild(outline);
            p.parentNode.removeChild(p);
            arrows.forEach(function (aa) { g.appendChild(aa); });   // own triangles, single-subpath
            vi = vj;
        } else {
            vi++;
        }
    }

    // 8. viewBox from width/height when missing.
    if (!root.getAttribute('viewBox') && root.getAttribute('width') && root.getAttribute('height')) {
        const w = parseFloat(root.getAttribute('width'));
        const h = parseFloat(root.getAttribute('height'));
        if (isFinite(w) && isFinite(h)) { root.setAttribute('viewBox', '0 0 ' + w + ' ' + h); }
    }

    // 9. Two global groups: all geometry under id="shapes", all node
    //    labels under id="texts". The serializer nests save/restore groups
    //    whose translate transforms vary per draw phase (nodes can even
    //    land outside the outer wrapper), so elements are collected with
    //    their CUMULATIVE transform and each distinct context becomes a
    //    transform-carrying subgroup inside its global group — rendering
    //    is unchanged. PowerPoint maps <g> to groups: the converted slide
    //    carries every shape in one group and every text box in another.
    const SVGNS = 'http://www.w3.org/2000/svg';
    const IDENT = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };
    function vpParseTf(t) {
        if (!t) { return { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 }; }
        let m = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };
        const re = /(translate|scale|matrix)\s*\(([^)]*)\)/g;
        let hit;
        while ((hit = re.exec(t)) !== null) {
            const args = hit[2].split(/[\s,]+/).map(Number);
            const n = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };
            if (hit[1] === 'translate') { n.e = args[0] || 0; n.f = args[1] || 0; }
            else if (hit[1] === 'scale') { n.a = args[0]; n.d = args.length > 1 ? args[1] : args[0]; }
            else if (hit[1] === 'matrix') { n.a = args[0]; n.b = args[1]; n.c = args[2]; n.d = args[3]; n.e = args[4]; n.f = args[5]; }
            m = {
                a: m.a * n.a + m.c * n.b, b: m.b * n.a + m.d * n.b,
                c: m.a * n.c + m.c * n.d, d: m.b * n.c + m.d * n.d,
                e: m.a * n.e + m.c * n.f + m.e, f: m.b * n.e + m.d * n.f + m.f
            };
        }
        return m;
    }
    function vpTfKey(m) {
        return [m.a, m.b, m.c, m.d, m.e, m.f].map(function (v) { return Math.round(v * 1e6) / 1e6; }).join(',');
    }
    function vpTfAttr(m) {
        const idn = m.a === 1 && m.b === 0 && m.c === 0 && m.d === 1 && m.e === 0 && m.f === 0;
        return idn ? null : 'matrix(' + [m.a, m.b, m.c, m.d, m.e, m.f].map(function (v) { return Math.round(v * 1e6) / 1e6; }).join(' ') + ')';
    }
    const buckets = new Map();   // tfKey -> { tf, shapes: [], texts: [] }
    function vpBucket(tf) {
        const key = vpTfKey(tf);
        if (!buckets.has(key)) { buckets.set(key, { tf: tf, shapes: [], texts: [] }); }
        return buckets.get(key);
    }
    (function vpCollect(el, tf) {
        Array.prototype.slice.call(el.children).forEach(function (k) {
            if (k.tagName === 'g') {
                if (k.getAttribute('data-vp-edge')) { vpBucket(tf).shapes.push(k); return; }
                vpCollect(k, (function () {
                    const outer = vpParseTf(k.getAttribute('transform'));
                    return {
                        a: tf.a * outer.a + tf.c * outer.b, b: tf.b * outer.a + tf.d * outer.b,
                        c: tf.a * outer.c + tf.c * outer.d, d: tf.b * outer.c + tf.d * outer.d,
                        e: tf.a * outer.e + tf.c * outer.f + tf.e,
                        f: tf.b * outer.e + tf.d * outer.f + tf.f
                    };
                })());
                return;
            }
            if (k.tagName === 'defs') { return; }
            (k.tagName === 'text' ? vpBucket(tf).texts : vpBucket(tf).shapes).push(k);
        });
    })(root, IDENT);
    if (buckets.size > 0) {
        const shapeG = doc.createElementNS(SVGNS, 'g');
        shapeG.setAttribute('id', 'shapes');
        const textG = doc.createElementNS(SVGNS, 'g');
        textG.setAttribute('id', 'texts');
        buckets.forEach(function (bucket) {
            const tAttr = vpTfAttr(bucket.tf);
            [['shapes', shapeG, bucket.shapes], ['texts', textG, bucket.texts]].forEach(function (pair) {
                const items = pair[2];
                if (!items.length) { return; }
                if (!tAttr) { items.forEach(function (el) { pair[1].appendChild(el); }); return; }
                const sub = doc.createElementNS(SVGNS, 'g');
                sub.setAttribute('transform', tAttr);
                items.forEach(function (el) { sub.appendChild(el); });
                pair[1].appendChild(sub);
            });
        });
        Array.prototype.slice.call(root.children).forEach(function (k) {
            if (k !== shapeG && k !== textG) { root.removeChild(k); }
        });
        root.appendChild(shapeG);
        if (textG.children.length) { root.appendChild(textG); }
    }

    return new XMLSerializer().serializeToString(root);
}

/* Cytoscape export backend (network). Honors the current background so
   the exported image matches the visible canvas, then normalizes the
   SVG for Office/PPT shape conversion. */
function exportCytoscapeToImage(cyObj, format, filename, scale, bg) {
    try {
        if (format === 'png') {
            const opts = { scale: scale, full: true };
            if (bg) { opts.bg = bg; }
            downloadDataUrl(cyObj.png(opts), filename);
        } else if (format === 'svg') {
            const opts = { full: true };
            if (bg) { opts.bg = bg; }
            const blob = new Blob([sanitizeSvgForOffice(cyObj.svg(opts))], { type: 'image/svg+xml' });
            downloadBlob(blob, filename);
        }
        console.log(format.toUpperCase() + ' exported (' + (format === 'png' ? scale + 'x' : 'native size') + ').');
    } catch (err) {
        console.error(format.toUpperCase() + ' export failed:', err);
        alert(format.toUpperCase() + ' export failed. Try lowering the scale (<=4) or exporting ' + (format === 'png' ? 'SVG' : 'PNG') + '. See console for details.');
    }
}

/* Generic localStorage save/load primitives (heatmap settings). */
function saveObjectToStorage(key, obj) {
    try {
        localStorage.setItem(key, JSON.stringify(obj));
        return true;
    } catch (err) {
        console.error('Save failed:', err);
        return false;
    }
}

function loadObjectFromStorage(key) {
    try {
        const raw = localStorage.getItem(key);
        return raw ? JSON.parse(raw) : null;
    } catch (err) {
        console.error('Load failed:', err);
        return null;
    }
}
"""
