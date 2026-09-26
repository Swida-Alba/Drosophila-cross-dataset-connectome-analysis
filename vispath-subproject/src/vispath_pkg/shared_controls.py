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

/* Cytoscape export backend (network). Honors the current background so
   the exported image matches the visible canvas (PPT-safe). */
function exportCytoscapeToImage(cyObj, format, filename, scale, bg) {
    try {
        if (format === 'png') {
            const opts = { scale: scale, full: true };
            if (bg) { opts.bg = bg; }
            downloadDataUrl(cyObj.png(opts), filename);
        } else if (format === 'svg') {
            const opts = { full: true };
            if (bg) { opts.bg = bg; }
            const blob = new Blob([cyObj.svg(opts)], { type: 'image/svg+xml' });
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
