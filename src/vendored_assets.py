"""Vendored browser libraries inlined into generated HTML reports.

Plan: ``_plan/plan-offline-html-exports.md`` — every exported HTML works
without internet. Shared by all HTML exporters (comparison report,
conserved-path networks, profile charts, interactive heatmap, statvis);
the paths-pair report's ``_vis_network_script`` delegates here.

Two rules come from the paths-pair report rounds:

1. Inlining a library into a ``<script>`` block must neutralize ONLY the
   sequence ``</script`` (a blanket ``</`` escape corrupts JS regex
   literals — "Invalid regular expression flags").
2. plotly is NOT vendored as a static asset: it is read from the
   installed plotly.py's own ``package_data/plotly.min.js``, which is
   version-matched to the figure producer (the old CDN tags referenced
   older plotly.js builds — a latent skew this module removes).

Every helper degrades to the historical CDN tag when its source file is
missing, so a broken checkout degrades to the old online behavior instead
of failing.
"""

from __future__ import annotations

import re
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent / 'assets'
_CYTOSCAPE_SCRIPTS = (
    'cytoscape.min.js',
    'dagre.min.js',
    'cytoscape-dagre.js',
    'layout-base.js',
    'cose-base.js',
    'cytoscape-cose-bilkent.js',
    'cytoscape-fcose.js',
    'klay.js',
    'cytoscape-klay.js',
    'cytoscape-svg.js',
)


def _escape_inline_js(lib: str) -> str:
    """Make *lib* safe to inline in a ``<script>`` block.

    HTML terminates a script block on the sequence ``</script`` only, so
    exactly that sequence is neutralized (``<\\\\/script`` — an identity
    escape inside JS strings/regexes). A blanket ``</`` escape would
    corrupt regex literals.
    """
    return re.sub(r'</(script)', r'<\\/\1', lib, flags=re.IGNORECASE)


def _inline_file(path: Path, cdn_fallback: str) -> str:
    try:
        return '<script>' + _escape_inline_js(
            path.read_text(encoding='utf-8')) + '</script>'
    except OSError:
        return cdn_fallback


def inline_vis_network() -> str:
    """vis-network (vendored ``src/assets/vis-network.min.js``) as an
    inline ``<script>`` block; CDN tag when the asset is missing."""
    return _inline_file(
        _ASSETS / 'vis-network.min.js',
        '<script src="https://unpkg.com/vis-network/standalone/umd/'
        'vis-network.min.js"></script>')


def inline_plotly() -> str:
    """plotly.js from the INSTALLED plotly.py package (version-matched to
    the figure producer); CDN tag when plotly is not importable."""
    try:
        import plotly  # noqa: PLC0415 - imported lazily, see docstring
        lib = Path(plotly.__file__).parent / 'package_data' / 'plotly.min.js'
        return _inline_file(lib, '<script src="https://cdn.plot.ly/'
                                  'plotly-2.35.2.min.js"></script>')
    except ImportError:
        return ('<script src="https://cdn.plot.ly/'
                'plotly-2.35.2.min.js"></script>')


def inline_cytoscape() -> str:
    """The full cytoscape bundle (core + dagre/CoSE/fcose/klay layouts +
    SVG export) as inline ``<script>`` blocks; CDN tags when any asset is
    missing. Mirrors the exact script list of
    ``vispath_pkg.vispath._plot_cytoscape_network``."""
    cdn = (
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/'
        '3.28.1/cytoscape.min.js"></script>\n'
        '<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js">'
        '</script>\n'
        '<script src="https://unpkg.com/cytoscape-dagre@2.5.0/'
        'cytoscape-dagre.js"></script>\n'
        '<script src="https://unpkg.com/layout-base@1.0.2/layout-base.js">'
        '</script>\n'
        '<script src="https://unpkg.com/cose-base@1.0.3/cose-base.js">'
        '</script>\n'
        '<script src="https://unpkg.com/cytoscape-cose-bilkent@4.1.0/'
        'cytoscape-cose-bilkent.js"></script>\n'
        '<script src="https://unpkg.com/cytoscape-fcose@2.2.0/'
        'cytoscape-fcose.js"></script>\n'
        '<script src="https://unpkg.com/klayjs@0.4.1/klay.js"></script>\n'
        '<script src="https://unpkg.com/cytoscape-klay@3.1.4/'
        'cytoscape-klay.js"></script>\n'
        '<script src="https://unpkg.com/cytoscape-svg@0.4.0/'
        'cytoscape-svg.js"></script>'
    )
    parts = []
    for name in _CYTOSCAPE_SCRIPTS:
        path = _ASSETS / 'cytoscape' / name
        try:
            parts.append('<script>' + _escape_inline_js(
                path.read_text(encoding='utf-8')) + '</script>')
        except OSError:
            return cdn
    return '\n'.join(parts)


def shared_plotly_src(output_html: Path) -> str:
    """A RELATIVE ``<script src>`` for plotly shared per output directory
    (round-8c variant for exporters that emit many HTMLs per run, e.g.
    statvis matrix pages): writes ``plotly.min.js`` beside *output_html*
    once per directory and references it relatively. Inline instead when
    the package file is unreadable. Callers embed the returned tag in the
    ``<head>`` of the HTML they are about to write to *output_html*."""
    try:
        import plotly  # noqa: PLC0415 - imported lazily, see docstring
        lib = Path(plotly.__file__).parent / 'package_data' / 'plotly.min.js'
        target = Path(output_html).resolve().parent / 'plotly.min.js'
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(lib.read_bytes())
        return '<script src="plotly.min.js"></script>'
    except Exception:  # noqa: BLE001 - degrade to inline, then CDN
        return inline_plotly()
