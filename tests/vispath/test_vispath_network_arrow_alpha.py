"""Translucent arrowheads composite in the canvas and in the SVG export.

Cytoscape 3.28.1's cut-out erased whatever sat under a below-full-opacity
arrowhead before painting it, so two coincident translucent arrows kept the
single-arrow value and the arrow could not blend with its own shaft.  The
vendored bundle is patched only for the cut-out (hollow arrows keep it), and
the Office sanitizer must not flatten the stroke/arrow alpha back to opaque.

The pixel expectations below are source-over composites over white.  They
deliberately include the shaft/arrow overlap: an edge body and its arrowhead
both carry real opacity and blend where they intersect, so an arrow interior
crossed by its own line is darker than the arrow alone.

Run:
    python -m pytest tests/vispath/test_vispath_network_arrow_alpha.py -q
"""

import base64
import io
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/vispath

from heatmap_cdp import Page, REQUIRES_CDP  # noqa: E402
from vispath_pkg.vispath import VisualizePath  # noqa: E402

NS = "{http://www.w3.org/2000/svg}"
MARKER = "DROCAT patch: arrow fill blends; hollow keeps cut-out"


def _inline(path):
    text = Path(path).read_text(encoding="utf-8")
    return re.sub(r"</(script)", r"<\\/\1", text, flags=re.IGNORECASE)


def _fixture_html():
    bundle = _inline(PROJECT_ROOT / "src" / "assets" / "cytoscape" / "cytoscape.min.js")
    svg_ext = _inline(PROJECT_ROOT / "src" / "assets" / "cytoscape" / "cytoscape-svg.js")
    return """<!doctype html><html><head><meta charset="utf-8">
<style>html,body{margin:0;background:#fff}#cy{width:420px;height:600px}</style>
<script>__BUNDLE__</script>
<script>__SVGEXT__</script>
</head><body><div id="cy"></div>
<script>
const cy = cytoscape({
  container: document.getElementById('cy'),
  style: [
    {selector: 'node',
     style: {'width': 1, 'height': 1, 'background-color': '#fff',
             'label': ''}},
    {selector: 'edge',
     style: {'width': 6, 'curve-style': 'straight',
             'target-arrow-shape': 'triangle', 'arrow-scale': 2,
             'line-color': '#000', 'target-arrow-color': '#000',
             'line-opacity': 0.4}}
  ],
  elements: [
    {data: {id: 'A1'}, position: {x: 70, y: 80}},
    {data: {id: 'B1'}, position: {x: 350, y: 80}},
    {data: {id: 'A2'}, position: {x: 70, y: 180}},
    {data: {id: 'B2'}, position: {x: 350, y: 180}},
    {data: {id: 'A3'}, position: {x: 70, y: 280}},
    {data: {id: 'B3'}, position: {x: 350, y: 280}},
    {data: {id: 'A4'}, position: {x: 70, y: 380}},
    {data: {id: 'B4'}, position: {x: 350, y: 380}},
    {data: {id: 'A5'}, position: {x: 70, y: 480}},
    {data: {id: 'B5'}, position: {x: 350, y: 480}},
    {data: {id: 'e1', source: 'A1', target: 'B1'}},
    {data: {id: 'e2a', source: 'A2', target: 'B2'}},
    {data: {id: 'e2b', source: 'A2', target: 'B2'}},
    {data: {id: 'e3a', source: 'A3', target: 'B3'}},
    {data: {id: 'e3b', source: 'A3', target: 'B3'}},
    {data: {id: 'e4', source: 'A4', target: 'B4'}},
    {data: {id: 'e5', source: 'A5', target: 'B5'}}
  ],
  layout: {name: 'preset'}, autoungrabify: true
});
cy.ready(function () {
  // Row 280: red drawn over black, both coincident.
  cy.edges('#e3b').style({'line-color': '#ff0000',
                          'target-arrow-color': '#ff0000'});
  // Row 380: hollow keeps the renderer's cut-out.
  cy.edges('#e4').style({'target-arrow-fill': 'hollow'});
  // Row 480: fully opaque stays fully opaque.
  cy.edges('#e5').style({'line-opacity': 1});
  cy.zoom(1);
  cy.pan({x: 0, y: 0});
  cy.forceRender();
  window.__arrowFixtureReady = true;
});
</script></body></html>""".replace(
        "__BUNDLE__", bundle
    ).replace(
        "__SVGEXT__", svg_ext
    )


def _close(actual, expected, tol=3):
    return all(abs(int(a) - int(e)) <= tol for a, e in zip(actual, expected))


def _png_rgb(browser):
    png = browser.js("return cy.png({output: 'base64', bg: 'white', scale: 2});")
    assert isinstance(png, str) and len(png) > 1000, png
    return Image.open(io.BytesIO(base64.b64decode(png))).convert("RGB")


def _rgb(img, x, y):
    return img.getpixel((2 * x, 2 * y))


# =============================================================================
# Canvas: true alpha on arrows, and body/arrow compositing
# =============================================================================


@REQUIRES_CDP
def test_canvas_arrowheads_blend_with_each_other_and_their_shaft(tmp_path):
    page = tmp_path / "arrow_alpha.html"
    page.write_text(_fixture_html(), encoding="utf-8")
    with Page(page, width=520, height=700) as browser:
        assert browser.wait("return typeof cy !== 'undefined' && cy.edges().length === 7;")
        assert browser.wait("return !!window.__arrowFixtureReady;")
        img = _png_rgb(browser)
        assert img.size == (840, 1200), img.size

        # (line midpoint, arrow interior over its own shaft, arrow interior
        #  clear of the shaft)
        cases = {
            "single": (80, (153, 153, 153), (92, 92, 92), (153, 153, 153)),
            "two coincident": (180, (92, 92, 92), (33, 33, 33), (92, 92, 92)),
            "red over black": (280, (194, 92, 92), (196, 33, 33), (194, 92, 92)),
        }
        for label, (y, line, overlap, arrow) in cases.items():
            assert _close(_rgb(img, 210, y), line), (label, "line", _rgb(img, 210, y))
            assert _close(_rgb(img, 321, y), overlap), (
                label,
                "shaft+arrow overlap",
                _rgb(img, 321, y),
            )
            assert _close(_rgb(img, 330, y - 8), arrow), (
                label,
                "arrow only",
                _rgb(img, 330, y - 8),
            )

        # Hollow: the cut-out is retained, so the interior stays white even
        # where the shaft passes under the arrowhead.
        assert _close(_rgb(img, 210, 380), (153, 153, 153))
        assert _close(_rgb(img, 321, 380), (255, 255, 255))
        assert _close(_rgb(img, 330, 372), (255, 255, 255))

        # Opaque pages are unchanged.
        for x, y in ((210, 480), (321, 480), (330, 472)):
            assert _close(_rgb(img, x, y), (0, 0, 0)), (x, y, _rgb(img, x, y))
        assert browser.errors() == [], browser.errors()


# =============================================================================
# VisPath page -> exported SVG: real alpha survives the Office sanitizer
# =============================================================================


def _render(tmp_path, frame, name, **kwargs):
    vp = VisualizePath(
        path_file=frame,
        output_folder=str(tmp_path),
        showfig=False,
        verbose=False,
        network_layout="dagre",
        **kwargs,
    )
    vp.base_filename = name
    vp.build_network()
    page = Path(vp.create_network())
    return page, page.read_text(encoding="utf-8")


def _translate(root):
    for el in root.iter():
        t = el.get("transform") or ""
        m = re.match(r"matrix\(1 0 0 1 (-?[\d.]+) (-?[\d.]+)\)", t)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = re.match(r"translate\((-?[\d.]+)[ ,](-?[\d.]+)\)", t)
        if m:
            return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0


_SVG_EXPORT_JS = """
  var svg = %s;
  var points = %s;
  var tx = %s, ty = %s;
  window.__exportSamples = null;
  var img = new Image();
  img.onload = function () {
    var c = document.createElement('canvas');
    c.width = 1000; c.height = 800;
    var ctx = c.getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight);
    window.__exportSamples = points.map(function (p) {
      var d = ctx.getImageData(Math.round(p[0] + tx),
                               Math.round(p[1] + ty), 1, 1).data;
      return [p[0], p[1], d[0], d[1], d[2]];
    });
  };
  img.onerror = function () { window.__exportSamples = 'load-error'; };
  img.src = URL.createObjectURL(new Blob([svg], {type: 'image/svg+xml'}));
  return true;
"""


@REQUIRES_CDP
def test_vispath_export_keeps_true_body_and_arrow_alpha(tmp_path):
    frame = pd.DataFrame(
        {
            "source": ["A1", "A2"],
            "target": ["B1", "B2"],
            "weight": [10.0, 10.0],
            "edge_alpha": [0.4, 0.4],
            "color": ["#000000", "#ff0000"],
            "arrow_color": ["#000000", "#ff0000"],
        }
    )
    page, html = _render(tmp_path, frame, "alpha")
    assert MARKER in html, "page did not inline the patched bundle"

    with Page(page, width=1200, height=900) as browser:
        assert browser.wait("return typeof cy !== 'undefined' && cy.edges().length === 2;")
        states = browser.js(
            """
          return cy.edges().map(function (e) {
            return [e.data('source'), e.data('target'),
                    parseFloat(e.style('line-opacity')),
                    e.style('line-color'), e.style('target-arrow-color')];
          });
        """
        )
        assert states == [
            ["A1", "B1", 0.4, "rgb(0,0,0)", "rgb(0,0,0)"],
            ["A2", "B2", 0.4, "rgb(255,0,0)", "rgb(255,0,0)"],
        ], states

        # Coincident endpoints + a 6 px shaft, so the shaft/arrow overlap is
        # wide enough to sample without antialiasing dominating.  Nodes are
        # shrunk to a hairline so they cannot cover the arrowheads.
        browser.js(
            """
          cy.nodes().style({'width': 1, 'height': 1, 'background-opacity': 0,
                            'border-width': 0, 'label': ''});
          cy.nodes().forEach(function (n) {
            n.position(n.id().charAt(0) === 'A' ? {x: 70, y: 80}
                                                : {x: 350, y: 80});
          });
          cy.edges().style({'width': 6, 'arrow-scale': 2});
          cy.zoom(1);
          cy.pan({x: 0, y: 0});
          cy.forceRender();
          return true;
        """
        )
        canvas_img = _png_rgb(browser)
        points = [(300, 80), (321, 80), (330, 80), (330, 72)]
        canvas = {p: _rgb(canvas_img, *p) for p in points}

        sanitized = browser.js("return sanitizeSvgForOffice(cy.svg({full: true, bg: 'white'}));")
        root = ET.fromstring(sanitized)

        groups = [g for g in root.iter(NS + "g") if g.get("data-vp-edge") == "1"]
        assert len(groups) == 2, sanitized[:500]
        arrow_ds = []
        for group in groups:
            paths = group.findall(NS + "path")
            assert len(paths) == 2, [p.get("d") for p in paths]
            outline, arrow = paths
            # The stroke is now a filled outline that keeps the stroke alpha.
            assert outline.get("fill-opacity") == "0.4", outline.attrib
            assert outline.get("fill") in ("#000000", "#ff0000"), outline.attrib
            assert arrow.get("fill-opacity") == "0.4", arrow.attrib
            arrow_ds.append(arrow.get("d"))

        # Coincident edges legitimately share one arrow geometry; what must
        # not survive is an opaque white twin with that same geometry.
        all_paths = list(root.iter(NS + "path"))
        for d in set(arrow_ds):
            assert not any(
                p.get("d") == d and p.get("fill") in ("#ffffff", "white") for p in all_paths
            )

        # The exported SVG renders the same composites as the canvas:
        # body-only, body+arrow overlap, arrow-only, arrow-only off-shaft.
        tx, ty = _translate(root)
        browser.js(_SVG_EXPORT_JS % (json.dumps(sanitized), json.dumps(points), tx, ty))
        assert browser.wait("return !!window.__exportSamples;", timeout=30)
        rendered = browser.js("return window.__exportSamples;")
        assert rendered != "load-error", rendered
        rendered_by_point = {tuple(row[:2]): tuple(row[2:]) for row in rendered}
        for p in points:
            assert _close(rendered_by_point[p], canvas[p], 3), (p, rendered, canvas)

        # A renderer that still emits the white cut-out twin: the sanitizer
        # drops the backing and leaves the arrow translucent.
        synthetic = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">'
            '<path d="M2 2 L18 2 L10 18 Z" fill="#ffffff"/>'
            '<path d="M2 2 L18 2 L10 18 Z" fill="#ff0000" '
            'fill-opacity="0.4"/></svg>'
        )
        cleaned = browser.js("return sanitizeSvgForOffice(%s);" % json.dumps(synthetic))
        cleaned_root = ET.fromstring(cleaned)
        cleaned_paths = list(cleaned_root.iter(NS + "path"))
        assert len(cleaned_paths) == 1, cleaned
        assert cleaned_paths[0].get("fill") == "#ff0000"
        assert cleaned_paths[0].get("fill-opacity") == "0.4"

        # A synthetic shaft + arrow pair keeps both alphas after fusion.
        pair = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="120">'
            '<rect width="200" height="120" fill="white"/>'
            '<path fill="none" stroke="#000000" stroke-opacity="0.4" '
            'stroke-width="6" d="M10 60 L 90 60"/>'
            '<path fill="#000000" fill-opacity="0.4" stroke="none" '
            'd="M80 70 L 100 60 L 80 50 Z"/></svg>'
        )
        fused = ET.fromstring(browser.js("return sanitizeSvgForOffice(%s);" % json.dumps(pair)))
        edge_groups = [g for g in fused.iter(NS + "g") if g.get("data-vp-edge") == "1"]
        assert len(edge_groups) == 1
        fused_paths = edge_groups[0].findall(NS + "path")
        assert [p.get("fill-opacity") for p in fused_paths] == ["0.4", "0.4"], [
            p.attrib for p in fused_paths
        ]
        assert not list(fused.iter(NS + "line"))

        assert browser.errors() == [], browser.errors()
