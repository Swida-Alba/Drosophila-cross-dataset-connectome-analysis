"""Translucent arrowheads composite in the canvas and in the SVG export.

Cytoscape 3.28.1's cut-out erased whatever sat under a below-full-opacity
arrowhead before painting it, so two coincident translucent arrows kept the
single-arrow value.  The vendored bundle is patched for the cut-out (hollow
arrows keep it) and for the triangle geometry: the shaft now ends at the
arrowhead's base instead of running inside it, so the translucent head has
no darker overlap stripe.  The Office sanitizer must not flatten the
stroke/arrow alpha back to opaque.

The pixel expectations below are source-over composites over white.  The
shaft and head meet at the base, so the head interior is uniform and equal
to the arrow-only value; parallel arrows still composite with each other.
The SVG export merges a same-colour geometrically straight shaft and head
into ONE single-subpath path (no butt-joint seam, one PowerPoint shape);
two-tone, curved, dashed and multi-head edges stay grouped.

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
MARKER = "DROCAT patch (Cytoscape 3.28.1)"


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

        # (line midpoint, arrow interior at the head's rim, arrow interior
        #  clear of the shaft); the rim value must equal the arrow-only
        #  value because the shaft ends at the arrowhead base.
        cases = {
            "single": (80, (153, 153, 153), (153, 153, 153), (153, 153, 153)),
            "two coincident": (180, (92, 92, 92), (92, 92, 92), (92, 92, 92)),
            "red over black": (280, (194, 92, 92), (194, 92, 92), (194, 92, 92)),
        }
        for label, (y, line, rim, arrow) in cases.items():
            assert _close(_rgb(img, 210, y), line), (label, "line", _rgb(img, 210, y))
            assert _close(_rgb(img, 321, y), rim), (
                label,
                "arrow rim",
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
        # The rim and the arrow-only interior read the same: no shaft ink
        # inside the head.
        assert _close(canvas[(321, 80)], canvas[(330, 80)])

        sanitized = browser.js("return sanitizeSvgForOffice(cy.svg({full: true, bg: 'white'}));")
        root = ET.fromstring(sanitized)

        # Both edges are solid straight shafts with one same-colour head,
        # so each exports as ONE single-subpath path (no group, no seam).
        edge_nodes = [el for el in root.iter() if el.get("data-vp-edge") == "1"]
        assert len(edge_nodes) == 2, sanitized[:500]
        for el in edge_nodes:
            assert el.tag == NS + "path", el.tag
            assert el.get("fill-opacity") == "0.4", el.attrib
            assert el.get("fill") in ("#000000", "#ff0000"), el.attrib
            d = el.get("d")
            assert d.count("M") == 1, d
            nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", d)]
            assert len(nums) == 14, nums  # 7 vertices: shaft + one head
            xs = nums[0::2]
            assert abs(max(xs) - 349.5) <= 0.01, xs  # arrow tip
            assert abs(sorted(set(xs))[-2] - 318.454) <= 0.01, xs  # head base
            assert not any(318.5 < x < 349.4 for x in xs), xs

        # The exported SVG renders the same composites as the canvas:
        # body-only, arrow rim, arrow-only, arrow-only off-shaft. The
        # merged path also removes the butt-joint seam at x = 318.454.
        seam = (318, 80)
        tx, ty = _translate(root)
        browser.js(_SVG_EXPORT_JS % (json.dumps(sanitized), json.dumps(points + [seam]), tx, ty))
        assert browser.wait("return !!window.__exportSamples;", timeout=30)
        rendered = browser.js("return window.__exportSamples;")
        assert rendered != "load-error", rendered
        rendered_by_point = {tuple(row[:2]): tuple(row[2:]) for row in rendered}
        for p in points:
            assert _close(rendered_by_point[p], canvas[p], 3), (p, rendered, canvas)
        assert _close(rendered_by_point[seam], canvas[(300, 80)], 3), (
            rendered_by_point[seam],
            canvas[(300, 80)],
        )

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

        def sanitized_nodes(svg_text):
            return [el for el in ET.fromstring(svg_text).iter() if el.get("data-vp-edge") == "1"]

        # Same-colour solid straight shaft + one head -> one path.
        pair = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="120">'
            '<rect width="200" height="120" fill="white"/>'
            '<path fill="none" stroke="#000000" stroke-opacity="0.4" '
            'stroke-width="6" d="M10 60 L 80 60"/>'
            '<path fill="#000000" fill-opacity="0.4" stroke="none" '
            'd="M80 70 L 100 60 L 80 50 Z"/></svg>'
        )
        fused = sanitized_nodes(browser.js("return sanitizeSvgForOffice(%s);" % json.dumps(pair)))
        assert len(fused) == 1 and fused[0].tag == NS + "path", fused
        assert fused[0].get("fill-opacity") == "0.4"
        fused_d = fused[0].get("d")
        assert fused_d.count("M") == 1
        fused_nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", fused_d)]
        assert len(fused_nums) == 14
        fused_xs = fused_nums[0::2]
        assert (min(fused_xs), sorted(set(fused_xs))[-2], max(fused_xs)) == (
            10.0,
            80.0,
            100.0,
        )

        # A bezier whose control point lies on the line is geometrically
        # straight (the article's edges) and merges too.
        bezier_straight = pair.replace("M10 60 L 80 60", "M10 60 Q 45 60 80 60")
        bezier_nodes = sanitized_nodes(
            browser.js("return sanitizeSvgForOffice(%s);" % json.dumps(bezier_straight))
        )
        assert len(bezier_nodes) == 1 and bezier_nodes[0].tag == NS + "path"

        # A source-only head merges from the other end too.
        source_head = pair.replace('d="M80 70 L 100 60 L 80 50 Z"', 'd="M10 70 L -10 60 L 10 50 Z"')
        source_nodes = sanitized_nodes(
            browser.js("return sanitizeSvgForOffice(%s);" % json.dumps(source_head))
        )
        assert len(source_nodes) == 1 and source_nodes[0].tag == NS + "path"
        source_xs = [
            float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", source_nodes[0].get("d"))[0::2]
        ]
        assert (min(source_xs), max(source_xs)) == (-10.0, 80.0)

        # Two-tone heads stay as a group (two shapes).
        two_tone = pair.replace(
            'fill="#000000" fill-opacity="0.4" stroke="none"',
            'fill="#ff0000" fill-opacity="0.4" stroke="none"',
        )
        two_tone_nodes = sanitized_nodes(
            browser.js("return sanitizeSvgForOffice(%s);" % json.dumps(two_tone))
        )
        assert len(two_tone_nodes) == 1 and two_tone_nodes[0].tag == NS + "g"
        two_tone_paths = two_tone_nodes[0].findall(NS + "path")
        assert [p.get("fill") for p in two_tone_paths] == ["#000000", "#ff0000"]
        assert [p.get("fill-opacity") for p in two_tone_paths] == ["0.4", "0.4"]

        # Curved shafts stay as a group.
        curved = pair.replace("M10 60 L 80 60", "M10 60 Q 45 30 80 60")
        curved_nodes = sanitized_nodes(
            browser.js("return sanitizeSvgForOffice(%s);" % json.dumps(curved))
        )
        assert len(curved_nodes) == 1 and curved_nodes[0].tag == NS + "g"
        assert len(curved_nodes[0].findall(NS + "path")) == 2

        # More than one head stays as a group.
        multi_head = pair.replace(
            "</svg>",
            '<path fill="#000000" fill-opacity="0.4" stroke="none" '
            'd="M80 70 L 100 60 L 80 50 Z"/></svg>',
        )
        multi_nodes = sanitized_nodes(
            browser.js("return sanitizeSvgForOffice(%s);" % json.dumps(multi_head))
        )
        assert len(multi_nodes) == 1 and multi_nodes[0].tag == NS + "g"
        assert len(multi_nodes[0].findall(NS + "path")) == 3

        # Opaque same-colour straight edges merge too, with no opacity attr.
        opaque = pair.replace(' stroke-opacity="0.4"', "").replace(' fill-opacity="0.4"', "")
        opaque_nodes = sanitized_nodes(
            browser.js("return sanitizeSvgForOffice(%s);" % json.dumps(opaque))
        )
        assert len(opaque_nodes) == 1 and opaque_nodes[0].tag == NS + "path"
        assert opaque_nodes[0].get("fill-opacity") is None

        assert browser.errors() == [], browser.errors()
