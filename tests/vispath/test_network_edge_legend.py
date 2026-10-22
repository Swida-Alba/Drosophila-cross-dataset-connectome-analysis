"""Caller-supplied edge-arrow legend on vispath network pages.

Pins the feature requester's contract from
`clock-production/_plan/plan-2026-10-22-network-edge-legend.md`:

* a spec of titled groups is drawn as a filled-``<rect>`` shaft (one solid
  segment, or gap-separated segments for dashed/dotted) + ``<polygon>`` arrow
  samples (never colour swatches) in a page overlay, in the network's own
  SVG export (legend appended BEFORE ``sanitizeSvgForOffice``) and in a
  standalone SVG download;
* the exported legend stays one standalone ``edgeLegend`` group with
  separate ``edgeLegendShapes`` / ``edgeLegendTexts`` subgroups, kept out
  of the network's global ``#shapes`` / ``#texts`` groups;
* every legend ``<text>`` carries ``font-family="Arial"`` explicitly and
  opacity is written as ``fill-opacity`` — the PPT-safe contract; shafts are
  never ``<line>`` (PowerPoint displaces their degenerate boxes).
* a page without ``network_edge_legend`` emits none of the overlay /
  spec / button / builder JS and keeps ``exportSVG()`` as its handler.

Run:
    python -m pytest tests/vispath/test_network_edge_legend.py -q
"""

import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/vispath

from heatmap_cdp import Page, REQUIRES_CDP  # noqa: E402
from vispath_pkg.fast_graph_core import FastGraph  # noqa: E402
from vispath_pkg.vispath import VisualizePath  # noqa: E402

NS = "{http://www.w3.org/2000/svg}"

SPEC = [
    {"title": "Neurotransmitter", "items": [
        {"label": "ACh", "color": "#FFA500"},
        {"label": "Glut", "color": "#008000"},
        {"label": "mixed", "color": "#95A5A6"},
    ]},
    {"title": "Synapse strength", "items": [
        {"label": "1–20 (20%)", "color": "#6b7280", "opacity": 0.2},
        {"label": "21–50 (40%)", "color": "#6b7280", "opacity": 0.4},
        {"label": "51–100 (75%)", "color": "#6b7280", "opacity": 0.75},
        {"label": ">100 (100%)", "color": "#6b7280", "opacity": 1.0},
    ]},
]
LEGEND_LABELS = [
    "Neurotransmitter", "ACh", "Glut", "mixed", "Synapse strength",
    "1–20 (20%)", "21–50 (40%)", "51–100 (75%)", ">100 (100%)",
]


def _render(tmp_path, name, **kwargs):
    """Generate a network page from a 4-node chain (the same fixture shape
    the other vispath network tests use) and return (vp, path, html)."""
    df = pd.DataFrame({"path_block": ["S>N1>N2>T"], "weights": [[10, 20, 30]]})
    vp = VisualizePath(
        path_file=df, output_folder=str(tmp_path), showfig=False,
        verbose=False, network_layout="dagre", **kwargs)
    G = FastGraph()
    for u, v, w in [("S", "N1", 10), ("N1", "N2", 20), ("N2", "T", 30)]:
        G.add_edge(u, v, w)
        G.node_attrs.setdefault(u, {})["node_type"] = "intermediate"
        G.node_attrs.setdefault(v, {})["node_type"] = "intermediate"
    G.node_attrs["S"]["node_type"] = "source"
    G.node_attrs["T"]["node_type"] = "target"
    out = tmp_path / f"{name}.html"
    vp._plot_cytoscape_network(G, output_path=str(out), layout="dagre",
                               open_browser=False)
    return vp, out, out.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def legend_page(tmp_path_factory):
    return _render(tmp_path_factory.mktemp("legend"), "legend",
                   network_edge_legend=SPEC)[1]


@pytest.fixture(scope="module")
def plain_page(tmp_path_factory):
    return _render(tmp_path_factory.mktemp("plain"), "plain")[1]


def _script_spec_line(html):
    return next(line for line in html.splitlines()
                if line.startswith("const edgeLegendSpec = "))


def _ancestor_translate(root, node):
    """(x, y) of the nearest ancestor carrying a matrix()/translate()."""
    parents = {child: parent for parent in root.iter()
               for child in parent}
    cursor = node
    while cursor is not None:
        tf = cursor.get("transform") or ""
        if tf.startswith("matrix("):
            values = [float(v) for v in tf[len("matrix("):-1].split()]
            return values[4], values[5]
        if tf.startswith("translate("):
            values = [float(v) for v in
                      tf[len("translate("):-1].replace(",", " ").split()]
            return values[0], values[1]
        cursor = parents.get(cursor)
    raise AssertionError("no translate ancestor")


def _wire_download_js(value_name):
    return (
        "window.%s = null;"
        "downloadBlob = function(blob, name) {"
        "  blob.text().then(function(t){ window.%s = {text: t, name: name}; });"
        "};" % (value_name, value_name))


# =============================================================================
# Generated page: spec present vs absent
# =============================================================================

class TestGeneratedPage:
    def test_legend_page_has_overlay_controls_and_builder(self, legend_page):
        html = legend_page.read_text(encoding="utf-8")
        assert 'id="vpEdgeLegendHost"' in html
        # The page card follows the pilot (top-left: the canvas bottom sits
        # below a 900 px viewport); the export keeps the requested
        # bottom-left anchor.
        assert 'class="vp-edge-legend-host vp-edge-legend-top-left"' in html
        assert 'const edgeLegendPosition = "bottom-left";' in html
        assert 'onclick="exportSVGWithLegend()"' in html
        assert 'onclick="exportEdgeLegendSvg()"' in html
        assert "Legend SVG" in html
        for name in ("normalizeEdgeLegend", "edgeLegendBody",
                     "standaloneEdgeLegendSvg", "appendEdgeLegendToSvg",
                     "renderEdgeLegendOverlay", "exportSVGWithLegend",
                     "exportEdgeLegendSvg"):
            assert f"function {name}(" in html, name
        assert 'const edgeLegendPosition = "bottom-left";' in html
        assert "renderEdgeLegendOverlay(edgeLegendSpec, edgeLegendPosition" in html
        # The overlay is a DOM host inside #cy, not a cytoscape element.
        assert "<div id=\"cy\"><div id=\"vpEdgeLegendHost\"" in html

    def test_default_page_has_no_legend_furniture(self, plain_page):
        html = plain_page.read_text(encoding="utf-8")
        for marker in ("vpEdgeLegendHost", "vp-edge-legend-host",
                       "edgeLegendSpec", "exportSVGWithLegend",
                       "Legend SVG", "normalizeEdgeLegend",
                       "function edgeLegendBody("):
            assert marker not in html, marker
        assert 'onclick="exportSVG()"' in html
        assert html.count('onclick="exportSVG()"') == 1

    def test_spec_is_normalised_and_position_is_clamped(self, tmp_path):
        spec = [
            {"title": "  NT  ", "items": [
                {"label": " ACh ", "opacity": 1.7},
                {"label": "", "color": "#ffffff"},
                {"label": "Glut", "line_style": "DASHED", "opacity": -0.2},
                {"label": "x", "opacity": "nope", "line_style": "wavy"},
            ]},
            {"title": "", "items": []},
            {"title": "Empty", "items": []},
            "not a group",
        ]
        vp, _, html = _render(tmp_path, "norm", network_edge_legend=spec,
                              edge_legend_position="TOP-RIGHT",
                              edge_legend_page_position="bottom-right")
        assert vp.network_edge_legend == [
            {"title": "NT", "items": [
                {"label": "ACh", "color": "#333333", "opacity": 1.0,
                 "line_style": "solid"},
                {"label": "Glut", "color": "#333333", "opacity": 0.0,
                 "line_style": "dashed"},
                {"label": "x", "color": "#333333", "opacity": 1.0,
                 "line_style": "solid"},
            ]},
        ]
        assert vp.edge_legend_position == "top-right"
        assert vp.edge_legend_page_position == "bottom-right"
        assert "vp-edge-legend-bottom-right" in html
        assert 'const edgeLegendPosition = "top-right";' in html

    def test_invalid_position_falls_back_to_each_default(self, tmp_path):
        vp, _, html = _render(tmp_path, "badpos", network_edge_legend=SPEC,
                              edge_legend_position="middle",
                              edge_legend_page_position="sideways")
        assert vp.edge_legend_position == "bottom-left"
        assert vp.edge_legend_page_position == "top-left"
        assert "vp-edge-legend-top-left" in html
        assert 'const edgeLegendPosition = "bottom-left";' in html

    def test_empty_spec_emits_no_legend(self, tmp_path):
        vp, _, html = _render(tmp_path, "empty", network_edge_legend=[
            {"title": "", "items": [{"label": ""}]}])
        assert vp.network_edge_legend == []
        assert "vpEdgeLegendHost" not in html
        assert 'onclick="exportSVG()"' in html

    def test_labels_cannot_break_out_of_the_inline_script(self, tmp_path):
        poison = '</script><img src=x onerror=alert(1)>S>&"'
        _, _, html = _render(tmp_path, "poison", network_edge_legend=[
            {"items": [{"label": poison}]}])
        line = _script_spec_line(html)
        assert "</script>" not in line
        assert "<img" not in line
        assert "\\u003c/script\\u003e" in line
        assert "\\u0026" in line

    def test_placeholder_text_in_labels_stays_in_the_spec(self, tmp_path):
        """A label may contain the position placeholder without corruption."""
        token = "__EDGE_LEGEND_POSITION__"
        _, _, html = _render(
            tmp_path, "placeholder",
            network_edge_legend=[{"title": token, "items": [
                {"label": token, "color": "#123456"}]}],
            edge_legend_page_position="bottom-left")
        spec_line = _script_spec_line(html)
        payload = json.loads(spec_line.split("=", 1)[1].strip().rstrip(";"))
        assert payload[0]["title"] == token
        assert payload[0]["items"][0]["label"] == token
        assert 'const edgeLegendPosition = "bottom-left";' in html


# =============================================================================
# Live page: builder, overlay geometry, exports
# =============================================================================

def _wait_ready(browser):
    assert browser.wait(
        "return typeof cy !== 'undefined' && cy.nodes().length === 4;"), \
        "network page did not initialise"


@REQUIRES_CDP
def test_live_standalone_legend_matches_the_ppt_contract(legend_page):
    with Page(legend_page) as browser:
        _wait_ready(browser)
        svg = browser.js("return standaloneEdgeLegendSvg(edgeLegendSpec);")
        root = ET.fromstring(svg)

        outer = root.find(NS + "g")
        assert outer.get("class") == "vp-edge-legend"
        assert [g.get("id") for g in outer.findall(NS + "g")] == [
            "edgeLegendShapes", "edgeLegendTexts"]
        shape_g = outer.find(NS + "g[@id='edgeLegendShapes']")
        text_g = outer.find(NS + "g[@id='edgeLegendTexts']")
        assert not shape_g.findall(".//" + NS + "line")
        assert len(shape_g.findall(".//" + NS + "polygon")) == 7
        shafts = [el for el in shape_g.findall(".//" + NS + "rect")
                  if el.get("data-vp-edge-legend-shaft") == "1"]
        assert len(shafts) == 7
        assert not shape_g.findall(".//" + NS + "text")
        assert not text_g.findall(".//" + NS + "line")

        texts = text_g.findall(".//" + NS + "text")
        polygons = shape_g.findall(".//" + NS + "polygon")
        assert [t.text for t in texts] == LEGEND_LABELS
        assert len(polygons) == 7
        assert not root.findall(".//" + NS + "image")
        assert not root.findall(".//" + NS + "foreignObject")
        assert all(t.get("font-family") == "Arial" for t in texts)
        assert [shaft.get("fill-opacity") for shaft in shafts] == [
            "1", "1", "1", "0.2", "0.4", "0.75", "1"]
        assert [shaft.get("fill") for shaft in shafts] == [
            "#FFA500", "#008000", "#95A5A6",
            "#6b7280", "#6b7280", "#6b7280", "#6b7280"]
        assert [p.get("fill") for p in polygons] == [
            "#FFA500", "#008000", "#95A5A6",
            "#6b7280", "#6b7280", "#6b7280", "#6b7280"]
        assert [p.get("fill-opacity") for p in polygons] == [
            "1", "1", "1", "0.2", "0.4", "0.75", "1"]
        assert root.get("viewBox") == (
            f'0 0 {root.get("width")} {root.get("height")}')
        assert browser.errors() == [], browser.errors()


@REQUIRES_CDP
def test_live_dashed_and_dotted_items(legend_page):
    spec = [{"title": "Route", "items": [
        {"label": "solid"}, {"label": "dashed", "line_style": "dashed"},
        {"label": "dotted", "line_style": "dotted"}]}]
    with Page(legend_page) as browser:
        _wait_ready(browser)
        svg = browser.js(
            "return standaloneEdgeLegendSvg(" + json.dumps(spec) + ");")
        assert "stroke-dasharray" not in svg
        root = ET.fromstring(svg)
        assert not root.findall(".//" + NS + "line")
        shafts = [el for el in root.findall(".//" + NS + "rect")
                  if el.get("data-vp-edge-legend-shaft") == "1"]
        # solid = one shaft; dashed = 4 segments; dotted = 8 segments.
        assert len(shafts) == 13, [s.get("x") for s in shafts]
        assert browser.errors() == [], browser.errors()


@REQUIRES_CDP
def test_live_overlay_sits_at_the_chosen_canvas_corner(legend_page):
    with Page(legend_page) as browser:
        _wait_ready(browser)
        geom = json.loads(browser.js("""
            var host = document.getElementById('vpEdgeLegendHost');
            var r = host.getBoundingClientRect();
            var c = document.getElementById('cy').getBoundingClientRect();
            return JSON.stringify({
                text: host.textContent,
                pointerEvents: getComputedStyle(host).pointerEvents,
                zIndex: getComputedStyle(host).zIndex,
                left: Math.round(r.left - c.left),
                top: Math.round(r.top - c.top),
                width: Math.round(r.width),
                height: Math.round(r.height)
            });
        """))
        for label in LEGEND_LABELS:
            assert label in geom["text"]
        assert geom["pointerEvents"] == "none"
        assert geom["zIndex"] == "45"
        # 1 px canvas border plus the 40 px top anchor clears the edit badge
        # and the Pan/Select hints.
        assert 11 <= geom["left"] <= 14, geom
        assert 39 <= geom["top"] <= 42, geom
        assert geom["width"] == 170, geom
        assert geom["height"] == 200, geom
        assert browser.errors() == [], browser.errors()


@REQUIRES_CDP
def test_live_network_export_carries_legend_after_sanitize(legend_page):
    with Page(legend_page) as browser:
        _wait_ready(browser)
        raw = browser.js("return cy.svg({full: true});")
        appended = browser.js(
            "return appendEdgeLegendToSvg(cy.svg({full: true}), "
            "edgeLegendSpec, 'bottom-left');")
        # Appending only adds the legend: the network's own paths/texts
        # are untouched (the sanitizer runs later, on the download path).
        assert appended.count("<path") == raw.count("<path")
        assert appended.count("<text") == raw.count("<text") + 9
        assert appended.count("<line") == raw.count("<line")
        assert appended.count("<rect") == raw.count("<rect") + 8
        assert appended.count("<polygon") == raw.count("<polygon") + 7

        browser.js(_wire_download_js("__svgOut") + "exportSVGWithLegend();")
        assert browser.wait(
            "return !!(window.__svgOut && window.__svgOut.text);"), \
            "SVG export produced nothing"
        out = browser.js("return window.__svgOut;")
        assert out["name"] == "network_selected_paths.svg"
        root = ET.fromstring(out["text"])
        texts = root.findall(".//" + NS + "text")
        labels = [t.text for t in texts]
        for node in ("S", "N1", "N2", "T"):
            assert node in labels
        assert [label for label in labels
                if label in LEGEND_LABELS] == LEGEND_LABELS
        arial = [t for t in texts if t.get("font-family") == "Arial"]
        assert len(arial) == 9
        assert not root.findall(".//" + NS + "line")
        assert len(root.findall(".//" + NS + "polygon")) == 7
        rects = root.findall(".//" + NS + "rect")
        # canvas background + legend card + seven filled shaft segments
        assert len(rects) == 9
        card = next(r for r in rects if r.get("rx") == "4")
        assert card.get("fill") == "#ffffff"
        assert card.get("fill-opacity") == "0.94"
        shafts = [r for r in rects
                  if r.get("data-vp-edge-legend-shaft") == "1"]
        assert len(shafts) == 7
        assert not root.findall(".//" + NS + "image")
        assert not root.findall(".//" + NS + "foreignObject")
        assert [g.get("id") for g in root.findall(NS + "g")
                if g.get("id")] == ["shapes", "texts", "edgeLegend"]

        # The legend is one standalone group with its own shapes/texts
        # subgroups, kept out of the network's global groups.
        legend = root.find(NS + "g[@id='edgeLegend']")
        assert legend.get("data-vp-edge-legend") == "1"
        shape_g = legend.find(NS + "g[@id='edgeLegendShapes']")
        text_g = legend.find(NS + "g[@id='edgeLegendTexts']")
        assert shape_g is not None and text_g is not None
        assert not shape_g.findall(".//" + NS + "line")
        assert len(shape_g.findall(".//" + NS + "polygon")) == 7
        shape_shafts = [r for r in shape_g.findall(".//" + NS + "rect")
                        if r.get("data-vp-edge-legend-shaft") == "1"]
        assert len(shape_shafts) == 7
        assert len(shape_g.findall(".//" + NS + "rect")) == 8
        assert not shape_g.findall(".//" + NS + "text")
        assert len(text_g.findall(".//" + NS + "text")) == 9
        assert not text_g.findall(".//" + NS + "line")
        assert root[-1].get("id") == "edgeLegend"  # paints above the network
        for global_group in ("shapes", "texts"):
            g = root.find(NS + f"g[@id='{global_group}']")
            assert not g.findall(".//*[@data-vp-edge-legend]"), global_group

        height = float(root.get("height"))
        ach = next(t for t in texts if t.text == "ACh")
        x, y = _ancestor_translate(root, ach)
        assert x == 16
        assert abs(y - (height - 200 - 16)) < 0.5
        # The common translate is lifted onto the legend group itself.
        assert legend.get("transform") == f"matrix(1 0 0 1 16 {y:g})"
        assert shape_g.get("transform") is None
        assert text_g.get("transform") is None
        assert browser.errors() == [], browser.errors()


@REQUIRES_CDP
def test_live_anchor_math_for_all_four_corners(legend_page):
    synthetic = ('<svg xmlns="http://www.w3.org/2000/svg" '
                 'width="800" height="600"></svg>')
    expected = {
        "bottom-left": (16, 384),
        "bottom-right": (614, 384),
        "top-left": (16, 16),
        "top-right": (614, 16),
    }
    with Page(legend_page) as browser:
        _wait_ready(browser)
        for position, (x, y) in expected.items():
            svg = browser.js(
                "return appendEdgeLegendToSvg(" + json.dumps(synthetic) + ","
                " edgeLegendSpec, " + json.dumps(position) + ");")
            root = ET.fromstring(svg)
            group = next(g for g in root.iter(NS + "g")
                         if g.get("data-vp-edge-legend") == "1")
            assert _ancestor_translate(root, group) == (x, y), position
        assert browser.errors() == [], browser.errors()


@REQUIRES_CDP
def test_live_standalone_download(legend_page):
    with Page(legend_page) as browser:
        _wait_ready(browser)
        browser.js(_wire_download_js("__legendOut") + "exportEdgeLegendSvg();")
        assert browser.wait(
            "return !!(window.__legendOut && window.__legendOut.text);"), \
            "legend download produced nothing"
        out = browser.js("return window.__legendOut;")
        assert out["name"] == "edge_legend.svg"
        root = ET.fromstring(out["text"])
        assert [t.text for t in root.findall(".//" + NS + "text")] == \
            LEGEND_LABELS
        assert not root.findall(".//" + NS + "line")
        assert len(root.findall(".//" + NS + "polygon")) == 7
        shafts = [el for el in root.findall(".//" + NS + "rect")
                  if el.get("data-vp-edge-legend-shaft") == "1"]
        assert len(shafts) == 7
        assert browser.errors() == [], browser.errors()
