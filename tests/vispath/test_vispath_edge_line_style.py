"""Per-edge stroke pattern on the vispath network canvas.

Filed from `clock-production` for the one-interlayer conserved network: an
indirect connection must be readable as indirect, so its edges are drawn
dashed while a direct edge stays solid. The network renderer had a live
*manual* line-pattern control (it styles the selected edges and persists in a
saved layout) but no way for a caller to declare the pattern in the edge list,
so every generated page opened solid.

Design decisions these tests pin:

* the pattern is a **per-edge data channel** (`line_style` column,
  `solid`|`dashed`|`dotted`), not a page-wide switch: one page may mix an
  indirect edge with a direct one, and a future caller decides per row;
* the dash/dot interval scales with the **rendered edge width**; dotted renders
  through the dashed path with round caps because Cytoscape hardcodes dotted
  to a 1px/1px dash;
* an unknown value is dropped at load time rather than emitting a style
  Cytoscape does not know (it would silently render solid and read as a real
  choice);
* a page whose edge list carries no `line_style` column gains **no** declared
  selector rule and **no** edge-data key; the manual line-pattern control is
  shared renderer code and now scales dashes/dots with edge width everywhere.

Run:
    conda run -n drocat-4.5.0 python -m pytest \\
        tests/vispath/test_vispath_edge_line_style.py -q
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vispath_pkg.vispath import VisualizePath  # noqa: E402
from heatmap_cdp import Page, REQUIRES_CDP  # noqa: E402


def _render(tmp_path, frame, name):
    vp = VisualizePath(path_file=frame, output_folder=str(tmp_path),
                       showfig=False, verbose=False,
                       network_layout="dagre")
    vp.base_filename = name
    vp.build_network()
    page = vp.create_network()
    return Path(page), Path(page).read_text(encoding="utf-8")


def _edge_payloads(html):
    match = re.search(r"const elements = \{\s*nodes:\s*(\[.*?\])\s*,"
                      r"\s*edges:\s*(\[.*?\])\s*\};", html, re.S)
    assert match, "embedded elements JSON not found"
    return json.loads(match.group(2))


def _frame(styles):
    return pd.DataFrame([
        {"source": src, "target": tgt, "weight": 10 + i,
         "ratio": 0.1, "ratio_adj": 0.2, "color": "#008048",
         **({"line_style": style} if style is not None else {})}
        for i, (src, tgt, style) in enumerate(zip(
            ["A", "R1", "C", "R2", "E"], ["R1", "B", "R2", "D", "F"],
            styles))
    ])


def test_dashed_and_dotted_strokes_are_declared_and_carried(tmp_path):
    page, html = _render(
        tmp_path, _frame(["dashed", "dashed", "dotted", "solid", None]),
        "stroke")
    assert page.is_file()
    for style in ("dashed", "dotted"):
        assert f'edge[line_style = "{style}"]' in html, style
    # Dotted renders through the dashed path so its interval can scale with
    # width; the static fallbacks keep the first paint sane.
    assert "'line-style': 'dashed'" in html
    assert "'line-dash-pattern': [6, 3]" in html
    assert "'line-cap': 'round'" in html
    assert "'line-dash-pattern': [0.01, 6]" in html
    assert 'function syncEdgeLinePattern' in html
    payloads = {(e["data"]["source"], e["data"]["target"]): e["data"]
                for e in _edge_payloads(html)}
    assert payloads[("A", "R1")]["line_style"] == "dashed"
    assert payloads[("R1", "B")]["line_style"] == "dashed"
    assert payloads[("C", "R2")]["line_style"] == "dotted"
    # An explicit solid is carried (the export states the intent) but needs
    # no selector rule; a blank cell carries no key at all.
    assert payloads[("R2", "D")]["line_style"] == "solid"
    assert "line_style" not in payloads[("E", "F")]


def test_no_column_emits_no_declared_style_machinery(tmp_path):
    _, html = _render(tmp_path, _frame([None, None, None, None, None]),
                      "plain")
    # No caller-declared channel: no selector rule and no edge-data key.
    assert "edge[line_style" not in html
    payloads = _edge_payloads(html)
    assert all("line_style" not in e["data"] for e in payloads)
    # The manual line-pattern control is shared renderer code; its width-aware
    # dash helper is present so manual styles scale on every page.
    assert 'function syncEdgeLinePattern' in html


def test_unknown_pattern_is_dropped_not_styled(tmp_path):
    _, html = _render(tmp_path, _frame(["wavy", None, None, None]), "wavy")
    assert "edge[line_style" not in html
    payloads = {(e["data"]["source"], e["data"]["target"]): e["data"]
                for e in _edge_payloads(html)}
    assert "line_style" not in payloads[("A", "R1")]


@REQUIRES_CDP
def test_dash_intervals_scale_with_rendered_width(tmp_path):
    """A thick dashed edge must not read as solid: the dash/gap lengths follow
    the rendered width. Dotted shares the dashed renderer (round caps) because
    Cytoscape hardcodes dotted to 1px/1px."""
    frame = pd.DataFrame([
        {"source": "A", "target": "B", "weight": 1.0, "line_style": "dashed"},
        {"source": "C", "target": "D", "weight": 100.0, "line_style": "dashed"},
        {"source": "E", "target": "F", "weight": 50.0, "line_style": "dotted"},
        {"source": "G", "target": "H", "weight": 25.0, "line_style": "solid"},
    ])
    page_path, _ = _render(tmp_path, frame, "dash_scale")
    read = r"""
    var out = {};
    cy.edges().forEach(function (e) {
        var parts = String(e.style('line-dash-pattern') || '')
            .trim().split(' ')
            .filter(function (x) { return x !== ''; })
            .map(Number);
        out[e.data('source') + '->' + e.data('target')] = {
            semantic: e.data('line_style'),
            width: Number(e.width()),
            lineStyle: e.style('line-style'),
            cap: e.style('line-cap'),
            pattern: parts.length >= 2
                ? [parts[0], parts[1]] : null
        };
    });
    return out;
    """
    with Page(page_path) as page:
        assert page.wait(
            "return typeof cy !== 'undefined' && cy.edges().length === 4;")
        assert page.wait(
            "var d = cy.edges().filter(function (e) {"
            " return e.data('line_style') === 'dashed'; });"
            "return d.length === 2 && d.every(function (e) {"
            " var p = String(e.style('line-dash-pattern') || '');"
            " return p.indexOf(' ') > 0; });")
        state = page.js(read)
        thin = state["A->B"]
        thick = state["C->D"]
        dotted = state["E->F"]
        solid = state["G->H"]

        assert thin["lineStyle"] == "dashed" and thick["lineStyle"] == "dashed"
        assert thick["width"] > thin["width"]
        assert thick["pattern"][0] > thin["pattern"][0]
        assert thin["pattern"][0] == pytest.approx(3 * thin["width"], rel=1e-6)
        assert thin["pattern"][1] == pytest.approx(2 * thin["width"], rel=1e-6)
        assert thick["pattern"][0] == pytest.approx(3 * thick["width"], rel=1e-6)

        assert dotted["semantic"] == "dotted"
        assert dotted["lineStyle"] == "dashed"
        assert dotted["cap"] == "round"
        assert dotted["pattern"][0] == pytest.approx(0.01)
        assert dotted["pattern"][1] == pytest.approx(2 * dotted["width"], rel=1e-6)

        assert solid["lineStyle"] == "solid"
        assert solid["cap"] == "butt"

        # The global Edge Width control changes the rendered widths, and the
        # intervals follow without any further interaction.
        page.js("var s = document.getElementById('edgeWidthSlider');"
                "s.value = '8'; s.dispatchEvent(new Event('input')); return true;")
        assert page.wait(
            "var e = cy.edges().filter(function (x) {"
            " return x.data('source') === 'A'; })[0];"
            "var p = String(e.style('line-dash-pattern') || '')"
            " .trim().split(' ').filter(function (x) { return x !== ''; })"
            " .map(Number);"
            "return p.length >= 2 && Math.abs(p[0] - 3 * e.width()) < 1e-6;")
        resized = page.js(read)["A->B"]
        assert resized["width"] > thin["width"]
        assert resized["pattern"][0] == pytest.approx(
            3 * resized["width"], rel=1e-6)
        assert resized["pattern"][0] > thin["pattern"][0]

        # The manual line-pattern panel uses the same width-aware rendering.
        page.js("var e = cy.edges().filter(function (x) {"
                " return x.data('source') === 'G'; })[0];"
                "e.select(); applyEdgeLineStyle({lineStyle: 'dotted'}); return true;")
        assert page.wait(
            "var e = cy.edges().filter(function (x) {"
            " return x.data('source') === 'G'; })[0];"
            "return e.data('line_style') === 'dotted'"
            " && e.style('line-style') === 'dashed'"
            " && e.style('line-cap') === 'round';")
        manual = page.js(read)["G->H"]
        assert manual["pattern"][1] == pytest.approx(
            2 * manual["width"], rel=1e-6)

        page.js("applyEdgeLineStyle({lineStyle: 'solid'}); return true;")
        assert page.wait(
            "var e = cy.edges().filter(function (x) {"
            " return x.data('source') === 'G'; })[0];"
            "return e.style('line-style') === 'solid'"
            " && e.style('line-cap') === 'butt';")
        assert page.errors() == [], page.errors()
