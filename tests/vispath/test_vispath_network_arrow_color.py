"""Arrowhead colour on vispath network pages.

Filed from `clock-production` for the conserved first-clock network: with
`color_edges_by_nt=True` the NT stylesheet rules coloured the lines, but every
edge payload also carried `arrow_color` defaulted to the global link colour,
and the load-time opacity pass wrote that value over the NT arrow rules.  The
page therefore showed orange/green/blue lines with grey arrowheads.

Design decisions these tests pin:

* an edge payload carries `arrow_color` only when its row declares one;
* arrowheads otherwise follow the rendered LINE colour (`data(color)`), which
  is what lets the NT (and negative-weight) rules reach both ends;
* a declared `arrow_color` gets its own late stylesheet rule, so an explicit
  two-tone value still wins over the NT rule;
* the alpha-only edge-style path and the highlight restore keep whatever
  colour the load pass stored, including a band page's pre-blended arrows.

Run:
    python -m pytest tests/vispath/test_vispath_network_arrow_color.py -q
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/vispath

from heatmap_cdp import Page, REQUIRES_CDP  # noqa: E402
from vispath_pkg.vispath import VisualizePath  # noqa: E402


def _render(tmp_path, frame, name, **kwargs):
    vp = VisualizePath(path_file=frame, output_folder=str(tmp_path),
                       showfig=False, verbose=False,
                       network_layout="dagre", **kwargs)
    vp.base_filename = name
    vp.build_network()
    page = Path(vp.create_network())
    return page, page.read_text(encoding="utf-8")


def _edge_payloads(html):
    match = re.search(r"const elements = \{\s*nodes:\s*(\[.*?\])\s*,"
                      r"\s*edges:\s*(\[.*?\])\s*\};", html, re.S)
    assert match, "embedded elements JSON not found"
    return {(e["data"]["source"], e["data"]["target"]): e["data"]
            for e in json.loads(match.group(2))}


def _rule_block(html, selector, span=500):
    start = html.index(f"selector: '{selector}'")
    return html[start:start + span]


def _nt_frame():
    return pd.DataFrame({
        "source": ["A", "B", "C", "D", "E"],
        "target": ["B", "C", "D", "E", "A"],
        "weight": [10.0, 20.0, 30.0, 40.0, 50.0],
        "nt_type": ["acetylcholine", "gaba", "glutamate", None, "histamine"],
    })


# =============================================================================
# Generated payload / stylesheet
# =============================================================================

def test_nt_payloads_leave_arrow_colour_to_the_line(tmp_path):
    _, html = _render(tmp_path, _nt_frame(), "nt", color_edges_by_nt=True)
    payloads = _edge_payloads(html)

    assert len(payloads) == 5, payloads
    assert all("arrow_color" not in data for data in payloads.values()), payloads
    # No explicit values -> no late rule.
    assert "selector: 'edge[arrow_color]'" not in html
    assert "data(arrow_color)" not in html

    # The base edge rule points both arrowheads at the line colour.
    base = _rule_block(html, "edge", span=1200)
    assert "'target-arrow-color': 'data(color)'" in base
    assert "'source-arrow-color': 'data(color)'" in base

    # Each NT rule still carries all three colours.
    for nt, colour in (
            ("acetylcholine", "#ffa500"), ("gaba", "#0000ff"),
            ("glutamate", "#008000"), ("histamine", "#00ffff"),
            ("unknown", "#95a5a6")):
        block = _rule_block(html, f'edge[nt_type = "{nt}"]')
        for prop in ("line-color", "target-arrow-color",
                     "source-arrow-color"):
            assert f"'{prop}': '{colour}'" in block, (nt, prop, block)


def test_explicit_arrow_colour_is_carried_and_has_its_own_rule(tmp_path):
    frame = pd.DataFrame({
        "source": ["A", "B", "C"],
        "target": ["B", "C", "A"],
        "weight": [10.0, 20.0, 30.0],
        "arrow_color": ["#112233", None, ""],
    })
    _, html = _render(tmp_path, frame, "explicit")
    payloads = _edge_payloads(html)

    assert payloads[("A", "B")]["arrow_color"] == "#112233"
    assert "arrow_color" not in payloads[("B", "C")]
    assert "arrow_color" not in payloads[("C", "A")]

    block = _rule_block(html, "edge[arrow_color]")
    assert "'target-arrow-color': 'data(arrow_color)'" in block
    assert "'source-arrow-color': 'data(arrow_color)'" in block
    # Still one base rule for every undeclared edge.
    assert "'target-arrow-color': 'data(color)'" in _rule_block(html, "edge", 1200)


def test_single_colour_page_has_no_arrow_machinery(tmp_path):
    frame = pd.DataFrame({
        "source": ["A", "B"], "target": ["B", "C"], "weight": [10.0, -5.0],
    })
    _, html = _render(tmp_path, frame, "plain")
    payloads = _edge_payloads(html)

    assert all(data["color"] == "#646464" for data in payloads.values())
    assert all("arrow_color" not in data for data in payloads.values())
    assert "selector: 'edge[arrow_color]'" not in html
    assert "data(arrow_color)" not in html


# =============================================================================
# Live page: what Cytoscape actually paints
# =============================================================================

_LIVE_STATE_JS = """
var out = [];
cy.edges().forEach(function (e) {
  out.push({
    key: e.data('source') + '>' + e.data('target'),
    nt: e.data('nt_type'),
    explicit: e.data('arrow_color') || null,
    baseArrow: e.data('__baseArrowColor') || null,
    line: e.style('line-color'),
    target: e.style('target-arrow-color'),
    source: e.style('source-arrow-color'),
    opacity: parseFloat(e.style('line-opacity'))
  });
});
return out;
"""


@REQUIRES_CDP
def test_nt_arrows_follow_each_nt_line_colour_live(tmp_path):
    page, _ = _render(tmp_path, _nt_frame(), "nt", color_edges_by_nt=True)
    with Page(page) as browser:
        assert browser.wait(
            "return typeof cy !== 'undefined' && cy.edges().length === 5;")
        rows = {row["nt"]: row for row in browser.js(_LIVE_STATE_JS)}
        assert set(rows) == {"acetylcholine", "gaba", "glutamate",
                             "histamine", "unknown"}, rows

        for nt, row in rows.items():
            assert row["explicit"] is None, (nt, row)
            assert row["target"] == row["line"], (nt, row)
            assert row["source"] == row["line"], (nt, row)
            # The load pass stored the rendered colour, so a later restore
            # cannot fall back to the base link colour.
            assert row["baseArrow"] == row["line"], (nt, row)

        assert {nt: row["line"] for nt, row in rows.items()} == {
            "acetylcholine": "rgb(255,165,0)",
            "gaba": "rgb(0,0,255)",
            "glutamate": "rgb(0,128,0)",
            "histamine": "rgb(0,255,255)",
            "unknown": "rgb(149,165,166)",
        }
        assert browser.errors() == [], browser.errors()


@REQUIRES_CDP
def test_explicit_arrow_colour_wins_and_survives_alpha_and_restore_live(
        tmp_path):
    frame = pd.DataFrame({
        "source": ["A", "B", "C", "D"],
        "target": ["B", "C", "D", "A"],
        "weight": [10.0, 20.0, 30.0, 40.0],
        "nt_type": ["acetylcholine", "gaba", "glutamate", None],
        # A band page declares its pre-blended arrow colours; the explicit
        # value must survive the alpha pass and later restores.
        "edge_alpha": [0.2, None, None, None],
        "arrow_color": ["#c8d2d2", None, "#112233", None],
    })
    page, _ = _render(tmp_path, frame, "two_tone", color_edges_by_nt=True)
    with Page(page) as browser:
        assert browser.wait(
            "return typeof cy !== 'undefined' && cy.edges().length === 4;")
        initial = {row["key"]: row for row in browser.js(_LIVE_STATE_JS)}

        band = initial["A>B"]
        assert band["line"] == "rgb(255,165,0)"
        assert (band["target"], band["source"]) == \
            ("rgb(200,210,210)", "rgb(200,210,210)")
        assert band["opacity"] == 0.2

        plain_nt = initial["B>C"]
        assert plain_nt["line"] == "rgb(0,0,255)"
        assert (plain_nt["target"], plain_nt["source"]) == \
            ("rgb(0,0,255)", "rgb(0,0,255)")

        two_tone = initial["C>D"]
        assert two_tone["line"] == "rgb(0,128,0)"
        assert (two_tone["target"], two_tone["source"]) == \
            ("rgb(17,34,51)", "rgb(17,34,51)")

        unknown = initial["D>A"]
        assert (unknown["target"], unknown["source"]) == \
            (unknown["line"], unknown["line"])

        # Alpha-only edit: repaints the line opacity, never the arrows.
        browser.js("cy.edges().select();"
                   "applyEdgeLineStyle({alpha: 0.45});"
                   "cy.edges().unselect(); return true;")
        after_alpha = {row["key"]: row for row in browser.js(_LIVE_STATE_JS)}
        for key, row in after_alpha.items():
            assert row["opacity"] == 0.45, (key, row)
            assert (row["target"], row["source"]) == \
                (initial[key]["target"], initial[key]["source"]), (key, row)

        # Highlight paints every edge, then the restore must bring back the
        # NT / explicit arrow colours, not the base link colour.
        browser.js("addHighlightToEdges(cy.edges()); return true;")
        highlighted = {row["key"]: row for row in browser.js(_LIVE_STATE_JS)}
        # The highlight really repainted the edge (both line and arrows),
        # so the restore assertion below is not comparing unchanged styles.
        assert highlighted["B>C"]["target"] != initial["B>C"]["target"]
        assert highlighted["B>C"]["target"] == highlighted["B>C"]["line"]
        browser.js("removeHighlightFromEdges(cy.edges()); return true;")
        restored = {row["key"]: row for row in browser.js(_LIVE_STATE_JS)}
        for key, row in restored.items():
            assert (row["target"], row["source"]) == \
                (initial[key]["target"], initial[key]["source"]), (key, row)
            assert row["line"] == initial[key]["line"], (key, row)
        assert browser.errors() == [], browser.errors()


@REQUIRES_CDP
def test_merged_bidirectional_source_arrow_follows_the_nt_line_live(tmp_path):
    """The bidirectional selector adds the source head but no colour: an
    NT-coloured double-headed edge must keep BOTH heads on its line."""
    frame = pd.DataFrame({
        "source": ["A", "B"], "target": ["B", "A"], "weight": [10.0, 20.0],
        "nt_type": ["acetylcholine", "acetylcholine"],
    })
    page, _ = _render(tmp_path, frame, "merged", color_edges_by_nt=True,
                      merge_reciprocal_edges=True)
    with Page(page) as browser:
        assert browser.wait(
            "return typeof cy !== 'undefined' && cy.edges().length === 2;")
        row = browser.js("""
          var e = cy.edges().filter(function (x) {
            return x.data('pair_canonical') === 1; });
          return {srcShape: e.style('source-arrow-shape'),
                  src: e.style('source-arrow-color'),
                  tgt: e.style('target-arrow-color'),
                  line: e.style('line-color'),
                  bidirectional: e.data('bidirectional')};
        """)
        assert row["bidirectional"] == 1, row
        assert row["srcShape"] == "triangle", row
        assert row["line"] == "rgb(255,165,0)", row
        assert (row["src"], row["tgt"]) == (row["line"], row["line"]), row
        assert browser.errors() == [], browser.errors()


@REQUIRES_CDP
def test_single_colour_and_negative_edges_follow_the_line_live(tmp_path):
    frame = pd.DataFrame({
        "source": ["A", "B"], "target": ["B", "C"], "weight": [10.0, -5.0],
    })
    # #008048 is the clock-production single-colour line; both heads must
    # keep it exactly as before.
    page, _ = _render(tmp_path, frame, "plain", link_color="#008048")
    with Page(page) as browser:
        assert browser.wait(
            "return typeof cy !== 'undefined' && cy.edges().length === 2;")
        rows = {row["key"]: row for row in browser.js(_LIVE_STATE_JS)}

        assert rows["A>B"]["line"] == "rgb(0,128,72)"
        assert (rows["A>B"]["target"], rows["A>B"]["source"]) == \
            ("rgb(0,128,72)", "rgb(0,128,72)")
        # Negative weights take the light-blue stylesheet rule; the base
        # arrow colour must follow it instead of the link colour.
        assert rows["B>C"]["line"] == "rgb(74,144,226)"
        assert (rows["B>C"]["target"], rows["B>C"]["source"]) == \
            ("rgb(74,144,226)", "rgb(74,144,226)")
        assert browser.errors() == [], browser.errors()
