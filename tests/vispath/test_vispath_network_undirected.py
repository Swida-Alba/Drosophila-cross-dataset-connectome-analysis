"""Undirected network view: hide both arrowheads.

Edge Route gains a Hide/Show Arrows button, and Arrow Size 0 is the same
hidden state.  Hiding leaves the arrow size alone (so showing restores the
exact previous heads), removes both the target and the source arrow of a
merged bidirectional edge, survives undo/redo and layout state, and drops
the arrow triangles from the SVG export.

Run:
    python -m pytest tests/vispath/test_vispath_network_undirected.py -q
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/vispath

from heatmap_cdp import Page, REQUIRES_CDP  # noqa: E402
from vispath_pkg.vispath import VisualizePath  # noqa: E402

NS = "{http://www.w3.org/2000/svg}"

_SHAPES_JS = """
  return cy.edges().map(function (e) {
    return [e.style('target-arrow-shape'), e.style('source-arrow-shape')];
  });
"""

_UI_JS = """
  return {
    size: document.getElementById('arrowSizeSlider').value,
    label: document.getElementById('edgeArrowsToggle').textContent
  };
"""


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
    return page


def _triangles(svg_text):
    """Count cytoscape-svg's single-subpath 3-point arrow triangles."""
    root = ET.fromstring(svg_text)
    count = 0
    for path in root.iter(NS + "path"):
        d = path.get("d") or ""
        if d.rstrip().endswith("Z") and len(re.findall(r"-?\d+(?:\.\d+)?", d)) == 6:
            count += 1
    return count


def _set_size(browser, value):
    browser.js(
        """
        var input = document.getElementById('arrowSizeSlider');
        input.value = %d;
        input.dispatchEvent(new Event('input', {bubbles: true}));
        return true;
        """
        % value
    )


@REQUIRES_CDP
def test_edge_route_hide_arrows_and_size_zero(tmp_path):
    frame = pd.DataFrame(
        {
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"],
            "weight": [10.0, 20.0, 30.0],
        }
    )
    page = _render(tmp_path, frame, "undirected")
    with Page(page, width=1200, height=900) as browser:
        assert browser.wait("return typeof cy !== 'undefined' && cy.edges().length === 3;")
        assert browser.wait("return document.getElementById('edgeArrowsToggle') !== null;")
        assert browser.js(
            "return document.getElementById('edgeRouteControls')"
            ".contains(document.getElementById('edgeArrowsToggle'));"
        )
        assert browser.js("return document.getElementById('arrowSizeSlider').min;") == "0"

        initial = browser.js(_SHAPES_JS)
        assert initial == [["triangle", "none"]] * 3, initial
        assert browser.js(_UI_JS) == {"size": "9", "label": "Hide Arrows"}
        assert _triangles(browser.js("return cy.svg({full: true});")) == 3

        # The button hides both ends but leaves the size untouched.
        browser.js("document.getElementById('edgeArrowsToggle').click(); return true;")
        assert browser.js(_SHAPES_JS) == [["none", "none"]] * 3
        assert browser.js(_UI_JS) == {"size": "9", "label": "Show Arrows"}
        assert _triangles(browser.js("return cy.svg({full: true});")) == 0

        # Undo/redo round-trips the hidden state.
        browser.js("undo(); return true;")
        assert browser.js(_SHAPES_JS) == [["triangle", "none"]] * 3
        assert browser.js(_UI_JS)["label"] == "Hide Arrows"
        browser.js("redo(); return true;")
        assert browser.js(_SHAPES_JS) == [["none", "none"]] * 3
        assert browser.js(_UI_JS)["label"] == "Show Arrows"

        # Showing restores the exact previous size.
        browser.js("document.getElementById('edgeArrowsToggle').click(); return true;")
        assert browser.js(_SHAPES_JS) == [["triangle", "none"]] * 3
        _set_size(browser, 5)
        browser.js("document.getElementById('edgeArrowsToggle').click(); return true;")
        assert browser.js(_SHAPES_JS) == [["none", "none"]] * 3
        assert browser.js(_UI_JS) == {"size": "5", "label": "Show Arrows"}
        browser.js("document.getElementById('edgeArrowsToggle').click(); return true;")
        assert browser.js(_SHAPES_JS) == [["triangle", "none"]] * 3
        assert browser.js(_UI_JS) == {"size": "5", "label": "Hide Arrows"}

        # A size change must not override the explicit Hide toggle.
        browser.js("document.getElementById('edgeArrowsToggle').click(); return true;")
        _set_size(browser, 7)
        assert browser.js(_SHAPES_JS) == [["none", "none"]] * 3
        assert browser.js(_UI_JS) == {"size": "7", "label": "Show Arrows"}
        browser.js("document.getElementById('edgeArrowsToggle').click(); return true;")
        assert browser.js(_SHAPES_JS) == [["triangle", "none"]] * 3
        assert browser.js(_UI_JS) == {"size": "7", "label": "Hide Arrows"}

        # Arrow Size 0 hides; a positive size shows again (unless the
        # explicit Hide toggle is on, as pinned above).
        _set_size(browser, 0)
        assert browser.js(_SHAPES_JS) == [["none", "none"]] * 3
        assert browser.js(_UI_JS) == {"size": "0", "label": "Show Arrows"}
        _set_size(browser, 5)
        assert browser.js(_SHAPES_JS) == [["triangle", "none"]] * 3
        assert browser.js(_UI_JS) == {"size": "5", "label": "Hide Arrows"}

        # Undo history carries the hidden flag.
        browser.js("document.getElementById('edgeArrowsToggle').click(); return true;")
        state = browser.js("return captureState();")
        assert state["globalStyles"]["arrowSize"] == 5, state["globalStyles"]
        assert state["globalStyles"]["arrowsHidden"] is True
        browser.js("document.getElementById('edgeArrowsToggle').click(); return true;")
        assert browser.js(_SHAPES_JS) == [["triangle", "none"]] * 3
        browser.js("restoreState(%s); return true;" % json.dumps(state))
        assert browser.js(_SHAPES_JS) == [["none", "none"]] * 3
        assert browser.js(_UI_JS) == {"size": "5", "label": "Show Arrows"}

        # Layout state carries the hidden flag too (still hidden after the
        # undo restore above).
        layout = browser.js("return JSON.stringify(captureNetworkState());")
        browser.js("document.getElementById('edgeArrowsToggle').click(); return true;")
        assert browser.js(_SHAPES_JS) == [["triangle", "none"]] * 3
        browser.js("applyNetworkState(%s); return true;" % layout)
        assert browser.js(_SHAPES_JS) == [["none", "none"]] * 3
        assert browser.js(_UI_JS) == {"size": "5", "label": "Show Arrows"}

        assert browser.errors() == [], browser.errors()


@REQUIRES_CDP
def test_merged_bidirectional_edge_restores_both_arrowheads(tmp_path):
    frame = pd.DataFrame({"source": ["A", "B"], "target": ["B", "A"], "weight": [10.0, 20.0]})
    page = _render(tmp_path, frame, "undirected_merged", merge_reciprocal_edges=True)
    with Page(page, width=1200, height=900) as browser:
        assert browser.wait("return typeof cy !== 'undefined' && cy.edges().length === 2;")
        assert browser.wait("return cy.edges('[bidirectional = 1]').length === 1;")
        shape = (
            "var e = cy.edges('[bidirectional = 1]');"
            "return [e.style('target-arrow-shape'), e.style('source-arrow-shape')];"
        )
        assert browser.js(shape) == ["triangle", "triangle"]

        browser.js("document.getElementById('edgeArrowsToggle').click(); return true;")
        assert browser.js(shape) == ["none", "none"]
        browser.js("document.getElementById('edgeArrowsToggle').click(); return true;")
        assert browser.js(shape) == ["triangle", "triangle"]

        _set_size(browser, 0)
        assert browser.js(shape) == ["none", "none"]
        browser.js("document.getElementById('edgeArrowsToggle').click(); return true;")
        assert browser.js(shape) == ["triangle", "triangle"]
        assert browser.errors() == [], browser.errors()
