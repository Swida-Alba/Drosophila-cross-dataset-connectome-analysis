"""Per-edge styling in the right panel.

The edge selection panel now carries the ribbon edge features as per-edge
overrides: Route (Follow global / Straight / Flow), Arrow Size (empty =
global; 0 hides), and Arrows (Follow global / Show / Hide).  A waypoint
edge's Flow route curves a Catmull-Rom spline through its bends instead of
the straight segments.  Overrides survive undo/redo, Save/Export Layout and
graph export/import (edge data), and the global Hide Arrows / Arrow Size 0
still wins over a per-edge Show.

Run:
    python -m pytest tests/vispath/test_vispath_network_edge_styling.py -q
"""

import json
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
    return Path(vp.create_network())


def _select(browser, *indexes):
    browser.js(
        """
        var indexes = %s;
        cy.$(':selected').unselect();
        indexes.forEach(function (i) { cy.edges()[i].select(); });
        if (indexes.length) syncSelectedGeometryInputs(cy.edges()[indexes[0]]);
        return true;
        """
        % json.dumps(list(indexes))
    )


def _set_input(browser, element_id, value):
    browser.js(
        """
        var el = document.getElementById(%s);
        el.value = %s;
        el.dispatchEvent(new Event('input', {bubbles: true}));
        return true;
        """
        % (json.dumps(element_id), json.dumps(str(value)))
    )


def _set_select(browser, element_id, value):
    browser.js(
        """
        var el = document.getElementById(%s);
        el.value = %s;
        el.dispatchEvent(new Event('change', {bubbles: true}));
        return true;
        """
        % (json.dumps(element_id), json.dumps(value))
    )


def _edge(browser, index):
    return browser.js(
        """
        var e = cy.edges()[%d];
        return {
            arrowSize: (e.data('arrowSize') === undefined || e.data('arrowSize') === null)
                ? null : e.data('arrowSize'),
            arrowMode: e.data('arrowMode') || null,
            routeStyle: e.data('routeStyle') || null,
            targetShape: e.style('target-arrow-shape'),
            arrowScale: e.style('arrow-scale'),
            curveStyle: e.style('curve-style'),
            controlPoints: e.style('control-point-distances'),
            segmentWeights: String(e.style('segment-weights') || ''),
            waypoints: e.data('hasWaypoints') ? (e.data('waypoints') || []).length : 0
        };
        """
        % index
    )


def _panel(browser):
    return browser.js(
        """
        return {
            route: document.getElementById('selEdgeRoute').value,
            size: document.getElementById('selEdgeArrowSize').value,
            sizePlaceholder: document.getElementById('selEdgeArrowSize').placeholder,
            mode: document.getElementById('selEdgeArrowMode').value
        };
        """
    )


def _segment_count(edge_state):
    return len([v for v in edge_state["segmentWeights"].split() if v])


def _triangle_frame():
    return pd.DataFrame(
        {
            "source": ["A", "B", "C"],
            "target": ["B", "C", "A"],
            "weight": [10.0, 20.0, 30.0],
        }
    )


@REQUIRES_CDP
def test_right_panel_per_edge_arrows(tmp_path):
    page = _render(tmp_path, _triangle_frame(), "per_edge_arrows")
    with Page(page, width=1400, height=900) as browser:
        assert browser.wait("return typeof cy !== 'undefined' && cy.edges().length === 3;")
        assert browser.js(
            "return ['selEdgeRoute', 'selEdgeArrowSize', 'selEdgeArrowMode']"
            ".every(function (id) { return !!document.getElementById(id); });"
        )

        _select(browser, 0)
        assert _panel(browser) == {
            "route": "",
            "size": "",
            "sizePlaceholder": "global",
            "mode": "",
        }

        # Per-edge size 18 -> arrow-scale 2 on edge 0 only.
        _set_input(browser, "selEdgeArrowSize", 18)
        assert _edge(browser, 0)["arrowSize"] == 18
        assert _edge(browser, 0)["arrowScale"] == "2"
        assert _edge(browser, 1)["arrowScale"] != "2"

        # Per-edge size 0 hides this edge only; empty follows the global.
        _set_input(browser, "selEdgeArrowSize", 0)
        assert _edge(browser, 0)["targetShape"] == "none"
        assert _edge(browser, 1)["targetShape"] == "triangle"
        _set_input(browser, "selEdgeArrowSize", "")
        assert _edge(browser, 0)["arrowSize"] is None
        assert _edge(browser, 0)["targetShape"] == "triangle"

        # Per-edge hide/show; the ribbon's global Hide wins over Show.
        _set_select(browser, "selEdgeArrowMode", "hide")
        assert _edge(browser, 0)["arrowMode"] == "hide"
        assert _edge(browser, 0)["targetShape"] == "none"
        assert _edge(browser, 1)["targetShape"] == "triangle"
        _set_select(browser, "selEdgeArrowMode", "show")
        assert _edge(browser, 0)["targetShape"] == "triangle"
        browser.js("document.getElementById('edgeArrowsToggle').click(); return true;")
        assert _edge(browser, 0)["targetShape"] == "none"
        assert _edge(browser, 1)["targetShape"] == "none"
        browser.js("document.getElementById('edgeArrowsToggle').click(); return true;")
        assert _edge(browser, 0)["targetShape"] == "triangle"
        assert _edge(browser, 1)["targetShape"] == "triangle"

        # Another edge seeds from its own state (empty = global).
        _select(browser, 1)
        assert _panel(browser) == {
            "route": "",
            "size": "",
            "sizePlaceholder": "global",
            "mode": "",
        }

        # Different per-edge sizes seed a mixed placeholder; applying one
        # value sets every selected edge.
        _select(browser, 0)
        _set_input(browser, "selEdgeArrowSize", 18)
        _select(browser, 1)
        _set_input(browser, "selEdgeArrowSize", 6)
        _select(browser, 0, 1)
        assert _panel(browser)["sizePlaceholder"] == "mixed"
        _set_input(browser, "selEdgeArrowSize", 12)
        assert _edge(browser, 0)["arrowSize"] == 12
        assert _edge(browser, 1)["arrowSize"] == 12
        assert _edge(browser, 2)["arrowSize"] is None
        _set_input(browser, "selEdgeArrowSize", "")
        assert _edge(browser, 0)["arrowSize"] is None
        assert _edge(browser, 1)["arrowSize"] is None

        assert browser.errors() == [], browser.errors()


@REQUIRES_CDP
def test_per_edge_route_and_curved_waypoints(tmp_path):
    page = _render(tmp_path, _triangle_frame(), "per_edge_route")
    with Page(page, width=1400, height=900) as browser:
        assert browser.wait("return typeof cy !== 'undefined' && cy.edges().length === 3;")
        _select(browser, 0)
        browser.js("addEdgeWaypoint(); return true;")
        bend = _edge(browser, 0)
        assert bend["waypoints"] == 1
        assert bend["curveStyle"] == "segments"
        bend_count = _segment_count(bend)
        assert bend_count == 1, bend

        # Flow curves the waypoint path (many sampled segment points) and
        # the panel seeds the edge's override.
        _set_select(browser, "selEdgeRoute", "flow")
        flow = _edge(browser, 0)
        assert flow["routeStyle"] == "flow"
        assert flow["curveStyle"] == "segments"
        assert _segment_count(flow) > 10, flow
        assert _panel(browser)["route"] == "flow"

        # Straight/follow-global keep the exact waypoint bends.
        _set_select(browser, "selEdgeRoute", "straight")
        straight = _edge(browser, 0)
        assert straight["routeStyle"] == "straight"
        assert _segment_count(straight) == 1, straight
        _set_select(browser, "selEdgeRoute", "")
        follow = _edge(browser, 0)
        assert follow["routeStyle"] is None
        assert _segment_count(follow) == 1, follow

        # Moving a waypoint keeps the chosen curved route.
        _set_select(browser, "selEdgeRoute", "flow")
        before = _segment_count(_edge(browser, 0))
        browser.js(
            """
            var e = cy.edges()[0];
            var wps = e.data('waypoints').map(function (p) {
                return {x: p.x + 25, y: p.y + 10};
            });
            e.data('waypoints', wps);
            renderEdgeRoute(e);
            return true;
            """
        )
        after = _edge(browser, 0)
        assert after["routeStyle"] == "flow"
        assert _segment_count(after) == before == _segment_count(flow)
        assert browser.errors() == [], browser.errors()


@REQUIRES_CDP
def test_per_edge_route_overrides_global_mode(tmp_path):
    frame = pd.DataFrame({"source": ["A", "C"], "target": ["B", "C"], "weight": [10.0, 1.0]})
    page = _render(tmp_path, frame, "per_edge_obstruction")
    with Page(page, width=1400, height=900) as browser:
        assert browser.wait("return typeof cy !== 'undefined' && cy.edges().length === 2;")
        browser.js(
            """
            cy.nodes().forEach(function (n) {
                n.position(n.id() === 'A' ? {x: 100, y: 300}
                          : n.id() === 'B' ? {x: 500, y: 300}
                                           : {x: 300, y: 300});
            });
            cy.zoom(1); cy.pan({x: 0, y: 0});
            refreshEdgeStyles(false);
            return true;
            """
        )
        browser.js(
            "var e = cy.edges('[source = \"A\"]'); e.select();"
            "syncSelectedGeometryInputs(e); return true;"
        )
        assert (
            browser.js("return cy.edges('[source = \"A\"]').style('curve-style');")
            == "unbundled-bezier"
        )  # global Flow bows around C

        _set_select(browser, "selEdgeRoute", "straight")
        assert browser.js("return cy.edges('[source = \"A\"]').style('curve-style');") == "straight"
        assert (
            browser.js("return cy.edges('[source = \"A\"]').style('control-point-distances');")
            is None
        )

        _set_select(browser, "selEdgeRoute", "flow")
        assert (
            browser.js("return cy.edges('[source = \"A\"]').style('curve-style');")
            == "unbundled-bezier"
        )

        # The per-edge override wins over a later global mode switch.
        browser.js("setEdgeRouteMode('straight'); return true;")
        assert (
            browser.js("return cy.edges('[source = \"A\"]').style('curve-style');")
            == "unbundled-bezier"
        )
        _set_select(browser, "selEdgeRoute", "")
        assert browser.js("return cy.edges('[source = \"A\"]').style('curve-style');") == "straight"

        # Self-loops have no per-edge Route: the select is disabled.
        browser.js(
            "var e = cy.edges().filter(function (x) {"
            " return x.source().id() === x.target().id(); });"
            "cy.$(':selected').unselect(); e.select();"
            "syncSelectedGeometryInputs(e); return true;"
        )
        assert browser.js("return document.getElementById('selEdgeRoute').disabled;") is True
        assert browser.errors() == [], browser.errors()


@REQUIRES_CDP
def test_route_control_disabled_for_reciprocal_pair(tmp_path):
    frame = pd.DataFrame(
        {
            "source": ["A", "B", "C"],
            "target": ["B", "A", "C"],
            "weight": [10.0, 20.0, 1.0],
        }
    )
    page = _render(tmp_path, frame, "route_scope")
    with Page(page, width=1400, height=900) as browser:
        assert browser.wait("return typeof cy !== 'undefined' && cy.edges().length === 3;")
        browser.js(
            "var e = cy.edges('[source = \"A\"]'); cy.$(':selected').unselect();"
            "e.select(); syncSelectedGeometryInputs(e); return true;"
        )
        assert browser.js("return document.getElementById('selEdgeRoute').disabled;") is True
        browser.js(
            "var e = cy.edges().filter(function (x) {"
            " return x.source().id() === x.target().id(); });"
            "cy.$(':selected').unselect(); e.select();"
            "syncSelectedGeometryInputs(e); return true;"
        )
        assert browser.js("return document.getElementById('selEdgeRoute').disabled;") is True

        # A waypoint edge routes per-edge even when a reciprocal half is
        # visible (the waypoint branch bypasses the pair router).
        browser.js(
            "var e = cy.edges('[source = \"A\"]'); cy.$(':selected').unselect();"
            "e.select(); syncSelectedGeometryInputs(e); addEdgeWaypoint();"
            "syncSelectedGeometryInputs(e); return true;"
        )
        assert browser.js("return document.getElementById('selEdgeRoute').disabled;") is False
        _set_select(browser, "selEdgeRoute", "flow")
        index = browser.js(
            "return cy.edges().toArray().findIndex(function (e) {"
            " return e.data('source') === 'A'; });"
        )
        assert _segment_count(_edge(browser, index)) > 10
        assert browser.errors() == [], browser.errors()


@REQUIRES_CDP
def test_per_edge_styling_persists_in_layout_and_undo(tmp_path):
    page = _render(tmp_path, _triangle_frame(), "per_edge_persist")
    with Page(page, width=1400, height=900) as browser:
        assert browser.wait("return typeof cy !== 'undefined' && cy.edges().length === 3;")
        _select(browser, 0)
        browser.js("addEdgeWaypoint(); return true;")
        _set_select(browser, "selEdgeRoute", "flow")
        _set_input(browser, "selEdgeArrowSize", 6)
        _set_select(browser, "selEdgeArrowMode", "hide")
        layout = browser.js("return JSON.stringify(captureNetworkState());")
        undo = browser.js("return JSON.stringify(captureState());")

        # Wipe the override and restore the layout state.
        browser.js(
            """
            var e = cy.edges()[0];
            e.removeData('routeStyle'); e.removeData('arrowSize');
            e.removeData('arrowMode'); e.removeData('waypoints');
            e.removeData('hasWaypoints');
            e.removeStyle('segment-distances'); e.removeStyle('segment-weights');
            e.removeStyle('curve-style'); e.removeStyle('arrow-scale');
            refreshEdgeStyles(false); return true;
            """
        )
        browser.js("applyNetworkState(%s); return true;" % layout)
        restored = _edge(browser, 0)
        assert restored["routeStyle"] == "flow"
        assert restored["arrowSize"] == 6
        assert restored["arrowMode"] == "hide"
        assert restored["waypoints"] == 1
        assert len(restored["segmentWeights"]) > 100
        assert restored["targetShape"] == "none"
        assert float(restored["arrowScale"]) == 6 / 9

        # Undo/redo carries the same per-edge fields.
        browser.js("restoreState(%s); return true;" % undo)
        undo_edge = _edge(browser, 0)
        assert undo_edge["routeStyle"] == "flow"
        assert undo_edge["arrowSize"] == 6
        assert undo_edge["arrowMode"] == "hide"
        assert len(undo_edge["segmentWeights"]) > 100
        assert undo_edge["targetShape"] == "none"
        assert browser.errors() == [], browser.errors()
