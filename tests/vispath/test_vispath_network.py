#!/usr/bin/env python
"""
Regression tests for the vispath network canvas.

Covers:
  - Structure of the generated network HTML: the operation-history
    dropdown, every mutating user operation recorded via pushHistory
    (hide node / hide edge / drag / filter / toggles / layout import),
    complete state snapshots (deep copies of data/positions plus the
    visibility toggles, edge filter and view), and the order-independent
    dead-end fixpoint over the current graph (as defined by the
    hide-edges filter).
  - Dead-end detection semantics executed in Node with headless
    Cytoscape, using the REAL functions extracted from the generated
    HTML: propagation to fixpoint, filter-defined current graph,
    hidden/self-loop edge exclusion, idempotence.
  - Undo/redo history semantics executed in Node with headless
    Cytoscape + DOM stubs, using the REAL functions extracted from the
    generated HTML: deep-copied snapshots, data/position/class restore,
    filter and toggle-flag restore, jump-to-history, redo clearing,
    history bound.

The Node harnesses read the generated HTML artifact (written by
vispath.py in this repository) and only execute code extracted from
that trusted, locally-produced file.
"""

import os
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))

from vispath_pkg.fast_graph_core import FastGraph  # noqa: E402
from vispath_pkg.vispath import VisualizePath  # noqa: E402

HERE = Path(__file__).parent


# =============================================================================
# Fixtures / helpers
# =============================================================================

def _build_network_html(output_path):
    """Generate a network HTML with the same small graph used by the
    Node harnesses: S->A->B->T source-target chain, a dead-end branch
    S->X->Y and an isolated start D->T."""
    df = pd.DataFrame(
        {
            "path_block": ["S>A>B>T", "S>X>Y", "D>T"],
            "weights": [[10, 20, 30], [5, 8], [3]],
        }
    )
    vp = VisualizePath(
        path_file=df,
        output_folder=str(output_path.parent),
        showfig=False,
        verbose=False,
        network_layout="dagre",
    )
    G = FastGraph()
    for u, v, w in [("S", "A", 10), ("A", "B", 20), ("B", "T", 30),
                    ("S", "X", 5), ("X", "Y", 8), ("D", "T", 3)]:
        G.add_edge(u, v, w)
        G.node_attrs.setdefault(u, {})["node_type"] = "intermediate"
        G.node_attrs.setdefault(v, {})["node_type"] = "intermediate"
    G.node_attrs["S"]["node_type"] = "source"
    G.node_attrs["T"]["node_type"] = "target"
    vp._plot_cytoscape_network(G, output_path=str(output_path), layout="dagre", open_browser=False)
    return output_path


@pytest.fixture(scope="module")
def network_html(tmp_path_factory):
    out = tmp_path_factory.mktemp("vispath_html") / "network_test.html"
    return _build_network_html(out)


@pytest.fixture(scope="module")
def declared_html(tmp_path_factory):
    """Network HTML in mapping-view mode: declared dataset groups replace
    the structural roles everywhere (legend, dropdown, quick actions)."""
    out = tmp_path_factory.mktemp("vispath_declared") / "declared_test.html"
    df = pd.DataFrame({"path_block": ["S>A>T"], "weights": [[5]]})
    vp = VisualizePath(
        path_file=df, output_folder=str(out.parent), showfig=False,
        verbose=False, network_layout="dagre",
        node_groups=[
            {"name": "F", "label": "FAFB", "color": "#22c55e"},
            {"name": "M", "label": "MCNS", "color": "#3b82f6"},
        ],
    )
    G = FastGraph()
    G.add_edge("S", "T", 5)
    G.node_attrs["S"] = {"node_type": "intermediate", "group": "F"}
    G.node_attrs["T"] = {"node_type": "intermediate", "group": "M"}
    vp._plot_cytoscape_network(G, output_path=str(out), layout="dagre", open_browser=False)
    return out


@pytest.fixture(scope="module")
def node_cache(tmp_path_factory):
    """Keep npm's test dependency cache inside pytest's disposable tree."""
    return tmp_path_factory.mktemp("vispath_node")


def _script_text(network_html):
    import re
    html = network_html.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert scripts, "no inline scripts found in generated HTML"
    return "\n".join(scripts)


def _ensure_node_with_cytoscape(node_cache):
    """Return the node executable, installing headless Cytoscape into a
    pytest-owned temporary directory once. Skips when node/npm or network
    access is missing."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    cy_path = node_cache / "node_modules" / "cytoscape"
    if cy_path.exists():
        return node
    npm = shutil.which("npm")
    if not npm:
        pytest.skip("npm not available for installing cytoscape")
    node_cache.mkdir(parents=True, exist_ok=True)
    npm_env = os.environ.copy()
    npm_env["npm_config_cache"] = str(node_cache / ".npm-cache")
    # Use the resolved npm path (npm.CMD on Windows): CreateProcess does not
    # resolve a bare "npm" through PATHEXT, so subprocess.run(["npm", ...])
    # raises FileNotFoundError on Windows even when npm is installed.
    res = subprocess.run(
        [npm, "install", "cytoscape@3.28.1", "--no-audit", "--no-fund", "--prefix", str(node_cache)],
        capture_output=True, text=True, timeout=600, env=npm_env,
    )
    if res.returncode != 0 or not cy_path.exists():
        pytest.skip(f"could not install cytoscape for Node tests: {res.stderr[-300:]}")
    return node


def _run_node_harness(node, harness_name, network_html, node_cache):
    harness = HERE / harness_name
    res = subprocess.run(
        [node, str(harness), str(node_cache), str(network_html)],
        capture_output=True, text=True, timeout=300,
        # Node writes UTF-8; on GBK-locale Windows the default decode would
        # raise in the reader thread and leave res.stdout as None.
        encoding="utf-8", errors="replace",
    )
    return res


# =============================================================================
# Structural checks on the generated HTML
# =============================================================================

class TestGeneratedHtmlStructure:
    def test_history_dropdown_present(self, network_html):
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        # dropdown markup lives in the HTML body; its logic in the script
        assert 'id="historyList"' in html
        assert "jumpToHistory(this.selectedIndex)" in html
        assert "▶ Current state" in html
        assert "function jumpToHistory" in js
        assert "function updateHistoryList" in js

    def test_all_mutating_operations_recorded(self, network_html):
        js = _script_text(network_html)
        # right-click hide node / hide edge (both context-menu paths)
        assert "pushHistory('Hide node')" in js
        assert "pushHistory('Hide edge')" in js
        # drag relocation is committed to history on dragfree; the pre-drag
        # stash MUST be on 'grab' — Cytoscape.js has no node-level
        # 'dragstart' event (only a core pan gesture), so wiring to
        # dragstart silently records nothing and undo cannot restore moves
        assert "pushStateHistory('Move nodes', pendingDragState)" in js
        assert "function registerDragHistory" in js
        assert "cy.on('grab', 'node'" in js
        assert "cy.on('dragstart', 'node'" not in js
        # edge filter changes are recorded per distinct value
        assert "pushHistory('Edge filter')" in js
        # layout import and label-position toggle are recorded
        assert "pushHistory('Import layout')" in js
        assert "pushHistory('Toggle label position')" in js
        # geometry editing (precise size/position) and alignment are recorded
        assert "pushHistory('Resize element')" in js
        assert "pushHistory('Align nodes')" in js
        # the three visibility toggles are recorded
        for label in ("Toggle self-loops", "Toggle orphans", "Toggle dead-ends"):
            assert f"pushHistory('{label}')" in js

    def test_visibility_control_remeasures_canvas(self, network_html):
        """Showing/restoring the visibility button must invalidate Cytoscape's
        cached container geometry before the next pointer event.

        The button is initially ``display:none`` and its first appearance can
        reflow the ribbon, moving the canvas.  Without ``cy.resize()`` the
        renderer keeps the old client-rect origin and hit-testing drifts away
        from the mouse position.
        """
        js = _script_text(network_html)
        assert "function resizeCanvasAfterVisibilityControlChange" in js
        assert "if (typeof cy !== 'undefined') cy.resize();" in js
        inline_show = re.findall(
            r"showAllBtn'\)\.style\.display = 'inline-block';\s+"
            r"resizeCanvasAfterVisibilityControlChange\(\);",
            js,
        )
        assert len(inline_show) == 4  # node/edge hide, H, and E
        assert re.search(
            r"showAllBtn'\)\.style\.display = 'none';\s+"
            r"resizeCanvasAfterVisibilityControlChange\(\);",
            js,
        )

    def test_geometry_editor_present(self, network_html):
        """Precise size/position editing and alignment helpers: numeric
        inputs in the Selected Element(s) panel, node-vs-edge row groups,
        align buttons, and the apply/align functions."""
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        # geometry inputs: X/Y/size for nodes, width for edges
        for elem_id in ('selGeomX', 'selGeomY', 'selGeomSize', 'selGeomWidth',
                        'geomNodeGroup', 'geomEdgeGroup',
                        'alignHBtn', 'alignVBtn'):
            assert f'id="{elem_id}"' in html, f'missing element {elem_id}'
        assert 'onclick="applySelectedGeometry()"' in html
        assert 'onclick="alignSelectedNodes(\'h\')"' in html
        assert 'onclick="alignSelectedNodes(\'v\')"' in html
        assert "function applySelectedGeometry" in js
        assert "function alignSelectedNodes" in js
        assert "function syncSelectedGeometryInputs" in js
        assert "function updateAlignButtons" in js
        # selection sync hooks: tap fills the inputs, dragfree refreshes
        # them after a manual drag, clearSelection resets the rows
        assert "syncSelectedGeometryInputs(element)" in js
        assert "syncSelectedGeometryInputs(evt.target)" in js
        assert "syncSelectedGeometryInputs(null)" in js
        # align-button enabled state follows any selection change (nodes
        # AND edges), and the size/position modifiers + confirm button are
        # hidden while nothing is selected
        assert "cy.on('select unselect', 'node, edge'" in js
        assert 'id="applyGeometryBtn"' in html
        assert "applyGeom.style.display = anySelected ? 'block' : 'none'" in js
        # manual edge widths are marked so they are recognizable as custom
        assert "e.data('customSize', true)" in js

    def test_selection_events_resync_geometry_controls(self, network_html):
        """A select event must re-show and repopulate geometry controls.

        Cytoscape can deliver the tap callback before it updates the
        element's selected state.  The selection listener therefore needs to
        re-sync after selection, while the final unselect must clear the rows.
        """
        js = _script_text(network_html)
        start = js.index("cy.on('select unselect', 'node, edge'")
        end = js.index("        });", start) + len("        });")
        selection_handler = js[start:end]
        assert "const selected = cy.$(':selected')" in selection_handler
        assert "syncSelectedGeometryInputs(null)" in selection_handler
        assert "syncSelectedGeometryInputs(primary)" in selection_handler

    def test_snapshots_are_complete_deep_copies(self, network_html):
        js = _script_text(network_html)
        # data()/position() return live references in Cytoscape; snapshots
        # must deep-copy them or later mutations corrupt earlier entries.
        assert "JSON.parse(JSON.stringify(n.data()))" in js
        assert "JSON.parse(JSON.stringify(e.data()))" in js
        assert "position: { x: n.position().x, y: n.position().y }" in js
        # snapshot carries the visibility toggles, filter and view so undo
        # restores the full UI state
        for field in ("deadEndsHidden", "orphansHidden", "selfLoopsHidden",
                      "hemisphereMirrorEnabled", "filterValue", "labelPosition",
                      "zoom", "pan"):
            assert field in js
        assert "syncToggleButtons()" in js
        # per-element style overrides (color/size bypasses) are captured
        # from _private.style — ele.json() does NOT expose bypasses in
        # Cytoscape 3.28.1 — and re-applied on restore, so individual
        # edits round-trip through undo/redo
        assert "function captureStyleBypass" in js
        assert "el._private && el._private.style" in js
        # computed (non-bypass) entries such as the default :active overlay
        # must never be snapshotted, or undo turns the transient drag
        # shading into a permanent bypass
        assert "if (!v || v.bypass !== true) return;" in js
        assert "style: captureStyleBypass(n)" in js
        assert "style: captureStyleBypass(e)" in js
        assert "if (n.style) cy.getElementById(n.data.id).style(n.style)" in js
        assert "if (e.style) cy.getElementById(e.data.id).style(e.style)" in js

    def test_selection_highlight_visible(self, network_html):
        """Selection feedback must be clearly visible: the default highlight
        color is a saturated orange (the old light-yellow #FFFFE0 was nearly
        invisible on the white canvas), selected nodes get a thick border
        plus an overlay halo, selected edges a thicker line."""
        js = _script_text(network_html)
        assert "#FFFFE0" not in js
        node_sel = js.index("selector: 'node:selected'")
        nblock = js[node_sel:node_sel + 800]
        assert "'border-width': '4px'" in nblock
        assert "'overlay-color': '#FF9800'" in nblock
        assert "'overlay-opacity': 0.25" in nblock
        edge_sel = js.index("selector: 'edge:selected'")
        eblock = js[edge_sel:edge_sel + 500]
        assert "'line-color': '#FF9800'" in eblock
        assert "3, 16" in eblock

    def test_dead_end_fixpoint_is_order_independent(self, network_html):
        js = _script_text(network_html)
        # additions are collected per pass and applied in batch, so the
        # result does not depend on node iteration order
        assert "const newlyDead = []" in js
        assert "newlyDead.forEach(id => deadEndSet.add(id))" in js
        # edges of the current graph are consistently defined
        assert "function isEdgeInCurrentGraph" in js
        assert "function reapplyOrphanHiding" in js
        # dead ends are re-detected whenever the graph changes
        assert "reapplyDeadEndHiding();" in js

    def test_dead_end_redetection_hooks(self, network_html):
        js = _script_text(network_html)
        # every operation that changes the visible graph must re-detect
        assert js.count("reapplyDeadEndHiding();") >= 5  # filter, hide node, self-loops, orphans, import

    def test_edge_anchors_follow_actual_node_sizes(self, network_html):
        """Individually resized nodes (geometry editor) must re-anchor their
        edges to their ACTUAL size, not the global node-size slider."""
        js = _script_text(network_html)
        # refreshEdgeStyles derives the anchor distances from each endpoint
        # node's computed width, with the global slider only as fallback
        assert "const sourceNodeSize = edge.source().numericStyle('width')" in js
        assert "const targetNodeSize = edge.target().numericStyle('width')" in js
        assert "let sourceDistance = sourceNodeSize / 2;" in js
        assert "let targetDistance = targetNodeSize / 2;" in js
        # the geometry editor refreshes edge styles after resizing
        assert "refreshEdgeStyles(false);  // keep endpoints/offsets attached to resized nodes" in js

    def test_reciprocal_detection_excludes_edge_itself(self, network_html):
        """A one-way edge must never be treated as its own parallel: the
        reciprocal check counts edges per direction, so only a DIFFERENT
        edge in the reverse direction triggers the offset branch. Otherwise
        every edge takes the reciprocal-offset branch and its arrows miss
        the node centers."""
        js = _script_text(network_html)
        # per-direction counts, not a set of all edges
        assert "const visibleEdgeCounts = new Map();" in js
        assert "const key = e.source().id() + '→' + e.target().id();" in js
        assert "visibleEdgeCounts.set(key, (visibleEdgeCounts.get(key) || 0) + 1);" in js
        # parallel check looks up the REVERSE direction only
        assert "const hasVisibleParallel = (visibleEdgeCounts.get(target + '→' + source) || 0) > 0;" in js

    def test_deadend_hidden_classes_have_display_none_style(self, network_html):
        """Regression: dead-end classes were assigned and counted, but no
        stylesheet rule hid them, so Hide Dead Ends reported counts without
        actually hiding anything."""
        js = _script_text(network_html)
        node_block = ("selector: 'node.deadend-hidden'" in js
                      and "'display': 'none'" in js)
        edge_block = ("selector: 'edge.deadend-hidden'" in js
                      and "'display': 'none'" in js)
        assert node_block, 'missing node.deadend-hidden { display: none } rule'
        assert edge_block, 'missing edge.deadend-hidden { display: none } rule'

    def test_layout_and_orphan_helpers_ignore_hidden_elements(self, network_html):
        """Layout algorithms must ignore hidden nodes/edges: the mirror
        placeholder builder and the layout/fit entry points filter through
        the class-based isVisibleElement, and orphans are recognized as dead
        ends (isOrphanNode, incl. self-loop-only nodes)."""
        js = _script_text(network_html)
        assert "function isVisibleElement" in js
        assert "function isOrphanNode" in js
        # mirror placeholders / positioning / fit / layout all use it
        assert js.count("filter(isVisibleElement)") >= 5
        # orphans are dead ends (isDeadEndNodeIn returns true on 0 in/out)
        assert "orphan: no connections in the current graph" in js
        # self-loops never count toward orphan connectivity
        assert "e.source().id() !== e.target().id()" in js

    def test_edge_list_csv_export_present(self, network_html):
        """The in-HTML Edge List CSV export button and its backing functions
        are present, and the documented column order is used."""
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        # button in the Export Controls column
        assert 'onclick="exportEdgeListCSV()"' in html
        assert "Edge List CSV" in html
        # export logic lives in the inline script
        assert "function exportEdgeListCSV" in js
        assert "function buildEdgeListCSV" in js
        assert "function getNtGroupCSV" in js
        assert "function csvEscapeField" in js
        # documented column order (source/target/weight re-importable, plus
        # color, NT, grouping info and the {key:val; ...} hover-info cells)
        assert "'source', 'target', 'weight', 'color', 'nt_type', 'nt_group'," in js
        assert "'source_group', 'target_group', 'custom_groups', 'ratio', 'probability'," in js
        assert "'edge info', 'source info', 'target info'" in js
        assert "const formatInfoCell" in js

    def test_global_style_adjustments_recorded_in_history(self, network_html):
        """Every size/adjustment control must record a history entry: node
        size, edge width, font size, arrow size, edge-width scaling method,
        metric and reciprocal offset."""
        js = _script_text(network_html)
        for label in ("Adjust node size", "Adjust edge width", "Adjust font size",
                      "Adjust arrow size", "Change edge width scale", "Change metric",
                      "Adjust reciprocal offset"):
            assert f"pushHistory('{label}')" in js, f"missing history entry: {label}"
        # snapshots carry the global style state so undo/redo restores it
        assert "globalStyles: {" in js
        assert "restoreGlobalStyles(state.globalStyles)" in js
        # restoring a snapshot must not create new history entries
        assert "restoringHistoryState = true;" in js

    def test_self_loop_curvature_adapts_to_size_and_width(self, network_html):
        """Self-loop geometry must scale with the rendered node size.

        Cytoscape renders every self-loop as a single cubic bezier (it
        forces curve-style 'bezier' on loops) whose control points sit at
        1.4 x control-point-step-size from the node center;
        control-point-distances is ignored for self-loops. With the default
        90deg sweep the endpoints land on the node circle exactly 90deg
        apart (top -> left) with tangents through the node center; a
        control-point distance of 3.0 x nodeRadius is the closest
        single-cubic approximation of the ideal 3/4 circle with the same
        radius as the node."""
        js = _script_text(network_html)
        # loop step size derives from the ACTUAL rendered node size
        assert "const loopNodeSize = edge.source().numericStyle('width')" in js
        assert "'control-point-step-size', (3.0 * loopNodeSize / 2) / 1.4" in js
        # control-point-distances must NOT be used for self-loops (ignored)
        assert "'control-point-distances', loopNodeSize" not in js
        # explicit loop orientation: 90deg sweep, start top / end left
        assert "'loop-direction', '-45deg'" in js
        assert "'loop-sweep', '-90deg'" in js
        # edge-width / node-size changes re-run refreshEdgeStyles so the
        # loop (and arrows/anchors) stay in sync
        assert "refreshEdgeStyles(false);" in js
        assert js.count("refreshEdgeStyles(false);") >= 3


class TestPanelCollapseAndRegrouping:
    """Panel collapse/show (top + right), the functional regrouping of the
    control cards, the compact export row, and systematic hover labels."""

    def test_panel_bar_present(self, network_html):
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        assert 'id="panelBar"' in html
        assert 'id="toggleControlsBtn"' in html
        assert 'id="togglePanelBtn"' in html
        assert 'onclick="toggleTopControls()"' in html
        assert 'onclick="toggleRightPanel()"' in html
        # both toggle functions exist and re-measure the canvas
        assert "function toggleTopControls" in js
        assert "function toggleRightPanel" in js
        assert "function initPanelBar" in js
        assert js.count("cy.resize()") >= 2
        # collapse classes exist in the stylesheet and are applied via JS
        assert ".controls.collapsed" in html
        assert ".main.palette-hidden" in html
        assert "body.controls-collapsed #cy" in html
        assert "classList.toggle('collapsed'" in js
        assert "classList.toggle('palette-hidden'" in js
        # panel preference persisted and restored
        assert "vispath_network_panels" in js
        assert "function persistPanelState" in js
        assert "initPanelBar();" in js

    def test_spacing_rotation_controls_live_in_layout_card(self, network_html):
        """The Horizontal/Vertical gap and Rotate spinners sit in the Layout
        ribbon page, directly after the layout algorithm selector (between the
        pageLayout and pageFilter containers). Gaps are absolute px distances
        with the icon inline in the label."""
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        order = [
            html.index('id="pageLayout"'),
            html.index('id="layoutSelector"'),
            html.index('id="nodeGapHSlider"'),
            html.index('id="nodeGapVSlider"'),
            html.index('id="rotateSlider"'),
            html.index('id="pageFilter"'),
        ]
        assert order == sorted(order), "spinners not between layoutSelector and the next card"
        for elem_id in ("nodeGapHSlider", "nodeGapVSlider", "rotateSlider"):
            tag = f'id="{elem_id}"'
            assert tag in html
            # spinner (number input), not a capped range slider
            seg = html[html.index(tag) - 60: html.index(tag) + 240]
            assert 'type="number"' in seg, f"{elem_id} is not a number spinner"
        # icon travels with the label text on one line
        assert ">Horizontal ↔</label>" in html
        assert ">Vertical ↕</label>" in html
        assert ">Rotate ↻</label>" in html
        assert 'onclick="resetSpacing()"' in html
        # single counter-clockwise button: each click applies -90°, repeated
        # clicks accumulate; snapRotation was removed with the 4-snap row
        assert 'onclick="rotateCounterClockwise()"' in html
        assert "function rotateCounterClockwise" in js
        assert "lastRotationDeg - 90" in js
        assert "snapRotation" not in html and "snapRotation" not in js

    def test_visibility_card_groups_hide_controls(self, network_html):
        """Connection Metric sits at the TOP of the Filter ribbon page,
        directly above the Hide Edges input, with Hide Orphans / Self-Loops /
        Dead Ends on the same page (Labels moved to the Layout page)."""
        html = network_html.read_text(encoding="utf-8")
        order = [
            html.index('id="pageFilter"'),
            html.index('id="metricSelect"'),
            html.index('id="ignoreEdgesInput"'),
            html.index('id="hideOrphansBtn"'),
            html.index('id="hideSelfLoopsBtn"'),
            html.index('id="hideDeadEndsBtn"'),
            html.index('id="pageShare"'),
        ]
        assert order == sorted(order), "metric + hide toggles not grouped in the Filter card"

    def test_labels_on_layout_and_canvas_simplification(self, network_html):
        """Labels (node labels + edge weights) live on the Layout page
        between Rotate and Canvas; the Canvas group is just Fit + Refresh
        Layout (the Reset button was removed); edge labels follow the active
        Connection Metric."""
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        # Refresh Layout is a STANDALONE top-bar button (cross-tab), not a
        # Canvas-group member: it renders before the Layout page content.
        assert html.index('id="refreshLayoutTopBtn"') < html.index('id="pageLayout"')
        layout_segment = html[html.index('id="pageLayout"'):html.index('id="pageFilter"')]
        assert 'onclick="refreshLayout()"' not in layout_segment
        order = [
            html.index('id="rotateSlider"'),
            html.index('id="toggleLabelsBtn"'),
            html.index('id="toggleEdgeWeightsBtn"'),
            html.index('onclick="fitGraph()"'),
            html.index('id="pageFilter"'),
        ]
        assert order == sorted(order), "Labels group not between Rotate and Canvas on the Layout page"
        # Reset button removed from the Canvas group
        assert 'onclick="resetLayout()"' not in html
        # the Filter page no longer hosts the Labels group
        filter_seg = html[html.index('id="pageFilter"'): html.index('id="pageShare"')]
        assert 'toggleLabelsBtn' not in filter_seg
        # metric-following edge labels
        assert "function updateEdgeMetricLabels" in js
        assert js.count("updateEdgeMetricLabels();") >= 2  # toggle + updateMetric
        assert "'label': 'data(display_label)'" in js

    def test_label_font_color_and_no_label_background(self, network_html):
        """Node labels never render a background: applyBackground no longer
        injects text-background-*. A Font Color control beside the background
        color drives the node label text color and adapts on theme flips
        unless the user chose a color."""
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        assert 'id="labelFontColor"' in html
        assert "applyLabelFontColor(this.value)" in html
        assert "let customLabelColor = null;" in js
        assert "if (!customLabelColor) {" in js
        assert ("const labelColor = isDark ? '#e5e7eb' : '#000000';" in js)
        assert "cy.nodes().style('color', labelColor)" in js
        # on-edge weight labels follow the same theme adaptation
        assert "cy.edges('.wlabel').style('color', labelColor)" in js
        # …and the explicit user choice recolors them together
        assert "cy.edges('.wlabel').style('color', hex)" in js
        # the old readability background is gone from applyBackground
        assert "text-background-color': isDark" not in js
        # persisted with the graph export and restored on import
        assert "labelFontColor: customLabelColor || ''" in js
        assert "if (settings.labelFontColor) {" in js

    def test_tab_cards_and_edge_label_cleanup(self, network_html):
        """Tabs use a light card background; each ribbon group is a standalone
        outlined card (no divider rules); edge weight labels are bare numbers
        (integer-safe, no unit, no background/outline box); Background and
        Font carry two distinct mini labels."""
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        # tabs: light card background, hover strengthens
        tab_rule = re.search(r"\.vp-tab \{[^}]*\}", html).group(0)
        assert "background: var(--vp-card-bg)" in tab_rule
        # ribbon groups: standalone outlined cards, divider rule gone
        group_rule = re.search(r"\.vp-ribbon-group \{[^}]*\}", html).group(0)
        assert "border: 1px solid var(--vp-border)" in group_rule
        assert "border-radius: 6px" in group_rule
        assert ".vp-ribbon-group:last-child" not in html
        assert "min-height: 127px" in html
        # edge labels: bare numbers, integer-safe (no '.0'), no unit
        assert "raw.toLocaleString('en-US')" in js
        assert "Number(edge.data('original_weight'))" in js
        # no edge label background/outline box
        assert "'text-background-color': '#fff'" not in js
        assert "'text-border-color': '#999'" not in js
        # Background & Font: two distinct mini labels
        assert ">Background</span>" in html
        assert ">Font</span>" in html
        assert ".vp-mini-label" in html

    def test_appearance_card_groups_style_controls(self, network_html):
        """Edge-width scale, size spinners, reciprocal offset and the
        background/font color controls live on the merged Layout & Style
        ribbon page as a SECOND card row (after the vp-ribbon-break); the
        metric moved to the Filter page; the refresh-edges card is gone."""
        html = network_html.read_text(encoding="utf-8")
        order = [
            html.index('id="pageLayout"'),
            html.index('id="edgeWidthScale"'),
            html.index('id="fontSizeSlider"'),
            html.index('id="nodeSizeSlider"'),
            html.index('id="edgeWidthSlider"'),
            html.index('id="arrowSizeSlider"'),
            html.index('id="reciprocalOffsetControls"'),
            html.index('id="bgToggleBtn"'),
            html.index('id="pageFilter"'),
        ]
        assert order == sorted(order), "style controls not grouped in the Style card"
        # the style cards open a deterministic second row inside the page
        assert 'class="vp-ribbon-break"' in html
        assert html.index('id="pageLayout"') < html.index('class="vp-ribbon-break"') \
            < html.index('id="edgeWidthScale"')

    def test_layout_persistence_grouped_with_exports(self, network_html):
        """Layout persistence (Save/Load browser storage, Export/Import
        Layout file) lives in the Import & Export ribbon page, grouped
        after Export/Import Graph and before the Edge List CSV."""
        html = network_html.read_text(encoding="utf-8")
        order = [
            html.index('id="pageShare"'),
            html.index('onclick="exportGraph()"'),
            html.index('onclick="saveLayout()"'),
            html.index('onclick="loadLayout()"'),
            html.index('onclick="exportLayout()"'),
            html.index('onclick="importLayout()"'),
            html.index('onclick="exportEdgeListCSV()"'),
        ]
        assert order == sorted(order), "layout persistence not grouped in the Import & Export page"
        # the Layout ribbon page keeps only layout actions (no persistence rows)
        layout_page = html[html.index('id="pageLayout"'):html.index('id="pageFilter"')]
        assert 'onclick="saveLayout()"' not in layout_page
        assert 'onclick="exportLayout()"' not in layout_page

    def test_layout_persistence_captures_all_parameters(self, network_html):
        """Save/Load must capture the FULL view state, not just positions
        and colors: per-element alpha (the opacity style bypass), edge base
        color + alpha, group definitions/memberships, background + label
        font color, metric, edge filter, hide toggles, and every global
        style control including edge-label size, spacing and rotation."""
        html = network_html.read_text(encoding="utf-8")
        assert "function captureNetworkState(" in html
        assert "function applyNetworkState(" in html
        # capture side: the alpha lives in the opacity style bypass, the
        # edge base appearance in the __base* data keys
        assert "captureStyleBypass(n)" in html
        assert "EDGE_BASE_COLOR_KEY" in html
        assert "EDGE_BASE_OPACITY_KEY" in html
        capture_block = html[html.index("function captureNetworkState("):html.index("function applyNetworkState(")]
        for key in ("version: 2", "edgeStyles", "assignedGroups", "groupDefaults",
                    "customGroups", "background: bgCtrl", "labelFontColor",
                    "edgeWeightLabels", "hemisphereMirrorEnabled", "reciprocal:",
                    "filter:", "hideToggles", "edgeLabelFontSize", "metric:",
                    "spacingX", "rotation", "background-opacity"):
            assert key in capture_block, f"capture payload missing {key!r}"
        # apply side: guarded restore paths + re-derivation of class hiding;
        # alpha restores as BODY-only (background-opacity) so the label font
        # keeps full opacity, and stale whole-element opacity is stripped
        apply_block = html[html.index("function applyNetworkState("):html.index("function saveLayout(")]
        assert "restoringHistoryState = true" in apply_block
        assert "node.style('background-opacity', item.opacity)" in apply_block
        assert "node.removeStyle('opacity')" in apply_block
        assert "setEdgeBaseAppearance(edge, item.baseColor, item.baseOpacity, canApplyBase)" in apply_block
        assert "reapplySelfLoopHiding()" in apply_block
        assert "syncToggleButtons()" in apply_block
        assert "bgCtrl.apply(state.background)" in apply_block

    def test_alpha_is_body_only_everywhere(self, network_html):
        """Alpha must never ride on whole-element 'opacity': that property
        multiplies the LABEL text too, so a faded node would fade its font.
        Every applier/read-back uses background-opacity (nodes) or
        line/arrow opacity (edges); element 'opacity' stays reserved for
        hiding (the .hidden stylesheet rule) and transient placeholders."""
        html = network_html.read_text(encoding="utf-8")
        # the alpha appliers are body-only
        assert "'background-opacity': opacity" in html          # individual + group appliers
        assert "node.style('background-opacity', sourceOpacity)" in html    # initial opacity
        assert "node.style('background-opacity', intermediateOpacity)" in html
        assert "node.style('background-opacity', targetOpacity)" in html
        assert "'line-opacity': edgeOpacity" in html            # initial edge opacity
        assert "'line-opacity': opacity" in html                # edge base appearance
        assert "'background-opacity': ((def.opacity" in html    # applyGroupLook
        # the alpha read-backs match
        assert "element.style('background-opacity')" in html    # tap read-back (node)
        assert "element.style('line-opacity')" in html          # tap read-back (edge)
        # no remaining applier SETS whole-element opacity on elements (reads
        # of the legacy key for old-build file compat are fine)
        assert "node.style('opacity'," not in html
        assert "edge.style('opacity'," not in html
        assert "'background-color': color, 'opacity':" not in html

    def test_graph_export_carries_metric_and_edge_label_size(self, network_html):
        """Import/Export Graph settings include the connection metric and
        the edge-label font size, so a graph round-trip restores them too
        (the restore runs under the history-suppression flag)."""
        html = network_html.read_text(encoding="utf-8")
        export_block = html[html.index("function exportGraph("):html.index("// ===== EDGE LIST CSV EXPORT =====")]
        assert "metric: currentMetric" in export_block
        assert "edgeLabelFontSize: globalEdgeLabelFontSize" in export_block
        import_block = html[html.index("function loadGraphFile("):html.index("function exportLayout(")]
        assert "settings.metric !== undefined" in import_block
        assert "settings.edgeLabelFontSize !== undefined" in import_block

    def test_layout_autorestore_on_open(self, network_html):
        """A saved layout for this generated file is re-applied on open
        (deferred past the init's hemisphere-caching timer), with a toast
        whose Reset action drops the saved state and reloads."""
        html = network_html.read_text(encoding="utf-8")
        init_at = html.index("initializeEdgeBaseStyles();")
        autorestore_at = html.index("Auto-restore: a layout saved for THIS generated file")
        assert autorestore_at > init_at, "auto-restore must run after the edge base init"
        block = html[autorestore_at:autorestore_at + 2000]
        assert "applyNetworkState(state)" in block
        assert "localStorage.removeItem(LAYOUT_STORAGE_KEY)" in block
        assert "window.location.reload()" in block
        assert "}, 250);" in block

    def test_layout_file_export_is_full_state_v2(self, network_html):
        """Export Layout emits the v2 full-state file (legacy positions map
        kept); Import Layout dispatches on `state` and still accepts v1
        positions-only files."""
        html = network_html.read_text(encoding="utf-8")
        export_block = html[html.index("function exportLayout("):html.index("function importLayout(")]
        assert "version: '2.0'" in export_block
        assert "type: 'network-state'" in export_block
        assert "state: captureNetworkState()" in export_block
        assert "layout: layoutData" in export_block
        import_block = html[html.index("function loadLayoutFile("):html.index("// Add CSS for edge-source indicator")]
        assert "if (!importData.layout && !importData.state)" in import_block
        assert "if (importData.state)" in import_block
        assert "applyNetworkState(importData.state)" in import_block

    def test_export_row_compact(self, network_html):
        """The narrowed scale input shares ONE flex row with the PNG and
        SVG buttons (buttons to the RIGHT of the input, not below it)."""
        html = network_html.read_text(encoding="utf-8")
        row_start = html.index('id="exportScale"')
        row_open = html.rindex("<div", 0, row_start)
        row_close = html.index("</div>", row_start)
        row = html[row_open:row_close]
        assert "exportPNG()" in row
        assert "exportSVG()" in row
        assert ">PNG<" in row and ">SVG<" in row
        assert "width: 56px" in row  # narrowed input
        assert "display: flex" in row

    def test_history_section_replaces_view_controls(self, network_html):
        """The right panel is three accordion sections (Edit / History /
        Selection & Color); the old View Controls section and the flat
        Color Settings stack are gone."""
        html = network_html.read_text(encoding="utf-8")
        assert "👁️ View Controls" not in html
        assert "🎨 Color Settings" not in html
        hist = html.index(">↩️ History<")
        assert hist < html.index('id="undoBtn"') < html.index('id="redoBtn"') \
            < html.index('id="historyList"')
        # Selection & Color accordion with live chip after History
        selection = html.index(">🎛️ Selection & Color<")
        assert hist < selection
        assert selection < html.index('id="selectionSummary"') < html.index('id="selectedInfo"')

    def test_every_control_has_hover_label(self, network_html):
        """Systematic hover labels: every button, select and input in the
        generated network page carries a non-empty title attribute."""
        html = network_html.read_text(encoding="utf-8")

        class _Collector(HTMLParser):
            def __init__(self):
                super().__init__()
                self.missing = []

            def handle_starttag(self, tag, attrs):
                if tag not in ("button", "select", "input"):
                    return
                attr = dict(attrs)
                if not (attr.get("title") or "").strip():
                    self.missing.append(
                        (tag, attr.get("id") or attr.get("onclick") or attr.get("oninput") or "?")
                    )

        collector = _Collector()
        collector.feed(html)
        assert collector.missing == [], f"controls without title: {collector.missing}"

    def test_layout_transform_history_recorded(self, network_html):
        """Spacing/rotation slider interactions commit ONE history entry per
        drag (pre-state captured on first input, pushed on change)."""
        js = _script_text(network_html)
        assert "pushStateHistory('Adjust node spacing', pendingTransformState)" in js
        assert "pushStateHistory('Rotate layout', pendingTransformState)" in js
        assert "pushHistory('Reset spacing')" in js
        assert "function onSpacingInput" in js
        assert "function onSpacingChange" in js
        assert "function onRotationInput" in js
        assert "function onRotationChange" in js
        # snapshots carry the transform trackers for exact undo/redo
        assert "spacingX: lastGapX" in js
        assert "spacingY: lastGapY" in js
        assert "rotation: lastRotationDeg" in js
        assert "resetLayoutTransformTrackers();" in js
        # layout re-runs reset the trackers (algorithm regenerates positions)
        assert js.count("resetLayoutTransformTrackers();") >= 5

    def test_layout_transforms_respect_visibility(self, network_html):
        """The transforms filter through the same isVisibleElement rule as
        the layout re-runs and operate via cy.batch()."""
        js = _script_text(network_html)
        assert js.count("cy.nodes().filter(isVisibleElement).forEach") >= 2
        assert "function visibleNodeCentroid" in js
        assert "function measureAxisGap" in js
        assert "function applyNodeGap" in js
        assert "function applyRotationDelta" in js

    def test_rotation_swaps_axis_gap_trackers(self, network_html):
        """A ±90°/270° rotation swaps the Horizontal/Vertical gap trackers
        AND refreshes the spinners — reachable from the typed Rotate field
        AND from the ↺ button (applyRotationDelta is the single path).
        Reset Spacing swaps the layout-run baseline targets to compensate
        for the current rotation."""
        js = _script_text(network_html)
        assert "function swapGapAxesIfQuarterTurn" in js
        assert "function syncGapDisplays" in js
        # the rotation applier performs the swap on its success path
        rot_seg = js[js.index("function applyRotationDelta"):js.index("function syncTransformInputs")]
        assert "swapGapAxesIfQuarterTurn(deltaDeg)" in rot_seg
        # the ↺ button reaches the swap through the same applier
        ccw_seg = js[js.index("function rotateCounterClockwise"):js.index("function resetSpacing")]
        assert "applyRotationDelta(lastRotationDeg - 90)" in ccw_seg
        # Reset Spacing maps the layout-frame baselines through the parity
        r_seg = js[js.index("function resetSpacing"):js.index("function resetLayoutTransformTrackers")]
        assert "gapAxesSwapped()" in r_seg

    def test_gaps_measured_in_layout_frame(self, network_html):
        """Gap measurement and scaling happen in the LAYOUT frame (visible
        coordinates unrotated by the accumulated rotation): a small tilt no
        longer makes the H/V spinners jump, spacing edits at a tilt scale the
        layout's own columns/rows (never shear), and the screen-axis mapping
        follows the same 45° parity as the quarter-turn tracker swap."""
        js = _script_text(network_html)
        assert "function gapAxesSwapped" in js
        # measurement unrotates the coordinates by the current rotation
        m_seg = js[js.index("function measureAxisGap"):js.index("function applyNodeGap")]
        assert "lastRotationDeg" in m_seg
        assert "visibleNodeCentroid()" in m_seg
        # scaling maps the screen axis onto the layout axis via the parity
        a_seg = js[js.index("function applyNodeGap"):js.index("function syncRotateDisplay")]
        assert "gapAxesSwapped()" in a_seg
        assert "layoutAxis" in a_seg
        # spinner sync maps the measured layout gaps through the parity
        s_seg = js[js.index("function syncTransformInputs"):js.index("function onSpacingInput")]
        assert "gapAxesSwapped()" in s_seg
        # Reset Spacing maps the layout-frame baselines through the parity
        r_seg = js[js.index("function resetSpacing"):js.index("function resetLayoutTransformTrackers")]
        assert "gapAxesSwapped()" in r_seg


class TestUiRedesign:
    """Full UI redesign: theme tokens, toast feedback, in-page dialogs,
    command-strip search, help overlay, accordions, edit-mode affordance."""

    def test_theme_tokens_and_dark_mode(self, network_html):
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        # custom properties defined for light + dark
        assert "--vp-card-bg" in html and "--vp-border" in html
        assert "body.vp-dark" in html
        # the ribbon derives from tokens via the shared page class
        assert ".vp-ribbon-page" in html and 'id="pageLayout"' in html
        # applyBackground flips the whole UI via the dark class
        assert "body.classList.toggle('vp-dark', isDark)" in js
        # the old per-selector label patching is gone
        assert "querySelectorAll('.info, .legend span, .controls label')" not in js

    def test_ribbon_tabs(self, network_html):
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        # three tabs on the command strip (Style merged into Layout & Style)
        for tab in ("tabLayout", "tabFilter", "tabShare"):
            assert f'id="{tab}"' in html
        assert 'id="tabStyle"' not in html and 'id="pageStyle"' not in html
        assert ">🔧 Layout &amp; Style</button>" in html
        assert html.count('onclick="switchTab(') == 3
        assert "function switchTab" in js
        # exactly one page open by default (Layout & Style), the others hidden
        assert 'id="pageLayout" class="vp-ribbon-page open"' in html or \
            'class="vp-ribbon-page open" id="pageLayout"' in html
        for pid in ("pageFilter", "pageShare"):
            seg_start = html.index(f'id="{pid}"')
            assert 'open' not in html[seg_start - 40:seg_start + 40].split('id=')[0], \
                f"{pid} should not be open by default"
        # active tab + collapsed ribbon persist; clicking the active tab
        # toggles the ribbon (first click collapses, the next expands) through
        # the same helper as the ⚙️ button
        assert "activeTab: activeTabName" in js
        assert "saved.activeTab" in js
        assert "ribbonCollapsed" in js
        assert "applyTopControlsCollapsed(!ribbon.classList.contains('collapsed'))" in js

    def test_div_balance(self, network_html):
        """Every opened div must be closed — a nesting slip (e.g. a group
        escaping its ribbon page) breaks tab switching while all regex
        tests on ids still pass."""
        html = network_html.read_text(encoding="utf-8")
        opens, closes = html.count("<div"), html.count("</div>")
        assert opens == closes, f"unbalanced divs: {opens} opens vs {closes} closes"

    def test_toast_system(self, network_html):
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        assert 'id="toastStack"' in html
        assert 'aria-live="polite"' in html
        assert "function createToastQueue" in js
        assert "function showToast" in js
        assert "function renderToasts" in js
        # legacy status channel delegates to toasts (hover box is
        # element-tooltips-only, written by the mouseover handlers)
        assert "function updateHoverInfo(text) {\n            showToast(text);" in js
        assert "document.getElementById('hoverInfo').textContent" not in js
        # layoutStatus line is gone; save/load report via toasts
        assert 'id="layoutStatus"' not in network_html.read_text(encoding="utf-8")
        assert "showToast('Layout saved', 'success')" in js
        assert "function showLayoutStatus" not in js

    def test_no_native_modals_left(self, network_html):
        """The network page must not open native prompt/confirm dialogs.
        The single remaining native confirm( is inside the embedded shared
        helper getExportScale (shared_controls.py, out of scope) — unused
        by the redesign's flows (they use the in-page dialog)."""
        js = _script_text(network_html)
        assert "prompt(" not in js
        # a native confirm is `confirm('...')` / `confirm("...")` — the
        # in-page confirmDialog()/confirm() calls never take a literal arg
        assert re.search(r"confirm\(['\"]", js) is None, "native confirm() call found"
        assert "function getExportScale" in js  # present but unused by flows

    def test_dialog_system(self, network_html):
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        assert 'id="vpDialogOverlay"' in html
        assert "function createDialogController" in js
        assert "function showDialog" in js
        assert "function confirmDialog" in js
        assert "function cancelDialog" in js
        # flows migrated to dialogs
        assert "function editNodeProperties" in js and "showDialog({" in js
        for flow in ("editNodeProperties", "editEdgeProperties", "addNode", "deleteCustomGroup"):
            start = js.index(f"function {flow}")
            body = js[start:start + 2400]
            assert "showDialog(" in body, f"{flow} does not open a dialog"
        # deletes are undoable via history, so they toast instead of asking
        assert "⌘Z/⌃Z to undo" in js
        # import layout offers Fit as a toast action
        assert "'Fit', onClick" in js or 'label: \'Fit\'' in js

    def test_command_strip_search(self, network_html):
        """The search box has visible ▲/▼ steppers; every navigation route
        (buttons, ↑/↓ keys) steps through the 1/N matches AND select+centers
        each one via the shared cycleSearchMatch helper."""
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        assert 'id="nodeSearchInput"' in html
        assert 'id="nodeSearchCount"' in html
        assert "function matchNodes" in js
        assert "function onSearchInput" in js
        assert "function onSearchKeydown" in js
        assert "function applySearchMatch" in js
        # visible steppers wired to the shared cycling helper
        assert 'id="searchPrevBtn"' in html and 'id="searchNextBtn"' in html
        assert 'onclick="cycleSearchMatch(-1)"' in html
        assert 'onclick="cycleSearchMatch(1)"' in html
        assert "function cycleSearchMatch" in js
        cycle_seg = js[js.index("function cycleSearchMatch"):js.index("function onSearchKeydown")]
        assert "applySearchMatch()" in cycle_seg
        # the arrow keys route through the same helper (which applies)
        key_seg = js[js.index("function onSearchKeydown"):js.index("function onSearchKeydown") + 900]
        assert "cycleSearchMatch(1)" in key_seg
        assert "cycleSearchMatch(-1)" in key_seg

    def test_metric_drives_edge_filter(self, network_html):
        """The Connection Metric lives in the Filter card and the edge
        filter evaluates the ACTIVE metric (weight / ratio / probability);
        changing the metric re-applies the filter and adapts the input
        placeholder to the filtered unit."""
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        # metric sits in the Filter card, directly above the hide input
        assert html.index(">👁️ Filter<") < html.index('id="metricSelect"') \
            < html.index('id="ignoreEdgesInput"')
        # metric-aware value + re-apply on change
        assert "function metricEdgeValue" in js
        assert "shouldIgnoreEdge(metricEdgeValue(edge))" in js
        start = js.index("function updateMetric")
        body = js[start:js.index("function updateEdgeWidths")]
        assert "applyEdgeFilter();" in body
        assert "updateIgnoredEdgesPlaceholder();" in body
        assert "function updateIgnoredEdgesPlaceholder" in js
        assert "ratio: 'OR: <0.2, >0.8 | AND: (>=0.1, <=0.5)'" in js

    def test_help_overlay(self, network_html):
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        assert 'id="helpOverlay"' in html
        assert 'id="helpBtn"' in html
        assert "function openHelp" in js
        assert "function closeHelp" in js
        assert "function toggleHelp" in js
        assert "e.key === '?'" in js
        assert "Press <strong>?</strong> for help" in html
        # filter card links into the recipes section
        assert "openHelp('recipes')" in html

    def test_accordions_and_persistence(self, network_html):
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        for acc in ("accEdit", "accHistory", "accSelection"):
            assert f'id="{acc}"' in html
        assert "function toggleAccordion" in js
        # accordion flags persist with the other panel preferences
        assert "accordions: {" in js
        assert "saved.accordions" in js
        # live selection chip wired into selection changes
        assert 'id="selectionSummary"' in html
        assert "function updateSelectionChip" in js
        assert js.count("updateSelectionChip();") >= 3
        # history entries get category icons (labels unchanged)
        assert "function historyIcon" in js
        assert "historyIcon(item.label)" in js

    def test_edit_mode_canvas_affordance(self, network_html):
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        assert 'id="editBadge"' in html
        assert "vp-editmode" in html
        assert "classList.add('vp-editmode')" in js
        assert "classList.remove('vp-editmode')" in js
        assert "document.getElementById('editBadge').style.display" in js

    def test_canvas_mouse_behavior_hints(self, network_html):
        """Two toggle buttons at the canvas TOP-RIGHT switch the drag mode:
        ✋ Pan (default — drag pans, Shift+drag box-selects) vs □ Select
        (plain drag box-selects). The switch turns cytoscape user panning
        off/on — the recipe cytoscape maps to box-select-on-drag — and the
        buttons carry active/aria-pressed state plus hover labels."""
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        assert 'id="cyHints"' in html
        # inside the canvas container, beside the edit badge
        assert html.index('id="cy"') < html.index('id="cyHints"') \
            < html.index('id="hoverInfo"')
        assert 'id="panModeBtn"' in html and 'id="selectModeBtn"' in html
        assert "onclick=\"setCanvasDragMode('pan')\"" in html
        assert "onclick=\"setCanvasDragMode('select')\"" in html
        # pan ships as the active default; select starts inactive
        pan_seg = html[html.index('id="panModeBtn"') - 60:html.index('id="panModeBtn"') + 300]
        assert "vp-cy-hint active" in pan_seg
        assert 'aria-pressed="true"' in pan_seg
        sel_seg = html[html.index('id="selectModeBtn"') - 60:html.index('id="selectModeBtn"') + 400]
        assert 'aria-pressed="false"' in sel_seg
        # the switch drives cytoscape: panning off in select mode, box
        # selection pinned on, button state + aria-pressed kept in sync
        assert "function setCanvasDragMode" in js
        fn_body = js[js.index("function setCanvasDragMode"):js.index("function setCanvasDragMode") + 1600]
        assert "cy.userPanningEnabled(mode === 'pan')" in fn_body
        assert "cy.boxSelectionEnabled(true)" in fn_body
        assert "'active-bg-opacity': mode === 'pan' ? 0.15 : 0" in fn_body
        assert "classList.toggle('active'" in fn_body
        assert "setAttribute('aria-pressed'" in fn_body
        # toggle styling + clickable buttons
        hint_rule = re.search(r"\.vp-cy-hint \{[^}]*\}", html).group(0)
        assert "cursor: pointer" in hint_rule
        assert re.search(r"\.vp-cy-hint\.active \{[^}]*\}", html)

    def test_directional_box_selection(self, network_html):
        """Box selection follows design-tool conventions: a LEFT→RIGHT frame
        selects only elements FULLY inside it (solid frame), a RIGHT→LEFT
        frame selects everything it TOUCHES (dashed frame) — and it is a
        combined select/deselect gesture: a frame over ONLY already-selected
        elements removes them from the selection, ⇧+drag per-element toggles,
        an empty frame is a no-op. Cytoscape's built-in frame is hidden
        (selection-box-opacity 0) and the result is re-derived on boxend —
        deferred, since cytoscape applies its own selection right after
        emitting the event."""
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        # overlay lives inside the canvas, hidden until a box gesture runs
        assert 'id="boxOverlay"' in html
        assert html.index('id="cy"') < html.index('id="boxOverlay"') \
            < html.index('id="cyHints"')
        overlay_rule = re.search(r"#boxOverlay \{[^}]*\}", html).group(0)
        assert "pointer-events: none" in overlay_rule
        assert "display: none" in overlay_rule
        touch_rule = re.search(r"#boxOverlay\.vp-touch \{[^}]*\}", html).group(0)
        assert "dashed" in touch_rule
        # the built-in cytoscape frame is hidden in the stylesheet; the
        # background-press circle stays as the pan affordance at half the
        # default radius (select mode hides it via setCanvasDragMode)
        assert "'selection-box-opacity': 0" in html
        assert "'active-bg-size': 15" in html
        assert "'active-bg-opacity': 0.15" in html
        # gesture wiring + deferred override
        for wire in ("cy.on('boxstart', onBoxFrameStart)",
                     "cy.on('tapdrag', onBoxFrameDrag)",
                     "cy.on('boxend', onBoxFrameEnd)"):
            assert wire in js
        assert "setTimeout(() => {" in js
        end_seg = js[js.index("function onBoxFrameEnd"):js.index("function onBoxFrameEnd") + 1200]
        assert "applyDirectionalBoxSelection" in end_seg
        # gesture start zeroes the overlay at the anchor (no stale-frame
        # flash from the previous gesture) and the dashed border has a
        # dead-zone so the style cannot flicker at the drag origin
        start_seg = js[js.index("function onBoxFrameStart"):js.index("function onBoxFrameStart") + 1400]
        assert "o.style.width = '0px'" in start_seg
        assert "o.style.height = '0px'" in start_seg
        drag_seg = js[js.index("function drawBoxOverlay"):js.index("function onBoxFrameStart")]
        assert "Math.abs(dx) > 4" in drag_seg
        # direction semantics: full containment vs touch
        assert "const full = x2 >= x1;" in js
        for fn in ("function boxNormalized", "function pointInRect",
                   "function segmentIntersectsRect", "function nodeRect",
                   "function nodeFullyInRect", "function nodeTouchesRect",
                   "function edgeSamplePoints", "function edgeFullyInRect",
                   "function edgeTouchesRect"):
            assert fn in js
        fn_seg = js[js.index("function applyDirectionalBoxSelection"):js.index("function drawBoxOverlay")]
        assert "nodeFullyInRect(n, rect) : nodeTouchesRect(n, rect)" in fn_seg
        assert "edgeFullyInRect(e, rect) : edgeTouchesRect(e, rect)" in fn_seg
        # combined select/deselect: subtract when the frame only covers
        # already-selected elements, ⇧ toggle, empty-frame no-op
        assert "inFrame.difference(base).empty()" in fn_seg
        assert "result = base.difference(inFrame);" in fn_seg
        assert "base.difference(inFrame).union(inFrame.difference(base))" in fn_seg
        assert "result = base;" in fn_seg

    def test_single_letter_shortcuts_skip_inputs_and_dialogs(self, network_html):
        js = _script_text(network_html)
        # H/E/L handlers ignore keystrokes while typing and while a dialog
        # is open (the E and L guards were missing before the redesign).
        assert js.count("if (dialogCtl.isActive()) return;") >= 4
        # the global Esc/? capture handler exists
        assert "dialogCtl.isActive()" in js
        assert "e.key === 'Escape'" in js


class TestGroupMembership:
    """Groups are live node memberships: every node carries a single-valued
    assigned_group field ('' = Unassigned), custom groups are look-only
    definitions, and the footer legend is rebuilt dynamically from that
    state (colors follow recoloring, chips include custom groups)."""

    def test_membership_field_and_helpers(self, network_html):
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        # server-side seed: every generated node carries its structural group
        # (the generated HTML embeds the nodes as JSON)
        assert '"assigned_group"' in html
        # JS backfill for imports of older exports
        assert "function normalizeAssignedGroups" in js
        assert "normalizeAssignedGroups();" in js
        # membership accessor + assignment pipeline
        assert "function groupMembers" in js
        assert "function assignNodesToGroup" in js
        assert "function assignSelectedToGroup" in js
        # group ops select via the membership field, not node_type
        assert "groupMembers('source')" in js
        assert "groupMembers('intermediate')" in js
        assert "groupMembers('target')" in js
        assert 'cy.nodes().filter(\'[node_type = "source"]\')' not in js
        assert 'cy.nodes().filter(\'[node_type = "intermediate"]\')' not in js
        assert 'cy.nodes().filter(\'[node_type = "target"]\')' not in js
        assert '[node_type = "\' + group' not in js

    def test_dynamic_legend_and_assign_row(self, network_html):
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        # legend containers: group chips rebuilt live, dataset chips static
        assert 'id="groupLegend"' in html
        assert 'id="datasetLegend"' in html
        assert 'data-group="source"' in html  # static seed for no-JS readers
        assert "function refreshLegend" in js
        # refreshLegend is wired into every color/membership mutation
        assert js.count("refreshLegend();") >= 4
        assert "function legendChip" in js
        # assign row in the Selected element(s) section
        assert 'id="assignGroupSelect"' in html
        assert 'onclick="assignSelectedToGroup()"' in html
        assert "function rebuildAssignSelect" in js
        assert "function syncAssignSelectToSelection" in js
        # Unassigned surfaces only when it has members
        assert "'unassigned'" in js
        assert "'Unassigned ('" in js
        # hover shows the group only when it differs from the structural type
        assert "data.assigned_group !== data.node_type" in js

    def test_custom_groups_are_definitions_only(self, network_html):
        html = network_html.read_text(encoding="utf-8")
        js = _script_text(network_html)
        # definitions carry look, not membership: the static ids list is gone
        assert "ids: ids" not in js
        assert "customGroups[groupName].ids" not in js
        # creation seeds a reset target (Reset All Colors restores it)
        assert "defaultColor: color" in js
        # Reset All Colors keeps groups/memberships — no more wipe
        assert "Object.keys(customGroups).forEach(key => delete customGroups[key]);" not in js
        # legacy exports: static id lists are re-materialized on import
        assert "(def.ids || [])" in js
        # edit-mode Add Node dialog carries a group choice
        assert "key: 'group', label: 'Group'" in js
        assert "assigned_group: assignedGroup" in js
        # Edge List CSV: group columns read the membership field
        assert "sourceNode.data('assigned_group') ?? sourceNode.data('node_type')" in js
        assert "membership[sourceNode.id()]" in js

    def test_declared_groups_replace_structural_roles(self, declared_html):
        """Mapping-view mode: declared dataset groups take over the node
        grouping — the structural role options disappear from the Groups
        dropdown (user convention), the legend seeds declared chips, and the
        JS switches to declared-only group lists."""
        html = declared_html.read_text(encoding="utf-8")
        js = _script_text(declared_html)
        # no template leakage; structural options are gone from the dropdown
        assert "standard_group_options_html" not in html
        assert '<option value="source">Source Nodes</option>' not in html
        # declared chips seed the legend with their group identity
        assert 'data-group="F"' in html
        assert 'data-group="M"' in html
        # structural role chips are not rendered in declared mode
        assert 'data-group="source"' not in html
        assert "const declaredGroupsActive = true;" in js
        # nodes carry their declared group as the initial membership
        assert '"assigned_group": "F"' in html
        assert '"assigned_group": "M"' in html


# =============================================================================
# Node-based logic tests (real functions extracted from the generated HTML)
# =============================================================================

class TestDeadEndLogicNode:
    def test_all_dead_end_scenarios(self, network_html, node_cache):
        node = _ensure_node_with_cytoscape(node_cache)
        res = _run_node_harness(node, "deadend_harness.js", network_html, node_cache)
        assert res.returncode == 0, (
            f"dead-end harness failed:\n{res.stdout}\n{res.stderr}"
        )
        assert "ALL DEAD-END TESTS PASSED" in res.stdout


class TestHistoryLogicNode:
    def test_all_history_scenarios(self, network_html, node_cache):
        node = _ensure_node_with_cytoscape(node_cache)
        res = _run_node_harness(node, "history_harness.js", network_html, node_cache)
        assert res.returncode == 0, (
            f"history harness failed:\n{res.stdout}\n{res.stderr}"
        )
        assert "ALL HISTORY TESTS PASSED" in res.stdout


class TestGlobalStyleHistoryNode:
    """Global style adjustments (size sliders + selects) recorded in the
    operation history, undo/redo restores them, and self-loop curvature
    follows the rendered node size / edge width."""

    def test_all_global_style_scenarios(self, network_html, node_cache):
        node = _ensure_node_with_cytoscape(node_cache)
        res = _run_node_harness(node, "globals_history_harness.js", network_html, node_cache)
        assert res.returncode == 0, (
            f"global-style harness failed:\n{res.stdout}\n{res.stderr}"
        )
        assert "ALL GLOBAL-STYLE TESTS PASSED" in res.stdout


class TestLayoutTransformsNode:
    """The inter-node spacing and rotation transforms extracted from the
    generated HTML: delta-multiplier semantics around the visible bounding
    box center, rotation composition, visible-only participation and the
    inverse reset."""

    def test_all_layout_transform_scenarios(self, network_html, node_cache):
        node = _ensure_node_with_cytoscape(node_cache)
        res = _run_node_harness(node, "layout_transform_harness.js", network_html, node_cache)
        assert res.returncode == 0, (
            f"layout-transform harness failed:\n{res.stdout}\n{res.stderr}"
        )
        assert "ALL LAYOUT-TRANSFORM TESTS PASSED" in res.stdout


class TestUiFeedbackNode:
    """The redesign's pure helpers extracted from the generated HTML: the
    toast queue (cap + ttl expiry), the dialog controller promise state
    machine, and the node-search matcher."""

    def test_all_ui_feedback_scenarios(self, network_html, node_cache):
        node = _ensure_node_with_cytoscape(node_cache)
        res = _run_node_harness(node, "ui_feedback_harness.js", network_html, node_cache)
        assert res.returncode == 0, (
            f"ui-feedback harness failed:\n{res.stdout}\n{res.stderr}"
        )
        assert "ALL UI-FEEDBACK TESTS PASSED" in res.stdout


class TestLayoutPersistenceNode:
    """The layout persistence capture/apply pair extracted from the
    generated HTML: a save -> scramble -> load round-trip restores every
    view parameter (per-element alpha, edge base color + alpha, filter,
    hide toggles, groups, global styles), old-format payloads still load,
    and loading records no history entries."""

    def test_all_persistence_scenarios(self, network_html, node_cache):
        node = _ensure_node_with_cytoscape(node_cache)
        res = _run_node_harness(node, "persistence_harness.js", network_html, node_cache)
        assert res.returncode == 0, (
            f"persistence harness failed:\n{res.stdout}\n{res.stderr}"
        )
        assert "ALL PERSISTENCE TESTS PASSED" in res.stdout

    def test_whole_page_script_parses(self, network_html, node_cache):
        """The ENTIRE inline script must parse as JavaScript. A single bad
        escape (e.g. a raw newline inside a string literal) kills the whole
        page while every regex/structural test stays green — only a real
        JS parse catches this class of bug."""
        node = _ensure_node_with_cytoscape(node_cache)
        script = _script_text(network_html)
        script_file = network_html.parent / "network_script_check.js"
        script_file.write_text(script, encoding="utf-8")
        res = subprocess.run(
            [node, "--check", str(script_file)],
            capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace",
        )
        assert res.returncode == 0, (
            f"inline network script has JS syntax errors:\n{res.stderr[-800:]}"
        )


class TestEdgeListExportNode:
    def test_all_edge_list_export_scenarios(self, network_html, node_cache):
        node = _ensure_node_with_cytoscape(node_cache)
        res = _run_node_harness(node, "edgelist_harness.js", network_html, node_cache)
        assert res.returncode == 0, (
            f"edge-list export harness failed:\n{res.stdout}\n{res.stderr}"
        )
        assert "ALL EDGE-LIST EXPORT TESTS PASSED" in res.stdout


# =============================================================================
# Input/export match on a REAL pathfinding result
# =============================================================================

class TestEdgeListExportMatchesPathfindingInput:
    """The in-HTML CSV export must carry the same (source, target, weight)
    rows that build_network aggregated from a genuine FindAllPath result.

    The network is plotted from the untrimmed graph so every conn_df edge is
    present, then the harness extracts the REAL buildEdgeListCSV from the
    generated HTML and dumps the exported rows for comparison.
    """

    ALLPATHS = (
        PROJECT_ROOT / "local_data" / "scratch_shortest_smoke"
        / "findallpath_crosscheck" / "aMe12_to_PPL101_etc_allpaths_type.csv"
    )

    def _exported_rows(self, node, node_cache, tmp_path):
        if not self.ALLPATHS.exists():
            pytest.skip(
                "pathfinding fixture not present: this test needs a prior "
                f"real FindAllPath smoke run output at {self.ALLPATHS}"
            )
        assert self.ALLPATHS.exists(), f"missing pathfinding fixture: {self.ALLPATHS}"
        vp = VisualizePath(
            path_file=str(self.ALLPATHS),
            output_folder=str(tmp_path),
            showfig=False,
            verbose=False,
            network_layout="dagre",
        )
        conn_df, G = vp.build_network()
        html_path = tmp_path / "pathfinding_network.html"
        # plot the full (untrimmed) graph so every conn_df edge is exported
        vp._plot_cytoscape_network(
            G, output_path=str(html_path), layout="dagre", open_browser=False
        )
        res = _run_node_harness(node, "edgelist_harness.js", html_path, node_cache)
        assert res.returncode == 0, (
            f"edge-list export harness failed:\n{res.stdout}\n{res.stderr}"
        )
        marker = "EXPORTED_CSV_JSON::"
        line = next(
            (l for l in res.stdout.splitlines() if l.startswith(marker)), None
        )
        assert line is not None, f"harness did not emit exported CSV:\n{res.stdout}"
        import json
        rows = json.loads(line[len(marker):])
        return conn_df, rows

    def test_export_matches_pathfinding_conn_df(self, node_cache, tmp_path):
        node = _ensure_node_with_cytoscape(node_cache)
        conn_df, rows = self._exported_rows(node, node_cache, tmp_path)

        header = rows[0]
        assert header[:3] == ["source", "target", "weight"]
        # one exported row per aggregated connection
        assert len(rows) - 1 == len(conn_df), (
            f"exported {len(rows) - 1} edges, conn_df has {len(conn_df)}"
        )

        exported = {(r[0], r[1], float(r[2])) for r in rows[1:]}
        expected = {
            (str(s), str(t), float(w))
            for s, t, w in zip(conn_df["source"], conn_df["target"], conn_df["weight"])
        }
        assert exported == expected, (
            "exported edge set differs from pathfinding input.\n"
            f"missing: {expected - exported}\nextra: {exported - expected}"
        )

    def test_export_carries_nt_and_grouping_columns(self, node_cache, tmp_path):
        node = _ensure_node_with_cytoscape(node_cache)
        _conn_df, rows = self._exported_rows(node, node_cache, tmp_path)
        header = rows[0]
        # every documented column is present and populated per row
        assert header == [
            "source", "target", "weight", "color", "nt_type", "nt_group",
            "source_group", "target_group", "custom_groups", "ratio", "probability",
            "edge info", "source info", "target info",
        ]
        for r in rows[1:]:
            assert r[3].startswith("#"), f"color not a hex string: {r[3]}"
            assert r[6] in ("source", "intermediate", "target"), r[6]
            assert r[7] in ("source", "intermediate", "target"), r[7]
            if r[4]:  # an nt_type implies a consistent nt_group
                assert r[5] in ("excitatory", "inhibitory", "modulatory", "unknown")


# =============================================================================
# Net-Viz must accept the EXPANDED edge list (same 11 columns the in-HTML
# Edge List CSV export writes) and reconstruct the same network.
# =============================================================================

class TestExpandedEdgeListReimport:
    """Feeding the expanded Edge List CSV back into VisualizePath recreates
    the same edges, metrics, NT types and node classification."""

    ALLPATHS = (
        PROJECT_ROOT / "local_data" / "scratch_shortest_smoke"
        / "findallpath_crosscheck" / "aMe12_to_PPL101_etc_allpaths_type.csv"
    )

    def _expanded_csv_from_pathfinding(self, tmp_path):
        """Build the expanded CSV exactly as the in-HTML export would."""
        vp = VisualizePath(
            path_file=str(self.ALLPATHS),
            output_folder=str(tmp_path / "orig"),
            showfig=False,
            verbose=False,
        )
        conn_df, G = vp.build_network()
        has_nt = "nt_type" in conn_df.columns
        rows = []
        for _, r in conn_df.iterrows():
            nt = r["nt_type"] if has_nt and pd.notna(r["nt_type"]) else ""
            rows.append({
                "source": r["source"],
                "target": r["target"],
                "weight": r["weight"],
                "color": "#646464",
                "nt_type": nt,
                "nt_group": "",
                "source_group": G.nodes[r["source"]].get("node_type", "intermediate"),
                "target_group": G.nodes[r["target"]].get("node_type", "intermediate"),
                "custom_groups": "",
                "ratio": r["ratio"] if pd.notna(r["ratio"]) else "",
                "probability": r["probability"] if pd.notna(r["probability"]) else "",
            })
        csv_path = tmp_path / "expanded_edge_list.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return conn_df, G, csv_path

    def test_pathfinding_export_reimports_identically(self, tmp_path):
        if not self.ALLPATHS.exists():
            pytest.skip(
                "pathfinding fixture not present: this test needs a prior "
                f"real FindAllPath smoke run output at {self.ALLPATHS}"
            )
        assert self.ALLPATHS.exists(), f"missing pathfinding fixture: {self.ALLPATHS}"
        conn_df, G, csv_path = self._expanded_csv_from_pathfinding(tmp_path)

        vp2 = VisualizePath(
            path_file=str(csv_path),
            output_folder=str(tmp_path / "reimport"),
            showfig=False,
            verbose=False,
        )
        conn2, G2 = vp2.build_network()

        # same edge set with the same weights
        assert len(conn2) == len(conn_df)
        exported = {(s, t, float(w)) for s, t, w in
                    zip(conn2["source"], conn2["target"], conn2["weight"])}
        expected = {(s, t, float(w)) for s, t, w in
                    zip(conn_df["source"], conn_df["target"], conn_df["weight"])}
        assert exported == expected

        # ratio/probability metrics survive the round trip
        assert "ratio" in conn2.columns and conn2["ratio"].notna().sum() == len(conn_df)
        assert "probability" in conn2.columns and conn2["probability"].notna().sum() == len(conn_df)

        # node classification is restored from source_group/target_group
        # (a plain edge list would classify EVERY node as "source")
        for node in G.nodes():
            assert G2.nodes[node]["node_type"] == G.nodes[node]["node_type"], node
        counts = {t for t in (G2.nodes[n]["node_type"] for n in G2.nodes())}
        assert counts == {"source", "intermediate", "target"}

    def test_nt_type_and_grouping_columns_accepted(self, tmp_path):
        """A populated nt_type column (plus the grouping columns) survives."""
        df = pd.DataFrame({
            "source": ["S", "A", "B"],
            "target": ["A", "B", "T"],
            "weight": [10, 20, 30],
            "color": ["#FF0000", "#00FF00", "#0000FF"],
            "nt_type": ["acetylcholine", "gaba", "dopamine"],
            "nt_group": ["excitatory", "inhibitory", "modulatory"],
            "source_group": ["source", "intermediate", "intermediate"],
            "target_group": ["intermediate", "intermediate", "target"],
            "custom_groups": ["", "", ""],
            "ratio": [0.5, 0.4, 0.3],
            "probability": [0.9, 0.8, 0.7],
        })
        vp = VisualizePath(
            path_file=df,
            output_folder=str(tmp_path),
            showfig=False,
            verbose=False,
        )
        conn, G = vp.build_network()

        # aggregation may reorder rows; compare per-edge and as a set
        assert set(conn["nt_type"]) == {"acetylcholine", "gaba", "dopamine"}
        nt_by_edge = {
            (s, t): nt for s, t, nt in
            zip(conn["source"], conn["target"], conn["nt_type"])
        }
        assert nt_by_edge == {("S", "A"): "acetylcholine",
                              ("A", "B"): "gaba",
                              ("B", "T"): "dopamine"}
        assert G.nodes["S"]["node_type"] == "source"
        assert G.nodes["T"]["node_type"] == "target"
        assert G.nodes["A"]["node_type"] == "intermediate"
        assert G.nodes["B"]["node_type"] == "intermediate"
        # grouping info columns never leak into conn_df as metrics
        assert "nt_group" not in conn.columns
        assert "custom_groups" not in conn.columns

    def test_hover_info_columns_round_trip(self, tmp_path):
        """The {key:val; ...} hover-info cells restore per-edge custom
        labels and one unique-union info map per node."""
        df = pd.DataFrame({
            "source": ["S1", "S2"],
            "target": ["T", "T"],
            "weight": [4, 2],
            "edge info": [
                "{weight:4 neurons; maps via:S1[male-cns·type]}",
                "{weight:2 neurons}",
            ],
            "source info": [
                "{M:S1 · male-cns (4 neurons)}",
                "{M:S2 · male-cns (2 neurons)}",
            ],
            "target info": [
                "{F:T · fafb (9 neurons)}",
                "{F:T · fafb (9 neurons)}",
            ],
        })
        vp = VisualizePath(
            path_file=df,
            output_folder=str(tmp_path),
            showfig=False,
            verbose=False,
        )
        conn, G = vp.build_network()

        # edge info restores the per-edge custom hover labels
        assert vp.edge_labels[("S1", "T")]["maps via"] == "S1[male-cns·type]"
        assert vp.edge_labels[("S1", "T")]["weight"] == "4 neurons"
        assert "maps via" not in vp.edge_labels[("S2", "T")]
        # node info is the unique union of every source/target info the
        # node appears with (T occurs as target twice — one merged entry)
        assert vp.node_dataset_info["S1"] == {"M": "S1 · male-cns (4 neurons)"}
        assert vp.node_dataset_info["S2"] == {"M": "S2 · male-cns (2 neurons)"}
        assert vp.node_dataset_info["T"] == {"F": "T · fafb (9 neurons)"}
        # info columns never leak into conn_df as metrics
        for column in ("edge info", "source info", "target info"):
            assert column not in conn.columns


class TestCustomNodeGroupsFromEdgeList:
    """Free-form source_group/target_group values become declared custom
    groups (legend chip, assign dropdown) carried by the node's 'group'
    attribute; structural role values keep overriding node_type."""

    def _build(self, tmp_path, rows):
        df = pd.DataFrame(rows)
        vp = VisualizePath(
            path_file=df,
            output_folder=str(tmp_path),
            showfig=False,
            verbose=False,
        )
        conn, G = vp.build_network()
        return vp, conn, G

    def test_free_form_groups_become_declared_groups(self, tmp_path):
        vp, conn, G = self._build(tmp_path, [
            {"source": "S", "target": "A", "weight": 10,
             "source_group": "PAM cluster", "target_group": ""},
            {"source": "A", "target": "T", "weight": 20,
             "source_group": "intermediate", "target_group": "target"},
        ])
        # The free-form name is registered as a declared group: selector-safe
        # name, raw label, palette color.
        registered = {g["name"]: g for g in vp.node_groups}
        assert set(registered) == {"PAM_cluster"}
        assert registered["PAM_cluster"]["label"] == "PAM cluster"
        assert registered["PAM_cluster"]["color"].startswith("#")
        # The node carries the group attribute; its structural role stays.
        assert G.nodes["S"].get("group") == "PAM_cluster"
        assert G.nodes["S"]["node_type"] == "source"
        # Structural roles were still applied to the other nodes and they
        # carry no custom group.
        assert G.nodes["A"]["node_type"] == "intermediate"
        assert "group" not in G.nodes["A"]
        assert G.nodes["T"]["node_type"] == "target"
        assert "group" not in G.nodes["T"]

    def test_group_registration_is_idempotent(self, tmp_path):
        vp, conn, G = self._build(tmp_path, [
            {"source": "S", "target": "T", "weight": 10,
             "source_group": "PAM cluster", "target_group": "target"},
        ])
        count = len(vp.node_groups)
        vp.build_network()
        assert len(vp.node_groups) == count

    def test_expanded_edge_list_with_extra_columns_converts(self, tmp_path):
        """Regression: an edge list carrying the optional metadata columns
        (color, groups, hover info) still converts to a network."""
        vp, conn, G = self._build(tmp_path, [
            {"source": "S", "target": "A", "weight": 10, "color": "#FF0000",
             "source_group": "PAM", "target_group": "linker",
             "edge info": "{nt:ACH}", "source info": "{M:S}",
             "target info": "{F:A}"},
            {"source": "A", "target": "T", "weight": 20, "color": "",
             "source_group": "", "target_group": "",
             "edge info": "", "source info": "", "target info": ""},
        ])
        assert len(conn) == 2
        assert vp.custom_edge_colors[("S", "A")] == "#FF0000"
        assert vp.edge_labels[("S", "A")] == {"nt": "ACH"}
        assert vp.node_dataset_info["S"] == {"M": "S"}
        assert {g["name"] for g in vp.node_groups} == {"PAM", "linker"}
        assert G.nodes["A"].get("group") == "linker"


# =============================================================================
# Uploaded edge-list colors: the file's 'color' column must win over the UI
# link color in the generated network canvas.
# =============================================================================

class TestEdgeListFileColors:
    """Per-edge colors from an uploaded edge list override the Net-Viz link
    color; edges without a file color keep the link-color fallback."""

    def _html_with_colors(self, tmp_path, colors=None):
        df = pd.DataFrame({
            "source": ["S", "A"],
            "target": ["A", "T"],
            "weight": [10, 20],
        })
        if colors is not None:
            df["color"] = colors
        vp = VisualizePath(
            path_file=df,
            output_folder=str(tmp_path),
            showfig=False,
            verbose=False,
            network_layout="dagre",
        )
        conn_df, G = vp.build_network()
        html_path = tmp_path / "colored_network.html"
        vp._plot_cytoscape_network(
            G, output_path=str(html_path), layout="dagre", open_browser=False
        )
        return vp, html_path

    def test_file_colors_reach_the_stylesheet(self, tmp_path):
        """The canvas edge style maps from per-edge data instead of baking
        the UI link color into every edge."""
        vp, html_path = self._html_with_colors(
            tmp_path, colors=["#FF0000", "#00FF00"]
        )
        js = _script_text(html_path)
        assert "'line-color': 'data(color)'" in js
        assert "'target-arrow-color': 'data(color)'" in js
        assert f"'line-color': '{vp.edge_color}'" not in js

    def test_file_colors_land_in_edge_data(self, tmp_path):
        vp, html_path = self._html_with_colors(
            tmp_path, colors=["#FF0000", "#00FF00"]
        )
        html = html_path.read_text(encoding="utf-8")
        for color in ("#FF0000", "#00FF00"):
            assert f'"color": "{color}"' in html, f"missing edge color {color}"

    def test_edges_without_file_color_keep_link_color(self, tmp_path):
        vp, html_path = self._html_with_colors(tmp_path)
        html = html_path.read_text(encoding="utf-8")
        js = _script_text(html_path)
        assert "'line-color': 'data(color)'" in js
        # both edges fall back to the configured link color
        assert html.count(f'"color": "{vp.edge_color}"') == 2, html


# =============================================================================
# Network trim for plotting: source/target reservation + threshold warning
# =============================================================================

class TestNetworkTrimForPlot:
    """Plot trimming keeps complete strong paths and avoids dangling edges."""

    def _make_vp(self, messages=None):
        vp = object.__new__(VisualizePath)
        vp._vprint = lambda msg: messages.append(msg)
        vp.path_df = None
        return vp

    def test_no_trim_when_within_limit(self):
        messages = []
        vp = self._make_vp(messages)
        vp.edgeN_limit = 500
        G = FastGraph()
        G.add_edge("S", "A", 3)
        G.add_edge("A", "T", 5)
        vp.G_network = G
        assert vp._trim_network_for_plot() is G  # unchanged
        assert messages == []

    def test_fallback_trim_keeps_source_target_corridor_and_reports_threshold(self):
        messages = []
        vp = self._make_vp(messages)
        vp.edgeN_limit = 2
        G = FastGraph()
        # Boundary edges are weak, but are part of a complete S -> T corridor.
        G.add_edge("S", "A", 1)
        G.add_edge("B", "T", 2)
        G.node_attrs["S"] = {"node_type": "source"}
        G.node_attrs["T"] = {"node_type": "target"}
        G.node_attrs["A"] = {"node_type": "intermediate"}
        G.node_attrs["B"] = {"node_type": "intermediate"}
        # Strong intermediate edges complete the corridor.  X -> Y is a
        # disconnected decoy and must not survive merely because it is strong.
        G.add_edge("A", "M", 100)
        G.add_edge("M", "B", 90)
        G.add_edge("X", "Y", 80)
        vp.G_network = G

        G_plot = vp._trim_network_for_plot()
        kept = set(G_plot.edges())
        # Endpoint edges survive only because the full corridor survives.
        assert {("S", "A"), ("B", "T")} <= kept
        # The complete corridor is retained; the disconnected decoy is cut.
        assert len(kept) == 4
        assert ("X", "Y") not in kept  # weakest non-reserved cut
        # The warning carries the weakest edge actually retained.
        assert any("applied threshold: weight >= 1" in m for m in messages), messages

    def test_trim_reservation_capped_for_degenerate_source_target_classification(self):
        """Regression for the network_early preview: with an edge-list input
        every node is classified as source/target, so the raw reservation
        would swallow the whole graph and the limit would do nothing. The
        auto-reservation is capped at edgeN_limit and the output stays
        bounded (<= 2 x edgeN_limit)."""
        messages = []
        vp = self._make_vp(messages)
        vp.edgeN_limit = 3
        G = FastGraph()
        # every node is a source or a target (degenerate classification)
        G.add_edge("A", "B", 1)
        G.add_edge("B", "C", 2)
        G.add_edge("C", "D", 3)
        G.add_edge("D", "E", 4)
        G.add_edge("E", "F", 5)
        G.add_edge("F", "G", 6)
        G.add_edge("G", "H", 7)
        G.add_edge("H", "A", 8)
        for n in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            G.node_attrs[n] = {"node_type": "source"}

        vp.G_network = G
        G_plot = vp._trim_network_for_plot()
        # No source-to-target corridor can be inferred from this degenerate
        # classification, so the ordinary edge-list limit applies.
        assert G_plot.number_of_edges() <= vp.edgeN_limit
        assert not any("source/target reservation" in m for m in messages), messages

    def test_path_based_trim_keeps_complete_paths_and_reports_threshold(self):
        messages = []
        vp = self._make_vp(messages)
        vp.edgeN_limit = 4
        G = FastGraph()
        G.add_edge("S", "A", 1)
        G.add_edge("B", "T", 2)
        G.node_attrs["S"] = {"node_type": "source"}
        G.node_attrs["T"] = {"node_type": "target"}
        G.node_attrs["A"] = {"node_type": "intermediate"}
        G.node_attrs["B"] = {"node_type": "intermediate"}
        G.add_edge("A", "M", 100)
        G.add_edge("M", "B", 90)
        G.add_edge("X", "Y", 80)
        vp.G_network = G
        # path_df: one complete path and one path whose edges are not all in
        # the graph.  The valid path must be selected as a unit.
        vp.path_df = pd.DataFrame(
            {
                "path_block": ["S->A->M->B->T", "S->A->X->Y->B->T"],
                "weights": [[1, 100, 90, 2], [1, 50, 60, 2]],
            }
        )

        G_plot = vp._trim_network_for_plot()
        kept = set(G_plot.edges())
        assert {("S", "A"), ("A", "M"), ("M", "B"), ("B", "T")} <= kept
        assert ("X", "Y") not in kept
        assert any("applied threshold" in m for m in messages), messages

    def test_path_trim_does_not_keep_a_target_tail_without_its_full_path(self):
        messages = []
        vp = self._make_vp(messages)
        vp.edgeN_limit = 3
        vp.path_df = pd.DataFrame(
            {
                "path_block": [
                    "S->Mi1->T",          # weak, independent target tail
                    "S->A->B->T",          # strong interior, weak endpoints
                ],
                "weights": [[100, 1], [1, 100, 2]],
                "path_prob": [0.01, 0.9],
            }
        )
        G = FastGraph()
        for u, v, weight in [
            ("S", "Mi1", 100), ("Mi1", "T", 1),
            ("S", "A", 1), ("A", "B", 100), ("B", "T", 2),
        ]:
            G.add_edge(u, v, weight)
        G.node_attrs["S"] = {"node_type": "source"}
        G.node_attrs["T"] = {"node_type": "target"}
        vp.G_network = G

        selected = vp._select_edges_for_plot()
        kept_edges, _boundary_capped, relaxed, selected_paths, _threshold = selected
        assert relaxed is False
        assert selected_paths == [1]
        assert set(kept_edges) == {("S", "A"), ("A", "B"), ("B", "T")}
        assert ("Mi1", "T") not in kept_edges
        visualized = vp.visualized_paths_for_export()
        assert list(visualized["path_block"]) == ["S->A->B->T"]


# =============================================================================
# save_data: the connMatrix exports can be skipped (FindAllPath keeps the
# canonical type-level matrices in data_details/conn_mat_type_*.csv)
# =============================================================================

class TestSaveDataMatrices:
    """save_data_matrices=False skips the connMatrix exports but still
    writes the connections/original_paths files."""

    def _make_vp(self, tmp_path, save_data_matrices):
        vp = object.__new__(VisualizePath)
        vp.conn_df = pd.DataFrame({
            "source": ["A", "A", "B"],
            "target": ["B", "C", "C"],
            "weight": [3, 5, 7],
            "ratio": [0.5, 0.4, 0.3],
            "probability": [0.9, 0.8, 0.7],
        })
        vp.path_df = pd.DataFrame({"path": ["A->B->C"], "length": [2], "path_prob": [0.72]})
        vp.output_folder = str(tmp_path)
        vp.base_filename = "run"
        vp.output_format = "csv"
        vp.save_data_matrices = save_data_matrices
        vp._vprint = lambda *a, **k: None
        return vp

    def test_skip_matrices_keeps_connections_and_paths(self, tmp_path):
        vp = self._make_vp(tmp_path, save_data_matrices=False)
        files = vp.save_data()
        names = [os.path.basename(f) for f in files]
        assert "run_data_connections.csv" in names
        assert "run_data_original_paths.csv" in names
        assert not any("connMatrix" in n for n in names), names

    def test_default_still_writes_matrices(self, tmp_path):
        vp = self._make_vp(tmp_path, save_data_matrices=True)
        files = vp.save_data()
        names = [os.path.basename(f) for f in files]
        assert "run_data_connMatrix_weight.csv" in names
        assert "run_data_connMatrix_ratio.csv" in names
        assert "run_data_connMatrix_prob.csv" in names


def test_empty_network_does_not_repeat_timestamp_in_folder_name(tmp_path):
    """A timestamped Net-Viz run folder contributes only one file timestamp."""
    run_folder = tmp_path / "plot-network_empty_network_20260814_170906"
    visualizer = VisualizePath(
        path_file=None,
        output_folder=str(run_folder),
        generate_empty_network=True,
        showfig=False,
        verbose=False,
    )

    output_path = Path(visualizer.generate_empty_network_html())

    assert output_path.name == "plot-network_empty_network_20260814_170906_network.html"
    assert re.findall(r"\d{8}_\d{6}", output_path.stem) == ["20260814_170906"]


def test_visualize_network_opens_generated_html_once(tmp_path, monkeypatch):
    """The network convenience method must not open the same HTML twice."""
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)

    visualizer = VisualizePath(
        path_file=pd.DataFrame({
            "path_block": ["S>T"],
            "weights": [[3]],
        }),
        output_folder=str(tmp_path),
        showfig=True,
        verbose=False,
    )
    graph = FastGraph()
    graph.add_edge("S", "T", 3)
    graph.node_attrs["S"]["node_type"] = "source"
    graph.node_attrs["T"]["node_type"] = "target"
    visualizer.conn_df = pd.DataFrame({
        "source": ["S"],
        "target": ["T"],
        "weight": [3],
    })
    visualizer.G_network = graph

    output_path = Path(visualizer.visualize_network())

    assert opened == [f"file://{output_path.resolve()}"]


# =============================================================================
# Visualization Edge Limit: the heatmap must consume the SAME complete-path /
# corridor edge set as the network.
# =============================================================================

class TestVisualizationEdgeLimitConsistency:
    """The shared selector gives the heatmap exactly the network edge set."""

    def _make_vp(self, messages=None):
        vp = object.__new__(VisualizePath)
        vp._vprint = lambda msg: messages.append(msg)
        vp.path_df = None
        return vp

    def _graph_and_conn(self):
        G = FastGraph()
        G.add_edge("S", "A", 1)     # weak source boundary
        G.add_edge("A", "T", 2)     # weak target boundary
        G.add_edge("A", "M", 100)   # strong intermediate
        G.add_edge("M", "B", 90)    # strong intermediate
        G.add_edge("B", "C", 80)    # weakest intermediate — cut
        G.node_attrs["S"] = {"node_type": "source"}
        G.node_attrs["T"] = {"node_type": "target"}
        conn_df = pd.DataFrame({
            "source": ["S", "A", "A", "M", "B"],
            "target": ["A", "T", "M", "B", "C"],
            "weight": [1, 2, 100, 90, 80],
        })
        return G, conn_df

    def test_heatmap_filter_uses_same_edge_set_as_network(self):
        messages = []
        vp = self._make_vp(messages)
        vp.edgeN_limit = 2
        vp.G_network, conn_df = self._graph_and_conn()

        filtered = vp._filter_conn_df_for_plot(conn_df)
        kept = set(zip(filtered["source"], filtered["target"]))
        # the weak boundary edges survive as part of the inferred corridor
        assert {("S", "A"), ("A", "T")} <= kept, kept
        # the weakest intermediate edge is cut in the heatmap too
        assert ("B", "C") not in kept, kept
        # identical to the edge set the network draws
        G_plot = vp._trim_network_for_plot()
        assert set(G_plot.edges()) == kept

    def test_selector_keeps_weak_source_target_edges_only_on_corridor(self):
        messages = []
        vp = self._make_vp(messages)
        vp.edgeN_limit = 2
        vp.G_network, _ = self._graph_and_conn()
        selected = vp._select_edges_for_plot()
        assert selected is not None
        kept_edges, boundary_capped, relaxed, selected_paths, threshold = selected
        assert {("S", "A"), ("A", "T")} <= set(kept_edges)
        assert ("B", "C") not in set(kept_edges)
        assert boundary_capped is False
        assert relaxed is False
        assert selected_paths is None          # weight-based fallback branch
        assert threshold == 1                   # weakest edge in the corridor
