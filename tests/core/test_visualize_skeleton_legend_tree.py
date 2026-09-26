"""Tests for the collapsible tree legend panel injected into viewer HTML.

``legend_mode='tree'`` renders exactly like ``'type'`` (the native
legend used by static exports) and additionally tags traces with
``drocatLegend`` meta so the exported interactive HTML can embed a
collapsible type -> neuron legend panel built client-side.
"""

from pathlib import Path
import sys

import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import visualize_skeleton  # noqa: E402
from visualize_skeleton import LEGEND_MODES, VisualizeSkeleton  # noqa: E402


def _make_visualizer(legend_mode='tree'):
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.background_color = 'white'
    visualizer.legend_mode = legend_mode
    return visualizer


def _tagged_figure():
    """A figure shaped like a tree export: tagged neurons, a paired
    synapse legend dummy, and a mesh pinned last."""
    fig = go.Figure()
    for i, (group, item) in enumerate(
        [('Tm3', '101'), ('Tm3', '102'), ('Mi1', '201')]
    ):
        trace = go.Scatter3d(
            x=[0, i + 1], y=[0, 1], z=[0, 1], mode='lines',
            line=dict(color='#1f77b4', width=4),
            name=group, legendgroup=group, showlegend=(i != 1),
        )
        trace.legendrank = i * 100
        trace.meta = {'drocatLegend': {
            'kind': 'neuron', 'group': group, 'item': item,
        }}
        fig.add_trace(trace)

    dummy = go.Scatter3d(
        x=[None], y=[None], z=[None], mode='markers',
        name='synapses 0 -> 1', legendgroup='synapses 0 -> 1',
        showlegend=True, marker=dict(size=10, color='rgb(44,160,44)'),
    )
    fig.add_trace(dummy)

    mesh = go.Mesh3d(
        x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0], j=[1], k=[2],
        color='#c8e6f0', opacity=0.05, name='Brain mesh', showlegend=True,
    )
    mesh.legendrank = visualize_skeleton.BRAIN_MESH_LEGEND_RANK
    fig.add_trace(mesh)
    return fig


def test_legend_modes_include_tree():
    assert 'tree' in LEGEND_MODES
    assert 'type' in LEGEND_MODES


def test_tree_custom_group_detection():
    visualizer = _make_visualizer()
    visualizer.custom_layer_names = []
    assert visualizer._tree_uses_custom_groups() is False
    visualizer.custom_layer_names = ['Group A', 'Group B']
    assert visualizer._tree_uses_custom_groups() is True


def test_tree_neuron_label_neuprint_uses_instance():
    import pandas as pd

    visualizer = _make_visualizer('male-cns:v1.0')
    visualizer.dataset = 'male-cns:v1.0'
    row = pd.Series({'bodyId': 11309, 'instance': 'aMe4_L'})
    assert visualizer._tree_neuron_label('11309', row) == '11309_aMe4_L'
    # no instance -> bare bodyId; no row -> raw neuron id
    assert visualizer._tree_neuron_label(
        '11309', pd.Series({'bodyId': 11309})) == '11309'
    assert visualizer._tree_neuron_label('11309', None) == '11309'


def test_tree_neuron_label_fafb_uses_type_and_hemisphere():
    import pandas as pd

    visualizer = _make_visualizer()
    visualizer.dataset = 'flywire_FAFB_v783'
    row = pd.Series({'bodyId': '7205759406', 'flywireType': 'Tm3',
                     'somaSide': 'L'})
    assert visualizer._tree_neuron_label('x', row) == '7205759406_Tm3_L'
    # hemisphere from the instance suffix when no side column exists
    row2 = pd.Series({'bodyId': '7205759406', 'flywireType': 'Tm3',
                      'instance': 'Tm3_R'})
    assert visualizer._tree_neuron_label('x', row2) == '7205759406_Tm3_R'
    # known type without hemisphere keeps just the type
    row3 = pd.Series({'bodyId': '7205759406', 'flywireType': 'Tm3'})
    assert visualizer._tree_neuron_label('x', row3) == '7205759406_Tm3'


def test_tree_neuron_label_cross_dataset_overlay_prefers_native_type():
    import pandas as pd

    visualizer = _make_visualizer()
    visualizer.dataset = 'flywire_FAFB_v783'
    row = pd.Series({
        'bodyId': 42274,
        'type': 'SMP227',
        'flywireType': 'CB1449,CB2843',
        'somaSide': 'L',
    })
    assert visualizer._tree_neuron_label(
        'x', row, source_dataset='male-cns:v1.0'
    ) == '42274_SMP227_L'


def test_tree_neuron_label_normalizes_legacy_names():
    """Legacy navis neuron names ('{instance} ({bodyId})') normalize to the
    same '{bodyId}_{instance}' shape as properly resolved labels."""
    import pandas as pd

    visualizer = _make_visualizer('male-cns:v1.0')
    # no matching metadata row: parse the legacy name
    assert visualizer._tree_neuron_label(
        'MeVPMe7_L (12805)', None) == '12805_MeVPMe7_L'
    # matching row wins with the proper instance column
    row = pd.Series({'bodyId': 12805, 'instance': 'MeVPMe7_L'})
    assert visualizer._tree_neuron_label(
        'MeVPMe7_L (12805)', row) == '12805_MeVPMe7_L'


def test_tree_hover_body_id_resolution():
    class _Vol:
        id = 720575940647731252  # FAFB-sized: int64-exact only

    assert VisualizeSkeleton._tree_hover_body_id(
        [_Vol()], 0, '7205759406477') == 720575940647731252
    # source_index beyond the list, empty list, un-int-able id, no list
    assert VisualizeSkeleton._tree_hover_body_id([_Vol()], 5, '1') is None
    assert VisualizeSkeleton._tree_hover_body_id([], 0, '1') is None
    assert VisualizeSkeleton._tree_hover_body_id([object()], 0, None) is None
    assert VisualizeSkeleton._tree_hover_body_id(None, 0, '1') is None


def test_tree_leaf_hover_embeds_bodyid_without_renaming_trace():
    """Ordinary tree scenes: the interactive hover reads like the bodyId
    leaf while the trace keeps its type-level name for the native legend
    (static exports never see hoverlabels)."""
    trace = go.Scatter3d(x=[1], y=[2], z=[3], mode='lines', name='Tm3')
    ok = VisualizeSkeleton._apply_tree_leaf_hover(
        trace, '720575940647731252_Tm3_L', 720575940647731252)
    assert ok is True
    assert '<b>720575940647731252_Tm3_L</b>' in trace.hovertemplate
    assert 'bodyId 720575940647731252' in trace.hovertemplate
    # legend identity untouched: hover-only change
    assert trace.name == 'Tm3'
    assert trace.customdata is None
    assert trace.legendgroup is None


def test_tree_leaf_hover_falls_back_to_leaf_label():
    """Without a resolvable bodyId the hover still names the leaf (the
    label itself starts with the bodyId) instead of the type name."""
    trace = go.Scatter3d(x=[1], y=[2], z=[3], mode='lines', name='Tm3')
    ok = VisualizeSkeleton._apply_tree_leaf_hover(
        trace, '11309_aMe4_L', None)
    assert ok is False
    assert trace.hovertemplate == '<b>11309_aMe4_L</b><extra></extra>'
    assert trace.name == 'Tm3'


def test_legend_tree_html_contains_panel_and_markers():
    html = _make_visualizer()._legend_tree_html()
    assert 'drocat-legend-tree' in html
    assert 'drocatLegend' in html  # meta key read client-side
    assert '"meshRankBase": 100000000' in html
    # theme-aware via the body class the theme switch toggles
    assert 'drocat-theme-dark' in html
    # replaces the native legend on injected pages only
    assert '.js-plotly-plot .legend{display:none !important;}' in html
    # banner-aware positioning + webdriver guard
    assert 'drocat-warning-container' in html
    assert 'navigator.webdriver' in html
    # panel sits on the right, below the theme switch; content-width (shrink
    # to fit short labels) capped at 420px / the viewport.
    assert 'position:fixed;right:10px;top:60px;' in html
    assert '"brainMeshRank": 200000000' in html
    assert '"vncMeshRank": 200000001' in html
    assert 'width:fit-content;min-width:180px;' in html
    assert 'max-width:min(420px,calc(100vw - 20px));' in html
    assert 'overflow-y:auto;overflow-x:hidden;box-sizing:border-box;' in html
    assert '.drocat-lt-items{padding-left:16px;min-width:max-content;}' in html
    # manual detector matches Plotly 6.4.0's native doubleClickDelay default
    assert '"doubleClickMs": 300' in html
    assert 'isDoubleClick' in html
    # the exported panel explains the configured timing to users
    assert 'Double-click window: under ' in html
    assert 'CONFIG.doubleClickMs' in html
    # Custom bodyId layers with skeleton + soma traces count one neuron, and
    # every group row stays expandable down to its bodyId-level children —
    # singleton groups included; only bodyId leaf rows are childless.
    assert 'custom: false' in html
    assert 'g.custom = g.custom || !!lg.customGroup;' in html
    assert 'function groupNeuronItemCount(g)' in html
    assert 'var neuronCount = neuronItemCount || (g.indices.length - g.sites.length);' in html
    assert 'attachDirectRow' not in html
    assert 'neuronItemCount === 1' not in html
    assert 'var typeNeuronCount = tt.itemOrder.length;' in html
    # explicit closed/open caret glyphs keep the direction in sync with state
    assert '\\u25B6' in html
    assert r"caret.textContent = open ? '\u25BC' : '\u25B6';" in html
    assert "row.setAttribute('aria-expanded', open ? 'true' : 'false');" in html
    # ROI meshes share an expandable group; brain and VNC remain root rows.
    assert "panel, 'ROI meshes'" in html
    assert 'm.rank < CONFIG.brainMeshRank' in html
    # pre/post site traces get leaf rows; long item lists are scroll-capped
    assert 'addSiteLeaves' in html
    assert "children.length > 10" in html
    assert '.drocat-lt-items.drocat-lt-scroll{max-height:224px;' in html
    assert 'min-width:0;overflow-y:auto;overflow-x:hidden;}' in html
    # horizontal overflow must stay mouse-reachable: the panel mounts its own
    # bar, and the axis is hidden so no engine can add a native bar under it.
    # Deleting the ::-webkit-scrollbar rule this replaced reproduced the
    # doubled bar at the same widths and the same gap, so that rule alone was
    # never enough.
    assert '::-webkit-scrollbar' not in html
    assert ('.drocat-lt-hscroll{position:sticky;bottom:0;left:0;'
            'display:none;height:10px;width:100%;cursor:pointer;'
            'background:rgba(255,255,255,0.92);border-radius:4px;}') in html
    assert ('body.drocat-theme-dark .drocat-lt-hscroll-thumb'
            '{background:rgba(255,255,255,0.4);}') in html
    assert 'function mountHScrollbar(container)' in html
    assert 'function updateHScrollbars()' in html
    assert 'mountHScrollbar(panel);' in html
    assert 'updateHScrollbars();' in html


def test_write_plotly_html_embeds_tree_only_when_requested(tmp_path):
    visualizer = _make_visualizer()

    on_path = tmp_path / 'on.html'
    visualizer._write_plotly_html(
        _tagged_figure(), str(on_path), legend_tree=True
    )
    on_html = on_path.read_text(encoding='utf-8')
    assert 'drocat-legend-tree' in on_html
    assert 'drocatLegend' in on_html  # meta serialized into the figure

    off_path = tmp_path / 'off.html'
    visualizer._write_plotly_html(_tagged_figure(), str(off_path))
    off_html = off_path.read_text(encoding='utf-8')
    assert 'drocat-legend-tree' not in off_html
    assert 'drocatLegend' in off_html  # meta is harmless without the panel


def test_singleton_custom_group_page_embeds_group_and_bodyid(tmp_path):
    """A single-neuron custom group keeps its bodyId-level data in the
    page: the client-side tree renders every group row expandable down
    to its bodyId leaf, so the figure must carry group and item labels."""
    fig = go.Figure()
    trace = go.Scatter3d(
        x=[0, 1], y=[0, 1], z=[0, 1], mode='lines',
        line=dict(color='#d62728', width=4),
        name='r1_SLP459_x1', legendgroup='r1_SLP459_x1', showlegend=True,
    )
    trace.legendrank = 0
    trace.meta = {'drocatLegend': {
        'kind': 'neuron', 'group': 'r1_SLP459_x1', 'type': 'SLP459',
        'item': '459_SLP459_R', 'color': '#d62728', 'customGroup': True,
    }}
    fig.add_trace(trace)

    visualizer = _make_visualizer()
    page = tmp_path / 'singleton.html'
    visualizer._write_plotly_html(fig, str(page), legend_tree=True)
    html = page.read_text(encoding='utf-8')
    assert 'drocat-legend-tree' in html
    assert 'r1_SLP459_x1' in html
    assert '459_SLP459_R' in html
    # no direct-row flattening: the singleton group renders like any group
    assert 'attachDirectRow' not in html


def test_tree_injection_is_idempotent(tmp_path):
    visualizer = _make_visualizer()
    page = tmp_path / 'idempotent.html'
    visualizer._write_plotly_html(
        _tagged_figure(), str(page), legend_tree=True
    )
    before = page.read_text(encoding='utf-8')
    visualizer._inject_page_extras(str(page), legend_tree=True)
    assert page.read_text(encoding='utf-8') == before


def test_tree_sort_only_when_sortkey_present():
    """The legend panel re-orders a leaf container only when its leaves
    carry an explicit sortKey (validation expansion bins). Other
    custom-group scenes keep insertion order."""
    html = _make_visualizer()._legend_tree_html()
    # the client sorts inside a truthiness guard on the stamped key
    assert 'if (lg.sortKey) {' in html
    assert 'if (!tt.sortKeys || !Object.keys(tt.sortKeys).length) { return; }' \
        in html
