"""Tests for line-mode soma visibility.

Line mode draws somas as navis spheres at ``neuron.soma`` (SWC ``label==1``).
Two mechanisms added on top: a radius-column fallback soma for skeletons that
carry no label-1 marker (male-cns marks only ~66% of neurons), and a
visibility floor that grows sub-floor tagged spheres to a fixed fraction of
the frozen scene's longest axis so the marker survives whole-CNS overview
zoom. See ``_plan/plan-line-mode-soma-visibility.md``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import navis  # noqa: E402

from skeleton_simplification import simplify_skeleton_nodes  # noqa: E402
from visualize_skeleton import (  # noqa: E402
    VisualizeSkeleton,
    classify_traces,
)


def make_vis(**attrs):
    vis = object.__new__(VisualizeSkeleton)
    merged = dict(
        verbose=False, dataset='male-cns:v1.0', client_type='neuprint',
        skeleton_mode='line',
    )
    merged.update(attrs)
    for key, value in merged.items():
        setattr(vis, key, value)
    return vis


def make_tree(rows):
    """Build a TreeNeuron from (node_id, label, x, y, z, radius, parent)."""
    df = pd.DataFrame(
        rows,
        columns=['node_id', 'label', 'x', 'y', 'z', 'radius', 'parent'])
    return navis.TreeNeuron(df)


def chain_tree(labels=None, radii=None):
    """A straight 4-node chain 1->2->3->4 with per-node overrides."""
    labels = labels or {}
    radii = radii or {}
    rows = []
    for node_id in (1, 2, 3, 4):
        parent = -1 if node_id == 1 else node_id - 1
        rows.append((node_id, labels.get(node_id, 0),
                     10.0 * node_id, 0.0, 0.0,
                     radii.get(node_id, 20.0), parent))
    return make_tree(rows)


def sphere_trace(center=0.0, radius=5.0, tag=True, name=None):
    t = go.Mesh3d(
        x=[center - radius, center + radius, center],
        y=[center - radius, center + radius, center],
        z=[center - radius, center + radius, center])
    if name:
        t.name = name
    if tag:
        meta = dict(getattr(t, 'meta', None) or {})
        meta['drocatTrace'] = {'kind': 'companion', 'soma': True}
        t.meta = meta
    return t


def trace_radius(trace):
    return VisualizeSkeleton._trace_sphere_radius(trace)


# ---------------------------------------------------------------- fallback

def test_fallback_marks_argmax_radius_node():
    n = chain_tree(radii={1: 20.0, 2: 90.0, 3: 20.0, 4: 20.0})
    assert n.soma is None  # no label-1 marker
    make_vis()._ensure_line_soma(n)
    assert int(np.atleast_1d(n.soma)[0]) == 2


def test_label1_soma_left_untouched():
    n = chain_tree(labels={3: 1}, radii={1: 20.0, 2: 90.0, 3: 30.0, 4: 20.0})
    assert int(np.atleast_1d(n.soma)[0]) == 3
    make_vis()._ensure_line_soma(n)
    assert int(np.atleast_1d(n.soma)[0]) == 3


def test_label1_marker_is_read_back_when_the_units_stamp_cleared_soma():
    """The way a cached skeleton ACTUALLY arrives: marker row, no `.soma`.

    `test_label1_soma_left_untouched` is self-consistent in a way the real
    generator is not — it builds its neuron with `navis.TreeNeuron(df)`, which
    resolves the soma from `label == 1` on the spot. Every DROCAT skeleton
    loader stamps `units` after `navis.read_swc`, and navis 1.5.0's units
    setter drops the soma assignment: measured on the local cache, 0 of 150
    male-cns and 0 of 150 hemibrain neurons arrive with a soma, while 62.7% of
    those male-cns files still carry the label-1 row. Without tier 0 the
    fallback resolves by radius and lands on the fatter branch point — 17 of
    120 sampled neurons that had the marker to begin with.
    """
    n = chain_tree(labels={3: 1}, radii={1: 20.0, 2: 90.0, 3: 30.0, 4: 20.0})
    n.soma = None                      # what the loader hands the renderer
    make_vis()._ensure_line_soma(n)
    assert int(np.atleast_1d(n.soma)[0]) == 3


def test_label1_wins_over_the_annotation_tier():
    """Precedence is the documented one: marker, then annotation, then radius.

    The annotation tier is a coordinate guess about the same soma; a label-1 row
    names the node, so it must be read first.
    """
    n = chain_tree(labels={3: 1}, radii={1: 20.0, 2: 90.0, 3: 30.0, 4: 20.0})
    n.soma = None
    vis = make_vis(neuron_dfs=[pd.DataFrame(
        {'bodyId': [int(n.id) if n.id is not None else 1],
         'somaLocation': ['[10.0, 0.0, 0.0]']})])   # node 1's coordinates
    vis._ensure_line_soma(n)
    assert int(np.atleast_1d(n.soma)[0]) == 3


def test_radius_tie_resolves_toward_root():
    # nodes 1 (root) and 4 (deep tip) share the max radius: the root-side
    # node wins because somas sit at the traced origin.
    n = chain_tree(radii={1: 50.0, 2: 10.0, 3: 10.0, 4: 50.0})
    make_vis()._ensure_line_soma(n)
    assert int(np.atleast_1d(n.soma)[0]) == 1


def test_missing_radius_column_is_noop():
    df = pd.DataFrame({
        'node_id': [1, 2], 'label': [0, 0],
        'x': [0.0, 1.0], 'y': [0.0, 0.0], 'z': [0.0, 0.0],
        'parent': [-1, 1]})
    n = navis.TreeNeuron(df)
    make_vis()._ensure_line_soma(n)
    assert n.soma is None


def test_banc_dataset_skipped():
    n = chain_tree(radii={2: 90.0})
    make_vis(dataset='banc_v888')._ensure_line_soma(n)
    assert n.soma is None


def test_tube_mode_skipped():
    n = chain_tree(radii={2: 90.0})
    make_vis(skeleton_mode='tube')._ensure_line_soma(n)
    assert n.soma is None


def test_show_soma_false_skipped():
    n = chain_tree(radii={2: 90.0})
    make_vis(show_soma=False)._ensure_line_soma(n)
    assert n.soma is None


def test_marked_soma_survives_node_reduction():
    # 300-node chain, fat soma at node 1, no label-1 marker anywhere.
    rows = [(1, 0, 0.0, 0.0, 0.0, 200.0, -1)]
    for i in range(2, 301):
        rows.append((i, 0, float(i), 0.0, 0.0, 5.0, i - 1))
    n = make_tree(rows)
    make_vis()._ensure_line_soma(n)
    assert int(np.atleast_1d(n.soma)[0]) == 1
    reduced, stats = simplify_skeleton_nodes(n, 0.5)
    assert stats['achieved_nodes'] < stats['raw_nodes']
    reduced_soma = np.atleast_1d(reduced.soma)
    assert len(reduced_soma) and int(reduced_soma[0]) == 1


def test_soma_location_annotation_wins_over_argmax():
    # All radii equal (argmax would pick node 1); the NeuPrint somaLocation
    # annotation sits at node 3's coordinates and must win.
    n = chain_tree(radii={1: 20.0, 2: 20.0, 3: 20.0, 4: 20.0})
    n.id = 100220
    vis = make_vis(neuron_dfs=[pd.DataFrame({
        'bodyId': [100220],
        'somaLocation': ['[30.0, 0.0, 0.0]'],
    })])
    vis._ensure_line_soma(n)
    assert int(np.atleast_1d(n.soma)[0]) == 3


def test_malformed_soma_location_falls_back_to_argmax():
    n = chain_tree(radii={2: 90.0})
    n.id = 100220
    vis = make_vis(neuron_dfs=[pd.DataFrame({
        'bodyId': [100220],
        'somaLocation': ['not-a-point'],
    })])
    vis._ensure_line_soma(n)
    assert int(np.atleast_1d(n.soma)[0]) == 2


def test_soma_location_lookup_is_cached_and_body_matched():
    vis = make_vis(neuron_dfs=[pd.DataFrame({
        'bodyId': [100220, '100133'],
        'somaLocation': ['[1, 2, 3]', None],
    })])
    first = vis._soma_location_lookup()
    assert first[100220] == [1.0, 2.0, 3.0]
    assert 100133 not in first  # None annotation skipped
    assert vis._soma_location_lookup() is first  # built once


# ------------------------------------------------- comma layer terms

def test_split_layer_terms_comma_bodyids():
    assert VisualizeSkeleton._split_layer_terms('100133,100218') == \
        ['100133', '100218']


def test_split_layer_terms_single_term_unchanged():
    assert VisualizeSkeleton._split_layer_terms('aMe12') == ['aMe12']


def test_split_layer_terms_path_chain_syntax_unchanged():
    assert VisualizeSkeleton._split_layer_terms('aMe12 -> LHCENT3') == \
        ['aMe12 -> LHCENT3']


def test_split_layer_terms_strips_whitespace_and_drops_empty():
    assert VisualizeSkeleton._split_layer_terms(' 100133 , , 100218 ') == \
        ['100133', '100218']


# ------------------------------------------------------------------ tag

def test_tag_stamps_unnamed_mesh3d_from_tree_source():
    vis = make_vis()
    trace = go.Mesh3d(x=[0, 1], y=[0, 1], z=[0, 1])
    vis._stamp_line_soma_tag(trace, True)
    identity = (getattr(trace, 'meta', None) or {}).get('drocatTrace', {})
    assert identity.get('soma') is True
    assert identity.get('kind') == 'companion' or 'kind' not in identity


def test_tag_skips_named_traces_and_non_mesh():
    vis = make_vis()
    named = go.Mesh3d(x=[0, 1], y=[0, 1], z=[0, 1], name='100220')
    vis._stamp_line_soma_tag(named, True)
    assert not (getattr(named, 'meta', None) or {}).get('drocatTrace', {})
    scatter = go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode='lines')
    vis._stamp_line_soma_tag(scatter, True)
    assert not (getattr(scatter, 'meta', None) or {}).get('drocatTrace', {})


def test_tag_skips_tube_mode_and_non_tree_source():
    vis = make_vis(skeleton_mode='tube')
    trace = go.Mesh3d(x=[0, 1], y=[0, 1], z=[0, 1])
    vis._stamp_line_soma_tag(trace, True)
    assert not (getattr(trace, 'meta', None) or {}).get('drocatTrace', {})
    vis = make_vis()
    vis._stamp_line_soma_tag(trace, False)
    assert not (getattr(trace, 'meta', None) or {}).get('drocatTrace', {})


def test_tag_survives_add_trace_copy_and_tree_meta_reassignment():
    # plotly's add_trace stores a copy: only pre-add stamps reach the figure.
    vis = make_vis()
    trace = go.Mesh3d(x=[0, 1], y=[0, 1], z=[0, 1])
    vis._stamp_line_soma_tag(trace, chain_tree())
    figure = go.Figure()
    figure.add_trace(trace)
    stored = figure.data[0]
    assert stored is not trace  # the copy is what the post-pass must read
    identity = (getattr(stored, 'meta', None) or {}).get('drocatTrace', {})
    assert identity.get('soma') is True

    # tree-mode legend handling rebuilds meta from the existing dict.
    tree_meta = dict(getattr(stored, 'meta', None) or {})
    tree_meta['drocatLegend'] = {'kind': 'neuron', 'group': 'L',
                                 'item': 'x'}
    stored.meta = tree_meta
    identity = (getattr(stored, 'meta', None) or {}).get('drocatTrace', {})
    assert identity.get('soma') is True


def test_tagged_sphere_classified_as_companion():
    vis = make_vis()
    neuron = go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode='lines',
                          name='100220')
    sphere = go.Mesh3d(x=[2, 3], y=[2, 3], z=[2, 3])
    vis._stamp_line_soma_tag(sphere, True)
    figure = go.Figure()
    figure.add_trace(neuron)
    figure.add_trace(sphere)
    roles = classify_traces(figure.data)
    assert roles[0].role == 'neuron'
    assert roles[1].role == 'companion'


def test_tree_mode_sphere_publishes_companion_kind():
    # Tree mode stamps drocatLegend on every trace of the layer. A tagged
    # soma sphere must publish kind='companion' so manifest counts and
    # profile plans never see the sphere as an extra neuron.
    vis = make_vis()
    sphere = go.Mesh3d(x=[0, 1], y=[0, 1], z=[0, 1])
    vis._stamp_line_soma_tag(sphere, True)
    tree_meta = dict(getattr(sphere, 'meta', None) or {})
    legend_kind = (
        'companion'
        if (tree_meta.get('drocatTrace') or {}).get('soma') else 'neuron')
    tree_meta['drocatLegend'] = {'kind': legend_kind, 'group': 'G',
                                 'item': 'leaf'}
    sphere.meta = tree_meta
    neuron = go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode='lines',
                          name='100220')
    figure = go.Figure()
    figure.add_trace(neuron)
    figure.add_trace(sphere)
    roles = classify_traces(figure.data)
    assert [r.role for r in roles] == ['neuron', 'companion']


# ---------------------------------------------------------------- floor

def test_floor_grows_undersized_sphere_to_exact_fraction():
    figure = go.Figure()
    figure.add_trace(go.Scatter3d(x=[0, 1000.0], y=[0, 1000.0], z=[0, 1000.0],
                                  mode='lines', name='n'))
    figure.add_trace(sphere_trace(center=500.0, radius=5.0))
    vis = make_vis()
    vis.fig_3d = figure
    vis._enforce_line_soma_visibility()
    ranges = VisualizeSkeleton._scene_data_ranges(figure)
    span = max(ranges[a][1] - ranges[a][0] for a in ('x', 'y', 'z'))
    floor = vis.LINE_SOMA_MIN_VISIBLE_FRACTION * span
    assert trace_radius(figure.data[1]) == pytest.approx(floor)


def test_floor_leaves_large_sphere_untouched():
    figure = go.Figure()
    figure.add_trace(go.Scatter3d(x=[0, 1000.0], y=[0, 1000.0], z=[0, 1000.0],
                                  mode='lines', name='n'))
    figure.add_trace(sphere_trace(center=500.0, radius=100.0))
    vis = make_vis()
    vis.fig_3d = figure
    vis._enforce_line_soma_visibility()
    assert trace_radius(figure.data[1]) == pytest.approx(100.0)


def test_floor_rescale_keeps_centroid():
    figure = go.Figure()
    figure.add_trace(go.Scatter3d(x=[0, 1000.0], y=[0, 1000.0], z=[0, 1000.0],
                                  mode='lines', name='n'))
    figure.add_trace(sphere_trace(center=400.0, radius=5.0))
    vis = make_vis()
    vis.fig_3d = figure
    vis._enforce_line_soma_visibility()
    stored = figure.data[1]
    for axis in ('x', 'y', 'z'):
        values = np.asarray(getattr(stored, axis), dtype=float)
        assert float(np.mean(values)) == pytest.approx(400.0)


def test_floor_noop_for_tube_mode_and_empty_figure():
    vis = make_vis(skeleton_mode='tube')
    vis.fig_3d = go.Figure()
    vis._enforce_line_soma_visibility()  # must not raise

    vis = make_vis()
    vis.fig_3d = go.Figure()  # no geometry -> _scene_data_ranges None
    vis._enforce_line_soma_visibility()  # must not raise


def test_floor_ignores_untagged_small_meshes():
    figure = go.Figure()
    figure.add_trace(go.Scatter3d(x=[0, 1000.0], y=[0, 1000.0], z=[0, 1000.0],
                                  mode='lines', name='n'))
    untagged = sphere_trace(center=500.0, radius=5.0, tag=False, name='roi')
    figure.add_trace(untagged)
    vis = make_vis()
    vis.fig_3d = figure
    vis._enforce_line_soma_visibility()
    assert trace_radius(figure.data[1]) == pytest.approx(5.0)


def test_floor_leaves_a_degenerate_sphere_alone_instead_of_dividing():
    """A zero-extent soma sphere used to cost stage 4 the WHOLE scene.

    The floor grows an undersized sphere by `floor / radius`, and a sphere whose
    vertices all coincide — which is what a marker node with radius 0 renders as
    — has radius 0, so the division raised straight through `save_figure`.
    Measured on the 2026-09-24 male-cns family run: `! scene s-CPDN3A failed:
    float division by zero`, then the same for s-CPDN3D, i.e. two of 21 parents
    lost their review scene entirely. Scaling a zero vector cannot produce extent
    either, so the honest outcome is "left as it was" and the scene survives.
    """
    figure = go.Figure()
    figure.add_trace(go.Scatter3d(x=[0, 1000.0], y=[0, 1000.0], z=[0, 1000.0],
                                  mode='lines', name='n'))
    figure.add_trace(sphere_trace(center=500.0, radius=0.0))
    vis = make_vis()
    vis.fig_3d = figure
    vis._enforce_line_soma_visibility()          # must not raise
    assert trace_radius(figure.data[1]) == 0.0


def test_a_zero_radius_marker_falls_through_to_a_usable_node():
    """`label == 1` with no thickness is not a usable marker.

    Marking it renders the degenerate sphere above, so tier 0 passes on it and
    the radius tier takes the node that actually has a profile.
    """
    rows = [(1, 0, 0.0, 0.0, 0.0, 5.0, -1),
            (2, 1, 10.0, 0.0, 0.0, 0.0, 1),      # the marker: zero radius
            (3, 0, 20.0, 0.0, 0.0, 40.0, 2)]
    n = make_tree(rows)
    n.soma = None                                # as the loader leaves it
    make_vis()._ensure_line_soma(n)
    assert int(np.atleast_1d(n.soma)[0]) == 3
