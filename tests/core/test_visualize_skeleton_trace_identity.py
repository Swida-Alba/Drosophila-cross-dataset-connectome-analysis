"""Tests for the stamped trace identity (``meta.drocatTrace``).

The renderer records what every trace *is* -- neuron skeleton, companion mesh
or connector site, plus its group/type/bodyId leaf identity -- so
``classify_traces`` never has to infer a role from a trace name, and the
profile granularities work from the stamp alone in all four legend modes and
in a page read back from disk.
"""

import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visualize_skeleton import (  # noqa: E402
    PROFILE_GRANULARITIES,
    VisualizeSkeleton,
    classify_traces,
)


def make_vis(**attrs):
    vis = object.__new__(VisualizeSkeleton)
    merged = dict(
        verbose=False, dataset='hemibrain:v1.2.1', client_type='neuprint',
        background_color='rgba(255, 255, 255, 1.0)', legend_mode='tree',
        layer_names=['KC layer'], custom_layer_names=None,
    )
    merged.update(attrs)
    for key, value in merged.items():
        setattr(vis, key, value)
    return vis


def neuron_trace(name, body_id='256', group='KC layer', type_label='KC1a',
                 item='256_aMe4', **kwargs):
    trace = go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode='lines',
                         name=name, legendgroup=kwargs.pop('legendgroup', name),
                         showlegend=True, **kwargs)
    make_vis()._stamp_trace_identity(
        trace, kind='neuron', group=group, type_label=type_label, item=item,
        body_id=body_id, layer_index=0)
    return trace


def site_trace(owner_label, owner_body_id='256', group='KC layer'):
    """A pre-site trace built the way _plot_site_group builds one."""
    trace = go.Scatter3d(
        x=[1.0, 2.0], y=[0.0, 0.0], z=[0.0, 0.0], mode='markers',
        name=f'{owner_label}_pre', showlegend=True,
        legendgroup=f'pre_post:pre:{owner_label}')
    make_vis()._stamp_site_identity(
        trace, group_layer=0, item=f'{owner_label}_pre', owner=owner_label,
        owner_body_id=owner_body_id)
    return trace


class TestStamp:
    def test_stamp_shape_and_string_ids(self):
        trace = go.Scatter3d(x=[0], y=[0], z=[0])
        make_vis()._stamp_trace_identity(
            trace, kind='neuron', group='g', type_label='t', item='i',
            body_id=12345, owner='o', owner_body_id=999, layer_index=2)
        stamped = trace.meta['drocatTrace']
        assert stamped == {
            'kind': 'neuron', 'group': 'g', 'type': 't', 'item': 'i',
            'body_id': '12345', 'owner': 'o', 'owner_body_id': '999',
            'layer_index': 2,
        }

    def test_stamp_keeps_existing_meta(self):
        trace = go.Scatter3d(x=[0], y=[0], z=[0])
        trace.meta = {'drocat_scatter_size_role': 'pre_post_site'}
        make_vis()._stamp_trace_identity(trace, kind='site', group='g')
        assert trace.meta['drocat_scatter_size_role'] == 'pre_post_site'
        assert trace.meta['drocatTrace']['kind'] == 'site'

    def test_site_stamp_uses_the_layer_root_as_group(self):
        vis = make_vis(layer_names=['APDN3 :: verified'])
        trace = go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0], j=[1],
                          k=[2])
        vis._stamp_site_identity(trace, group_layer=0, item='256_pre',
                                 owner='256', owner_body_id='256')
        assert trace.meta['drocatTrace']['group'] == 'APDN3'

    def test_site_stamp_survives_a_missing_layer_name(self):
        trace = go.Scatter3d(x=[0], y=[0], z=[0])
        make_vis(layer_names=[])._stamp_site_identity(
            trace, group_layer=7, item='x', owner='x', owner_body_id=None)
        assert trace.meta['drocatTrace']['group'] is None

    def test_stamp_round_trips_through_plotly_json(self):
        """A page read back from disk must carry the same identity."""
        figure = go.Figure(data=[neuron_trace('KC1a')])
        parsed = json.loads(figure.to_json())['data'][0]
        assert parsed['meta']['drocatTrace']['body_id'] == '256'
        roles = classify_traces([parsed])
        assert roles[0].role == 'neuron'
        assert roles[0].item == '256_aMe4'


class TestClassifierPrefersTheStamp:
    def test_neuron_named_like_an_roi_stays_a_neuron(self):
        """F3: 'LH' is also a brain region name."""
        roles = classify_traces(
            [neuron_trace('LH')], mesh_roi_names=['LH'])
        assert roles[0].role == 'neuron'
        assert roles[0].rule == 'stamped_identity'

    def test_neuron_named_like_a_dataset_template_stays_a_neuron(self):
        roles = classify_traces([neuron_trace('query_transformed_SMP227_fafb')])
        assert roles[0].role == 'neuron'

    def test_unstamped_traces_still_fall_back_to_structure(self):
        mesh = go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0], j=[1],
                         k=[2], name='Brain mesh', legendrank=2e8)
        assert classify_traces([mesh])[0].role == 'mesh'

    def test_companion_keeps_its_owner_identity(self):
        companion = go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0],
                              j=[1], k=[2])
        make_vis()._stamp_trace_identity(
            companion, kind='companion', group='KC layer',
            type_label='KC1a', item='256_aMe4', body_id='256', layer_index=0)
        roles = classify_traces([neuron_trace('KC1a'), companion])
        assert [r.role for r in roles] == ['neuron', 'companion']
        assert roles[1].group == roles[0].group
        assert roles[1].item == roles[0].item

    def test_site_role_carries_owner_and_body_id(self):
        roles = classify_traces([site_trace('256_aMe4')])
        site = roles[0]
        assert site.role == 'site'
        assert site.rule == 'site_group'
        assert site.owner == '256_aMe4'
        assert site.owner_body_id == '256'
        assert site.group == 'KC layer'


class TestGranularityPlans:
    """The four granularity levels the re-exporters offer."""

    def two_neuron_scene(self):
        """Same layer + type, two bodyIds, one unnamed companion each."""
        traces = [
            neuron_trace('KC1a', body_id='256', item='256_aMe4'),
            neuron_trace('KC1a', body_id='257', item='257_aMe5'),
        ]
        vis = make_vis()
        # The renderer assigns every companion the owner's legend group before
        # stamping, so a companion never becomes a profile of its own.
        companion = go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0],
                              j=[1], k=[2], legendgroup='KC1a')
        vis._stamp_trace_identity(companion, kind='companion',
                                  group='KC layer', type_label='KC1a',
                                  item='256_aMe4', body_id='256',
                                  layer_index=0)
        traces.insert(1, companion)
        traces.append(site_trace('KC1a', owner_body_id='257'))
        traces.append(go.Scatter3d(x=[0], y=[0], z=[0], name='synapses 1 -> 2',
                                   legendgroup='synapses 1 -> 2'))
        return vis, classify_traces(traces)

    def test_legend_granularity_uses_the_legend_group(self):
        vis, roles = self.two_neuron_scene()
        entries, background, unlabeled = vis._build_profile_plan(roles)
        assert set(entries) == {'KC1a'}
        assert sorted(entries['KC1a']) == [0, 1, 2, 3]
        assert background == [4]
        assert unlabeled == []

    def test_layer_granularity_merges_types_within_a_layer(self):
        vis, roles = self.two_neuron_scene()
        entries, background, _ = vis._build_profile_plan(roles, 'layer')
        assert set(entries) == {'KC layer'}
        assert sorted(entries['KC layer']) == [0, 1, 2, 3]
        assert background == [4]

    def test_type_granularity_splits_by_neuron_type(self):
        vis = make_vis()
        traces = [
            neuron_trace('KC1a', body_id='256', item='256_a',
                         type_label='KC1a', legendgroup='merged layer'),
            neuron_trace('KC2b', body_id='257', item='257_b',
                         type_label='KC2b', legendgroup='merged layer'),
        ]
        entries, _, _ = vis._build_profile_plan(
            classify_traces(traces), 'type')
        assert set(entries) == {'KC1a', 'KC2b'}

    def test_body_granularity_yields_one_profile_per_bodyId(self):
        vis, roles = self.two_neuron_scene()
        entries, background, unlabeled = vis._build_profile_plan(roles, 'body')
        assert sorted(entries) == sorted([
            'KC layer__KC1a__256_aMe4', 'KC layer__KC1a__257_aMe5'])
        assert unlabeled == []
        # The companion follows its skeleton, the site follows its owner's
        # bodyId, and only the synapse group is background.
        assert background == [4]
        assert sorted(entries['KC layer__KC1a__256_aMe4']) == [0, 1]
        assert sorted(entries['KC layer__KC1a__257_aMe5']) == [2, 3]

    def test_body_granularity_without_an_item_falls_back_to_the_label(self):
        vis = make_vis()
        trace = go.Scatter3d(x=[0], y=[0], z=[0], name='unnamed',
                             legendgroup='unnamed')
        vis._stamp_trace_identity(trace, kind='neuron', group='g')
        entries, _, unlabeled = vis._build_profile_plan(
            classify_traces([trace]), 'body')
        assert entries == {'unnamed': [0]}
        assert unlabeled == []

    def test_companion_without_a_legend_group_keys_off_the_stamp(self):
        vis = make_vis()
        companion = go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0],
                              j=[1], k=[2])
        vis._stamp_trace_identity(companion, kind='companion',
                                  group='KC layer', type_label='KC1a',
                                  item='256_a', body_id='256')
        entries, _, unlabeled = vis._build_profile_plan(
            classify_traces([companion]))
        assert entries == {'KC layer': [0]}
        assert unlabeled == []

    def test_invalid_granularity_is_rejected(self):
        vis, roles = self.two_neuron_scene()
        with pytest.raises(ValueError, match='granularity must be one of'):
            vis._build_profile_plan(roles, 'neuron')

    def test_granularity_levels_are_declared_once(self):
        assert PROFILE_GRANULARITIES == ('legend', 'layer', 'type', 'body')

    def test_sites_without_a_matching_owner_become_background(self):
        vis = make_vis()
        traces = [neuron_trace('KC1a', body_id='256', item='256_a'),
                  site_trace('999_orphan', owner_body_id='999')]
        entries, background, _ = vis._build_profile_plan(
            classify_traces(traces), 'body')
        assert background == [1]
        assert all(1 not in v for v in entries.values())


def test_neuron_leaf_without_stamped_body_id_reads_it_off_the_item():
    """TM VEV overlay scenes never reach the drocatTrace stamping loop —
    their drocatLegend carries kind/group/type/item but no body_id, so
    every neuron trace of a TM VEV manifest used to record body_id=None
    (measured on the 2026-09-29 reverse MCNS→FAFB scenes AND the 09-28
    forward control).  The item label is bodyId-prefixed by contract, so
    the identity resolver recovers it; non-neuron kinds are left alone."""
    import re as _re
    from visualize_skeleton import _identity_from_meta

    def legend_trace(kind, item, with_body_id=True):
        trace = go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode='lines')
        stamped = {'kind': kind, 'group': 'g', 'type': 't', 'item': item}
        if with_body_id:
            stamped['body_id'] = '256'
        trace.meta = {'drocatLegend': stamped}
        return trace

    # neuron leaf, body_id absent -> recovered from the item prefix
    got = _identity_from_meta(legend_trace('neuron', '16634_s-LNv_L',
                                           with_body_id=False))
    assert got['body_id'] == '16634'
    # explicit stamp still wins
    got = _identity_from_meta(legend_trace('neuron', '16634_s-LNv_L'))
    assert got['body_id'] == '256'
    # non-neuron kinds keep None (site/companion owners are not bodies)
    got = _identity_from_meta(legend_trace('site', '16634_pre',
                                           with_body_id=False))
    assert got['body_id'] is None
    # a non-numeric item prefix recovers nothing
    got = _identity_from_meta(legend_trace('neuron', 'ROI-alpha',
                                           with_body_id=False))
    assert got['body_id'] is None
    assert _re.match(r'^\d+$', got['body_id'] or '') is None
