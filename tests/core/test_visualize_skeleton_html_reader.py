"""Tests for reading a figure back out of an exported DROCAT HTML page.

``plotly.io.read_html`` does not exist in the pinned plotly release, so the
skeleton re-exporters parse ``Plotly.newPlot(...)`` themselves. A self-contained
page carries that literal twice (the embedded plotly.js bundle mentions it),
which is the trap these tests pin down.
"""

import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visualize_skeleton import (  # noqa: E402
    VisualizeSkeleton,
    classify_traces,
    figure_from_plotly_html,
    figure_payload_from_html,
)


def make_vis(**attrs):
    vis = object.__new__(VisualizeSkeleton)
    merged = dict(
        verbose=False, dataset='hemibrain:v1.2.1', client_type='neuprint',
        background_color='rgba(255, 255, 255, 1.0)', legend_mode='tree',
        layer_names=['KC layer'], custom_layer_names=None, save_folder='',
    )
    merged.update(attrs)
    for key, value in merged.items():
        setattr(vis, key, value)
    vis._in_page_warning_html = lambda: ''
    return vis


def tree_figure():
    """Two tagged neurons (one with a companion), a swatch and a brain mesh."""
    fig = go.Figure()
    vis = make_vis()
    for body_id, item in (('256', '256_aMe4'), ('257', '257_aMe5')):
        trace = go.Scatter3d(
            x=[0, 100, 200], y=[0, 10, 20], z=[0, 5, 300], mode='lines',
            name='KC1a', legendgroup='KC1a', showlegend=(body_id == '256'),
            line=dict(color='rgba(31,119,180,0.85)'))
        trace.legendrank = 100
        trace.meta = {'drocatLegend': {'kind': 'neuron', 'group': 'KC1a',
                                       'item': item}}
        vis._stamp_trace_identity(trace, kind='neuron', group='KC layer',
                                  type_label='KC1a', item=item,
                                  body_id=body_id)
        fig.add_trace(trace)
    soma = go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0], j=[1],
                     k=[2], legendgroup='KC1a', opacity=0.5,
                     color='rgba(31,119,180,0.85)')
    vis._stamp_trace_identity(soma, kind='companion', group='KC layer',
                              type_label='KC1a', item='257_aMe5',
                              body_id='257')
    fig.add_trace(soma)
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                              name='synapses 256 -> 257', showlegend=True,
                              legendgroup='synapses 256 -> 257'))
    mesh = go.Mesh3d(x=[0, 500, 500, 0], y=[0, 0, 500, 500], z=[0, 0, 0, 400],
                     i=[0, 0], j=[1, 2], k=[2, 3], name='Brain mesh',
                     color='rgba(200,230,240,0.3)', showlegend=True)
    mesh.legendrank = 2e8
    fig.add_trace(mesh)
    fig.update_layout(scene=dict(aspectmode='data',
                                camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))),
                      title='tree scene')
    return fig


@pytest.fixture(scope='module')
def page(tmp_path_factory):
    path = tmp_path_factory.mktemp('page') / 'scene.html'
    make_vis()._write_plotly_html(
        tree_figure(), str(path), theme_toggle=True, legend_tree=True,
        freeze_view=True, include_plotlyjs=True,
        config={'displayModeBar': True, 'scrollZoom': True})
    return path


class TestPayloadReader:
    def test_ignores_the_bundle_occurrence_of_the_marker(self, page):
        text = page.read_text()
        assert text.count('Plotly.newPlot(') >= 2
        data, layout, config = figure_payload_from_html(str(page))
        assert len(data) == 5
        assert layout['title']['text'] == 'tree scene'
        assert config.get('scrollZoom') is True

    def test_trace_identity_survives_the_round_trip(self, page):
        data, _layout, _config = figure_payload_from_html(str(page))
        roles = classify_traces(data)
        # The synapse legend dummy is a coordinate-less swatch that carries a
        # 'synapses ...' group, so it reads as synapse background either way.
        assert [r.role for r in roles] == [
            'neuron', 'neuron', 'companion', 'synapse', 'mesh']
        assert [r.item for r in roles][:2] == ['256_aMe4', '257_aMe5']
        assert [r.body_id for r in roles][:2] == ['256', '257']
        assert roles[2].group == 'KC layer'
        # The tree panel's own tags are still there for a browser to consume.
        assert data[0]['meta']['drocatLegend']['item'] == '256_aMe4'

    def test_geometry_colors_and_ranks_are_lossless(self, page):
        original = [t.to_plotly_json() for t in tree_figure().data]
        data, _layout, _config = figure_payload_from_html(str(page))
        for index, (before, after) in enumerate(zip(original, data)):
            assert after['type'] == before['type'], index
            for axis in ('x', 'y', 'z'):
                assert len(after.get(axis) or []) == len(
                    before.get(axis) or []), (index, axis)
            assert after.get('legendrank') == before.get('legendrank'), index
            assert after.get('showlegend') == before.get('showlegend'), index
            assert after.get('name') == before.get('name'), index
        for index in (0, 1, 2, 4):
            before, after = original[index], data[index]
            colour = (after.get('line') or {}).get('color') or \
                (after.get('marker') or {}).get('color') or after.get('color')
            assert colour, index

    def test_rebuilt_figure_carries_the_same_payload(self, page):
        figure = figure_from_plotly_html(str(page))
        assert isinstance(figure, go.Figure)
        assert [t.type for t in figure.data] == [
            'scatter3d', 'scatter3d', 'mesh3d', 'scatter3d', 'mesh3d']
        assert figure.layout.scene.aspectmode == 'data'
        # A rebuilt figure can be written again without losing the freeze
        # input or the neuron roles.
        ranges = VisualizeSkeleton._scene_data_ranges(figure)
        assert ranges['x'][0] < 0 and ranges['z'][1] > 300
        assert [r.role for r in classify_traces(figure.data)][:2] == [
            'neuron', 'neuron']

    def test_rewritten_page_is_valid_input_again(self, page, tmp_path):
        target = tmp_path / 'again.html'
        make_vis()._write_plotly_html(
            figure_from_plotly_html(str(page)), str(target),
            freeze_view=True, include_plotlyjs=False)
        data, _layout, _config = figure_payload_from_html(str(target))
        assert len(data) == 5
        assert 'drocat-freeze-toggle' in target.read_text()

    def test_non_html_and_markerless_files_raise_valueerror(self, tmp_path):
        empty = tmp_path / 'plain.html'
        empty.write_text('<html><body>nothing here</body></html>')
        with pytest.raises(ValueError, match='Plotly.newPlot'):
            figure_payload_from_html(str(empty))

    def test_marker_with_unparseable_args_is_skipped(self, tmp_path):
        page_path = tmp_path / 'broken_then_good.html'
        page_path.write_text(
            '<html><body>'
            'Plotly.newPlot(THIS IS NOT JSON'
            '\nPlotly.newPlot("gd", [], {"title": "ok"}, {"scrollZoom": true});'
            '</body></html>')
        data, layout, config = figure_payload_from_html(str(page_path))
        assert data == []
        assert layout == {'title': 'ok'}
        assert config == {'scrollZoom': True}
