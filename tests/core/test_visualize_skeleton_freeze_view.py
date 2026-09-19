"""Tests for the frozen 3D view injected into skeleton viewer HTML.

``freeze_view=True`` (the default) makes the permanent pages pin their scene
axes to the extents of the whole scene, so showing or hiding a trace in the
viewer cannot rescale the anatomy.  The pin is *injected JavaScript only*: the
figure layout keeps autoranging, because the same ``_simplified.html`` page
feeds the WebDriver session that renders per-neuron profiles.
"""

import json
import math
import re
import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visualize_skeleton import VisualizeSkeleton  # noqa: E402

RANGES = {'x': [-1.0, 2.0], 'y': [-1.0, 2.0], 'z': [-1.0, 2.0]}


def make_vis(**attrs):
    """VisualizeSkeleton carrying only what the page-writing helpers read."""
    vis = object.__new__(VisualizeSkeleton)
    merged = dict(
        verbose=False,
        dataset='hemibrain:v1.2.1',
        client_type='neuprint',
        background_color='rgba(255, 255, 255, 1.0)',
        legend_mode='tree',
        freeze_view=True,
        html_theme_toggle=False,
        save_folder='',
    )
    merged.update(attrs)
    for key, value in merged.items():
        setattr(vis, key, value)
    # The warning banner has its own tests; keep this module about the freeze.
    vis._in_page_warning_html = lambda: ''
    return vis


def scatter(x, y, z, **kwargs):
    return go.Scatter3d(x=x, y=y, z=z, mode='lines', **kwargs)


def swatch(name='synapses 0 -> 1'):
    """A legend-only trace: every coordinate is None, as DROCAT emits them."""
    return go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                        name=name, showlegend=True)


def scene_figure():
    """A figure shaped like ``save_figure``'s output: scene chrome, no ranges."""
    fig = go.Figure(data=[scatter([0, 10], [0, 10], [0, 10], name='neuron A')])
    fig.update_layout(scene=dict(
        xaxis=dict(title='x'), yaxis=dict(title='y'), zaxis=dict(title='z'),
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)),
        aspectmode='data',
    ))
    return fig


class TestSceneDataRanges:
    def test_pads_each_axis_by_its_own_span_over_32(self):
        fig = go.Figure(data=[scatter([0, 10], [0, 20], [0, 40])])
        ranges = VisualizeSkeleton._scene_data_ranges(fig)
        # 1/32 of the span is the margin plotly.js adds when it autoranges a
        # 3D scene, so the frozen frame is the frame users see today.
        assert ranges['x'] == pytest.approx([-10 / 32.0, 10 + 10 / 32.0])
        assert ranges['y'] == pytest.approx([-20 / 32.0, 20 + 20 / 32.0])
        assert ranges['z'] == pytest.approx([-40 / 32.0, 40 + 40 / 32.0])

    def test_unions_across_traces(self):
        fig = go.Figure(data=[
            scatter([0, 1], [0, 1], [0, 1]),
            scatter([-5, 0], [2, 3], [10, 11]),
        ])
        ranges = VisualizeSkeleton._scene_data_ranges(fig)
        assert ranges['x'][0] < -5 and ranges['x'][1] > 1
        assert ranges['y'][0] < 0 and ranges['y'][1] > 3
        assert ranges['z'][0] < 0 and ranges['z'][1] > 11

    def test_includes_traces_that_start_hidden(self):
        default_off = scatter([0, 1], [0, 1], [0, 1], visible=False)
        fig = go.Figure(data=[scatter([0, 1], [0, 1], [0, 1]), default_off])
        ranges = VisualizeSkeleton._scene_data_ranges(fig)
        assert ranges['x'] == pytest.approx([-1 / 32.0, 1 + 1 / 32.0])

    def test_ignores_legend_only_swatch(self):
        fig = go.Figure(data=[scatter([0, 1], [0, 1], [0, 1]), swatch()])
        assert (VisualizeSkeleton._scene_data_ranges(fig)
                == VisualizeSkeleton._scene_data_ranges(
                    go.Figure(data=[fig.data[0]])))

    def test_holes_in_a_column_do_not_drop_the_trace(self):
        fig = go.Figure(data=[go.Mesh3d(
            x=[0, 10, None], y=[0, None, 5], z=[None, 1, 2],
            i=[0], j=[1], k=[2])])
        ranges = VisualizeSkeleton._scene_data_ranges(fig)
        assert ranges['x'] == pytest.approx([-10 / 32.0, 10 + 10 / 32.0])
        assert ranges['y'] == pytest.approx([-5 / 32.0, 5 + 5 / 32.0])
        assert ranges['z'] == pytest.approx([1 - 1 / 32.0, 2 + 1 / 32.0])

    def test_non_numeric_and_nan_entries_are_skipped(self):
        fig = go.Figure(data=[scatter([0, 'x', float('nan'), 8],
                                      [0, 1, 8], [0, 1, 8])])
        ranges = VisualizeSkeleton._scene_data_ranges(fig)
        assert ranges['x'] == pytest.approx([-8 / 32.0, 8 + 8 / 32.0])

    def test_zero_span_axis_gets_unit_padding(self):
        fig = go.Figure(data=[scatter([1, 1], [0, 2], [3, 3])])
        ranges = VisualizeSkeleton._scene_data_ranges(fig)
        assert ranges['x'] == [0.0, 2.0]
        assert ranges['z'] == [2.0, 4.0]

    def test_returns_none_without_usable_coordinates(self):
        assert VisualizeSkeleton._scene_data_ranges(
            go.Figure(data=[swatch()])) is None
        assert VisualizeSkeleton._scene_data_ranges(go.Figure()) is None
        assert VisualizeSkeleton._scene_data_ranges(
            go.Figure(data=[scatter([None], [None], [None])])) is None

    def test_accepts_trace_dicts_parsed_from_a_page(self):
        # Phases 5-7 feed this the payload read back out of an HTML file.
        traces = scatter([0, 8], [0, 8], [0, 8]).to_plotly_json()
        ranges = VisualizeSkeleton._scene_data_ranges(
            go.Figure(data=[scatter([0, 8], [0, 8], [0, 8])]))
        assert ranges == VisualizeSkeleton._scene_data_ranges(
            _Payload([traces]))


class _Payload:
    """Minimal stand-in for a Figure: only ``data`` is read."""

    def __init__(self, data):
        self.data = data


class TestFreezeViewHtml:
    def test_script_pins_the_baked_ranges(self):
        script = make_vis()._freeze_view_html(RANGES)
        assert 'drocat-freeze-toggle' in script
        assert json.dumps({'ranges': RANGES}) in script
        assert "'scene.xaxis.range': r.x" in script
        assert "'scene.xaxis.autorange': false" in script

    def test_script_hands_autoscaling_back_on_demand(self):
        script = make_vis()._freeze_view_html(RANGES)
        # A 3D axis keeps honouring a manual range after autorange is re-enabled
        # (measured in Chrome), so Fit must clear the range as well.
        assert "'scene.xaxis.autorange': true, 'scene.xaxis.range': null" in script
        assert "e.key === 'f' || e.key === 'F'" in script

    def test_guard_matches_the_theme_and_tree_extras(self):
        script = make_vis()._freeze_view_html(RANGES)
        assert 'navigator.webdriver' in script
        assert 'window.__drocatFreezeView' in script
        assert '__DROCAT_FREEZE_CONFIG__' not in script

    def test_button_sits_left_of_the_mode_bar_and_tree_panel(self):
        # The theme switch owns top-right:14px and the tree panel top-right:60px.
        script = make_vis()._freeze_view_html(RANGES)
        assert '#drocat-freeze-toggle{position:fixed;left:14px;top:14px;' in script


class TestInjection:
    def _page(self, tmp_path):
        page = tmp_path / 'page.html'
        page.write_text('<html><body><div id="gd"></div></body></html>')
        return page

    def test_inject_writes_one_block_after_body(self, tmp_path):
        page = self._page(tmp_path)
        make_vis()._inject_page_extras(str(page), freeze_ranges=RANGES)
        html = page.read_text()
        assert html.count('id="drocat-freeze-toggle"') == 1
        assert html.index('<body') < html.index('drocat-freeze-toggle')

    def test_injection_is_idempotent(self, tmp_path):
        page = self._page(tmp_path)
        vis = make_vis()
        vis._inject_page_extras(str(page), freeze_ranges=RANGES)
        first = page.read_text()
        vis._inject_page_extras(str(page), freeze_ranges=RANGES)
        assert page.read_text() == first

    def test_missing_ranges_write_no_block(self, tmp_path):
        page = self._page(tmp_path)
        vis = make_vis()
        vis._inject_page_extras(str(page))
        vis._inject_page_extras(str(page), freeze_ranges={})
        assert 'drocat-freeze-toggle' not in page.read_text()

    def test_survives_the_theme_and_tree_extras(self, tmp_path):
        page = tmp_path / 'page.html'
        page.write_text('<html><body></body></html>')
        vis = make_vis(html_theme_toggle=True)
        vis._inject_page_extras(str(page), theme_toggle=True,
                                legend_tree=True, freeze_ranges=RANGES)
        html = page.read_text()
        for marker in ('drocat-theme-toggle', 'drocat-legend-tree',
                       'drocat-freeze-toggle'):
            assert marker in html


class TestWriterThreading:
    def _write(self, tmp_path, vis, figure, **kwargs):
        path = tmp_path / 'plot.html'
        vis._write_plotly_html(
            figure, str(path), include_plotlyjs=False, **kwargs)
        return path.read_text()

    def test_writer_pins_ranges_into_the_page(self, tmp_path):
        html = self._write(tmp_path, make_vis(), scene_figure(),
                           freeze_view=True)
        assert 'drocat-freeze-toggle' in html
        assert '10.3125' in html  # 10 + 10/32: real ranges, not a placeholder

    def test_freeze_view_off_leaves_the_page_untouched(self, tmp_path):
        html = self._write(tmp_path, make_vis(), scene_figure(),
                           freeze_view=False)
        assert 'drocat-freeze-toggle' not in html

    def test_the_page_alone_carries_the_freeze(self, tmp_path):
        vis = make_vis()
        figure = scene_figure()
        self._write(tmp_path, vis, figure, freeze_view=True)
        scene = figure.to_plotly_json()['layout']['scene']
        assert scene['xaxis'].get('range') is None
        assert scene['xaxis'].get('autorange') is not False
        assert all(
            axis.get('range') is None
            for axis in (scene['xaxis'], scene['yaxis'], scene['zaxis']))
        # Nothing about a frozen page may reach a kaleido/WebDriver render.
        assert json.loads(figure.to_json())['layout']['scene'] == scene

    def test_degenerate_scene_writes_no_ranges(self, tmp_path):
        figure = go.Figure(data=[swatch()])
        html = self._write(tmp_path, make_vis(), figure, freeze_view=True)
        assert 'drocat-freeze-toggle' not in html

    def test_every_viewer_page_threads_the_field(self):
        """Every tree-mode page is also a freeze-capable page.

        ``_simplified.html`` is written from three separate places, so a new
        site that forgets ``freeze_view`` would silently keep relayouting.
        """
        source = (PROJECT_ROOT / 'src' / 'visualize_skeleton.py').read_text()
        sites = re.findall(
            r"legend_tree=\(self\.legend_mode == 'tree'\),\n\s*(\S+),", source)
        assert sites and set(sites) == {'freeze_view=self.freeze_view'}


class TestContract:
    def test_field_defaults_on(self):
        field = VisualizeSkeleton.__dataclass_fields__['freeze_view']
        assert field.default is True

    def test_ranges_are_finite(self):
        ranges = VisualizeSkeleton._scene_data_ranges(scene_figure())
        assert all(math.isfinite(v) for axis in ranges for v in ranges[axis])
