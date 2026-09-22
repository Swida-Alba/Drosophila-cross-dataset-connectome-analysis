"""Tests for the frozen 3D view injected into skeleton viewer HTML.

``freeze_view=True`` (the default) makes the permanent pages pin their scene
axes to the extents of the whole scene, so showing or hiding a trace in the
viewer cannot rescale the anatomy.  The pin is *injected JavaScript only*: the
figure layout keeps autoranging, because the same ``_simplified.html`` page
feeds the WebDriver session that renders per-neuron profiles.

Pinning the three axis ranges is not sufficient on its own, and
``TestFrozenAspectRatio`` is the regression for that: ``aspectmode='data'``
re-derives the box proportions from the *visible* traces on every restyle, so a
page whose ranges are held still changes shape when a long mesh is hidden.
"""

import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visualize_skeleton import (  # noqa: E402
    VIEWER_MODEBAR_BUTTONS_TO_REMOVE,
    VisualizeSkeleton,
)

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


def run_page_js(expression, ranges, *, gd='null', preamble='', reveal=None):
    """Execute the emitted freeze helpers under node against plain stubs.

    The slice starts at ``contextIsShown()`` and stops at ``decorate()``, which
    is everything in the script that reads no DOM: the box selector, the
    pin/fit builders and the pivot helpers. ``frozen``, ``revealed`` and
    ``pinnedCenter`` live outside that window in the real IIFE, so the program
    declares them itself and a test can seed ``preamble`` to set state.
    """
    script = make_vis()._freeze_view_html(
        ranges, {'traces': [1], 'ranges': reveal} if reveal else None)
    start = script.index('function contextIsShown() {')
    end = script.index('function decorate() {')
    config = {'ranges': ranges}
    if reveal:
        config['reveal'] = {'traces': [1], 'ranges': reveal}
    program = (
        f'var CONFIG = {json.dumps(config)};\n'
        "var AXES = ['x', 'y', 'z'];\n"
        f'var GD = {gd};\n'
        'var frozen = true;\n'
        'var pinnedCenter = null;\n'
        'var revealed = false;\n'
        'var APPLIED = [];\n'
        'function graphDiv() { return GD; }\n'
        'function update(u) { APPLIED.push(u); return true; }\n'
        f'{script[start:end]}\n'
        f'{preamble}\n'
        f'console.log(JSON.stringify({expression}));'
    )
    if shutil.which('node') is None:
        pytest.skip('node is not installed')
    out = subprocess.run(['node', '-e', program], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


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
        # Both view tools stack down the left edge, the recenter one below.
        script = make_vis()._freeze_view_html(RANGES)
        assert '#drocat-freeze-toggle,#drocat-recenter{position:fixed;left:14px;' \
            in script
        assert '#drocat-freeze-toggle{top:14px;}' in script
        assert '#drocat-recenter{top:50px;}' in script

    def test_the_hover_hints_are_drawn_by_css_not_a_native_title(self):
        """Each button explains its current state on hover, instantly.

        A native ``title`` costs about a second to appear and would stack on
        top of the styled hint, so the copy lives in ``data-drocat-tip`` and
        ``::after`` renders it.
        """
        script = make_vis()._freeze_view_html(RANGES)
        assert 'content:attr(data-drocat-tip)' in script
        assert '#drocat-freeze-toggle:hover::after' in script
        assert 'body.drocat-theme-dark #drocat-recenter::after' in script
        assert ' title=' not in script
        # decorate() rewrites the hint with the button's state, so the hover
        # text and the screen-reader text cannot drift apart.
        assert "setAttribute('data-drocat-tip'" in script
        assert "setAttribute('aria-label'" in script
        # The longest state sentence is ~374 px on one line, which ran off a
        # narrow window, so the tip wraps inside a capped box.
        assert 'max-width:min(340px,58vw);white-space:normal;' in script
        assert 'white-space:nowrap' not in script


class TestFrozenAspectRatio:
    """The freeze holds the box *proportions*, not just the axis ranges.

    Measured in Chrome on a real page: with the ranges pinned, hiding one
    long mesh still moved ``scene.aspectratio`` from ``{0.97, 0.73, 1.41}`` to
    ``{1.67, 0.93, 0.64}`` -- ``aspectmode='data'`` re-derives the ratio from
    the visible traces on every restyle, which is the rescaling the freeze
    exists to prevent. So the pin carries the ratio the ranges already imply,
    and Fit hands ``'data'`` back.
    """

    RANGES = {'x': [0.0, 4.0], 'y': [0.0, 2.0], 'z': [0.0, 8.0]}

    @pytest.fixture(autouse=True)
    def _require_node(self):
        if shutil.which('node') is None:
            pytest.skip('node is not installed')

    def _run(self, ranges=None):
        """Execute the emitted pin/fit builders under node."""
        ranges = ranges or self.RANGES
        script = make_vis()._freeze_view_html(ranges)
        start = script.index('function contextIsShown() {')
        end = script.index('function decorate() {')
        program = (
            f'var CONFIG = {json.dumps({"ranges": ranges})};\n'
            'var revealed = false;\n'
            f'{script[start:end]}\n'
            'console.log(JSON.stringify({pin: pinUpdate(), '
            'fit: fitUpdate()}));'
        )
        out = subprocess.run(['node', '-e', program], capture_output=True,
                             text=True)
        assert out.returncode == 0, out.stderr
        return json.loads(out.stdout)

    def test_the_pin_freezes_the_ratio_and_fit_gives_it_back(self):
        state = self._run()
        assert state['pin']['scene.aspectmode'] == 'manual'
        assert state['pin']['scene.aspectratio.x'] == pytest.approx(0.5)
        assert state['pin']['scene.aspectratio.y'] == pytest.approx(0.25)
        assert state['pin']['scene.aspectratio.z'] == pytest.approx(1.0)
        # Ranges and ratio travel in one relayout: pinning only the ratio would
        # hold the shape while the box still slid around the scene.
        assert state['pin']['scene.zaxis.range'] == [0.0, 8.0]
        assert state['fit']['scene.aspectmode'] == 'data'
        assert state['fit']['scene.aspectratio'] is None

    def test_the_ratio_reads_only_the_baked_ranges(self):
        """No trace data may reach the ratio, or it is trace-dependent again.

        The box it reads is one of two baked ones, and choosing between them is
        a flag rather than a measurement, so the ratio still cannot see what is
        currently drawn.
        """
        script = make_vis()._freeze_view_html(self.RANGES)
        body = script[script.index('function ratio() {'):
                      script.index('function pinUpdate() {')]
        code = '\n'.join(line.split('//')[0] for line in body.splitlines())
        assert 'activeRanges()' in code
        for leak in ('gd.data', 'visible', '_fullLayout', 'restyle'):
            assert leak not in code, f'{leak} makes the ratio trace-dependent'

        selectors = script[script.index('function activeRanges() {'):
                           script.index('function ratio() {')]
        assert 'CONFIG.ranges' in selectors
        assert 'CONFIG.reveal.ranges' in selectors
        for leak in ('gd.data', 'visible', '_fullLayout', 'restyle',
                     'Math.min', 'values'):
            assert leak not in selectors, f'{leak} measures the scene'

    def test_the_range_padding_keeps_the_raw_axis_ratios(self):
        """A proportional pad, so freezing cannot change the anatomy's shape.

        Each axis grows by its own span over 32 at both ends, which scales all
        three spans by the same factor -- the ratio the pin carries is therefore
        the scene's real one, and the first frame looks like the unfrozen page.
        """
        fig = _Payload([scatter([0, 10], [0, 5], [0, 20], name='neuron A')])
        ranges = make_vis()._scene_data_ranges(fig)
        raw = {'x': 10.0, 'y': 5.0, 'z': 20.0}
        padded = {a: abs(ranges[a][1] - ranges[a][0]) for a in raw}
        assert [padded[a] / max(padded.values()) for a in 'xyz'] == \
            pytest.approx([raw[a] / max(raw.values()) for a in 'xyz'])


class TestRecenterAndRepin:
    """A frozen box needs a movable pivot and a pin that survives Plotly's resets.

    Two things follow from pinning the box to the *whole* scene. The rotation
    pivot sits at the scene centre, so hiding the ventral nerve cord leaves the
    brain orbiting around empty space -- hence the recenter control. And
    Plotly's modebar "reset camera to last save" restores the layout snapshot
    taken before this script ran: measured in Chrome it put ``aspectmode`` back
    to ``'data'`` and the ratio back to the visible traces' ``1.67/0.93/0.64``,
    while the button still showed the lock. So a relayout that drifted re-pins.
    """

    RANGES = {'x': [0.0, 4.0], 'y': [0.0, 2.0], 'z': [0.0, 8.0]}

    @pytest.fixture(autouse=True)
    def _require_node(self):
        if shutil.which('node') is None:
            pytest.skip('node is not installed')

    def _run(self, expression, gd='null', preamble='', reveal=None):
        """Run *expression* under node against the page's own helper functions."""
        return run_page_js(expression, self.RANGES, gd=gd, preamble=preamble,
                           reveal=reveal)

    def test_the_pivot_follows_the_visible_traces(self):
        """Hidden content must not pull the pivot, and neither do placeholders.

        The stub feeds ``_fullData`` rather than ``data`` because that is where
        plotly 3.x keeps the resolved coordinates: ``gd.data[i].x`` is a
        ``{dtype, bdata}`` container with no ``.length``, so reading it yields
        nothing and the control would sit dead on every real page.
        """
        traces = [
            {'x': [1, 3], 'y': [0, 2], 'z': [1, 3], 'visible': True},
            {'x': [0, 4], 'y': [0, 2], 'z': [6, 8], 'visible': False},
            {'x': [None], 'y': [None], 'z': [None], 'visible': True},
            {'x': [-9, -9], 'y': [-9, -9], 'z': [-9, -9], 'visible': True},
        ]
        hidden = self._run('centerUpdate()',
                           json.dumps({'_fullData': traces}))
        # Only the first trace counts: the hidden one is off, the swatch has no
        # numbers, and the last sits outside the pinned box (a placeholder some
        # traces carry at the dataset origin).
        assert hidden['scene.camera.center.x'] == pytest.approx(0.0)
        assert hidden['scene.camera.center.y'] == pytest.approx(0.0)
        assert hidden['scene.camera.center.z'] == pytest.approx(-0.25)
        assert set(hidden) == {'scene.camera.center.x', 'scene.camera.center.y',
                               'scene.camera.center.z'}

        shown = self._run('centerUpdate()', json.dumps({'_fullData': [
            dict(t, visible=True) for t in traces]}))
        # z now spans 1..8, whose midpoint sits above the box centre at 4.
        assert shown['scene.camera.center.z'] == pytest.approx(0.0625)

    def test_recenter_is_inert_in_fit_mode(self):
        """In fit mode the box already follows the visible traces, and the
        pinned ranges this math reads are stale, so the control does nothing."""
        script = make_vis()._freeze_view_html(self.RANGES)
        body = script[script.index('function recenter() {'):
                      script.index('function centerIsDefault() {')]
        code = '\n'.join(line.split('//')[0] for line in body.splitlines())
        assert 'if (!frozen) { return; }' in code
        assert 'centerUpdate()' in code and 'update(updateObj)' in code

    def test_fit_clears_the_pivot_and_freeze_takes_it_back(self):
        """``camera.center`` is normalized to the *pinned* box.

        Measured on a real page: hide the VNC, recenter (center goes to
        ``0 / -0.122 / -0.346``), then press Fit. The axes returned to
        autorange -- so the box shrank onto the visible traces, whose midpoint
        is now normalized 0 -- while the stale offset stayed on the camera. The
        anatomy sat off the rotation centre by 12% of the y-span and 35% of the
        z-span, which is the bug this pins.
        """
        gd = json.dumps({
            '_fullData': [{'x': [1, 3], 'y': [0, 2], 'z': [1, 3],
                           'visible': True}],
            '_fullLayout': {'scene': {'camera': {'center': {
                'x': 0, 'y': 0, 'z': 0}}}},
        })
        seen = self._run(
            '{clears: fitUpdate()[\'scene.camera.center\'], '
            'set: APPLIED.length}', gd)
        assert seen['clears'] is None
        assert seen['set'] == 0

        round_trip = self._run(
            '(recenter(), '
            '{remembered: pinnedCenter, '
            'afterFit: fitUpdate()[\'scene.camera.center\']})', gd)
        assert round_trip['remembered'] == {'x': 0.0, 'y': 0.0, 'z': -0.25}
        assert round_trip['afterFit'] is None

        restored = self._run(
            '(recenter(), restoreCenter(), APPLIED[1])', gd)
        assert restored['scene.camera.center.z'] == pytest.approx(-0.25)

    def test_a_pan_during_fit_mode_beats_the_remembered_pivot(self):
        """Plotly's pan writes camera.center, so a non-default pivot is the
        user's own and re-freezing must not overwrite it."""
        gd = json.dumps({
            '_fullData': [{'x': [1, 3], 'y': [0, 2], 'z': [1, 3],
                           'visible': True}],
            '_fullLayout': {'scene': {'camera': {'center': {
                'x': 0.1, 'y': -0.2, 'z': 0.05}}}},
        })
        seen = self._run(
            '(recenter(), restoreCenter(), '
            '{applies: APPLIED.length, stillDefault: centerIsDefault()})', gd)
        assert seen['applies'] == 1        # only the recenter itself
        assert seen['stillDefault'] is False

    def test_a_lost_pin_is_reapplied_after_plotlys_own_resets(self):
        script = make_vis()._freeze_view_html(self.RANGES)
        assert 'function repaint() {' in script
        assert 'if (!frozen || repinning) { return; }' in script
        assert 'if (!switched && !drifted()) { return; }' in script

        pinned = json.dumps({'_fullLayout': {'scene': {
            'aspectmode': 'manual',
            'xaxis': {'autorange': False, 'range': self.RANGES['x']},
            'yaxis': {'autorange': False, 'range': self.RANGES['y']},
            'zaxis': {'autorange': False, 'range': self.RANGES['z']}}}})
        assert self._run('drifted()', pinned) is False
        # What the modebar reset actually leaves behind: ranges held, but
        # aspectmode back on 'data'.
        reset = json.dumps({'_fullLayout': {'scene': {
            'aspectmode': 'data',
            'xaxis': {'autorange': False, 'range': self.RANGES['x']},
            'yaxis': {'autorange': False, 'range': self.RANGES['y']},
            'zaxis': {'autorange': False, 'range': self.RANGES['z']}}}})
        assert self._run('drifted()', reset) is True
        fitted = json.dumps({'_fullLayout': {'scene': {
            'aspectmode': 'data',
            'xaxis': {'autorange': True, 'range': None},
            'yaxis': {'autorange': True, 'range': None},
            'zaxis': {'autorange': True, 'range': None}}}})
        assert self._run('drifted()', fitted) is True

    def test_revealing_through_the_tree_reaches_the_repaint(self):
        """A legend-tree eye is a restyle, and restyle is not relayout.

        The first version of the box switch listened for relayout only, so on a
        live page the embedded nerve cord came on with the framing unchanged --
        which, given that 0.3% of it lies inside a brain-only box, is the same
        as not shipping the toggle at all.
        """
        script = make_vis()._freeze_view_html(self.RANGES)
        assert "gd.on('plotly_relayout', repaint);" in script
        assert "gd.on('plotly_restyle', repaint);" in script

    def test_the_doubleclick_handler_is_gone_now_that_relayout_listens(self):
        """One re-pin path, not two racing each other over the same gesture."""
        script = make_vis()._freeze_view_html(self.RANGES)
        assert 'plotly_doubleclick' not in script


class TestContextMeshRevealBox:
    """The pinned box grows while a viewer is looking at an embedded envelope.

    A page that ships a brain/VNC half it did not show has to keep that half
    out of the first frame's box -- otherwise a brain-only run opens zoomed out
    to the nerve cord. But the two do not overlap: measured on a real male-cns
    brain-only page, 0.3% of the embedded nerve cord's vertices fall inside the
    box pinned to the brain's framing, so revealing it there would draw
    almost nothing at all and the checkbox would be cosmetic. Hence a second
    baked box, switched to only while such a mesh is on view.
    """

    RANGES = {'x': [0.0, 4.0], 'y': [0.0, 2.0], 'z': [0.0, 8.0]}
    REVEAL = {'x': [-8.0, 4.0], 'y': [0.0, 2.0], 'z': [0.0, 24.0]}

    # Index 1 is the embedded mesh: ``run_page_js`` bakes ``reveal.traces``.
    @staticmethod
    def _gd(context_visible):
        traces = [
            {'x': [1, 3], 'y': [0, 2], 'z': [1, 3], 'visible': True},
            {'x': [1, 3], 'y': [0, 2], 'z': [1, 24],
             'visible': context_visible},
        ]
        return json.dumps({'data': traces, '_fullData': traces})

    def test_the_box_follows_the_embedded_mesh(self):
        off = run_page_js('(syncBox(), {switched: revealed, '
                          'z: pinUpdate()[\'scene.zaxis.range\'], '
                          'again: syncBox()})', self.RANGES,
                          gd=self._gd(False), reveal=self.REVEAL)
        assert off == {'switched': False, 'z': [0.0, 8.0], 'again': False}

        shown = run_page_js('(syncBox(), {switched: revealed, '
                            'z: pinUpdate()[\'scene.zaxis.range\'], '
                            'ratioZ: ratio().z, ratioX: ratio().x})',
                            self.RANGES, gd=self._gd(True), reveal=self.REVEAL)
        assert shown['switched'] is True
        assert shown['z'] == [0.0, 24.0]
        # The proportions belong to the box in force, not to the first frame:
        # holding the old ratio while widening z would stretch the anatomy.
        assert shown['ratioZ'] == pytest.approx(1.0)
        assert shown['ratioX'] == pytest.approx(12.0 / 24.0)

    def test_hiding_it_again_shrinks_the_box_back(self):
        """The switch is a state, not a ratchet -- the run's framing returns."""
        back = run_page_js('(revealed = true, '
                           '{switched: syncBox(), '
                           'z: pinUpdate()[\'scene.zaxis.range\']})',
                           self.RANGES, gd=self._gd(False), reveal=self.REVEAL)
        assert back == {'switched': True, 'z': [0.0, 8.0]}

    def test_a_page_without_an_embedded_mesh_never_switches(self):
        """No second box baked means no second box, however the traces read."""
        plain = run_page_js('(syncBox(), {switched: revealed, '
                            'z: pinUpdate()[\'scene.zaxis.range\'], '
                            'ratio: ratio().z})', self.RANGES,
                            gd=self._gd(True))
        assert plain == {'switched': False, 'z': [0.0, 8.0],
                         'ratio': pytest.approx(1.0)}

    def test_legendonly_is_not_on_view(self):
        """A swatch collapsed to the legend is not something the box must fit."""
        traces = json.dumps({'data': [
            {'x': [1, 3], 'y': [0, 2], 'z': [1, 3], 'visible': True},
            {'x': [1, 3], 'y': [0, 2], 'z': [1, 24], 'visible': 'legendonly'},
        ]})
        assert run_page_js('contextIsShown()', self.RANGES, gd=traces,
                           reveal=self.REVEAL) is False

    def test_the_pivot_and_the_drift_read_the_active_box(self):
        """Both normalize against whichever box is in force.

        ``camera.center`` is in normalized scene units, so a pivot computed
        against the other box points somewhere else entirely; and comparing the
        live ranges against the wrong box would have the page re-pin itself on
        every redraw.
        """
        pivot = run_page_js('(revealed = true, centerUpdate())', self.RANGES,
                            gd=self._gd(True), reveal=self.REVEAL)
        # Visible now is brain plus cord: x spans 1..3 inside a box centred on
        # -2, and z spans 1..24 around the box's own midpoint of 12.
        assert pivot['scene.camera.center.x'] == pytest.approx((2 + 2) / 12)
        assert pivot['scene.camera.center.z'] == pytest.approx((12.5 - 12) / 24)

        pinned_wide = json.dumps({'_fullLayout': {'scene': {
            'aspectmode': 'manual',
            'xaxis': {'autorange': False, 'range': self.REVEAL['x']},
            'yaxis': {'autorange': False, 'range': self.REVEAL['y']},
            'zaxis': {'autorange': False, 'range': self.REVEAL['z']}}}})
        assert run_page_js('(revealed = true, drifted())', self.RANGES,
                           gd=pinned_wide, reveal=self.REVEAL) is False
        assert run_page_js('drifted()', self.RANGES, gd=pinned_wide,
                           reveal=self.REVEAL) is True


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

    def test_the_viewer_modebar_keeps_exactly_one_reset_button(self):
        """Plotly gives a 3D scene two home-shaped resets; keep one.

        ``resetCameraLastSave3d`` restores the framing the page opened with, so
        it is the useful half. The names here are Plotly's button-registry keys:
        sending the rendered ``data-attr`` instead -- ``resetDefault`` -- is
        accepted by the config schema and ignored with no warning, which is how
        the first version of this failed a live page while passing here.
        """
        assert 'resetCameraDefault3d' in VIEWER_MODEBAR_BUTTONS_TO_REMOVE
        assert 'resetCameraLastSave3d' not in VIEWER_MODEBAR_BUTTONS_TO_REMOVE
        assert not {'resetDefault', 'resetLastSave'} & set(
            VIEWER_MODEBAR_BUTTONS_TO_REMOVE)
        source = (PROJECT_ROOT / 'src' / 'visualize_skeleton.py').read_text(
            encoding='utf-8')
        assert re.search(
            r"'modeBarButtonsToRemove':\s*list\(\s*"
            r"VIEWER_MODEBAR_BUTTONS_TO_REMOVE\s*\)", source), \
            'the viewer page must build its modebar from the constant'
