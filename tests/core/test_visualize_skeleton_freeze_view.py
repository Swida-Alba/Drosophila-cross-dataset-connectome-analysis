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


def run_page_js(expression, ranges, *, gd='null', preamble='',
                update_body=None, frozen=True):
    """Execute the emitted freeze helpers under node against plain stubs.

    The slice starts at ``ratio()`` and stops at ``decorate()``, which is
    everything in the script that reads no DOM beyond ``graphDiv()``: the two
    box builders, the drawn-content bounds, the pivot/dolly maths and the
    camera-shadow plumbing. ``frozen``, ``frameRadius`` and ``extentCache`` live
    outside that window in the real IIFE, so the program declares them itself --
    ``frozen`` as a parameter, because the fit box depends on it and a test must
    be able to ask either state for its box. ``preamble`` seeds state, and
    ``update_body`` replaces the default record-and-return ``update()`` stub --
    the view-reset hook test needs one that also lands the write in
    ``_fullLayout``, the way plotly would.
    """
    script = make_vis()._freeze_view_html(ranges)
    start = script.index('function ratioFor(r) {')
    end = script.index('function decorate() {')
    update_js = (update_body
                 or 'function update(u) { APPLIED.push(u); return true; }')
    program = (
        f'var CONFIG = {json.dumps({"ranges": ranges})};\n'
        "var AXES = ['x', 'y', 'z'];\n"
        f'var GD = {gd};\n'
        f'var frozen = {json.dumps(bool(frozen))};\n'
        'var frameRadius = 0;\n'
        'var extentCache = {};\n'
        'var APPLIED = [];\n'
        'function graphDiv() { return GD; }\n'
        f'{update_js}\n'
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
        baked = json.loads(script.split('var CONFIG = ')[1].split('};')[0]
                           + '}')
        assert baked['ranges'] == RANGES
        assert baked['hiddenHalf'] is False
        assert "'scene.xaxis.range': r.x" in script
        assert "'scene.xaxis.autorange': false" in script

    def test_a_half_hidden_at_write_time_moves_the_opening_pivot(self):
        """The box spans what the page does not draw, so say so.

        A frozen box fixes the rotation pivot at its own centre, and when the
        run embedded the other context half that centre is the gap between the
        two groups of anatomy: a brain-only male-cns page opened with the brain
        pushed to the top of a 1.46x wider frame. The page already has the
        control that fixes it (⌖), so setup runs it once -- and only when a
        half is actually hidden, because otherwise the pivot is already right
        and moving it would be a framing change no one asked for.
        """
        hidden = make_vis()._freeze_view_html(RANGES, hidden_half=True)
        assert json.loads(hidden.split('var CONFIG = ')[1].split('};')[0]
                          + '}')['hiddenHalf'] is True
        shown = make_vis()._freeze_view_html(RANGES)
        assert json.loads(shown.split('var CONFIG = ')[1].split('};')[0]
                          + '}')['hiddenHalf'] is False
        for script in (hidden, shown):
            start = script.index('var opening = cameraParts();')
            # decorate() guards on `if (button)` too, so end the slice at the
            # next one: what the page does at open is what sits in between.
            end = script.index('if (button) {', start)
            setup = script[start:end]
            # The boot camera is the anchor every later dolly scales from, and
            # it is captured before anything moves.
            assert 'frameRadius = opening ? opening.radius : 0;' in setup
            assert "if (CONFIG.hiddenHalf) {" in setup
            # The pivot half of ⌖ only: a page that also dollied at setup would
            # open zoomed onto one half instead of on the whole box.
            assert setup.count('recenter(false);') == 1
            assert 'recenter(true' not in setup

    def test_fit_recomputes_the_box_instead_of_autoranging(self):
        """Fit promises "the framing follows the visible traces", and Plotly's
        autorange cannot deliver it.

        The embedded context halves ship ``visible: true`` with ``opacity: 0``
        so a reveal can never re-normalise a frozen scene -- and Plotly counts a
        trace that draws nothing. Measured on a male-cns-shaped page, handing the
        axes back left the Fit box at 191.2 where the drawn half wanted ~40. The
        script therefore builds Fit's box itself, over the drawn traces, with the
        writer's own padding, and keeps autorange off in both states.
        """
        script = make_vis()._freeze_view_html(RANGES)
        assert "'scene.xaxis.autorange': true" not in script
        assert "'scene.xaxis.range': null" not in script
        assert "'scene.xaxis.autorange': false, 'scene.xaxis.range': r.x," \
            in script
        assert "e.key === 'f' || e.key === 'F'" in script
        fit = script[script.index('function fitBox() {'):
                     script.index('function currentBox() {')]
        assert 'visibleBounds()' in fit
        assert 'span / 32.0' in fit, 'fit must pad exactly like the writer'

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
        # The 36 px button is the tip's containing block, so shrink-to-fit
        # collapses the box onto it and the hint stacks one word per line.
        # max-content opts out, which is what keeps a hint on one row; the cap
        # only wraps it on a window narrower than the longest sentence.
        assert ('width:max-content;max-width:min(560px,88vw);'
                'white-space:normal;') in script
        assert 'white-space:nowrap' not in script


class TestFrozenAspectRatio:
    """The freeze holds the box *proportions*, not just the axis ranges.

    Measured in Chrome on a real page: with the ranges pinned, hiding one
    long mesh still moved ``scene.aspectratio`` from ``{0.97, 0.73, 1.41}`` to
    ``{1.67, 0.93, 0.64}`` -- ``aspectmode='data'`` re-derives the ratio from
    the visible traces on every restyle, which is the rescaling the freeze
    exists to prevent. So both states write ``'manual'`` with the ratio of the
    box they are using; only which box that is changes.
    """

    RANGES = {'x': [0.0, 4.0], 'y': [0.0, 2.0], 'z': [0.0, 8.0]}
    # The drawn content the fit box is sized from: exactly the baked ranges,
    # so any difference between the two boxes is the padding and nothing else.
    GD = json.dumps({'_fullData': [{'x': [0, 4], 'y': [0, 2], 'z': [0, 8],
                                    'visible': True}]})

    @pytest.fixture(autouse=True)
    def _require_node(self):
        if shutil.which('node') is None:
            pytest.skip('node is not installed')

    def _run(self, ranges=None, frozen=True):
        """Both box builders, under node, against the drawn traces above."""
        ranges = ranges or self.RANGES
        return run_page_js('{pin: pinUpdate(), fit: fitUpdate(), '
                           'fitBox: fitBox()}', ranges, gd=self.GD,
                           frozen=frozen)

    def test_the_pin_freezes_the_ratio_and_fit_keeps_its_own(self):
        state = self._run()
        fitted = self._run(frozen=False)
        assert state['pin']['scene.aspectmode'] == 'manual'
        assert state['pin']['scene.aspectratio.x'] == pytest.approx(0.5)
        assert state['pin']['scene.aspectratio.y'] == pytest.approx(0.25)
        assert state['pin']['scene.aspectratio.z'] == pytest.approx(1.0)
        # Ranges and ratio travel in one relayout: pinning only the ratio would
        # hold the shape while the box still slid around the scene.
        assert state['pin']['scene.zaxis.range'] == [0.0, 8.0]
        # The frozen page's own box, unchanged, is what fitBox() would not return
        # -- currentBox() reads the state, and freezing must not re-derive.
        assert fitted['pin'] == state['pin']
        # Fit hands over no part of that to Plotly: it is the same three writes
        # against the box it computed from the drawn content.
        assert fitted['fit']['scene.aspectmode'] == 'manual'
        assert fitted['fit']['scene.xaxis.autorange'] is False
        assert fitted['fit']['scene.zaxis.range'] == pytest.approx(
            [-8 / 32.0, 8 + 8 / 32.0])
        assert fitted['fit']['scene.aspectratio.x'] == pytest.approx(0.5)
        assert fitted['fitBox'] == {'x': pytest.approx([-0.125, 4.125]),
                                   'y': pytest.approx([-0.0625, 2.0625]),
                                   'z': pytest.approx([-0.25, 8.25])}
        assert state['fitBox'] == fitted['fitBox']

    def test_the_fit_box_follows_the_drawn_traces_not_the_pinned_one(self):
        """One box per state, and the frozen one is the only trace-blind one.

        Hidden traces -- ``visible: false`` or ``opacity: 0`` -- size the pinned
        box (the reveal contract needs one frame that holds both halves) and must
        not size the fit box, which is the whole promise Fit makes.
        """
        drawn = json.dumps({'_fullData': [
            {'x': [0, 1], 'y': [0, 1], 'z': [0, 2], 'visible': True},
            {'x': [0, 4], 'y': [0, 2], 'z': [0, 8], 'visible': False},
            {'x': [0, 4], 'y': [0, 2], 'z': [0, 8], 'visible': True,
             'opacity': 0},
        ]})
        state = run_page_js('{pin: pinUpdate(), fit: fitBox()}', self.RANGES,
                            gd=drawn, frozen=False)
        assert state['fit'] == {'x': [-1 / 32.0, 1 + 1 / 32.0],
                                'y': [-1 / 32.0, 1 + 1 / 32.0],
                                'z': [-2 / 32.0, 2 + 2 / 32.0]}
        assert state['pin']['scene.zaxis.range'] == [0.0, 8.0]

    def test_a_toggle_does_not_rescan_the_coordinates(self):
        """The fit box is rebuilt on every legend toggle, so the per-trace
        extents are cached: re-walking every coordinate array per toggle costs
        more than the toggle it is answering for."""
        gd = json.dumps({'_fullData': [{'x': [0, 4], 'y': [0, 2], 'z': [0, 8],
                                        'visible': True}]})
        before = run_page_js('visibleBounds()', self.RANGES, gd=gd)
        grown = json.dumps({'_fullData': [{'x': [0, 40], 'y': [0, 2],
                                           'z': [0, 8], 'visible': True}]})
        after = run_page_js('visibleBounds()', self.RANGES, gd=grown,
                            preamble='extentCache[0] = '
                                     '{lo: {x: 0, y: 0, z: 0}, '
                                     'hi: {x: 4, y: 2, z: 8}};')
        assert before == after == {'lo': {'x': 0, 'y': 0, 'z': 0},
                                   'hi': {'x': 4, 'y': 2, 'z': 8}}

    def test_the_ratio_reads_only_the_baked_ranges(self):
        """No trace data may reach the ratio, or it is trace-dependent again.

        There is one baked box and the ratio reads it directly, so the pin
        cannot see what is currently drawn -- and has no second framing to be
        talked into when a hidden half comes on.
        """
        script = make_vis()._freeze_view_html(self.RANGES)
        # The whole pin path: the ratio, the box writer, and the baked ranges
        # it reads. Nothing in it may look at a trace.
        body = script[script.index('function ratioFor(r) {'):
                      script.index('function fitBox() {')]
        code = '\n'.join(line.split('//')[0] for line in body.splitlines())
        assert 'CONFIG.ranges' in code
        for leak in ('gd.data', 'visible', '_fullLayout', 'restyle'):
            assert leak not in code, f'{leak} makes the ratio trace-dependent'
        # The box selector is gone, not merely unused: a page with one box has
        # nothing to choose between.
        for gone in ('activeRanges', 'contextIsShown', 'syncBox', 'revealed',
                     'CONFIG.reveal'):
            assert gone not in script, f'{gone} is the two-box machinery'
        baked = script.split('var CONFIG = ')[1].split('};')[0] + '}'
        # One box, plus the note that says whether the box spans more than the
        # page draws -- which is a pivot question, not a second box.
        # viewCameras (the frontal orientations the reset watcher matches
        # against) rides along with the framing pair.
        assert {'ranges', 'hiddenHalf'} <= set(json.loads(baked))

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

    def _run(self, expression, gd='null', preamble='', update_body=None,
             frozen=True):
        """Run *expression* under node against the page's own helper functions."""
        return run_page_js(expression, self.RANGES, gd=gd, preamble=preamble,
                           update_body=update_body, frozen=frozen)

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
        hidden = self._run('pivotAndSpans().centre',
                           json.dumps({'_fullData': traces}))
        # Only the first trace counts: the hidden one is off, the swatch has no
        # numbers, and the last sits outside the pinned box (a placeholder some
        # traces carry at the dataset origin).
        assert hidden == {'x': pytest.approx(0.0),
                          'y': pytest.approx(0.0),
                          'z': pytest.approx(-0.25)}

        shown = self._run('pivotAndSpans().centre', json.dumps({'_fullData': [
            dict(t, visible=True) for t in traces]}))
        # z now spans 1..8, whose midpoint sits above the box centre at 4.
        assert shown['z'] == pytest.approx(0.0625)

    def test_the_dolly_ratio_is_the_largest_spans_not_the_pivot(self):
        """The one number ⌖ scales the camera radius by: max drawn span over max
        box span, in data units. Measured against a page whose box IS the
        content, this is what lands within ~7% of its framing; the per-axis
        ratio (0.94 here) and the diagonal ratio (0.18) both miss it badly.
        """
        gd = json.dumps({'_fullData': [
            {'x': [1, 3], 'y': [0, 2], 'z': [1, 3], 'visible': True}]})
        assert self._run('pivotAndSpans().ratio', gd) == pytest.approx(2 / 8)
        # A box already tight on the content gives ~1, so fit mode cannot
        # double-apply the dolly.
        tight = json.dumps({
            '_fullData': [{'x': [1, 3], 'y': [0, 2], 'z': [1, 3],
                           'visible': True}],
            '_fullLayout': {'scene': {
                'xaxis': {'range': [1, 3]}, 'yaxis': {'range': [0, 2]},
                'zaxis': {'range': [1, 3]}}}})
        assert self._run('pivotAndSpans().ratio', tight) == pytest.approx(1.0)

    def test_recenter_acts_in_both_states_and_writes_no_range(self):
        """⌖ owns the camera, so the freeze state cannot be a precondition.

        The gate this replaces returned without acting in fit mode on the grounds
        that the pinned ranges its maths read were stale there. The denominator
        is now the box in use, so one computation serves both states -- and it
        touches no axis key, which is what lets it re-frame a frozen view without
        breaking the freeze.
        """
        script = make_vis()._freeze_view_html(self.RANGES)
        body = script[script.index('function recenter(dolly, box) {'):
                      script.index('function centerOf(cam) {')]
        code = '\n'.join(line.split('//')[0] for line in body.splitlines())
        assert 'if (!frozen)' not in code
        assert 'scene.camera.eye.' in code and 'scene.camera.center.' in code
        for leak in ('xaxis.range', 'aspectmode', 'autorange', 'aspectratio'):
            assert leak not in code, f'{leak} would move the box, not the view'

        gd = json.dumps({
            '_fullData': [{'x': [1, 3], 'y': [0, 2], 'z': [1, 3],
                           'visible': True}],
            '_fullLayout': {'scene': {'camera': {
                'eye': {'x': 1.25, 'y': 1.25, 'z': 1.25},
                'center': {'x': 0, 'y': 0, 'z': 0}}}},
        })
        writes = {'frozen': self._run('(recenter(true), APPLIED)', gd),
                  'fit': self._run('(recenter(true), APPLIED)', gd,
                                   frozen=False)}
        # Same box in use, same numbers: the state is not an input.
        assert writes['frozen'] == writes['fit']
        written = writes['frozen'][0]
        assert (written['scene.camera.center.x'],
                written['scene.camera.center.y'],
                written['scene.camera.center.z']) == pytest.approx(
                    (0.0, 0.0, -0.25))

    def test_recenter_writes_eye_and_centre_together(self):
        """A pivot write that leaves the eye where it was changes the zoom.

        Measured on the shipped control: pressing ⌖ moved the radius from
        2.1651 to 2.4286 -- a 12% unintended zoom-out on every press, because
        ``camera.eye`` is read relative to ``camera.center``. Both are now
        written from one computation, so the radius is exactly the number chosen:
        the anchor (``frameRadius``, the radius that frames the box at this
        orientation) scaled by the ratio of the largest data spans.
        """
        gd = json.dumps({
            '_fullData': [{'x': [1, 3], 'y': [0, 2], 'z': [1, 3],
                           'visible': True}],
            '_fullLayout': {'scene': {'camera': {
                'eye': {'x': 1.25, 'y': 1.25, 'z': 1.25},
                'center': {'x': 0, 'y': 0, 'z': 0}}}},
        })
        default_radius = math.sqrt(3) * 1.25
        seed = 'frameRadius = %r;' % default_radius

        def radius(write):
            return math.sqrt(sum(
                (write['scene.camera.eye.%s' % a]
                 - write['scene.camera.center.%s' % a]) ** 2
                for a in ('x', 'y', 'z')))

        seen = self._run('{dolly: APPLIED[0], plain: APPLIED[1]}', gd,
                         preamble='recenter(true); recenter(false);')
        # drawn span 2 / box span 8 = 0.25 of the framing radius.
        assert radius(seen['dolly']) == pytest.approx(default_radius * 0.25)
        # Without the dolly the radius is held exactly: a pure translation.
        assert radius(seen['plain']) == pytest.approx(default_radius)

        twice = self._run('APPLIED', gd, preamble='recenter(true); '
                          'recenter(true);')
        assert twice[0] == twice[1], 'a second press must not compound the zoom'

    def test_fit_hands_the_camera_over_rather_than_nulling_it(self):
        """``camera.center`` is normalized to the box in use, and Fit now knows
        what that box is.

        Measured on a real page before: hide the VNC, ⌖ (centre goes to
        ``0 / -0.122 / -0.346``), then press Fit -- the box shrank onto the
        visible traces while the stale offset stayed on the camera, leaving the
        anatomy 12% of the y-span and 35% of the z-span off its rotation centre.
        The old fix was to clear the centre on the way out, which also threw away
        the aim. Fit now recomputes against the box it just wrote, so there is
        nothing to remember and nothing to clear.
        """
        gd = json.dumps({
            '_fullData': [{'x': [1, 3], 'y': [0, 2], 'z': [1, 3],
                           'visible': True}],
            '_fullLayout': {'scene': {'camera': {
                'center': {'x': 0, 'y': 0, 'z': -0.25}}}},
        })
        seen = self._run('{fit: fitUpdate(), pin: pinUpdate()}', gd)
        for state in ('fit', 'pin'):
            assert not [k for k in seen[state] if 'camera' in k], \
                'the box builders must not touch the camera'

        # And the pivot maths read the box the scene is USING: with the fitted
        # box in _fullLayout the same visible content normalizes to its centre.
        fitted = json.dumps({
            '_fullData': [{'x': [1, 3], 'y': [0, 2], 'z': [1, 3],
                           'visible': True}],
            '_fullLayout': {'scene': {
                'xaxis': {'range': [1, 3]}, 'yaxis': {'range': [0, 2]},
                'zaxis': {'range': [1, 3]},
                'camera': {'eye': {'x': 1.25, 'y': 1.25, 'z': 1.25},
                           'center': {'x': 0, 'y': 0, 'z': 0}}}},
        })
        assert self._run('pivotAndSpans().centre', fitted) == pytest.approx(
            {'x': 0.0, 'y': 0.0, 'z': 0.0}, rel=1e-3, abs=1e-3)

    def test_a_pan_beats_a_recomputed_pivot(self):
        """Plotly's pan writes camera.center, so a pivot the viewer moved is
        theirs: a camera write that is not a default-pivot preset must not be
        answered by one.
        """
        script = make_vis()._freeze_view_html(self.RANGES)
        head = "gd.on('plotly_relayout', function(event) {"
        start = script.index(head)
        end = script.index('});', start) + len('});')
        hook = script[start:end]
        gd_stub = (
            "var gd = { handlers: [],\n"
            "  _fullData: [{x: [1, 3], y: [0, 2], z: [1, 3], visible: true}],\n"
            "  _fullLayout: {scene: {camera: "
            "{eye: {x: 0, y: 0, z: -2.5}, up: {x: 0, y: -1, z: 0}, "
            "center: {x: 0, y: 0, z: 0}}}} };\n"
            "gd.on = function (ev, fn) { this.handlers.push([ev, fn]); };\n"
            "GD = gd;\n"
            "CONFIG.viewCameras = [{eye: {x: 0, y: 0, z: -2.5}, "
            "up: {x: 0, y: -1, z: 0}}];\n"
            + hook + "\n"
        )
        expression = """
(function(){
  var fn = null;
  for (var i = 0; i < gd.handlers.length; i++) {
    if (gd.handlers[i][0] === 'plotly_relayout') { fn = gd.handlers[i][1]; }
  }
  // A pan: plotly moves the centre and keeps the eye, and asks for nothing else.
  fn({'scene.camera': {eye: {x: 0, y: 0, z: -2.5}, up: {x: 0, y: -1, z: 0},
       center: {x: 0.18, y: 0, z: -0.4}}});
  return {afterPan: APPLIED.length};
})()
"""
        assert self._run(expression, preamble=gd_stub) == {'afterPan': 0}

    def test_a_lost_box_is_reapplied_after_plotlys_own_resets(self):
        script = make_vis()._freeze_view_html(self.RANGES)
        assert 'function repaint() {' in script
        # Both states re-assert, because both now hold a box the page owns; the
        # guard is only the re-entrancy flag.
        assert 'if (repinning) { return; }' in script
        assert 'if (!drifted()) { return; }' in script
        assert 'update(frozen ? pinUpdate() : fitUpdate())' in script

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

    def test_a_fit_box_change_is_reasserted_but_a_reveal_is_not_rescaled(self):
        """A legend-tree eye is a restyle, and a restyle must not move the view
        while the box is frozen.

        Both events reach ``repaint``, and in each state the box in force is only
        ever *restored*: frozen from the baked ranges, fitted from what the
        toggle left drawn. The two versions this replaces both failed the same
        promise differently: one grew the frozen box on a reveal (a camera move
        under a freeze), the earlier one listened for relayout only and so left
        the revealed half outside the framing.
        """
        script = make_vis()._freeze_view_html(self.RANGES)
        assert "gd.on('plotly_relayout', repaint);" in script
        assert "gd.on('plotly_restyle', repaint);" in script
        repaint = script[script.index('function repaint() {'):
                         script.index("gd.on('plotly_relayout'")]
        assert 'drifted()' in repaint
        assert 'recenter(' not in repaint, 'a restyle may not move the camera'
        # The frozen box is decided when the page is written: nothing between the
        # pin builder and the fit box asks what is on screen.
        pinning = script[script.index('function ratioFor(r) {'):
                         script.index('function fitBox() {')]
        code = '\n'.join(line.split('//')[0] for line in pinning.splitlines())
        assert 'visible' not in code, 'the pin must be trace-blind'
        # ...and the fit box is the one builder that may not be.
        fitting = script[script.index('function fitBox() {'):
                         script.index('function traceExtent(')]
        assert 'visibleBounds()' in fitting

    def test_the_doubleclick_handler_is_gone_now_that_relayout_listens(self):
        """One re-pin path, not two racing each other over the same gesture."""
        script = make_vis()._freeze_view_html(self.RANGES)
        assert 'plotly_doubleclick' not in script

    def test_recenter_writes_even_when_the_pivot_is_the_box_centre(self):
        """The write is also the internal-orbit-camera sync, so no skip.

        With a half revealed the visible midpoint can equal the box centre,
        and a view click has then already set the pivot to that same point --
        but plotly's gl scene keeps its own orbit camera, and only a dotted
        write syncs it. Skipping the "redundant" write let the next view click
        re-sync the layout back from the stale internal pivot. The write's own
        echo carries dotted keys, so this cannot loop.
        """
        gd = json.dumps({
            '_fullData': [{'x': [1, 3], 'y': [0, 2], 'z': [3, 5],
                           'visible': True}],
            '_fullLayout': {'scene': {'camera': {
                'eye': {'x': 0, 'y': 0, 'z': -2.5},
                'up': {'x': 0, 'y': -1, 'z': 0},
                'center': {'x': 0, 'y': 0, 'z': 0}}}},
        })
        seen = self._run('(recenter(false), {writes: APPLIED.length, '
                         'w: APPLIED[0]})', gd)
        assert seen['writes'] == 1, 'a centre-at-zero content still gets its write'
        written = seen['w']
        assert (written['scene.camera.center.x'], written['scene.camera.center.y'],
                written['scene.camera.center.z']) == pytest.approx((0, 0, 0))
        # The eye travels with it: without a dolly the radius is held exactly,
        # so a pivot write cannot zoom the viewer out behind their back.
        assert written['scene.camera.eye.z'] == pytest.approx(-2.5)

    # A plotly 3D write is eased: while it is animating, ``_fullLayout`` holds
    # the camera the scene is passing *through*, not the one the write asked
    # for. The watcher is therefore stubbed with a holder that disagrees with
    # its event -- reading the holder is exactly how every preset used to slip
    # past unre-aimed (measured live: the modebar reset left the pivot at zero).
    PRESET_EYE = "{x: 0, y: 0, z: -2.5}"
    PRESET_UP = "{x: 0, y: -1, z: 0}"

    def _watcher_stub(self, hook, holder_eye="{x: 0, y: 0, z: -3.5}"):
        return (
            "var gd = { handlers: [],\n"
            "  _fullData: [{x: [1, 3], y: [0, 2], z: [1, 3], visible: true}],\n"
            "  _fullLayout: {scene: {camera: {eye: " + holder_eye + ",\n"
            "    up: " + self.PRESET_UP + ",\n"
            "    center: {x: 0, y: 0, z: 0}}}} };\n"
            "gd.on = function (ev, fn) { this.handlers.push([ev, fn]); };\n"
            "GD = gd;\n"
            "CONFIG.viewCameras = [{eye: " + self.PRESET_EYE + ", up: "
            + self.PRESET_UP + "}];\n"
            "frameRadius = 99;\n"      # the boot anchor, to prove a refresh
            + hook + "\n"
        )

    WATCHER_UPDATE_BODY = """
function update(u) {
  APPLIED.push(u);
  for (var k in u) {
    var m = /^(scene\\.camera\\.(center|eye|up))\\.([xyz])$/.exec(k);
    if (m && u[k] !== null) {
      GD._fullLayout.scene.camera[m[2]][m[3]] = u[k];
    }
  }
  return true;
}
"""

    WATCHER_FIRE = """
(function(){
  var fn = null;
  for (var i = 0; i < gd.handlers.length; i++) {
    if (gd.handlers[i][0] === 'plotly_relayout') { fn = gd.handlers[i][1]; }
  }
  if (!fn) { return {error: 'hook not registered'}; }
  function fire(event) { fn(event); }
%s
})()
"""

    def _hook_slice(self):
        script = make_vis()._freeze_view_html(self.RANGES)
        head = "gd.on('plotly_relayout', function(event) {"
        start = script.index(head)
        end = script.index('});', start) + len('});')
        return script, script[start:end]

    def test_a_view_preset_reset_is_reaimed_on_the_visible_traces(self):
        """The View presets rewrite the whole camera with the box pivot.

        The emitted hook answers a relayout whose *asked-for* camera is a baked
        preset with the box-centred pivot: a hidden VNC must not win the pivot
        back just because the viewer asked for the top view. An orbit drag's eye
        belongs to no preset, an event without a camera is not a camera write,
        and ⌖'s own echo carries per-axis dotted keys the watcher does not
        answer.
        """
        script, hook = self._hook_slice()
        assert 'targetCamera(event)' in hook and 'centerOf(cam)' in hook
        assert 'recenter(false)' in hook and 'cameraMatchesViewPreset' in hook

        seen = self._run(self.WATCHER_FIRE % ("""
  fire({});                                       // not a camera write
  var plain = APPLIED.length;
  fire({'scene.camera': {eye: {x: 0.7, y: 0.7, z: 0.7},
        up: """ + self.PRESET_UP + """, center: {x: 0, y: 0, z: 0}}});  // orbit
  var orbit = APPLIED.length - plain;
  fire({'scene.camera': {eye: """ + self.PRESET_EYE + """,
        up: """ + self.PRESET_UP + """, center: {x: 0, y: 0, z: 0}}});  // preset
  var preset = APPLIED.length - orbit - plain;
  fire({'scene.camera.center.x': 0, 'scene.camera.center.y': 0,
        'scene.camera.eye.z': -3.75});            // our own echo
  var echo = APPLIED.length - preset - orbit - plain;
  var landed = {pivot: {x: GD._fullLayout.scene.camera.center.x,
                        y: GD._fullLayout.scene.camera.center.y,
                        z: GD._fullLayout.scene.camera.center.z},
                eye: {x: GD._fullLayout.scene.camera.eye.x,
                      y: GD._fullLayout.scene.camera.eye.y,
                      z: GD._fullLayout.scene.camera.eye.z}};
  recenter(true);                                 // the refreshed anchor's dolly
  return {plain: plain, orbit: orbit, preset: preset, echo: echo,
          pivot: landed.pivot, eye: landed.eye, anchor: frameRadius,
          dollyEye: APPLIED[APPLIED.length - 1]['scene.camera.eye.z']};
"""), preamble=self._watcher_stub(hook),
                         update_body=self.WATCHER_UPDATE_BODY)
        assert seen == {'plain': 0, 'orbit': 0, 'preset': 1, 'echo': 0,
                        'pivot': {'x': 0.0, 'y': 0.0, 'z': -0.25},
                        # A preset re-aims and never re-frames: the eye keeps the
                        # radius in hand (3.5) and travels with the pivot.
                        'eye': {'x': 0.0, 'y': 0.0, 'z': -3.75},
                        # The preset's own radius -- not the mid-animation one --
                        # is what every later dolly scales from.
                        'anchor': 2.5,
                        # drawn span 2 / box span 8 of that anchor.
                        'dollyEye': pytest.approx(-0.25 - 2.5 * 0.25)}

    def test_the_modebar_reset_dotted_event_reaims_the_pivot(self):
        """The modebar's reset emits dotted scene.camera.* keys -- not the
        whole-object form a preset click uses -- and restores the page's opening
        camera. The watcher must answer it too, or the frozen page's pivot dies
        on the modebar's home button.
        """
        _, hook = self._hook_slice()
        seen = self._run(self.WATCHER_FIRE % ("""
  fn({'scene.camera.eye': """ + self.PRESET_EYE + """,
      'scene.camera.up': """ + self.PRESET_UP + """,
      'scene.camera.center': {x: 0, y: 0, z: 0}});
  return {writes: APPLIED.length, anchor: frameRadius,
          pivot: GD._fullLayout.scene.camera.center};
"""), preamble=self._watcher_stub(hook),
                         update_body=self.WATCHER_UPDATE_BODY)
        assert seen == {'writes': 1, 'anchor': 2.5,
                        'pivot': {'x': 0.0, 'y': 0.0, 'z': -0.25}}

    def test_an_orbit_drag_ending_on_the_default_pivot_is_left_alone(self):
        """Orbiting never moves the centre, so on a page that never
        re-centred every drag ends with the default pivot -- and must not be
        'corrected' into a frame jump just because the eye belongs to no preset.
        """
        _, hook = self._hook_slice()
        stub = self._watcher_stub(hook).replace(
            "  _fullData: [{x: [1, 3], y: [0, 2], z: [1, 3], visible: true}],",
            "  _fullData: [{x: [0.5, 1.5], y: [0.5, 1.5], z: [3, 5],"
            " visible: true}],")   # visible midpoint off the box centre
        seen = self._run(self.WATCHER_FIRE % ("""
  fn({'scene.camera': {eye: {x: 1.1, y: 0.6, z: 0.9},
       up: """ + self.PRESET_UP + """, center: {x: 0, y: 0, z: 0}}});
  return {writes: APPLIED.length, anchor: frameRadius,
          centerStillDefault: centerOf(GD._fullLayout.scene.camera)};
"""), preamble=stub)
        assert seen == {'writes': 0, 'anchor': 99, 'centerStillDefault': True}

    def test_freezing_leaves_the_camera_entirely_alone(self):
        """The two buttons are independent axes, and only one of them frames.

        Re-freezing used to re-derive the pivot on entry (and Fit cleared it on
        the way out), which made ⌖'s answer depend on what Freeze had last done.
        Now freezing writes the box and nothing else, leaving un-freeze the one
        transition that frames -- because that is the frame the viewer asked for
        at the moment they asked for it.
        """
        script = make_vis()._freeze_view_html(self.RANGES)
        body = script[script.index('function setFrozen(next) {'):
                      script.index('function toggle() {')]
        assert 'recenter(true, box)' in body
        frozen_branch = body[body.index('} else {'):]
        assert 'recenter(' not in frozen_branch, 'freeze owns the box, not the view'
        for gone in ('restoreCenter', 'pinnedCenter',
                    'pinnedPivotMatchesCurrent'):
            assert gone not in script, f'{gone} is the remembered-pivot machinery'

    def test_the_reset_hook_sits_beside_the_repin_not_inside_it(self):
        """``repaint`` still never moves the camera; the hook is its own path.

        The reveal contract (a restyle may only restore the box) is what the
        older listener is for; the pivot re-aim must not leak into it.
        """
        script = make_vis()._freeze_view_html(self.RANGES)
        assert "gd.on('plotly_relayout', repaint);" in script
        repaint = script[script.index('function repaint() {'):
                         script.index("gd.on('plotly_relayout'")]
        assert 'recenter()' not in repaint
        assert "gd.on('plotly_relayout', function(event) {" in script


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
