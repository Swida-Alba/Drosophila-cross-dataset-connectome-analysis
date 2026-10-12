"""The two top-left controls, measured in a browser instead of asserted in prose.

``freeze_view`` gives a page two independent axes: **Freeze/Fit owns the box**,
**⌖ owns the camera**. Everything here is a number that used to be a paragraph --
and one of them (a ⌖ that re-frames a frozen view) only exists because a camera
write turns out not to move a box.

The framing target is not an opinion either. Each scenario is measured against
its own *fitted reference page*: the same content written alone with
``freeze_view=True``, so the page's pinned box IS that content's padded extent and
its boot camera is Plotly's own answer to "the drawn content fills the frame".
The coloured ink box of the anatomy is then compared between the two pages.

Measured on the shipped control before this change, pressing ⌖ on a frozen page
left the anatomy at 21x29 px against a fitted page's 195x233 -- a 9x miss, and
the pivot write also drifted the radius 2.1651 -> 2.4286 (a 12% unintended
zoom-out on every press).

Run:
    conda run -n drocat-4.5.0 python -m pytest \
        tests/core/test_visualize_skeleton_view_controls.py -q
"""

import base64
import contextlib
import io
import math
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "core"))

from visualize_skeleton import (  # noqa: E402
    BRAIN_MESH_LEGEND_RANK,
    VisualizeSkeleton,
    dataset_view_cameras,
)

from test_visualize_skeleton_camera_shadow import (  # noqa: E402
    PANEL_READY,
    Chrome,
    requires_chrome,
)

# The fixture's own geometry, so the expected numbers are arithmetic rather
# than a transcription of a screenshot: each bundle is a 9-unit-radius helix
# spanning 20 units along z, one at +75 and one at -75.
BUNDLE_XY = (21.0, 39.0)          # 18 wide, padded to 19.125
BUNDLE_Z = 20.0                   # padded to 21.25
PINNED_Z_SPAN = (170.0 + 2 * (170.0 / 32.0))            # 180.625
FIT_Z_SPAN = BUNDLE_Z * (1 + 2 / 32.0)                  # 21.25
FRONT_RADIUS = 2.5                 # the male-cns Front preset's eye distance

STATE = r"""
var gd = document.querySelector('.js-plotly-plot');
var S = gd._fullLayout.scene;
var c = S._scene.getCamera();
var q = gd.querySelector('canvas').getBoundingClientRect();
var p = document.getElementById('drocat-legend-tree').getBoundingClientRect();
var ranges = {};
['x', 'y', 'z'].forEach(function (a) {
  ranges[a] = [S[a + 'axis'].range[0], S[a + 'axis'].range[1]];
});
return {
  aspectmode: S.aspectmode, ranges: ranges,
  pivot: [c.center.x, c.center.y, c.center.z],
  radius: Math.hypot(c.eye.x - c.center.x, c.eye.y - c.center.y,
                     c.eye.z - c.center.z),
  layoutRadius: (function () {
    var L = (gd.layout.scene && gd.layout.scene.camera);
    if (!L || !L.eye || !L.center) { return null; }
    return Math.hypot(L.eye.x - L.center.x, L.eye.y - L.center.y,
                      L.eye.z - L.center.z);
  })(),
  canvas: [q.left, q.top, q.width, q.height],
  panel: [p.left, p.top, p.width, p.height],
  doc: [document.documentElement.scrollWidth,
        document.documentElement.scrollHeight]
};
"""

EYE = r"""
window.clickEye = function (label) {
  var rows = [].slice.call(document.querySelectorAll(
    '#drocat-legend-tree .drocat-lt-row'));
  var row = rows.filter(function (r) {
    return (r.querySelector('.drocat-lt-label') || {}).textContent === label;
  })[0];
  if (!row) { throw new Error('no legend row ' + label); }
  row.querySelector('.drocat-lt-eye').click();
  return label;
};
return true;
"""

FREEZE = "document.getElementById('drocat-freeze-toggle').click(); return 1;"
RECENTER = ("document.getElementById('drocat-recenter').click(); return 1;")


def bundle(z, colour, name, rank):
    points = [(30 + 9 * math.cos(k / 120.0 * math.pi * 6),
               30 + 9 * math.sin(k / 120.0 * math.pi * 6),
               z - 10 + 20.0 * k / 119.0) for k in range(120)]
    trace = go.Scatter3d(x=[p[0] for p in points], y=[p[1] for p in points],
                         z=[p[2] for p in points], mode='lines', name=name,
                         legendgroup=name, line=dict(color=colour, width=6))
    trace.legendrank = rank
    trace.meta = {'drocatLegend': {'kind': 'neuron', 'group': name,
                                   'item': name, 'color': colour}}
    return trace


def write_page(tmp, name, traces, with_envelope=False):
    """A permanent viewer page exactly as the writer ships it, including the
    Front opening camera the view dropdown bakes in."""
    fig = go.Figure(data=list(traces))
    if with_envelope:
        envelope = go.Mesh3d(x=[0, 60, 30, 30], y=[0, 0, 60, 30],
                             z=[60, 60, 60, 90], i=[0, 0, 0, 1], j=[1, 2, 3, 2],
                             k=[2, 3, 1, 3], color='#74a8d6', opacity=0,
                             name='ctx (brain)', showlegend=True)
        envelope.legendrank = BRAIN_MESH_LEGEND_RANK
        fig = go.Figure(data=list(traces) + [envelope])
    front = dataset_view_cameras('male-cns:v1.0', 'none')['Front']
    fig.update_layout(scene=dict(camera=dict(
        eye=dict(front['eye']), up=dict(front['up']),
        center=dict(front.get('center', dict(x=0, y=0, z=0))))))
    vis = object.__new__(VisualizeSkeleton)
    vis.background_color = 'rgba(255, 255, 255, 1.0)'
    vis.legend_mode = 'tree'
    vis.freeze_view = True
    vis.html_theme_toggle = False
    vis.save_folder = str(tmp)
    vis.saveas = name
    vis.dataset = 'male-cns:v1.0'
    vis.brain_mesh = 'none'
    vis.fig_3d = fig
    vis._in_page_warning_html = lambda: ''
    path = Path(tmp) / ('%s.html' % name)
    vis._write_plotly_html(fig, str(path), legend_tree=True, freeze_view=True,
                           auto_open=False, include_plotlyjs=True)
    return path


@contextlib.contextmanager
def page(path):
    chrome = Chrome(path)
    try:
        assert chrome.wait_for(PANEL_READY), 'the legend tree never booted'
        chrome.js(EYE)
        yield chrome
    finally:
        chrome.close()


def measure(chrome, settle=1.5):
    """The scene's own numbers plus a screenshot, at the plot's own rect."""
    time.sleep(settle)
    state = chrome.js(STATE)
    raw = chrome.send('Page.captureScreenshot', {'format': 'png'})
    return state, Image.open(io.BytesIO(base64.b64decode(raw['data'])))


def ink(image, state, colour):
    """The bounding box of one bundle's saturated pixels inside the plot rect.

    A frame-wide metric measures the modebar, the legend and the corner buttons
    here, so the crop is the canvas minus the tree panel, and the count is hue
    specific: the two bundles sit at opposite ends of the scene and only one is
    meant to be drawn.
    """
    img = np.asarray(image.convert('RGB'), dtype=np.int16)
    h, w = img.shape[:2]
    dw, dh = state['doc']
    sx, sy = w / float(dw), h / float(dh)
    cx, cy, cw, ch = state['canvas']
    x0, y0 = int(cx * sx), int(cy * sy)
    sub = img[y0:min(int((cy + ch) * sy), h),
              x0:min(int((cx + cw) * sx), w)]
    px, py, pw, ph = state['panel']
    hidden = np.zeros(sub.shape[:2], dtype=bool)
    hidden[int(max(0, py - cy) * sy):int(max(0, py - cy) * sy + ph * sy),
           int(max(0, px - cx) * sx):int(max(0, px - cx) * sx + pw * sx)] = True
    red, blue = sub[:, :, 0], sub[:, :, 2]
    chroma = (sub.max(axis=2) - sub.min(axis=2)) > 45
    mask = chroma & ((blue - red > 15) if colour == 'blue' else (red - blue > 15))
    mask &= ~hidden
    ys, xs = np.where(mask)
    if not len(ys):
        return {'n': 0, 'w': 0, 'h': 0}
    return {'n': int(len(ys)), 'w': int(xs.max() - xs.min() + 1),
            'h': int(ys.max() - ys.min() + 1)}


RED, BLUE = '#d62728', '#1f77b4'


@pytest.fixture(scope='module')
def pages(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('view-controls')
    return {
        'pinned': write_page(tmp, 'pinned', [
            bundle(-75, RED, 'LOWER', 0), bundle(75, BLUE, 'UPPER', 100)]),
        'fitted': write_page(tmp, 'fitted', [bundle(-75, RED, 'LOWER', 0)]),
        'half': write_page(tmp, 'half', [bundle(-75, RED, 'LOWER', 0)],
                           with_envelope=True),
    }


@requires_chrome
class TestRecenterFramesWhileFrozen:
    """⌖ must reach the framing without touching the box (Q1)."""

    def test_the_box_is_the_union_while_both_halves_are_carried(self, pages):
        with page(pages['pinned']) as chrome:
            state, _ = measure(chrome, settle=0.4)
        assert state['aspectmode'] == 'manual'
        assert state['ranges']['z'] == pytest.approx(
            [-PINNED_Z_SPAN / 2, PINNED_Z_SPAN / 2], abs=1e-6)
        # The opening camera frames the box; it is the anchor every dolly scales
        # from, so its value is a precondition of the gates below.
        assert state['radius'] == pytest.approx(FRONT_RADIUS, rel=1e-3)

    def test_the_fitted_reference_draws_the_bundle_across_the_frame(self, pages):
        with page(pages['fitted']) as chrome:
            state, image = measure(chrome, settle=0.4)
        # The reference page's box is the bundle's own padded extent, centred on
        # the bundle at z = -75 -- not on zero.
        assert state['ranges']['z'] == pytest.approx(
            [-75 - FIT_Z_SPAN / 2, -75 + FIT_Z_SPAN / 2], abs=1e-6)
        box = ink(image, state, 'red')
        assert box['w'] > 120 and box['h'] > 150, box

    def test_recenter_scales_the_dolly_by_the_largest_data_spans(self, pages):
        """radius = the framing radius x drawn span / box span.

        The whole rule in one number: 2.5 x (20 / 180.625). Anything else the
        control could mean -- a fixed radius, or the radius the pivot leaves
        behind -- lands somewhere else, which is how the 12% drift went unnoticed.
        """
        want = FRONT_RADIUS * BUNDLE_Z / PINNED_Z_SPAN
        with page(pages['pinned']) as chrome:
            chrome.js("return window.clickEye('UPPER');")
            before, _ = measure(chrome)
            chrome.js(RECENTER)
            state, image = measure(chrome)
        assert state['radius'] == pytest.approx(want, rel=0.02)
        assert state['layoutRadius'] == pytest.approx(state['radius'], rel=1e-6)
        assert before['radius'] == pytest.approx(FRONT_RADIUS, rel=1e-3)

    def test_the_box_does_not_move_when_the_camera_reframes(self, pages):
        """The frozen contract is what makes ⌖ safe: a camera write cannot be a
        box write, and the ranges prove it is not one."""
        with page(pages['pinned']) as chrome:
            chrome.js("return window.clickEye('UPPER');")
            state, _ = measure(chrome)
            pinned = state['ranges']
            chrome.js(RECENTER)
            state, _ = measure(chrome)
            assert state['ranges'] == pinned
            chrome.js("return window.clickEye('LOWER');")
            state, _ = measure(chrome)
            assert state['ranges'] == pinned
            assert state['aspectmode'] == 'manual'

    def test_the_visible_traces_fill_the_frame_as_a_fitted_page_draws_them(
            self, pages):
        """The acceptance gate in pixels: within 15% of the reference's ink box,
        measured from a page whose box is the content itself."""
        with page(pages['fitted']) as chrome:
            state, image = measure(chrome, settle=0.4)
            reference = ink(image, state, 'red')
        with page(pages['pinned']) as chrome:
            chrome.js("return window.clickEye('UPPER');")
            state, image = measure(chrome)
            unfixed = ink(image, state, 'red')
            chrome.js(RECENTER)
            state, image = measure(chrome)
            framed = ink(image, state, 'red')
        assert unfixed['w'] < reference['w'] * 0.3, \
            'the control is not the only thing that can frame'
        assert framed['w'] == pytest.approx(reference['w'], rel=0.15), framed
        assert framed['h'] == pytest.approx(reference['h'], rel=0.15), framed

    def test_a_second_press_is_the_same_view(self, pages):
        """The dolly scales the framing anchor, not the radius in hand, so ⌖ is
        idempotent instead of compounding a zoom on every press."""
        with page(pages['pinned']) as chrome:
            chrome.js("return window.clickEye('UPPER');")
            chrome.js(RECENTER)
            first, _ = measure(chrome)
            chrome.js(RECENTER)
            second, _ = measure(chrome)
        assert second['radius'] == pytest.approx(first['radius'], abs=1e-9)
        assert second['pivot'] == pytest.approx(first['pivot'], abs=1e-9)
        assert second['ranges'] == first['ranges']

    def test_the_pivot_write_does_not_zoom(self, pages):
        """A preset or the boot pivot travels with the eye.

        The shipped control wrote ``camera.center`` alone, so the radius grew
        with the offset -- 2.1651 to 2.4286 on the old fixture, a 12% zoom-out
        nobody asked for. The opening page exercises the same write through
        ``hidden_half`` and must hold the Front radius exactly.
        """
        with page(pages['half']) as chrome:
            state, _ = measure(chrome, settle=0.6)
        assert state['radius'] == pytest.approx(FRONT_RADIUS, rel=1e-6)
        # ...while the pivot really did move onto the half the page draws.
        assert state['pivot'][2] < -0.3
        # LOWER spans z -85..-65 and the carried envelope spans 60..90, so the
        # one box spans -90.46875..95.46875 (175 padded by its own 1/32).
        assert state['ranges']['z'] == pytest.approx([-90.46875, 95.46875],
                                                    abs=1e-6)


@requires_chrome
class TestFreezeAndFitOwnTheBox:
    """Freeze stops automatic framing changes; Fit is the control that follows
    content, and it follows *drawn* content."""

    def test_unfreeze_frames_the_drawn_content_at_once(self, pages):
        with page(pages['pinned']) as chrome:
            chrome.js("return window.clickEye('UPPER');")
            state, _ = measure(chrome)
            assert state['ranges']['z'][1] - state['ranges']['z'][0] == \
                pytest.approx(PINNED_Z_SPAN, abs=1e-6)
            chrome.js(FREEZE)
            state, image = measure(chrome)
        span = state['ranges']['z'][1] - state['ranges']['z'][0]
        assert span == pytest.approx(FIT_Z_SPAN, abs=1e-3), \
            'Fit must shrink onto the drawn half, not the carried one'
        # Autorange is not what does it: the embedded half is visible, and
        # Plotly's autorange counts it (that was the 191.2 symptom).
        assert state['aspectmode'] == 'manual'
        assert ink(image, state, 'red')['w'] > 150

    def test_fit_follows_every_later_toggle_both_ways(self, pages):
        with page(pages['pinned']) as chrome:
            chrome.js(FREEZE)                       # both halves drawn
            both, _ = measure(chrome)
            chrome.js("return window.clickEye('UPPER');")
            one, image = measure(chrome)
            chrome.js("return window.clickEye('UPPER');")
            back, _ = measure(chrome)
        assert one['ranges']['z'][1] - one['ranges']['z'][0] == \
            pytest.approx(FIT_Z_SPAN, abs=1e-3)
        assert back['ranges']['z'][1] - back['ranges']['z'][0] == \
            pytest.approx(both['ranges']['z'][1] - both['ranges']['z'][0],
                          abs=1e-3)
        assert one['aspectmode'] == 'manual'

    def test_freezing_back_leaves_the_camera_alone(self, pages):
        """The two buttons are independent: Freeze decides the box, and whatever
        ⌖ did to the view stays."""
        with page(pages['pinned']) as chrome:
            chrome.js("return window.clickEye('UPPER');")
            chrome.js(RECENTER)
            dollied, _ = measure(chrome)
            chrome.js(FREEZE)                        # -> fit, frames once
            fitted, _ = measure(chrome)
            chrome.js(FREEZE)                        # -> frozen again
            state, _ = measure(chrome)
        # Freezing back restores the pinned box and touches no camera value at
        # all -- the frame the fit transition produced is the frame that stays.
        assert state['ranges'] == dollied['ranges']
        assert state['pivot'] == pytest.approx(fitted['pivot'], abs=1e-9)
        assert state['radius'] == pytest.approx(fitted['radius'], abs=1e-9)
        # ...and ⌖'s dolly is not still in there: the fit transition re-framed.
        assert fitted['radius'] > dollied['radius'] * 2

    def test_recenter_is_live_in_fit_mode_too(self, pages):
        """Un-gated, and the same computation: in fit mode the box IS the
        content, so the ratio is ~1 and ⌖ undoes a manual zoom-out."""
        with page(pages['pinned']) as chrome:
            chrome.js("return window.clickEye('UPPER');")
            chrome.js(FREEZE)
            fitted, _ = measure(chrome)
            chrome.js("Plotly.relayout(document.querySelector("
                      "'.js-plotly-plot'), {'scene.camera.eye.z': -8});"
                      "return 1;")
            zoomed, _ = measure(chrome)
            chrome.js(RECENTER)
            state, _ = measure(chrome)
        assert zoomed['radius'] > fitted['radius'] * 2
        assert state['radius'] == pytest.approx(fitted['radius'], rel=0.02)
        assert state['ranges'] == fitted['ranges']


@requires_chrome
class TestControlsAreIndependent:
    """Copy and affordance: neither button mentions the other."""

    def test_neither_hint_makes_the_other_a_precondition(self, pages):
        with page(pages['pinned']) as chrome:
            frozen_tips = chrome.js("return [document.getElementById("
                                    "'drocat-freeze-toggle')"
                                    ".getAttribute('data-drocat-tip'), "
                                    "document.getElementById('drocat-recenter')"
                                    ".getAttribute('data-drocat-tip')];")
            chrome.js(FREEZE)
            fit_tips = chrome.js("return [document.getElementById("
                                 "'drocat-freeze-toggle')"
                                 ".getAttribute('data-drocat-tip'), "
                                 "document.getElementById('drocat-recenter')"
                                 ".getAttribute('data-drocat-tip')];")
            both_visible = chrome.js(
                "return ['drocat-freeze-toggle', 'drocat-recenter'].map("
                "function (id) { return getComputedStyle("
                "document.getElementById(id)).display; });")
        assert all(tip and 'Freeze the view first' not in tip
                   for tip in frozen_tips + fit_tips)
        assert fit_tips[1] == frozen_tips[1], '⌖ says the same thing in both states'
        assert 'zoom' in fit_tips[1].lower()
        assert both_visible == ['flex', 'flex']

    def test_the_key_shortcuts_reach_the_same_actions(self, pages):
        with page(pages['pinned']) as chrome:
            chrome.js("return window.clickEye('UPPER');")
            chrome.js("document.dispatchEvent(new KeyboardEvent("
                      "'keydown', {key: 'c'})); return 1;")
            keyed, _ = measure(chrome)
        want = FRONT_RADIUS * BUNDLE_Z / PINNED_Z_SPAN
        assert keyed['radius'] == pytest.approx(want, rel=0.02)
