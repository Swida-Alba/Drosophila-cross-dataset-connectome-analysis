"""A viewer's own zoom must not make the legend move the scene.

The frozen view pins the box (ranges + ``aspectmode='manual'``) and the legend
tree toggles context envelopes through ``opacity`` rather than ``visible``, so
nothing DROCAT writes should be able to re-frame the scene. Yet toggling a row
after wheel-zooming jumped the view. The cause is inside Plotly, not in those
pins: a 3D dolly keeps easing the scene camera after the last camera relayout
Plotly emits, so ``gd.layout.scene.camera`` holds a mid-animation value while
the viewer is looking at the eased end. The next ``Plotly.restyle`` -- which is
what every legend toggle is, tree eye or native legend click -- re-supplies
``_fullLayout`` from that stale layout and pushes it into the scene.

Measured before the guard, with the harness below:
  +59.9% of the settled dolly radius on a frozen BANC page,
  +68.2% on a Plotly page that carries no DROCAT script,
  and exactly 0.0 over the same window with no restyle (so it is not inertia).

The page-level fix is a camera shadow: copy the live scene camera into
``layout.scene.camera`` every frame until it stops moving, which makes any
later flush an identity. This module drives real headless Chrome to prove it,
because a DOM stub cannot see a camera that only disagrees after a gesture --
which is also how the earlier "camera bit-identical across every toggle" check
passed on a page that still jumped for users.

Run:
    conda run -n drocat-4.5.0 python -m pytest \
        tests/core/test_visualize_skeleton_camera_shadow.py -q
"""

import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import plotly.graph_objects as go
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visualize_skeleton import BRAIN_MESH_LEGEND_RANK, VisualizeSkeleton  # noqa: E402

# Float equality is safe for the copies themselves (the same numbers), but a
# slow machine can sample while the dolly is still easing; 1e-9 is far below any
# visible motion and far above the noise of a copied camera.
EXACT = 1e-9

CHROME_CANDIDATES = (
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta',
)


def _find_chrome():
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for name in ('google-chrome', 'chromium', 'chromium-browser', 'chrome'):
        found = shutil.which(name)
        if found:
            return found
    return None


CHROME = _find_chrome()
pytest.importorskip('websocket', reason='needs a websocket client for CDP')

requires_chrome = pytest.mark.skipif(
    CHROME is None, reason='no Chrome/Chromium binary found for the CDP probe')


def build_figure():
    """A skeleton plus one context envelope, the smallest scene that has both
    a legend-tree eye and a 3D camera to move."""
    xs = [0, 10, 20, 30, 40]
    neuron = go.Scatter3d(x=xs, y=xs, z=[0, 5, 10, 5, 0], mode='lines',
                         name='neuronA', legendgroup='neuronA',
                         line=dict(color='#1f77b4', width=4))
    envelope = go.Mesh3d(x=[0, 60, 30, 30], y=[0, 0, 60, 30], z=[0, 0, 0, 60],
                         i=[0, 0, 0, 1], j=[1, 2, 3, 2], k=[2, 3, 1, 3],
                         color='#74a8d6', opacity=0.04, name='ctx (brain)',
                         showlegend=True)
    envelope.legendrank = BRAIN_MESH_LEGEND_RANK
    return go.Figure(data=[neuron, envelope])


def write_guarded_page(tmp_path):
    """A viewer page as the permanent copies ship them: tree legend + freeze."""
    vis = object.__new__(VisualizeSkeleton)
    vis.background_color = 'rgba(255, 255, 255, 1.0)'
    vis.legend_mode = 'tree'
    vis.freeze_view = True
    vis.html_theme_toggle = False
    vis.save_folder = str(tmp_path)
    vis.saveas = 'scene'
    vis.dataset = 'hemibrain:v1.2.1'
    vis.fig_3d = build_figure()
    vis._in_page_warning_html = lambda: ''
    path = Path(tmp_path) / 'guarded.html'
    vis._write_plotly_html(build_figure(), str(path), legend_tree=True,
                           freeze_view=True, auto_open=False,
                           include_plotlyjs=True)
    return path


def write_naked_page(tmp_path):
    """The same figure as a plain Plotly page: no DROCAT script at all."""
    fig = build_figure()
    fig.update_layout(scene=dict(xaxis=dict(range=[-10, 70]),
                                 yaxis=dict(range=[-10, 70]),
                                 zaxis=dict(range=[-10, 70])))
    path = Path(tmp_path) / 'naked.html'
    fig.write_html(str(path), include_plotlyjs=True, auto_open=False)
    return path


CAM_STATE = r"""
var gd = document.querySelector('.js-plotly-plot');
var c = gd._fullLayout.scene._scene.getCamera();
var L = gd.layout.scene && gd.layout.scene.camera;
var F = gd._fullLayout.scene.camera;
var rad = function (e, ce) {
  return Math.hypot(e.x - ce.x, e.y - ce.y, e.z - ce.z);
};
return {
  live: rad(c.eye, c.center),
  layout: L && L.eye ? rad(L.eye, L.center) : null,
  fullLayout: rad(F.eye, F.center),
  liveEye: [c.eye.x, c.eye.y, c.eye.z],
  layoutEye: (L && L.eye) ? [L.eye.x, L.eye.y, L.eye.z] : null,
  shadowed: !!(L && L.eye),
  // the envelope is trace 1 of this fixture; the eye click is what has to
  // change it, and a guard that silently broke the toggle would also produce
  // a beautiful zero camera delta
  opacity: gd.data[1].opacity,
  visible: gd.data[1].visible === undefined ? null : gd.data[1].visible,
  errors: window.__probeErrors || []
};
"""

ARM_ERROR_WATCH = ("window.__probeErrors=[];"
                   "window.addEventListener('error',function(e){"
                   "window.__probeErrors.push(String(e.message));});"
                   "window.addEventListener('unhandledrejection',function(e){"
                   "window.__probeErrors.push('rejection: '+String(e.reason));});"
                   "return true;")


class Chrome:
    """Minimal CDP driver: launch, evaluate, wheel-dolly, click one eye."""

    def __init__(self, page_path):
        self.uid = tempfile.mkdtemp(prefix='drocat-chrome-')
        self.port = 9900 + random.randint(0, 99)
        self.ws = None
        self._id = 0
        self.proc = subprocess.Popen([
            CHROME, f'--user-data-dir={self.uid}', '--headless=new',
            f'--remote-debugging-port={self.port}', '--window-size=900,700',
            '--disable-blink-features=AutomationControlled',
            '--allow-file-access-from-files', '--hide-scrollbars',
            'file://' + str(page_path),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(80):
            try:
                with urllib.request.urlopen(
                        f'http://127.0.0.1:{self.port}/json', timeout=3) as r:
                    target = next(t for t in json.load(r)
                                  if t['type'] == 'page')
                import websocket
                self.ws = websocket.create_connection(
                    target['webSocketDebuggerUrl'], suppress_origin=True,
                    max_size=None)
                self.ws.settimeout(90)
                break
            except Exception:
                time.sleep(0.4)
        if self.ws is None:
            self.close()
            pytest.skip('Chrome exposed no DevTools page endpoint')
        self.send('Page.enable')
        self.send('Runtime.enable')

    def send(self, method, params=None):
        self._id += 1
        self.ws.send(json.dumps({'id': self._id, 'method': method,
                                 'params': params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get('id') == self._id:
                if 'error' in msg:
                    raise RuntimeError(msg['error'])
                return msg.get('result', {})

    def js(self, expr):
        out = self.send('Runtime.evaluate', {
            'expression': '(function(){' + expr + '})()', 'returnByValue': True})
        if out.get('exceptionDetails'):
            raise RuntimeError(json.dumps(out['exceptionDetails'])[:200])
        return out['result']['value']

    def wait_for(self, predicate_js, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            if self.js(predicate_js):
                return True
            time.sleep(0.5)
        return False

    def camera(self):
        return self.js(CAM_STATE)

    def arm_error_watch(self):
        self.js(ARM_ERROR_WATCH)

    def dolly(self, steps=6, delta=-240):
        """Wheel over the canvas: the gesture a viewer uses to zoom in."""
        rect = self.js("var q=document.querySelector('.js-plotly-plot canvas')"
                       ".getBoundingClientRect();"
                       "return [q.left, q.top, q.width, q.height];")
        x, y = rect[0] + rect[2] / 2, rect[1] + rect[3] / 2
        for _ in range(steps):
            self.send('Input.dispatchMouseEvent', {
                'type': 'mouseWheel', 'x': x, 'y': y, 'deltaX': 0,
                'deltaY': delta, 'button': 'none'})
            time.sleep(0.1)

    def click_envelope_eye(self, label='ctx (brain)'):
        point = self.js(
            "var rows=Array.from(document.querySelectorAll("
            "'#drocat-legend-tree .drocat-lt-row'));"
            "var row=rows.filter(function (r) {"
            "return (r.querySelector('.drocat-lt-label') || {})"
            ".textContent === __LABEL__; })[0];"
            "if (!row) { return null; }"
            "var e = row.querySelector('.drocat-lt-eye').getBoundingClientRect();"
            "return [e.left + e.width / 2, e.top + e.height / 2];"
            .replace('__LABEL__', json.dumps(label)))
        assert point, f'the tree panel has no row for {label!r}'
        for kind in ('mouseMoved', 'mousePressed', 'mouseReleased'):
            self.send('Input.dispatchMouseEvent', {
                'type': kind, 'x': point[0], 'y': point[1],
                'button': 'left' if kind != 'mouseMoved' else 'none',
                'clickCount': 1})

    def raw_restyle(self, index=1, opacity=0):
        """The restyle a legend click is, issued without any DROCAT code."""
        self.js("var gd=document.querySelector('.js-plotly-plot');"
                f"Plotly.restyle(gd, {{opacity:[{opacity}]}}, [{index}]);"
                "return 1;")

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()
        shutil.rmtree(self.uid, ignore_errors=True)


PANEL_READY = ("var g=document.querySelector('.js-plotly-plot');"
               "var p=document.getElementById('drocat-legend-tree');"
               "return !!(g && g._fullLayout && g._fullLayout.scene"
               " && g._fullLayout.scene._scene && p"
               " && p.style.display === 'block'"
               " && p.querySelectorAll('.drocat-lt-row').length);")

SCENE_READY = ("var g=document.querySelector('.js-plotly-plot');"
               "return !!(g && g._fullLayout && g._fullLayout.scene"
               " && g._fullLayout.scene._scene);")


@requires_chrome
class TestCameraShadowLive:
    """The gate: after a viewer zooms in, a legend toggle must not move them."""

    def test_tree_toggle_after_a_zoom_holds_the_framing(self, tmp_path):
        page = write_guarded_page(tmp_path)
        chrome = Chrome(page)
        try:
            assert chrome.wait_for(PANEL_READY), 'the legend tree never booted'
            chrome.arm_error_watch()
            start = chrome.camera()
            chrome.dolly()
            time.sleep(2.5)
            settled = chrome.camera()
            assert settled['shadowed'], 'no camera in gd.layout to shadow'
            # the guard's invariant: all three holders agree, so the flush a
            # restyle performs has nothing to change. A single-holder write
            # would leave _fullLayout on the opening camera and make the next
            # toggle apply the whole dolly at once.
            assert abs(settled['layout'] - settled['live']) <= EXACT, (
                f'the layout camera lags the scene: {settled["layout"]} vs '
                f'{settled["live"]}')
            assert abs(settled['fullLayout'] - settled['live']) <= EXACT, (
                f'the resolved camera lags the scene: '
                f'{settled["fullLayout"]} vs {settled["live"]}')

            chrome.click_envelope_eye()
            time.sleep(2.0)
            after = chrome.camera()
            # teeth inside the guarded case: the click has to have done the one
            # thing it claims, or a silent breakage would also read as "no
            # camera moved".
            assert after['opacity'] == 0, (
                f'the envelope eye did not hide the envelope '
                f'(opacity still {after["opacity"]})')
            assert after['visible'] is not False, (
                'the envelope eye must toggle opacity, not visibility -- the '
                'frozen framing depends on the trace staying in the scene')
            assert not after['errors'], f'the page raised: {after["errors"]}'
            jump = after['live'] - settled['live']
            assert abs(jump) <= EXACT, (
                f'the envelope eye moved the view by {jump:+g} '
                f'({100.0 * jump / settled["live"]:+.2f}% of the zoomed '
                f'radius {settled["live"]:g})')
            assert after['live'] < start['live'], (
                'the dolly itself never happened, so the assertion above '
                'would pass for a page that ignored the wheel')
        finally:
            chrome.close()

    def test_the_probe_has_teeth_without_the_guard(self, tmp_path):
        """A page without the shadow must still jump.

        This is not a feature request for the jump: it is the only proof that
        the measurement above can fail. If Plotly ever fixes the dolly/layout
        lag upstream, this test goes red -- and that is the moment to re-check
        whether the shadow is still load-bearing, not to delete the assertion.
        """
        page = write_naked_page(tmp_path)
        chrome = Chrome(page)
        try:
            assert chrome.wait_for(SCENE_READY), 'the scene never rendered'
            chrome.arm_error_watch()
            chrome.dolly()
            time.sleep(2.5)
            settled = chrome.camera()
            assert settled['layoutEye'] is not None, (
                'the dolly never reached gd.layout, so there is nothing to '
                'compare; the harness would pass vacuously')
            lag = abs(settled['layout'] - settled['live'])
            assert lag > 1e-3, (
                'Plotly no longer leaves the layout camera behind the eased '
                'dolly; re-check whether the page-side shadow is still needed')
            chrome.raw_restyle(index=1, opacity=0)
            time.sleep(2.0)
            after = chrome.camera()
            assert not after['errors'], f'the page raised: {after["errors"]}'
            assert after['opacity'] == 0, 'the restyle did not reach the trace'
            jump = after['live'] - settled['live']
            assert abs(jump) > 1e-3, (
                f'the restyle flush did not move the view ({jump}); the '
                f'measured lag of {lag:g} did not become a jump, so the '
                'guarded assertion above is not being tested here')
        finally:
            chrome.close()


class TestCameraShadowIsEmitted:
    """CI-portable pins: the guard exists, is armed on gestures, and is bounded.

    A browser check can only run where Chrome does; these assertions are what
    keeps the mechanism from being edited out silently in the meantime.
    """

    @staticmethod
    def _script():
        vis = object.__new__(VisualizeSkeleton)
        vis.background_color = 'rgba(255, 255, 255, 1.0)'
        vis.legend_mode = 'tree'
        ranges = {'x': [-1.0, 2.0], 'y': [-1.0, 2.0], 'z': [-1.0, 2.0]}
        return vis._freeze_view_html(ranges)

    def test_the_shadow_copies_the_live_camera_into_both_holders(self):
        script = self._script()
        assert 'function startCameraShadow()' in script
        assert 'function liveCamera()' in script
        assert 'gd._fullLayout.scene._scene' in script
        # A restyle re-supplies _fullLayout FROM the layout, so the layout write
        # is what makes the flush yield the current camera; the _fullLayout
        # write keeps the resolved holder from lagging an entire dolly and being
        # applied as one large move at toggle time.
        body = script.split('function shadowOnce() {')[1]
        body = body.split('function startCameraShadow()')[0]
        assert 'gd.layout.scene.camera = cam;' in body
        assert 'gd._fullLayout.scene.camera = cam;' in body
        assert 'Plotly.relayout' not in body, (
            'the shadow must not redraw: it only has to make the next flush '
            'an identity')

    def test_the_shadow_is_bounded_and_stops(self):
        script = self._script()
        assert 'var SHADOW_STABLE_FRAMES = 3;' in script
        assert 'var SHADOW_MAX_MS = 4000;' in script
        assert 'Date.now() - start < SHADOW_MAX_MS' in script
        assert 'shadowing = false;' in script

    def test_gestures_arm_the_shadow(self):
        script = self._script()
        # plotly's own camera relayouts (presets, the modebar reset, dolly
        # emissions) ...
        assert 'startCameraShadow();' in script.split(
            "gd.on('plotly_relayout', function(event) {")[1].split(
            "});")[0]
        # ... and the gestures that start before plotly has emitted anything
        assert "'wheel', 'pointerdown', 'touchstart'" in script
        assert "gd.addEventListener(type, startCameraShadow," in script
