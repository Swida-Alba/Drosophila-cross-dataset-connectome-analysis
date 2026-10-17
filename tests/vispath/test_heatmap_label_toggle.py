"""The interactive heatmap's "Hide/Show Labels" button must actually toggle labels.

Filed by `clock-production` (`_plan/issues-2026-10-13-heatmap-toggle-labels-dead.md`)
and implemented in `_plan/plan-2026-10-13-heatmap-label-toggle-dead.md`: the emitted
`toggleLabels()` handed plotly nested `*.title.text` keys while the page stores every
one of those titles as a plain string, so `Plotly.restyle` threw `container is not an
object` inside the click handler - silently, after the button caption had already been
rewritten. Two reader symptoms, both covered here:

  * a large matrix (labels off at load) never showed its row/column names;
  * a small matrix (labels on) kept showing them while the button claimed otherwise.

These run the exported page in headless Chrome, because the template passes every
string-level check while the handler is dead - the reason `_plan/probes/` measured it
in a browser first. Marked `REQUIRES_CDP`, so a host without Chrome reports SKIPPED
rather than a false pass.

Run:
    conda run -n drocat-4.5.0 python -m pytest \\
        tests/vispath/test_heatmap_label_toggle.py -q
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vispath_pkg.vispath import VisConnMatInteractive  # noqa: E402

from heatmap_cdp import PAGE_READY, Page, REQUIRES_CDP  # noqa: E402

# above VisConnMatInteractive's is_large threshold (vispath.py), so the page opens
# with the tick labels hidden and the button reading "Show Labels"
LARGE_N = 120
SMALL_N = 21


def _cmat(rows, cols, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.random((rows, cols)),
        index=[f'S{i}' for i in range(rows)],
        columns=[f'T{i}' for i in range(cols)],
    )


def _export(tmp_path, name, rows, cols=None, seed=0, **kwargs):
    cols = rows if cols is None else cols
    out = tmp_path / f'{name}.html'
    VisConnMatInteractive(_cmat(rows, cols, seed), str(out),
                          title=f'{name} labels',
                          showfig=False, verbose=False,
                          metric_name='Jaccard', **kwargs)
    assert out.exists() and out.stat().st_size > 1000, f'{name} was not exported'
    return out


#: One read of everything the toggle is allowed to move, so an assertion can
#: compare picture and control in the same instant.
#: The square-cells lock is applied 250 ms after window load (a setTimeout in
#: the emitted page, by design). PAGE_READY says nothing about it, so a probe
#: that reads the layout straight after load is racing the timer — green on a
#: quiet host, red under the full suite (2026-10-16). Wait for the lock itself.
SQUARE_LOCKED = ("var g=document.querySelector('.js-plotly-plot');"
                 "return !!(g && g._fullLayout && g.layout"
                 " && g.layout.xaxis"
                 " && g.layout.xaxis.scaleanchor === 'y');")

STATE = r"""
var gd = document.querySelector('.js-plotly-plot');
return {
  ticks: document.querySelectorAll('.xtick text, .ytick text').length,
  xshow: gd.layout.xaxis.showticklabels,
  // Plotly writes the layout model first and rebuilds the SVG after; _fullLayout
  // only agrees once the redraw has landed. Reading the DOM before that agrees
  // yields a sample that is neither the old nor the new state, which is how a
  // correct page can look like a failed toggle.
  fullXshow: gd._fullLayout.xaxis.showticklabels,
  yshow: gd.layout.yaxis.showticklabels,
  btn: document.getElementById('toggleLabelsBtn').textContent.trim(),
  showLabels: showLabels,
  cbTitle: String((gd.data[0].colorbar || {}).title),
  xTitle: String(gd.layout.xaxis.title),
  yTitle: String(gd.layout.yaxis.title),
  xTitlefont: JSON.stringify(gd.layout.xaxis.titlefont || null),
  anchor: String(gd.layout.xaxis.scaleanchor),
  ratio: String(gd.layout.xaxis.scaleratio),
  errors: window.__probeErrors || []
};
"""

CLICK = "try{document.getElementById('toggleLabelsBtn').click();return 'clicked';}" \
        "catch(e){return 'THREW: '+String(e);}"

#: Canvas size plus the Plot Size panel's own reading of it - the fit is only
#: honest if the two agree.
FIT_STATE = r"""
var gd = document.querySelector('.js-plotly-plot');
return {
  width: gd.layout.width,
  height: gd.layout.height,
  wControl: parseInt(document.getElementById('widthInput').value),
  hControl: parseInt(document.getElementById('heightInput').value),
  wValue: document.getElementById('widthValue').textContent,
  hValue: document.getElementById('heightValue').textContent,
  fontSize: currentFontSize,
  errors: window.__probeErrors || []
};
"""


def _press(page):
    """Press the real button, as the reader does, and read the page back once
    Plotly says it has finished drawing.

    Sampling the model is not enough. `Plotly.restyle` lands its trace edit on a
    later render pass than the `relayout` that follows it, so a read taken between
    the two shows the axes hidden while the colourbar still carries its old title -
    measured on a real 21x21 export, where it made a correct page look broken. So
    count the graph div's own `plotly_afterplot` events and wait for that stream to
    go quiet. Nothing here presupposes the outcome: a dead handler fires no events,
    the wait expires, and the assertions below still judge the state it left.
    """
    if not page.js("return !!window.__drocatAfterPlot;"):
        page.js("window.__drocatAfterPlot = 0;"
                "document.querySelector('.js-plotly-plot')"
                ".on('plotly_afterplot', function(){ window.__drocatAfterPlot++; });"
                "return true;")

    def renders():
        return page.js("return window.__drocatAfterPlot;")

    # wait for the page to be idle before acting on it: a click landing in the
    # middle of the initial render sequence gets its restyle dropped by Plotly's
    # own re-supply pass, which is the mixed reading (axes hidden, colourbar still
    # titled) this used to catch a correct page for
    last, idle = renders(), 0
    deadline = time.time() + 30
    while idle < 3 and time.time() < deadline:
        time.sleep(0.25)
        now = renders()
        idle = idle + 1 if now == last else 0
        last = now
    before = last
    page.js(CLICK)
    state = page.js(STATE)
    last, quiet = None, 0
    for _ in range(60):
        time.sleep(0.25)
        state = page.js(STATE)
        assert isinstance(state, dict), f'page state unreadable: {state!r}'
        count = renders()
        if count == before:
            continue                       # nothing has redrawn yet
        if count == last:
            quiet += 1
            if quiet >= 4:                 # the render stream has stopped
                return state
        else:
            quiet = 0
        last = count
    return state


class TestLabelToggleBehaviour:
    """The button, the tick labels and the caption must agree, on both matrix
    sizes, with nothing thrown inside the handler."""

    @REQUIRES_CDP
    def test_large_matrix_shows_labels_on_press(self, tmp_path):
        page = Page(_export(tmp_path, 'large', LARGE_N, seed=0))
        try:
            assert page.wait(PAGE_READY), 'large heatmap never rendered'
            load = page.js(STATE)
            assert load['ticks'] == 0, \
                f'a >100 matrix opens with labels hidden, saw {load["ticks"]} ticks'
            assert load['xshow'] is False and load['btn'].startswith('🏷️ Show'), load

            shown = _press(page)
            assert shown['ticks'] == 2 * LARGE_N, \
                f'the click must draw {2 * LARGE_N} tick labels, drew {shown["ticks"]}'
            assert shown['xshow'] is True and shown['yshow'] is True, shown
            assert shown['btn'].startswith('🏷️ Hide'), shown
            assert shown['showLabels'] is True, shown

            hidden = _press(page)
            assert hidden['ticks'] == 0 and hidden['xshow'] is False, hidden
            assert hidden['btn'].startswith('🏷️ Show'), hidden
            assert page.errors() == [], page.errors()
        finally:
            page.close()

    @REQUIRES_CDP
    def test_small_matrix_hides_labels_on_press(self, tmp_path):
        """Symptom 2: the caption flipped while the picture stayed put."""
        page = Page(_export(tmp_path, 'small', SMALL_N, seed=1))
        try:
            assert page.wait(PAGE_READY), 'small heatmap never rendered'
            load = page.js(STATE)
            assert load['ticks'] == 2 * SMALL_N, load
            assert load['xshow'] is True and load['btn'].startswith('🏷️ Hide'), load

            hidden = _press(page)
            assert hidden['ticks'] == 0, \
                f'the click must remove the tick labels, {hidden["ticks"]} remain'
            assert hidden['xshow'] is False and hidden['btn'].startswith('🏷️ Show'), hidden

            shown = _press(page)
            assert shown['ticks'] == 2 * SMALL_N, shown
            assert shown['xshow'] is True and shown['btn'].startswith('🏷️ Hide'), shown
            assert page.errors() == [], page.errors()
        finally:
            page.close()

    @REQUIRES_CDP
    def test_caption_never_lies_about_the_picture(self, tmp_path):
        """Every press: the caption is exactly what `showLabels` says, which is
        what the picture shows. The defect wrote the caption before the update
        could throw, so control and render desynced permanently."""
        page = Page(_export(tmp_path, 'sync', SMALL_N, seed=2))
        try:
            assert page.wait(PAGE_READY), 'page never rendered'
            for _ in range(5):
                state = _press(page)
                wants_hide = state['showLabels'] is True
                assert state['btn'].startswith(
                    '🏷️ Hide' if wants_hide else '🏷️ Show'), state
                assert state['xshow'] is wants_hide, state
                assert (state['ticks'] > 0) is wants_hide, state
            assert page.errors() == [], page.errors()
        finally:
            page.close()

    @REQUIRES_CDP
    def test_colourbar_and_axis_titles_survive_the_round_trip(self, tmp_path):
        """Hiding writes '' and showing restores **the string the page drew**, not
        a recomputed one: the handler used the page-level `metricType` constant, so
        a labels round trip renamed a colourbar reading `Jaccard` to `Weight`."""
        page = Page(_export(tmp_path, 'titles', LARGE_N, seed=3))
        try:
            assert page.wait(PAGE_READY), 'page never rendered'
            load = page.js(STATE)
            assert load['cbTitle'], 'the colourbar is titled at load'
            assert load['xTitle'], 'the axis is titled at load'
            titlefont = load['xTitlefont']
            # Pinned asymmetry: a page that LOADS with the tick labels hidden
            # (is_large) still draws its colourbar and axis titles, because
            # createHeatmap does not consult showLabels for those. Toggling to the
            # hidden state clears them. Deliberate for now - the toggle follows the
            # template's stated intent ("hide ALL text elements") - so change it
            # here and in createHeatmap together if that ever gets overruled.
            assert load['cbTitle'] != '' and load['xshow'] is False, load

            other = _press(page)          # the opposite of the load state
            assert other['showLabels'] is (not load['showLabels']), other
            back = _press(page)           # same state class as load
            assert back['showLabels'] == load['showLabels'], back

            for state, shown in ((other, other['showLabels']), (back, back['showLabels'])):
                # tie the title claim to the picture, or this test would pass on a
                # page whose toggle threw and moved nothing at all
                assert (state['ticks'] > 0) is shown, state
                if shown:
                    assert state['cbTitle'] == load['cbTitle'], \
                        f'colourbar title drifted: {load["cbTitle"]!r} -> {state["cbTitle"]!r}'
                    assert state['xTitle'] == load['xTitle'], \
                        f'axis title drifted: {load["xTitle"]!r} -> {state["xTitle"]!r}'
                else:
                    assert state['cbTitle'] == '' and state['xTitle'] == '', state
            # the flat `xaxis.title` write must leave its sibling attributes alone
            assert back['xTitlefont'] == titlefont, back
            assert page.errors() == [], page.errors()
        finally:
            page.close()

    @REQUIRES_CDP
    def test_square_cells_lock_survives_the_toggle(self, tmp_path):
        """The handler also writes width/height/margins, so on a square-cells page
        it must not let the cell geometry come loose."""
        page = Page(_export(tmp_path, 'square', LARGE_N, seed=4, square_cells=True))
        try:
            assert page.wait(PAGE_READY), 'square page never rendered'
            assert page.wait(SQUARE_LOCKED), 'square lock never landed'
            load = page.js(STATE)
            assert load['anchor'] == 'y' and load['ratio'] == '1', load
            for _ in range(3):
                state = _press(page)
                assert state['anchor'] == 'y', state
                assert state['ratio'] == '1', state
            assert page.errors() == [], page.errors()
        finally:
            page.close()

    @REQUIRES_CDP
    def test_revealing_labels_on_a_large_map_fits_the_canvas(self, tmp_path):
        """A 245-name axis at the default 800 px is one label per 3 px, which is
        exactly why such a page opens hidden - so showing the labels has to make
        room for them, and hiding it has to give the room back. The size controls
        must follow, or the panel reports a plot size the page does not have."""
        rows, cols = LARGE_N, LARGE_N
        page = Page(_export(tmp_path, 'fit', rows, seed=6))
        try:
            assert page.wait(PAGE_READY), 'fit page never rendered'
            before = page.js(FIT_STATE)
            assert before['hControl'] == before['height'], before
            assert before['hValue'] == f"{before['height']}px", before

            _press(page)                       # press, and wait for the redraw
            shown = page.js(FIT_STATE)
            pitch = shown['fontSize'] + 2
            assert shown['width'] >= cols * pitch, \
                f'{cols} labels need >= {cols * pitch}px, canvas is {shown["width"]}'
            assert shown['height'] >= rows * pitch, \
                f'{rows} labels need >= {rows * pitch}px, canvas is {shown["height"]}'
            assert shown['height'] <= 4000 and shown['width'] <= 4000, shown
            # the panel must show the size the page actually drew
            assert shown['hControl'] == shown['height'], shown
            assert shown['wControl'] == shown['width'], shown
            assert shown['hValue'] == f"{shown['height']}px", shown
            assert shown['wValue'] == f"{shown['width']}px", shown

            _press(page)
            hidden = page.js(FIT_STATE)
            assert hidden['height'] == before['height'], hidden
            assert hidden['width'] == before['width'], hidden
            assert hidden['hControl'] == before['height'], hidden
            assert page.errors() == [], page.errors()
        finally:
            page.close()

    @REQUIRES_CDP
    def test_rectangular_matrix_toggles_every_label(self, tmp_path):
        """The cross-dataset morphology maps are rectangular (a 242 x 245 bodyId
        matrix is real output), where the tick total is rows + columns and each
        axis title counts a different length. A square-only test would miss the
        axis whose count is small."""
        rows, cols = LARGE_N + 10, 40
        page = Page(_export(tmp_path, 'rect', rows, cols, seed=5))
        try:
            assert page.wait(PAGE_READY), 'rectangular page never rendered'
            load = page.js(STATE)
            assert load['ticks'] == 0, load
            assert f'({cols} neurons)' in load['xTitle'], load['xTitle']
            assert f'({rows} neurons)' in load['yTitle'], load['yTitle']

            shown = _press(page)
            assert shown['ticks'] == rows + cols, \
                f'a {rows}x{cols} map must draw {rows + cols} tick labels, ' \
                f'drew {shown["ticks"]}'
            assert shown['xshow'] is True and shown['yshow'] is True, shown

            hidden = _press(page)
            assert hidden['ticks'] == 0, hidden
            assert page.errors() == [], page.errors()
        finally:
            page.close()
