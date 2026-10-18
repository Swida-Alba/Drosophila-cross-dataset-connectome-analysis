"""Per-cell fill and outline on the interactive connection-matrix heatmap.

Filed by `clock-production` (`clock-production/DROCAT_issue_heatmap_per_cell_style.md`)
and widened by the user the same day: the channels must be caller-supplied AND
reader-adjustable from any per-cell field. They are the one part of the heatmap a
caller cannot express today — a Plotly heatmap colours a cell from
`colorscale(z)` and has no per-cell border at all.

Design decisions these tests pin (see
`_plan/plan-2026-10-17-heatmap-cell-style.md`):

* style is keyed **by label**, never by position: `displayXLabels`/`displayYLabels`
  are rewritten by transpose, clustering, custom order and the axis subset, and a
  node that is both source and target sits on BOTH axes, so a positional matrix
  desyncs. The key is always (source-side, target-side) in the caller's direction,
  whichever axis those currently sit on;
* a page whose caller passes nothing gains **no** style code at all — not an empty
  array, not a hidden control. That is what keeps the exported page and
  `tests/core/test_profiling_multi_dataset.py`'s `"xgap" not in` law untouched
  (certified byte-identical against the pre-feature module; see the plan);
* colour and border go through the ONE `Plotly.newPlot` `createHeatmap` already
  performs — no `Plotly.restyle` of nested keys, the trap the label-toggle defect
  showed (`_plan/plan-2026-10-13-heatmap-label-toggle-dead.md`);
* the unmapped colour and an unpriced category are loud: a category the caller did
  not price raises rather than rendering grey and reading like a real class;
* the metric keeps its numbers: `trace.text`/hover still carry the value, so this
  is a rendering channel and changes no computed number.

Run:
    conda run -n drocat-4.5.0 python -m pytest \\
        tests/vispath/test_heatmap_cell_style.py -q
Chrome-free laws still run on a host without it; the DOM laws report SKIPPED.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vispath_pkg.vispath import (  # noqa: E402
    CELL_OUTLINE_DEFAULT,
    CELL_UNMAPPED_COLOR,
    VisConnMatInteractive,
    _cell_style_js_chunks,
    _numeric_bins,
    normalize_cell_style,
)

from heatmap_cdp import PAGE_READY, Page, REQUIRES_CDP  # noqa: E402

VISPATH_SRC = (PROJECT_ROOT / "vispath-subproject" / "src"
               / "vispath_pkg" / "vispath.py").read_text(encoding='utf-8')

ROWS = ['S0', 'S1', 'S2', 'S3']
COLS = ['T0', 'T1', 'T2', 'T3']

#: Every style the fixture asks for. Keys are (row_label, col_label) — sources are
#: heatmap rows and targets are columns (`create_heatmap` pivots index=source).
FILL = {
    ('S0', 'T0'): '#e15759',   # red
    ('S0', 'T1'): '#4e79a7',   # blue
    ('S1', 'T1'): '#4e79a7',
    ('S1', 'T2'): '#59a14f',   # green  (3 distinct colours + unmapped)
    ('S3', 'T3'): '#e15759',
}
OUTLINED = {
    ('S0', 'T0'): True,
    ('S0', 'T1'): {'width': 4, 'color': '#000000'},
    ('S3', 'T3'): True,
}
FILL_KEY = lambda r, c: f'{r}\t{c}'  # noqa: E731  the documented key form


def _cmat(rows=None, cols=None, seed=3):
    rows = rows or ROWS
    cols = cols or COLS
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.integers(1, 90, size=(len(rows), len(cols))).astype(float),
                        index=rows, columns=cols)


def _export(tmp_path, name, **kwargs):
    out = tmp_path / f'{name}.html'
    VisConnMatInteractive(_cmat(), str(out), title=name, showfig=False,
                          verbose=False, init_clustered=False, **kwargs)
    assert out.exists() and out.stat().st_size > 1000, f'{name} was not exported'
    return out


def _styled_kwargs():
    return dict(cell_fill=dict(FILL), cell_outline=dict(OUTLINED))


# ---------------------------------------------------------------------------
# the normalizer (no browser): keying, defaults, and the two loud failures
# ---------------------------------------------------------------------------

def test_style_is_keyed_by_label_with_the_documented_joiner():
    fields, outline = normalize_cell_style(
        ROWS, COLS, cell_fill=dict(FILL), cell_outline=dict(OUTLINED))
    fill = fields[0]
    assert fill['kind'] == 'color', fill['kind']
    # colour categories are stored as the colour itself; the map keys are labels
    assert fill['values'][FILL_KEY('S0', 'T0')] == '#e15759', fill['values']
    assert FILL_KEY('S1', 'T2') in fill['values']
    assert (outline[FILL_KEY('S0', 'T1')]
            == {'width': 4, 'color': '#000000'}), outline
    # a bare True takes the DEFAULT border, not an unspecified one
    assert outline[FILL_KEY('S0', 'T0')] == CELL_OUTLINE_DEFAULT, outline
    # cells nobody styled are still drawn — in the unmapped colour
    assert fill['unmapped'] == CELL_UNMAPPED_COLOR, fill


def test_categories_are_priced_by_the_callers_scale_and_a_missing_one_raises():
    _fields, _outline = normalize_cell_style(
        ROWS, COLS, cell_fill={FILL_KEY('S0', 'T0'): 1},
        fill_scale={0: '#9aa0a6', 1: '#4e79a7'})
    with pytest.raises(ValueError) as exc:
        normalize_cell_style(ROWS, COLS,
                             cell_fill={FILL_KEY('S0', 'T0'): 7},
                             fill_scale={0: '#9aa0a6'})
    # the refusal names the offender: silently greying an unpriced category would
    # read to the reader as a real class
    assert '7' in str(exc.value), str(exc.value)


def test_a_cell_neither_row_nor_col_can_show_is_reported_not_dropped():
    """A style whose label no axis carries cannot render; saying nothing would
    let a renamed row look like an unstyled cell forever."""
    with pytest.warns(UserWarning) as rec:
        normalize_cell_style(ROWS, COLS,
                             cell_fill={FILL_KEY('S9', 'T0'): '#ffffff'})
    assert any('S9' in str(w.message) for w in rec), [str(w.message) for w in rec]


def test_a_numeric_field_bins_over_the_displayed_values():
    values = [float(i) for i in range(16)]
    bins = _numeric_bins(values, 4)
    assert len(bins) == 4
    assert bins[0][0] == 0.0 and bins[-1][1] == 15.0, bins
    # the bins cover the range without a gap, or a value would paint nothing
    for low, high in bins:
        assert low <= high, (low, high)


# ---------------------------------------------------------------------------
# an unstyled page gains nothing (the mechanism, then the emitted page)
# ---------------------------------------------------------------------------

def test_no_style_means_no_chunks_at_all():
    """The mechanism behind the byte-identity promise: every placeholder in the
    template interpolates the empty string, so an unstyled export is not merely
    similar to the old page, it is the old page."""
    chunks = _cell_style_js_chunks([], {})
    assert chunks and all(value == '' for value in chunks.values()), chunks


def test_each_style_placeholder_is_used_exactly_once():
    src = VISPATH_SRC
    for name in ('consts', 'functions', 'panel', 'context', 'trace', 'shapes',
                 'contrast', 'settings_save', 'settings_load'):
        assert src.count('{cell_style_' + name + '}') == 1, name
        assert f"chunks['{name}']" in src, name


@pytest.mark.parametrize('kwargs', [
    {},                                    # the default page
    {'cell_fill': {}, 'cell_outline': {}},  # present but empty == absent
])
def test_a_page_without_cell_style_carries_no_new_identifiers(tmp_path, kwargs):
    """Byte-identity checked the strong way: the emitted text must not contain
    the style code at all (not even an empty version of it), so every later law
    about the default figure still describes the real page."""
    html = _export(tmp_path, 'plain', **kwargs).read_text(encoding='utf-8')
    for token in ('cellStyleFields', 'cellOutlineMap', 'cellFillSource',
                  'layout.shapes', 'cellStyleState', 'cellStyleColorFor',
                  'cellStyleSection', 'normalize_cell_style'):
        assert token not in html, (token, html[:200])


# ---------------------------------------------------------------------------
# the rendered page (headless Chrome): what the reader actually sees
# ---------------------------------------------------------------------------

#: Read the figure through the page's own label lookup AND through Plotly's
#: rendered trace model, so "the S1/T2 cell is green" cannot pass on state that
#: never reached the figure. `ys[i]|xs[j]` is the DISPLAYED position of the pair,
#: which mirrors under transpose — the lookup below is what must not.
CELL_STYLE = r"""
var gd = document.querySelector('.js-plotly-plot');
var xs = gd.layout.xaxis.ticktext, ys = gd.layout.yaxis.ticktext;
function drawnColor(i, j) {
  var fd = gd._fullData[0];
  if (!fd || !fd.z || !fd.z[i]) return null;
  var cs = fd.colorscale;
  if (!cs || !cs.length) return null;
  var lo = (fd.zmin === undefined) ? 0 : fd.zmin;
  var hi = (fd.zmax === undefined) ? 1 : fd.zmax;
  var t = (hi === lo) ? 0 : (fd.z[i][j] - lo) / (hi - lo);
  for (var s = 0; s < cs.length; s++) { if (t <= cs[s][0] + 1e-9) return cs[s][1]; }
  return cs[cs.length - 1][1];
}
var out = {};
for (var i = 0; i < ys.length; i++) for (var j = 0; j < xs.length; j++) {
  out[ys[i] + '|' + xs[j]] = {
    color: (typeof cellStyleColorFor === 'function') ? cellStyleColorFor(i, j) : null,
    drawn: drawnColor(i, j),
    outlined: (typeof cellStyleOutlineFor === 'function') ? cellStyleOutlineFor(i, j) : null
  };
}
return {cells: out,
        shapes: gd.layout.shapes || [],
        traces: gd.data.length,
        xOrder: xs.join(','), yOrder: ys.join(','),
        z: gd.data[0].z, text: gd.data[0].text,
        colorscale: gd.data[0].colorscale,
        cbTicks: (gd.data[0].colorbar || {}).ticktext || null,
        panel: !!document.getElementById('cellFillSource'),
        swatches: document.querySelectorAll('#cellStyleColors input[type=color]').length,
        errors: window.__probeErrors || []};
"""

#: A restyle re-runs `createHeatmap`, which purges and rebuilds the figure, so a
#: poll on `_fullLayout` alone can read the previous render. The figure's own
#: shapes are what prove the new render landed.
RENDERED = """
var g = document.querySelector('.js-plotly-plot');
if (!g || !g._fullLayout || !g.data || !g.data[0]) return false;
var want = (typeof cellStyleShapes === 'function') ? cellStyleShapes().length : 0;
return (g.layout.shapes || []).length === want;
"""


def _state(page):
    page.wait(PAGE_READY)
    state = page.js(CELL_STYLE)
    assert '__eval_error__' not in state, state['__eval_error__']
    return state


def _rendered(page):
    assert page.wait(RENDERED), 'the restyled render never landed'
    assert page.wait(PAGE_READY)


def _fill_of(row, col):
    return FILL.get((row, col), CELL_UNMAPPED_COLOR)


def _assert_styles_follow_labels(state, transposed=False):
    """Every cell, read by the labels it is currently displayed under."""
    for key, cell in state['cells'].items():
        left, right = key.split('|')
        row, col = (right, left) if transposed else (left, right)
        want = _fill_of(row, col)
        assert str(cell['color']).lower() == want, (key, cell, want)
        assert str(cell['drawn']).lower() == want, (key, cell, want)
        assert bool(cell['outlined']) == ((row, col) in OUTLINED), (key, cell)
    drawn = sum(1 for cell in state['cells'].values() if cell['outlined'])
    assert len(state['shapes']) == drawn, (len(state['shapes']), drawn)


@REQUIRES_CDP
def test_each_cell_renders_its_caller_colour_and_only_outlined_cells_bordered(
        tmp_path):
    page_path = _export(tmp_path, 'styled', **_styled_kwargs())
    with Page(page_path) as page:
        page.arm_error_watch()
        st = _state(page)
        assert not st['errors'], st['errors']
        assert st['panel'], 'the style control was not rendered at all'
        _assert_styles_follow_labels(st)
        # three outlined cells, and each border is the cell's own unit rect
        assert st['shapes'] and len(st['shapes']) == len(OUTLINED), st['shapes']
        for shape in st['shapes']:
            assert shape['x1'] - shape['x0'] == 1.0, shape
            assert shape['y1'] - shape['y0'] == 1.0, shape
            assert shape['layer'] == 'above', shape
        assert st['traces'] == 1, 'the style must not add a trace'
        # the metric number survives the recolouring: this channel renders, it
        # does not re-compute
        assert st['text'][0][0], st['text'][0][0]
        # z is the cell's position on the stitched class scale: a flat class
        # sits at its band's middle (3 colours + unmapped = 0..3.5)
        assert max(max(row) for row in st['z']) == 3.5, st['z']
        # a discrete scale is two hard stops per category (3 colours + unmapped)
        assert len(st['colorscale']) == 8, st['colorscale']
        assert st['cbTicks'], 'a styled page should label its discrete colourbar'


@REQUIRES_CDP
def test_transpose_custom_order_and_clustering_keep_style_on_its_label(tmp_path):
    """The reorder-proof claim: the same label pair carries the same colour and
    border however the axes have been permuted."""
    page_path = _export(tmp_path, 'reorder', **_styled_kwargs())
    with Page(page_path) as page:
        page.arm_error_watch()
        _state(page)

        page.js("transposeMatrix(); return 1;")
        _rendered(page)
        transposed = _state(page)
        assert transposed['yOrder'] == ','.join(COLS), transposed['yOrder']
        _assert_styles_follow_labels(transposed, transposed=True)

        page.js("transposeMatrix(); return 1;")
        _rendered(page)
        before = _state(page)
        _assert_styles_follow_labels(before)

        # clustering has to actually move the axes, or this branch proves nothing
        page.js("toggleClustering('clustered'); return 1;")
        _rendered(page)
        clustered = _state(page)
        assert (clustered['xOrder'], clustered['yOrder']) != (
            before['xOrder'], before['yOrder']), clustered['xOrder']
        assert len(clustered['cells']) == 16, len(clustered['cells'])
        _assert_styles_follow_labels(clustered)

        # custom order: the same mutation `applyDragOrder()` performs
        page.js("toggleClustering('original'); return 1;")
        _rendered(page)
        page.js("""
            currentXLabels = ['T2', 'T3', 'T0', 'T1'];
            currentYLabels = ['S3', 'S0', 'S2', 'S1'];
            createHeatmap(); return 1;""")
        _rendered(page)
        reordered = _state(page)
        assert reordered['xOrder'] == 'T2,T3,T0,T1', reordered['xOrder']
        _assert_styles_follow_labels(reordered)

        assert not page.errors(), page.errors()


@REQUIRES_CDP
def test_the_axis_subset_keeps_style_only_on_the_cells_it_still_shows(tmp_path):
    """The value filter rewrites the display labels the same way the reorder
    paths do — and it is data-determined, so the law is 'every cell still shown
    keeps its own style', not a hand-picked 2x2."""
    page_path = _export(tmp_path, 'subset', **_styled_kwargs())
    with Page(page_path) as page:
        page.arm_error_watch()
        _state(page)
        page.js("""
            document.getElementById('dataFilterInput').value = '>60';
            dataFilterMode = 'hide';
            applyDataFilter(); return 1;""")
        _rendered(page)
        st = _state(page)
        assert 0 < len(st['cells']) < 16, len(st['cells'])
        _assert_styles_follow_labels(st)


@REQUIRES_CDP
def test_the_reader_can_recolour_a_category_and_get_the_metric_colour_back(tmp_path):
    page_path = _export(tmp_path, 'panel', **_styled_kwargs())
    with Page(page_path) as page:
        page.arm_error_watch()
        _state(page)
        # recolour the category the first swatch names, through the panel's own
        # control — a programmatic `.value =` would not fire the page's onchange
        first = page.js("""
            var sel = document.getElementById('cellFillSource');
            sel.value = '__fill__';
            sel.dispatchEvent(new Event('change'));
            var box = document.querySelector('#cellStyleColors input[type=color]');
            box.value = '#123456';
            box.dispatchEvent(new Event('input'));
            return box.dataset.valueKey;""")
        _rendered(page)
        assert str(first).lower() == '#4e79a7', first
        st = _state(page)
        for key, cell in st['cells'].items():
            row, col = key.split('|')
            want = ('#123456' if _fill_of(row, col) == '#4e79a7'
                    else _fill_of(row, col))
            assert str(cell['color']).lower() == want, (key, cell, want)
            assert str(cell['drawn']).lower() == want, (key, cell, want)

        page.js("""
            var sel = document.getElementById('cellFillSource');
            sel.value = 'metric';
            sel.dispatchEvent(new Event('change'));
            return 1;""")
        _rendered(page)
        st = _state(page)
        assert all(cell['color'] is None for cell in st['cells'].values()), st
        assert st['cbTicks'] is None, st['cbTicks']

        # one click back to the caller's default, unpriced category included
        page.js("document.getElementById('cellStyleReset').click(); return 1;")
        _rendered(page)
        st = _state(page)
        _assert_styles_follow_labels(st)
        assert st['cbTicks'], 'a styled page should label its discrete colourbar'

        # and the outline can be cleared to none without losing the fill
        page.js("""
            var o = document.getElementById('cellOutlineSource');
            o.value = 'none';
            o.dispatchEvent(new Event('change'));
            return 1;""")
        _rendered(page)
        st = _state(page)
        assert not st['shapes'], st['shapes']
        assert all(cell['outlined'] is None for cell in st['cells'].values()), st
        assert st['cells']['S0|T0']['color'], 'the fill must survive the outline'


@REQUIRES_CDP
def test_a_field_label_cannot_end_the_option_list(tmp_path):
    """Field names are caller text: a quote or a `<` must reach the panel as a
    label, never as markup (`json_safe` escapes the consts, `escapeHtml` the
    option rows the panel builds with innerHTML)."""
    nasty = 'a"b<c>\'d'
    page_path = _export(tmp_path, 'escaping',
                        cell_data={nasty: {FILL_KEY('S0', 'T0'): '#e15759'}})
    consts = page_path.read_text(encoding='utf-8').split(
        'const cellStyleFields = ')[1].split(';')[0]
    assert '\\u003c' in consts and '<' not in consts, consts[:200]
    with Page(page_path) as page:
        page.arm_error_watch()
        _state(page)
        got = page.js("""
            var opts = document.querySelectorAll('#cellFillSource option');
            return {n: opts.length,
                    texts: Array.prototype.map.call(opts, function (o) {
                        return o.textContent; }),
                    leaked: document.querySelectorAll('#cellFillSource svg').length,
                    errors: window.__probeErrors || []};""")
        assert got['n'] == 2, got
        assert got['texts'] == ['the metric', nasty], got
        assert got['leaked'] == 0, got
        assert not got['errors'], got


@REQUIRES_CDP
def test_a_numeric_field_is_binned_and_rebinnable(tmp_path):
    strength = {FILL_KEY(r, c): float(i * 4 + j)
                for i, r in enumerate(ROWS) for j, c in enumerate(COLS)}
    page_path = _export(tmp_path, 'numeric', cell_data={'strength': strength},
                        cell_outline={FILL_KEY('S0', 'T0'): True})
    with Page(page_path) as page:
        page.arm_error_watch()
        st = _state(page)
        assert not st['errors'], st['errors']
        assert len(st['shapes']) == 1, st['shapes']
        low = st['cells']['S0|T0']
        high = st['cells']['S3|T3']
        assert low['color'] != high['color'], (low, high)
        assert low['drawn'] == low['color'], low
        # 4 bins and no unmapped bucket: every cell carries a strength
        assert len(st['cbTicks']) == 4, st['cbTicks']
        assert st['swatches'] == 4, st['swatches']

        page.js("document.getElementById('cellBinCount').value = '6';"
                "setCellBinCount('6'); return 1;")
        _rendered(page)
        st = _state(page)
        assert len(st['cbTicks']) == 6, st['cbTicks']
        assert st['swatches'] == 6, st['swatches']


@REQUIRES_CDP
def test_the_default_page_still_draws_one_unshaped_trace(tmp_path):
    """Same browser, same read, no maps: the figure is untouched."""
    with Page(_export(tmp_path, 'plain2')) as page:
        page.arm_error_watch()
        st = _state(page)
        assert st['traces'] == 1, st['traces']
        assert len(st['shapes']) == 0, st['shapes']
        assert not st['panel'], 'the control appeared on a page nobody styled'
        assert st['z'] and st['z'][0][0] is not None
        assert st['cbTicks'] is None, st['cbTicks']
        assert not st['errors'], st['errors']


def test_the_caller_names_its_fill_field_and_the_default_stays_generic():
    """A categorical legend titled "per-cell fill" tells a reader nothing about
    what its classes mean; the caller may name the field it is encoding."""
    named, _ = normalize_cell_style(ROWS, COLS, cell_fill=dict(FILL),
                                    fill_label='Conservation (datasets)')
    assert named[0]['label'] == 'Conservation (datasets)', named[0]
    default, _ = normalize_cell_style(ROWS, COLS, cell_fill=dict(FILL))
    assert default[0]['label'] == 'per-cell fill', default[0]


@REQUIRES_CDP
def test_a_fully_styled_page_carries_no_unmapped_swatch(tmp_path):
    """The "—" bucket exists only when a shown cell is outside the field. On a
    page where every cell is classed, it would sit in the legend beside the
    caller's own grey class and read as a fifth category."""
    palette = ['#e15759', '#4e79a7', '#59a14f']
    every = {(r, c): palette[(i + j) % 3]
             for i, r in enumerate(ROWS) for j, c in enumerate(COLS)}
    page_path = _export(tmp_path, 'fully_styled', cell_fill=every,
                        fill_label='Class')
    with Page(page_path) as page:
        st = _state(page)
        assert not st['errors'], st['errors']
        assert '—' not in st['cbTicks'], st['cbTicks']
        assert len(st['cbTicks']) == 3, st['cbTicks']
        # three categories, two hard stops each, no unmapped bucket
        assert len(st['colorscale']) == 6, st['colorscale']
        assert max(max(row) for row in st['z']) == 2.5, st['z']


def test_a_category_can_carry_a_ramp_keyed_by_the_cells_value():
    fill = {FILL_KEY('S0', 'T0'): 'unique', FILL_KEY('S0', 'T1'): 'unique'}
    vals = {FILL_KEY('S0', 'T0'): 10.0, FILL_KEY('S0', 'T1'): 30.0}
    fields, _ = normalize_cell_style(
        ROWS, COLS, cell_fill=fill,
        fill_scale={'unique': ['#ffffff', '#000000']}, fill_value=vals)
    f = fields[0]
    assert f['ramps']['unique'] == ['#ffffff', '#000000'], f
    assert f['ranges']['unique'] == [10.0, 30.0], f
    assert f['values_num'][FILL_KEY('S0', 'T1')] == 30.0, f
    with pytest.raises(ValueError) as exc:
        normalize_cell_style(ROWS, COLS, cell_fill=fill,
                             fill_scale={'unique': ['#ffffff']})
    assert 'two' in str(exc.value), str(exc.value)


@REQUIRES_CDP
def test_a_ramp_class_shades_by_value_and_names_its_scale(tmp_path):
    """One class, four values: the shade must move along the ramp and the
    legend band must name the value range, not a single swatch."""
    matrix = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]],
                          index=['S0', 'S1'], columns=['T0', 'T1'])
    fill = {(r, c): 'class' for r in ('S0', 'S1') for c in ('T0', 'T1')}
    value = {('S0', 'T0'): 10.0, ('S0', 'T1'): 20.0,
             ('S1', 'T0'): 30.0, ('S1', 'T1'): 40.0}
    ramp = ['#f7f7f7', '#cccccc', '#969696', '#525252', '#000000']
    out = tmp_path / 'ramp.html'
    VisConnMatInteractive(matrix, str(out), title='ramp', showfig=False,
                          verbose=False, init_clustered=False,
                          cell_fill=fill,
                          fill_scale={'class': ramp},
                          fill_value=value, fill_label='Class')
    with Page(out) as page:
        page.arm_error_watch()
        st = _state(page)
        assert not st['errors'], st['errors']
        colours = {k: cell['color'] for k, cell in st['cells'].items()}
        assert colours['S0|T0'] == '#f7f7f7', colours
        assert colours['S1|T1'] == '#000000', colours
        assert len(set(colours.values())) == 4, colours
        assert len(st['colorscale']) == 5, st['colorscale']
        assert st['cbTicks'] == ['class (10–40)'], st['cbTicks']
        # one band: every position inside [0, 1)
        assert 0 <= min(min(r) for r in st['z']), st['z']
        assert max(max(r) for r in st['z']) < 1, st['z']


def test_a_pinned_range_overrides_the_per_class_domain():
    fill = {FILL_KEY('S0', 'T0'): 'a', FILL_KEY('S0', 'T1'): 'b'}
    vals = {FILL_KEY('S0', 'T0'): 100.0, FILL_KEY('S0', 'T1'): 300.0}
    fields, _ = normalize_cell_style(
        ROWS, COLS, cell_fill=fill,
        fill_scale={'a': ['#ffffff', '#000000'],
                    'b': ['#ffffff', '#2ca02c']},
        fill_value=vals, fill_range=(0, 800))
    f = fields[0]
    assert f['range'] == [0.0, 800.0], f
    # the per-class ranges are still recorded; the pinned one wins
    assert f['ranges']['a'] == [100.0, 100.0], f['ranges']
    with pytest.raises(ValueError):
        normalize_cell_style(ROWS, COLS, cell_fill=fill,
                             fill_scale={'a': ['#ffffff', '#000000'],
                                         'b': ['#ffffff', '#2ca02c']},
                             fill_range=(800, 0))


PARALLEL = """
var g = document.querySelector('.js-plotly-plot');
var bars = g.data.filter(function (t) { return t.showscale; });
return {traces: g.data.length,
        bars: bars.map(function (t) {
            var z = [];
            t.z.forEach(function (r) { z = z.concat(r); });
            return {title: t.colorbar.title.text, x: t.colorbar.x,
                    zmin: t.zmin, zmax: t.zmax,
                    first: t.colorscale[0][1],
                    n: z.filter(function (v) { return v !== null; }).length,
                    nulls: z.filter(function (v) { return v === null; }).length};
        }),
        marginRight: g.layout.margin.r, width: g.layout.width};
"""


@REQUIRES_CDP
def test_a_pinned_range_draws_one_parallel_bar_per_class(tmp_path):
    """Same scale, one bar each: a stitched bar cannot show three windows of one
    0..800 domain without lying about one of them."""
    matrix = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]],
                          index=['S0', 'S1'], columns=['T0', 'T1'])
    fill = {('S0', 'T0'): 'a', ('S0', 'T1'): 'b',
            ('S1', 'T0'): 'a', ('S1', 'T1'): 'b'}
    out = tmp_path / 'parallel.html'
    VisConnMatInteractive(matrix, str(out), title='parallel', showfig=False,
                          verbose=False, init_clustered=False,
                          cell_fill=fill,
                          fill_scale={'a': ['#ffffff', '#000000'],
                                      'b': ['#ffffff', '#2ca02c']},
                          fill_range=(0, 800), fill_label='Classes')
    with Page(out) as page:
        page.arm_error_watch()
        assert page.wait(PAGE_READY)
        st = page.js(PARALLEL)
        assert not page.errors(), page.errors()
        assert len(st['bars']) == 2, st
        assert [b['title'] for b in st['bars']] == ['a', 'b'], st
        for bar in st['bars']:
            assert (bar['zmin'], bar['zmax']) == (0, 800), bar
            assert bar['first'] == '#ffffff', bar
            assert bar['n'] == 2 and bar['nulls'] == 2, bar
        assert st['bars'][0]['x'] != st['bars'][1]['x'], st
        # the canvas grows for the bars instead of squeezing the plot
        assert st['width'] > 800 and st['marginRight'] > 40, st


@REQUIRES_CDP
def test_a_pinned_domain_follows_the_scale_and_its_ramp_is_editable(tmp_path):
    """The Scale buttons must move a pinned domain with them (log ticks still
    name the original values), and a ramp must be editable like a flat class."""
    import math
    matrix = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]],
                          index=['S0', 'S1'], columns=['T0', 'T1'])
    fill = {(r, c): 'c' for r in ('S0', 'S1') for c in ('T0', 'T1')}
    out = tmp_path / 'scaled.html'
    VisConnMatInteractive(matrix, str(out), title='scaled', showfig=False,
                          verbose=False, init_clustered=False,
                          cell_fill=fill,
                          fill_scale={'c': ['#ffffff', '#969696', '#000000']},
                          fill_range=(0, 800), fill_label='Class')
    bar = """
    var g = document.querySelector('.js-plotly-plot');
    var t = g.data.filter(function (x) { return x.showscale; })[0];
    return {zmin: t.zmin, zmax: t.zmax, first: t.colorscale[0][1],
            ticks: t.colorbar.ticktext, tickvals: t.colorbar.tickvals,
            cell: cellStyleColorFor(0, 0)};
    """
    with Page(out) as page:
        page.arm_error_watch()
        assert page.wait(PAGE_READY)
        lin = page.js(bar)
        assert (lin['zmin'], lin['zmax']) == (0, 800), lin
        assert lin['ticks'] == ['0', '200', '400', '600', '800'], lin

        page.js("window.setScale('log10'); return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return g.data.some(function (x) {"
                         " return x.showscale && x.zmax > 0 && x.zmax < 10; });")
        logged = page.js(bar)
        assert abs(logged['zmax'] - math.log10(801)) < 1e-6, logged
        assert abs(logged['tickvals'][1] - math.log10(201)) < 1e-6, logged
        assert logged['ticks'] == ['0', '200', '400', '600', '800'], logged

        page.js("cellStyleSetRampEnd('__fill__', 'c', 0, '#ff0000'); return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "var t=g.data.filter(function(x){return x.showscale;})[0];"
                         "return t.colorscale[0][1] === '#ff0000';")
        edited = page.js(bar)
        assert edited['first'] == '#ff0000', edited
        ramps = page.js("return cellStyleState.ramps;")
        assert ramps['__fill__']['c'][0] == '#ff0000', ramps
        assert page.errors() == [], page.errors()


@REQUIRES_CDP
def test_each_colourbar_owns_its_value_range(tmp_path):
    """A per-class range override must move that bar alone, with ticks rounded
    to the range, and still follow the Scale buttons."""
    matrix = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]],
                          index=['S0', 'S1'], columns=['T0', 'T1'])
    fill = {('S0', 'T0'): 'a', ('S0', 'T1'): 'b',
            ('S1', 'T0'): 'a', ('S1', 'T1'): 'b'}
    out = tmp_path / 'ranges.html'
    VisConnMatInteractive(matrix, str(out), title='ranges', showfig=False,
                          verbose=False, init_clustered=False,
                          cell_fill=fill,
                          fill_scale={'a': ['#ffffff', '#000000'],
                                      'b': ['#ffffff', '#2ca02c']},
                          fill_range=(0, 800), fill_label='Classes')
    bar = """
    var g = document.querySelector('.js-plotly-plot');
    var t = g.data.filter(function (x) { return x.showscale
        && x.colorbar.title.text === '%s'; })[0];
    return {zmax: t.zmax, ticks: t.colorbar.ticktext};
    """
    with Page(out) as page:
        page.arm_error_watch()
        assert page.wait(PAGE_READY)
        assert page.js(bar % 'a')['ticks'] == ['0', '200', '400', '600', '800']

        page.js("cellStyleSetRange('__fill__', 'a', 0, 100); return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "var t=g.data.filter(function(x){return x.showscale"
                         " && x.colorbar.title.text==='a';})[0];"
                         "return t.zmax === 100;")
        a = page.js(bar % 'a')
        assert a['ticks'] == ['0', '25', '50', '75', '100'], a
        b = page.js(bar % 'b')
        assert b['ticks'] == ['0', '200', '400', '600', '800'], b
        # the override is stored against the metric it was made in
        assert page.js("return cellStyleState.ranges;") == {
            '__fill__': {'a\tweight': [0, 100]}}, page.js("return cellStyleState.ranges;")

        page.js("window.setScale('log10'); return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "var t=g.data.filter(function(x){return x.showscale"
                         " && x.colorbar.title.text==='a';})[0];"
                         "return t.zmax < 3;")
        logged = page.js(bar % 'a')
        assert logged['ticks'] == ['0', '25', '50', '75', '100'], logged
        assert page.errors() == [], page.errors()


@REQUIRES_CDP
def test_square_cells_keep_the_pinned_range_and_draw_no_grid(tmp_path):
    """The square-cells lock sizes the canvas from the width the page ACTUALLY
    renders — parallel colourbars make it wider than the window setting — so the
    anchor has nothing to pad. Padding the x range was the blank, grid-lined band
    between the row labels and the matrix."""
    matrix = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]],
                          index=['S0', 'S1'], columns=['T0', 'T1'])
    fill = {('S0', 'T0'): 'a', ('S0', 'T1'): 'b',
            ('S1', 'T0'): 'a', ('S1', 'T1'): 'b'}
    out = tmp_path / 'square.html'
    VisConnMatInteractive(matrix, str(out), title='square', showfig=False,
                          verbose=False, init_clustered=False,
                          cell_fill=fill,
                          fill_scale={'a': ['#ffffff', '#000000'],
                                      'b': ['#ffffff', '#2ca02c']},
                          fill_range=(0, 800), fill_label='Classes')
    geo = """
    var gd = document.getElementById('heatmap');
    var fl = gd._fullLayout;
    return {xr: fl.xaxis.range, xl: fl.xaxis._length, yl: fl.yaxis._length,
            gx: fl.xaxis.showgrid, gy: fl.yaxis.showgrid,
            w: fl.width, h: fl.height};
    """
    with Page(out) as page:
        assert page.wait(PAGE_READY)
        before = page.js(geo)
        assert before['gx'] is False and before['gy'] is False, before

        page.js("document.getElementById('squareCellsBtn').click(); return true;")
        assert page.wait("var fl=document.getElementById('heatmap')._fullLayout;"
                         "return Math.abs(fl.xaxis._length - fl.yaxis._length) < 1;")
        after = page.js(geo)
        assert after['xr'] == [-0.5, 1.5], after
        assert after['gx'] is False and after['gy'] is False, after
        assert page.errors() == [], page.errors()


@REQUIRES_CDP
def test_svg_export_keeps_the_raster_and_regroups_for_office(tmp_path):
    """The export must write the RENDERED width (a widened layout with parallel
    colourbars is wider than the window setting), leave the cells as Plotly's
    one-pixel-per-cell raster at any size, and hand Office three sibling
    objects: the cell image(s), the label text and the standalone colourbars."""
    import json
    rng = np.random.default_rng(3)
    matrix = pd.DataFrame(rng.integers(1, 90, size=(21, 21)).astype(float),
                          index=[f'R{i}' for i in range(21)],
                          columns=[f'C{i}' for i in range(21)])
    out = tmp_path / 'export.html'
    VisConnMatInteractive(matrix, str(out), title='export', showfig=False,
                          verbose=False, init_clustered=False)
    expr = """(async () => {
      window.__captured = null;
      const orig = window.downloadBlob;
      window.downloadBlob = function (blob) {
        return blob.text().then(function (t) { window.__captured = t; });
      };
      exportSVG();
      const t0 = Date.now();
      while (window.__captured === null && Date.now() - t0 < 25000) {
        await new Promise(r => setTimeout(r, 250));
      }
      window.downloadBlob = orig;
      const gd = document.getElementById('heatmap');
      return JSON.stringify({w: gd._fullLayout.width, h: gd._fullLayout.height,
                             svg: window.__captured});
    })()"""
    with Page(out) as page:
        assert page.wait(PAGE_READY)
        res = page.send('Runtime.evaluate', {'expression': expr,
                                            'awaitPromise': True,
                                            'returnByValue': True})
        d = json.loads(res['result']['value'])
        assert d['svg'], 'the export button produced nothing'
        layout = page.js("var fl=document.getElementById('heatmap')._fullLayout;"
                         "return {w: fl.width, h: fl.height};")
        assert (d['w'], d['h']) == (layout['w'], layout['h']), (d['w'], d['h'], layout)
        svg = d['svg']
        assert '<image' in svg, 'the cells stay an image'
        assert 'heatmap-cells-vector' not in svg, 'the cells are never rebuilt as shapes'
        assert 'class="drocat-cell-images"' in svg, 'the image family is its own group'
        assert 'class="drocat-labels"' in svg, 'the text family is its own group'
        assert svg.count('colorbar') >= 1, 'the colourbars stay standalone'
        assert '<text' in svg, 'labels are text'
        assert page.errors() == [], page.errors()


@REQUIRES_CDP
def test_node_universe_keeps_a_type_with_no_edge_on_the_axis(tmp_path):
    """A 21-type matrix built from an edge subset may not touch every type; the
    declared universe keeps the axis square instead of silently losing a row."""
    from vispath_pkg import VisualizePath
    edges = pd.DataFrame([('A', 'A', 1.0), ('A', 'B', 2.0), ('B', 'A', 3.0)],
                         columns=['source', 'target', 'weight'])
    vp = VisualizePath(path_file=edges, output_folder=str(tmp_path),
                       showfig=False, verbose=False)
    vp.base_filename = 'universe'
    vp.build_network()
    out = vp.create_heatmap(node_universe=['A', 'B', 'C'])
    with Page(out) as page:
        assert page.wait(PAGE_READY)
        shape = page.js("var t=document.querySelector('.js-plotly-plot').data[0];"
                        "return {x: t.x.length, y: t.y.length,"
                        " rows: t.z.length, cols: t.z[0].length};")
        assert shape == {'x': 3, 'y': 3, 'rows': 3, 'cols': 3}, shape
        assert page.errors() == [], page.errors()
