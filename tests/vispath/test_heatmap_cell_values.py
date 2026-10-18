"""Cell values on the interactive connection-matrix heatmap: one layer, in-cell.

Found 2026-10-18 from `clock-production`'s conservation page, two defects in the
same block of `createHeatmap`:

* `trace.texttemplate = '%{text}'` was set *and* one `layout.annotation` per cell
  was pushed. Plotly drew both — `g.heatmap-label` (the TRANSFORMED value) under
  the annotation (the datum). Linear showed the same number twice, a pixel apart
  (a "shadow"); log₁₀/√ showed two different numbers in one cell.
* the annotation text was `String(originalValue)`, so a raw weight printed as
  `1.475572836012` — 14 characters in a 24 px cell, which reads as several
  numbers run together.

Values now come from the annotations only: synapse counts as whole numbers
and ratio metrics as percentages at one decimal,
and the font is re-hugged once to the cell the axis actually drew. The reader's
"Cell Value Size" stays the upper bound.

Run:
    conda run -n drocat-4.5.0 python -m pytest \\
        tests/vispath/test_heatmap_cell_values.py -q
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

from vispath_pkg.vispath import VisConnMatInteractive  # noqa: E402
from heatmap_cdp import PAGE_READY, Page, REQUIRES_CDP  # noqa: E402

ROWS = ['S0', 'S1', 'S2']
COLS = ['T0', 'T1', 'T2']

#: A long float and values whose log₁₀ differs from the datum, so a scaled or
#: unrounded label is obvious.
VALUES = np.array([[10.0, 0.0, 100.0],
                   [0.0, 1000.0, 1.475572836012],
                   [5.0, 25.0, 20.0]])
EXPECTED = ['0', '0', '1', '10', '100', '1000', '20', '25', '5']

SHOW_VALUES = "window.toggleCellValues(); return true;"
STATE = """
var g = document.querySelector('.js-plotly-plot');
var ann = (g.layout.annotations || []).map(function (a) { return String(a.text); });
return {annotations: ann.sort(),
        heatLabels: document.querySelectorAll('g.heatmap-label').length,
        texttemplate: g.data[0].texttemplate === undefined ? null
                                                          : g.data[0].texttemplate,
        z11: g.data[0].z[1][1]};
"""


def _export(tmp_path, name, matrix):
    out = tmp_path / f'{name}.html'
    VisConnMatInteractive(pd.DataFrame(matrix), str(out), title=name,
                          showfig=False, verbose=False, init_clustered=False)
    assert out.exists() and out.stat().st_size > 1000
    return out


@REQUIRES_CDP
def test_cell_values_are_one_annotation_layer_and_untransformed(tmp_path):
    page_path = _export(tmp_path, 'values',
                        pd.DataFrame(VALUES, index=ROWS, columns=COLS))
    with Page(page_path) as page:
        assert page.wait(PAGE_READY)
        hidden = page.js(STATE)
        assert hidden['annotations'] == [] and hidden['heatLabels'] == 0

        page.js(SHOW_VALUES)
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return (g.layout.annotations || []).length === 9;")
        linear = page.js(STATE)
        # One text per cell: Plotly's heatmap-label layer must be empty ...
        assert linear['heatLabels'] == 0, linear
        assert linear['texttemplate'] is None, linear
        # ... and the one layer that remains shows the datum at hover precision.
        assert linear['annotations'] == EXPECTED, linear

        page.js("window.setScale('log10'); return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return g.data[0].z[1][1] < 10;")
        logged = page.js(STATE)
        assert logged['z11'] < 10, 'the log₁₀ scale did not land'
        assert logged['heatLabels'] == 0, logged
        assert logged['texttemplate'] is None, logged
        # The colour is log-scaled; the number in the cell is still the datum.
        assert logged['annotations'] == EXPECTED, logged

        page.js(SHOW_VALUES)
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return (g.layout.annotations || []).length === 0;")
        assert page.js(STATE)['heatLabels'] == 0
        assert page.errors() == []


@REQUIRES_CDP
def test_a_dense_matrix_shrinks_its_labels_to_the_cell(tmp_path):
    rng = np.random.default_rng(7)
    matrix = rng.integers(0, 4000, size=(40, 40)).astype(float)
    page_path = _export(tmp_path, 'dense', matrix)
    with Page(page_path) as page:
        assert page.wait(PAGE_READY)
        page.js(SHOW_VALUES)
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return (g.layout.annotations || []).length === 1600;")
        # The re-hug runs after the first paint; wait for it to land.
        assert page.wait(FIT_LANDED, timeout=30), 'the cell labels never shrank'
        fit = page.js(FIT)
        assert fit['fontSize'] < 10, fit
        assert fit['maxWidth'] <= fit['pitch'] * 1.02, fit
        assert page.js(STATE)['heatLabels'] == 0
        assert page.errors() == []


FIT = """
var g = document.querySelector('.js-plotly-plot');
var pitch = g._fullLayout.xaxis._length / g.data[0].x.length;
var ws = [].map.call(document.querySelectorAll('g.annotation text'),
                     function (t) { return t.getBoundingClientRect().width; });
return {pitch: pitch, maxWidth: Math.max.apply(null, ws), n: ws.length,
        fontSize: g.layout.annotations[0].font.size};
"""

FIT_LANDED = ("var g=document.querySelector('.js-plotly-plot');"
              "return (g.layout.annotations || []).length === 1600"
              " && g.layout.annotations[0].font.size < 10;")


@REQUIRES_CDP
def test_ratio_cell_values_and_bars_read_as_percentages(tmp_path):
    """Ratio metrics label cells and colourbars as percentages (one
    decimal); synapse counts stay whole numbers."""
    weight = pd.DataFrame([[10.0, 20.0], [30.0, 40.0]],
                          index=['S0', 'S1'], columns=['T0', 'T1'])
    ratio = weight / 1000.0
    out = tmp_path / 'ratio_values.html'
    VisConnMatInteractive(weight, str(out), title='ratio_values',
                          showfig=False, verbose=False, init_clustered=False,
                          matrices_dict={'weight': weight, 'ratio': ratio})
    assert out.exists()
    with Page(out) as page:
        assert page.wait(PAGE_READY)
        page.js("var s=document.getElementById('metricSelect');"
                "s.value='ratio'; s.dispatchEvent(new Event('change'));"
                "return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return g.data[0].z[0][0] > 0.009 && "
                         "g.data[0].z[0][0] < 0.011;")
        page.js(SHOW_VALUES)
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return (g.layout.annotations || []).length === 4;")
        ann = page.js("var g=document.querySelector('.js-plotly-plot');"
                      "return (g.layout.annotations || []).map(function (a) {"
                      " return String(a.text); }).sort();")
        assert ann == ['1.0%', '2.0%', '3.0%', '4.0%'], ann
        cb = page.js("var t=document.querySelector('.js-plotly-plot')"
                     ".data[0];"
                     "return {title: (t.colorbar.title && t.colorbar.title.text)"
                     " || t.colorbar.title, ticks: t.colorbar.ticktext || null,"
                     " format: t.colorbar.tickformat};")
        assert '%' in str(cb['title']), cb
        assert cb['format'] == '.1%', cb
        assert page.errors() == [], page.errors()
