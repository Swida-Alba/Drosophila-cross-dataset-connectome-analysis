"""Cell values on the interactive connection-matrix heatmap are drawn ONCE.

Found 2026-10-18 from `clock-production`'s conservation page: with "Show
Values" on, `createHeatmap` set `trace.texttemplate` (which Plotly draws as
`g.heatmap-label`) *and* pushed one `layout.annotation` per cell. Linear showed
the same number twice, offset a pixel (a "shadow"); log10 and sqrt showed the
transformed value underneath the original, i.e. two different numbers in one
cell. Values now come from the annotations only, which is also what makes the
contrast colour work.

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

#: Values whose log₁₀ differs from the datum, so a scaled label is obvious.
VALUES = np.array([[10.0, 0.0, 100.0],
                   [0.0, 1000.0, 0.0],
                   [5.0, 0.0, 20.0]])


def _export(tmp_path):
    out = tmp_path / 'values.html'
    VisConnMatInteractive(pd.DataFrame(VALUES, index=ROWS, columns=COLS),
                          str(out), title='values', showfig=False,
                          verbose=False, init_clustered=False)
    assert out.exists() and out.stat().st_size > 1000
    return out


SHOW_VALUES = "window.toggleCellValues(); return true;"
STATE = """
var g = document.querySelector('.js-plotly-plot');
var ann = (g.layout.annotations || []).map(function (a) { return String(a.text); });
return {annotations: ann.sort(),
        values: g.data[0].z.map(function (r) { return r.map(String); })
                 .reduce(function (a, b) { return a.concat(b); }).sort(),
        heatLabels: document.querySelectorAll('g.heatmap-label').length,
        texttemplate: g.data[0].texttemplate === undefined ? null
                                                          : g.data[0].texttemplate,
        z11: g.data[0].z[1][1]};
"""


@REQUIRES_CDP
def test_cell_values_are_annotations_only_and_untransformed(tmp_path):
    page_path = _export(tmp_path)
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
        # ... and the one layer that remains carries the untransformed values.
        assert linear['texttemplate'] is None, linear
        assert linear['annotations'] == linear['values'], linear

        page.js("window.setScale('log10'); return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return g.data[0].z[1][1] < 10;")
        logged = page.js(STATE)
        assert logged['z11'] < 10, 'the log₁₀ scale did not land'
        assert logged['heatLabels'] == 0, logged
        assert logged['texttemplate'] is None, logged
        # The colour is log-scaled; the number in the cell is still the datum.
        assert logged['annotations'] == linear['values'], (logged, linear)

        page.js(SHOW_VALUES)
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return (g.layout.annotations || []).length === 0;")
        assert page.js(STATE)['heatLabels'] == 0
        assert page.errors() == []
