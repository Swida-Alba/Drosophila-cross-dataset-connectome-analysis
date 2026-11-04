"""The Scale buttons move colour POSITIONS; labels stay in metric units.

A log10/sqrt heatmap used to relabel the colourbar with the transformed value:
a 901-synapse maximum read "2.95" over a bar ticking 0..3, and the colour
panel's Min/Max fields showed the same transformed numbers. The transform is a
positioning device only, so the matrix, zmin/zmax and tickvals are transformed
while every reader-facing number (colorbar ticks, panel fields) is the inverse.
"""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vispath_pkg.vispath import VisConnMatInteractive  # noqa: E402

from heatmap_cdp import PAGE_READY, Page, REQUIRES_CDP  # noqa: E402

WEIGHT = pd.DataFrame(
    [[10.0, 901.0, 5.0],
     [40.0, 300.0, 0.0],
     [1.0, 70.0, 120.0]],
    index=['S0', 'S1', 'S2'], columns=['T0', 'T1', 'T2'])

BAR = """
var g = document.querySelector('.js-plotly-plot');
var t = g.data[0];
var f = g._fullData[0];
return {ticks: t.colorbar.ticktext || null,
        tickvals: t.colorbar.tickvals || null,
        zmin: f.zmin, zmax: f.zmax};
"""

#: Wait for a scale re-render to land, not just for the click handler to run.
SCALED = """
var g = document.querySelector('.js-plotly-plot');
var v = g.data[0].colorbar.tickvals;
return !!(v && Math.abs(v[v.length - 1] - %s) < 1e-9);
"""


def _export(tmp_path, name, **kwargs):
    out = tmp_path / f'{name}.html'
    VisConnMatInteractive(WEIGHT, str(out), title=name, showfig=False,
                          verbose=False, init_clustered=False,
                          inline_plotly=True, **kwargs)
    assert out.is_file()
    return out


@REQUIRES_CDP
def test_linear_log_and_sqrt_bars_label_the_same_original_ticks(tmp_path):
    out = _export(tmp_path, 'scale_labels')

    with Page(out) as page:
        assert page.wait(PAGE_READY)
        linear = page.js(BAR)
        assert linear['ticks'] == ['0', '250', '500', '750', '901'], linear
        assert linear['tickvals'] == [0, 250, 500, 750, 901], linear

        page.js("window.setScale('log10'); return true;")
        assert page.wait(SCALED % math.log10(902))
        logged = page.js(BAR)
        assert logged['ticks'] == linear['ticks'], logged
        assert logged['tickvals'][0] == 0, logged
        assert logged['tickvals'][4] == pytest.approx(math.log10(902)), logged
        # The colour domain still spans the transformed maximum.
        assert logged['zmax'] == pytest.approx(math.log10(902)), logged

        page.js("window.setScale('sqrt'); return true;")
        assert page.wait(SCALED % math.sqrt(901))
        rooted = page.js(BAR)
        assert rooted['ticks'] == linear['ticks'], rooted
        assert rooted['tickvals'][4] == pytest.approx(math.sqrt(901)), rooted
        assert rooted['zmax'] == pytest.approx(math.sqrt(901)), rooted
        assert page.errors() == [], page.errors()


@REQUIRES_CDP
def test_custom_range_is_stored_in_metric_units_and_positioned_by_the_scale(tmp_path):
    out = _export(tmp_path, 'custom_range', zmin=0, zmax=900,
                  color_scale=[[0.0, '#ffffff'], [1.0, '#68379c']])

    with Page(out) as page:
        assert page.wait(PAGE_READY)
        linear = page.js(BAR)
        assert linear['ticks'] == ['0', '250', '500', '750', '900'], linear
        assert (linear['zmin'], linear['zmax']) == (0, 900), linear

        page.js("window.setScale('log10'); return true;")
        assert page.wait(SCALED % math.log10(901))
        logged = page.js(BAR)
        assert logged['ticks'] == linear['ticks'], logged
        assert logged['tickvals'][1] == pytest.approx(math.log10(251)), logged
        assert logged['tickvals'][4] == pytest.approx(math.log10(901)), logged
        # The panel speaks metric units too, not the log10 domain.
        panel = page.js("return [document.getElementById('valueMin2').value,"
                        " document.getElementById('valueMax2').value];")
        assert panel == ['0', '900'], panel
        assert page.errors() == [], page.errors()


@REQUIRES_CDP
def test_fractional_metric_ticks_keep_their_decimals(tmp_path):
    """Jaccard/cosine pages are 'weight' metrics with a display name; rounding
    every tick to a whole number would collapse 0/0.25/0.5/0.75/0.9 to 0/0/1/1."""
    jaccard = pd.DataFrame([[0.0, 0.25], [0.5, 0.9]],
                           index=['A', 'B'], columns=['A', 'B'])
    out = tmp_path / 'jaccard.html'
    VisConnMatInteractive(jaccard, str(out), title='jaccard', showfig=False,
                          verbose=False, init_clustered=False,
                          inline_plotly=True, metric_name='Jaccard')
    with Page(out) as page:
        assert page.wait(PAGE_READY)
        linear = page.js(BAR)
        assert linear['ticks'] == ['0', '0.25', '0.5', '0.75', '0.9'], linear

        page.js("window.setScale('log10'); return true;")
        assert page.wait(SCALED % math.log10(1.9))
        logged = page.js(BAR)
        assert logged['ticks'] == linear['ticks'], logged
        assert page.errors() == [], page.errors()


@REQUIRES_CDP
def test_ratio_bars_read_original_percentages_under_log(tmp_path):
    ratio = WEIGHT / 1000.0
    out = _export(tmp_path, 'ratio_labels',
                  matrices_dict={'weight': WEIGHT, 'ratio': ratio})

    with Page(out) as page:
        assert page.wait(PAGE_READY)
        page.js("var s=document.getElementById('metricSelect');"
                "s.value='ratio'; s.dispatchEvent(new Event('change'));"
                "return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return g.data[0].z[1][1] > 0.29 && "
                         "g.data[0].z[1][1] < 0.31;")

        page.js("window.setScale('log10'); return true;")
        assert page.wait(SCALED % math.log10(1.901))
        logged = page.js(BAR)
        assert logged['ticks'] == ['0.0%', '25.0%', '50.0%', '75.0%', '90.1%'], logged
        assert logged['tickvals'][4] == pytest.approx(math.log10(1.901)), logged
        assert page.errors() == [], page.errors()
