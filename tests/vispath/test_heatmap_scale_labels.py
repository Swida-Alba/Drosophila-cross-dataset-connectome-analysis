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


def _export(tmp_path, name, matrix=WEIGHT, **kwargs):
    out = tmp_path / f'{name}.html'
    VisConnMatInteractive(matrix, str(out), title=name, showfig=False,
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


TINY = pd.DataFrame([[1e-11, 2e-11], [3e-11, 4e-11]],
                    index=['S0', 'S1'], columns=['T0', 'T1'])


@REQUIRES_CDP
def test_tiny_values_get_one_tick_per_value_not_hundreds_of_zeros(tmp_path):
    """A 1e-11 metric must not trip the tick epsilon: the absolute 1e-9 used
    to zero every tick and run the loop ~1e6 iterations past hi."""
    out = _export(tmp_path, 'tiny', matrix=TINY)

    with Page(out) as page:
        assert page.wait(PAGE_READY)
        linear = page.js(BAR)
        assert len(linear['ticks']) <= 6, linear
        assert linear['ticks'] == ['1e-11', '2e-11', '3e-11', '4e-11'], linear
        assert linear['tickvals'][-1] == pytest.approx(4e-11), linear

        page.js("window.setScale('log10'); return true;")
        assert page.wait(SCALED % math.log10(1 + 4e-11))
        logged = page.js(BAR)
        assert logged['ticks'] == linear['ticks'], logged
        assert logged['tickvals'][-1] == pytest.approx(math.log10(1 + 4e-11)), logged
        assert page.errors() == [], page.errors()


@REQUIRES_CDP
def test_constant_matrix_names_its_constant_not_a_fabricated_range(tmp_path):
    const = pd.DataFrame([[5.0, 5.0], [5.0, 5.0]],
                         index=['S0', 'S1'], columns=['T0', 'T1'])
    out = _export(tmp_path, 'constant', matrix=const)

    with Page(out) as page:
        assert page.wait(PAGE_READY)
        linear = page.js(BAR)
        assert linear['ticks'] == ['5'], linear
        assert linear['tickvals'] == [5], linear

        page.js("window.setScale('log10'); return true;")
        assert page.wait(SCALED % math.log10(6))
        logged = page.js(BAR)
        assert logged['ticks'] == ['5'], logged
        assert logged['tickvals'] == [pytest.approx(math.log10(6))], logged
        assert page.errors() == [], page.errors()


@REQUIRES_CDP
def test_applied_three_point_mapping_follows_the_scale(tmp_path):
    """The named mid value (100 of 0..900) must move to its transformed
    position: 0.111 linear, 0.678 log10, 1/3 sqrt."""
    matrix = pd.DataFrame([[0.0, 100.0], [450.0, 900.0]],
                          index=['S0', 'S1'], columns=['T0', 'T1'])
    out = _export(tmp_path, 'triplet', matrix=matrix)
    pos = """
    var g=document.querySelector('.js-plotly-plot');
    return {positions: customColorScale.map(function (c) { return c[0]; }),
            ticks: g.data[0].colorbar.ticktext,
            values: [document.getElementById('valueMin3').value,
                     document.getElementById('valueMid3').value,
                     document.getElementById('valueMax3').value]};
    """
    with Page(out) as page:
        assert page.wait(PAGE_READY)
        page.js("""
            document.getElementById('use3PointScale').checked = true;
            toggle3PointScale();
            document.getElementById('colorMin3').value = '#ffffff';
            document.getElementById('colorMid3').value = '#ff0000';
            document.getElementById('colorMax3').value = '#000000';
            document.getElementById('valueMin3').value = '0';
            document.getElementById('valueMid3').value = '100';
            document.getElementById('valueMax3').value = '900';
            applyCustomColors(); return true;""")
        assert page.wait("return !!customColorTriplets[currentMetric];")
        linear = page.js(pos)
        assert linear['positions'] == [pytest.approx(0.0),
                                       pytest.approx(100 / 900),
                                       pytest.approx(1.0)], linear
        assert linear['values'] == ['0', '100', '900'], linear

        page.js("window.setScale('log10'); return true;")
        logged_pos = math.log10(101) / math.log10(901)
        assert page.wait("return currentScale === 'log10' && "
                         f"Math.abs(customColorScale[1][0] - {logged_pos!r}) < 1e-9;")
        logged = page.js(pos)
        assert logged['positions'][1] == pytest.approx(logged_pos), logged
        assert logged['ticks'] == ['0', '250', '500', '750', '900'], logged

        page.js("window.setScale('sqrt'); return true;")
        assert page.wait("return currentScale === 'sqrt' && "
                         "Math.abs(customColorScale[1][0] - 1/3) < 1e-9;")
        rooted = page.js(pos)
        assert rooted['positions'][1] == pytest.approx(1 / 3), rooted

        # The triplet survives a save/reload and still follows later scales.
        page.js("saveSettings(); return true;")
        page.js("location.reload(); return true;")
        assert page.wait_reload(PAGE_READY)
        assert page.wait("return Math.abs(customColorScale[1][0] - 1/3) < 1e-9;")
        page.js("window.setScale('log10'); return true;")
        assert page.wait("return currentScale === 'log10' && "
                         f"Math.abs(customColorScale[1][0] - {logged_pos!r}) < 1e-9;")
        assert page.errors() == [], page.errors()


@REQUIRES_CDP
def test_custom_ranges_are_kept_per_metric(tmp_path):
    ratio = WEIGHT / 1000.0
    out = _export(tmp_path, 'per_metric',
                  matrices_dict={'weight': WEIGHT, 'ratio': ratio})
    read = """
    var g=document.querySelector('.js-plotly-plot');
    var f=g._fullData[0];
    return {metric: currentMetric, zmin: f.zmin, zmax: f.zmax,
            ticks: g.data[0].colorbar.ticktext,
            panel: [document.getElementById('valueMin2').value,
                    document.getElementById('valueMax2').value],
            active: window.customColorRange,
            stored: (customColorRanges || {})[currentMetric] || null};
    """
    with Page(out) as page:
        assert page.wait(PAGE_READY)
        page.js("""
            document.getElementById('use3PointScale').checked = false;
            toggle3PointScale();
            document.getElementById('valueMin2').value = '0';
            document.getElementById('valueMax2').value = '900';
            applyCustomColors(); return true;""")
        assert page.wait("return !!((customColorRanges||{}).weight);")
        weight = page.js(read)
        assert (weight['zmin'], weight['zmax']) == (0, 900), weight
        assert weight['panel'] == ['0', '900'], weight

        # The weight pin must not deform the ratio metric.
        page.js("var s=document.getElementById('metricSelect');"
                "s.value='ratio'; s.dispatchEvent(new Event('change'));"
                "return true;")
        assert page.wait(
            "var g=document.querySelector('.js-plotly-plot');"
            "return currentMetric === 'ratio' && "
            "Math.abs(g._fullData[0].zmax - 0.901) < 1e-12 && "
            "window.customColorRange === null;")
        ratio_view = page.js(read)
        assert ratio_view['ticks'][-1] == '90.1%', ratio_view
        assert ratio_view['panel'] == ['0', '0.901'], ratio_view

        # A ratio pin is remembered when the reader comes back to ratio.
        page.js("document.getElementById('valueMax2').value = '0.5';"
                "applyCustomColors(); return true;")
        assert page.wait("return !!((customColorRanges||{}).ratio);")
        page.js("var s=document.getElementById('metricSelect');"
                "s.value='weight'; s.dispatchEvent(new Event('change'));"
                "return true;")
        assert page.wait(
            "var g=document.querySelector('.js-plotly-plot');"
            "return currentMetric === 'weight' && "
            "Math.abs(g._fullData[0].zmax - 900) < 1e-12;")
        back = page.js(read)
        assert back['panel'] == ['0', '900'], back
        page.js("var s=document.getElementById('metricSelect');"
                "s.value='ratio'; s.dispatchEvent(new Event('change'));"
                "return true;")
        assert page.wait(
            "var g=document.querySelector('.js-plotly-plot');"
            "return currentMetric === 'ratio' && "
            "Math.abs(g._fullData[0].zmax - 0.5) < 1e-12;")
        again = page.js(read)
        assert again['panel'] == ['0', '0.5'], again
        assert again['active'] == {'min': 0, 'max': 0.5}, again
        assert page.errors() == [], page.errors()


@REQUIRES_CDP
def test_old_saved_transformed_range_is_migrated_to_metric_units(tmp_path):
    out = _export(tmp_path, 'old_saved', zmin=0, zmax=900,
                  color_scale=[[0.0, '#ffffff'], [1.0, '#68379c']])
    with Page(out) as page:
        assert page.wait(PAGE_READY)
        key = page.js("return storageKey;")
        old = ('{"scale":"log10","colorscale":"Custom","useAutoRange":false,'
               '"customColorRange":{"min":0,"max":2.954724790979063},'
               '"customColorScale":[[0,"rgb(255,255,255)"],'
               '[1,"rgb(104,55,164)"]],'
               '"use3PointScale":false,"currentMetric":"weight"}')
        page.js("localStorage.setItem(" + repr(key).replace("'", '"')
                + ", " + repr(old) + "); location.reload(); return true;")
        assert page.wait_reload(PAGE_READY)
        migrated = page.js(BAR)
        assert migrated['ticks'][-1] == '900', migrated
        assert migrated['zmax'] == pytest.approx(math.log10(901)), migrated
        stored = page.js("return customColorRanges.weight;")
        assert stored['min'] == pytest.approx(0), stored
        assert stored['max'] == pytest.approx(900), stored

        # Saving again writes the new per-metric format and round-trips.
        page.js("saveSettings(); return true;")
        page.js("location.reload(); return true;")
        assert page.wait_reload(PAGE_READY)
        again = page.js(BAR)
        assert again['ticks'] == ['0', '250', '500', '750', '900'], again
        assert again['zmax'] == pytest.approx(math.log10(901)), again
        assert page.errors() == [], page.errors()


@REQUIRES_CDP
def test_empty_or_reversed_three_point_values_are_refused(tmp_path):
    out = _export(tmp_path, 'bad_triplet',
                  color_scale=[[0.0, '#ffffff'], [0.5, '#ff0000'],
                               [1.0, '#000000']])
    with Page(out) as page:
        assert page.wait(PAGE_READY)
        page.js("window.__lastAlert = null;"
                "window.alert = function (m) { window.__lastAlert = String(m); };"
                "return true;")
        before = page.js("return customColorScale.map(function (c) {"
                         " return c[0]; });")
        page.js("document.getElementById('valueMid3').value = '';"
                "applyCustomColors(); return true;")
        assert 'numbers' in page.js("return window.__lastAlert;")
        after = page.js("return customColorScale.map(function (c) {"
                        " return c[0]; });")
        assert after == before, (after, before)

        page.js("document.getElementById('valueMin3').value = '900';"
                "document.getElementById('valueMid3').value = '100';"
                "document.getElementById('valueMax3').value = '100';"
                "applyCustomColors(); return true;")
        assert 'greater than min' in page.js("return window.__lastAlert;")
        assert page.errors() == [], page.errors()


@REQUIRES_CDP
def test_cell_style_bar_uses_the_same_tiny_value_ticks(tmp_path):
    fill = {(r, c): 'c' for r in ('S0', 'S1') for c in ('T0', 'T1')}
    out = _export(tmp_path, 'tiny_bars', matrix=TINY,
                  cell_fill=fill, fill_scale={'c': ['#ffffff', '#000000']},
                  fill_range=(0, 4e-11), fill_label='Class')
    bar = """
    var g=document.querySelector('.js-plotly-plot');
    var t=g.data.filter(function (x) { return x.showscale; })[0];
    return {ticks: t.colorbar.ticktext, tickvals: t.colorbar.tickvals};
    """
    with Page(out) as page:
        assert page.wait(PAGE_READY)
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return g.data.some(function (x) {"
                         " return x.showscale; });")
        got = page.js(bar)
        assert len(got['ticks']) <= 6, got
        assert got['ticks'][-1] == '4e-11', got
        assert got['tickvals'][-1] == pytest.approx(4e-11), got
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
