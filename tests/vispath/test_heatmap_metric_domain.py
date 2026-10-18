"""A classed heatmap's colourbars follow the METRIC's own unit.

A caller pins `fill_range` in the metric it built the ramps against — synapses,
0..800. The page also carries a `ratio` metric on 0..1, and following the pin
there would wash every cell white. The bars switch to 0..1 when the reader picks
the ratio.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vispath_pkg import VisualizePath  # noqa: E402
from vispath_pkg.vispath import VisConnMatInteractive  # noqa: E402
from heatmap_cdp import PAGE_READY, Page, REQUIRES_CDP  # noqa: E402

LABELS = ['A', 'B', 'C']


@REQUIRES_CDP
def test_the_ratio_metric_scales_the_bars_to_its_own_domain(tmp_path):
    edges = pd.DataFrame(
        [(a, b, float((i + 1) * 100), float((i + 1) / 20))
         for i, (a, b) in enumerate(
             [(a, b) for a in LABELS for b in LABELS])],
        columns=['source', 'target', 'weight', 'ratio'])
    vp = VisualizePath(path_file=edges, output_folder=str(tmp_path),
                       showfig=False, verbose=False)
    vp.base_filename = 'metric_domain'
    vp.build_network()
    out = vp.create_heatmap(
        cell_fill={(a, b): 'c' for a in LABELS for b in LABELS},
        fill_scale={'c': ['#ffffff', '#000000']},
        fill_range=(0, 800), fill_label='Class')
    bar = """
    var g = document.querySelector('.js-plotly-plot');
    var t = g.data.filter(function (x) { return x.showscale; })[0];
    return {zmin: t.zmin, zmax: t.zmax, ticks: t.colorbar.ticktext};
    """
    with Page(out) as page:
        assert page.wait(PAGE_READY)
        weight = page.js(bar)
        assert (weight['zmin'], weight['zmax']) == (0, 800), weight

        page.js("var s=document.getElementById('metricSelect');"
                "s.value='ratio'; s.dispatchEvent(new Event('change'));"
                "return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return g.data.some(function (x) {"
                         " return x.showscale && x.zmax === 1; });")
        ratio = page.js(bar)
        assert (ratio['zmin'], ratio['zmax']) == (0, 1), ratio
        assert ratio['ticks'] == ['0.0%', '25.0%', '50.0%', '75.0%', '100.0%'], ratio

        # The Cell style panel's own Min/Max follow the metric too.
        panel = ("return [...document.querySelectorAll("
                 "'#cellStyleColors input[type=number]')]"
                 ".map(function (i) { return i.value; });")
        assert page.js(panel) == ['0', '1'], page.js(panel)

        # The transforms must follow the METRIC: switching to ratio and then a
        # scale used to paint log10/sqrt of the SYNAPSE matrix against the ratio
        # domain (z 0.75..2.34 under log10), so every cell saturated.
        zmax_of = ("var g=document.querySelector('.js-plotly-plot');"
                   "var t=g.data.filter(function (x) { return x.showscale; })[0];"
                   "var z=[]; t.z.forEach(function (r) { r.forEach(function (v) {"
                   " if (v !== null) z.push(v); }); });"
                   "return Math.max.apply(null, z);")
        page.js("window.setScale('log10'); return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return g.data.some(function (x) {"
                         " return x.showscale && Math.abs(x.zmax - 0.30103) < 1e-4; });")
        assert page.js(zmax_of) < 0.5, page.js(zmax_of)

        page.js("window.setScale('sqrt'); return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return g.data.some(function (x) {"
                         " return x.showscale && x.zmax === 1; });")
        assert page.js(zmax_of) < 2, page.js(zmax_of)
        page.js("window.setScale('linear'); return true;")

        # An override is stored against the metric it was made in: a ratio edit
        # does not deform the synapse view, and vice versa.
        page.js("cellStyleSetRange('__fill__', 'c', 0.2, 0.5); return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return g.data.some(function (x) {"
                         " return x.showscale && x.zmax === 0.5; });")
        page.js("var s=document.getElementById('metricSelect');"
                "s.value='weight'; s.dispatchEvent(new Event('change'));"
                "return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return g.data.some(function (x) {"
                         " return x.showscale && x.zmax === 800; });")
        assert page.js(panel) == ['0', '800'], page.js(panel)

        # Contrast is decided against each cell's OWN colormap colour: a
        # near-white cell gets dark text, a near-black one light text.
        page.js("window.toggleCellValues(); return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return (g.layout.annotations || []).length === 9;")
        colours = page.js("var a=document.querySelector('.js-plotly-plot')"
                          ".layout.annotations;"
                          "return a.map(function (x) { return x.font.color; });")
        assert 'black' in colours and 'white' in colours, colours
        assert page.errors() == [], page.errors()


@REQUIRES_CDP
def test_the_adjusted_ratio_metric_is_selectable_on_its_own_domain(tmp_path):
    """A `ratio_adj` column becomes a THIRD metric option, on the ratio's own
    0..1 domain and with its own override key (not the plain ratio's)."""
    edges = pd.DataFrame(
        [(a, b, float((i + 1) * 100), float((i + 1) / 20),
          float((i + 1) / 10))
         for i, (a, b) in enumerate(
             [(a, b) for a in LABELS for b in LABELS])],
        columns=['source', 'target', 'weight', 'ratio', 'ratio_adj'])
    vp = VisualizePath(path_file=edges, output_folder=str(tmp_path),
                       showfig=False, verbose=False)
    vp.base_filename = 'metric_adj'
    vp.build_network()
    out = vp.create_heatmap(
        cell_fill={(a, b): 'c' for a in LABELS for b in LABELS},
        fill_scale={'c': ['#ffffff', '#000000']},
        fill_range=(0, 800), fill_label='Class')
    bar = """
    var g = document.querySelector('.js-plotly-plot');
    var t = g.data.filter(function (x) { return x.showscale; })[0];
    return {zmin: t.zmin, zmax: t.zmax, ticks: t.colorbar.ticktext};
    """
    panel = ("return [...document.querySelectorAll("
             "'#cellStyleColors input[type=number]')]"
             ".map(function (i) { return i.value; });")
    with Page(out) as page:
        assert page.wait(PAGE_READY)
        opts = page.js(
            "return [...document.querySelectorAll('#metricSelect option')]"
            ".map(function (o) { return [o.value, o.disabled]; });")
        assert opts[:3] == [['weight', False], ['ratio', False],
                            ['ratio_adj', False]], opts
        assert page.js("return (metricsData['ratio_adj'] || []).length;") == 3

        page.js("var s=document.getElementById('metricSelect');"
                "s.value='ratio_adj'; s.dispatchEvent(new Event('change'));"
                "return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return g.data.some(function (x) {"
                         " return x.showscale && x.zmax === 1; });")
        adj = page.js(bar)
        assert (adj['zmin'], adj['zmax']) == (0, 1), adj
        assert adj['ticks'] == ['0.0%', '25.0%', '50.0%', '75.0%', '100.0%'], adj
        assert page.js(panel) == ['0', '1'], page.js(panel)

        # Its override is keyed separately from the plain ratio's.
        page.js("cellStyleSetRange('__fill__', 'c', 0.3, 0.6); return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return g.data.some(function (x) {"
                         " return x.showscale && x.zmax === 0.6; });")
        page.js("var s=document.getElementById('metricSelect');"
                "s.value='ratio'; s.dispatchEvent(new Event('change'));"
                "return true;")
        assert page.wait("var g=document.querySelector('.js-plotly-plot');"
                         "return g.data.some(function (x) {"
                         " return x.showscale && x.zmax === 1; });")
        assert page.js(panel) == ['0', '1'], page.js(panel)
        assert page.errors() == [], page.errors()


@REQUIRES_CDP
def test_saved_metric_swaps_the_matrix_on_initialization(tmp_path):
    """A saved settings file naming `ratio` must load the RATIO matrix, not
    keep the default weight matrix under ratio labels (the bug showed the
    synapse count x100 in the percent cells)."""
    weight = pd.DataFrame([[10.0, 1000.0], [30.0, 40.0]],
                          index=['S0', 'S1'], columns=['T0', 'T1'])
    ratio = weight / 100000.0
    out = tmp_path / 'saved_metric.html'
    VisConnMatInteractive(weight, str(out), title='saved', showfig=False,
                          verbose=False, init_clustered=False,
                          matrices_dict={'weight': weight, 'ratio': ratio})
    assert out.exists()
    with Page(out) as page:
        assert page.wait(PAGE_READY)
        key = page.js("return storageKey;")
        saved = ('{"scale":"sqrt","currentMetric":"ratio",'
                 '"showCellValues":true,"showLabels":true}')
        page.js("localStorage.setItem(" + repr(key).replace("'", '"')
                + ", " + repr(saved) + "); location.reload(); return true;")
        assert page.wait(PAGE_READY)
        d = page.js("""
          var g = document.querySelector('.js-plotly-plot');
          var ann = (g.layout.annotations || []).map(function (a) {
              return String(a.text); });
          var t = g.data[0];
          return {metric: currentMetric,
                  linear: dataLinear[0][1],
                  ratio: metricsData['ratio'][0][1],
                  weight: metricsData['weight'][0][1],
                  ann: ann.sort(),
                  ticks: (t.colorbar || {}).ticktext || null};
        """)
        assert d['metric'] == 'ratio', d
        assert abs(d['linear'] - d['ratio']) < 1e-12, d
        assert '1.0' in d['ann'], d          # 0.01 as a percentage number
        assert '100000.0' not in d['ann'], d  # the weight x100 bug
        assert page.errors() == [], page.errors()
