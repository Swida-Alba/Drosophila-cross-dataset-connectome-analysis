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
        assert ratio['ticks'] == ['0', '0.25', '0.5', '0.75', '1'], ratio
        assert page.errors() == [], page.errors()
