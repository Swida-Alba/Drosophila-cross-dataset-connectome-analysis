"""Shared report kit: clustering, metric styles, fragments, grids, heatmaps."""

import numpy as np
import pandas as pd
import pytest

from comparison import report_kit


def test_metric_style_selects_scale_by_key():
    positive = report_kit.metric_style('jaccard', 'Jaccard Similarity')
    assert positive.zmin == 0.0 and positive.zmax == 1.0
    assert positive.colorscale == report_kit.REPORT_POSITIVE_COLORSCALE

    diverging = report_kit.metric_style('rank_corr_union', 'Rank Correlation (Union)')
    assert diverging.zmin == -1.0 and diverging.zmax == 1.0
    assert diverging.colorscale == report_kit.REPORT_DIVERGING_COLORSCALE

    # unknown keys default to the positive scale
    assert report_kit.metric_style('weird').zmin == 0.0


def test_morph_v2_style_uses_diverging_scale():
    """vector_v2 is a whitened cosine that can go negative."""
    from comparison.morph_cross_dataset import _MORPH_V2_STYLE
    assert _MORPH_V2_STYLE.zmin == -1.0
    assert _MORPH_V2_STYLE.colorscale == report_kit.REPORT_DIVERGING_COLORSCALE


def test_cluster_heatmap_matrix_ward_order():
    pytest.importorskip("scipy")
    matrix = pd.DataFrame(
        [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
        index=["r0", "r1", "r2", "r3"],
        columns=["c0", "c1"],
    )
    ordered, clustered = report_kit.cluster_heatmap_matrix(matrix)
    assert clustered is True
    assert ordered.index.tolist() == ["r0", "r2", "r1", "r3"]
    assert ordered.columns.tolist() == ["c0", "c1"]
    # NaN cells survive clustering as blanks (ordering uses a finite copy)
    matrix.iloc[0, 0] = np.nan
    ordered, clustered = report_kit.cluster_heatmap_matrix(matrix)
    assert clustered is True
    assert int(ordered.isna().sum().sum()) == 1


def test_plotly_heatmap_fragment_uses_style_scale():
    pytest.importorskip("plotly")
    matrix = pd.DataFrame([[-1.0, 0.0, 1.0]],
                          index=["r"], columns=["n", "z", "p"])
    style = report_kit.MetricStyle(
        'morph_v2', 'Vector v2', report_kit.REPORT_DIVERGING_COLORSCALE,
        -1.0, 1.0)
    fragment, clustered = report_kit.plotly_heatmap_fragment(
        matrix, 't', style, 'x', 'y')
    assert fragment is not None
    assert '"colorscale":[[0.0,"#053061"]' in fragment
    assert 'Vector v2' in fragment
    assert clustered is True


def test_metric_grid_renders_one_card_per_style(tmp_path):
    pytest.importorskip("plotly")
    matrix = pd.DataFrame([[0.5]], index=["a"], columns=["b"])
    lines = []
    report_kit.append_report_metric_grid(
        lines, tmp_path, {'morph_v2': matrix}, 'MCNS → FAFB · Type level',
        {'morph_v2': 'MCNS_to_FAFB/results/morph_type_matrix.csv'},
        {'morph_v2': None}, 'Target type', 'Source type',
        {'include_plotlyjs': False},
        styles={'morph_v2': report_kit.MetricStyle(
            'morph_v2', 'Vector v2',
            report_kit.REPORT_DIVERGING_COLORSCALE, -1.0, 1.0)})
    text = '\n'.join(lines)
    assert "Vector v2" in text
    assert "MCNS_to_FAFB/results/morph_type_matrix.csv" in text
    assert "heatmap-card" in text


def test_generate_standalone_heatmaps_writes_and_falls_back(tmp_path):
    """Non-finite cells are rendered as 0 and the count is returned."""
    matrix = pd.DataFrame(
        [[1.0, np.nan], [np.inf, 0.5]],
        index=["a", "b"], columns=["a", "b"],
    )
    saved = {'heatmaps_generated': []}
    replaced = report_kit.generate_standalone_heatmaps(
        {'MCNS_to_FAFB': {'morph_v2': matrix}},
        tmp_path / 'viz',
        {'morph_v2': report_kit.MetricStyle(
            'morph_v2', 'Vector v2',
            report_kit.REPORT_DIVERGING_COLORSCALE, -1.0, 1.0)},
        filename_builder=lambda group, key: f'heatmap_morph_{group}_type.html',
        group_display=lambda g: g.replace('_to_', ' → '),
        vispath_title=lambda g, k, gd: f'Vector v2 — {gd}',
        fallback_title=lambda g, k, gd: f'Cross-Dataset Morphology - {gd}',
        tqdm_desc='test', verbose=False,
        saved_files=saved)
    assert replaced == 2
    out = tmp_path / 'viz' / 'heatmap_morph_MCNS_to_FAFB_type.html'
    assert out.exists()
    assert saved['heatmaps_generated'] == [str(out)]
    # The caller-provided MetricStyle must reach the renderer: the embedded
    # custom scale is the diverging one (its dark end is #67001f), not the
    # default positive scale (whose dark end #a50f15 would silently appear).
    html = out.read_text(encoding='utf-8')
    assert 'customColorScale = [[0.0, "#053061"]' in html or \
           'customColorScale = [[0.0,"#053061"]' in html
