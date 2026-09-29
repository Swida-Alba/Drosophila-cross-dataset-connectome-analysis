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


def test_cluster_heatmap_matrix_with_linkage_returns_ward_linkages():
    pytest.importorskip("scipy")
    from scipy.cluster.hierarchy import leaves_list

    rng = np.random.default_rng(21)
    matrix = pd.DataFrame(
        rng.random((5, 4)),
        index=[f"r{i}" for i in range(5)],
        columns=[f"c{j}" for j in range(4)],
    )
    ordered, clustered, row_linkage, col_linkage = (
        report_kit.cluster_heatmap_matrix_with_linkage(matrix))
    assert clustered is True
    assert row_linkage.shape == (4, 4)
    assert col_linkage.shape == (3, 4)
    # The applied leaf order comes from exactly the returned linkage
    assert ordered.index.tolist() == [
        f"r{i}" for i in leaves_list(row_linkage).tolist()]
    assert ordered.columns.tolist() == [
        f"c{i}" for i in leaves_list(col_linkage).tolist()]


def test_cluster_heatmap_matrix_wrapper_matches_linkage_variant():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(22)
    matrix = pd.DataFrame(rng.random((4, 3)), index=list("abcd"),
                          columns=list("xyz"))
    ordered, clustered, row_linkage, _ = (
        report_kit.cluster_heatmap_matrix_with_linkage(matrix))
    wrapped_ordered, wrapped_clustered = report_kit.cluster_heatmap_matrix(matrix)
    assert wrapped_clustered == clustered
    assert wrapped_ordered.equals(ordered)
    assert row_linkage.shape == (3, 4)


def test_plotly_heatmap_fragment_draws_dendrograms_when_clustered():
    pytest.importorskip("plotly")
    import re

    rng = np.random.default_rng(23)
    matrix = pd.DataFrame(
        rng.random((5, 4)),
        index=[f"r{i}" for i in range(5)],
        columns=[f"c{j}" for j in range(4)],
    )
    style = report_kit.metric_style('weight', 'Synapses')
    fragment, clustered = report_kit.plotly_heatmap_fragment(
        matrix, 't', style, 'x', 'y')
    assert clustered is True
    assert fragment is not None
    # Column tree on y2 (shared x), row tree on x2 (shared y)
    assert '"yaxis":"y2"' in fragment
    assert '"xaxis":"x2"' in fragment
    assert '"yaxis2"' in fragment and '"xaxis2"' in fragment
    # Heatmap moved to numeric ticks with the labels as ticktext
    assert '"ticktext"' in fragment and '"tickvals"' in fragment
    # Orientation: the right tree's y values are leaf slots (0..n-1; internal
    # nodes sit at child midpoints); its x values are merge heights. A swapped
    # pair would clip the tree at xaxis2's height range and truncate it.
    match = re.search(r'"xaxis":"x2","y":\[([^\]]*)\]', fragment)
    assert match, "right dendrogram trace not found"
    slots = [float(v) for v in match.group(1).split(',') if v != 'null']
    assert min(slots) == 0.0 and max(slots) == 4.0  # first/last leaf stems


def test_plotly_heatmap_fragment_without_dendrogram_has_no_trees():
    pytest.importorskip("plotly")
    rng = np.random.default_rng(24)
    matrix = pd.DataFrame(
        rng.random((5, 4)),
        index=[f"r{i}" for i in range(5)],
        columns=[f"c{j}" for j in range(4)],
    )
    style = report_kit.metric_style('weight', 'Synapses')
    fragment, clustered = report_kit.plotly_heatmap_fragment(
        matrix, 't', style, 'x', 'y', show_dendrogram=False)
    assert clustered is True
    assert '"yaxis":"y2"' not in fragment
    assert '"xaxis":"x2"' not in fragment
    assert '"yaxis2"' not in fragment and '"xaxis2"' not in fragment
    assert '"ticktext"' not in fragment


def test_plotly_heatmap_fragment_single_row_has_column_tree_only():
    pytest.importorskip("plotly")
    matrix = pd.DataFrame([[0.2, 0.9, 0.4, 0.7]], index=["only"],
                          columns=["c0", "c1", "c2", "c3"])
    style = report_kit.metric_style('weight', 'Synapses')
    fragment, clustered = report_kit.plotly_heatmap_fragment(
        matrix, 't', style, 'x', 'y')
    assert clustered is True
    assert '"yaxis":"y2"' in fragment
    assert '"xaxis":"x2"' not in fragment


def test_plotly_heatmap_fragment_degenerate_matrix_stays_flat():
    pytest.importorskip("plotly")
    matrix = pd.DataFrame([[0.5]], index=["a"], columns=["b"])
    style = report_kit.metric_style('weight', 'Synapses')
    fragment, clustered = report_kit.plotly_heatmap_fragment(
        matrix, 't', style, 'x', 'y')
    assert clustered is True
    # 1x1: clustering "succeeds" trivially but both linkages are None, so
    # no dendrogram bands may appear
    assert '"yaxis":"y2"' not in fragment


def test_square_cells_anchors_cells_and_caps_natural_width():
    """square_cells locks cells 1:1 via the scaleanchor at any container
    width; small matrices keep the natural-width wrapper so a wide
    single-metric card does not center the band away from the header."""
    pytest.importorskip("plotly")
    matrix = pd.DataFrame(
        np.eye(4), index=[f"r{i}" for i in range(4)],
        columns=[f"c{i}" for i in range(4)])
    style = report_kit.metric_style('jaccard')
    fragment, _ = report_kit.plotly_heatmap_fragment(
        matrix, 't', style, 'x', 'y', square_cells=True)
    assert 'scaleanchor' in fragment
    assert 'heatmap-square-fit' in fragment


def test_dendrogram_band_alignment_tracks_constrained_domain(monkeypatch):
    """The band snap must map the cell edge through the LIVE _fullLayout
    domain, not the user layout domain.

    With square_cells the y axis uses constrain:'domain', so plotly
    satisfies the aspect ratio by shrinking the y DOMAIN (ranges stay
    exact): the pre-fix JS only compensated range padding and read the
    stale user domain, leaving the top tree band floating as much as a
    quarter of the plot height above the cells. Rendered here through the
    real production fragment (kaleido resolves the constraint), the ported
    VisPath math lands on the true cell edge while the retired formula
    leaves a visible gap.
    """
    pytest.importorskip("plotly")
    pytest.importorskip("scipy")
    pytest.importorskip("kaleido")
    import plotly.graph_objects as go

    rng = np.random.default_rng(1)
    n = 21
    matrix = pd.DataFrame(
        rng.random((n, n)), index=[f"r{i}" for i in range(n)],
        columns=[f"c{i}" for i in range(n)])
    style = report_kit.metric_style('jaccard')

    captured = {}
    original_to_html = go.Figure.to_html

    def spy(self, *args, **kwargs):
        captured['fig'] = self
        return original_to_html(self, *args, **kwargs)

    monkeypatch.setattr(go.Figure, 'to_html', spy)
    fragment, clustered = report_kit.plotly_heatmap_fragment(
        matrix, 't', style, 'x', 'y', square_cells=True)
    assert fragment is not None and clustered
    user_fig = captured['fig']
    assert user_fig.layout.yaxis2 is not None  # a top tree band exists

    width = 530  # two-per-row metric card
    user_fig.layout.width = width
    full = user_fig.full_figure_for_development()
    full_ya = full.layout.yaxis
    user_yd = user_fig.layout.yaxis.domain
    assert full_ya.domain[1] < user_yd[1] - 0.02, (
        "expected the constrained regime: plotly shrinks the y domain")

    def data_to_paper(value, axis):
        d0, d1 = axis.domain
        r0, r1 = axis.range
        return d0 + (value - r0) / (r1 - r0) * (d1 - d0)

    true_edge = data_to_paper(-0.5, full_ya)  # top row's top edge (reversed)

    # The ported VisPath math (report_script's alignDendrogramBands): frac
    # from the live range, mapped through the live domain.
    frac = ((-0.5) - full_ya.range[0]) / (full_ya.range[1] - full_ya.range[0])
    new_edge = full_ya.domain[0] + frac * (full_ya.domain[1] - full_ya.domain[0])
    assert new_edge == pytest.approx(true_edge, abs=1e-9)

    # The retired formula (range-padding only, user domain): in this regime
    # it either under-fires (padY == 0 when the range is exact) or maps
    # through the stale user domain — both leave a material gap.
    y_hi, y_lo = max(full_ya.range), min(full_ya.range)
    pad_y = (y_hi - (n - 0.5)) / (y_hi - y_lo)
    old_edge = user_yd[1] - max(pad_y, 0.0) * (user_yd[1] - user_yd[0])
    assert old_edge - true_edge > 0.05, (
        "the retired formula must fail visibly in the constrained regime")


def test_report_script_aligns_from_live_domains():
    """Source-level pins for the snap wiring: live-domain reads inside the
    update math, the retired user-layout read gone, a relayout re-align
    hook, and resize alignment chained onto the resize promise."""
    pytest.importorskip("plotly")
    script = report_kit.report_script()
    assert 'var yd = ya.domain;' in script
    assert 'var xd = xa.domain;' in script
    assert '((-0.5) - ya.range[0])' in script
    assert '((nX - 0.5) - xa.range[0])' in script
    assert 'var layout = plot.layout' not in script
    assert "plot.on('plotly_relayout'" in script
    assert '.then(done, done)' in script
    assert '__lastDendroAlign' in script


def test_square_cells_large_matrix_anchors_without_wrapper():
    pytest.importorskip("plotly")
    n = 35
    matrix = pd.DataFrame(
        np.eye(n), index=[f"r{i}" for i in range(n)],
        columns=[f"c{i}" for i in range(n)])
    style = report_kit.metric_style('jaccard')
    fragment, _ = report_kit.plotly_heatmap_fragment(
        matrix, 't', style, 'x', 'y', square_cells=True)
    assert 'scaleanchor' in fragment
    assert 'heatmap-square-fit' not in fragment


def test_non_square_cells_stay_free_aspect():
    """square_cells=False (e.g. the profiling inter-dataset grids) keeps the
    rectangular rendering — no anchor, no width cap."""
    pytest.importorskip("plotly")
    matrix = pd.DataFrame(
        np.eye(4), index=[f"r{i}" for i in range(4)],
        columns=[f"c{i}" for i in range(4)])
    style = report_kit.metric_style('jaccard')
    fragment, _ = report_kit.plotly_heatmap_fragment(
        matrix, 't', style, 'x', 'y', square_cells=False)
    assert 'scaleanchor' not in fragment
    assert 'heatmap-square-fit' not in fragment


def test_dendrogram_ticks_thin_on_large_matrices():
    """A 200-row dendrogram matrix thins the shared leaf-slot ticks to a
    readable stride (first and last leaf kept) instead of drawing every
    label at ~3px pitch."""
    pytest.importorskip("plotly")
    n = 200
    rng = np.random.default_rng(25)
    matrix = pd.DataFrame(
        rng.random((n, n)),
        index=[f"r{i}" for i in range(n)],
        columns=[f"c{i}" for i in range(n)],
    )
    style = report_kit.metric_style('jaccard')
    fragment, _ = report_kit.plotly_heatmap_fragment(
        matrix, 't', style, 'x', 'y')
    import re
    tick_arrays = re.findall(r'"tickvals":\[([0-9.,\s]+)\]', fragment)
    assert tick_arrays, 'dendrogram mode must carry explicit tickvals'
    for array in tick_arrays:
        ticks = [t for t in array.split(',') if t.strip()]
        assert len(ticks) <= 47
        assert ticks[0] == '0' and ticks[-1] == str(n - 1)
    # Small matrices keep every leaf labelled.
    small = pd.DataFrame(
        np.eye(5), index=[f"r{i}" for i in range(5)],
        columns=[f"c{i}" for i in range(5)])
    fragment, _ = report_kit.plotly_heatmap_fragment(
        small, 't', style, 'x', 'y')
    assert '"tickvals":[0,1,2,3,4]' in fragment
