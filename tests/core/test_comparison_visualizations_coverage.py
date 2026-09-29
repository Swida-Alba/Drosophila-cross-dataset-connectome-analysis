"""Coverage tests for comparison.visualizations (ComparisonVisualizer).

Hermetic: synthetic pandas data only, matplotlib Agg backend, no network.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported anywhere

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from comparison.visualizations import ComparisonVisualizer

DATASETS = ["ds_one", "ds_two"]
THRESHOLDS = [1, 5]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def viz():
    return ComparisonVisualizer(verbose=False)


def _edge_df(n=4, weight_shift=0):
    """Result DataFrame per (dataset, threshold) with type columns + weight."""
    return pd.DataFrame(
        {
            "type_pre": [f"P{i}" for i in range(n)],
            "type_post": [f"Q{i}" for i in range(n)],
            "weight": [float(i + 1 + weight_shift) for i in range(n)],
            "has_valid_path": [True, False, True, True][:n],
        }
    )


@pytest.fixture
def results():
    return {
        "ds_one": {1: _edge_df(4, 0), 5: _edge_df(2, 1)},
        "ds_two": {1: _edge_df(4, 2), 5: _edge_df(3, 0)},
    }


@pytest.fixture
def aligned():
    idx = ["A -> B", "B -> C", "A -> C", "C -> D"]
    return pd.DataFrame(
        {"ds_one": [10.0, 0.0, 5.0, 3.0], "ds_two": [8.0, 4.0, 0.0, 3.0]},
        index=idx,
    )


@pytest.fixture
def similarities():
    return pd.DataFrame(
        [
            {
                "dataset_1": "ds_one",
                "dataset_2": "ds_two",
                "threshold": 1,
                "jaccard_similarity": 0.6,
                "svd_similarity": 0.7,
                "cosine_similarity": 0.8,
                "edge_rank_correlation": 0.5,
                "spearman_rank_correlation": 0.4,
                "weighted_jaccard": 0.55,
                "kernel_similarity": 0.65,
            }
        ]
    )


@pytest.fixture
def path_data():
    idx = ["A -> B", "A -> C", "B -> D"]
    return pd.DataFrame(
        {"ds_one": [9.0, 0.0, 6.0], "ds_two": [7.0, 5.0, 0.0]}, index=idx
    )


@pytest.fixture
def prob_data():
    idx = ["A -> B", "A -> C"]
    return pd.DataFrame(
        {"ds_one": [0.5, 0.01], "ds_two": [0.3, 0.2]}, index=idx
    )


@pytest.fixture
def ratio_data():
    idx = ["A -> B", "A -> C"]
    return pd.DataFrame(
        {"ds_one": [0.9, 0.1], "ds_two": [0.8, 0.0]}, index=idx
    )


# ---------------------------------------------------------------------------
# Path count plots
# ---------------------------------------------------------------------------

def test_plot_path_counts(viz, results):
    fig = viz.plot_path_counts(results, THRESHOLDS, nickname_map={"ds_one": "D1"})
    assert fig is not None
    assert len(fig.axes) >= 1


def test_plot_path_counts_missing_threshold(viz, results):
    # threshold 9 absent from results -> zero counts handled
    fig = viz.plot_path_counts(results, [1, 9])
    assert fig is not None


def test_plot_path_counts_stacked(viz, results):
    fig = viz.plot_path_counts_stacked(results, THRESHOLDS)
    assert fig is not None


# ---------------------------------------------------------------------------
# Edge heatmaps
# ---------------------------------------------------------------------------

def test_plot_edge_weight_heatmap(viz, aligned):
    fig = viz.plot_edge_weight_heatmap(
        aligned, DATASETS, max_edges=3, nickname_map={"ds_one": "D1"}
    )
    assert fig is not None


def test_plot_edge_weight_heatmap_no_columns(viz, aligned):
    fig = viz.plot_edge_weight_heatmap(aligned, ["missing_ds"])
    assert fig is not None  # fallback "No data available" figure


def test_plot_edge_weight_heatmap_all_thresholds(viz, results, aligned):
    fig = viz.plot_edge_weight_heatmap_all_thresholds(
        results, THRESHOLDS, align_func=lambda t: aligned, max_edges=3
    )
    assert fig is not None


def test_plot_edge_weight_heatmap_all_thresholds_empty(viz, results):
    empty = pd.DataFrame()
    fig = viz.plot_edge_weight_heatmap_all_thresholds(
        results, THRESHOLDS, align_func=lambda t: empty
    )
    assert fig is not None


def test_plot_edge_weight_heatmap_all_thresholds_raising(viz, results):
    def bad(t):
        raise RuntimeError("boom")

    fig = viz.plot_edge_weight_heatmap_all_thresholds(
        results, THRESHOLDS, align_func=bad
    )
    assert fig is not None


# ---------------------------------------------------------------------------
# Similarity matrices
# ---------------------------------------------------------------------------

def test_plot_similarity_matrix(viz, similarities):
    fig = viz.plot_similarity_matrix(similarities, metric="jaccard_similarity")
    assert fig is not None


def test_plot_dual_similarity_matrices(viz):
    sims = pd.DataFrame(
        [
            {
                "dataset_1": "ds_one",
                "dataset_2": "ds_two",
                "jaccard_similarity": 0.5,
                "svd_similarity": np.nan,  # exercises NaN branch
            }
        ]
    )
    fig = viz.plot_dual_similarity_matrices(sims)
    assert fig is not None
    assert len(fig.axes) >= 2  # two heatmaps (+ colorbar axes)


@pytest.mark.parametrize(
    "metric", ["jaccard", "weighted_jaccard", "svd", "edge_rank", "kernel"]
)
def test_plot_similarity_per_threshold(viz, similarities, metric):
    sims_t1 = similarities.copy()
    sims_t5 = pd.DataFrame()  # empty entry exercises fallback text
    fig = viz.plot_similarity_per_threshold(
        {1: sims_t1, 5: sims_t5}, metric=metric
    )
    assert fig is not None


def test_plot_similarity_per_threshold_none_entry(viz):
    fig = viz.plot_similarity_per_threshold({1: None})
    assert fig is not None


def test_plot_similarity_per_threshold_empty(viz):
    fig = viz.plot_similarity_per_threshold({})
    assert fig is not None


# ---------------------------------------------------------------------------
# Overlap / venn
# ---------------------------------------------------------------------------

def test_plot_edge_overlap(viz, aligned):
    fig = viz.plot_edge_overlap(aligned, DATASETS, threshold=1)
    assert fig is not None


def test_plot_edge_overlap_single_dataset(viz, aligned):
    fig = viz.plot_edge_overlap(aligned, ["ds_one"])
    assert fig is not None  # fallback figure


def test_plot_venn2_style(viz, aligned):
    fig = viz.plot_venn2_style(aligned, DATASETS, threshold=1)
    assert fig is not None


def test_plot_venn2_style_wrong_count(viz, aligned):
    with pytest.raises(ValueError):
        viz.plot_venn2_style(aligned, ["ds_one"])


def test_plot_venn2_style_missing_column(viz, aligned):
    fig = viz.plot_venn2_style(aligned, ["ds_one", "nope"])
    assert fig is not None  # fallback figure


# ---------------------------------------------------------------------------
# Scatter / distributions
# ---------------------------------------------------------------------------

def test_plot_weight_scatter(viz, aligned):
    fig = viz.plot_weight_scatter(aligned, "ds_one", "ds_two")
    assert fig is not None


def test_plot_weight_scatter_log(viz, aligned):
    fig = viz.plot_weight_scatter(aligned, "ds_one", "ds_two", log_scale=True)
    assert fig is not None


def test_plot_fold_change_distribution(viz):
    diff = pd.DataFrame({"max_fold_change": [0.5, 1.2, 2.4, 5.0]})
    fig = viz.plot_fold_change_distribution(diff)
    assert fig is not None


def test_plot_fold_change_distribution_missing_column(viz):
    fig = viz.plot_fold_change_distribution(pd.DataFrame({"x": [1, 2]}))
    assert fig is not None


def test_plot_similarity_vs_threshold(viz):
    df = pd.DataFrame(
        {
            "threshold": [1, 5, 1, 5],
            "dataset_1": ["a", "a", "b", "b"],
            "dataset_2": ["c", "c", "d", "d"],
            "jaccard_similarity": [0.4, 0.6, 0.2, 0.8],
        }
    )
    fig = viz.plot_similarity_vs_threshold(df)
    assert fig is not None


def test_plot_similarity_vs_threshold_empty(viz):
    fig = viz.plot_similarity_vs_threshold(pd.DataFrame())
    assert fig is not None


# ---------------------------------------------------------------------------
# Utility / saving
# ---------------------------------------------------------------------------

def test_save_figure(viz, tmp_path):
    fig = viz.plot_path_counts(
        {"d": {1: _edge_df(2)}}, [1]
    )
    out = tmp_path / "fig.png"
    viz.save_figure(fig, str(out), dpi=50)
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Composite subplots
# ---------------------------------------------------------------------------

def test_plot_threshold_comparison_subplots(viz, results):
    fig = viz.plot_threshold_comparison_subplots(results, THRESHOLDS)
    assert fig is not None


def test_plot_threshold_comparison_subplots_single_threshold(viz, results):
    fig = viz.plot_threshold_comparison_subplots(results, [1])
    assert fig is not None


def test_plot_similarity_matrix_per_threshold(viz, results, aligned, similarities):
    # Provide cached similarities including the advanced metric columns
    sims = similarities.copy()
    fig = viz.plot_similarity_matrix_per_threshold(
        results,
        THRESHOLDS,
        align_func=lambda t: aligned,
        similarity_func=lambda t: sims,
        show_progress=False,
    )
    assert fig is not None


def test_plot_similarity_matrix_per_threshold_recompute(viz, results, aligned):
    # No similarity_func -> metrics are recomputed from aligned data
    fig = viz.plot_similarity_matrix_per_threshold(
        results,
        [1],
        align_func=lambda t: aligned,
        show_progress=False,
    )
    assert fig is not None


def test_plot_similarity_matrix_per_threshold_empty(viz, results):
    fig = viz.plot_similarity_matrix_per_threshold(
        results, THRESHOLDS, align_func=lambda t: pd.DataFrame(),
        show_progress=False,
    )
    assert fig is not None


# ---------------------------------------------------------------------------
# Path-level heatmaps
# ---------------------------------------------------------------------------

def test_plot_path_heatmap(viz, path_data):
    fig = viz.plot_path_heatmap(
        path_data, DATASETS, nickname_map={"ds_one": "D1"}
    )
    assert fig is not None


def test_plot_path_heatmap_no_columns(viz, path_data):
    fig = viz.plot_path_heatmap(path_data, ["nope"])
    assert fig is not None


def test_plot_path_heatmap_all_thresholds(viz, path_data):
    fig = viz.plot_path_heatmap_all_thresholds(
        lambda t: path_data, THRESHOLDS, DATASETS
    )
    assert fig is not None


def test_plot_path_heatmap_all_thresholds_empty(viz):
    fig = viz.plot_path_heatmap_all_thresholds(
        lambda t: pd.DataFrame(), THRESHOLDS, DATASETS
    )
    assert fig is not None


def test_plot_path_heatmap_all_thresholds_raising(viz):
    def bad(t):
        raise RuntimeError("no data")

    fig = viz.plot_path_heatmap_all_thresholds(bad, THRESHOLDS, DATASETS)
    assert fig is not None


def test_plot_ratio_heatmap(viz, ratio_data):
    fig = viz.plot_ratio_heatmap(ratio_data, DATASETS)
    assert fig is not None


def test_plot_ratio_heatmap_no_columns(viz, ratio_data):
    fig = viz.plot_ratio_heatmap(ratio_data, ["nope"])
    assert fig is not None


def test_plot_traversal_probability_heatmap(viz, prob_data):
    fig = viz.plot_traversal_probability_heatmap(prob_data, DATASETS)
    assert fig is not None


def test_plot_traversal_probability_heatmap_no_columns(viz, prob_data):
    fig = viz.plot_traversal_probability_heatmap(prob_data, ["nope"])
    assert fig is not None


def test_plot_ratio_heatmap_all_thresholds(viz, ratio_data):
    fig = viz.plot_ratio_heatmap_all_thresholds(
        lambda t: ratio_data, THRESHOLDS, DATASETS
    )
    assert fig is not None


def test_plot_ratio_heatmap_all_thresholds_none(viz):
    fig = viz.plot_ratio_heatmap_all_thresholds(
        lambda t: None, THRESHOLDS, DATASETS
    )
    assert fig is not None


def test_plot_traversal_probability_heatmap_all_thresholds(viz, prob_data):
    fig = viz.plot_traversal_probability_heatmap_all_thresholds(
        lambda t: prob_data, THRESHOLDS, DATASETS
    )
    assert fig is not None


def test_plot_traversal_probability_heatmap_all_thresholds_none(viz):
    fig = viz.plot_traversal_probability_heatmap_all_thresholds(
        lambda t: None, THRESHOLDS, DATASETS
    )
    assert fig is not None


# ---------------------------------------------------------------------------
# Conservation / trend plots
# ---------------------------------------------------------------------------

def test_plot_conservation_across_thresholds(viz, results, aligned):
    fig, df = viz.plot_conservation_across_thresholds(
        results, THRESHOLDS, align_func=lambda t: aligned
    )
    assert fig is not None
    assert isinstance(df, pd.DataFrame) and not df.empty
    assert "edge_count" in df.columns


def test_plot_conservation_across_thresholds_presence_matrix(viz, results, aligned):
    matrix = pd.DataFrame(
        {
            "ds_one_t1": ["True", "False"],
            "ds_one_t5": ["True", "True"],
            "ds_two_t1": [True, False],
            "ds_two_t5": [1, 0],
        }
    )
    fig, df = viz.plot_conservation_across_thresholds(
        results,
        THRESHOLDS,
        align_func=lambda t: aligned,
        path_presence_matrix=matrix,
    )
    assert fig is not None
    assert not df.empty


def test_plot_jaccard_similarity_trend(viz, aligned):
    fig, df = viz.plot_jaccard_similarity_trend(
        lambda t: aligned, THRESHOLDS, DATASETS
    )
    assert fig is not None
    assert not df.empty and "jaccard" in df.columns


def test_plot_topk_overlap_trend(viz, aligned):
    fig, df = viz.plot_topk_overlap_trend(
        lambda t: aligned, THRESHOLDS, DATASETS
    )
    assert fig is not None
    assert not df.empty and "top20_overlap" in df.columns


def test_plot_netsimile_trend(viz, aligned):
    fig, df = viz.plot_netsimile_trend(
        lambda t: aligned, THRESHOLDS, DATASETS
    )
    assert fig is not None
    assert not df.empty and "netsimile_similarity" in df.columns


def test_plot_netsimile_trend_empty(viz):
    fig, df = viz.plot_netsimile_trend(
        lambda t: pd.DataFrame(), THRESHOLDS, DATASETS
    )
    assert fig is not None
    assert df.empty


def test_plot_cosine_similarity_trend(viz, aligned):
    fig, df = viz.plot_cosine_similarity_trend(
        lambda t: aligned, THRESHOLDS, DATASETS
    )
    assert fig is not None
    assert not df.empty


def test_plot_conservation_across_thresholds_plotly(viz, results, aligned):
    payload = viz.plot_conservation_across_thresholds_plotly(
        results, THRESHOLDS, align_func=lambda t: aligned
    )
    assert isinstance(payload, str)
    parsed = json.loads(payload)
    assert "data" in parsed and len(parsed["data"]) > 0


# ---------------------------------------------------------------------------
# save_all_plots orchestration
# ---------------------------------------------------------------------------

def test_save_all_plots(viz, results, aligned, similarities, path_data,
                        ratio_data, prob_data, tmp_path):
    out_dir = str(tmp_path / "plots")
    sims = similarities.copy()
    presence = pd.DataFrame(
        {
            "ds_one_t1": ["True"],
            "ds_one_t5": ["True"],
            "ds_two_t1": ["True"],
            "ds_two_t5": ["False"],
        }
    )
    viz.save_all_plots(
        results,
        aligned,
        sims,
        out_dir,
        THRESHOLDS,
        align_func=lambda t: aligned,
        similarity_func=lambda t: sims,
        path_data_func=lambda t: path_data,
        ratio_data_func=lambda t: ratio_data,
        prob_data_func=lambda t: prob_data,
        nickname_map={"ds_one": "D1", "ds_two": "D2"},
        path_presence_matrix=presence,
        silent=True,
    )
    files = os.listdir(out_dir)
    assert "path_counts.png" in files
    assert any(f.endswith(".png") for f in files)
    vis_data = os.path.join(out_dir, "visualization_data")
    assert os.path.isdir(vis_data)
    assert os.path.exists(os.path.join(vis_data, "path_counts.csv"))
    # subfolders for ratio / probability heatmaps
    assert os.path.isdir(os.path.join(out_dir, "by_ratio"))
    assert os.path.isdir(os.path.join(out_dir, "by_probability"))


def test_generate_vis_summary_pdf(viz, tmp_path):
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    # Create two tiny PNGs
    for name in ("a.png", "b.png"):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        fig.savefig(str(plots_dir / name), dpi=40)
        plt.close(fig)
    viz._generate_vis_summary_pdf(str(plots_dir))
    assert (tmp_path / "vis_summary.pdf").exists()


def test_generate_vis_summary_pdf_no_pngs(viz, tmp_path):
    plots_dir = tmp_path / "empty"
    plots_dir.mkdir()
    viz._generate_vis_summary_pdf(str(plots_dir))  # early return, no crash
    assert not (tmp_path / "vis_summary.pdf").exists()


# ============================================================================
# interactive_heatmap module
# ============================================================================

from comparison.interactive_heatmap import generate_interactive_heatmap


def _metric_df(n=4, with_nan=False):
    vals = np.linspace(0.1, 1.0, n * n).reshape(n, n)
    if with_nan:
        vals[0, 1] = np.nan
    labels = [f"T{i}" for i in range(n)]
    return pd.DataFrame(vals, index=labels, columns=labels)


def test_interactive_heatmap_basic(tmp_path):
    matrices = {"jaccard": _metric_df(with_nan=True), "cosine": _metric_df()}
    out = tmp_path / "heatmap.html"
    generate_interactive_heatmap(
        matrices, str(out), title="Test Heatmap", showfig=False, verbose=True
    )
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "Test Heatmap" in html
    assert "Jaccard" in html  # display name of metric option
    assert "clusteringAvailable" in html
    assert "cdn.plot.ly" in html


def test_interactive_heatmap_empty_dict(tmp_path):
    with pytest.raises(ValueError):
        generate_interactive_heatmap({}, str(tmp_path / "x.html"), showfig=False)


def test_interactive_heatmap_single_cell(tmp_path):
    matrices = {"m": pd.DataFrame([[0.5]], index=["a"], columns=["b"])}
    out = tmp_path / "single.html"
    generate_interactive_heatmap(matrices, str(out), showfig=False, verbose=False)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "metricsData" in html
    assert "let showLabels = true" in html  # small matrix: labels shown


def test_interactive_heatmap_large_branch(tmp_path):
    # >100 rows triggers the is_large branch (labels hidden by default)
    n = 101
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.random((n, 3)),
        index=[f"r{i}" for i in range(n)],
        columns=["a", "b", "c"],
    )
    out = tmp_path / "large.html"
    generate_interactive_heatmap({"m": df}, str(out), showfig=False, verbose=False)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "Show Labels" in html  # large matrices start with labels hidden


# ============================================================================
# Additional coverage: dependency guards, fallbacks, error paths
# ============================================================================

import comparison.visualizations as vis_module
from comparison import metrics as comparison_metrics_module


def _raiser(*args, **kwargs):
    raise RuntimeError("boom")


def _df_types_no_hvp():
    """Tiny result DataFrame without has_valid_path column."""
    return pd.DataFrame({"type_pre": ["A"], "type_post": ["B"], "weight": [1.0]})


# --- Dependency guards ------------------------------------------------------

def test_init_requires_matplotlib(monkeypatch):
    monkeypatch.setattr(vis_module, "HAS_MATPLOTLIB", False)
    with pytest.raises(ImportError):
        ComparisonVisualizer()


def test_vprint_verbose():
    v = ComparisonVisualizer(verbose=True)
    v._vprint("hello", 42)  # exercises tqdm.write branch


def test_heatmap_functions_require_seaborn(
    monkeypatch, viz, results, aligned, similarities, path_data, ratio_data, prob_data
):
    monkeypatch.setattr(vis_module, "HAS_SEABORN", False)
    with pytest.raises(ImportError):
        viz.plot_edge_weight_heatmap(aligned, DATASETS)
    with pytest.raises(ImportError):
        viz.plot_edge_weight_heatmap_all_thresholds(
            results, THRESHOLDS, align_func=lambda t: aligned
        )
    with pytest.raises(ImportError):
        viz.plot_similarity_matrix(similarities)
    with pytest.raises(ImportError):
        viz.plot_dual_similarity_matrices(similarities)
    with pytest.raises(ImportError):
        viz.plot_similarity_per_threshold({1: similarities})
    with pytest.raises(ImportError):
        viz.plot_similarity_matrix_per_threshold(
            results, THRESHOLDS, align_func=lambda t: aligned
        )
    with pytest.raises(ImportError):
        viz.plot_path_heatmap(path_data, DATASETS)
    with pytest.raises(ImportError):
        viz.plot_path_heatmap_all_thresholds(lambda t: path_data, THRESHOLDS, DATASETS)
    with pytest.raises(ImportError):
        viz.plot_ratio_heatmap(ratio_data, DATASETS)
    with pytest.raises(ImportError):
        viz.plot_traversal_probability_heatmap(prob_data, DATASETS)
    with pytest.raises(ImportError):
        viz.plot_ratio_heatmap_all_thresholds(
            lambda t: ratio_data, THRESHOLDS, DATASETS
        )
    with pytest.raises(ImportError):
        viz.plot_traversal_probability_heatmap_all_thresholds(
            lambda t: prob_data, THRESHOLDS, DATASETS
        )


def test_trend_functions_require_matplotlib(monkeypatch, viz, results, aligned, path_data):
    monkeypatch.setattr(vis_module, "HAS_MATPLOTLIB", False)
    fig, df = viz.plot_jaccard_similarity_trend(lambda t: aligned, THRESHOLDS, DATASETS)
    assert fig is None and df.empty
    fig, df = viz.plot_topk_overlap_trend(lambda t: aligned, THRESHOLDS, DATASETS)
    assert fig is None and df.empty
    fig, df = viz.plot_netsimile_trend(lambda t: aligned, THRESHOLDS, DATASETS)
    assert fig is None and df.empty
    fig, df = viz.plot_cosine_similarity_trend(lambda t: aligned, THRESHOLDS, DATASETS)
    assert fig is None and df.empty
    fig, df = viz.plot_conservation_across_thresholds(results, THRESHOLDS)
    assert fig is None and df is None


# --- Threshold comparison subplot fallbacks ----------------------------------

def test_threshold_comparison_subplots_no_seaborn_fallback(monkeypatch, viz):
    monkeypatch.setattr(vis_module, "HAS_SEABORN", False)
    df_w = pd.DataFrame({"weight": [1.0, 2.0, 3.0]})  # no type_pre/type_post
    res = {
        "d1": {1: df_w, 2: pd.DataFrame()},
        "d2": {1: pd.DataFrame({"weight": [2.0, 4.0]}), 2: pd.DataFrame()},
    }
    fig = viz.plot_threshold_comparison_subplots(res, [1, 2])
    assert fig is not None


def test_edge_heatmap_all_thresholds_no_dataset_columns(viz, results):
    # aligned frame without any dataset column -> index-based top_edges fallback
    other = pd.DataFrame({"zzz": [1.0, 2.0]}, index=["e1", "e2"])
    fig = viz.plot_edge_weight_heatmap_all_thresholds(
        results, THRESHOLDS, align_func=lambda t: other
    )
    assert fig is not None  # "No data available" fallback


# --- Similarity subplot branches ---------------------------------------------

def test_similarity_per_threshold_nan_value_and_unused_axes(viz, similarities):
    sims_nan = similarities.copy()
    sims_nan.loc[0, "jaccard_similarity"] = np.nan  # NaN -> 0 branch
    sims = similarities.copy()
    # 4 thresholds on a 3-col grid -> two unused axes hidden
    fig = viz.plot_similarity_per_threshold(
        {1: sims_nan, 2: sims, 3: sims, 4: sims}, metric="jaccard"
    )
    assert fig is not None


def test_similarity_matrix_per_threshold_branches(
    viz, results, aligned, similarities, monkeypatch
):
    # path_data_func raising -> swallowed
    fig = viz.plot_similarity_matrix_per_threshold(
        results, THRESHOLDS,
        align_func=lambda t: aligned,
        similarity_func=lambda t: similarities.copy(),
        path_data_func=_raiser,
        show_progress=False,
    )
    assert fig is not None
    # empty cached similarities -> recompute path
    fig = viz.plot_similarity_matrix_per_threshold(
        results, [1],
        align_func=lambda t: aligned,
        similarity_func=lambda t: pd.DataFrame(),
        show_progress=False,
    )
    assert fig is not None

    # recompute yields empty -> skip threshold
    class _EmptyMetrics:
        def calculate_all_pairwise_similarities(self, *a, **k):
            return pd.DataFrame()

    monkeypatch.setattr(comparison_metrics_module, "ComparisonMetrics", _EmptyMetrics)
    fig = viz.plot_similarity_matrix_per_threshold(
        results, [1], align_func=lambda t: aligned, show_progress=False
    )
    assert fig is not None
    # align_func raising -> per-threshold warning branch
    fig = viz.plot_similarity_matrix_per_threshold(
        results, [1], align_func=_raiser, show_progress=False
    )
    assert fig is not None


# --- Path / ratio / probability all-threshold heatmaps ------------------------

def test_path_heatmap_all_thresholds_dup_index_and_bad_value(viz):
    df = pd.DataFrame(
        {"ds_one": [5.0, 4.0, 9.0], "ds_two": [3.0, 2.0, "bad"]},
        index=["p", "p", "q"],  # duplicate index -> Series.sum branch
    )
    fig = viz.plot_path_heatmap_all_thresholds(lambda t: df, THRESHOLDS, DATASETS)
    assert fig is not None


def test_path_heatmap_all_thresholds_no_dataset_columns(viz):
    df = pd.DataFrame({"zzz": [1.0, 2.0]}, index=["p1", "p2"])
    fig = viz.plot_path_heatmap_all_thresholds(lambda t: df, THRESHOLDS, ["nope"])
    assert fig is not None  # fallback figure


def test_path_heatmap_all_thresholds_no_thresholds(viz, path_data):
    fig = viz.plot_path_heatmap_all_thresholds(lambda t: path_data, [], DATASETS)
    assert fig is not None


def test_path_heatmap_all_thresholds_many_paths(viz):
    n = 40
    df = pd.DataFrame(
        {"ds_one": np.linspace(1, 40, n), "ds_two": np.linspace(2, 30, n)},
        index=[f"p{i}" for i in range(n)],
    )
    # >30 rows -> annotations disabled
    fig = viz.plot_path_heatmap_all_thresholds(
        lambda t: df.copy(), THRESHOLDS, DATASETS, max_paths=32
    )
    assert fig is not None


def test_ratio_heatmap_all_thresholds_raising(viz):
    fig = viz.plot_ratio_heatmap_all_thresholds(_raiser, THRESHOLDS, DATASETS)
    assert fig is not None  # "No ratio data available"


def test_ratio_heatmap_all_thresholds_no_dataset_columns(viz):
    df = pd.DataFrame({"zzz": [0.5, 0.6]}, index=["e1", "e2"])
    fig = viz.plot_ratio_heatmap_all_thresholds(lambda t: df, THRESHOLDS, ["nope"])
    assert fig is not None


def test_ratio_heatmap_all_thresholds_dup_index_and_bad_value(viz):
    df = pd.DataFrame(
        {"ds_one": [0.5, 0.4, 0.9], "ds_two": [0.3, 0.2, "bad"]},
        index=["e", "e", "f"],
    )
    fig = viz.plot_ratio_heatmap_all_thresholds(lambda t: df, THRESHOLDS, DATASETS)
    assert fig is not None


def test_ratio_heatmap_all_thresholds_many_edges(viz):
    n = 40
    df = pd.DataFrame(
        {"ds_one": np.linspace(0.1, 0.9, n), "ds_two": np.linspace(0.2, 0.8, n)},
        index=[f"e{i}" for i in range(n)],
    )
    fig = viz.plot_ratio_heatmap_all_thresholds(
        lambda t: df.copy(), THRESHOLDS, DATASETS, max_edges=32
    )
    assert fig is not None


def test_traversal_prob_all_thresholds_raising(viz):
    fig = viz.plot_traversal_probability_heatmap_all_thresholds(
        _raiser, THRESHOLDS, DATASETS
    )
    assert fig is not None  # "No probability data available"


def test_traversal_prob_all_thresholds_no_dataset_columns(viz):
    df = pd.DataFrame({"zzz": [0.5, 0.6]}, index=["p1", "p2"])
    fig = viz.plot_traversal_probability_heatmap_all_thresholds(
        lambda t: df, THRESHOLDS, ["nope"]
    )
    assert fig is not None


def test_traversal_prob_all_thresholds_dup_index_and_bad_value(viz):
    df = pd.DataFrame(
        {"ds_one": [0.5, 0.4, 0.9], "ds_two": [0.3, 0.2, "bad"]},
        index=["p", "p", "q"],
    )
    fig = viz.plot_traversal_probability_heatmap_all_thresholds(
        lambda t: df, THRESHOLDS, DATASETS
    )
    assert fig is not None


def test_traversal_prob_all_thresholds_many_paths(viz):
    n = 40
    df = pd.DataFrame(
        {"ds_one": np.linspace(0.01, 0.9, n), "ds_two": np.linspace(0.02, 0.8, n)},
        index=[f"p{i}" for i in range(n)],
    )
    fig = viz.plot_traversal_probability_heatmap_all_thresholds(
        lambda t: df.copy(), THRESHOLDS, DATASETS, max_paths=32
    )
    assert fig is not None


# --- Conservation across thresholds -------------------------------------------

def test_conservation_presence_matrix_sanitized_names(viz):
    # dataset name with ':' -> sanitized column lookup; second dataset column missing
    res = {
        "a:b": {1: _edge_df(3), 5: _edge_df(2)},
        "c": {1: _edge_df(3), 5: _edge_df(2)},
    }
    matrix = pd.DataFrame(
        {"a_b_t1": ["True", "False"], "a_b_t5": ["True", "True"]}
    )
    align = pd.DataFrame({"a:b": [1.0, 0.0], "c": [1.0, 1.0]}, index=["x", "y"])
    fig, df = viz.plot_conservation_across_thresholds(
        res, THRESHOLDS, align_func=lambda t: align, path_presence_matrix=matrix
    )
    assert fig is not None and not df.empty


def test_conservation_presence_matrix_missing_col_no_hvp(viz):
    res = {"d1": {1: _df_types_no_hvp(), 5: _df_types_no_hvp()}}
    matrix = pd.DataFrame({"other_t1": ["True"]})  # no matching column
    fig, df = viz.plot_conservation_across_thresholds(
        res, THRESHOLDS,
        align_func=lambda t: pd.DataFrame(),
        path_presence_matrix=matrix,
    )
    assert fig is not None


def test_conservation_fallback_common_paths(viz):
    df_types = _df_types_no_hvp()
    df_hvp_false = pd.DataFrame(
        {"type_pre": ["A"], "type_post": ["B"], "weight": [1.0],
         "has_valid_path": [False]}
    )
    df_notypes = pd.DataFrame({"weight": [1.0, 2.0]})
    res = {
        "d1": {1: df_types, 5: df_types},
        "d2": {1: df_hvp_false, 5: pd.DataFrame()},
        "d3": {1: df_notypes, 5: df_types},
    }
    fig, df = viz.plot_conservation_across_thresholds(
        res, THRESHOLDS, align_func=_raiser
    )
    assert fig is not None and not df.empty


def test_conservation_empty_results(viz):
    fig, df = viz.plot_conservation_across_thresholds({}, THRESHOLDS)
    assert fig is not None and not df.empty


# --- Trend plot fallbacks -------------------------------------------------------

def test_jaccard_and_edge_rank_trend_empty_aligned(viz):
    fig, df = viz.plot_jaccard_similarity_trend(
        lambda t: pd.DataFrame(), THRESHOLDS, DATASETS
    )
    assert fig is not None and df.empty
    fig, df = viz.plot_topk_overlap_trend(
        lambda t: pd.DataFrame(), THRESHOLDS, DATASETS
    )
    assert fig is not None and df.empty


def test_netsimile_trend_raising(viz):
    fig, df = viz.plot_netsimile_trend(_raiser, THRESHOLDS, DATASETS)
    assert fig is not None and df.empty


def test_cosine_trend_align_raise_and_none(viz):
    fig, df = viz.plot_cosine_similarity_trend(_raiser, THRESHOLDS, DATASETS)
    assert fig is not None and df.empty
    fig, df = viz.plot_cosine_similarity_trend(lambda t: None, THRESHOLDS, DATASETS)
    assert fig is not None and df.empty


# --- Plotly conservation branches -----------------------------------------------

def test_plotly_conservation_presence_branches(viz):
    res = {
        "a:b": {1: _edge_df(2), 5: _edge_df(2)},
        "c": {1: _edge_df(2), 5: _edge_df(2)},
    }
    matrix = pd.DataFrame(
        {
            "a_b_t1": [True, False],     # bool dtype
            "a_b_t5": [1, 0],            # numeric dtype
            "c_t5": [True, True],        # bool; c_t1 missing -> fallback
        }
    )
    payload = viz.plot_conservation_across_thresholds_plotly(
        res, THRESHOLDS, align_func=_raiser, path_presence_matrix=matrix
    )
    parsed = json.loads(payload)
    assert "data" in parsed


def test_plotly_conservation_missing_col_no_hvp(viz):
    # presence matrix exists but has no matching column and df lacks has_valid_path
    res = {"d1": {1: _df_types_no_hvp(), 5: _df_types_no_hvp()}}
    matrix = pd.DataFrame({"unrelated": [1]})
    payload = viz.plot_conservation_across_thresholds_plotly(
        res, THRESHOLDS, path_presence_matrix=matrix
    )
    parsed = json.loads(payload)
    assert "data" in parsed


def test_plotly_conservation_common_path_fallbacks(viz):
    df_hvp_false = pd.DataFrame(
        {"type_pre": ["A"], "type_post": ["B"], "weight": [1.0],
         "has_valid_path": [False]}
    )
    df_notypes = pd.DataFrame({"weight": [1.0]})
    res = {
        "d1": {1: _df_types_no_hvp(), 5: _df_types_no_hvp()},
        "d2": {1: df_notypes, 5: _df_types_no_hvp()},
        "d3": {1: df_hvp_false, 5: pd.DataFrame()},
    }
    payload = viz.plot_conservation_across_thresholds_plotly(res, THRESHOLDS)
    parsed = json.loads(payload)
    assert "data" in parsed


def test_plotly_conservation_empty_results(viz):
    payload = viz.plot_conservation_across_thresholds_plotly({}, THRESHOLDS)
    parsed = json.loads(payload)
    assert "data" in parsed


# --- save_all_plots error / misc branches ----------------------------------------

def test_save_all_plots_error_paths(viz, results, aligned, tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "plot_conservation_across_thresholds", _raiser)
    monkeypatch.setattr(viz, "plot_netsimile_trend", _raiser)
    monkeypatch.setattr(viz, "plot_cosine_similarity_trend", _raiser)
    viz.save_all_plots(
        results, aligned, pd.DataFrame(), str(tmp_path / "out"), THRESHOLDS,
        align_func=_raiser,
        similarity_func=None,
        path_data_func=_raiser,
        ratio_data_func=_raiser,
        prob_data_func=_raiser,
        nickname_map=None,
        silent=True,
    )
    assert os.path.isdir(str(tmp_path / "out"))


def test_save_all_plots_plotly_failure(
    viz, results, aligned, similarities, path_data, ratio_data, prob_data,
    tmp_path, monkeypatch
):
    monkeypatch.setattr(viz, "plot_conservation_across_thresholds_plotly", _raiser)
    sims = similarities.copy()
    viz.save_all_plots(
        results, aligned, sims, str(tmp_path / "out"), THRESHOLDS,
        align_func=lambda t: aligned,
        similarity_func=lambda t: sims,
        path_data_func=lambda t: path_data,
        ratio_data_func=lambda t: ratio_data,
        prob_data_func=lambda t: prob_data,
        silent=True,
    )
    assert os.path.isdir(str(tmp_path / "out"))


def test_save_all_plots_similarity_plot_failure(
    viz, results, aligned, similarities, tmp_path, monkeypatch
):
    monkeypatch.setattr(viz, "plot_similarity_matrix_per_threshold", _raiser)
    sims = similarities.copy()
    viz.save_all_plots(
        results, aligned, sims, str(tmp_path / "out"), THRESHOLDS,
        align_func=lambda t: aligned,
        similarity_func=lambda t: sims,
        silent=True,
    )
    assert os.path.isdir(str(tmp_path / "out"))


class _FakeMetricsSVD:
    """Fake ComparisonMetrics returning sims that include svd_similarity."""

    def calculate_all_pairwise_similarities(self, aligned, datasets, threshold=1, **kw):
        return pd.DataFrame([{
            "dataset_1": "ds_one", "dataset_2": "ds_two",
            "jaccard_similarity": 0.5, "svd_similarity": 0.6,
            "cosine_similarity": 0.7, "edge_rank_correlation": 0.4,
            "spearman_rank_correlation": 0.3,
        }])

    def calculate_edge_list_rank_correlation(self, a, b):
        return 0.5

    def calculate_path_list_rank_correlation(self, a, b):
        return 0.5

    def calculate_cosine_similarity(self, a, b):
        return 0.5


def test_save_all_plots_misc_branches(viz, results, tmp_path, monkeypatch):
    monkeypatch.setattr(comparison_metrics_module, "ComparisonMetrics", _FakeMetricsSVD)
    aligned_two = pd.DataFrame(
        {"ds_one": [3.0, 1.0], "ds_two": [2.0, 0.0]}, index=["e1", "e2"]
    )
    aligned_one = pd.DataFrame({"ds_one": [3.0, 1.0]}, index=["e1", "e2"])
    aligned_odd = pd.DataFrame({"zzz": [1.0]}, index=["e1"])

    def align(t):
        if t == 1:
            return aligned_one
        if t == 2:
            return aligned_odd
        return pd.DataFrame()

    def path_data(t):
        if t == 1:
            raise RuntimeError("no path data")
        return pd.DataFrame()

    viz.save_all_plots(
        results, aligned_two, pd.DataFrame(), str(tmp_path / "out"), [1, 2, 5],
        align_func=align,
        similarity_func=None,
        path_data_func=path_data,
        silent=True,
    )
    vis_data = tmp_path / "out" / "visualization_data"
    assert (vis_data / "similarity_per_threshold.csv").exists()


# --- PDF summary generation --------------------------------------------------------

def test_generate_vis_summary_pdf_categorized(viz, tmp_path):
    from PIL import Image

    plots = tmp_path / "plots"
    for sub in ("threshold_2", "minsyn_5", "by_ratio", "visualization_data"):
        (plots / sub).mkdir(parents=True)

    def save(img, rel):
        img.save(str(plots / rel), "PNG")

    save(Image.new("RGB", (3000, 2000), "red"), "big.png")   # scale < 1 resize
    save(Image.new("L", (40, 40)), "gray.png")               # non-RGB convert
    save(Image.new("P", (40, 40)), "pal.png")                # palette -> RGBA paste
    save(Image.new("LA", (40, 40)), "la.png")                # LA -> plain paste
    save(Image.new("RGB", (30, 30), "blue"), "threshold_2/t.png")
    save(Image.new("RGB", (30, 30), "green"), "minsyn_5/m.png")
    save(Image.new("RGB", (30, 30), "red"), "by_ratio/r.png")
    save(Image.new("RGB", (30, 30), "red"), "visualization_data/v.png")  # skipped
    (plots / "notes.txt").write_text("not an image")          # skipped
    (plots / "bad.png").write_bytes(b"definitely-not-a-png")  # processing error

    viz._generate_vis_summary_pdf(str(plots))
    assert (tmp_path / "vis_summary.pdf").exists()


def test_generate_vis_summary_pdf_all_unreadable(viz, tmp_path):
    plots = tmp_path / "plots"
    plots.mkdir()
    (plots / "bad1.png").write_bytes(b"junk")
    (plots / "bad2.png").write_bytes(b"junk")
    viz._generate_vis_summary_pdf(str(plots))
    assert not (tmp_path / "vis_summary.pdf").exists()


def test_interactive_heatmap_cells_are_square(tmp_path):
    """The standalone fallback (used when VisPath is unavailable) locks cells
    1:1 with the same scaleanchor contract as the report-kit figures."""
    from comparison.interactive_heatmap import generate_interactive_heatmap

    out = tmp_path / "square.html"
    generate_interactive_heatmap(
        {"jaccard": _metric_df()}, str(out), title="Square",
        showfig=False, verbose=False)
    html = out.read_text(encoding="utf-8")
    assert "scaleanchor: 'x'" in html
    assert "constrain: 'domain'" in html


def test_interactive_heatmap_honors_metric_scale_and_domain(tmp_path):
    """The fallback must carry the metric's diverging colormap and fixed
    [-1, 1] domain — sequential Viridis on an auto-ranged axis would strip
    the negative half of its meaning."""
    from comparison.interactive_heatmap import generate_interactive_heatmap
    from comparison.report_kit import REPORT_DIVERGING_COLORSCALE

    out = tmp_path / "signed.html"
    generate_interactive_heatmap(
        {"morph_similarity": _metric_df(with_nan=True)}, str(out),
        title="Signed", showfig=False, verbose=False,
        color_scale=REPORT_DIVERGING_COLORSCALE, zmin=-1.0, zmax=1.0)
    html = out.read_text(encoding="utf-8")
    assert '"__metric__"' in html and "Metric scale</option>" in html
    assert "const zminFixed = -1.0" in html
    assert "const zmaxFixed = 1.0" in html
    assert '"#053061"' in html  # the diverging ramp's dark-blue stop
    # Callers without a style keep the sequential default, unfixed domain.
    plain = tmp_path / "plain.html"
    generate_interactive_heatmap(
        {"jaccard": _metric_df()}, str(plain), title="Plain",
        showfig=False, verbose=False)
    plain_html = plain.read_text(encoding="utf-8")
    assert "const zminFixed = null" in plain_html
    assert "const zmaxFixed = null" in plain_html
    # The update handler always knows the sentinel; what must be absent is
    # the dropdown option (plain pages start on Viridis).
    assert 'value="__metric__"' not in plain_html
    assert "const metricScale = null" in plain_html
