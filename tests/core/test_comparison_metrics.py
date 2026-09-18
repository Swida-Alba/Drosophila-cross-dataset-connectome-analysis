"""Coverage tests for comparison.metrics.ComparisonMetrics.

Hermetic: purely in-memory synthetic DataFrames/Series. No I/O, no network.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.metrics import ComparisonMetrics


@pytest.fixture
def metrics():
    return ComparisonMetrics()


def _edge_series(pairs):
    return pd.Series({f"{a} -> {b}": w for (a, b), w in pairs.items()})


def _aligned_df():
    """Aligned frame: edges present with varying overlap."""
    return pd.DataFrame(
        {
            "d1": [10.0, 5.0, 0.0, 3.0],
            "d2": [8.0, 0.0, 4.0, 3.0],
        },
        index=["A -> B", "A -> C", "B -> C", "C -> D"],
    )


# ---------------------------------------------------------------------------
# Path count / edge weight comparisons
# ---------------------------------------------------------------------------

def test_compare_path_counts(metrics):
    results = {
        "d1": {1: pd.DataFrame({"hop_count": [1, 2, 2]}), 3: pd.DataFrame()},
        "d2": {1: pd.DataFrame({"other": [1]})},
    }
    out = metrics.compare_path_counts(results, group_by="hop_count")
    assert not out.empty
    row_all = out[(out["dataset"] == "d1") & (out["threshold"] == 1) & (out["group"] == "all")]
    assert row_all.iloc[0]["path_count"] == 3
    # empty df produces a zero-count row
    assert ((out["dataset"] == "d1") & (out["threshold"] == 3) & (out["path_count"] == 0)).any()


def test_compare_edge_weights(metrics):
    aligned = _aligned_df()
    stats = metrics.compare_edge_weights(aligned, ["d1", "d2"])
    assert "mean_weight" in stats.columns and "cv" in stats.columns

    # empty input
    assert metrics.compare_edge_weights(pd.DataFrame(), ["d1"]).empty
    # single available dataset -> unchanged copy
    single = metrics.compare_edge_weights(aligned, ["d1"])
    assert "mean_weight" not in single.columns


# ---------------------------------------------------------------------------
# Basic similarity metrics
# ---------------------------------------------------------------------------

def test_jaccard(metrics):
    assert metrics.calculate_jaccard_similarity(set(), set()) == 1.0
    assert metrics.calculate_jaccard_similarity({("A", "B")}, {("A", "B")}) == 1.0
    assert metrics.calculate_jaccard_similarity({("A", "B")}, {("C", "D")}) == 0.0
    assert metrics.calculate_jaccard_similarity({("A", "B"), ("C", "D")}, {("A", "B")}) == pytest.approx(0.5)


def test_ruzicka(metrics):
    empty = pd.Series(dtype=float)
    assert metrics.calculate_ruzicka_similarity(empty, empty) == 1.0
    w1 = _edge_series({("A", "B"): 4.0, ("B", "C"): 6.0})
    w2 = _edge_series({("A", "B"): 4.0, ("B", "C"): 6.0})
    assert metrics.calculate_ruzicka_similarity(w1, w2) == pytest.approx(1.0)
    w3 = _edge_series({("X", "Y"): 1.0})
    assert metrics.calculate_ruzicka_similarity(w1, w3) == pytest.approx(0.0)
    # unnormalized branch
    val = metrics.calculate_ruzicka_similarity(w1, w2, use_normalized=False)
    assert val == pytest.approx(1.0)


def test_weighted_correlation(metrics):
    w1 = _edge_series({("A", "B"): 1.0, ("B", "C"): 2.0, ("C", "D"): 3.0})
    w2 = _edge_series({("A", "B"): 2.0, ("B", "C"): 4.0, ("C", "D"): 6.0})
    assert metrics.calculate_weighted_correlation(w1, w2) == pytest.approx(1.0)
    # too few shared edges
    assert np.isnan(metrics.calculate_weighted_correlation(w1.iloc[:1], w2.iloc[:1]))
    # zero variance
    const = _edge_series({("A", "B"): 1.0, ("B", "C"): 1.0, ("C", "D"): 1.0})
    assert np.isnan(metrics.calculate_weighted_correlation(const, w2))


def test_pairwise_similarities(metrics):
    aligned = _aligned_df()
    sims = metrics.calculate_all_pairwise_similarities(aligned, ["d1", "d2"], threshold=1)
    assert len(sims) == 1
    row = sims.iloc[0]
    assert row["common_edges"] == 2  # A->B and C->D
    assert "ruzicka_similarity" in sims.columns
    assert "cosine_similarity" in sims.columns
    assert "rv_coefficient" in sims.columns
    assert "edge_rank_correlation" in sims.columns
    assert "spearman_rank_correlation" in sims.columns
    assert np.isnan(row["path_rank_correlation"])

    # with path_data -> path rank correlation computed
    path_data = pd.DataFrame(
        {"d1": [5.0, 2.0, 1.0], "d2": [4.0, 2.5, 1.5]},
        index=["A->B", "B->C", "C->D"],
    )
    sims2 = metrics.calculate_all_pairwise_similarities(
        aligned, ["d1", "d2"], threshold=1, path_data=path_data
    )
    assert not np.isnan(sims2.iloc[0]["path_rank_correlation"])

    # without advanced metrics
    sims3 = metrics.calculate_all_pairwise_similarities(
        aligned, ["d1", "d2"], include_advanced_metrics=False
    )
    assert "cosine_similarity" not in sims3.columns

    # fewer than 2 datasets -> empty
    assert metrics.calculate_all_pairwise_similarities(aligned, ["d1"]).empty
    assert metrics.calculate_all_pairwise_similarities(aligned, ["zzz"]).empty


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def test_align_results_at_threshold(metrics):
    results = {
        "d1": {1: pd.DataFrame({"type_pre": ["A", "A"], "type_post": ["B", "B"], "weight": [2.0, 3.0]})},
        "d2": {1: pd.DataFrame({"type_pre": ["A"], "type_post": ["B"], "weight": [4.0]})},
    }
    aligned = metrics._align_results_at_threshold(results, ["d1", "d2"], 1)
    assert aligned.loc["A -> B", "d1"] == pytest.approx(5.0)
    assert aligned.loc["A -> B", "d2"] == pytest.approx(4.0)

    # std_label columns take priority
    results_std = {
        "d1": {1: pd.DataFrame({
            "std_label_pre": ["S"], "std_label_post": ["T"],
            "type_pre": ["x"], "type_post": ["y"], "weight": [1.0],
        })},
    }
    aligned_std = metrics._align_results_at_threshold(results_std, ["d1"], 1)
    assert "S -> T" in aligned_std.index

    # bodyId fallback + no weight column -> count aggregation
    results_bodyid = {
        "d1": {1: pd.DataFrame({"bodyId_pre": [1, 1], "bodyId_post": [2, 3]})},
    }
    aligned_b = metrics._align_results_at_threshold(results_bodyid, ["d1"], 1)
    assert len(aligned_b) == 2

    # missing threshold/dataset -> empty
    assert metrics._align_results_at_threshold(results, ["d1", "d2"], 99).empty
    assert metrics._align_results_at_threshold({}, ["d1"], 1).empty
    # empty df skipped
    results_empty = {"d1": {1: pd.DataFrame()}}
    assert metrics._align_results_at_threshold(results_empty, ["d1"], 1).empty


def test_align_with_label_mapper(metrics):
    class StubMapper:
        def apply_to_dataframe(self, df, dataset):
            df["std_label_pre"] = df["type_pre"].map({"A": "grpA"})
            df["std_label_post"] = df["type_post"].map({"B": "grpB"})
            return df

    results = {
        "d1": {1: pd.DataFrame({"type_pre": ["A"], "type_post": ["B"], "weight": [2.0]})},
    }
    aligned = metrics._align_results_at_threshold(results, ["d1"], 1, label_mapper=StubMapper())
    assert "grpA -> grpB" in aligned.index


def test_align_with_type_mapper(metrics):
    class StubTypeMapper:
        """Fake mapper on the shared resolver contract: metrics canonicalize
        edge endpoints through ``canonical_merge_key`` (which consumes
        ``get_mapping_decision``), not the raw ``get_canonical_type``."""

        _loaded = True

        def _get_type_mapping_key(self, dataset):
            return dataset

        def get_mapping_decision(self, source_type, source_dataset,
                                 target_dataset, include_bridges=False):
            mapped = {"MTe46": "MeVPaMe1"}.get(source_type, source_type)
            return {'status': 'mapped', 'source_type': source_type,
                    'target_type': mapped, 'target_types': [mapped],
                    'relationship': '1-to-1', 'conflicts': []}

        def get_canonical_type(self, t, dataset):
            return {"MTe46": "MeVPaMe1"}.get(t, t)

        def get_display_name(self, canonical, datasets):
            return f"{canonical}(disp)"

    results = {
        "d1": {1: pd.DataFrame({"type_pre": ["MTe46"], "type_post": ["B"], "weight": [2.0]})},
        "d2": {1: pd.DataFrame({"type_pre": ["MeVPaMe1"], "type_post": ["B"], "weight": [3.0]})},
    }
    aligned = metrics._align_results_at_threshold(
        results, ["d1", "d2"], 1, type_mapper=StubTypeMapper()
    )
    # both datasets align under canonical display name
    assert len(aligned) == 1
    assert aligned.iloc[0]["d1"] == pytest.approx(2.0)
    assert aligned.iloc[0]["d2"] == pytest.approx(3.0)


def test_similarity_across_thresholds(metrics):
    results = {
        "d1": {
            1: pd.DataFrame({"type_pre": ["A", "B"], "type_post": ["B", "C"], "weight": [3.0, 2.0]}),
            3: pd.DataFrame({"type_pre": ["A"], "type_post": ["B"], "weight": [3.0]}),
        },
        "d2": {
            1: pd.DataFrame({"type_pre": ["A", "B"], "type_post": ["B", "C"], "weight": [2.0, 2.0]}),
            3: pd.DataFrame({"type_pre": ["A"], "type_post": ["B"], "weight": [3.0]}),
        },
    }
    sims = metrics.calculate_similarity_across_thresholds(
        results, ["d1", "d2"], [1, 3], show_progress=False
    )
    assert not sims.empty
    assert set(sims["threshold"]) == {1, 3}

    # with path_data_func
    sims2 = metrics.calculate_similarity_across_thresholds(
        results, ["d1", "d2"], [1], show_progress=True,
        path_data_func=lambda t: None,
    )
    assert not sims2.empty

    # path_data_func raising is tolerated
    def bad_func(t):
        raise RuntimeError("nope")

    sims3 = metrics.calculate_similarity_across_thresholds(
        results, ["d1", "d2"], [1], show_progress=False, path_data_func=bad_func
    )
    assert not sims3.empty

    # max_edges_for_metrics skips large thresholds entirely
    sims_skip = metrics.calculate_similarity_across_thresholds(
        results, ["d1", "d2"], [1], show_progress=False, max_edges_for_metrics=1
    )
    assert sims_skip.empty


# ---------------------------------------------------------------------------
# Connection classification
# ---------------------------------------------------------------------------

def test_find_common_connections(metrics):
    aligned = _aligned_df()
    common = metrics.find_common_connections(aligned, ["d1", "d2"], threshold=1)
    assert list(common.index) == ["A -> B", "C -> D"]
    assert metrics.find_common_connections(aligned, ["zzz"]).empty


def test_find_unique_connections(metrics):
    aligned = _aligned_df()
    unique = metrics.find_unique_connections(aligned, ["d1", "d2"], threshold=1)
    assert list(unique["d1"].index) == ["A -> C"]
    assert list(unique["d2"].index) == ["B -> C"]
    assert metrics.find_unique_connections(aligned, ["zzz"]) == {}


def test_find_differential_connections(metrics):
    aligned = _aligned_df()
    diff = metrics.find_differential_connections(aligned, ["d1", "d2"], fold_threshold=2.0)
    assert not diff.empty
    assert "max_fold_change" in diff.columns
    # single dataset -> empty
    assert metrics.find_differential_connections(aligned, ["d1"]).empty


def test_find_conserved_strong_connections(metrics):
    aligned = _aligned_df()
    conserved = metrics.find_conserved_strong_connections(aligned, ["d1", "d2"], top_n=2)
    assert "A -> B" in conserved.index
    assert metrics.find_conserved_strong_connections(aligned, ["zzz"]).empty
    # no conserved edges possible
    separated = pd.DataFrame({"d1": [5.0, 0.0], "d2": [0.0, 5.0]}, index=["A -> B", "C -> D"])
    res = metrics.find_conserved_strong_connections(separated, ["d1", "d2"], top_n=1)
    assert res.empty


# ---------------------------------------------------------------------------
# Summary statistics / comparison summary
# ---------------------------------------------------------------------------

def test_calculate_summary_statistics(metrics):
    aligned = _aligned_df()
    stats = metrics.calculate_summary_statistics(aligned, ["d1", "d2"], threshold=1)
    assert len(stats) == 2
    d1 = stats[stats["dataset"] == "d1"].iloc[0]
    assert d1["total_edges"] == 3


def test_generate_comparison_summary(metrics):
    results = {
        "d1": {1: pd.DataFrame({"type_pre": ["A", "B"], "type_post": ["B", "C"], "weight": [4.0, 2.0]}),
               3: pd.DataFrame({"type_pre": ["A", "B"], "type_post": ["B", "C"], "weight": [4.0, 2.0]})},
        "d2": {1: pd.DataFrame({"type_pre": ["A", "B"], "type_post": ["B", "C"], "weight": [3.0, 2.0]}),
               3: pd.DataFrame({"type_pre": ["A", "B"], "type_post": ["B", "C"], "weight": [4.0, 2.0]})},
    }
    summary = metrics.generate_comparison_summary(
        results, ["d1", "d2"], [1, 3], show_progress=False
    )
    assert summary["datasets"] == ["d1", "d2"]
    assert summary["common_connections_count"] >= 1
    assert any("Jaccard" in f for f in summary["key_findings"])
    assert 1 in summary["key_findings_per_threshold"]
    per_t = summary["key_findings_per_threshold"][1]
    assert per_t["edge_counts"]["d1"] == 2
    assert per_t["common_edges"] == 2
    assert per_t["conservation_rate"] == pytest.approx(1.0)

    # max_edges_for_metrics skips similarity computation
    summary2 = metrics.generate_comparison_summary(
        results, ["d1", "d2"], [1, 3], show_progress=False, max_edges_for_metrics=1
    )
    assert summary2["pairwise_similarities"].empty

    # empty results
    summary3 = metrics.generate_comparison_summary({}, ["d1", "d2"], [1], show_progress=False)
    assert summary3["key_findings"] == ["No data available for comparison"]


# ---------------------------------------------------------------------------
# Density / coverage / degrees / top edges
# ---------------------------------------------------------------------------

def test_calculate_edge_density(metrics):
    aligned = _aligned_df()
    dens = metrics.calculate_edge_density(aligned, ["d1", "d2"], source_count=3, target_count=3, threshold=1)
    assert len(dens) == 2
    d1 = dens[dens["dataset"] == "d1"].iloc[0]
    assert d1["actual_edges"] == 3
    # zero possible edges -> empty
    assert metrics.calculate_edge_density(aligned, ["d1"], 0, 0).empty


def test_calculate_type_coverage(metrics):
    results = {
        "d1": {1: pd.DataFrame({"type_pre": ["A"], "type_post": ["B"], "weight": [1.0]})},
        "d2": {1: pd.DataFrame()},
    }
    cov = metrics.calculate_type_coverage(results, ["d1", "d2", "missing"], 1, ["A", "Z"], ["B"])
    assert len(cov) == 3
    d1 = cov[cov["dataset"] == "d1"].iloc[0]
    assert d1["source_coverage_pct"] == pytest.approx(50.0)
    assert d1["target_coverage_pct"] == pytest.approx(100.0)
    d2 = cov[cov["dataset"] == "d2"].iloc[0]
    assert d2["source_types_connected"] == 0


def test_degree_distribution_and_statistics(metrics):
    results = {
        "d1": {1: pd.DataFrame({
            "type_pre": ["A", "A", "B"],
            "type_post": ["B", "C", "C"],
            "weight": [1.0, 2.0, 3.0],
        })},
        "d2": {1: pd.DataFrame()},
    }
    deg = metrics.calculate_degree_distribution(results, ["d1", "d2"], 1)
    assert not deg["out_degree"].empty
    assert not deg["in_degree"].empty
    stats = metrics.calculate_degree_statistics(deg)
    assert len(stats) == 2  # out + in for d1
    # empty degree data
    empty_deg = metrics.calculate_degree_distribution({}, ["d1"], 1)
    assert empty_deg["out_degree"].empty
    assert metrics.calculate_degree_statistics(empty_deg).empty
    # weight-less df branch
    results_now = {"d1": {1: pd.DataFrame({"bodyId_pre": [1, 1], "bodyId_post": [2, 3]})}}
    deg_now = metrics.calculate_degree_distribution(results_now, ["d1"], 1)
    assert not deg_now["out_degree"].empty


def test_top_edges(metrics):
    aligned = _aligned_df()
    top = metrics.get_top_edges_per_dataset(aligned, ["d1", "d2"], top_n=2)
    assert not top.empty
    assert "rank_in_dataset" in top.columns
    assert "present_in_d2" in top.columns

    # top_n <= 0 -> all positive edges (N4 layout: per-dataset weight
    # columns are dropped; use the dataset + weight columns)
    top_all = metrics.get_top_edges_per_dataset(aligned, ["d1"], top_n=0)
    d1_rows = top_all[top_all["dataset"] == "d1"]
    assert (d1_rows["weight"] > 0).all()

    assert metrics.get_top_edges_per_dataset(aligned, ["zzz"]).empty

    overlap = metrics.compare_top_edges_overlap(aligned, ["d1", "d2"], top_n=2)
    assert len(overlap) == 1
    assert overlap.iloc[0]["overlap_count"] >= 1

    # 3 datasets -> includes ALL row
    aligned3 = aligned.copy()
    aligned3["d3"] = [9.0, 4.0, 1.0, 3.0]
    overlap3 = metrics.compare_top_edges_overlap(aligned3, ["d1", "d2", "d3"], top_n=2)
    assert (overlap3["dataset_1"] == "ALL").any()

    # top_n <= 0 branch for overlap
    overlap_all = metrics.compare_top_edges_overlap(aligned, ["d1", "d2"], top_n=-1)
    assert not overlap_all.empty

    assert metrics.compare_top_edges_overlap(aligned, ["d1"]).empty


# ---------------------------------------------------------------------------
# Advanced metrics
# ---------------------------------------------------------------------------

def test_frobenius(metrics):
    w1 = _edge_series({("A", "B"): 1.0, ("B", "C"): 2.0})
    assert metrics.calculate_frobenius_similarity(w1, w1) == pytest.approx(1.0)
    empty = pd.Series(dtype=float)
    assert metrics.calculate_frobenius_similarity(empty, empty) == 0.0
    w2 = _edge_series({("X", "Y"): 5.0})
    val = metrics.calculate_frobenius_similarity(w1, w2)
    assert 0.0 <= val < 1.0


def test_spearman(metrics):
    w1 = _edge_series({("A", "B"): 1.0, ("B", "C"): 2.0, ("C", "D"): 3.0, ("D", "E"): 4.0})
    w2 = _edge_series({("A", "B"): 2.0, ("B", "C"): 4.0, ("C", "D"): 6.0, ("D", "E"): 8.0})
    assert metrics.calculate_spearman_rank_correlation(w1, w2) == pytest.approx(1.0)
    # fewer than 3 shared -> NaN
    assert np.isnan(metrics.calculate_spearman_rank_correlation(w1.iloc[:2], w2.iloc[:2]))
    # union mode
    val = metrics.calculate_spearman_rank_correlation(w1, w2, use_shared_edges=False, use_normalized=False)
    assert val == pytest.approx(1.0)


def test_rv_coefficient(metrics):
    w1 = _edge_series({("A", "B"): 1.0, ("B", "C"): 2.0})
    w2 = _edge_series({("A", "B"): 3.0, ("B", "C"): 6.0})
    assert metrics.calculate_rv_coefficient(w1, w2) == pytest.approx(1.0, abs=1e-6)
    empty = pd.Series(dtype=float)
    assert metrics.calculate_rv_coefficient(empty, empty) == 0.0
    val = metrics.calculate_rv_coefficient(w1, w2, use_normalized=False)
    assert 0.0 <= val <= 1.0


def test_mantel_test(metrics):
    # needs >= 3 nodes to avoid the n<3 fallback branch
    w1 = _edge_series({("A", "B"): 1.0, ("B", "C"): 2.0, ("A", "C"): 3.0, ("C", "D"): 1.5})
    w2 = _edge_series({("A", "B"): 2.0, ("B", "C"): 4.0, ("A", "C"): 6.0, ("C", "D"): 3.0})
    corr_sim, p = metrics.calculate_mantel_test(w1, w2, n_permutations=50)
    assert 0.0 <= corr_sim <= 1.0
    assert 0.0 < p <= 1.0
    corr_s, p_s = metrics.calculate_mantel_test(w1, w2, method="spearman", n_permutations=20)
    assert 0.0 <= corr_s <= 1.0
    empty = pd.Series(dtype=float)
    assert metrics.calculate_mantel_test(empty, empty) == (0.0, 1.0)
    # identical sparse matrices -> perfect observed correlation -> sim 1.0
    sim_c, p_c = metrics.calculate_mantel_test(w1, w1, n_permutations=10)
    assert sim_c == pytest.approx(1.0)
    assert 0.0 < p_c <= 1.0


def test_parse_edge_and_adjacency(metrics):
    assert metrics._parse_edge(("A", "B")) == ("A", "B")
    assert metrics._parse_edge(("A", "B", "C")) == ("A", "B")
    assert metrics._parse_edge("A -> B") == ("A", "B")
    assert metrics._parse_edge("A -> B -> C") is None
    assert metrics._parse_edge(123) is None
    assert metrics._parse_edge("no-arrow") is None

    w1 = _edge_series({("A", "B"): 1.0})
    w2 = pd.Series({("A", "B"): 2.0, ("B", "C"): 1.0})
    adj_a, adj_b = metrics._build_aligned_adjacency_matrices(w1, w2)
    assert adj_a.shape == adj_b.shape == (3, 3)
    adj_e1, adj_e2 = metrics._build_aligned_adjacency_matrices(
        pd.Series(dtype=float), pd.Series(dtype=float)
    )
    assert adj_e1.size == 0


def test_cosine_similarity(metrics):
    w1 = _edge_series({("A", "B"): 1.0, ("B", "C"): 2.0})
    w2 = _edge_series({("A", "B"): 2.0, ("B", "C"): 4.0})
    assert metrics.calculate_cosine_similarity(w1, w2) == pytest.approx(1.0)
    empty = pd.Series(dtype=float)
    assert np.isnan(metrics.calculate_cosine_similarity(empty, empty))
    assert metrics.calculate_cosine_similarity(w1, empty) == 0.0
    # duplicate indices handled via _safe_series_get
    dup = pd.Series([1.0, 2.0], index=["A -> B", "A -> B"])
    val = metrics.calculate_cosine_similarity(dup, w1)
    assert 0.0 <= val <= 1.0
    assert metrics._safe_series_get(dup, "A -> B") == pytest.approx(3.0)
    assert metrics._safe_series_get(dup, "missing", default=7) == 7.0


def test_edge_list_rank_correlation(metrics):
    w1 = _edge_series({("A", "B"): 1.0, ("B", "C"): 2.0, ("C", "D"): 3.0})
    w2 = _edge_series({("A", "B"): 2.0, ("B", "C"): 4.0, ("C", "D"): 6.0})
    corr = metrics.calculate_edge_list_rank_correlation(w1, w2)
    assert corr == pytest.approx(1.0)
    norm = metrics.calculate_edge_list_rank_correlation(w1, w2, normalize=True)
    assert norm == pytest.approx(1.0)
    # insufficient data
    assert np.isnan(metrics.calculate_edge_list_rank_correlation(w1.iloc[:1], w2.iloc[:1]))
    # constant vector -> NaN
    const = _edge_series({("A", "B"): 1.0, ("B", "C"): 1.0, ("C", "D"): 1.0})
    assert np.isnan(metrics.calculate_edge_list_rank_correlation(const, w2))


def test_path_list_rank_correlation(metrics):
    p1 = pd.Series({"A->B": 1.0, "B->C": 2.0, "C->D": 3.0})
    p2 = pd.Series({"A->B": 3.0, "B->C": 2.0, "C->D": 1.0})
    corr = metrics.calculate_path_list_rank_correlation(p1, p2)
    assert -1.0 <= corr <= 1.0
    norm = metrics.calculate_path_list_rank_correlation(p1, p2, normalize=True)
    assert 0.0 <= norm <= 1.0
    assert np.isnan(metrics.calculate_path_list_rank_correlation(p1.iloc[:1], p2.iloc[:1]))


def test_graph_kernel_similarity(metrics):
    w1 = _edge_series({("A", "B"): 1.0, ("B", "C"): 2.0, ("C", "D"): 3.0})
    w2 = _edge_series({("A", "B"): 1.5, ("B", "C"): 2.5, ("C", "D"): 3.5})
    sim = metrics.calculate_graph_kernel_similarity(w1, w2)
    assert 0.0 <= sim <= 1.0
    sim_same = metrics.calculate_graph_kernel_similarity(w1, w1)
    assert sim_same == pytest.approx(1.0, abs=1e-6)
    # empty graphs
    empty = pd.Series(dtype=float)
    assert metrics.calculate_graph_kernel_similarity(empty, empty) == 1.0
    # one-sided empty: shared node labels still yield a partial match
    assert 0.0 <= metrics.calculate_graph_kernel_similarity(w1, empty) <= 1.0
    assert 0.0 <= metrics.calculate_graph_kernel_similarity(empty, w1) <= 1.0
    # unknown kernel type
    assert np.isnan(metrics.calculate_graph_kernel_similarity(w1, w2, kernel_type="xx"))

    # _wl_features with simple (unweighted) adjacency
    adj = {0: [1], 1: [0, 2], 2: [1]}
    labels = {0: "A", 1: "B", 2: "C"}
    feats = metrics._wl_features(adj, labels, n_iterations=2)
    assert sum(feats.values()) == 9  # 3 nodes x (1 init + 2 iterations)

    # _cosine_similarity_dicts edge cases
    assert metrics._cosine_similarity_dicts({}, {}) == 1.0
    assert metrics._cosine_similarity_dicts({"a": 1}, {"b": 1}) == 0.0


# ---------------------------------------------------------------------------
# v2.2 representatives (plan-similarity-matrix-schema-v2)
# ---------------------------------------------------------------------------

def test_overlap_coefficients(metrics):
    a = {"A -> B", "B -> C", "X -> Y"}
    b = {"A -> B", "C -> D"}
    cov_a, cov_b, cov_min = metrics.calculate_overlap_coefficients(a, b)
    assert cov_a == pytest.approx(1 / 3)
    assert cov_b == pytest.approx(0.5)
    assert cov_min == pytest.approx(1 / 3)
    # both empty = identical
    assert metrics.calculate_overlap_coefficients(set(), set()) == (1.0, 1.0, 1.0)
    # one-sided empty leaves that direction undefined
    cov_a, cov_b, cov_min = metrics.calculate_overlap_coefficients({"A -> B"}, set())
    assert np.isnan(cov_b) and np.isnan(cov_min) and cov_a == 0.0


def test_topk_overlap(metrics):
    wa = pd.Series([5.0, 3.0, 1.0], index=["A -> B", "B -> C", "A -> C"])
    wb = pd.Series([4.0, 2.0, 0.5, 7.0], index=["A -> B", "B -> C", "A -> C", "C -> D"])
    # k=2: A's top-2 {A->B, B->C}; B's top-2 {C->D, A->B}; intersection 1/2
    assert metrics.calculate_topk_overlap(wa, wb, k=2) == pytest.approx(0.5)
    # identical vectors = 1.0
    assert metrics.calculate_topk_overlap(wa, wa, k=2) == 1.0
    # empty side = NaN
    empty = pd.Series(dtype=float)
    assert np.isnan(metrics.calculate_topk_overlap(empty, wb))


def test_path_jaccard_and_diagnostics(metrics):
    path_data = pd.DataFrame(
        {"d1": [3.0, 2.0, 0.0, np.nan], "d2": [1.0, 0.0, 4.0, 5.0]},
        index=["A -> B -> C", "B -> C", "C -> D", "D -> E"],
    )
    # presence: d1 = {A->B->C, B->C}, d2 = {A->B->C, C->D, D->E}
    # jaccard = 1 / 4 (with the small-sample gate lowered for the fixture)
    assert metrics.calculate_path_jaccard(
        path_data, "d1", "d2", min_support=2) == pytest.approx(0.25)
    # default min_support=5 gates the tiny fixture -> NaN
    assert np.isnan(metrics.calculate_path_jaccard(path_data, "d1", "d2"))
    # top-k overlap on bottleneck weights: d1's top-1 is A->B->C (3.0),
    # d2's is D->E (5.0) — disjoint heads at k=1
    assert metrics.calculate_path_topk_overlap(path_data, "d1", "d2", k=1) == 0.0
    # hop W1: d1 hops [3, 2] vs d2 hops [3, 2, 2, 2]
    w1 = metrics.calculate_hop_profile_w1(path_data, "d1", "d2")
    assert 0 <= w1 <= 1.0
    # missing frame/columns -> NaN
    assert np.isnan(metrics.calculate_path_jaccard(None, "d1", "d2"))
    assert np.isnan(metrics.calculate_hop_profile_w1(path_data, "d1", "zz"))


def test_netsimile_similarity_and_strength_w1(metrics):
    wa = pd.Series([5.0, 3.0, 1.0], index=["A -> B", "B -> C", "A -> C"])
    wb = pd.Series([4.0, 2.0, 0.5, 7.0], index=["A -> B", "B -> C", "A -> C", "C -> D"])
    sim_same = metrics.calculate_netsimile_similarity(wa, wa)
    assert sim_same == pytest.approx(1.0)
    sim_diff = metrics.calculate_netsimile_similarity(wa, wb)
    assert 0.0 <= sim_diff < 1.0
    empty = pd.Series(dtype=float)
    assert np.isnan(metrics.calculate_netsimile_similarity(empty, wb))
    w1_out, w1_in = metrics.calculate_strength_w1(wa, wb)
    assert w1_out >= 0 and w1_in >= 0
    assert np.isnan(metrics.calculate_strength_w1(empty, wb)[0])


def test_spearman_min_shared_gate(metrics):
    wa = pd.Series([5.0, 3.0, 1.0], index=["A -> B", "B -> C", "A -> C"])
    wb = pd.Series([4.0, 2.0], index=["A -> B", "B -> C"])
    # legacy behavior (no gate): 2 shared edges -> NaN via the <3 rule
    assert np.isnan(metrics.calculate_spearman_rank_correlation(wa, wb))
    # perfectly rank-correlated larger sample passes the gate
    big_a = pd.Series(range(40, 0, -1), index=[f"n{i} -> m{i}" for i in range(40)])
    big_b = pd.Series(range(40, 0, -1), index=[f"n{i} -> m{i}" for i in range(40)])
    gated = metrics.calculate_spearman_rank_correlation(big_a, big_b, min_shared=30)
    assert gated == pytest.approx(1.0)
    # same sample against a 30-edge gate with only 20 shared -> NaN
    small = big_a.iloc[:20]
    assert np.isnan(metrics.calculate_spearman_rank_correlation(small, big_b, min_shared=30))


class TestSimilarityV22RepresentativeMetrics:
    """v2.2 representative metrics (plan-similarity-matrix-schema-v2 §4):
    per-metric unit tests including floor/NaN behavior, tie safety and the
    conserved-only-vs-union path distinction."""

    # -- overlap coefficients ------------------------------------------

    def test_overlap_coefficients_basic(self, metrics):
        a = {('x', 'y'), ('p', 'q')}
        b = {('x', 'y'), ('r', 's'), ('t', 'u')}
        cov_a, cov_b, cov_min = metrics.calculate_overlap_coefficients(a, b)
        assert cov_a == 0.5
        assert cov_b == pytest.approx(1 / 3)
        assert cov_min == min(cov_a, cov_b)

    def test_overlap_coefficients_nan_and_identity(self, metrics):
        cov_a, cov_b, cov_min = metrics.calculate_overlap_coefficients(
            set(), {('a', 'b')})
        assert cov_a != cov_a            # NaN for the empty direction
        assert cov_b == 0.0              # nothing of B is covered by A
        assert cov_min != cov_min
        assert metrics.calculate_overlap_coefficients(
            set(), set()) == (1.0, 1.0, 1.0)

    # -- top-k overlap ---------------------------------------------------

    def test_topk_overlap_counts(self, metrics):
        wa = pd.Series({'e1': 5.0, 'e2': 4.0, 'e3': 1.0})
        wb = pd.Series({'e1': 7.0, 'e2': 6.0, 'e9': 1.0})
        assert metrics.calculate_topk_overlap(wa, wb, k=2) == 1.0
        assert metrics.calculate_topk_overlap(wa, wb, k=3) == 2 / 3

    def test_topk_overlap_tie_safe_deterministic_head(self, metrics):
        # equal-weight heads: sort order is stable, so the same input
        # gives the same head set and the same score on both calls
        wa = pd.Series({'e1': 5.0, 'e2': 5.0, 'e3': 5.0, 'e4': 5.0})
        wb = pd.Series({'e1': 5.0, 'e2': 5.0, 'e9': 5.0, 'e10': 5.0})
        first = metrics.calculate_topk_overlap(wa, wb, k=2)
        assert first == metrics.calculate_topk_overlap(wa, wb, k=2)
        assert 0.0 <= first <= 1.0

    def test_topk_overlap_nan_and_shrink_and_nonpositive(self, metrics):
        empty = pd.Series(dtype=float)
        assert metrics.calculate_topk_overlap(
            empty, pd.Series({'a': 1.0})) != metrics.calculate_topk_overlap(
                empty, pd.Series({'a': 1.0}))  # NaN != NaN
        # k shrinks to the smaller side
        wa = pd.Series({'a': 2.0})
        wb = pd.Series({'a': 3.0, 'b': 1.0})
        assert metrics.calculate_topk_overlap(wa, wb, k=20) == 1.0
        # non-positive weights are floor-excluded before ranking
        zero = pd.Series({'a': 0.0, 'b': -1.0})
        assert metrics.calculate_topk_overlap(
            zero, pd.Series({'a': 1.0})) != metrics.calculate_topk_overlap(
                zero, pd.Series({'a': 1.0}))

    # -- path metrics: union frame vs conserved-only ---------------------

    def _path_frame(self):
        return pd.DataFrame(
            {'A': [3, 0, 2, 1], 'B': [1, 2, 0, 4]},
            index=['p1 -> p2', 'p2 -> p3', 'p3 -> p4', 'p1 -> p4'])

    def test_path_jaccard_union_frame(self, metrics):
        df = self._path_frame()
        # present A = {p1->p2, p3->p4, p1->p4}; B = {p1->p2, p2->p3, p1->p4}
        # jaccard = |{p12, p14}| / |union 4| = 0.5
        assert metrics.calculate_path_jaccard(
            df, 'A', 'B', min_support=2) == pytest.approx(0.5)

    def test_path_jaccard_min_support_nan(self, metrics):
        df = pd.DataFrame({'A': [1, 0], 'B': [1, 1]},
                          index=['a -> b', 'b -> c'])
        assert metrics.calculate_path_jaccard(
            df, 'A', 'B', min_support=5) != metrics.calculate_path_jaccard(
                df, 'A', 'B', min_support=5)

    def test_path_jaccard_conserved_only_frame_trivially_one(self, metrics):
        # the exported path_presence_matrix CSV is pre-intersected
        # (conserved-only) — the metric is trivially 1.0 there, which is
        # exactly why the contract demands the UNION frame
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]},
                          index=['x -> y', 'y -> z'])
        assert metrics.calculate_path_jaccard(
            df, 'A', 'B', min_support=1) == 1.0

    def test_path_topk_overlap_bottleneck_ranking(self, metrics):
        df = pd.DataFrame({'A': [5, 1], 'B': [4, 0]},
                          index=['a -> b', 'b -> c'])
        assert metrics.calculate_path_topk_overlap(
            df, 'A', 'B', k=1) == 1.0

    def test_hop_profile_w1_value(self, metrics):
        idx = ['a -> b', 'b -> c', 'a -> b -> c', 'a -> b -> c -> d']
        df = pd.DataFrame({'A': [1, 1, 1, 1], 'B': [1, 1, 0, 0]}, index=idx)
        # A hop profile [1, 1, 2, 3]; B hop profile [1, 1]
        assert metrics.calculate_hop_profile_w1(
            df, 'A', 'B') == pytest.approx(0.75)

    # -- NetSimile-lite ----------------------------------------------------

    def test_netsimile_identical_graphs_score_one(self, metrics):
        w = _edge_series({('n1', 'n2'): 3.0, ('n2', 'n3'): 1.5,
                          ('n1', 'n3'): 2.0})
        assert metrics.calculate_netsimile_similarity(w, w) == 1.0

    def test_netsimile_empty_nan(self, metrics):
        empty = pd.Series(dtype=float)
        w = _edge_series({('n1', 'n2'): 1.0})
        assert metrics.calculate_netsimile_similarity(
            empty, w) != metrics.calculate_netsimile_similarity(empty, w)

    def test_netsimile_signature_shape_and_content(self, metrics):
        w = _edge_series({('n1', 'n2'): 3.0, ('n2', 'n3'): 1.5})
        sig = metrics.netsimile_signature(w)
        assert sig is not None and sig.shape == (8,)
        assert metrics.netsimile_signature(pd.Series(dtype=float)) is None
