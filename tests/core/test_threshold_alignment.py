"""Tests for the cross-dataset threshold alignment work
(plan-cross-dataset-threshold-alignment.md).

Covers:
- Feature C prober: precompute + count(t) vs brute-force filtering;
  monotonicity; bisection best-match vs exhaustive argmin on random
  monotone curves (including plateaus — the BANC >= 3 truncation case);
  cap respected; ±2 neighbors checked for near-ties.
- Feature C metrics: edge-count distance formula; Jaccard empty cases;
  rank similarity blank when shared edges < 3.
- Feature C export: alignment files produced from an analyzer with fake
  raw_results; the typed-only matrix shape; the summary log line.
- Feature E: dataset_thresholds validation, per-dataset lookup, union
  thresholds, to_dict/from_dict round-trip.
- Feature A: metadata density (flywire derivation fixture + neuron-table
  path) — see test_metadata_density.py.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.threshold_alignment import (  # noqa: E402
    EdgeDensityProber,
    build_alignment_matrix,
    edge_count_distance,
    jaccard,
    rank_similarity,
    ALIGNMENT_TOLERANCE,
)


# ---------------------------------------------------------------------------
# Prober: count(t) vs brute force + monotonicity
# ---------------------------------------------------------------------------

def _extract_from_pairs(pairs):
    """Expand {(pre, post): max_weight} into a bodyId-level extract frame
    (two bodyId rows per type pair, as in real extracts)."""
    rows = []
    for i, ((pre, post), w) in enumerate(pairs.items()):
        rows.append((f"b{i}a", f"b{i}x", pre, post, w))
        rows.append((f"b{i}b", f"b{i}y", pre, post, max(1, w - 1)))
    return pd.DataFrame(rows, columns=[
        "bodyId_pre", "bodyId_post", "type_pre", "type_post", "weight"])


def test_count_matches_brute_force():
    pairs = {
        ("A", "B"): 5, ("A", "C"): 3, ("B", "C"): 8,
        ("C", "D"): 1, ("D", "A"): 12,
    }
    extract = _extract_from_pairs(pairs)
    prober = EdgeDensityProber(extract)
    for t in range(0, 14):
        brute = sum(1 for w in pairs.values() if w >= t)
        assert prober.count(t) == brute, f"t={t}"
    assert prober.total_pairs == len(pairs)


def test_count_monotone_non_increasing():
    rng = np.random.default_rng(42)
    pairs = {(f"p{i}", f"q{i}"): int(w) for i, w in enumerate(
        rng.integers(1, 40, size=200))}
    prober = EdgeDensityProber(_extract_from_pairs(pairs))
    counts = [prober.count(t) for t in range(0, 45)]
    assert all(a >= b for a, b in zip(counts, counts[1:]))


def test_empty_extract_is_safe():
    prober = EdgeDensityProber(pd.DataFrame(
        columns=["type_pre", "type_post", "weight"]))
    assert prober.count(3) == 0
    assert prober.best_match(5)["best_t"] is None
    assert prober.edge_set(3) == set()


# ---------------------------------------------------------------------------
# Prober: bisection best-match vs exhaustive argmin (incl. plateaus)
# ---------------------------------------------------------------------------

def _random_prober(rng, n_pairs=150, max_w=30):
    pairs = {(f"p{i}", f"q{i}"): int(w)
             for i, w in enumerate(rng.integers(1, max_w + 1, size=n_pairs))}
    return EdgeDensityProber(_extract_from_pairs(pairs)), pairs


@pytest.mark.parametrize("seed", range(8))
def test_best_match_matches_exhaustive_argmin(seed):
    rng = np.random.default_rng(seed)
    prober, _ = _random_prober(rng)
    cap = 30
    for anchor_count in (1, 5, 17, 40, 120):
        match = prober.best_match(anchor_count, cap=cap)
        # exhaustive argmin over the grid
        best_d = min(abs(prober.count(t) - anchor_count) for t in range(1, cap + 1))
        assert abs(prober.count(match["best_t"]) - anchor_count) == best_d
        assert match["count_at_best_t"] == prober.count(match["best_t"])
        assert match["count_distance"] == pytest.approx(
            edge_count_distance(anchor_count, match["count_at_best_t"]))


def test_best_match_handles_plateau_banc_truncation():
    # BANC-like: download pre-truncated at weight >= 3, so counts at
    # t = 1/2/3 are identical (a plateau the bisection must not stumble on)
    pairs = {(f"p{i}", f"q{i}"): int(w) for i, w in enumerate(
        [3, 3, 4, 7, 9, 15, 22, 30])}
    prober = EdgeDensityProber(_extract_from_pairs(pairs))
    assert prober.count(1) == prober.count(2) == prober.count(3) == 8
    match = prober.best_match(8, cap=30)
    assert match["best_t"] in (1, 2, 3)
    assert match["count_distance"] == 0.0


def test_best_match_checks_neighbors_for_near_ties():
    # weights {10,8,6,4,2} -> counts t=1..2:5, t=3..4:4, t=5..6:3, t=7..8:2,
    # t=9..10:1. Anchor count 2 has an EXACT match on a plateau (7..8); the
    # ±2 window around the bisection crossover must keep it.
    pairs = {("a", "b"): 10, ("c", "d"): 8, ("e", "f"): 6, ("g", "h"): 4,
             ("i", "j"): 2}
    prober = EdgeDensityProber(_extract_from_pairs(pairs))
    match = prober.best_match(2, cap=10)
    assert match["count_at_best_t"] == 2
    assert match["count_distance"] == 0.0
    assert match["best_t"] in (7, 8)

    # anchor between plateaus: count 4 exists (t=3..4) but count 0 does not;
    # asking for the closest to a nonexistent 4.5-equivalent still returns
    # a real argmin over the grid.
    match = prober.best_match(5, cap=10)
    assert match["best_t"] == 1  # only t in (1,2) has count 5


def test_best_match_respects_cap():
    rng = np.random.default_rng(7)
    prober, _ = _random_prober(rng, max_w=60)
    match = prober.best_match(1, cap=30)
    assert 1 <= match["best_t"] <= 30


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_edge_count_distance_formula():
    assert edge_count_distance(10, 10) == 0.0
    assert edge_count_distance(0, 0) == 0.0
    assert edge_count_distance(8, 10) == pytest.approx(0.2)
    assert edge_count_distance(10, 8) == pytest.approx(0.2)  # symmetric


def test_jaccard_empty_sets():
    assert jaccard(set(), set()) is None
    assert jaccard({("a", "b")}, {("a", "b")}) == 1.0
    assert jaccard({("a", "b")}, set()) == 0.0


def test_rank_similarity_blank_below_shared_floor():
    wa = {("a", "b"): 5.0, ("c", "d"): 3.0}
    wb = {("a", "b"): 7.0, ("c", "d"): 2.0}
    assert rank_similarity(wa, wb) is None  # 2 shared < 3

    wa = {("a", "b"): 5.0, ("c", "d"): 3.0, ("e", "f"): 1.0}
    wb = {("a", "b"): 7.0, ("c", "d"): 2.0, ("e", "f"): 4.0}
    val = rank_similarity(wa, wb)
    assert val is not None and -1.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# Feature C export: alignment files on a fake analyzer
# ---------------------------------------------------------------------------

def _make_analyzer(tmp_path, datasets=("male-cns:v1.0", "flywire_FAFB_v783"),
                   thresholds=(3, 5, 10)):
    from comparison.comparison_analyzer import ComparisonAnalyzer
    from comparison.comparison_parameters import ComparisonParameters

    params = ComparisonParameters(
        datasets=list(datasets),
        source_neurons=["L2"],
        target_neurons=["l-LNv"],
        max_interlayer=2,
        thresholds=list(thresholds),
        output_folder=str(tmp_path),
        auto_type_mapping=False,
        verbose=False,
    )
    return ComparisonAnalyzer(params, verbose=False)


def _frame(rows):
    return pd.DataFrame(rows, columns=["type_pre", "type_post", "weight"])


def test_export_threshold_alignment_outputs(tmp_path):
    an = _make_analyzer(tmp_path)
    ds_a, ds_b = "male-cns:v1.0", "flywire_FAFB_v783"
    # Dataset A: 10 pairs at t=3; dataset B: pairs engineered so its
    # best match for A@3 sits at an extended threshold (not 3).
    an.raw_results = {
        ds_a: {3: _frame([(f"a{i}", f"b{i}", 5) for i in range(10)])},
        ds_b: {3: _frame([(f"a{i}", f"b{i}", 9) for i in range(9)] + [
            ("ax", "bx", 20)])},
    }
    out = tmp_path / "comparison_results"
    out.mkdir()

    an._export_threshold_alignment(str(out))

    best = pd.read_csv(out / "threshold_alignment_best_matches.csv")
    assert {"reference_dataset", "anchor_threshold", "target_dataset",
            "best_t", "count_distance", "match_kind"} <= set(best.columns)
    # named anchor rows exist for every (anchor, other-dataset) pair in
    # BOTH directions (each dataset serves as reference against the other)
    anchors = best[best["match_kind"] == "anchor"]
    assert len(anchors) == 3 * 2  # 3 typed thresholds x 1 other, both refs
    # B's pair_max is 9 for 9 pairs and 20 for 1: matching A@3 (10 pairs)
    # lands on B's plateau count=10 (t=1..9) — any best_t there with
    # count_at_best_t=10 is a valid argmin (exact match, d=0).
    row = anchors[(anchors.reference_dataset == ds_a)
                  & (anchors.anchor_threshold == 3)
                  & (anchors.target_dataset == ds_b)].iloc[0]
    assert int(row["count_at_best_t"]) == 10
    assert float(row["count_distance"]) == 0.0
    assert 1 <= int(row["best_t"]) <= 9

    matrix = pd.read_csv(out / "threshold_alignment_matrix.csv")
    # typed grid: 2 datasets x 3 thresholds = 6 points -> C(6,2) = 15 pairs
    assert len(matrix) == 15
    assert (matrix["edge_count_distance"] <= 1.0 + 1e-9).all()

    density = pd.read_csv(out / "edge_density_per_threshold.csv")
    assert set(density.columns) >= {"dataset", "threshold", "pair_count",
                                    "is_typed_threshold"}
    # extended grid: cap = max(3*10, 30) = 30 points per dataset
    assert len(density[density.dataset == ds_a]) == 30


def test_alignment_log_line_emitted(tmp_path, monkeypatch):
    an = _make_analyzer(tmp_path)
    ds_a, ds_b = "male-cns:v1.0", "flywire_FAFB_v783"
    an.raw_results = {
        ds_a: {3: _frame([(f"a{i}", f"b{i}", 5) for i in range(6)])},
        ds_b: {3: _frame([(f"a{i}", f"b{i}", 4) for i in range(6)])},
    }
    out = tmp_path / "comparison_results"
    out.mkdir()

    messages = []
    monkeypatch.setattr(an, "_log", lambda msg, level="info": messages.append(msg))
    an._export_threshold_alignment(str(out))
    # identical densities -> d=0.00 summary line
    summary = [m for m in messages if "aligns best with" in m]
    assert summary, messages[-5:]
    assert f"{ds_a}@3" in summary[0]


# ---------------------------------------------------------------------------
# Feature E: per-dataset thresholds
# ---------------------------------------------------------------------------

def test_dataset_thresholds_validation_and_union():
    from comparison.comparison_parameters import ComparisonParameters

    params = ComparisonParameters(
        datasets=["banc:v626", "flywire_FAFB_v783", "male-cns:v1.0"],
        thresholds=[3, 5, 10],
        dataset_thresholds={
            "flywire_FAFB_v783": [7, 11],
            "male-cns:v1.0": [8, 15, 8],   # dup 8 collapses
            "unknown-dataset:v9": [1],      # dropped with a warning
        },
        verbose=False,
        allow_single_dataset=True,
    )
    assert params.dataset_thresholds == {
        "flywire_FAFB_v783": [7, 11],
        "male-cns:v1.0": [8, 15],
    }
    # thresholds = sorted union of global + overrides
    assert params.thresholds == [3, 5, 7, 8, 10, 11, 15]
    assert params.get_thresholds_for_dataset("banc:v626") == [3, 5, 10]
    assert params.get_thresholds_for_dataset("flywire_FAFB_v783") == [7, 11]
    assert params.get_thresholds_for_dataset("male-cns:v1.0") == [8, 15]


def test_dataset_thresholds_round_trip():
    from comparison.comparison_parameters import ComparisonParameters

    params = ComparisonParameters(
        datasets=["banc:v626", "flywire_FAFB_v783"],
        thresholds=[3, 5],
        dataset_thresholds={"flywire_FAFB_v783": [7, 11]},
        verbose=False,
    )
    data = params.to_dict()
    assert data["dataset_thresholds"] == {"flywire_FAFB_v783": [7, 11]}
    restored = ComparisonParameters.from_dict(dict(data))
    assert restored.dataset_thresholds == {"flywire_FAFB_v783": [7, 11]}
    assert restored.get_thresholds_for_dataset("flywire_FAFB_v783") == [7, 11]
    assert restored.thresholds == [3, 5, 7, 11]


def test_run_loops_honor_per_dataset_lists(tmp_path, monkeypatch):
    """Per-dataset lists drive the path run loop: BANC@{3,5} + FAFB@{7}."""
    an = _make_analyzer(tmp_path, datasets=("banc:v626", "flywire_FAFB_v783"),
                        thresholds=(3, 5, 10, 7, 11))
    an.parameters.dataset_thresholds = {
        "banc:v626": [3, 5],
        "flywire_FAFB_v783": [7],
    }
    # Feature E only: disable the Feature F replay batch so the loop uses
    # run_path_analysis (monkeypatched) per threshold.
    an.parameters.replay_paths = False

    calls = []

    def fake_run(dataset_name, threshold, verbose_mode="simple"):
        calls.append((dataset_name, threshold))
        return _frame([("X", "Y", 9)])

    monkeypatch.setattr(an, "run_path_analysis", fake_run)
    an._run_all_path_analyses(skip_existing=True)

    assert sorted(calls) == [("banc:v626", 3), ("banc:v626", 5),
                             ("flywire_FAFB_v783", 7)]


# ---------------------------------------------------------------------------
# Alignment-derived combinations (plan §7C)
# ---------------------------------------------------------------------------

def test_suggest_combination_rows_builds_aligned_query_rows():
    from comparison.threshold_alignment import suggest_combination_rows

    best = pd.DataFrame([
        {'reference_dataset': 'mcns', 'anchor_threshold': 19,
         'target_dataset': 'banc', 'best_t': 6, 'count_distance': 0.02,
         'match_kind': 'anchor'},
        {'reference_dataset': 'mcns', 'anchor_threshold': 19,
         'target_dataset': 'fafb', 'best_t': 17, 'count_distance': 0.03,
         'match_kind': 'anchor'},
        {'reference_dataset': 'mcns', 'anchor_threshold': 30,
         'target_dataset': 'banc', 'best_t': 10, 'count_distance': 0.05,
         'match_kind': 'anchor'},
        {'reference_dataset': 'mcns', 'anchor_threshold': 30,
         'target_dataset': 'fafb', 'best_t': 30, 'count_distance': 0.01,
         'match_kind': 'anchor'},
    ])
    rows = suggest_combination_rows(best, 'mcns', ['mcns', 'banc', 'fafb'])
    assert len(rows) == 2
    assert rows[0]['thresholds'] == {'mcns': 19, 'banc': 6, 'fafb': 17}
    assert rows[1]['thresholds'] == {'mcns': 30, 'banc': 10, 'fafb': 30}
    assert all(r['aligned_within_tolerance'] for r in rows)


def test_suggest_combination_rows_skips_incomplete():
    from comparison.threshold_alignment import suggest_combination_rows

    best = pd.DataFrame([
        {'reference_dataset': 'mcns', 'anchor_threshold': 19,
         'target_dataset': 'banc', 'best_t': 6, 'count_distance': 0.02,
         'match_kind': 'anchor'},
        # No fafb row -> incomplete, dropped.
    ])
    rows = suggest_combination_rows(best, 'mcns', ['mcns', 'banc', 'fafb'])
    assert rows == []


def test_suggest_combination_rows_flags_out_of_tolerance():
    from comparison.threshold_alignment import suggest_combination_rows

    best = pd.DataFrame([
        {'reference_dataset': 'mcns', 'anchor_threshold': 19,
         'target_dataset': 'banc', 'best_t': 6, 'count_distance': 0.8,
         'match_kind': 'anchor'},
    ])
    rows = suggest_combination_rows(best, 'mcns', ['mcns', 'banc'])
    assert len(rows) == 1
    assert rows[0]['aligned_within_tolerance'] is False
