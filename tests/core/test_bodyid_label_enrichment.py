"""BodyId-level export label enrichment in the connectivity similarity tab.

The bodyId axes of ``ConnectivityProfileComparer`` matrices and the
HomologFinder bodyId tables carry the tree-legend suffixes
('{bodyId}_{instance}' or '{bodyId}_{type}_{L|R}'); the neuron tables are
monkeypatched so no real dataset folders are read.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import comparison.profile_comparator as pc  # noqa: E402
from comparison.connectivity_profiler import ConnectivityProfile  # noqa: E402
from comparison.profile_comparator import (  # noqa: E402
    ConnectivityProfileComparer,
    HomologFinder,
    _body_id_display_label,
    _body_id_instance_name,
)

DS_A = "flywire_FAFB_v783"
DS_B = "male-cns:v1.0"


def _profile(bid, ds):
    """Minimal valid profile (rich enough for combined_score)."""
    upstream = {"A": 10.0, "B": 5.0}
    downstream = {"C": 8.0}
    profile = ConnectivityProfile(
        neuron_id=bid, dataset=ds,
        upstream_partners=upstream, downstream_partners=downstream,
        upstream_ranks={"A": 1, "B": 2}, downstream_ranks={"C": 1},
        total_upstream_weight=15.0, total_downstream_weight=8.0,
    )
    profile.actual_upstream_count = len(upstream)
    profile.actual_downstream_count = len(downstream)
    profile.top_k_bodyid_used = 2
    profile.unique_types_upstream = len(upstream)
    profile.unique_types_downstream = len(downstream)
    profile._connectivity_status = profile._compute_connectivity_status().value
    profile.is_weak_connectivity = (
        not profile.connectivity_status.is_valid_for_comparison())
    return profile


def _patch_maps(monkeypatch, type_map, instance_map, side_map=None):
    monkeypatch.setattr(
        pc, "_load_body_id_label_maps",
        lambda ds: (type_map, instance_map, side_map or {}))


# ------------------------------------------------------- _body_id_display_label
def test_display_label_uses_instance_on_neuprint(monkeypatch):
    _patch_maps(monkeypatch, {1: "Mi1"}, {1: "Mi1_A3"})
    assert _body_id_display_label(DS_B, 1, fallback_type="Mi1") == "1_Mi1_A3"


def test_display_label_uses_type_hemisphere_on_fafb(monkeypatch):
    _patch_maps(monkeypatch, {"9": "aMe4"}, {"9": "AA_R"})
    assert _body_id_display_label(DS_A, "9", fallback_type="Mi1") == "9_aMe4_R"


def test_display_label_falls_back_to_resolved_type(monkeypatch):
    _patch_maps(monkeypatch, {}, {})
    assert _body_id_display_label(DS_B, 7, fallback_type="Mi1") == "7_Mi1"


def test_bodyid_instance_name_reads_map(monkeypatch):
    _patch_maps(monkeypatch, {}, {1: "Mi1_A3", "9": "AA_R"})
    assert _body_id_instance_name(DS_B, 1) == "Mi1_A3"
    assert _body_id_instance_name(DS_A, "9") == "AA_R"
    assert _body_id_instance_name(DS_B, 999) == ""


# --------------------------------- _compute_bodyid_similarity_matrices
def test_bodyid_similarity_matrix_axes_are_enriched(monkeypatch):
    _patch_maps(monkeypatch, {}, {1: "Mi1_A3", 2: "Mi1_B7"})
    comparer = ConnectivityProfileComparer(
        query=["Mi1"], dataset=DS_B, verbose=False)
    matrices = comparer._compute_bodyid_similarity_matrices({
        ("Mi1", 1): _profile(1, DS_B),
        ("Mi1", 2): _profile(2, DS_B),
    })
    df = matrices["overall"]["jaccard"]
    assert list(df.index) == ["1_Mi1_A3", "2_Mi1_B7"]
    assert list(df.columns) == ["1_Mi1_A3", "2_Mi1_B7"]


def test_bodyid_similarity_matrix_axes_without_metadata(monkeypatch):
    # No neuron table -> the resolved comparison type is the fallback, so
    # the axes never regress to bare bodyIds.
    _patch_maps(monkeypatch, {}, {})
    comparer = ConnectivityProfileComparer(
        query=["Mi1"], dataset=DS_B, verbose=False)
    matrices = comparer._compute_bodyid_similarity_matrices({
        ("Mi1", 1): _profile(1, DS_B),
        ("Mi1", 2): _profile(2, DS_B),
    })
    assert list(matrices["overall"]["jaccard"].index) == ["1_Mi1", "2_Mi1"]


# ------------------------------------------------- anchor rows / aggregates
def test_anchor_row_label_enriches_digit_anchors():
    assert ConnectivityProfileComparer._anchor_row_label(
        "12345", {DS_B: ("Mi1", None), DS_A: ("Mi1", None)}) == "12345_Mi1"
    # Named anchors and unresolvable anchors pass through.
    assert ConnectivityProfileComparer._anchor_row_label("Mi1", {}) == "Mi1"
    assert ConnectivityProfileComparer._anchor_row_label("12345", None) == "12345"


def test_aggregate_inter_matrices_label_digit_anchor_rows():
    comparer = ConnectivityProfileComparer(
        query=["12345"], datasets=[DS_A, DS_B], verbose=False)
    inter_matrices = {
        "12345": {
            "overall": {
                "jaccard": pd.DataFrame(
                    [[0.4, 0.5], [0.5, np.nan]],
                    index=[DS_A, DS_B], columns=[DS_A, DS_B], dtype=float),
            },
        },
    }
    anchor_profiles = {
        "12345": {DS_A: ("Mi1", None), DS_B: ("Mi1", None)},
    }
    aggregate = comparer._aggregate_inter_dataset_matrices(
        inter_matrices, anchor_profiles)
    frame = aggregate["overall"]["jaccard"]
    assert list(frame.index) == ["12345_Mi1"]
    assert list(frame.columns) == [f"{DS_A} vs {DS_B}"]


# ------------------------------------------------- HomologFinder columns
def test_homolog_finder_attach_instance_columns(monkeypatch):
    _patch_maps(monkeypatch, {}, {1: "Mi1_A3", 22: "PN_B1"})
    finder = HomologFinder.__new__(HomologFinder)  # no __init__ side effects
    df = pd.DataFrame({
        "source_bodyId": [1],
        "source_type": ["Mi1"],
        "target_bodyId": [22],
        "target_type": ["PN"],
        "target_dataset": [DS_A],
    })
    out = finder._attach_instance_columns(df, DS_B, DS_A)
    assert list(out.columns)[:6] == [
        "source_bodyId", "source_type", "source_instance",
        "target_bodyId", "target_type", "target_instance"]
    assert out["source_instance"].iloc[0] == "Mi1_A3"
    # The target resolves through the row's dataset (FAFB).
    assert out["target_instance"].iloc[0] == "PN_B1"


def test_homolog_finder_attach_instance_columns_idempotent(monkeypatch):
    _patch_maps(monkeypatch, {}, {1: "Mi1_A3"})
    finder = HomologFinder.__new__(HomologFinder)
    df = pd.DataFrame({
        "source_bodyId": [1],
        "source_type": ["Mi1"],
        "source_instance": ["already"],
    })
    out = finder._attach_instance_columns(df, DS_B, DS_B)
    assert out["source_instance"].iloc[0] == "already"


def test_homolog_finder_attach_instance_columns_empty_frame():
    finder = HomologFinder.__new__(HomologFinder)
    df = pd.DataFrame()
    out = finder._attach_instance_columns(df, DS_B, DS_B)
    assert out.empty


# ------------------------------------------------- profile filename + type fill
def test_bodyid_profile_stem_avoids_doubling():
    from comparison.profile_comparator import _bodyid_profile_stem
    # bodyid aggregation: the label already carries the bodyId prefix.
    assert _bodyid_profile_stem(12211, "12211_aMe12_L") == "12211_aMe12_L"
    # type aggregation: prefix the resolved type.
    assert _bodyid_profile_stem(12211, "aMe12") == "12211_aMe12"
    # label equal to the bare bid still gets the prefix form.
    assert _bodyid_profile_stem(12211, "12211") == "12211_12211"


def test_profiler_fill_neuron_type():
    from comparison.connectivity_profiler import ConnectivityProfiler

    profiler = ConnectivityProfiler(datasets=[DS_B], verbose=False)
    profiler._bodyid_type_maps[DS_B] = {"12211": "aMe12"}

    profile = _profile(12211, DS_B)
    assert profile.neuron_type is None
    profiler._fill_neuron_type(profile, DS_B)
    assert profile.neuron_type == "aMe12"

    # Type-queried profiles pass through untouched.
    named = _profile(12211, DS_B)
    named.neuron_type = "Mi1"
    profiler._fill_neuron_type(named, DS_B)
    assert named.neuron_type == "Mi1"

    # Unknown bodyId: no crash, stays None.
    unknown = _profile(999999, DS_B)
    profiler._fill_neuron_type(unknown, DS_B)
    assert unknown.neuron_type is None


# ------------------------------------------------- same_label_only filter
def test_direct_comparison_same_label_only_uses_resolved_type(monkeypatch):
    """The Cross-Dataset tab's same-type filter compares the RESOLVED type
    name (same_label_only=True). BodyId profiles carry their type only via
    the profiler backfill; before it existed the filter compared the raw
    per-dataset keys ('12211' vs '5813058431') and silently dropped every
    pair. Hermetic: a fake profiler + a stubbed FindNeuronConnection keep
    the prewarm offline.
    """
    import coana
    from comparison.profile_comparator import ProfileComparator

    class _DummyFNC:
        def __init__(self, *a, **k):
            pass

        def build_connection_cache(self, *a, **k):
            return True

        _conn_df_cache = None

    monkeypatch.setattr(coana, "FindNeuronConnection", _DummyFNC)

    class _FakeProfiler:
        token = None

        def __init__(self, typed):
            self.typed = typed

        def get_profile(self, neuron, dataset):
            profile = _profile(int(neuron), dataset)
            if self.typed:
                profile.neuron_type = "aMe12"
            return profile

        def consolidate_profile_cache(self, dataset):
            pass

        def _load_cache_dataframe(self, dataset):
            pass

        def _get_cached_conn_df(self, dataset):
            pass

    kwargs = dict(
        neurons_a=["12211"], neurons_b=["5813058431"],
        dataset_a=DS_B, dataset_b="hemibrain:v1.2.1",
        direction="both", same_label_only=True, verbose=False,
    )
    fixed = ProfileComparator.direct_comparison(
        profiler=_FakeProfiler(typed=True), **kwargs)
    old = ProfileComparator.direct_comparison(
        profiler=_FakeProfiler(typed=False), **kwargs)

    # Fixed: the pair is retained and typed by its resolved name.
    assert len(fixed["results"]) == 1
    row = fixed["results"].iloc[0]
    assert row["type_a"] == "aMe12" and row["type_b"] == "aMe12"
    assert bool(row["is_same_type"]) is True
    # Old behavior (no backfill): every pair silently dropped.
    assert len(old["results"]) == 0
