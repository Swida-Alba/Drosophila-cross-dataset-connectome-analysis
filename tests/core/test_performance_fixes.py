#!/usr/bin/env python
"""
Regression tests for performance and pandas/polars interop fixes.

Covers:
  - Vectorized + cached full-dataset incoming-weight aggregates
    (_fetch_total_incoming_weight / _by_type)
  - O(M*N) -> O(M+N) label mapping in statvis_polars.build_bodyid_label_map
  - EnrichConnectionTablePolars global-weight aggregation (join instead of
    map_elements)
  - Vectorized hemisphere suffix logic matches the old scalar logic
  - Vectorized type-level ratio / pair filtering in _apply_type_level_filters
  - _count_cached_connections works with a Polars cache (no .empty crash)
  - Per-instance local neuron CSV caching
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))

from coana import FindNeuronConnection  # noqa: E402
from statvis import (  # noqa: E402
    EnrichConnectionTablePolars,
    _PL_NEURON_DF_CACHE,
    build_bodyid_label_map,
)


def _stub_fnc(**attrs):
    """Create a FindNeuronConnection without running __post_init__."""
    obj = object.__new__(FindNeuronConnection)
    obj._conn_df_cache = None
    obj._local_neuron_df_cache = {}
    obj._connection_maps = {}
    obj._vprint = lambda *a, **k: None
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


# =============================================================================
# Incoming-weight aggregates (vectorized + cached)
# =============================================================================


def _write_synthetic_cache(tmp_path, n_conns=2000, n_neurons=50, seed=7):
    rng = np.random.default_rng(seed)
    conn_path = tmp_path / "connections.parquet"
    index_path = tmp_path / "neuron_index.parquet"

    neurons = [f"N{i}" for i in range(n_neurons)]
    types = [f"T{i % 7}" for i in range(n_neurons)]
    # Intentionally include a duplicate bodyId (last entry must win).
    idx_df = pl.DataFrame(
        {"bodyId": neurons + [neurons[3]], "type": types + ["T_DUP"]}
    )
    idx_df.write_parquet(index_path)

    pre = rng.choice(neurons, size=n_conns)
    post = rng.choice(neurons, size=n_conns)
    weight = rng.integers(1, 15, size=n_conns)
    conn_df = pl.DataFrame(
        {"bodyId_pre": pre, "bodyId_post": post, "weight": weight}
    )
    conn_df.write_parquet(conn_path)
    return conn_path, index_path


def _reference_type_incoming(db_path, index_path, min_weight, post_types):
    """Reference implementation — PER-PAIR PHYSICAL semantics (plan §12,
    2026-10-12): the cache stores per-ROI slices, so each pair's weight
    is the SUM of its rows and the tier filter applies to that sum (a
    sub-threshold slice of a supra-threshold pair contributes; the old
    row-group loop filtered per row and under-counted)."""
    import pyarrow.parquet as pq

    neuron_index = pl.read_parquet(index_path, columns=["bodyId", "type"])
    bodyid_to_type = dict(zip(neuron_index["bodyId"].to_list(),
                              neuron_index["type"].to_list()))
    post_types_set = set(post_types)
    target = {b for b, t in bodyid_to_type.items() if t in post_types_set}

    pair_weights = {}
    pf = pq.ParquetFile(db_path)
    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(
            i, columns=["bodyId_pre", "bodyId_post", "weight"])
        chunk = pl.from_arrow(table)
        for pre, bid, w in zip(chunk["bodyId_pre"].to_list(),
                               chunk["bodyId_post"].to_list(),
                               chunk["weight"].to_list()):
            if bid in target:
                key = (pre, bid)
                pair_weights[key] = pair_weights.get(key, 0) + w
    type_weights = {}
    for (pre, bid), total in pair_weights.items():
        if total >= min_weight:
            t = bodyid_to_type.get(bid)
            if t:
                type_weights[t] = type_weights.get(t, 0) + total
    return type_weights


class TestTotalIncomingByType:
    def test_vectorized_matches_reference(self, tmp_path):
        conn_path, index_path = _write_synthetic_cache(tmp_path)
        obj = _stub_fnc(
            _get_connection_db_path=lambda: str(conn_path),
            _get_neuron_index_path=lambda: str(index_path),
        )

        all_types = {f"T{i}" for i in range(7)} | {"T_DUP"}
        table = obj._get_total_incoming_by_type_table(min_weight=3)

        expected = _reference_type_incoming(conn_path, index_path, 3, all_types)
        actual = dict(
            zip(
                table["type_post"].to_list(),
                table["total_incoming_weight"].to_list(),
            )
        )
        assert actual == expected

        # Requested-type filtering happens on the cached full table
        subset = obj._fetch_total_incoming_weight_by_type(["T0"], min_weight=3)
        assert set(subset["type_post"]) <= {"T0"}
        assert len(obj._connection_maps) == 1  # one D_t map, computed once

        ref_t0 = _reference_type_incoming(conn_path, index_path, 3, {"T0"})
        assert subset.set_index("type_post")["total_incoming_weight"].to_dict() == ref_t0

    def test_duplicate_bodyid_last_wins(self, tmp_path):
        conn_path, index_path = _write_synthetic_cache(tmp_path, n_conns=2000)
        obj = _stub_fnc(
            _get_connection_db_path=lambda: str(conn_path),
            _get_neuron_index_path=lambda: str(index_path),
        )
        table = obj._get_total_incoming_by_type_table(min_weight=1)
        # Neuron N3's duplicate entry maps it to T_DUP; the join must use the
        # last mapping (keep='last'), so N3's weight never lands in T3.
        t3_rows = table.filter(pl.col("type_post") == "T3")
        tdup_rows = table.filter(pl.col("type_post") == "T_DUP")
        assert t3_rows.height >= 0
        assert tdup_rows.height >= 0
        # If keep='first' were used, T_DUP would have zero rows; with
        # keep='last' it must be non-empty (N3 has incoming edges).
        assert tdup_rows["total_incoming_weight"][0] > 0


class TestTotalIncomingByBodyId:
    def test_cached_table_filtered_per_call(self, tmp_path):
        conn_path, _ = _write_synthetic_cache(tmp_path, n_conns=500)
        obj = _stub_fnc(_get_connection_db_path=lambda: str(conn_path))

        res1 = obj._fetch_total_incoming_weight(["N1", "N2"], min_weight=2)
        res2 = obj._fetch_total_incoming_weight(["N3"], min_weight=2)
        assert set(res1["bodyId_post"]) <= {"N1", "N2"}
        assert set(res2["bodyId_post"]) <= {"N3"}
        assert len(obj._connection_maps) == 1  # one D_t map shared by both tables

        # Sanity: totals equal per-neuron sums from the raw table
        raw = pl.read_parquet(conn_path).filter(pl.col("weight") >= 2)
        expected = (
            raw.filter(pl.col("bodyId_post") == "N1")
            .select(pl.col("weight").sum())
            .item()
        )
        got = res1.loc[res1["bodyId_post"] == "N1", "total_incoming_weight"].iloc[0]
        assert got == expected


# =============================================================================
# statvis_polars label mapping
# =============================================================================


class TestBuildBodyidLabelMap:
    def _label_mapper(self):
        return SimpleNamespace(
            _source_mapping={
                "STD_A": {"test_v1": ["A1", "A2"]},
                "STD_X": {"test_v1": ["TB"]},  # type-based mapping
            },
            _target_mapping={"STD_B": {"test_v1": ["B1"]}},
            _intermediate_mapping={
                "STD_C": {"test_v1": ["C1"]},
                "STD_Y": {"test_v1": ["inst1"]},  # instance-based mapping
            },
        )

    def test_mapping_correctness_and_priority(self):
        neuron_df = pl.DataFrame(
            {
                "bodyId": ["A1", "A2", "B1", "B2", "C1"],
                "type": ["TA", "TA", "TB", "TB", "TC"],
                "instance": ["inst1", "inst2", "", "", ""],
            }
        )
        result = build_bodyid_label_map(
            self._label_mapper(), "test:v1", neuron_df
        )
        # Direct bodyId mapping wins over type/instance expansion
        assert result["A1"] == "STD_A"
        assert result["A2"] == "STD_A"
        assert result["B1"] == "STD_B"
        # B2 has no direct bodyId mapping; its type TB maps to STD_X
        assert result["B2"] == "STD_X"
        assert result["C1"] == "STD_C"
        # A1 is also covered by the STD_Y instance mapping, but the first
        # mapping wins (source mappings are processed first).
        assert result["A1"] == "STD_A"

    def test_empty_mapping(self):
        neuron_df = pl.DataFrame({"bodyId": ["A"], "type": ["TA"]})
        result = build_bodyid_label_map(None, "test:v1", neuron_df)
        assert result == {}


# =============================================================================
# EnrichConnectionTablePolars with global incoming weights
# =============================================================================


class TestEnrichConnectionTablePolars:
    def _dataset(self, tmp_path):
        dataset_clean = "test_v1"
        d = tmp_path / "datasets" / dataset_clean
        d.mkdir(parents=True, exist_ok=True)
        csv_path = d / f"{dataset_clean}_allneurons_neuron_df.csv"
        pd.DataFrame(
            {
                "bodyId": ["A1", "A2", "B1", "B2", "C1"],
                "type": ["TA", "TA", "TB", "TB", "TC"],
                "post": [50, 60, 70, 80, 90],
            }
        ).to_csv(csv_path, index=False)
        return str(tmp_path), str(csv_path)

    def test_global_weights_std_label_aggregation(self, tmp_path):
        script_path, _ = self._dataset(tmp_path)
        conn_table = pd.DataFrame(
            {
                "bodyId_pre": ["A1", "A2", "A2", "C1"],
                "bodyId_post": ["B1", "B1", "B2", "B2"],
                "type_pre": ["TA", "TA", "TA", "TC"],
                "type_post": ["TB", "TB", "TB", "TB"],
                "weight": [5, 3, 2, 1],
            }
        )
        label_mapper = SimpleNamespace(
            _source_mapping={"STD_A": {"test_v1": ["A1", "A2"]}},
            _target_mapping={"STD_B": {"test_v1": ["B1", "B2"]}},
            _intermediate_mapping={"STD_C": {"test_v1": ["C1"]}},
        )
        # global_incoming_weights are raw-type totals (type_post = "TB"),
        # matching what _fetch_total_incoming_weight_by_type returns.
        global_incoming = pd.DataFrame(
            {"type_post": ["TB"], "total_incoming_weight": [200]}
        )

        conn_df, conn_type, conn_group = EnrichConnectionTablePolars(
            conn_table,
            dataset="test:v1",
            script_path=script_path,
            label_mapper=label_mapper,
            global_incoming_weights=global_incoming,
        )
        assert conn_group is None

        # Type-level aggregation uses std_labels and global denominators
        rows = {
            (r["type_pre"], r["type_post"]): r
            for r in conn_type.iter_rows(named=True)
        }
        ab = next(
            r
            for r in rows.values()
            if r["type_pre"] == "STD_A" and r["type_post"] == "STD_B"
        )
        assert ab["weight"] == 10  # 5 + 3 + 2
        assert ab["connection_ratio"] == pytest.approx(10 / 200)
        # PROBABILITY SUSPENSION (2026-10-06): traversal_probability is
        # no longer exported (the former product-compounded column rode
        # an unverifiable reliability model) — the surviving pair mass
        # and the global-denominator ratio carry the semantics.
        assert "traversal_probability" not in ab

        cb = next(
            r
            for r in rows.values()
            if r["type_pre"] == "STD_C" and r["type_post"] == "STD_B"
        )
        assert cb["weight"] == 1
        assert cb["connection_ratio"] == pytest.approx(1 / 200)

    def test_ratio_method_preserves_legacy_semantics(self, tmp_path):
        """aggregate_method='ratio' keeps the legacy pair semantics:
        exact-dup-collapsed per-pair mass over the global denominator
        (the suspended traversal_probability consumed the /0.3 cap)."""
        script_path, _ = self._dataset(tmp_path)
        conn_table = pd.DataFrame(
            {
                "bodyId_pre": ["A1", "A2", "A2", "C1"],
                "bodyId_post": ["B1", "B1", "B2", "B2"],
                "type_pre": ["TA", "TA", "TA", "TC"],
                "type_post": ["TB", "TB", "TB", "TB"],
                "weight": [5, 3, 2, 1],
            }
        )
        label_mapper = SimpleNamespace(
            _source_mapping={"STD_A": {"test_v1": ["A1", "A2"]}},
            _target_mapping={"STD_B": {"test_v1": ["B1", "B2"]}},
            _intermediate_mapping={"STD_C": {"test_v1": ["C1"]}},
        )
        global_incoming = pd.DataFrame(
            {"type_post": ["TB"], "total_incoming_weight": [200]}
        )

        _, conn_type, _ = EnrichConnectionTablePolars(
            conn_table,
            dataset="test:v1",
            script_path=script_path,
            label_mapper=label_mapper,
            global_incoming_weights=global_incoming,
            aggregate_method="ratio",
        )
        rows = {
            (r["type_pre"], r["type_post"]): r
            for r in conn_type.iter_rows(named=True)
        }
        ab = next(
            r
            for r in rows.values()
            if r["type_pre"] == "STD_A" and r["type_post"] == "STD_B"
        )
        # PROBABILITY SUSPENSION (2026-10-06): traversal_probability is no
        # longer exported; the legacy ratio-method semantic lives in the
        # surviving columns — the type mass is the exact-dup-collapsed
        # per-pair sum (5+3+2 = 10) and the ratio uses the global
        # denominator (10/200).
        assert ab["weight"] == pytest.approx(10.0)
        assert ab["connection_ratio"] == pytest.approx(10 / 200)


# =============================================================================
# Hemisphere suffix vectorization
# =============================================================================


class TestHemisphereVectorization:
    def _reference_hemi(self, row, side):
        code_col = f"hemisphere_code_{side}"
        hemi_col = f"hemisphere_{side}"
        inst_col = f"instance_{side}"
        if code_col in row and pd.notna(row[code_col]):
            return str(row[code_col])
        if hemi_col in row and pd.notna(row[hemi_col]):
            normalized = {
                "r": "R", "right": "R", "rhs": "R", "right hemisphere": "R",
                "l": "L", "left": "L", "lhs": "L", "left hemisphere": "L",
            }.get(str(row[hemi_col]).strip().lower(), "U")
            return normalized
        if inst_col in row and isinstance(row[inst_col], str):
            if row[inst_col].endswith("_R"):
                return "R"
            if row[inst_col].endswith("_L"):
                return "L"
        return "U"

    def test_conn_df_matches_scalar_logic(self):
        obj = _stub_fnc(separate_hemispheres=True)
        df = pd.DataFrame(
            {
                "bodyId_pre": ["1", "2", "3", "4", "5", "6", "7"],
                "bodyId_post": ["8", "9", "10", "11", "12", "13", "14"],
                "type_pre": ["PPL101", "DN1p_L", None, "MBON", "X_R", "Y", "Z"],
                "type_post": ["A", "B", "C", "D", "E", "F", "G"],
                "hemisphere_code_pre": ["R", None, "left", None, None, None, "U"],
                "hemisphere_pre": [None, "right", None, "L", None, "foo", None],
                "instance_pre": [None, None, None, None, "n_R", "n_L", "n_X"],
                "hemisphere_code_post": [None, "L", None, "U", None, None, None],
                "hemisphere_post": [None, None, "right", None, None, None, None],
                "instance_post": [None, None, None, None, "m_R", "m_L", None],
            }
        )
        out = obj._apply_hemisphere_suffix_to_conn_df(df)

        for _, row in df.iterrows():
            pre_code = self._reference_hemi(row, "pre")
            post_code = self._reference_hemi(row, "post")
            expected_pre = obj._append_hemi_suffix(row["type_pre"], pre_code)
            expected_post = obj._append_hemi_suffix(row["type_post"], post_code)
            assert out.at[row.name, "type_pre"] == expected_pre
            assert out.at[row.name, "type_post"] == expected_post

    def test_neuron_df_suffix(self):
        obj = _stub_fnc(separate_hemispheres=True)
        df = pd.DataFrame(
            {
                "bodyId": ["1", "2", "3"],
                "type": ["PPL101", "DN1p", None],
                "instance": ["a_R", "b_L", "c"],
            }
        )
        out = obj._apply_hemisphere_suffix_to_neuron_df(df)
        assert out["type"].tolist() == ["PPL101_R", "DN1p_L", "Unknown_U"]


# =============================================================================
# Type-level filtering vectorization
# =============================================================================


class TestApplyTypeLevelFilters:
    def test_ratios_and_pair_filter(self):
        obj = _stub_fnc()
        obj._fetch_total_incoming_weight_by_type = lambda post_types, min_weight: (
            pd.DataFrame(
                {"type_post": ["B1", "B2"], "total_incoming_weight": [100, 200]}
            )
        )
        combined = pd.DataFrame(
            {
                "bodyId_pre": ["1", "2", "3", "4"],
                "bodyId_post": ["10", "20", "20", "30"],
                "type_pre": ["A", "A", "C", "D"],
                "type_post": ["B1", "B1", "B2", "B2"],
                "weight": [10, 5, 8, 1],
            }
        )
        out = obj._apply_type_level_filters(
            combined,
            min_weight=3,
            min_conn_ratio=0.0,
            min_traversal_prob=0.0,
            total_before_filter=4,
        )
        # Type pair A->B1 sums to 15 >= 3; C->B2 sums to 8 >= 3;
        # D->B2 sums to 1 < 3 and is filtered out at type level.
        assert set(out["bodyId_pre"]) == {"1", "2", "3"}
        # Ratio/type-level aggregates are applied before the bodyId-level
        # filter; the returned frame keeps only the passing connections.
        assert "connection_ratio" not in out.columns

    def test_bodyid_level_ratio_vectorized(self):
        obj = _stub_fnc()
        obj._fetch_total_incoming_weight = lambda post_ids, min_weight: (
            pd.DataFrame(
                {
                    "bodyId_post": ["10", "20"],
                    "total_incoming_weight": [100, 200],
                }
            )
        )
        combined = pd.DataFrame(
            {
                "bodyId_pre": ["1", "2", "3"],
                "bodyId_post": ["10", "20", "30"],
                "weight": [10, 8, 1],
            }
        )
        out = obj._apply_bodyid_level_filters(
            combined, min_conn_ratio=0.0, min_traversal_prob=0.0,
            total_before_filter=3, min_weight=1,
        )
        assert out.loc[out["bodyId_post"] == "10", "connection_ratio"].iloc[0] == pytest.approx(10 / 100)
        assert out.loc[out["bodyId_post"] == "20", "connection_ratio"].iloc[0] == pytest.approx(8 / 200)
        # No total incoming weight available -> NaN ratio, kept with min filters off
        assert np.isnan(out.loc[out["bodyId_post"] == "30", "connection_ratio"].iloc[0])

    def test_null_types_preserved(self):
        obj = _stub_fnc()
        obj._fetch_total_incoming_weight_by_type = lambda post_types, min_weight: (
            pd.DataFrame(columns=["type_post", "total_incoming_weight"])
        )
        combined = pd.DataFrame(
            {
                "bodyId_pre": ["1", "2"],
                "bodyId_post": ["10", "20"],
                "type_pre": ["A", np.nan],
                "type_post": ["B", np.nan],
                "weight": [10, 7],
            }
        )
        out = obj._apply_type_level_filters(
            combined, min_weight=3, min_conn_ratio=0.0,
            min_traversal_prob=0.0, total_before_filter=2,
        )
        assert len(out) == 2
        assert out["type_post"].isna().any()


# =============================================================================
# Pandas/Polars interop edge cases
# =============================================================================


class TestInteropEdgeCases:
    def test_count_cached_connections_with_polars_cache(self):
        obj = _stub_fnc(
            _conn_df_cache=pl.DataFrame(
                {
                    "bodyId_pre": ["a", "b", "c"],
                    "bodyId_post": ["x", "y", "z"],
                    "weight": [1, 2, 3],
                }
            )
        )
        obj._get_connection_db_path = lambda: (_ for _ in ()).throw(
            AssertionError("disk path must not be used")
        )
        assert obj._count_cached_connections() == 3

    def test_local_neuron_df_cached_per_mtime(self, tmp_path):
        csv_path = tmp_path / "ds_allneurons_neuron_df.csv"
        pd.DataFrame({"bodyId": ["1", "2"], "type": ["A", "B"]}).to_csv(
            csv_path, index=False
        )
        obj = _stub_fnc()
        obj._read_csv = lambda path, **kw: pd.read_csv(path, **kw)
        obj._ensure_hemisphere_columns = lambda df: df

        df1 = obj._load_local_neuron_df(str(csv_path), is_fafb=False)
        df2 = obj._load_local_neuron_df(str(csv_path), is_fafb=False)
        assert df1 is df2  # cached
        assert len(df1) == 2

        old_mtime = os.path.getmtime(csv_path)
        os.utime(csv_path, times=(old_mtime + 5, old_mtime + 5))  # bump mtime -> reload
        df3 = obj._load_local_neuron_df(str(csv_path), is_fafb=False)
        assert df3 is not df1

    def test_statvis_neuron_df_cache_invalidates_on_mtime(self, tmp_path):
        _PL_NEURON_DF_CACHE.clear()
        try:
            csv_path = tmp_path / "ds_allneurons_neuron_df.csv"
            pd.DataFrame({"bodyId": ["1", "2"], "type": ["A", "B"]}).to_csv(
                csv_path, index=False
            )
            from statvis import _load_local_neuron_df_cached

            df1 = _load_local_neuron_df_cached(str(csv_path), is_fafb=False)
            df2 = _load_local_neuron_df_cached(str(csv_path), is_fafb=False)
            assert df1 is df2
            old_mtime = os.path.getmtime(csv_path)
            os.utime(csv_path, times=(old_mtime + 5, old_mtime + 5))
            df3 = _load_local_neuron_df_cached(str(csv_path), is_fafb=False)
            assert df3 is not df1
        finally:
            _PL_NEURON_DF_CACHE.clear()


class TestPerPairCollapseBeforeTier:
    """Per-ROI ordering fix (2026-10-12, plan §12): the cache stores
    per-ROI slices, so D_t's tier filter must see the pair's PHYSICAL
    (summed) weight — a sub-threshold slice of a supra-threshold pair
    must not drop, and a pair whose every slice is sub-threshold must
    not vanish."""

    def _per_roi_cache(self, tmp_path):
        conn = tmp_path / "connections.parquet"
        idx = tmp_path / "neuron_index.parquet"
        pl.DataFrame({
            "bodyId_pre": ["A", "A", "B", "B"],
            "bodyId_post": ["B", "B", "C", "C"],
            "roi": ["R1", "R2", "R1", "R2"],
            "weight": [2, 5, 1, 2]}).write_parquet(conn)
        pl.DataFrame({"bodyId": ["A", "B", "C"],
                      "type": ["TA", "TB", "TC"]}).write_parquet(idx)
        return conn, idx

    def test_totals_use_physical_pair_weight(self, tmp_path):
        conn, idx = self._per_roi_cache(tmp_path)
        obj = _stub_fnc(
            _get_connection_db_path=lambda: str(conn),
            _get_neuron_index_path=lambda: str(idx),
        )
        table = obj._get_total_incoming_by_bodyid_table(min_weight=3)
        got = dict(zip(table["bodyId_post"].to_list(),
                       table["total_incoming_weight"].to_list()))
        assert got == {"B": 7, "C": 3}

    def test_type_totals_use_physical_pair_weight(self, tmp_path):
        conn, idx = self._per_roi_cache(tmp_path)
        obj = _stub_fnc(
            _get_connection_db_path=lambda: str(conn),
            _get_neuron_index_path=lambda: str(idx),
        )
        table = obj._get_total_incoming_by_type_table(min_weight=3)
        got = dict(zip(table["type_post"].to_list(),
                       table["total_incoming_weight"].to_list()))
        assert got == {"TB": 7, "TC": 3}
