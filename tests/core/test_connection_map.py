#!/usr/bin/env python
"""
Tests for the ThresholdedConnectionMap (D_t) model and the unified
type-level probability aggregation.

Covers:
  - Per-cutoff aggregates: each map owns the totals of its own thresholded
    graph (D_t = { edges with weight >= t }), computed once and cached
  - Source-signature invalidation: a rebuilt/replaced cache discards stale maps
  - Type-filter consistency: _apply_type_level_filters thresholds EDGES first
    so numerator and denominator come from the same D_t
  - Aggregate-method parity: 'product' / 'average' / 'ratio' produce the same
    type-level traversal_probability in the pandas and Polars engines
  - Compound product semantics: 1 - prod(1 - p_pair) over deduplicated pairs
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))

from coana import FindNeuronConnection  # noqa: E402
from connection_map import ThresholdedConnectionMap  # noqa: E402
from statvis import EnrichConnectionTable, EnrichConnectionTablePolars  # noqa: E402


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


def _write_cache(tmp_path):
    """Small deterministic cache: neurons N0..N5 (types T0..T2), edges below."""
    conn_path = tmp_path / "connections.parquet"
    index_path = tmp_path / "neuron_index.parquet"

    idx = pl.DataFrame(
        {
            "bodyId": ["N0", "N1", "N2", "N3", "N4", "N5"],
            "type": ["T0", "T1", "T0", "T2", "T2", "T1"],
        }
    )
    idx.write_parquet(index_path)

    conn = pl.DataFrame(
        {
            "bodyId_pre": ["N0", "N1", "N2", "N0", "N3", "N4", "N5"],
            "bodyId_post": ["N3", "N3", "N3", "N4", "N5", "N5", "N0"],
            "weight": [5, 2, 1, 4, 3, 6, 2],
        }
    )
    conn.write_parquet(conn_path)
    return conn_path, index_path


def _reference_totals(conn_path, index_path, min_weight):
    """Reference: filter edges by weight >= t, then aggregate."""
    conn = pl.read_parquet(conn_path).filter(pl.col("weight") >= min_weight)
    by_bodyid = conn.group_by("bodyId_post").agg(
        pl.col("weight").sum().alias("total_incoming_weight")
    )
    idx = pl.read_parquet(index_path, columns=["bodyId", "type"])
    by_type = (
        conn.join(idx, left_on="bodyId_post", right_on="bodyId", how="inner")
        .group_by("type")
        .agg(pl.col("weight").sum().alias("total_incoming_weight"))
        .rename({"type": "type_post"})
    )
    return by_bodyid, by_type


# =============================================================================
# ThresholdedConnectionMap: per-cutoff aggregates + caching
# =============================================================================


class TestThresholdedConnectionMap:
    def test_per_cutoff_aggregates(self, tmp_path):
        conn_path, index_path = _write_cache(tmp_path)

        map_t1 = ThresholdedConnectionMap(str(conn_path), str(index_path), min_weight=1)
        map_t3 = ThresholdedConnectionMap(str(conn_path), str(index_path), min_weight=3)

        # The two cutoffs describe different graphs -> different totals
        b1, t1 = map_t1.total_incoming_by_bodyid(), map_t1.total_incoming_by_type()
        b3, t3 = map_t3.total_incoming_by_bodyid(), map_t3.total_incoming_by_type()
        assert not b1.equals(b3)
        assert not t1.equals(t3)

        # Each matches a direct reference computation on its own D_t
        ref_b1, ref_t1 = _reference_totals(conn_path, index_path, 1)
        ref_b3, ref_t3 = _reference_totals(conn_path, index_path, 3)
        assert b1.sort("bodyId_post").equals(ref_b1.sort("bodyId_post"))
        assert t1.sort("type_post").equals(ref_t1.sort("type_post"))
        assert b3.sort("bodyId_post").equals(ref_b3.sort("bodyId_post"))
        assert t3.sort("type_post").equals(ref_t3.sort("type_post"))

        # N3 receives 5+2+1 = 8 at t=1 but only 5 at t=3 (the weight-1 and
        # weight-2 edges are below the cutoff)
        n3_t1 = b1.filter(pl.col("bodyId_post") == "N3")["total_incoming_weight"][0]
        n3_t3 = b3.filter(pl.col("bodyId_post") == "N3")["total_incoming_weight"][0]
        assert n3_t1 == 8
        assert n3_t3 == 5

    def test_aggregates_cached_and_shared(self, tmp_path):
        conn_path, index_path = _write_cache(tmp_path)
        cm = ThresholdedConnectionMap(str(conn_path), str(index_path), min_weight=2)
        first = cm.total_incoming_by_bodyid()
        second = cm.total_incoming_by_bodyid()
        assert first is second  # cached object, not recomputed
        # Both aggregate tables come from the same D_t (single map)
        assert cm.total_incoming_by_type() is not None

    def test_in_memory_frame_supported(self):
        frame = pd.DataFrame(
            {
                "bodyId_pre": ["A", "B"],
                "bodyId_post": ["X", "X"],
                "weight": [3, 1],
            }
        )
        cm = ThresholdedConnectionMap(
            db_path="", neuron_index_path="", min_weight=2, conn_frame=frame
        )
        # Edge with weight 1 is excluded from D_2
        totals = cm.total_incoming_by_bodyid().to_dict(as_series=False)
        assert totals == {"bodyId_post": ["X"], "total_incoming_weight": [3]}

    def test_batch_only_cache_and_mixed_id_dtypes_are_read(self, tmp_path):
        """A resumable batch cache remains queryable before consolidation."""
        index_path = tmp_path / "neuron_index.parquet"
        pl.DataFrame({
            "bodyId": ["1", "2", "3"],
            "type": ["T1", "T2", "T1"],
        }).write_parquet(index_path)

        cache_dir = tmp_path / "cache"
        batch_dir = cache_dir / "_batch_files"
        batch_dir.mkdir(parents=True)
        # Simulate an old batch with numeric IDs and a current batch with
        # string IDs.  The loader must normalize both before concatenating.
        pl.DataFrame({
            "bodyId_pre": [1, 2],
            "bodyId_post": [2, 3],
            "weight": [4, 5],
        }).write_parquet(batch_dir / "batch_000000.parquet")
        pl.DataFrame({
            "bodyId_pre": ["1", "2"],
            "bodyId_post": ["2", "3"],
            "weight": [4, 6],
            "roi": ["AL", "MB"],
            "cached_date": ["today", "today"],
        }).write_parquet(batch_dir / "batch_000001.parquet")

        cm = ThresholdedConnectionMap(
            str(cache_dir / "connections.parquet"),
            str(index_path),
            min_weight=1,
        )
        body = cm.total_incoming_by_bodyid().sort("bodyId_post")
        assert body.to_dict(as_series=False) == {
            "bodyId_post": ["2", "3"],
            "total_incoming_weight": [8, 11],
        }


class TestConnectionCacheLoader:
    def test_check_cache_accepts_pending_batch_cache(self, tmp_path):
        import coana

        dataset = "pending:v1"
        cache_dir = tmp_path / "cache" / "pending_v1" / "_batch_files"
        cache_dir.mkdir(parents=True)
        pl.DataFrame({
            "bodyId_pre": [1],
            "bodyId_post": [2],
            "weight": [3],
        }).write_parquet(cache_dir / "batch_000000.parquet")

        index_path = tmp_path / "neuron_indexes" / "pending_v1" / "neuron_index.parquet"
        index_path.parent.mkdir(parents=True)
        pl.DataFrame({"bodyId": ["1"], "type": ["T"]}).write_parquet(index_path)

        fc = object.__new__(coana.FindNeuronConnection)
        fc.dataset = dataset
        fc.script_path = str(tmp_path)
        status = fc._check_cache_exists()

        # Resumable batch files count as connection data and feed the
        # row counts even without a merged connections.parquet.
        assert status["has_connections"] is True
        assert status["has_neuron_index"] is True
        assert status["connection_count"] == 1
        # is_usable additionally requires the dataset neuron table
        # (cache-only runs crash later in type resolution without it):
        # this fixture has none, so the gate stays closed.
        assert status["has_dataset"] is False
        assert status["is_usable"] is False

    def test_load_normalizes_mixed_main_and_batch_files(self, tmp_path):
        import coana

        db_path = tmp_path / "connections.parquet"
        batch_dir = tmp_path / "_batch_files"
        batch_dir.mkdir()
        # Legacy main cache: numeric IDs and no optional columns.
        pl.DataFrame({
            "bodyId_pre": [1, 2],
            "bodyId_post": [2, 3],
            "weight": [4, 5],
        }).write_parquet(db_path)
        # New batch cache: string IDs and optional columns.  The first row is
        # a duplicate of the legacy row and should not be double-counted.
        pl.DataFrame({
            "bodyId_pre": ["1", "2"],
            "bodyId_post": ["2", "3"],
            "weight": [4, 6],
            "roi": ["", "MB"],
            "cached_date": ["today", "today"],
        }).write_parquet(batch_dir / "batch_000000.parquet")

        dataset_key = f"test_cache_loader_{tmp_path.name}"
        fc = object.__new__(coana.FindNeuronConnection)
        fc._conn_df_cache = None
        fc._conn_index = None
        fc._conn_index_post = None
        fc._conn_db_pre_id_cache = None
        fc._dataset_safe = dataset_key
        fc._get_connection_db_path = lambda: str(db_path)
        fc._vprint = lambda *args, **kwargs: None
        try:
            out = fc._load_connection_db()
            assert out.schema == {
                "bodyId_pre": pl.String,
                "bodyId_post": pl.String,
                "weight": pl.Int64,
                "roi": pl.String,
                "cached_date": pl.String,
            }
            assert out.height == 3
            assert out.filter(
                (pl.col("bodyId_pre") == "2")
                & (pl.col("bodyId_post") == "3")
            )["weight"].sort().to_list() == [5, 6]
        finally:
            coana._FNC_CACHE.pop(dataset_key, None)


class TestConnectionMapInvalidation:
    def test_source_signature_rebuilds_map(self, tmp_path):
        conn_path, index_path = _write_cache(tmp_path)
        obj = _stub_fnc(_get_connection_db_path=lambda: str(conn_path))

        m1 = obj._connection_map(min_weight=2)
        assert obj._connection_map(min_weight=2) is m1  # cached

        # Rebuilding the cache (bumped mtimes) must discard the stale map
        old = os.path.getmtime(conn_path)
        os.utime(conn_path, times=(old + 5, old + 5))
        m2 = obj._connection_map(min_weight=2)
        assert m2 is not m1

        # Different cutoffs live in separate map entries
        assert obj._connection_map(min_weight=1) is not m2
        assert len(obj._connection_maps) == 2

    def test_both_tables_come_from_same_map(self, tmp_path):
        conn_path, index_path = _write_cache(tmp_path)
        obj = _stub_fnc(
            _get_connection_db_path=lambda: str(conn_path),
            _get_neuron_index_path=lambda: str(index_path),
        )
        by_type = obj._get_total_incoming_by_type_table(min_weight=3)
        by_bodyid = obj._get_total_incoming_by_bodyid_table(min_weight=3)
        assert len(obj._connection_maps) == 1  # one D_3 map serves both tables
        # Cross-check: type totals equal per-neuron sums of the bodyId table
        # for typed neurons (N3->T2: 5, N4->T2: 4 at t=3)
        t2 = by_type.filter(pl.col("type_post") == "T2")["total_incoming_weight"][0]
        assert t2 == 5 + 4


# =============================================================================
# Type-filter consistency: numerator and denominator from the same D_t
# =============================================================================


class TestTypeFilterConsistency:
    def _obj(self):
        obj = _stub_fnc()
        obj._fetch_total_incoming_weight_by_type = lambda post_types, min_weight: (
            pd.DataFrame(
                {"type_post": ["B1", "B2"], "total_incoming_weight": [100, 200]}
            )
        )
        return obj

    def test_edges_thresholded_before_pair_aggregation(self):
        obj = self._obj()
        combined = pd.DataFrame(
            {
                "bodyId_pre": ["1", "2", "3", "4", "5"],
                "bodyId_post": ["10", "10", "20", "30", "30"],
                "type_pre": ["A", "A", "C", "D", "D"],
                "type_post": ["B1", "B1", "B2", "B2", "B2"],
                "weight": [10, 4, 8, 2, 3],
            }
        )
        out = obj._apply_type_level_filters(
            combined,
            min_weight=3,
            min_conn_ratio=0.0,
            min_traversal_prob=0.0,
            total_before_filter=5,
        )
        # D_3 = {edges >= 3}: the weight-2 edge (D->B2) is below the cutoff
        # and must NOT survive by summing with the weight-3 edge of the same
        # pair (old bug: pair sum 2+3=5 >= 3 kept it; numerator from a
        # different graph than the edge-thresholded denominator).
        assert set(out["bodyId_pre"]) == {"1", "2", "3", "5"}

    def test_product_aggregation_matches_statvis_engine(self):
        obj = self._obj()
        obj._fetch_total_incoming_weight = lambda post_ids, min_weight: (
            pd.DataFrame(
                {
                    "bodyId_post": ["10", "20", "30"],
                    "total_incoming_weight": [100, 200, 300],
                }
            )
        )
        combined = pd.DataFrame(
            {
                "bodyId_pre": ["1", "2", "3", "4", "5", "6"],
                "bodyId_post": ["10", "10", "10", "20", "20", "30"],
                "type_pre": ["A", "A", "A", "C", "C", "D"],
                "type_post": ["B1", "B1", "B1", "B2", "B2", "B2"],
                "weight": [10, 4, 6, 8, 2, 3],
            }
        )
        # min_traversal_prob=0.5: with the compound product model only
        # A->B1 (1 - (1-0.333)(1-0.133)(1-0.2) = 0.538) passes; C->B2 and
        # D->B2 stay below 0.5. Under the legacy ratio model A->B1 would be
        # 0.667 (pass) and C->B2 0.167 (fail) - the same pair selection here,
        # but the product values are what the enriched conn_type reports, so
        # filter and output agree.
        out = obj._apply_type_level_filters(
            combined,
            min_weight=1,
            min_conn_ratio=0.0,
            min_traversal_prob=0.5,
            total_before_filter=6,
        )
        assert set(out["bodyId_pre"]) == {"1", "2", "3"}  # only A->B1 passes

        # Same aggregation through EnrichConnectionTable (pandas engine)
        # with matching global weights must agree with the filter.
        global_type = pd.DataFrame(
            {"type_post": ["B1", "B2"], "total_incoming_weight": [100, 200]}
        )
        global_body = pd.DataFrame(
            {
                "bodyId_post": ["10", "20", "30"],
                "total_incoming_weight": [100, 200, 300],
            }
        )
        _, conn_type, _ = EnrichConnectionTable(
            combined,
            global_incoming_weights=global_type,
            global_incoming_body_weights=global_body,
        )
        ab = conn_type[conn_type["type_post"] == "B1"].iloc[0]
        assert ab["traversal_probability"] == pytest.approx(
            1 - (1 - 10 / 100 / 0.3) * (1 - 4 / 100 / 0.3) * (1 - 6 / 100 / 0.3)
        )
        assert ab["traversal_probability"] > 0.5  # passes the filter threshold
        cb = conn_type[conn_type["type_post"] == "B2"].iloc[0]
        assert cb["traversal_probability"] < 0.5  # filtered out above


# =============================================================================
# Engine parity: pandas vs Polars across aggregate methods
# =============================================================================


class TestEngineParity:
    """EnrichConnectionTable (pandas) and EnrichConnectionTablePolars must
    emit the same type-level traversal_probability for every method."""

    @staticmethod
    def _input():
        # One row per bodyId pair (the cache guarantees unique pairs)
        return pd.DataFrame(
            {
                "bodyId_pre": ["A1", "A2", "A2", "B1"],
                "bodyId_post": ["C1", "C1", "C2", "C2"],
                "type_pre": ["T1", "T1", "T1", "T2"],
                "type_post": ["T3", "T3", "T3", "T3"],
                "weight": [5, 3, 2, 4],
            }
        )

    @staticmethod
    def _globals():
        global_type = pd.DataFrame(
            {"type_post": ["T3"], "total_incoming_weight": [100]}
        )
        global_body = pd.DataFrame(
            {
                "bodyId_post": ["C1", "C2"],
                "total_incoming_weight": [20, 10],
            }
        )
        return global_type, global_body

    @pytest.mark.parametrize("method", ["product", "average", "ratio"])
    def test_type_level_probability_parity(self, method):
        global_type, global_body = self._globals()
        _, pd_type, _ = EnrichConnectionTable(
            self._input(),
            aggregate_method=method,
            global_incoming_weights=global_type,
            global_incoming_body_weights=global_body,
        )
        _, pl_type, _ = EnrichConnectionTablePolars(
            self._input(),
            aggregate_method=method,
            global_incoming_weights=global_type,
            global_incoming_body_weights=global_body,
        )
        pd_rows = pd_type.set_index(["type_pre", "type_post"])
        pl_rows = {
            (r["type_pre"], r["type_post"]): r
            for r in pl_type.iter_rows(named=True)
        }
        for (pre, post), prow in pd_rows.iterrows():
            rrow = pl_rows[(pre, post)]
            assert rrow["traversal_probability"] == pytest.approx(
                prow["traversal_probability"], abs=1e-12
            )
            assert rrow["block_probability"] == pytest.approx(
                prow["block_probability"], abs=1e-12
            )
            assert rrow["connection_ratio"] == pytest.approx(
                prow["connection_ratio"], abs=1e-12
            )

    def test_product_compounds_deduplicated_pairs(self):
        """Default method: 1 - prod(1 - p_pair) over the deduplicated pairs.

        Pair probabilities (global bodyId denominators, capped at 1.0):
          A1->C1: 5/20 / 0.3 = 0.8333     A2->C1: 3/20 / 0.3 = 0.5
          A2->C2: 2/10 / 0.3 = 0.6667     B1->C2: 4/10 / 0.3 = 1.0 (capped)
        """
        global_type, global_body = self._globals()
        _, conn_type, _ = EnrichConnectionTable(
            self._input(),
            global_incoming_weights=global_type,
            global_incoming_body_weights=global_body,
        )
        rows = conn_type.set_index(["type_pre", "type_post"])
        t1t3 = rows.loc[("T1", "T3")]
        assert t1t3["traversal_probability"] == pytest.approx(
            1 - (1 - 0.8333333333) * (1 - 0.5) * (1 - 0.6666666667), abs=1e-9
        )
        t2t3 = rows.loc[("T2", "T3")]
        assert t2t3["traversal_probability"] == pytest.approx(1.0)  # capped pair

    def test_all_null_type_post_is_joinable_with_global_type_totals(self):
        """Untyped post neurons must fall back to their bodyId without a
        Polars Null-vs-String join error during type aggregation."""
        conn = pl.DataFrame({
            "bodyId_pre": ["A"],
            "bodyId_post": ["U"],
            "type_pre": ["T1"],
            "type_post": [None],
            "weight": [3],
        })
        global_type = pl.DataFrame({
            "type_post": ["T2"],
            "total_incoming_weight": [100],
        })
        _, conn_type, _ = EnrichConnectionTablePolars(
            conn,
            global_incoming_weights=global_type,
        )
        assert conn_type.height == 1
        assert conn_type["type_post"].to_list() == ["U"]

    def test_average_is_weight_weighted_mean(self):
        global_type, global_body = self._globals()
        _, conn_type, _ = EnrichConnectionTable(
            self._input(),
            aggregate_method="average",
            global_incoming_weights=global_type,
            global_incoming_body_weights=global_body,
        )
        rows = conn_type.set_index(["type_pre", "type_post"])
        t1t3 = rows.loc[("T1", "T3")]
        expected = (5 * 0.8333333333 + 3 * 0.5 + 2 * 0.6666666667) / 10
        assert t1t3["traversal_probability"] == pytest.approx(expected, abs=1e-9)

    def test_ratio_matches_legacy_model(self):
        global_type, global_body = self._globals()
        _, conn_type, _ = EnrichConnectionTable(
            self._input(),
            aggregate_method="ratio",
            global_incoming_weights=global_type,
            global_incoming_body_weights=global_body,
        )
        rows = conn_type.set_index(["type_pre", "type_post"])
        t1t3 = rows.loc[("T1", "T3")]
        assert t1t3["traversal_probability"] == pytest.approx((10 / 100) / 0.3)
        t2t3 = rows.loc[("T2", "T3")]
        assert t2t3["traversal_probability"] == pytest.approx((4 / 100) / 0.3)
