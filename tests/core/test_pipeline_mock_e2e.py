#!/usr/bin/env python
"""
End-to-end pipeline verification on hand-computed mock data.

Tracks one small dataset through every stage of the metric pipeline and
checks the OUTPUT of each stage against hand-computed values:

  mock cache (connections.parquet + neuron_index.parquet)
    -> ThresholdedConnectionMap aggregates per cutoff (D_1, D_3)
    -> EnrichConnectionTable (pandas AND Polars engines) with the map's
       global weights -> conn_type (product / average / ratio)
    -> path builder -> path_prob / min_ratio / min_weight

Dataset (all weights are synapse counts; types in parentheses):

    A1(T1)->X(T3): 3      A1(T1)->Y(T4): 7
    A2(T1)->X(T3): 4      A2(T1)->Y(T4): 3
    B (T2)->X(T3): 5      F (T3)->Y(T4): 5
    C (T2)->X(T3): 10     G (T3)->Y(T4): 5
    D (T2)->X(T3): 2      H (T3)->Y(T4): 4
    E (T1)->X(T3): 6

D_1: total_incoming(X) = 30, total_incoming(Y) = 24
D_3: total_incoming(X) = 28 (D->X 2 excluded), total_incoming(Y) = 24

Query at t=3 (sources A1, A2, B, D, H): A1->X 3, A2->X 4, B->X 5,
A1->Y 7, A2->Y 3, H->Y 4  (D->X 2 is below the cutoff)

Pair probabilities p = min((w / total_incoming(post)) / 0.3, 1):
    A1->X: (3/28)/0.3 = 0.35714    A1->Y: (7/24)/0.3 = 0.97222
    A2->X: (4/28)/0.3 = 0.47619    A2->Y: (3/24)/0.3 = 0.41667
    B->X:  (5/28)/0.3 = 0.59524    H->Y:  (4/24)/0.3 = 0.55556

Type-level traversal_probability per aggregate method:
    product:  T1->T3: 1-(1-.35714)(1-.47619) = 0.66327   T1->T4: 1-(1-.97222)(1-.41667) = 0.98380
              T2->T3: 0.59524                            T3->T4: 0.55556
    average:  T1->T3: (3*.35714+4*.47619)/7 = 0.42517    T1->T4: (7*.97222+3*.41667)/10 = 0.80556
    ratio:    T1->T3: (7/28)/0.3 = 0.83333               T1->T4: (10/24)/0.3 = 1.0 (capped)
"""

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
from statvis import (  # noqa: E402
    EnrichConnectionTable,
    EnrichConnectionTablePolars,
    build_path_dataframe_from_paths as sv_build,
    build_path_dataframe_from_paths_polars as svp_build,
)

# ---------------------------------------------------------------------------
# Mock dataset
# ---------------------------------------------------------------------------


def _write_mock_cache(tmp_path):
    conn_path = tmp_path / "connections.parquet"
    index_path = tmp_path / "neuron_index.parquet"

    idx = pl.DataFrame(
        {
            "bodyId": ["A1", "A2", "B", "C", "D", "E", "F", "G", "H", "X", "Y"],
            "type": ["T1", "T1", "T2", "T2", "T2", "T1", "T3", "T3", "T3", "T3", "T4"],
        }
    )
    idx.write_parquet(index_path)

    conn = pl.DataFrame(
        {
            "bodyId_pre": ["A1", "A2", "B", "C", "D", "E", "A1", "A2", "F", "G", "H"],
            "bodyId_post": ["X", "X", "X", "X", "X", "X", "Y", "Y", "Y", "Y", "Y"],
            "weight": [3, 4, 5, 10, 2, 6, 7, 3, 5, 5, 4],
        }
    )
    conn.write_parquet(conn_path)
    return conn_path, index_path


def _mock_query():
    """The query table at t=3 (D->X with weight 2 is below the cutoff)."""
    return pd.DataFrame(
        {
            "bodyId_pre": ["A1", "A2", "B", "A1", "A2", "H"],
            "bodyId_post": ["X", "X", "X", "Y", "Y", "Y"],
            "type_pre": ["T1", "T1", "T2", "T1", "T1", "T3"],
            "type_post": ["T3", "T3", "T3", "T4", "T4", "T4"],
            "weight": [3, 4, 5, 7, 3, 4],
        }
    )


def _global_tables(conn_path, index_path, min_weight=3):
    cm = ThresholdedConnectionMap(str(conn_path), str(index_path), min_weight=min_weight)
    return cm.total_incoming_by_bodyid(), cm.total_incoming_by_type()


def _stub_fnc(**attrs):
    obj = object.__new__(FindNeuronConnection)
    obj._conn_df_cache = None
    obj._local_neuron_df_cache = {}
    obj._connection_maps = {}
    obj._vprint = lambda *a, **k: None
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


# ---------------------------------------------------------------------------
# Stage 1: ThresholdedConnectionMap aggregates per cutoff
# ---------------------------------------------------------------------------


class TestMapStage:
    def test_per_cutoff_totals_hand_computed(self, tmp_path):
        conn_path, index_path = _write_mock_cache(tmp_path)

        for t, expected_x, expected_y in [(1, 30, 24), (3, 28, 24)]:
            by_body, by_type = _global_tables(conn_path, index_path, min_weight=t)
            totals = dict(
                zip(by_body["bodyId_post"].to_list(), by_body["total_incoming_weight"].to_list())
            )
            assert totals["X"] == expected_x, f"D_{t} X total"
            assert totals["Y"] == expected_y, f"D_{t} Y total"

            type_totals = dict(
                zip(by_type["type_post"].to_list(), by_type["total_incoming_weight"].to_list())
            )
            # T3 (neurons X) incoming from all pre neurons of every type
            assert type_totals["T3"] == expected_x
            assert type_totals["T4"] == expected_y

        # D_3 excludes the weight-2 edge D->X
        by_body_1, _ = _global_tables(conn_path, index_path, min_weight=1)
        by_body_3, _ = _global_tables(conn_path, index_path, min_weight=3)
        x1 = by_body_1.filter(pl.col("bodyId_post") == "X")["total_incoming_weight"][0]
        x3 = by_body_3.filter(pl.col("bodyId_post") == "X")["total_incoming_weight"][0]
        assert x1 - x3 == 2  # exactly the D->X edge


# ---------------------------------------------------------------------------
# Stage 2: Enrichment - conn_type per aggregate method (both engines)
# ---------------------------------------------------------------------------


class TestEnrichmentStage:
    @pytest.fixture()
    def globals_(self, tmp_path):
        conn_path, index_path = _write_mock_cache(tmp_path)
        by_body, by_type = _global_tables(conn_path, index_path, min_weight=3)
        return by_type.to_pandas(), by_body.to_pandas()

    @pytest.mark.parametrize("method", ["product", "average", "ratio"])
    @pytest.mark.parametrize("engine", ["pandas", "polars"])
    def test_conn_type_matches_hand_computed(self, globals_, method, engine):
        global_type, global_body = globals_
        kwargs = dict(
            aggregate_method=method,
            global_incoming_weights=global_type,
            global_incoming_body_weights=global_body,
        )
        if engine == "pandas":
            _, conn_type, _ = EnrichConnectionTable(_mock_query(), **kwargs)
        else:
            _, conn_type, _ = EnrichConnectionTablePolars(_mock_query(), **kwargs)
            conn_type = conn_type.to_pandas()

        # PROBABILITY SUSPENSION (2026-10-06): the aggregate methods still
        # run internally (the F9 threshold filter consumes them), but the
        # suspended probability columns are no longer EXPORTED — the former
        # hand-computed traversal_probability assertions are replaced by
        # absence checks (the aggregate math itself is exercised by
        # TestFilterStage through the min_traversal_probability filter).
        assert "traversal_probability" not in conn_type.columns
        assert "block_probability" not in conn_type.columns
        # Ratios are unaffected and remain exported.
        rows = conn_type.set_index(["type_pre", "type_post"])
        assert set(rows.index) == {
            ("T1", "T3"), ("T2", "T3"), ("T1", "T4"), ("T3", "T4"),
        }

    def test_conn_type_ratios_global(self, globals_):
        """connection_ratio always uses the global D_3 denominators."""
        global_type, global_body = globals_
        _, conn_type, _ = EnrichConnectionTable(
            _mock_query(),
            global_incoming_weights=global_type,
            global_incoming_body_weights=global_body,
        )
        rows = conn_type.set_index(["type_pre", "type_post"])
        assert rows.loc[("T1", "T3"), "connection_ratio"] == pytest.approx(7 / 28)
        assert rows.loc[("T2", "T3"), "connection_ratio"] == pytest.approx(5 / 28)
        assert rows.loc[("T1", "T4"), "connection_ratio"] == pytest.approx(10 / 24)
        assert rows.loc[("T3", "T4"), "connection_ratio"] == pytest.approx(4 / 24)


# ---------------------------------------------------------------------------
# Stage 3: Path builder - path metrics from product-aggregated conn_type
# ---------------------------------------------------------------------------


class TestPathStage:
    def _enriched_type_table(self, tmp_path):
        conn_path, index_path = _write_mock_cache(tmp_path)
        by_body, by_type = _global_tables(conn_path, index_path, min_weight=3)
        _, conn_type, _ = EnrichConnectionTablePolars(
            _mock_query(),
            global_incoming_weights=by_type.to_pandas(),
            global_incoming_body_weights=by_body.to_pandas(),
        )
        return conn_type

    def test_path_prob_product_of_type_edges(self, tmp_path):
        conn_type = self._enriched_type_table(tmp_path)
        paths = [["T1", "T3", "T4"]]
        df_pd = sv_build(paths, conn_type.to_pandas(), ["T4"], real_layer_map=None, level="type")
        row = df_pd.iloc[0]
        # PROBABILITY SUSPENSION (2026-10-06): the per-hop probabilities and
        # the product path_prob are no longer exported.
        assert "path_prob" not in df_pd.columns
        assert "probabilities" not in df_pd.columns
        assert row["min_ratio"] == pytest.approx(4 / 24, abs=1e-6)
        assert row["min_weight"] == pytest.approx(4.0)
        assert row["length"] == 2

    def test_polars_path_builder_agrees(self, tmp_path):
        conn_type = self._enriched_type_table(tmp_path)
        paths = [["T1", "T3", "T4"], ["T1", "T4"]]
        df_pd = sv_build(paths, conn_type.to_pandas(), ["T4"], real_layer_map=None, level="type")
        df_pl = svp_build(paths, conn_type, ["T4"], real_layer_map=None, level="type")
        df_pl = df_pl.to_pandas().sort_values("path").reset_index(drop=True)
        df_pd = df_pd.sort_values("path").reset_index(drop=True)
        # PROBABILITY SUSPENSION (2026-10-06): neither engine exports
        # path_prob/probabilities anymore.
        for col in ["min_ratio", "min_weight", "length"]:
            assert df_pl[col].tolist() == pytest.approx(df_pd[col].tolist(), abs=1e-9)
        assert "path_prob" not in df_pd.columns
        assert "path_prob" not in df_pl.columns


# ---------------------------------------------------------------------------
# Stage 4: coana type-level filter - product semantics on the same D_t
# ---------------------------------------------------------------------------


class TestFilterStage:
    def _obj(self, tmp_path):
        conn_path, index_path = _write_mock_cache(tmp_path)
        by_body, by_type = _global_tables(conn_path, index_path, min_weight=3)

        obj = _stub_fnc()
        obj._fetch_total_incoming_weight_by_type = lambda post_types, min_weight: (
            by_type.filter(pl.col("type_post").is_in(post_types)).to_pandas()
        )
        obj._fetch_total_incoming_weight = lambda post_ids, min_weight: (
            by_body.filter(pl.col("bodyId_post").is_in([str(p) for p in post_ids])).to_pandas()
        )
        return obj

    def test_product_filter_selects_expected_pairs(self, tmp_path):
        """min_traversal_probability=0.6 on product probabilities: only the
        type pairs with compound prob >= 0.6 survive (T1->T3 0.663, T1->T4
        0.984); T2->T3 0.595 and T3->T4 0.556 are filtered out."""
        obj = self._obj(tmp_path)
        out = obj._apply_type_level_filters(
            _mock_query().copy(),
            min_weight=3,
            min_conn_ratio=0.0,
            min_traversal_prob=0.6,
            total_before_filter=6,
        )
        kept = sorted(zip(out["bodyId_pre"], out["bodyId_post"]))
        assert kept == [("A1", "X"), ("A1", "Y"), ("A2", "X"), ("A2", "Y")]
        # The weight-2 D->X edge was already gone (below the D_3 cutoff)
        assert "D" not in out["bodyId_pre"].tolist()

    def test_ratio_filter_keeps_more_pairs(self, tmp_path):
        """aggregate_method='ratio' at the same threshold keeps T3->T4
        (0.556 >= 0.5) in addition to the product survivors."""
        obj = self._obj(tmp_path)
        out = obj._apply_type_level_filters(
            _mock_query().copy(),
            min_weight=3,
            min_conn_ratio=0.0,
            min_traversal_prob=0.5,
            total_before_filter=6,
            aggregate_method="ratio",
        )
        kept = sorted(zip(out["bodyId_pre"], out["bodyId_post"]))
        assert ("H", "Y") in kept  # T3->T4 survives under the ratio model
