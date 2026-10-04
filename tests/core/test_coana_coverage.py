#!/usr/bin/env python
"""Coverage-focused hermetic tests for src/coana.py.

All tests run fully offline: connection tables are tiny synthetic
DataFrames injected directly or served by monkeypatched fetchers. File
I/O goes to pytest's tmp_path only. Existing tests already cover the
FindAllPath graph cache and path-edge/layer matching; this file targets
the remaining uncovered helpers:

  - connection-table-to-matrix pandas-2.x fallback
  - layer-table helpers (pandas branches), cache clearers
  - hemisphere normalization / suffix / filtering helpers
  - ratio & probability derivation
  - export normalizers and CSV/Excel matrix writers
  - query/export metadata helpers (all_neurons token, group labels,
    readable names, secret sanitization)
  - hemisphere conservation filtering + symmetry analysis
  - path-derivation helpers, warning notes, neuron enrollment
  - FindDirectConnections end-to-end with fake fetchers
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

import coana  # noqa: E402


# =============================================================================
# fixture helpers
# =============================================================================

def make_fc(**attrs):
    """Build an uninitialized FindNeuronConnection with silent logging."""
    fc = object.__new__(coana.FindNeuronConnection)
    logs = []
    fc._vprint = lambda msg="", level="full", end="\n", flush=False: logs.append(str(msg))
    fc.verbose_mode = "silent"
    fc.progress_events = False
    fc._warn_notes = []
    for key, value in attrs.items():
        setattr(fc, key, value)
    return fc, logs


def conn_df(rows, extra=None):
    """rows: list of (pre, post, weight[, type_pre, type_post])"""
    data = {
        "bodyId_pre": [str(r[0]) for r in rows],
        "bodyId_post": [str(r[1]) for r in rows],
        "weight": [r[2] for r in rows],
    }
    if rows and len(rows[0]) > 3:
        data["type_pre"] = [r[3] for r in rows]
        data["type_post"] = [r[4] for r in rows]
    if extra:
        data.update(extra)
    return pd.DataFrame(data)


# =============================================================================
# connection_table_to_matrix fallback + layer helpers + cache clearers
# =============================================================================

class TestPatchedConnectionTableToMatrix:
    def test_pandas2_fallback_pivot(self, monkeypatch):
        rows = [("1", "2", 5), ("1", "2", 3), ("2", "3", 7)]
        df = conn_df(rows)

        def raiser(conn_df, group_cols="bodyId", weight_col="weight",
                   sort_by=None, make_square=False):
            raise TypeError("pivot() got an unexpected keyword argument")

        monkeypatch.setattr(coana, "_original_connection_table_to_matrix", raiser)
        mat = coana._patched_connection_table_to_matrix(df)
        assert mat.loc["1", "2"] == 8  # summed duplicate edge
        assert mat.loc["2", "3"] == 7

    def test_pandas2_fallback_make_square(self, monkeypatch):
        df = conn_df([("1", "2", 5)])

        def raiser(*a, **k):
            raise TypeError("nested pivot() failure")

        monkeypatch.setattr(coana, "_original_connection_table_to_matrix", raiser)
        mat = coana._patched_connection_table_to_matrix(
            df, group_cols="bodyId", make_square=True)
        assert list(mat.index) == ["1", "2"]
        assert list(mat.columns) == ["1", "2"]
        assert mat.loc["2", "1"] == 0

    def test_non_pivot_typeerror_propagates(self, monkeypatch):
        def raiser(*a, **k):
            raise TypeError("something else entirely")

        monkeypatch.setattr(coana, "_original_connection_table_to_matrix", raiser)
        with pytest.raises(TypeError, match="something else"):
            coana._patched_connection_table_to_matrix(conn_df([("1", "2", 1)]))


class TestLayerAndCacheHelpers:
    def test_layer_table_edge_pairs_pandas_branches(self):
        fc, _ = make_fc()
        assert coana._layer_table_edge_pairs(None) == set()
        empty = pd.DataFrame(columns=["bodyId_pre", "bodyId_post"])
        assert coana._layer_table_edge_pairs(empty) == set()
        pairs = coana._layer_table_edge_pairs(conn_df([(1, 2, 3), (3, 4, 1)]))
        assert pairs == {("1", "2"), ("3", "4")}

    def test_clear_findallpath_cache_dataset_normalized(self):
        cache = coana._FINDALLPATH_GRAPH_CACHE
        cache.clear()
        try:
            cache["hemibrain_v1_2_1_a"] = {"x": 1}
            cache["male_cns_v1_0_b"] = {"x": 2}
            coana.clear_findallpath_cache("hemibrain:v1.2.1")
            assert "hemibrain_v1_2_1_a" not in cache
            assert "male_cns_v1_0_b" in cache
            coana.clear_findallpath_cache()
            assert cache == {}
        finally:
            cache.clear()

    def test_clear_fnc_cache(self):
        cache = coana._FNC_CACHE
        cache["ds_a"] = {"conn_df": None}
        cache["ds_b"] = {"conn_df": None}
        try:
            coana.clear_fnc_cache("ds_a")
            assert "ds_a" not in cache and "ds_b" in cache
            coana.clear_fnc_cache()
            assert cache == {}
        finally:
            cache.clear()


# =============================================================================
# hemisphere helpers
# =============================================================================

class TestHemisphereHelpers:
    def test_normalize_hemisphere_value(self):
        fc, _ = make_fc()
        assert fc._normalize_hemisphere_value("right") == "R"
        assert fc._normalize_hemisphere_value(" L ") == "L"
        assert fc._normalize_hemisphere_value("lhs") == "L"
        assert fc._normalize_hemisphere_value(None) == "U"
        assert fc._normalize_hemisphere_value(float("nan")) == "U"
        assert fc._normalize_hemisphere_value("middle") == "U"

    def test_find_hemisphere_column(self):
        FNC = coana.FindNeuronConnection
        assert FNC._find_hemisphere_column(pd.DataFrame({"Soma side": ["R"]})) == "Soma side"
        assert FNC._find_hemisphere_column(pd.DataFrame({"somaSide": ["R"]})) == "somaSide"
        assert FNC._find_hemisphere_column(pd.DataFrame({"hemisphere": ["R"], "somaSide": ["L"]})) == "hemisphere"
        assert FNC._find_hemisphere_column(pd.DataFrame({"rootSide": ["R"]})) == "rootSide"
        assert FNC._find_hemisphere_column(pd.DataFrame({"type": ["x"]})) is None

    def test_ensure_hemisphere_columns(self):
        fc, _ = make_fc()
        # from Soma side column
        df = fc._ensure_hemisphere_columns(pd.DataFrame({"Soma side": ["R", "left"]}))
        assert list(df["hemisphere_code"]) == ["R", "L"]
        # from instance suffixes
        df = fc._ensure_hemisphere_columns(pd.DataFrame({"instance": ["a_R", "b_L", "c"]}))
        assert list(df["hemisphere_code"]) == ["R", "L", "U"]
        # no source at all
        df = fc._ensure_hemisphere_columns(pd.DataFrame({"type": ["x"]}))
        assert list(df["hemisphere_code"]) == ["U"]
        # empty passthrough
        assert fc._ensure_hemisphere_columns(pd.DataFrame()).empty

    def test_append_hemi_suffix(self):
        fc, _ = make_fc()
        assert fc._append_hemi_suffix("aMe12", "R") == "aMe12_R"
        assert fc._append_hemi_suffix("aMe12_L", "R") == "aMe12_L"
        assert fc._append_hemi_suffix(None, "U") == "Unknown_U"

    def test_hemi_code_series_priority(self):
        fc, _ = make_fc()
        df = pd.DataFrame({
            "hemisphere_code_pre": ["R", None, None],
            "hemisphere_pre": [None, "left", None],
            "instance_pre": ["x_L", "y_R", "z_L"],
        })
        codes = fc._hemi_code_series(df, "pre")
        # code column wins verbatim, then hemisphere normalization, then
        # instance suffix for the unhandled row
        assert list(codes) == ["R", "L", "L"]

    def test_hemi_code_series_default_side(self):
        fc, _ = make_fc()
        df = pd.DataFrame({"instance": ["n_R", "n_L"]})
        assert list(fc._hemi_code_series(df, "")) == ["R", "L"]

    def test_append_hemi_suffix_series(self):
        fc, _ = make_fc()
        labels = pd.Series(["A", "B_R", None])
        codes = pd.Series(["L", "R", "U"])
        out = fc._append_hemi_suffix_series(labels, codes)
        assert list(out) == ["A_L", "B_R", "Unknown_U"]

    def test_apply_hemisphere_suffix_to_neuron_df(self):
        fc, _ = make_fc(separate_hemispheres=True, hemisphere_filter="both")
        df = pd.DataFrame({
            "type": ["T1", "T2", "T3"],
            "custom_group": ["g", "g", "h"],
            "hemisphere": ["left", "right", ""],
        })
        out = fc._apply_hemisphere_suffix_to_neuron_df(df)
        assert list(out["type"]) == ["T1_L", "T2_R", "T3_U"]
        assert list(out["custom_group"]) == ["g_L", "g_R", "h_U"]

    def test_apply_hemisphere_filter_to_neuron_df(self):
        fc, _ = make_fc(separate_hemispheres=False, hemisphere_filter="left")
        df = pd.DataFrame({
            "type": ["T1", "T2", "T3"],
            "hemisphere": ["left", "right", ""],
        })
        out = fc._apply_hemisphere_suffix_to_neuron_df(df)
        # left filter keeps L and U, drops R
        assert list(out["type"]) == ["T1", "T3"]

    def test_ensure_ratio_prob_columns_pandas(self):
        fc, _ = make_fc()
        df = pd.DataFrame({
            "bodyId_pre": ["1", "2"],
            "bodyId_post": ["X", "X"],
            "weight": [10, 30],
        })
        out = fc._ensure_ratio_prob_columns(df, "bodyId_pre", "bodyId_post")
        assert list(out["connection_ratio"]) == [0.25, 0.75]
        assert out["traversal_probability"].iloc[0] == pytest.approx(0.25 / 0.3)
        assert out["traversal_probability"].iloc[1] == 1.0  # clipped

    def test_ensure_ratio_prob_columns_keeps_existing(self):
        fc, _ = make_fc()
        df = pd.DataFrame({
            "bodyId_pre": ["1"], "bodyId_post": ["X"], "weight": [10],
            "connection_ratio": [0.4], "traversal_probability": [0.9],
        })
        out = fc._ensure_ratio_prob_columns(df, "bodyId_pre", "bodyId_post")
        assert out["connection_ratio"].iloc[0] == 0.4

    def test_ensure_ratio_prob_columns_polars(self):
        fc, _ = make_fc()
        df = pl.DataFrame({
            "bodyId_pre": ["1", "2"],
            "bodyId_post": ["X", "X"],
            "weight": [10, 30],
        })
        out = fc._ensure_ratio_prob_columns(df, "bodyId_pre", "bodyId_post")
        ratios = sorted(out["connection_ratio"].to_list())
        assert ratios == [0.25, 0.75]

    def test_ensure_ratio_prob_columns_edge_cases(self):
        fc, _ = make_fc()
        assert fc._ensure_ratio_prob_columns(None, "a", "b") is None
        assert fc._ensure_ratio_prob_columns(pd.DataFrame(), "a", "b").empty
        no_weight = pd.DataFrame({"bodyId_pre": ["1"]})
        assert fc._ensure_ratio_prob_columns(no_weight, "a", "b") is no_weight

    def test_apply_hemisphere_suffix_to_conn_df(self):
        fc, _ = make_fc(separate_hemispheres=True, hemisphere_filter="both")
        df = conn_df(
            [("1", "2", 5, "A", "B")],
            extra={
                "custom_group_pre": ["g1"],
                "custom_group_post": ["g2"],
                "hemisphere_pre": ["left"],
                "hemisphere_post": ["right"],
            },
        )
        out = fc._apply_hemisphere_suffix_to_conn_df(df)
        assert out["type_pre"].iloc[0] == "A_L"
        assert out["type_post"].iloc[0] == "B_R"
        assert out["custom_group_pre"].iloc[0] == "g1_L"
        assert out["custom_group_post"].iloc[0] == "g2_R"

    def test_apply_hemisphere_filter_to_conn_df(self):
        fc, _ = make_fc(separate_hemispheres=False, hemisphere_filter="right")
        df = conn_df(
            [("1", "2", 5), ("3", "4", 6)],
            extra={
                "hemisphere_pre": ["right", "left"],
                "hemisphere_post": ["right", "right"],
            },
        )
        out = fc._apply_hemisphere_suffix_to_conn_df(df)
        assert list(out["bodyId_pre"]) == ["1"]

    def test_query_has_hemisphere_suffix(self):
        fc, _ = make_fc()
        assert fc._query_has_hemisphere_suffix("aMe12_L") is True
        assert fc._query_has_hemisphere_suffix("aMe12") is False
        assert fc._query_has_hemisphere_suffix(None) is False
        assert fc._query_has_hemisphere_suffix(["x", ["y_R"]]) is True
        assert fc._query_has_hemisphere_suffix(["x", "y"]) is False
        assert fc._query_has_hemisphere_suffix({"endswith": "_R"}) is True
        assert fc._query_has_hemisphere_suffix({"endswith": ["_X"]}) is False
        assert fc._query_has_hemisphere_suffix({"type": {"regex": "a_L"}}) is True
        assert fc._query_has_hemisphere_suffix({"type": "plain"}) is False


# =============================================================================
# export normalizers + writers
# =============================================================================

class TestExportNormalizers:
    def test_normalize_weight_list_for_export(self):
        norm = coana.FindNeuronConnection._normalize_weight_list_for_export
        out = norm([1.0, 2.5, "x", np.inf])
        assert out[0] == 1 and isinstance(out[0], int)
        assert out[1] == 2.5
        assert out[2] == "x"
        assert norm("scalar") == "scalar"

    def test_normalize_export_count_columns_pandas(self):
        FNC = coana.FindNeuronConnection
        df = pd.DataFrame({
            "weight": [5.0, 3.0],
            "total_weight": [8.0, 3.0],
            "fraction": [0.5, 0.25],          # not a count column
            "weights": [[1.0, 2.0], [3.5]],
        })
        out = FNC._normalize_export_count_columns_pandas(df)
        assert str(out["weight"].dtype) == "Int64"
        assert str(out["total_weight"].dtype) == "Int64"
        assert out["fraction"].iloc[0] == 0.5  # untouched
        assert out["weights"].iloc[0] == [1, 2]
        assert out["weights"].iloc[1] == [3.5]

    def test_normalize_export_count_columns_pandas_nonintegral(self):
        FNC = coana.FindNeuronConnection
        df = pd.DataFrame({"weight": [5.5]})
        out = FNC._normalize_export_count_columns_pandas(df)
        # non-integral values keep the original float column
        assert out["weight"].iloc[0] == 5.5

    def test_normalize_export_count_columns_polars(self):
        FNC = coana.FindNeuronConnection
        df = pl.DataFrame({
            "weight": [5.0, 3.0],
            "weights": [[1.0, 2.0], [3.0]],
        })
        out = FNC._normalize_export_count_columns_polars(df)
        assert out.schema["weight"] == pl.Int64
        assert out["weights"].to_list() == [[1, 2], [3]]
        # no matching columns -> returned unchanged
        same = pl.DataFrame({"x": [1]})
        assert FNC._normalize_export_count_columns_polars(same) is same

    def test_save_df_to_csv_polars_variants(self, tmp_path):
        fc, _ = make_fc()
        # pandas non-empty
        df = pd.DataFrame({"weight": [5.0], "type_pre": ["A"]})
        p1 = tmp_path / "pandas.csv"
        fc._save_df_to_csv_polars(df, str(p1))
        text = p1.read_text(encoding="utf-8")
        assert "weight" in text and "5" in text
        # pandas empty -> header only
        p2 = tmp_path / "empty_pd.csv"
        fc._save_df_to_csv_polars(pd.DataFrame(columns=["a", "b"]), str(p2))
        assert p2.read_text(encoding="utf-8").strip() == "a,b"
        # polars empty -> header only
        p3 = tmp_path / "empty_pl.csv"
        fc._save_df_to_csv_polars(pl.DataFrame({"a": [], "b": []}), str(p3))
        assert p3.read_text(encoding="utf-8").strip() == "a,b"
        # polars non-empty
        p4 = tmp_path / "polars.csv"
        fc._save_df_to_csv_polars(pl.DataFrame({"weight": [7]}), str(p4))
        assert "7" in p4.read_text(encoding="utf-8")
        # index=True keeps the index column
        p5 = tmp_path / "indexed.csv"
        mat = pd.DataFrame({"c1": [1]}, index=["r1"])
        fc._save_df_to_csv_polars(mat, str(p5), index=True)
        assert "r1" in p5.read_text(encoding="utf-8")
        # None is a no-op
        fc._save_df_to_csv_polars(None, str(tmp_path / "none.csv"))
        assert not (tmp_path / "none.csv").exists()

    def test_save_matrices_to_excel(self, tmp_path):
        fc, _ = make_fc()
        df = conn_df(
            [("1", "2", 5, "A", "B"), ("2", "3", 7, "B", "C")],
            extra={
                "connection_ratio": [0.5, 0.25],
                "traversal_probability": [0.9, 0.4],
                "nt_type": ["ACh", "GABA"],
            },
        )
        xlsx = tmp_path / "mat.xlsx"
        with pd.ExcelWriter(xlsx, mode="w", engine="xlsxwriter") as writer:
            fc._save_matrices_to_excel(df, writer, level="bodyId")
            fc._save_matrices_to_excel(pl.from_pandas(df), writer, level="type")
            fc._save_matrices_to_excel(pd.DataFrame(), writer)  # no-op
        sheets = pd.ExcelFile(xlsx).sheet_names
        for name in ("conn_mat_bodyId_weight", "conn_mat_bodyId_ratio",
                     "conn_mat_bodyId_prob", "conn_mat_bodyId_nt",
                     "conn_mat_type_weight", "conn_mat_type_ratio",
                     "conn_mat_type_prob", "conn_mat_type_nt"):
            assert name in sheets

    def test_save_matrices_to_csv(self, tmp_path):
        fc, _ = make_fc()
        folder = tmp_path / "mats"
        folder.mkdir()
        df = conn_df(
            [("A", "B", 5, "A", "B")],
            extra={
                "connection_ratio": [0.5],
                "traversal_probability": [0.9],
                "nt_type": ["ACh"],
            },
        )
        fc._save_matrices_to_csv(df, str(folder), level="type")
        assert (folder / "conn_mat_type_weight.csv").exists()
        assert (folder / "conn_mat_type_ratio.csv").exists()
        assert (folder / "conn_mat_type_prob.csv").exists()
        assert (folder / "conn_mat_type_nt.csv").exists()
        # empty frame writes nothing
        fc._save_matrices_to_csv(pd.DataFrame(), str(folder), level="type")


# =============================================================================
# query / export metadata helpers
# =============================================================================

class TestQueryAndExportMetadata:
    def test_query_uses_all_neurons(self):
        FNC = coana.FindNeuronConnection
        assert FNC._query_uses_all_neurons("all_neurons") is True
        assert FNC._query_uses_all_neurons("  ALL_NEURONS ") is True
        assert FNC._query_uses_all_neurons(["x", ["all_neurons"]]) is True
        assert FNC._query_uses_all_neurons({"contains": "all_neurons"}) is False
        assert FNC._query_uses_all_neurons("other") is False
        assert FNC._query_uses_all_neurons(None) is False
        assert FNC._query_uses_all_neurons(123) is False

    def test_apply_all_neurons_query(self):
        fc, logs = make_fc(dataset="test:v1")
        assert fc._apply_all_neurons_query("all_neurons", "source") == []
        assert logs  # detection notice printed
        assert fc._apply_all_neurons_query(["Mi1"], "target") == ["Mi1"]

    def test_expand_group_labels(self):
        class Mapper:
            is_empty = False

            def get_neurons_for_label(self, label, dataset, role):
                if label == "grp":
                    return [1, "2"]
                return None

        fc, _ = make_fc(label_mapper=Mapper(), dataset="test:v1")
        out = fc._expand_group_labels(["grp", "plain", "grp"], "source")
        assert out == ["1", "2", "plain"]
        # no mapper -> passthrough
        fc2, _ = make_fc(label_mapper=None, dataset="test:v1")
        assert fc2._expand_group_labels(["x"], "source") == ["x"]
        # dict query -> passthrough
        assert fc._expand_group_labels({"type": "x"}, "source") == {"type": "x"}

    def test_readable_query_name(self):
        name = coana.FindNeuronConnection._readable_query_name
        assert name({"type": "x"}, "fallback") == "fallback"
        assert name(["aMe12", "PPL1"]) == "aMe12_etc"
        assert name(["aMe12"]) == "aMe12"
        assert name("solo") == "solo"
        assert name([["a", "b"]]) == "a_etc"
        assert name([], "fb") == "fb"
        assert name([".*"], "fb") == "fb"

    def test_is_sensitive_export_key(self):
        sens = coana.FindNeuronConnection._is_sensitive_export_key
        assert sens("token") is True
        assert sens("API-Key") is True
        assert sens("neuprint token") is True
        assert sens("my_secret") is True
        assert sens("password") is True
        assert sens("dataset") is False
        assert sens(None) is False

    def test_sanitize_export_value(self):
        sanitize = coana.FindNeuronConnection._sanitize_export_value
        payload = {
            "dataset": "x",
            "token": "abc",
            "nested": {"api_key": "k", "keep": 1},
            "list": [{"password": "p"}, "ok"],
        }
        out = sanitize(payload)
        assert out == {
            "dataset": "x",
            "nested": {"keep": 1},
            "list": [{}, "ok"],
        }
        assert "token" in payload  # original never mutated

    def test_run_export_attributes(self, tmp_path):
        fc, _ = make_fc(
            dataset="test:v1",
            min_synapse_num=5,
            sourceNeurons=["aMe12"],
            targetNeurons=["PPL1"],
            source_df=pd.DataFrame({"bodyId": [1]}),
            target_df=pd.DataFrame({"bodyId": [2, 3]}),
            label_mapper=None,
            secret_token="hidden",
        )
        fc._requested_source_neurons = ["aMe12"]
        attrs = fc._run_export_attributes(path_mode="shortest")
        assert attrs["dataset"] == "test:v1"
        assert attrs["path_mode"] == "shortest"
        assert attrs["requested_source_neurons"] == ["aMe12"]
        assert attrs["resolved_source_bodyIds"] == ["1"]
        assert attrs["resolved_target_bodyIds"] == ["2", "3"]
        assert "secret_token" not in attrs
        assert "_warn_notes" not in attrs

    def test_custom_group_payload_and_parameters(self):
        class Mapper:
            is_empty = True

        fc, _ = make_fc(label_mapper=Mapper(), dataset="d")
        assert fc._custom_group_export_payload() is None

        class Mapper2:
            is_empty = False

            def get_all_std_labels(self, role):
                return ["grp"] if role == "source" else []

            def get_neurons_for_label(self, label, dataset, role):
                return [10, 20]

        fc2, _ = make_fc(
            label_mapper=Mapper2(), dataset="d",
            custom_mapping_file="/tmp/map.json",
            parameter_dict={},
            sourceNeurons=["grp"], targetNeurons=[],
            source_df=pd.DataFrame({"bodyId": [10, 20]}),
            target_df=pd.DataFrame({"bodyId": []}),
        )
        payload = fc2._custom_group_export_payload()
        assert payload["source_groups"] == [
            {"label": "grp", "members": ["10", "20"], "member_count": 2}]
        fc2._add_custom_group_parameters()
        assert "custom mapping file" in fc2.parameter_dict
        assert fc2.parameter_dict["resolved source bodyIds"] == "['10', '20']"


# =============================================================================
# hemisphere conservation + symmetry analysis
# =============================================================================

class TestHemisphereConservation:
    def test_is_symmetric_dataset(self):
        fc, _ = make_fc(dataset="male-cns:v1.0")
        assert fc._is_symmetric_dataset() is True
        fc.dataset = "hemibrain:v1.2.1"
        assert fc._is_symmetric_dataset() is False

    def test_extract_hemi_from_label(self):
        fc, _ = make_fc()
        assert fc._extract_hemi_from_label("aMe12_L") == ("aMe12", "L")
        assert fc._extract_hemi_from_label("aMe12_R (x)") == ("aMe12", "R")
        assert fc._extract_hemi_from_label("plain") == ("plain", None)

    def test_filter_hemisphere_unconserved_edges_pandas(self):
        fc, _ = make_fc()
        df = pd.DataFrame({
            "type_pre": ["A_L", "A_R", "A_L", "D", "F_U"],
            "type_post": ["B_L", "B_R", "C_R", "E", "G_L"],
            "weight": [10, 12, 5, 7, 3],
        })
        kept, removed = fc._filter_hemisphere_unconserved_edges(df)
        assert set(zip(kept["type_pre"], kept["type_post"])) == {
            ("A_L", "B_L"), ("A_R", "B_R"), ("D", "E"), ("F_U", "G_L")}
        assert set(zip(removed["type_pre"], removed["type_post"])) == {("A_L", "C_R")}

    def test_filter_hemisphere_unconserved_edges_polars_and_empty(self):
        fc, _ = make_fc()
        df = pl.DataFrame({
            "type_pre": ["A_L", "A_R"],
            "type_post": ["B_L", "B_R"],
            "weight": [1, 2],
        })
        kept, removed = fc._filter_hemisphere_unconserved_edges(df)
        assert isinstance(kept, pl.DataFrame)
        assert removed is None and len(kept) == 2
        empty_pd = pd.DataFrame(columns=["type_pre", "type_post", "weight"])
        out, none = fc._filter_hemisphere_unconserved_edges(empty_pd)
        assert out is empty_pd and none is None

    def test_count_hemisphere_from_df(self):
        fc, _ = make_fc()
        assert fc._count_hemisphere_from_df(None) == {"L": 0, "R": 0, "U": 0}
        df = pd.DataFrame({"hemisphere_code": ["L", "r", "x"]})
        assert fc._count_hemisphere_from_df(df) == {"L": 1, "R": 1, "U": 1}
        df = pd.DataFrame({"Soma side": ["right", "left", None]})
        assert fc._count_hemisphere_from_df(df) == {"L": 1, "R": 1, "U": 1}
        df = pd.DataFrame({"type": ["A_L", "B_R", "C"]})
        assert fc._count_hemisphere_from_df(df) == {"L": 1, "R": 1, "U": 1}

    def test_run_hemisphere_symmetry_analysis(self, tmp_path):
        fc, _ = make_fc(
            symmetry_analysis=True,
            dataset="male-cns:v1.0",
            allpath_folder=str(tmp_path),
            min_synapse_num=1,
            source_df=pd.DataFrame({"type": ["A_L", "A_R"],
                                    "hemisphere_code": ["L", "R"]}),
            target_df=pd.DataFrame({"type": ["B_L"],
                                    "hemisphere_code": ["L"]}),
        )
        conn_types = pd.DataFrame({
            "type_pre": ["A_L", "A_R", "A_L", "D"],
            "type_post": ["B_L", "B_R", "C_R", "E"],
            "weight": [10, 20, 5, 7],
        })
        paths = pd.DataFrame({
            "path_block": ["A_L -> B_L", "A_R -> B_R", "X_L -> Y_L",
                            "A_L -> C_R"],
        })
        fc._run_hemisphere_symmetry_analysis(conn_types, paths_df=paths)

        sym_dir = tmp_path / "hemisphere_symmetry"
        import json
        summary = json.loads((sym_dir / "symmetry_summary.json").read_text())
        assert summary["ipsi"]["conserved"] == 1
        assert summary["ipsi"]["union"] == 1
        assert summary["ipsi"]["jaccard"] == pytest.approx(1.0)
        assert summary["contra"]["union"] == 1
        assert summary["contra"]["conserved"] == 0
        assert summary["paths"]["total_L"] == 2
        assert summary["paths"]["total_R"] == 1
        assert summary["paths"]["conserved"] == 1
        assert summary["paths"]["unconserved_L_only"] == 1
        assert summary["hemisphere_counts"]["source"] == {"L": 1, "R": 1, "U": 0}

        for name in ("symmetry_ipsi.csv", "symmetry_contra.csv",
                     "conserved_edges.csv", "unconserved_edges.csv",
                     "pairwise_strength.csv", "type_counts_by_role.csv",
                     "conserved_paths.csv", "unconserved_paths.csv"):
            assert (sym_dir / name).exists(), name
        ipsi = pd.read_csv(sym_dir / "symmetry_ipsi.csv")
        row = ipsi.iloc[0]
        assert row["weight_L"] == 10 and row["weight_R"] == 20
        assert row["ratio"] == pytest.approx(0.5)

    def test_run_hemisphere_symmetry_analysis_skips(self, tmp_path):
        base = dict(
            symmetry_analysis=True, dataset="male-cns:v1.0",
            allpath_folder=str(tmp_path), min_synapse_num=1,
            source_df=pd.DataFrame(), target_df=pd.DataFrame(),
        )
        # disabled flag
        fc, _ = make_fc(**{**base, "symmetry_analysis": False})
        fc._run_hemisphere_symmetry_analysis(pd.DataFrame(
            {"type_pre": ["A_L"], "type_post": ["B_L"], "weight": [1]}))
        assert not (tmp_path / "hemisphere_symmetry").exists()
        # non-symmetric dataset
        fc, _ = make_fc(**{**base, "dataset": "hemibrain:v1.2.1"})
        fc._run_hemisphere_symmetry_analysis(pd.DataFrame(
            {"type_pre": ["A_L"], "type_post": ["B_L"], "weight": [1]}))
        assert not (tmp_path / "hemisphere_symmetry").exists()
        # empty frame / missing columns
        fc, _ = make_fc(**base)
        fc._run_hemisphere_symmetry_analysis(pd.DataFrame())
        fc._run_hemisphere_symmetry_analysis(pd.DataFrame({"type_pre": ["A"]}))
        assert not (tmp_path / "hemisphere_symmetry").exists()


# =============================================================================
# path helpers, warning notes, enrollment
# =============================================================================

class TestPathHelpers:
    def test_derive_label_paths_from_bodyid_paths(self):
        fc, _ = make_fc()
        label = {"1": "S", "2": "M", "3": "T", "4": "T"}
        kept_edges = {("S", "M"), ("M", "T")}
        paths = [["1", "2", "3"],      # valid
                 ["1", "2", "4"],      # duplicate type sequence
                 ["2", "3"],           # starts outside source set
                 ["1", "3"]]           # missing type edge S->T
        out = fc._derive_label_paths_from_bodyid_paths(
            paths, label.get, kept_edges, ["S"], ["T"])
        assert out == [["S", "M", "T"]]

    def test_keep_shortest_bodyid_paths(self):
        keep = coana.FindNeuronConnection._keep_shortest_bodyid_paths
        paths = [
            ["a", "b", "t"],   # pair (a,t) length 2
            ["a", "x", "y", "t"],  # longer alternative for (a,t)
            ["b", "t"],        # pair (b,t) length 1
            [],                # dropped
        ]
        out = keep(paths)
        assert ["a", "b", "t"] in out
        assert ["b", "t"] in out
        assert ["a", "x", "y", "t"] not in out
        assert len(out) == 2

    def test_build_bodyid_type_map(self):
        fc, _ = make_fc(
            all_connections_filtered=[conn_df([("1", "2", 5, "TA", "TB")])],
            source_df=pd.DataFrame({"bodyId": ["1", "3"], "type": ["TA", ""]}),
            target_df=pd.DataFrame({"bodyId": ["2"], "type": ["TB"]}),
        )
        type_map = fc._build_bodyid_type_map()
        assert type_map == {"1": "TA", "2": "TB"}

    def test_extract_nodes_from_path_graph(self):
        fc, _ = make_fc()
        assert fc._extract_nodes_from_path_graph(None) == []
        assert fc._extract_nodes_from_path_graph(pd.DataFrame()) == []
        nodes = fc._extract_nodes_from_path_graph(conn_df([("1", "2", 3)]))
        assert set(nodes) == {"1", "2"}
        nodes_pl = fc._extract_nodes_from_path_graph(
            pl.DataFrame({"bodyId_pre": ["1"], "bodyId_post": ["2"]}))
        assert set(nodes_pl) == {"1", "2"}

    def test_reset_temp_columns(self):
        fc, _ = make_fc(
            source_df=pd.DataFrame({"bodyId": ["1"], "isInPath": [True]}),
            target_df=pd.DataFrame({"bodyId": ["2"], "Checked": [True],
                                    "Layer": [1]}),
        )
        fc._reset_temp_columns()
        assert "isInPath" not in fc.source_df.columns
        assert "Checked" not in fc.target_df.columns
        assert fc._edgeN_limit_reached is False
        assert fc._shortest_target_hop_limits == {}

    def test_record_search_priority_warnings(self):
        fc, _ = make_fc()
        fc._record_search_priority_warnings("source", [
            {"matched_column": "type", "search_term": "Mi1", "match_count": 3},
            {"matched_column": "hemiline", "search_term": "Hb", "match_count": 12},
            {},
        ])
        assert len(fc._warn_notes) == 1
        assert "hemiline" in fc._warn_notes[0]

    def test_fetch_direct_connections_for_nodes(self):
        fc, _ = make_fc(min_synapse_num=1, min_ratio=0.0,
                        min_traversal_probability=0.0)
        table = conn_df([("1", "2", 5), ("1", "9", 5)])
        fc._fetch_connections_with_cache = (
            lambda upstream_bodyIds=None, downstream_bodyIds=None,
            min_weight=None, min_conn_ratio=None, min_traversal_prob=None:
            table.copy()
        )
        assert fc._fetch_direct_connections_for_nodes([]).empty
        out = fc._fetch_direct_connections_for_nodes(["1", "2"])
        assert list(out["bodyId_post"]) == ["2"]

    def test_write_user_warning_notes(self, tmp_path):
        fc, _ = make_fc(
            min_ratio=0.1,
            min_traversal_probability=0.05,
            keyword_in_path_to_remove=["GABA"],
            edgeN_limit=100,
            _edgeN_limit_reached=True,
            max_interlayer=5,
            _depth_cap_reached=True,
            _shortest_backward_active=True,
            _shortest_scope_limited=True,
            separate_hemispheres=True,
            hemisphere_filter="left",
            keep_only_hemisphere_conserved_connections=True,
            symmetry_analysis=True,
            find_reciprocal=True,
            skip_bodyId=True,
            pathN_to_show=3,
            cache_only=True,
        )
        folder = tmp_path / "run"
        folder.mkdir()
        fc._write_user_warning_notes(str(folder))
        text = (folder / "user_warning_notes.txt").read_text(encoding="utf-8")
        for token in ("min_ratio=0.1", "keyword_in_path_to_remove",
                      "edge limit", "depth", "hemisphere", "symmetry",
                      "reciprocal", "skip_bodyId", "pathN_to_show",
                      "cache_only"):
            assert token in text

    def test_write_user_warning_notes_empty(self, tmp_path):
        # skip_bodyId=False keeps the config-derived '[output]' note out so
        # the writer has nothing to record.
        fc, _ = make_fc(min_ratio=0.0, min_traversal_probability=0.0,
                        hemisphere_filter="both", skip_bodyId=False)
        fc._write_user_warning_notes(str(tmp_path))
        assert not (tmp_path / "user_warning_notes.txt").exists()

    def test_save_path_neuron_enrollment(self, tmp_path):
        fc, _ = make_fc(
            source_df=pd.DataFrame({"bodyId": ["1"], "type": ["S"]}),
            target_df=pd.DataFrame({"bodyId": ["2"], "type": ["T"]}),
        )
        src_path, tgt_path = fc._save_path_neuron_enrollment(str(tmp_path))
        src = pd.read_csv(src_path)
        tgt = pd.read_csv(tgt_path)
        assert "isInPath" in src.columns
        assert {"Checked", "Layer"}.issubset(tgt.columns)

    def test_record_viz_edge_trim(self):
        fc, _ = make_fc(_edgeN_limit_reached=False)
        from types import SimpleNamespace
        fc._record_viz_edge_trim(SimpleNamespace(edge_limit_trimmed=True))
        assert fc._edgeN_limit_reached is True
        fc._record_viz_edge_trim(SimpleNamespace(edge_limit_trimmed=False))
        assert fc._edgeN_limit_reached is True

    def test_is_empty_df(self):
        empty = coana.FindNeuronConnection._is_empty_df
        assert empty(None) is True
        assert empty(pd.DataFrame()) is True
        assert empty(pl.DataFrame()) is True
        assert empty(pd.DataFrame({"a": [1]})) is False


# =============================================================================
# FindDirectConnections offline pipeline
# =============================================================================

def _make_direct_fc(monkeypatch, tmp_path, fetch_rows, *, types,
                    keep_conserved=False, separate=False, target_ids=None):
    """Wire a FindNeuronConnection for an offline FindDirectConnections run.

    fetch_rows: list of (pre, post, weight); types: {bodyId: type}
    """
    def fake_fetch_path_connections(upstream_bodyIds=None,
                                    downstream_bodyIds=None, **kwargs):
        rows = [(str(u), str(v), w) for (u, v, w) in fetch_rows]
        if not rows:
            return pd.DataFrame(columns=["bodyId_pre", "bodyId_post",
                                         "weight", "type_pre", "type_post"])
        return pd.DataFrame({
            "bodyId_pre": [r[0] for r in rows],
            "bodyId_post": [r[1] for r in rows],
            "weight": [r[2] for r in rows],
            "type_pre": [types[r[0]] for r in rows],
            "type_post": [types[r[1]] for r in rows],
        })

    def fake_enrich(conn_df, traversal_probability_threshold=0, dataset=None,
                    script_path=None, aggregate_method="product",
                    label_mapper=None, global_incoming_weights=None,
                    separate_hemispheres=False,
                    global_incoming_body_weights=None, **kwargs):
        conn_e = conn_df.copy()
        conn_e["connection_ratio"] = 0.5
        conn_e["traversal_probability"] = 0.5
        conn_t = (conn_e.groupby(["type_pre", "type_post"], as_index=False)
                  ["weight"].sum())
        conn_t["connection_ratio"] = 0.5
        conn_t["traversal_probability"] = 0.5
        return conn_e, conn_t, None

    monkeypatch.setattr(coana.sv, "EnrichConnectionTable", fake_enrich)

    source_ids = sorted({str(u) for (u, _, _) in fetch_rows})[:2] or ["S"]
    if target_ids is None:
        target_ids = sorted({str(v) for (_, v, _) in fetch_rows})[:2] or ["T"]
    fc, logs = make_fc(
        saveas="",
        output_dir=str(tmp_path),
        source_fname="src",
        target_fname="tgt",
        dataset="test:v1",
        max_interlayer=0,
        min_synapse_num=1,
        min_ratio=0.0,
        min_traversal_probability=0.0,
        parameter_dict={"k": "v"},
        parameter_df=pd.DataFrame({"p": [1]}),
        source_df=pd.DataFrame({"bodyId": source_ids,
                                "type": [types.get(i, "T" + str(i)) for i in source_ids]}),
        target_df=pd.DataFrame({"bodyId": target_ids,
                                "type": [types.get(i, "T" + str(i)) for i in target_ids]}),
        aggregate_method="product",
        label_mapper=None,
        separate_hemispheres=separate,
        keep_only_hemisphere_conserved_connections=keep_conserved,
        output_format="csv",
        largeTargetSet=False,
        script_path=str(PROJECT_ROOT),
    )
    fc._fetch_path_connections = fake_fetch_path_connections
    fc._fetch_total_incoming_weight_by_type = lambda *a, **k: None
    fc._fetch_total_incoming_weight = lambda *a, **k: None
    fc.VisualizeDirectConnections_simple = lambda: None
    return fc, logs


class TestTypePathRealizedWeightsAndMembershipN:
    """Round-10: per-hop path Weights scoped to the realized bodyId edges
    at each hop's position (not the pair's cross-layer total), and ONE
    canonical membership N shared by the ratio recompute's type_coverage
    and the allpaths coverage lists."""

    def test_realized_hop_weights_scope_to_position(self):
        # Pair A->B connects at two depths: edge (a1,b1) backs the 2-hop
        # sequence, edge (a2,b2) backs the 4-hop sequence. The streaming
        # join stamps BOTH masses (160+168=328) on every hop of the pair;
        # the realized walk must give each sequence only its own edges.
        label = {"s": "S", "m": "M", "a1": "A", "a2": "A",
                 "b1": "B", "b2": "B", "t": "T"}
        paths = [
            ["s", "a1", "b1"],             # S->A->B
            ["s", "m", "a2", "b2", "t"],   # S->M->A->B->T
        ]
        lut = {("s", "a1"): 5.0, ("a1", "b1"): 160.0,
               ("s", "m"): 7.0, ("m", "a2"): 3.0, ("a2", "b2"): 168.0,
               ("b2", "t"): 9.0}
        out = coana.FindNeuronConnection._type_path_realized_hop_weights(
            paths, label.get, lut)
        assert out[("S", "A", "B")] == [5.0, 160.0]
        assert out[("S", "M", "A", "B", "T")] == [7.0, 3.0, 168.0, 9.0]

    def test_realized_hop_weights_no_double_count_across_paths(self):
        # Two bodyId paths sharing one hop edge count that edge ONCE — the
        # same de-dup the per-node coverage n applies to bodyIds.
        label = {"a": "A", "b": "B", "c": "C"}
        paths = [["a", "b", "c"], ["a", "b", "c"]]
        lut = {("a", "b"): 4.0, ("b", "c"): 6.0}
        out = coana.FindNeuronConnection._type_path_realized_hop_weights(
            paths, label.get, lut)
        assert out[("A", "B", "C")] == [4.0, 6.0]

    def test_realized_hop_weights_missing_edge_yields_none(self):
        # Defensive contract: a realizing edge absent from the conn tables
        # marks the whole sequence None so the caller keeps the join value.
        label = {"a": "A", "b": "B"}
        out = coana.FindNeuronConnection._type_path_realized_hop_weights(
            [["a", "b"]], label.get, {})
        assert out[("A", "B")] is None

    def test_realized_ratio_readout_realized_shares(self):
        # Round-11: hop ratio = realized mass / the post type's
        # full-membership mass (the SAME denominator the pair-level
        # recompute uses); probability = min(1, r/0.3); product; min.
        out = coana.FindNeuronConnection._realized_ratio_readout(
            [41.0, 40.0], ("R8", "R7", "aMe6a"),
            {"R7": 66293.0, "aMe6a": 2181.0})
        ratios, probs, pp, mr = out
        assert abs(ratios[0] - 41.0 / 66293.0) < 1e-15
        assert abs(ratios[1] - 40.0 / 2181.0) < 1e-15
        assert probs == [min(1.0, ratios[0] / 0.3),
                         min(1.0, ratios[1] / 0.3)]
        assert abs(pp - probs[0] * probs[1]) < 1e-18
        assert mr == ratios[0]  # 41/66293 << 40/2181

    def test_realized_ratio_readout_missing_mass_yields_none(self):
        # Unknown post-type mass (or unrealized weights) keeps the
        # pair-level join values — never a fabricated share.
        assert coana.FindNeuronConnection._realized_ratio_readout(
            None, ("A", "B"), {"B": 100.0}) is None
        assert coana.FindNeuronConnection._realized_ratio_readout(
            [5.0], ("A", "B"), {}) is None
        assert coana.FindNeuronConnection._realized_ratio_readout(
            [5.0], ("A", "B"), {"B": 0.0}) is None


class TestRatioAdjColumns:
    """Round-12: connection_ratio_adj — the threshold-CONDITIONED share
    (denominator counts only edges at/above the run's Min Synapse
    Count), at both type and bodyId levels."""

    @staticmethod
    def _tot_frame(rows, key):
        return pd.DataFrame({
            key: [r[0] for r in rows],
            'total_incoming_weight': [float(r[1]) for r in rows],
        })

    def test_type_level_uses_threshold_totals(self):
        fc, _ = make_fc(min_synapse_num=3)
        # threshold-conditioned type totals: TT receives 50 above thr 3
        fc._fetch_total_incoming_weight_by_type = (
            lambda types, min_weight: self._tot_frame(
                [('TT', 50)], 'type_post'))
        df = pd.DataFrame({
            'type_pre': ['TA', 'TB'], 'type_post': ['TT', 'TT'],
            'weight': [10, 5],
            'connection_ratio': [0.1, 0.05],
        })
        out = fc._attach_ratio_adj_columns(df, 'type_pre', 'type_post')
        assert abs(out['connection_ratio_adj'].iloc[0] - 10 / 50) < 1e-12
        assert abs(out['connection_ratio_adj'].iloc[1] - 5 / 50) < 1e-12
        # the min_weight argument is the run's threshold, not 1
        fc._fetch_total_incoming_weight_by_type = (
            lambda types, min_weight: (
                self._tot_frame([('TT', 50)], 'type_post')
                if min_weight == 3 else pd.DataFrame()))
        out2 = fc._attach_ratio_adj_columns(df, 'type_pre', 'type_post')
        assert abs(out2['connection_ratio_adj'].iloc[0] - 10 / 50) < 1e-12

    def test_bodyid_level_uses_threshold_totals(self):
        fc, _ = make_fc(min_synapse_num=3)
        fc._fetch_total_incoming_weight = (
            lambda posts, min_weight: self._tot_frame(
                [('42', 25)], 'bodyId_post'))
        df = pd.DataFrame({
            'bodyId_pre': ['1'], 'bodyId_post': ['42'], 'weight': [5],
        })
        out = fc._attach_ratio_adj_columns(
            df, 'bodyId_pre', 'bodyId_post')
        assert abs(out['connection_ratio_adj'].iloc[0] - 5 / 25) < 1e-12

    def test_skip_at_threshold_one(self):
        # thr 1: the adjusted denominator IS the F9 one — no column.
        fc, _ = make_fc(min_synapse_num=1)
        fc._fetch_total_incoming_weight_by_type = (
            lambda types, min_weight: self._tot_frame([('TT', 100.0)],
                                                      'type_post'))
        df = pd.DataFrame({
            'type_pre': ['TA'], 'type_post': ['TT'], 'weight': [10],
            'connection_ratio': [0.1],
        })
        out = fc._attach_ratio_adj_columns(df, 'type_pre', 'type_post')
        # Round-13: ALWAYS emitted — at Min Synapse 1 the adjusted
        # denominator IS the threshold-free one, so adj == plain ratio
        # (a redundant duplicate by design, disclosed in the notes).
        assert abs(out['connection_ratio_adj'].iloc[0] - 0.1) < 1e-12
        # ratio basis without a synapse co-threshold — same redundancy
        fc2, _ = make_fc(min_synapse_num=1, weight_basis='connection_ratio')
        fc2._fetch_total_incoming_weight_by_type = (
            lambda types, min_weight: self._tot_frame([('TT', 100.0)],
                                                      'type_post'))
        out2 = fc2._attach_ratio_adj_columns(df, 'type_pre', 'type_post')
        assert abs(out2['connection_ratio_adj'].iloc[0] - 0.1) < 1e-12

    def test_inframe_fallback_when_fetch_fails(self):
        fc, _ = make_fc(min_synapse_num=3)

        def _boom(*a, **k):
            raise RuntimeError('offline')
        fc._fetch_total_incoming_weight_by_type = _boom
        df = pd.DataFrame({
            'type_pre': ['TA', 'TB', 'TC'], 'type_post': ['TT', 'TT', 'TS'],
            'weight': [10, 5, 7],
        })
        out = fc._attach_ratio_adj_columns(df, 'type_pre', 'type_post')
        # in-frame totals: TT 15, TS 7
        assert abs(out['connection_ratio_adj'].iloc[0] - 10 / 15) < 1e-12
        assert abs(out['connection_ratio_adj'].iloc[1] - 5 / 15) < 1e-12
        assert abs(out['connection_ratio_adj'].iloc[2] - 1.0) < 1e-12

    def test_invariant_adj_ge_plain(self):
        # the adjusted denominator counts a SUBSET of edges, so
        # adj >= plain holds row-wise (equal only when nothing is cut).
        fc, _ = make_fc(min_synapse_num=3)
        fc._fetch_total_incoming_weight_by_type = (
            lambda types, min_weight: self._tot_frame(
                [('TT', 60 if min_weight == 3 else 100)], 'type_post'))
        df = pd.DataFrame({
            'type_pre': ['TA'], 'type_post': ['TT'], 'weight': [10],
            'connection_ratio': [10 / 100],
        })
        out = fc._attach_ratio_adj_columns(df, 'type_pre', 'type_post')
        assert (out['connection_ratio_adj']
                >= out['connection_ratio']).all()

    def test_ratio_and_adj_uses_pair_level_numerator(self):
        # Round-15 (real MCNS ratio+AND run): the plain column is
        # PAIR-level after the recompute, so the thr-conditioned adj must
        # sum the pair's conn-layer slices for its numerator — a row
        # slice over a pair-level plain compared two granularities.
        fc, _ = make_fc(min_synapse_num=3,
                        weight_basis='connection_ratio')
        fc._fetch_total_incoming_weight_by_type = (
            lambda types, min_weight: self._tot_frame([('TB', 200.0)],
                                                      'type_post'))
        df = pd.DataFrame({
            'type_pre': ['TA', 'TA'], 'type_post': ['TB', 'TB'],
            'conn_layer': ['1->2', '2->3'], 'weight': [60, 40],
            'connection_ratio': [0.5, 0.5],
        })
        out = fc._attach_ratio_adj_columns(df, 'type_pre', 'type_post')
        # pair numerator 100 over the thr-3-conditioned 200 on BOTH rows
        assert abs(out['connection_ratio_adj'].iloc[0] - 0.5) < 1e-12
        assert abs(out['connection_ratio_adj'].iloc[1] - 0.5) < 1e-12

    def test_ratio_basis_reuses_recompute_mass(self):
        # Round-14: under the ratio basis at Min Synapse 1 the adj column
        # must equal connection_ratio BY CONSTRUCTION — even when a fresh
        # by-type fetch would disagree (the recompute settled on its
        # frame-membership fallback; a real hemibrain delegate hit this).
        fc, _ = make_fc(min_synapse_num=1,
                        weight_basis='connection_ratio')
        fc._ratio_lane_type_mass = {'TT': 100.0}

        def _flaky(types, min_weight):
            raise RuntimeError('transient fetch failure')
        fc._fetch_total_incoming_weight_by_type = _flaky
        df = pd.DataFrame({
            'type_pre': ['TA'], 'type_post': ['TT'], 'weight': [10],
            'connection_ratio': [0.1],
        })
        out = fc._attach_ratio_adj_columns(df, 'type_pre', 'type_post')
        assert abs(out['connection_ratio_adj'].iloc[0] - 0.1) < 1e-12

    def test_polars_branch(self):
        fc, _ = make_fc(min_synapse_num=3)
        fc._fetch_total_incoming_weight_by_type = (
            lambda types, min_weight: self._tot_frame(
                [('TT', 50)], 'type_post'))
        df = pl.DataFrame({
            'type_pre': ['TA'], 'type_post': ['TT'], 'weight': [10],
        })
        out = fc._attach_ratio_adj_columns(df, 'type_pre', 'type_post')
        assert isinstance(out, pl.DataFrame)
        assert abs(out['connection_ratio_adj'][0] - 0.2) < 1e-12

    def test_network_type_membership_counts_union(self):
        # N = bodyIds on emitted paths ∪ discovery layers, counted per
        # final type label (duplicates collapse).
        label = {"a1": "A", "a2": "A", "b": "B", "c": "C"}
        counts = coana.FindNeuronConnection._network_type_membership_counts(
            [["a1", "b"]], [["a1", "a2"], ["b"], []], label.get)
        assert counts == {"A": 2, "B": 1}

    def test_recompute_type_coverage_uses_canonical_n(self):
        fc, _ = make_fc(
            weight_basis="connection_ratio",
            min_ratio=0.01,
            _ratio_lane_totals={"b1": 100.0, "b2": 100.0},
            _ratio_lane_type_members={"A": ["a1"], "B": ["b1", "b2"]},
        )

        def _fail_fetch(types, min_weight):
            raise RuntimeError("dataset fetch unavailable in test")

        fc._fetch_total_incoming_weight_by_type = _fail_fetch
        conn_types = pd.DataFrame({
            "type_pre": ["A"], "type_post": ["B"], "weight": [50],
            "connection_ratio": [0.25], "traversal_probability": [0.8],
            "block_probability": [0.2],
        })
        conn_inpath = pd.DataFrame({
            "type_pre": ["A"], "type_post": ["B"],
            "bodyId_post": ["b1"], "weight": [50],
        })
        out = fc._recompute_ratio_lane_type_ratios(
            conn_types, conn_inpath, total_n_map={"B": 5})
        # canonical N wins over the frame membership (2 members)
        assert out["type_coverage"].iloc[0] == "1/5"
        # numerator/denominator untouched: 50 / (100+100)
        assert abs(out["connection_ratio"].iloc[0] - 0.25) < 1e-12

    def test_recompute_numerator_dedupes_recurring_pairs(self):
        # Round-15: conn_inpath keeps ALL layer-specific occurrences; the
        # recompute's numerator must count a recurring physical edge ONCE
        # (the exported weight column's semantics). Real-data exposure:
        # SMP053->PPL101 summed 307 vs the table's 187.
        fc, _ = make_fc(
            weight_basis='connection_ratio',
            min_ratio=0.01,
            _ratio_lane_totals={'b1': 100.0},
            _ratio_lane_type_members={'A': ['a1'], 'B': ['b1']},
        )
        fc._fetch_total_incoming_weight_by_type = (
            lambda types, min_weight: self._tot_frame([('B', 100.0)],
                                                      'type_post'))
        conn_types = pd.DataFrame({
            'type_pre': ['A'], 'type_post': ['B'], 'weight': [50],
            'connection_ratio': [0.5], 'traversal_probability': [1.0],
            'block_probability': [0.0],
        })
        # the SAME bodyId edge recurring at two depths (307-vs-187 class)
        conn_inpath = pd.DataFrame({
            'type_pre': ['A', 'A'], 'type_post': ['B', 'B'],
            'bodyId_pre': ['a1', 'a1'], 'bodyId_post': ['b1', 'b1'],
            'weight': [30, 25],
        })
        out = fc._recompute_ratio_lane_type_ratios(conn_types, conn_inpath)
        assert abs(out['connection_ratio'].iloc[0] - 30 / 100) < 1e-12

    def test_recompute_type_coverage_falls_back_to_frame_members(self):
        fc, _ = make_fc(
            weight_basis="connection_ratio",
            min_ratio=0.01,
            _ratio_lane_totals={"b1": 100.0},
            _ratio_lane_type_members={"A": ["a1"], "B": ["b1"]},
        )
        fc._fetch_total_incoming_weight_by_type = (
            lambda types, min_weight: pd.DataFrame())
        conn_types = pd.DataFrame({
            "type_pre": ["A"], "type_post": ["B"], "weight": [50],
            "connection_ratio": [0.5], "traversal_probability": [1.0],
            "block_probability": [0.0],
        })
        conn_inpath = pd.DataFrame({
            "type_pre": ["A"], "type_post": ["B"],
            "bodyId_post": ["b1"], "weight": [50],
        })
        out = fc._recompute_ratio_lane_type_ratios(conn_types, conn_inpath)
        assert out["type_coverage"].iloc[0] == "1/1"
        assert abs(out["connection_ratio"].iloc[0] - 0.5) < 1e-12


class TestFindDirectConnections:
    def test_full_run_csv_outputs(self, monkeypatch, tmp_path):
        types = {"S": "TS", "A": "TA", "X": "TX"}
        fc, _ = _make_direct_fc(
            monkeypatch, tmp_path,
            [("S", "A", 10), ("S", "X", 50)],  # X outside target set
            types=types, target_ids=["A"])
        result = fc.FindDirectConnections()
        assert result == 0

        folder = Path(fc.direct_folder)
        assert folder.name.startswith("finddirect_")
        params = (folder / "parameters.txt").read_text()
        assert "k: v" in params
        # enrollment files at run root
        assert (folder / "source_neurons.csv").exists()
        assert (folder / "target_neurons.csv").exists()
        details = folder / "data_details"
        base = "src_to_tgt_info_snp1"
        for suffix in ("_parameters.csv", "_source_info.csv",
                       "_target_info.csv", "_source_in_connection.csv",
                       "_target_in_connection.csv",
                       "_connection_groupby_type.csv",
                       "_connectionMatrix_type.csv", "_connMat_type_full.csv",
                       "_transmissionMat_type.csv",
                       "_connectionRatioMat_type.csv",
                       "_ratioMat_type_full.csv"):
            assert (details / (base + suffix)).exists(), suffix
        body_csv = folder / "src_to_tgt_bodyId_connections_snp1.csv"
        assert body_csv.exists()
        body = pd.read_csv(body_csv)
        # X is not a target: only S->A survives
        assert set(zip(body["bodyId_pre"].astype(str),
                       body["bodyId_post"].astype(str))) == {("S", "A")}
        assert (folder / "src_to_tgt_connectionMatrix_bodyId.csv").exists()
        # target enrollment marks A checked at layer 1
        tgt = pd.read_csv(folder / "target_neurons.csv")
        assert tgt.loc[tgt["bodyId"].astype(str) == "A", "Layer"].iloc[0] == 1

    def test_empty_result_early_return(self, monkeypatch, tmp_path):
        fc, _ = _make_direct_fc(monkeypatch, tmp_path, [], types={"S": "TS"})
        result = fc.FindDirectConnections()
        assert result is None  # early return path
        folder = Path(fc.direct_folder)
        assert (folder / "source_neurons.csv").exists()
        assert not (folder / "data_details").exists()

    def test_hemisphere_conserved_filtering(self, monkeypatch, tmp_path):
        # TS_L->TA_L mirrors TS_R->TA_R (conserved pair); TS_L->TB_L has no
        # TS_R->TB_R counterpart (unconserved -> exported).
        types = {"S1": "TS_L", "S2": "TS_R", "A": "TA_L", "B": "TA_R",
                 "C": "TB_L"}
        rows = [("S1", "A", 10), ("S2", "B", 9), ("S1", "C", 8)]
        fc, _ = _make_direct_fc(
            monkeypatch, tmp_path, rows, types=types,
            keep_conserved=True, separate=True)
        fc.source_df = pd.DataFrame({"bodyId": ["S1", "S2"],
                                     "type": ["TS_L", "TS_R"]})
        fc.target_df = pd.DataFrame({"bodyId": ["A", "B", "C"],
                                     "type": ["TA_L", "TA_R", "TB_L"]})
        fc.FindDirectConnections()
        details = Path(fc.direct_folder) / "data_details"
        type_csv = details / "src_to_tgt_info_snp1_connection_groupby_type.csv"
        kept = pd.read_csv(type_csv)
        unconserved = details / "hemisphere_unconserved_edges.csv"
        assert unconserved.exists()
        assert len(kept) == 2
        assert len(pd.read_csv(unconserved)) == 1


    def test_untyped_edges_dropped_from_direct_outputs(self, monkeypatch, tmp_path):
        """REPO-2 (2026-09-25): FindDirectConnections was the one coana entry
        point without the drop_untyped filter — untyped edges entered its
        matrices and summaries. The shared predicate now runs after the
        target-set filter (before the enrollment marks), and the removal is
        published like every other tool (warning note + records CSV)."""
        types = {"S": "TS", "A": "TA", "U": "Unknown"}
        fc, _ = _make_direct_fc(
            monkeypatch, tmp_path,
            [("S", "A", 10), ("S", "U", 5)],
            types=types, target_ids=["A", "U"])
        assert fc.FindDirectConnections() == 0
        folder = Path(fc.direct_folder)
        body = pd.read_csv(folder / "src_to_tgt_bodyId_connections_snp1.csv")
        got = set(zip(body["bodyId_pre"].astype(str),
                      body["bodyId_post"].astype(str)))
        assert got == {("S", "A")}, "the untyped target edge is gone"
        # enrollment describes the analysis set: U was never 'checked'
        tgt = pd.read_csv(folder / "target_neurons.csv")
        checked = set(tgt.loc[tgt["Checked"].astype(bool), "bodyId"].astype(str))
        assert checked == {"A"}
        # the removal is published like the path/network tools
        rec = pd.read_csv(folder / "data_details" / "untyped_dropped_records.csv",
                          dtype=str)
        assert set(rec["bodyId_post"]) == {"U"}
        notes = (folder / "user_warning_notes.txt").read_text()
        assert "untyped dropped" in notes

# =============================================================================
# FindPath offline pipeline (layer discovery + path reconstruction)
# =============================================================================

class _FakeVisualizePath:
    """Records every VisualizePath construction; visualize() is a no-op."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.edge_limit_trimmed = False
        _FakeVisualizePath.instances.append(self)

    def visualize(self, **kwargs):
        pass


def _make_path_fc(monkeypatch, tmp_path, edges, types, *, max_interlayer=5,
                  source_ids=("S",), target_ids=("T",), skip_bodyId=False,
                  min_synapse=1):
    """Wire a FindNeuronConnection for an offline FindPath run.

    Connection layers are served from the static ``edges`` list filtered by
    the requested upstream set; neuron metadata comes from ``types``.
    """
    edge_set = [(str(u), str(v), w) for (u, v, w) in edges]

    def fake_fetch(upstream_bodyIds=None, downstream_bodyIds=None, **kwargs):
        ups = {str(u) for u in (upstream_bodyIds or [])}
        rows = [(u, v, w) for (u, v, w) in edge_set if u in ups]
        if not rows:
            return pd.DataFrame(columns=["bodyId_pre", "bodyId_post",
                                         "weight", "type_pre", "type_post"])
        return pd.DataFrame({
            "bodyId_pre": [r[0] for r in rows],
            "bodyId_post": [r[1] for r in rows],
            "weight": [r[2] for r in rows],
            "type_pre": [types.get(r[0], r[0]) for r in rows],
            "type_post": [types.get(r[1], r[1]) for r in rows],
        })

    def fake_enrich(conn, traversal_probability_threshold=0, dataset=None,
                    script_path=None, target_neurons_df=None, label_mapper=None,
                    global_incoming_weights=None, separate_hemispheres=False,
                    **kwargs):
        conn_e = conn.copy()
        conn_e["connection_ratio"] = 0.5
        conn_e["traversal_probability"] = 0.5
        conn_t = (conn_e.groupby(["type_pre", "type_post"], as_index=False)
                  ["weight"].sum())
        conn_t["connection_ratio"] = 0.5
        conn_t["traversal_probability"] = 0.5
        return conn_e, conn_t, None

    _FakeVisualizePath.instances = []
    monkeypatch.setattr(coana.sv, "EnrichConnectionTable", fake_enrich)
    monkeypatch.setattr(coana, "VisualizePath", _FakeVisualizePath)

    fc, logs = make_fc(
        saveas="",
        save_folder=str(tmp_path / "base"),
        output_dir=str(tmp_path),
        source_fname="src",
        target_fname="tgt",
        dataset="test:v1",
        max_interlayer=max_interlayer,
        min_synapse_num=min_synapse,
        min_ratio=0.0,
        min_traversal_probability=0.0,
        parameter_dict={"k": "v"},
        parameter_df=pd.DataFrame({"p": [1]}),
        source_df=pd.DataFrame({"bodyId": list(source_ids),
                                "type": [types.get(s, s) for s in source_ids]}),
        target_df=pd.DataFrame({"bodyId": list(target_ids),
                                "type": [types.get(t, t) for t in target_ids]}),
        sourceNeurons=list(source_ids),
        targetNeurons=list(target_ids),
        separate_hemispheres=False,
        output_format="csv",
        script_path=str(tmp_path),
        graph_edge_limit_groups=0,
        graph_edge_limit_bodyid=None,
        edgeN_limit=0,
        network_layout="hierarchical",
        showfig=False,
        skip_bodyId=skip_bodyId,
        keyword_in_path_to_remove=["None"],
        label_mapper=None,
        node_color="#1f77b4",
        target_color="#d62728",
        link_color="rgba(100,100,100,0.3)",
    )
    os.makedirs(fc.save_folder, exist_ok=True)
    fc._fetch_path_connections = fake_fetch
    fc._fetch_neurons_local_or_api = lambda ids, columns=None: pd.DataFrame({
        "bodyId": [str(b) for b in ids],
        "type": [types.get(str(b), str(b)) for b in ids],
        "post": [100] * len(ids),
    })
    fc._fetch_total_incoming_weight_by_type = lambda *a, **k: None
    fc._fetch_total_incoming_weight = lambda *a, **k: None
    fc._ensure_neuprint_client = lambda: None
    fc._fetch_neurons_batched = lambda ids: pd.DataFrame({
        "bodyId": [str(b) for b in ids],
        "type": [types.get(str(b), str(b)) for b in ids],
        "instance": [str(b) + "_i" for b in ids],
    })
    return fc, logs


class TestFindPathPipeline:
    CHAIN = [("S", "A", 10), ("A", "B", 8), ("B", "T", 6)]
    TYPES = {"S": "TS", "A": "TA", "B": "TB", "T": "TT"}

    def test_full_run_csv_outputs(self, monkeypatch, tmp_path):
        fc, _ = _make_path_fc(monkeypatch, tmp_path, self.CHAIN, self.TYPES)
        fc.FindPath()

        folder = Path(fc.path_folder)
        assert folder.name.startswith("find-paths-complete_")
        assert "r0p0" not in folder.name
        assert (folder / "source_neurons.csv").exists()
        assert (folder / "target_neurons.csv").exists()
        assert (folder / "all_attributes.json").exists()
        params = (folder / "parameters.txt").read_text()
        assert "Analysis Parameters for FindPath" in params

        # type-level path output at run root
        path_type = folder / "src_to_tgt_path_type.csv"
        assert path_type.exists()
        ptdf = pd.read_csv(path_type)
        assert len(ptdf) == 1

        # bodyId-level outputs (skip_bodyId=False -> find_bodyId_path True)
        path_body = folder / "src_to_tgt_path_bodyId.csv"
        assert path_body.exists()
        details = folder / "data_details"
        bodyid = pd.read_csv(details / "connection_info_bodyId.csv")
        assert set(zip(bodyid["bodyId_pre"].astype(str),
                       bodyid["bodyId_post"].astype(str))) == {
            ("S", "A"), ("A", "B"), ("B", "T")}
        # interlayer neuron sheets (one per hop)
        for i in (1, 2, 3):
            assert (details / f"layer_{i}.csv").exists()
        assert (details / "parameters.csv").exists()
        assert (details / "total_weight_layer.csv").exists()
        assert (details / "connection_type.csv").exists()

        # enrollment status columns
        tgt = pd.read_csv(folder / "target_neurons.csv")
        assert tgt.loc[tgt["bodyId"].astype(str) == "T", "Checked"].iloc[0] in (True, "True")
        src = pd.read_csv(folder / "source_neurons.csv")
        assert src.loc[src["bodyId"].astype(str) == "S", "isInPath"].iloc[0] in (True, "True")
        # both type-level and bodyId-level visualizations constructed
        assert len(_FakeVisualizePath.instances) == 2

    def test_frontier_dried_no_paths(self, monkeypatch, tmp_path):
        # T is unreachable: only S->A exists.
        fc, _ = _make_path_fc(
            monkeypatch, tmp_path, [("S", "A", 10)], self.TYPES)
        # No path connects source to target: FindPath completes gracefully —
        # the type-level analysis is skipped when conn_types has no columns
        # (previously a KeyError('type_pre') crashed the run here).
        fc.FindPath()
        assert fc._depth_cap_reached is False
        folder = Path(fc.path_folder)
        # enrollment + type-level CSVs are still saved
        assert (folder / "source_neurons.csv").exists()
        assert (folder / "data_details" / "connection_type.csv").exists()

    def test_depth_cap_reached(self, monkeypatch, tmp_path):
        # The chain needs 3 hops; depth cap of 1 leaves T untraced with a
        # live frontier -> the run records a truncated search.
        fc, _ = _make_path_fc(
            monkeypatch, tmp_path, self.CHAIN, self.TYPES, max_interlayer=1)
        # No complete path exists; the run now completes without crashing.
        fc.FindPath()
        assert fc._depth_cap_reached is True

    def test_skip_bodyid_flag(self, monkeypatch, tmp_path):
        fc, _ = _make_path_fc(
            monkeypatch, tmp_path, self.CHAIN, self.TYPES, skip_bodyId=True)
        fc.FindPath(find_bodyId_path=False)
        folder = Path(fc.path_folder)
        assert (folder / "src_to_tgt_path_type.csv").exists()
        assert not (folder / "src_to_tgt_path_bodyId.csv").exists()
        # only the type-level visualization is constructed
        assert len(_FakeVisualizePath.instances) == 1


# =============================================================================
# _enrich_connections_with_neuron_info (local table + API fallback)
# =============================================================================

class TestEnrichConnectionsWithNeuronInfo:
    DATASET = "hemibrain:v1.2.1"
    SAFE = "hemibrain_v1_2_1"

    def _fc(self, tmp_path, *, use_cache=True):
        fc, logs = make_fc(
            dataset=self.DATASET,
            script_path=str(tmp_path),
            use_cache=use_cache,
            separate_hemispheres=False,
        )
        fc._local_neuron_df_cache = {}
        fetch_calls = []

        def fake_fetch(ids, columns=None):
            fetch_calls.append(list(ids))
            return pd.DataFrame({
                "bodyId": [str(i) for i in ids],
                "type": ["API_" + str(i) for i in ids],
                "instance": ["APII_" + str(i) for i in ids],
            })

        fc._fetch_neurons_local_or_api = fake_fetch
        fc._fetch_calls = fetch_calls
        return fc

    def _write_local_table(self, tmp_path, bodyids, *, nt=True):
        d = tmp_path / "datasets" / self.SAFE
        d.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "bodyId": [str(b) for b in bodyids],
            "type": ["LOC_" + str(b) for b in bodyids],
            "instance": ["LOCI_" + str(b) for b in bodyids],
            "hemisphere": ["R"] * len(bodyids),
        })
        if nt:
            df["nt_type"] = ["ACh"] * len(bodyids)
        df.to_csv(d / f"{self.SAFE}_allneurons_neuron_df.csv", index=True)

    def test_empty_frame_returns_unchanged(self, tmp_path):
        fc = self._fc(tmp_path)
        out = fc._enrich_connections_with_neuron_info(pd.DataFrame())
        assert out.empty

    def test_online_only_fetches_from_api(self, tmp_path):
        fc = self._fc(tmp_path, use_cache=False)
        out = fc._enrich_connections_with_neuron_info(
            conn_df([("1", "2", 5), ("2", "3", 3)]))
        assert len(fc._fetch_calls) == 1
        assert out.loc[0, "type_pre"] == "API_1"
        assert out.loc[0, "type_post"] == "API_2"
        assert out.loc[1, "type_post"] == "API_3"
        # hemisphere columns ensured on the API frame
        assert "hemisphere_pre" in out.columns

    def test_local_csv_enrichment_with_nt(self, tmp_path):
        self._write_local_table(tmp_path, ["1", "2", "3"])
        fc = self._fc(tmp_path)
        out = fc._enrich_connections_with_neuron_info(
            conn_df([("1", "2", 5), ("2", "3", 3)]))
        assert out.loc[0, "type_pre"] == "LOC_1"
        assert out.loc[1, "type_post"] == "LOC_3"
        assert out.loc[0, "nt_type_pre"] == "ACh"
        assert out.loc[0, "instance_post"] == "LOCI_2"
        assert fc._fetch_calls == []  # nothing fetched from API

    def test_missing_bodyids_fetched_from_api(self, tmp_path):
        # Local table only knows 1 and 2; neuron 9 must come from the API.
        self._write_local_table(tmp_path, ["1", "2"])
        fc = self._fc(tmp_path)
        out = fc._enrich_connections_with_neuron_info(conn_df([("1", "9", 4)]))
        assert len(fc._fetch_calls) == 1
        assert sorted(fc._fetch_calls[0]) == ["9"]
        assert out.loc[0, "type_pre"] == "LOC_1"
        assert out.loc[0, "type_post"] == "API_9"

    def test_custom_group_columns_merged(self, tmp_path):
        self._write_local_table(tmp_path, ["1", "2"])
        fc = self._fc(tmp_path)
        fc.source_df = pd.DataFrame({"bodyId": ["1"], "custom_group": ["G_src"]})
        fc.target_df = pd.DataFrame({"bodyId": ["2"], "custom_group": ["G_tgt"]})
        # pre-existing enrichment columns are dropped before the merge
        df = conn_df([("1", "2", 7)])
        df["type_pre"] = "stale"
        df["type_post"] = "stale"
        out = fc._enrich_connections_with_neuron_info(df)
        assert out.loc[0, "type_pre"] == "LOC_1"
        assert out.loc[0, "custom_group_pre"] == "G_src"
        assert out.loc[0, "custom_group_post"] == "G_tgt"


# =============================================================================
# _fetch_connections_with_cave_api (mocked CAVEDataFetcher + API cache)
# =============================================================================

class TestFetchConnectionsWithCaveApi:
    UP = "720575940614131061"
    DOWN1 = "720575940614131062"
    DOWN2 = "720575940614131063"

    def _install_fake_cave(self, monkeypatch, *, behavior="ok"):
        import types as _types

        fetcher_log = []

        class FakeCAVEDataFetcher:
            def __init__(self, dataset=None, materialization_version=None,
                         cache_enabled=False, verbose=False):
                fetcher_log.append(("init", dataset, materialization_version,
                                    cache_enabled, verbose))

            def fetch_connections(self, ids, direction="pre"):
                fetcher_log.append(("fetch", tuple(ids), direction))
                if behavior == "importerror":
                    raise ImportError("caveclient missing")
                if behavior == "error":
                    raise RuntimeError("CAVE server unreachable")
                return pd.DataFrame({
                    "pre_pt_root_id": [int(self.UP)] * 2,
                    "post_pt_root_id": [int(self.DOWN1), int(self.DOWN2)],
                    "weight": [12, 2],
                })

        FakeCAVEDataFetcher.UP = self.UP
        FakeCAVEDataFetcher.DOWN1 = self.DOWN1
        FakeCAVEDataFetcher.DOWN2 = self.DOWN2
        mod = _types.ModuleType("cave_data_fetcher")
        mod.CAVEDataFetcher = FakeCAVEDataFetcher
        monkeypatch.setitem(sys.modules, "cave_data_fetcher", mod)
        return fetcher_log

    def _fc(self, tmp_path, **attrs):
        base = dict(
            dataset="fafb:v1",
            script_path=str(tmp_path),
            use_cache=True,
            version=1,
            min_synapse_num=1,
            min_ratio=0.0,
            min_traversal_probability=0.0,
            exclude_intra_type_connections=False,
            label_mapper=None,
        )
        base.update(attrs)
        fc, logs = make_fc(**base)
        fc._cave_fetcher = None
        # Isolate the CAVE fetch from the enrichment engine (tested above).
        def fake_enrich(conn_df):
            conn_df = conn_df.copy()
            conn_df["type_pre"] = "PT_" + conn_df["bodyId_pre"].astype(str).str[-1]
            conn_df["type_post"] = "PT_" + conn_df["bodyId_post"].astype(str).str[-1]
            return conn_df
        fc._enrich_connections_with_neuron_info = fake_enrich
        return fc

    def test_fetch_caches_and_serves_from_cache(self, monkeypatch, tmp_path):
        log = self._install_fake_cave(monkeypatch)
        fc = self._fc(tmp_path)
        out = fc._fetch_connections_with_cave_api([self.UP])
        assert len(out) == 2
        assert set(out["bodyId_post"]) == {self.DOWN1, self.DOWN2}
        assert out.loc[0, "type_pre"].startswith("PT_")
        # fetcher constructed once via _get_cave_fetcher, one API fetch
        assert log[0][0] == "init" and log[0][3] is False
        assert len([e for e in log if e[0] == "fetch"]) == 1
        # API cache written under the hermetic script_path
        cache_dir = tmp_path / "cache" / "fafb_v1" / "API_cache"
        assert (cache_dir / "connections.parquet").exists()
        assert (cache_dir / "neuron_index.parquet").exists()

        # Second call: neuron already in the API cache -> no new API fetch.
        out2 = fc._fetch_connections_with_cave_api([self.UP])
        assert len(out2) == 2
        assert len([e for e in log if e[0] == "fetch"]) == 1

    def test_downstream_filter(self, monkeypatch, tmp_path):
        self._install_fake_cave(monkeypatch)
        fc = self._fc(tmp_path, use_cache=False)
        out = fc._fetch_connections_with_cave_api(
            [self.UP], downstream_bodyIds=[self.DOWN1])
        assert len(out) == 1
        assert out.iloc[0]["bodyId_post"] == self.DOWN1

    def test_min_weight_excludes_and_sets_flag(self, monkeypatch, tmp_path):
        self._install_fake_cave(monkeypatch)
        fc = self._fc(tmp_path, use_cache=False, min_synapse_num=5)
        out = fc._fetch_connections_with_cave_api([self.UP])
        # weight>=5 keeps only the weight-12 edge; the weight-2 edge is gone
        assert len(out) == 1
        assert out.iloc[0]["bodyId_post"] == self.DOWN1
        assert fc._min_synapse_excluded is True

    def test_intra_type_exclusion_and_label_mapper(self, monkeypatch, tmp_path):
        self._install_fake_cave(monkeypatch)

        class FakeLabelMapper:
            def apply_to_dataframe(self, df, dataset):
                df = df.copy()
                # row 0 maps both ends to the same label -> intra-type;
                # row 1 keeps distinct labels and survives.
                df["std_label_pre"] = ["STD", "STD"]
                df["std_label_post"] = ["STD", "SAME2"]
                return df

        fc = self._fc(tmp_path, use_cache=False,
                      exclude_intra_type_connections=True,
                      label_mapper=FakeLabelMapper())

        def same_type_enrich(conn_df):
            conn_df = conn_df.copy()
            conn_df["type_pre"] = "SAME"
            conn_df["type_post"] = ["SAME", "OTHER"]
            return conn_df

        fc._enrich_connections_with_neuron_info = same_type_enrich
        out = fc._fetch_connections_with_cave_api([self.UP])
        # label mapping applied first, then the intra-type row removed
        assert len(out) == 1
        assert out.iloc[0]["type_pre"] == "STD"
        assert out.iloc[0]["type_post"] == "SAME2"

    def test_importerror_returns_empty_frame(self, monkeypatch, tmp_path):
        self._install_fake_cave(monkeypatch, behavior="importerror")
        fc = self._fc(tmp_path, use_cache=False)
        out = fc._fetch_connections_with_cave_api([self.UP])
        assert out.empty
        assert list(out.columns) == ["bodyId_pre", "bodyId_post", "weight", "roi"]

    def test_generic_error_returns_empty_frame(self, monkeypatch, tmp_path):
        self._install_fake_cave(monkeypatch, behavior="error")
        fc = self._fc(tmp_path, use_cache=False)
        out = fc._fetch_connections_with_cave_api([self.UP])
        assert out.empty
        assert "type_pre" in out.columns


# =============================================================================
# VisualizeDirectConnections_simple (fake VisualizePath recorder)
# =============================================================================

class TestVisualizeDirectConnectionsSimple:
    def _fc(self, monkeypatch, tmp_path, *, conn_type=None, conn_df=None,
            conn_group=None, raise_on_visualize=False):
        calls = []

        class FakeVP:
            def __init__(self, path_file=None, output_folder=None, **kwargs):
                self.path_file = path_file
                self.output_folder = output_folder
                self.edge_limit_trimmed = False
                calls.append(self)

            def visualize(self, **kwargs):
                if raise_on_visualize:
                    raise RuntimeError("plotly export failed")

        monkeypatch.setattr(coana, "VisualizePath", FakeVP)
        fc, logs = make_fc(
            conn_type=conn_type if conn_type is not None else pd.DataFrame(),
            conn_df=conn_df if conn_df is not None else pd.DataFrame(),
            conn_group=conn_group,
            direct_folder=str(tmp_path),
            showfig=False,
            edgeN_limit=0,
            network_layout="hierarchical",
            output_format="csv",
        )
        return fc, logs, calls

    def test_type_bodyid_and_group_visualizations(self, monkeypatch, tmp_path):
        conn_type = pd.DataFrame({
            "type_pre": ["TS", "TS"], "type_post": ["TA", "TB"],
            "weight": [10, 5], "connection_ratio": [0.4, 0.2],
            "traversal_probability": [0.6, 0.3],
        })
        conn_df = pd.DataFrame({
            "bodyId_pre": ["1", "1"], "bodyId_post": ["2", "3"],
            "weight": [10, 5], "connection_ratio": [0.4, 0.2],
            "traversal_probability": [0.6, 0.3],
            "type_pre": ["TS", "TS"], "type_post": ["TA", "TB"],
            "hemisphere_code_pre": ["R", "R"],
            "hemisphere_code_post": ["R", "L"],
        })
        conn_group = pd.DataFrame({
            "custom_group_pre": ["G1"], "custom_group_post": ["G2"],
            "weight": [9], "connection_ratio": [0.5],
            "traversal_probability": [0.5],
        })
        fc, logs, calls = self._fc(
            monkeypatch, tmp_path, conn_type=conn_type, conn_df=conn_df,
            conn_group=conn_group)
        fc.VisualizeDirectConnections_simple()
        assert len(calls) == 3
        type_blocks = calls[0].path_file["path_block"].tolist()
        assert "TS -> TA" in type_blocks
        body_blocks = calls[1].path_file["path_block"].tolist()
        assert any(b.startswith("1_TS") for b in body_blocks)
        group_blocks = calls[2].path_file["path_block"].tolist()
        assert group_blocks == ["G1 -> G2"]
        assert calls[1].output_folder.endswith("bodyId_visualization")
        assert calls[2].output_folder.endswith("custom_groups")

    def test_no_connections_no_visualization(self, monkeypatch, tmp_path):
        fc, logs, calls = self._fc(monkeypatch, tmp_path)
        fc.VisualizeDirectConnections_simple()
        assert calls == []
        assert any("No connections to visualize" in m for m in logs)

    def test_visualization_failure_is_swallowed(self, monkeypatch, tmp_path):
        conn_type = pd.DataFrame({
            "type_pre": ["TS"], "type_post": ["TA"], "weight": [10],
            "connection_ratio": [0.4], "traversal_probability": [0.6],
        })
        fc, logs, calls = self._fc(
            monkeypatch, tmp_path, conn_type=conn_type,
            raise_on_visualize=True)
        fc.VisualizeDirectConnections_simple()  # must not raise
        assert any("visualization failed" in m for m in logs)


# =============================================================================
# cache loaders / warm-up / cache status on real tmp_path parquet caches
# =============================================================================

def _cache_fc(tmp_path, **attrs):
    """FindNeuronConnection wired to a tmp_path-backed cache layout."""
    cache_folder = tmp_path / "cache" / "test_v1"
    cache_folder.mkdir(parents=True, exist_ok=True)
    fc, logs = make_fc(
        dataset="test:v1",
        script_path=str(tmp_path),
        cache_folder=str(cache_folder),
        use_cache=True,
        client_type="neuprint",
        _conn_df_cache=None,
        _conn_index=None,
        _conn_index_post=None,
        _conn_db_pre_id_cache=None,
        _conn_cache_signature=None,
        _neuron_index_cache=None,
        _neuron_index_dict=None,
        _neuron_index_signature_value=None,
        _local_neuron_df_cache=dict(),
    )
    for key, value in attrs.items():
        setattr(fc, key, value)
    return fc, logs


def _write_conn_parquet(fc, pre, post, weight):
    pl.DataFrame({
        "bodyId_pre": pre,
        "bodyId_post": post,
        "weight": weight,
        "roi": ["r"] * len(pre),
        "cached_date": ["2026-01-01"] * len(pre),
    }).write_parquet(fc._get_connection_db_path())


def _write_index_parquet(fc, rows):
    path = fc._get_neuron_index_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


class TestCacheLoaders:
    def test_load_connection_db_and_build_indexes(self, tmp_path):
        fc, logs = _cache_fc(tmp_path)
        _write_conn_parquet(fc, ["1", "1", "2"], ["2", "3", "3"], [5, 6, 7])
        db = fc._load_connection_db()
        assert db.height == 3
        assert fc._conn_index["1"] == [0, 1]
        assert fc._conn_index_post["3"] == [1, 2]
        # second call with matching signature -> served from memory
        fc._conn_cache_signature = fc._connection_cache_signature()
        db2 = fc._load_connection_db()
        assert db2 is db
        # warm-up reports the loaded cache once the neuron index is ready
        _write_index_parquet(fc, {
            "bodyId": ["1"], "type": ["T1"], "instance": ["I1"], "post": [10],
            "downstream_complete": [True], "last_fetched": [""],
            "connection_count": [2],
        })
        fc._load_neuron_index()
        status = fc.warm_up_cache(quiet=True)
        assert status["connections_loaded"] == 3
        assert status["index_ready"] is True

    def test_load_connection_db_includes_batch_files(self, tmp_path):
        fc, _ = _cache_fc(tmp_path)
        _write_conn_parquet(fc, ["1"], ["2"], [5])
        batch_dir = Path(fc.cache_folder) / "_batch_files"
        batch_dir.mkdir(parents=True)
        pl.DataFrame({
            "bodyId_pre": ["3"], "bodyId_post": ["4"], "weight": [9],
            "roi": ["r"], "cached_date": ["2026-01-01"],
        }).write_parquet(batch_dir / "batch_0001.parquet")
        fc._conn_df_cache = None
        db = fc._load_connection_db()
        assert db.height == 2
        assert set(db["bodyId_pre"].to_list()) == {"1", "3"}

    def test_load_connection_db_no_cache_mode(self, tmp_path):
        fc, _ = _cache_fc(tmp_path, use_cache=False)
        db = fc._load_connection_db()
        assert db.is_empty()
        assert fc._conn_index == {}

    def test_load_connection_db_pandas_memory_frame_converted(self, tmp_path):
        fc, _ = _cache_fc(tmp_path)
        fc._conn_df_cache = pd.DataFrame({
            "bodyId_pre": ["1"], "bodyId_post": ["2"], "weight": [5],
            "roi": ["r"], "cached_date": ["2026-01-01"],
        })
        # matching signature: signature None + empty on-disk cache
        fc._conn_cache_signature = fc._connection_cache_signature()
        db = fc._load_connection_db()
        assert isinstance(db, pl.DataFrame)
        assert db.height == 1

    def test_load_neuron_index_and_dict(self, tmp_path):
        fc, _ = _cache_fc(tmp_path)
        _write_index_parquet(fc, {
            "bodyId": ["1", "2"], "type": ["T1", "T2"],
            "instance": ["I1", "I2"], "post": [10, 20],
            "downstream_complete": [True, False],
            "last_fetched": ["", ""], "connection_count": [1, 0],
        })
        idx = fc._load_neuron_index()
        assert len(idx) == 2
        assert fc._neuron_index_dict["1"]["type"] == "T1"
        # warm-up sees the index too
        status = fc.warm_up_cache(quiet=True)
        assert status["neurons_indexed"] == 2

    def test_get_cache_status_and_all_dataset_bodyids(self, tmp_path):
        fc, _ = _cache_fc(tmp_path)
        _write_conn_parquet(fc, ["1"], ["2"], [5])
        _write_index_parquet(fc, {
            "bodyId": ["1", "2"], "type": ["T1", "T2"],
            "instance": ["I1", "I2"], "post": [10, 20],
            "downstream_complete": [True, False],
            "last_fetched": ["", ""], "connection_count": [1, 0],
        })
        ds_dir = tmp_path / "datasets" / "test_v1"
        ds_dir.mkdir(parents=True)
        pd.DataFrame({
            "bodyId": ["1", "2", "3"], "type": ["a", "b", "c"],
            "instance": ["x", "y", "z"], "post": [1, 2, 3],
        }).to_csv(ds_dir / "test_v1_allneurons_neuron_df.csv", index=True)

        # Fixed: get_cache_status handles the polars connection frame from
        # _load_connection_db (n_unique(), null-safe) instead of raising
        # AttributeError on pandas-only nunique().
        status_polars = fc.get_cache_status()
        assert status_polars["connections_cached"] == 1
        assert status_polars["unique_upstream"] == 1

        # Serve a pandas frame to exercise the rest of the status report.
        fc._load_connection_db = lambda force_reload=False: pd.DataFrame({
            "bodyId_pre": ["1"], "bodyId_post": ["2"], "weight": [5],
        })
        status = fc.get_cache_status()
        assert status["neuron_df_exists"] is True
        assert status["neuron_df_count"] == 3
        assert status["neurons_indexed"] == 2
        assert status["neurons_complete"] == 1
        assert status["connections_cached"] == 1
        assert status["unique_upstream"] == 1


class TestGetAllDatasetBodyids:
    def test_from_csv(self, tmp_path):
        fc, _ = _cache_fc(tmp_path)
        fc._get_neuron_index_path = lambda: str(tmp_path / "nope" / "index.parquet")
        ds_dir = tmp_path / "datasets" / "test_v1"
        ds_dir.mkdir(parents=True)
        pd.DataFrame({"bodyId": [1, 2, 2], "type": ["a", "b", "b"]}).to_csv(
            ds_dir / "test_v1_allneurons_neuron_df.csv", index=True)
        assert fc._get_all_dataset_bodyids() == ["1", "2"]

    def test_from_parquet(self, tmp_path):
        fc, _ = _cache_fc(tmp_path)
        fc._get_neuron_index_path = lambda: str(tmp_path / "nope" / "index.parquet")
        ds_dir = tmp_path / "datasets" / "test_v1"
        ds_dir.mkdir(parents=True)
        pd.DataFrame({"bodyId": ["7", "8"]}).to_parquet(
            ds_dir / "test_v1_allneurons_neuron_df.parquet", index=False)
        assert fc._get_all_dataset_bodyids() == ["7", "8"]

    def test_no_table_returns_empty(self, tmp_path):
        fc, _ = _cache_fc(tmp_path)
        fc._get_neuron_index_path = lambda: str(tmp_path / "nope" / "index.parquet")
        assert fc._get_all_dataset_bodyids() == []


class TestUpdateNeuronIndexAfterFetch:
    def _fc(self):
        fc, logs = make_fc(dataset="test:v1", script_path="/tmp/no_such_dir",
                           use_cache=True)
        saved = []
        fc._load_neuron_index = lambda force_reload=False: pd.DataFrame({
            "bodyId": ["1", "2"], "type": ["T1", "T2"],
            "instance": ["I1", "I2"], "post": [10, 20],
            "downstream_complete": [False, False],
            "last_fetched": ["", ""], "connection_count": [0, 0],
        })
        fc._save_neuron_index_state = (
            lambda df, touched_bodyids=None, force=False: saved.append(df.copy()))
        return fc, logs, saved

    def test_marks_counts_and_appends_missing(self):
        fc, _, saved = self._fc()
        fc._fetch_neurons_batched = lambda ids: pd.DataFrame({
            "bodyId": ["3"], "type": ["T3"], "instance": ["I3"], "post": [30],
        })
        connections = pd.DataFrame({
            "bodyId_pre": ["1", "1", "3"], "bodyId_post": ["2", "3", "4"],
            "weight": [1, 2, 3],
        })
        fc._update_neuron_index_after_fetch(
            connections, ["1", "3"], downstream_bodyIds=None)
        assert len(saved) == 1
        out = saved[0].set_index("bodyId")
        assert int(out.loc["1", "connection_count"]) == 2
        # neuron 3 fetched via the batched-API fallback and appended
        assert "3" in out.index
        assert int(out.loc["3", "connection_count"]) == 1
        assert out.loc["3", "type"] == "T3"

    def test_zero_outdegree_marked_complete(self):
        fc, _, saved = self._fc()
        fc._update_neuron_index_after_fetch(
            pd.DataFrame(columns=["bodyId_pre", "bodyId_post", "weight"]),
            ["2"], downstream_bodyIds=None)
        out = saved[0]
        row = out[out["bodyId"] == "2"].iloc[0]
        assert bool(row["downstream_complete"]) is True
        assert int(row["connection_count"]) == 0

    def test_partial_downstream_not_marked_complete(self):
        fc, _, saved = self._fc()
        fc._fetch_neurons_batched = lambda ids: pd.DataFrame({
            "bodyId": list(ids), "type": [""] * len(ids),
            "instance": [""] * len(ids), "post": [0] * len(ids),
        })
        fc._update_neuron_index_after_fetch(
            pd.DataFrame({"bodyId_pre": ["1"], "bodyId_post": ["2"],
                          "weight": [1]}),
            ["1"], downstream_bodyIds=["2"])
        out = saved[0]
        row = out[out["bodyId"] == "1"].iloc[0]
        assert bool(row["downstream_complete"]) is False

    def test_missing_ids_resolved_from_local_dataset(self, tmp_path):
        fc, logs = make_fc(dataset="test:v1", script_path=str(tmp_path),
                           use_cache=True)
        fc._local_neuron_df_cache = {}
        saved = []
        fc._load_neuron_index = lambda force_reload=False: pd.DataFrame({
            "bodyId": ["1"], "type": ["T1"], "instance": ["I1"], "post": [10],
            "downstream_complete": [False], "last_fetched": [""],
            "connection_count": [0],
        })
        fc._save_neuron_index_state = (
            lambda df, touched_bodyids=None, force=False: saved.append(df.copy()))
        ds_dir = tmp_path / "datasets" / "test_v1"
        ds_dir.mkdir(parents=True)
        pd.DataFrame({
            "bodyId": ["9"], "type": ["T9"], "instance": ["I9"], "post": [99],
        }).to_csv(ds_dir / "test_v1_allneurons_neuron_df.csv", index=True)
        fc._update_neuron_index_after_fetch(
            pd.DataFrame({"bodyId_pre": ["9"], "bodyId_post": ["1"],
                          "weight": [4]}),
            ["9"], downstream_bodyIds=None)
        out = saved[0]
        row = out[out["bodyId"] == "9"].iloc[0]
        assert row["type"] == "T9"
        assert int(row["connection_count"]) == 1


class TestValidateAndRepairCache:
    def test_cache_disabled_short_circuits(self, tmp_path):
        fc, _ = _cache_fc(tmp_path, use_cache=False)
        summary = fc.validate_and_repair_cache(quiet=True)
        assert summary["total_indexed"] == 0

    def test_no_index_nothing_to_repair(self, tmp_path):
        fc, _ = _cache_fc(tmp_path)
        summary = fc.validate_and_repair_cache(quiet=True)
        assert summary == {
            "total_indexed": 0, "total_with_connections": 0,
            "falsely_complete": 0, "repaired": 0, "types_updated": 0,
        }

    def test_repair_false_completion_and_enrich_types(self, tmp_path):
        fc, _ = _cache_fc(tmp_path)
        _write_index_parquet(fc, {
            "bodyId": ["1", "2", "3"], "type": ["", "", "T3"],
            "instance": ["", "", ""], "post": [0, 0, 0],
            "downstream_complete": [True, True, False],
            "last_fetched": ["", "", ""], "connection_count": [5, 0, -1],
        })
        # only neuron 3 has cached downstream rows -> 1 is falsely complete
        _write_conn_parquet(fc, ["3"], ["9"], [5])
        ds_dir = tmp_path / "datasets" / "test_v1"
        ds_dir.mkdir(parents=True)
        pd.DataFrame({
            "bodyId": ["1", "2"], "type": ["TA", "TB"],
            "instance": ["IA", "IB"], "post": [1, 2],
        }).to_csv(ds_dir / "test_v1_allneurons_neuron_df.csv", index=False)

        summary = fc.validate_and_repair_cache(quiet=True)
        assert summary["total_indexed"] == 3
        assert summary["total_with_connections"] == 1
        assert summary["falsely_complete"] == 1
        assert summary["repaired"] == 1
        assert summary["types_updated"] >= 1

        repaired = fc._read_neuron_index_disk().set_index("bodyId")
        assert bool(repaired.loc["1", "downstream_complete"]) is False
        assert int(repaired.loc["1", "connection_count"]) == -1
        # zero-outdegree neuron keeps its completion flag
        assert bool(repaired.loc["2", "downstream_complete"]) is True
        assert repaired.loc["1", "type"] == "TA"

    def test_clean_cache_needs_no_repair(self, tmp_path):
        fc, _ = _cache_fc(tmp_path)
        _write_index_parquet(fc, {
            "bodyId": ["3"], "type": ["T3"], "instance": [""], "post": [0],
            "downstream_complete": [True], "last_fetched": [""],
            "connection_count": [1],
        })
        _write_conn_parquet(fc, ["3"], ["9"], [5])
        summary = fc.validate_and_repair_cache(quiet=True)
        assert summary["falsely_complete"] == 0
        assert summary["repaired"] == 0


class TestVisualizeSelectedPaths:
    def test_wrapper_forwards_colors_and_limits(self, monkeypatch, tmp_path):
        calls = []

        class FakeVP:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                calls.append(self)

            def visualize(self):
                return ("CONN", "GRAPH")

        monkeypatch.setattr(coana, "VisualizePath", FakeVP)
        fc, _ = make_fc(
            source_color="#111111", intermediate_color="#222222",
            target_color="#333333", link_color="rgba(1,1,1,0.5)",
            node_color=["#111111", "#222222"], edgeN_limit=77,
            verbose_mode="silent",
        )
        paths = pd.DataFrame({"path_block": ["A -> B"], "weights": [[5]]})
        out = fc.VisualizeSelectedPaths(
            paths, sheet_name="path_type", output_folder=str(tmp_path))
        assert out == ("CONN", "GRAPH")
        kw = calls[0].kwargs
        assert kw["source_color"] == "#111111"
        assert kw["target_color"] == "#333333"
        assert kw["edgeN_limit"] == 77
        assert kw["network_layout"] == "hierarchical"
        assert kw["showfig"] is False


class TestSaveNeuronInfo:
    def test_saves_source_target_parameters(self, tmp_path):
        fc, _ = make_fc(
            source_fname="src", target_fname="tgt",
            source_df=pd.DataFrame({"bodyId": ["1"], "type": ["T"]}),
            target_df=pd.DataFrame({"bodyId": ["2"], "type": ["U"]}),
            parameter_df=pd.DataFrame(
                {"parameter": ["p"], "value": ["v"]}),
            _source_target_identical=True,
        )
        # missing source_df raises
        bare = object.__new__(coana.FindNeuronConnection)
        with pytest.raises(RuntimeError):
            bare.SaveNeuronInfo()
        out = fc.SaveNeuronInfo(output_dir=str(tmp_path))
        files = sorted(p.name for p in Path(out).glob("*.csv"))
        assert "src_to_tgt_source_neurons.csv" in files
        assert "src_to_tgt_target_neurons.csv" in files
        assert "src_to_tgt_parameters.csv" in files

    def test_no_parameter_df_skips_params(self, monkeypatch, tmp_path):
        # Remove the class-level default so hasattr() really misses it
        monkeypatch.delattr(coana.FindNeuronConnection, "parameter_df")
        fc, _ = make_fc(
            source_fname="s", target_fname="t",
            source_df=pd.DataFrame({"bodyId": ["1"]}),
            target_df=pd.DataFrame({"bodyId": ["2"]}),
        )
        out = fc.SaveNeuronInfo(output_dir=str(tmp_path),
                                filename_prefix="custom")
        files = sorted(p.name for p in Path(out).glob("*.csv"))
        assert files == ["custom_source_neurons.csv", "custom_target_neurons.csv"]


class TestFetchIncomingAndBackward:
    def test_incoming_online_flywire_cave(self, monkeypatch, tmp_path):
        fc, _ = make_fc(dataset="fafb:v1", script_path=str(tmp_path),
                        use_cache=False, verbose_mode="silent")

        class FakeFetcher:
            def fetch_connections(self, ids, direction=None,
                                  show_progress=False):
                assert direction == "post"
                return pd.DataFrame({
                    "pre_pt_root_id": [720575940614131061],
                    "post_pt_root_id": [ids[0]], "weight": [3],
                })

        fc._cave_fetcher = FakeFetcher()
        out = fc._fetch_incoming_connections_online(["720575940614131062"])
        assert list(out.columns) >= ["bodyId_pre", "bodyId_post", "weight"]
        assert (out["roi"] == "WholeBrain").all()
        assert fc._fetch_incoming_connections_online([]).empty

    def test_backward_layer_cached_plus_api(self, monkeypatch, tmp_path):
        fc, logs = make_fc(dataset="test:v1", script_path=str(tmp_path),
                           use_cache=True, cache_only=False)
        cached = pl.DataFrame({
            "bodyId_pre": ["1"], "bodyId_post": ["10"], "weight": [5],
            "roi": ["r"],
        })
        fc._conn_df_cache = cached
        # keep the test focused on the backward-layer logic
        fc._enrich_connections_with_neuron_info = lambda df: df.copy()
        fc._conn_index = None
        fc._conn_index_post = None
        fc._neuron_index_dict = {}
        fc._load_connection_db = lambda force_reload=False: cached
        api_calls = []
        fc._fetch_incoming_connections_online = lambda posts: (
            api_calls.append(list(posts)) or pd.DataFrame({
                "bodyId_pre": ["2"], "bodyId_post": ["10"], "weight": [4],
                "roi": ["r"],
            }))
        out = fc._fetch_path_connections_backward(["10"], source_bodyIds=None)
        assert set(out["bodyId_pre"]) == {"1", "2"}
        assert api_calls == [["10"]]

    def test_backward_source_cache_complete_skips_api(self, tmp_path):
        """F-PERF-03 fix (2026-09-08): a source's ``downstream_complete``
        flag says nothing about a post's incoming completeness. The online
        incoming query must still run for a post that is not proven
        incoming-complete, and the source-oriented cached row still merges
        into the layer."""
        fc, logs = make_fc(dataset="test:v1", script_path=str(tmp_path),
                           use_cache=True, cache_only=False)
        cached = pl.DataFrame({
            "bodyId_pre": ["1"], "bodyId_post": ["10"], "weight": [5],
            "roi": ["r"],
        })
        fc._conn_df_cache = cached
        # keep the test focused on the backward-layer logic
        fc._enrich_connections_with_neuron_info = lambda df: df.copy()
        fc._conn_index = None
        fc._conn_index_post = None
        fc._neuron_index_dict = {
            "1": {"downstream_complete": True, "connection_count": 1},
        }
        fc._load_connection_db = lambda force_reload=False: cached
        api_calls = []

        def _online(posts):
            api_calls.append(list(posts))
            # a real incoming query returns the post's complete incoming set
            return pd.DataFrame({
                "bodyId_pre": ["1"], "bodyId_post": ["10"], "weight": [5],
                "roi": ["r"],
            })

        fc._fetch_incoming_connections_online = _online
        out = fc._fetch_path_connections_backward(["10"], source_bodyIds=["1"])
        # the query ran for the unproven post (the old source-gate skipped it)
        assert api_calls == [["10"]]
        # the successful incoming query proves post 10's incoming set ...
        assert fc._incoming_cache["complete"] == {"10"}
        # ...and the row merged in
        assert len(out) == 1

        # once incoming-complete, the post is served from the incoming
        # cache without re-querying
        out2 = fc._fetch_path_connections_backward(["10"], source_bodyIds=["1"])
        assert api_calls == [["10"]]
        assert len(out2) == 1

    def test_backward_cache_only_warns_when_unproven(self, tmp_path):
        fc, logs = make_fc(dataset="test:v1", script_path=str(tmp_path),
                           use_cache=True, cache_only=True)
        cached = pl.DataFrame({
            "bodyId_pre": ["1"], "bodyId_post": ["10"], "weight": [5],
            "roi": ["r"],
        })
        fc._conn_df_cache = cached
        # keep the test focused on the backward-layer logic
        fc._enrich_connections_with_neuron_info = lambda df: df.copy()
        fc._conn_index = None
        fc._conn_index_post = None
        fc._neuron_index_dict = {}
        fc._load_connection_db = lambda force_reload=False: cached
        fc._fetch_incoming_connections_online = lambda posts: pd.DataFrame()
        out = fc._fetch_path_connections_backward(["10"], source_bodyIds=["1"])
        assert len(out) == 1
        assert any("cache_only" in note for note in fc._warn_notes)


class TestFetchConnectionsBulk:
    def test_empty_upstream_returns_empty(self, tmp_path):
        fc, _ = make_fc(dataset="test:v1", script_path=str(tmp_path))
        assert fc._fetch_connections_bulk([]).empty

    def test_flywire_online_via_cave(self, monkeypatch, tmp_path):
        fc, _ = make_fc(dataset="fafb:v1", script_path=str(tmp_path),
                        use_cache=False)

        class FakeFetcher:
            def fetch_connections(self, ids, direction=None,
                                  show_progress=False):
                return pd.DataFrame({
                    "pre_pt_root_id": [ids[0]] * 2,
                    "post_pt_root_id": [720575940614131062,
                                          720575940614131063],
                    "weight": [3, 4],
                })

        fc._cave_fetcher = FakeFetcher()
        statuses = []
        out = fc._fetch_connections_bulk(
            ["720575940614131061"],
            downstream_bodyIds=["720575940614131062"],
            status_callback=statuses.append)
        assert len(out) == 1
        assert "roi" in out.columns

    def test_flywire_online_empty_result(self, tmp_path):
        fc, _ = make_fc(dataset="fafb:v1", script_path=str(tmp_path),
                        use_cache=False)

        class FakeFetcher:
            def fetch_connections(self, *a, **k):
                return None

        fc._cave_fetcher = FakeFetcher()
        assert fc._fetch_connections_bulk(["720575940614131061"]).empty


class TestInitializeNeuronInfo:
    def _base_attrs(self, tmp_path):
        return dict(
            client_type="flywire", client_flywire=None, client_hemibrain=None,
            dataset="test:v1", script_path=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            max_interlayer=1, separate_hemispheres=False,
            hemisphere_filter="both", label_mapper=None,
            custom_source_group_names=None, custom_target_group_names=None,
            custom_source_name=None, custom_target_name=None,
            search_columns=None, saveas="", save_folder=None,
            folder_prefix=None, min_synapse_num=1, min_ratio=0.0,
            min_traversal_probability=0.0, aggregate_method="sum",
            filter_by="bodyId", exclude_intra_type_connections=False,
            find_reciprocal=False,
            keyword_in_path_to_remove=["None"], server="",
            run_date="2026-01-01", kwargs_fetch={},
            simple_fetch=True, largeTargetSet=False,
            sourceNeurons=["A"], targetNeurons=["B"],
        )

    def _fake_getNeurons(self, monkeypatch):
        calls = []

        def fake(query, dataset=None, custom_group_names=None, client=None,
                 verbose=False, search_columns=None, search_info_sink=None):
            calls.append(list(query))
            df = pd.DataFrame({
                "bodyId": ["1"], "type": ["T"], "instance": ["I"],
            })
            return df, None, "_".join(str(q) for q in query), None

        monkeypatch.setattr(coana.sv, "getNeurons", fake)
        return calls

    def test_identical_source_target_single_fetch(self, monkeypatch, tmp_path):
        calls = self._fake_getNeurons(monkeypatch)
        attrs = self._base_attrs(tmp_path)
        attrs.update(sourceNeurons=["A"], targetNeurons=["A"],
                     max_interlayer=-1)
        fc, logs = make_fc(**attrs)
        fc.InitializeNeuronInfo()
        assert calls == [["A"]]  # fetched once only
        assert fc._source_target_identical is True
        assert fc.source_fname == "A"
        assert fc.target_fname == "A"
        assert fc.target_df is not fc.source_df  # copy, not alias

    def test_two_sided_with_label_mapper(self, monkeypatch, tmp_path):
        calls = self._fake_getNeurons(monkeypatch)

        class FakeMapper:
            is_empty = False

            def get_std_label(self, dataset, value, role):
                return f"STD_{role}"

            def get_neurons_for_label(self, label, dataset, role):
                return []

        attrs = self._base_attrs(tmp_path)
        attrs.update(label_mapper=FakeMapper(), custom_source_name="mySrc",
                     custom_target_name="myTgt")
        fc, logs = make_fc(**attrs)
        fc.InitializeNeuronInfo()
        assert calls == [["A"], ["B"]]
        assert fc.source_df.iloc[0]["type"] == "STD_source"
        assert fc.target_df.iloc[0]["type"] == "STD_target"
        assert fc.source_fname == "mySrc"
        assert fc.target_fname == "myTgt"
        assert fc.parameter_dict["source name"] == "mySrc"
        assert "parameter" in fc.parameter_df.columns

    def test_all_neurons_both_sides_rejected(self, monkeypatch, tmp_path):
        self._fake_getNeurons(monkeypatch)
        attrs = self._base_attrs(tmp_path)
        attrs.update(sourceNeurons=["all_neurons"],
                     targetNeurons=["all_neurons"])
        fc, _ = make_fc(**attrs)
        with pytest.raises(ValueError):
            fc.InitializeNeuronInfo()


# =============================================================================
# neuron fetching: by type / from dataset or API / bulk profile cache
# =============================================================================

class TestFetchNeuronsByTypes:
    def _write_local(self, tmp_path, rows):
        ds_dir = tmp_path / "datasets" / "test_v1"
        ds_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            ds_dir / "test_v1_allneurons_neuron_df.csv", index=True)

    def test_local_csv_with_columns(self, tmp_path):
        self._write_local(tmp_path, {
            "bodyId": ["1", "2", "3"], "type": ["T1", "T1", "T2"],
            "instance": ["I1", "I2", "I3"], "post": [10, 20, 30],
        })
        fc, _ = make_fc(dataset="test:v1", script_path=str(tmp_path),
                        use_cache=True, separate_hemispheres=False)
        fc._local_neuron_df_cache = {}
        out = fc._fetch_neurons_by_types(
            ["T1"], columns=["bodyId", "type", "post", "side"])
        assert len(out) == 2
        # missing 'side' column filled with ''
        assert (out["side"] == "").all()

    def test_local_csv_hemisphere_columns_added(self, tmp_path):
        self._write_local(tmp_path, {
            "bodyId": ["1"], "type": ["T1"], "instance": ["I1"], "post": [10],
            "hemisphere": ["R"], "hemisphere_code": ["R"],
        })
        fc, _ = make_fc(dataset="test:v1", script_path=str(tmp_path),
                        use_cache=True, separate_hemispheres=True)
        fc._local_neuron_df_cache = {}
        out = fc._fetch_neurons_by_types(["T1"], columns=["bodyId", "type"])
        assert {"instance", "hemisphere", "hemisphere_code"} <= set(out.columns)

    def test_flywire_online_via_cave_fetcher(self, monkeypatch, tmp_path):
        fc, _ = make_fc(dataset="fafb:v1", script_path=str(tmp_path),
                        use_cache=False, client_flywire=None,
                        separate_hemispheres=False)

        class FakeCave:
            def fetch_neurons_by_types(self, types, show_progress=False):
                return pd.DataFrame({
                    "root_id": [720575940614131061, 720575940614131062],
                    "type": list(types) * 1, "tag": ["a", "b"],
                })

        fc._cave_fetcher = FakeCave()
        out = fc._fetch_neurons_by_types(
            ["TA", "TB"], columns=["bodyId", "type", "instance", "post"])
        assert len(out) == 2
        assert "bodyId" in out.columns
        assert list(out["post"]) == [0, 0]

    def test_flywire_online_empty_result(self, tmp_path):
        fc, _ = make_fc(dataset="fafb:v1", script_path=str(tmp_path),
                        use_cache=False, client_flywire=None)

        class FakeCave:
            def fetch_neurons_by_types(self, types, show_progress=False):
                return None

        fc._cave_fetcher = FakeCave()
        out = fc._fetch_neurons_by_types(["TA"])
        assert out.empty

    def test_flywire_cached_but_no_local_table(self, tmp_path):
        # cache enabled but no local neuron table -> documented empty result
        fc, logs = make_fc(dataset="fafb:v1", script_path=str(tmp_path),
                           use_cache=True)
        fc._local_neuron_df_cache = {}
        out = fc._fetch_neurons_by_types(["TA"], columns=["bodyId"])
        assert out.empty


class TestFetchFromDatasetOrApiBranches:
    def test_flywire_online_only(self, tmp_path):
        fc, _ = make_fc(dataset="fafb:v1", script_path=str(tmp_path),
                        use_cache=False)
        calls = []
        fc._fetch_flywire_neurons_online = lambda ids, columns=None: (
            calls.append(list(ids)) or pd.DataFrame({
                "bodyId": [str(i) for i in ids], "type": ["T"] * len(ids),
            }))
        out = fc._fetch_from_dataset_or_api(
            ["720575940614131061"], columns=["bodyId", "type"])
        assert calls == [["720575940614131061"]]
        assert len(out) == 1

    def test_flywire_cache_enabled_missing_table_warns(self, tmp_path):
        fc, logs = make_fc(dataset="fafb:v1", script_path=str(tmp_path),
                           use_cache=True)
        fc._local_neuron_df_cache = {}
        out = fc._fetch_from_dataset_or_api(
            ["720575940614131061"], columns=["bodyId"])
        assert out.empty
        assert any("Local neuron data not found" in m for m in logs)

    def test_neuprint_batched_fallback(self, tmp_path):
        fc, _ = make_fc(dataset="test:v1", script_path=str(tmp_path),
                        use_cache=False, client_type="neuprint")
        fc._ensure_neuprint_client = lambda: None
        fc._fetch_neurons_batched = lambda ids: pd.DataFrame({
            "bodyId": [str(i) for i in ids], "type": ["T"] * len(ids),
            "instance": ["I"] * len(ids),
        })
        out = fc._fetch_from_dataset_or_api(
            [5813, 5814], columns=["bodyId", "type", "instance"])
        assert len(out) == 2
        assert list(out.columns) == ["bodyId", "type", "instance"]


class TestBuildConnectivityProfileCache:
    def test_delegates_to_profiler(self, monkeypatch, tmp_path, capsys):
        import types as _types

        built = {"T1": object()}

        class FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeProfiler:
            def __init__(self, datasets=None, config=None):
                self.datasets = datasets
                self.config = config

            def build_connectivity_profile_cache(self, **kwargs):
                self.kwargs = kwargs
                return built

        fake_module = _types.ModuleType("comparison.connectivity_profiler")
        fake_module.ConnectivityProfiler = FakeProfiler
        fake_module.ProfilerConfig = FakeConfig
        monkeypatch.setitem(
            sys.modules, "comparison.connectivity_profiler", fake_module)

        fc, _ = make_fc(dataset="test:v1", script_path=str(tmp_path),
                        verbose_mode="silent")
        result = fc.build_connectivity_profile_cache(
            neuron_types=["T1", "T2"], top_k=3, top_m=2, max_neurons=10)
        assert result["total_profiles"] == 1
        assert result["failed_types"] == ["T2"]
        assert result["profiles"] == built


# =============================================================================
# FlyWire one-time imports (merged connections + enriched neuron CSV)
# =============================================================================

def _flywire_fc(tmp_path, **attrs):
    cache_folder = tmp_path / "cache" / "FAFB_v1_0"
    cache_folder.mkdir(parents=True, exist_ok=True)
    ds_dir = tmp_path / "datasets" / "FAFB_v1_0"
    ds_dir.mkdir(parents=True, exist_ok=True)
    fc, logs = make_fc(
        dataset="FAFB:v1.0", script_path=str(tmp_path),
        cache_folder=str(cache_folder), use_cache=True,
        client_type="flywire", force_API_fetching=False,
        _conn_df_cache=None, _conn_index=None, _conn_index_post=None,
        _conn_db_pre_id_cache=None, _conn_cache_signature=None,
        _neuron_index_cache=None, _neuron_index_dict=None,
        _neuron_index_signature_value=None,
        _local_neuron_df_cache=dict())
    for key, value in attrs.items():
        setattr(fc, key, value)
    return fc, logs


class TestFlywireOneTimeImports:
    def test_import_merged_connections_parquet(self, tmp_path):
        fc, logs = _flywire_fc(tmp_path)
        merged = tmp_path / "datasets" / "FAFB_v1_0" / "fafb_merged_connections.parquet"
        pl.DataFrame({
            "pre_root_id": ["720575940000000011", "720575940000000012"],
            "post_root_id": ["720575940000000021", "720575940000000022"],
            "syn_count": [7, 9],
            "neuropil": ["ME", "ME"],
        }).write_parquet(merged)
        df = fc._load_connection_db()
        assert df is not None and df.height == 2
        assert set(df.columns) >= {"bodyId_pre", "bodyId_post", "weight", "roi"}
        assert os.path.exists(fc._get_connection_db_path())
        # index was built from the imported frame
        assert fc._conn_index is not None
        assert "720575940000000011" in fc._conn_index

    def test_import_merged_connections_csv(self, tmp_path):
        fc, logs = _flywire_fc(tmp_path)
        merged = tmp_path / "datasets" / "FAFB_v1_0" / "fafb_merged_connections.csv"
        pd.DataFrame({
            "pre": ["720575940000000031"],
            "post": ["720575940000000041"],
            "synapses": [4],
        }).to_csv(merged, index=False)
        df = fc._load_connection_db()
        assert df.height == 1
        assert df["weight"][0] == 4
        # roi/cached_date were backfilled
        assert "roi" in df.columns and "cached_date" in df.columns

    def test_import_neuron_index_from_enriched_csv(self, tmp_path):
        fc, logs = _flywire_fc(tmp_path)
        table = tmp_path / "datasets" / "FAFB_v1_0" / "FAFB_v1_0_allneurons_neuron_df.csv"
        pd.DataFrame({
            "bodyId": ["720575940000000051", "720575940000000052"],
            "type": ["Mi1", "Tm3"],
            "name": ["Mi1_L", "Tm3_R"],
            "post": [12, 0],
        }).to_csv(table, index=False)
        # the import writes the index parquet; its folder must exist
        os.makedirs(os.path.dirname(fc._get_neuron_index_path()),
                    exist_ok=True)
        df = fc._load_neuron_index()
        assert df is not None and len(df) == 2
        assert set(df["downstream_complete"]) == {True}
        assert os.path.exists(fc._get_neuron_index_path())
        assert fc._neuron_index_dict is not None


# =============================================================================
# _prepare_flywire_data decision tree
# =============================================================================

class TestPrepareFlywireData:
    def test_non_flywire_noop(self):
        fc, _ = make_fc(client_type="neuprint", dataset="hemibrain:v1.2.1")
        assert fc._prepare_flywire_data() is None

    def test_online_mode_banc_raises(self, tmp_path):
        fc, _ = _flywire_fc(tmp_path, dataset="BANC:v1.0", use_cache=False)
        with pytest.raises(RuntimeError):
            fc._prepare_banc_data()

    def test_online_mode_fafb_forces_api(self, tmp_path):
        fc, _ = _flywire_fc(tmp_path, use_cache=False)
        fc.force_API_fetching = False
        fc._prepare_flywire_data()
        assert fc.force_API_fetching is True

    def test_banc_force_api_falls_back_to_local(self, monkeypatch, tmp_path):
        fc, _ = _flywire_fc(tmp_path, dataset="BANC:v1.0",
                            force_API_fetching=True)

        class FakeBANC:
            @staticmethod
            def ensure_banc_data(dataset, dataset_dir):
                return True

            @staticmethod
            def build_connection_cache_from_tables(dataset_dir, cache_dir):
                return True

        monkeypatch.setattr(coana, "BANC_file_converter", FakeBANC)
        fc._prepare_banc_data()
        assert fc.force_API_fetching is False

    def test_force_api_with_api_cache(self, tmp_path):
        fc, _ = _flywire_fc(tmp_path, force_API_fetching=True)
        api_dir = tmp_path / "cache" / "FAFB_v1_0" / "API_cache"
        api_dir.mkdir(parents=True)
        (api_dir / "connections.parquet").write_bytes(b"x")
        (api_dir / "neuron_index.parquet").write_bytes(b"x")
        assert fc._prepare_flywire_data() is None

    def test_force_api_without_api_cache(self, tmp_path):
        fc, logs = _flywire_fc(tmp_path, force_API_fetching=True)
        assert fc._prepare_flywire_data() is None

    def test_valid_existing_cache(self, tmp_path):
        fc, _ = _flywire_fc(tmp_path)
        cache_dir = tmp_path / "cache" / "FAFB_v1_0"
        pl.DataFrame({"bodyId_pre": ["1"], "bodyId_post": ["2"],
                      "weight": [1]}).write_parquet(
            cache_dir / "connections.parquet")
        idx_dir = tmp_path / "neuron_indexes" / "FAFB_v1_0"
        idx_dir.mkdir(parents=True)
        pd.DataFrame({"bodyId": ["1"]}).to_parquet(
            idx_dir / "neuron_index.parquet", index=False)
        assert fc._prepare_flywire_data() is None

    def test_invalid_cache_converter_failure_raises(self, monkeypatch, tmp_path):
        fc, _ = _flywire_fc(tmp_path)
        cache_dir = tmp_path / "cache" / "FAFB_v1_0"
        (cache_dir / "connections.parquet").write_text("not parquet")
        idx_dir = tmp_path / "neuron_indexes" / "FAFB_v1_0"
        idx_dir.mkdir(parents=True)
        (idx_dir / "neuron_index.parquet").write_text("not parquet")

        class FakeFAFB:
            @staticmethod
            def ensure_flywire_data(dataset, dataset_dir):
                return False

        monkeypatch.setattr(coana, "FAFB_file_converter", FakeFAFB)
        with pytest.raises(RuntimeError):
            fc._prepare_flywire_data()


class TestBancLocalDataFailure:
    """Missing local data is an error the caller can surface, never a
    process exit."""

    class _FakeBANC:
        ensure_ok = False
        build_ok = True

        @classmethod
        def ensure_banc_data(cls, dataset, dataset_dir):
            return cls.ensure_ok

        @classmethod
        def build_connection_cache_from_tables(cls, dataset_dir, cache_dir):
            return cls.build_ok

    @staticmethod
    def _banc_fc(tmp_path, monkeypatch, fake):
        fc, _ = make_fc(
            dataset="banc_v888", script_path=str(tmp_path),
            client_type="banc", use_cache=True, force_API_fetching=False,
            cache_folder=str(tmp_path / "cache" / "banc_v888"))
        monkeypatch.setattr(coana, "BANC_file_converter", fake)
        monkeypatch.setattr(coana, "HAS_BANC_CONVERTER", True)
        return fc

    def test_prepare_banc_data_raises(self, tmp_path, monkeypatch):
        fc = self._banc_fc(tmp_path, monkeypatch, self._FakeBANC)
        with pytest.raises(RuntimeError, match="banc_v888"):
            fc._prepare_banc_data()

    def test_connection_cache_absent_tables_raises(self, tmp_path, monkeypatch):
        class Fake(self._FakeBANC):
            build_ok = False

        fc = self._banc_fc(tmp_path, monkeypatch, Fake)
        with pytest.raises(RuntimeError, match="banc_v888"):
            fc._ensure_banc_connection_cache()

    def test_connection_cache_underived_raises(self, tmp_path, monkeypatch):
        class Fake(self._FakeBANC):
            ensure_ok = True
            build_ok = False

        fc = self._banc_fc(tmp_path, monkeypatch, Fake)
        with pytest.raises(RuntimeError, match="connection cache"):
            fc._ensure_banc_connection_cache()


class TestEnsureCompleteDatasetFolder:
    """A failed or skipped pull must not leave an empty dataset folder:
    the directory is created by the pull itself, lazily."""

    class _FakeSV:
        DatasetPullCancelled = type("DatasetPullCancelled", (Exception,), {})

        @staticmethod
        def roi_count_table_path(prefix):
            return prefix + "_roi_count_df.parquet"

        @staticmethod
        def pull_dataset(*args, **kwargs):
            raise RuntimeError("no token")

    def _fc(self, tmp_path):
        return make_fc(
            dataset="male-cns:v1.0", client_type="neuprint", cache_only=False,
            use_cache=True, script_path=str(tmp_path))

    def test_failed_pull_creates_no_dataset_folder(self, tmp_path, monkeypatch):
        fc, _ = self._fc(tmp_path)
        monkeypatch.setattr(coana, "sv", self._FakeSV)
        fc._ensure_neuprint_client = lambda: None
        fc._ensure_complete_dataset()
        assert (tmp_path / "datasets").is_dir()
        assert not (tmp_path / "datasets" / "male-cns_v1_0").exists()

    def test_existing_tables_create_no_folder(self, tmp_path, monkeypatch):
        fc, _ = self._fc(tmp_path)
        ds_dir = tmp_path / "datasets" / "male-cns_v1_0"
        ds_dir.mkdir(parents=True)
        (ds_dir / "male-cns_v1_0_allneurons_neuron_df.csv").write_text("x")
        (ds_dir / "male-cns_v1_0_allneurons_roi_count_df.parquet").write_bytes(
            b"x")
        monkeypatch.setattr(coana, "sv", self._FakeSV)
        fc._ensure_neuprint_client = lambda: None
        fc._ensure_complete_dataset()
        assert sorted(p.name for p in (tmp_path / "datasets").iterdir()) == [
            "male-cns_v1_0"]


# =============================================================================
# build_connection_cache: neuron_types resolution + parallel batches
# =============================================================================

def _bulk_conn_frame(batch):
    return pd.DataFrame({
        "bodyId_pre": [str(b) for b in batch],
        "bodyId_post": ["900"] * len(batch),
        "weight": [5] * len(batch),
    })


class TestBuildConnectionCacheModes:
    def _dataset_table(self, tmp_path):
        ds_dir = tmp_path / "datasets" / "test_v1"
        ds_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "bodyId": ["1", "2", "3", "4"],
            "type": ["T1", "T1", "T2", "T2"],
        }).to_csv(ds_dir / "test_v1_allneurons_neuron_df.csv", index=True)

    def test_neuron_types_sequential(self, tmp_path):
        self._dataset_table(tmp_path)
        fc, _ = _cache_fc(tmp_path)
        fc._fetch_connections_bulk = (
            lambda upstream_bodyIds=None, downstream_bodyIds=None,
            cancel_event=None, status_callback=None:
            _bulk_conn_frame(upstream_bodyIds or []))
        result = fc.build_connection_cache(neuron_types=["T1"], quiet=True)
        assert result["total_neurons"] == 2
        assert result["newly_cached"] == 2
        assert result["total_connections"] == 2
        assert os.path.exists(fc._get_connection_db_path())

    def test_bodyids_parallel_workers(self, tmp_path):
        self._dataset_table(tmp_path)
        fc, _ = _cache_fc(tmp_path)
        fc._fetch_connections_bulk = (
            lambda upstream_bodyIds=None, downstream_bodyIds=None,
            cancel_event=None, status_callback=None:
            _bulk_conn_frame(upstream_bodyIds or []))
        result = fc.build_connection_cache(
            neuron_bodyIds=["1", "2", "3", "4"], batch_size=2,
            max_workers=2, quiet=True)
        assert result["newly_cached"] == 4
        assert result["total_connections"] == 4

    def test_force_rebuild_reimports_from_metadata(self, tmp_path):
        self._dataset_table(tmp_path)
        fc, _ = _cache_fc(tmp_path)
        # pre-existing index with progress flags -> join-back path
        idx_path = fc._get_neuron_index_path()
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        pd.DataFrame({
            "bodyId": ["1", "2"],
            "type": ["T1", "T1"],
            "instance": ["a", "b"],
            "post": [1, 1],
            "downstream_complete": [True, True],
            "last_fetched": ["", ""],
            "connection_count": [1, 1],
        }).to_parquet(idx_path, index=False)
        fc._fetch_connections_bulk = (
            lambda upstream_bodyIds=None, downstream_bodyIds=None,
            cancel_event=None, status_callback=None:
            _bulk_conn_frame(upstream_bodyIds or []))
        result = fc.build_connection_cache(
            neuron_bodyIds=["1"], force_rebuild=True, quiet=True)
        assert result["newly_cached"] == 1
        assert result["already_cached"] == 0


# =============================================================================
# FindAllPath: find_reciprocal pipeline + graph-cache reuse
# =============================================================================

class _RecipFakeVP(_FakeVisualizePath):
    """VisualizePath stand-in supporting the reciprocal viz call shape."""

    def build_network(self):
        self.G_network = None

    def create_heatmap(self):
        pass

    def create_network(self):
        pass


class TestFindAllPathReciprocal:
    CHAIN = [("S", "A", 10), ("A", "B", 8), ("B", "T", 6)]
    TYPES = {"S": "TS", "A": "TA", "B": "TB", "T": "TT"}

    def _wire(self, monkeypatch, tmp_path):
        fc, logs = _make_path_fc(monkeypatch, tmp_path, self.CHAIN,
                                 self.TYPES)
        fc._dataset_safe = "test_v1"

        # richer enrichment: also produce a custom-group frame
        def rich_enrich(conn, traversal_probability_threshold=0, dataset=None,
                        script_path=None, target_neurons_df=None,
                        label_mapper=None, global_incoming_weights=None,
                        separate_hemispheres=False, **kwargs):
            was_polars = isinstance(conn, pl.DataFrame)
            base = conn.to_pandas() if was_polars else conn.copy()
            base["connection_ratio"] = 0.5
            base["traversal_probability"] = 0.5
            conn_e = pl.from_pandas(base) if was_polars else base
            conn_t = (base.groupby(["type_pre", "type_post"],
                                   as_index=False)["weight"].sum())
            conn_t["connection_ratio"] = 0.5
            conn_t["traversal_probability"] = 0.5
            conn_g = pd.DataFrame({
                "custom_group_pre": ["GA"],
                "custom_group_post": ["GB"],
                "weight": [10],
                "connection_ratio": [0.5],
                "traversal_probability": [0.5],
            })
            if was_polars:
                return conn_e, pl.from_pandas(conn_t), pl.from_pandas(conn_g)
            return conn_e, conn_t, conn_g

        monkeypatch.setattr(coana.sv, "EnrichConnectionTable", rich_enrich)
        monkeypatch.setattr(coana, "VisualizePath", _RecipFakeVP)
        _FakeVisualizePath.instances = []

        # reciprocal edges among graph nodes
        fc._fetch_direct_connections_for_nodes = lambda node_ids: pd.DataFrame({
            "bodyId_pre": ["A", "B"],
            "bodyId_post": ["B", "T"],
            "weight": [8, 6],
            "type_pre": ["TA", "TB"],
            "type_post": ["TB", "TT"],
        })
        return fc, logs

    def test_find_reciprocal_outputs_and_viz(self, monkeypatch, tmp_path):
        coana.clear_findallpath_cache()
        fc, _ = self._wire(monkeypatch, tmp_path)
        fc.FindAllPath(find_reciprocal=True)

        folder = Path(fc.allpath_folder)
        recip = folder / "find_reciprocal"
        assert recip.is_dir()
        assert (recip / "reciprocal_connection_type.csv").exists()
        assert (recip / "reciprocal_connection_bodyId.csv").exists()
        assert (recip / "parameters.csv").exists()
        # type/group/bodyId reciprocal visualizations were constructed
        assert len(_FakeVisualizePath.instances) >= 3
        coana.clear_findallpath_cache()

    def test_graph_cache_reuse_with_higher_threshold(self, monkeypatch,
                                                     tmp_path):
        coana.clear_findallpath_cache()
        fc, _ = self._wire(monkeypatch, tmp_path)
        fc.FindAllPath(find_reciprocal=False)
        # second run: same query, higher threshold -> reuse + weight filter
        fc.min_synapse_num = 5
        fc.FindAllPath(find_reciprocal=False)
        folder = Path(fc.allpath_folder)
        assert (folder / "all_attributes.json").exists()
        coana.clear_findallpath_cache()


class TestFindPathDeterminism:
    """Determinism pin: two runs of the same query export a byte-identical
    type-level connection table (the shared total-key stable sort,
    _sort_connection_export, applied AFTER _ensure_ratio_prob_columns —
    the totals join reorders rows, so the sort must be the last
    operation before the write)."""

    CHAIN = [("S", "A", 10), ("A", "B", 8), ("B", "T", 6),
             ("S", "B", 9), ("A", "T", 4)]
    TYPES = {"S": "TS", "A": "TA", "B": "TB", "T": "TT"}

    def test_connection_table_byte_reproducible(self, monkeypatch, tmp_path):
        contents = []
        for run in ("a", "b"):
            fc, _ = _make_path_fc(
                monkeypatch, tmp_path / run, self.CHAIN, self.TYPES)
            fc.FindPath()
            path = Path(fc.path_folder) / "data_details" / "connection_type.csv"
            contents.append(path.read_bytes())
        assert contents[0] == contents[1]


# ---------------------------------------------------------------------------
# Per-pair bodyId coverage summary (2026-10-04): persisted at run time
# from the in-memory bodyId paths, so skip_bodyId runs (which never save
# the bodyId paths table) still get exact pair-scope coverage in the
# pair report.
# ---------------------------------------------------------------------------
def test_pair_endpoint_bodyids_distinct_counts():
    import coana
    paths = [
        ['1', '10', '20'],
        ['2', '11', '20'],
        ['3', '11', '21'],
    ]

    def label(node):
        return {'1': 'S', '2': 'S', '3': 'S',
                '20': 'T', '21': 'T', '10': 'M', '11': 'M'}[node]

    out = coana.FindNeuronConnection._pair_endpoint_bodyids(paths, label)
    assert out[('S', 'T')] == ({'1', '2', '3'}, {'20', '21'})


def test_materialize_paths_writes_pair_coverage_summary():
    """The run-time write is unconditional (skip_bodyId-independent)."""
    import inspect
    import coana
    src = inspect.getsource(coana.FindNeuronConnection._materialize_paths)
    assert 'pair_bodyid_coverage.csv' in src
    assert '_pair_endpoint_bodyids(' in src
