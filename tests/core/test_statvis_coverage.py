"""Coverage-focused unit tests for statvis.py.

All tests are hermetic: synthetic in-memory DataFrames, monkeypatched
loaders, and file I/O restricted to pytest tmp_path.  No network access,
no reads from datasets/ or cache/.
"""
import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

import statvis as sv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Escape / print helpers
# =============================================================================

class TestEscapeHelpers:
    def test_html_escape(self):
        assert sv._statvis_html_escape(None) == ""
        assert sv._statvis_html_escape("<a & 'b' \"c\">") == (
            "&lt;a &amp; &#39;b&#39; &quot;c&quot;&gt;"
        )
        assert sv._statvis_html_escape(5) == "5"

    def test_js_escape(self):
        assert sv._statvis_js_escape(None) == ""
        out = sv._statvis_js_escape("a'b\"c\\d\ne<f>g&h\r")
        assert "\\'" in out and '\\"' in out and "\\\\" in out
        assert "\\n" in out and "\\r" in out
        assert "\\u003c" in out and "\\u003e" in out and "\\u0026" in out

    def test_tqdm_print_default_end(self, capsys):
        sv._tqdm_print("hello", "world", sep="-")
        captured = capsys.readouterr()
        assert "hello-world" in captured.out

    def test_tqdm_print_custom_end(self, capsys):
        sv._tqdm_print("line", end="!")
        captured = capsys.readouterr()
        assert "line!" in captured.out


# =============================================================================
# drop_leading_index_columns
# =============================================================================

class TestDropLeadingIndexColumns:
    def test_pandas_unnamed_dropped(self):
        df = pd.DataFrame({"Unnamed: 0": [0, 1], "type": ["a", "b"]})
        out = sv.drop_leading_index_columns(df)
        assert list(out.columns) == ["type"]

    def test_polars_column_1_dropped(self):
        df = pl.DataFrame({"column_1": [0, 1], "type": ["a", "b"]})
        out = sv.drop_leading_index_columns(df)
        assert list(out.columns) == ["type"]

    def test_non_integer_first_column_kept(self):
        df = pd.DataFrame({"Unnamed: 0": ["x", "y"], "type": ["a", "b"]})
        out = sv.drop_leading_index_columns(df)
        assert list(out.columns) == ["Unnamed: 0", "type"]

    def test_only_index_column_kept(self):
        df = pd.DataFrame({"index": [0, 1]})
        out = sv.drop_leading_index_columns(df)
        assert list(out.columns) == ["index"]

    def test_no_index_columns_noop(self):
        df = pd.DataFrame({"type": ["a"], "bodyId": [1]})
        out = sv.drop_leading_index_columns(df)
        assert list(out.columns) == ["type", "bodyId"]

    def test_leading_run_stops_at_real_column(self):
        df = pd.DataFrame({"Unnamed: 0": [0], "index": [1], "type": ["a"]})
        out = sv.drop_leading_index_columns(df)
        assert list(out.columns) == ["type"]


# =============================================================================
# _load_dataframe_fast
# =============================================================================

class TestLoadDataframeFast:
    def test_csv_with_dtype_overrides(self, tmp_path):
        csv = tmp_path / "t.csv"
        pd.DataFrame({"bodyId": [1, 2], "type": ["a", "b"]}).to_csv(
            csv, index=False)
        df = sv._load_dataframe_fast(str(csv), dtype_overrides={"bodyId": str})
        assert df["bodyId"].dtype == object
        assert list(df["bodyId"]) == ["1", "2"]

    def test_newer_parquet_sibling_preferred(self, tmp_path):
        csv = tmp_path / "t.csv"
        pd.DataFrame({"v": [1]}).to_csv(csv, index=False)
        pq = tmp_path / "t.parquet"
        pd.DataFrame({"v": [99]}).to_parquet(pq)
        # ensure parquet strictly newer
        now = os.stat(csv).st_mtime_ns
        os.utime(pq, ns=(now + 10**6, now + 10**6))
        df = sv._load_dataframe_fast(str(csv))
        assert list(df["v"]) == [99]

    def test_newer_csv_wins_over_parquet(self, tmp_path):
        pq = tmp_path / "t.parquet"
        pd.DataFrame({"v": [99]}).to_parquet(pq)
        csv = tmp_path / "t.csv"
        pd.DataFrame({"v": [1]}).to_csv(csv, index=False)
        now = os.stat(pq).st_mtime_ns
        os.utime(csv, ns=(now + 10**6, now + 10**6))
        df = sv._load_dataframe_fast(str(csv))
        assert list(df["v"]) == [1]

    def test_parquet_only(self, tmp_path):
        pq = tmp_path / "t.parquet"
        pd.DataFrame({"v": [7]}).to_parquet(pq)
        df = sv._load_dataframe_fast(str(tmp_path / "t.csv"),
                                     dtype_overrides={"v": int})
        assert list(df["v"]) == [7]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            sv._load_dataframe_fast(str(tmp_path / "nope.csv"))

    def test_pandas_fallback_when_no_polars(self, tmp_path, monkeypatch):
        csv = tmp_path / "t.csv"
        pd.DataFrame({"bodyId": [1], "type": ["a"]}).to_csv(csv, index=False)
        monkeypatch.setattr(sv, "HAS_POLARS", False)
        df = sv._load_dataframe_fast(str(csv),
                                     dtype_overrides={"bodyId": str})
        assert list(df["bodyId"]) == ["1"]


# =============================================================================
# _get_cached_neuron_df / clear_neuron_cache / roi_count_table_path
# =============================================================================

class TestCachedNeuronDf:
    def test_load_cache_and_clear(self, tmp_path):
        body = str(tmp_path / "synth_allneurons")
        pd.DataFrame({"bodyId": [1, 2], "type": ["a", "b"]}).to_csv(
            body + "_neuron_df.csv", index=False)
        pd.DataFrame({"bodyId": [1, 2], "roi": ["r", "r"]}).to_csv(
            body + "_roi_count_df.csv", index=False)

        ndf, rdf = sv._get_cached_neuron_df("synth", body)
        assert ndf["bodyId"].dtype == np.int64
        assert rdf["bodyId"].dtype == np.int64
        # second call served from memory cache
        ndf2, rdf2 = sv._get_cached_neuron_df("synth", body)
        assert list(ndf2["type"]) == ["a", "b"]
        assert list(rdf2["roi"]) == ["r", "r"]

        sv.clear_neuron_cache("synth")
        assert "synth" not in sv._NEURON_DF_CACHE
        sv._get_cached_neuron_df("synth", body)
        sv.clear_neuron_cache()  # clear-all branch
        assert sv._NEURON_DF_CACHE == {}

    def test_roi_count_table_path(self, tmp_path):
        body = str(tmp_path / "x_allneurons")
        # no parquet -> csv path
        assert sv.roi_count_table_path(body).endswith("_roi_count_df.csv")
        pd.DataFrame({"a": [1]}).to_parquet(body + "_roi_count_df.parquet")
        assert sv.roi_count_table_path(body).endswith("_roi_count_df.parquet")


# =============================================================================
# Query helpers: get_types / get_bodyIds / get_instances / get_info
# =============================================================================

def _synthetic_ndf():
    return pd.DataFrame({
        "bodyId": [1, 2, 3, 4],
        "type": ["Mi1", "Mi1", "Tm3", "DN1"],
        "instance": ["Mi1_L", "Mi1_R", "Tm3_L", "DN1_R"],
    })


@pytest.fixture
def patched_neuron_df(monkeypatch):
    monkeypatch.setattr(sv, "_get_neuron_df",
                        lambda dataset="x", verbose=False: _synthetic_ndf())
    monkeypatch.setattr(sv, "_get_cached_neuron_search", lambda dataset: None)


class TestQueryHelpers:
    def test_get_types_legacy(self, patched_neuron_df):
        types = sv.get_types("Mi1", dataset="test:v1", verbose=False,
                             return_simple=True)
        assert types == ["Mi1"]
        type_list, map_dict, ds = sv.get_types(["Mi1", "Tm3"],
                                               dataset="test:v1",
                                               verbose=False)
        assert type_list == ["Mi1", "Tm3"]
        assert map_dict["Mi1"] == ["Mi1"]
        assert ds == "test_v1"

    def test_get_types_dict_filter(self, patched_neuron_df):
        types = sv.get_types({"contains": "DN"}, dataset="test:v1",
                             verbose=False, return_simple=True)
        assert types == ["DN1"]
        type_list, map_dict, ds = sv.get_types({"contains": "DN"},
                                               dataset="test:v1",
                                               verbose=False)
        assert type_list == ["DN1"]
        assert "DN1" in map_dict

    def test_get_bodyIds_legacy_and_dict(self, patched_neuron_df):
        ids = sv.get_bodyIds("Mi1", dataset="test:v1", verbose=False,
                             return_simple=True)
        assert sorted(ids) == [1, 2]
        body_list, map_dict, ds = sv.get_bodyIds(["Mi1", 3],
                                                 dataset="test:v1",
                                                 verbose=False)
        assert sorted(body_list) == [1, 2, 3]
        assert map_dict[3] == 3
        ids = sv.get_bodyIds({"startswith": "DN"}, dataset="test:v1",
                             verbose=False, return_simple=True)
        assert ids == [4]

    def test_get_instances_legacy(self, patched_neuron_df):
        insts = sv.get_instances("Mi1", dataset="test:v1", verbose=False,
                                 return_simple=True)
        assert insts == ["Mi1_L", "Mi1_R"]
        inst_list, map_dict, ds = sv.get_instances(["Tm3"], dataset="test:v1",
                                                   verbose=False)
        assert inst_list == ["Tm3_L"]
        assert map_dict["Tm3_L"] == ["Tm3"]

    def test_get_instances_dict(self, patched_neuron_df):
        insts = sv.get_instances({"endswith": "_R"}, dataset="test:v1",
                                 verbose=False, return_simple=True)
        assert insts == ["DN1_R", "Mi1_R"]

    def test_get_instances_missing_column(self, monkeypatch):
        df = _synthetic_ndf().drop(columns=["instance"])
        monkeypatch.setattr(sv, "_get_neuron_df",
                            lambda dataset="x", verbose=False: df)
        assert sv.get_instances("Mi1", verbose=False,
                                return_simple=True) == []
        # normalization replaces ':' and '.' only ('-' is kept)
        assert sv.get_instances("Mi1", verbose=False) == (
            [], {}, "male-cns_v0_9")

    def test_get_info_none_query_and_columns(self, patched_neuron_df):
        df = sv.get_info(None, dataset="test:v1",
                         columns=["bodyId", "type", "nope"], verbose=False)
        assert list(df.columns) == ["bodyId", "type"]
        assert len(df) == 4

    def test_get_info_dict_and_legacy(self, patched_neuron_df):
        df = sv.get_info({"contains": "DN"}, dataset="test:v1", verbose=False)
        assert list(df["bodyId"]) == [4]
        df = sv.get_info(["Mi1"], dataset="test:v1",
                         columns=["bodyId"], verbose=False)
        assert sorted(df["bodyId"]) == [1, 2]


# =============================================================================
# Matrix helpers: removeSearchedNeurons / Conn2FullMat / calRC / filtMat / stMat
# =============================================================================

class TestMatrixHelpers:
    def test_remove_searched_neurons(self):
        conn = pd.DataFrame({"bodyId_pre": [1, 1, 2],
                             "bodyId_post": [2, 3, 3],
                             "weight": [5, 6, 7]})
        out = sv.removeSearchedNeurons(conn, [2, 3])
        assert out.empty
        out = sv.removeSearchedNeurons(conn, [2, 3], exempt_neurons=[3])
        assert sorted(out["bodyId_post"].unique()) == [3]

    def test_conn2fullmat_with_types(self):
        source_df = pd.DataFrame({"bodyId": [1], "type": ["S"]})
        target_df = pd.DataFrame({"bodyId": [2, 3], "type": ["T1", "T2"]})
        conn_df = pd.DataFrame({"bodyId_pre": [1], "bodyId_post": [2],
                                "weight": [9]})
        conn_type = pd.DataFrame({"type_pre": ["S"], "type_post": ["T1"],
                                  "weight": [9]})
        cmat_b, cmat_t = sv.Conn2FullMat(source_df, target_df, conn_df,
                                         conn_type)
        assert cmat_b.index.tolist() == ["1_S"]
        assert cmat_b.at["1_S", "2_T1"] == 9
        assert cmat_b.at["1_S", "3_T2"] == 0
        assert cmat_t.at["S", "T1"] == 9
        assert cmat_t.at["S", "T2"] == 0

    def test_conn2fullmat_alternate_weight_col(self):
        source_df = pd.DataFrame({"bodyId": [1], "type": ["S"]})
        target_df = pd.DataFrame({"bodyId": [2], "type": ["T1"]})
        conn_df = pd.DataFrame({"bodyId_pre": [1], "bodyId_post": [2],
                                "weight": [3],
                                "traversal_probability": [0.7]})
        conn_type = pd.DataFrame({"type_pre": ["S"], "type_post": ["T1"],
                                  "weight": [3],
                                  "traversal_probability": [0.7]})
        cmat_b, cmat_t = sv.Conn2FullMat(source_df, target_df, conn_df,
                                         conn_type,
                                         weight_col="traversal_probability")
        assert cmat_b.at["1_S", "2_T1"] == 0.7
        assert cmat_t.at["S", "T1"] == 0.7

    def test_conn2fullmat_without_type_column(self):
        # Regression: frames without a 'type' column used to crash on
        # source_df.type.unique(); they now fall back to bodyId labels.
        source_df = pd.DataFrame({"bodyId": [1]})
        target_df = pd.DataFrame({"bodyId": [2, 3]})
        conn_df = pd.DataFrame({"bodyId_pre": [1], "bodyId_post": [2],
                                "weight": [4]})
        conn_type = pd.DataFrame(columns=["type_pre", "type_post", "weight"])
        cmat_b, cmat_t = sv.Conn2FullMat(source_df, target_df, conn_df,
                                         conn_type)
        assert cmat_b.index.tolist() == ["1"]
        assert cmat_b.at["1", "2"] == 4
        assert cmat_b.at["1", "3"] == 0
        assert cmat_t.at["1", "2"] == 0  # empty conn_type fills nothing

    def test_calrc(self):
        cmat = pd.DataFrame([[1, 0], [2, 3]], index=["s1", "s2"],
                            columns=["t1", "t2"])
        out = sv.calRC(cmat, threshold=0)
        assert out.index.tolist() == ["s1", "s2", "sourceN", "sum_col"]
        assert out.columns.tolist() == ["t1", "t2", "targetN", "sum_row"]
        assert out.loc["sum_col", "t1"] == 3
        assert out.loc["sourceN", "t2"] == 1
        assert out.loc["s1", "targetN"] == 1
        assert out.loc["s2", "sum_row"] == 5

    def test_filtmat_mr_axis0_range(self):
        cmat = pd.DataFrame([[0.2, 0.9], [0.1, 0.5]], columns=["t1", "t2"])
        out = sv.filtMat(cmat, axis=0, filt_range=[0.15, 1.0], by="MR")
        assert out.columns.tolist() == ["t1", "t2"]
        out = sv.filtMat(cmat, axis=0, filt_range=[0.5, 1.0], by="MR")
        assert out.columns.tolist() == ["t2"]

    def test_filtmat_mr_equality_branch(self):
        cmat = pd.DataFrame([[0.5, 0.9]], columns=["t1", "t2"])
        out = sv.filtMat(cmat, axis=0, filt_range=[0.5, 0.5], by="MR")
        assert out.columns.tolist() == ["t1"]

    def test_filtmat_mr_axis1(self):
        cmat = pd.DataFrame([[0.2, 0.1], [0.9, 0.05]],
                            index=["s1", "s2"], columns=["t1", "t2"])
        out = sv.filtMat(cmat, axis=1, filt_range=[0.15, 1.0], by="MR")
        assert out.index.tolist() == ["s1", "s2"]
        out = sv.filtMat(cmat, axis=1, filt_range=[0.5, 1.0], by="MR")
        assert out.index.tolist() == ["s2"]

    def test_filtmat_by_n(self):
        cmat = pd.DataFrame([[1, 0], [2, 3]], index=["s1", "s2"],
                            columns=["t1", "t2"])
        meta_cols = {"targetN", "sum_row"}
        meta_rows = {"sourceN", "sum_col"}
        # sourceN: t1 -> 2 sources, t2 -> 1 source
        out = sv.filtMat(cmat, axis=0, filt_range=[2, 2], by="N")
        assert [c for c in out.columns if c not in meta_cols] == ["t1"]
        out = sv.filtMat(cmat, axis=0, filt_range=[None, 1], by="N")
        assert [c for c in out.columns if c not in meta_cols] == ["t2"]
        out = sv.filtMat(cmat, axis=0, filt_range=[2, None], by="N")
        assert [c for c in out.columns if c not in meta_cols] == ["t1"]
        # targetN: s1 -> 1 target, s2 -> 2 targets
        out = sv.filtMat(cmat, axis=1, filt_range=[2, None], by="N")
        assert [i for i in out.index if i not in meta_rows] == ["s2"]
        out = sv.filtMat(cmat, axis=1, filt_range=[None, 1], by="N")
        assert [i for i in out.index if i not in meta_rows] == ["s1"]

    def test_stmat(self):
        mat = pd.DataFrame([[1.0, 0.0], [3.0, 2.0]], index=["s1", "s2"],
                           columns=["t1", "t2"])
        out = sv.stMat(mat, axis=0)  # column normalization
        assert out.iat[0, 0] == pytest.approx(0.25)
        assert out.iat[1, 0] == pytest.approx(0.75)
        assert out.iat[1, 1] == pytest.approx(1.0)
        out = sv.stMat(mat, axis=1)  # row normalization
        assert out.iat[0, 0] == pytest.approx(1.0)
        assert out.iat[1, 0] == pytest.approx(0.6)


# =============================================================================
# VisConnMat + CreateHeatmap
# =============================================================================

@pytest.fixture
def patched_colorbar(monkeypatch):
    """Work around source/plotly-version incompatibilities so the
    visualization code paths themselves can be exercised:
    - VisConnMat passes 'titleside' to the plotly ColorBar (rejected by the
      installed plotly version).
    - The layout uses the deprecated 'titlefont' axis property.
    """
    real_heatmap = sv.go.Heatmap

    def factory(*args, **kwargs):
        cb = kwargs.get("colorbar")
        if isinstance(cb, dict) and "titleside" in cb:
            kwargs["colorbar"] = {k: v for k, v in cb.items()
                                  if k != "titleside"}
        return real_heatmap(*args, **kwargs)

    monkeypatch.setattr(sv.go, "Heatmap", factory)

    real_figure = sv.go.Figure

    class FixedFigure(real_figure):
        def update_layout(self, *args, **kwargs):
            for axis_key in ("xaxis", "yaxis"):
                axis = kwargs.get(axis_key)
                if isinstance(axis, dict):
                    axis = {k: v for k, v in axis.items()
                            if k != "titlefont"}
                    kwargs[axis_key] = axis
            return super().update_layout(*args, **kwargs)

    monkeypatch.setattr(sv.go, "Figure", FixedFigure)


class TestVisConnMat:
    def _matrix(self):
        return pd.DataFrame([[10, 0], [3, 7]], index=["s1", "s2"],
                            columns=["t1", "t2"])

    def test_writes_html_synapses(self, tmp_path, patched_colorbar):
        out = tmp_path / "hm.html"
        sv.VisConnMat(self._matrix(), str(out), title="Connection Matrix",
                      showfig=False)
        assert out.exists()
        content = out.read_text()
        assert "Connection Matrix" in content

    def test_ratio_metric_and_log_scales(self, tmp_path, patched_colorbar):
        mat = pd.DataFrame([[0.5, 0.2], [0.0, 0.9]])
        for i, (title, scale) in enumerate([
            ("Ratio matrix", "linear"),
            ("Transmission probability", "log2"),
            ("Synapses", "log10"),
        ]):
            out = tmp_path / f"hm{i}.html"
            sv.VisConnMat(mat, str(out), title=title, showfig=False,
                          scale=scale)
            assert out.exists() and out.stat().st_size > 0

    def test_create_heatmap_workflow(self, tmp_path, patched_colorbar):
        hm = sv.CreateHeatmap(output_folder=str(tmp_path / "heat"),
                              showfig=False)
        assert (tmp_path / "heat").is_dir()
        mat = self._matrix()
        ret = hm.add_heatmap(mat, "conn_matrix_type", color_scale="green")
        assert ret is hm
        hm.add_heatmap(mat, "unknown_preset", color_scale="nope")
        hm.add_heatmaps({
            "ratio_matrix_type": mat,
            "transmission_mat": mat,
            "conn_matrix_bodyid": mat,
            "plain_matrix": mat,
        })
        assert len(hm.heatmaps) == 6
        files = hm.create_all()
        assert len(files) == 6
        for f in files:
            assert Path(f).exists()
        assert hm.heatmaps == []
        # empty queue branch
        assert hm.create_all() == []
        # clear() branch
        hm.add_heatmap(mat, "x")
        hm.clear()
        assert hm.heatmaps == []


# =============================================================================
# Path builders: split_path / path_filter / build_path_dataframe_from_paths
# =============================================================================

class TestPathBuilders:
    def test_split_path(self):
        assert sv.split_path(pd.DataFrame()).empty
        df = pd.DataFrame({"path": [[1, 2, 3], [4, 5]]})
        out = sv.split_path(df.copy())
        assert out["path"].tolist() == ["1->2->3", "4->5"]
        assert [list(x) for x in out["path_str"]] == [[1, 2, 3], [4, 5]]
        # already has path_str: list path gets stringified, path_str kept
        df2 = pd.DataFrame({"path": [[1, 2]], "path_str": [[1, 2]]})
        out2 = sv.split_path(df2.copy())
        assert out2["path"].tolist() == ["1->2"]

    def test_path_filter(self):
        df = pd.DataFrame({"path": ["A->B", "C->D"],
                           "path_str": ["A->B", "C->D"]})
        kept, excluded = sv.path_filter(df, None)
        assert len(kept) == 2 and excluded.empty
        kept, excluded = sv.path_filter(df.copy(), "B")
        assert kept["path"].tolist() == ["C->D"]
        assert excluded["path"].tolist() == ["A->B"]
        kept, excluded = sv.path_filter(df.copy(), ["B", "C"])
        assert kept.empty and len(excluded) == 2
        kept, excluded = sv.path_filter(pd.DataFrame(), "B")
        assert kept.empty and excluded.empty

    def test_path_filter_after_split_path(self):
        # Regression: split_path leaves the original list in path_str and the
        # string form in path; filtering must match against the strings
        # instead of silently no-op'ing on the list column.
        df = sv.split_path(pd.DataFrame({"path": [[1, 2, 3], [4, 5]]}))
        kept, excluded = sv.path_filter(df, "2")
        assert kept["path"].tolist() == ["4->5"]
        assert excluded["path"].tolist() == ["1->2->3"]

    def test_integer_synapse_count(self):
        assert sv._integer_synapse_count(5.0) == 5
        assert isinstance(sv._integer_synapse_count(5.0), int)
        assert sv._integer_synapse_count(5.5) == 5.5
        assert sv._integer_synapse_count("abc") == "abc"

    def _type_conn_df(self):
        return pd.DataFrame({
            "type_pre": ["A", "A", "B"],
            "type_post": ["B", "B", "C"],
            "weight": [10, 2, 4],
            "traversal_probability": [0.5, 0.5, 0.8],
            "connection_ratio": [0.4, 0.4, 0.2],
            "nt_type": ["ACH", "ACH", "GABA"],
        })

    def test_build_path_dataframe_pandas(self):
        df = sv.build_path_dataframe_from_paths(
            [["A", "B", "C"], ["A", "X", "C"]],
            self._type_conn_df(), targets=["C"], level="type",
            engine="pandas")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["path"] == "A->B->C"
        assert row["weights"] == [12, 4]
        assert row["min_weight"] == 4
        assert row["path_prob"] == pytest.approx(0.5 * 0.8)
        assert row["min_ratio"] == pytest.approx(0.2)
        assert row["length"] == 2
        assert row["nt_types"] == ["ACH|", "GABA"] or row["nt_types"][1] == "GABA"

    def test_build_path_dataframe_with_type_lookup(self):
        conn = pd.DataFrame({
            "bodyId_pre": [1, 2],
            "bodyId_post": [2, 3],
            "weight": [5, 6],
            "traversal_probability": [0.5, 0.5],
            "connection_ratio": [0.3, 0.3],
        })
        lookup = {"1": "A", "2": "B", "3": "C"}
        df = sv.build_path_dataframe_from_paths(
            [["1", "2", "3"]], conn, targets=["3"], level="bodyId",
            type_lookup=lookup, engine="pandas")
        assert df.iloc[0]["path"] == "1_A->2_B->3_C"
        assert df.iloc[0]["path_types"] == ["A", "B", "C"]

    def test_build_path_dataframe_empty(self):
        out = sv.build_path_dataframe_from_paths([], pd.DataFrame(),
                                                 targets=[], engine="pandas")
        assert isinstance(out, pd.DataFrame) and out.empty

    def test_build_path_dataframe_polars_engine(self):
        df = sv.build_path_dataframe_from_paths(
            [["A", "B"]], self._type_conn_df(), targets=["B"],
            level="type", engine="polars")
        assert isinstance(df, pl.DataFrame)
        assert df["path"].to_list() == ["A->B"]
        # auto engine with polars conn data
        df2 = sv.build_path_dataframe_from_paths(
            [["A", "B"]], pl.from_pandas(self._type_conn_df()),
            targets=["B"], level="type", engine="auto")
        assert isinstance(df2, pl.DataFrame)

    def test_enrich_engines_keep_untyped_neurons_as_bodyid(self):
        """Both enrichment engines resolve labels by the exclusive chain
        label -> type -> bodyId (groups: custom_group -> type -> bodyId):
        untyped neurons survive as their bodyId, typed-but-ungrouped
        neurons keep their type as the group label, and null group keys
        never reach the aggregated tables."""
        rows = {
            "bodyId_pre": ["1", "2", "5"],
            "bodyId_post": ["3", "4", "6"],
            "weight": [5, 6, 7],
            "post": [100, 100, 100],
            "type_pre": [None, "A", None],
            "type_post": ["B", None, None],
            "custom_group_pre": ["G1", None, None],
            "custom_group_post": [None, "G2", None],
        }
        expected_type = {("1", "B"), ("A", "4"), ("5", "6")}
        expected_group = {("G1", "B"), ("A", "G2"), ("5", "6")}
        frames = [("polars", pl.DataFrame(rows)),
                  ("pandas", pd.DataFrame(rows))]
        for engine, frame in frames:
            _, conn_t, conn_g = sv.EnrichConnectionTable(
                frame, dataset=None, script_path=None, engine=engine)
            type_edges = set(zip(conn_t["type_pre"], conn_t["type_post"]))
            assert type_edges == expected_type
            assert conn_g is not None
            group_edges = set(zip(conn_g["custom_group_pre"],
                                  conn_g["custom_group_post"]))
            assert group_edges == expected_group
            pre_nulls = (conn_g["custom_group_pre"].is_null()
                         if hasattr(conn_g["custom_group_pre"], "is_null")
                         else conn_g["custom_group_pre"].isna())
            post_nulls = (conn_g["custom_group_post"].is_null()
                          if hasattr(conn_g["custom_group_post"], "is_null")
                          else conn_g["custom_group_post"].isna())
            assert pre_nulls.sum() == 0
            assert post_nulls.sum() == 0


# =============================================================================
# Streaming polars pipeline
# =============================================================================

class TestStreamingPipeline:
    def _conn_df(self):
        return pd.DataFrame({
            "type_pre": ["A", "B"],
            "type_post": ["B", "C"],
            "weight": [10, 4],
            "traversal_probability": [0.5, 0.8],
            "connection_ratio": [0.4, 0.2],
            "nt_type": ["ACH", "GABA"],
        })

    def test_prepare_connection_data(self):
        conn = pd.DataFrame({
            "type_pre": ["A", "A"],
            "type_post": ["B", "B"],
            "weight": [3.0, 4.0],
            "traversal_probability": [0.4, 0.6],
            "connection_ratio": [0.2, 0.4],
            "nt_type": ["ACH", "ACH"],
        })
        out = sv.prepare_connection_data(conn, "type")
        assert isinstance(out, pl.DataFrame)
        row = out.row(0, named=True)
        assert row["src"] == "A" and row["tgt"] == "B"
        assert row["weight"] == 7
        assert row["traversal_probability"] == pytest.approx(0.5)
        assert row["nt_type"] == "ACH"
        # polars input passes through
        out2 = sv.prepare_connection_data(pl.from_pandas(conn), "type")
        assert out2.height == 1

    def test_process_batch_polars(self):
        df_conn = sv.prepare_connection_data(self._conn_df(), "type")
        # empty batch
        a, b = sv.process_batch_polars([], df_conn, "type")
        assert a.is_empty() and b.is_empty()
        # second path has an unknown edge -> zero weight -> filtered out
        df, excl = sv.process_batch_polars([["A", "B", "C"], ["A", "Z"]],
                                           df_conn, "type")
        assert df.height == 1
        assert df["path"].to_list() == ["A->B->C"]
        assert df["min_weight"].to_list() == [4]
        assert df["length"].to_list() == [2]
        assert df["nt_types"].to_list() == ['["ACH", "GABA"]']
        assert excl.is_empty()

    def test_process_batch_polars_null_nt_keeps_hop_slot(self):
        """A hop with no NT annotation must keep its (empty) slot in the
        serialized nt_types list: vispath reads the list POSITIONALLY, one
        entry per hop, and Polars list.join silently drops null entries —
        the shortened list shifted every later hop's NT onto the wrong
        edge (cholinergic aMe12 edges rendered GLUT/GABA)."""
        conn = pd.DataFrame({
            "type_pre": ["R", "A", "B"],
            "type_post": ["A", "B", "C"],
            "weight": [10, 4, 6],
            "traversal_probability": [0.5, 0.8, 0.9],
            "connection_ratio": [0.4, 0.2, 0.3],
            "nt_type": [None, "ACH", "GLUT"],  # R->A unannotated (photoreceptor-style)
        })
        df_conn = sv.prepare_connection_data(conn, "type")
        df, _ = sv.process_batch_polars([["R", "A", "B", "C"]], df_conn, "type")
        assert df.height == 1
        hops = len(df["path"][0].split("->")) - 1
        nt_list = ast.literal_eval(df["nt_types"][0])
        assert hops == 3
        assert len(nt_list) == hops            # one slot per hop, nulls kept
        assert nt_list == ["", "ACH", "GLUT"]  # positions preserved

    def test_process_batch_keyword_and_label_map(self):
        df_conn = sv.prepare_connection_data(self._conn_df(), "type")
        df, excl = sv.process_batch_polars(
            [["A", "B", "C"], ["A", "B"]], df_conn, "type",
            keyword_in_path_to_remove="C")
        assert df["path"].to_list() == ["A->B"]
        assert excl["path"].to_list() == ["A->B->C"]
        df2, _ = sv.process_batch_polars(
            [["A", "B"]], df_conn, "type",
            type_to_label_map={"A": "GA", "B": "GB"})
        assert df2["path"].to_list() == ["GA->GB"]

    def test_write_buffer_to_csv(self, tmp_path):
        d1 = pl.DataFrame({"a": [1], "b": ["x"]})
        d2 = pl.DataFrame({"a": [2], "b": ["y"]})
        out = tmp_path / "buf.csv"
        sv._write_buffer_to_csv([], out)  # empty buffer no-op
        assert not out.exists()
        sv._write_buffer_to_csv([d1], out)
        sv._write_buffer_to_csv([d2], out, append=True)
        df = pl.read_csv(out)
        assert df["a"].to_list() == [1, 2]

    def test_process_paths_streaming(self, tmp_path):
        paths = [["A", "B", "C"]] * 25
        out = tmp_path / "paths.csv"
        total = sv.process_paths_streaming(
            iter(paths), self._conn_df(), targets=["C"], output_path=str(out),
            level="type", batch_size=1, verbose=False)
        assert total == 25
        df = pl.read_csv(out)
        assert df.height == 25

    def test_process_paths_streaming_excluded(self, tmp_path):
        paths = [["A", "B", "C"]] * 3
        out = tmp_path / "paths.csv"
        excl = tmp_path / "excluded.csv"
        total = sv.process_paths_streaming(
            iter(paths), self._conn_df(), targets=["C"], output_path=str(out),
            excluded_path=str(excl), level="type", batch_size=2,
            keyword_in_path_to_remove="B", verbose=False)
        assert total == 0
        assert pl.read_csv(excl).height == 3


# =============================================================================
# _type_probability_series
# =============================================================================

class TestTypeProbabilitySeries:
    def _pairs(self):
        return pd.DataFrame({
            "type_pre": ["A", "A", "B"],
            "type_post": ["B", "B", "C"],
            "weight": [10, 2, 5],
            "block_probability": [0.5, 0.25, 0.4],
            "traversal_probability": [0.5, 0.75, 0.6],
        })

    def test_product(self):
        s = sv._type_probability_series(self._pairs(), "type_pre",
                                        "type_post", "product")
        assert s[("A", "B")] == pytest.approx(1 - 0.5 * 0.25)
        assert s[("B", "C")] == pytest.approx(0.6)

    def test_average(self):
        s = sv._type_probability_series(self._pairs(), "type_pre",
                                        "type_post", "average")
        expected = (10 * 0.5 + 2 * 0.75) / 12
        assert s[("A", "B")] == pytest.approx(expected)

    def test_ratio_returns_none(self):
        assert sv._type_probability_series(self._pairs(), "type_pre",
                                           "type_post", "ratio") is None


# =============================================================================
# build_bodyid_label_map
# =============================================================================

class FakeLabelMapper:
    def __init__(self, source=None, target=None, intermediate=None):
        self._source_mapping = source or {}
        self._target_mapping = target or {}
        self._intermediate_mapping = intermediate or {}


class TestBuildBodyIdLabelMap:
    def _neuron_df(self):
        return pl.DataFrame({
            "bodyId": [1, 2, 3, 4],
            "type": ["Mi1", "Mi1", "Tm3", "X"],
            "instance": ["Mi1_L", "Mi1_R", "Tm3_L", "X_L"],
        })

    def test_empty_inputs(self):
        assert sv.build_bodyid_label_map(None, "d", self._neuron_df()) == {}
        mapper = FakeLabelMapper()
        assert sv.build_bodyid_label_map(mapper, "d", pl.DataFrame()) == {}

    def test_no_bodyid_column(self):
        mapper = FakeLabelMapper(source={"G": {"d": [1]}})
        out = sv.build_bodyid_label_map(mapper, "d",
                                        pl.DataFrame({"type": ["a"]}))
        assert out == {}

    def test_direct_type_instance_expansion(self):
        mapper = FakeLabelMapper(
            source={"G1": {"test:v1": [1, "Mi1"]}},
            target={
                "G2": {"test_v1": ["Tm3_L"]},     # sanitized dataset key
                "G4": {"test:v1": ["nosuch"]},    # unknown -> stored as-is
            },
            intermediate={"G3": {"other_ds": ["99"]}},  # dataset mismatch
        )
        out = sv.build_bodyid_label_map(mapper, "test:v1", self._neuron_df())
        assert out["1"] == "G1"       # direct bodyId
        assert out["2"] == "G1"       # type expansion (Mi1)
        assert out["3"] == "G2"       # instance expansion (Tm3_L)
        assert out["nosuch"] == "G4"
        assert "4" not in out
        assert "99" not in out


# =============================================================================
# getCriteriaAndName / _get_coverage_notes
# =============================================================================

class TestMiscHelpers:
    def test_get_criteria_and_name(self, monkeypatch):
        import neuprint

        class FakeNC:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # neuprint.NeuronCriteria requires a live client; use a stand-in
        monkeypatch.setattr(neuprint, "NeuronCriteria", FakeNC)
        criteria, fname = sv.getCriteriaAndName(None)
        assert criteria is None and fname == "ALL"
        criteria, fname = sv.getCriteriaAndName([123])
        assert fname == "123" and criteria.kwargs == {"bodyId": [123]}
        criteria, fname = sv.getCriteriaAndName(["aMe.*"])
        assert fname == "aMe" and criteria.kwargs == {"instance": ["aMe.*"]}
        criteria, fname = sv.getCriteriaAndName(["Mi1"])
        assert fname == "Mi1" and criteria.kwargs == {"type": ["Mi1"]}
        criteria, fname = sv.getCriteriaAndName(["Mi1", "Tm3"])
        assert fname == "Mi1_etc"
        with pytest.raises(ValueError):
            sv.getCriteriaAndName("not-a-list")

    def test_get_coverage_notes(self):
        assert "Central brain" in sv._get_coverage_notes("hemibrain:v1.2.1")
        assert "not available" in sv._get_coverage_notes("unknown-ds")
        assert "FAFB" in sv._get_coverage_notes("flywire_70k")


# =============================================================================
# _VisConnMatInteractive_local (the big HTML generator, lines 2726-5590)
# =============================================================================
#
# VisConnMatInteractive prefers vispath_pkg when importable; setting
# sys.modules["vispath_pkg"] = None makes the ``from vispath_pkg import ...``
# raise ImportError so execution falls through to the local implementation.

@pytest.fixture
def local_interactive(monkeypatch):
    monkeypatch.setitem(sys.modules, "vispath_pkg", None)


class TestVisConnMatInteractiveLocal:
    def _cmat(self, rows, cols, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            rng.integers(0, 20, size=(rows, cols)).astype(float),
            index=[f"r{i}" for i in range(rows)],
            columns=[f"c{j}" for j in range(cols)],
        )

    def test_single_matrix_small(self, tmp_path, local_interactive):
        out = tmp_path / "conn_weight.html"
        cmat = self._cmat(5, 4)
        sv.VisConnMatInteractive(cmat, str(out), title="Synapse counts",
                                 showfig=False, verbose=True)
        assert out.exists() and out.stat().st_size > 1000
        text = out.read_text(encoding="utf-8")
        assert "plotly" in text.lower()
        assert "r0" in text and "c0" in text

    def test_ratio_title_metric(self, tmp_path, local_interactive):
        out = tmp_path / "conn_ratio.html"
        cmat = self._cmat(4, 4, seed=1) / 20.0
        sv.VisConnMatInteractive(cmat, str(out), title="Connection ratio",
                                 showfig=False, verbose=False)
        assert out.exists()
        assert "'ratio'" in out.read_text(encoding="utf-8")

    def test_probability_filename_metric(self, tmp_path, local_interactive):
        out = tmp_path / "transmission_prob.html"
        cmat = self._cmat(4, 3, seed=2) / 20.0
        sv.VisConnMatInteractive(cmat, str(out), title="", showfig=False,
                                 verbose=False)
        assert out.exists()

    def test_matrices_dict_mode(self, tmp_path, local_interactive):
        out = tmp_path / "multi.html"
        base = self._cmat(4, 4, seed=3)
        matrices = {
            "weight": base,
            "ratio": base / base.values.max(),
            "probability": base / base.values.max(),
        }
        sv.VisConnMatInteractive(base, str(out), title="Multi",
                                 matrices_dict=matrices, showfig=False,
                                 verbose=False)
        text = out.read_text(encoding="utf-8")
        assert out.exists() and "weight" in text and "probability" in text

    def test_conn_df_type_lookup(self, tmp_path, local_interactive):
        out = tmp_path / "bodyid.html"
        cmat = pd.DataFrame(
            [[5.0, 0.0], [3.0, 2.0]], index=["10", "20"], columns=["30", "40"])
        conn = pd.DataFrame({
            "bodyId_pre": [10, 20], "type_pre": ["Mi1", "Tm3"],
            "bodyId_post": [30, 40], "type_post": ["DN1", "DN2"],
        })
        sv.VisConnMatInteractive(cmat, str(out), title="bodyId",
                                 conn_df=conn, showfig=False, verbose=False)
        text = out.read_text(encoding="utf-8")
        assert "Mi1" in text and "DN1" in text

    def test_clustering_failure_falls_back(self, tmp_path, local_interactive,
                                           monkeypatch):
        import scipy.cluster.hierarchy as hierarchy

        def broken_linkage(*a, **k):
            raise RuntimeError("clustering unavailable")

        monkeypatch.setattr(hierarchy, "linkage", broken_linkage)
        out = tmp_path / "noclust.html"
        sv.VisConnMatInteractive(self._cmat(4, 4, seed=4), str(out),
                                 title="noclust", showfig=False, verbose=True)
        assert out.exists()

    def test_large_matrix_paths(self, tmp_path, local_interactive):
        # >100 rows triggers is_large; keep cols small so hover loop is cheap.
        out = tmp_path / "large.html"
        cmat = self._cmat(101, 5, seed=5)
        sv.VisConnMatInteractive(cmat, str(out), title="big", showfig=False,
                                 verbose=False)
        assert out.exists()
        out2 = tmp_path / "large_ratio.html"
        sv.VisConnMatInteractive(cmat / 20.0, str(out2), title="ratio",
                                 showfig=False, verbose=False)
        assert out2.exists()

    def test_wrapper_falls_back_to_local(self, tmp_path, local_interactive):
        out = tmp_path / "wrapper.html"
        sv.VisConnMatInteractive(self._cmat(3, 3, seed=6), str(out),
                                 title="wrap", showfig=False, verbose=False)
        assert out.exists()


class TestVisConnMatInteractiveDendrogram:
    """Dendrogram support in the live vispath_pkg heatmap page.

    These target vispath_pkg directly (not the deprecated statvis local
    copy) because the dendrogram template changes live there.
    """

    def _cmat(self, rows, cols, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            rng.integers(0, 20, size=(rows, cols)).astype(float),
            index=[f"r{i}" for i in range(rows)],
            columns=[f"c{j}" for j in range(cols)],
        )

    def _render(self, tmp_path, cmat, **kwargs):
        import vispath_pkg
        out = tmp_path / "dendro.html"
        vispath_pkg.VisConnMatInteractive(
            cmat, str(out), title="dendro", showfig=False, verbose=False,
            **kwargs)
        assert out.exists()
        return out.read_text(encoding="utf-8")

    def _clustering_results(self, text):
        import json
        import re
        match = re.search(r"const clusteringResults = (\{.*?\});\n", text)
        assert match, "clusteringResults payload not found"
        return json.loads(match.group(1))

    def test_linkage_matrices_embedded_per_method(self, tmp_path):
        pytest.importorskip("vispath_pkg")
        text = self._render(tmp_path, self._cmat(6, 5, seed=11))
        results = self._clustering_results(text)
        assert set(results) == {"ward", "average", "complete", "single"}
        for method, payload in results.items():
            assert len(payload["row_order"]) == 6
            assert len(payload["col_order"]) == 5
            # scipy linkage matrices: (n-1) merges x 4 columns
            assert len(payload["row_linkage"]) == 5
            assert all(len(row) == 4 for row in payload["row_linkage"])
            assert len(payload["col_linkage"]) == 4
            assert all(len(row) == 4 for row in payload["col_linkage"])

    def test_toggle_and_builder_markers_present(self, tmp_path):
        pytest.importorskip("vispath_pkg")
        text = self._render(tmp_path, self._cmat(5, 4, seed=12))
        assert "buildDendrogramSegments" in text
        assert "buildDendrogramTrace" in text
        assert "alignDendrogramBands" in text
        # Toggle button (active by default when clustered) + line-width slider
        assert 'id="dendrogramToggleBtn" class="active" onclick="toggleDendrogram()"' in text
        assert 'id="dendroLineWidthSlider"' in text
        assert "updateDendroLineWidth" in text
        assert "dendroLineWidth = 1" in text
        assert 'id="dendroBandSizeSlider"' in text
        assert "updateDendroBandSize" in text
        assert "dendroBandSize = 90" in text
        assert "showDendrogram = true" in text
        # The square-cells lock must not touch dendrogram visibility: the
        # suppression machinery must stay gone
        assert "dendroSuppressedByLock" not in text

    def test_init_clustered_false_leaves_toggle_off(self, tmp_path):
        pytest.importorskip("vispath_pkg")
        text = self._render(tmp_path, self._cmat(5, 4, seed=13),
                            init_clustered=False)
        # Linkage data still ships so switching to Clustered works client-side
        assert '"row_linkage"' in text
        assert "showDendrogram = false" in text
        assert 'id="dendrogramToggleBtn" class="active"' not in text

    def test_single_row_matrix_has_column_linkage_only(self, tmp_path):
        pytest.importorskip("vispath_pkg")
        single = pd.DataFrame(
            [[5.0, 0.0, 2.0, 9.0, 1.0]], index=["only"], columns=[f"c{j}" for j in range(5)])
        text = self._render(tmp_path, single)
        results = self._clustering_results(text)
        for payload in results.values():
            assert "row_linkage" not in payload
            assert payload["row_order"] == [0]
            assert len(payload["col_linkage"]) == 4

    def test_clustering_failure_omits_dendrogram_data(self, tmp_path,
                                                      monkeypatch):
        pytest.importorskip("vispath_pkg")
        import scipy.cluster.hierarchy as hierarchy

        def broken_linkage(*a, **k):
            raise RuntimeError("clustering unavailable")

        monkeypatch.setattr(hierarchy, "linkage", broken_linkage)
        text = self._render(tmp_path, self._cmat(4, 4, seed=14))
        results = self._clustering_results(text)
        assert results == {}
        assert "clusteringAvailable = false" in text


# =============================================================================
# VisConnMat large-matrix optimization branch (2444, 2510-2632)
# =============================================================================

class TestVisConnMatLarge:
    def test_large_matrix_heatmap(self, tmp_path, patched_colorbar):
        rng = np.random.default_rng(7)
        cmat = pd.DataFrame(
            rng.integers(0, 30, size=(101, 4)).astype(float),
            index=[f"s{i}" for i in range(101)],
            columns=[f"t{j}" for j in range(4)])
        out = tmp_path / "large_heatmap.html"
        sv.VisConnMat(cmat, filename=str(out), title="Synapses",
                      showfig=False, scale="log2")
        assert out.exists()

    def test_large_sparse_rounding(self, tmp_path, patched_colorbar):
        cmat = pd.DataFrame(np.zeros((101, 4)))
        cmat.iloc[0, 0] = 5.0
        out = tmp_path / "sparse.html"
        sv.VisConnMat(cmat, filename=str(out), title="sparse", showfig=False)
        assert out.exists()


# =============================================================================
# SankeyDirect (6672-6725)
# =============================================================================

class TestSankeyDirect:
    def test_writes_html(self, tmp_path):
        cmat = pd.DataFrame([[3.0, 0.0], [1.0, 4.0]],
                            index=["A", "B"], columns=["X", "Y"])
        out = tmp_path / "sankey.html"
        sv.SankeyDirect(cmat, str(out), showfig=False)
        assert out.exists() and "plotly" in out.read_text().lower()

    def test_no_positive_values_returns_early(self, tmp_path):
        cmat = pd.DataFrame([[0.0, 0.0]], index=["A"], columns=["X", "Y"])
        out = tmp_path / "empty_sankey.html"
        sv.SankeyDirect(cmat, str(out), showfig=False)
        assert not out.exists()


# =============================================================================
# Vis3S soma path (6733-6979)
# =============================================================================

class TestVis3S:
    def test_soma_plot_saves_png(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        df = pd.DataFrame({
            "bodyId": [1, 2, 3],
            "type": ["Mi1", "Mi1", "Tm3"],
            "somaLocation": ["[100, 200, 300]", "[150, 250, 350]",
                             "[400, 500, 600]"],
            "somaRadius": [100, 120, 80],
        })
        save_path = str(tmp_path / "vis3s")
        sv.Vis3S(df, toPlot="soma", save_path=save_path, showfig=False,
                 dataset="test", show_mesh=False, dpi=50)
        assert os.path.exists(save_path + ".png")


# =============================================================================
# build_synapse_mesh / build_site_mesh (7018-7380)
# =============================================================================

class TestMeshBuilders:
    def test_synapse_mesh_sphere(self):
        pre = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        post = np.array([[1.0, 1.0, 1.0], [11.0, 11.0, 11.0]])
        mesh = sv.build_synapse_mesh(pre, post, mode="sphere", size=50.0)
        assert mesh is not None

    def test_synapse_mesh_cone_and_tetra(self):
        pre = np.array([[0.0, 0.0, 0.0]])
        post = np.array([[5.0, 5.0, 5.0]])
        assert sv.build_synapse_mesh(pre, post, mode="cone", size=20.0) is not None
        assert sv.build_synapse_mesh(pre, post, mode="tetrahedron", size=20.0) is not None

    def test_synapse_mesh_array_size(self):
        pre = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        post = np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        mesh = sv.build_synapse_mesh(pre, post, mode="sphere",
                                     size=np.array([10.0, 20.0]))
        assert mesh is not None

    def test_synapse_mesh_empty(self):
        pre = np.empty((0, 3))
        post = np.empty((0, 3))
        mesh = sv.build_synapse_mesh(pre, post, mode="sphere")
        assert mesh is not None

    def test_synapse_mesh_unknown_mode(self):
        with pytest.raises(ValueError):
            sv.build_synapse_mesh(np.array([[0.0, 0, 0]]),
                                  np.array([[1.0, 1, 1]]), mode="bogus")

    def test_site_mesh_sphere_and_cone(self):
        coords = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
        assert sv.build_site_mesh(coords, mode="sphere") is not None
        assert sv.build_site_mesh(coords, mode="cone") is not None

    def test_site_mesh_empty_and_1d(self):
        assert sv.build_site_mesh(np.empty((0, 3)), mode="sphere") is not None
        assert sv.build_site_mesh(np.array([1.0, 2.0, 3.0]), mode="sphere") is not None

    def test_site_mesh_unknown_mode(self):
        with pytest.raises(ValueError):
            sv.build_site_mesh(np.array([[0.0, 0, 0]]), mode="bogus")


# =============================================================================
# _get_neuron_df flywire + missing branches (502-539)
# =============================================================================

class TestGetNeuronDf:
    def test_flywire_cache_hit(self, monkeypatch):
        df = pd.DataFrame({"bodyId": ["1"], "type": ["Mi1"]})
        sv._NEURON_DF_CACHE["flywire_fafb"] = {"neuron_df": df}
        try:
            out = sv._get_neuron_df("fafb")
            assert list(out["bodyId"]) == ["1"]
        finally:
            sv._NEURON_DF_CACHE.pop("flywire_fafb", None)

    def test_flywire_load_failure(self, monkeypatch):
        sv._NEURON_DF_CACHE.pop("flywire_fafb", None)
        # fafb_utils import / resolve path failing raises FileNotFoundError
        monkeypatch.setattr(sv, "resolve_flywire_dataset_dir",
                            lambda root, ds: None)
        with pytest.raises(FileNotFoundError):
            sv._get_neuron_df("fafb")

    def test_standard_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sv.os.path, "exists", lambda p: False)
        with pytest.raises(FileNotFoundError):
            sv._get_neuron_df("nonexistent-dataset:v9.9")


# =============================================================================
# _ensure_local_dataset_files (445-478)
# =============================================================================

class TestEnsureLocalDatasetFiles:
    def test_failed_registry_raises(self, monkeypatch):
        sv._FAILED_DATASET_DOWNLOADS.add("missing-ds_v1")
        try:
            with pytest.raises(FileNotFoundError):
                sv._ensure_local_dataset_files("missing-ds:v1")
        finally:
            sv._FAILED_DATASET_DOWNLOADS.discard("missing-ds_v1")

    def test_pull_success_creates_files(self, tmp_path, monkeypatch):
        dataset_dir = tmp_path / "datasets" / "fake_ds_v1"
        body = str(dataset_dir / "fake_ds_v1_allneurons")

        def fake_path_body(dataset):
            return "fake_ds_v1", str(dataset_dir), body

        def fake_pull(dataset, save_path=None, omitNoneType=False, client=None,
                      **kwargs):
            # Mirror the real pull_dataset contract: it creates the
            # destination folder itself right before writing.
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            pd.DataFrame({"bodyId": [1]}).to_csv(
                save_path + "_neuron_df.csv", index=False)
            pd.DataFrame({"roi": ["R"]}).to_parquet(
                save_path + "_roi_count_df.parquet")

        monkeypatch.setattr(sv, "_get_dataset_path_body", fake_path_body)
        monkeypatch.setattr(sv, "pull_dataset", fake_pull)
        monkeypatch.setattr(sv, "clear_neuron_cache", lambda *a, **k: None)
        normalized, out_body = sv._ensure_local_dataset_files(
            "fake-ds:v1", verbose=False)
        assert normalized == "fake_ds_v1"
        assert os.path.exists(out_body + "_neuron_df.csv")

    def test_pull_failure_recorded(self, tmp_path, monkeypatch):
        dataset_dir = tmp_path / "datasets" / "fail_ds_v1"

        def fake_path_body(dataset):
            return "fail_ds_v1", str(dataset_dir), str(
                dataset_dir / "fail_ds_v1_allneurons")

        def failing_pull(*a, **k):
            raise RuntimeError("offline")

        monkeypatch.setattr(sv, "_get_dataset_path_body", fake_path_body)
        monkeypatch.setattr(sv, "pull_dataset", failing_pull)
        with pytest.raises(RuntimeError):
            sv._ensure_local_dataset_files("fail-ds:v1")
        assert "fail_ds_v1" in sv._FAILED_DATASET_DOWNLOADS
        sv._FAILED_DATASET_DOWNLOADS.discard("fail_ds_v1")


# =============================================================================
# _build_dataset_metadata branches (1345, 1358, 1364-1372)
# =============================================================================

class TestBuildDatasetMetadata:
    def test_no_type_column(self):
        ndf = pd.DataFrame({"bodyId": [1, 2], "pre": [10, 20], "post": [5, 6]})
        roi = pd.DataFrame({"bodyId": [1, 2], "roi": ["AL", "AL"],
                            "pre": [10, 20], "post": [5, 6]})
        meta = sv._build_dataset_metadata("hemibrain", ndf, roi)
        assert meta["neuron_counts"]["typed"] == 0
        assert meta["synapse_counts"]["total"] == 41
        assert meta["roi_coverage"]["roi_list"] == ["AL"]
        assert meta["roi_coverage"]["neuron_counts_per_roi"]["AL"] == 2

    def test_client_primary_rois(self):
        class ClientStub:
            primary_rois = ["LH", "AL"]

        ndf = pd.DataFrame({"bodyId": [1], "type": ["Mi1"], "pre": [1],
                            "post": [0]})
        roi = pd.DataFrame({"bodyId": [1], "roi": ["LH"], "pre": [1],
                            "post": [0]})
        meta = sv._build_dataset_metadata("hemibrain", ndf, roi,
                                          client=ClientStub())
        assert meta["roi_coverage"]["roi_list"] == ["LH", "AL"]


# =============================================================================
# LogInHemibrain (1274-1290)
# =============================================================================

class TestLogInHemibrain:
    def test_returns_client_and_dataset(self, monkeypatch):
        class ClientStub:
            def __init__(self, address, dataset=None, token=None):
                self.address = address
                self.dataset = dataset
                self.token = token

        monkeypatch.setattr(sv, "Client", ClientStub)
        client, dataset = sv.LogInHemibrain("sometoken")
        assert dataset == "hemibrain:v1.2.1"
        assert isinstance(client, ClientStub)


# =============================================================================
# pull_dataset with injected fetch_fn (1497-1728)
# =============================================================================

class TestPullDataset:
    def _frames(self, body_ids):
        ndf = pd.DataFrame({"bodyId": list(body_ids),
                            "type": ["Mi1"] * len(body_ids),
                            "pre": [1] * len(body_ids),
                            "post": [2] * len(body_ids),
                            "roiInfo": ["AL"] * len(body_ids)})
        roi = pd.DataFrame({"bodyId": list(body_ids),
                            "roi": ["AL"] * len(body_ids),
                            "pre": [1] * len(body_ids),
                            "post": [2] * len(body_ids)})
        return ndf, roi

    def test_cancelled_before_start(self):
        import threading
        ev = threading.Event()
        ev.set()
        with pytest.raises(sv.DatasetPullCancelled):
            sv.pull_dataset("hemibrain:v1.2.1", cancel_event=ev)

    def test_sequential_pull_with_fetch_fn(self, tmp_path, monkeypatch):
        import neuprint

        class CriteriaStub:
            def __init__(self, bodyId=None, **kwargs):
                self.bodyId = bodyId

        monkeypatch.setattr(neuprint, "NeuronCriteria", CriteriaStub)

        class ClientStub:
            def fetch_custom(self, query):
                return pd.DataFrame({"bodyId": [1, 2, 3, 4, 5]})

        def fetch_fn(criteria, client=None):
            return self._frames(criteria.bodyId)

        save_path = str(tmp_path / "ds_allneurons")
        sv.pull_dataset("hemibrain:v1.2.1", save_path=save_path,
                        client=ClientStub(), fetch_fn=fetch_fn,
                        batch_size=2, omitNoneType=True)
        assert os.path.exists(save_path + "_neuron_df.csv")
        assert os.path.exists(save_path + "_roi_count_df.parquet")
        assert os.path.exists(str(tmp_path / "ds_metadata.json"))

    def test_parallel_pull_with_progress(self, tmp_path, monkeypatch):
        import neuprint

        class CriteriaStub:
            def __init__(self, bodyId=None, **kwargs):
                self.bodyId = bodyId

        monkeypatch.setattr(neuprint, "NeuronCriteria", CriteriaStub)

        class ClientStub:
            def fetch_custom(self, query):
                return pd.DataFrame({"bodyId": [1, 2, 3, 4]})

        def fetch_fn(criteria, client=None):
            return self._frames(criteria.bodyId)

        progress = []
        save_path = str(tmp_path / "ds_allneurons")
        sv.pull_dataset("hemibrain:v1.2.1", save_path=save_path,
                        client=ClientStub(), fetch_fn=fetch_fn,
                        batch_size=2, max_workers=2,
                        progress_callback=lambda cur, tot: progress.append((cur, tot)))
        assert os.path.exists(save_path + "_neuron_df.csv")
        assert progress and progress[-1][0] == 4


# =============================================================================
# getNeurons legacy query pipeline (nested groups, dict filters, None query)
# =============================================================================

def _getneurons_rdf():
    return pd.DataFrame({"bodyId": [1, 2, 3, 4], "roi": ["r"] * 4})


def _patch_getneurons_local(monkeypatch):
    monkeypatch.setattr(
        sv, "_ensure_local_dataset_files",
        lambda dataset, client=None, verbose=True: ("synth_ds", "ignored"))
    monkeypatch.setattr(
        sv, "_get_cached_neuron_df",
        lambda dn, path: (_synthetic_ndf(), _getneurons_rdf()))
    monkeypatch.setattr(sv, "_get_cached_neuron_search", lambda dataset: None)


class TestGetNeuronsLegacyPipeline:
    def test_none_query_uses_api_fetch(self, monkeypatch):
        calls = {}

        def fake_fetch(criteria, client=None):
            calls["criteria"] = criteria
            return _synthetic_ndf(), _getneurons_rdf()

        monkeypatch.setattr(sv, "fetch_neurons", fake_fetch)
        ndf, rdf, name, criteria = sv.getNeurons(
            None, dataset="hemibrain:v1.2.1", verbose=False)
        assert name == "ALL"
        assert criteria is None
        assert len(ndf) == 4
        assert calls["criteria"] is None

    def test_empty_list_returns_all_typed(self, monkeypatch):
        _patch_getneurons_local(monkeypatch)
        ndf, rdf, name, _ = sv.getNeurons(
            [], dataset="hemibrain:v1.2.1", verbose=False)
        assert name == "allneurons"
        assert len(ndf) == 4
        assert set(ndf["type"]) == {"Mi1", "Tm3", "DN1"}

    def test_nested_list_custom_groups(self, monkeypatch):
        _patch_getneurons_local(monkeypatch)
        ndf, rdf, name, _ = sv.getNeurons(
            [["Mi1", "Tm3"], "DN1"],
            dataset="hemibrain:v1.2.1",
            custom_group_names=["vis_group"],
            verbose=False)
        assert "custom_group" in ndf.columns
        assert len(ndf) == 4
        grouped = set(ndf.loc[ndf["type"].isin(["Mi1", "Tm3"]),
                              "custom_group"])
        assert grouped == {"vis_group"}
        # two group names -> joined auto name
        assert name == "vis_group_DN1"

    def test_nested_list_auto_group_name(self, monkeypatch):
        _patch_getneurons_local(monkeypatch)
        ndf, _, name, _ = sv.getNeurons(
            [["Mi1", "Tm3"]], dataset="hemibrain:v1.2.1", verbose=False)
        assert name == "Mi1_etc"
        assert set(ndf["custom_group"]) == {"Mi1_etc"}

    def test_dict_filter_branch(self, monkeypatch):
        rdf = _getneurons_rdf()
        monkeypatch.setattr(
            sv, "_get_cached_neuron_df",
            lambda dn, path: (_synthetic_ndf(), rdf))
        matched, roi, name, criteria = sv.getNeurons(
            {"startswith": ["Mi"]},
            dataset="hemibrain:v1.2.1", verbose=False)
        assert criteria is None
        assert len(matched) == 2
        assert set(matched["type"]) == {"Mi1"}


# =============================================================================
# EnrichConnectionTable dataset-discovery and target-neuron branches
# =============================================================================

class _GetLabelMapper:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_label(self, dataset, key):
        return self._mapping.get(str(key), str(key))


class TestEnrichDatasetDiscoveryBranches:
    def _conn(self):
        return pd.DataFrame({
            "bodyId_pre": ["1", "2"],
            "bodyId_post": ["3", "4"],
            "weight": [5, 6],
            "post": [100, 120],
        })

    def test_legacy_root_dataset_path(self, tmp_path):
        # dataset table only at the legacy root-level location
        clean = "synth_ds"
        datasets = tmp_path / "datasets"
        datasets.mkdir()
        table = datasets / f"{clean}_allneurons_neuron_df.csv"
        pd.DataFrame({
            "bodyId": ["1", "2", "3", "4"],
            "type": ["A", "A", "B", "B"],
            "post": [100, 100, 120, 120],
        }).to_csv(table, index=False)
        conn_e, conn_t, conn_g = sv.EnrichConnectionTable(
            self._conn(), dataset="synth:ds", script_path=str(tmp_path),
            engine="pandas")
        assert conn_e is not None and len(conn_e) == 2
        assert conn_t is not None

    def test_globbed_subdir_dataset_path(self, tmp_path):
        clean = "synth_ds"
        subdir = tmp_path / "datasets" / clean
        subdir.mkdir(parents=True)
        pd.DataFrame({
            "bodyId": ["1", "2", "3", "4"],
            "type": ["A", "A", "B", "B"],
            "post": [100, 100, 120, 120],
        }).to_csv(subdir / "custom_allneurons_neuron_df.csv", index=False)
        conn_e, conn_t, _ = sv.EnrichConnectionTable(
            self._conn(), dataset="synth:ds", script_path=str(tmp_path),
            engine="pandas")
        assert len(conn_e) == 2

    def test_target_neurons_df_with_label_mapper(self, tmp_path):
        # No local dataset at all -> use_local=False -> target_neurons_df path
        target = pd.DataFrame({
            "bodyId": ["3", "4"],
            "type": ["B", "B"],
            "post": [120, 120],
        })
        mapper = _GetLabelMapper({"B": "Bstd"})
        conn_e, conn_t, _ = sv.EnrichConnectionTable(
            self._conn(), dataset="synth:ds", script_path=str(tmp_path),
            target_neurons_df=target, label_mapper=mapper, engine="pandas")
        assert len(conn_e) == 2
        assert "connection_ratio" in conn_e.columns

    def test_no_local_no_target_warns(self, tmp_path, capsys):
        conn_e, conn_t, _ = sv.EnrichConnectionTable(
            self._conn(), dataset="synth:ds", script_path=str(tmp_path),
            engine="pandas")
        assert "Could not fetch neuron info" in capsys.readouterr().out
        assert len(conn_e) == 2
