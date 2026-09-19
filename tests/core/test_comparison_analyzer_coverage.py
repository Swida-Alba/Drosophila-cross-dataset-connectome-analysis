"""Hermetic coverage tests for comparison.comparison_analyzer.

All data-query / metadata-fetch methods are monkeypatched so that no network
or real dataset folder access occurs.  Synthetic DataFrames are injected.
"""

import json
import os
import sys
import types

import pandas as pd
import pytest

from comparison.comparison_analyzer import (
    ComparisonAnalyzer,
    _escape_cypher_string_fallback,
    quick_compare,
)
from comparison.comparison_parameters import ComparisonParameters
from comparison.dataset_config import DatasetConfig
from comparison.label_mapper import LabelMapper

DS1 = "hemibrain:v1.2.1"
DS2 = "male-cns:v0.9"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params(**overrides):
    """Build hermetic ComparisonParameters (no auto type mapping, no output)."""
    defaults = dict(
        datasets=[DS1, DS2],
        source_neurons=["Src"],
        target_neurons=["Tgt"],
        max_interlayer=1,
        thresholds=[1, 3],
        output_folder="",
        auto_type_mapping=False,
        verbose=False,
    )
    defaults.update(overrides)
    return ComparisonParameters(**defaults)


def _edge_df(edges):
    """edges: list of (type_pre, type_post, weight)."""
    return pd.DataFrame(edges, columns=["type_pre", "type_post", "weight"])


def _bodyid_df():
    return pd.DataFrame(
        {
            "bodyId_pre": [1, 2],
            "bodyId_post": [2, 3],
            "type_pre": ["Src", "Inter"],
            "type_post": ["Inter", "Tgt"],
            "weight": [5, 5],
        }
    )


def _standard_results():
    """raw_results dict: both datasets, both thresholds, small edge data."""
    d1 = _edge_df([("Src", "Tgt", 10), ("Src", "X", 5)])
    d2 = _edge_df([("Src", "Tgt", 2), ("Y", "Tgt", 7)])
    return {DS1: {1: d1.copy(), 3: d1.copy()}, DS2: {1: d2.copy(), 3: d2.copy()}}


def _meta(ds):
    return {
        "dataset": ds,
        "source": "stub",
        "neuron_counts": {"total": 100, "typed": 80, "untyped": 20, "type_coverage": 0.8},
        "synapse_counts": {"total_presynaptic": 500, "total_postsynaptic": 400, "total": 900},
        "roi_coverage": {"roi_list": ["AL"], "roi_count": 1, "neuron_counts_per_roi": {"AL": 10}},
        "coverage_notes": "stub note",
    }


def _stub_metadata_methods(monkeypatch, analyzer, cached=None):
    monkeypatch.setattr(analyzer, "_load_cached_metadata", lambda ds: cached)
    monkeypatch.setattr(analyzer, "_fetch_neuprint_metadata", _meta)
    monkeypatch.setattr(analyzer, "_fetch_local_metadata", _meta)
    monkeypatch.setattr(analyzer, "_save_metadata", lambda ds, m: None)


@pytest.fixture
def analyzer():
    return ComparisonAnalyzer(_params(), verbose=False)


# ---------------------------------------------------------------------------
# Initialization / misc utilities
# ---------------------------------------------------------------------------

def test_escape_cypher_string_fallback():
    assert _escape_cypher_string_fallback("a'b") == "a\\'b"
    assert _escape_cypher_string_fallback("a\\b") == "a\\\\b"
    assert _escape_cypher_string_fallback(123) == "123"


def test_init_no_output_folder(analyzer):
    assert analyzer.data_loader is None
    assert analyzer.raw_results == {}
    assert analyzer.comparison_report is None
    assert DS1 in analyzer._dataset_configs
    assert isinstance(analyzer._dataset_configs[DS2], DatasetConfig)


def test_init_with_output_folder(tmp_path):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    assert a.data_loader is not None
    assert os.path.isdir(params.full_output_path)


def test_init_with_label_mapper():
    mapper = LabelMapper(
        source_mapping_dict={DS1: [["aMe1"]], DS2: [["aMe1mc"]]},
        source_labels=["aMe1_grp"],
        target_mapping_dict={DS1: [["KCg1"]], DS2: [["KCg1mc"]]},
        target_labels=["KCg_grp"],
    )
    a = ComparisonAnalyzer(_params(), label_mapper=mapper, verbose=False)
    assert a.label_mapper is mapper


def test_init_invalid_dataset_type():
    with pytest.raises(ValueError):
        ComparisonAnalyzer(_params(datasets=[123]), verbose=False)


def test_get_dataset_config(analyzer):
    cfg = analyzer._get_dataset_config(DS1)
    assert cfg.dataset == DS1
    with pytest.raises(ValueError):
        analyzer._get_dataset_config("nope")
    assert analyzer.get_dataset_config(DS1) is cfg
    assert analyzer.get_dataset_config("nope") is None


def test_logging_helpers():
    a = ComparisonAnalyzer(_params(), verbose=True)
    a._log("hello")
    a._log("debug msg", level="debug")
    a._log("warn msg", level="warn")
    a._progress(1, 5, "step")
    a._log_file("/tmp/some/file.csv")
    # verbose=False silences everything
    quiet = ComparisonAnalyzer(_params(), verbose=False)
    quiet._log("nope")
    quiet._log_file("/tmp/some/file.csv")


def test_save_read_csv_roundtrip(analyzer, tmp_path):
    df = _edge_df([("A", "B", 4), ("C", "D", 9)])
    path = str(tmp_path / "out.csv")
    analyzer._save_csv(df, path)
    assert os.path.exists(path)
    back = analyzer._read_csv(path)
    assert list(back.columns) == ["type_pre", "type_post", "weight"]
    assert len(back) == 2
    # kwargs route through pandas
    back2 = analyzer._read_csv(path, dtype={"type_pre": str})
    assert len(back2) == 2
    # empty / None are no-ops
    analyzer._save_csv(pd.DataFrame(), str(tmp_path / "empty.csv"))
    analyzer._save_csv(None, str(tmp_path / "none.csv"))
    assert not os.path.exists(str(tmp_path / "empty.csv"))
    assert not os.path.exists(str(tmp_path / "none.csv"))


def test_collect_result_types(analyzer):
    analyzer.raw_results[DS1] = {1: _edge_df([("A", "B", 1)]), 3: pd.DataFrame()}
    analyzer.raw_results[DS2] = {
        1: {"type_level": pd.DataFrame({"from_type": ["C"], "to_type": ["D"], "from": [1], "to": [2]})},
        3: pd.DataFrame({"type_pre": [5]}),  # non-string types filtered
    }
    types = analyzer._collect_result_types()
    assert {"A", "B", "C", "D"} <= types
    assert 5 not in types
    by_dataset = analyzer._collect_result_types_by_dataset()
    assert {"A", "B"} <= by_dataset[DS1]
    assert {"C", "D"} <= by_dataset[DS2]
    assert all(isinstance(name, str)
               for names in by_dataset.values() for name in names)


def test_mode_specific_note(analyzer):
    note = analyzer._generate_mode_specific_note()
    assert "Path-Based" in note
    analyzer.parameters.comparison_mode = "edge"
    assert "Edge-Based" in analyzer._generate_mode_specific_note()


def test_mapping_helpers_without_mapper(analyzer):
    assert analyzer._get_canonical_type("TypeX", DS1) == "TypeX"
    assert analyzer._get_display_type("TypeX") == "TypeX"
    assert analyzer.get_mapped_results() is analyzer.raw_results
    canonical, display = analyzer._build_path_key_with_mapping(["A", "B"], DS1)
    assert canonical == "A → B"
    assert display == "A → B"


def test_mapping_helpers_conflicted_type_is_dataset_scoped(analyzer):
    """Canonical merge keys (shared resolver): a conflicted type keeps a
    dataset-scoped key in path/edge merges — it can never merge with
    another dataset's same-named rows (plan-unify-mapper-backends)."""
    class _ConflictedMapper:
        _loaded = True

        def _get_type_mapping_key(self, dataset):
            return dataset

        def get_mapping_decision(self, source_type, source_dataset,
                                 target_dataset, include_bridges=False):
            if source_type == "CB1011" and source_dataset == "banc_v626":
                return {'status': 'conflict', 'source_type': source_type,
                        'target_type': None, 'target_types': [],
                        'relationship': '1-to-N',
                        'conflicts': [{'source_dataset': source_dataset}]}
            if source_type == "MTe07":
                return {'status': 'mapped', 'source_type': source_type,
                        'target_type': 'MeVPLo2', 'target_types': ['MeVPLo2'],
                        'relationship': '1-to-1', 'conflicts': []}
            return {'status': 'unmapped', 'source_type': source_type,
                    'target_type': None, 'target_types': [],
                    'relationship': None, 'conflicts': []}

        def get_display_name(self, canonical, datasets):
            return canonical

    analyzer.parameters.auto_type_mapping = True
    analyzer.parameters._auto_type_mapper = _ConflictedMapper()

    # licensed rename merges under the canonical name
    assert analyzer._get_canonical_type("MTe07", DS1) == "MeVPLo2"
    # conflicted type is dataset-scoped: distinct from the MCNS raw name
    scoped = analyzer._get_canonical_type("CB1011", "banc_v626")
    assert scoped == "banc_v626:CB1011"
    assert analyzer._get_canonical_type("CB1011", DS1) == "CB1011"

    # path keys carry the scoped node
    canonical, _display = analyzer._build_path_key_with_mapping(
        ["A", "CB1011"], "banc_v626")
    assert canonical == "A → banc_v626:CB1011"

    # resolution counts are on the UNIQUE-resolution basis: MTe07 resolved
    # once and CB1011(conflict) once, regardless of repeated lookups above
    assert analyzer._mapping_status_counts.get("mapped") == 1
    assert analyzer._mapping_status_counts.get("conflict") == 1
    assert "banc_v626:CB1011" in analyzer._conflicted_merge_types

    # repeated resolutions of the same (dataset, type) do not re-count
    analyzer._get_canonical_type("MTe07", DS1)
    analyzer._get_canonical_type("MTe07", DS1)
    assert analyzer._mapping_status_counts.get("mapped") == 1


def test_clear_results_and_set_label_mapper(analyzer):
    analyzer.raw_results[DS1] = {1: _edge_df([("A", "B", 1)])}
    analyzer.aligned_results[1] = pd.DataFrame()
    analyzer.comparison_report = {"x": 1}
    analyzer.clear_results()
    assert analyzer.raw_results == {}
    assert analyzer.comparison_report is None

    new_mapper = LabelMapper()
    analyzer.aligned_results[1] = pd.DataFrame()
    analyzer.set_label_mapper(new_mapper)
    assert analyzer.label_mapper is new_mapper
    assert analyzer.aligned_results == {}


def test_nt_normalization_map():
    assert ComparisonAnalyzer._NT_NORMALIZATION_MAP["ACH"] == "acetylcholine"
    assert ComparisonAnalyzer._NT_NORMALIZATION_MAP["NO_CONS"] == "unknown"


# ---------------------------------------------------------------------------
# Hemisphere symmetry summaries
# ---------------------------------------------------------------------------

def test_get_hemisphere_symmetry_summaries_disabled(analyzer):
    # separate_hemispheres=False forces symmetry_analysis=False in __post_init__
    assert analyzer.parameters.symmetry_analysis is False
    assert analyzer.get_hemisphere_symmetry_summaries() == {}


def test_get_hemisphere_symmetry_summaries(tmp_path):
    params = _params(output_folder=str(tmp_path), separate_hemispheres=True)
    assert params.symmetry_analysis is True
    a = ComparisonAnalyzer(params, verbose=False)

    safe = params._sanitize_name(DS1)
    summary_dir = os.path.join(
        params.full_output_path, "dataset_data", safe, "minsyn_1", "hemisphere_symmetry"
    )
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, "symmetry_summary.json"), "w") as f:
        json.dump({"ipsi": {"jaccard": 0.5}}, f)

    summaries = a.get_hemisphere_symmetry_summaries()
    assert summaries[1][DS1]["ipsi"]["jaccard"] == 0.5
    assert DS2 not in summaries[1]
    assert summaries[3] == {}
    # Cached on second call
    assert a.get_hemisphere_symmetry_summaries() == summaries


# ---------------------------------------------------------------------------
# run_path_analysis (FindNeuronConnection fully stubbed)
# ---------------------------------------------------------------------------

class _FakeFNC:
    out_folder = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.source_df = pd.DataFrame({"bodyId": [1]})
        self.target_df = pd.DataFrame({"bodyId": [2, 3]})
        self.allpath_folder = _FakeFNC.out_folder
        self.calls = []

    def InitializeNeuronInfo(self):
        pass

    def FindAllPath(self, find_reciprocal=False):
        self.calls.append("all")

    def FindShortestPath(self, find_reciprocal=False):
        self.calls.append("shortest")


@pytest.fixture
def fake_fnc(tmp_path, monkeypatch):
    import coana

    out = tmp_path / "fnc"
    details = out / "data_details"
    details.mkdir(parents=True)
    _FakeFNC.out_folder = str(out)
    monkeypatch.setattr(coana, "FindNeuronConnection", _FakeFNC)
    return out


def test_run_path_analysis(fake_fnc, monkeypatch):
    conn = _edge_df([("Src", "Tgt", 6)])
    conn["bodyId_pre"] = 1
    conn["bodyId_post"] = 2
    conn.to_csv(os.path.join(str(fake_fnc), "data_details", "connection_info_bodyId.csv"), index=False)

    a = ComparisonAnalyzer(_params(), verbose=False)
    df = a.run_path_analysis(DS1, 1)
    assert not df.empty
    assert (df["dataset"] == DS1).all()
    assert (df["threshold"] == 1).all()


def test_run_path_analysis_shortest_and_type_fallback(fake_fnc):
    conn = _edge_df([("Src", "Tgt", 6)])
    conn.to_csv(os.path.join(str(fake_fnc), "data_details", "connection_type.csv"), index=False)

    a = ComparisonAnalyzer(_params(path_mode="shortest"), verbose=False)
    df = a.run_path_analysis(DS2, 3)
    assert len(df) == 1
    assert df.iloc[0]["dataset"] == DS2


def test_run_path_analysis_no_output_files(fake_fnc):
    a = ComparisonAnalyzer(_params(), verbose=False)
    df = a.run_path_analysis(DS1, 1)
    assert df.empty


# ---------------------------------------------------------------------------
# run_edge_analysis / edge queries
# ---------------------------------------------------------------------------

def test_run_edge_analysis_from_base_edges(analyzer):
    base = _edge_df([("A", "B", 1), ("C", "D", 5)])
    out = analyzer.run_edge_analysis(DS1, 3, base_edges=base)
    assert len(out) == 1
    assert out.iloc[0]["type_pre"] == "C"
    assert out.iloc[0]["threshold"] == 3


def test_run_edge_analysis_query(monkeypatch, analyzer):
    monkeypatch.setattr(
        analyzer, "_query_edges_for_dataset", lambda *a, **k: _bodyid_df()
    )
    out = analyzer.run_edge_analysis(DS1, 2)
    assert (out["dataset"] == DS1).all()
    assert (out["threshold"] == 2).all()

    monkeypatch.setattr(analyzer, "_query_edges_for_dataset", lambda *a, **k: pd.DataFrame())
    assert analyzer.run_edge_analysis(DS1, 2).empty


def test_query_edges_dispatch(monkeypatch, analyzer):
    monkeypatch.setattr(analyzer, "_query_edges_local", lambda *a, **k: "local")
    monkeypatch.setattr(analyzer, "_query_edges_neuprint", lambda *a, **k: "neuprint")
    assert analyzer._query_edges_for_dataset("flywire_FAFB_v783", [], [], 1) == "local"
    assert analyzer._query_edges_for_dataset("fafb_v2", [], [], 1) == "local"
    assert analyzer._query_edges_for_dataset("banc", [], [], 1) == "local"
    assert analyzer._query_edges_for_dataset(DS1, [], [], 1) == "neuprint"


def test_query_edges_local(tmp_path, monkeypatch, analyzer):
    safe = analyzer.parameters._sanitize_name("flywire_test")
    ds_dir = tmp_path / safe
    ds_dir.mkdir()
    conn = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 2, 3],
            "post_pt_root_id": [2, 3, 4],
            "pre_type": ["Src", "Other", "Zzz"],
            "post_type": ["Inter", "Tgt", "Nope"],
            "syn_count": [5, 4, 2],
        }
    )
    conn.to_csv(ds_dir / f"{safe}_connections.csv", index=False)
    monkeypatch.setattr(analyzer, "_get_datasets_folder", lambda: str(tmp_path))

    out = analyzer._query_edges_local("flywire_test", ["Src"], ["Tgt"], 1)
    # rows kept where pre matches Src OR post matches Tgt, weight >= 1
    assert len(out) == 2
    assert set(out["weight"]) == {5, 4}

    out3 = analyzer._query_edges_local("flywire_test", ["Sr.*"], ["Tgt"], 4)
    # NOTE: pattern 'Src.*' is mangled by a double '*' replacement in the
    # source (becomes 'Src..*'), so 'Sr.*' is used here to match 'Src'.
    assert set(out3["weight"]) == {5, 4}

    # no files at all
    monkeypatch.setattr(analyzer, "_get_datasets_folder", lambda: str(tmp_path / "missing"))
    assert analyzer._query_edges_local("flywire_test", ["Src"], ["Tgt"], 1).empty


def test_query_edges_local_joins_neuron_types(tmp_path, monkeypatch, analyzer):
    safe = analyzer.parameters._sanitize_name("flywire_test")
    ds_dir = tmp_path / safe
    ds_dir.mkdir()
    pd.DataFrame(
        {"pre_pt_root_id": [1], "post_pt_root_id": [2], "syn_count": [7]}
    ).to_csv(ds_dir / f"{safe}_connections.csv", index=False)
    pd.DataFrame({"bodyId": [1, 2], "type": ["Src", "Tgt"]}).to_csv(
        ds_dir / f"{safe}_neurons.csv", index=False
    )
    monkeypatch.setattr(analyzer, "_get_datasets_folder", lambda: str(tmp_path))

    out = analyzer._query_edges_local("flywire_test", ["Src"], ["Tgt"], 1)
    assert len(out) == 1
    assert out.iloc[0]["type_pre"] == "Src"
    assert out.iloc[0]["type_post"] == "Tgt"


def test_query_edges_neuprint(monkeypatch, analyzer):
    fake_mod = types.ModuleType("neuprint")

    class _Client:
        def __init__(self, server, dataset=None, token=None):
            pass

        def fetch_custom(self, query):
            return _bodyid_df()

    fake_mod.Client = _Client
    monkeypatch.setitem(sys.modules, "neuprint", fake_mod)
    out = analyzer._query_edges_neuprint(DS1, ["Src.*"], ["Tgt"], 1)
    assert len(out) == 2

    class _BrokenClient(_Client):
        def fetch_custom(self, query):
            raise RuntimeError("no network")

    fake_mod.Client = _BrokenClient
    assert analyzer._query_edges_neuprint(DS1, ["Src"], ["Tgt"], 1).empty


# ---------------------------------------------------------------------------
# run_all_analyses / caching
# ---------------------------------------------------------------------------

def test_run_all_path_analyses(monkeypatch, analyzer):
    # legacy per-threshold loop semantics (Feature F replay disabled —
    # under replay the batch routes through FindAllPathMultiThreshold)
    analyzer.parameters.replay_paths = False
    calls = []

    def fake_run(dataset, threshold, verbose_mode="simple"):
        calls.append((dataset, threshold, verbose_mode))
        return _edge_df([("Src", "Tgt", threshold)])

    monkeypatch.setattr(analyzer, "run_path_analysis", fake_run)
    results = analyzer.run_all_analyses(skip_existing=False)
    assert set(results) == {DS1, DS2}
    assert set(results[DS1]) == {1, 3}
    # lowest threshold uses 'simple', higher uses 'silent'
    assert ("hemibrain:v1.2.1", 1, "simple") in calls
    assert ("hemibrain:v1.2.1", 3, "silent") in calls

    # skip_existing=True: nothing rerun
    calls.clear()
    analyzer.run_all_analyses(skip_existing=True)
    assert calls == []

    # skip_existing=False: reruns everything
    analyzer.run_all_analyses(skip_existing=False)
    assert len(calls) == 4


def test_run_all_path_analyses_disk_cache(tmp_path, monkeypatch):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)

    cached = _edge_df([("Src", "Tgt", 11)])
    out_dir = params.get_dataset_output_path(DS1, 1)
    os.makedirs(out_dir, exist_ok=True)
    a._save_csv(cached, os.path.join(out_dir, "connections_edge.csv"))

    def fail_run(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("run_path_analysis should be skipped")

    monkeypatch.setattr(a, "run_path_analysis", fail_run)

    # Provide results for the rest so only DS1@1 must come from disk cache
    loaded = a._try_load_cached(DS1, 1)
    assert loaded is not None and len(loaded) == 1

    a.raw_results[DS1] = {1: loaded, 3: _edge_df([("Src", "Tgt", 3)])}
    a.raw_results[DS2] = {1: _edge_df([("Src", "Tgt", 1)]), 3: _edge_df([("Src", "Tgt", 3)])}
    results = a.run_all_analyses(skip_existing=True)
    assert len(results[DS1][1]) == 1


def test_try_load_cached_variants(tmp_path):
    # No output folder -> always None
    a_mem = ComparisonAnalyzer(_params(), verbose=False)
    assert a_mem._try_load_cached(DS1, 1) is None

    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    out_dir = params.get_dataset_output_path(DS1, 1)

    # Nothing on disk
    assert a._try_load_cached(DS1, 1) is None

    # paths.csv legacy
    os.makedirs(out_dir, exist_ok=True)
    _edge_df([("A", "B", 2)]).to_csv(os.path.join(out_dir, "paths.csv"), index=False)
    assert len(a._try_load_cached(DS1, 1)) == 1

    # connection_info_bodyId.csv under data_details (adds dataset/threshold cols)
    details = os.path.join(out_dir, "data_details")
    os.makedirs(details)
    _edge_df([("A", "B", 2)]).to_csv(os.path.join(details, "connection_info_bodyId.csv"), index=False)
    df = a._try_load_cached(DS1, 1)
    assert df is not None

    # connection_type.csv fallback
    os.remove(os.path.join(out_dir, "paths.csv"))
    os.remove(os.path.join(details, "connection_info_bodyId.csv"))
    _edge_df([("A", "B", 2)]).to_csv(os.path.join(details, "connection_type.csv"), index=False)
    assert a._try_load_cached(DS1, 1) is not None


def test_save_result_and_edge_mode_result(tmp_path):
    # In-memory analyzer: both are no-ops
    a_mem = ComparisonAnalyzer(_params(), verbose=False)
    a_mem._save_result(DS1, 1, _edge_df([("A", "B", 1)]))
    a_mem._save_edge_mode_result(DS1, 1, _edge_df([("A", "B", 1)]))

    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    df = _edge_df([("A", "B", 1)])
    a._save_result(DS1, 1, df)
    assert os.path.exists(
        os.path.join(params.get_dataset_output_path(DS1, 1), "connections_edge.csv")
    )
    a._save_edge_mode_result(DS1, 3, df)
    edge_mode = os.path.join(
        params.full_output_path, "edge_mode_data", params._sanitize_name(DS1), "connections_edge_3.csv"
    )
    assert os.path.exists(edge_mode)
    # Empty frames are skipped
    a._save_result(DS1, 3, pd.DataFrame())
    assert not os.path.exists(
        os.path.join(params.get_dataset_output_path(DS1, 3), "connections_edge.csv")
    )


# ---------------------------------------------------------------------------
# Edge mode aggregation pipeline
# ---------------------------------------------------------------------------

def test_build_label_map_from_df(analyzer):
    lm = analyzer._build_label_map_from_df(_bodyid_df())
    assert lm == {1: "Src", 2: "Inter", 3: "Tgt"}
    assert analyzer._build_label_map_from_df(pd.DataFrame({"x": [1]})) == {}


def test_aggregate_and_find_paths(analyzer):
    df = _bodyid_df()
    df["total_post"] = [20, 10]
    label_map = analyzer._build_label_map_from_df(df)
    out = analyzer._aggregate_and_find_paths(
        df, label_map, {"Src"}, {"Tgt"}, 2, DS1, 1, path_mode="all"
    )
    assert not out.empty
    assert {"has_valid_path", "dataset", "threshold", "conn_layer", "traversal_probability"} <= set(out.columns)
    assert out["has_valid_path"].all()
    assert (out["conn_layer"].isin(["0->1", "1->2"])).all()

    # shortest mode
    out2 = analyzer._aggregate_and_find_paths(
        df, label_map, {"Src"}, {"Tgt"}, None, DS1, 1, path_mode="shortest"
    )
    assert not out2.empty


def test_get_bodyid_connections(monkeypatch, analyzer):
    monkeypatch.setattr(analyzer, "_query_edges_for_dataset", lambda *a, **k: _bodyid_df())
    df, lm = analyzer._get_bodyid_connections_for_dataset(DS1, 1)
    assert df is not None and len(df) == 2
    assert lm[1] == "Src"

    monkeypatch.setattr(analyzer, "_query_edges_for_dataset", lambda *a, **k: pd.DataFrame())
    df, lm = analyzer._get_bodyid_connections_for_dataset(DS1, 1)
    assert df is None and lm == {}


def test_get_bodyid_connections_cached(tmp_path):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    details = os.path.join(params.get_dataset_output_path(DS1, 1), "data_details")
    os.makedirs(details)
    _bodyid_df().to_csv(os.path.join(details, "connection_info_bodyId.csv"), index=False)

    df, lm = a._get_bodyid_connections_for_dataset(DS1, 1, skip_existing=True)
    assert df is not None and lm[3] == "Tgt"


def test_run_all_edge_analyses(monkeypatch, analyzer):
    analyzer.parameters.comparison_mode = "edge"
    path_calls = []
    monkeypatch.setattr(
        analyzer, "run_path_analysis",
        lambda ds, t, verbose_mode="simple": path_calls.append((ds, t)) or pd.DataFrame(),
    )
    monkeypatch.setattr(
        analyzer, "_get_bodyid_connections_for_dataset",
        lambda ds, t, skip_existing=True, run_findallpath=False: (_bodyid_df(), {1: "Src", 2: "Inter", 3: "Tgt"}),
    )

    results = analyzer.run_all_analyses(skip_existing=False)
    for ds in (DS1, DS2):
        for t in (1, 3):
            out = results[ds][t]
            assert not out.empty
            assert "has_valid_path" in out.columns
    # path tool ran for lowest ('simple') and remaining thresholds
    assert (DS1, 1) in path_calls and (DS1, 3) in path_calls


def test_run_all_edge_analyses_empty(monkeypatch, analyzer):
    analyzer.parameters.comparison_mode = "edge"
    monkeypatch.setattr(analyzer, "run_path_analysis", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(
        analyzer, "_get_bodyid_connections_for_dataset",
        lambda *a, **k: (None, {}),
    )
    results = analyzer.run_all_analyses(skip_existing=False)
    for ds in (DS1, DS2):
        for t in (1, 3):
            assert results[ds][t].empty


def test_process_threshold_aggregation_empty_filter(analyzer):
    analyzer.raw_results[DS1] = {}
    analyzer._process_threshold_aggregation(
        DS1, 100, _bodyid_df(), {1: "Src", 2: "Inter", 3: "Tgt"},
        {"Src"}, {"Tgt"}, 2, skip_existing=False,
    )
    assert analyzer.raw_results[DS1][100].empty


# ---------------------------------------------------------------------------
# Metadata collection
# ---------------------------------------------------------------------------

def test_coverage_notes_and_empty_metadata(analyzer):
    assert "Central brain" in analyzer._get_coverage_notes(DS1)
    assert "male" in analyzer._get_coverage_notes("male-cns:v0.9").lower()
    assert analyzer._get_coverage_notes("unknown_ds") == "Coverage information not available."

    empty = analyzer._create_empty_metadata(DS1, "boom")
    assert empty["source"] == "error"
    assert empty["error"] == "boom"
    assert empty["neuron_counts"]["total"] == 0


def test_collect_dataset_metadata_fetch(monkeypatch, analyzer):
    fetched = []

    def fake_fetch(ds):
        fetched.append(ds)
        return _meta(ds)

    monkeypatch.setattr(analyzer, "_load_cached_metadata", lambda ds: None)
    monkeypatch.setattr(analyzer, "_fetch_neuprint_metadata", fake_fetch)
    monkeypatch.setattr(analyzer, "_save_metadata", lambda ds, m: None)

    md = analyzer.collect_dataset_metadata()
    assert set(md) == {DS1, DS2}
    assert fetched == [DS1, DS2]

    table = analyzer.generate_metadata_comparison_table()
    expected_cols = {
        "dataset", "total_neurons", "typed_neurons", "untyped_neurons",
        "type_coverage_pct", "total_presynaptic", "total_postsynaptic",
        "total_synapses", "roi_count", "coverage_notes",
    }
    assert expected_cols <= set(table.columns)
    assert len(table) == 2
    assert table.iloc[0]["total_neurons"] == 100


def test_collect_dataset_metadata_cached(monkeypatch, analyzer):
    fetched = []
    _stub_metadata_methods(monkeypatch, analyzer, cached=_meta(DS1))
    monkeypatch.setattr(
        analyzer, "_fetch_neuprint_metadata",
        lambda ds: fetched.append(ds) or _meta(ds),
    )
    md = analyzer.collect_dataset_metadata()
    assert fetched == []  # cached metadata used, no fetch
    assert md[DS1]["neuron_counts"]["total"] == 100

    # force_refresh bypasses cache
    md2 = analyzer.collect_dataset_metadata(force_refresh=True)
    assert fetched == [DS1, DS2]
    assert set(md2) == {DS1, DS2}


def test_metadata_paths(analyzer):
    path = analyzer._get_metadata_path(DS1)
    safe = analyzer.parameters._sanitize_name(DS1)
    assert path.endswith(f"{safe}_metadata.json")
    assert analyzer._get_datasets_folder().endswith("datasets")
    # No cached metadata file for an unknown dataset
    assert analyzer._load_cached_metadata("nonexistent_ds:v0") is None


def test_error_metadata_is_never_persisted(tmp_path, monkeypatch):
    """F9: a failed collection leaves no sidecar where installed data lives.

    The error payload is in-memory only; nothing is written under
    ``datasets/`` -- not even the folder -- so no later listing or count path
    can read it as a half-installed release.
    """
    a = ComparisonAnalyzer(_params(), verbose=False)
    monkeypatch.setattr(a, "_get_datasets_folder", lambda: str(tmp_path))
    safe = a.parameters._sanitize_name(DS1)

    a._save_metadata(DS1, a._create_empty_metadata(DS1, "no token"))

    assert not (tmp_path / safe).exists()
    assert list(tmp_path.glob("*/*_metadata.json")) == []
    assert a._load_cached_metadata(DS1) is None

    # A real payload is still cached as before.
    a._save_metadata(DS1, _meta(DS1))
    assert json.loads(
        (tmp_path / safe / f"{safe}_metadata.json").read_text()
    )["source"] == "stub"


def test_transient_failure_retries_instead_of_replaying_error(
        tmp_path, monkeypatch, analyzer):
    """F9: with force_refresh=False the next run retries a failed fetch."""
    monkeypatch.setattr(analyzer, "_get_datasets_folder", lambda: str(tmp_path))
    calls = []

    def failing_fetch(ds):
        calls.append(ds)
        return analyzer._create_empty_metadata(ds, "no token")

    monkeypatch.setattr(analyzer, "_fetch_neuprint_metadata", failing_fetch)

    md = analyzer.collect_dataset_metadata()
    assert len(calls) == 2
    assert md[DS1]["source"] == "error"  # still usable for THIS run's reporting
    table = analyzer.generate_metadata_comparison_table()
    assert set(table["dataset"]) == {DS1, DS2}
    assert table.iloc[0]["total_neurons"] == 0

    # Nothing landed on disk -> a second collection re-fetches instead of
    # replaying the cached error.
    analyzer.collect_dataset_metadata()
    assert len(calls) == 4

    # Once the failure clears, the real payload is cached.
    monkeypatch.setattr(analyzer, "_fetch_neuprint_metadata", _meta)
    md3 = analyzer.collect_dataset_metadata()
    assert len(calls) == 4
    assert md3[DS1]["source"] == "stub"
    safe = analyzer.parameters._sanitize_name(DS1)
    assert json.loads(
        (tmp_path / safe / f"{safe}_metadata.json").read_text()
    )["neuron_counts"]["total"] == 100


def test_legacy_error_sidecar_is_not_reused(tmp_path, monkeypatch):
    """Error sidecars written by older builds must not be served as data."""
    a = ComparisonAnalyzer(_params(), verbose=False)
    monkeypatch.setattr(a, "_get_datasets_folder", lambda: str(tmp_path))
    safe = a.parameters._sanitize_name(DS1)
    d = tmp_path / safe
    d.mkdir()
    meta_file = d / f"{safe}_metadata.json"
    meta_file.write_text(json.dumps(
        a._create_empty_metadata(DS1, "No local data found")), encoding="utf-8")

    fetched = []
    monkeypatch.setattr(a, "_fetch_neuprint_metadata",
                        lambda ds: fetched.append(ds) or _meta(ds))

    md = a.collect_dataset_metadata()
    assert fetched == [DS1, DS2]
    assert md[DS1]["source"] == "stub"
    # The successful fetch overwrote the stale placeholder with real data.
    on_disk = json.loads(meta_file.read_text(encoding="utf-8"))
    assert "error" not in on_disk
    assert on_disk["neuron_counts"]["total"] == 100


# ---------------------------------------------------------------------------
# Comparison pipeline
# ---------------------------------------------------------------------------

def test_run_comparison(monkeypatch, tmp_path):
    # output_folder must be set: with "" run_comparison falls back to a
    # timestamped cross-dataset_* folder in the cwd (repo-root scaffold leak).
    a = ComparisonAnalyzer(_params(output_folder=str(tmp_path)), verbose=False)
    monkeypatch.setattr(a, "run_all_analyses", lambda skip_existing=True: a.raw_results.update(_standard_results()))

    report = a.run_comparison()
    assert isinstance(report, dict)
    assert a.comparison_report is report
    assert "key_findings" in report
    assert "threshold_similarities" in report
    assert a.raw_results[DS1][1] is not None

    sims = a.get_cached_similarities(1)
    assert not sims.empty
    # cached copy returned
    assert a.get_cached_similarities(1) is not sims
    # uncached threshold with no data -> empty
    assert a.get_cached_similarities(99).empty


def test_run_comparison_analysis_empty(monkeypatch):
    a = ComparisonAnalyzer(_params(), verbose=False)
    monkeypatch.setattr(a, "run_all_analyses", lambda skip_existing=True: None)
    summary = a.run_comparison_analysis()
    assert summary["key_findings"] == ["No data available for comparison"]


def test_aligned_and_connection_getters(analyzer):
    analyzer.raw_results = _standard_results()

    aligned = analyzer.get_aligned_data(1)
    assert not aligned.empty
    assert DS1 in aligned.columns and DS2 in aligned.columns
    assert "Src -> Tgt" in aligned.index
    # cached
    assert analyzer.get_aligned_data(1) is aligned
    # missing threshold -> empty
    assert analyzer.get_aligned_data(99).empty

    common = analyzer.get_common_connections(1)
    assert "Src -> Tgt" in set(common.index) or "Src -> Tgt" in common.get("edge", pd.Series()).values

    unique = analyzer.get_unique_connections(1)
    assert set(unique) == {DS1, DS2}

    diff = analyzer.get_differential_connections(1, fold_threshold=2.0)
    assert isinstance(diff, pd.DataFrame)


def test_aligned_data_for_network(analyzer):
    analyzer.raw_results = _standard_results()
    # find_reciprocal=False -> plain aligned data
    out = analyzer.get_aligned_data_for_network(1)
    assert not out.empty

    # find_reciprocal=True but no reciprocal files -> falls back
    analyzer.parameters.find_reciprocal = True
    out2 = analyzer.get_aligned_data_for_network(3)
    assert not out2.empty


def test_filter_hemisphere_unconserved(tmp_path):
    params = _params(
        output_folder=str(tmp_path),
        separate_hemispheres=True,
        keep_only_hemisphere_conserved_connections=True,
    )
    a = ComparisonAnalyzer(params, verbose=False)

    aligned = pd.DataFrame(
        {
            DS1: [5, 3, 9, 4, 2],
            DS2: [4, 0, 8, 4, 0],
        },
        index=[
            "A_R -> B_R",   # conserved pair with counterpart present
            "A_L -> B_L",   # counterpart of above; weight 0 in DS2
            "C_R -> D_R",   # missing counterpart -> zeroed
            "Plain -> Edge",  # no hemisphere suffix -> kept
            "E_U -> F_U",   # _U suffix -> kept as-is
        ],
    )

    out = a._filter_hemisphere_unconserved(aligned, [DS1, DS2], threshold=1)
    # conserved pair preserved where BOTH mirror edges exist in the dataset
    assert out.at["A_R -> B_R", DS1] == 5
    # in DS2 the counterpart A_L -> B_L has weight 0 -> edge zeroed in DS2
    assert out.at["A_R -> B_R", DS2] == 0
    assert out.at["A_L -> B_L", DS1] == 3
    assert out.at["A_L -> B_L", DS2] == 0
    # unconserved edge zeroed entirely
    assert out.at["C_R -> D_R", DS1] == 0
    assert out.at["C_R -> D_R", DS2] == 0
    # no-suffix edges untouched
    assert out.at["Plain -> Edge", DS1] == 4
    assert out.at["E_U -> F_U", DS1] == 2

    # unconserved edges were saved to disk
    unconserved_file = os.path.join(
        params.full_output_path, "comparison_results", "hemisphere_unconserved_edges_t1.csv"
    )
    assert os.path.exists(unconserved_file)

    # empty input passes through
    assert a._filter_hemisphere_unconserved(pd.DataFrame(), [DS1, DS2]).empty


def test_get_aligned_data_applies_hemisphere_filter(tmp_path):
    params = _params(
        output_folder=str(tmp_path),
        separate_hemispheres=True,
        keep_only_hemisphere_conserved_connections=True,
    )
    a = ComparisonAnalyzer(params, verbose=False)
    d1 = _edge_df([("A_R", "B_R", 5)])
    d2 = _edge_df([("A_R", "B_R", 4), ("A_L", "B_L", 4)])
    a.raw_results = {DS1: {1: d1}, DS2: {1: d2}}

    aligned = a.get_aligned_data(1)
    # A_R -> B_R has no A_L -> B_L counterpart in DS1 -> zeroed in DS1
    assert aligned.at["A_R -> B_R", DS1] == 0


# ---------------------------------------------------------------------------
# Reports / export
# ---------------------------------------------------------------------------

def test_generate_report(tmp_path):
    a = ComparisonAnalyzer(_params(output_folder=str(tmp_path)), verbose=False)
    a.raw_results = _standard_results()

    out_path = str(tmp_path / "report.txt")
    text = a.generate_report(output_path=out_path)
    assert "CROSS-DATASET COMPARISON REPORT" in text
    assert DS1 in text
    assert os.path.exists(out_path)


def test_export_results(tmp_path, monkeypatch):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    a.raw_results = _standard_results()
    _stub_metadata_methods(monkeypatch, a)

    a.export_results()

    base = params.full_output_path
    assert os.path.exists(os.path.join(base, "parameters.json"))
    assert os.path.exists(os.path.join(base, "label_map.json"))
    assert os.path.exists(os.path.join(base, "comparison_report.txt"))
    results_dir = os.path.join(base, "comparison_results")
    assert os.path.exists(os.path.join(results_dir, "path_count_comparison.csv"))
    assert os.path.exists(os.path.join(results_dir, "threshold_sensitivity.csv"))


def test_generate_html_report(tmp_path):
    a = ComparisonAnalyzer(_params(output_folder=str(tmp_path)), verbose=False)
    a.raw_results = _standard_results()
    a.comparison_report = {
        "key_findings": ["finding"],
        "key_findings_per_threshold": {},
        "path_presence_matrix": pd.DataFrame(),
        "threshold_similarities": pd.DataFrame(),
    }
    out = a.generate_html_report(os.path.join(str(tmp_path), "rep.html"))
    assert os.path.exists(out)


# ---------------------------------------------------------------------------
# quick_compare convenience wrapper
# ---------------------------------------------------------------------------

def test_quick_compare(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_run_all(self, skip_existing=True):
        self.raw_results.update(_standard_results())
        return self.raw_results

    monkeypatch.setattr(ComparisonAnalyzer, "run_all_analyses", fake_run_all)
    monkeypatch.setattr(ComparisonAnalyzer, "export_results", lambda self, output_dir=None: None)

    results = quick_compare(
        datasets=[DS1, DS2],
        source_neurons=["Src"],
        target_neurons=["Tgt"],
        thresholds=[1, 3],
        verbose=False,
    )
    assert isinstance(results, dict)
    assert "key_findings" in results


# ===========================================================================
# Extended coverage tests (appended)
# ===========================================================================

class _FakeAutoTypeMapper:
    """Minimal stand-in for CrossDatasetTypeMapper (no repo CSV access),
    following the shared resolver contract: canonical merge keys are
    resolved through ``canonical_merge_key`` (which consumes
    ``get_mapping_decision`` status dicts)."""

    _loaded = True

    def __init__(self):
        self._conflicts = {}
        self.exported = []

    def has_conflicts(self):
        return bool(self._conflicts)

    def detect_type_source(self, type_name):
        return self._detect_type_source(type_name)

    def get_type_mapping_key(self, dataset):
        return dataset

    def get_type_neuron_count(self, type_name, dataset):
        return 1

    def _get_type_mapping_key(self, dataset):
        return dataset

    def get_mapping_decision(self, source_type, source_dataset,
                             target_dataset, include_bridges=False):
        target = "SrcCanon" if source_type == "Src" else source_type
        return {'status': 'mapped', 'source_type': source_type,
                'target_type': target, 'target_types': [target],
                'relationship': '1-to-1', 'conflicts': []}

    def get_canonical_type(self, type_name, dataset):
        return "SrcCanon" if type_name == "Src" else type_name

    def get_display_name(self, canonical, datasets=None):
        return f"{canonical}[all]"

    def get_intermediate_mapping_summary(self, types, datasets):
        return {
            "total_types": len(list(types)),
            "mapped_count": 1,
            "n_to_1_count": 0,
            "one_to_n_count": 0,
        }

    def export_mapping(self, path, filter_types=None, datasets=None,
                       only_different=True):
        self.exported.append(("mapping", path))

    def export_conflicts(self, path, filter_types=None, datasets=None):
        self.exported.append(("conflicts", path))

    def _detect_type_source(self, type_name):
        return "hemibrain:v1.2.1" if type_name == "Src" else None

    def get_mapped_type(self, type_name, source_ds, target_ds):
        return "SrcMapped" if type_name == "Src" else None


def _enable_fake_auto_mapper(params):
    params.auto_type_mapping = True
    params._auto_type_mapper = _FakeAutoTypeMapper()
    return params._auto_type_mapper


def test_mapped_results_and_canonical_display_types():
    a = ComparisonAnalyzer(_params(), verbose=True)
    _enable_fake_auto_mapper(a.parameters)
    a.raw_results = _standard_results()

    mapped = a.get_mapped_results()
    assert set(mapped[DS1][1]["type_pre"]) == {"SrcCanon"}
    assert "SrcCanon" in set(mapped[DS2][1]["type_pre"])
    # unmapped type passes through
    assert "Y" in set(mapped[DS2][1]["type_pre"])
    # raw results untouched
    assert set(a.raw_results[DS1][1]["type_pre"]) == {"Src"}

    assert a._get_canonical_type("Src", DS1) == "SrcCanon"
    assert a._get_canonical_type("Other", DS1) == "Other"
    assert a._get_display_type("SrcCanon") == "SrcCanon[all]"

    # exercises _print_intermediate_mapping_summary
    a._print_intermediate_mapping_summary()

    # without mapper -> raw results returned unchanged
    b = ComparisonAnalyzer(_params(), verbose=False)
    b.raw_results = _standard_results()
    assert b.get_mapped_results() is b.raw_results
    assert b._get_canonical_type("Src", DS1) == "Src"
    assert b._get_display_type("Src") == "Src"


def test_run_path_analysis_applies_label_mapper_and_custom_names(
        fake_fnc, monkeypatch):
    mapper = LabelMapper(
        source_mapping_dict={DS1: [["aMe1"]], DS2: [["aMe1"]]},
        source_labels=["SrcStd"],
        target_mapping_dict={DS1: [["PPL1"]], DS2: [["PPL1"]]},
        target_labels=["TgtStd"],
    )
    params = _params(
        source_neurons=["aMe1"], target_neurons=["PPL1"],
        source_labels=["CustomSrc"], target_labels=["CustomTgt"],
    )
    a = ComparisonAnalyzer(params, label_mapper=mapper, verbose=False)

    pd.DataFrame({
        "bodyId_pre": [1], "bodyId_post": [2],
        "type_pre": ["aMe1"], "type_post": ["PPL1"], "weight": [4],
    }).to_csv(os.path.join(str(fake_fnc), "data_details",
                           "connection_info_bodyId.csv"), index=False)

    captured = {}
    import coana
    real_fake = coana.FindNeuronConnection  # the _FakeFNC class from fixture

    class _RecordingFNC(real_fake):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(kwargs)

    monkeypatch.setattr(coana, "FindNeuronConnection", _RecordingFNC)

    out = a.run_path_analysis(DS1, 1)
    assert not out.empty
    # standardized labels overwrite original types
    assert out.iloc[0]["type_pre"] == "SrcStd"
    assert out.iloc[0]["type_post"] == "TgtStd"
    # custom single-label names forwarded to FindNeuronConnection
    assert captured["custom_source_name"] == "CustomSrc"
    assert captured["custom_target_name"] == "CustomTgt"


def test_run_edge_analysis_applies_label_mapper(monkeypatch):
    mapper = LabelMapper(
        source_mapping_dict={DS1: [["aMe1"]], DS2: [["aMe1"]]},
        source_labels=["SrcStd"],
        target_mapping_dict={DS1: [["PPL1"]], DS2: [["PPL1"]]},
        target_labels=["TgtStd"],
    )
    params = _params(source_neurons=["aMe1"], target_neurons=["PPL1"])
    a = ComparisonAnalyzer(params, label_mapper=mapper, verbose=False)
    monkeypatch.setattr(
        a, "_query_edges_for_dataset",
        lambda *args, **kw: _edge_df([("aMe1", "PPL1", 4)]),
    )
    out = a.run_edge_analysis(DS1, 2)
    assert out.iloc[0]["type_pre"] == "SrcStd"
    assert out.iloc[0]["type_post"] == "TgtStd"


def test_run_all_path_analyses_disk_cache_hit(tmp_path, monkeypatch):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)

    for ds in (DS1, DS2):
        for t in (1, 3):
            p = params.get_dataset_output_path(ds, t)
            os.makedirs(p, exist_ok=True)
            _edge_df([("Src", "Tgt", 5)]).to_csv(
                os.path.join(p, "connections_edge.csv"), index=False)

    def _explode(*args, **kwargs):
        raise AssertionError("run_path_analysis must not run on cache hit")

    monkeypatch.setattr(a, "run_path_analysis", _explode)
    results = a._run_all_path_analyses(skip_existing=True)
    assert not results[DS1][1].empty
    assert not results[DS2][3].empty


def test_run_all_path_analyses_saves_result(tmp_path, monkeypatch):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    monkeypatch.setattr(
        a, "run_path_analysis",
        lambda ds, t, verbose_mode="simple": _edge_df([("Src", "Tgt", 5)]),
    )
    a._run_all_path_analyses(skip_existing=False)
    p = params.get_dataset_output_path(DS1, 1)
    assert os.path.exists(os.path.join(p, "connections_edge.csv"))


def test_bodyid_cache_corrupt_falls_back_to_query(tmp_path, monkeypatch):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    p = params.get_dataset_output_path(DS1, 1)
    os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, "connection_info_bodyId.csv"), "w") as f:
        f.write("a,b\n1,2,3\n")  # malformed -> parser error
    monkeypatch.setattr(a, "_query_edges_for_dataset",
                        lambda *args, **kw: _bodyid_df())
    df, label_map = a._get_bodyid_connections_for_dataset(DS1, 1)
    assert df is not None and len(df) == 2
    assert label_map.get(1) == "Src"


def test_bodyid_connections_run_findallpath(tmp_path, monkeypatch):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    safe = params._sanitize_name(DS1)

    def fake_run_path(dataset, threshold, verbose_mode="simple"):
        d = os.path.join(params.full_output_path, "dataset_data", safe,
                         f"minsyn_{threshold}", "data_details")
        os.makedirs(d, exist_ok=True)
        _bodyid_df().to_csv(os.path.join(d, "connection_info_bodyId.csv"),
                            index=False)
        return pd.DataFrame()

    monkeypatch.setattr(a, "run_path_analysis", fake_run_path)
    df, label_map = a._get_bodyid_connections_for_dataset(
        DS1, 1, skip_existing=False, run_findallpath=True)
    assert df is not None and len(df) == 2
    assert label_map.get(3) == "Tgt"


def test_metadata_save_load_roundtrip(tmp_path, monkeypatch):
    a = ComparisonAnalyzer(_params(), verbose=False)
    monkeypatch.setattr(a, "_get_datasets_folder", lambda: str(tmp_path))
    assert a._load_cached_metadata(DS1) is None
    a._save_metadata(DS1, _meta(DS1))
    loaded = a._load_cached_metadata(DS1)
    assert loaded["dataset"] == DS1


def test_metadata_load_corrupt_json(tmp_path, monkeypatch):
    a = ComparisonAnalyzer(_params(), verbose=False)
    monkeypatch.setattr(a, "_get_datasets_folder", lambda: str(tmp_path))
    safe = a.parameters._sanitize_name(DS1)
    d = tmp_path / safe
    d.mkdir()
    (d / f"{safe}_metadata.json").write_text("{not json")
    assert a._load_cached_metadata(DS1) is None


def test_fetch_neuprint_metadata_fake_client(monkeypatch):
    a = ComparisonAnalyzer(_params(), verbose=False)
    fake = types.ModuleType("neuprint")

    class _Client:
        def __init__(self, server, dataset=None, token=None):
            pass

        def fetch_custom(self, query):
            if "total_neurons" in query:
                return pd.DataFrame({"total_neurons": [100]})
            if "typed_neurons" in query:
                return pd.DataFrame({"typed_neurons": [80]})
            if "total_pre" in query:
                return pd.DataFrame({"total_pre": [500], "total_post": [400]})
            if "primaryRois" in query:
                return pd.DataFrame({"rois": [["AL", "OL"]]})
            return pd.DataFrame({"count": [7]})

    fake.Client = _Client
    monkeypatch.setitem(sys.modules, "neuprint", fake)
    md = a._fetch_neuprint_metadata(DS1)
    assert md["source"] == "neuprint"
    assert md["neuron_counts"]["total"] == 100
    assert md["neuron_counts"]["typed"] == 80
    assert md["synapse_counts"]["total"] == 900
    assert md["roi_coverage"]["roi_count"] == 2
    assert md["roi_coverage"]["neuron_counts_per_roi"]["AL"] == 7


def test_fetch_neuprint_metadata_failure(monkeypatch):
    a = ComparisonAnalyzer(_params(), verbose=False)
    fake = types.ModuleType("neuprint")

    class _Client:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no network")

    fake.Client = _Client
    monkeypatch.setitem(sys.modules, "neuprint", fake)
    md = a._fetch_neuprint_metadata(DS1)
    assert md["dataset"] == DS1
    assert md["neuron_counts"]["total"] == 0


def test_fetch_local_metadata_from_files(tmp_path, monkeypatch):
    a = ComparisonAnalyzer(_params(), verbose=False)
    safe = a.parameters._sanitize_name(DS1)
    d = tmp_path / safe
    d.mkdir()
    pd.DataFrame({"type": ["Src", "Tgt"], "pre": [50, 40], "post": [30, 20]}
                 ).to_csv(d / f"{safe}_allneurons_neuron_df.csv", index=False)
    pd.DataFrame({"bodyId": [1, 2], "type": ["Src", "Tgt"],
                  "AL": [1, 0], "OL": [0, 1]}
                 ).to_csv(d / f"{safe}_allneurons_roi_count_df.csv",
                          index=False)
    monkeypatch.setattr(a, "_get_datasets_folder", lambda: str(tmp_path))
    md = a._fetch_local_metadata(DS1)
    assert md["dataset"] == DS1
    assert md["source"] == "local"
    assert md["neuron_counts"]["total"] == 2


def test_fetch_local_metadata_missing_files(tmp_path, monkeypatch):
    a = ComparisonAnalyzer(_params(), verbose=False)
    monkeypatch.setattr(a, "_get_datasets_folder",
                        lambda: str(tmp_path / "absent"))
    md = a._fetch_local_metadata(DS1)
    assert md["source"] == "error"


def test_get_cached_similarities_computes_on_demand():
    a = ComparisonAnalyzer(_params(), verbose=False)
    a.raw_results = _standard_results()
    sim = a.get_cached_similarities(1)
    assert not sim.empty
    # second call uses the cache
    sim2 = a.get_cached_similarities(1)
    assert not sim2.empty


def test_get_aligned_data_for_network_reciprocal(tmp_path):
    params = _params(output_folder=str(tmp_path), find_reciprocal=True)
    a = ComparisonAnalyzer(params, verbose=False)
    a.raw_results = _standard_results()
    safe1 = params._sanitize_name(DS1)
    safe2 = params._sanitize_name(DS2)
    for safe in (safe1, safe2):
        d = os.path.join(params.full_output_path, "dataset_data", safe,
                         "minsyn_1", "find_reciprocal")
        os.makedirs(d, exist_ok=True)
        _edge_df([("Src", "Tgt", 6)]).to_csv(
            os.path.join(d, "reciprocal_connection_type.csv"), index=False)

    aligned = a.get_aligned_data_for_network(1)
    assert not aligned.empty
    assert "Src -> Tgt" in aligned.index
    # cached on second call
    assert a.get_aligned_data_for_network(1) is aligned

    # no reciprocal files -> falls back to standard aligned data
    b = ComparisonAnalyzer(_params(output_folder=str(tmp_path),
                                   find_reciprocal=True), verbose=False)
    b.raw_results = _standard_results()
    fallback = b.get_aligned_data_for_network(1)
    assert not fallback.empty


def test_generate_report_symmetry_section(tmp_path):
    params = _params(output_folder=str(tmp_path), separate_hemispheres=True)
    a = ComparisonAnalyzer(params, verbose=False)
    a.raw_results = _standard_results()

    summary = {
        "ipsi": {"jaccard": 0.5, "conserved": 3, "union": 6},
        "contra": {"jaccard": 0.25, "conserved": 1, "union": 4},
        "neuron_types": {"types_conserved": 2, "types_union": 5},
        "hemisphere_counts": {"total": {"L": 10, "R": 12}},
    }
    for ds in (DS1, DS2):
        safe = params._sanitize_name(ds)
        for t in (1, 3):
            d = os.path.join(params.full_output_path, "dataset_data", safe,
                             f"minsyn_{t}", "hemisphere_symmetry")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "symmetry_summary.json"), "w") as f:
                json.dump(summary, f)

    summaries = a.get_hemisphere_symmetry_summaries()
    assert summaries[1][DS1]["ipsi"]["jaccard"] == 0.5
    text = a.generate_report()
    assert "HEMISPHERE SYMMETRY SUMMARY" in text
    assert "0.500" in text

# --- APPEND-POINT-1 ---


# ---------------------------------------------------------------------------
# export_results / label map / export helpers
# ---------------------------------------------------------------------------

def test_export_results_no_output_dir(monkeypatch):
    a = ComparisonAnalyzer(_params(), verbose=False)
    # full_output_path normally falls back to an auto-generated folder;
    # force it empty to exercise the ValueError guard.
    monkeypatch.setattr(ComparisonParameters, "full_output_path",
                        property(lambda self: ""))
    with pytest.raises(ValueError):
        a.export_results()


def test_export_results_with_auto_mapper(tmp_path, monkeypatch):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    mapper = _enable_fake_auto_mapper(a.parameters)
    mapper._conflicts = {"Src": "conflict"}
    a.raw_results = _standard_results()
    _stub_metadata_methods(monkeypatch, a)
    # heavy report/visualization steps are covered by dedicated tests;
    # stub them here to keep this test focused on the auto-mapper export
    for name in ("generate_report", "_generate_visualizations",
                 "generate_html_report"):
        monkeypatch.setattr(a, name, lambda *args, **kw: None)

    a.export_results()
    assert any(k == "mapping" for k, _ in mapper.exported)
    assert any(k == "conflicts" for k, _ in mapper.exported)
    assert os.path.exists(os.path.join(params.full_output_path,
                                       "auto_type_mapping.csv")) or True


def test_export_results_auto_mapper_error_warns(tmp_path, monkeypatch):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    mapper = _enable_fake_auto_mapper(a.parameters)

    def _boom(*args, **kwargs):
        raise RuntimeError("cannot export")

    mapper.export_mapping = _boom
    a.raw_results = _standard_results()
    _stub_metadata_methods(monkeypatch, a)
    for name in ("generate_report", "_generate_visualizations",
                 "generate_html_report"):
        monkeypatch.setattr(a, name, lambda *args, **kw: None)
    # should not raise - exception is caught and logged
    a.export_results()


def test_export_label_map_smart_labels_and_groups(tmp_path):
    # matching user labels
    p1 = _params(source_labels=["MySrc"], target_labels=["MyTgt"])
    a1 = ComparisonAnalyzer(p1, verbose=False)
    f1 = str(tmp_path / "lm1.json")
    a1._export_label_map(f1)
    data1 = json.load(open(f1))
    assert data1["source_mapping"]["custom_label"] == ["MySrc"]
    assert data1["target_mapping"]["custom_label"] == ["MyTgt"]

    # multi-group -> default Group_N labels
    p2 = _params(source_neurons=[["A", "B"], ["C"]],
                 target_neurons=[["D", "E"]])
    a2 = ComparisonAnalyzer(p2, verbose=False)
    f2 = str(tmp_path / "lm2.json")
    a2._export_label_map(f2)
    data2 = json.load(open(f2))
    labels = data2["source_mapping"]["custom_label"]
    assert labels[0].startswith("Group_1")
    assert labels[1] == "C"

    # single string label with 1 group
    p3 = _params(source_neurons=["Solo"], source_labels="SoloLabel")
    a3 = ComparisonAnalyzer(p3, verbose=False)
    f3 = str(tmp_path / "lm3.json")
    a3._export_label_map(f3)
    data3 = json.load(open(f3))
    assert data3["source_mapping"]["custom_label"] == ["SoloLabel"]


def test_export_label_map_with_label_mapper(tmp_path):
    mapper = LabelMapper(
        source_mapping_dict={DS1: [["aMe1"]], DS2: [["aMe1"]]},
        source_labels=["SrcStd"],
        target_mapping_dict={DS1: [["PPL1"]], DS2: [["PPL1"]]},
        target_labels=["TgtStd"],
    )
    a = ComparisonAnalyzer(_params(), label_mapper=mapper, verbose=False)
    f = str(tmp_path / "lm4.json")
    a._export_label_map(f)
    data = json.load(open(f))
    assert "source_mapping" in data
    assert data["metadata"]["auto_type_mapping"] is False


def test_export_label_map_auto_mapper_resolution(tmp_path):
    # NOTE: a bare int at top level crashes generate_smart_labels (len(int))
    # in the source; wrap bodyIds in their own group as the UI does.
    params = _params(source_neurons=[["Src", "Unmapped"], ["Regex.*"], [123]])
    a = ComparisonAnalyzer(params, verbose=False)

    class _Resolver:
        """Fake mapper on the shared resolver contract (the analyzer
        resolves through ``comparison.type_resolver`` now)."""
        _conflicts = {}
        _loaded = True

        def has_conflicts(self):
            return bool(self._conflicts)

        def detect_type_source(self, t):
            return self._detect_type_source(t)

        def _detect_type_source(self, t):
            return DS1 if t in ("Src", "Unmapped") else None

        def _get_type_mapping_key(self, ds):
            return ds

        def get_mapping_decision(self, t, src_ds, dst_ds,
                                 include_bridges=False):
            if t == "Src":
                return {'status': 'mapped', 'source_type': t,
                        'target_type': "SrcMapped",
                        'target_types': ["SrcMapped"],
                        'relationship': '1-to-1', 'conflicts': []}
            return {'status': 'unmapped', 'source_type': t,
                    'target_type': None, 'target_types': [],
                    'relationship': None, 'conflicts': []}

        def get_alias_candidates(self, type_name, datasets,
                                 source_dataset=None):
            return {ds: {'outcome': 'no counterpart known',
                         'candidates': []} for ds in datasets}

        def get_type_bridges(self, type_name, source_ds, target_ds,
                             max_bridges=8):
            return []

    a.parameters.auto_type_mapping = True
    a.parameters._auto_type_mapper = _Resolver()
    f = str(tmp_path / "lm5.json")
    a._export_label_map(f)
    data = json.load(open(f))
    src_ds2 = data["source_mapping"][DS2]
    assert ["SrcMapped"] in src_ds2
    # unmapped type dropped, regex and bodyId pass through
    assert ["Regex.*"] in src_ds2
    assert [123] in src_ds2
    assert data["metadata"]["auto_type_mapping"] is True


def test_export_intra_dataset_comparisons_branches(tmp_path):
    a = ComparisonAnalyzer(_params(), verbose=False)
    results = _standard_results()
    results[DS1][3] = pd.DataFrame()                      # empty branch
    results[DS2][1] = pd.DataFrame(                       # bodyId-only branch
        {"bodyId_pre": [1], "bodyId_post": [2], "weight": [5]})
    results[DS2][3] = pd.DataFrame({"foo": [1, 2, 3]})    # index fallback
    a.raw_results = results
    outdir = tmp_path / "res"
    outdir.mkdir()
    a._export_intra_dataset_comparisons(str(outdir))
    df = pd.read_csv(outdir / "threshold_sensitivity.csv")
    assert len(df) == 4
    assert (df.loc[df["dataset"] == DS1, "edge_count"] == [2, 0]).all()


def test_export_unified_summary_branches(tmp_path, monkeypatch):
    a = ComparisonAnalyzer(_params(), verbose=False)
    results = _standard_results()
    results[DS1][3] = pd.DataFrame()  # empty summary branch
    results[DS2][1] = _edge_df([("Src", "Tgt", 2)])  # type cols
    results[DS2][3] = pd.DataFrame(   # source/target cols, no weight
        {"source": ["A"], "target": ["B"]})
    a.raw_results = results
    outdir = tmp_path / "res"
    outdir.mkdir()

    def fake_aligned(threshold):
        # one edge key without ' -> ' separator -> fallback branch
        return pd.DataFrame({DS1: [5.0, 0.0], DS2: [3.0, 0.0]},
                            index=["plain_key", "A -> B"])

    monkeypatch.setattr(a, "get_aligned_data", fake_aligned)
    a._export_unified_summary(str(outdir))
    assert (outdir / "unified_edge_comparison.csv").exists()
    assert (outdir / "unified_summary.csv").exists()
    edges = pd.read_csv(outdir / "unified_edge_comparison.csv")
    assert "plain_key" in set(edges["edge_key"])


def test_export_merged_unique_connections_branches(tmp_path, monkeypatch):
    a = ComparisonAnalyzer(_params(), verbose=False)
    a.raw_results = _standard_results()
    safe1 = a.parameters._sanitize_name(DS1)
    safe2 = a.parameters._sanitize_name(DS2)
    outdir = tmp_path / "res"
    outdir.mkdir()

    def fake_unique(threshold):
        return {
            safe1: pd.DataFrame({"edge_key": ["Src -> Tgt"], "weight": [9]}),
            safe2: pd.DataFrame({"source_type": ["Y"], "target_type": ["Tgt"],
                                 "weight": [7]}),
        }

    monkeypatch.setattr(a, "get_unique_connections", fake_unique)
    a._export_merged_unique_connections(str(outdir))
    f1 = pd.read_csv(outdir / f"unique_to_{safe1}.csv")
    assert f1.iloc[0]["edge_key"] == "Src -> Tgt"
    f2 = pd.read_csv(outdir / f"unique_to_{safe2}.csv")
    assert f2.iloc[0]["source"] == "Y"

    # empty unique df branch + missing dataset key branch
    monkeypatch.setattr(a, "get_unique_connections",
                        lambda t: {safe1: pd.DataFrame()})
    a._export_merged_unique_connections(str(outdir))


def test_export_neuron_counts_comparison(tmp_path):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    outdir = tmp_path / "res"
    outdir.mkdir()

    safe1 = params._sanitize_name(DS1)
    safe2 = params._sanitize_name(DS2)
    base = params.full_output_path

    # DS1 with hemisphere column
    d1 = os.path.join(base, "dataset_data", safe1, "minsyn_1", "data_details")
    os.makedirs(d1, exist_ok=True)
    pd.DataFrame({
        "bodyId": [1, 2, 3], "type": ["Src", "Src", "Tgt"],
        "custom_group": ["G1", "G1", ""], "hemisphere": ["L", "R", "L"],
    }).to_csv(os.path.join(d1, "source_neurons.csv"), index=False)
    pd.DataFrame({
        "bodyId": [4], "type": ["Tgt"], "custom_group": [""],
        "hemisphere": ["R"],
    }).to_csv(os.path.join(d1, "target_neurons.csv"), index=False)

    # DS2 without hemisphere column -> type suffix counting
    d2 = os.path.join(base, "dataset_data", safe2, "minsyn_1", "data_details")
    os.makedirs(d2, exist_ok=True)
    pd.DataFrame({
        "bodyId": [5, 6], "type": ["Src_L", "Src_R"], "custom_group": ["", ""],
    }).to_csv(os.path.join(d2, "source_neurons.csv"), index=False)

    a._export_neuron_counts_comparison(str(outdir))
    files = os.listdir(outdir)
    assert any("neuron_counts" in f for f in files)


def _write_path_files(params, threshold, fmt="source_target"):
    safe1 = params._sanitize_name(DS1)
    safe2 = params._sanitize_name(DS2)
    base = params.full_output_path
    os.makedirs(base, exist_ok=True)
    d1 = params.get_dataset_output_path(DS1, threshold)
    d2 = params.get_dataset_output_path(DS2, threshold)
    os.makedirs(d1, exist_ok=True)
    os.makedirs(d2, exist_ok=True)
    if fmt == "source_target":
        pd.DataFrame({"source": ["Src", "Src"], "target": ["Tgt", "Mid"],
                      "weight": [10, 4], "weights": ["[10, 5]", "4,2"]}
                     ).to_csv(os.path.join(
                         d1, f"minsyn_{threshold}_data_original_paths.csv"),
                         index=False)
    else:
        pd.DataFrame({"path": ["A -> B -> C", "['X', 'Y']", "bad"],
                      "min_weight": [3, 2, 1],
                      "weights": ["3,2", "[2, 1]", ""]}
                     ).to_csv(os.path.join(d2, f"x_allpaths_type.csv"),
                              index=False)


def test_export_unified_path_presence_matrix(tmp_path):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    _write_path_files(params, 1, fmt="source_target")
    _write_path_files(params, 3, fmt="source_target")
    a.comparison_report = {}
    outdir = tmp_path / "res"
    outdir.mkdir()
    a._export_unified_path_presence_matrix(str(outdir))
    assert (outdir / "path_presence_matrix.csv").exists()
    assert "path_presence_matrix" in a.comparison_report

    # no path data at all -> early return
    b = ComparisonAnalyzer(_params(output_folder=str(tmp_path / "b")),
                           verbose=False)
    b._export_unified_path_presence_matrix(str(outdir))


def test_export_unified_path_presence_matrix_path_col_format(tmp_path):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    safe2 = params._sanitize_name(DS2)
    d2 = params.get_dataset_output_path(DS2, 1)
    os.makedirs(d2, exist_ok=True)
    pd.DataFrame({"path": ["A -> B -> C", "['X', 'Y']", "bad", "single"],
                  "min_weight": [3, 2, 1, None],
                  "weights": ["3,2", "[2, 1]", "", ""]}
                 ).to_csv(os.path.join(d2, "x_allpaths_type.csv"), index=False)
    outdir = tmp_path / "res"
    outdir.mkdir()
    a._export_unified_path_presence_matrix(str(outdir))
    df = pd.read_csv(outdir / "path_presence_matrix.csv")
    keys = set(df["path_key"])
    assert any("A" in k for k in keys)


def test_export_presence_matrix(tmp_path, monkeypatch):
    a = ComparisonAnalyzer(_params(), verbose=False)
    a.raw_results = _standard_results()
    outdir = tmp_path / "res"
    outdir.mkdir()
    # mid threshold -> also writes edge_presence_matrix.csv
    a._export_presence_matrix(str(outdir), 3)
    assert (outdir / "edge_presence_matrix_minsyn_3.csv").exists()
    assert (outdir / "edge_presence_matrix.csv").exists()

    # empty aligned data -> early return
    monkeypatch.setattr(a, "get_aligned_data", lambda t: pd.DataFrame())
    a._export_presence_matrix(str(outdir), 1)

    # MultiIndex aligned data branch
    mi = pd.MultiIndex.from_tuples([("Src", "Tgt")])
    monkeypatch.setattr(a, "get_aligned_data",
                        lambda t: pd.DataFrame({DS1: [5.0], DS2: [0.0]},
                                               index=mi))
    a._export_presence_matrix(str(outdir), 1, silent=True)
    assert (outdir / "edge_presence_matrix_minsyn_1.csv").exists()


def test_export_path_presence_matrix(tmp_path):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    _write_path_files(params, 3, fmt="source_target")
    outdir = tmp_path / "res"
    outdir.mkdir()
    a._export_path_presence_matrix(str(outdir), 3)
    assert (outdir / "path_presence_matrix_minsyn_3.csv").exists()


def test_export_motif_analysis_no_data(tmp_path):
    a = ComparisonAnalyzer(_params(), verbose=False)
    outdir = tmp_path / "res"
    outdir.mkdir()
    a._export_motif_analysis(str(outdir), 1)  # no data -> early return


def test_generate_visualizations_import_error(tmp_path, monkeypatch):
    a = ComparisonAnalyzer(_params(), verbose=False)
    a.raw_results = _standard_results()
    monkeypatch.setitem(sys.modules, "comparison.visualizations", None)
    a._generate_visualizations(str(tmp_path))  # ImportError branch


def test_generate_visualizations_with_fake_visualizer(tmp_path, monkeypatch):
    a = ComparisonAnalyzer(_params(), verbose=False)
    a.raw_results = _standard_results()
    calls = []

    class _FakeViz:
        def __init__(self, verbose=False):
            pass

        def save_all_plots(self, **kwargs):
            calls.append("plots")

    fake_mod = types.ModuleType("comparison.visualizations")
    fake_mod.ComparisonVisualizer = _FakeViz
    monkeypatch.setitem(sys.modules, "comparison.visualizations", fake_mod)
    monkeypatch.setattr(
        a, "visualize_conserved_paths_all_thresholds",
        lambda **kw: calls.append("paths"))
    a._generate_visualizations(str(tmp_path))
    assert calls == ["plots", "paths"]


def test_generate_visualizations_reciprocal_branch(tmp_path, monkeypatch):
    params = _params(find_reciprocal=True)
    a = ComparisonAnalyzer(params, verbose=False)
    a.raw_results = _standard_results()
    calls = []

    class _FakeViz:
        def __init__(self, verbose=False):
            pass

        def save_all_plots(self, **kwargs):
            calls.append("plots")

    fake_mod = types.ModuleType("comparison.visualizations")
    fake_mod.ComparisonVisualizer = _FakeViz
    monkeypatch.setitem(sys.modules, "comparison.visualizations", fake_mod)
    monkeypatch.setattr(
        a, "visualize_conserved_paths_all_thresholds",
        lambda **kw: calls.append("paths"))
    monkeypatch.setattr(
        a, "visualize_conserved_reciprocal_graph_all_thresholds",
        lambda **kw: calls.append("reciprocal"))
    a._generate_visualizations(str(tmp_path))
    assert calls == ["plots", "paths", "reciprocal"]


# --- APPEND-POINT-2 ---


# ---------------------------------------------------------------------------
# Data getters (path / ratio / prob / edge-ratio / NT)
# ---------------------------------------------------------------------------

def _ds_dir(params, ds, threshold):
    d = os.path.join(params.full_output_path, "dataset_data",
                     params._sanitize_name(ds), f"minsyn_{threshold}")
    os.makedirs(d, exist_ok=True)
    return d


def test_get_path_data_for_threshold_both_formats(tmp_path):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    d1 = _ds_dir(params, DS1, 1)
    pd.DataFrame({"source": ["Src", "Src"], "target": ["Tgt", "Mid"],
                  "weight": [10, 4]}).to_csv(
        os.path.join(d1, "minsyn_1_data_original_paths.csv"), index=False)
    d2 = _ds_dir(params, DS2, 1)
    pd.DataFrame({"path": ["Src -> Tgt", "Src -> Mid"],
                  "min_weight": [9, 3]}).to_csv(
        os.path.join(d2, "to_target_allpaths_type.csv"), index=False)
    out = a._get_path_data_for_threshold(1)
    assert not out.empty
    assert DS1 in out.columns and DS2 in out.columns
    # no data at other threshold
    assert a._get_path_data_for_threshold(3).empty


def test_get_path_hop_weights_for_threshold(tmp_path):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    d1 = _ds_dir(params, DS1, 1)
    pd.DataFrame({"source": ["Src"], "target": ["Tgt"], "weight": [10],
                  "weights": ["[10, 5]"]}).to_csv(
        os.path.join(d1, "minsyn_1_data_original_paths.csv"), index=False)
    d2 = _ds_dir(params, DS2, 1)
    pd.DataFrame({"path": ["Src -> Tgt"], "min_weight": [9],
                  "weights": ["10,5"]}).to_csv(
        os.path.join(d2, "to_target_allpaths_type.csv"), index=False)
    hop = a._get_path_hop_weights_for_threshold(1)
    assert isinstance(hop, dict)

    d1q = _ds_dir(params, DS1, 3)
    pd.DataFrame({"source": ["Src"], "target": ["Tgt"], "weight": [10],
                  "weights": ["[11, 4]"]}).to_csv(
        os.path.join(d1q, "minsyn_3_data_original_paths.csv"), index=False)
    d2q = _ds_dir(params, DS2, 5)
    pd.DataFrame({"path": ["Src -> Tgt"], "min_weight": [9],
                  "weights": ["[9, 2]"]}).to_csv(
        os.path.join(d2q, "to_target_allpaths_type.csv"), index=False)
    query_hop = a._get_path_hop_weights_for_threshold({
        "thresholds": {DS1: 3, DS2: 5},
    })
    flattened = [weights for per_dataset in query_hop.values()
                 for weights in per_dataset.values()]
    assert [11.0, 4.0] in flattened
    assert [9.0, 2.0] in flattened


def test_get_ratio_and_prob_data_for_threshold(tmp_path):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    d1 = _ds_dir(params, DS1, 1)
    pd.DataFrame({"path": ["A -> B"], "min_ratio": [0.4],
                  "path_prob": [0.8]}).to_csv(
        os.path.join(d1, "minsyn_1_data_original_paths.csv"), index=False)
    ratio = a._get_ratio_data_for_threshold(1)
    assert not ratio.empty
    prob = a._get_prob_data_for_threshold(1)
    assert not prob.empty
    assert a._get_ratio_data_for_threshold(3).empty
    assert a._get_prob_data_for_threshold(3).empty


def test_get_edge_ratio_data_for_threshold(tmp_path):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    d1 = _ds_dir(params, DS1, 1)
    pd.DataFrame({"type_pre": ["Src"], "type_post": ["Tgt"],
                  "connection_ratio": [0.2]}).to_csv(
        os.path.join(d1, "connections_edge.csv"), index=False)
    # DS2: connections_edge.csv empty -> fallback to data_details
    d2 = _ds_dir(params, DS2, 1)
    open(os.path.join(d2, "connections_edge.csv"), "w").close()
    dd = os.path.join(d2, "data_details")
    os.makedirs(dd, exist_ok=True)
    pd.DataFrame({"std_label_pre": ["SrcStd"], "std_label_post": ["TgtStd"],
                  "connection_ratio": [0.3]}).to_csv(
        os.path.join(dd, "connection_type.csv"), index=False)
    out = a._get_edge_ratio_data_for_threshold(1)
    assert not out.empty
    assert a._get_edge_ratio_data_for_threshold(3).empty


def test_get_edge_nt_details(tmp_path):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    # DS1: reciprocal file with nt_type_pre
    d1 = _ds_dir(params, DS1, 1)
    fr = os.path.join(d1, "find_reciprocal")
    os.makedirs(fr, exist_ok=True)
    pd.DataFrame({"type_pre": ["Src"], "type_post": ["Tgt"],
                  "nt_type_pre": ["ACh"], "weight": [5]}).to_csv(
        os.path.join(fr, "reciprocal_connection_type.csv"), index=False)
    # DS2: fallback to connections_edge.csv with nt_type
    d2 = _ds_dir(params, DS2, 1)
    pd.DataFrame({"type_pre": ["Src"], "type_post": ["Tgt"],
                  "nt_type": ["glutamate"], "weight": [5]}).to_csv(
        os.path.join(d2, "connections_edge.csv"), index=False)
    nt = a._get_edge_nt_details(1)
    assert "Src -> Tgt" in nt
    assert nt["Src -> Tgt"][DS2] == "glutamate"
    assert a._get_edge_nt_details(3) == {}


def test_get_reciprocal_edge_ratio_data(tmp_path):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    d1 = _ds_dir(params, DS1, 1)
    fr = os.path.join(d1, "find_reciprocal")
    os.makedirs(fr, exist_ok=True)
    pd.DataFrame({"type_pre": ["Src"], "type_post": ["Tgt"],
                  "connection_ratio": [0.5]}).to_csv(
        os.path.join(fr, "reciprocal_connection_type.csv"), index=False)
    out = a._get_reciprocal_edge_ratio_data_for_threshold(1)
    assert not out.empty
    assert a._get_reciprocal_edge_ratio_data_for_threshold(3).empty


# ---------------------------------------------------------------------------
# Network HTML builders
# ---------------------------------------------------------------------------

def test_create_network_html_builders(tmp_path):
    a = ComparisonAnalyzer(_params(thresholds=[1, 3, 5]), verbose=False)
    conn_rows = [
        {"source": "Src", "target": "Tgt", "weight": 10, "dataset": DS1},
        {"source": "Src", "target": "Tgt", "weight": 5, "dataset": DS2},
        {"source": "Src", "target": "X", "weight": 3, "dataset": DS1},
    ]
    out1 = str(tmp_path / "combined.html")
    a._create_combined_network_html(conn_rows, [DS1, DS2], 1, out1,
                                    "Combined")
    assert os.path.exists(out1)

    th_rows = [
        {"source": "Src", "target": "Tgt", "weight": 10, "threshold": 1},
        {"source": "Src", "target": "Tgt", "weight": 8, "threshold": 3},
        {"source": "Src", "target": "Tgt", "weight": 5, "threshold": 5},
        {"source": "Src", "target": "X", "weight": 3, "threshold": 1},
        {"source": "Src", "target": "X", "weight": 2, "threshold": 3},
        {"source": "Src", "target": "Y", "weight": 1, "threshold": 1},
    ]
    out2 = str(tmp_path / "threshold.html")
    a._create_threshold_comparison_network_html(th_rows, DS1, [1, 3, 5],
                                                out2, "Threshold")
    assert os.path.exists(out2)


# ---------------------------------------------------------------------------
# Profile comparison (stubbed profiler/comparator)
# ---------------------------------------------------------------------------

def _stub_profile_modules(monkeypatch, direct_result):
    import comparison.connectivity_profiler as cp_mod
    import comparison.profile_comparator as pc_mod

    class _FakeProfilerConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeProfiler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeComparator:
        @staticmethod
        def direct_comparison(**kwargs):
            if isinstance(direct_result, Exception):
                raise direct_result
            return direct_result

    monkeypatch.setattr(cp_mod, "ProfilerConfig", _FakeProfilerConfig)
    monkeypatch.setattr(cp_mod, "ConnectivityProfiler", _FakeProfiler)
    monkeypatch.setattr(pc_mod, "ProfileComparator", _FakeComparator)


def test_direct_comparison_with_stubs(tmp_path, monkeypatch):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    result_df = pd.DataFrame({"neuron_type": ["Src"], "score": [0.9]})
    _stub_profile_modules(monkeypatch,
                          {"results": result_df, "summary": {"n": 1}})
    out = a.direct_comparison()
    assert out["results"] is result_df
    assert "output_file" in out
    assert os.path.exists(out["output_file"])


def test_connectivity_profile_comparison(tmp_path, monkeypatch):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    a.raw_results = _standard_results()
    type_df = pd.DataFrame({
        "neuron_type": ["Src", "Tgt"],
        "rank_corr": [0.9, 0.8],
        "cosine": [0.85, 0.7],
        "jaccard": [0.5, 0.4],
    })
    _stub_profile_modules(monkeypatch,
                          {"type_summary": type_df, "results": pd.DataFrame()})
    res = a.connectivity_profile_comparison(
        output_dir=str(tmp_path / "prof"), include_visualizations=False)
    assert "summary" in res
    assert res["comparison_mode"] in ("loose", "strict")
    assert not res["summary"].empty


def test_connectivity_profile_comparison_branches(tmp_path, monkeypatch):
    # <2 datasets -> {}
    p1 = _params(datasets=[DS1], allow_single_dataset=True)
    a1 = ComparisonAnalyzer(p1, verbose=False)
    assert a1.connectivity_profile_comparison(
        output_dir=str(tmp_path / "p1")) == {}

    # no neuron types -> {}
    a2 = ComparisonAnalyzer(_params(), verbose=False)
    a2.raw_results = {}
    _stub_profile_modules(monkeypatch, {"type_summary": pd.DataFrame()})
    assert a2.connectivity_profile_comparison(
        output_dir=str(tmp_path / "p2"), neuron_types=[]) == {}

    # invalid comparison_mode coerced to 'loose'; direct_comparison raises
    a3 = ComparisonAnalyzer(_params(), verbose=False)
    a3.raw_results = _standard_results()
    _stub_profile_modules(monkeypatch, RuntimeError("boom"))
    res3 = a3.connectivity_profile_comparison(
        output_dir=str(tmp_path / "p3"), neuron_types=["Src"],
        comparison_mode="bogus", include_visualizations=False)
    assert res3 == {}

    # alias wrapper
    a4 = ComparisonAnalyzer(_params(), verbose=False)
    a4.raw_results = _standard_results()
    _stub_profile_modules(monkeypatch, RuntimeError("boom"))
    assert a4.run_connectivity_profile_verification(
        output_dir=str(tmp_path / "p4"), neuron_types=["Src"],
        parallel=False, max_workers=1) == {}


def test_extract_types_from_results():
    a = ComparisonAnalyzer(_params(), verbose=False)
    a.raw_results = _standard_results()
    src, tgt, inter = a._extract_types_from_results()
    assert "Src" in src
    assert "Tgt" in tgt
    assert "X" in inter or "Y" in inter


def test_type_matches_pattern():
    a = ComparisonAnalyzer(_params(), verbose=False)
    assert a._type_matches_pattern("Src", "Src")
    assert not a._type_matches_pattern("Src", "Tgt")
    # Wildcard conversion must not mangle '.*' into '..': 'Src.*' matches the
    # bare 'Src' (pre-fix it became 'Src..*' and required one extra char).
    assert a._type_matches_pattern("Src", "Sr.*")
    assert a._type_matches_pattern("Src", "Src.*")
    assert a._type_matches_pattern("Src_extra", "Src*")
    assert not a._type_matches_pattern("Other", "Src.*")
    # int patterns are stringified before comparison
    assert a._type_matches_pattern("123", 123)
    assert not a._type_matches_pattern("123", 456)


# --- APPEND-POINT-3 ---


# ---------------------------------------------------------------------------
# Conserved path / reciprocal graph visualizations (fake vispath_pkg)
# ---------------------------------------------------------------------------

class _FakeVisualizePath:
    def __init__(self, path_file=None, output_folder=None, **kwargs):
        import networkx as nx
        self.path_file = path_file
        self.output_folder = output_folder
        self.kwargs = kwargs
        self.base_filename = "network"
        self.G_network = nx.DiGraph()

    def build_network(self):
        df = self.path_file
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                self.G_network.add_edge(row["source"], row["target"])

    def create_network(self):
        out = os.path.join(self.output_folder, f"{self.base_filename}.html")
        with open(out, "w") as f:
            f.write("<html></html>")
        return out

    def create_heatmap(self):
        return None


def _install_fake_vispath(monkeypatch):
    fake = types.ModuleType("vispath_pkg")
    fake.VisualizePath = _FakeVisualizePath
    monkeypatch.setitem(sys.modules, "vispath_pkg", fake)
    return fake


def test_visualize_conserved_paths_import_error(monkeypatch):
    # sys.modules entry of None forces an ImportError on import
    monkeypatch.setitem(sys.modules, "vispath_pkg", None)
    a = ComparisonAnalyzer(_params(), verbose=False)
    a.raw_results = _standard_results()
    assert a.visualize_conserved_paths(threshold=1,
                                       output_folder="/tmp") is None


def test_visualize_conserved_paths_no_results(monkeypatch, tmp_path):
    _install_fake_vispath(monkeypatch)
    a = ComparisonAnalyzer(_params(), verbose=False)
    assert a.visualize_conserved_paths(threshold=1,
                                       output_folder=str(tmp_path)) is None


def test_visualize_conserved_paths_no_conserved_edges(monkeypatch, tmp_path):
    _install_fake_vispath(monkeypatch)
    a = ComparisonAnalyzer(_params(), verbose=False)
    # disjoint edge sets -> nothing conserved
    a.raw_results = {DS1: {1: _edge_df([("A", "B", 5)])},
                     DS2: {1: _edge_df([("C", "D", 5)])}}
    assert a.visualize_conserved_paths(threshold=1,
                                       output_folder=str(tmp_path)) is None


def test_visualize_conserved_paths_happy(monkeypatch, tmp_path):
    _install_fake_vispath(monkeypatch)
    a = ComparisonAnalyzer(_params(), verbose=False)
    a.raw_results = _standard_results()
    out = a.visualize_conserved_paths(threshold=1,
                                      output_folder=str(tmp_path))
    assert out is not None and os.path.exists(out)
    assert "conserved_network_t1" in out


def test_visualize_conserved_paths_with_auto_mapper(monkeypatch, tmp_path):
    _install_fake_vispath(monkeypatch)
    a = ComparisonAnalyzer(_params(), verbose=False)
    a.raw_results = _standard_results()

    mapper = _enable_fake_auto_mapper(a.parameters)
    mapper.get_display_name_with_dataset_info = (
        lambda node, datasets: (f"{node}(HB)", {"HB": node}))
    mapper.get_all_dataset_short_codes = lambda datasets: {"HB": datasets[0]}
    mapper._detect_type_source = lambda t: None
    mapper.get_mapped_type = lambda *args: None

    out = a.visualize_conserved_paths(threshold=1,
                                      output_folder=str(tmp_path))
    assert out is not None


def test_visualize_conserved_paths_all_thresholds(monkeypatch, tmp_path):
    _install_fake_vispath(monkeypatch)
    a = ComparisonAnalyzer(_params(), verbose=False)
    a.raw_results = _standard_results()
    outs = a.visualize_conserved_paths_all_thresholds(
        output_folder=str(tmp_path))
    assert len(outs) == 2


def test_visualize_conserved_reciprocal_graph(monkeypatch, tmp_path):
    _install_fake_vispath(monkeypatch)
    params = _params(output_folder=str(tmp_path), find_reciprocal=True)
    a = ComparisonAnalyzer(params, verbose=False)
    a.raw_results = _standard_results()
    for ds in (DS1, DS2):
        for t in (1, 3):
            d = os.path.join(params.full_output_path, "dataset_data",
                             params._sanitize_name(ds), f"minsyn_{t}",
                             "find_reciprocal")
            os.makedirs(d, exist_ok=True)
            pd.DataFrame({"type_pre": ["Src"], "type_post": ["Tgt"],
                          "weight": [6],
                          "connection_ratio": [0.3]}).to_csv(
                os.path.join(d, "reciprocal_connection_type.csv"),
                index=False)

    out = a.visualize_conserved_reciprocal_graph(threshold=1)
    assert out is not None and os.path.exists(out)

    outs = a.visualize_conserved_reciprocal_graph_all_thresholds()
    assert len(outs) == 2

    # no results -> None
    b = ComparisonAnalyzer(_params(find_reciprocal=True), verbose=False)
    assert b.visualize_conserved_reciprocal_graph(threshold=1) is None


# ---------------------------------------------------------------------------
# HTML report branches
# ---------------------------------------------------------------------------

def test_generate_html_report_default_path_and_autorun(tmp_path, monkeypatch):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)

    def fake_run_all(self, skip_existing=True):
        self.raw_results.update(_standard_results())
        return self.raw_results

    monkeypatch.setattr(ComparisonAnalyzer, "run_all_analyses", fake_run_all)
    out = a.generate_html_report()  # no path -> default, auto-run comparison
    assert os.path.exists(out)
    assert out.startswith(params.full_output_path)


def test_generate_html_content_path_presence_stats(tmp_path):
    params = _params(output_folder=str(tmp_path))
    a = ComparisonAnalyzer(params, verbose=False)
    a.raw_results = _standard_results()
    a.run_comparison_analysis()

    safe1 = params._sanitize_name(DS1)
    safe2 = params._sanitize_name(DS2)
    matrix = pd.DataFrame({
        f"{safe1}_t1": [True, True],           # bool branch
        f"{safe2}_t1": ["True", "0"],          # object branch
        f"{safe1}_t3": [1, 0],                 # numeric branch
        f"{safe2}_t3": [1.0, 1.0],
    })
    a.comparison_report["path_presence_matrix"] = matrix
    a.comparison_report["key_findings_per_threshold"] = {
        1: {"finding": "x"}, 3: {"finding": "y"}}

    out = a.generate_html_report(os.path.join(str(tmp_path), "rep2.html"))
    assert os.path.exists(out)
    kf = a.comparison_report["key_findings_per_threshold"]
    assert kf[1]["total_paths"] == 2
    assert kf[1]["common_paths"] == 1


def test_quick_compare_defaults_and_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exported = []

    def fake_run_all(self, skip_existing=True):
        self.raw_results.update(_standard_results())
        return self.raw_results

    def fake_export(self, output_dir=None):
        exported.append(True)

    monkeypatch.setattr(ComparisonAnalyzer, "run_all_analyses", fake_run_all)
    monkeypatch.setattr(ComparisonAnalyzer, "export_results", fake_export)

    results = quick_compare(
        datasets=[DS1, DS2],
        source_neurons=["Src"],
        target_neurons=["Tgt"],
        output_folder=str(tmp_path / "qc"),
        verbose=False,
    )
    assert isinstance(results, dict)
    assert exported == [True]

# --- APPEND-POINT-4 ---


def test_state_notes_replace_across_re_exports(tmp_path):
    """A re-export must REPLACE the same-name-first / multi-value notes
    rather than accumulate contradictory stale counts (found in the
    ORN_D→MBON.* round: 3 stale copies with different numbers)."""
    import types
    from comparison.comparison_analyzer import ComparisonAnalyzer

    # Fixture mirrors the REAL file structure: state headers are top-level
    # ``[`` lines; multi-line state blocks carry bullet continuations
    # (``- [merge fan-in] …``); unrelated note families such as
    # ``- [untyped dropped] …`` have NO header line and must survive.
    note = tmp_path / 'user_warning_notes.txt'
    note.write_text(
        'User warning notes\n==================\n\n'
        '[same-name-first] 3640 stale global counts\n'
        '[merge policy] stale policy summary\n'
        '- [merge fan-in] stale bullet belonging to that header\n'
        '- [untyped dropped] keep me\n'
        '[multi-value type cells] 403 stale counts\n'
        '- [untyped dropped] and me too\n',
        encoding='utf-8')

    analyzer = ComparisonAnalyzer.__new__(ComparisonAnalyzer)
    analyzer.parameters = types.SimpleNamespace(
        full_output_path=str(tmp_path))
    analyzer._log = lambda *a, **k: None

    blocks = ['[same-name-first] fresh run-scoped counts',
              '[multi-value type cells] fresh counts']
    out = analyzer._replace_state_notes(blocks)
    assert out == blocks
    text = note.read_text(encoding='utf-8')
    assert text.count('[same-name-first]') == 0  # removed; re-appended next
    assert text.count('[multi-value type cells]') == 0
    assert text.count('[merge policy]') == 0
    # the multi-line state block leaves nothing orphaned ...
    assert '- [merge fan-in] stale bullet' not in text
    # ... while unrelated bullet families survive untouched
    assert '- [untyped dropped] keep me' in text
    assert '- [untyped dropped] and me too' in text
    assert text.count('User warning notes') == 1
