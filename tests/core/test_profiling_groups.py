"""Connectivity Profiling custom grouping and aggregation levels.

The profiling tab drives ``ConnectivityProfileComparer`` (profile_comparator).
Its ``aggregation_level`` must now be functional:
- 'type' (default): pattern query items ('aMe.*') expand into their matched
  types, each an independent row
- 'bodyid': every individual neuron is its own row ({bodyId}_{type})
- 'custom': rows are user-defined custom groups from a LabelMapper preset
  (custom_mapping_file), a nested-list query or group_map_csv
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.connectivity_profiler import ConnectivityProfiler  # noqa: E402
from comparison.profile_comparator import ConnectivityProfileComparer  # noqa: E402

DATASET = "hemibrain:v1.2.1"


def _make_comparer(**kwargs):
    """Build a comparer with a fake profiler (no network, no disk access)."""
    params = dict(query=["Mi1"], dataset=DATASET, verbose=False)
    params.update(kwargs)
    comparer = ConnectivityProfileComparer(**params)
    comparer.profiler.get_bodyids_for_type = lambda t, ds: {"Mi1": [1]}.get(t, [])
    comparer.profiler.get_types_for_bodyids = lambda bids, ds: {}
    comparer.profiler.list_types = lambda p, ds: []
    return comparer


# ---------------------------------------------------------------------------
# ConnectivityProfiler.list_types (pattern -> types)
# ---------------------------------------------------------------------------

def test_list_types_pattern_matching_and_cache(monkeypatch):
    profiler = ConnectivityProfiler(datasets=[DATASET], verbose=False)
    monkeypatch.setattr(profiler, "_load_all_types", lambda ds: ["aMe12", "aMe10", "Mi1", "DN1p"])
    assert profiler.list_types("aMe.*") == ["aMe12", "aMe10"]
    assert profiler.list_types(".*DN.*") == ["DN1p"]
    assert profiler.list_types("DN") == ["DN1p"]  # re.match anchored at start
    assert profiler.list_types(None) == ["aMe12", "aMe10", "Mi1", "DN1p"]
    # cached: a later change of the loader does not alter results
    monkeypatch.setattr(profiler, "_load_all_types", lambda ds: ["CHANGED"])
    assert profiler.list_types(None) == ["aMe12", "aMe10", "Mi1", "DN1p"]


def test_list_types_invalid_pattern_falls_back_to_literal():
    profiler = ConnectivityProfiler(datasets=[DATASET], verbose=False)
    profiler._load_all_types = lambda ds: ["aMe12", "Mi1"]
    assert profiler.list_types("[") == []  # invalid regex -> literal compare
    assert profiler.list_types("aMe12") == ["aMe12"]


# ---------------------------------------------------------------------------
# Aggregation level = type: patterns expand to independent types
# ---------------------------------------------------------------------------

def test_type_aggregation_expands_patterns_to_independent_types():
    comparer = _make_comparer(query=["aMe.*"])

    def fake_bodyids(t, ds):
        return {"aMe12": [1, 2], "aMe10": [3], "aMe.*": []}.get(t, [])

    comparer.profiler.get_bodyids_for_type = fake_bodyids
    comparer.profiler.list_types = lambda p, ds: ["aMe12", "aMe10"] if p == "aMe.*" else []

    neurons = comparer._get_neurons_to_compare()
    assert neurons == {"aMe12": [1, 2], "aMe10": [3]}
    assert "aMe.*" not in neurons


def test_type_aggregation_keeps_exact_types_and_bodyids():
    comparer = _make_comparer(query=["Mi1", 42])
    comparer.profiler.get_types_for_bodyids = lambda bids, ds: {42: "X"}
    neurons = comparer._get_neurons_to_compare()
    assert neurons == {"Mi1": [1], "X": [42]}


def test_type_aggregation_unmatched_pattern_keeps_raw_item():
    comparer = _make_comparer(query=["NoMatch.*"])
    neurons = comparer._get_neurons_to_compare()
    assert neurons == {"NoMatch.*": ["NoMatch.*"]}


# ---------------------------------------------------------------------------
# Aggregation level = bodyid: every neuron is its own row
# ---------------------------------------------------------------------------

def test_bodyid_aggregation_rows_per_neuron():
    comparer = _make_comparer(query=["Mi1", 42], aggregation_level="bodyid")
    comparer.profiler.get_types_for_bodyids = lambda bids, ds: {42: "X"}
    neurons = comparer._get_neurons_to_compare()
    assert neurons == {"1_Mi1": [1], "42_X": [42]}


def test_bodyid_aggregation_expands_patterns_to_bodyids():
    comparer = _make_comparer(query=["aMe.*"], aggregation_level="bodyid")

    def fake_bodyids(t, ds):
        return {"aMe12": [1, 2], "aMe10": [3], "aMe.*": []}.get(t, [])

    comparer.profiler.get_bodyids_for_type = fake_bodyids
    comparer.profiler.list_types = lambda p, ds: ["aMe12", "aMe10"] if p == "aMe.*" else []
    neurons = comparer._get_neurons_to_compare()
    assert neurons == {"1_aMe12": [1], "2_aMe12": [2], "3_aMe10": [3]}


# ---------------------------------------------------------------------------
# Aggregation level = custom: LabelMapper presets / groups / literal items
# ---------------------------------------------------------------------------

def _write_mapping(tmp_path, dataset_key=DATASET):
    preset = {
        "source_mapping": {
            "custom_label": ["grp1", "grp2", "grp3"],
            dataset_key: [["aMe12", "aMe12_R"], ["aMe12_L"], []],
        }
    }
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(preset), encoding="utf-8")
    return str(path)


def test_custom_aggregation_loads_labelmapper_groups(tmp_path):
    mapping_path = _write_mapping(tmp_path)
    comparer = _make_comparer(query=["ignored"], aggregation_level="custom group",
                              custom_mapping_file=mapping_path)
    # the UI label 'custom group' is normalized to 'custom'
    assert comparer.aggregation_level == "custom"
    assert comparer._custom_group_names == ["grp1", "grp2"]
    neurons = comparer._get_neurons_to_compare()
    assert neurons == {"grp1": ["aMe12", "aMe12_R"], "grp2": ["aMe12_L"]}


def test_custom_mapping_matches_normalized_dataset_key(tmp_path):
    # mapping key uses the normalized name; profiling dataset uses ':' — the
    # loader must still find the groups
    mapping_path = _write_mapping(tmp_path, dataset_key="hemibrain_v1_2_1")
    comparer = _make_comparer(query=["x"], aggregation_level="custom",
                              custom_mapping_file=mapping_path)
    assert comparer._custom_group_names == ["grp1", "grp2"]


def test_custom_mapping_forces_custom_aggregation(tmp_path):
    mapping_path = _write_mapping(tmp_path)
    comparer = _make_comparer(query=["x"], aggregation_level="type",
                              custom_mapping_file=mapping_path)
    assert comparer.aggregation_level == "custom"


def test_custom_aggregation_nested_groups_unchanged():
    comparer = _make_comparer(query=[["G1", ["aMe12"]], ["G2", [42]]],
                              aggregation_level="custom")
    neurons = comparer._get_neurons_to_compare()
    assert neurons == {"G1": ["aMe12"], "G2": [42]}


def test_custom_aggregation_flat_items_taken_literally():
    comparer = _make_comparer(query=["aMe.*", "Mi1"], aggregation_level="custom")

    def boom(*args, **kwargs):
        raise AssertionError("no type resolution may happen under 'custom'")

    comparer.profiler.get_bodyids_for_type = boom
    comparer.profiler.list_types = boom
    neurons = comparer._get_neurons_to_compare()
    assert neurons == {"aMe.*": ["aMe.*"], "Mi1": ["Mi1"]}


# ---------------------------------------------------------------------------
# Aggregation level = type: a coarse taxonomy label expands into its real types
# ---------------------------------------------------------------------------

def test_type_aggregation_expands_coarse_label_to_real_types():
    """A cell-class / cell-type label that maps to several real types becomes
    one independent row per real type, like the network tab."""
    comparer = _make_comparer(query=["circadian_clock"])
    # The label has no direct bodyId match and is not a pattern.
    comparer.profiler.get_bodyids_for_type = lambda t, ds: {}.get(t, [])
    comparer.profiler.list_types = lambda p, ds: []
    comparer.profiler.get_types_for_label = lambda label, ds: {
        "aMe12": [1, 2], "DN1p": [3],
    }
    neurons = comparer._get_neurons_to_compare()
    assert neurons == {"aMe12": [1, 2], "DN1p": [3]}
    assert "circadian_clock" not in neurons


def test_type_aggregation_single_real_type_from_label():
    """A coarse label that collapses to one real type still becomes one row."""
    comparer = _make_comparer(query=["clock_type"])
    comparer.profiler.get_bodyids_for_type = lambda t, ds: {}.get(t, [])
    comparer.profiler.list_types = lambda p, ds: []
    comparer.profiler.get_types_for_label = lambda label, ds: {"DN1p": [5, 6]}
    neurons = comparer._get_neurons_to_compare()
    assert neurons == {"DN1p": [5, 6]}


def test_type_aggregation_unmatched_label_keeps_raw_item():
    """A label the resolver cannot map is kept literally so the user sees it."""
    comparer = _make_comparer(query=["NoSuchLabel"])
    comparer.profiler.get_bodyids_for_type = lambda t, ds: []
    comparer.profiler.list_types = lambda p, ds: []
    comparer.profiler.get_types_for_label = lambda label, ds: {}
    neurons = comparer._get_neurons_to_compare()
    assert neurons == {"NoSuchLabel": ["NoSuchLabel"]}


def test_type_aggregation_resolver_absent_keeps_raw_item():
    """Older profilers without get_types_for_label degrade to the raw literal."""
    comparer = _make_comparer(query=["Unmapped"])
    comparer.profiler.get_bodyids_for_type = lambda t, ds: []
    comparer.profiler.list_types = lambda p, ds: []
    comparer.profiler.get_types_for_label = None  # resolver not available
    neurons = comparer._get_neurons_to_compare()
    assert neurons == {"Unmapped": ["Unmapped"]}


def test_bodyid_aggregation_expands_coarse_label_to_neurons():
    """The same coarse label under bodyid aggregation rows every neuron."""
    comparer = _make_comparer(query=["circadian_clock"], aggregation_level="bodyid")
    comparer.profiler.get_bodyids_for_type = lambda t, ds: {}.get(t, [])
    comparer.profiler.list_types = lambda p, ds: []
    comparer.profiler.get_types_for_label = lambda label, ds: {
        "aMe12": [1, 2], "DN1p": [3],
    }
    neurons = comparer._get_neurons_to_compare()
    assert neurons == {"1_aMe12": [1], "2_aMe12": [2], "3_DN1p": [3]}


# ---------------------------------------------------------------------------
# run(): bodyid aggregation FILES the main matrices as bodyId-level results
# ---------------------------------------------------------------------------

def test_run_bodyid_aggregation_files_bodyid_matrices_and_type_average(monkeypatch):
    """The compared rows ARE individual neurons, so the main matrices are the
    bodyId-level results. Neither the separate pair loop nor a re-scored type
    average may run: both would recompute the pairs just scored."""
    comparer = _make_comparer(query=["Mi1", "Tm3"], aggregation_level="bodyid")

    profile = object()
    monkeypatch.setattr(comparer, "_extract_all_profiles",
                        lambda: ({"1_Mi1": profile, "2_Tm3": profile},
                                 {("Mi1", 1): profile, ("Tm3", 2): profile}))
    monkeypatch.setattr(comparer, "_compute_similarity_matrices",
                        lambda profiles: {"combined": {"jaccard": None}})
    monkeypatch.setattr(comparer, "_compute_bodyid_similarity_matrices",
                        lambda *a: (_ for _ in ()).throw(AssertionError(
                            "the bodyId pair loop must not re-score the main matrices")))
    monkeypatch.setattr(comparer, "_compute_type_avg_bodyid_matrices",
                        lambda *a: (_ for _ in ()).throw(AssertionError(
                            "the type average must be folded from existing scores")))
    monkeypatch.setattr(comparer, "_type_avg_from_pair_matrices",
                        lambda pair_matrices, bodyid: {"combined": {"jaccard": 1.0}})
    saved = {}

    def _save(type_profiles, bodyid_profiles, type_matrices,
              bodyid_matrices, type_avg_matrices):
        saved.update(type_matrices=type_matrices,
                     bodyid_matrices=bodyid_matrices,
                     type_avg_matrices=type_avg_matrices)
        return {"output_path": "/tmp/x", "matrices_saved": []}

    monkeypatch.setattr(comparer, "_save_results", _save)

    result = comparer.run()
    assert saved["type_matrices"] == {}
    assert saved["bodyid_matrices"] == {"combined": {"jaccard": None}}
    assert saved["type_avg_matrices"] == {"combined": {"jaccard": 1.0}}
    assert result["bodyid_level_skipped"] is False


def test_run_type_aggregation_still_computes_bodyid_matrices(monkeypatch):
    comparer = _make_comparer(query=["Mi1", "Tm3"], aggregation_level="type")

    profile = object()
    monkeypatch.setattr(comparer, "_extract_all_profiles",
                        lambda: ({"Mi1": profile, "Tm3": profile},
                                 {("Mi1", 1): profile, ("Tm3", 2): profile}))
    monkeypatch.setattr(comparer, "_compute_similarity_matrices",
                        lambda profiles: {"combined": {"jaccard": None}})
    monkeypatch.setattr(comparer, "_compute_bodyid_similarity_matrices",
                        lambda profiles: {"combined": {"jaccard": None}})
    monkeypatch.setattr(comparer, "_compute_type_avg_bodyid_matrices",
                        lambda profiles: {"combined": {"jaccard": None}})
    monkeypatch.setattr(comparer, "_save_results",
                        lambda *a, **k: {"output_path": "/tmp/x", "matrices_saved": []})

    result = comparer.run()
    assert result["bodyid_level_skipped"] is False


def test_run_single_row_keeps_the_bodyid_pass_despite_the_cost_skip(monkeypatch):
    """skip_bodyId_level cannot leave a one-row run with nothing but that row
    pooled against itself (1.0 on every metric)."""
    comparer = _make_comparer(query=["Mi1"], aggregation_level="type",
                              skip_bodyId_level=True)

    profile = object()
    monkeypatch.setattr(comparer, "_extract_all_profiles",
                        lambda: ({"Mi1": profile},
                                 {("Mi1", 1): profile, ("Mi1", 2): profile}))
    called = []
    monkeypatch.setattr(comparer, "_compute_similarity_matrices",
                        lambda profiles: {"combined": {"jaccard": None}})
    monkeypatch.setattr(comparer, "_compute_bodyid_similarity_matrices",
                        lambda profiles: called.append("bodyid") or {"combined": {}})
    monkeypatch.setattr(comparer, "_compute_type_avg_bodyid_matrices",
                        lambda profiles: called.append("type_avg") or {"combined": {}})
    monkeypatch.setattr(comparer, "_save_results",
                        lambda *a, **k: {"output_path": "/tmp/x", "matrices_saved": []})

    result = comparer.run()
    assert called == ["bodyid", "type_avg"]
    assert result["bodyid_level_skipped"] is False


def test_single_row_keeps_the_cost_skip_beyond_the_bodyid_budget(monkeypatch):
    """The override saves a run from a meaningless 1.0 cell; it must not burn
    hours doing it, so past the 1000-bodyId budget the auto skip stands."""
    comparer = _make_comparer(query=["Mi1"], aggregation_level="type")

    profile = object()
    monkeypatch.setattr(comparer, "_extract_all_profiles",
                        lambda: ({"Mi1": profile},
                                 {("Mi1", i): profile for i in range(1001)}))
    called = []
    monkeypatch.setattr(comparer, "_compute_similarity_matrices",
                        lambda profiles: {"combined": {"jaccard": None}})
    monkeypatch.setattr(comparer, "_compute_bodyid_similarity_matrices",
                        lambda profiles: called.append("bodyid") or {})
    monkeypatch.setattr(comparer, "_compute_type_avg_bodyid_matrices",
                        lambda profiles: called.append("type_avg") or {})
    monkeypatch.setattr(comparer, "_save_results",
                        lambda *a, **k: {"output_path": "/tmp/x", "matrices_saved": []})

    result = comparer.run()
    assert called == []
    assert result["bodyid_level_skipped"] is True


def test_run_refuses_when_fewer_than_two_neurons_are_in_scope(monkeypatch):
    """One neuron is not a comparison — and the refusal must raise, because the
    runner discards the return dict and would report Completed with no files."""
    comparer = _make_comparer(query=["Mi1"], aggregation_level="bodyid")
    profile = object()
    monkeypatch.setattr(comparer, "_extract_all_profiles",
                        lambda: ({"1_Mi1": profile}, {("Mi1", 1): profile}))
    with pytest.raises(ValueError, match="at least two neurons"):
        comparer.run()


# ---------------------------------------------------------------------------
# Shared LabelMapper group parser (used by BOTH comparison tabs)
# ---------------------------------------------------------------------------
def test_load_labelmapper_source_groups_returns_the_raw_side(tmp_path):
    """The shared function returns (groups, names, side), so a caller that needs
    another dataset's column does not re-read the file."""
    from utils.naming_utils import load_labelmapper_source_groups

    path = tmp_path / "preset.json"
    path.write_text(json.dumps({"source_mapping": {
        "custom_label": ["g1", "g2", "g3"],
        DATASET: [["Mi1", "Mi2"], [], ["7"]],
        "other:v1": [["X"]],
    }}), encoding="utf-8")

    groups, names, side = load_labelmapper_source_groups(str(path), DATASET)
    assert names == ["g1", "g3"]
    assert groups == [["Mi1", "Mi2"], [7]]          # digit strings become ints
    assert side["custom_label"] == ["g1", "g2", "g3"]
    assert "other:v1" in side


def test_load_labelmapper_source_groups_degrades_like_the_method(tmp_path):
    """Empty cases return the empty triple, warning through the injected log —
    the two comparers both treat 'no groups here' as warn-and-continue."""
    from utils.naming_utils import load_labelmapper_source_groups

    logs = []
    no_labels = tmp_path / "a.json"
    no_labels.write_text(json.dumps({"source_mapping": {DATASET: [["x"]]}}))
    assert load_labelmapper_source_groups(
        str(no_labels), DATASET, log=logs.append) == ([], [], {DATASET: [["x"]]})

    other_ds = tmp_path / "b.json"
    other_ds.write_text(json.dumps({"source_mapping": {
        "custom_label": ["g"], "nope:v1": [["x"]]}}))
    groups, names, _side = load_labelmapper_source_groups(
        str(other_ds), DATASET, log=logs.append)
    assert (groups, names) == ([], [])

    unreadable = tmp_path / "missing.json"
    with pytest.raises(ValueError):
        load_labelmapper_source_groups(str(unreadable), DATASET)


def test_connectivity_delegate_matches_the_shared_function(tmp_path):
    """The profiling method is a delegate: identical (groups, names)."""
    from utils.naming_utils import load_labelmapper_source_groups

    path = tmp_path / "preset.json"
    path.write_text(json.dumps({"source_mapping": {
        "custom_label": ["g1", "g2"],
        DATASET.replace(':', '_').replace('.', '_'): [["Mi1"], ["5"]],
    }}), encoding="utf-8")
    comparer = _make_comparer(query=["Mi1"], aggregation_level="custom")
    via_method = comparer._load_custom_groups_from_mapping(str(path), DATASET)
    shared = load_labelmapper_source_groups(str(path), DATASET)[:2]
    assert via_method == tuple(shared) == ([["Mi1"], [5]], ["g1", "g2"])


def test_get_types_for_bodyids_maps_untyped_table_rows_to_none(
        tmp_path, monkeypatch):
    """The local-table path must honour its own docstring ('type name or
    None'): a missing annotation reaches the table as pandas NaN and used
    to stringify to the literal 'nan', which is truthy and defeated every
    consumer's `or` guard (user 2026-09-27: 54/370 target_matches rows)."""
    import comparison.connectivity_profiler as cpmod

    table = tmp_path / "a" / "b" / "c"  # stands in for the profiler file:
    # src_dir = table.parent.parent = tmp/'a', project_root = tmp_path
    safe = (cpmod.canonical_dataset_name("probe-ds")
            .replace(':', '_').replace('.', '_'))
    folder = tmp_path / "datasets" / safe
    folder.mkdir(parents=True)
    pd.DataFrame({
        "bodyId": ["1", "2", "3", "4"],
        # empty cell -> NaN; literal 'nan' spelling; sentinel 'Unknown'
        "type": ["CL125", "", "nan", "Unknown"],
    }).to_csv(folder / f"{safe}_neurons.csv", index=False)

    monkeypatch.setattr(cpmod, "Path", lambda p=None: table)
    prof = ConnectivityProfiler.__new__(ConnectivityProfiler)
    prof._has_local_table = lambda ds: True

    got = prof.get_types_for_bodyids([1, 2, 3, 4], "probe-ds")
    assert got[1] == "CL125"
    assert got[2] is None  # NaN annotation -> None, never the string 'nan'
    assert got[3] is None  # literal 'nan' spelling
    assert got[4] is None  # 'Unknown' sentinel


def test_get_types_for_bodyids_neuprint_nan_type_also_none(monkeypatch):
    """The NeuPrint path leaks the same way the local table did: a missing
    type property reaches fetch_custom's DataFrame as pandas NaN, which is
    TRUTHY, so `str(ntype) if ntype else None` published the literal
    'nan' (review find 2026-09-27)."""
    import comparison.connectivity_profiler as cpmod

    monkeypatch.setattr(cpmod, "is_local_connectome_dataset",
                        lambda ds: False)
    prof = ConnectivityProfiler.__new__(ConnectivityProfiler)
    prof._has_local_table = lambda ds: False

    class FakeClient:
        def fetch_custom(self, query):
            # None in the frame mimics an absent type property after the
            # NeuPrint -> DataFrame hop (pandas turns it into NaN)
            return pd.DataFrame({"bodyId": [7, 8],
                                 "type": [None, "Mi1"]})

    prof._get_client_for_dataset = lambda ds: FakeClient()
    got = prof.get_types_for_bodyids([7, 8], "hemibrain:v1.2.1")
    assert got[7] is None  # NaN type -> None, never the string 'nan'
    assert got[8] == "Mi1"
