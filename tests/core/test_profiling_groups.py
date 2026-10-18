"""Connectivity Profiling custom grouping and aggregation levels.

The profiling tab drives ``ConnectivityProfileComparer`` (profile_comparator).
Its ``aggregation_level`` must now be functional:
- 'type' (default): pattern query items ('aMe.*') expand into their matched
  types, each an independent row
- 'bodyid': every individual neuron is its own row ({bodyId}_{type})
- 'custom': rows are user-defined custom groups from a LabelMapper preset
  (custom_mapping_file) or a nested-list query
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
from comparison.label_mapper import PRESET_FORMAT_TAG  # noqa: E402
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
        "format": PRESET_FORMAT_TAG,
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


# ---------------------------------------------------------------------------
# Tier-0 user label mapper (label_mapping_file) — type/bodyid modes
# ---------------------------------------------------------------------------

def _make_profile(up=None, down=None):
    from comparison.connectivity_profiler import (
        ConnectivityProfile, compute_ranks)
    up, down = dict(up or {}), dict(down or {})
    return ConnectivityProfile(
        neuron_id=1, dataset=DATASET,
        upstream_partners=up, downstream_partners=down,
        upstream_ranks=compute_ranks(up), downstream_ranks=compute_ranks(down),
        upstream_top_k=len(up), downstream_top_k=len(down),
        total_upstream_weight=sum(up.values()),
        total_downstream_weight=sum(down.values()),
        actual_upstream_count=len(up), actual_downstream_count=len(down),
        unique_types_upstream=len(up), unique_types_downstream=len(down))


def test_label_mapping_file_does_not_force_custom_aggregation(tmp_path):
    """U1: the tier-0 mapper overlay never changes the rows — only
    custom_mapping_file forces aggregation_level='custom'."""
    mapping_path = _write_mapping(tmp_path)
    comparer = _make_comparer(query=["grp1", "Mi1"],
                              label_mapping_file=mapping_path)
    assert comparer.aggregation_level == "type"
    assert comparer._custom_group_names is None
    assert comparer._user_label_mapper is not None
    assert "grp1" in comparer._user_label_governing_labels()


def test_partner_fold_applies_whether_or_not_a_preset_forces_custom(tmp_path):
    """The control's hint once read 'Type/bodyid rows only', but the board's
    preset forces aggregation_level='custom' — so the fold has to survive that,
    while the overlay path leaves the level at type. Both are the contract the
    hint now states."""
    mapping_path = _write_mapping(tmp_path)
    forced = _make_comparer(query=["grp1"], custom_mapping_file=mapping_path,
                            group_partners_by_mapping=True)
    assert forced.aggregation_level == "custom"
    assert forced.group_partners_by_mapping is True
    overlay = _make_comparer(query=["grp1", "Mi1"],
                             label_mapping_file=mapping_path,
                             group_partners_by_mapping=True)
    assert overlay.aggregation_level == "type"
    assert overlay.group_partners_by_mapping is True


def test_type_mode_user_label_token_expands_to_pooled_row(tmp_path):
    """U2: a query chip naming a user label pools that label's members for
    the dataset into one row, recorded as a tier-0 resolution."""
    mapping_path = _write_mapping(tmp_path)
    comparer = _make_comparer(query=["grp1", "Mi1"],
                              label_mapping_file=mapping_path)
    neurons = comparer._get_neurons_to_compare()
    assert neurons["grp1"] == ["aMe12", "aMe12_R"]
    assert neurons["Mi1"] == [1]
    assert comparer.mapping_resolution_counts.get("user_label", 0) >= 1
    record = comparer.mapping_resolution_record[DATASET]["grp1"]
    assert record["status"] == "user_label"
    assert record["targets"] == ["grp1"]


def test_type_mode_label_row_survives_generic_resolution(tmp_path):
    """The generic union lookup must not overwrite the tier-0 row with the
    raw literal when the label matches no type."""
    mapping_path = _write_mapping(tmp_path)
    comparer = _make_comparer(query=["grp3", "Mi1"],
                              label_mapping_file=mapping_path)
    # grp3 has no members for this dataset in the preset — but grp1 does;
    # use a label whose members resolve while its name matches no type.
    comparer = _make_comparer(query=["grp1"], label_mapping_file=mapping_path)
    neurons = comparer._get_neurons_to_compare()
    assert neurons == {"grp1": ["aMe12", "aMe12_R"]}


def test_bodyid_mode_user_label_token_rows_per_member(tmp_path):
    """U3: bodyid mode rows a label's members individually; type-name
    members expand to their bodyIds."""
    mapping_path = _write_mapping(tmp_path)
    comparer = _make_comparer(query=["grp1"], aggregation_level="bodyid",
                              label_mapping_file=mapping_path)
    comparer.profiler.get_types_for_bodyids = lambda bids, ds: {}
    comparer.profiler.get_bodyids_for_type = (
        lambda t, ds: {"aMe12": [1], "aMe12_R": [2]}.get(t, []))
    neurons = comparer._get_neurons_to_compare()
    assert neurons == {"1_aMe12": [1], "2_aMe12_R": [2]}


def test_canonical_label_tier0_verbatim(tmp_path):
    """U4: a user label keys verbatim even when a type mapper is present
    (the fake mapper would crash the branch if tier-0 were bypassed)."""
    mapping_path = _write_mapping(tmp_path)
    comparer = _make_comparer(label_mapping_file=mapping_path)
    comparer._type_mapper = object()  # would explode if consulted
    assert comparer._canonical_label("grp1", DATASET) == "grp1"
    # mapper off + non-label keeps the raw name
    comparer2 = _make_comparer(label_mapping_file=mapping_path)
    comparer2._type_mapper = None
    assert comparer2._canonical_label("Mi1", DATASET) == "Mi1"


def test_licensed_anchor_label_returns_user_label(tmp_path):
    """U5: a user label anchors as itself in every dataset."""
    mapping_path = _write_mapping(tmp_path)
    comparer = _make_comparer(label_mapping_file=mapping_path)
    comparer._type_mapper = object()  # would explode if consulted
    assert comparer._licensed_anchor_label("grp1", DATASET) == "grp1"


def test_standardize_profile_protects_user_label_keys(tmp_path):
    """U6: tier-0 partner keys pass canonicalization verbatim while the
    remaining keys are standardized."""
    from comparison.profile_comparator import ProfileComparator

    mapping_path = _write_mapping(tmp_path)
    comparer = _make_comparer(label_mapping_file=mapping_path)

    class FakeMapper:
        @staticmethod
        def standardize_partner_types(partners, ds):
            return {f"canon_{k}": v for k, v in partners.items()}

    profile = _make_profile(up={"grp1": 5.0, "Tm3": 3.0})
    out = ProfileComparator.standardize_profile(
        profile, DATASET, FakeMapper(),
        protected=comparer._user_label_governing_labels())
    assert out.upstream_partners["grp1"] == 5.0
    assert out.upstream_partners["canon_Tm3"] == 3.0
    assert "Tm3" not in out.upstream_partners
    # without protection the old behavior applies (everything canonicalized)
    plain = ProfileComparator.standardize_profile(profile, DATASET, FakeMapper())
    assert plain.upstream_partners == {"canon_grp1": 5.0, "canon_Tm3": 3.0}


def test_a_T_star_query_key_selects_the_residual_class(tmp_path):
    """U10: a `T*` key copied out of an export names a CLASS, not a label.

    The preset enrolls bodyId 1 of native type aMe12 (neurons 1,2), so
    ``aMe12*`` is that type's unenrolled half — neuron 2. Before this branch the
    token was claimed as a tier-0 identity, ``get_neurons_for_label('aMe12*')``
    answered nothing, and the row silently disappeared
    (_plan/pending_issues.md, "a `T*` key cannot be named in a query").
    """
    preset = tmp_path / "residual_preset.json"
    preset.write_text(json.dumps({
        "format": PRESET_FORMAT_TAG,
        "source_mapping": {"custom_label": ["clock"], DATASET: [[1]]},
    }), encoding="utf-8")
    comparer = _make_comparer(query=["aMe12*"],
                              label_mapping_file=str(preset))
    comparer.profiler.get_bodyids_for_type = lambda t, ds: {
        "aMe12": [1, 2]}.get(t, [])
    assert comparer._get_neurons_to_compare(aggregation_level="type") == {
        "aMe12*": [2]}
    # the resolution record says which class the key resolved to
    record = comparer.mapping_resolution_record[DATASET]["aMe12*"]
    assert record["members"] == ["2"] and "residual class" in record["reason"]


def test_an_untouched_type_has_no_residual_class_to_query(tmp_path):
    """Collision is dataset-scoped: with nothing enrolled the key names nothing.

    A `T*` spelling of a type the preset does not govern at all is NOT that
    type's population — an untouched native type keeps its raw name — so the
    lane must not invent a row of neurons under a star the export never printed.
    Declined, the token falls to the ordinary lanes, which keep it as the raw
    item they keep any unresolved pattern.
    """
    preset = tmp_path / "residual_preset2.json"
    preset.write_text(json.dumps({
        "format": PRESET_FORMAT_TAG,
        "source_mapping": {"custom_label": ["clock"], DATASET: [["OTHER"]]},
    }), encoding="utf-8")
    comparer = _make_comparer(query=["aMe12*"],
                              label_mapping_file=str(preset))
    comparer.profiler.get_bodyids_for_type = lambda t, ds: {
        "aMe12": [2, 3]}.get(t, [])
    neurons = comparer._get_neurons_to_compare(aggregation_level="type")
    flat = [bid for ids in neurons.values() for bid in ids]
    assert 2 not in flat and 3 not in flat, (
        f"an untouched type was rowed under a star: {neurons}")


def test_a_star_that_is_a_real_type_name_is_still_the_type(tmp_path):
    """A dataset type spelled with a star (FAFB's `PPL1*`) is not a class.

    The tier-0 claim used to swallow it: the token reads as a terminal identity
    by spelling, `get_neurons_for_label` answers nothing, and the row of a valid
    type query disappears. Declining it leaves the type lanes to resolve it.
    """
    preset = tmp_path / "residual_preset3.json"
    preset.write_text(json.dumps({
        "format": PRESET_FORMAT_TAG,
        "source_mapping": {"custom_label": ["clock"], DATASET: [[1]]},
    }), encoding="utf-8")
    comparer = _make_comparer(query=["PPL1*"],
                              label_mapping_file=str(preset))
    comparer.profiler.get_bodyids_for_type = lambda t, ds: {
        "PPL1*": [7, 8], "PPL1": [1, 2]}.get(t, [])
    comparer.profiler.list_types = lambda p, ds: (
        ["PPL1*"] if p == "PPL1*" else [])
    assert comparer._get_neurons_to_compare(aggregation_level="type") == {
        "PPL1*": [7, 8]}


def test_collapse_partners_sums_member_weights(tmp_path):
    """U7: partner-side grouping folds member types into their label with
    summed weights; unmapped keys and true labels keep their keys."""
    mapping_path = _write_mapping(tmp_path)
    comparer = _make_comparer(label_mapping_file=mapping_path)
    profile = _make_profile(up={"aMe12": 2.0, "aMe12_R": 3.0, "Tm3": 1.0})
    collapsed = comparer.collapse_partners_by_label_mapper({"grp2": profile},
                                                           DATASET)
    out = collapsed["grp2"]
    assert out.upstream_partners == {"grp1": 5.0, "Tm3": 1.0}
    assert out.total_upstream_weight == 6.0
    # ranks rebuilt from the collapsed weights
    assert out.upstream_ranks == {"grp1": 1, "Tm3": 2}


def test_collapse_reports_the_number_of_folded_partners(tmp_path):
    """The folding disclosure counts what disappeared (pre - post), so a run
    never logs a negative number of folded partner entries."""
    mapping_path = _write_mapping(tmp_path)
    comparer = _make_comparer(label_mapping_file=mapping_path)
    lines: list = []
    comparer._log = lambda msg: lines.append(str(msg))
    profile = _make_profile(up={"aMe12": 2.0, "aMe12_R": 3.0, "Tm3": 1.0})
    comparer.collapse_partners_by_label_mapper({"grp2": profile}, DATASET)
    # three partner keys became two (aMe12 + aMe12_R -> grp1) => 1 folded
    assert lines and "1 partner entr(ies) folded into group labels" in lines[-1]


def test_collapse_without_mapper_is_identity():
    """U9-lite: no mapper -> the profiles pass through untouched."""
    comparer = _make_comparer()
    assert comparer._user_label_mapper is None
    sentinel = {"row": object()}
    assert comparer.collapse_partners_by_label_mapper(
        sentinel, DATASET) is sentinel


def test_group_partners_requires_a_mapper():
    """The toggle is silently dropped (with a log) when no mapping was
    provided — a run must not claim grouping it cannot apply."""
    comparer = _make_comparer(group_partners_by_mapping=True)
    assert comparer.group_partners_by_mapping is False


def test_preset_without_format_tag_refuses_the_run(tmp_path):
    """F4: a preset declaring no grammar is refused, not parsed on shape — and
    the profiling comparer does NOT swallow that into a raw-name run (which
    would publish labels it never applied)."""
    from comparison.label_mapper import PresetFormatError

    untagged = tmp_path / "untagged.json"
    untagged.write_text(json.dumps({"source_mapping": {
        "custom_label": ["grp1"], DATASET: [["aMe12"]],
    }}), encoding="utf-8")
    with pytest.raises(PresetFormatError, match="no 'format' key"):
        _make_comparer(query=["grp1"], label_mapping_file=str(untagged))

    # A genuinely broken file still degrades to raw names with a warning.
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    degraded = _make_comparer(query=["grp1"], label_mapping_file=str(broken))
    assert degraded._user_label_mapper is None


def test_anchor_profiles_align_by_user_label(tmp_path):
    """U11: type-mode user labels anchor across datasets by the label
    itself (each dataset's profile row IS the label)."""
    mapping_path = _write_mapping(tmp_path)
    comparer = _make_comparer(query=["grp1"], datasets=[DATASET, "banc_v626"],
                              label_mapping_file=mapping_path)
    profiles_by_dataset = {
        DATASET: {"grp1": _make_profile(up={"aMe12": 4.0})},
        "banc_v626": {"grp1": _make_profile(up={"R1": 9.0})},
    }
    anchors = comparer._build_anchor_profiles(profiles_by_dataset)
    assert set(anchors) == {"grp1"}
    assert set(anchors["grp1"]) == {DATASET, "banc_v626"}


def _write_metachar_mapping(tmp_path):
    """Preset whose labels contain regex metacharacters.

    The spellings are the clock set's (``LNd_CRY+``, ``LNd_CRY+_ITP+``,
    ``LNd_CRY-``) plus a parenthesised variant, so every entry is a literal
    that ``_looks_like_pattern`` would route to the regex branch."""
    preset = {
        "format": PRESET_FORMAT_TAG,
        "source_mapping": {
            "custom_label": ["LNd_CRY+", "LNd_CRY+_ITP+", "LNd_CRY-",
                             "DN1pA(II)"],
            DATASET: [["1"], ["2"], ["3"], ["4"]],
        }
    }
    path = tmp_path / "metachar_mapping.json"
    path.write_text(json.dumps(preset), encoding="utf-8")
    return str(path)


def _two_dataset_profiles(labels):
    return {ds: {lbl: _make_profile(up={ds: 1.0}) for lbl in labels}
            for ds in (DATASET, "banc_v626")}


def test_anchor_profile_tier0_metachar_label_never_hits_the_pool(tmp_path):
    """U11b: a tier-0 label IS the comparison identity — a label containing
    ``+`` must anchor as itself, and the dataset type pool may not be
    consulted for it at all."""
    comparer = _make_comparer(
        query=["LNd_CRY+", "LNd_CRY+_ITP+"], datasets=[DATASET, "banc_v626"],
        label_mapping_file=_write_metachar_mapping(tmp_path))

    def boom(pattern, ds):
        raise AssertionError(
            f"tier-0 label pattern-matched: {pattern!r} in {ds}")

    comparer.profiler.list_types = boom
    anchors = comparer._build_anchor_profiles(
        _two_dataset_profiles(["LNd_CRY+", "LNd_CRY+_ITP+", "LNd_CRY-"]))
    assert set(anchors) == {"LNd_CRY+", "LNd_CRY+_ITP+"}
    assert set(anchors["LNd_CRY+_ITP+"]) == {DATASET, "banc_v626"}


def test_anchor_profile_tier0_label_does_not_leak_its_regex_family(tmp_path):
    """U11c: querying one label publishes ONE anchor — not the family a
    regex expansion of its ``+`` spelling would have matched."""
    comparer = _make_comparer(
        query=["LNd_CRY+"], datasets=[DATASET, "banc_v626"],
        label_mapping_file=_write_metachar_mapping(tmp_path))
    comparer.profiler.list_types = (
        lambda p, ds: ["LNd_CRY+", "LNd_CRY+_ITP+", "LNd_CRY-"]
        if p == "LNd_CRY+" else [])
    anchors = comparer._build_anchor_profiles(
        _two_dataset_profiles(["LNd_CRY+", "LNd_CRY+_ITP+", "LNd_CRY-"]))
    assert set(anchors) == {"LNd_CRY+"}


def test_anchor_profile_exact_type_with_metachar_beats_the_pool(tmp_path):
    """U11d: even without a preset, an exact type name that happens to
    contain ``( resolves exactly — _looks_like_pattern's documented order.
    """
    comparer = _make_comparer(query=["DN1pA(II)"],
                              datasets=[DATASET, "banc_v626"])
    comparer.profiler.list_types = (
        lambda p, ds: ["DN1pA(II)R", "DN1pA(II)L"])
    anchors = comparer._build_anchor_profiles(
        _two_dataset_profiles(["DN1pA(II)", "DN1pA(II)R", "DN1pA(II)L"]))
    assert set(anchors) == {"DN1pA(II)"}


def test_anchor_profile_dot_bearing_label_resolves_exactly(tmp_path):
    """U11d2: a label containing '.' (a version-suffix spelling) is a literal
    too — '.' as a regex wildcard must not pull in same-length neighbours."""
    preset = tmp_path / "dot_preset.json"
    preset.write_text(json.dumps({
        "format": PRESET_FORMAT_TAG,
        "source_mapping": {
            "custom_label": ["DN1pA.1", "DN1pAxl"],
            DATASET: [["1"], ["2"]],
        }
    }), encoding="utf-8")
    comparer = _make_comparer(query=["DN1pA.1"],
                              datasets=[DATASET, "banc_v626"],
                              label_mapping_file=str(preset))

    def boom(pattern, ds):
        raise AssertionError(
            f"tier-0 label pattern-matched: {pattern!r} in {ds}")

    comparer.profiler.list_types = boom
    # 'DN1pAxl' is what re.match('DN1pA.1') would have matched as a sibling.
    anchors = comparer._build_anchor_profiles(
        _two_dataset_profiles(["DN1pA.1", "DN1pAxl"]))
    assert set(anchors) == {"DN1pA.1"}


def test_anchor_profile_pattern_item_still_expands(tmp_path):
    """U11e: the exact-first reorder must not kill pattern support — an item
    with no exact row still expands through the dataset's type pool."""
    comparer = _make_comparer(query=["CRY.*"],
                              datasets=[DATASET, "banc_v626"])
    comparer.profiler.list_types = (
        lambda p, ds: ["LNd_CRY+", "LNd_CRY-"] if p == "CRY.*" else [])
    anchors = comparer._build_anchor_profiles(
        _two_dataset_profiles(["LNd_CRY+", "LNd_CRY-"]))
    assert set(anchors) == {"LNd_CRY+", "LNd_CRY-"}


def test_anchor_profile_plain_type_name_unchanged_without_preset(tmp_path):
    """U11f: the no-metacharacter, no-preset baseline stays byte-identical."""
    comparer = _make_comparer(query=["Mi1", "Tm3"],
                              datasets=[DATASET, "banc_v626"])
    anchors = comparer._build_anchor_profiles(
        _two_dataset_profiles(["Mi1", "Tm3"]))
    assert set(anchors) == {"Mi1", "Tm3"}
    assert comparer.inter_anchor_dropped == []


def test_dropped_anchor_listing_is_capped_but_not_in_json(tmp_path):
    """U11j: the log/README name a bounded number of drops; the full record
    stays in parameters.json (a long-tail type-level run can lose hundreds)."""
    comparer = _make_comparer(query=[f"T{i}" for i in range(25)],
                              datasets=[DATASET, "banc_v626"])
    assert comparer._build_anchor_profiles(
        {DATASET: {}, "banc_v626": {}}) == {}
    assert len(comparer.inter_anchor_dropped) == 25
    lines = comparer._dropped_anchor_lines()
    assert len(lines) == 21                       # 20 named + the summary
    assert "and 5 more" in lines[-1]
    assert "dropped_inter_anchors" in lines[-1]
    # a short list is not padded with a summary line
    other = _make_comparer(query=["Mi1", "Tm3"],
                           datasets=[DATASET, "banc_v626"])
    other._build_anchor_profiles({DATASET: {"Mi1": _make_profile()},
                                  "banc_v626": {}})
    assert len(other._dropped_anchor_lines()) == len(other.inter_anchor_dropped)


def test_anchor_profile_records_dropped_anchor(tmp_path):
    """U11g: an item with a row in only ONE dataset drops — legitimately, but
    never silently: the drop names the item, its kind and the reason."""
    comparer = _make_comparer(query=["Mi1", "Tm3"],
                              datasets=[DATASET, "banc_v626"])
    profiles = {
        DATASET: {"Mi1": _make_profile(up={"a": 1.0}),
                  "Tm3": _make_profile(up={"a": 1.0})},
        "banc_v626": {"Mi1": _make_profile(up={"b": 1.0})},
    }
    anchors = comparer._build_anchor_profiles(profiles)
    assert set(anchors) == {"Mi1"}
    assert [(d["item"], d["kind"]) for d in comparer.inter_anchor_dropped] \
        == [("Tm3", "type name")]
    dropped = comparer.inter_anchor_dropped[0]
    assert dropped["datasets_with_anchor_row"] == [DATASET]
    assert "1 of 2" in dropped["reason"]


def test_anchor_profile_tier0_label_without_rows_is_disclosed(tmp_path):
    """U11h: a tier-0 label that anchors nothing is reported as a tier-0
    drop, and the type pool is still never consulted for it."""
    comparer = _make_comparer(
        query=["LNd_CRY+"], datasets=[DATASET, "banc_v626"],
        label_mapping_file=_write_metachar_mapping(tmp_path))

    def boom(pattern, ds):
        raise AssertionError(
            f"tier-0 label pattern-matched: {pattern!r} in {ds}")

    comparer.profiler.list_types = boom
    assert comparer._build_anchor_profiles(
        {DATASET: {}, "banc_v626": {}}) == {}
    dropped = comparer.inter_anchor_dropped[0]
    assert dropped["kind"] == "user label (tier 0)"
    assert dropped["datasets_with_anchor_row"] == []


def test_anchor_profile_dropped_pattern_names_the_pool_size(tmp_path):
    """U11i: a pattern that matches nothing is disclosed as a pattern drop
    with the dataset whose pool was searched."""
    comparer = _make_comparer(query=["CRY.*"],
                              datasets=[DATASET, "banc_v626"])
    comparer.profiler.list_types = lambda p, ds: []
    assert comparer._build_anchor_profiles(
        _two_dataset_profiles(["Mi1"])) == {}
    dropped = comparer.inter_anchor_dropped[0]
    assert dropped["kind"] == "pattern"
    assert f"matched no type in {DATASET}" in dropped["reason"]


def test_collapse_bodyid_keyed_map_type_bridge(tmp_path):
    """U7b: bodyId-keyed maps bridge partner TYPE names through the type's
    bodyIds — all resolved members must agree on one label."""
    import json as _json
    bodyid_map = tmp_path / "bodyid_map.json"
    bodyid_map.write_text(_json.dumps({
        "format": PRESET_FORMAT_TAG,
        "source_mapping": {
            "custom_label": ["grpA", "grpB"],
            DATASET: [["10", "11"], ["20"]],
        }
    }), encoding="utf-8")
    comparer = _make_comparer(label_mapping_file=str(bodyid_map))
    comparer.profiler.get_bodyids_for_type = (
        lambda t, ds: {"Tm3": [10, 11], "Mi1": [20], "Split": [10, 20]}.get(t, []))
    profile = _make_profile(up={"Tm3": 4.0, "Mi1": 2.0, "Split": 3.0})
    collapsed = comparer.collapse_partners_by_label_mapper({"row": profile},
                                                           DATASET)
    out = collapsed["row"].upstream_partners
    assert out["grpA"] == 4.0          # Tm3's bodyIds {10,11} -> all grpA
    assert out["grpB"] == 2.0          # Mi1's bodyId 20 -> grpB
    assert out["Split"] == 3.0         # spans grpA+grpB -> stays raw
    assert "Tm3" not in out and "Mi1" not in out


def test_collapse_label_keyed_partner_keeps_its_own_key(tmp_path):
    """A partner key that IS a label never folds into another label."""
    mapping_path = _write_mapping(tmp_path)
    comparer = _make_comparer(label_mapping_file=mapping_path)
    profile = _make_profile(up={"grp1": 5.0})
    collapsed = comparer.collapse_partners_by_label_mapper({"grp2": profile},
                                                           DATASET)
    assert collapsed["grp2"].upstream_partners == {"grp1": 5.0}


def test_bodyid_axes_carry_standard_labels_with_preset(tmp_path):
    """F2 fix: with a preset loaded, bodyId-level axes upgrade to
    {bodyId}_{label}({native}); without one they stay legacy."""
    import json as _json
    preset = tmp_path / "axis_map.json"
    preset.write_text(_json.dumps({
        "format": PRESET_FORMAT_TAG,
        "source_mapping": {
            "custom_label": ["grpM"],
            DATASET: [["1"]],
        }
    }), encoding="utf-8")
    labeled = _make_comparer(query=["Mi1"], aggregation_level="bodyid",
                             label_mapping_file=str(preset))
    neurons = labeled._get_neurons_to_compare()
    assert neurons == {"1_grpM(Mi1)": [1]}
    # without the preset the legacy axis is unchanged (compatibility gate)
    legacy = _make_comparer(query=["Mi1"], aggregation_level="bodyid")
    assert legacy._get_neurons_to_compare() == {"1_Mi1": [1]}


def test_connectionless_profile_is_dropped_whatever_the_batch_size(monkeypatch):
    """A neuron with no connections has no profile to compare.

    The >100-neuron batch path returns None for it; the per-neuron path used
    to return an all-zero profile, which entered the bodyId matrix as a row
    of zeros. Both paths must drop it, or a standalone bodyId comparison's
    axes and means depend on how many neurons happened to be in the query.
    """
    from comparison.connectivity_profiler import ConnectivityProfile

    comparer = _make_comparer(query=["Mi1", "Tm3"])
    monkeypatch.setattr(
        comparer, "_get_neurons_to_compare",
        lambda **kwargs: {"Mi1": [1, 2], "Tm3": [3]})
    live = ConnectivityProfile(
        neuron_id=1, dataset=DATASET,
        upstream_partners={"A": 5.0}, downstream_partners={"B": 3.0})
    connectionless = ConnectivityProfile(neuron_id=2, dataset=DATASET)
    other = ConnectivityProfile(
        neuron_id=3, dataset=DATASET,
        upstream_partners={"A": 2.0}, downstream_partners={"C": 1.0})
    monkeypatch.setattr(
        comparer.profiler, "get_profiles_batch",
        lambda ids, ds, **kwargs: {1: live, 2: connectionless, 3: other})
    monkeypatch.setattr(comparer.profiler, "_get_cached_conn_df",
                        lambda ds: None)

    type_profiles, bodyid_profiles = comparer._extract_profiles_for_dataset(
        DATASET)

    assert ("Mi1", 2) not in bodyid_profiles
    assert ("Mi1", 1) in bodyid_profiles
    assert ("Tm3", 3) in bodyid_profiles
    assert "Mi1" in type_profiles and "Tm3" in type_profiles
