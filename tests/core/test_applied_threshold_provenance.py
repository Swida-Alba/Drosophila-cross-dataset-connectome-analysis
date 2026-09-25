"""Plan 2026-09-07 Phase B: canonical applied-threshold/bottleneck
provenance.

Covers the shared ``applied_threshold_provenance`` formula (complete,
StrongestFirst-bitten, Edge-Budget-floored, and combined runs), the
instance-level finalization (all_attributes.json fields +
data_details/parameters.csv rows), the parameters.txt provenance block with
its backward-compatible aliases, and the shortest-mode contract (StrongestFirst
budget metadata exported, Edge Budget never applied).
"""

import json
import os

import pandas as pd
import polars as pl
import pytest

import coana
from coana import applied_threshold_provenance
from utils.threshold_state import applied_threshold_provenance as shared

from tests.core.test_pathfinding import _make_pipeline_fc


def test_coana_reexports_shared_formula():
    """The canonical formula lives in utils.threshold_state; the coana
    import path is a re-export of the SAME object (single source)."""
    assert coana.applied_threshold_provenance is shared


# ---------------------------------------------------------------------------
# ComparisonAnalyzer delegation (Phase B.1 residue fix)
# ---------------------------------------------------------------------------

def _analyzer_with_meta(meta, key_threshold=3):
    from comparison.comparison_analyzer import ComparisonAnalyzer

    analyzer = object.__new__(ComparisonAnalyzer)
    analyzer._path_run_meta = {("banc_v888", key_threshold): meta}
    return analyzer


def test_analyzer_floor_only_run_reports_natural_tau_not_asked():
    """The reported residue: a floored-but-unbitten run's materialized set
    is a complete run at the natural tau (>= w0), not at the asked
    threshold. The old local formula returned max(asked, floor); the
    shared formula matches the folder's parameters.txt."""
    analyzer = _analyzer_with_meta({
        "tau": 9.0, "tau_canonical": None,
        "strongest_dropped_bottleneck": None,
        "budget_bitten": False, "paths_complete": True,
        "skipped": False, "duplicate_of": None,
        "applied_folder": 3, "edge_weight_floor": 6.0,
    })
    applied, pruned, floor, source = analyzer._applied_state_for(
        "banc_v888", 3)
    assert applied == 9            # natural tau, not max(3, 6) == 6
    assert pruned is True
    assert floor == 6.0
    assert source == "edge_budget"


def test_analyzer_source_names_bite_and_floor():
    analyzer = _analyzer_with_meta({
        "tau": 14.0, "tau_canonical": 12,
        "strongest_dropped_bottleneck": 11.0,
        "budget_bitten": True, "paths_complete": False,
        "skipped": False, "duplicate_of": None,
        "applied_folder": 3, "edge_weight_floor": 6.0,
    })
    applied, _pruned, _floor, source = analyzer._applied_state_for(
        "banc_v888", 3)
    assert applied == 12
    assert source == "strongest_first_budget+edge_budget"


def test_analyzer_complete_run_source_is_requested():
    analyzer = _analyzer_with_meta({
        "tau": 5.0, "tau_canonical": 5,
        "strongest_dropped_bottleneck": None,
        "budget_bitten": False, "paths_complete": True,
        "skipped": False, "duplicate_of": None,
        "applied_folder": 3, "edge_weight_floor": None,
    })
    applied, pruned, floor, source = analyzer._applied_state_for(
        "banc_v888", 3)
    assert applied == 3
    assert pruned is False and floor is None
    assert source == "requested"


def test_comparison_provenance_row_exposes_threshold_and_budget_fields():
    analyzer = _analyzer_with_meta({
        "requested_threshold": 3,
        "tau": 14.0,
        "tau_canonical": 12,
        "budget_bitten": True,
        "strongest_dropped_bottleneck": 11.0,
        "strongest_first_budget": 500,
        "edge_budget": 1000,
        "edge_budget_landing": 5,
        "edge_weight_floor": 6,
        "strongest_retained_bottleneck": 15,
        "paths_complete": False,
        "skipped": False,
        "applied_folder": 3,
    })
    row = analyzer._path_provenance_row("banc_v888", 3)

    assert row["requested_threshold"] == 3
    assert row["applied_threshold"] == 12
    assert row["applied_threshold_source"] == \
        "strongest_first_budget+edge_budget"
    assert row["strongest_first_budget"] == 500
    assert row["edge_budget"] == 1000
    assert row["edge_budget_applied"] is True
    assert row["edge_weight_floor"] == 6
    assert row["strongest_dropped_bottleneck"] == 11.0
    assert row["strongest_retained_bottleneck"] == 15


def test_analyzer_bare_bite_source_is_strongest_first_budget():
    analyzer = _analyzer_with_meta({
        "tau": 12.0, "tau_canonical": 8,
        "strongest_dropped_bottleneck": 7.0,
        "budget_bitten": True, "paths_complete": False,
        "skipped": False, "duplicate_of": None,
        "applied_folder": 3, "edge_weight_floor": None,
    })
    applied, pruned, floor, source = analyzer._applied_state_for(
        "banc_v888", 3)
    assert applied == 8
    assert pruned is False
    assert source == "strongest_first_budget"


def test_analyzer_skipped_row_without_canonical_uses_folder():
    analyzer = _analyzer_with_meta({
        "tau": 25.0, "tau_canonical": None,
        "strongest_dropped_bottleneck": None,
        "budget_bitten": True, "paths_complete": False,
        "skipped": True, "duplicate_of": 10,
        "applied_folder": 24, "edge_weight_floor": None,
    }, key_threshold=20)
    applied, _pruned, _floor, source = analyzer._applied_state_for(
        "banc_v888", 20)
    assert applied == 24  # the applied folder, never the landing tau
    assert source == "strongest_first_budget"


# ---------------------------------------------------------------------------
# Shared formula
# ---------------------------------------------------------------------------

def test_complete_run_applied_is_requested_tau_reported_separately():
    prov = applied_threshold_provenance(
        requested_threshold=3,
        strongest_first_tau=9,
        strongest_first_budget_bitten=False,
        tau_canonical=9,
    )
    assert prov['applied_threshold'] == 3
    assert prov['applied_threshold_source'] == 'requested'
    assert prov['paths_complete'] is True
    assert prov['strongest_first_tau'] == 9
    assert prov['tau_canonical'] == 9


def test_sf_bitten_run_applied_is_w2_plus_one():
    prov = applied_threshold_provenance(
        requested_threshold=3,
        strongest_first_tau=12,
        strongest_first_budget_bitten=True,
        strongest_dropped_bottleneck=7,
        tau_canonical=8,
    )
    assert prov['applied_threshold'] == 8
    assert prov['applied_threshold_source'] == 'strongest_first_budget'
    assert prov['paths_complete'] is False
    assert prov['strongest_dropped_bottleneck'] == 7


def test_edge_budget_only_run():
    prov = applied_threshold_provenance(
        requested_threshold=3,
        strongest_first_tau=10,
        strongest_first_budget_bitten=False,
        tau_canonical=10,
        edge_weight_floor=6,
        edge_budget_landing=5,
        edge_budget=1000,
    )
    assert prov['applied_threshold'] == 10
    assert prov['applied_threshold_source'] == 'edge_budget'
    assert prov['edge_budget_applied'] is True
    assert prov['paths_complete'] is False


def test_combined_run_names_both_sources():
    prov = applied_threshold_provenance(
        requested_threshold=3,
        strongest_first_tau=14,
        strongest_first_budget_bitten=True,
        strongest_dropped_bottleneck=11,
        tau_canonical=12,
        edge_weight_floor=6,
        edge_budget_landing=5,
        edge_budget=1000,
    )
    assert prov['applied_threshold'] == 12
    assert prov['applied_threshold_source'] == \
        'strongest_first_budget+edge_budget'
    assert prov['paths_complete'] is False


def test_floor_guard_against_degenerate_canonical():
    # A canonical below the floor is impossible; the formula must not
    # report a contradiction.
    prov = applied_threshold_provenance(
        requested_threshold=3,
        strongest_first_tau=9,
        strongest_first_budget_bitten=True,
        tau_canonical=4,
        edge_weight_floor=6,
        edge_budget=1000,
    )
    assert prov['applied_threshold'] == 6


# ---------------------------------------------------------------------------
# Contract 2026-09-11 (rev 2): applied = the canonical equivalent threshold.
# NO max(canonical, requested) clamp. The pipeline order (filter by requested
# -> edge budget -> graph -> StrongestFirst) makes a per-slice run's canonical
# structurally >= its own requested threshold; a canonical below the request
# means the slice inherited a lower threshold's provenance (fixed in the
# replay, not by clamping the label).
# ---------------------------------------------------------------------------

def test_applied_is_canonical_not_clamped():
    """A canonical below the request is reported as-is: it is the minimal
    equivalent threshold for the emitted set (the replay must not hand a
    higher slice a lower threshold's canonical)."""
    prov = applied_threshold_provenance(
        requested_threshold=30,
        strongest_first_tau=None,
        strongest_first_budget_bitten=True,
        strongest_dropped_bottleneck=3,
        tau_canonical=4,
        edge_weight_floor=4,
        edge_budget=1000000,
    )
    assert prov['applied_threshold'] == 4
    assert prov['applied_threshold_source'] == 'strongest_first_budget'
    assert prov['tau_canonical'] == 4


def test_floor_below_requested_is_inactive():
    """w0 <= requested: the floored graph reproduces the unfiltered
    output at this threshold, so the run is untouched and complete."""
    prov = applied_threshold_provenance(
        requested_threshold=30,
        strongest_first_tau=30,
        strongest_first_budget_bitten=False,
        tau_canonical=30,
        edge_weight_floor=9,
        edge_budget_landing=8,
        edge_budget=1000000,
    )
    assert prov['applied_threshold'] == 30
    assert prov['applied_threshold_source'] == 'requested'
    assert prov['edge_budget_applied'] is False
    assert prov['edge_floor_binding'] is False
    assert prov['paths_complete'] is True


def test_floor_equal_requested_is_inactive():
    prov = applied_threshold_provenance(
        requested_threshold=9,
        strongest_first_tau=9,
        strongest_first_budget_bitten=False,
        tau_canonical=9,
        edge_weight_floor=9,
        edge_budget=1000000,
    )
    assert prov['applied_threshold'] == 9
    assert prov['applied_threshold_source'] == 'requested'
    assert prov['edge_budget_applied'] is False
    assert prov['paths_complete'] is True


def test_floor_above_requested_is_binding():
    """w0 > requested: the floor removed material the threshold needs, so
    the run is exactly a complete run at w0."""
    prov = applied_threshold_provenance(
        requested_threshold=3,
        strongest_first_tau=9,
        strongest_first_budget_bitten=False,
        tau_canonical=9,
        edge_weight_floor=9,
        edge_budget_landing=8,
        edge_budget=1000000,
    )
    assert prov['applied_threshold'] == 9
    assert prov['applied_threshold_source'] == 'edge_budget'
    assert prov['edge_budget_applied'] is True
    assert prov['edge_floor_binding'] is True
    assert prov['paths_complete'] is False


def test_analyzer_banc_30_complete_slice_reports_requested():
    """The reported BANC-30 case: a COMPLETE slice (paths_complete=True, no
    SF bite) with a graph floor below the request must report applied = 30,
    not the inherited floor. This is the correct per-slice provenance, not a
    clamp: no SF bite and floor inactive => untouched run."""
    analyzer = _analyzer_with_meta({
        "requested_threshold": 30,
        "tau": 30.0,
        "tau_canonical": 30,
        "strongest_dropped_bottleneck": None,
        "budget_bitten": False, "paths_complete": True,
        "skipped": False, "duplicate_of": None,
        "applied_folder": 30, "edge_weight_floor": 4.0,
        "edge_budget": 1000000, "edge_budget_landing": 3.0,
    }, key_threshold=30)
    applied, pruned, floor, source = analyzer._applied_state_for(
        "banc_v888", 30)
    assert applied == 30
    assert pruned is False
    assert floor == 4.0
    assert source == "requested"


def test_complete_slice_reports_requested_with_natural_tau():
    """A complete slice reports the requested threshold; its natural tau is
    the minimal equivalent, reported separately (not applied)."""
    prov = applied_threshold_provenance(
        requested_threshold=30,
        strongest_first_tau=45,
        strongest_first_budget_bitten=False,
        tau_canonical=45,
        edge_weight_floor=4,
        edge_budget=1000000,
    )
    assert prov['applied_threshold'] == 30
    assert prov['strongest_first_tau'] == 45
    assert prov['paths_complete'] is True
    assert prov['edge_budget_applied'] is False




# ---------------------------------------------------------------------------
# Instance-level finalization
# ---------------------------------------------------------------------------

def _state_fc(tmp_path):
    fc = object.__new__(coana.FindNeuronConnection)
    fc.allpath_folder = str(tmp_path)
    fc.source_fname = 'src'
    fc.target_fname = 'tgt'
    fc.dataset = 'test:v1'
    fc.min_synapse_num = 3
    fc.max_paths_bodyid = 0
    fc.graph_edge_limit_bodyid = 0
    fc.drop_untyped = True
    fc.parameter_dict = {'min synapse number': '3'}
    fc.parameter_df = pd.DataFrame(
        [('min synapse number', '3')], columns=['parameter', 'value'])
    fc.strongest_first_cutoff = None
    fc.strongest_first_budget_bitten = False
    fc.strongest_dropped_bottleneck = None
    fc.tau_canonical = None
    fc.edge_weight_floor = None
    fc.edge_budget_landing = None
    fc.strongest_retained_bottleneck = None
    fc._vprint = lambda *a, **k: None
    return fc


def test_finalize_records_full_contract(tmp_path):
    fc = _state_fc(tmp_path)
    fc.strongest_first_cutoff = 12
    fc.strongest_first_budget_bitten = True
    fc.strongest_dropped_bottleneck = 7
    fc.tau_canonical = 8
    fc.strongest_retained_bottleneck = 15

    prov = fc._finalize_threshold_provenance(path_mode='all')

    assert prov['requested_threshold'] == 3
    assert prov['applied_threshold'] == 8
    assert prov['applied_threshold_source'] == 'strongest_first_budget'
    assert prov['strongest_first_budget'] == 1000000  # auto budget
    assert prov['paths_complete'] is False
    # public instance attributes -> all_attributes.json export
    for attr in ('requested_threshold', 'applied_threshold',
                 'applied_threshold_source', 'strongest_first_budget',
                 'strongest_first_budget_bitten', 'strongest_first_tau',
                 'tau_canonical', 'strongest_dropped_bottleneck',
                 'edge_budget', 'edge_budget_applied', 'edge_budget_landing',
                 'edge_weight_floor', 'strongest_retained_bottleneck',
                 'paths_complete'):
        assert hasattr(fc, attr), attr
    # structured parameter export carries the same contract names
    assert fc.parameter_dict['applied_threshold'] == '8'
    assert fc.parameter_dict['applied_threshold_source'] == \
        'strongest_first_budget'
    assert 'applied_threshold' in set(fc.parameter_df['parameter'])


def test_finalize_shortest_mode_never_floored(tmp_path):
    fc = _state_fc(tmp_path)
    fc.graph_edge_limit_bodyid = 1000000  # even with a (ignored) budget
    fc.strongest_first_cutoff = 5
    fc.strongest_first_budget_bitten = True
    fc.strongest_dropped_bottleneck = 3
    fc.tau_canonical = 4

    prov = fc._finalize_threshold_provenance(path_mode='shortest')

    assert prov['edge_budget'] is None
    assert prov['edge_budget_applied'] is False
    assert prov['edge_weight_floor'] is None
    assert prov['applied_threshold'] == 4
    assert prov['applied_threshold_source'] == 'strongest_first_budget'


def test_write_run_metadata_writes_provenance_block(tmp_path):
    fc = _state_fc(tmp_path)
    fc.strongest_first_cutoff = 12
    fc.strongest_first_budget_bitten = True
    fc.strongest_dropped_bottleneck = 7
    fc.tau_canonical = 8
    fc._finalize_threshold_provenance(path_mode='all')
    fc._write_run_metadata('all')

    text = (tmp_path / 'parameters.txt').read_text(encoding='utf-8')
    for key in ('requested_threshold', 'applied_threshold',
                'applied_threshold_source', 'strongest_first_budget',
                'strongest_first_budget_bitten', 'strongest_first_tau',
                'tau_canonical', 'strongest_dropped_bottleneck',
                'edge_budget', 'edge_budget_applied', 'edge_budget_landing',
                'edge_weight_floor', 'strongest_retained_bottleneck',
                'paths_complete', 'path_mode'):
        assert f'{key}:' in text, key
    # backward-compatible aliases
    assert 'applied_tau (min path bottleneck):' in text
    assert 'edge_weight_floor:' in text

    attrs = json.loads((tmp_path / 'all_attributes.json').read_text())
    assert attrs['applied_threshold'] == 8
    assert attrs['applied_threshold_source'] == 'strongest_first_budget'
    assert attrs['paths_complete'] is False


# ---------------------------------------------------------------------------
# Pipeline-level provenance (offline harness)
# ---------------------------------------------------------------------------

# Three parallel 2-hop routes with distinct bottlenecks 9 / 7 / 3.
_BUDGET_EDGES = [
    ('S', 'A', 10), ('A', 'T', 9),
    ('S', 'B', 7), ('B', 'T', 8),
    ('S', 'C', 3), ('C', 'T', 9),
]


def test_complete_run_bitten_provenance_flows_to_files(
        monkeypatch, tmp_path):
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc, fetch_calls, logs = _make_pipeline_fc(
        monkeypatch, tmp_path, _BUDGET_EDGES, max_interlayer=2)
    fc.pathfinding = 'StrongestFirst'
    fc.max_paths_bodyid = 1  # bite: only the bottleneck-9 path is kept
    fc.FindAllPath()

    assert fc.paths_complete is False
    assert fc.applied_threshold_source == 'strongest_first_budget'
    assert fc.applied_threshold == fc.tau_canonical
    assert fc.strongest_dropped_bottleneck == 7
    assert fc.strongest_first_tau == 9
    assert fc.strongest_retained_bottleneck == 9  # W* from lossless pruning

    text = open(os.path.join(fc.allpath_folder, 'parameters.txt'),
                encoding='utf-8').read()
    assert 'applied_threshold:' in text
    assert 'applied_threshold_source:' in text
    assert 'strongest_first_budget+edge_budget' not in text
    attrs = json.loads(open(os.path.join(
        fc.allpath_folder, 'all_attributes.json'),
        encoding='utf-8').read())
    assert attrs['applied_threshold'] == fc.applied_threshold
    assert attrs['paths_complete'] is False
    # data_details/parameters.csv carries the same block
    params_csv = pl.read_csv(os.path.join(
        fc.allpath_folder, 'data_details', 'parameters.csv'))
    keys = set(params_csv['parameter'].to_list())
    assert {'applied_threshold', 'applied_threshold_source',
            'paths_complete', 'strongest_retained_bottleneck'} <= keys


def test_complete_run_unbitten_provenance_is_requested(
        monkeypatch, tmp_path):
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc, fetch_calls, logs = _make_pipeline_fc(
        monkeypatch, tmp_path, _BUDGET_EDGES, max_interlayer=2)
    fc.pathfinding = 'StrongestFirst'
    fc.FindAllPath()  # auto budget 1M — no bite

    assert fc.paths_complete is True
    assert fc.applied_threshold == 1  # the requested threshold
    assert fc.applied_threshold_source == 'requested'
    assert fc.strongest_first_tau == 3  # natural weakest emitted bottleneck
    assert fc.tau_canonical == 3
    text = open(os.path.join(fc.allpath_folder, 'parameters.txt'),
                encoding='utf-8').read()
    assert 'requested' in text


def test_shortest_run_exports_budget_metadata_and_no_floor(
        monkeypatch, tmp_path):
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc, fetch_calls, logs = _make_pipeline_fc(
        monkeypatch, tmp_path, _BUDGET_EDGES, max_interlayer=2,
        graph_edge_limit=500000)
    fc.max_paths_bodyid = 1  # bite inside the shortest enumeration
    fc.FindShortestPath()

    assert fc.paths_complete is False
    assert fc.applied_threshold_source == 'strongest_first_budget'
    assert fc.edge_budget_applied is False
    assert fc.edge_weight_floor is None  # shortest mode is NEVER floored
    assert fc.edge_budget is None
    text = open(os.path.join(fc.allpath_folder, 'parameters.txt'),
                encoding='utf-8').read()
    assert 'edge_weight_floor:' in text
    assert 'n/a' in text  # floor rendered as n/a for shortest mode


def test_reconcile_quarantines_instead_of_deleting(tmp_path, monkeypatch):
    """REPO-1 (2026-09-25 handoff): the reconcile's _replace_with_marker
    rmtree'd a "redundant" data folder, while plan §10 (user-CONFIRMED)
    promises "data is never deleted, only renamed" — and the plan's own §4
    documents the mislabeled-folder bugs that can make a "duplicate"
    grouping wrong.  The folder must move to _replaced/ beside the marker,
    invisible to every minsyn_-prefix scan, and recoverable by rename-back."""
    from comparison.comparison_analyzer import ComparisonAnalyzer
    base = tmp_path / 'ds_out'
    folder = base / 'minsyn_50'
    folder.mkdir(parents=True)
    (folder / 'connections.csv').write_text('keep me')
    analyzer = ComparisonAnalyzer.__new__(ComparisonAnalyzer)
    marker_calls = []
    logs = []
    monkeypatch.setattr(
        analyzer, '_write_skipped_marker',
        lambda ds, req, m: marker_calls.append((ds, req)))
    monkeypatch.setattr(analyzer, '_log', lambda msg='': logs.append(msg))
    analyzer._replace_with_marker('ds', 50, 19, str(folder))
    assert marker_calls == [('ds', 50)]
    assert not folder.exists(), 'the original name is vacated for the marker'
    replaced = list((base / '_replaced').iterdir())
    assert len(replaced) == 1 and replaced[0].name.startswith('minsyn_50_')
    assert (replaced[0] / 'connections.csv').read_text() == 'keep me'
    # every folder scan keys on the minsyn_ prefix: the quarantined copy
    # is invisible to the reconcile and to data_loader's threshold listing
    visible = [n for n in os.listdir(base) if n.startswith('minsyn_')]
    assert visible == [], f'quarantine leaked into scans: {visible}'
    # a second call on an already-vacated folder is a no-op, never a raise
    analyzer._replace_with_marker('ds', 50, 19, str(folder))
    assert any('nothing is deleted' in m for m in logs)
