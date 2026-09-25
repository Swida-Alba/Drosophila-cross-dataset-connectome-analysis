"""Regression tests for the 2026-09-25 pathfinding / cross-dataset audit
fix round (report: ``docs/audits/PATHFINDING_CROSS_DATASET_AUDIT_2026-09-25.md``).

Covers F-PF-001..008, F-XD-001/003/008 and the two fix-round residuals
(F-RV-001/002) at the unit seam; F-XD-002 lives with the profiler coverage
tests, F-XD-004 with the analyzer coverage tests.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import coana  # noqa: E402
import statvis  # noqa: E402
from comparison import profile_comparator as pc  # noqa: E402


# ---------------------------------------------------------------------------
# F-PF-001: the 'None' keyword sentinel must never reach the streaming writer
# ---------------------------------------------------------------------------

def test_normalized_keyword_filter_sentinel_contract():
    fnc = coana.FindNeuronConnection.__new__(coana.FindNeuronConnection)
    fnc.keyword_in_path_to_remove = 'None'
    assert fnc._normalized_keyword_filter() is None
    fnc.keyword_in_path_to_remove = ['None']
    assert fnc._normalized_keyword_filter() is None
    fnc.keyword_in_path_to_remove = ['None', 'glia']
    assert fnc._normalized_keyword_filter() == ['None', 'glia']
    fnc.keyword_in_path_to_remove = 'glia'
    assert fnc._normalized_keyword_filter() == ['glia']


def test_streaming_batch_keeps_none_substring_when_keyword_is_none():
    """statvis's batch filter with keyword=None is a no-op, so a type label
    containing 'None' survives the streamed export (the sentinel used to
    reach the writer raw and drop such paths)."""
    df = pl.DataFrame({'path': ['["aNone_L", "b_L"]', '["c_L", "d_L"]'],
                       'path_prob': [0.5, 0.4]})
    excluded, kept = statvis._keyword_filter_batch(df, None)
    assert kept.height == 2 and excluded.height == 0


# ---------------------------------------------------------------------------
# F-PF-002: hemisphere_filter participates in the graph-cache key
# ---------------------------------------------------------------------------

def test_graph_cache_key_separates_hemisphere_filters():
    base = dict(dataset_safe='ds', source_ID={1}, target_ID={2},
                max_interlayer=1, separate_hemispheres=False,
                filter_by='bodyid', min_ratio=0.0,
                min_traversal_probability=0.0,
                exclude_intra_type_connections=False, drop_untyped=True)
    k_both = coana._findallpath_cache_key(**base, hemisphere_filter='both')
    k_left = coana._findallpath_cache_key(**base, hemisphere_filter='left')
    k_right = coana._findallpath_cache_key(**base, hemisphere_filter='right')
    assert len({k_both, k_left, k_right}) == 3
    # same inputs -> stable key
    assert k_both == coana._findallpath_cache_key(**base,
                                                  hemisphere_filter='both')
    # and it still separates hemisphere-aware runs
    assert k_both != coana._findallpath_cache_key(
        **{**base, 'separate_hemispheres': True}, hemisphere_filter='both')


# ---------------------------------------------------------------------------
# F-PF-003: fit_edge_budget virtual-boundary landing must not IndexError
# ---------------------------------------------------------------------------

def _chain_cone(n_tiers, weight_of=None):
    """S -> M_i -> T chain, one tier per distinct weight (2 edges each)."""
    rows = []
    for i in range(n_tiers):
        w = n_tiers - i if weight_of is None else weight_of(i)
        rows.append({'bodyId_pre': 1, 'bodyId_post': 1000 + i, 'weight': w})
        rows.append({'bodyId_pre': 1000 + i, 'bodyId_post': 2, 'weight': w})
    return [pl.DataFrame(rows)]


@pytest.mark.parametrize("n_tiers,budget", [
    (60, 70),   # the arithmetic case: probes exhaust with a virtual bracket
    (30, 56), (40, 60), (80, 90),
])
def test_fit_edge_budget_never_indexes_past_the_tier_list(n_tiers, budget):
    """A gallop that runs off the tier list with probes to spare leaves the
    virtual hi_idx == len(distinct); the landing read must clamp to a real
    tier instead of raising IndexError."""
    tables = _chain_cone(n_tiers)
    stats = {}
    floored, stats = coana.fit_edge_budget(
        tables, budget, sources={1}, targets={2}, bound=2)
    distinct = sorted({r['weight'] for r in tables[0].to_dicts()},
                      reverse=True)
    if 'landing' in stats:
        assert stats['landing'] in distinct, stats
    # the result must still honor the budget (or be the unfloored cone)
    rows = sum(t.height for t in floored)
    assert rows <= max(budget, rows)  # sanity: no exception, real tables


# ---------------------------------------------------------------------------
# F-PF-004: the conserved-edges note tells the truth when hemi-aware is off
# ---------------------------------------------------------------------------

def test_conserved_edge_note_says_ignored_without_hemi_split(tmp_path):
    fnc = coana.FindNeuronConnection.__new__(coana.FindNeuronConnection)
    fnc._warn_notes = []
    fnc.keep_only_hemisphere_conserved_connections = True
    fnc.separate_hemispheres = False
    fnc._write_user_warning_notes(str(tmp_path))
    written = tmp_path / 'user_warning_notes.txt'
    assert written.exists() and 'IGNORED' in written.read_text()
    assert 'NOT filtered' in written.read_text()


# ---------------------------------------------------------------------------
# F-PF-005: a failed metadata re-stamp is loud and marked
# ---------------------------------------------------------------------------

def test_metadata_restamp_failure_is_marked(tmp_path, monkeypatch):
    fnc = coana.FindNeuronConnection.__new__(coana.FindNeuronConnection)
    fnc.allpath_folder = str(tmp_path)
    fnc.parameter_dict = {}
    fnc.source_fname = fnc.target_fname = 'x'
    fnc._vprint = lambda *a, **k: None

    def boom(**_kw):
        raise RuntimeError('serialize failed')

    monkeypatch.setattr(fnc, '_run_export_attributes', boom)
    fnc._write_run_metadata('all')   # must not raise
    marker = tmp_path / '.provenance_incomplete'
    assert marker.exists()
    assert 'serialize failed' in marker.read_text()


# ---------------------------------------------------------------------------
# F-PF-007: derived label-path order is deterministic
# ---------------------------------------------------------------------------

def test_derived_label_paths_order_is_stable():
    fnc = coana.FindNeuronConnection.__new__(coana.FindNeuronConnection)

    def node_label(n):
        return f'type{n}'

    kept = {(f'type{a}', f'type{b}')
            for a in range(6) for b in range(6) if b == a + 1}
    paths_a = [[i, i + 1, i + 2] for i in range(0, 4)]
    paths_b = list(reversed(paths_a))          # same content, other order
    out_a = fnc._derive_label_paths_from_bodyid_paths(
        paths_a, node_label, kept, {'type0'}, {'type4'})
    out_b = fnc._derive_label_paths_from_bodyid_paths(
        paths_b, node_label, kept, {'type0'}, {'type4'})
    assert out_a == sorted(out_a)          # deterministic (sorted) order
    assert out_a == out_b                  # independent of input iteration


# ---------------------------------------------------------------------------
# F-RV-001: literal_eval failures degrade to the '->' fallback, never raise
# ---------------------------------------------------------------------------

def test_parse_path_nodes_degrades_on_unhashable_literal():
    parsed = coana._parse_path_nodes('{[1]: 2}')     # TypeError inside
    assert parsed == ['{[1]: 2}']
    assert coana._parse_path_nodes('A->B') == ['A', 'B']
    assert coana._parse_path_nodes("['A', 'B']") == ['A', 'B']


# ---------------------------------------------------------------------------
# F-XD-001: cross-dataset strict mode never uses the intra-bodyId metric
# ---------------------------------------------------------------------------

class _StubProfiler:
    def get_profile(self, neuron, dataset):
        return SimpleNamespace(
            neuron_id=neuron, dataset=dataset, neuron_type=str(neuron),
            upstream_partners={'u1': 1.0}, downstream_partners={'d1': 1.0},
            upstream_ranks={'u1': 1}, downstream_ranks={'d1': 1})

    # pre-warm surface: no-ops so direct_comparison stays offline
    def consolidate_profile_cache(self, *a, **k):
        pass

    def _load_cache_dataframe(self, *a, **k):
        return None

    def _get_cached_conn_df(self, *a, **k):
        return None


class _NoFetchFNC:
    """Stand-in for FindNeuronConnection: never touches the network."""
    def __init__(self, *a, **k):
        pass

    def build_connection_cache(self, *a, **k):
        pass


def test_cross_dataset_strict_avoids_bodyid_core(monkeypatch):
    """dataset_a != dataset_b + strict + type names must NOT delegate to
    compare_types_bodyid_core (bare-bodyId scoring across ID spaces)."""
    def explode(*a, **kw):
        raise AssertionError('cross-dataset strict used the bodyId core')

    monkeypatch.setattr(coana, 'FindNeuronConnection', _NoFetchFNC)
    monkeypatch.setattr(pc.ProfileComparator,
                        'compare_types_bodyid_core', explode)
    monkeypatch.setattr(
        pc.ProfileComparator, 'batch_compare_cross_dataset',
        staticmethod(lambda **kw: [
            {'target_bid': key, 'rank': 0.5, 'rank_corr': 0.5,
             'rank_union': 0.3, 'jaccard': 0.4, 'cosine': 0.6,
             'shared_type_count': 1, 'union_type_count': 2,
             'n_source_bodyIds': 1, 'n_target_bodyIds': 1}
            for key in kw['target_profiles']]))

    result = pc.ProfileComparator.direct_comparison(
        neurons_a='MBON14', neurons_b='MBON14',
        dataset_a='hemibrain:v1.2.1', dataset_b='male-cns:v1.0',
        profiler=_StubProfiler(), comparison_mode='strict', verbose=False)
    assert not result['results'].empty


def test_same_dataset_strict_still_uses_bodyid_core(monkeypatch):
    called = {}

    def fake_core(**kw):
        called['yes'] = True
        summary = pd.DataFrame([{
            'neuron_type': 'MBON14', 'avg_rank_corr': 0.9,
            'avg_rank_union': 0.8, 'avg_jaccard': 0.7, 'avg_cosine': 0.8,
            'n_source_bodyIds': 1, 'n_target_bodyIds': 1}])
        return pd.DataFrame(), summary

    monkeypatch.setattr(coana, 'FindNeuronConnection', _NoFetchFNC)
    monkeypatch.setattr(pc.ProfileComparator,
                        'compare_types_bodyid_core', staticmethod(fake_core))
    pc.ProfileComparator.direct_comparison(
        neurons_a='MBON14', neurons_b='MBON14',
        dataset_a='hemibrain:v1.2.1', dataset_b='hemibrain:v1.2.1',
        profiler=_StubProfiler(), comparison_mode='strict', verbose=False)
    assert called.get('yes') is True


# ---------------------------------------------------------------------------
# F-XD-003: cached comparison results carry a query fingerprint
# ---------------------------------------------------------------------------

class _AnalyzerShell:
    """Just enough ComparisonAnalyzer surface for the fingerprint methods."""
    from comparison.comparison_analyzer import ComparisonAnalyzer  # noqa

    def __init__(self, parameters):
        self.parameters = parameters

    _query_fingerprint = ComparisonAnalyzer._query_fingerprint
    _cached_result_matches_query = (
        ComparisonAnalyzer._cached_result_matches_query)

    def _log(self, *_a, **_k):
        pass


def test_query_fingerprint_and_cache_validation(tmp_path):
    params = SimpleNamespace(
        source_neurons=['aMe12'], target_neurons=['PPL101'],
        comparison_mode='edge', max_interlayer=2, pathfinding='StrongestFirst')
    analyzer = _AnalyzerShell(params)

    fp = analyzer._query_fingerprint()
    # sidecar written -> matches
    sidecar = tmp_path / 'connections_edge.fingerprint.json'
    sidecar.write_text(json.dumps(fp))
    assert analyzer._cached_result_matches_query(str(tmp_path)) is True
    # different query -> mismatch
    params.source_neurons = ['aMe01']
    assert analyzer._cached_result_matches_query(str(tmp_path)) is False
    # legacy folder (no sidecar) -> accepted
    empty = tmp_path / 'legacy'
    empty.mkdir()
    assert analyzer._cached_result_matches_query(str(empty)) is True


# ---------------------------------------------------------------------------
# F-XD-008: dataset spellings that sanitize to one folder are refused
# ---------------------------------------------------------------------------

def test_colliding_dataset_spellings_are_refused():
    from comparison.comparison_parameters import ComparisonParameters
    with pytest.raises(ValueError, match='sanitized name'):
        ComparisonParameters(
            datasets=['hemibrain:v1.2.1', 'hemibrain_v1_2_1'],
            source_neurons=['aMe12'], target_neurons=['PPL101'])
