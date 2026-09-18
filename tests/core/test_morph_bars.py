"""Unit tests for the unified morph-qualification bar engine (floors v3)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from comparison.morph_bars import (  # noqa: E402
    CANDIDATE_BACKUP,
    CANDIDATE_NATIVE,
    CANDIDATE_NULL,
    SUSPICIOUS_TRACK_A,
    BarSet,
    candidate_qualified,
    compute_branch_bars,
    suspicious_qualified,
)


def test_native_floor_binds_when_two_refs():
    bars = compute_branch_bars(ref_pool_sim_mean=0.80, n_native_refs=2,
                               pool_track_a_baseline=0.60, n_scored_pool=3,
                               null_bar=0.60)
    assert bars.candidate_kind == CANDIDATE_NATIVE
    assert bars.native_floor == 0.75
    # native compares pool_ref; Track-A is irrelevant on this rung
    assert candidate_qualified(bars, pool_ref=0.751, track_a=None)
    assert not candidate_qualified(bars, pool_ref=0.749, track_a=0.99)


def test_native_floor_needs_two_refs():
    bars = compute_branch_bars(ref_pool_sim_mean=0.80, n_native_refs=1,
                               pool_track_a_baseline=0.60, n_scored_pool=2)
    assert bars.candidate_kind == CANDIDATE_BACKUP
    assert bars.native_floor is None
    assert bars.backup_floor == pytest.approx(0.55)


def test_backup_floor_compares_track_a():
    bars = compute_branch_bars(pool_track_a_baseline=0.60, n_scored_pool=1)
    assert bars.candidate_kind == CANDIDATE_BACKUP
    assert bars.backup_floor == pytest.approx(0.55)
    assert candidate_qualified(bars, pool_ref=None, track_a=0.56)
    assert not candidate_qualified(bars, pool_ref=0.99, track_a=0.54)


def test_null_fallback_when_no_refs_and_no_pool_scores():
    bars = compute_branch_bars(null_bar=0.60)
    assert bars.candidate_kind == CANDIDATE_NULL
    assert candidate_qualified(bars, pool_ref=None, track_a=0.61)
    assert not candidate_qualified(bars, pool_ref=None, track_a=0.59)
    assert candidate_qualified(bars, pool_ref=None, track_a=None) is False


def test_suspicious_level_derives_from_same_dial():
    bars = compute_branch_bars(pool_track_a_baseline=0.60, n_scored_pool=1,
                               suspicious_level=3)
    assert bars.suspicious_kind == 'track_a_suspicious'
    assert bars.suspicious_floor == 0.60 - 3 * 0.05
    assert suspicious_qualified(bars, track_a=0.46)
    assert not suspicious_qualified(bars, track_a=0.44)


def test_suspicious_fallback_uses_null_p50():
    bars = compute_branch_bars(null_bar=0.60, null_bar_lo=0.40)
    assert bars.suspicious_kind == 'null_lo'
    assert bars.suspicious_floor == 0.40
    assert suspicious_qualified(bars, track_a=0.41)


def test_suspicious_requires_scored_pool_or_null_lo():
    bars = compute_branch_bars()
    assert bars.suspicious_floor is None
    assert suspicious_qualified(bars, track_a=0.9) is False


def test_random_null_floor_clamps_every_bar():
    bars = compute_branch_bars(ref_pool_sim_mean=0.52, n_native_refs=2,
                               pool_track_a_baseline=0.52, n_scored_pool=2,
                               random_null_floor=0.50)
    assert bars.clamp_applied
    assert bars.candidate_kind == CANDIDATE_NATIVE
    assert bars.native_floor == 0.50
    # native refs exist -> native binds, no backup rung is computed
    assert bars.backup_floor is None


def test_no_clamp_when_bars_above_floor():
    bars = compute_branch_bars(ref_pool_sim_mean=0.80, n_native_refs=2,
                               random_null_floor=0.50)
    assert not bars.clamp_applied
    assert bars.native_floor == 0.75


def test_bars_without_any_basis_admit_nothing():
    bars = compute_branch_bars()
    assert bars.candidate_kind == CANDIDATE_NULL
    assert bars.null_bar is None
    assert candidate_qualified(bars, pool_ref=0.9, track_a=0.9) is False


def test_barser_defaults():
    b = BarSet()
    assert b.candidate_kind == CANDIDATE_NULL
    assert b.track_a_offset == 0.05
    assert b.suspicious_level == 3


# ---------------------------------------------------------------------------
# Floors v3 wiring: ladder, aggressive split, scope stability (plan §5)
# ---------------------------------------------------------------------------

def test_ladder_includes_verified_in_references(monkeypatch):
    """Floors v3: the reference tier is matched+verified whenever the two
    together have >= 2 members — verified neurons are no longer excluded
    when >= 2 matched exist (the run_morphology ladder, exercised through
    the Track-B mock harness)."""
    import numpy as np
    import morphology
    from comparison.mapping_validation import (MappingValidationConfig,
                                               MappingValidator)

    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(
        source_dataset='flywire_FAFB_v783', target_dataset='male-cns:v1.0',
        query_types=['T'], visualize=False, pool_ref_floor_margin=0.05)
    v.notes = []
    v.log = lambda m='': None

    class FakeMapper:
        def get_mapping_decision(self, t, src, tgt):
            return {'status': 'mapped', 'target_type': 'TX',
                    'target_types': ['TX']}

    v.mapper = FakeMapper()
    v.pairs = []
    v._source_type_counts = {'TX': 5}
    v._source_add_counts = {}

    unit = np.zeros(8); unit[0] = 1.0
    # 21 is the invader — it must be in the cache or the whole Track-B
    # block aborts (fail-closed by design).
    vecs = {11: unit, 12: unit, 13: unit, 14: unit, 21: unit}

    class FakeCache:
        def vectors_for(self, ids, compute_missing=True):
            return np.stack([vecs[i] for i in ids]), [True] * len(ids), None

        def load(self):
            return {'whiten': np.eye(8)}

    monkeypatch.setattr(morphology, 'find_similar_dataset_cache_v2',
                        lambda ds, verbose=False, **k: FakeCache())
    monkeypatch.setattr(morphology, 'apply_whitening', lambda w, X: X)
    monkeypatch.setattr(morphology, 'v2_similarity_matrix',
                        lambda A, B, weights: A @ B.T)
    monkeypatch.setattr(morphology, 'DEFAULT_V2_BLOCK_WEIGHTS',
                        {'all': 1.0}, raising=False)

    pool_detail = [
        {'target_bodyId': 11, 'category': 'matched',
         'best_source_bodyId': 1, 'source_type': 'T', 'target_type': 'T'},
        {'target_bodyId': 12, 'category': 'matched',
         'best_source_bodyId': 1, 'source_type': 'T', 'target_type': 'T'},
        {'target_bodyId': 13, 'category': 'verified',
         'best_source_bodyId': 1, 'source_type': 'T', 'target_type': 'T'},
        {'target_bodyId': 14, 'category': 'verified',
         'best_source_bodyId': 1, 'source_type': 'T', 'target_type': 'T'},
    ]
    sus = [{'source_bodyId': 1, 'source_type': 'T', 'target_type': 'T',
            'ahead_target_bodyId': 21, 'ahead_target_type': 'TX',
            'invader_class': 'unmapped'}]
    out = v.run_morphology([], sus, [], pool_detail, deep_rows=[])
    bars = out['branch_bars']['T->T']
    assert bars['n_native_refs'] == 4          # 2 matched + 2 verified
    assert bars['candidate_kind'] == 'native'
    assert bars['native_floor'] == pytest.approx(0.95)
    assert out['pool_ref_tiers']['T->T'] == 'matched+verified'
    assert out['bar_params']['track_a_offset'] == 0.05
    assert out['bar_params']['suspicious_level'] == 3


def test_aggressive_deep_window_two_gate_split():
    """Floors v3 aggressive: a deep row passing either bar is suspicious;
    below both it is out of scope AND morph-failed; restrictive mode never
    admits deep rows (unchanged)."""
    from comparison.mapping_validation import classify_category
    common = dict(target_bid=500, branch_pool=frozenset({101}),
                  in_map=frozenset({101}), target_type='OTHER',
                  branch_target_type='SMP220', in_map_types=frozenset({'SMP220'}),
                  candidate_types=frozenset(), connectivity_qualified=False,
                  tier=None)
    # strict pass -> suspicious (below-pool-best but fully qualified)
    cat, scope, failed = classify_category(
        morph_ok=True, suspicious_morph_ok=False, is_deep=True,
        mode='aggressive', **common)
    assert (cat, scope, failed) == ('examinees', True, False)
    # same-type deep row -> family (rule 4 precedes rule 6, per §4.5)
    same_type = dict(common, target_type='SMP220')
    cat, scope, failed = classify_category(
        morph_ok=True, suspicious_morph_ok=False, is_deep=True,
        mode='aggressive', **same_type)
    assert (cat, scope, failed) == ('family', True, False)
    # between bars: strict fail, loose pass -> suspicious
    cat, scope, failed = classify_category(
        morph_ok=False, suspicious_morph_ok=True, is_deep=True,
        mode='aggressive', **common)
    assert (cat, scope, failed) == ('examinees', True, False)
    # below both -> out of scope, morph-failed
    cat, scope, failed = classify_category(
        morph_ok=False, suspicious_morph_ok=False, is_deep=True,
        mode='aggressive', **common)
    assert (cat, scope, failed) == ('', False, True)
    # restrictive: deep rows are never admitted (unchanged), not flagged
    cat, scope, failed = classify_category(
        morph_ok=False, suspicious_morph_ok=True, is_deep=True,
        mode='restrictive', **common)
    assert (cat, scope, failed) == ('', False, False)


def test_backup_floor_ignores_null_bar_scope_stability():
    """80536 scope-stability fixture: a branch with a Track-A backup floor
    decides on B_b - Δ alone — the per-run null bar cannot flip the
    verdict when the query scope changes."""
    bars = compute_branch_bars(pool_track_a_baseline=0.6177, n_scored_pool=1,
                               null_bar=0.6058, track_a_offset=0.05)
    assert bars.candidate_kind == CANDIDATE_BACKUP
    assert bars.backup_floor == pytest.approx(0.5677)
    assert candidate_qualified(bars, pool_ref=None, track_a=0.582)
    assert not candidate_qualified(bars, pool_ref=None, track_a=0.55)


def test_morph_qualified_none_bars_fail_closed():
    from comparison.mapping_validation import morph_qualified,         morph_qualified_suspicious
    row = {'morph_pool_ref': 0.9, 'morph_v2_similarity': 0.9}
    assert morph_qualified(row, None) is False
    assert morph_qualified_suspicious(row, None) is False


# ---------------------------------------------------------------------------
# Plan I §1–2: profile pre-flight + progress JSONL
# ---------------------------------------------------------------------------

def test_preflight_builds_only_missing_and_emits_jsonl(tmp_path):
    """The stage-2 pre-flight builds exactly universe-minus-cached through
    the profiler backend, batch-saves, emits profiles_progress events, and
    is resumable (a second pass builds nothing)."""
    from comparison.mapping_validation import (MappingValidationConfig,
                                               MappingValidator,
                                               ProgressReporter)

    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='flywire_FAFB_v783',
                                    target_dataset='banc_v888',
                                    query_types=['T'], visualize=False)
    v.notes = []
    built = []

    class FakeProfile:
        top_k_bodyid_used = 25

    class FakeProfiler:
        def get_profile(self, bid, dataset, force_refresh=False):
            built.append(int(bid))
            return FakeProfile()

        def _load_cache_dataframe(self, dataset):
            import pandas as pd
            return pd.DataFrame({'neuron_id': [11, 12]})

        def _save_profiles_to_cache_batch(self, profiles, dataset,
                                          silent=True):
            pass

        def consolidate_profile_cache(self, dataset):
            pass

    v.profiler = FakeProfiler()
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    v.progress = ProgressReporter(run_dir)

    universe = [11, 12, 13, 14, 15]
    stats = v._preflight_target_profiles('banc_v888', universe=universe,
                                         cached_ids={11, 12})
    assert stats == {'universe': 5, 'cached': 2, 'built': 3}
    assert built == [13, 14, 15]                    # only the missing
    events = [json.loads(l) for l in
              (run_dir / 'pipeline_progress.jsonl').read_text().splitlines()]
    kinds = [e['event'] for e in events]
    assert 'profiles_progress' in kinds
    last = [e for e in events if e['event'] == 'profiles_progress'][-1]
    assert last['done'] == 3 and last['total'] == 3 and last['note'] == 'complete'

    # resumable: cached now covers the universe
    stats2 = v._preflight_target_profiles('banc_v888', universe=universe,
                                          cached_ids=set(universe))
    assert stats2['built'] == 0

import json


def test_expand_out_map_sources_topk_excludes_in_map(monkeypatch, tmp_path):
    """Out-map expansion: scans each unpaired source, drops in-map targets,
    keeps the top-k connectivity-ranked new candidates; progress events
    emitted; connectivity-only (no morph columns)."""
    import json
    import pandas as pd
    import comparison.mapping_validation as mv
    from comparison.mapping_validation import (MappingValidationConfig,
                                               MappingValidator,
                                               ProgressReporter)

    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='flywire_FAFB_v783',
                                    target_dataset='male-cns:v1.0',
                                    query_types=['T'], visualize=False,
                                    out_map_top_k=2)
    v.notes = []

    class _Status:
        name = 'Traced'

    class _Profile:
        connectivity_status = _Status()

    class FakeMapper:
        pass

    class FakeProfiler:
        def get_profile(self, bid, dataset, force_refresh=False):
            return _Profile()

    v.profiler = FakeProfiler()
    v.mapper = FakeMapper()
    v.log = lambda m='': None
    v.progress = ProgressReporter(tmp_path)

    monkeypatch.setattr(mv, 'expanded_vector', lambda sp, mapper: 'vec')
    monkeypatch.setattr(mv, 'scan_source', lambda vec, stats, bids: pd.DataFrame([
        # rank 1: in-map claim -> excluded
        {'target_bid': 100, 'rank_union': 0.9, 'rank_union_rank': 1,
         'jaccard': 0.5, 'jaccard_rank': 1},
        # rank 2 and 3: new candidates
        {'target_bid': 101, 'rank_union': 0.5, 'rank_union_rank': 2,
         'jaccard': 0.3, 'jaccard_rank': 2},
        {'target_bid': 102, 'rank_union': 0.4, 'rank_union_rank': 3,
         'jaccard': 0.2, 'jaccard_rank': 3},
        # rank 4: beyond top-k=2
        {'target_bid': 103, 'rank_union': 0.3, 'rank_union_rank': 4,
         'jaccard': 0.1, 'jaccard_rank': 4},
        # rank 5: untyped (NaN type) -> filtered even at rank 2
        {'target_bid': 104, 'rank_union': 0.45, 'rank_union_rank': 2.5,
         'jaccard': 0.25, 'jaccard_rank': 2},
    ]))

    out_map_by_type = {('q', 'T'): [10]}
    in_map = {100}
    rows, cand_rows = v._expand_out_map_sources(
        out_map_by_type, in_map,
        target_stats=None, target_bids=[100, 101, 102, 103],
        target_id2type={101: 'NEW1', 102: 'NEW2',
                        103: 'NEW3',
                        104: float('nan')},
        top_k=2)
    # the in-map claim 100 has no pool owner here -> still excluded
    assert cand_rows == []
    assert len(rows) == 2
    assert [r['target_bodyId'] for r in rows] == [101, 102]
    assert all(r['target_bodyId'] != 104 for r in rows)
    assert all(r['target_type'] in ('NEW1', 'NEW2') for r in rows)
    assert all('rank_union' in r and 'jaccard' in r for r in rows)
    events = [json.loads(l) for l in
              (tmp_path / 'pipeline_progress.jsonl').read_text().splitlines()]
    assert any(e['event'] == 'out_map_progress' for e in events)
