"""Coverage tests for comparison.profile_comparator.

Hermetic: all comparisons run on synthetic ConnectivityProfile objects built
directly in memory. No profiler network methods are exercised.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from comparison.connectivity_profiler import ConnectivityProfile, compute_ranks
from comparison.profile_comparator import (
    DEFAULT_SCORE_WEIGHTS,
    ComparisonResult,
    ConnectivityProfileComparer,
    ProfileComparator,
)

DS_A = 'flywire_FAFB_v783'
DS_B = 'male-cns:v1.0'


def _profile(bid, ds=DS_A, upstream=None, downstream=None, **kwargs):
    """Build a ConnectivityProfile from synthetic partner dicts."""
    upstream = upstream or {}
    downstream = downstream or {}
    params = dict(
        neuron_id=bid,
        dataset=ds,
        upstream_partners=upstream,
        downstream_partners=downstream,
        upstream_ranks=compute_ranks(upstream),
        downstream_ranks=compute_ranks(downstream),
        total_upstream_weight=float(sum(upstream.values())),
        total_downstream_weight=float(sum(downstream.values())),
    )
    params.update(kwargs)
    profile = ConnectivityProfile(**params)
    # Derive counts so _compute_connectivity_status classifies rich profiles
    # as COMPLETE instead of NONE (which would skip them in comparisons).
    if not upstream and not downstream:
        return profile
    profile.actual_upstream_count = len(upstream)
    profile.actual_downstream_count = len(downstream)
    profile.top_k_bodyid_used = min(len(upstream), len(downstream))
    profile.unique_types_upstream = len(upstream)
    profile.unique_types_downstream = len(downstream)
    profile._connectivity_status = profile._compute_connectivity_status().value
    profile.is_weak_connectivity = not profile.connectivity_status.is_valid_for_comparison()
    return profile


def _rich_profile(bid, ds=DS_A, scale=1.0):
    """A profile with 6 partners per direction (enough for rank correlation)."""
    up = {f'U{i}': float((10 - i) * scale) for i in range(1, 7)}
    down = {f'D{i}': float((8 - i) * scale) for i in range(1, 7)}
    return _profile(bid, ds=ds, upstream=up, downstream=down)


# ---------------------------------------------------------------------------
# ComparisonResult
# ---------------------------------------------------------------------------

def _result(**overrides):
    params = dict(
        profile_a_id='a', profile_b_id='b', dataset_a='d1', dataset_b='d2',
        direction='both', jaccard=0.8, cosine=0.9, rank_correlation=0.6,
        overlap_a_in_b=0.5, overlap_b_in_a=0.6,
    )
    params.update(overrides)
    return ComparisonResult(**params)


def test_comparison_result_to_dict_and_summary():
    r = _result(weak_connectivity_a=True, notes='some note')
    d = r.to_dict()
    assert d['profile_a'] == 'a'
    assert d['jaccard'] == 0.8
    assert d['weak_connectivity_warning'] is True
    assert d['notes'] == 'some note'
    r_nan = _result(rank_correlation=np.nan, rank_union=np.nan)
    d_nan = r_nan.to_dict()
    assert np.isnan(d_nan['rank_corr'])
    assert np.isnan(d_nan['rank_union'])

    s = r.summary()
    assert 'RankCorr' in s
    s_nan = r_nan.summary()
    assert 'NaN' in s_nan


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

def test_jaccard_similarity():
    empty_a, empty_b = _profile('a'), _profile('b')
    assert ProfileComparator.jaccard_similarity(empty_a, empty_b) == 0.0

    pa = _profile('a', upstream={'X': 1.0, 'Y': 2.0})
    pb = _profile('b', upstream={'Y': 1.0, 'Z': 1.0})
    assert ProfileComparator.jaccard_similarity(pa, pb, 'upstream') == pytest.approx(1 / 3)
    assert ProfileComparator.jaccard_similarity(pa, pa, 'upstream') == 1.0
    # direction with no partners
    assert ProfileComparator.jaccard_similarity(pa, pb, 'downstream') == 0.0


def test_weighted_cosine_similarity():
    empty_a, empty_b = _profile('a'), _profile('b')
    assert ProfileComparator.weighted_cosine_similarity(empty_a, empty_b) == 0.0

    pa = _profile('a', upstream={'X': 2.0, 'Y': 1.0})
    pb = _profile('b', upstream={'X': 4.0, 'Y': 2.0})
    # proportional weights -> cosine 1
    assert ProfileComparator.weighted_cosine_similarity(pa, pb, 'upstream') == pytest.approx(1.0)

    # one side empty -> zero norm -> 0
    pc = _profile('c')
    assert ProfileComparator.weighted_cosine_similarity(pa, pc, 'upstream') == 0.0

    # disjoint partners -> cosine 0
    pd_ = _profile('d', upstream={'Q': 1.0})
    assert ProfileComparator.weighted_cosine_similarity(pa, pd_, 'upstream') == pytest.approx(0.0)


def test_rank_correlation_identical_and_reversed():
    pa = _rich_profile('a')
    pb = _rich_profile('b')
    assert ProfileComparator.rank_correlation(pa, pb, 'upstream') == pytest.approx(1.0)
    assert ProfileComparator.rank_correlation(pa, pb, 'both') == pytest.approx(1.0)

    reversed_down = {f'D{i}': float(i) for i in range(1, 7)}  # increasing -> reversed ranks
    pb_rev = _profile('b', upstream=pb.upstream_partners, downstream=reversed_down)
    both_corr = ProfileComparator.rank_correlation(pa, pb_rev, 'both')
    assert both_corr == pytest.approx(0.0)  # (1 + -1) / 2

    # kendall also supported
    kt = ProfileComparator.rank_correlation(pa, pb, 'upstream', method='kendall')
    assert kt == pytest.approx(1.0)


def test_rank_correlation_nan_paths():
    # fewer than 2 total partners
    pa = _profile('a', upstream={'X': 1.0})
    pb = _profile('b')
    assert np.isnan(ProfileComparator.rank_correlation(pa, pb, 'upstream'))

    # constant rank arrays (all ties) -> NaN
    tied = {f'T{i}': 5.0 for i in range(6)}
    p1 = _profile('p1', upstream=tied)
    p2 = _profile('p2', upstream=tied)
    assert np.isnan(ProfileComparator.rank_correlation(p1, p2, 'upstream'))

    # both directions NaN -> NaN for 'both'
    assert np.isnan(ProfileComparator.rank_correlation(pa, pb, 'both'))


def test_rank_correlation_expansion_and_union_mode():
    # shared < 5 triggers union expansion path
    pa = _profile('a', upstream={'A': 6.0, 'B': 5.0, 'C': 4.0, 'D': 3.0})
    pb = _profile('b', upstream={'A': 6.0, 'B': 5.0, 'E': 4.0, 'F': 3.0})
    corr = ProfileComparator.rank_correlation(pa, pb, 'upstream')
    assert isinstance(corr, float) and -1.0 <= corr <= 1.0

    # use_all_partners mode with defaults for missing partners
    corr_all = ProfileComparator.rank_correlation(pa, pb, 'upstream', use_all_partners=True)
    assert isinstance(corr_all, float)
    # union < 3 in all-partners mode -> NaN
    small_a = _profile('a', upstream={'A': 1.0})
    small_b = _profile('b', upstream={'B': 1.0})
    assert np.isnan(ProfileComparator.rank_correlation(
        small_a, small_b, 'upstream', use_all_partners=True))


def test_rank_correlation_both_falls_back_to_valid_direction():
    # only upstream has data -> 'both' returns upstream correlation
    pa_down_empty = _profile('a', upstream=_rich_profile('a').upstream_partners)
    pb_down_empty = _profile('b', upstream=_rich_profile('b').upstream_partners)
    assert ProfileComparator.rank_correlation(
        pa_down_empty, pb_down_empty, 'both') == pytest.approx(1.0)


def test_overlap_fraction():
    pa = _profile('a', upstream={'X': 1.0, 'Y': 1.0})
    pb = _profile('b', upstream={'Y': 1.0, 'Z': 1.0, 'W': 1.0})
    ov_a, ov_b = ProfileComparator.overlap_fraction(pa, pb, 'upstream')
    assert ov_a == pytest.approx(0.5)
    assert ov_b == pytest.approx(1 / 3)
    empty = _profile('e')
    assert ProfileComparator.overlap_fraction(pa, empty, 'upstream') == (0.0, 0.0)


# ---------------------------------------------------------------------------
# combined_score
# ---------------------------------------------------------------------------

def test_combined_score_identical():
    pa = _rich_profile('a')
    pb = _rich_profile('b')
    scores = ProfileComparator.combined_score(pa, pb)
    assert scores['jaccard'] == 1.0
    assert scores['weighted_jaccard'] == pytest.approx(1.0)
    assert scores['cosine'] == pytest.approx(1.0)
    assert scores['rank'] == pytest.approx(1.0)
    assert scores['rank_union'] == pytest.approx(1.0)
    assert scores['overlap_avg'] == pytest.approx(1.0)
    assert 'combined' not in scores
    assert 'rank_norm' not in scores
    assert 'rank_union_norm' not in scores


def test_combined_score_empty_and_custom_weights():
    pa, pb = _profile('a'), _profile('b')
    scores = ProfileComparator.combined_score(pa, pb)
    assert scores['jaccard'] == 0.0
    assert scores['weighted_jaccard'] == 0.0
    assert np.isnan(scores['rank'])
    assert np.isnan(scores['rank_union'])
    assert 'combined' not in scores

    weights = {'jaccard': 1.0, 'rank': 0.0}
    pa2 = _profile('a', upstream={'X': 2.0})
    pb2 = _profile('b', upstream={'X': 4.0})
    scores2 = ProfileComparator.combined_score(pa2, pb2, weights=weights)
    assert scores2['jaccard'] == pytest.approx(1.0)
    assert scores2['weighted_jaccard'] == pytest.approx(0.5)  # min(2,4)/max(2,4)


# ---------------------------------------------------------------------------
# Expanded types (2-hop)
# ---------------------------------------------------------------------------

def test_get_expanded_types_1hop_only_when_typed_present():
    p = ConnectivityProfile(
        neuron_id='x', dataset=DS_A,
        upstream_partners={'A': 10.0}, upstream_ranks={'A': 1},
        total_upstream_weight=10.0,
        untyped_upstream_bodyids={7: 2.0},
        untyped_upstream_2hop={7: {'H2': 4.0}},
    )
    types = ProfileComparator._get_expanded_types(p, 'upstream')
    # typed partner present in top-N -> 2-hop NOT added
    assert types == {'A': 10.0}


def test_get_expanded_types_2hop_when_all_untyped():
    p = ConnectivityProfile(
        neuron_id='x', dataset=DS_A,
        untyped_upstream_bodyids={7: 6.0},
        untyped_upstream_2hop={7: {'H2': 3.0, 'H3': 1.0}},
    )
    types = ProfileComparator._get_expanded_types(p, 'upstream')
    # scaled = (hop2_weight / hop2_total) * untyped_weight
    assert types['2hop:H2'] == pytest.approx(4.5)
    assert types['2hop:H3'] == pytest.approx(1.5)
    types_noprefix = ProfileComparator._get_expanded_types(p, 'upstream', prefix_2hop=False)
    assert 'H2' in types_noprefix

    # downstream path as well
    p2 = ConnectivityProfile(
        neuron_id='y', dataset=DS_A,
        untyped_downstream_bodyids={8: 2.0},
        untyped_downstream_2hop={8: {'T1': 1.0}},
    )
    types_down = ProfileComparator._get_expanded_types(p2, 'both')
    assert types_down['2hop:T1'] == pytest.approx(2.0)


def test_get_expanded_types_standardized_with_mapper():
    p = _profile('x', upstream={'LocalType': 5.0})

    class FakeMapper:
        def standardize_partner_types(self, partners, source_dataset):
            return {f'canon_{k}': v for k, v in partners.items()}

    out_none = ProfileComparator._get_expanded_types_standardized(p, 'upstream', None)
    assert out_none == {'LocalType': 5.0}
    out = ProfileComparator._get_expanded_types_standardized(p, 'upstream', FakeMapper())
    assert out == {'canon_LocalType': 5.0}


# ---------------------------------------------------------------------------
# bodyId-level metrics (intra-dataset)
# ---------------------------------------------------------------------------

def _bodyid_profile(bid, typed_up=None, untyped_up=None, typed_down=None):
    return ConnectivityProfile(
        neuron_id=bid, dataset=DS_A,
        typed_upstream_bodyids=typed_up,
        untyped_upstream_bodyids=untyped_up,
        typed_downstream_bodyids=typed_down,
    )


def test_bodyid_jaccard():
    pa = _bodyid_profile('a', typed_up={1: 5.0, 2: 3.0, 3: 1.0})
    pb = _bodyid_profile('b', typed_up={2: 3.0, 3: 1.0, 4: 1.0})
    assert ProfileComparator.bodyid_jaccard(pa, pb, 'upstream') == pytest.approx(0.5)
    empty = _bodyid_profile('e')
    assert ProfileComparator.bodyid_jaccard(empty, empty, 'upstream') == 0.0
    # untyped bodyIds also count
    pc = _bodyid_profile('c', typed_up={1: 5.0}, untyped_up={9: 2.0})
    assert ProfileComparator.bodyid_jaccard(pa, pc, 'upstream') == pytest.approx(1 / 4)


def test_bodyid_rank_correlation():
    pa = _bodyid_profile('a', typed_up={1: 6.0, 2: 5.0, 3: 4.0, 4: 3.0})
    pb = _bodyid_profile('b', typed_up={1: 6.0, 2: 5.0, 3: 4.0, 4: 3.0})
    assert ProfileComparator.bodyid_rank_correlation(pa, pb, 'upstream') == pytest.approx(1.0)

    # fewer than 3 shared -> NaN
    pc = _bodyid_profile('c', typed_up={1: 6.0, 2: 5.0})
    assert np.isnan(ProfileComparator.bodyid_rank_correlation(pa, pc, 'upstream'))

    # constant weights -> NaN
    const_a = _bodyid_profile('ca', typed_up={1: 5.0, 2: 5.0, 3: 5.0})
    const_b = _bodyid_profile('cb', typed_up={1: 2.0, 2: 2.0, 3: 2.0})
    assert np.isnan(ProfileComparator.bodyid_rank_correlation(const_a, const_b, 'upstream'))


def test_bodyid_rank_correlation_union():
    pa = _bodyid_profile('a', typed_up={1: 6.0, 2: 5.0, 3: 4.0})
    pb = _bodyid_profile('b', typed_up={1: 6.0, 2: 5.0, 4: 3.0})
    corr = ProfileComparator.bodyid_rank_correlation_union(pa, pb, 'upstream')
    assert isinstance(corr, float)

    # union < 3 -> NaN
    small_a = _bodyid_profile('a', typed_up={1: 1.0})
    small_b = _bodyid_profile('b', typed_up={2: 1.0})
    assert np.isnan(ProfileComparator.bodyid_rank_correlation_union(small_a, small_b))


# ---------------------------------------------------------------------------
# Expanded-type metrics (cross-dataset)
# ---------------------------------------------------------------------------

def test_expanded_type_metrics():
    up = {f'T{i}': float(10 - i) for i in range(1, 7)}
    pa = _profile('a', upstream=up)
    pb = _profile('b', upstream=up)
    assert ProfileComparator.expanded_type_jaccard(pa, pb, 'upstream') == 1.0
    assert ProfileComparator.expanded_type_rank_correlation(pa, pb, 'upstream') == pytest.approx(1.0)
    assert ProfileComparator.expanded_type_rank_correlation_union(pa, pb, 'upstream') == pytest.approx(1.0)

    # disjoint sets
    pc = _profile('c', upstream={'Q1': 1.0, 'Q2': 2.0})
    assert ProfileComparator.expanded_type_jaccard(pa, pc, 'upstream') == 0.0
    # shared < 3 -> NaN
    assert np.isnan(ProfileComparator.expanded_type_rank_correlation(pa, pc, 'upstream'))

    empty_a, empty_b = _profile('e1'), _profile('e2')
    assert ProfileComparator.expanded_type_jaccard(empty_a, empty_b, 'upstream') == 0.0
    assert np.isnan(ProfileComparator.expanded_type_rank_correlation_union(empty_a, empty_b))


# ---------------------------------------------------------------------------
# Combined scores: intra / cross dataset
# ---------------------------------------------------------------------------

def test_combined_score_intra_dataset_empty():
    pa = _bodyid_profile('a', typed_up={1: 5.0})
    pb = _bodyid_profile('b')
    scores = ProfileComparator.combined_score_intra_dataset(pa, pb)
    assert 'combined' not in scores
    assert np.isnan(scores['jaccard'])
    assert scores['shared_type_count'] == 0


def test_combined_score_intra_dataset_full():
    typed = {i: float(10 - i) for i in range(1, 7)}
    pa = _bodyid_profile('a', typed_up=typed)
    pb = _bodyid_profile('b', typed_up=typed)
    scores = ProfileComparator.combined_score_intra_dataset(pa, pb, direction='upstream')
    assert scores['jaccard'] == pytest.approx(1.0)
    assert scores['weighted_jaccard'] == pytest.approx(1.0)
    assert scores['cosine'] == pytest.approx(1.0)
    assert scores['rank'] == pytest.approx(1.0)
    assert scores['rank_union'] == pytest.approx(1.0)
    assert 'combined' not in scores
    assert scores['shared_type_count'] == 6
    assert scores['union_type_count'] == 6


def test_combined_score_cross_dataset_empty_and_full():
    pa = _profile('a', upstream={'X': 1.0})
    pb = _profile('b')
    scores = ProfileComparator.combined_score_cross_dataset(pa, pb)
    assert 'combined' not in scores
    assert scores['union_type_count'] == 1

    up = {f'T{i}': float(10 - i) for i in range(1, 7)}
    pa2 = _profile('a', upstream=up)
    pb2 = _profile('b', upstream=up)
    scores2 = ProfileComparator.combined_score_cross_dataset(pa2, pb2, direction='upstream')
    assert scores2['jaccard'] == pytest.approx(1.0)
    assert 'combined' not in scores2
    assert scores2['shared_type_count'] == 6


# ---------------------------------------------------------------------------
# batch_compare_cross_dataset
# ---------------------------------------------------------------------------

class FakeMapper:
    """Minimal mapper canonicalizing type names."""

    def standardize_partner_types(self, partners, source_dataset):
        return {k.replace('_local', ''): v for k, v in partners.items()}


def test_batch_compare_cross_dataset_empty_source():
    src = _profile('src')
    assert ProfileComparator.batch_compare_cross_dataset(
        src, {1: _profile(1)}, {1: 5}) == []


def test_batch_compare_cross_dataset_variants():
    up = {f'T{i}': float(10 - i) for i in range(1, 7)}
    src = _profile('src', upstream=up)
    same = _profile(1, ds=DS_B, upstream=up)
    disjoint = _profile(2, ds=DS_B, upstream={'Q': 1.0})
    empty_target = _profile(3, ds=DS_B)

    targets = {1: same, 2: disjoint, 3: empty_target}
    candidate_map = {1: 10, 2: 8, 3: 4, 99: 1}  # 99 has no profile -> skipped

    results = ProfileComparator.batch_compare_cross_dataset(src, targets, candidate_map)
    assert len(results) == 3
    by_bid = {r['target_bid']: r for r in results}

    assert by_bid[1]['jaccard'] == pytest.approx(1.0)
    assert by_bid[1]['rank'] == pytest.approx(1.0)

    # no overlap -> jaccard 0
    assert by_bid[2]['jaccard'] == 0.0

    # empty target -> NaN row
    assert 'combined' not in by_bid[3]
    assert by_bid[3]['target_type_count'] == 0

    # with type mapper standardizing names
    up_local = {f'T{i}_local': float(10 - i) for i in range(1, 7)}
    src_local = _profile('src', upstream=up_local)
    results_mapped = ProfileComparator.batch_compare_cross_dataset(
        src_local, {1: same}, {1: 5}, type_mapper=FakeMapper())
    assert results_mapped[0]['jaccard'] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compare_profiles / compare_profiles_simple / overlap details / multiple
# ---------------------------------------------------------------------------

def test_compare_profiles():
    pa = _rich_profile('a')
    pb = _rich_profile('b', ds=DS_B)
    result = ProfileComparator.compare_profiles(pa, pb)
    assert isinstance(result, ComparisonResult)
    assert result.profile_a_id == 'a'
    assert result.dataset_b == DS_B
    assert result.rank_correlation == pytest.approx(1.0)

    # score_weights alias + NaN rank note
    tied = {f'T{i}': 5.0 for i in range(6)}
    p1 = _profile('p1', upstream=tied)
    p2 = _profile('p2', upstream=tied)
    result_nan = ProfileComparator.compare_profiles(
        p1, p2, direction='upstream', score_weights=DEFAULT_SCORE_WEIGHTS)
    assert np.isnan(result_nan.rank_correlation)
    assert 'constant' in result_nan.notes.lower()


def test_compare_profiles_simple():
    pa = _rich_profile('a')
    pb = _rich_profile('b')
    d = ProfileComparator.compare_profiles_simple(pa, pb)
    assert set(d.keys()) == {'jaccard', 'cosine', 'rank_correlation', 'shared_count'}
    assert d['shared_count'] == 12  # 6 up + 6 down shared
    d_up = ProfileComparator.compare_profiles_simple(pa, pb, direction='upstream')
    assert d_up['shared_count'] == 6


def test_get_partner_overlap_details():
    pa0 = _profile('a', upstream={'X': 3.0})
    pb0 = _profile('b')
    # Fixed: with an empty partner union the detail table is returned with
    # the documented schema instead of raising on the sort-key assignment.
    df0 = ProfileComparator.get_partner_overlap_details(pa0, pb0, 'downstream')
    assert df0.empty
    assert list(df0.columns) == ['partner_type', 'in_a', 'in_b', 'weight_a',
                                 'weight_b', 'rank_a', 'rank_b', 'status']

    pa = _profile('a', upstream={'X': 3.0, 'Y': 2.0, 'Z': 1.0})
    pb = _profile('b', upstream={'X': 1.0, 'W': 1.0})
    df = ProfileComparator.get_partner_overlap_details(pa, pb, 'upstream')
    assert list(df.columns) == ['partner_type', 'in_a', 'in_b', 'weight_a',
                                'weight_b', 'rank_a', 'rank_b', 'status']
    assert df.iloc[0]['status'] == 'shared'  # shared rows sorted first
    assert set(df['status']) == {'shared', 'a_only', 'b_only'}


def test_compare_multiple_profiles():
    profiles = {
        'a': _rich_profile('a'),
        'b': _rich_profile('b'),
        'c': _rich_profile('c', ds=DS_B),
    }
    df = ProfileComparator.compare_multiple_profiles(profiles)
    assert len(df) == 3
    assert 'pair' in df.columns
    assert set(df['pair']) == {'a vs b', 'a vs c', 'b vs c'}


# ---------------------------------------------------------------------------
# ConnectivityProfileComparer helpers (called unbound with synthetic data)
# ---------------------------------------------------------------------------

def test_compute_similarity_from_types():
    types_a = {'X': 5.0, 'Y': 3.0, 'Z': 1.0}
    same = ConnectivityProfileComparer._compute_similarity_from_types(None, types_a, types_a)
    assert same['jaccard'] == pytest.approx(1.0)
    assert same['weighted_jaccard'] == pytest.approx(1.0)
    assert same['cosine'] == pytest.approx(1.0)
    assert same['rank'] == pytest.approx(1.0)
    assert same['rank_union'] == pytest.approx(1.0)
    assert 'combined' not in same

    empty = ConnectivityProfileComparer._compute_similarity_from_types(None, {}, {})
    assert empty['jaccard'] == 0.0
    assert empty['cosine'] == 0.0
    assert 'rank_norm' not in empty
    assert 'combined' not in empty

    partial = ConnectivityProfileComparer._compute_similarity_from_types(
        None, types_a, {'X': 5.0, 'Q': 1.0})
    assert partial['jaccard'] == pytest.approx(1 / 4)
    assert np.isnan(partial['rank'])  # only 1 shared type (< 3)
    assert 'combined' not in partial


def test_same_name_flag():
    p = _rich_profile('a')
    per_ds = {DS_A: ('TypeX', p), DS_B: ('TypeX', p)}
    assert ConnectivityProfileComparer._same_name_flag(per_ds, [DS_A, DS_B]) == 1
    per_ds_diff = {DS_A: ('TypeX', p), DS_B: ('TypeY', p)}
    assert ConnectivityProfileComparer._same_name_flag(per_ds_diff, [DS_A, DS_B]) == 0
    per_ds_missing = {DS_A: ('TypeX', p)}
    assert ConnectivityProfileComparer._same_name_flag(per_ds_missing, [DS_A, DS_B]) == 0
    per_ds_none = {DS_A: (None, p), DS_B: ('TypeX', p)}
    assert ConnectivityProfileComparer._same_name_flag(per_ds_none, [DS_A, DS_B]) == 0


def test_comparer_compute_ranks():
    assert ConnectivityProfileComparer._compute_ranks({}) == {}
    ranks = ConnectivityProfileComparer._compute_ranks({'a': 3.0, 'b': 5.0})
    assert ranks == {'b': 1, 'a': 2}


# ---------------------------------------------------------------------------
# ProfileComparator: type-level similarity matrices
# ---------------------------------------------------------------------------

def test_build_similarity_matrix_metrics():
    profiles = {
        'T1': _rich_profile('T1'),
        'T2': _rich_profile('T2'),  # identical to T1
        'T3': _profile('T3', upstream={'Z9': 1.0}),  # disjoint
    }
    m = ProfileComparator.build_similarity_matrix(profiles, metric='rank')
    assert m.shape == (3, 3)
    assert np.allclose(np.diag(m.to_numpy()), 1.0)
    assert m.loc['T1', 'T2'] == pytest.approx(1.0)
    assert m.loc['T2', 'T1'] == pytest.approx(m.loc['T1', 'T2'])  # symmetric
    # disjoint types: jaccard 0, small residual from rank expansion
    assert m.loc['T1', 'T3'] < 0.2

    for metric in ['jaccard', 'cosine', 'rank']:
        mm = ProfileComparator.build_similarity_matrix(profiles, metric=metric)
        assert mm.loc['T1', 'T2'] == pytest.approx(1.0)


def test_compute_inter_type_rank_correlation_matrix():
    profiles_by_dataset = {
        DS_A: {'T1': _rich_profile('T1'), 'T2': _profile('T2', upstream={'Z': 1.0})},
        DS_B: {'T1': _rich_profile('T1')},
    }
    out = ProfileComparator.compute_inter_type_rank_correlation_matrix(
        profiles_by_dataset, metric='rank')
    assert DS_A in out['intra_dataset']          # >= 2 types -> intra matrix
    assert DS_B not in out['intra_dataset']      # single type -> skipped
    assert out['all_types_matrix'].shape == (3, 3)
    cross = out['cross_dataset']
    assert not cross.empty
    same = cross[cross['same_type']]
    assert same.iloc[0]['similarity'] == pytest.approx(1.0)

    # empty input -> empty results
    out_empty = ProfileComparator.compute_inter_type_rank_correlation_matrix({})
    assert out_empty['cross_dataset'] is None
    assert out_empty['intra_dataset'] == {}

    # other metrics run through the same paths
    for metric in ['rank', 'jaccard', 'cosine']:
        o = ProfileComparator.compute_inter_type_rank_correlation_matrix(
            profiles_by_dataset, metric=metric)
        assert not o['cross_dataset'].empty


# ---------------------------------------------------------------------------
# HomologFinder (hermetic: profiler replaced by a stub)
# ---------------------------------------------------------------------------

class _FakeProfilerForFinder:
    """Stub profiler covering the methods HomologFinder delegates to."""

    def __init__(self, profiles=None, bodyids=None, available=None):
        self.profiles = profiles or {}
        self.bodyids = bodyids or {}
        self.available = available or {}
        self._defer_cache_writes = False
        self._pending_cache_writes = {}
        self.calls = {'consolidate': 0, 'load_cache': 0, 'cached_conn': 0}

    def get_profile(self, neuron, dataset, force_refresh=False):
        return self.profiles.get((str(neuron), dataset))

    def get_bodyids_for_type(self, neuron_type, dataset):
        return self.bodyids.get((neuron_type, dataset))

    def get_available_types(self, dataset):
        return self.available.get(dataset)

    def ensure_data_available(self, dataset, raise_on_missing=True):
        if dataset not in self.available:
            raise RuntimeError(f'no data for {dataset}')
        return True

    def get_data_status(self, datasets):
        return {ds: {'available': ds in self.available} for ds in datasets}

    def consolidate_profile_cache(self, dataset):
        self.calls['consolidate'] += 1

    def _load_cache_dataframe(self, dataset, force_reload=False):
        self.calls['load_cache'] += 1

    def _get_cached_conn_df(self, dataset):
        self.calls['cached_conn'] += 1
        return None

    def flush_pending_cache_writes(self, silent=False):
        pass


@pytest.fixture
def finder(tmp_path):
    from comparison.profile_comparator import HomologFinder
    f = HomologFinder(output_dir=str(tmp_path), verbose=False)
    return f


def test_homolog_finder_type_mapper_variants(finder, tmp_path):
    from comparison.cross_dataset_type_mapper import CrossDatasetTypeMapper

    # disabled entirely
    finder.use_auto_type_mapping = False
    assert finder._get_type_mapper_for_comparison(True) is None

    # intra-dataset -> always None
    finder.use_auto_type_mapping = True
    assert finder._get_type_mapper_for_comparison(False) is None

    # pre-seeded loaded mapper is returned
    csv = tmp_path / 'n.csv'
    csv.write_text(
        "bodyId,type,flywireType,hemibrainType,mancType\n1,aMe12,MTe07,,\n",
        encoding='utf-8')
    loaded = CrossDatasetTypeMapper(neuron_df_path=str(csv), verbose=False)
    loaded.load()
    finder._type_mapper = loaded
    assert finder._get_type_mapper_for_comparison(True) is loaded

    # unloaded mapper -> None
    unloaded = CrossDatasetTypeMapper(
        neuron_df_path=str(tmp_path / 'missing.csv'), verbose=False)
    finder._type_mapper = unloaded
    assert finder._get_type_mapper_for_comparison(True) is None


def test_homolog_finder_profile_helpers(finder):
    p = _rich_profile('T', DS_A)
    fake = _FakeProfilerForFinder(
        profiles={('T', DS_A): p},
        bodyids={('T', DS_A): [11, 12]},
        available={DS_A: ['T']},
    )
    finder.profiler = fake

    assert finder.get_profile('T', DS_A) is p
    assert finder.get_profile('X', DS_A) is None

    assert finder.get_available_types(DS_A) == ['T']
    assert finder.get_available_types(DS_A) == ['T']  # cached
    assert finder.get_bodyids_for_type('T', DS_A) == [11, 12]
    assert finder.get_bodyids_for_type('T', DS_A) == [11, 12]  # cached
    assert finder.get_bodyids_for_type('Missing', DS_A) == []

    finder._prewarm_profile_cache(DS_A)
    assert fake.calls == {'consolidate': 1, 'load_cache': 1, 'cached_conn': 1}


def test_homolog_finder_get_profiles_batch(finder):
    p1, p2 = _rich_profile('A', DS_A), _rich_profile('B', DS_A)
    fake = _FakeProfilerForFinder(profiles={('A', DS_A): p1, ('B', DS_A): p2})
    finder.profiler = fake

    out = finder.get_profiles_batch(['A', 'B', 'Missing'], DS_A,
                                    show_progress=False)
    assert out == {'A': p1, 'B': p2}
    # deferred-write mode reset afterwards
    assert fake._defer_cache_writes is False


def test_homolog_finder_data_availability(finder):
    fake = _FakeProfilerForFinder(available={DS_A: ['T']})
    finder.profiler = fake
    finder.source_dataset = DS_A
    finder.target_dataset = DS_B

    avail = finder.ensure_data_available(raise_on_missing=False)
    assert avail == {DS_A: True, DS_B: False}

    with pytest.raises(RuntimeError):
        finder.ensure_data_available(raise_on_missing=True)

    # explicit dataset list
    assert finder.ensure_data_available([DS_A]) == {DS_A: True}

    status = finder.get_data_status()
    assert status[DS_A]['available'] is True
    assert status[DS_B]['available'] is False


def _install_fake_neuprint(monkeypatch, client_cls):
    """Register a fake ``neuprint`` module so client init is hermetic."""
    import sys
    import types

    fake = types.ModuleType('neuprint')
    fake.Client = client_cls
    monkeypatch.setitem(sys.modules, 'neuprint', fake)


def test_fetch_neuprint_available_datasets_excludes_hidden(monkeypatch):
    from comparison import profile_comparator as pc

    server = 'https://neuprint.test'
    monkeypatch.setattr(pc, '_NEUPRINT_AVAILABLE_CACHE', {})

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                'male-cns:v1.0': {'hidden': False},
                'hemibrain:v1.2.1': {},
                # hidden entries are the only ones excluded
                'banc:v888': {'hidden': 'true'},
            }

    calls = []
    monkeypatch.setattr(
        'requests.get',
        lambda url, headers=None, timeout=None: calls.append(url) or FakeResponse(),
    )

    result = pc.fetch_neuprint_available_datasets(server=server, token='tok')
    assert result == ['hemibrain:v1.2.1', 'male-cns:v1.0']
    assert len(calls) == 1

    # Cached within the TTL: no second HTTP call
    assert pc.fetch_neuprint_available_datasets(server=server, token='tok') == result
    assert len(calls) == 1


def test_fetch_neuprint_available_datasets_failure_paths(monkeypatch):
    from comparison import profile_comparator as pc
    from utils.token_manager import token_manager

    server = 'https://neuprint.test'
    cache = {}
    monkeypatch.setattr(pc, '_NEUPRINT_AVAILABLE_CACHE', cache)

    # No token anywhere -> None without hitting the network
    monkeypatch.setattr(token_manager, 'get_neuprint_token', lambda: None)
    assert pc.fetch_neuprint_available_datasets(server=server) is None

    # Network failure -> None and nothing cached (later calls retry)
    def boom(url, headers=None, timeout=None):
        raise RuntimeError('offline')

    monkeypatch.setattr('requests.get', boom)
    assert pc.fetch_neuprint_available_datasets(server=server, token='tok') is None
    assert cache == {}


def test_homolog_finder_skips_neuprint_client_for_flywire(tmp_path, capsys):
    from comparison.profile_comparator import HomologFinder

    finder = HomologFinder(
        source_dataset='flywire_FAFB_v783',
        target_dataset='banc_v888',
        output_dir=str(tmp_path),
        verbose=True,
    )
    out = capsys.readouterr().out
    assert finder.clients == {}
    assert finder.client is None
    # Local releases skip the client with an explicit reason (message from
    # commit 8dbcd8d9; the historical 'is a FlyWire dataset' text is gone).
    assert 'no NeuPrint client needed' in out
    # No misleading neuprint error for FlyWire identifiers
    assert 'does not exist' not in out
    assert 'Could not initialize client' not in out


def test_homolog_finder_graceful_for_unknown_neuprint_dataset(tmp_path, capsys, monkeypatch):
    from comparison import profile_comparator as pc

    class BoomClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError('Client must not be constructed for unknown datasets')

    _install_fake_neuprint(monkeypatch, BoomClient)
    monkeypatch.setattr(
        pc, 'fetch_neuprint_available_datasets',
        lambda token=None, **kwargs: ['hemibrain:v1.2.1', 'male-cns:v1.0'],
    )

    finder = pc.HomologFinder(
        source_dataset='hemibrain:v9.9.9',
        output_dir=str(tmp_path),
        verbose=True,
    )
    out = capsys.readouterr().out
    assert finder.clients == {}
    assert "does not exist on the NeuPrint server" in out
    assert "Available datasets: ['hemibrain:v1.2.1', 'male-cns:v1.0']" in out
    # Hidden server entries never leak into the available listing
    assert 'banc:v888' not in out


def test_homolog_finder_trims_long_client_error(tmp_path, capsys, monkeypatch):
    from comparison import profile_comparator as pc

    class FailingClient:
        def __init__(self, server, dataset=None, token=None):
            raise RuntimeError(
                f"Dataset '{dataset}' does not exist on the neuprint server. "
                "Available datasets: ['very', 'long', 'server', 'dump']"
            )

    _install_fake_neuprint(monkeypatch, FailingClient)
    # Listing unavailable -> fall through to the client constructor
    monkeypatch.setattr(
        pc, 'fetch_neuprint_available_datasets', lambda token=None, **kwargs: None
    )

    finder = pc.HomologFinder(
        source_dataset='hemibrain:v9.9.9',
        output_dir=str(tmp_path),
        verbose=True,
    )
    out = capsys.readouterr().out
    assert finder.clients == {}
    assert 'Could not initialize client for hemibrain:v9.9.9' in out
    # The long server-side dump is stripped from the warning
    assert 'Available datasets:' not in out


def test_homolog_finder_get_neuron_info(finder):
    class FakeFNC:
        def _fetch_neurons_local_or_api(self, ids, columns=None):
            return pd.DataFrame({'bodyId': ids})

        def _fetch_neurons_by_types(self, types, columns=None):
            return pd.DataFrame({'type': types})

    finder._fnc_cache[DS_A] = FakeFNC()
    df_bid = finder.get_neuron_info([1, 2], DS_A)
    assert list(df_bid['bodyId']) == [1, 2]
    df_type = finder.get_neuron_info(['T1'], DS_A)
    assert list(df_type['type']) == ['T1']

    # no FNC available -> empty frame
    assert finder.get_neuron_info([1], DS_B).empty


def test_resolve_queries_to_types_expands_coarse_label(finder):
    """A coarse / higher-category label must expand into one unit per type."""
    fake = _FakeProfilerForFinder(
        bodyids={('T1', DS_A): [11, 12], ('T2', DS_A): [20]},
    )
    fake.get_types_for_label = lambda label, ds: (
        {'R1': [1, 2], 'R2': [3]} if label == 'cellclass' else {}
    )
    finder.profiler = fake

    units = finder._resolve_queries_to_types(['cellclass'], DS_A)
    assert [u['type'] for u in units] == ['R1', 'R2']
    assert units[0]['query'] == 'cellclass'
    assert units[0]['bodyids'] == [1, 2]
    assert units[1]['bodyids'] == [3]

    # Plain exact type -> one unit holding all bodyIds of the type.
    units2 = finder._resolve_queries_to_types(['T1'], DS_A)
    assert len(units2) == 1
    assert units2[0]['type'] == 'T1'
    assert units2[0]['bodyids'] == [11, 12]

    # bodyId item -> one unit with a single bodyId.
    units3 = finder._resolve_queries_to_types([123], DS_A)
    assert units3[0]['type'] is None
    assert units3[0]['bodyids'] == [123]

    # Unresolvable item -> placeholder with no bodyIds (skipped downstream).
    units4 = finder._resolve_queries_to_types(['NoSuch'], DS_A)
    assert units4[0]['bodyids'] == []

    # Mixed list keeps order and expands the coarse label inline.
    units5 = finder._resolve_queries_to_types(['cellclass', 'T1', 'NoSuch'], DS_A)
    assert [u['type'] for u in units5] == ['R1', 'R2', 'T1', 'NoSuch']


def test_find_homologs_multi_writes_combined_grouped_output(finder, tmp_path, monkeypatch):
    """find_homologs_multi runs one search per resolved type and writes a single
    combined output folder grouped per query_type, with per-type subfolders."""
    fake = _FakeProfilerForFinder(
        bodyids={('T1', DS_A): [11, 12], ('T2', DS_A): [20]},
    )
    # 'cellclass' is a coarse label that expands into two real types.
    fake.get_types_for_label = lambda label, ds: (
        {'R1': [1, 2], 'R2': [3]} if label == 'cellclass' else {}
    )
    finder.profiler = fake
    finder.source_dataset = DS_A
    finder.target_dataset = DS_B

    seen_sources = []

    def fake_fast(**kwargs):
        src = kwargs.get('source')
        seen_sources.append(src)
        # Mimic the real per-type writer: create the by_type/<slug>/ folders
        # (results + visualization) that each per-type run produces.
        base = Path(kwargs['output_dir']) / kwargs['saveas']
        (base / 'results').mkdir(parents=True, exist_ok=True)
        (base / 'results' / 'bodyid_results.csv').write_text('dummy\n', encoding='utf-8')
        (base / 'visualization').mkdir(parents=True, exist_ok=True)
        return pd.DataFrame({
            'source_bodyId': [1, 2],
            'source_type': [src, src],
            'target_bodyId': [101, 102],
            'target_type': ['X', 'X'],
            'rank_corr': [0.9, 0.8],
            'jaccard': [0.7, 0.6],
            'cosine': [0.5, 0.4],
            'rank_union': [0.3, 0.2],
        })

    monkeypatch.setattr(finder, 'find_homologs_fast', fake_fast)

    out = finder.find_homologs_multi(
        source=['cellclass', 'T2'], output_dir=str(tmp_path), saveas='combined'
    )

    # Every resolved real type was searched (cellclass -> R1, R2; plus T2).
    assert seen_sources == ['R1', 'R2', 'T2']

    combined_path = tmp_path / 'combined'
    # Combined top-level outputs (written by the real _save_multi_results).
    assert (combined_path / 'README.txt').exists()
    assert (combined_path / 'results' / 'bodyid_results.csv').exists()
    assert (combined_path / 'results' / 'type_summary.csv').exists()
    assert (combined_path / 'results' / 'homolog_results.csv').exists()

    body_df = pd.read_csv(combined_path / 'results' / 'bodyid_results.csv')
    assert 'query_type' in body_df.columns
    summary_df = pd.read_csv(combined_path / 'results' / 'type_summary.csv')
    assert 'query_type' in summary_df.columns
    assert 'source_type' in summary_df.columns
    assert 'target_type' in summary_df.columns

    # A coarse label expands into several real-type units sharing the same
    # 'query' label; the combined outputs must list each query only once.
    readme = (combined_path / 'README.txt').read_text(encoding='utf-8')
    assert 'Query Types: cellclass, T2\n' in readme
    assert summary_df['query'].eq('cellclass,T2').all()

    # Per-type subfolders for every resolved type, each with per-type visualization.
    for sub in ['R1', 'R2', 'T2']:
        assert (combined_path / 'by_type' / sub / 'results').is_dir()
        assert (combined_path / 'by_type' / sub / 'visualization').is_dir()

    # The combined frame is grouped by query_type.
    assert set(out['query_type']) == {'cellclass', 'T2'}
    assert sorted(out['query_type'].drop_duplicates().tolist()) == ['T2', 'cellclass']


def test_find_homologs_multi_fetches_all_then_visualizes(finder, tmp_path, monkeypatch):
    """Refinement: all per-type searches are fetched first (visualization
    disabled), then each resolved type is visualized in a second phase."""
    fake = _FakeProfilerForFinder(
        bodyids={('T1', DS_A): [11, 12], ('T2', DS_A): [20]},
    )
    fake.get_types_for_label = lambda label, ds: {}
    finder.profiler = fake
    finder.source_dataset = DS_A
    finder.target_dataset = DS_B

    fetch_calls = []

    def fake_fast(**kwargs):
        fetch_calls.append((kwargs.get('source'), kwargs.get('visualize_skeleton')))
        return pd.DataFrame({
            'source_bodyId': [1],
            'source_type': [kwargs.get('source')],
            'target_bodyId': [101],
            'target_type': ['X'],
            'rank_corr': [0.5],
            'rank_union': [0.1],
            'jaccard': [0.2],
            'cosine': [0.3],
        })

    monkeypatch.setattr(finder, 'find_homologs_fast', fake_fast)
    viz_calls = []
    monkeypatch.setattr(
        finder, '_visualize_homolog_candidates',
        lambda **kw: viz_calls.append((kw['query'], str(kw['visualization_dir']))),
    )

    out = finder.find_homologs_multi(
        source=['T1', 'T2'], output_dir=str(tmp_path), saveas='combined',
        visualize_skeleton=True, visualize_top_n=2,
    )

    # Phase 1: every type fetched with visualization disabled.
    assert [src for src, _viz in fetch_calls] == ['T1', 'T2']
    assert all(viz is False for _src, viz in fetch_calls)

    # Phase 2: one visualization per resolved type, in its by_type folder.
    assert [q for q, _d in viz_calls] == ['T1', 'T2']
    for q, viz_dir in viz_calls:
        assert viz_dir.endswith(f'/by_type/{q}/visualization')

    # Combined output is unaffected.
    assert set(out['query_type']) == {'T1', 'T2'}


def test_save_multi_results_writes_combined_files(tmp_path):
    """The combined saver writes per-query-type grouped CSV outputs."""
    from comparison.profile_comparator import HomologFinder

    finder = HomologFinder(output_dir=str(tmp_path), verbose=False)
    df = pd.DataFrame({
        'query_type': ['T1', 'T1', 'T2'],
        'source_bodyId': [1, 2, 20],
        'source_type': ['T1', 'T1', 'T2'],
        'target_bodyId': [101, 102, 201],
        'target_type': ['X', 'X', 'Y'],
        'rank_corr': [0.9, 0.8, 0.7],
        'jaccard': [0.7, 0.6, 0.5],
        'cosine': [0.5, 0.4, 0.3],
        'rank_union': [0.3, 0.2, 0.1],
    })
    combined_path = tmp_path / 'combined'
    finder._save_multi_results(
        df, combined_path, DS_A, DS_B, ['T1', 'T2'], 'rank_corr'
    )

    assert (combined_path / 'results' / 'bodyid_results.csv').exists()
    assert (combined_path / 'results' / 'type_summary.csv').exists()
    assert (combined_path / 'results' / 'homolog_results.csv').exists()
    assert (combined_path / 'README.txt').exists()

    body_df = pd.read_csv(combined_path / 'results' / 'bodyid_results.csv')
    assert 'query_type' in body_df.columns

    summary_df = pd.read_csv(combined_path / 'results' / 'type_summary.csv')
    assert 'query_type' in summary_df.columns
    assert 'source_type' in summary_df.columns
    assert 'target_type' in summary_df.columns


def test_compare_types_bodyid_core(finder):
    def bodyid_profile(bid, ds, offset=0):
        up_bids = {100 + i + offset: float(10 - i) for i in range(6)}
        down_bids = {200 + i + offset: float(8 - i) for i in range(6)}
        return ConnectivityProfile(
            neuron_id=bid, dataset=ds,
            typed_upstream_bodyids=up_bids,
            typed_downstream_bodyids=down_bids,
            upstream_ranks={}, downstream_ranks={},
            total_upstream_weight=float(sum(up_bids.values())),
            total_downstream_weight=float(sum(down_bids.values())),
        )

    pa1 = bodyid_profile(1, DS_A)
    pa2 = bodyid_profile(2, DS_A)
    pb1 = bodyid_profile(3, DS_B)  # identical bodyId sets -> perfect match
    fake = _FakeProfilerForFinder(
        profiles={('1', DS_A): pa1, ('2', DS_A): pa2, ('3', DS_B): pb1},
        bodyids={('T', DS_A): [1, 2], ('T', DS_B): [3]},
    )
    finder.profiler = fake

    bodyid_df, type_summary = ProfileComparator.compare_types_bodyid_core(
        profiler=fake, neuron_types=['T'], dataset_a=DS_A, dataset_b=DS_B,
        min_common_partners=3, verbose=False)
    assert len(bodyid_df) == 2  # 2 x 1 bodyId pairs
    assert set(bodyid_df.columns) >= {'neuron_type', 'source_bodyId',
                                      'target_bodyId', 'jaccard', 'cosine',
                                      'rank_corr', 'rank_union'}
    assert bodyid_df['jaccard'].iloc[0] == pytest.approx(1.0)
    # Fixed behavior: overlap is computed from the same typed+untyped bodyId
    # sets the score uses, so identical profiles report full overlap and the
    # min_common_partners=3 filter keeps the rank metrics.
    assert bodyid_df['overlap_partners'].iloc[0] == 12
    assert bodyid_df['rank_corr'].notna().all()
    assert type_summary.iloc[0]['n_source_bodyIds'] == 2
    assert type_summary.iloc[0]['avg_jaccard'] == pytest.approx(1.0)

    # min_common_partners=0 keeps rank metrics
    bodyid_df_ok, _ = ProfileComparator.compare_types_bodyid_core(
        profiler=fake, neuron_types=['T'], dataset_a=DS_A, dataset_b=DS_B,
        min_common_partners=0, verbose=False)
    assert bodyid_df_ok['rank_corr'].notna().all()

    # type with no bodyIds anywhere -> empty results
    fake3 = _FakeProfilerForFinder()
    bodyid_df3, summary3 = ProfileComparator.compare_types_bodyid_core(
        profiler=fake3, neuron_types=['Ghost'], dataset_a=DS_A,
        dataset_b=DS_B, verbose=False)
    assert bodyid_df3.empty and summary3.empty


def test_homolog_finder_get_partner_overlap(finder):
    pa = _rich_profile('Q', DS_A)
    pb = _rich_profile('C', DS_B)
    fake = _FakeProfilerForFinder(
        profiles={('Q', DS_A): pa, ('C', DS_B): pb})
    finder.profiler = fake

    df = finder.get_partner_overlap('Q', DS_A, 'C', DS_B, direction='upstream')
    assert not df.empty

    # missing profile on either side -> empty frame
    assert finder.get_partner_overlap('Nope', DS_A, 'C', DS_B).empty


# ===========================================================================
# Extended coverage: hermetic fakes for coana / profiler internals
# ===========================================================================

import json
import os
import sys
import warnings as _warnings_mod
from types import ModuleType, SimpleNamespace

from comparison import connectivity_profiler as cp_module
from comparison import profile_comparator as pc_module
from comparison.profile_comparator import HomologFinder

SAFE_A = DS_A.replace(':', '_').replace('.', '_')
SAFE_B = DS_B.replace(':', '_').replace('.', '_')


class _FakeFNC:
    """Hermetic stand-in for coana.FindNeuronConnection."""

    instances = []
    write_cache_path = None     # Path: if set, build_connection_cache writes parquet here
    total_neurons = 5
    raise_on_build = False

    @classmethod
    def reset(cls):
        cls.instances = []
        cls.write_cache_path = None
        cls.total_neurons = 5
        cls.raise_on_build = False

    def __init__(self, dataset=None, token=None, use_cache=True,
                 verbose_mode='normal', simple_fetch=False):
        self.dataset = dataset
        self.token = token
        self.verbose_mode = verbose_mode
        self.simple_fetch = simple_fetch
        self._conn_df_cache = None
        self._conn_index = {}
        self._conn_index_post = {}
        self._in_progress_bar = False
        _FakeFNC.instances.append(self)

    def build_connection_cache(self, batch_size=1000, quiet=True):
        if _FakeFNC.raise_on_build:
            raise RuntimeError('boom')
        if _FakeFNC.write_cache_path is not None:
            path = _FakeFNC.write_cache_path
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                'bodyId_pre': [1, 2],
                'bodyId_post': [2, 3],
                'weight': [5.0, 4.0],
            }).to_parquet(path)
        return {
            'total_neurons': _FakeFNC.total_neurons,
            'already_cached': 2,
            'newly_cached': 0,
            'total_connections': 2,
        }


def _install_fake_coana(monkeypatch, fnc_cache=None):
    """Register a fake top-level `coana` module used by every FNC import chain."""
    mod = ModuleType('coana')
    mod.FindNeuronConnection = _FakeFNC
    mod._FNC_CACHE = fnc_cache if fnc_cache is not None else {}
    monkeypatch.setitem(sys.modules, 'coana', mod)
    return mod


@pytest.fixture(autouse=True)
def _reset_fake_fnc():
    _FakeFNC.reset()
    yield
    _FakeFNC.reset()


class _FakeProfilerFull(_FakeProfilerForFinder):
    """Stub profiler extended with the methods used by finder/comparer flows."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = SimpleNamespace(
            top_k_bodyid=15, top_m_type=5, include_untyped_partners=True)
        self._in_progress_bar = False
        self.from_cache = {}          # (bid, dataset) -> cached profile
        self.types_for_bodyids = {}   # dataset -> {bid: type}
        self.type_lists = {}          # (pattern, dataset) -> [types]
        self.label_groups = {}        # (label, dataset) -> {type: [ids]}

    def _load_from_cache(self, bid, dataset, required_top_k=15):
        return self.from_cache.get((bid, dataset))

    def get_types_for_bodyids(self, bids, dataset):
        lookup = self.types_for_bodyids.get(dataset, {})
        return {b: lookup[b] for b in bids if b in lookup}

    def list_types(self, pattern, dataset):
        return list(self.type_lists.get((pattern, dataset), []))

    def get_types_for_label(self, label, dataset):
        return dict(self.label_groups.get((label, dataset), {}))

    def get_profiles_batch(self, ids, dataset, force_refresh=False,
                           skip_profile_cache=False, show_progress=False):
        out = {}
        for nid in ids:
            p = self.profiles.get((str(nid), dataset))
            if p is not None:
                out[nid] = p
        return out


class _FakeLabelMapper:
    def __init__(self, mapping):
        self.mapping = mapping  # (name, dataset) -> mapped

    def get_mapped_label(self, name, dataset):
        return self.mapping.get((name, dataset))


@pytest.fixture
def pc_fake_repo(tmp_path, monkeypatch):
    """Redirect Path(__file__) lookups into tmp_path (project root)."""
    fake_file = tmp_path / 'src' / 'comparison' / 'profile_comparator.py'
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text('', encoding='utf-8')
    monkeypatch.setattr(pc_module, '__file__', str(fake_file))
    return tmp_path


# ---------------------------------------------------------------------------
# ProfileComparator.direct_comparison (static)
# ---------------------------------------------------------------------------

def _dc_profiler():
    return _FakeProfilerFull(
        profiles={
            ('T1', DS_A): _rich_profile('T1', DS_A),
            ('T1', DS_B): _rich_profile('T1', DS_B),
            ('T2', DS_B): _profile('T2', ds=DS_B, upstream={'Z': 1.0}),
        },
        available={DS_A: ['T1'], DS_B: ['T1', 'T2']},
    )


def test_direct_comparison_static_loose_mode(monkeypatch):
    _install_fake_coana(monkeypatch)
    fake = _dc_profiler()
    out = ProfileComparator.direct_comparison(
        'T1', 'T1', dataset_a=DS_A, dataset_b=DS_B,
        profiler=fake, verbose=False)
    assert out['comparison_mode'] == 'loose'
    assert not out['results'].empty
    row = out['results'].iloc[0]
    assert row['rank_corr'] == pytest.approx(1.0)
    assert out['summary']['n_comparisons'] == 1
    assert out['summary']['same_type_matches'] == 1
    # legacy alias 'type' normalizes to loose
    out2 = ProfileComparator.direct_comparison(
        'T1', ['T1', 'T2'], source_dataset=DS_A, target_dataset=DS_B,
        profiler=fake, comparison_mode='type', verbose=False)
    assert len(out2['results']) == 2


def test_direct_comparison_static_label_mapper(monkeypatch):
    _install_fake_coana(monkeypatch)
    fake = _dc_profiler()
    mapper = _FakeLabelMapper({('T1x', DS_A): 'T1', ('T1x', DS_B): 'T1'})
    out = ProfileComparator.direct_comparison(
        'T1x', None, dataset_a=DS_A, dataset_b=DS_B,
        profiler=fake, label_mapper=mapper, verbose=False)
    assert not out['results'].empty
    assert out['results'].iloc[0]['type_a'] == 'T1'


def test_direct_comparison_static_strict_types(monkeypatch):
    _install_fake_coana(monkeypatch)

    def bodyid_profile(bid, ds):
        up = {100 + i: float(10 - i) for i in range(6)}
        down = {200 + i: float(8 - i) for i in range(6)}
        return ConnectivityProfile(
            neuron_id=bid, dataset=ds,
            typed_upstream_bodyids=up, typed_downstream_bodyids=down,
            upstream_ranks={}, downstream_ranks={},
            total_upstream_weight=float(sum(up.values())),
            total_downstream_weight=float(sum(down.values())),
        )

    fake = _FakeProfilerFull(
        profiles={
            ('1', DS_A): bodyid_profile(1, DS_A),
            ('3', DS_B): bodyid_profile(3, DS_B),
        },
        bodyids={('T', DS_A): [1], ('T', DS_B): [3]},
    )
    # Cross-dataset (F-XD-001): per-bodyId scoring intersects partner
    # bodyIds across two ID spaces — not a defined metric — so strict mode
    # compares at the type level with standardized partner names instead.
    out = ProfileComparator.direct_comparison(
        'T', 'T', dataset_a=DS_A, dataset_b=DS_B,
        profiler=fake, comparison_mode='strict', verbose=False)
    assert out['comparison_mode'] == 'strict'
    assert 'bodyid_results' not in out
    # Same-dataset strict still delegates to the per-bodyId core.
    fake_same = _FakeProfilerFull(
        profiles={
            ('1', DS_A): bodyid_profile(1, DS_A),
            ('3', DS_A): bodyid_profile(3, DS_A),
        },
        bodyids={('T', DS_A): [1, 3]},
    )
    out_same = ProfileComparator.direct_comparison(
        'T', 'T', dataset_a=DS_A, dataset_b=DS_A,
        profiler=fake_same, comparison_mode='strict', verbose=False)
    assert out_same['comparison_mode'] == 'strict'
    assert 'type_summary' in out_same and 'bodyid_results' in out_same
    assert len(out_same['results']) == 1
    assert out_same['results'].iloc[0]['n_source_bodyIds'] >= 1


def test_direct_comparison_static_all_bodyid(monkeypatch):
    _install_fake_coana(monkeypatch)
    fake = _FakeProfilerFull(
        profiles={
            ('5', DS_A): _rich_profile(5, DS_A),
            ('101', DS_B): _rich_profile(101, DS_B),
        },
    )
    out = ProfileComparator.direct_comparison(
        [5], [101], dataset_a=DS_A, dataset_b=DS_B,
        profiler=fake, comparison_mode='loose', verbose=False)
    # all-int input forces strict mode even when 'loose' requested
    assert out['comparison_mode'] == 'strict'
    assert not out['results'].empty


def test_direct_comparison_static_empty_source_nan_rows(monkeypatch):
    _install_fake_coana(monkeypatch)
    fake = _FakeProfilerFull(
        profiles={
            ('Empty', DS_A): _profile('Empty', DS_A),
            ('T1', DS_B): _rich_profile('T1', DS_B),
            ('T2', DS_B): _rich_profile('T2', DS_B),
        },
    )
    out = ProfileComparator.direct_comparison(
        'Empty', ['T1', 'T2'], dataset_a=DS_A, dataset_b=DS_B,
        profiler=fake, verbose=False)
    # empty source -> NaN rows for every target
    assert len(out['results']) == 2
    assert out['results']['rank_corr'].isna().all()

    # same_label_only filters non-matching labels
    out_same = ProfileComparator.direct_comparison(
        'Empty', ['T1'], dataset_a=DS_A, dataset_b=DS_B,
        profiler=fake, same_label_only=True, verbose=False)
    assert out_same['results'].empty

    # insufficient profiles on one side -> empty return dict
    out_none = ProfileComparator.direct_comparison(
        'Empty', 'Ghost', dataset_a=DS_A, dataset_b=DS_B,
        profiler=fake, verbose=False)
    assert out_none['results'].empty
    assert out_none['summary'] == {}


def test_direct_comparison_static_missing_datasets():
    with pytest.raises(ValueError):
        ProfileComparator.direct_comparison('A', 'B', dataset_a=DS_A)


def test_homolog_finder_direct_comparison_wrapper(finder, monkeypatch, tmp_path):
    _install_fake_coana(monkeypatch)
    fake = _dc_profiler()
    finder.profiler = fake
    finder.source_dataset = DS_A
    finder.target_dataset = DS_B

    out = finder.direct_comparison('T1', 'T1', output_dir=str(tmp_path),
                                   save_results=True)
    assert not out['results'].empty
    assert os.path.exists(out['output_file'])
    params_path = out['output_file'].replace('.csv', '_params.json')
    assert os.path.exists(params_path)
    with open(params_path) as f:
        params = json.load(f)
    assert params['dataset_a'] == DS_A

    # missing datasets -> ValueError
    finder.source_dataset = None
    finder.target_dataset = None
    with pytest.raises(ValueError):
        finder.direct_comparison('T1', 'T1')


# ---------------------------------------------------------------------------
# HomologFinder._get_target_bodyids_and_types
# ---------------------------------------------------------------------------

def test_get_target_bodyids_and_types_neuron_index(finder, pc_fake_repo):
    idx_dir = pc_fake_repo / 'neuron_indexes' / SAFE_A
    idx_dir.mkdir(parents=True)
    pd.DataFrame({'bodyId': [1, 2], 'type': ['Cand', None]}).to_parquet(
        idx_dir / 'neuron_index.parquet')
    bodyids, lookup = finder._get_target_bodyids_and_types(DS_A)
    assert bodyids == [1, 2]
    assert lookup[1] == 'Cand'
    assert lookup[2] == ''


def test_get_target_bodyids_and_types_csv_fallback(finder, pc_fake_repo):
    ds_dir = pc_fake_repo / 'datasets' / SAFE_B
    ds_dir.mkdir(parents=True)
    (ds_dir / f'{SAFE_B}_allneurons_neuron_df.csv').write_text(
        'bodyId,type\n101,Cand\n102,\n', encoding='utf-8')
    bodyids, lookup = finder._get_target_bodyids_and_types(DS_B)
    assert bodyids == [101, 102]
    assert lookup[101] == 'Cand'
    assert lookup[102] == ''


def test_get_target_bodyids_and_types_missing(finder, pc_fake_repo):
    bodyids, lookup = finder._get_target_bodyids_and_types(DS_A)
    assert bodyids == [] and lookup == {}


# ---------------------------------------------------------------------------
# HomologFinder._build_profiles_batch / _build_profiles_memory_safe
# ---------------------------------------------------------------------------

def test_build_profiles_batch_cached_and_built(finder):
    p_cached = _rich_profile(11, DS_A)
    p_built = _rich_profile(12, DS_A)
    fake = _FakeProfilerFull(
        profiles={('12', DS_A): p_built},
        available={DS_A: ['T']},
    )
    fake.from_cache = {(11, DS_A): p_cached}
    finder.profiler = fake
    finder._batch_size = 1  # force mid-batch flush branch

    out = finder._build_profiles_batch([11, 12], DS_A, show_progress=False)
    assert out == {11: p_cached, 12: p_built}
    assert fake.calls['consolidate'] >= 2  # pre + final consolidation
    assert fake._defer_cache_writes is False

    # everything cached -> early return without building
    fake2 = _FakeProfilerFull(available={DS_A: ['T']})
    fake2.from_cache = {(11, DS_A): p_cached}
    finder.profiler = fake2
    out2 = finder._build_profiles_batch([11], DS_A, show_progress=False)
    assert out2 == {11: p_cached}


def test_build_profiles_memory_safe_clears_caches(finder, monkeypatch):
    coana_mod = _install_fake_coana(monkeypatch)
    p = _rich_profile(11, DS_A)
    fake = _FakeProfilerFull(
        profiles={('11', DS_A): p}, available={DS_A: ['T']})
    finder.profiler = fake
    finder._conn_cache[DS_A] = pd.DataFrame({'x': [1]})
    cp_module._PROFILER_CONN_CACHE[SAFE_A] = {'conn_df': 'junk'}
    coana_mod._FNC_CACHE[SAFE_A] = {'conn_df': 'junk'}
    try:
        out = finder._build_profiles_memory_safe(
            [11], DS_A, show_progress=False)
        assert out == {11: p}
        assert cp_module._PROFILER_CONN_CACHE[SAFE_A] == {}
        assert coana_mod._FNC_CACHE[SAFE_A] == {}
        assert DS_A not in finder._conn_cache
    finally:
        cp_module._PROFILER_CONN_CACHE.pop(SAFE_A, None)


def test_build_profiles_memory_safe_ensure_cache_complete(finder, monkeypatch):
    _install_fake_coana(monkeypatch)
    fake = _FakeProfilerFull(available={DS_A: ['T']})
    finder.profiler = fake
    finder.ensure_cache_complete = True
    out = finder._build_profiles_memory_safe([99], DS_A, show_progress=False)
    assert out == {}  # no profile available for 99
    assert len(_FakeFNC.instances) == 1  # cache completeness check ran


# ---------------------------------------------------------------------------
# HomologFinder.find_homologs (end-to-end with synthetic data)
# ---------------------------------------------------------------------------

def _finder_for_homologs(finder, monkeypatch, target_bodyids=(101, 102)):
    fake = _FakeProfilerFull(
        profiles={
            ('Q', DS_A): _rich_profile('Q', DS_A),
            ('5', DS_A): _rich_profile(5, DS_A),
            ('101', DS_B): _rich_profile(101, DS_B),
            ('102', DS_B): _rich_profile(102, DS_B),
            ('Cand', DS_B): _rich_profile('Cand', DS_B),
        },
        bodyids={('Q', DS_A): [5]},
        available={DS_A: ['Q'], DS_B: ['Cand']},
    )
    finder.profiler = fake
    finder.source_dataset = DS_A
    finder.target_dataset = DS_B
    finder.use_auto_type_mapping = False
    lookup = {101: 'Cand', 102: ''}
    monkeypatch.setattr(
        finder, '_get_target_bodyids_and_types',
        lambda dataset: (list(target_bodyids), dict(lookup)))
    return fake


def test_find_homologs_end_to_end(finder, monkeypatch, tmp_path):
    _finder_for_homologs(finder, monkeypatch)
    df = finder.find_homologs(source='Q', show_progress=False,
                              include_partner_details=False)
    assert not df.empty
    assert {'source_bodyId', 'target_bodyId', 'rank_corr'} <= set(df.columns)
    assert set(df['target_bodyId']) == {101, 102}
    # identical synthetic partners -> perfect similarity
    assert df['rank_corr'].max() == pytest.approx(1.0)
    # morphological enrichment is disabled: no morph columns on results
    assert not {'morph_v2_similarity', 'morph_nblast'} & set(df.columns)
    # results saved under the default output dir
    assert list(tmp_path.iterdir())


def test_find_homologs_shuffle_branch(finder, monkeypatch):
    _finder_for_homologs(finder, monkeypatch)
    stats = {
        'p_value': 0.01, 'z_score': 2.5, 'effect_size': 1.1,
        'is_significant': True, 'summary': 'sig',
    }
    monkeypatch.setattr(finder, 'run_random_control_test',
                        lambda **kwargs: dict(stats))
    df = finder.find_homologs(source='Q', show_progress=False,
                              include_partner_details=False,
                              run_shuffle_test=True, n_shuffles=3)
    assert 'shuffle_p_value' in df.columns
    assert df['shuffle_significant'].all()


def test_find_homologs_validation_and_empty(finder, monkeypatch):
    _finder_for_homologs(finder, monkeypatch)
    finder.source_dataset = None
    finder.target_dataset = None
    with pytest.raises(ValueError):
        finder.find_homologs()
    with pytest.raises(ValueError):
        finder.find_homologs(source='Q')
    with pytest.raises(ValueError):
        finder.find_homologs(source='Q', source_dataset=DS_A)

    # unknown type -> no bodyIds -> empty frame
    df = finder.find_homologs(source='Ghost', source_dataset=DS_A,
                              target_dataset=DS_B, show_progress=False)
    assert df.empty


def test_find_novel_homologs_delegates(finder, monkeypatch):
    calls = {}

    def fake_find(**kwargs):
        calls.update(kwargs)
        return pd.DataFrame({'x': [1]})

    monkeypatch.setattr(finder, 'find_homologs', fake_find)
    out = finder.find_novel_homologs('Q', DS_A, top_n=5, min_score=0.4)
    assert calls['source_dataset'] == DS_A and calls['target_dataset'] == DS_A
    assert calls['top_n'] == 5 and calls['min_score'] == 0.4
    assert not out.empty


# ---------------------------------------------------------------------------
# HomologFinder._compare_candidates_core
# ---------------------------------------------------------------------------

def _core_inputs(include_weak=False):
    src_profiles = {5: _rich_profile(5, DS_A)}
    status = {5: src_profiles[5].connectivity_status}
    if include_weak:
        src_profiles[6] = _profile(6, DS_A)  # empty -> NONE status
        status[6] = src_profiles[6].connectivity_status
    tgt_profiles = {101: _rich_profile(101, DS_B),
                    102: _rich_profile(102, DS_B)}
    return src_profiles, status, tgt_profiles


def test_compare_candidates_core_cross(finder):
    src_profiles, status, tgt_profiles = _core_inputs(include_weak=True)
    candidate_map = {bid: {101: 3, 102: 1} for bid in src_profiles}
    results, intra_df, skipped, warned, s_map, t_map, t_counts = \
        finder._compare_candidates_core(
            source_bodyids=list(src_profiles),
            source_profiles_cache=src_profiles,
            source_status_map=status,
            target_profiles_cache=tgt_profiles,
            target_type_lookup={101: 'Cand', 102: ''},
            source_type_lookup={5: 'Q', 6: 'Q'},
            candidate_map=candidate_map,
            is_cross_dataset=True,
            target_dataset=DS_B,
            show_progress=False,
            similarity_metric='rank_corr',
            top_n=5,
            include_intra_type=False,
        )
    assert not results.empty
    assert set(results['target_bodyId']) == {101, 102}
    assert skipped['none'] == [6]  # empty source skipped
    assert intra_df.empty
    assert t_map[101] == tgt_profiles[101].connectivity_status


def test_compare_candidates_core_vector_prefilter_and_min_score(finder):
    src_profiles, status, tgt_profiles = _core_inputs()
    candidate_map = {5: {101: 3, 102: 1}}
    # prune to top fraction by adjacency count
    results, *_ = finder._compare_candidates_core(
        source_bodyids=[5], source_profiles_cache=src_profiles,
        source_status_map=status, target_profiles_cache=tgt_profiles,
        target_type_lookup={101: 'Cand', 102: ''},
        source_type_lookup={5: 'Q'}, candidate_map=candidate_map,
        is_cross_dataset=True, target_dataset=DS_B, show_progress=False,
        similarity_metric='rank_corr', top_n=5,
        vector_prefiltering=True, vector_prune_fraction=0.5)
    assert list(results['target_bodyId']) == [101]

    # fraction >= 1.0 keeps every cosine-positive candidate
    results_all, *_ = finder._compare_candidates_core(
        source_bodyids=[5], source_profiles_cache=src_profiles,
        source_status_map=status, target_profiles_cache=tgt_profiles,
        target_type_lookup={101: 'Cand', 102: ''},
        source_type_lookup={5: 'Q'}, candidate_map=candidate_map,
        is_cross_dataset=True, target_dataset=DS_B, show_progress=False,
        similarity_metric='rank_corr', top_n=5,
        vector_prefiltering=True, vector_prune_fraction=1.0)
    assert len(results_all) == 2

    # min_score filter removes lower matches
    results_min, *_ = finder._compare_candidates_core(
        source_bodyids=[5], source_profiles_cache=src_profiles,
        source_status_map=status, target_profiles_cache=tgt_profiles,
        target_type_lookup={101: 'Cand', 102: ''},
        source_type_lookup={5: 'Q'}, candidate_map=candidate_map,
        is_cross_dataset=True, target_dataset=DS_B, show_progress=False,
        similarity_metric='rank_corr', top_n=5, min_score=0.999)
    assert len(results_min) >= 1
    assert (results_min['rank_corr'] >= 0.999).all()


def test_compare_candidates_core_guaranteed_same_type(finder):
    """Same-type homologs bypass vector-prefilter pruning."""
    src_profiles, status, tgt_profiles = _core_inputs()
    # Give 102 strictly lower cosine than 101 (same partners as the source
    # plus a dominant unrelated ones), so the default cosine-ranked prune
    # deterministically drops it (top 50% of 2 -> keep 1).
    tgt_profiles[102] = _profile(
        102, DS_B,
        upstream={'U1': 9.0, 'U2': 8.0, 'W1': 60.0},
        downstream={'D1': 7.0, 'D2': 6.0, 'W2': 50.0})
    # Prune keeps only the top candidate (101), but 102 is flagged same-type
    # under pruning.
    candidate_map = {5: {101: 3, 102: 1}}
    results, *_ = finder._compare_candidates_core(
        source_bodyids=[5], source_profiles_cache=src_profiles,
        source_status_map=status, target_profiles_cache=tgt_profiles,
        target_type_lookup={101: 'Cand', 102: 'Q'},
        source_type_lookup={5: 'Q'}, candidate_map=candidate_map,
        is_cross_dataset=True, target_dataset=DS_B, show_progress=False,
        similarity_metric='rank_corr', top_n=5,
        vector_prefiltering=True, vector_prune_fraction=0.5,
        guaranteed_candidates={5: {102}})
    assert set(results['target_bodyId']) == {101, 102}
    row = results[results['target_bodyId'] == 102].iloc[0]
    # adjacency_score preserved from the original candidate map
    assert int(row['adjacency_score']) == 1
    assert bool(row['is_same_type'])

    # Guaranteed bids absent from the profile cache are silently ignored.
    results2, *_ = finder._compare_candidates_core(
        source_bodyids=[5], source_profiles_cache=src_profiles,
        source_status_map=status, target_profiles_cache=tgt_profiles,
        target_type_lookup={101: 'Cand', 102: 'Q'},
        source_type_lookup={5: 'Q'}, candidate_map=candidate_map,
        is_cross_dataset=True, target_dataset=DS_B, show_progress=False,
        similarity_metric='rank_corr', top_n=5,
        vector_prefiltering=True, vector_prune_fraction=0.5,
        guaranteed_candidates={5: {102, 999}})
    assert set(results2['target_bodyId']) == {101, 102}

    # Without prefiltering nothing changes for the unflagged candidates.
    results3, *_ = finder._compare_candidates_core(
        source_bodyids=[5], source_profiles_cache=src_profiles,
        source_status_map=status, target_profiles_cache=tgt_profiles,
        target_type_lookup={101: 'Cand', 102: 'Q'},
        source_type_lookup={5: 'Q'}, candidate_map=candidate_map,
        is_cross_dataset=True, target_dataset=DS_B, show_progress=False,
        similarity_metric='rank_corr', top_n=5,
        vector_prefiltering=False,
        guaranteed_candidates=None)
    assert set(results3['target_bodyId']) == {101, 102}


def test_compare_candidates_core_prefilter_rank_metric(finder):
    """The prune step ranks by weighted cosine by default; 'adjacency' opts out."""
    src = _profile(9, DS_A,
                   upstream={'U1': 100.0, 'U2': 90.0},
                   downstream={'D1': 80.0, 'D2': 70.0})
    # Hub candidate: shares all four source partner TYPES (adjacency 4) but
    # its weight mass sits on unrelated partners -> low cosine.
    hub = _profile(101, DS_B,
                   upstream={'U1': 1.0, 'U2': 1.0, 'X1': 5000.0},
                   downstream={'D1': 1.0, 'D2': 1.0, 'X2': 4000.0})
    # Near candidate: shares fewer types (adjacency 2) but with matching
    # weights -> high cosine.
    near = _profile(102, DS_B,
                    upstream={'U1': 95.0, 'U3': 40.0},
                    downstream={'D1': 85.0, 'D4': 35.0})
    cos_hub = ProfileComparator.weighted_cosine_similarity(src, hub, 'both')
    cos_near = ProfileComparator.weighted_cosine_similarity(src, near, 'both')
    assert cos_near > cos_hub

    common = dict(
        source_bodyids=[9], source_profiles_cache={9: src},
        source_status_map={9: src.connectivity_status},
        source_type_lookup={9: 'Q'},
        target_profiles_cache={101: hub, 102: near},
        target_type_lookup={101: 'Hub', 102: 'Near'},
        candidate_map={9: {101: 4, 102: 2}},
        is_cross_dataset=True, target_dataset=DS_B, show_progress=False,
        similarity_metric='rank_corr', top_n=5,
        vector_prefiltering=True, vector_prune_fraction=0.5)

    res_cos, *_ = finder._compare_candidates_core(**common)
    assert list(res_cos['target_bodyId']) == [102]

    res_adj, *_ = finder._compare_candidates_core(
        **common, prefilter_rank_metric='adjacency')
    assert list(res_adj['target_bodyId']) == [101]

    # Invalid metric names are rejected at construction time.
    from comparison.profile_comparator import HomologFinder
    with pytest.raises(ValueError):
        HomologFinder(output_dir=str(finder.output_dir),
                      prefilter_rank_metric='cosnie')


def test_expand_partner_types_snapshot_reuse_and_dedupe(finder):
    """P2/P1.4: one snapshot per run reused across expansion calls, and
    unique-resolution counting deduped across calls."""
    import comparison.type_resolver as tr_mod

    class _Mapper:
        _loaded = True

        def _get_type_mapping_key(self, ds):
            return ds

        def get_mapping_decision(self, t, src, dst, include_bridges=False):
            return {'status': 'mapped', 'source_type': t,
                    'target_type': f'{t}_mapped',
                    'target_types': [f'{t}_mapped'],
                    'relationship': '1-to-1', 'conflicts': []}

        def get_alias_candidates(self, type_name, datasets,
                                 source_dataset=None):
            return {ds: {'outcome': 'no counterpart known',
                         'candidates': []} for ds in datasets}

        def get_type_bridges(self, *a, **k):
            return []

    calls = {'n': 0}
    real = tr_mod.get_mapper_snapshot

    def _counting_snapshot(type_mapper, **kwargs):
        calls['n'] += 1
        return real(type_mapper, **kwargs)

    tr_mod.get_mapper_snapshot = _counting_snapshot
    try:
        finder._mapper_snapshot = None
        finder._counted_expansion_resolutions = set()
        counts: dict = {}
        finder._expand_partner_types_to_target(
            ['A', 'B'], DS_A, DS_B, _Mapper(), cache={},
            status_counts_out=counts)
        finder._expand_partner_types_to_target(
            ['A', 'C'], DS_A, DS_B, _Mapper(), cache={},
            status_counts_out=counts)
    finally:
        tr_mod.get_mapper_snapshot = real

    # one snapshot built for the run despite two expansion calls
    assert calls['n'] == 1
    # A resolved twice across calls counts once; B and C fold in once each
    assert counts == {'mapped': 3}


def test_expand_partner_types_counts_statuses(finder):
    """Per-run mapping-resolution counting (Workstream E closure): each
    UNIQUE partner type's resolver status is counted once during candidate
    expansion; unmapped fallback drives the now-MEASURED raw_fallback_used
    flag in the saved homolog metadata."""
    from comparison.type_resolver import STATUS_BRIDGED

    class _Mapper:
        _loaded = True

        def _get_type_mapping_key(self, ds):
            return ds

        def get_mapping_decision(self, t, src, dst, include_bridges=False):
            if t == 'RenamedLocal':
                return {'status': 'mapped', 'source_type': t,
                        'target_type': 'CanonName', 'target_types':
                        ['CanonName'], 'relationship': '1-to-1',
                        'conflicts': []}
            if t == 'Conflicty':
                return {'status': 'conflict', 'source_type': t,
                        'target_type': None, 'target_types': [],
                        'relationship': '1-to-N', 'conflicts': [{}]}
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

    counts: dict = {}
    out = finder._expand_partner_types_to_target(
        ['RenamedLocal', 'Conflicty', 'UnknownPartner', 'UnknownPartner'],
        DS_A, DS_B, _Mapper(), cache={},
        status_counts_out=counts)

    # policy unchanged: renamed expands, conflict excluded, unmapped raw
    assert out['RenamedLocal'] == ('CanonName',)
    assert out['Conflicty'] == ()
    assert out['UnknownPartner'] == ('UnknownPartner',)
    # counting: once per UNIQUE type (the duplicated UnknownPartner counts 1)
    assert counts == {'mapped': 1, 'conflict': 1, 'unmapped': 1}
    # the derivation the save path uses for raw_fallback_used
    assert bool(counts.get('unmapped')) is True


def test_compute_same_type_candidates(finder):
    """The helper matches raw names plus mapper-standardized canon names.

    The stub follows the PRODUCTION mapper contract: ``get_canonical_type``
    maps raw -> canonical and ``standardize_partner_types`` returns
    ``{canonical_type: weight}``.  The historical stub returned
    raw->canonical dict items — the inverse contract — which masked the
    backwards unpacking that made the real mapper return an empty set for
    valid renames (see test_type_mapper_real_datasets.py)."""

    class _StubMapper:
        def get_canonical_type(self, type_name, source_dataset=None):
            return 'MeVPLo2' if type_name == 'MTe07' else type_name

        def standardize_partner_types(self, partner_types, source_dataset):
            canonical: dict = {}
            for t, w in partner_types.items():
                key = 'MeVPLo2' if t == 'MTe07' else t
                canonical[key] = canonical.get(key, 0.0) + w
            return canonical

        def get_mapping_decision(self, source_type, source_dataset,
                                 target_dataset, include_bridges=False):
            return {'status': 'unmapped', 'target_type': None,
                    'target_types': [], 'conflicts': [],
                    'relationship': None}

    lookup = {
        1: 'l-LNv',          # verbatim match
        2: 'MTe07',          # maps to canonical 'MeVPLo2'
        3: '',               # untyped
        4: None,             # untyped
        5: 'Other',
    }
    out = finder._compute_same_type_candidates(
        ['l-LNv', 'MeVPLo2'], lookup, _StubMapper(), DS_A)
    assert out == {1, 2}

    # Without a mapper only verbatim matches count.
    out_raw = finder._compute_same_type_candidates(
        ['l-LNv'], lookup, None, DS_A)
    assert out_raw == {1}

    # Empty query names yield no guarantees.
    assert finder._compute_same_type_candidates([''], lookup, None, DS_A) == set()


def test_compare_candidates_core_intra(finder):
    profiles = {
        5: _rich_profile(5, DS_A),
        6: _rich_profile(6, DS_A),
    }
    status = {b: p.connectivity_status for b, p in profiles.items()}
    candidate_map = {5: {6: 1}, 6: {5: 1}}
    results, intra_df, skipped, warned, *_ = finder._compare_candidates_core(
        source_bodyids=[5, 6], source_profiles_cache=profiles,
        source_status_map=status, target_profiles_cache=profiles,
        target_type_lookup={5: 'Q', 6: 'Q'},
        source_type_lookup={5: 'Q', 6: 'Q'},
        candidate_map=candidate_map,
        is_cross_dataset=False, target_dataset=DS_A, show_progress=False,
        similarity_metric='jaccard', top_n=2, include_intra_type=True)
    assert not results.empty
    assert results['is_same_dataset'].all()
    # intra-type pairwise comparison between the two sources
    assert len(intra_df) == 1
    # synthetic type-level profiles carry no typed bodyId structure, so
    # combined_score_intra_dataset yields NaN scores (code path exercised)
    assert pd.isna(intra_df.iloc[0]['rank_corr'])
    assert intra_df.iloc[0]['is_same_type']


# ---------------------------------------------------------------------------
# HomologFinder._save_homolog_results_internal
# ---------------------------------------------------------------------------

def _homolog_results_df():
    rows = []
    for tgt_bid, tgt_type, rc in [(101, 'Cand', 0.95), (102, '', 0.40)]:
        rows.append({
            'source_bodyId': 5, 'source_type': 'Q',
            'target_bodyId': tgt_bid, 'target_type': tgt_type,
            'target_dataset': DS_B,
            'rank_corr': rc, 'rank_union': rc - 0.05,
            'jaccard': rc - 0.1, 'cosine': rc - 0.15,
            'adjacency_score': 3, 'shared_type_count': 6,
            'union_type_count': 6,
            'is_same_type': False, 'is_same_dataset': False,
            'source_status': 'complete', 'target_status': 'complete',
            'weak_source': False, 'weak_target': False,
            'source_partner_count': 12, 'target_partner_count': 12,
        })
    return pd.DataFrame(rows)


def test_save_homolog_results_internal_full(finder, monkeypatch, tmp_path):
    _finder_for_homologs(finder, monkeypatch)
    results_df = _homolog_results_df()
    intra_df = pd.DataFrame({'source_bodyId': [5], 'other_bodyId': [6],
                             'rank_corr': [1.0], 'jaccard': [0.9]})
    shuffle_stats = {
        'n_shuffles': 3,
        'observed_score': np.float64(0.9),
        'mean_shuffled': 0.2, 'std_shuffled': 0.1,
        'p_value': 0.0, 'z_score': 2.0, 'effect_size': 1.2,
        'interpretation': 'significant',
        'real_results': results_df,       # excluded key (DataFrame)
        'shuffled_results': [results_df], # excluded key
        'extra_df': results_df,           # DataFrame -> skipped
        'scores': [np.float64(0.1), 0.2], # list with numpy items
        'arr': np.array([0.5]),           # ndarray -> .item()
        'misc': object(),                 # falls back to str()
    }
    out = finder._save_homolog_results_internal(
        results_df=results_df, query='Q', source_dataset=DS_A,
        target_dataset=DS_B, output_dir=str(tmp_path),
        saveas='fixed_folder', direction='both',
        include_partner_details=True, top_n_details=2,
        params={'top_n': 5, 'metric': 'rank_corr'},
        shuffle_stats=shuffle_stats, intra_type_df=intra_df,
        similarity_metric='rank_corr',
        source_status_summary={'complete': 1, 'incomplete': 0},
        visualize_top_n=1)

    base = tmp_path / 'fixed_folder'
    for rel in ['README.txt',
                'results/bodyid_results.csv',
                'results/homolog_results.csv',
                'results/intra_type_results.csv',
                'results/shuffle_test.json',
                'results/source_status_summary.json',
                'results/type_summary.csv',
                'profiles/query/Q.csv',
                'profiles/query/source_bodyids.csv',
                'profiles/matches/top_target_bodyids.csv',
                'profiles/matches/Cand.csv',
                'overlaps/Q_vs_Cand.csv']:
        assert (base / rel).exists(), rel

    with open(base / 'results' / 'shuffle_test.json') as f:
        saved = json.load(f)
    assert 'real_results' not in saved
    assert 'shuffled_results' not in saved
    assert 'extra_df' not in saved
    assert saved['observed_score'] == pytest.approx(0.9)
    assert saved['scores'] == pytest.approx([0.1, 0.2])
    assert saved['arr'] == pytest.approx(0.5)

    ts = pd.read_csv(base / 'results' / 'type_summary.csv')
    assert 'avg_rank_corr' not in ts.columns
    # visualize_top_n alone does NOT claim visualization: without
    # visualize_skeleton no scenes rendered, so the flag columns stay out.
    assert 'visualized' not in ts.columns
    assert 'visualization_rank' not in ts.columns
    assert out['query_profile'] is not None
    assert 'Cand' in out['match_profiles']


def test_save_homolog_results_internal_nan_and_sort_fallback(finder, monkeypatch, tmp_path):
    _finder_for_homologs(finder, monkeypatch)
    results_df = _homolog_results_df()
    results_df['rank_corr'] = np.nan
    # drop metrics so the requested sort metric must fall back
    results_df = results_df.drop(columns=['rank_union', 'jaccard', 'cosine'])
    finder._save_homolog_results_internal(
        results_df=results_df, query='Q', source_dataset=DS_A,
        target_dataset=DS_B, output_dir=str(tmp_path), saveas='nan_folder',
        direction='upstream', include_partner_details=False,
        top_n_details=1, params={}, similarity_metric='rank_union')
    ts = pd.read_csv(tmp_path / 'nan_folder' / 'results' / 'type_summary.csv')
    # all rows had NaN rank_corr -> empty aggregated summary (header only)
    assert ts.empty
    assert 'avg_rank_corr' not in ts.columns


def test_save_homolog_results_internal_type_level_and_fallback(finder, monkeypatch, tmp_path):
    _finder_for_homologs(finder, monkeypatch)
    # type-level rows (no source_bodyId column)
    type_df = pd.DataFrame({
        'source_neuron': ['Q', 'Q'],
        'target_type': ['Cand', 'Other'],
        'rank_corr': [0.8, np.nan],
        'rank_union': [0.7, np.nan],
        'jaccard': [0.6, np.nan],
        'adjacency_score': [2, 0],
        'is_same_type': [False, False],
    })
    finder._save_homolog_results_internal(
        results_df=type_df, query='Q', source_dataset=DS_A,
        target_dataset=DS_B, output_dir=str(tmp_path), saveas='type_folder',
        direction='both', include_partner_details=False, top_n_details=1,
        params={}, visualize_top_n=1)
    ts = pd.read_csv(tmp_path / 'type_folder' / 'results' / 'type_summary.csv')
    assert len(ts) == 1  # NaN row filtered
    assert ts.iloc[0]['target_type'] == 'Cand'
    # No scenes rendered (visualize_skeleton off): no visualized claims.
    assert 'visualized' not in ts.columns

    # Visualization rank follows the sorted type order, not the input index.
    ranked_type_df = pd.DataFrame({
        'source_neuron': ['Q', 'Q', 'Q'],
        'target_type': ['Low', 'High', 'Mid'],
        'rank_corr': [0.4, 0.9, 0.7],
        'rank_union': [0.3, 0.8, 0.6],
        'jaccard': [0.2, 0.9, 0.5],
        'adjacency_score': [1, 3, 2],
        'is_same_type': [False, False, False],
    })
    finder._save_homolog_results_internal(
        results_df=ranked_type_df, query='Q', source_dataset=DS_A,
        target_dataset=DS_B, output_dir=str(tmp_path), saveas='ranked_folder',
        direction='both', include_partner_details=False, top_n_details=1,
        params={}, visualize_top_n=2)
    ranked = pd.read_csv(tmp_path / 'ranked_folder' / 'results' / 'type_summary.csv')
    assert list(ranked['target_type']) == ['High', 'Mid', 'Low']
    # Still no rendered scenes: the flag columns must stay absent even
    # though visualize_top_n is set.
    assert 'visualized' not in ranked.columns
    assert 'visualization_rank' not in ranked.columns

    # fallback branch: neither target_type nor target column present
    bare_df = pd.DataFrame({'foo': [1], 'rank_corr': [0.5]})
    finder._save_homolog_results_internal(
        results_df=bare_df, query='Q', source_dataset=DS_A,
        target_dataset=DS_B, output_dir=str(tmp_path), saveas=None,
        direction='both', include_partner_details=False, top_n_details=1,
        params={})
    # saveas=None -> auto-generated timestamped folder
    folders = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert any(f.name.startswith(finder.output_folder_prefix) for f in folders)


def test_type_summary_visualized_reflects_rendered_scenes(
        finder, monkeypatch, tmp_path):
    """`visualized` / `visualization_rank` come from the types the scenes
    actually rendered (returned by _visualize_homolog_candidates), not from
    the table's own row order; untyped targets stay out of type_summary."""
    _finder_for_homologs(finder, monkeypatch)
    results_df = _homolog_results_df()  # targets: 101 'Cand', 102 untyped
    monkeypatch.setattr(
        HomologFinder, '_visualize_homolog_candidates',
        lambda self, **kwargs: ['Elsewhere', 'Cand'])
    finder._save_homolog_results_internal(
        results_df=results_df, query='Q', source_dataset=DS_A,
        target_dataset=DS_B, output_dir=str(tmp_path), saveas='vis_folder',
        direction='both', include_partner_details=False, top_n_details=1,
        params={}, visualize_skeleton=True, visualize_top_n=2)
    ts = pd.read_csv(tmp_path / 'vis_folder' / 'results' / 'type_summary.csv')
    # the untyped target ('' type) is excluded from the aggregation
    assert list(ts['target_type']) == ['Cand']
    # 'Cand' was rendered (rank 2 of the returned scene order); the
    # untyped row is gone, so no False rows remain.
    assert list(ts['visualized']) == [True]
    assert list(ts['visualization_rank']) == [2]


# ---------------------------------------------------------------------------
# HomologFinder._load_connection_cache and friends
# ---------------------------------------------------------------------------

def _write_neuron_index(pc_fake_repo, safe_name, bodyids, types):
    idx_dir = pc_fake_repo / 'neuron_indexes' / safe_name
    idx_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'bodyId': bodyids, 'type': types}).to_parquet(
        idx_dir / 'neuron_index.parquet')


def test_load_connection_cache_internal_hit(finder):
    df = pd.DataFrame({'bodyId_pre': [1]})
    finder._conn_cache[DS_A] = df
    assert finder._load_connection_cache(DS_A) is df


def test_load_connection_cache_fnc_branch(finder, monkeypatch, pc_fake_repo):
    raw = pd.DataFrame({'bodyId_pre': [1, 2], 'bodyId_post': [2, 1],
                        'weight': [5, 3]})
    coana_mod = _install_fake_coana(monkeypatch, fnc_cache={
        SAFE_A: {'conn_df': raw}})
    _write_neuron_index(pc_fake_repo, SAFE_A, [1, 2], ['T1', 'T2'])
    out = finder._load_connection_cache(DS_A)
    assert out is not None
    assert list(out['type_pre']) == ['T1', 'T2']
    assert list(out['type_post']) == ['T2', 'T1']
    # enrichment happens on a copy: the shared FNC frame stays untouched
    assert 'type_pre' not in raw.columns
    assert coana_mod._FNC_CACHE[SAFE_A]['conn_df'] is raw


def test_load_connection_cache_disk_parquet(finder, monkeypatch, pc_fake_repo):
    _install_fake_coana(monkeypatch)  # empty _FNC_CACHE -> falls to disk
    cache_dir = pc_fake_repo / 'cache' / SAFE_A
    cache_dir.mkdir(parents=True)
    pd.DataFrame({'bodyId_pre': [1], 'bodyId_post': [2],
                  'weight': [4]}).to_parquet(cache_dir / 'connections.parquet')
    _write_neuron_index(pc_fake_repo, SAFE_A, [1, 2], ['T1', 'T2'])
    out = finder._load_connection_cache(DS_A)
    assert out is not None
    assert out.iloc[0]['type_pre'] == 'T1'
    assert out.iloc[0]['type_post'] == 'T2'


def test_load_connection_cache_missing_and_no_typemap(finder, monkeypatch, pc_fake_repo):
    _install_fake_coana(monkeypatch)
    # auto_build disabled -> None when cache file absent
    assert finder._load_connection_cache(DS_A, auto_build=False) is None

    # cache exists but no type mapping anywhere -> None
    cache_dir = pc_fake_repo / 'cache' / SAFE_A
    cache_dir.mkdir(parents=True)
    pd.DataFrame({'bodyId_pre': [1], 'bodyId_post': [2],
                  'weight': [4]}).to_parquet(cache_dir / 'connections.parquet')
    assert finder._load_connection_cache(DS_A, auto_build=False) is None


def test_load_connection_cache_auto_build(finder, monkeypatch, tmp_path, pc_fake_repo):
    _install_fake_coana(monkeypatch)
    cache_path = tmp_path / 'cache' / 'connections.parquet'
    monkeypatch.setattr(finder, '_get_connection_cache_path',
                        lambda dataset: cache_path)
    _FakeFNC.write_cache_path = cache_path
    _FakeFNC.total_neurons = 2
    _write_neuron_index(pc_fake_repo, SAFE_A, [1, 2], ['T1', 'T2'])
    out = finder._load_connection_cache(DS_A, auto_build=True)
    assert out is not None
    assert 'type_pre' in out.columns
    assert len(_FakeFNC.instances) == 1


def test_ensure_connection_cache_complete(finder, monkeypatch):
    coana_mod = _install_fake_coana(monkeypatch)
    coana_mod._FNC_CACHE[SAFE_A] = {'conn_df': 'junk'}
    assert finder._ensure_connection_cache_complete(DS_A) is True
    assert coana_mod._FNC_CACHE[SAFE_A] == {}

    _FakeFNC.total_neurons = 0
    assert finder._ensure_connection_cache_complete(DS_A) is False

    _FakeFNC.total_neurons = 5
    _FakeFNC.raise_on_build = True
    assert finder._ensure_connection_cache_complete(DS_A) is False


# ---------------------------------------------------------------------------
# HomologFinder shuffle / random control test
# ---------------------------------------------------------------------------

def test_shuffle_profile_deterministic(finder):
    cp_module._PROFILER_CONN_CACHE[SAFE_A] = {
        'type_lookup': {i: f'PT{i}' for i in range(12)}}
    try:
        profile = _rich_profile(5, DS_A)
        s1 = finder.shuffle_profile(profile, seed=42)
        s2 = finder.shuffle_profile(profile, seed=42)
        assert s1.upstream_partners == s2.upstream_partners
        assert s1.downstream_partners == s2.downstream_partners
        assert sorted(s1.upstream_partners.values()) == \
            sorted(profile.upstream_partners.values())
        assert s1.neuron_id.endswith('_shuffled')
    finally:
        cp_module._PROFILER_CONN_CACHE.pop(SAFE_A, None)


def test_shuffle_profile_fallback_pool(finder):
    # no cache / no profiler loaders -> warns and shuffles within own types
    profile = _rich_profile(5, DS_A)
    with pytest.warns(UserWarning):
        shuffled = finder.shuffle_profile(profile, seed=7)
    pool = set(profile.upstream_partners) | set(profile.downstream_partners)
    assert set(shuffled.upstream_partners) <= pool


def test_get_type_pool_variants(finder):
    # branch 1: profiler connection cache
    cp_module._PROFILER_CONN_CACHE[SAFE_A] = {
        'type_lookup': {1: 'T1', 2: 'T2', 3: 'T1'}}
    try:
        assert set(finder._get_type_pool(DS_A)) == {'T1', 'T2'}
    finally:
        cp_module._PROFILER_CONN_CACHE.pop(SAFE_A, None)

    # branch 2: profiler neuron index
    fake = _FakeProfilerForFinder(available={})
    fake._load_neuron_index = lambda dataset: pd.DataFrame(
        {'type': ['A', 'B', None]})
    finder.profiler = fake
    assert set(finder._get_type_pool(DS_A)) == {'A', 'B'}

    # branch 3: connection data fallback
    fake._load_neuron_index = lambda dataset: None
    fake._load_connection_data = lambda dataset: pd.DataFrame(
        {'type_pre': ['A', None], 'type_post': ['C', 'C']})
    assert set(finder._get_type_pool(DS_A)) == {'A', 'C'}

    # branch 4: nothing available
    fake._load_connection_data = lambda dataset: None
    assert finder._get_type_pool(DS_A) == []


def test_run_random_control_test(finder, monkeypatch, tmp_path):
    _finder_for_homologs(finder, monkeypatch)
    real_df = pd.DataFrame({'target_type': ['Cand'], 'rank_corr': [0.9]})
    monkeypatch.setattr(finder, 'find_homologs_fast', lambda **kw: real_df.copy())
    cp_module._PROFILER_CONN_CACHE[SAFE_B] = {'type_lookup': {101: 'Cand'}}
    try:
        res = finder.run_random_control_test(
            source='Q', source_dataset=DS_A, target_dataset=DS_B,
            n_shuffles=3, seed=42, show_progress=False,
            output_dir=str(tmp_path))
    finally:
        cp_module._PROFILER_CONN_CACHE.pop(SAFE_B, None)
    assert 'error' not in res
    assert res['n_shuffles'] == 3
    assert 0.0 <= res['p_value'] <= 1.0
    out_dir = tmp_path / res['output_path'].split('/')[-1]
    for fname in ['real_results.csv', 'control_test_stats.csv',
                  'shuffled_score_distribution.csv', 'summary.txt']:
        assert (out_dir / fname).exists()

    # empty real results -> early return with p_value 1.0
    monkeypatch.setattr(finder, 'find_homologs_fast',
                        lambda **kw: pd.DataFrame())
    res_empty = finder.run_random_control_test(
        source='Q', source_dataset=DS_A, target_dataset=DS_B,
        n_shuffles=2, show_progress=False)
    assert res_empty['p_value'] == 1.0
    assert not res_empty['is_significant']

    # profiler raises -> error dict
    class _RaisingProfiler:
        def get_profile(self, neuron, dataset):
            raise RuntimeError('boom')
    finder.profiler = _RaisingProfiler()
    res_err = finder.run_random_control_test(
        source='Q', source_dataset=DS_A, target_dataset=DS_B)
    assert 'error' in res_err and not res_err['is_significant']


def test_run_homolog_with_profile_empty_pool(finder, monkeypatch):
    _finder_for_homologs(finder, monkeypatch)
    # no type pool for target dataset -> empty frame
    out = finder._run_homolog_with_profile(_rich_profile(5, DS_A), DS_B, 5)
    assert out.empty


# ===========================================================================
# ConnectivityProfileComparer coverage (chunk 5)
# ===========================================================================

from pathlib import Path


def _comparer(query, tmp_path=None, dataset=DS_A, **kwargs):
    """Build a comparer with hermetic defaults (no type-mapper file loads)."""
    kwargs.setdefault('verbose', False)
    kwargs.setdefault('generate_heatmaps', False)
    kwargs.setdefault('use_auto_type_mapping', False)
    if tmp_path is not None:
        kwargs.setdefault('output_dir', str(tmp_path))
    return ConnectivityProfileComparer(query, dataset=dataset, **kwargs)


def _install_fake_heatmap(monkeypatch, calls, kwargs_seen=None):
    """Fake comparison.interactive_heatmap used by the vispath fallback.

    Matches the real signature incl. the metric-scale passthrough
    (color_scale/zmin/zmax) the fallback has forwarded since the
    diverging-[-1, 1] round.
    """
    fake_mod = ModuleType('comparison.interactive_heatmap')

    def gen(matrices_dict, filename, title=None, showfig=False, verbose=False,
            color_scale=None, zmin=None, zmax=None):
        calls.append(filename)
        if kwargs_seen is not None:
            kwargs_seen.append(
                {'color_scale': color_scale, 'zmin': zmin, 'zmax': zmax})
        Path(filename).write_text('<html></html>', encoding='utf-8')

    fake_mod.generate_interactive_heatmap = gen
    monkeypatch.setitem(sys.modules, 'comparison.interactive_heatmap', fake_mod)


def _block_vispath(monkeypatch):
    """Force the vispath import to fail so the heatmap fallback runs.

    Other tests leak the real vispath-subproject src into sys.path, so a
    None sys.modules entry is used to hard-block the import chain.
    """
    for name in ('vispath_pkg', 'vispath_pkg.vispath'):
        sys.modules.pop(name, None)
        monkeypatch.setitem(sys.modules, name, None)


def test_comparer_init_modes_and_errors(tmp_path):
    # cross-dataset dict with <2 datasets -> ValueError
    with pytest.raises(ValueError):
        ConnectivityProfileComparer({DS_A: ['T1']}, verbose=False)

    # dict query + dataset parameter -> UserWarning, forced bodyId skip,
    # flattened 'ds:name' query
    with pytest.warns(UserWarning):
        comp = ConnectivityProfileComparer(
            {DS_A: ['T1'], DS_B: ['T2']}, dataset=DS_A,
            output_dir=str(tmp_path), verbose=False,
            use_auto_type_mapping=False)
    assert comp.is_cross_dataset is True
    assert comp.skip_bodyId_level is True
    assert comp.query == [f'{DS_A}:T1', f'{DS_B}:T2']
    assert comp.dataset == DS_A

    # intra-dataset without dataset -> ValueError
    with pytest.raises(ValueError):
        ConnectivityProfileComparer(['T1'], verbose=False)

    # aggregation level normalization ('custom group' UI label)
    comp2 = _comparer(['T1'], tmp_path, aggregation_level='custom group')
    assert comp2.aggregation_level == 'custom'
    comp2b = _comparer(['T1'], tmp_path, aggregation_level='bogus')
    assert comp2b.aggregation_level == 'type'

    # one-element datasets list is accepted for single-dataset mode
    comp3 = ConnectivityProfileComparer(
        ['T1'], datasets=[DS_A], verbose=False, use_auto_type_mapping=False)
    assert comp3.dataset == DS_A and comp3.is_multi_dataset is False

    # multi-dataset mode
    comp4 = ConnectivityProfileComparer(
        ['T1'], datasets=[DS_A, DS_B], verbose=False,
        use_auto_type_mapping=False)
    assert comp4.is_multi_dataset is True
    assert comp4.datasets == [DS_A, DS_B]


def test_comparer_query_helpers(tmp_path):
    comp = _comparer(['T1'], tmp_path)

    # _format_query_for_log: full and truncated
    assert comp._format_query_for_log(['a', 'b']) == "['a', 'b']"
    assert comp._format_query_for_log(42) == '42'
    long_q = [f't{i}' for i in range(12)]
    s = comp._format_query_for_log(long_q)
    assert '(2 more)' in s and '(12 total items)' in s

    # _sort_types_string_first
    assert comp._sort_types_string_first(['2', 'B', '1', 'A']) == \
        ['A', 'B', '1', '2']

    # _normalize_query variants
    assert comp._normalize_query('Mi1') == (['Mi1'], None)
    assert comp._normalize_query(5) == ([5], None)
    assert comp._normalize_query([]) == ([], None)
    assert comp._normalize_query(['a', 'b']) == (['a', 'b'], None)
    nested, names = comp._normalize_query([['G1', [1, 2]], ['G2', [3]], 'bad'])
    assert nested == [[1, 2], [3]] and names == ['G1', 'G2']

    # _looks_like_pattern
    assert comp._looks_like_pattern('aMe.*') is True
    assert comp._looks_like_pattern('Mi1') is False

    # _generate_query_name
    assert comp._generate_query_name() == 'T1'
    comp_multi = _comparer(['T1', 'T2'], tmp_path)
    assert comp_multi._generate_query_name() == 'T1_etc'
    comp_grp = _comparer(['T1'], tmp_path)
    comp_grp._custom_group_names = ['My:Group']
    assert comp_grp._generate_query_name() == 'My_Group'
    comp_empty = _comparer(['T1'], tmp_path)
    comp_empty.query = []
    assert comp_empty._generate_query_name() == 'unnamed'

    # _safe_folder_name
    assert ConnectivityProfileComparer._safe_folder_name('a/b c:d.e') == 'a_b_c_d_e'

    # query mapping pass-throughs (no type mapper loaded)
    assert comp._map_query_item(5, DS_A) == 5
    assert comp._map_query_item('17', DS_A) == '17'
    assert comp._map_query_item('aMe.*', DS_A) == 'aMe.*'
    assert comp._map_query_item('Mi1', DS_A) == 'Mi1'
    assert comp._mapped_query_for(DS_A) == comp.query
    assert comp._canonical_label('Mi1', DS_A) == 'Mi1'


def test_comparer_parse_group_map_csv(tmp_path):
    comp = _comparer(['T1'], tmp_path)

    with pytest.raises(FileNotFoundError):
        comp._parse_group_map_csv(str(tmp_path / 'nope.csv'))

    bad = tmp_path / 'bad.csv'
    bad.write_text('a,b\n1,2\n', encoding='utf-8')
    with pytest.raises(ValueError):
        comp._parse_group_map_csv(str(bad))

    good = tmp_path / 'groups.csv'
    good.write_text(
        'group,id_type_instance\n'
        'G1,Mi1\nG1,12345\nG2,Tm3\n', encoding='utf-8')
    normalized, names = comp._parse_group_map_csv(str(good))
    assert names == ['G1', 'G2']
    assert normalized == [['Mi1', 12345], ['Tm3']]

    # group_map_csv at init overrides the query
    comp2 = _comparer(['ignored'], tmp_path, group_map_csv=str(good))
    assert comp2._custom_group_names == ['G1', 'G2']


def test_comparer_load_custom_groups_from_mapping(tmp_path):
    comp = _comparer(['T1'], tmp_path)

    # unreadable / invalid JSON -> ValueError
    with pytest.raises(ValueError):
        comp._load_custom_groups_from_mapping(str(tmp_path / 'missing.json'), DS_A)
    bad = tmp_path / 'bad.json'
    bad.write_text('{oops', encoding='utf-8')
    with pytest.raises(ValueError):
        comp._load_custom_groups_from_mapping(str(bad), DS_A)

    def _write(data):
        p = tmp_path / 'preset.json'
        p.write_text(json.dumps(data), encoding='utf-8')
        return str(p)

    # no labels -> ([], [])
    assert comp._load_custom_groups_from_mapping(
        _write({'source_mapping': {DS_A: [['x']]}}), DS_A) == ([], [])
    # labels but dataset missing -> ([], [])
    assert comp._load_custom_groups_from_mapping(
        _write({'source_mapping': {'custom_label': ['g1'],
                                   'other_ds': [['x']]}}), DS_A) == ([], [])

    # valid preset (safe-name dataset key + digit conversion + empty rows dropped)
    safe_a = DS_A.replace(':', '_').replace('.', '_')
    normalized, names = comp._load_custom_groups_from_mapping(
        _write({'source_mapping': {
            'custom_label': ['g1', 'g2', 'g3'],
            safe_a: [['T1', 'T2'], [], ['5']]}
        }), DS_A)
    assert names == ['g1', 'g3']
    assert normalized == [['T1', 'T2'], [5]]

    # multi-dataset mode records per-dataset member lists
    comp_multi = ConnectivityProfileComparer(
        ['T1'], datasets=[DS_A, DS_B], verbose=False,
        use_auto_type_mapping=False)
    comp_multi._load_custom_groups_from_mapping(
        _write({'source_mapping': {
            'custom_label': ['g1'],
            safe_a: [['T1']],
            DS_B: [[7]]}
        }), DS_A)
    assert DS_A in comp_multi._mapping_ds_members
    assert comp_multi._mapping_ds_members[DS_B] == [[7]]

    # custom_mapping_file at init forces custom aggregation
    comp_init = _comparer(['T1'], tmp_path, custom_mapping_file=_write(
        {'source_mapping': {'custom_label': ['g1'], safe_a: [['T1']]}}))
    assert comp_init.aggregation_level == 'custom'
    assert comp_init._custom_group_names == ['g1']


def test_comparer_get_neurons_to_compare_variants(tmp_path):
    comp = _comparer(['T1'], tmp_path)
    fake = _FakeProfilerFull(
        bodyids={('T1', DS_A): [1, 2], ('aMe1', DS_A): [10],
                 ('aMe2', DS_A): [11]},
    )
    fake.types_for_bodyids[DS_A] = {1: 'T1'}
    fake.type_lists[('aMe.*', DS_A)] = ['aMe1', 'aMe2']
    fake.label_groups[('clock', DS_A)] = {'cA': [20], 'cB': ['21']}
    comp.profiler = fake

    # explicit custom groups win over everything
    out = comp._get_neurons_to_compare(
        query=[[1, 2], [3]], custom_group_names=['G1', 'G2'])
    assert out == {'G1': [1, 2], 'G2': [3]}

    # custom aggregation: every flat item is its own literal group
    comp.aggregation_level = 'custom'
    assert comp._get_neurons_to_compare(query=['T1', '7']) == \
        {'T1': ['T1'], '7': [7]}
    comp.aggregation_level = 'type'

    # bodyIds: known type aggregates, unknown keeps bodyId label
    out = comp._get_neurons_to_compare(query=[1, 9])
    assert out == {'T1': [1], '9': [9]}

    # bodyid aggregation: per-neuron labels {bid}_{type}
    out = comp._get_neurons_to_compare(query=[1, 9], aggregation_level='bodyid')
    assert out == {'1_T1': [1], '9': [9]}

    # exact type resolution
    assert comp._get_neurons_to_compare(query=['T1']) == {'T1': [1, 2]}
    out = comp._get_neurons_to_compare(query=['T1'], aggregation_level='bodyid')
    assert out == {'1_T1': [1], '2_T1': [2]}

    # pattern expansion (types and bodyid levels)
    out = comp._get_neurons_to_compare(query=['aMe.*'])
    assert out == {'aMe1': [10], 'aMe2': [11]}
    out = comp._get_neurons_to_compare(query=['aMe.*'], aggregation_level='bodyid')
    assert out == {'10_aMe1': [10], '11_aMe2': [11]}

    # taxonomy label expansion (string ids coerced to int)
    out = comp._get_neurons_to_compare(query=['clock'])
    assert out == {'cA': [20], 'cB': [21]}

    # unresolved literal fallbacks
    assert comp._get_neurons_to_compare(query=['Ghost']) == {'Ghost': ['Ghost']}
    assert comp._get_neurons_to_compare(query=['zz.*']) == {'zz.*': ['zz.*']}


def test_comparer_coarse_label_expands_before_union_lookup(tmp_path):
    """A coarse taxonomy label that maps to several real types must expand
    into one comparison row per type, even when the single-column union
    lookup would also return bodyIds (FAFB 'circadian_clock' case)."""
    comp = _comparer(['circadian_clock'], tmp_path)
    fake = _FakeProfilerFull(
        bodyids={
            ('circadian_clock', DS_A): [1, 2, 3, 4, 5],  # union result
            ('aMe12', DS_A): [1, 2],
        },
    )
    fake.label_groups[('circadian_clock', DS_A)] = {
        's-LNv': [1, 2], 'DN1pC': [3, 4],
    }
    comp.profiler = fake

    # Multi-type taxonomy label: per-real-type rows win over the union row
    assert comp._get_neurons_to_compare(query=['circadian_clock']) == \
        {'s-LNv': [1, 2], 'DN1pC': [3, 4]}

    # bodyid aggregation still produces per-neuron rows under real types
    assert comp._get_neurons_to_compare(
        query=['circadian_clock'], aggregation_level='bodyid') == \
        {'1_s-LNv': [1], '2_s-LNv': [2], '3_DN1pC': [3], '4_DN1pC': [4]}

    # An exact type (single taxonomy group) is still one row
    fake.label_groups[('Mi1', DS_A)] = {'Mi1': [7, 8]}
    assert comp._get_neurons_to_compare(query=['Mi1']) == {'Mi1': [7, 8]}

    # No taxonomy grouping (e.g. NeuPrint): the union lookup keeps one row
    assert comp._get_neurons_to_compare(query=['aMe12']) == \
        {'aMe12': [1, 2]}


def test_comparer_ensure_connection_cache(tmp_path, monkeypatch):
    _install_fake_coana(monkeypatch)
    comp = _comparer(['T1'], tmp_path)

    assert comp._ensure_connection_cache_complete() is True
    _FakeFNC.total_neurons = 0
    assert comp._ensure_connection_cache_complete() is False
    _FakeFNC.reset()
    _FakeFNC.raise_on_build = True
    assert comp._ensure_connection_cache_complete() is False
    assert comp._ensure_connection_cache_complete_for_dataset(DS_A) is False
    _FakeFNC.reset()
    assert comp._ensure_connection_cache_complete_for_dataset(DS_A) is True


def test_comparer_matrix_helpers(tmp_path):
    comp = _comparer(['T1', 'T2'], tmp_path)

    # _aggregate_profiles_from_list
    assert comp._aggregate_profiles_from_list([], 'X') is None
    p1 = _rich_profile(1)
    assert comp._aggregate_profiles_from_list([p1], 'X') is p1
    agg = comp._aggregate_profiles_from_list(
        [_rich_profile(1), _rich_profile(2)], 'X')
    assert agg.num_neurons_aggregated == 2
    assert agg.dataset == DS_A
    assert agg.neuron_id == 'X'

    # _compute_similarity_from_types edge cases
    s = comp._compute_similarity_from_types({'a': 1.0}, {'a': 1.0})
    assert s['jaccard'] == pytest.approx(1.0)
    assert np.isnan(s['rank'])  # <3 shared -> no rank
    weights_a = {f'p{i}': float(6 - i) for i in range(4)}
    weights_b = {f'p{i}': float(6 - i) for i in range(4)}
    s2 = comp._compute_similarity_from_types(weights_a, weights_b)
    assert s2['rank'] == pytest.approx(1.0)

    # similarity matrices (3 directions, 5 metrics, diagonal 1.0)
    profiles = {'T1': _rich_profile(1, DS_A), 'T2': _rich_profile(2, DS_A)}
    mats = comp._compute_similarity_matrices(profiles)
    assert set(mats) == {'overall', 'upstream', 'downstream'}
    assert set(mats['overall']) == {'jaccard', 'weighted_jaccard', 'cosine',
                                    'rank_corr', 'rank_corr_union'}
    assert mats['overall']['jaccard'].loc['T1', 'T1'] == 1.0
    assert list(mats['overall']['jaccard'].index) == ['T1', 'T2']

    # single-direction comparer
    comp_up = _comparer(['T1', 'T2'], tmp_path, direction='upstream')
    mats_up = comp_up._compute_similarity_matrices(profiles)
    assert set(mats_up) == {'upstream'}

    # bodyId-level matrices: labels {bid}_{type}
    bodyid_profiles = {
        ('T1', 1): _rich_profile(1, DS_A),
        ('T1', 2): _rich_profile(2, DS_A),
        ('T2', 3): _rich_profile(3, DS_A),
    }
    bm = comp._compute_bodyid_similarity_matrices(bodyid_profiles)
    assert '1_T1' in bm['overall']['jaccard'].index

    # type-avg-bodyId matrices: intra avg for T1 (2 bids), 1.0 for T2 (1 bid)
    tm = comp._compute_type_avg_bodyid_matrices(bodyid_profiles)
    tav = tm['overall']['jaccard']
    assert tav.loc['T2', 'T2'] == 1.0
    assert 0.0 <= tav.loc['T1', 'T1'] <= 1.0

    # inter-dataset matrices + consolidation (multi-dataset layout)
    comp.datasets = [DS_A, DS_B]
    anchors = {
        'T1': {DS_A: ('T1', _rich_profile(1, DS_A)),
               DS_B: ('T1', _rich_profile(1, DS_B))},
    }
    inter = comp._compute_inter_dataset_matrices(anchors)
    assert 'overall' in inter['T1']
    m = inter['T1']['overall']['jaccard']
    assert np.isnan(m.loc[DS_A, DS_A])  # diagonal stays NaN
    assert m.loc[DS_A, DS_B] == pytest.approx(1.0)
    consolidated = comp._aggregate_inter_dataset_matrices(inter)
    assert 'overall' in consolidated
    frame = consolidated['overall']['jaccard']
    assert list(frame.index) == ['T1']
    assert frame.iloc[0, 0] == pytest.approx(1.0)
    assert comp._aggregate_inter_dataset_matrices({}) == {}


def test_comparer_get_output_path_default(pc_fake_repo):
    comp = _comparer(['T1'])  # no output_dir -> local_data default
    path = comp._get_output_path()
    assert str(path).startswith(str(pc_fake_repo))
    assert 'local_data' in path.parts and 'connectivity_profiling' in path.parts


def test_comparer_run_single_dataset(tmp_path, pc_fake_repo, monkeypatch):
    calls, heatmap_kwargs = [], []
    _block_vispath(monkeypatch)
    _install_fake_heatmap(monkeypatch, calls, heatmap_kwargs)

    comp = _comparer(['T1', 'T2'], tmp_path,
                     skip_bodyId_level=False, generate_heatmaps=True)
    comp.profiler = _FakeProfilerFull(
        profiles={
            ('1', DS_A): _rich_profile(1, DS_A, scale=1.0),
            ('2', DS_A): _rich_profile(2, DS_A, scale=0.9),
            ('3', DS_A): _rich_profile(3, DS_A, scale=1.1),
        },
        bodyids={('T1', DS_A): [1, 2], ('T2', DS_A): [3]},
    )
    res = comp.run()

    assert res['n_type_profiles'] == 2
    assert res['n_bodyid_profiles'] == 3
    assert res['bodyid_level_skipped'] is False
    assert res['is_cross_dataset'] is False
    assert set(res['type_matrices']) == {'overall', 'upstream', 'downstream'}

    out = Path(res['output_path'])
    assert (out / 'parameters.json').exists()
    assert (out / 'README.txt').exists()
    assert (out / 'report.html').exists()
    assert len(list((out / 'type_level' / 'results').glob(
        'type_similarity_*.csv'))) == 15
    assert len(list((out / 'bodyid_level' / 'results').glob(
        'bodyid_similarity_*.csv'))) == 15
    assert len(list((out / 'bodyid_level' / 'results').glob(
        'type_avg_bodyid_similarity_*.csv'))) == 15
    assert len(list((out / 'profiles' / 'individual').glob('*.json'))) == 3
    assert len(list((out / 'profiles' / 'aggregated').glob('*.json'))) == 2
    # heatmap fallback path (vispath blocked above); every call carries the
    # metric's colormap and fixed domain (jaccard et al. are [0, 1])
    assert calls and len(res['heatmaps_generated']) == 45
    assert len(heatmap_kwargs) == len(calls)
    assert all(kw['color_scale'] is not None and kw['zmin'] is not None
               and kw['zmax'] is not None for kw in heatmap_kwargs)
    # jaccard-family metrics are [0, 1]; rank_corr_union is diverging
    # [-1, 1] — the passthrough preserves each metric's own domain.
    zmins = {kw['zmin'] for kw in heatmap_kwargs}
    assert zmins == {0.0, -1.0}


def test_comparer_run_single_dataset_insufficient(tmp_path):
    comp = _comparer(['OnlyOne'], tmp_path)
    comp.profiler = _FakeProfilerFull(
        profiles={('1', DS_A): _rich_profile(1, DS_A)},
        bodyids={('OnlyOne', DS_A): [1]},
    )
    # Raise, do not return an error dict: the runner discards return values,
    # so a silent refusal shows up in the panel as Completed with no files.
    with pytest.raises(ValueError, match='at least two neurons in scope'):
        comp.run()


def test_comparer_run_bodyid_aggregation(tmp_path, pc_fake_repo):
    """One type at bodyId level is a legitimate run: the pairwise detail is the
    result, and it is filed under bodyid_level/ — not under type_level/, whose
    pooled-profile semantics this selection never produces."""
    comp = _comparer([1, 2], tmp_path, aggregation_level='bodyid')
    fake = _FakeProfilerFull(
        profiles={
            ('1', DS_A): _rich_profile(1, DS_A),
            ('2', DS_A): _rich_profile(2, DS_A),
        },
    )
    fake.types_for_bodyids[DS_A] = {1: 'T1', 2: 'T1'}
    comp.profiler = fake
    res = comp.run()
    assert res['n_type_profiles'] == 2
    assert res['bodyid_level_skipped'] is False
    assert res['type_matrices'] == {}
    assert set(res['bodyid_matrices']) == {'overall', 'upstream', 'downstream'}
    assert set(res['type_avg_matrices']) == {'overall', 'upstream', 'downstream'}
    assert '1_T1' in res['type_labels']

    out = Path(res['output_path'])
    assert len(list((out / 'bodyid_level' / 'results').glob(
        'bodyid_similarity_*.csv'))) == 15
    assert len(list((out / 'bodyid_level' / 'results').glob(
        'type_avg_bodyid_similarity_*.csv'))) == 15
    # Nothing was pooled, so neither the type matrices nor their folder exist
    assert not (out / 'type_level').exists()
    assert not (out / 'profiles' / 'aggregated').exists()
    assert len(list((out / 'profiles' / 'individual').glob('*.json'))) == 2
    readme = (out / 'README.txt').read_text(encoding='utf-8')
    assert 'Levels computed: bodyid-level, type-average of bodyId pairs' in readme
    assert 'type_level/results' not in readme
    report = (out / 'report.html').read_text(encoding='utf-8')
    assert 'Type-level' not in report
    assert 'BodyId-level' in report and 'Type-average BodyId' in report
    assert 'not computed for this run' not in report
    params = json.loads((out / 'parameters.json').read_text(encoding='utf-8'))
    assert params['row_kind'] == 'neuron'
    assert params['levels_computed'] == ['bodyid-level',
                                         'type-average of bodyId pairs']


def test_comparer_run_custom_group_aggregation(tmp_path, pc_fake_repo):
    """A group×group matrix is filed under group_level/, not type_level/.

    The folder and file names describe the axes, so the custom level cannot
    read as a type comparison — the same rule the morphology comparer applies
    to the same selection. ``bodyid_profiles`` is keyed by the GROUP and its
    member item (measured on real data: ``('pair-set', 'APL')``), which is why
    the row-average below reads as group-average.
    """
    comp = _comparer([['T1', 'T2'], ['T3']], tmp_path,
                     aggregation_level='custom group', skip_bodyId_level=False)
    p1, p2, p3 = (_rich_profile(1, DS_A), _rich_profile(2, DS_A),
                  _rich_profile(3, DS_A))
    p1.neuron_type = 'T1'
    p2.neuron_type = 'T2'
    p3.neuron_type = 'T3'
    comp._extract_all_profiles = lambda: (
        {'GroupA': p1, 'GroupB': p3},
        {('GroupA', 'T1'): p1, ('GroupA', 'T2'): p2, ('GroupB', 'T3'): p3})
    res = comp.run()

    out = Path(res['output_path'])
    assert len(list((out / 'group_level' / 'results').glob(
        'group_similarity_*.csv'))) == 15
    assert not (out / 'type_level').exists()
    assert len(list((out / 'bodyid_level' / 'results').glob(
        'bodyid_similarity_*.csv'))) == 15
    avg = pd.read_csv(out / 'bodyid_level' / 'results'
                      / 'type_avg_bodyid_similarity_jaccard_overall.csv',
                      index_col=0)
    assert list(avg.index.astype(str)) == ['GroupA', 'GroupB']
    params = json.loads((out / 'parameters.json').read_text(encoding='utf-8'))
    assert params['row_kind'] == 'group'
    assert params['levels_computed'] == ['group-level', 'bodyId-level',
                                         'Group-average of bodyId pairs']
    report = (out / 'report.html').read_text(encoding='utf-8')
    assert 'Group-level' in report and 'Type-level' not in report
    assert 'group_level/results/group_similarity_jaccard_overall.csv' in report
    assert 'Custom group' in report, "the group axes must be titled as groups"
    readme = (out / 'README.txt').read_text(encoding='utf-8')
    assert 'group_level/results/' in readme and 'type_level/' not in readme
    assert 'Group-average of bodyId pairs' in readme


def test_comparer_run_multi_dataset(tmp_path, pc_fake_repo):
    comp = ConnectivityProfileComparer(
        ['T1', 'T2'], datasets=[DS_A, DS_B],
        output_dir=str(tmp_path), verbose=False,
        generate_heatmaps=False, use_auto_type_mapping=False,
        skip_bodyId_level=False)
    comp.profiler = _FakeProfilerFull(
        profiles={
            ('1', DS_A): _rich_profile(1, DS_A),
            ('2', DS_A): _rich_profile(2, DS_A),
            ('3', DS_A): _rich_profile(3, DS_A),
            ('10', DS_B): _rich_profile(10, DS_B),
            ('11', DS_B): _rich_profile(11, DS_B),
        },
        bodyids={
            ('T1', DS_A): [1, 2], ('T2', DS_A): [3],
            ('T1', DS_B): [10], ('T2', DS_B): [11],
        },
    )
    res = comp.run()

    assert res['is_multi_dataset'] is True
    assert res['is_cross_dataset'] is False
    assert res['bodyid_level_skipped'] is False
    assert sorted(res['inter_anchors']) == ['T1', 'T2']
    assert set(res['intra_matrices_by_dataset']) == {DS_A, DS_B}
    assert set(res['inter_type_matrices']) == \
        {'overall', 'upstream', 'downstream'}

    out = Path(res['output_path'])
    assert (out / 'parameters.json').exists()
    assert (out / 'README.txt').exists()
    assert (out / 'report.html').exists()
    intra_res = out / 'intra_dataset' / SAFE_A / 'results'
    assert (intra_res / 'similarity_overall_jaccard.csv').exists()
    assert (intra_res / 'bodyid_similarity_jaccard_overall.csv').exists()
    assert (intra_res / 'type_avg_bodyid_similarity_jaccard_overall.csv').exists()
    cross_res = out / 'cross_dataset' / 'all_types' / 'results'
    assert (cross_res / 'similarity_overall_jaccard.csv').exists()
    assert (out / 'cross_dataset' / 'mapping_summary.csv').exists()
    assert (out / 'profiles' / SAFE_A / 'aggregated' / 'T1_profile.json').exists()
    assert (out / 'profiles' / SAFE_A / 'individual' / '1_T1_profile.json').exists()

    mapping_df = pd.read_csv(out / 'cross_dataset' / 'mapping_summary.csv')
    assert list(mapping_df['anchor']) == ['T1', 'T2']
    assert (mapping_df['same name'] == 1).all()


def test_comparer_run_cross_dataset(tmp_path):
    comp = ConnectivityProfileComparer(
        {DS_A: ['T1', 5], DS_B: ['T2']},
        output_dir=str(tmp_path), verbose=False,
        generate_heatmaps=False, use_auto_type_mapping=False)
    comp.profiler = _FakeProfilerFull(
        profiles={
            ('1', DS_A): _rich_profile(1, DS_A),
            ('5', DS_A): _rich_profile(5, DS_A),
            ('7', DS_B): _rich_profile(7, DS_B),
        },
        bodyids={('T1', DS_A): [1], ('T2', DS_B): [7]},
    )
    res = comp.run()

    assert res['is_cross_dataset'] is True
    assert res['n_type_profiles'] == 3
    assert res['bodyid_level_skipped'] is True
    assert res['row_labels'] == ['T1', '5']
    assert res['col_labels'] == ['T2']
    assert 'overall' in res['cross_matrices']
    assert res['cross_matrices']['overall']['jaccard'].loc['T1', 'T2'] == \
        pytest.approx(1.0)

    out = Path(res['output_path'])
    assert (out / 'metadata.json').exists()
    assert (out / 'cross_dataset' / 'overall_jaccard.csv').exists()
    assert (out / 'profiles' / DS_A / 'T1_profile.json').exists()
    meta = json.loads((out / 'metadata.json').read_text(encoding='utf-8'))
    assert meta['datasets'] == [DS_A, DS_B]

    # no resolvable profiles -> raises rather than exiting 0 with no files
    comp_empty = ConnectivityProfileComparer(
        {DS_A: ['Nope'], DS_B: ['Nada']},
        output_dir=str(tmp_path), verbose=False,
        generate_heatmaps=False, use_auto_type_mapping=False)
    comp_empty.profiler = _FakeProfilerFull()
    with pytest.raises(ValueError, match='no profiles extracted'):
        comp_empty.run()


def test_comparer_compare_intra_inter_type(tmp_path):
    comp = _comparer(['T1', 'T2'], tmp_path)
    comp.profiler = _FakeProfilerFull(
        profiles={
            ('1', DS_A): _rich_profile(1, DS_A, scale=1.0),
            ('2', DS_A): _rich_profile(2, DS_A, scale=0.95),
            ('3', DS_A): _rich_profile(3, DS_A, scale=1.0),
            ('4', DS_A): _rich_profile(4, DS_A, scale=1.05),
        },
        bodyids={('T1', DS_A): [1, 2], ('T2', DS_A): [3, 4]},
    )
    res = comp.compare_intra_inter_type()
    assert len(res['intra_type']) == 2     # T1 pair + T2 pair
    assert len(res['inter_type']) == 4     # 2x2 sampled pairs
    assert set(res['intra_type'].columns) >= {'type', 'bodyId_a', 'bodyId_b',
                                              'jaccard', 'cosine'}

    # all-numeric query -> nothing to compare
    comp_num = _comparer([1, 2], tmp_path)
    comp_num.profiler = _FakeProfilerFull()
    res_num = comp_num.compare_intra_inter_type()
    assert res_num['intra_type'].empty and res_num['inter_type'].empty


# ---------------------------------------------------------------------------
# Chunk 6: remaining comparer branches (mappers, anchors, extraction edges)
# ---------------------------------------------------------------------------

class _FakeTypeMapper:
    """Stand-in for CrossDatasetTypeMapper following the shared resolver
    contract: ``get_mapping_decision`` returns the status dict that
    ``comparison.type_resolver`` consumes (the legacy
    ``resolve_type_across_datasets`` one-target API is compatibility-only
    now)."""

    _loaded = True

    def _detect_type_source(self, name):
        return 'detected_source_ns'

    def _get_type_mapping_key(self, dataset):
        return dataset

    def get_mapping_decision(self, source_type, source_dataset,
                             target_dataset, include_bridges=False):
        return {
            'status': 'mapped',
            'source_type': source_type,
            'target_type': f'{source_type}_mapped',
            'target_types': [f'{source_type}_mapped'],
            'relationship': '1-to-1',
            'conflicts': [],
        }

    def get_alias_candidates(self, type_name, datasets, source_dataset=None):
        return {ds: {'outcome': 'no counterpart known', 'candidates': []}
                for ds in datasets}

    def get_type_bridges(self, type_name, source_ds, target_ds,
                         max_bridges=8):
        return []

    def get_canonical_type(self, label, dataset):
        return f'{label}_canon'


class _SelectiveRaisingProfiler(_FakeProfilerFull):
    """get_profile raises for bodyIds listed in raise_for."""

    def __init__(self, raise_for=(), **kwargs):
        super().__init__(**kwargs)
        self.raise_for = set(raise_for)

    def get_profile(self, neuron, dataset, force_refresh=False):
        try:
            if int(neuron) in self.raise_for:
                raise RuntimeError('profile fetch failed')
        except (TypeError, ValueError):
            pass
        return super().get_profile(neuron, dataset, force_refresh)


def test_comparer_verbose_log_and_progress(tmp_path):
    comp = _comparer(['T1'], tmp_path, verbose=True)
    comp._log('hello')            # verbose print branch
    comp._progress(1, 2, 'step')  # structured progress branch


def test_comparer_multi_init_with_mappings(tmp_path):
    csv = tmp_path / 'groups.csv'
    csv.write_text('group,id_type_instance\nG1,Mi1\nG2,Tm3\n', encoding='utf-8')
    comp = ConnectivityProfileComparer(
        ['ignored'], datasets=[DS_A, DS_B], group_map_csv=str(csv),
        verbose=False, use_auto_type_mapping=False)
    assert comp._custom_group_names == ['G1', 'G2']

    preset = tmp_path / 'preset.json'
    preset.write_text(json.dumps({'source_mapping': {
        'custom_label': ['g1'], DS_A: [['T1']]}
    }), encoding='utf-8')
    comp2 = ConnectivityProfileComparer(
        ['ignored'], datasets=[DS_A, DS_B],
        custom_mapping_file=str(preset),
        verbose=False, use_auto_type_mapping=False)
    assert comp2.aggregation_level == 'custom'
    assert comp2._custom_group_names == ['g1']


def test_comparer_custom_groups_mapping_edges(tmp_path):
    comp = _comparer(['T1'], tmp_path)

    def _write(data):
        p = tmp_path / 'preset.json'
        p.write_text(json.dumps(data), encoding='utf-8')
        return str(p)

    # string member list (single member not wrapped in a list)
    normalized, names = comp._load_custom_groups_from_mapping(
        _write({'source_mapping': {'custom_label': ['g1'],
                                   DS_A: ['T1']}}), DS_A)
    assert (normalized, names) == ([['T1']], ['g1'])

    # all rows empty -> warning + ([], [])
    assert comp._load_custom_groups_from_mapping(
        _write({'source_mapping': {'custom_label': ['g1'],
                                   DS_A: [[]]}}), DS_A) == ([], [])

    # multi-dataset mode with a bare-string row
    comp_multi = ConnectivityProfileComparer(
        ['T1'], datasets=[DS_A, DS_B], verbose=False,
        use_auto_type_mapping=False)
    comp_multi._load_custom_groups_from_mapping(
        _write({'source_mapping': {'custom_label': ['g1'],
                                   DS_A: [['T1']],
                                   DS_B: ['plain']}}), DS_A)
    assert comp_multi._mapping_ds_members[DS_B] == [['plain']]


def test_comparer_query_name_and_neuron_edges(tmp_path):
    comp = _comparer(['T1'], tmp_path)

    # query whose first item is a list
    comp.query = [[1, 2]]
    assert comp._generate_query_name() == '1'

    # defaults from self.query / self.dataset
    fake = _FakeProfilerFull(bodyids={('T1', DS_A): [1]})
    comp.profiler = fake
    comp.query = ['T1']
    assert comp._get_neurons_to_compare() == {'T1': [1]}

    # non-list group member gets wrapped
    out = comp._get_neurons_to_compare(query=[5], custom_group_names=['G'])
    assert out == {'G': [5]}

    # pattern whose matched type has no bodyIds is skipped
    fake.type_lists[('p.*', DS_A)] = ['Empty']
    assert comp._get_neurons_to_compare(query=['p.*']) == {}

    # taxonomy expansion at bodyid aggregation level
    fake.label_groups[('clock', DS_A)] = {'cA': [20]}
    out = comp._get_neurons_to_compare(query=['clock'],
                                       aggregation_level='bodyid')
    assert out == {'20_cA': [20]}


def test_comparer_type_mapper_branches(tmp_path):
    comp = _comparer(['T1'], tmp_path)
    comp._type_mapper = _FakeTypeMapper()
    assert comp._map_query_item('Mi1', DS_A) == 'Mi1_mapped'
    assert comp._map_query_item(5, DS_A) == 5
    assert comp._map_query_item('Mi.*', DS_A) == 'Mi.*'
    # anchor merge key via the shared resolver: a licensed rename keys
    # under its canonical target (not the legacy raw get_canonical_type)
    assert comp._canonical_label('Mi1', DS_A) == 'Mi1_mapped'

    # multi-dataset mapped query (flat and custom-group forms)
    comp_m = ConnectivityProfileComparer(
        ['T1'], datasets=[DS_A, DS_B], verbose=False,
        use_auto_type_mapping=False)
    comp_m._type_mapper = _FakeTypeMapper()
    assert comp_m._mapped_query_for(DS_A) == ['T1_mapped']
    comp_m._custom_group_names = ['G']
    comp_m.query = [['Mi1']]
    assert comp_m._mapped_query_for(DS_A) == [['Mi1_mapped']]
    comp_m._mapping_ds_members = {DS_A: [['X', 5]]}
    assert comp_m._mapped_query_for(DS_A) == [['X', 5]]


def test_comparer_multi_run_bodyid_and_pattern_anchors(tmp_path, pc_fake_repo):
    comp = ConnectivityProfileComparer(
        [1, 'aMe.*'], datasets=[DS_A, DS_B],
        output_dir=str(tmp_path), verbose=False,
        generate_heatmaps=False, use_auto_type_mapping=False,
        skip_bodyId_level=True)
    comp.profiler = _FakeProfilerFull(
        profiles={
            ('1', DS_A): _rich_profile(1, DS_A),
            ('10', DS_A): _rich_profile(10, DS_A),
            ('1', DS_B): _rich_profile(1, DS_B),
            ('10', DS_B): _rich_profile(10, DS_B),
        },
        bodyids={
            ('T1', DS_A): [1], ('aMe1', DS_A): [10],
            ('T1', DS_B): [1], ('aMe1', DS_B): [10],
        },
    )
    comp.profiler.types_for_bodyids[DS_A] = {1: 'T1'}
    comp.profiler.types_for_bodyids[DS_B] = {1: 'T1'}
    comp.profiler.type_lists[('aMe.*', DS_A)] = ['aMe1']
    comp.profiler.type_lists[('aMe.*', DS_B)] = ['aMe1']

    res = comp.run()
    assert res['is_multi_dataset'] is True
    # bodyId anchor '1' (via its type) + pattern anchor 'aMe1'
    assert sorted(res['inter_anchors']) == ['1', 'aMe1']


def test_comparer_anchor_fails_closed_on_conflict():
    """A conflicted type contributes NO anchor across datasets (fail closed):
    the anchor path must not form a raw same-name merge the resolver
    refuses."""
    comp = ConnectivityProfileComparer.__new__(ConnectivityProfileComparer)
    comp.query = ['CB1011']
    comp._custom_group_names = None
    comp.aggregation_level = 'type'
    comp.datasets = [DS_A, DS_B]

    class _ConflictMapper:
        _loaded = True

        def _get_type_mapping_key(self, ds):
            return ds

        def _detect_type_source(self, name):
            # the item lives in a third namespace; toward the queried
            # datasets its verdict is a conflict
            return 'banc_v626'

        def get_mapping_decision(self, source_type, source_dataset,
                                 target_dataset, include_bridges=False):
            return {'status': 'conflict', 'source_type': source_type,
                    'target_type': None, 'target_types': [],
                    'relationship': '1-to-N', 'conflicts': [{}]}

        def get_alias_candidates(self, type_name, datasets,
                                 source_dataset=None):
            return {ds: {'outcome': 'no counterpart known',
                         'candidates': []} for ds in datasets}

        def get_type_bridges(self, *a, **k):
            return []

    comp._type_mapper = _ConflictMapper()
    comp._mapper_snapshot = None
    comp._merge_key_cache = {}
    comp.mapping_resolution_record = {}
    comp.mapping_resolution_counts = {}
    comp.mapping_raw_fallback_used = False
    comp._counted_query_resolutions = set()

    profiles = {
        DS_A: {'CB1011': _rich_profile('a', DS_A)},
        DS_B: {'CB1011': _rich_profile('b', DS_B)},
    }
    anchors = comp._build_anchor_profiles(profiles)
    assert anchors == {}  # no raw same-name anchor for a conflicted type


def test_comparer_anchor_no_mapper_passthrough():
    """With auto mapping disabled the anchor uses raw names as-is."""
    comp = ConnectivityProfileComparer.__new__(ConnectivityProfileComparer)
    comp.query = ['T1']
    comp._custom_group_names = None
    comp.aggregation_level = 'type'
    comp.datasets = [DS_A, DS_B]
    comp._type_mapper = None
    comp._merge_key_cache = {}

    profiles = {
        DS_A: {'T1': _rich_profile('a', DS_A)},
        DS_B: {'T1': _rich_profile('b', DS_B)},
    }
    anchors = comp._build_anchor_profiles(profiles)
    assert list(anchors.keys()) == ['T1']


def test_comparer_multi_run_custom_group_anchors(tmp_path, pc_fake_repo):
    comp = ConnectivityProfileComparer(
        [['G1', [1]], ['G2', [2]]], datasets=[DS_A, DS_B],
        output_dir=str(tmp_path), verbose=False,
        generate_heatmaps=False, use_auto_type_mapping=False,
        skip_bodyId_level=True)
    comp.profiler = _FakeProfilerFull(
        profiles={
            ('1', DS_A): _rich_profile(1, DS_A),
            ('2', DS_A): _rich_profile(2, DS_A),
            ('1', DS_B): _rich_profile(1, DS_B),
            ('2', DS_B): _rich_profile(2, DS_B),
        },
    )
    res = comp.run()
    assert res['is_multi_dataset'] is True
    assert sorted(res['inter_anchors']) == ['G1', 'G2']


def test_comparer_cross_extraction_edges(tmp_path):
    # calling the cross extraction outside cross mode raises
    comp_plain = _comparer(['T1'], tmp_path)
    with pytest.raises(RuntimeError):
        comp_plain._extract_cross_dataset_profiles()

    comp = ConnectivityProfileComparer(
        {DS_A: ['5', 'T1', 'Ghost'], DS_B: ['T2']},
        output_dir=str(tmp_path), verbose=False,
        generate_heatmaps=False, use_auto_type_mapping=False)
    comp.profiler = _SelectiveRaisingProfiler(
        raise_for={3},
        profiles={
            ('5', DS_A): _rich_profile(5, DS_A),
            ('1', DS_A): _rich_profile(1, DS_A),
            ('2', DS_A): _rich_profile(2, DS_A),
            ('7', DS_B): _rich_profile(7, DS_B),
        },
        bodyids={('T1', DS_A): [1, 2, 3], ('Ghost', DS_A): [9],
                 ('T2', DS_B): [7]},
    )
    profiles_by_ds, row_labels, col_labels = comp._extract_cross_dataset_profiles()
    # digit-string query kept as label, T1 aggregated from 2 surviving bids,
    # Ghost dropped (profile fetch failures)
    assert row_labels == ['5', 'T1']
    assert col_labels == ['T2']
    assert profiles_by_ds[DS_A]['T1'].num_neurons_aggregated == 2


def test_comparer_run_ensure_cache_branch(tmp_path, monkeypatch):
    _install_fake_coana(monkeypatch)
    comp = _comparer(['OnlyOne'], tmp_path,
                     use_cache=True, ensure_cache_complete=True)
    comp.profiler = _FakeProfilerFull(
        profiles={('1', DS_A): _rich_profile(1, DS_A)},
        bodyids={('OnlyOne', DS_A): [1]},
    )
    # cache ensured first, then the population refusal raises
    with pytest.raises(ValueError, match='at least two neurons in scope'):
        comp.run()


def test_comparer_intra_inter_ensure_cache_and_profile_error(
        tmp_path, monkeypatch):
    _install_fake_coana(monkeypatch)
    comp = _comparer(['T1'], tmp_path,
                     use_cache=True, ensure_cache_complete=True)
    comp.profiler = _SelectiveRaisingProfiler(
        raise_for={2},
        profiles={('1', DS_A): _rich_profile(1, DS_A)},
        bodyids={('T1', DS_A): [1, 2]},
    )
    res = comp.compare_intra_inter_type()
    # bodyId 2 fails to extract -> only 1 profile -> no pairs
    assert res['intra_type'].empty and res['inter_type'].empty


# ---------------------------------------------------------------------------
# Coarse class-label backfill in dataset `type` columns (e.g. FAFB v783
# writes `type='optic_lobes'` / `cell_class='optic_lobes'` on untyped
# neurons) must not leak into the homolog type lookup.
# ---------------------------------------------------------------------------

def test_coarse_type_labels_detected_from_neuron_index(tmp_path):
    from comparison.profile_comparator import HomologFinder
    safe = "flywire_FAFB_v783"
    idx_dir = tmp_path / "neuron_indexes" / safe
    idx_dir.mkdir(parents=True)
    pd.DataFrame({
        "bodyId": ["1", "2", "3"],
        "type": ["optic_lobes", "aMe12", "Optic"],
        "cell_class": ["optic_lobes", "optic_lobes", "optic_lobes"],
        "super_class": ["optic", "optic", "visual"],
    }).to_parquet(idx_dir / "neuron_index.parquet")
    finder = HomologFinder.__new__(HomologFinder)
    coarse = finder._coarse_type_labels("flywire_FAFB_v783",
                                        project_root=str(tmp_path))
    # Only 'optic_lobes' is a backfill: row 1 carries it as both type and
    # cell_class. Row 3's 'Optic' never equals one of its own class labels,
    # and the super_class value 'optic' never appears as a type.
    assert coarse == frozenset({"optic_lobes"})


def test_build_bodyid_type_lookup_skips_coarse_labels(tmp_path, monkeypatch):
    from comparison.profile_comparator import HomologFinder
    finder = HomologFinder.__new__(HomologFinder)
    monkeypatch.setattr(
        HomologFinder, "_coarse_type_labels",
        lambda self, ds, project_root=None: frozenset({"optic_lobes"}))
    conn = pd.DataFrame({
        "bodyId_pre": [1, 2, 3],
        "bodyId_post": [10, 11, 12],
        "type_pre": ["aMe12", "optic_lobes", "aMe12"],
        "type_post": ["Mi15", "Mi15", "optic_lobes"],
    })
    lookup = finder._build_bodyid_type_lookup(conn, "flywire_FAFB_v783")
    # bodyId 2 only had the coarse pre-type (skipped); bodyId 12 only had
    # the coarse post-type (skipped).
    assert lookup == {1: "aMe12", 3: "aMe12", 10: "Mi15", 11: "Mi15"}


def test_build_bodyid_type_lookup_keeps_everything_without_dataset():
    from comparison.profile_comparator import HomologFinder
    finder = HomologFinder.__new__(HomologFinder)
    conn = pd.DataFrame({
        "bodyId_pre": [1],
        "bodyId_post": [2],
        "type_pre": ["optic_lobes"],
        "type_post": ["Mi15"],
    })
    lookup = finder._build_bodyid_type_lookup(conn)
    assert lookup == {1: "optic_lobes", 2: "Mi15"}
