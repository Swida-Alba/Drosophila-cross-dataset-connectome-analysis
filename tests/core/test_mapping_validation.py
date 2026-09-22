"""Tests for the type-mapping validation pipeline
(plan-type-mapping-validation-pipeline.md).

Offline unit tests on synthetic ConnectivityProfile objects:
- scorer parity with ProfileComparator.batch_compare_cross_dataset
- global ranks + tiered verdicts (verified_strong / verified / borderline /
  unmatched)
- suspicious-candidate detection (non-pool neurons ranked ahead)
- mutual-best greedy assignment (contest-loser case; confident verdicts
  only)
- gap rule (min-pool operand, >20% and >5 triggers, boundaries)
- gap fill with in_pool / out_of_pool tagging
- int64 bodyId integrity through scoring and DataFrames
- morphology AUC calibration + Youden threshold helpers

No network, no caches: profiles are constructed in memory and the
CrossDatasetTypeMapper is not needed (type_mapper=None exercises the raw
expanded-type path on both sides).

Scenario geometry: the source neuron's merged partner-type vector is
{A:10, B:8, P:6, C:4, Q:3, D:2}.  Note Jaccard is set-based, so vectors
sharing the source's key SET all tie at jaccard = 1.0; rank separation
must come from weight ORDER (rank_union) or from set differences.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.connectivity_profiler import (  # noqa: E402
    ConnectivityProfile,
)
from comparison.mapping_validation import (  # noqa: E402
    MappingValidationConfig,
    MappingValidator,
    TypePair,
    annotate_pair_branches,
    best_threshold,
    build_target_vectors,
    expanded_vector,
    gap_check,
    measure_branch_disjointness,
    morph_auc,
    mutual_best_assignment,
    categorize_pool_targets,
    prep_target_stats,
    scan_source,
    score_one_candidate,
    score_one_candidate_fast,
    _SideStats,
    _write_csv,
)
from comparison.profile_comparator import ProfileComparator  # noqa: E402

FAFB_SIZED_ID = 720575940647731252  # > 2**53: float64 cannot represent it


def make_profile(bid, up, dn):
    return ConnectivityProfile(
        neuron_id=bid, dataset='test',
        upstream_partners=dict(up),
        downstream_partners=dict(dn),
        actual_upstream_count=len(up),
        actual_downstream_count=len(dn),
    )


def vec(up=None, dn=None):
    return expanded_vector(make_profile(1, up or {}, dn or {}), None)


def run_validator(source_pool, target_pool, target_vectors_by_bid,
                  source_profiles_by_bid):
    cfg = MappingValidationConfig(
        source_dataset='dsA', target_dataset='dsB',
        query_types=['T'], morph_enabled=False, visualize=False)
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = cfg
    v.mapper = None
    v.notes = []

    pair = TypePair(
        source_dataset='dsA', source_type='T', source_pool=source_pool,
        target_dataset='dsB', target_type='T', target_pool=target_pool)

    class FakeProfiler:
        def get_profile(self, bid, dataset):
            return source_profiles_by_bid.get(bid)

        def get_types_for_bodyids(self, bids, dataset):
            return {}

    v.profiler = FakeProfiler()
    target_stats = prep_target_stats(target_vectors_by_bid)
    scans = {bid: scan_source(expanded_vector(p, None), target_stats)
             for bid, p in source_profiles_by_bid.items()}
    res = v.validate_pair(pair, scans)
    return v, pair, res


SRC_UP = {'A': 10, 'B': 8, 'C': 4, 'D': 2}
SRC_DN = {'P': 6, 'Q': 3}


def scenario(targets_updown, pool, src_bid=1):
    src = make_profile(src_bid, SRC_UP, SRC_DN)
    tgt_vectors = {bid: expanded_vector(make_profile(bid, up, dn), None)
                   for bid, (up, dn) in targets_updown.items()}
    _, _, res = run_validator([src_bid], pool, tgt_vectors, {src_bid: src})
    return res


# ---------------------------------------------------------------------------
# Scorer parity with the production comparator
# ---------------------------------------------------------------------------

def test_scorer_parity_with_batch_compare():
    src = make_profile(101, {'A': 12, 'B': 5, 'C': 2, 'D': 1},
                       {'X': 8, 'Y': 3})
    tgts = {
        201: make_profile(201, {'A': 9, 'B': 6, 'C': 3, 'E': 2},
                          {'X': 7, 'Z': 1}),
        202: make_profile(202, {'B': 4, 'F': 2}, {'Y': 5, 'W': 2}),
    }
    got = score_one_candidate(expanded_vector(src, None),
                              expanded_vector(tgts[201], None))
    prod = ProfileComparator.batch_compare_cross_dataset(
        src, tgts, {201: 0, 202: 0}, direction='both', type_mapper=None)
    prod201 = next(r for r in prod if r['target_bid'] == 201)
    for metric in ('jaccard', 'rank_union', 'cosine', 'weighted_jaccard'):
        a, b = got[metric], prod201[metric]
        if a is None or (isinstance(a, float) and np.isnan(a)):
            assert b is None or (isinstance(b, float) and np.isnan(b)), metric
        else:
            assert a == pytest.approx(b, abs=1e-12), metric


def test_fast_scorer_matches_reference():
    """The numpy fast path must equal the reference dict scorer exactly."""
    src = {'A': 10, 'B': 8, 'P': 6, 'C': 4, 'Q': 3, 'D': 2}
    targets = [
        {'A': 10, 'B': 8, 'P': 6, 'C': 4, 'Q': 3, 'D': 2},      # identical
        {'A': 2, 'B': 3, 'P': 6, 'C': 4, 'Q': 30, 'D': 5},      # scrambled
        {'A': 9, 'B': 7, 'P': 5, 'C': 3},                       # subset
        {'Z': 1},                                               # disjoint
        {'F': 4, 'G': 2, 'H': 1},                               # disjoint 2
    ]
    ref = score_one_candidate(src, targets[0])
    fast = score_one_candidate_fast(_SideStats(src), _SideStats(targets[0]))
    for metric in ('jaccard', 'rank_union', 'cosine', 'weighted_jaccard'):
        a, b = ref[metric], fast[metric]
        assert (a is None or np.isnan(a)) == \
            (b is None or (isinstance(b, float) and np.isnan(b))), metric
        if a is not None and not np.isnan(a):
            assert a == pytest.approx(b, abs=1e-9), metric
    # and the full production comparator agrees on rank_union too
    sp = make_profile(1, src, {})
    tgts = {i + 200: make_profile(i + 200, t, {})
            for i, t in enumerate(targets)}
    prod = ProfileComparator.batch_compare_cross_dataset(
        sp, tgts, {b: 0 for b in tgts}, direction='both', type_mapper=None)
    prod_by_bid = {r['target_bid']: r for r in prod}
    for bid, t in enumerate(targets):
        fast = score_one_candidate_fast(_SideStats(src), _SideStats(t))
        p = prod_by_bid[bid + 200]
        a, b = fast['rank_union'], p['rank_union']
        a_nan = a is None or np.isnan(a)
        b_nan = b is None or np.isnan(b)
        if a_nan or b_nan:
            assert a_nan and b_nan, bid
        else:
            assert a == pytest.approx(b, abs=1e-9), bid


# ---------------------------------------------------------------------------
# Global ranks + tiered verdicts
# ---------------------------------------------------------------------------

def test_verified_strong_requires_both_metrics_top1():
    # 201 keeps the source's weight ORDER (A>B>P>C>Q>D) on the same set:
    # rank_union 1.0 (rank 1) and jaccard 1.0 (rank 1).
    res = scenario(
        {201: ({'A': 11, 'B': 7, 'C': 5, 'D': 2}, {'P': 6, 'Q': 3}),
         202: ({'Z': 3}, {}),
         203: ({'E': 5, 'F': 4, 'G': 1}, {}),
         204: ({'H': 8, 'I': 2, 'J': 2}, {'R': 4}),
         205: ({'K': 6, 'L': 6}, {'S': 9})},
        pool=[201])
    row = res['rows'][0]
    assert row['verdict'] == 'verified_strong'
    assert row['target_bodyId'] == 201
    assert row['rank_union_rank'] == 1 and row['jaccard_rank'] == 1
    assert res['pairs'] == [(1, 201)]


def test_the_verdict_is_the_pools_not_the_orderings():
    """D-A (plan-tmvev-jaccard-primary-bodyid-ranking) + its two corrections:
    the ladder quantifies over the POOL so a label cannot ride on the ordering
    key's accident, but a rank-1 claim lifts only the row that PUBLISHES the
    partner holding it.  Here 201 is the pool's rank_union top-1 while the
    chain-best (202) leads on jaccard, so no single member holds both:
    `verified_strong` cannot fire (r17's correction) and neither can
    `verified` (user 2026-09-21: "tighten verified too") — the rank_union win
    stays visible as `ru_top_target_bodyId` evidence instead.
    """
    res = scenario(
        {201: ({'A': 1, 'B': 2, 'C': 3, 'D': 4}, {'P': 1, 'Q': 2}),   # same SET, reversed order
         202: ({'A': 10, 'B': 8}, {'P': 6})},                         # same ORDER, partial set
        pool=[201, 202])
    row = res['rows'][0]
    assert row['verdict'] == 'verified'
    assert row['metric_top1'] == 'jaccard'
    # the chain (Jaccard first) publishes 201; 202 carries the rank_union win
    assert row['target_bodyId'] == 201
    assert row['jaccard_rank'] == 1 and row['rank_union_rank'] == 2
    assert row['ru_top_target_bodyId'] == 202


def test_a_claim_on_an_unpublished_member_is_evidence_not_a_verdict():
    """The discriminating shape for the tightened `verified`: the pool's
    rank_union top-1 (201) is NOT the published partner (202, the pool's
    Jaccard best), and 202 itself is behind an out-of-pool neuron on Jaccard,
    so the published row holds no rank-1 claim at all.  Under the old
    ∃-over-pool reading this row was `verified`; now it is `borderline` with
    the rank_union winner still published as evidence.  r18 measured 2 of 223
    real rows in exactly this shape (user 2026-09-21: "tighten verified too").
    """
    res = scenario(
        {201: ({'A': 10, 'B': 8}, {}),                              # pool, rank_union top-1
         202: ({'A': 10, 'B': 8, 'C': 4, 'D': 2, 'E': 3}, {}),      # pool, Jaccard best of the two
         203: ({'A': 10, 'B': 8, 'C': 4, 'D': 2}, {})},             # out of pool, global Jaccard top-1
        pool=[201, 202])
    row = res['rows'][0]
    assert row['verdict'] == 'borderline'
    assert row['metric_top1'] == ''
    assert row['target_bodyId'] == 202
    assert row['jaccard_rank'] == 2 and row['rank_union_rank'] == 3
    assert row['ru_top_target_bodyId'] == 201


def test_verified_single_metric_top1():
    # 201 shares the source's SET (jaccard 1.0 -> tied rank 1) but with a
    # scrambled weight ordering (poor rank_union); only 201 is in the pool
    # so the best pool member is top-1 under jaccard alone -> 'verified'
    # with the metric recorded, plus 202 outranking it under rank_union.
    res = scenario(
        {201: ({'A': 2, 'B': 3, 'C': 4, 'D': 5}, {'P': 6, 'Q': 30}),
         202: ({'A': 20, 'B': 16, 'C': 8, 'D': 4}, {'P': 12, 'Q': 6})},
        pool=[201])
    row = res['rows'][0]
    assert row['verdict'] == 'verified'
    assert row['metric_top1'] == 'jaccard'
    assert row['rank_union_rank'] == 2  # 202 (2x the source) takes rank 1
    assert 'suspicious_ahead' in row['flags']


def test_borderline_window():
    # 201 differs slightly from the source (Q swapped for R: jaccard 6/7,
    # near-perfect order) while non-pool 202 IS the source exactly ->
    # 201 is rank 2 under BOTH metrics -> borderline (top-5, not top-1).
    res = scenario(
        {201: ({'A': 10, 'B': 8, 'C': 4, 'D': 2}, {'P': 6, 'R': 3}),
         202: ({'A': 10, 'B': 8, 'C': 4, 'D': 2}, {'P': 6, 'Q': 3})},
        pool=[201])
    row = res['rows'][0]
    assert row['verdict'] == 'borderline'
    assert row['rank_union_rank'] == 2
    assert row['jaccard_rank'] == 2
    assert row['suspicious_count'] == 1


def test_unmatched_when_pool_member_outside_window():
    # pool member 201 is a near-empty profile; six non-pool neurons are
    # progressively closer to the source -> 201 ranks ~7th globally.
    res = scenario(
        {201: ({'Z': 1}, {}),
         202: ({'A': 10, 'B': 8, 'C': 4, 'D': 2}, {'P': 6, 'Q': 3}),
         203: ({'A': 9, 'B': 7, 'C': 3, 'D': 1}, {'P': 5}),
         204: ({'A': 8, 'B': 6, 'C': 2}, {'P': 4}),
         205: ({'A': 7, 'B': 5, 'C': 1}, {'P': 3}),
         206: ({'A': 6, 'B': 4}, {'P': 2}),
         207: ({'A': 5, 'B': 3}, {})},
        pool=[201])
    row = res['rows'][0]
    assert row['verdict'] == 'unmatched'
    # an unassigned confident pool is required for M: unmatched sources are
    # never assigned
    assert res['pairs'] == []


def test_positive_rank_union_required_for_evidence():
    """Only a POSITIVE rank_union counts: a top-1 rank earned by a
    less-negative score on a weak profile is not evidence (user policy
    2026-09-11).  Pool member P is the reversed source (rank_union -1.0,
    rank 1 when all candidates are negative) with jaccard 1.0."""
    # pool member 201 shares the source's key SET (jaccard 1.0) but with
    # an anti-correlated weight order -> rank_union ~ -1, still rank 1
    # because the only other candidate is disjoint.  The negative top-1
    # must NOT count as rank_union evidence (positivity policy).
    res = scenario(
        {201: ({'A': 1, 'B': 2, 'P': 4, 'C': 6, 'Q': 8, 'D': 10}, {}),
         202: ({'A': 2, 'B': 4, 'P': 8, 'C': 12, 'Q': 16, 'D': 20}, {})},
        pool=[201])
    row = res['rows'][0]
    assert row['verdict'] == 'verified'
    assert row['metric_top1'] == 'jaccard'
    assert row['rank_union_rank'] == 1          # tied at the top...
    assert row['rank_union'] < 0                # ...but negative: no weight
    # 'verified' sources still participate in mutual-best assignment
    assert res['pairs'] == [(1, 201)]


def test_suspicious_negative_rank_union_noise_filtered():
    """A non-pool neuron that wins on negative rank_union only is
    filtered from the suspicious list (noise), while a positive
    rank_union win is kept."""
    # 202 anti-correlates with the source (negative rank_union); 203 is
    # an exact 2x copy (rank_union +1).  Both rank ahead of the weak pool
    # member 201, but only 203's positive win survives the filter.
    res = scenario(
        {201: ({'Z': 1}, {}),
         202: ({'A': 1, 'B': 2, 'P': 4, 'C': 6, 'Q': 8, 'D': 10}, {}),
         203: ({'A': 20, 'B': 16, 'P': 12, 'C': 8, 'Q': 6, 'D': 4}, {})},
        pool=[201])
    assert res['rows'][0]['verdict'] in ('unmatched', 'borderline')
    sus_types = {(s['ahead_target_bodyId'], s['ahead_metric'])
                 for s in res['suspicious']}
    # 203 wins positively -> kept under rank_union
    assert (203, 'rank_union') in sus_types
    # 202 wins with a negative score -> filtered as noise
    assert (202, 'rank_union') not in sus_types
    assert all(s['ahead_rank_union'] > 0
               for s in res['suspicious']
               if s['ahead_metric'] == 'rank_union')


def test_suspicious_ahead_detection():
    # 202 is the source scaled 2x: rank_union rank 1, ahead of pool member
    # 201 (whose P/Q weights are flipped -> rank 2).  201 still holds
    # jaccard rank 1 (same set, tie at 1.0) -> verdict 'verified' but with
    # a suspicious candidate reported under rank_union.
    res = scenario(
        {201: ({'A': 10, 'B': 8, 'C': 4, 'D': 2}, {'P': 2, 'Q': 9}),
         202: ({'A': 20, 'B': 16, 'C': 8, 'D': 4}, {'P': 12, 'Q': 6})},
        pool=[201])
    row = res['rows'][0]
    assert row['verdict'] == 'verified'
    assert row['metric_top1'] == 'jaccard'
    assert row['suspicious_count'] == 1
    assert 'suspicious_ahead' in row['flags']
    sus = res['suspicious']
    assert {s['ahead_target_bodyId'] for s in sus} == {202}
    assert {s['ahead_metric'] for s in sus} == {'rank_union'}
    assert all(isinstance(s['ahead_target_bodyId'], int) for s in sus)


# ---------------------------------------------------------------------------
# Assignment + gap rule + fill
# ---------------------------------------------------------------------------

def test_mutual_best_assignment_contest_loser():
    df1 = pd.DataFrame([
        {'target_bid': 11, 'rank_union': 0.9, 'rank_union_rank': 1,
         'jaccard': 0.9, 'jaccard_rank': 1},
        {'target_bid': 12, 'rank_union': 0.3, 'rank_union_rank': 3,
         'jaccard': 0.3, 'jaccard_rank': 3},
    ])
    df2 = pd.DataFrame([
        {'target_bid': 11, 'rank_union': 0.95, 'rank_union_rank': 1,
         'jaccard': 0.95, 'jaccard_rank': 1},
        {'target_bid': 13, 'rank_union': 0.2, 'rank_union_rank': 4,
         'jaccard': 0.2, 'jaccard_rank': 4},
    ])
    df3 = pd.DataFrame([
        {'target_bid': 12, 'rank_union': 0.5, 'rank_union_rank': 2,
         'jaccard': 0.5, 'jaccard_rank': 2},
        {'target_bid': 13, 'rank_union': 0.4, 'rank_union_rank': 3,
         'jaccard': 0.4, 'jaccard_rank': 3},
    ])
    per_source = {1: df1, 2: df2, 3: df3}
    val_rows = [{'source_bodyId': b, 'verdict': 'verified'}
                for b in (1, 2, 3)]
    assigned = mutual_best_assignment({11, 12, 13}, per_source, val_rows)
    # t11's best source is s2 (higher ru at the same rank) -> mutual (2, 11)
    # s3's best is t12 and t12's best source is s3 -> mutual (3, 12)
    # s1 is the contest loser and stays unassigned
    assert sorted(assigned) == [(2, 11), (3, 12)]


def test_assignment_skips_unmatched_verdicts():
    per_source = {
        1: pd.DataFrame([{'target_bid': 11, 'rank_union': 0.9,
                          'rank_union_rank': 1, 'jaccard': 0.9,
                          'jaccard_rank': 1}]),
        2: pd.DataFrame([{'target_bid': 11, 'rank_union': 0.5,
                          'rank_union_rank': 2, 'jaccard': 0.5,
                          'jaccard_rank': 2}]),
    }
    val_rows = [
        {'source_bodyId': 1, 'verdict': 'unmatched'},
        {'source_bodyId': 2, 'verdict': 'verified'},
    ]
    assigned = mutual_best_assignment({11}, per_source, val_rows)
    # s1 is unmatched (not assignable) so t11's mutual best is s2
    assert assigned == [(2, 11)]


def test_the_assignment_follows_the_chain_not_rank_union():
    """S4 (plan-tmvev-jaccard-primary-bodyid-ranking): who a target belongs
    to is the same key the reverse pass uses — Jaccard first, rank_union as
    the tie-break.  Here s1 leads on Jaccard and loses on rank_union; under
    the old rank_union-first key the pair would have been (2, 11) and s1
    would have stayed unassigned."""
    df1 = pd.DataFrame([
        {'target_bid': 11, 'rank_union': 0.40, 'rank_union_rank': 2,
         'jaccard': 0.90, 'jaccard_rank': 1},
        {'target_bid': 12, 'rank_union': 0.10, 'rank_union_rank': 3,
         'jaccard': 0.10, 'jaccard_rank': 3},
    ])
    df2 = pd.DataFrame([
        {'target_bid': 11, 'rank_union': 0.80, 'rank_union_rank': 1,
         'jaccard': 0.30, 'jaccard_rank': 2},
    ])
    val_rows = [{'source_bodyId': b, 'verdict': 'verified'}
                for b in (1, 2)]
    assert mutual_best_assignment({11, 12}, {1: df1, 2: df2},
                                  val_rows) == [(1, 11)]


def test_gap_rule_gt_one_always():
    # Revision 3.4: gap > 1 always (the '> 20% or > 5' rule is retired)
    assert gap_check(6, 7, 4, 1)['gap_triggered'] is True
    # gap exactly 1 -> NOT > 1 -> tolerated
    assert gap_check(3, 3, 2, 1)['gap_triggered'] is False
    assert gap_check(25, 25, 24, 1)['gap_triggered'] is False
    # gap 2 -> triggered
    assert gap_check(20, 20, 18, 1)['gap_triggered'] is True
    # gap 6 -> triggered
    assert gap_check(20, 20, 14, 1)['gap_triggered'] is True
    # empty pool side -> no gap semantics
    assert gap_check(0, 5, 0, 1)['gap_triggered'] is False


def test_summary_gap_min_rule():
    pair = TypePair('dsA', 'T', [1, 2, 3], 'dsB', 'T', [11, 12, 13, 14])
    val_rows = [
        {'verdict': 'verified_strong', 'source_bodyId': 1,
         'suspicious_count': 0},
        {'verdict': 'unmatched', 'source_bodyId': 2, 'suspicious_count': 0},
        {'verdict': 'verified', 'source_bodyId': 3, 'suspicious_count': 0},
    ]
    summary = MappingValidator._summary(pair, val_rows, [(1, 11), (3, 12)],
                                        gap_min=1)
    assert summary['source_pool'] == 3 and summary['target_pool'] == 4
    assert summary['matched'] == 2
    assert summary['gap'] == 1 and summary['gap_ratio'] == 0.3333
    # gap 1 is tolerated under 'gap > 1 always'
    assert summary['gap_triggered'] is False


def test_gap_fill_tags_out_of_pool():
    """Three strong-ish sources face a weak 3-neuron pool: every source
    ends unmatched (pool members rank deep), gap = 3 - 0 = 3 > 1 fires,
    and expansion proposes the strong out-of-pool neuron 999."""
    targets = {}
    for bid in (201, 202, 203):
        targets[bid] = ({'Z': 1}, {})
    # decoys share the key set but with P/Q swapped -> rank_union strictly
    # below the exact match
    for bid in (204, 205, 206):
        targets[bid] = ({'A': 10, 'B': 8, 'C': 4, 'D': 2},
                        {'P': 2, 'Q': 9})
    targets[999] = (dict(SRC_UP), dict(SRC_DN))

    src_profiles = {}
    for bid in (1, 2, 3):
        src_profiles[bid] = make_profile(bid, dict(SRC_UP), dict(SRC_DN))
    tgt_vectors = {bid: expanded_vector(make_profile(bid, up, dn), None)
                   for bid, (up, dn) in targets.items()}
    _, pair, res = run_validator(
        [1, 2, 3], [201, 202, 203], tgt_vectors, src_profiles)

    # pool members are weak: at most one borderline assignment, the
    # rest of the sources stay unpaired -> gap fires, out-of-pool fills
    assert len(res['pairs']) <= 1
    fills = res['fills']
    src_fills = [f for f in fills if f['side'] == 'source']
    assert src_fills, 'gap must fire while sources remain unpaired'
    assert all(f['fill_class'] == 'out_of_pool' for f in src_fills)
    # every source's best out-of-pool candidate is the exact-match 999
    assert all(f['proposal_bodyId'] == 999 for f in src_fills)
    tgt_fills = [f for f in fills if f['side'] == 'target']
    assert tgt_fills and all(f['fill_class'] == 'in_pool'
                             for f in tgt_fills)


# ---------------------------------------------------------------------------
# Revision 3.3 category ladder
# ---------------------------------------------------------------------------

def _scan_df(rows):
    """rows: (bid, ru_rank, ru, ja_rank, ja) -> scan-like DataFrame."""
    return pd.DataFrame(
        [{'target_bid': b, 'rank_union_rank': ru, 'rank_union': ruv,
          'jaccard_rank': ja, 'jaccard': jav}
         for b, ru, ruv, ja, jav in rows])


def test_ladder_matched_top_level():
    # 11 is the ru top-1 with ru 0.3 > 0.1 -> matched (top level)
    per_source = {1: _scan_df([(11, 1, 0.3, 1, 0.6),
                               (99, 2, 0.1, 2, 0.1)])}
    cats, _ = categorize_pool_targets(per_source, {11})
    assert cats[11] == 'matched'


def test_ladder_matched_reads_the_bar_off_the_chain_claimant():
    """Two sources claim target 11.  S4 makes the claimant the CHAIN best
    (Jaccard first), so the `matched` bar is read off THAT row — the
    rank_union-best source is no longer the one whose score decides the
    level, and `best_source_bodyId` says which one did."""
    per_source = {
        1: _scan_df([(11, 2, 0.05, 1, 0.80)]),   # chain best, ru below bar
        2: _scan_df([(11, 1, 0.60, 2, 0.20)]),   # ru best, not the claimant
    }
    cats, detail = categorize_pool_targets(per_source, {11})
    assert [d['best_source_bodyId'] for d in detail] == [1]
    assert cats[11] == 'verified'    # the claimant's 0.05 does not clear 0.1


def test_ladder_verified_top1_below_matched_threshold():
    # ru top-1 but 0.05 <= 0.1 -> verified, not matched
    per_source = {1: _scan_df([(11, 1, 0.05, 1, 0.6),
                               (99, 2, 0.01, 2, 0.1)])}
    cats, _ = categorize_pool_targets(per_source, {11})
    assert cats[11] == 'verified'


def test_ladder_verified_negative_top1_falls_to_jaccard():
    # ru top-1 with a negative score does not give matched; the Jaccard
    # top-1 still verifies (per the inspected fixture behavior)
    per_source = {1: _scan_df([(11, 1, -0.5, 1, 0.6),
                               (99, 2, -0.9, 2, 0.1)])}
    cats, _ = categorize_pool_targets(per_source, {11})
    assert cats[11] == 'verified'


def test_ladder_continuous_top_n_same_type():
    # ranks 1 AND 2 are both in-pool -> 12 (not a literal top-1) is
    # verified via the continuous ordered top-N path
    per_source = {1: _scan_df([(11, 1, 0.4, 1, 0.5),
                               (12, 2, 0.05, 2, 0.4),
                               (99, 3, 0.1, 3, 0.1)])}
    cats, _ = categorize_pool_targets(per_source, {11, 12})
    assert cats[11] == 'matched'
    # 12 is not a literal top-1; the continuous top-2 (both pool) path
    # verifies it, and its ru 0.05 <= 0.1 keeps it at verified
    assert cats[12] == 'verified'


def test_ladder_borderline_and_unmatched_by_invasion_count():
    # 13 has exactly 3 invaders ahead -> borderline
    per_source = {1: _scan_df([
        (91, 1, 0.2, 1, 0.1), (92, 2, 0.15, 2, 0.1),
        (93, 3, 0.1, 3, 0.1), (13, 4, 0.05, 4, 0.05)])}
    cats, _ = categorize_pool_targets(per_source, {13})
    assert cats[13] == 'borderline'
    # a 4th invader ahead -> unmatched (outside)
    per_source = {1: _scan_df([
        (91, 1, 0.2, 1, 0.1), (92, 2, 0.15, 2, 0.1),
        (93, 3, 0.1, 3, 0.1), (94, 4, 0.05, 4, 0.05),
        (13, 5, 0.01, 5, 0.01)])}
    cats, _ = categorize_pool_targets(per_source, {13})
    assert cats[13] == 'unmatched'


def test_ladder_precedence_matched_beats_invasion():
    # matched tier wins even with invaders ahead (precedence)
    per_source = {1: _scan_df([
        (91, 1, 0.2, 2, 0.1), (11, 2, 0.3, 1, 0.5),
        (92, 3, 0.1, 3, 0.1), (93, 4, 0.05, 4, 0.05),
        (94, 5, 0.01, 5, 0.01)])}
    cats, _ = categorize_pool_targets(per_source, {11})
    assert cats[11] == 'matched'  # top-1 jaccard AND ru 0.3 > 0.1


# ---------------------------------------------------------------------------
# bodyId integrity
# ---------------------------------------------------------------------------

def test_fafb_sized_ids_survive_scoring():
    src = make_profile(1, SRC_UP, SRC_DN)
    tgt_vectors = {
        FAFB_SIZED_ID: expanded_vector(make_profile(
            FAFB_SIZED_ID, dict(SRC_UP), dict(SRC_DN)), None)}
    _, _, res = run_validator([1], [FAFB_SIZED_ID], tgt_vectors, {1: src})
    row = res['rows'][0]
    assert row['verdict'] == 'verified_strong'
    assert row['target_bodyId'] == FAFB_SIZED_ID
    df = pd.DataFrame(res['rows'])
    assert int(df['target_bodyId'].iloc[0]) == FAFB_SIZED_ID


def test_csv_writer_keeps_int64(tmp_path):
    rows = [{'source_bodyId': 1, 'target_bodyId': FAFB_SIZED_ID,
             'verdict': 'verified_strong'}]
    out = tmp_path / 'v.csv'
    _write_csv(out, rows)
    back = pd.read_csv(out)
    assert int(back['target_bodyId'].iloc[0]) == FAFB_SIZED_ID


# ---------------------------------------------------------------------------
# Morphology calibration helpers
# ---------------------------------------------------------------------------

def test_morph_auc_perfect_and_none():
    verified = [0.9, 0.85, 0.8, 0.95, 0.88]
    suspicious = [0.1, 0.2, 0.15, 0.05, 0.3]
    auc = morph_auc(verified, suspicious)
    assert auc == pytest.approx(1.0)
    assert best_threshold(verified, suspicious) is not None
    assert morph_auc([0.9], suspicious) is None  # too few samples
    auc2 = morph_auc([0.5, 0.52, 0.48, 0.51, 0.49],
                     [0.5, 0.52, 0.48, 0.51, 0.49])
    assert 0.3 < auc2 < 0.7  # overlapping -> mid AUC


def test_build_target_vectors_missing_cache_raises(tmp_path):
    class FakeProfiler:
        cache_dir = tmp_path

        def _get_cache_parquet_path(self, ds):
            return tmp_path / 'nope.parquet'

    with pytest.raises(FileNotFoundError):
        build_target_vectors(FakeProfiler(), 'dsX', None, verbose=False)


# ---------------------------------------------------------------------------
# Revision 2: branch annotation helpers
# ---------------------------------------------------------------------------

def _branch_pair(target, pool, basis='linker rows'):
    return TypePair('dsA', 'T', list(pool), 'dsB', target, list(pool),
                    pool_basis=basis, source_type_total=12)


def test_branch_disjointness_and_annotation():
    # APDN3-like: 4 disjoint linker-refined branches over 12 neurons
    pairs = [_branch_pair('CL125', [1, 2, 3, 4]),
             _branch_pair('PLP080', [5, 6]),
             _branch_pair('SLP249', [7, 8, 9, 10]),
             _branch_pair('SLP250', [11, 12])]
    assert measure_branch_disjointness(pairs) is True
    annotate_pair_branches(pairs)
    assert all(p.branch_annotation for p in pairs)
    assert all('bifurcation' in p.branch_annotation for p in pairs)
    assert all('resolved_by_linkers' in p.branch_annotation
               for p in pairs)
    assert all('branches_disjoint' in p.branch_annotation for p in pairs)
    assert sorted(p.branch_index for p in pairs) == [1, 2, 3, 4]


def test_branch_overlap_flags_unresolved():
    pairs = [_branch_pair('A', [1, 2, 3]),
             _branch_pair('B', [3, 4, 5])]
    assert measure_branch_disjointness(pairs) is False
    annotate_pair_branches(pairs)
    assert all('unresolved_fanout' in p.branch_annotation
               for p in pairs)
    assert all('branches_overlap' in p.branch_annotation for p in pairs)


def test_full_population_branch_cannot_measure_disjointness():
    pairs = [_branch_pair('A', [1, 2, 3]),
             _branch_pair('B', [4, 5, 6], basis='full population')]
    assert measure_branch_disjointness(pairs) is None
    annotate_pair_branches(pairs)
    assert all('unresolved_fanout' in p.branch_annotation
               for p in pairs)


def test_single_branch_has_no_fanout_annotation():
    pairs = [_branch_pair('A', [1, 2, 3])]
    annotate_pair_branches(pairs)
    assert pairs[0].branch_annotation == ''
    assert pairs[0].branch_index == 1


def test_mapping_export_rows_carry_branch_context():
    v = MappingValidator.__new__(MappingValidator)
    v.pairs = [_branch_pair('CL125', [1, 2, 3, 4])]
    v.pairs[0].query = 'APDN3'
    v.pairs[0].status = 'evidence_only'
    v.pairs[0].relationship = 'N-to-1'
    v.pairs[0].selected_chain = [
        {'dataset': 'flywire_FAFB_v783', 'column': 'type',
         'value': 'APDN3'},
        {'dataset': 'flywire_FAFB_v783', 'column': 'additional_type(s)',
         'value': 'LMTe01'},
        {'dataset': 'male-cns:v1.0', 'column': 'flywireType',
         'value': 'CL125'},
    ]
    v.pairs[0].linkers = [
        {'column': 'additional_type(s)', 'raw_value': 'LMTe01',
         'canonical_value': 'LMTe01', 'home': 'flywire_FAFB_v783'},
        {'column': 'flywireType', 'raw_value': 'LMTe01',
         'canonical_value': 'LMTe01', 'home': 'male-cns:v1.0'},
    ]
    v.pairs[0].pool_basis = 'linker rows'
    v.pairs[0].parent_source_pool = [1, 2, 3, 4, 5, 6]
    annotate_pair_branches(v.pairs)
    rows = v._mapping_export_rows()
    row = rows[0]
    assert row['source_body_ids'] == '{1, 2, 3, 4}'
    assert row['parent_source_body_ids'] == '{1, 2, 3, 4, 5, 6}'
    assert row['is_selected'] is True
    assert row['pool_basis'] == 'linker rows'
    assert 'LMTe01' in row['selected_linker_values']
    assert row['mapping_status'] == 'evidence_only'


def test_null_sample_selection_deterministic():
    """Rev 3.9 null-sample selection (deterministic rework, user
    2026-09-17): the small sampled set for the arbitrary floor gate is a
    pure function of the scan rows — sorted by target bid before the
    per-source cap, invariant to row/insertion order, honoring pool /
    seen exclusions and the cap."""
    from comparison.mapping_validation import select_null_sample

    def mk(rows):
        return pd.DataFrame(rows, columns=['target_bid', 'jaccard'])

    s1 = mk([(30, 0.01), (10, 0.0), (20, 0.02), (40, 0.04)])
    s2 = mk([(25, 0.01), (15, 0.03)])
    # (sources without best evidence are filtered by the CALLER via
    # best_by_src before selection — mirrors _scan_pair)

    picked = select_null_sample(
        {11: s1, 12: s2}, {40}, set(), 0.05, 2)
    # pool bid 40 excluded; cap 2 per source; targets sorted per source
    assert picked == [(11, 10), (11, 20), (12, 15), (12, 25)]

    # same rows, different insertion orders -> identical selection
    shuffled = {11: s1.sample(frac=1.0, random_state=7),
                12: s2.sample(frac=1.0, random_state=7)}
    assert select_null_sample(
        shuffled, {40}, set(), 0.05, 2) == picked

    # targets consumed by an earlier bin (seen_null) are skipped and
    # consumed for later sources too
    picked2 = select_null_sample(
        {11: s1, 12: s2}, {40}, {10}, 0.05, 2)
    assert picked2 == [(11, 20), (11, 30), (12, 15), (12, 25)]

    # thin window: fewer eligible rows than the cap is fine
    s3 = mk([(50, 0.01)])
    assert select_null_sample({11: s3}, set(), set(), 0.05, 5) \
        == [(11, 50)]


def _ps_frame(rows):
    import pandas as pd
    return pd.DataFrame(rows, columns=['target_bid', 'rank_union',
                                       'rank_union_rank', 'jaccard',
                                       'jaccard_rank'])


def test_categorize_pool_sources_mirror_rules():
    """plan-backward-source-status.md: the column view grades sources by
    the same pair scores — column-top-1 → source-verified (source-matched
    when it is also the source's row-best with ru above the matched bar);
    out-of-pool competitors above → source-borderline/unmatched."""
    from comparison.mapping_validation import categorize_pool_sources

    target_pool = {101, 102}
    source_pool = {1, 2, 3, 4}
    per_source = {
        # source 1: row-best 101, column-top-1 of 101, ru above bar
        1: _ps_frame([(101, 0.40, 1, 0.40, 1),
                      (102, 0.10, 2, 0.10, 2),
                      (301, 0.05, 3, 0.05, 3)]),
        # source 2: column-top-1 of 102 (by ru) but its row-best is 102
        # with ru BELOW the bar -> source-verified
        2: _ps_frame([(102, 0.05, 1, 0.30, 1),
                      (101, 0.02, 2, 0.05, 2)]),
        # source 3: never column-top-1; one OUT-OF-BRANCH source (310)
        # ranks above it in its best column -> source-borderline
        3: _ps_frame([(101, 0.20, 3, 0.20, 3),
                      (302, 0.25, 1, 0.25, 1)]),
        310: _ps_frame([(101, 0.30, 1, 0.30, 1)]),
        # source 4: two out-of-pool competitors above in its best column
        4: _ps_frame([(101, 0.01, 3, 0.01, 3),
                      (302, 0.30, 1, 0.30, 1),
                      (303, 0.20, 2, 0.20, 2)]),
    }
    statuses, detail = categorize_pool_sources(
        per_source, target_pool, source_pool,
        top_n=2, invader_max=3, matched_ru_min=0.1)
    assert statuses[1] == 'source-matched'
    assert statuses[2] == 'source-verified'
    assert statuses[3] == 'source-borderline'
    # source 4 ranks below three IN-POOL sources in column 101 -> the
    # out-of-pool competitor count above it is 0 -> source-borderline
    assert statuses[4] == 'source-borderline'
    det = {d['source_bodyId']: d for d in detail}
    assert det[1]['best_column_target'] == 101 and det[1]['col_rank'] == 1
    # competitor above source 4: the out-of-branch source 310
    assert det[4]['n_competitors'] == 1


def test_categorize_pool_sources_topn_all_in_pool():
    """The forward top-N mirror: a column whose top-N sources are ALL
    in-pool admits those N sources as source-verified (the top-1 also
    upgrades to source-matched when it is its own row-best above the
    bar)."""
    from comparison.mapping_validation import categorize_pool_sources

    per_source = {
        1: _ps_frame([(101, 0.30, 1, 0.30, 1),
                      (102, 0.25, 2, 0.25, 2)]),
        2: _ps_frame([(101, 0.20, 1, 0.20, 1),
                      (102, 0.15, 2, 0.15, 2)]),
    }
    statuses, _ = categorize_pool_sources(
        per_source, {101, 102}, {1, 2}, top_n=2, invader_max=3,
        matched_ru_min=0.1)
    assert statuses == {1: 'source-matched', 2: 'source-verified'}


def test_categorize_pool_sources_in_branch_only():
    """D-B7: only sources in source_pool receive statuses. Source 1 is
    column runner-up behind the out-of-branch source 9 -> borderline;
    source 9 itself gets NO status."""
    from comparison.mapping_validation import categorize_pool_sources
    per_source = {
        1: _ps_frame([(101, 0.30, 1, 0.30, 1)]),
        9: _ps_frame([(101, 0.99, 1, 0.99, 1)]),   # out-of-branch source
    }
    statuses, detail = categorize_pool_sources(
        per_source, {101}, {1}, top_n=2, invader_max=3,
        matched_ru_min=0.1)
    assert statuses == {1: 'source-borderline'}
    assert all(d['source_bodyId'] == 1 for d in detail)


def test_categorize_pool_sources_unmatched_when_dominated():
    """More out-of-branch sources above than invader_max -> the source
    is dominated in every column it appears in -> source-unmatched."""
    from comparison.mapping_validation import categorize_pool_sources
    per_source = {
        1: _ps_frame([(101, 0.50, 1, 0.50, 1)]),
        2: _ps_frame([(101, 0.20, 3, 0.20, 3)]),
        3: _ps_frame([(101, 0.10, 5, 0.10, 5)]),
        4: _ps_frame([(101, 0.01, 8, 0.01, 8)]),   # dominated source
        # four out-of-branch sources rank between source 1 and 4
        30: _ps_frame([(101, 0.40, 2, 0.40, 2)]),
        31: _ps_frame([(101, 0.35, 3, 0.35, 3)]),
        32: _ps_frame([(101, 0.28, 4, 0.28, 4)]),
        33: _ps_frame([(101, 0.25, 5, 0.25, 5)]),
    }
    statuses, _ = categorize_pool_sources(
        per_source, {101}, {1, 2, 3, 4}, top_n=2, invader_max=3,
        matched_ru_min=0.1)
    assert statuses[4] == 'source-unmatched'
    assert statuses[1] == 'source-matched'


def test_a_run_that_scores_nothing_says_why():
    """r25's hemibrain run graded 36 branches on connectivity alone, and the
    only trace in its notes was "Track-A null sample too thin (n=0)".  A
    morphology pass that scored NONE of what it asked for must name the
    degradation: the null bar has no sample because no target skeleton was
    loadable, so the verdicts are not merely un-calibrated."""
    from comparison.mapping_validation import morph_coverage_warning

    # nothing requested -> nothing to explain
    assert morph_coverage_warning({}, 'hemibrain_v1_2_1') is None
    # partial coverage is ordinary (thin caches, missing neurons)
    assert morph_coverage_warning(
        {(1, 11): 0.4, (2, 12): None}, 'hemibrain_v1_2_1') is None
    warn = morph_coverage_warning(
        {(1, 11): None, (2, 12): None}, 'hemibrain_v1_2_1')
    assert warn is not None
    assert '0 of 2' in warn
    assert 'hemibrain_v1_2_1' in warn
    assert 'connectivity-only' in warn


def test_gap_fill_proposals_name_their_pair_by_side():
    """A proposal is spelled from the side it fills, so reading
    bodyId/proposal_bodyId in one fixed order asked Track A to render a
    target bodyId out of the SOURCE dataset: every target-side fill row
    scored nothing (0 of 76 in the male-cns family baselines, 0 of 2 in the
    hemibrain probe) while its source-side twin scored 87 of 88.  The pair
    frame and the score merge now read it through one helper, so they cannot
    drift apart."""
    from comparison.mapping_validation import (MappingValidationConfig,
                                               MappingValidator, fill_pair)

    assert fill_pair({'side': 'source', 'bodyId': 1,
                      'proposal_bodyId': 11}) == (1, 11)
    assert fill_pair({'side': 'target', 'bodyId': 12,
                      'proposal_bodyId': 2}) == (2, 12)

    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='dsA', target_dataset='dsB',
                                    query_types=['T'], visualize=False)
    frame = v._morph_pair_frame([], [], [
        {'side': 'source', 'bodyId': 1, 'proposal_bodyId': 11},
        {'side': 'target', 'bodyId': 12, 'proposal_bodyId': 2}])
    got = sorted(zip(frame['source_bodyId'], frame['target_bodyId']))
    assert got == [(1, 11), (2, 12)], \
        'a target-side proposal must score the real pair, not a swapped one'
    assert set(frame['pair_kind']) == {'fill'}


def test_skeleton_preflight_is_cache_first_bounded_and_fails_open(monkeypatch,
                                                                 tmp_path):
    """Stage 5 must fetch the target skeletons IT scores against (plan
    §17.2): its target reader is cache-only while the stage-4 scenes — which
    run later — are what fetch. Without the pre-flight a dataset's first run
    scores 0 pairs and its second run gets different bars.
    """
    import types
    from comparison import mapping_validation as mv

    fetched = []

    class _Cache:
        def find_skeleton_file(self, bid):
            bid = int(bid)
            if bid in (11, 12, 13) or bid >= 100:
                return None
            return tmp_path / f'{bid}.swc.zst'

    def _fetch(dataset, bid):
        fetched.append(int(bid))
        return None if int(bid) == 13 else object()

    fake = types.ModuleType('morphology')
    fake.find_similar_raw_cache = lambda dataset, **kw: _Cache()
    fake.fetch_skeleton_on_demand = _fetch
    monkeypatch.setitem(sys.modules, 'morphology', fake)

    v = mv.MappingValidator.__new__(mv.MappingValidator)
    v.cfg = mv.MappingValidationConfig(source_dataset='dsA', target_dataset='dsB',
                                       query_types=['T'], visualize=False,
                                       morph_enabled=True)
    v.notes = []
    v.progress = types.SimpleNamespace(emit=lambda *a, **k: None)

    stats = v._preflight_target_skeletons('dsB', [11, 12, 13, 14])
    assert stats['requested'] == 4 and stats['cached'] == 1
    assert stats['fetched'] == 2 and stats['failed'] == 1
    assert sorted(fetched) == [11, 12, 13]

    # FAFB targets resolve through the healed-zip path, never this cache
    assert v._preflight_target_skeletons('flywire_FAFB_v783', [11])['fetched'] == 0
    assert sorted(fetched) == [11, 12, 13]

    # a bounded fetch says what it left unscored instead of scoring less
    monkeypatch.setattr(mv, 'MORPH_SKELETON_PREFLIGHT_CAP', 2)
    st = v._preflight_target_skeletons('dsB', list(range(100, 110)))
    assert st['skipped_cap'] == 8 and st['fetched'] == 2
    assert any('capped at 2' in n for n in v.notes)

    # fail-open: an unreadable cache must not kill the morphology stage
    def _boom(dataset, **kw):
        raise RuntimeError('no cache namespace')
    fake.find_similar_raw_cache = _boom
    v.notes = []
    assert v._preflight_target_skeletons('dsB', [11])['fetched'] == 0
    assert any('skeleton pre-flight unavailable' in n for n in v.notes)
