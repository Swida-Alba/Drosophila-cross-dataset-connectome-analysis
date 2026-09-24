"""Scan-kernel parity: the indexed whole-universe scan vs the reference loop.

``plan-tmvev-run-acceleration`` P1/P2.  ``prep_target_stats`` now returns a
:class:`~comparison.body_id_resolver.TargetScanIndex`, which reaches the
Jaccard-positive block through a per-type postings list and fills the other
99.3 % of the frame arithmetically (``rank_union`` for a disjoint pair has a
closed form — :func:`_disjoint_rank_union`).

The contract is NOT "close enough".  The frame must be the same frame: same
rows in the same order, same columns, same values, so every consumer of a
rank window (``metric == 1``, ``nsmallest(top_n)``, ``invaders_ahead``), the
null sampler and ``chain_pos`` reads exactly as it did before the
acceleration.  These tests pin that contract at three levels:

1. the frame itself, over randomized vectors that include every corner case
   the closed form has to survive;
2. the consumers, on the same universe through both paths;
3. the cost model, by counting how many pairs reach the per-pair scorer —
   the guard that the speedup is real rather than a different formula.

The end-to-end proof is a run-level one: an optimized run must be
verdict-identical to the pre-acceleration baselines r27b/r28/r29
(``_plan/probes/r17_delta.py``), which compares 242 real sources against
~100k real targets — 24.6M pairs — rather than a sample.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from comparison.body_id_resolver import (  # noqa: E402
    TargetScanIndex, _SideStats, _disjoint_rank_union, _ranks_squared_sum,
    prep_target_stats, scan_source, score_one_candidate_fast)


def reference_scan(src_vec, stats_by_bid, bids):
    """The pre-P2 scan: one :func:`score_one_candidate_fast` per pair."""
    src = _SideStats(src_vec)
    rows = []
    for tbid in bids:
        tgt = stats_by_bid.get(tbid)
        if tgt is None or not tgt.vec:
            continue
        m = score_one_candidate_fast(src, tgt)
        m['target_bid'] = tbid
        rows.append(m)
    from comparison.body_id_resolver import _finish_frame
    return _finish_frame(pd.DataFrame(rows))


def assert_same_frame(a, b, label=''):
    assert list(a.columns) == list(b.columns), \
        f'{label} column order: {list(a.columns)} vs {list(b.columns)}'
    assert len(a) == len(b), f'{label} row count {len(a)} vs {len(b)}'
    if len(a) == 0:
        return
    assert list(a['target_bid']) == list(b['target_bid']), \
        f'{label} row ORDER differs (the chain sort must be identical)'
    for col in a.columns:
        if col == 'target_bid':
            continue
        x = a[col].to_numpy(dtype=float)
        y = b[col].to_numpy(dtype=float)
        same = np.array_equal(x, y, equal_nan=True)
        if not same:
            d = np.nonzero(~((x == y)
                             | (np.isnan(x) & np.isnan(y))))[0][:3]
            pytest.fail(f'{label} column {col!r}: {len(d)}+ rows differ, '
                        f'first {x[d].tolist()} vs {y[d].tolist()}')


VOCAB = [f't{i}' for i in range(60)]


def _vec(rng, n_keys, weights=None, extra_zero=False):
    keys = rng.choice(len(VOCAB),
                      size=min(n_keys, len(VOCAB)), replace=False)
    vals = (weights if weights is not None else
            rng.choice([1.0, 1.0, 2.0, 3.5, 7.0, 0.25], size=len(keys)))
    out = {VOCAB[i]: float(v) for i, v in zip(keys, np.asarray(vals)[:len(keys)])}
    if extra_zero:
        out['zero_key'] = 0.0
    return out


def test_the_indexed_frame_is_identical_to_the_reference_frame():
    """Randomized universes × sources: the fast path must not change a value.

    Covers the corners the closed form breaks on if it is wrong — weight
    ties, 1-key vectors, the ``|U| < 3`` NaN rule, everything overlapping,
    nothing overlapping, and a zero weight (which must fall back, not
    mis-score).
    """
    rng = np.random.default_rng(20260922)
    for trial in range(30):
        vectors = {1000 + b: _vec(rng, int(rng.integers(1, 9)))
                   for b in range(int(rng.integers(3, 300)))}
        index = prep_target_stats(vectors)
        plain = {b: _SideStats(v) for b, v in vectors.items() if v}
        bids = sorted(vectors)
        for _ in range(4):
            src = _vec(rng, int(rng.integers(1, 10)),
                       extra_zero=(trial % 7 == 3))
            assert_same_frame(
                scan_source(src, index, bids),
                reference_scan(src, plain, bids),
                label=f'trial {trial}:')
            # the default (whole-index) call shape must agree too
            assert_same_frame(
                scan_source(src, index),
                reference_scan(src, plain, bids),
                label=f'trial {trial} (no bid list):')


def test_prep_target_stats_indexes_but_still_maps_bid_to_stats():
    """Every existing consumer reads it as a ``bid -> _SideStats`` dict."""
    vectors = {7: {'a': 1.0, 'b': 2.0}, 8: {'c': 3.0}, 9: {}}
    stats = prep_target_stats(vectors)
    assert isinstance(stats, TargetScanIndex)
    assert list(stats) == [7, 8]                   # empty vectors dropped
    assert stats[7].sum1 == 3.0 and stats[8].norm == 3.0
    assert {b: s.sum1 for b, s in stats.items()} == {7: 3.0, 8: 3.0}
    assert len(stats) == 2


def test_the_disjoint_closed_form_matches_the_scorer_pair_by_pair():
    """``_disjoint_rank_union`` is an identity, not a heuristic.

    Sweeps small key-count pairs exhaustively (including every tie pattern
    reachable there) against the reference scorer on disjoint vectors.
    """
    rng = np.random.default_rng(5)
    for p in range(1, 6):
        for q in range(1, 6):
            for _ in range(40):
                sv = _vec(rng, p, weights=np.round(
                    rng.random(p) * 3 + 1, 2))
                tv = _vec(rng, q, weights=np.round(
                    rng.random(q) * 3 + 1, 2))
                if set(sv) & set(tv):
                    continue
                s, t = _SideStats(sv), _SideStats(tv)
                want = score_one_candidate_fast(s, t)['rank_union']
                got = _disjoint_rank_union(
                    len(s.keys), _ranks_squared_sum(
                        np.array(list(sv.values()))),
                    np.array([len(t.keys)], dtype=np.int64),
                    np.array([_ranks_squared_sum(
                        np.array(list(tv.values())))]))[0]
                if np.isnan(want):
                    assert np.isnan(got), (p, q, sv, tv)
                else:
                    assert got == pytest.approx(want, rel=1e-12), (p, q)


def test_the_index_scores_only_the_positive_block_with_the_pair_scorer():
    """The speedup guard: the per-pair scorer must see the positives (and the
    few non-analytic rows), never the whole universe.

    A regression that quietly re-enumerates every target would still be
    correct — and still cost 3.6 s per source neuron, which is the number
    this module exists to remove.
    """
    import comparison.body_id_resolver as b

    rng = np.random.default_rng(3)
    vectors = {}
    for i in range(2000):
        keys = rng.choice(500, size=int(rng.integers(1, 8)), replace=False)
        vectors[1000 + i] = {f't{k}': float(k + 1) for k in keys}
    index = prep_target_stats(vectors)
    src = {f't{k}': 2.0 for k in (0, 1, 2)}
    calls = []
    real = b.score_one_candidate_fast

    def spy(s, t):
        calls.append((len(s.keys), len(t.keys)))
        return real(s, t)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(b, 'score_one_candidate_fast', spy)
    try:
        df = scan_source(src, index)
    finally:
        monkey.undo()
    positives = int((df['jaccard'] > 0).sum())
    assert 0 < positives < len(df) / 10, \
        f'fixture should be mostly-disjoint, got {positives}/{len(df)}'
    assert len(calls) == positives, \
        f'{len(calls)} pairs reached the scorer, {positives} were positive'


def test_the_rank_windows_and_the_null_sample_read_the_same_frame():
    """Consumer level: the two things a shrunken frame would have broken.

    ``select_null_sample`` deliberately samples the ``jaccard == 0`` block
    (the null calibration bar), and the tier ladder reads a global
    ``rank == 1`` / top-N window — both are only safe because the frame
    still carries every zero row with its original rank.
    """
    from comparison.mapping_validation import (
        categorize_pool_targets, select_null_sample)

    rng = np.random.default_rng(11)
    vectors = {1000 + b: _vec(rng, int(rng.integers(2, 8)))
               for b in range(400)}
    index = prep_target_stats(vectors)
    plain = {b: _SideStats(v) for b, v in vectors.items()}
    src = _vec(rng, 4)
    fa = scan_source(src, index)
    fb = scan_source(src, dict(plain))
    assert_same_frame(fa, fb, label='consumer frame:')

    pool = {int(x) for x in fa['target_bid'].tolist()[:5]}
    picked_a = select_null_sample({9: fa}, pool, set(), 0.05, 5)
    picked_b = select_null_sample({9: fb}, pool, set(), 0.05, 5)
    assert picked_a and picked_a == picked_b
    cats_a, detail_a = categorize_pool_targets({9: fa}, pool)
    cats_b, detail_b = categorize_pool_targets({9: fb}, pool)
    assert cats_a == cats_b
    assert [d['invaders_ahead'] for d in detail_a] == \
        [d['invaders_ahead'] for d in detail_b]


def test_a_plain_dict_still_uses_the_reference_loop():
    """Pool-scoped callers and the test fixtures hand in plain dicts; the
    fast path is a property of the object, not of the call site."""
    vectors = {1: {'a': 1.0, 'b': 5.0}, 2: {'c': 2.0}, 3: {'a': 3.0}}
    plain = {b: _SideStats(v) for b, v in vectors.items()}
    src = {'a': 1.0, 'z': 9.0}
    df = scan_source(src, plain)
    assert not isinstance(plain, TargetScanIndex)
    # Jaccard 0.5 (bid 3) > 1/3 (bid 1) > 0 (bid 2), then bodyId
    assert list(df['target_bid']) == [3, 1, 2]
    assert_same_frame(df, reference_scan(src, plain, [1, 2, 3]),
                      label='plain dict:')
    assert_same_frame(scan_source(src, prep_target_stats(vectors), [1, 2, 3]),
                      df, label='indexed vs plain:')
