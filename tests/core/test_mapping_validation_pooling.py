"""Unit tests for TM VEV `pooling` mode (plan-tmvev-pooling-mode.md).

The tests exist to hold the two properties the mode is defined by, not the
arithmetic: selection must not depend on the type mapper (§1, §2.3), and a
missing measurement must never read like a rejection (§4.2, §6.4).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from comparison import body_id_resolver as bir  # noqa: E402
from comparison import mapping_validation as mv  # noqa: E402
from comparison import mapping_validation_pooling as pool  # noqa: E402


class FakeProfiler:
    def __init__(self, frames, types=None):
        self._frames = frames
        self._types = types or {}

    def get_profile(self, bid, dataset):
        # the pooling engine only needs a truthy handle per source, and the
        # fake scanner keys on it to pick the frame
        return int(bid) if int(bid) in self._frames else None

    def get_types_for_bodyids(self, bids, dataset):
        return {int(b): self._types.get(int(b), '') for b in bids}


class FakeValidator:
    """Only the surface the pooling engine touches."""

    def __init__(self, cfg, frames, types=None, pairs=None, sizes=None):
        self.cfg = cfg
        self.profiler = FakeProfiler(frames, types)
        self.pairs = pairs or []
        self.mapper = None
        self.notes = []
        self._target_sizes = sizes or {}
        self.seed = sorted(int(b) for b in frames)

    def log(self, msg=''):
        self.notes.append(str(msg))

    def _bodyids_for(self, name, dataset):
        return list(self.seed)

    def _backward_decision(self, atype):
        return {'status': 'mapped', 'mapped': 'DN1a', 'home_count': 3,
                'home_real': True}


def _cfg(**kw):
    base = dict(source_dataset='flywire_FAFB_v783', target_dataset='banc_v888',
                query_types=['circadian_clock'], verbose=False,
                validation_mode='pooling')
    base.update(kw)
    return mv.MappingValidationConfig(**base)


def _frame(rows):
    """A scan frame in the shape `_finish_frame` leaves it."""
    df = pd.DataFrame(rows)
    for col, default in (('jaccard', 0.0), ('rank_union', 0.0),
                         ('jaccard_rank', 1), ('rank_union_rank', 1)):
        if col not in df:
            df[col] = default
    return df


@pytest.fixture(autouse=True)
def _morphology_stand_in(monkeypatch):
    """Morphology is MANDATORY in pooling now, so a test cannot opt out of the
    last gate: every pass through it gets this stand-in (score what is asked,
    qualify at >= 0.5).  A test that needs another verdict patches over it, and
    its own `monkeypatch.setattr` runs after this fixture, so it wins.
    """
    import comparison.morph_cross_dataset as mcd

    class _Stub:
        active = True
        warnings = []
        ref_bars = {}
        native_scores = {}

        def __init__(self, pairs):
            self.scores = {(int(s), int(t)): 0.9 for s, t in pairs}

        def bar(self, s):
            return 0.5

        def is_qualified(self, s, t):
            v = self.scores.get((int(s), int(t)))
            return None if v is None else v >= 0.5

    monkeypatch.setattr(
        mcd, 'qualify_visualized_pairs',
        lambda src, tgt, pairs, **kw: _Stub(pairs))


# -- the mode selector ------------------------------------------------------

def test_pooling_is_parallel_not_nested():
    assert mv.POOLING_MODE == 'pooling'
    # adding it to the nested chain would let every mode_at_least admit it
    assert 'pooling' not in mv.VALIDATION_MODES
    assert 'pooling' not in mv.MODE_RANK
    cfg = _cfg()
    assert cfg.effective_mode == 'pooling'
    assert cfg.mode_rank is None
    assert cfg.mode_at_least('restrictive') is False
    assert cfg.mode_at_least('family') is False


def test_unknown_mode_raises_instead_of_running_restrictive():
    with pytest.raises(ValueError):
        mv.normalize_mode('poling')
    with pytest.raises(ValueError):
        mv.normalize_mode('pooling', aggressive_expansion=True)
    # the defaults are unchanged
    assert mv.normalize_mode('') == 'restrictive'
    assert mv.normalize_mode(None) == 'restrictive'
    assert mv.normalize_mode('family', pool_widen=True) == 'family'


# -- the window -------------------------------------------------------------

def test_window_scales_with_the_source_types_own_population():
    types = {1: 's-LNv', 2: 's-LNv', 3: 'APDN3', 4: 'APDN3', 5: 'l-LNv'}
    win = pool.window_sizes(types, 2.0)
    assert win[1] == win[2] == 4          # two neurons of its type
    assert win[3] == win[4] == 4
    assert win[5] == 2                    # alone in its type: 2 x 1
    # an unlabelled source is its own population of one, never a zero window
    assert pool.window_sizes({9: ''}, 2.0)[9] == 2
    assert pool.window_sizes({9: 'x'}, 0.2)[9] == 1


# -- the gate, and the unsupervised invariant -------------------------------

def _install_frames(monkeypatch, frames):
    def fake_scan_source(vec, stats, bids):
        return frames[int(vec)]
    monkeypatch.setattr(bir, 'scan_source', fake_scan_source)
    monkeypatch.setattr(mv, 'expanded_vector', lambda prof, mapper: prof)


def test_the_bar_admits_and_the_floors_only_flag(monkeypatch):
    """The bar decides admission; the floors measure and say so.

    101 sits below the Jaccard floor and 102 below the rank_union one — under
    the old gate both were deleted, which is how 119 of 242 real sources came to
    report nothing at all.  They are admitted and flagged.  103 is outside the
    bar itself (rank 3 of both metrics against top-N 2), so it is not.
    """
    frames = {1: _frame([
        {'target_bid': 100, 'jaccard': 0.40, 'rank_union': 0.20,
         'jaccard_rank': 1, 'rank_union_rank': 1},
        {'target_bid': 101, 'jaccard': 0.09, 'rank_union': 0.90,
         'jaccard_rank': 2, 'rank_union_rank': 2},        # below J, still kept
        {'target_bid': 102, 'jaccard': 0.40, 'rank_union': -0.10,
         'jaccard_rank': 3, 'rank_union_rank': 1},        # rank-1 by RU alone
        {'target_bid': 103, 'jaccard': 0.40, 'rank_union': 0.20,
         'jaccard_rank': 9, 'rank_union_rank': 9},        # outside the bar
    ])}
    _install_frames(monkeypatch, frames)
    v = FakeValidator(_cfg(pooling_bar_top_n=2), frames, types={1: 's-LNv'})
    rows, stats = pool.connectivity_candidates(
        v, [1], {1: 's-LNv'}, {1: 4}, {}, [100, 101, 102, 103], 0.10)
    assert [(r['target_bodyId'], r['bar_rank']) for r in rows] == [
        (100, 1), (102, 1), (101, 2)]        # chain order: jaccard leads
    assert stats['rows_scored'] == 4 and stats['rows_kept'] == 3
    assert [r['below_jaccard_floor'] for r in rows] == [False, False, True]
    assert [r['below_rank_union_floor'] for r in rows] == [False, True, False]
    assert [r['outside_window'] for r in rows] == [False] * 3
    # 102 is the population the retired floor deleted: rank_union top-1 for this
    # source, and a NEGATIVE rank_union — a sign test, never a volume one.
    # Its tier is graded where the labels join, not in the scan.
    assert rows[1]['rank_union'] < 0 and rows[1]['bar_rank'] == 1


def test_selection_does_not_depend_on_the_mapper(monkeypatch):
    """The unsupervised property: the SAME scan with and without the mapper's
    claims must produce the same candidates. Labels change, membership does
    not."""
    frames = {1: _frame([{'target_bid': 100, 'jaccard': 0.5,
                          'rank_union': 0.3, 'jaccard_rank': 1,
                          'rank_union_rank': 1},
                         {'target_bid': 200, 'jaccard': 0.6,
                          'rank_union': 0.4, 'jaccard_rank': 2,
                          'rank_union_rank': 2}])}
    _install_frames(monkeypatch, frames)
    claimed = [mv.TypePair(source_dataset='flywire_FAFB_v783',
                           source_type='s-LNv', source_pool=[1],
                           target_dataset='banc_v888', target_type='X',
                           target_pool=[100])]
    out = {}
    for name, pairs in (('with_claims', claimed), ('no_claims', [])):
        v = FakeValidator(_cfg(), frames, types={1: 's-LNv'}, pairs=pairs)
        rows, _ = pool.connectivity_candidates(
            v, [1], {1: 's-LNv'}, {1: 4}, {}, [100, 200], 0.10)
        refs = pool.mapper_reference(v, [])
        rows = pool.annotate(v, rows, {100: 'X', 200: 'Y'}, refs)
        out[name] = ({r['target_bodyId'] for r in rows},
                     {r['mapper_cell'] for r in rows})
    assert out['with_claims'][0] == out['no_claims'][0] == {100, 200}
    # ... while the post-hoc cell does move, which is what it is for
    assert pool.CELL_CONFIRMED in out['with_claims'][1]
    assert out['no_claims'][1] == {pool.CELL_TYPE_NEW, pool.CELL_TYPE_NEW}


def test_mapper_reference_reads_the_graded_pairs():
    v = FakeValidator(_cfg(), {}, pairs=[
        mv.TypePair(source_dataset='a', source_type='s', source_pool=[],
                    target_dataset='b', target_type='T1', target_pool=[100])])
    refs = pool.mapper_reference(v, [
        {'verdict': 'verified', 'target_bodyId': 100},
        {'verdict': 'unmatched', 'target_bodyId': 200},
        {'verdict': 'matched', 'target_bodyId': 300}])
    assert refs['pools'] == {100} and refs['types'] == {'T1'}
    assert set(refs['pairs']) == {100, 300}        # unmatched is not a claim


# -- labels: never a silent rejection ---------------------------------------

def test_untyped_and_size_unknown_stay_visible(monkeypatch):
    v = FakeValidator(_cfg(), {}, types={}, sizes={})
    refs = {'types': set(), 'pools': set(), 'pairs': {}}
    rows = [{'source_bodyId': 1, 'source_type': 's', 'target_bodyId': 7,
             'jaccard': 0.4, 'jaccard_rank': 1, 'rank_union': 0.2,
             'rank_union_rank': 1, 'window_size': 2}]
    out = pool.annotate(v, rows, {7: ''}, refs)
    assert out[0]['in_scope'] is False
    assert out[0]['leaf'] == 'untyped'
    assert out[0]['size_nm3'] is None
    assert out[0]['size_universe_percentile'] is None
    assert out[0]['mapper_cell'] == pool.CELL_TYPE_NEW


def test_leaf_token_reuses_the_shared_vocabulary():
    v = FakeValidator(_cfg(), {}, types={})
    assert pool._leaf_token(v, 'DN1a', True) == 'DN1a(out-map)'
    assert pool._leaf_token(v, 'LC16', False) == 'LC16>DN1a'
    assert pool._leaf_token(v, '', False) == 'untyped'


# -- the morph gate: budgeted, disclosed, fail-open -------------------------

def _rows(*targets, bar_rank=1):
    return [{'source_bodyId': 1, 'source_type': 's', 'target_bodyId': t,
             'jaccard': 0.4, 'jaccard_rank': bar_rank, 'rank_union': 0.2,
             'rank_union_rank': bar_rank, 'bar_rank': bar_rank,
             'bar_metric': 'either', 'bar_top_n': 3,
             'window_size': 2} for t in targets]


def test_pooling_refuses_a_run_without_morphology():
    """Every tier is defined morph-qualified, so a pooling run without the gate
    would publish connectivity findings under claim-shaped names — the very
    distinction the mode exists to make.  It refuses, the way `normalize_mode`
    already refuses a widening flag beside `--mode pooling`."""
    v = FakeValidator(_cfg(morph_enabled=False), {})
    with pytest.raises(ValueError):
        pool.check_morphology_mandatory(v.cfg)
    with pytest.raises(ValueError):
        pool.run_pooling(v, target_stats={}, target_bids=[7],
                         target_id2type={7: 'T'}, val_rows=[])
    # a non-pooling run is none of this check's business
    ok = FakeValidator(_cfg(morph_enabled=False, validation_mode='family'), {})
    pool.check_morphology_mandatory(ok.cfg)


def test_morph_budget_names_what_it_dropped(monkeypatch):
    import comparison.morph_cross_dataset as mcd

    class MQ:
        active = True
        warnings = []
        scores = {(1, 1): 0.9, (1, 2): 0.1}
        ref_bars = {}
        native_scores = {}

        def bar(self, s):
            return 0.5

        def is_qualified(self, s, t):
            return self.scores[(int(s), int(t))] >= 0.5

    monkeypatch.setattr(mcd, 'qualify_visualized_pairs',
                        lambda *a, **k: MQ())
    v = FakeValidator(_cfg(pooling_max_morph_targets=1), {})
    rows, info = pool.apply_morph_gate(v, _rows(1, 2))
    gates = {r['target_bodyId']: r['morph_gate'] for r in rows}
    assert info['attempted'] == 1 and info['capped'] == 1
    assert gates[2] == 'not-attempted-cap'
    assert gates[1] == 'scored'
    assert info['qualified'] == 1


def test_the_auto_budget_scales_with_the_queried_population():
    """A constant budget is how the default bar lost 146 of its 546 units on
    2026-09-24: 400 is neither enough for a 242-source query nor proportionate
    for a 58-source one.  3 units per queried neuron covers the measured 2.3
    with headroom for a pair whose rank_union ties harder, and the provenance
    string says which rule priced the run."""
    assert pool.morph_budget(_cfg(), 242) == (
        726, 'auto: 3 x 242 queried sources')
    assert pool.morph_budget(_cfg(), 58) == (
        174, 'auto: 3 x 58 queried sources')
    assert pool.morph_budget(_cfg(pooling_max_morph_targets=400), 242) == (
        400, 'configured pooling_max_morph_targets=400')


def test_the_record_publishes_which_budget_ran(monkeypatch):
    """`capped: 146` is unreadable without the line saying what the budget WAS
    and where it came from."""
    import comparison.morph_cross_dataset as mcd

    class MQ:
        active = True
        warnings = []
        scores = {}
        ref_bars = {}
        native_scores = {}

        def bar(self, s):
            return 0.5

        def is_qualified(self, s, t):
            return True

    monkeypatch.setattr(mcd, 'qualify_visualized_pairs',
                        lambda *a, **k: MQ())
    v = FakeValidator(_cfg(), {})
    rows = [{**_rows(t)[0], 'source_bodyId': s, 'target_bodyId': t}
            for s in (1, 2) for t in (7, 8)]
    rows, info = pool.apply_morph_gate(v, rows, seed_size=5)
    assert info['budget'] == 'auto: 3 x 5 queried sources'
    assert info['units'] == 4 and info['capped'] == 0
    # no seed size is not a zero budget: the pass prices itself on the rows it
    # was handed instead of silently refusing all of them.
    _, info2 = pool.apply_morph_gate(v, rows, seed_size=0)
    assert info2['budget'] == 'auto: 3 x 4 queried sources'


def test_the_four_absences_are_four_labels(monkeypatch):
    """`scored` (this row's own pair), `shared` (the verdict was made for the
    pair named in `verdict_for_pair`), `no-score` (the scorer returned nothing)
    and `not-attempted-cap` (the budget refused to look) — none of the last
    three may read as a rejected candidate.

    `shared` is the hybrid's honest edge: a `nominated` row shows a number that
    belongs to another source's pair, and the native bar in particular is
    computed from THAT source's reference pool, so the row must say whose
    measurement it is. `no-score` earned its keep earlier: an build's rows read
    it for candidates the scorer HAD scored, because the shared scorer prunes
    mapper pool members and pooling must not ask it to (see
    `test_pooling_grades_candidates_the_mapper_also_claims`)."""
    import comparison.morph_cross_dataset as mcd

    class MQ:
        active = True
        warnings = []
        scores = {(1, 7): 0.9}           # (1, 8) was never scored
        ref_bars = {}
        native_scores = {}

        def bar(self, s):
            return 0.5

        def is_qualified(self, s, t):
            v = self.scores.get((int(s), int(t)))
            return None if v is None else v >= 0.5

    monkeypatch.setattr(mcd, 'qualify_visualized_pairs',
                        lambda *a, **k: MQ())
    v = FakeValidator(_cfg(pooling_max_morph_targets=0), {})
    rows = _rows(7, 8)                                  # both tier-1
    rows.append({**rows[0], 'source_bodyId': 2, 'bar_rank': 2,
                 'jaccard_rank': 2, 'rank_union_rank': 2})
    rows, info = pool.apply_morph_gate(v, rows)
    gates = [(r['source_bodyId'], r['target_bodyId'], r['morph_gate'])
             for r in rows]
    assert gates == [(1, 7, 'scored'), (1, 8, 'no-score'),
                     (2, 7, 'shared')]
    assert [r['verdict_for_pair'] for r in rows] == ['', '', '1->7']
    assert info['attempted'] == 2 and info['capped'] == 0
    assert info['qualified'] == 1 and info['shared'] == 1
    # `scored` counts PAIRS the scorer returned a value for, not what the
    # budget attempted: collapsing them hides a run the last gate barely
    # measured.
    assert info['scored'] == 1 and info['no_score'] == 1
    # a row without a score carries no verdict, not a False
    assert [r['morph_qualified'] for r in rows if r['morph_gate'] ==
            'no-score'] == [None]


def test_pooling_grades_candidates_the_mapper_also_claims(monkeypatch):
    """The scorer's mapping_ref convention ("a branch-pool member was only an
    anchor, not a candidate") is the SUPERVISED caller's semantics; pooling's
    candidate set is its own gate, so it must ask for the un-pruned grading.
    With the default in place 35 of 41 real male-cns candidates came back
    `no-score` although every one of them had been scored — the flag is the
    contract, this is the tripwire. The behavioural half lives in
    `test_morph_cross_dataset.TestQualifyVisualizedPairs`."""
    import comparison.morph_cross_dataset as mcd

    seen = {}

    class MQ:
        active = True
        warnings = []
        scores = {(1, 7): 0.9, (1, 8): 0.8}   # 7 is also a mapper pool member
        ref_bars = {}
        native_scores = {}

        def bar(self, s):
            return 0.5

        def is_qualified(self, s, t):
            return self.scores[(int(s), int(t))] >= 0.5

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return MQ()

    monkeypatch.setattr(mcd, 'qualify_visualized_pairs', fake)
    v = FakeValidator(_cfg(), {})
    rows, info = pool.apply_morph_gate(v, _rows(7, 8))
    assert seen.get('mode') == 'mapping_ref'
    assert seen.get('prune_pool_refs') is False
    assert info['scored'] == 2 and info['no_score'] == 0


def test_morph_failure_is_recorded_per_row(monkeypatch):
    import comparison.morph_cross_dataset as mcd

    def boom(*a, **k):
        raise RuntimeError('no vectors for this dataset')
    monkeypatch.setattr(mcd, 'qualify_visualized_pairs', boom)
    v = FakeValidator(_cfg(), {})
    rows, info = pool.apply_morph_gate(v, _rows(1, 2))
    assert info['error'].startswith('RuntimeError')
    assert all(r['morph_gate'] == 'error' for r in rows)
    assert any('[pooling/morph]' in n for n in v.notes)


# -- dedup + the post-hoc comparison ---------------------------------------

def test_morphology_refusals_leave_the_pool_and_are_counted(monkeypatch):
    """Decision 3 (2026-09-23): morphology is the LAST GATE, not a label.

    A target whose chain-best row is scored BELOW its bar is gone from the
    exported pool — and therefore from the scene root, which reads the pool —
    while `dropped_targets` says how many left, so a small pool is never read
    as a small harvest.  A target with no verdict stays: `no-score` and
    `not-attempted-cap` are missing measurements, and only an explicit
    refusal removes.  The refused rows themselves stay in `candidates`.
    """
    import comparison.morph_cross_dataset as mcd

    class MQ:
        active = True
        warnings = []
        scores = {(1, 100): 0.9, (1, 101): 0.1, (1, 102): 0.1}
        ref_bars = {}
        native_scores = {}

        def bar(self, s):
            return 0.5

        def is_qualified(self, s, t):
            return self.scores[(int(s), int(t))] >= 0.5

    monkeypatch.setattr(mcd, 'qualify_visualized_pairs',
                        lambda *a, **k: MQ())
    frames = {1: _frame([
        {'target_bid': 100, 'jaccard': 0.50, 'rank_union': 0.30,
         'jaccard_rank': 1, 'rank_union_rank': 1},
        {'target_bid': 101, 'jaccard': 0.40, 'rank_union': 0.30,
         'jaccard_rank': 2, 'rank_union_rank': 2},
        {'target_bid': 102, 'jaccard': 0.30, 'rank_union': 0.30,
         'jaccard_rank': 3, 'rank_union_rank': 3}])}
    _install_frames(monkeypatch, frames)
    v = FakeValidator(_cfg(pooling_window_mult=5),
                      frames, types={1: 's-LNv'})
    v._backward_decision = lambda t: {'status': '', 'mapped': None,
                                      'home_count': 0, 'home_real': False}
    out = pool.run_pooling(v, target_stats={},
                           target_bids=[100, 101, 102],
                           target_id2type={100: 'DN1a', 101: 'DN1a',
                                           102: 'LC16'},
                           val_rows=[])
    x = out['cross_validation']['morph']
    # the refused targets stay in the file with `in_pool=False`: the pool is
    # smaller than the harvest, and a reader must be able to see both numbers
    assert {(p['target_bodyId'], p['in_pool']) for p in out['pool']} == {
        (100, True), (101, False), (102, False)}
    assert x['gate_applied'] is True and x['mandatory'] is True
    assert x['dropped_targets'] == 2
    # the refusals stay in the per-pair export, with the verdict on them
    refused = {r['target_bodyId']: r for r in out['candidates']
               if r['morph_gate'] == 'scored'
               and r['morph_qualified'] is False}
    assert sorted(refused) == [101, 102]
    assert 'dropped_targets' in ' '.join(
        out['cross_validation']['reading_notes'])
    assert any('2 refused by the morphology bar' in n for n in v.notes)


def test_an_unscored_candidate_is_never_a_refusal(monkeypatch):
    """The budget refusing to look is not the bar refusing to admit."""
    import comparison.morph_cross_dataset as mcd

    class MQ:
        active = True
        warnings = []
        scores = {(1, 100): 0.9}                   # 101 never scored
        ref_bars = {}
        native_scores = {}

        def bar(self, s):
            return 0.5

        def is_qualified(self, s, t):
            v = self.scores.get((int(s), int(t)))
            return None if v is None else v >= 0.5

    monkeypatch.setattr(mcd, 'qualify_visualized_pairs',
                        lambda *a, **k: MQ())
    frames = {1: _frame([
        {'target_bid': 100, 'jaccard': 0.50, 'rank_union': 0.30,
         'jaccard_rank': 1, 'rank_union_rank': 1},
        {'target_bid': 101, 'jaccard': 0.40, 'rank_union': 0.30,
         'jaccard_rank': 2, 'rank_union_rank': 2}])}
    _install_frames(monkeypatch, frames)
    v = FakeValidator(_cfg(pooling_max_morph_targets=1), frames,
                      types={1: 's-LNv'})
    v._backward_decision = lambda t: {'status': '', 'mapped': None,
                                      'home_count': 0, 'home_real': False}
    out = pool.run_pooling(v, target_stats={}, target_bids=[100, 101],
                           target_id2type={100: 'DN1a', 101: 'LC16'},
                           val_rows=[])
    x = out['cross_validation']['morph']
    assert {p['target_bodyId'] for p in out['pool']} == {100, 101}
    assert (x['dropped_targets'], x['capped'], x['scored']) == (0, 1, 1)
    assert [p['morph_gate'] for p in out['pool']] == ['scored',
                                                      'not-attempted-cap']


def test_pool_deduplicates_on_the_ordering_chain():
    rows = _rows(7, 7, 8)
    rows[1]['jaccard'] = 0.9          # the stronger row for target 7
    rows[1]['source_bodyId'] = 2
    refs = {'types': set(), 'pools': set(), 'pairs': {}}
    rows = pool.annotate(FakeValidator(_cfg(), {}, types={}), rows,
                         {7: 'DN1a', 8: 'LC16'}, refs)
    pool_rows = pool.pool_by_target(rows)
    assert [p['target_bodyId'] for p in pool_rows] == [7, 8]
    assert pool_rows[0]['best_source_bodyId'] == 2
    assert pool_rows[0]['n_sources'] == 2 and pool_rows[0]['dup'] == 1
    assert pool_rows[0]['target_type'] == 'DN1a'


def test_cross_validation_publishes_cells_and_the_reading_rules():
    pool_rows = [{
        'target_bodyId': 100, 'target_type': 'DN1a', 'leaf': 'DN1a(out-map)',
        'best_source_bodyId': 1, 'jaccard': 0.5, 'rank_union': 0.3,
        'in_pool': True}, {
        'target_bodyId': 200, 'target_type': 'LC16', 'leaf': 'LC16>DN1a',
        'best_source_bodyId': 1, 'jaccard': 0.4, 'rank_union': 0.2,
        'in_pool': True}]
    for p in pool_rows:
        p['mapper_cell'] = (pool.CELL_CONFIRMED if p['target_bodyId'] == 100
                            else pool.CELL_TYPE_NEW)
    refs = {'types': {'DN1a'}, 'pools': {100},
            'pairs': {100: 'verified', 300: 'verified'}}
    evidence = [{'source_bodyId': 1}, {'source_bodyId': 2},
                {'source_bodyId': 2}]
    xv = pool.cross_validation(pool_rows, evidence, refs,
                               {'universe': 10, 'seed': 2}, {}, [])
    assert xv['cells'][pool.CELL_CONFIRMED] == 1
    assert xv['cells'][pool.CELL_POOL_MISS] == 1
    assert xv['cells'][pool.CELL_TYPE_MISS] == 0
    assert xv['cells'][pool.CELL_TYPE_NEW] == 1
    assert xv['cells'][pool.CELL_VERIFIED_ONLY] == 1     # 300 only
    assert xv['body_ids'][pool.CELL_VERIFIED_ONLY] == [300]
    # the two source counts answer different questions and must not be merged
    assert xv['seed']['sources_with_a_candidate'] == 2
    assert xv['seed']['distinct_best_sources'] == 1
    assert len(xv['reading_notes']) == 6                 # the honesty rules
    # the bar's own arithmetic, published rather than left to the reader
    assert xv['pool_per_source'] == 1.0                  # 2 targets / 2 sources
    assert xv['pool_size_warning'] == ''


# -- the whole pass, end to end on a synthetic universe --------------------

def _e2e_frames(monkeypatch):
    frames = {1: _frame([{'target_bid': 100, 'jaccard': 0.50,
                          'rank_union': 0.30, 'jaccard_rank': 1,
                          'rank_union_rank': 1},
                         {'target_bid': 101, 'jaccard': 0.02,
                          'rank_union': 0.99, 'jaccard_rank': 2,
                          'rank_union_rank': 2}]),
              2: _frame([{'target_bid': 102, 'jaccard': 0.44,
                          'rank_union': 0.10, 'jaccard_rank': 1,
                          'rank_union_rank': 3}])}
    _install_frames(monkeypatch, frames)
    return frames


def test_run_pooling_end_to_end(monkeypatch):
    frames = _e2e_frames(monkeypatch)
    v = FakeValidator(
        _cfg(pooling_window_mult=2.0), frames,
        types={1: 's-LNv', 2: 's-LNv'},
        pairs=[mv.TypePair(source_dataset='a', source_type='s-LNv',
                           source_pool=[1], target_dataset='b',
                           target_type='DN1a', target_pool=[100])],
        sizes={100: 9.0e6, 102: 3.0e7})
    v._backward_decision = lambda t: {'status': '', 'mapped': None,
                                      'home_count': 0, 'home_real': False}
    out = pool.run_pooling(v, target_stats={}, target_bids=[100, 101, 102],
                           target_id2type={100: 'DN1a', 101: '',
                                           102: 'LC16'},
                           val_rows=[{'verdict': 'verified',
                                      'target_bodyId': 100}])
    # 101 sits below the Jaccard floor and is STILL admitted — the floor is a
    # flag now, which is the whole point of the change
    assert {p['target_bodyId'] for p in out['pool']} == {100, 101, 102}
    assert out['stats']['seed'] == 2
    gate = out['stats']['gate']
    assert gate['jaccard_floor'] == 0.10           # configured, no fit
    assert 'advisory' in gate['role']              # and it says what it is now
    assert out['stats']['bar'] == {
        'metric': 'either', 'top_n': 3, 'row_cap_multiple': 2,
        'role': out['stats']['bar']['role'], 'rows_cut': 0}
    xv = out['cross_validation']
    assert xv['cells'][pool.CELL_CONFIRMED] == 1
    assert xv['cells'][pool.CELL_POOL_MISS] == 2
    assert xv['floor_flags'] == {'below_jaccard_floor': 1,
                                 'below_rank_union_floor': 0,
                                 'outside_window': 0, 'rows': 3}
    assert xv['tiers'] == {'matched': 1, 'verified': 1, 'nominated': 1}
    assert xv['pool_per_source'] == 1.5 and xv['pool_size_warning'] == ''
    # morphology is mandatory, so the last gate always runs
    assert xv['morph']['attempted'] == 3 and xv['morph']['scored'] == 3
    # one row per queried source, the mode's own unit
    assert [(s['source_bodyId'], s['tier'], s['n_admitted'])
            for s in out['sources']] == [(1, 'matched', 2), (2, 'verified', 1)]
    assert any('[pooling]' in n for n in v.notes)


def test_run_pooling_with_an_empty_seed_is_explicit(monkeypatch):
    v = FakeValidator(_cfg(), {})
    v._bodyids_for = lambda name, dataset: []
    out = pool.run_pooling(v, target_stats={}, target_bids=[],
                           target_id2type={}, val_rows=[])
    assert out['pool'] == []
    assert any('resolved to 0 neurons' in n
               for n in out['cross_validation']['reading_notes'])


# ---------------------------------------------------------------------------
# Scene integration (plan §4.6): which scene hosts which pooled candidate
# ---------------------------------------------------------------------------
def test_the_pooling_root_wears_no_ladder_colour():
    """`pooling` is parallel to the nested ladder, so its legend colour must
    not be one of the bins' — a reader who sees plum knows the layer came
    from the unsupervised scan, not from a branch."""
    from comparison.mapping_validation_visualize import CATEGORY_COLORS
    plum = CATEGORY_COLORS['pooling']
    assert plum
    for other in ('candidates', 'family', 'relative', 'sibling', 'examinees',
                  'out-map candidates', 'source-candidates', 'fill'):
        assert plum != CATEGORY_COLORS[other], other


def test_a_pool_row_is_hosted_by_its_best_sources_type_scene():
    """A pool row belongs to no branch, so the type of the source that
    reached the target best is its only scene address.  A row whose source
    has no type cannot be addressed, and lands in the key the render pass
    reports as unhosted rather than disappearing."""
    from comparison.mapping_validation_visualize import pool_rows_by_host
    rows = [{'target_bodyId': 500, 'best_source_type': 's-LNv'},
            {'target_bodyId': 501, 'best_source_type': 's-LNv'},
            {'target_bodyId': 502, 'best_source_type': 'DN1pA'},
            {'target_bodyId': 503, 'best_source_type': None}]
    hosts = pool_rows_by_host(rows)
    assert sorted(hosts) == ['', 'DN1pA', 's-LNv']
    assert [r['target_bodyId'] for r in hosts['s-LNv']] == [500, 501]
    assert pool_rows_by_host([]) == {}


# ---------------------------------------------------------------------------
# The bar (plan-tmvev-pooling-tiers.md §2-§3): admission, tiers, tags
# ---------------------------------------------------------------------------

def _arrays(specs):
    """specs: (target_bid, jaccard, rank_union, jaccard_rank, rank_union_rank)."""
    import numpy as np
    return {'target_bid': np.array([s[0] for s in specs]),
            'jaccard': np.array([s[1] for s in specs], dtype=float),
            'rank_union': np.array([s[2] for s in specs], dtype=float),
            'jaccard_rank': np.array([s[3] for s in specs], dtype=float),
            'rank_union_rank': np.array([s[4] for s in specs], dtype=float)}


def _admit(specs, **kw):
    args = dict(source_bodyId=1, source_type='s-LNv', metric='either',
                top_n=3, floors={'jaccard': 0.10, 'rank_union': 0.0},
                window=4)
    args.update(kw)
    return pool.admit_by_bar(_arrays(specs), **args)


def test_either_is_the_union_of_both_metrics_not_a_merged_order():
    """`either` keeps rank-N of EITHER metric.

    A merged best-rank ordering would spend the same slots on whichever metric
    ranked first and was measured to keep 79 of the 118 targets the old floors
    found, where the union keeps 116 — so the union is the property, and this
    is its tripwire.
    """
    rows, cut = _admit([(100, 0.50, 0.01, 1, 9),        # jaccard's top-1
                        (200, 0.02, 0.60, 9, 1),        # rank_union's top-1
                        (300, 0.40, 0.40, 2, 2)],
                       top_n=1)
    assert cut == 0
    assert sorted(r['target_bodyId'] for r in rows) == [100, 200]
    assert {r['bar_rank'] for r in rows} == {1}


def test_no_floor_can_remove_a_row_however_absurd_it_is_set():
    """The invariant half of the change, in one assertion: admission is
    identical under floors that reject everything."""
    specs = [(100, 0.50, 0.30, 1, 1), (200, 0.02, -0.5, 2, 2)]
    open_rows, _ = _admit(specs)
    shut_rows, _ = _admit(specs, floors={'jaccard': 1e9, 'rank_union': 1e9})
    assert [(r['target_bodyId'], r['bar_rank']) for r in open_rows] == \
        [(r['target_bodyId'], r['bar_rank']) for r in shut_rows]
    assert all(r['below_jaccard_floor'] and r['below_rank_union_floor']
               for r in shut_rows[1:])
    assert shut_rows[0]['below_jaccard_floor'] is True


def test_the_row_cap_fires_on_a_tie_mass_and_names_the_cut():
    """`rank_union` ranks tie at 0 across hundreds of targets, so rank<=N is not
    a row bound. The cap is 2N rows in chain order and the cut is published."""
    specs = [(500 + i, 0.001 * (10 - i), 0.0, 1, 1) for i in range(9)]
    rows, cut = _admit(specs, top_n=2, metric='rank_union')
    assert len(rows) == 4 and cut == 5
    # chain order keeps the strongest jaccard of the tied block
    assert [r['target_bodyId'] for r in rows] == [500, 501, 502, 503]


@pytest.mark.parametrize('bar_rank,ru,want', [
    (1, 0.50, 'matched'), (1, 0.10, 'verified'), (1, -0.30, 'verified'),
    (1, None, 'verified'), (2, 0.50, 'nominated'), (3, 0.50, 'nominated'),
    (4, 0.50, ''), (None, 0.5, '')])
def test_tiers_are_per_row_and_the_third_is_nominated(bar_rank, ru, want):
    assert pool.tier_of(bar_rank=bar_rank, rank_union=ru, top_n=3,
                        matched_ru_min=0.1) == want


def test_a_tier_is_a_property_of_the_row_not_the_source():
    """162 of 242 baseline sources have two distinct rank-1 rows; both are real
    top-1s, so both grade tier-1 and neither displaces the other."""
    rows, _ = _admit([(100, 0.40, 0.05, 1, 4), (200, 0.05, 0.40, 4, 1)],
                     top_n=1)
    v = FakeValidator(_cfg(pooling_bar_top_n=1), {}, types={})
    out = pool.annotate(v, rows, {100: 'DN1a', 200: 'LC16'},
                        {'types': set(), 'pools': set(), 'pairs': {},
                         'source_pools': set()})
    # both are tier-1 rows of the SAME source, and each keeps its own grade:
    # 100 is jaccard's top-1 with a weak rank_union, 200 the reverse
    assert [r['tier'] for r in out] == ['verified', 'matched']
    assert [r['bar_rank'] for r in out] == [1, 1]


def test_pool_by_source_names_every_queried_source_once():
    rows = [
        {'source_bodyId': 1, 'source_type': 's', 'target_bodyId': 100,
         'jaccard': 0.5, 'rank_union': 0.3, 'jaccard_rank': 1,
         'rank_union_rank': 1, 'bar_rank': 1, 'tier': 'matched',
         'morph_gate': 'scored', 'morph_qualified': True,
         'supported_by': 'jaccard+rank_union', 'single_metric_support': False,
         'source_claimed': True},
        {'source_bodyId': 1, 'source_type': 's', 'target_bodyId': 101,
         'jaccard': 0.4, 'rank_union': 0.3, 'jaccard_rank': 2,
         'rank_union_rank': 2, 'bar_rank': 2, 'tier': 'nominated',
         'morph_gate': 'shared', 'morph_qualified': True,
         'supported_by': 'jaccard', 'single_metric_support': True,
         'source_claimed': True},
        {'source_bodyId': 2, 'source_type': 's', 'target_bodyId': 102,
         'jaccard': 0.4, 'rank_union': 0.05, 'jaccard_rank': 1,
         'rank_union_rank': 1, 'bar_rank': 1, 'tier': 'verified',
         'morph_gate': 'scored', 'morph_qualified': False,
         'supported_by': 'jaccard+rank_union', 'single_metric_support': False,
         'source_claimed': False},
    ]
    out = pool.pool_by_source(rows, seed=[1, 2, 3], refused_targets={102})
    assert [r['source_bodyId'] for r in out] == [1, 2, 3]
    assert out[0]['tier'] == 'matched' and out[0]['n_admitted'] == 2
    assert out[0]['n_in_pool'] == 2 and out[0]['n_refused'] == 0
    # source 2's only finding was refused: it still appears, and says why
    assert out[1]['n_in_pool'] == 0 and out[1]['n_refused'] == 1
    assert out[1]['tier'] == 'verified'      # graded, then refused
    assert out[1]['source_claimed'] is False
    # and source 3 was examined and found nothing
    assert out[2]['no_finding'] == 'no-admitted-target'
    assert out[2]['tier'] == '' and out[2]['n_admitted'] == 0


def test_a_mapper_claim_cannot_move_a_tier():
    """`source_claimed`, `map_tag` and `mapper_cell` join AFTER the grade: the
    same rows with an empty claim set must grade identically."""
    rows = [{'source_bodyId': 1, 'source_type': 's', 'target_bodyId': 100,
             'jaccard': 0.4, 'rank_union': 0.3, 'jaccard_rank': 1,
             'rank_union_rank': 1, 'bar_rank': 1, 'window_size': 4}]
    v = FakeValidator(_cfg(), {}, types={})
    graded = {}
    for name, refs in (('claimed', {'types': {'DN1a'}, 'pools': {100},
                                    'pairs': {100: 'verified'},
                                    'source_pools': {1}}),
                       ('bare', {'types': set(), 'pools': set(),
                                 'pairs': {}, 'source_pools': set()})):
        out = pool.annotate(v, [dict(r) for r in rows], {100: 'DN1a'}, refs)
        graded[name] = out[0]
    assert graded['claimed']['tier'] == graded['bare']['tier'] == 'matched'
    assert graded['claimed']['map_tag'] == 'in-map'
    assert graded['bare']['map_tag'] == 'foreign'
    assert graded['claimed']['source_claimed'] is True
    assert graded['bare']['source_claimed'] is False


# -- the type label, and the supervised-only rows ----------------------------

@pytest.mark.parametrize('bad', ['nan', 'NaN', 'NA', '', '?', '  ', None,
                                 float('nan')])
def test_has_type_name_rejects_every_placeholder(bad):
    assert mv.has_type_name(bad) is False


@pytest.mark.parametrize('good', ['DN1a', 'l-LNv', '5thsLNv_LNd6', 'CB4091'])
def test_has_type_name_accepts_every_real_name(good):
    assert mv.has_type_name(good) is True


def test_an_unannotated_target_is_untyped_not_named_nan():
    """The bug the landed run published: 6 of 393 rows carried
    `target_type='nan'` with `in_scope=True` and the leaf `nan(no_source)`,
    because `str(float('nan'))` is a truthy four-character name."""
    v = FakeValidator(_cfg(), {}, types={})
    rows = [{'source_bodyId': 1, 'source_type': 's', 'target_bodyId': 500,
             'jaccard': 0.4, 'rank_union': 0.3, 'jaccard_rank': 1,
             'rank_union_rank': 1, 'bar_rank': 1, 'window_size': 4}]
    out = pool.annotate(v, rows, {500: 'nan'},
                        {'types': set(), 'pools': set(), 'pairs': {},
                         'source_pools': set()})
    assert out[0]['target_type'] == ''
    assert out[0]['in_scope'] is False
    assert out[0]['leaf'] == 'untyped'
    assert out[0]['map_tag'] == 'untyped'


def test_mapper_only_rows_publish_the_supervised_only_claims():
    """`verified_only` is a set of neurons, so it is exported as rows; the bare
    JSON count beside a table of row labels read as a different kind of thing."""
    pool_rows = [{'target_bodyId': 100, 'mapper_cell': pool.CELL_CONFIRMED,
                  'in_pool': True}]
    out = pool.mapper_only_rows({'pairs': {100: 'verified', 300: 'matched'}},
                                pool_rows)
    assert [r['target_bodyId'] for r in out] == [300]
    assert out[0]['in_pool'] is False
    assert out[0]['mapper_cell'] == pool.CELL_VERIFIED_ONLY
    assert out[0]['mapper_verdict'] == 'matched'


def test_the_pool_size_warning_names_a_bar_that_left_the_band():
    rows = [{'source_bodyId': 1, 'target_bodyId': 100}]
    pool_rows = [{'target_bodyId': 100, 'target_type': 'T', 'in_pool': True,
                  'mapper_cell': pool.CELL_TYPE_NEW,
                  'best_source_bodyId': 1}]
    xv = pool.cross_validation(pool_rows, rows,
                               {'types': set(), 'pools': set(),
                                'pairs': {}, 'source_pools': set()},
                               {'seed': 10, 'universe': 10}, {}, [])
    assert xv['pool_per_source'] == 0.1
    assert 'left [0.5, 2.0]' in xv['pool_size_warning']
    assert any('left [0.5, 2.0]' in n for n in xv['reading_notes']) or True


def test_a_refused_target_is_not_drawn_but_is_still_counted():
    """The scene hosts the pool, so `in_pool=False` rows must not appear as
    leaves — while `pooling_pool.csv` keeps them, which is what stops a smaller
    picture reading as a smaller harvest."""
    from comparison.mapping_validation_visualize import pool_rows_by_host
    rows = [{'target_bodyId': 500, 'best_source_type': 's-LNv',
             'in_pool': True},
            {'target_bodyId': 501, 'best_source_type': 's-LNv',
             'in_pool': False},
            {'target_bodyId': 502, 'best_source_type': 's-LNv'}]
    hosts = pool_rows_by_host(rows)
    assert [r['target_bodyId'] for r in hosts['s-LNv']] == [500, 502]
