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
                validation_mode='pooling', pooling_morph_gate=False)
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


def test_gate_keeps_only_absolute_evidence(monkeypatch):
    frames = {1: _frame([
        # jaccard, rank_union, ranks — target 100 passes, the rest do not
        {'target_bid': 100, 'jaccard': 0.40, 'rank_union': 0.20,
         'jaccard_rank': 1, 'rank_union_rank': 1},
        {'target_bid': 101, 'jaccard': 0.09, 'rank_union': 0.90,
         'jaccard_rank': 2, 'rank_union_rank': 2},        # below J
        {'target_bid': 102, 'jaccard': 0.40, 'rank_union': -0.10,
         'jaccard_rank': 3, 'rank_union_rank': 3},        # below RU
        {'target_bid': 103, 'jaccard': 0.40, 'rank_union': 0.20,
         'jaccard_rank': 9, 'rank_union_rank': 9},        # outside window
    ])}
    _install_frames(monkeypatch, frames)
    v = FakeValidator(_cfg(), frames, types={1: 's-LNv'})
    rows, stats = pool.connectivity_candidates(
        v, [1], {1: 's-LNv'}, {1: 4}, {}, [100, 101, 102, 103], 0.10)
    assert [r['target_bodyId'] for r in rows] == [100]
    assert stats['rows_scored'] == 4 and stats['rows_kept'] == 1


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

def _rows(*targets):
    return [{'source_bodyId': 1, 'source_type': 's', 'target_bodyId': t,
             'jaccard': 0.4, 'jaccard_rank': 1, 'rank_union': 0.2,
             'rank_union_rank': 1, 'window_size': 2} for t in targets]


def test_morph_gate_disabled_is_labelled_not_blank():
    v = FakeValidator(_cfg(pooling_morph_gate=False), {})
    rows, info = pool.apply_morph_gate(v, _rows(1, 2))
    assert all(r['morph_gate'] == 'disabled' for r in rows)
    assert info['attempted'] == 0


def test_run_wide_structural_switch_also_disables_the_pooling_gate():
    """`--no-morphology` means no morphology is spent, here either."""
    v = FakeValidator(_cfg(pooling_morph_gate=True, morph_enabled=False), {})
    rows, info = pool.apply_morph_gate(v, _rows(1, 2))
    assert all(r['morph_gate'] == 'disabled' for r in rows)
    assert info['attempted'] == 0


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
    v = FakeValidator(_cfg(pooling_morph_gate=True,
                           pooling_max_morph_targets=1), {})
    rows, info = pool.apply_morph_gate(v, _rows(1, 2))
    gates = {r['target_bodyId']: r['morph_gate'] for r in rows}
    assert info['attempted'] == 1 and info['capped'] == 1
    assert gates[2] == 'not-attempted-cap'
    assert gates[1] == 'scored'
    assert info['qualified'] == 1


def test_the_three_absences_are_three_labels(monkeypatch):
    """`not-selected` (the verdict lives on the chain-best row),
    `not-attempted-cap` (the budget refused), `no-score` (the scorer returned
    nothing for the pair) — none of them may read as a rejected candidate.

    The third label earned its keep: an earlier build's rows read `no-score`
    for candidates the scorer HAD scored, because the shared scorer prunes
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
    v = FakeValidator(_cfg(pooling_morph_gate=True,
                           pooling_max_morph_targets=0), {})
    rows = _rows(7, 7, 8)                # target 7 has two evidence rows
    rows, info = pool.apply_morph_gate(v, rows)
    gates = [(r['target_bodyId'], r['morph_gate']) for r in rows]
    assert gates == [(7, 'scored'), (7, 'not-selected'), (8, 'no-score')]
    assert info['attempted'] == 2 and info['capped'] == 0
    assert info['qualified'] == 1
    # `scored` counts what the scorer RETURNED a value for, not what the
    # budget attempted: collapsing them (the first version said scored 2)
    # hides a run where the last gate measured almost nothing.
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
    v = FakeValidator(_cfg(pooling_morph_gate=True), {})
    rows, info = pool.apply_morph_gate(v, _rows(7, 8))
    assert seen.get('mode') == 'mapping_ref'
    assert seen.get('prune_pool_refs') is False
    assert info['scored'] == 2 and info['no_score'] == 0


def test_morph_failure_is_recorded_per_row(monkeypatch):
    import comparison.morph_cross_dataset as mcd

    def boom(*a, **k):
        raise RuntimeError('no vectors for this dataset')
    monkeypatch.setattr(mcd, 'qualify_visualized_pairs', boom)
    v = FakeValidator(_cfg(pooling_morph_gate=True), {})
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
    v = FakeValidator(_cfg(pooling_morph_gate=True, pooling_window_mult=5),
                      frames, types={1: 's-LNv'})
    v._backward_decision = lambda t: {'status': '', 'mapped': None,
                                      'home_count': 0, 'home_real': False}
    out = pool.run_pooling(v, target_stats={},
                           target_bids=[100, 101, 102],
                           target_id2type={100: 'DN1a', 101: 'DN1a',
                                           102: 'LC16'},
                           val_rows=[])
    x = out['cross_validation']['morph']
    assert {p['target_bodyId'] for p in out['pool']} == {100}
    assert x['gate_applied'] is True
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
    v = FakeValidator(_cfg(pooling_morph_gate=True,
                           pooling_max_morph_targets=1), frames,
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
        'best_source_bodyId': 1, 'jaccard': 0.5, 'rank_union': 0.3}, {
        'target_bodyId': 200, 'target_type': 'LC16', 'leaf': 'LC16>DN1a',
        'best_source_bodyId': 1, 'jaccard': 0.4, 'rank_union': 0.2}]
    for p in pool_rows:
        p['mapper_cell'] = (pool.CELL_CONFIRMED if p['target_bodyId'] == 100
                            else pool.CELL_TYPE_NEW)
    refs = {'types': {'DN1a'}, 'pools': {100},
            'pairs': {100: 'verified', 300: 'verified'}}
    evidence = [{'source_bodyId': 1}, {'source_bodyId': 2},
                {'source_bodyId': 2}]
    xv = pool.cross_validation(pool_rows, evidence, refs,
                               {'universe': 10}, {}, [])
    assert xv['cells'][pool.CELL_CONFIRMED] == 1
    assert xv['cells'][pool.CELL_POOL_MISS] == 1
    assert xv['cells'][pool.CELL_TYPE_MISS] == 0
    assert xv['cells'][pool.CELL_TYPE_NEW] == 1
    assert xv['cells'][pool.CELL_VERIFIED_ONLY] == 1     # 300 only
    assert xv['body_ids'][pool.CELL_VERIFIED_ONLY] == [300]
    # the two source counts answer different questions and must not be merged
    assert xv['seed']['sources_with_a_candidate'] == 2
    assert xv['seed']['distinct_best_sources'] == 1
    assert len(xv['reading_notes']) == 4                 # the honesty rules


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
    # 101 is below the J floor and must not appear at all
    assert {p['target_bodyId'] for p in out['pool']} == {100, 102}
    assert out['stats']['seed'] == 2
    gate = out['stats']['gate']
    assert gate['jaccard_floor'] == 0.10           # configured, no fit
    assert 'volume' in gate['role']                # and it says so
    assert out['cross_validation']['cells'][pool.CELL_CONFIRMED] == 1
    assert out['cross_validation']['cells'][pool.CELL_POOL_MISS] == 1
    assert out['cross_validation']['morph']['attempted'] == 0   # gate off
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
