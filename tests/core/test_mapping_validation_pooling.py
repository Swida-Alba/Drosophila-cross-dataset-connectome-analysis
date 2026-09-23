"""Unit tests for TM VEV `pooling` mode (plan-tmvev-pooling-mode.md).

The tests exist to hold the two properties the mode is defined by, not the
arithmetic: selection must not depend on the type mapper (§1, §2.3), and a
missing measurement must never read like a rejection (§4.2, §6.4).
"""
import json
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
                validation_mode='pooling', pooling_morph_gate=False,
                # the J-floor store lives in the project's cache/; unit tests
                # never write there, so the fit is exercised explicitly below
                pooling_floor_from_evidence=False)
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
    `not-attempted-cap` (the budget refused), `no-score` (the scorer had no
    vector) — none of them may read as a rejected candidate."""
    import comparison.morph_cross_dataset as mcd

    class MQ:
        active = True
        warnings = []
        scores = {(1, 7): 0.9}           # (1, 8) was never scored
        ref_bars = {}

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
    # a row without a score carries no verdict, not a False
    assert [r['morph_qualified'] for r in rows if r['morph_gate'] ==
            'no-score'] == [None]


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
    assert len(xv['reading_notes']) == 3                 # the honesty rules


# -- cross-target corroboration (advisory ranking, plan §2.5) ---------------

def _sibling_run(tmp_path, name, dataset, pairs):
    """A sibling pooling run folder: its target dataset + its pool."""
    d = tmp_path / name
    (d / 'pooling').mkdir(parents=True)
    (d / 'parameters.json').write_text(json.dumps({'target_dataset': dataset}))
    (d / 'pooling' / 'pooling_pool.csv').write_text(
        'best_source_bodyId,target_type,target_bodyId\n'
        + ''.join(f'{s},{t},{9999 + i}\n'
                  for i, (s, t) in enumerate(pairs)))
    return str(d)


def test_corroboration_is_blank_until_the_join_is_asked_for(tmp_path):
    rows = [{'source_bodyId': 1, 'target_type': 'DN1a'},
            {'source_bodyId': 2, 'target_type': 'LC16'}]
    pool_rows = [dict(r, best_source_bodyId=r['source_bodyId']) for r in rows]
    # One run has one target, so with no siblings named the honest answer is
    # blank: a 0 would claim "no other target agrees", about runs that were
    # never run.
    summary = pool.corroborate_pool(rows, pool_rows,
                                    pool.corroborate([], 'banc_v888'))
    assert [r['targets_corroborated'] for r in rows] == [None, None]
    assert summary['status'] == 'not-computed'
    assert summary['pool_size_histogram'] == {'not-computed': 2}

    sib = _sibling_run(tmp_path, 'run_mcns', 'male-cns:v1.0',
                       [(1, 'DN1a'), (2, 'SMP228')])
    summary = pool.corroborate_pool(rows, pool_rows,
                                    pool.corroborate([sib], 'banc_v888'))
    assert rows[0]['targets_corroborated'] == 2      # this run + the sibling
    assert rows[1]['targets_corroborated'] == 1      # this run only
    assert pool_rows[0]['targets_corroborated'] == 2
    assert summary['sibling_runs'] == 1
    assert summary['pool_size_histogram'] == {'2': 1, '1': 1}
    # what the count is keyed on must be stated, not guessed
    assert 'TYPE' in summary['key']


def test_corroboration_keys_on_the_type_name_not_a_target_bodyId(tmp_path):
    # target bodyIds do not transfer across datasets, so the sibling's own
    # bodyIds (9999+) are never part of the key.
    sib = _sibling_run(tmp_path, 'run_hemi', 'hemibrain', [(1, 'DN1a')])
    corr = pool.corroborate([sib], 'banc_v888')
    assert pool._corroborated(corr, 1, 'DN1a') == 2
    assert pool._corroborated(corr, 1, 'dn1a') == 1     # names are exact
    assert pool._corroborated(corr, 99, 'DN1a') == 1    # another source


def test_unreadable_sibling_dirs_are_named_not_silently_dropped(tmp_path):
    missing = str(tmp_path / 'no_such_run')
    corr = pool.corroborate([missing], 'banc_v888')
    assert corr['status'] == 'not-computed'
    assert corr['sibling_runs'] == 0
    assert corr['unreadable'] == [missing]


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
    assert out['stats']['seed'] == 2 and out['stats']['gate'][
        'jaccard_floor'] == 0.10
    assert out['cross_validation']['cells'][pool.CELL_CONFIRMED] == 1
    assert out['cross_validation']['cells'][pool.CELL_POOL_MISS] == 1
    assert out['cross_validation']['morph']['attempted'] == 0   # gate off
    assert any('[pooling]' in n for n in v.notes)


def test_run_pooling_publishes_the_sibling_join(monkeypatch, tmp_path):
    frames = _e2e_frames(monkeypatch)
    sib = _sibling_run(tmp_path, 'run_mcns', 'male-cns:v1.0', [(1, 'DN1a')])
    v = FakeValidator(_cfg(pooling_sibling_runs=[sib]), frames,
                      types={1: 's-LNv', 2: 's-LNv'})
    v._backward_decision = lambda t: {'status': '', 'mapped': None,
                                      'home_count': 0, 'home_real': False}
    out = pool.run_pooling(v, target_stats={}, target_bids=[100, 101, 102],
                           target_id2type={100: 'DN1a', 102: 'LC16'},
                           val_rows=[])
    corr = out['cross_validation']['corroboration']
    assert corr['status'] == 'computed' and corr['sibling_runs'] == 1
    # 100 is the DN1a candidate source 1 reaches, which the sibling also
    # pooled; 102 is only this run's.
    assert {p['target_bodyId']: p['targets_corroborated']
            for p in out['pool']} == {100: 2, 102: 1}
    assert corr['pool_size_histogram'] == {'2': 1, '1': 1}


def test_run_pooling_with_an_empty_seed_is_explicit(monkeypatch):
    v = FakeValidator(_cfg(), {})
    v._bodyids_for = lambda name, dataset: []
    out = pool.run_pooling(v, target_stats={}, target_bids=[],
                           target_id2type={}, val_rows=[])
    assert out['pool'] == []
    assert 'queried population resolved to 0 neurons' in \
        out['cross_validation']['reading_notes'][3]


# -- the per-dataset Jaccard floor (P4) --------------------------------------

def _store(tmp_path):
    return pool.JaccardFloorStore('flywire_FAFB_v783', 'banc_v888',
                                 project_root=str(tmp_path))


def _graded(n, low=0.02):
    return [{'verdict': 'verified', 'source_bodyId': 10 + i,
             'target_bodyId': 100 + i, 'jaccard': low + i / 100}
            for i in range(n)]


def test_the_floor_fits_to_the_low_tail_of_the_recorded_evidence(tmp_path):
    # a pair whose verified rows bottom out near 0.02 must not be gated at the
    # global 0.10 — that floor would reject this dataset's own verified evidence
    st = _store(tmp_path)
    assert st.observe(_graded(30)) == 30
    floor, meta = st.fit(0.10)
    assert meta['source'] == pool.FLOOR_FITTED and meta['n'] == 30
    assert floor == pytest.approx(meta['q05'])
    assert floor < 0.10


def test_a_thin_sample_leaves_the_configured_floor_alone(tmp_path):
    st = _store(tmp_path)
    st.observe(_graded(st.MIN_N - 1))
    floor, meta = st.fit(0.10)
    assert floor == 0.10 and meta['source'] == pool.FLOOR_THIN
    assert 'below min_n' in meta['why']


def test_evidence_is_deduplicated_by_pair(tmp_path):
    st = _store(tmp_path)
    same_pair = [{'verdict': 'verified', 'source_bodyId': 1,
                  'target_bodyId': 100, 'jaccard': j} for j in (0.2, 0.9)]
    assert st.observe(same_pair) == 1
    assert st.observe(same_pair) == 0            # already on record
    assert list(st.load().values()) == [0.9]     # the later grade stands


def test_the_cap_keeps_the_low_tail_because_q05_reads_it(tmp_path, monkeypatch):
    monkeypatch.setattr(pool.JaccardFloorStore, 'CAP', 3)
    st = _store(tmp_path)
    st.observe(_graded(10))
    assert sorted(st.load().values()) == [0.02, 0.03, 0.04]


def test_ungraded_rows_are_not_evidence(tmp_path):
    st = _store(tmp_path)
    rows = _graded(3) + [{'verdict': 'borderline', 'source_bodyId': 9,
                          'target_bodyId': 99, 'jaccard': 0.4},
                         {'verdict': 'unmatched', 'source_bodyId': 8,
                          'target_bodyId': 98, 'jaccard': 0.4}]
    assert st.observe(rows) == 3


def test_a_run_cannot_move_its_own_floor(monkeypatch, tmp_path):
    """The ordering IS the unsupervised property, expressed in time.

    The evidence is read before the scan and written after it, so the first run
    on a store gates on the configured floor even though it contributes rows
    that would fit far lower, and only the second run sees a fitted floor.
    """
    frames = {1: _frame([{'target_bid': 100, 'jaccard': 0.50,
                          'rank_union': 0.30, 'jaccard_rank': 1,
                          'rank_union_rank': 1},
                         {'target_bid': 101, 'jaccard': 0.05,
                          'rank_union': 0.40, 'jaccard_rank': 2,
                          'rank_union_rank': 2}])}
    _install_frames(monkeypatch, frames)
    v = FakeValidator(_cfg(pooling_floor_from_evidence=True), frames,
                      types={1: 's-LNv'})
    v.project_root = str(tmp_path)
    v._backward_decision = lambda t: {'status': '', 'mapped': None,
                                      'home_count': 0, 'home_real': False}
    val = _graded(30)

    def run():
        return pool.run_pooling(v, target_stats={}, target_bids=[100, 101],
                                target_id2type={100: 'DN1a', 101: 'DN1a'},
                                val_rows=val)

    first = run()
    g1 = first['cross_validation']['gate']
    assert g1['jaccard_floor'] == 0.10
    assert g1['jaccard_floor_source'] == pool.FLOOR_THIN
    assert g1['jaccard_floor_pairs_added'] == 30       # written AFTER the scan
    assert {r['target_bodyId'] for r in first['candidates']} == {100}

    second = run()
    g2 = second['cross_validation']['gate']
    assert g2['jaccard_floor_source'] == pool.FLOOR_FITTED
    assert g2['jaccard_floor'] == pytest.approx(g2['jaccard_floor_q05'])
    assert g2['jaccard_floor'] < 0.05
    # and the gate really moved: 101 (jaccard 0.05) is below the configured
    # floor and above the fitted one
    assert {r['target_bodyId'] for r in second['candidates']} == {100, 101}


def test_the_floor_stay_can_be_switched_off(monkeypatch, tmp_path):
    st = _store(tmp_path)
    st.observe(_graded(30))
    floor, meta = pool.resolve_jaccard_floor(
        _cfg(pooling_floor_from_evidence=False), project_root=str(tmp_path))
    assert (floor, meta['source']) == (0.10, pool.FLOOR_CONFIG)
    floor, meta = pool.resolve_jaccard_floor(
        _cfg(pooling_floor_from_evidence=True), project_root=str(tmp_path))
    assert meta['source'] == pool.FLOOR_FITTED and floor < 0.10
