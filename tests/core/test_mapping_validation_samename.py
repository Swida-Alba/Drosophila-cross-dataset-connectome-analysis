"""Tests for the TM VEV same-name-first consumers
(plan-tmvev-samename-first-consumers.md).

- P0: examinees rename (report reader legacy fallback)
- P1: fired-pair provenance (TypePair.same_name_first, row/summary
  columns, set_coverage counters)
- P2: held/excluded same-name fan-out accounting (same_name_excluded.csv)
- P3: suspects verification — opt-in default OFF, isolation, skips,
  partial-mapper guards
- P4: multivalue type cells accounted, never split
- the backward-decision improvement from fired reverse decisions

Offline, synthetic; partial validator instances (the standing test
pattern) exercise the getattr guards.
"""

import sys
from pathlib import Path

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
    compute_set_coverage,
    expanded_vector,
    prep_target_stats,
    run_file_path,
    scan_source,
    _write_csv,
)
from comparison.mapping_validation_report import _read_examinees  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def make_profile(bid, up, dn):
    return ConnectivityProfile(
        neuron_id=bid, dataset='test',
        upstream_partners=dict(up),
        downstream_partners=dict(dn),
        actual_upstream_count=len(up),
        actual_downstream_count=len(dn),
    )


SNF_FIRED = {
    'status': 'mapped', 'target_type': 'aMe9', 'target_types': ['aMe9'],
    'relationship': 'suspects', 'suspects': True,
    'same_name_first': {'fires': True, 'selected': 'aMe9',
                        'rivals': ['aMe12', 'aMe6'],
                        'path': 'valid_split_evidence',
                        'disposition': 'broad_selection'},
}
SNF_HELD = {
    'status': 'conflict', 'target_type': None,
    'target_types': ['ORN_D', 'ORN_DA1', 'ORN_DA4m'],
    'relationship': '1-to-N',
    'same_name_first': {'fires': False, 'selected': 'ORN_D',
                        'rivals': ['ORN_DA1', 'ORN_DA4m'],
                        'path': 'conflict', 'disposition': 'gated_held'},
}


class FakeMapper:
    """Decision table keyed by (type, src_ds, tgt_ds); P3 accessors
    optional via flags so partial-mapper paths stay testable."""

    def __init__(self, decisions, with_accessors=True):
        self._decisions = decisions
        self.include_suspects_in_targets = False
        self.with_accessors = with_accessors

    def get_mapping_decision(self, t, src, tgt, route_scope=None):
        return self._decisions.get(
            (t, src, tgt),
            {'status': 'unmapped', 'target_type': None,
             'target_types': []})

    def standardize_partner_types(self, expanded, dataset):
        return expanded

    def same_name_first_fires(self, t, src, tgt):
        if not self.with_accessors:
            raise AttributeError('removed accessor')
        dec = self._decisions.get((t, src, tgt))
        return (dec or {}).get('same_name_first')

    def get_same_name_conflict_detail(self, t, src, tgt):
        dec = self._decisions.get((t, src, tgt)) or {}
        snf = dec.get('same_name_first') or {}
        return {'source_type': t, 'selected': snf.get('selected'),
                'rivals': list(snf.get('rivals') or []),
                'disposition': snf.get('disposition'),
                'rival_evidence': [
                    {'rival': r, 'rival_has_own_clean_pair': r == 'aMe6',
                     'reverse_target': t if r == 'aMe6' else None,
                     'votes': '2/3'} for r in (snf.get('rivals') or [])]}


class FakeResolver:
    def __init__(self, pools):
        self._pools = pools

    def type_bodyid_pool(self, name, dataset):
        return list(self._pools.get((name, dataset), []))


class FakeProfiler:
    def __init__(self, profiles):
        self._profiles = profiles

    def get_profile(self, bid, dataset):
        return self._profiles.get(bid)

    def get_types_for_bodyids(self, bids, dataset):
        return {}


def make_validator(decisions=None, pools=None, profiles=None,
                   with_accessors=True, **cfg_overrides):
    cfg_kwargs = dict(source_dataset='dsA', target_dataset='dsB',
                      query_types=['T'], morph_enabled=False,
                      visualize=False, verbose=False)
    cfg_kwargs.update(cfg_overrides)
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(**cfg_kwargs)
    v.notes = []
    v.mapper = FakeMapper(decisions or {}, with_accessors=with_accessors)
    v.resolver = FakeResolver(pools or {})
    v.profiler = FakeProfiler(profiles or {})
    v.pairs = []
    return v


# ---------------------------------------------------------------------------
# P0 — examinees rename: report reader fallback
# ---------------------------------------------------------------------------

def test_read_examinees_legacy_fallback(tmp_path):
    assert _read_examinees(tmp_path) == []
    (tmp_path / 'suspicious_candidates.csv').write_text('a\n1\n')
    assert _read_examinees(tmp_path) == [{'a': '1'}]
    (tmp_path / 'examinees.csv').write_text('a\n2\n')
    assert _read_examinees(tmp_path) == [{'a': '2'}]


# ---------------------------------------------------------------------------
# P1 — fired-pair provenance
# ---------------------------------------------------------------------------

def test_p1_fired_pair_carries_provenance():
    v = make_validator(
        decisions={('aMe9', 'dsA', 'dsB'): SNF_FIRED},
        pools={('aMe9', 'dsB'): [201, 202]})
    pairs = v._pairs_for_type('aMe9', [11, 12], 'aMe9')
    assert len(pairs) == 1
    p = pairs[0]
    assert p.same_name_first == {
        'selected': 'aMe9', 'rivals': ['aMe12', 'aMe6'],
        'path': 'valid_split_evidence',
        'disposition': 'broad_selection'}
    assert p.relationship == 'suspects'
    assert p.status == 'mapped'
    # the advisory columns ride on the row/summary surfaces
    row = v._val_row(p, 11, 'unmatched')
    assert row['same_name_first'] is True
    assert row['same_name_rivals'] == 'aMe12;aMe6'
    summary = MappingValidator._summary(p, [], [])
    assert summary['same_name_first'] is True
    assert summary['same_name_rivals'] == 'aMe12;aMe6'


def test_p1_ordinary_pair_unmarked():
    v = make_validator(
        decisions={('T1', 'dsA', 'dsB'):
                   {'status': 'mapped', 'target_type': 'T1',
                    'target_types': ['T1']}},
        pools={('T1', 'dsB'): [201]})
    pairs = v._pairs_for_type('T1', [11], 'T1')
    assert len(pairs) == 1
    assert pairs[0].same_name_first is None
    row = v._val_row(pairs[0], 11, 'unmatched')
    assert row['same_name_first'] is False
    assert row['same_name_rivals'] == ''


def test_p1_set_coverage_counters_additive():
    v = make_validator(
        decisions={('aMe9', 'dsA', 'dsB'): SNF_FIRED},
        pools={('aMe9', 'dsB'): [201, 202]})
    pairs = v._pairs_for_type('aMe9', [11], 'aMe9')
    cov = compute_set_coverage(pairs, {}, [])
    assert cov['same_name_first_pairs'] == 1
    assert cov['same_name_first_types'] == 1
    # ordinary pairs contribute zero but the keys always exist
    pair_plain = TypePair(
        source_dataset='dsA', source_type='T1', source_pool=[1],
        target_dataset='dsB', target_type='T1', target_pool=[2])
    cov2 = compute_set_coverage([pair_plain], {}, [])
    assert cov2['same_name_first_pairs'] == 0
    assert cov2['same_name_first_types'] == 0


# ---------------------------------------------------------------------------
# P2 — held / evidence-only accounting
# ---------------------------------------------------------------------------

def test_p2_held_type_recorded_and_still_excluded():
    v = make_validator(
        decisions={('ORN_D', 'dsA', 'dsB'): SNF_HELD},
        pools={('ORN_D', 'dsB'): [301]})
    pairs = v._pairs_for_type('ORN_D', [21], 'ORN_D')
    assert pairs == []  # fail-closed unchanged
    rows = getattr(v, '_same_name_excluded', [])
    assert len(rows) == 1
    r = rows[0]
    assert r['source_type'] == 'ORN_D'
    assert r['disposition'] == 'gated_held'
    assert r['decision_status'] == 'conflict'
    assert r['n_rivals'] == 2
    assert r['rivals'] == 'ORN_DA1;ORN_DA4m'
    assert r['reason'] == 'same_name_fanout'
    # the improved log line says WHY and WHERE to look
    held_notes = [n for n in v.notes if 'held' in n
                  and 'auto_type_mapping_suspects.csv' in n]
    assert held_notes


def test_p2_same_name_excluded_csv_export(tmp_path):
    v = make_validator()
    v.run_dir = tmp_path
    v._same_name_excluded = [{
        'query': 'ORN_D', 'source_type': 'ORN_D',
        'decision_status': 'conflict', 'disposition': 'gated_held',
        'selected': 'ORN_D', 'n_rivals': 2,
        'rivals': 'ORN_DA1;ORN_DA4m', 'reason': 'same_name_fanout'}]
    MappingValidator._write_outputs(
        v, [], [], [], None, [], [], set_coverage={})
    out = pd.read_csv(run_file_path(tmp_path, 'same_name_excluded.csv'))
    assert len(out) == 1
    assert out.iloc[0]['disposition'] == 'gated_held'


def test_p2_held_counters_land_in_coverage():
    v = make_validator(
        decisions={('ORN_D', 'dsA', 'dsB'): SNF_HELD})
    v._pairs_for_type('ORN_D', [21], 'ORN_D')
    held = [r for r in v._same_name_excluded
            if r['disposition'] == 'gated_held']
    assert len(held) == 1  # the run() merge maps this to the counter


# ---------------------------------------------------------------------------
# P3 — suspects verification (opt-in)
# ---------------------------------------------------------------------------

def test_p3_default_off():
    assert MappingValidationConfig(
        source_dataset='dsA', target_dataset='dsB').verify_suspects is False


def test_p3_partial_mapper_accessors_guarded():
    v = make_validator(with_accessors=False)
    assert v._suspects_decision('T1') is None
    # no decision -> the pass is a silent no-op
    v._verify_suspects_for_type('T1', [1], {}, None)
    assert not hasattr(v, '_suspects_verification_rows')


def test_p3_rival_with_empty_pool_skipped():
    v = make_validator(
        decisions={('aMe9', 'dsA', 'dsB'): SNF_FIRED})
    v._verify_suspects_for_type('aMe9', [11], {}, SNF_FIRED['same_name_first'])
    # aMe12/aMe6 have no target pools in the fake resolver
    rows = getattr(v, '_suspects_verification_rows', [])
    assert rows == []
    assert any('no target pool' in n for n in v.notes)


def test_p3_rival_verification_rows_isolated_and_annotated():
    # geometry: the source clearly ranks the rival's pool member BELOW
    # the selection's — the rival relation comes out weak, and every row
    # lands in the SEPARATE suspects accumulator.
    src = make_profile(11, {'A': 10, 'B': 8}, {'P': 6})
    sel = make_profile(201, {'A': 10, 'B': 8}, {'P': 6})   # mirror
    rival = make_profile(301, {'Z': 5}, {'W': 4})          # dissimilar
    tgt_vectors = {201: expanded_vector(sel, None),
                   301: expanded_vector(rival, None)}
    stats = prep_target_stats(tgt_vectors)
    scans = {11: scan_source(expanded_vector(src, None), stats)}
    v = make_validator(
        decisions={('aMe9', 'dsA', 'dsB'): SNF_FIRED},
        pools={('aMe12', 'dsB'): [201], ('aMe6', 'dsB'): [301]},
        profiles={11: src})
    v._current_query = 'aMe9'
    v._verify_suspects_for_type(
        'aMe9', [11], scans, SNF_FIRED['same_name_first'])
    rows = getattr(v, '_suspects_verification_rows', [])
    assert rows, 'rival rows must be produced'
    for r in rows:
        assert r['rival_of'] == 'aMe9'
        assert r['disposition'] == 'broad_selection'
        assert 'rival_has_own_clean_pair' in r
    # isolation: the main validation accumulators were never touched
    assert not hasattr(v, '_same_name_excluded') or \
        all(r.get('reason') != 'suspects_verification'
            for r in v._same_name_excluded)
    by_rival = {r['target_type'] for r in rows}
    assert by_rival == {'aMe12', 'aMe6'}
    # the mirror rival (aMe12 vs bodyId 201) verifies; the dissimilar
    # one does not earn verified_strong against 301
    strong = [r for r in rows
              if r['verdict'] == 'verified_strong']
    assert strong and all(r['target_type'] == 'aMe12' for r in strong)
    # the mapper-side populations ride along (interpretation aid)
    assert 'rival_population_source' in rows[0]
    assert 'rival_population_target' in rows[0]


def test_p3_excluded_types_get_their_own_scans():
    """D-A: a held type has no validated pairs — the second hook builds
    its scans and verifies its rivals too."""
    src = make_profile(21, {'A': 10}, {'P': 6})
    rival_tgt = make_profile(301, {'A': 9, 'Q': 1}, {'P': 5})
    tgt_vectors = {301: expanded_vector(rival_tgt, None)}
    stats = prep_target_stats(tgt_vectors)
    v = make_validator(
        decisions={('ORN_D', 'dsA', 'dsB'): SNF_HELD,
                   ('ORN_DA1', 'dsA', 'dsB'):
                   {'status': 'unmapped', 'target_type': None,
                    'target_types': []}},
        pools={('ORN_D', 'dsA'): [21], ('ORN_DA1', 'dsB'): [301]},
        profiles={21: src})
    v._pairs_for_type('ORN_D', [21], 'ORN_D')  # records the held row
    v._verify_suspects_for_excluded(
        target_stats=stats, target_bids=[301])
    rows = getattr(v, '_suspects_verification_rows', [])
    assert rows, 'held-type rivals must be verified'
    assert {r['source_type'] for r in rows} == {'ORN_D'}
    assert all(r['target_type'] == 'ORN_DA1' for r in rows)


def test_p3_csv_only_written_when_enabled(tmp_path):
    v = make_validator()
    v.run_dir = tmp_path
    v._suspects_verification_rows = [
        {'source_type': 'T', 'target_type': 'R', 'verdict': 'unmatched'}]
    MappingValidator._write_outputs(
        v, [], [], [], None, [], [], set_coverage={})
    assert not run_file_path(
        tmp_path, 'suspects_verification.csv').exists()
    v.cfg.verify_suspects = True
    MappingValidator._write_outputs(
        v, [], [], [], None, [], [], set_coverage={})
    assert run_file_path(
        tmp_path, 'suspects_verification.csv').exists()


# ---------------------------------------------------------------------------
# P4 — multivalue accounting
# ---------------------------------------------------------------------------

def test_p4_multivalue_source_accounted_never_split():
    class MVMapper(FakeMapper):
        def is_multivalue_type(self, name, dataset):
            return name == 'A, B'

        def get_mapping_decision(self, t, src, tgt, route_scope=None):
            return {'status': 'unmapped', 'target_type': None,
                    'target_types': []}

    v = make_validator()
    v.mapper = MVMapper({}, with_accessors=False)
    pairs = v._pairs_for_type('A, B', [31], 'A, B')
    assert pairs == []  # kept atomic, fail-closed as before
    rows = getattr(v, '_same_name_excluded', [])
    assert len(rows) == 1
    assert rows[0]['reason'] == 'multivalue_cell'
    assert rows[0]['disposition'] == 'multivalue_cell'


def test_p4_multivalue_target_skipped_and_counted():
    class MVMapper(FakeMapper):
        def is_multivalue_type(self, name, dataset):
            return name == 'X, Y'

        def get_mapping_decision(self, t, src, tgt, route_scope=None):
            if t == 'T1':
                return {'status': 'mapped', 'target_type': 'T1',
                        'target_types': ['X, Y', 'T1']}
            return {'status': 'unmapped', 'target_type': None,
                    'target_types': []}

    v = make_validator(
        decisions={('T1', 'dsA', 'dsB'):
                   {'status': 'mapped', 'target_type': 'T1',
                    'target_types': ['X, Y', 'T1']}},
        pools={('T1', 'dsB'): [201]})
    v.mapper = MVMapper({}, with_accessors=False)
    pairs = v._pairs_for_type('T1', [11], 'T1')
    # the ordinary target survives; the multivalue cell is skipped
    assert [p.target_type for p in pairs] == ['T1']
    assert getattr(v, '_multivalue_target_skips', 0) == 1
    assert any('multi-value target' in n for n in v.notes)


# ---------------------------------------------------------------------------
# backward-decision improvement (fired reverse decision)
# ---------------------------------------------------------------------------

def test_backward_decision_benefits_from_fired_reverse():
    v = make_validator()
    reverse_fired = {
        ('T1', 'dsB', 'dsA'): {
            'status': 'mapped', 'target_type': 's-CPDN3C',
            'target_types': ['s-CPDN3C'],
            'same_name_first': {'fires': True, 'selected': 's-CPDN3C',
                                'rivals': [], 'path': 'conflict',
                                'disposition': 'gated_selection'}}}
    v.mapper = FakeMapper(reverse_fired)
    v._source_type_counts = {'s-CPDN3C': 3}
    v._source_add_counts = {}
    dec = v._backward_decision('T1')
    assert dec['status'] == 'mapped'
    assert dec['mapped'] == 's-CPDN3C'
    assert dec['home_real'] is True


# ---------------------------------------------------------------------------
# source-candidates: RE-AIMED to the out-map expansion route (option 2)
# ---------------------------------------------------------------------------

def test_source_candidates_expansion_route():
    """Option 2: candidates come from the out-map expansion's in-pool
    rows — only morph-qualified rows become candidates, grouped by the
    branch owning the pool; multi-branch sources are marked dup."""
    v = make_validator()
    cand_rows = [
        # qualified hit into branch (q, sA, T1)'s pool -> candidate
        {'source_bodyId': 999, 'source_type': 'sX',
         'target_bodyId': 601, 'target_type': 'T1',
         'rank_union': 0.6, 'jaccard': 0.5,
         'morph_v2_similarity': 0.4, 'morph_bar': 0.143,
         'morph_bar_kind': 'null', 'morph_qualified': True,
         'in_map': True, 'branch_key': ('q', 'sA', 'T1')},
        # unqualified hit -> never a candidate
        {'source_bodyId': 888, 'source_type': 'sY',
         'target_bodyId': 601, 'target_type': 'T1',
         'rank_union': 0.7, 'jaccard': 0.6,
         'morph_v2_similarity': 0.05, 'morph_qualified': False,
         'in_map': True, 'branch_key': ('q', 'sA', 'T1')},
        # same source 999 also reaches branch B's pool -> dup
        {'source_bodyId': 999, 'source_type': 'sX',
         'target_bodyId': 602, 'target_type': 'T2',
         'rank_union': 0.5, 'jaccard': 0.4,
         'morph_v2_similarity': 0.35, 'morph_qualified': True,
         'in_map': True, 'branch_key': ('q', 'sB', 'T2')},
    ]
    v._collect_source_candidates(cand_rows)
    cands = v._source_candidates
    assert set(cands) == {('q', 'sA', 'T1'), ('q', 'sB', 'T2')}
    by = {c['source_bodyId']: c for c in cands[('q', 'sA', 'T1')]}
    assert set(by) == {999}          # unqualified 888 dropped
    assert by[999]['morph_qualified'] is True
    # the ✓ travels with the number it was compared against: the row is
    # recomputable (`morph_v2_similarity >= morph_bar`), which it was not
    # until 2026-09-25
    assert by[999]['morph_bar'] == 0.143
    assert by[999]['morph_bar_kind'] == 'null'
    assert v._source_candidates_multi == {999}
    assert any('distinct out-of-map source(s)' in n for n in v.notes)


def test_source_candidates_csv_export(tmp_path):
    v = make_validator()
    v.run_dir = tmp_path
    v._source_candidates = {
        ('q', 'sA', 'T1'): [
            {'source_bodyId': 999, 'source_type': 'sX',
             'target_bodyId': 601, 'target_type': 'T1',
             'rank_union': 0.5, 'jaccard': 0.4,
             'morph_v2_similarity': 0.4, 'morph_qualified': True}],
    }
    v._source_candidates_multi = set()
    MappingValidator._write_outputs(
        v, [], [], [], None, [], [], set_coverage={})
    out = pd.read_csv(run_file_path(tmp_path, 'source_candidates.csv'))
    assert len(out) == 1
    assert out.iloc[0]['branch_source_type'] == 'sA'
    assert out.iloc[0]['branch_target_type'] == 'T1'
    assert bool(out.iloc[0]['morph_qualified']) is True


def test_expansion_collects_in_pool_candidate_rows():
    """_expand_out_map_sources records the unclaimed source's in-pool
    hits (branch-attributed) alongside the non-in-map expansion rows."""
    from comparison.mapping_validation import (
        expanded_vector, prep_target_stats, scan_source, SOURCE_STATUS_SKIP,
    )
    src = make_profile(71, {'A': 10, 'B': 8}, {'P': 6})
    pool_tgt = make_profile(601, {'A': 9, 'B': 7}, {'P': 5})
    far_tgt = make_profile(701, {'Z': 5}, {'W': 4})
    tgt_vectors = {601: expanded_vector(pool_tgt, None),
                   701: expanded_vector(far_tgt, None)}
    stats = prep_target_stats(tgt_vectors)
    v = make_validator(profiles={71: src})

    class _StubProgress:
        def emit(self, *a, **k):
            pass

    v.progress = _StubProgress()
    out_map_by_type = {('q', 'sX'): [71]}
    # pool owner: 601 belongs to branch (q, sA, T1); 701 is nobody's
    pool_owner = {601: ('q', 'sA', 'T1')}
    rows, cand = v._expand_out_map_sources(
        out_map_by_type, in_map={601}, target_stats=stats,
        target_bids=[601, 701],
        target_id2type={601: 'T1', 701: 'Z1'}, top_k=5,
        pool_owner=pool_owner)
    # the expansion rows exclude the in-map target (unchanged contract)
    assert all(int(r['target_bodyId']) == 701 for r in rows)
    # the in-pool hit is captured as a branch-attributed candidate row
    assert len(cand) == 1
    assert cand[0]['source_bodyId'] == 71
    assert cand[0]['target_bodyId'] == 601
    assert cand[0]['branch_key'] == ('q', 'sA', 'T1')
    assert cand[0]['in_map'] is True


def test_mapping_export_carries_same_name_first():
    v = make_validator()
    pair = TypePair('dsA', 'aMe9', [11], 'dsB', 'aMe9', [201],
                    relationship='suspects', query='aMe9')
    pair.same_name_first = {'selected': 'aMe9', 'rivals': ['aMe12'],
                            'path': 'conflict',
                            'disposition': 'gated_selection'}
    v.pairs = [pair]
    rows = MappingValidator._mapping_export_rows(v)
    assert rows[0]['same_name_first'] is True
    assert rows[0]['same_name_rivals'] == 'aMe12'
    plain = TypePair('dsA', 'T1', [1], 'dsB', 'T1', [2])
    v.pairs = [plain]
    rows2 = MappingValidator._mapping_export_rows(v)
    assert rows2[0]['same_name_first'] is False
    assert rows2[0]['same_name_rivals'] == ''


def test_source_candidates_color_is_category10_brown():
    """User adjustment of D-B12 (2026-09-18): the re-aimed
    source-candidates render in bokeh Category10's brown."""
    from comparison.mapping_validation_visualize import CATEGORY_COLORS
    assert CATEGORY_COLORS['source-candidates'] == '#8c564d'


def test_empty_exports_are_header_only(tmp_path):
    """User 2026-09-18: a run folder must never contain zero-byte CSVs —
    empty exports carry the registry's header line so pd.read_csv yields
    an empty frame instead of raising EmptyDataError."""
    v = make_validator()
    v.run_dir = tmp_path
    MappingValidator._write_outputs(
        v, [], [], [], None, [], [], set_coverage={})
    for name in ('deep_candidates.csv', 'same_name_excluded.csv',
                 'source_candidates.csv', 'out_map_expansion.csv',
                 'examinees.csv', 'pair_summary.csv',
                 'backward_matches.csv'):
        df = pd.read_csv(run_file_path(tmp_path, name))  # must not raise
        assert len(df) == 0, name
    hdr = run_file_path(
        tmp_path, 'same_name_excluded.csv').read_text().strip()
    assert hdr.startswith('query,source_type')


def test_a_published_verdict_travels_with_the_bar_it_was_made_against(tmp_path):
    """#58 proved the rule on pooling, #61 applies it to the two exports that
    were still missing it: a ✓/✗ whose bar lives only in
    `morphology_calibration.json` cannot be recomputed from the row.

    The pairing is asserted on the SCHEMA, because that is where a column can
    silently stop being filled: `morph_qualified` without `morph_bar` beside it
    is the shape of the defect, not any one run's numbers.
    """
    import comparison.mapping_validation as mv
    for name in ('out_map_expansion.csv', 'source_candidates.csv'):
        cols = mv._RUN_CSV_SCHEMAS[name]
        i = cols.index('morph_qualified')
        assert cols[i - 1] == 'morph_bar_kind', (name, cols[max(0, i - 2):i + 1])
        assert cols[i - 2] == 'morph_bar', (name, cols[max(0, i - 2):i + 1])
        assert cols[i - 3] == 'morph_v2_similarity', name
    # and an empty export still writes the header, so a reader of a
    # header-only file sees the columns exist
    mv._write_run_csv(tmp_path, 'out_map_expansion.csv', [])
    head = (tmp_path / 'expansion' / 'out_map_expansion.csv').read_text(
        encoding='utf-8').splitlines()[0]
    assert 'morph_bar,morph_bar_kind,morph_qualified' in head


def test_a_bar_value_never_ships_without_the_rule_that_produced_it():
    """#62's invariant, in the form a future export cannot dodge.

    A row may carry a bar VALUE without a bar NAME only where the name is
    published under another column: `gap_fill_levels.csv` calls it `evidence`
    (its `high`/`medium`/`low` level is derived from that same kind). Measured
    on the 2026-09-24 aggressive and 2026-09-25 family runs, no exported row
    breaks the pairing — this test is what keeps it that way.
    """
    import comparison.mapping_validation as mv
    for name, cols in mv._RUN_CSV_SCHEMAS.items():
        if 'bar_value' not in cols:
            continue
        assert ('bar_kind' in cols or 'evidence' in cols), (
            name, [c for c in cols if c.startswith('bar_') or c == 'evidence'])
    assert 'bar_kind' not in mv._RUN_CSV_SCHEMAS[
        'gap_fill_levels.csv']    # the kind rides on `evidence`, by design
