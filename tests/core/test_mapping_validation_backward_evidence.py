"""Stage 5d: the reciprocal (target -> source) homolog evidence.

Covers ``_plan/plan-tmvev-backward-expansion-evidence.md`` at the seams that
are easy to break silently:

* the grade classifier (high / medium / low by branch-type rank),
* the single-column top-N payload the report hovers,
* the ADVISORY contract -- a reverse label never moves a bin or a level,
* ``run_file_path`` (new layout + legacy flat fallback),
* the reverse column that retires ``categorize_pool_sources``' structurally
  zero ``n_competitors``,
* the report panel and the scene leaf suffix agreeing with the CSV.
"""

from pathlib import Path
import shutil

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolver():
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from comparison import body_id_resolver as m
    return m


def _mv():
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from comparison import mapping_validation as m
    return m


def _scan_df(rows):
    """A reverse scan result in the shape `scan_source` returns.

    A row is ``(bid, rank_union, jaccard, rank_union_rank, jaccard_rank)``
    and may carry ``(shared_type_count, union_type_count)`` after it — the
    two counts :func:`score_one_candidate_fast` always returns."""
    out = []
    for r in rows:
        d = dict(zip(('target_bid', 'rank_union', 'jaccard',
                      'rank_union_rank', 'jaccard_rank'), r[:5]))
        if len(r) > 5:
            d['shared_type_count'], d['union_type_count'] = r[5], r[6]
        out.append(d)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# classify_backward_scan
# ---------------------------------------------------------------------------

def test_branch_source_type_as_rank_union_top1_is_high():
    m = _resolver()
    df = _scan_df([(101, 0.42, 0.30, 1, 1), (999, 0.10, 0.08, 2, 2)])
    out = m.classify_backward_scan(df, branch_pool={101, 102},
                                   id2type={101: 's-CPDN3D'},
                                   branch_source_type='s-CPDN3D')
    assert out['backward_evidence'] == 'high'
    assert out['backward_top1_source_bodyId'] == 101
    assert out['backward_top1_in_branch'] is True
    assert out['backward_top1_source_type'] == 's-CPDN3D'
    assert out['backward_scanned_at'] == 'run'


def test_same_type_top1_outside_the_pool_still_grades_high():
    """The 40165 lesson (r13): pool membership is context, never the
    verdict -- a same-type top-1 grades `high` even when it lives outside
    the claiming branch's refined pool."""
    m = _resolver()
    df = _scan_df([(777, 0.55, 0.40, 1, 1), (101, 0.20, 0.15, 2, 2)])
    out = m.classify_backward_scan(df, branch_pool={101},
                                   id2type={777: 'A', 101: 'A'},
                                   branch_source_type='A')
    assert out['backward_evidence'] == 'high'
    assert out['backward_top1_source_bodyId'] == 777
    assert out['backward_top1_in_branch'] is False
    # the branch's own source still travels in the neighbourhood payload,
    # flagged as the in-branch one, so the hover answers "then where is ours?"
    recs = {r.split('|')[1]: r.split('|') for r in
            out['backward_topN'].split(';')}
    assert set(recs) == {'101', '777'}
    assert recs['101'][-1] == '1' and recs['777'][-1] == '0'


def test_there_is_no_score_bar_any_more():
    """The old matched_ru_min=0.1 manufactured `none` from a same-type
    top-1 with a weak rank_union; ranks alone decide now."""
    m = _resolver()
    df = _scan_df([(101, -0.014, 0.30, 1, 1)])
    out = m.classify_backward_scan(df, branch_pool={101},
                                   id2type={101: 'A'},
                                   branch_source_type='A')
    assert out['backward_evidence'] == 'high'
    assert out['backward_rank_union'] == pytest.approx(-0.014)


def test_medium_when_the_branch_type_ranks_second():
    m = _resolver()
    df = _scan_df([(999, 0.55, 0.40, 1, 1), (101, 0.20, 0.15, 2, 2),
                   (998, 0.10, 0.10, 3, 3), (997, 0.05, 0.05, 4, 4)])
    out = m.classify_backward_scan(df, branch_pool={101},
                                   id2type={101: 'A'},
                                   branch_source_type='A')
    assert out['backward_evidence'] == 'medium'
    # a jaccard top-1 lifts the grade to high even when rank_union does not
    df2 = _scan_df([(999, 0.55, 0.10, 1, 3), (998, 0.50, 0.20, 2, 2),
                    (101, 0.20, 0.45, 4, 1)])
    out2 = m.classify_backward_scan(df2, branch_pool={101},
                                    id2type={101: 'A'},
                                    branch_source_type='A')
    assert out2['backward_evidence'] == 'high'


def test_low_is_a_graded_negative():
    m = _resolver()
    df = _scan_df([(999, 0.55, 0.40, 1, 1), (998, 0.20, 0.15, 2, 2),
                   (997, 0.10, 0.10, 3, 3)])
    out = m.classify_backward_scan(df, branch_pool={101},
                                   id2type={999: 'X', 998: 'Y',
                                            997: 'Z'},
                                   branch_source_type='A')
    assert out['backward_evidence'] == 'low'
    assert out['backward_scanned_at'] == 'run'


@pytest.mark.parametrize('df', [None, pd.DataFrame()])
def test_nothing_ranked_is_silence_not_contradiction(df):
    m = _resolver()
    out = m.classify_backward_scan(df, branch_pool={101})
    assert out['backward_evidence'] == 'low'
    assert out['backward_top1_source_bodyId'] is None
    assert out['backward_topN'] == ''


def test_spatial_caliber_is_context_never_a_verdict():
    """A top-1 far smaller than the branch pool's best is flagged, but the
    caliber no longer overwrites the grade (user 2026-09-19: advisory
    hints, not gates)."""
    m = _resolver()
    df = _scan_df([(101, 0.42, 0.30, 1, 1)])
    out = m.classify_backward_scan(df, branch_pool={101},
                                   id2type={101: 'A'},
                                   branch_source_type='A',
                                   sizes={101: 4}, pool_best_size=1000.0,
                                   min_size_ratio=0.1)
    assert out['backward_evidence'] == 'high'
    assert out['backward_size_filtered'] is True
    assert out['backward_size_ratio'] == pytest.approx(0.004)


def test_out_of_branch_competitors_count_over_the_whole_universe():
    m = _resolver()
    df = _scan_df([(1, 0.9, 0.9, 1, 1), (2, 0.8, 0.8, 2, 3),
                   (3, 0.7, 0.7, 3, 4), (101, 0.6, 0.6, 4, 2)])
    out = m.classify_backward_scan(df, branch_pool={101},
                                   id2type={101: 'A'},
                                   branch_source_type='A')
    assert out['backward_evidence'] == 'medium'
    assert out['backward_top1_source_bodyId'] == 1
    # the branch's own source is the 4th-ranked column entry, with three
    # out-of-branch rivals above it
    assert out['backward_n_out_of_branch'] == 3


# ---------------------------------------------------------------------------
# the top-N payload + the blank state
# ---------------------------------------------------------------------------

def test_serialize_topN_carries_membership_flags_in_rank_order():
    m = _resolver()
    df = _scan_df([(500, 0.2, 0.1, 2, 2), (101, 0.6, 0.5, 1, 1)])
    raw = m.serialize_backward_topN(df, id2type={101: 'A', 500: 'B'},
                                    top_n=5, branch_pool={101})
    recs = raw.split(';')
    assert [r.split('|')[0] for r in recs] == ['1', '2']
    assert recs[0].split('|')[1] == '101'
    assert recs[0].split('|')[-1] == '1'      # in the claiming branch
    assert recs[1].split('|')[-1] == '0'      # elsewhere
    assert len(recs[1].split('|')) == 6


def test_serialize_topN_respects_the_cap_and_empty_inputs():
    m = _resolver()
    df = _scan_df([(i, 0.5, 0.4, i, i) for i in range(1, 8)])
    assert len(m.serialize_backward_topN(df, top_n=3).split(';')) == 3
    assert m.serialize_backward_topN(df, top_n=0) == ''
    assert m.serialize_backward_topN(None, top_n=5) == ''


def test_blank_fields_name_the_reason_nothing_was_scored():
    m = _resolver()
    for reason in ('disabled', 'cap', 'run', 'no_profile', 'error'):
        blank = m.blank_backward_fields(reason)
        assert set(blank) == set(m.BACKWARD_COLUMNS)
        assert blank['backward_evidence'] == 'not-checked'
        assert blank['backward_scanned_at'] == reason


# ---------------------------------------------------------------------------
# the published evidence base (plan §17b)
# ---------------------------------------------------------------------------

def test_the_verdict_publishes_what_the_score_was_computed_over():
    """rank_union ranks the UNION and scores an absent partner 0.0, so the
    number itself cannot say whether it rested on 1 shared type or 20. The
    scorer counts both; the exports now carry them."""
    m = _resolver()
    out = m.classify_backward_scan(
        _scan_df([(101, 0.42, 0.30, 1, 1, 2, 9)]), branch_pool={101})
    assert m.THIN_SHARED_TYPE_COUNT == 3
    assert out['backward_shared_type_count'] == 2
    assert out['backward_union_type_count'] == 9
    assert out['backward_thin_evidence'] is True
    fat = m.classify_backward_scan(
        _scan_df([(101, 0.42, 0.30, 1, 1, 4, 9)]), branch_pool={101})
    assert fat['backward_thin_evidence'] is False


def test_never_scored_reads_as_no_count_and_no_warning():
    m = _resolver()
    for out in (m.classify_backward_scan(None),
                m.classify_backward_scan(_scan_df([])),
                m.blank_backward_fields('cap')):
        assert out['backward_shared_type_count'] is None
        assert out['backward_thin_evidence'] is False


def test_thin_evidence_never_moves_a_verdict_or_a_fill_level():
    """Advisory only: the two rows below score identically and differ in
    nothing but the three published-evidence fields."""
    m = _resolver()
    kw = dict(branch_pool={101}, branch_source_type='A',
              id2type={101: 'A'})
    thin = m.classify_backward_scan(
        _scan_df([(101, 0.42, 0.30, 1, 1, 1, 3)]), **kw)
    thick = m.classify_backward_scan(
        _scan_df([(101, 0.42, 0.30, 1, 1, 20, 40)]), **kw)
    assert thin['backward_evidence'] == thick['backward_evidence'] == 'high'
    assert {k for k in thin if thin[k] != thick[k]} == {
        'backward_shared_type_count', 'backward_union_type_count',
        'backward_thin_evidence'}
    mv = _mv()
    rows, counts = mv.build_gap_fill_levels([
        {'dedup_category': 'family', 'target_bodyId': 11, 'target_type': 'T1',
         'backward_evidence': 'high', 'backward_thin_evidence': True},
        {'dedup_category': 'family', 'target_bodyId': 12, 'target_type': 'T1',
         'backward_evidence': 'high', 'backward_thin_evidence': False},
    ], [])
    assert [r['level'] for r in rows] == ['type_gated', 'type_gated']
    assert counts['type_gated'] == 2


def test_the_evidence_columns_reach_every_backward_export(tmp_path):
    """The header registry and the row dicts must agree: an empty
    `backward_matches.csv` still lists them, and the dedup rollup carries
    the two that matter there (not the per-scan ranks)."""
    m = _resolver()
    mv = _mv()
    new = ['backward_shared_type_count', 'backward_union_type_count',
           'backward_thin_evidence']
    assert new[0] in m.BACKWARD_COLUMNS and new[2] in m.BACKWARD_COLUMNS
    for name in ('backward_matches.csv', 'examinees.csv',
                 'deep_candidates.csv', 'family_candidates.csv',
                 'relatives.csv'):
        assert all(c in mv._RUN_CSV_SCHEMAS[name] for c in new), name
    dedup = mv._RUN_CSV_SCHEMAS['gap_fill_dedup.csv']
    assert 'backward_shared_type_count' in dedup
    assert 'backward_thin_evidence' in dedup
    assert 'backward_union_type_count' not in dedup
    assert mv._RUN_CSV_SCHEMAS['gap_fill_levels.csv'][-1] == \
        'backward_evidence'
    header = tmp_path / 'backward_matches.csv'
    mv._write_csv(header, [],
                  columns=mv._RUN_CSV_SCHEMAS['backward_matches.csv'])
    text = header.read_text(encoding='utf-8')
    assert all(c in text for c in new)
    assert len(text.strip().split(',')) == 7 + len(m.BACKWARD_COLUMNS)


def test_dedup_rollup_carries_the_evidence_base():
    mv = _mv()
    v = mv.MappingValidator.__new__(mv.MappingValidator)
    row = v._expansion_row(('q', 'A', 'X'), 11, 'X1', 'family')
    row.update({'backward_evidence': 'high',
                'backward_shared_type_count': 2,
                'backward_union_type_count': 9,
                'backward_thin_evidence': True})
    d = {r['target_bodyId']: r
         for r in v._build_dedup_rows([], [], [row], [])}
    assert d[11]['backward_shared_type_count'] == 2
    assert d[11]['backward_thin_evidence'] is True


# ---------------------------------------------------------------------------
# the advisory contract
# ---------------------------------------------------------------------------

def test_reverse_label_rides_evidence_not_the_level_ladder():
    """D7: the level ladder stays the morph-bar strength ordering, so the
    reverse grade lands in `evidence` and `backward_evidence`, and a
    `high` label does not promote a type_gated family row."""
    mv = _mv()
    dedup = [
        {'dedup_category': 'family', 'target_bodyId': 11,
         'target_type': 'T1', 'backward_evidence': 'high'},
        {'dedup_category': 'family', 'target_bodyId': 12,
         'target_type': 'T1', 'backward_evidence': 'not-checked'},
        {'dedup_category': 'relative', 'target_bodyId': 13,
         'target_type': 'T2', 'backward_evidence': 'medium'},
    ]
    rows, counts = mv.build_gap_fill_levels(dedup, [])
    by_bid = {r['target_bodyId']: r for r in rows}
    assert by_bid[11]['level'] == 'type_gated'
    assert by_bid[11]['evidence'] == 'backward_high'
    assert by_bid[12]['evidence'] == 'type_membership'
    assert by_bid[13]['level'] == 'advice'
    assert by_bid[13]['evidence'] == 'backward_medium'
    assert by_bid[13]['backward_evidence'] == 'medium'
    assert counts['type_gated'] == 2 and counts['advice'] == 1


def test_backward_pass_is_off_by_default_and_skips_without_a_stash():
    mv = _mv()
    v = mv.MappingValidator.__new__(mv.MappingValidator)
    v.cfg = mv.MappingValidationConfig(
        source_dataset='dsA', target_dataset='dsB', query_types=['q'])
    v.notes = []
    v.log = lambda *a, **k: None
    assert v._backward_expansion_pass() is None       # default: disabled
    v.cfg.backward_evidence_enabled = True
    assert v._backward_expansion_pass() is None       # nothing stashed
    assert v.cfg.skip_backward_pass is False
    v.cfg.skip_backward_pass = True
    assert v._backward_expansion_pass() is None


def _backward_stub(mv, fam):
    """A validator wired just enough to reach the scan loop of stage 5d."""
    from types import SimpleNamespace
    v = mv.MappingValidator.__new__(mv.MappingValidator)
    v.cfg = mv.MappingValidationConfig(
        source_dataset='dsA', target_dataset='dsB', query_types=['q'])
    v.cfg.backward_evidence_enabled = True
    v.cfg.skip_profile_build = True
    v.log = lambda *a, **k: None
    v.progress = SimpleNamespace(emit=lambda *a, **k: None)
    v.pairs = []
    v.mapper = None
    v.profiler = None
    v._source_sizes = {}
    v._cat_evidence = []
    v._cat_pool_detail = []
    v._cat_family_rows = [fam]
    v._cat_relative_rows = []
    v._cat_per_pair_res = {}
    v._rebuild_dedup_rows = lambda: 'rebuilt'
    return v


def test_unavailable_source_universe_says_error_not_disabled(monkeypatch):
    """An ON run whose vectors fail must NOT keep the pass-off default
    `disabled` — 'we tried and could not' and 'this run never asked' read
    completely differently in the report."""
    mv = _mv()
    v = mv.MappingValidator.__new__(mv.MappingValidator)
    fam = v._expansion_row(('q', 'A', 'X'), 2, 'X', 'family')
    v = _backward_stub(mv, fam)

    def boom(*a, **k):
        raise RuntimeError('no vectors')

    monkeypatch.setattr(mv, 'build_target_vectors', boom)
    assert v._backward_expansion_pass() == 'rebuilt'
    assert fam['backward_scanned_at'] == 'error'
    assert fam['backward_evidence'] == 'not-checked'


def test_member_without_a_profile_is_unscanned_not_none(monkeypatch):
    """`none` claims "we scanned and nothing ranked"; a neuron with no
    profile was never scanned, so it must not land in that bucket."""
    from types import SimpleNamespace
    mv = _mv()
    v = mv.MappingValidator.__new__(mv.MappingValidator)
    fam = v._expansion_row(('q', 'A', 'X'), 2, 'X', 'family')
    v = _backward_stub(mv, fam)
    v.profiler = SimpleNamespace(get_profile=lambda bid, ds: None)
    v._bodyid_types = lambda ids, ds: {}
    monkeypatch.setattr(mv, 'build_target_vectors',
                        lambda *a, **k: {'vectors': {}})
    monkeypatch.setattr(mv, 'prep_target_stats', lambda vectors: {101: {}})
    assert v._backward_expansion_pass() == 'rebuilt'
    assert fam['backward_evidence'] == 'not-checked'
    assert fam['backward_scanned_at'] == 'no_profile'


def test_a_broken_scan_is_error_and_an_empty_one_is_none(monkeypatch):
    """Three different claims: `no_profile` = not scannable, `error` = the
    scan itself blew up, `none` = scanned and nothing ranked. Only the last
    one is a negative result."""
    from types import SimpleNamespace
    import pandas as pd
    mv = _mv()
    v = mv.MappingValidator.__new__(mv.MappingValidator)
    broke = v._expansion_row(('q', 'A', 'X'), 2, 'X', 'family')
    empty = v._expansion_row(('q', 'A', 'X'), 3, 'X', 'family')
    v = _backward_stub(mv, broke)
    v._cat_family_rows = [broke, empty]
    v.profiler = SimpleNamespace(
        get_profile=lambda bid, ds: SimpleNamespace(
            bid=bid, connectivity_status=SimpleNamespace(name='OK')))
    v._bodyid_types = lambda ids, ds: {}
    monkeypatch.setattr(mv, 'build_target_vectors',
                        lambda *a, **k: {'vectors': {}})
    monkeypatch.setattr(mv, 'prep_target_stats', lambda vectors: {101: {}})
    monkeypatch.setattr(mv, 'expanded_vector', lambda sp, mapper: sp)

    def _scan(sp, stats, bids):
        if sp.bid == 2:
            raise RuntimeError('scan blew up')
        return pd.DataFrame(columns=['target_bid'])

    monkeypatch.setattr(mv, 'scan_source', _scan)
    assert v._backward_expansion_pass() == 'rebuilt'
    assert (broke['backward_evidence'],
            broke['backward_scanned_at']) == ('not-checked', 'error')
    assert (empty['backward_evidence'],
            empty['backward_scanned_at']) == ('low', 'run')
    # both were ATTEMPTED, neither is a cap cut
    assert v._backward_counters['distinct_scanned'] == 1


def test_the_budget_prefers_the_fill_bins_and_skips_mapped_controls(
        monkeypatch):
    """A 1-scan budget must buy a bin label, not an unmatched pool row:
    unmatched pool members only follow the bins, and matched / verified /
    borderline pool members are NOT scanned at all — they are already
    mapped, and the symmetric forward score is their evidence
    (user 2026-09-19)."""
    from types import SimpleNamespace
    mv = _mv()
    v = mv.MappingValidator.__new__(mv.MappingValidator)
    fam = v._expansion_row(('q', 'A', 'X'), 900, 'X', 'family')
    v = _backward_stub(mv, fam)
    v.cfg.backward_max_neurons = 1
    v._cat_pool_detail = [{'target_bodyId': 5, 'target_type': 'X',
                           'category': 'matched'},
                          {'target_bodyId': 7, 'target_type': 'X',
                           'category': 'unmatched'}]
    v.profiler = SimpleNamespace(get_profile=lambda bid, ds: None)
    v._bodyid_types = lambda ids, ds: {}
    monkeypatch.setattr(mv, 'build_target_vectors',
                        lambda *a, **k: {'vectors': {}})
    monkeypatch.setattr(mv, 'prep_target_stats', lambda vectors: {101: {}})
    v._backward_expansion_pass()
    assert fam['backward_scanned_at'] == 'no_profile'   # got the one slot
    assert v._backward_counters['beyond_cap'] == 1      # the unmatched one
    # the matched control was never even ELIGIBLE (beyond_cap counts only
    # the unmatched member); the stub builds no profiles, so nothing is
    # actually scanned
    assert v._backward_counters['distinct_scanned'] == 0


def test_the_scorer_is_exactly_symmetric_under_transpose():
    """plan §12's transpose test.  rank_union / jaccard / cosine compare two
    partner-type vectors, so scoring A-against-B must equal scoring
    B-against-A — if it ever differs, a forward and a reverse pass are being
    fed DIFFERENT vectors, which is a vector-construction bug, not a
    directional property.  (Verified on real data too: rescoring every r9 +
    r11 pair from the cache parquet reproduced the stored forward rank_union
    on 272/272 pairs.)"""
    mv = _mv()
    from comparison.body_id_resolver import _SideStats, score_one_candidate_fast
    cases = [
        ({'A': 10.0, 'B': 5.0, 'C': 1.0}, {'A': 8.0, 'D': 2.0, 'E': 1.0}),
        ({'A': 10.0}, {'A': 1.0, 'B': 2.0, 'C': 3.0, 'D': 4.0}),
        ({f't{i}': float(i) for i in range(1, 26)},
         {f't{i}': float(26 - i) for i in range(1, 26)}),
        ({'X': 3.0, 'Y': 3.0, 'Z': 1.0}, {'X': 3.0, 'Y': 3.0, 'W': 9.0}),
    ]
    for va, vb in cases:
        a, b = _SideStats(va), _SideStats(vb)
        assert score_one_candidate_fast(a, b) == score_one_candidate_fast(b, a)


def test_run_csv_keeps_60_bit_bodyids_exact(tmp_path):
    """A FAFB bodyId in a column that also has blanks must not round-trip
    through float64: `...74019` once exported as `...74048`, which is a
    different (non-existent) neuron. Covers int, numpy int and digit-string
    inputs, since any of the three can reach a writer."""
    import numpy as np
    mv = _mv()
    bid = 720575940623474019
    path = tmp_path / 'x.csv'
    mv._write_csv(path, [
        {'target_bodyId': 11, 'backward_top1_source_bodyId': bid},
        {'target_bodyId': 12, 'backward_top1_source_bodyId': None},
        {'target_bodyId': 13, 'backward_top1_source_bodyId': bid},
        {'target_bodyId': 14, 'backward_top1_source_bodyId': str(bid)},
        {'target_bodyId': 15,
         'backward_top1_source_bodyId': np.int64(bid)},
    ])
    text = path.read_text(encoding='utf-8')
    assert str(bid) in text and 'e+' not in text.lower()
    back = pd.read_csv(path, dtype=str)
    assert list(back['backward_top1_source_bodyId'].fillna('')) == [
        str(bid), '', str(bid), str(bid), str(bid)]


def test_expansion_row_defaults_to_not_checked_so_headers_stay_stable():
    """A run without the pass must emit byte-stable `backward_*` columns."""
    mv = _mv()
    v = mv.MappingValidator.__new__(mv.MappingValidator)
    v.pairs = []
    row = v._expansion_row(('q', 'A', 'X'), 2, 'X', 'family')
    assert row['backward_evidence'] == 'not-checked'
    assert row['backward_scanned_at'] == 'disabled'
    assert row['backward_topN'] == ''
    assert row['category'] == 'family'


# ---------------------------------------------------------------------------
# run-folder layout
# ---------------------------------------------------------------------------

def test_run_file_path_uses_the_registry_and_creates_parents(tmp_path):
    mv = _mv()
    assert mv.RUN_FILE_LAYOUT['backward_matches.csv'] == 'expansion'
    assert mv.RUN_FILE_LAYOUT['report.html'] == ''
    p = mv.run_file_path(tmp_path, 'gap_fill_dedup.csv',
                         create_parent=True)
    assert p == tmp_path / 'gap_fill' / 'gap_fill_dedup.csv'
    assert p.parent.is_dir()
    # a meta deliverable stays at the root
    assert mv.run_file_path(tmp_path, 'parameters.json') == \
        tmp_path / 'parameters.json'


def test_run_file_path_falls_back_to_a_legacy_flat_folder(tmp_path):
    """Folders written before the layout change must still resolve."""
    mv = _mv()
    (tmp_path / 'examinees.csv').write_text('query\nq1\n',
                                            encoding='utf-8')
    assert mv.run_file_path(tmp_path, 'examinees.csv') == \
        tmp_path / 'examinees.csv'


def test_every_registered_run_csv_has_a_schema(tmp_path):
    """A registered file with no schema writes a bare-newline file when
    empty (`columns=None`) — exactly what suspects_verification.csv did on
    the r13 real-data run (--verify-suspects, zero rivals to verify)."""
    mv = _mv()
    for name in mv.RUN_FILE_LAYOUT:
        if name.endswith('.csv'):
            assert name in mv._RUN_CSV_SCHEMAS, name
    # the composed suspects ledger: ordinary verdict rows + the rival
    # provenance columns the suspects pass appends
    sus = mv._RUN_CSV_SCHEMAS['suspects_verification.csv']
    base = mv._RUN_CSV_SCHEMAS['validation_results.csv']
    assert sus[:len(base)] == base
    assert sus[len(base):] == [
        'rival_of', 'disposition', 'rival_has_own_clean_pair',
        'rival_reverse_target', 'rival_votes',
        'rival_population_source', 'rival_population_target']
    # the empty export is header-only and re-reads as an empty frame
    mv._write_run_csv(tmp_path, 'suspects_verification.csv', [])
    path = tmp_path / 'mapping' / 'suspects_verification.csv'
    assert path.read_text(encoding='utf-8').startswith('query,')
    back = pd.read_csv(path)
    assert back.empty
    assert list(back.columns) == sus


def test_report_reads_the_new_layout_and_the_legacy_one(tmp_path):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from comparison.mapping_validation_report import (
        _run_file, _run_file_rel, collect_run_data, _reciprocal_tab)

    rows = [{'query': 'q', 'branch_source_type': 'A', 'branch_target_type':
             'X', 'member_bodyId': 11, 'member_type': 'X1',
             'member_category': 'family', 'scan_role': 'family',
             'backward_evidence': 'high',
             'backward_top1_source_bodyId': 101,
             'backward_top1_source_type': 'A',
             'backward_top1_in_branch': True, 'backward_rank_union': 0.42,
             'backward_jaccard': 0.3, 'backward_rank_union_rank': 1,
             'backward_jaccard_rank': 1, 'backward_n_out_of_branch': 0,
             'backward_size_ratio': None, 'backward_size_filtered': False,
             'backward_shared_type_count': 2, 'backward_union_type_count': 9,
             'backward_thin_evidence': True,
             'backward_topN': '1|101|A|0.4200|0.3000|1',
             'backward_scanned_at': 'run'}]
    import json
    (tmp_path / 'parameters.json').write_text(
        json.dumps({'backward_evidence_enabled': True, 'backward_top_n': 5,
                    'backward_max_neurons': 300,
                    'backward_per_branch_cap': 40}),
        encoding='utf-8')
    (tmp_path / 'set_coverage.json').write_text(json.dumps(
        {'mcns': {'backward_evidence': {
            'matched': 1, 'foreign': 0, 'none': 0, 'unscanned': 0,
            'distinct_scanned': 1, 'beyond_cap': 0}}}), encoding='utf-8')
    exp = tmp_path / 'expansion'
    exp.mkdir()
    pd.DataFrame(rows).to_csv(exp / 'backward_matches.csv', index=False,
                              encoding='utf-8')
    assert _run_file(tmp_path, 'backward_matches.csv') == \
        exp / 'backward_matches.csv'
    assert _run_file_rel(tmp_path, 'backward_matches.csv') == \
        'expansion/backward_matches.csv'

    d = collect_run_data(tmp_path)
    assert [r['member_bodyId'] for r in d['backward_rows']] == ['11']
    assert d['backward_bins']['family'] == {
        'high': 1, 'medium': 0, 'low': 0, 'not-checked': 0,
        'neurons': 1}
    assert d['backward_counters']['beyond_cap'] == 0
    html = _reciprocal_tab(d)
    assert 'Per-neuron evidence, grouped by type' in html
    assert 'family · X1 — 1 neuron' in html          # grouped by type
    assert 'top-N reverse hits' in html              # the hover payload
    assert "class='bev bev-high'>high" in html
    # the evidence base of that score, and its thin marker
    assert '2/9' in html and "class='bev bev-thin'>thin" in html
    assert 'shared/union types' in html              # the column header
    # the artifact index lists the folder-qualified path
    from comparison import mapping_validation_report as rep
    assert 'expansion/backward_matches.csv' in dict(rep.ARTIFACT_LINES)

    # legacy flat folder: same panel, found one level up
    flat = tmp_path.parent / 'flat'
    flat.mkdir()
    shutil.copy(tmp_path / 'parameters.json', flat / 'parameters.json')
    shutil.copy(tmp_path / 'set_coverage.json', flat / 'set_coverage.json')
    (exp / 'backward_matches.csv').rename(flat / 'backward_matches.csv')
    assert _run_file(flat, 'backward_matches.csv') == \
        flat / 'backward_matches.csv'
    d_flat = collect_run_data(flat)
    assert d_flat['backward_bins']['family']['high'] == 1
    assert 'family · X1 — 1 neuron' in _reciprocal_tab(d_flat)


# ---------------------------------------------------------------------------
# P4: the reverse column retires the structurally-zero competitor count
# ---------------------------------------------------------------------------

def test_forward_only_columns_cannot_see_competitors():
    mv = _mv()
    df = _scan_df([(201, 0.5, 0.4, 1, 1)])
    statuses, detail = mv.categorize_pool_sources(
        {101: df}, target_pool={201}, source_pool={101})
    assert statuses[101] == 'source-matched'
    assert detail[0]['n_competitors'] == 0     # structurally, not by merit


def test_reverse_column_supplies_the_out_of_branch_competitors():
    mv = _mv()
    df = _scan_df([(201, 0.5, 0.4, 1, 1)])
    reverse = {201: [{'source': 900, 'ru': 0.9, 'jac': 0.9,
                      'in_pool': False},
                     {'source': 901, 'ru': 0.8, 'jac': 0.8,
                      'in_pool': False},
                     {'source': 902, 'ru': 0.7, 'jac': 0.7,
                      'in_pool': False},
                     {'source': 101, 'ru': 0.5, 'jac': 0.4,
                      'in_pool': True}]}
    statuses, detail = mv.categorize_pool_sources(
        {101: df}, target_pool={201}, source_pool={101},
        invader_max=2, reverse_columns=reverse)
    assert detail[0]['n_competitors'] == 3
    assert statuses[101] == 'source-unmatched'
    # the reverse ranking replaces the branch-blind column, so the source is
    # now read at its true universe position
    assert detail[0]['col_rank'] == 4


# ---------------------------------------------------------------------------
# the scene suffix agrees with the CSV vocabulary
# ---------------------------------------------------------------------------

def test_scene_suffixes_cover_every_scanned_verdict_only():
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from comparison.mapping_validation_visualize import _BACKWARD_LEAF_TAG
    m = _resolver()

    scanned = {'high', 'medium', 'low'}
    assert set(_BACKWARD_LEAF_TAG) == scanned
    assert set(scanned) <= set(m.BACKWARD_EVIDENCE_VALUES)
    # an unchecked member keeps a bare leaf: no suffix may imply a negative
    assert 'not-checked' not in _BACKWARD_LEAF_TAG
    assert all(v.startswith('· ') for v in _BACKWARD_LEAF_TAG.values())


# ---------------------------------------------------------------------------
# the advisory contract, proven on the mechanism
# ---------------------------------------------------------------------------

def _no_bev(row):
    """A row with every `backward_*` key removed — what a pass-off run has."""
    return {k: v for k, v in row.items()
            if not str(k).startswith('backward_')}


def _wire_scan(mv, monkeypatch, v, rows):
    """Give the stage-5d stub a profiler and a fixed reverse scan."""
    from types import SimpleNamespace
    v.profiler = SimpleNamespace(
        get_profile=lambda bid, ds: SimpleNamespace(
            bid=bid, connectivity_status=SimpleNamespace(name='OK')))
    v._bodyid_types = lambda ids, ds: {i: 'A' for i in ids}
    monkeypatch.setattr(mv, 'build_target_vectors',
                        lambda *a, **k: {'vectors': {}})
    monkeypatch.setattr(mv, 'prep_target_stats', lambda vectors: {101: {}})
    monkeypatch.setattr(mv, 'expanded_vector', lambda sp, mapper: sp)
    monkeypatch.setattr(mv, 'scan_source',
                        lambda sp, stats, bids: _scan_df(rows))


def test_the_pass_writes_nothing_but_the_backward_columns(monkeypatch):
    """The plan's central promise, checked on the mechanism rather than in
    prose: stage 5d may only ADD `backward_*` keys.  A label that reaches a
    `category` or a `counts_toward_*` flag would be a gating bug, and so
    would a rollup that changes shape when rebuilt."""
    mv = _mv()
    probe = mv.MappingValidator.__new__(mv.MappingValidator)
    fam = probe._expansion_row(('q', 'A', 'X'), 2, 'X', 'family')
    fam['counts_toward_family_fill'] = True
    v = _backward_stub(mv, fam)
    del v._rebuild_dedup_rows                      # the REAL rollup
    from types import SimpleNamespace
    v.pairs = [SimpleNamespace(query='q', source_type='A', target_type='X',
                               source_pool=[101])]
    before_row = _no_bev(fam)
    before_dedup = v._build_dedup_rows(v._cat_evidence, v._cat_pool_detail,
                                       v._cat_family_rows,
                                       v._cat_relative_rows)
    _wire_scan(mv, monkeypatch, v, [(101, 0.5, 0.4, 1, 1)])
    rebuilt = v._backward_expansion_pass()

    assert fam['backward_evidence'] == 'high'   # the pass DID run
    assert _no_bev(fam) == before_row
    assert fam['category'] == 'family'
    assert fam['counts_toward_family_fill'] is True

    def ident(r):
        return (str(r['target_bodyId']), r['dedup_category'])

    assert len(rebuilt) == len(before_dedup) >= 1
    assert sorted(map(ident, rebuilt)) == sorted(map(ident, before_dedup))
    now = {ident(r): _no_bev(r) for r in rebuilt}
    for was in before_dedup:
        assert now[ident(was)] == _no_bev(was)
    # D4 mechanism 1: re-calling the rollup is deterministic, so the pass
    # does not have to run once per label to converge
    assert rebuilt == v._rebuild_dedup_rows()


def test_stripping_the_reverse_labels_leaves_the_fill_ladder_untouched():
    """The advisory proof at the layered report: an ON run and an OFF run
    agree on `level` and on every count; only a family / relative row's
    `evidence` string may differ (plan D7)."""
    mv = _mv()
    dedup = [
        {'dedup_category': 'candidates', 'target_bodyId': 13,
         'target_type': 'T3', 'backward_evidence': 'low'},
        {'dedup_category': 'family', 'target_bodyId': 11,
         'target_type': 'T1', 'backward_evidence': 'high'},
        {'dedup_category': 'relative', 'target_bodyId': 12,
         'target_type': 'T2', 'backward_evidence': 'medium'},
    ]
    on_rows, on_counts = mv.build_gap_fill_levels(
        [dict(r) for r in dedup], [])
    off_rows, off_counts = mv.build_gap_fill_levels(
        [_no_bev(dict(r)) for r in dedup], [])

    def by_bid(rows):
        return {r['target_bodyId']: r for r in rows}

    on, off = by_bid(on_rows), by_bid(off_rows)
    assert on_counts == off_counts
    assert [r['level'] for r in on_rows] == [r['level'] for r in off_rows]
    for bid, lab in ((11, 'high'), (12, 'medium'), (13, 'low')):
        assert on[bid]['level'] == off[bid]['level']
        assert on[bid]['backward_evidence'] == lab
        assert off[bid]['backward_evidence'] == ''
    # the reverse fact lands in `evidence`, and only there
    assert on[11]['evidence'] == 'backward_high'
    assert off[11]['evidence'] == 'type_membership'
    assert on[12]['evidence'] == 'backward_medium'
    assert off[12]['evidence'] == 'candidate_type_mate'
    assert on[13]['evidence'] == off[13]['evidence']


def test_no_morphometry_rides_the_reverse_columns():
    """D5: the reverse pass is connectivity-only, so no writer, schema or
    verdict may introduce a morph key.  Checking the vocabulary plus the
    classifier's output covers every path one could arrive by."""
    m = _resolver()
    assert not [c for c in m.BACKWARD_COLUMNS if 'morph' in str(c)]
    out = m.classify_backward_scan(_scan_df([(101, 0.5, 0.4, 1, 1)]),
                                   branch_pool={101})
    assert set(out) == set(m.BACKWARD_COLUMNS)


def test_the_topN_payload_round_trips_through_the_report_hover():
    """`backward_topN` is written by the resolver and re-parsed by the
    report's hover; only both ends together pin the record format."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from comparison.mapping_validation_report import _topn_hover
    m = _resolver()
    df = _scan_df([(500, 0.2, 0.1, 2, 2), (101, 0.6, 0.5, 1, 1)])
    raw = m.serialize_backward_topN(df, id2type={101: 's-CPDN3D',
                                                 500: 'DP01'},
                                    top_n=5, branch_pool={101})
    html = _topn_hover(raw)
    assert '101' in html and 's-CPDN3D' in html and '0.6000' in html
    assert '500' in html and 'DP01' in html and '0.2000' in html
    assert html.count('>this branch<') == 1
    assert html.count('>elsewhere<') == 1
    assert _topn_hover('') == '' and _topn_hover(None) == ''


# ---------------------------------------------------------------------------
# the same sentence in both sinks, and paths that exist
# ---------------------------------------------------------------------------

def _bev_row(bid, mtype, cat, evidence, top1):
    return {
        'query': 'q', 'branch_source_type': 'A', 'branch_target_type': mtype,
        'member_bodyId': bid, 'member_type': mtype, 'member_category': cat,
        'scan_role': cat, 'backward_evidence': evidence,
        'backward_top1_source_bodyId': top1,
        'backward_top1_source_type': 'A',
        'backward_top1_in_branch': evidence == 'high',
        'backward_rank_union': 0.5, 'backward_jaccard': 0.4,
        'backward_shared_type_count': 12 if evidence == 'high' else 2,
        'backward_union_type_count': 40,
        'backward_thin_evidence': evidence != 'high',
        'backward_topN': f'1|{top1}|A|0.5000|0.4000|1',
        'backward_scanned_at': 'run',
    }


def _bev_run(run_dir, nested=True):
    """The smallest run folder a stage-5d run leaves behind: one reciprocal
    family member and one foreign-homolog relative."""
    import json
    run_dir.mkdir()
    (run_dir / 'parameters.json').write_text(json.dumps({
        'source_dataset': 'dsA', 'target_dataset': 'dsB',
        'backward_evidence_enabled': True, 'backward_top_n': 5,
        'backward_max_neurons': 300, 'backward_per_branch_cap': 40,
        'backward_scan_pool_targets': True, 'skip_backward_pass': False,
    }), encoding='utf-8')
    (run_dir / 'set_coverage.json').write_text(json.dumps(
        {'mcns': {'backward_evidence': {
            'high': 1, 'medium': 1, 'low': 0, 'not-checked': 0,
            'distinct_scanned': 2, 'beyond_cap': 4}}}), encoding='utf-8')
    out = run_dir / 'expansion'
    if nested:
        out.mkdir()
    else:
        out = run_dir
    pd.DataFrame([_bev_row(11, 'T1', 'family', 'high', 101),
                  _bev_row(12, 'T2', 'relative', 'medium', 900)]
                 ).to_csv(out / 'backward_matches.csv', index=False,
                          encoding='utf-8')
    return run_dir


def _rep():
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from comparison import mapping_validation_report as rep
    return rep


def test_the_reciprocal_advisory_reaches_the_report_and_the_notes(tmp_path):
    """report.html used to carry only the ``!`` log lines, so the one
    advisory a reader must not miss lived solely in a side file.  Both sinks
    must now hold the SAME sentence."""
    rep = _rep()
    d = rep.collect_run_data(_bev_run(tmp_path / 'run'))
    warns = [w for w in rep.collect_warnings(d)
             if w.startswith('[reciprocal]')]
    assert len(warns) == 1
    assert warns[0].startswith('[reciprocal] 2/2 scanned gap-fill member(s)')
    assert '4 member(s) beyond the scan cap are unchecked' in warns[0]
    html = rep.build_report_document(d)
    assert warns[0] in html
    assert 'reciprocal: 2 member(s) rank their own branch source type top-3' in html


def test_the_fill_tab_splits_each_bin_by_reverse_evidence(tmp_path):
    """D7: the reverse fact gets its own axis next to the level ladder —
    per bin, since `level` is a morph-bar property and stays untouched."""
    rep = _rep()
    d = rep.collect_run_data(_bev_run(tmp_path / 'run'))
    html = rep.build_report_document(d)
    assert 'Reverse evidence by bin' in html
    assert 'family 1 scanned' in html and 'medium 1' in html


def test_the_file_index_prints_paths_that_exist(tmp_path):
    """The artifact index is registry-derived, so a folder written before
    the layout change used to advertise `expansion/…` paths it never had."""
    rep = _rep()
    nested = rep.build_report_document(
        rep.collect_run_data(_bev_run(tmp_path / 'nested')))
    assert '<td>expansion/backward_matches.csv</td>' in nested
    flat = rep.build_report_document(
        rep.collect_run_data(_bev_run(tmp_path / 'flat', nested=False)))
    assert '<td>backward_matches.csv</td>' in flat
    # the file-glossary panel still documents the canonical layout path; the
    # index of what THIS run wrote must not
    assert '<td>expansion/backward_matches.csv</td>' not in flat


# ---------------------------------------------------------------------------
# a scope a real run measured wrong
# ---------------------------------------------------------------------------

def test_reverse_column_keeps_pool_members_below_the_cut():
    """An in-pool source ranked past the cap must not vanish from its own
    column: the caller reads a missing row as `no reverse evidence`, not as
    `ranked low`, and silently downgrades the source's status."""
    m = _resolver()
    df = _scan_df([
        (10, 0.9, 0.8, 1, 1),
        (11, 0.8, 0.7, 2, 2),
        (12, 0.7, 0.6, 3, 3),
        (99, 0.1, 0.1, 4, 4),     # the branch's own source, ranked last
    ])
    col = m.reverse_source_column(df, [99], top_rows=3)
    assert [e['source'] for e in col] == [10, 11, 12, 99]
    assert [e['in_pool'] for e in col] == [False, False, False, True]
