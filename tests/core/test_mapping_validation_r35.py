"""Revision 3.5 tests for the type-mapping validation pipeline.

Covers the six R5-review issues (plan
plan-type-mapping-validation-pipeline.md, "Revision 3.5 — implementation
brief"):

- Issue 5a: target-side profile-quality gate — fragments (tiny weight,
  single partner) never enter a scan (``passes_target_quality_gate`` +
  ``build_target_vectors`` on a synthetic parquet cache).
- Issue 5b: jaccard sanity on suspicious claims — a positive
  rank_union win whose jaccard is far below the best pool member's
  jaccard is filtered as noise.
- Issue 3c: candidate_morph_factor recalibrated to 0.25 and the chosen
  rule + per-branch thresholds recorded by ``run_morphology``.
- Issues 2/3/3b: invader cross-referencing helpers — sibling/backward
  bucket keys, ``{N} types`` root labels, single-bucket rule.
- Issue 6c: scene identity self-check flags geometry rendered under the
  wrong legend leaf (the R5 50274/61430 slip) without false positives.

No network, no real caches: profiles are constructed in memory (the
parquet test writes its own tiny cache file).
"""

import json
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
    build_target_vectors,
    compute_set_coverage,
    expanded_vector,
    passes_target_quality_gate,
    prep_target_stats,
    run_file_path,
    scan_source,
    scene_styling_record,
)
from comparison.mapping_validation_visualize import (  # noqa: E402
    bucket_root_label,
    check_scene_identities,
)

SRC_UP = {'A': 10, 'B': 8, 'C': 4, 'D': 2}
SRC_DN = {'P': 6, 'Q': 3}


def make_profile(bid, up, dn):
    return ConnectivityProfile(
        neuron_id=bid, dataset='test',
        upstream_partners=dict(up), downstream_partners=dict(dn),
        actual_upstream_count=len(up), actual_downstream_count=len(dn),
    )


def stub_skeleton_store_all_cached(monkeypatch):
    """Stage 5 pre-flights the target raw-skeleton cache before scoring
    (plan §17.2).  These tests fake the Track-A scorer, not the store, so
    the store reports every target as already on disk: nothing is fetched
    and no network is reached.  The pre-flight's own fetch/bound/fail-open
    behavior is covered by
    test_skeleton_preflight_is_cache_first_bounded_and_fails_open.
    """
    import morphology

    class _AllCached:
        def find_skeleton_file(self, bid):
            return Path(f'/stub-cache/{int(bid)}.swc.zst')

    monkeypatch.setattr(morphology, 'find_similar_raw_cache',
                        lambda dataset, **kw: _AllCached())


def run_validator(source_pool, target_pool, target_vectors_by_bid,
                  source_profiles_by_bid, sizes=None, weights=None,
                  source_sides=None, target_sides=None, **cfg_kwargs):
    cfg = MappingValidationConfig(
        source_dataset='dsA', target_dataset='dsB',
        query_types=['T'], morph_enabled=False, visualize=False,
        **cfg_kwargs)
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = cfg
    v.mapper = None
    v.notes = []

    from comparison.mapping_validation import TypePair
    pair = TypePair(
        source_dataset='dsA', source_type='T', source_pool=source_pool,
        target_dataset='dsB', target_type='T', target_pool=target_pool)

    class FakeProfiler:
        def get_profile(self, bid, dataset):
            return source_profiles_by_bid.get(bid)

        def get_types_for_bodyids(self, bids, dataset):
            return {b: 'TX' for b in bids}

    v.profiler = FakeProfiler()
    target_stats = prep_target_stats(target_vectors_by_bid)
    scans = {bid: scan_source(expanded_vector(p, None), target_stats)
             for bid, p in source_profiles_by_bid.items()}
    res = v.validate_pair(pair, scans,
                          {b: 'TX' for b in target_vectors_by_bid},
                          sizes=sizes, weights=weights,
                          source_sides=source_sides,
                          target_sides=target_sides)
    return v, pair, res


# ---------------------------------------------------------------------------
# Issue 5a — target-side profile-quality gate
# ---------------------------------------------------------------------------

def test_quality_gate_filters_fragments():
    strong = expanded_vector(
        make_profile(1, SRC_UP, SRC_DN), None)
    fragment = expanded_vector(
        make_profile(2, {}, {'DNpe035': 3}), None)
    assert passes_target_quality_gate(strong, 10.0, 2)
    assert not passes_target_quality_gate(fragment, 10.0, 2)
    # no gate -> everything passes
    assert passes_target_quality_gate(fragment)
    # single-partner floors apply independently
    assert not passes_target_quality_gate(fragment, min_partner_types=2)
    assert not passes_target_quality_gate(fragment, min_weight=10.0)


def test_build_target_vectors_applies_quality_gate(tmp_path):
    rows = pd.DataFrame({
        'neuron_id': np.array([301, 302], dtype=np.int64),
        'upstream_partners': [json.dumps(SRC_UP), json.dumps({})],
        'downstream_partners': [json.dumps(SRC_DN),
                                json.dumps({'DNpe035': 3})],
    })
    parquet = tmp_path / 'connectivity_profiles.parquet'
    rows.to_parquet(parquet, index=False)

    class FakeProfiler:
        cache_dir = tmp_path

        def _get_cache_parquet_path(self, ds):
            return parquet

    # gated: the 3-synapse single-partner fragment is excluded
    vectors = build_target_vectors(FakeProfiler(), 'dsX', None,
                                   verbose=False,
                                   min_weight=10.0, min_partner_types=2)
    assert set(vectors) == {301}

    # ungated: both cached profiles produce vectors
    vectors_all = build_target_vectors(FakeProfiler(), 'dsX', None,
                                       verbose=False)
    assert set(vectors_all) == {301, 302}


# ---------------------------------------------------------------------------
# Issue 5b — jaccard sanity on suspicious claims
# ---------------------------------------------------------------------------

def _sanity_scenario():
    """202 wins rank_union (+1.0) with jaccard 6/14 ~ 0.43; 203 wins
    rank_union (+1.0) with jaccard 6/8 = 0.75; the best pool member 201
    holds jaccard 1.0.  With factor 0.5 only 203 survives the filter."""
    extra8 = {k: 0.5 for k in 'EFGHIJKL'}
    extra2 = {'M': 0.5, 'N': 0.5}
    targets = {
        201: ({'A': 10, 'B': 8, 'C': 4, 'D': 2}, {'P': 2, 'Q': 9}),
        202: ({**SRC_UP, **SRC_DN, **extra8}, {}),
        203: ({'A': 20, 'B': 16, 'P': 12, 'C': 8, 'Q': 6, 'D': 4,
               **extra2}, {}),
    }
    tgt_vectors = {bid: expanded_vector(make_profile(bid, up, dn), None)
                   for bid, (up, dn) in targets.items()}
    src = make_profile(1, SRC_UP, SRC_DN)
    return tgt_vectors, src


def test_jaccard_sanity_filters_low_agreement_rank_union_win():
    tgt_vectors, src = _sanity_scenario()
    v, _, res = run_validator([1], [201], tgt_vectors, {1: src})
    sus = {(s['ahead_target_bodyId'], s['ahead_metric'])
           for s in res['suspicious']}
    # 203's positive win survives (jaccard 0.75 >= 0.5 x 1.0)...
    assert (203, 'rank_union') in sus
    # ...202's is noise (jaccard 0.43 < 0.5 x 1.0 despite rank_union +1)
    assert (202, 'rank_union') not in sus
    row = res['rows'][0]
    assert row['verdict'] == 'verified'
    assert 'noise_filtered=1' in row['flags']
    assert 'neg_ru_or_jac=1' in row['flags']
    assert row['suspicious_noise_filtered'] == 1
    # the dropped row travels in the noise sink with its reason
    noise = res['noise']
    assert len(noise) == 1
    assert noise[0]['ahead_target_bodyId'] == 202
    assert noise[0]['noise_reason'] == 'jaccard_below_pool'


def test_jaccard_sanity_factor_zero_disables_filter():
    tgt_vectors, src = _sanity_scenario()
    _, _, res = run_validator([1], [201], tgt_vectors, {1: src},
                              suspicious_jaccard_factor=0.0)
    sus = {(s['ahead_target_bodyId'], s['ahead_metric'])
           for s in res['suspicious']}
    assert {(202, 'rank_union'), (203, 'rank_union')} <= sus
    assert res['rows'][0]['suspicious_noise_filtered'] == 0


# ---------------------------------------------------------------------------
# Issue 3c — recalibrated morph factor + recorded rule
# ---------------------------------------------------------------------------

def test_config_defaults_recalibrated():
    cfg = MappingValidationConfig(source_dataset='a', target_dataset='b')
    assert cfg.candidate_morph_factor == 0.25
    assert cfg.target_min_weight == 10.0
    assert cfg.target_min_partner_types == 2
    assert cfg.suspicious_jaccard_factor == 0.5
    assert cfg.scene_selfcheck is False


def test_run_morphology_records_candidate_rule(monkeypatch):
    import morphology
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(
        source_dataset='dsA', target_dataset='dsB', query_types=['T'],
        visualize=False)
    v.notes = []
    v.log = lambda m='': None
    val_rows = [{'source_bodyId': 1, 'target_bodyId': 11,
                 'verdict': 'verified_strong', 'flags': ''}]
    pool_detail = [
        {'target_bodyId': 11, 'category': 'verified',
         'best_source_bodyId': 1, 'source_type': 'T', 'target_type': 'T'},
        {'target_bodyId': 12, 'category': 'borderline',
         'best_source_bodyId': 1, 'source_type': 'T', 'target_type': 'T'},
    ]

    def fake_enrich(pair_df, src, tgt, verbose=False):
        df = pair_df.copy()
        df['morph_v2_similarity'] = 0.5
        df['morph_nblast'] = 0.4
        return df

    monkeypatch.setattr(morphology, 'enrich_homolog_results', fake_enrich)
    stub_skeleton_store_all_cached(monkeypatch)
    out = v.run_morphology(val_rows, [], [], pool_detail)
    # threshold = factor (0.25) x mean pool morph (0.5)
    assert out['candidate_thresholds']['T->T'] == pytest.approx(0.125)
    assert out['candidate_pool_avg_morph']['T->T'] == pytest.approx(0.5)
    assert 'factor=0.25' in out['candidate_morph_rule']


def test_run_morphology_scores_pool_reference_pairs(monkeypatch):
    """Review 2026-09-16 P1: ``pool_pairs`` were never passed to
    ``_morph_pair_frame``, so pool reference pairs that are not the
    mutual-best 'assigned' pair were never Track-A scored — the
    reference-tier mean (B_b) and the floors-v3 bars built on it were
    computed over a biased subset, and verified-only branches silently
    degraded to the null bar."""
    import morphology
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(
        source_dataset='dsA', target_dataset='dsB', query_types=['T'],
        visualize=False)
    v.notes = []
    v.log = lambda m='': None
    val_rows = [{'source_bodyId': 1, 'target_bodyId': 11,
                 'verdict': 'verified_strong', 'flags': ''}]
    # Target 12's best source is 2 — NOT the assigned (1, 11) pair.
    pool_detail = [
        {'target_bodyId': 11, 'category': 'verified',
         'best_source_bodyId': 1, 'source_type': 'T', 'target_type': 'T'},
        {'target_bodyId': 12, 'category': 'verified',
         'best_source_bodyId': 2, 'source_type': 'T', 'target_type': 'T'},
    ]

    def fake_enrich(pair_df, src, tgt, verbose=False):
        df = pair_df.copy()
        df['morph_v2_similarity'] = [
            0.9 if int(t) == 12 else 0.5 for t in df['target_bodyId']]
        df['morph_nblast'] = 0.4
        return df

    monkeypatch.setattr(morphology, 'enrich_homolog_results', fake_enrich)
    stub_skeleton_store_all_cached(monkeypatch)
    out = v.run_morphology(val_rows, [], [], pool_detail)
    scored = {d['target_bodyId']: d['morph_v2_similarity']
              for d in pool_detail}
    assert scored[12] == pytest.approx(0.9), \
        'pool ref pair (2, 12) must be scored even though it is not ' \
        'the assigned (1, 11) pair'
    # the reference mean includes the pool score: (0.9 + 0.5) / 2
    assert out['candidate_pool_avg_morph']['T->T'] == pytest.approx(0.7)


def test_finalize_keeps_same_type_branches_from_different_queries():
    """Review 2026-09-16 P2: per-branch results were keyed by
    (source_type, target_type) only, so two queries resolving the same
    concrete type pair overwrote each other's pools/tiers/scenes. The
    branch key carries the query now; the target type is always the last
    element (in_map_types must see T, not S)."""
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(
        source_dataset='dsA', target_dataset='dsB',
        query_types=['Q1', 'Q2'], visualize=False)
    v.notes = []
    v.log = lambda m='': None
    per_pair_res = {
        ('Q1', 'S', 'T'): {'_pool_set': [101], 'target_categories': {}},
        ('Q2', 'S', 'T'): {'_pool_set': [202], 'target_categories': {}},
    }
    v.finalize_categories(per_pair_res, [], [], [], [])
    assert len(v._branch_pools) == 2
    assert v._branch_pools[('Q1', 'S', 'T')] == {101}
    assert v._branch_pools[('Q2', 'S', 'T')] == {202}
    assert v._in_map_types == {'T'}


# ---------------------------------------------------------------------------
# Issues 2/3/3b — invader cross-referencing helpers
# ---------------------------------------------------------------------------


def test_bucket_root_label_counts_types():
    # Revision 3.12: the four expansion categories are ONE root each (the
    # type detail rides on the leaf), so a bare key returns itself.
    assert bucket_root_label('examinees', {1: 'SMP223'}) == 'examinees'
    assert bucket_root_label(
        'examinees', {1: 'CB4091', 2: 'SMP223', 3: 'CB3508'}) \
        == 'examinees'
    assert bucket_root_label('candidates', {1: 'SMP223'}) == 'candidates'
    assert bucket_root_label('family', {1: 'SMP219'}) == 'family'
    assert bucket_root_label('relative', {1: 'CB4091'}) == 'relative'
    assert bucket_root_label('candidates', {}) == 'candidates'
    # sibling stays a counted root
    assert bucket_root_label('sibling', {1: 'SMP223', 2: 'SMP222'}) \
        == 'sibling · 2 members'
    # legacy/verbatim keys pass through
    assert bucket_root_label('sibling · verified · SMP221', {9: 'X'}) \
        == 'sibling · verified · SMP221'
    assert bucket_root_label('backward · s-CPDN3D', {9: 'X'}) \
        == 'backward · s-CPDN3D'


# ---------------------------------------------------------------------------
# Issue 6c — scene identity self-check
# ---------------------------------------------------------------------------

class _FakeTrace:
    def __init__(self, mode, item, pts):
        self.mode = mode
        self.meta = {'drocatLegend': {'kind': 'neuron', 'item': item}}
        self.x = pts[:, 0]
        self.y = pts[:, 1]
        self.z = pts[:, 2]


class _FakeFig:
    def __init__(self, traces):
        self.data = traces


class _FakeViz:
    def __init__(self, traces):
        self.fig_3d = _FakeFig(traces)


def test_check_scene_identities_flags_mislabeled_leaf():
    # R5 slip: the leaf labeled 61430 (left hemisphere, low x) actually
    # carries 50274's geometry (right hemisphere, high x)
    bbox_61430 = (np.array([503_000., 100., 100.]),
                  np.array([508_000., 200., 200.]))
    bbox_50274 = (np.array([531_000., 300., 100.]),
                  np.array([538_000., 400., 200.]))
    scene_bboxes = {61430: bbox_61430, 50274: bbox_50274}
    true_61430 = np.random.default_rng(7).uniform(
        bbox_61430[0], bbox_61430[1], size=(50, 3))
    ghost_50274 = np.random.default_rng(9).uniform(
        bbox_50274[0], bbox_50274[1], size=(50, 3))
    viz = _FakeViz([
        _FakeTrace('lines', '61430_SMP223_L', true_61430),
        _FakeTrace('lines', '61430_None', ghost_50274),  # the slip
        _FakeTrace('markers', '50274_CB3508_R',
                   np.zeros((1, 3))),  # legend-fix marker: skipped
    ])
    problems = check_scene_identities(viz, scene_bboxes)
    assert len(problems) == 1
    item, bid, why = problems[0]
    assert item == '61430_None' and bid == 61430
    assert 'nearest neuron 50274' in why


def test_check_scene_identities_passes_clean_scene():
    bbox = (np.array([0., 0., 0.]), np.array([1000., 1000., 1000.]))
    pts = np.random.default_rng(3).uniform(bbox[0], bbox[1], size=(30, 3))
    viz = _FakeViz([_FakeTrace('lines', '18706_CL125_L', pts)])
    assert check_scene_identities(viz, {18706: bbox}) == []


def test_check_scene_identities_reports_missing_bbox():
    pts = np.array([[1., 1., 1.], [2., 2., 2.]])
    viz = _FakeViz([_FakeTrace('lines', '999_T_X', pts)])
    problems = check_scene_identities(viz, {18706: (
        np.zeros(3), np.ones(3) * 10)})
    assert problems and problems[0][1] == 999


def test_check_scene_identities_undoes_render_rotation():
    """The backend warps every rendered layer with the FAFB tilt
    correction (rigid rotation about the template center); the self-check
    must map trace coordinates back to raw space before comparing with
    the loaded bboxes."""
    import math

    bbox = (np.array([400_000., 100_000., 50_000.]),
            np.array([650_000., 350_000., 250_000.]))
    rng = np.random.default_rng(11)
    raw = rng.uniform(bbox[0], bbox[1], size=(80, 3))

    center = np.array([527652.0, 240039.0, 148110.0])
    ang = math.radians(-12)
    def rot(axis):
        c, s = math.cos(ang), math.sin(ang)
        if axis == 'z':
            return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    R = np.eye(4)
    R[:3, :3] = rot('z') @ rot('y')
    T = np.eye(4)
    T[:3, 3] = center
    M = T @ R @ np.linalg.inv(T)

    rendered = (M @ np.c_[raw, np.ones(len(raw))].T).T[:, :3]
    viz = _FakeViz([_FakeTrace('lines', '18706_CL125_L', rendered)])

    # without the undo matrix the rotated geometry looks outside
    assert len(check_scene_identities(viz, {18706: bbox})) == 1
    # with the inverse correction applied, the leaf passes
    undo = np.linalg.inv(M)
    assert check_scene_identities(viz, {18706: bbox},
                                  undo_matrix=undo) == []


# ---------------------------------------------------------------------------
# Revision 3.6 — spatial-caliber gate (PRIMARY), tie margin, untyped policy
# ---------------------------------------------------------------------------

def test_spatial_caliber_gate_drops_tiny_fragment():
    """A 293154-shaped invader (tiny spatial size vs the pool) is noise
    at any metric; size columns travel on the kept and noise rows."""
    tgt_vectors = {
        201: expanded_vector(make_profile(
            201, {'A': 10, 'B': 8, 'C': 4, 'D': 2}, {'P': 2, 'Q': 9}), None),
        202: expanded_vector(make_profile(
            202, dict(SRC_UP), dict(SRC_DN)), None),
    }
    src = make_profile(1, SRC_UP, SRC_DN)
    sizes = {201: 2.6e9, 202: 14e6}   # pool: huge; invader: tiny
    _, _, res = run_validator([1], [201], tgt_vectors, {1: src},
                              sizes=sizes)
    # 202 would normally be suspicious (verified pool, ru top-1 for 202)
    assert all(s['ahead_target_bodyId'] != 202 for s in res['suspicious'])
    noise = res['noise']
    assert noise and noise[0]['ahead_target_bodyId'] == 202
    assert 'spatial_caliber' in noise[0]['noise_reason']
    assert noise[0]['size_ratio'] < 0.1
    row = res['rows'][0]
    assert row['suspicious_size_filtered'] == 1


def test_caliber_fallback_to_weight_ratio_when_size_missing():
    tgt_vectors = {
        201: expanded_vector(make_profile(
            201, {'A': 10, 'B': 8, 'C': 4, 'D': 2}, {'P': 2, 'Q': 9}), None),
        202: expanded_vector(make_profile(
            202, dict(SRC_UP), dict(SRC_DN)), None),
    }
    src = make_profile(1, SRC_UP, SRC_DN)
    # no size map: the weight-ratio fallback must still drop the tiny one
    _, _, res = run_validator([1], [201], tgt_vectors, {1: src},
                              weights={201: 500.0, 202: 8.0})
    assert all(s['ahead_target_bodyId'] != 202 for s in res['suspicious'])
    assert res['noise'][0]['noise_reason'] == 'spatial_caliber'


def test_tie_margin_filters_numerical_ties():
    # 22 wins by a hair (+0.005 < 0.02 margin) over the pool best 11,
    # while 23's +0.10 margin survives
    import pandas as pd
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(
        source_dataset='dsA', target_dataset='dsB', query_types=['T'],
        morph_enabled=False, visualize=False)
    v.mapper = None
    v.notes = []

    class FakeProfiler:
        def get_profile(self, bid, dataset):
            return None

        def get_types_for_bodyids(self, bids, dataset):
            return {b: 'TX' for b in bids}

    v.profiler = FakeProfiler()
    pair = TypePair('dsA', 'T', [1], 'dsB', 'T', [11])
    scans = {1: pd.DataFrame([
        {'target_bid': 23, 'rank_union': 0.60, 'rank_union_rank': 1,
         'jaccard': 0.55, 'jaccard_rank': 2},
        {'target_bid': 22, 'rank_union': 0.505, 'rank_union_rank': 2,
         'jaccard': 0.30, 'jaccard_rank': 3},
        {'target_bid': 11, 'rank_union': 0.50, 'rank_union_rank': 3,
         'jaccard': 0.60, 'jaccard_rank': 1},
    ])}
    res = v.validate_pair(pair, scans, {11: 'T', 22: 'TX', 23: 'TX'})
    sus_bids = {s['ahead_target_bodyId'] for s in res['suspicious']}
    assert 23 in sus_bids                      # +0.10 margin survives
    assert 22 not in sus_bids                  # +0.005 is a numerical tie
    tie_rows = [n for n in res['noise']
                if n['ahead_target_bodyId'] == 22]
    assert tie_rows and any('tie_margin' in n['noise_reason']
                            for n in tie_rows)
    assert res['rows'][0]['suspicious_tie_filtered'] >= 1


def test_always_fire_proposes_without_gap_trigger():
    """Rev 3.8 scope (user 2026-09-12): DEFAULT is invader-only +
    gap-gated fills (no proposal at gap=0); `aggressive_expansion`
    opts in to unconditional proposals."""
    # 2 sources, 1 pool target + 1 out-of-pool target: 1 source always
    # ends unpaired and must receive a proposal (here: the out-of-pool
    # near-match 202)
    tgt_vectors = {
        201: expanded_vector(make_profile(201, dict(SRC_UP), dict(SRC_DN)),
                             None),
        202: expanded_vector(make_profile(
            202, {'A': 10, 'B': 8, 'C': 4, 'D': 2}, {'P': 2, 'Q': 9}), None),
    }
    src1 = make_profile(1, SRC_UP, SRC_DN)
    src2 = make_profile(2, {'A': 10, 'B': 8, 'C': 4, 'D': 2},
                        {'P': 2, 'Q': 9})  # slightly different
    # default: gap-gated — gap=0 -> no fill proposals
    _, pair, res = run_validator([1, 2], [201], tgt_vectors,
                                 {1: src1, 2: src2})
    assert res['summary']['gap'] <= 1        # no gap by the old rule
    assert not [f for f in res['fills'] if f['side'] == 'source']
    # aggressive opt-in: the unpaired source gets its proposal
    _, pair, res = run_validator([1, 2], [201], tgt_vectors,
                                 {1: src1, 2: src2},
                                 aggressive_expansion=True)
    src_fills = [f for f in res['fills'] if f['side'] == 'source']
    assert src_fills, 'unpaired source must get a proposal even at gap=0'
    assert all(f['fill_class'] == 'out_of_pool' for f in src_fills)


# ---------------------------------------------------------------------------
# Revision 3.6 — annotate_invaders classification
# ---------------------------------------------------------------------------

def test_annotate_invaders_sibling_altchain_hollow():
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='flywire_FAFB_v783',
                                    target_dataset='male-cns:v1.0',
                                    query_types=['s-CPDN3D'],
                                    visualize=False, morph_enabled=False)
    v.mapper = None      # no backward decisions in this unit test
    v.pairs = []         # no bridge chains: chain index stays empty
    v.notes = []
    # sibling pools: 50274 verified in CB3508 branch; 61868 verified in
    # SMP223 branch
    per_pair_res = {
        ('s-CPDN3D', 'CB3508'): {'target_categories': {50274: 'verified',
                                                       511870: 'verified'}},
        ('s-CPDN3D', 'SMP223'): {'target_categories': {61868: 'verified',
                                                       53640: 'borderline'}},
    }
    sus = [
        {'source_type': 's-CPDN3D', 'target_type': 'SMP222',
         'ahead_target_bodyId': 50274, 'ahead_target_type': 'CB3508'},
        {'source_type': 's-CPDN3D', 'target_type': 'SMP222',
         'ahead_target_bodyId': 293154, 'ahead_target_type': None},
    ]
    fills = [{'side': 'source', 'fill_class': 'out_of_pool',
              'source_type': 's-CPDN3D', 'target_type': 'SMP222',
              'proposal_bodyId': 61868, 'proposal_type': 'SMP223'}]
    v.annotate_invaders(sus, fills, per_pair_res)
    assert sus[0]['invader_class'] == 'sibling'
    assert sus[0]['invader_label'] == 'sibling · verified · CB3508'
    assert sus[1]['invader_class'] == 'untyped'
    assert fills[0]['invader_class'] == 'sibling'
    assert fills[0]['sibling_category'] == 'verified'


def test_annotate_invaders_cross_parent_becomes_backward():
    """User refinement 2026-09-12: sibling = same parent ONLY. A member
    of a s-CPDN3D branch seen in an s-CPDN3C scene is NOT a sibling — it
    falls through to the backward classification (SMP222 -> s-CPDN3D)."""
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='flywire_FAFB_v783',
                                    target_dataset='male-cns:v1.0',
                                    query_types=['s-CPDN3C'],
                                    visualize=False, morph_enabled=False)
    v.notes = []

    class FakeMapper:
        def get_mapping_decision(self, t, src, tgt):
            return {'status': 'mapped', 'target_type': 's-CPDN3D',
                    'target_types': ['s-CPDN3D']}

    v.mapper = FakeMapper()
    v.pairs = []
    v._source_type_counts = {'s-CPDN3D': 37}
    v._source_add_counts = {}
    per_pair_res = {
        ('s-CPDN3D', 'SMP222'): {'target_categories': {46295: 'verified'}},
        ('s-CPDN3C', 'SMP220'): {'target_categories': {154435: 'verified'}},
    }
    sus = [{'source_type': 's-CPDN3C', 'target_type': 'SMP218',
            'ahead_target_bodyId': 46295, 'ahead_target_type': 'SMP222'},
           {'source_type': 's-CPDN3C', 'target_type': 'SMP218',
            'ahead_target_bodyId': 154435, 'ahead_target_type': 'SMP220'}]
    v.annotate_invaders(sus, [], per_pair_res)
    # 46295: member of a DIFFERENT parent's branch -> backward
    assert sus[0]['invader_class'] == 'backward'
    assert sus[0]['invader_label'] == 'backward · s-CPDN3D'
    assert sus[0]['sibling_pool_of'] == ''
    # 154435: member of the SAME parent's OTHER branch -> sibling
    assert sus[1]['invader_class'] == 'sibling'
    assert sus[1]['invader_label'] == 'sibling · verified · SMP220'


def test_hemisphere_asymmetry_fires_gap():
    """User 2026-09-12 (corrected): every neuron has a hemisphere
    identity, so an L != R imbalance in either pool fires the gap even
    at arithmetic gap 0 — and fills fire on that trigger."""
    tgt_vectors = {
        201: expanded_vector(make_profile(201, dict(SRC_UP), dict(SRC_DN)),
                             None),
        202: expanded_vector(make_profile(
            202, {'A': 10, 'B': 8, 'C': 4, 'D': 2}, {'P': 2, 'Q': 9}), None),
    }
    src1 = make_profile(1, SRC_UP, SRC_DN)
    # pool: 1 neuron (L) -> L=1 vs R=0 asymmetry fires even if M covers it
    _, pair, res = run_validator([1, 2], [201], tgt_vectors,
                                 {1: src1, 2: src2_profile()},
                                 source_sides={1: 'L', 2: 'R'},
                                 target_sides={201: 'L'})
    fired = res['summary']['gap_triggered']
    hemi = res['summary'].get('hemisphere', {})
    assert hemi.get('hemisphere_asymmetry') is True, hemi
    assert fired is True
    # fills fired on the asymmetry trigger (default scope)
    assert any(f['side'] == 'source' for f in res['fills'])


def src2_profile():
    return make_profile(2, {'A': 10, 'B': 8, 'C': 4, 'D': 2},
                        {'P': 2, 'Q': 9})


def test_sibling_proposal_not_counted_as_fill():
    """User 2026-09-12: a sibling-classified proposal is cross-branch
    convergence, NOT a fill of the gap — excluded from gap accounting
    (counts_toward_gap_fill=False), still visible in the CSV."""
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='flywire_FAFB_v783',
                                    target_dataset='male-cns:v1.0',
                                    query_types=['s-CPDN3C'],
                                    visualize=False, morph_enabled=False)
    v.mapper = None
    v.pairs = []
    v.notes = []
    per_pair_res = {
        ('s-CPDN3C', 'SMP218'): {'target_categories': {28988: 'verified'}},
    }
    fills = [{'side': 'source', 'fill_class': 'out_of_pool',
              'source_type': 's-CPDN3C', 'target_type': 'SMP221',
              'proposal_bodyId': 28988, 'proposal_type': 'SMP218'}]
    v.annotate_invaders([], fills, per_pair_res)
    assert fills[0]['invader_class'] == 'sibling'
    assert fills[0]['counts_toward_gap_fill'] is False


def test_rev310_fill_accounting_rules():
    """Rev 3.10: class- and scope-aware fill accounting — in-pool exempt
    from classification; sibling never counts; same-type alternate-chain
    counts; cross-target alternate-chain and backward never count
    (scope note distinguishes convergence from expansion advice)."""
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='flywire_FAFB_v783',
                                    target_dataset='male-cns:v1.0',
                                    query_types=['s-CPDN3C'],
                                    visualize=False, morph_enabled=False)

    class FakeMapper:
        def get_mapping_decision(self, t, src, tgt):
            return {'status': 'mapped', 'target_type': 's-CPDN3C',
                    'target_types': ['s-CPDN3C']}

        def get_type_bridges(self, parent, src, tgt, max_bridges=0):
            chains = []
            for linker, final in (('CB1709', 'SMP220'),
                                  ('CB2438', 'SMP220'),
                                  ('CB3767', 'SMP220'),
                                  ('CB2568', 'SMP221')):
                chains.append([
                    {'dataset': src, 'column': 'type', 'value': parent},
                    {'dataset': src, 'column': 'additional_type(s)',
                     'value': linker, 'via': parent},
                    {'dataset': tgt, 'column': 'flywireType',
                     'value': final}])
            return chains

    v.mapper = FakeMapper()
    v._source_type_counts = {'s-CPDN3C': 32}
    v._source_add_counts = {}
    sel = TypePair('flywire_FAFB_v783', 's-CPDN3C', [1], 'male-cns:v1.0',
                   'SMP220', [2])
    sel.linkers = [{'column': 'additional_type(s)', 'raw_value': 'CB1709',
                    'home': 'flywire_FAFB_v783'}]
    v.pairs = [sel]
    v.notes = []
    per_pair_res = {
        ('s-CPDN3C', 'SMP218'): {'target_categories': {28988: 'verified'}},
        ('s-CPDN3C', 'SMP220'): {'target_categories': {295020: 'verified'}},
        ('s-CPDN3D', 'SMP223'): {'target_categories': {61868: 'verified'}},
    }
    fills = [
        # in-pool: never classified, always counts (distinct proposal id
        # from the sibling row below — same id would overwrite by_bid)
        {'side': 'source', 'fill_class': 'in_pool',
         'source_type': 's-CPDN3C', 'target_type': 'SMP218',
         'proposal_bodyId': 28989, 'proposal_type': 'SMP218'},
        # sibling: never counts
        {'side': 'source', 'fill_class': 'out_of_pool',
         'source_type': 's-CPDN3C', 'target_type': 'SMP221',
         'proposal_bodyId': 28988, 'proposal_type': 'SMP218'},
        # same-type alternate-chain residue: counts
        {'side': 'source', 'fill_class': 'out_of_pool',
         'source_type': 's-CPDN3C', 'target_type': 'SMP220',
         'proposal_bodyId': 295020, 'proposal_type': 'SMP220'},
        # cross-target alternate-chain: does not count
        {'side': 'source', 'fill_class': 'out_of_pool',
         'source_type': 's-CPDN3C', 'target_type': 'SMP222',
         'proposal_bodyId': 40165, 'proposal_type': 'SMP223'},
        # backward inside the query family: never counts, convergence
        {'side': 'source', 'fill_class': 'out_of_pool',
         'source_type': 's-CPDN3C', 'target_type': 'SMP227',
         'proposal_bodyId': 61868, 'proposal_type': 'SMP223'},
    ]
    family = {'s-CPDN3C', 's-CPDN3D'}
    v.annotate_invaders([], fills, per_pair_res, family_types=family)
    by_prop = {f['proposal_bodyId']: f for f in fills}
    assert by_prop[28989]['invader_class'] == 'in-pool'
    assert by_prop[28989]['counts_toward_gap_fill'] is True
    # both sibling-keyed rows (in-pool was overwritten by sibling key)
    sib = [f for f in fills if f['invader_class'] == 'sibling']
    assert sib and all(f['counts_toward_gap_fill'] is False
                       for f in sib)
    assert by_prop[295020]['invader_class'] == 'same-type'
    # Rev 3.11: same-type extras are the only legitimate fills; their
    # counting defers to the morph qualification in stage 5
    assert by_prop[295020]['counts_toward_gap_fill'] is None
    assert by_prop[295020]['same_type_residue'] is True
    assert by_prop[40165]['counts_toward_gap_fill'] is False
    assert by_prop[40165]['fill_scope_note'] == 'cross_branch_convergence'
    assert by_prop[61868]['counts_toward_gap_fill'] is False
    assert by_prop[61868]['in_query_family'] is True
    assert by_prop[61868]['fill_scope_note'] == 'cross_branch_convergence'


def test_rev310_backward_out_of_family_is_expansion_advice():
    """A backward fill whose mapped type is NOT in the query family is
    query-expansion advice: never counts, flagged for widening."""
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='flywire_FAFB_v783',
                                    target_dataset='male-cns:v1.0',
                                    query_types=['DN1pD'],
                                    visualize=False, morph_enabled=False)

    class FakeMapper:
        def get_mapping_decision(self, t, src, tgt):
            return {'status': 'mapped', 'target_type': 'R1-6',
                    'target_types': ['R1-6']}

    v.mapper = FakeMapper()
    v.pairs = []
    v.notes = []
    v._source_type_counts = {'R1-6': 100}
    v._source_add_counts = {}
    fills = [{'side': 'source', 'fill_class': 'out_of_pool',
              'source_type': 'DN1pD', 'target_type': 'SMP539',
              'proposal_bodyId': 167595, 'proposal_type': 'R1-R6'}]
    v.annotate_invaders([], fills, {}, family_types={'DN1pD'})
    assert fills[0]['invader_class'] == 'backward'
    assert fills[0]['counts_toward_gap_fill'] is False
    assert fills[0]['in_query_family'] is False
    assert fills[0]['fill_scope_note'] == 'query_expansion_advice'


def test_rev310_compute_set_coverage_holes():
    """Set-level coverage: holes = mapped-target-set neurons in no
    branch pool and never reached; FAFB rollup counts assigned /
    proposed / unpaired."""
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='dsA',
                                    target_dataset='dsB',
                                    query_types=['T'],
                                    visualize=False, morph_enabled=False)
    v.notes = []
    pair_a = TypePair('dsA', 'T1', [1, 2], 'dsB', 'X', [11, 12, 99])
    pair_a.parent_source_pool = [1, 2, 3]
    pair_a.parent_target_pool = [11, 12, 99]
    pair_b = TypePair('dsA', 'T2', [4], 'dsB', 'Y', [21])
    pair_b.parent_source_pool = [4]
    pair_b.parent_target_pool = [21]
    pairs = [pair_a, pair_b]
    per_pair_res = {
        ('T1', 'X'): {
            'pairs': [(1, 11)],
            'target_categories': {11: 'matched', 12: 'verified'},
            'fills': [],
        },
        ('T2', 'Y'): {
            'pairs': [],
            'target_categories': {21: 'unmatched'},
            'fills': [],
        },
    }
    # source 2 and 3 get a counted fill proposal; neuron 99 is a hole
    fills = [
        {'side': 'source', 'fill_class': 'out_of_pool',
         'source_type': 'T1', 'target_type': 'X',
         'bodyId': 2, 'proposal_bodyId': 30,
         'counts_toward_gap_fill': True,
         'proposal_type': 'Z'},
        {'side': 'source', 'fill_class': 'out_of_pool',
         'source_type': 'T1', 'target_type': 'X',
         'bodyId': 3, 'proposal_bodyId': 99,
         'counts_toward_gap_fill': False,   # sibling: not counted
         'proposal_type': 'W'},
    ]
    cov = compute_set_coverage(pairs, per_pair_res, fills)
    f = cov['source']
    assert f['total_queried'] == 4        # {1,2,3} u {4}
    assert f['assigned'] == 1             # source 1
    assert f['fill_proposed_only'] == 1   # source 2 (3 is sibling-blocked)
    assert f['unpaired_unproposed'] == 2  # source 3 + 4
    m = cov['target']
    assert m['mapped_target_set'] == 4    # {11,12,99} u {21}
    assert m['in_branch_pool'] == 3       # 11, 12, 21
    assert m['reached_as_candidates_only'] == 1  # 30
    # 99's proposal is sibling-blocked (not counted) -> 99 is a hole
    assert m['holes'] == 1
    assert m['per_type']['X']['hole_body_ids'] == [99]
    # once its proposal counts, the hole closes
    fills[1]['counts_toward_gap_fill'] = True
    cov = compute_set_coverage(pairs, per_pair_res, fills)
    assert cov['target']['holes'] == 0
    assert cov['target']['per_type']['X']['hole_body_ids'] == []
    # the blocks are named by ROLE, not by a hard-coded dataset: a
    # FAFB->BANC run that reports its target as `mcns` is a lie the
    # report then repeats (plan-tmvev-jaccard-primary-bodyid-ranking.md
    # §14, r22).
    assert cov['source_dataset'] == 'dsA' and cov['target_dataset'] == 'dsB'


def test_rev312_set_coverage_candidate_evidence_closes_hole():
    """A bodyId the run claimed as a `candidates` row (invader route,
    no fill proposal) is reached, not a hole: compute_set_coverage must
    read the post-finalization evidence rows, not only the fills."""
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='dsA',
                                    target_dataset='dsB',
                                    query_types=['T'],
                                    visualize=False, morph_enabled=False)
    v.notes = []
    pair_a = TypePair('dsA', 'T1', [1], 'dsB', 'X', [11, 99])
    pair_a.parent_source_pool = [1]
    pair_a.parent_target_pool = [11, 99]
    pairs = [pair_a]
    per_pair_res = {
        ('T1', 'X'): {
            'pairs': [(1, 11)],
            'target_categories': {11: 'matched'},
            'fills': [],
        },
    }
    evidence = [
        # invader-surfaced candidate of the mapped type X: claims 99
        {'source_type': 'T1', 'target_type': 'X',
         'ahead_target_bodyId': 99, 'ahead_target_type': 'X',
         'category': 'candidates', 'in_scope': True},
        # morph-failed rows claim nothing
        {'source_type': 'T1', 'target_type': 'X',
         'ahead_target_bodyId': 98, 'ahead_target_type': 'X',
         'category': '', 'in_scope': False},
    ]
    cov = compute_set_coverage(pairs, per_pair_res, [], evidence_rows=evidence)
    m = cov['target']
    assert m['mapped_target_set'] == 2      # {11, 99}
    assert m['in_branch_pool'] == 1         # 11
    assert m['reached_as_candidates_only'] == 1  # 99 via the candidate row
    assert m['holes'] == 0
    assert m['per_type']['X']['hole_body_ids'] == []


def test_rev312_category_buckets_from_exported_category():
    """Revision 3.12: the scene buckets by the EXPORTED category, not by
    re-parsing label prefixes.  A row renders only when in scope and its
    category is an expansion bin; candidates split by annotation; out-of-
    scope (morph-failed) rows never render."""
    from comparison.mapping_validation_visualize import (
        build_category_buckets)
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(
        source_dataset='flywire_FAFB_v783', target_dataset='male-cns:v1.0',
        query_types=['x'], visualize=False)
    res = {'suspicious': [
        {'source_bodyId': 1, 'ahead_target_bodyId': 56425,
         'ahead_target_type': 'SMP223', 'category': 'candidates',
         'candidate_annotation': 'SMP223>s-CPDN3C', 'in_scope': True,
         'source_type': 's-CPDN3C', 'target_type': 'SMP223'},
        {'source_bodyId': 1, 'ahead_target_bodyId': 65631,
         'ahead_target_type': 'SMP223', 'category': 'candidates',
         'candidate_annotation': 'SMP223(no_source)', 'in_scope': True,
         'source_type': 's-CPDN3C', 'target_type': 'SMP223'},
        {'source_bodyId': 1, 'ahead_target_bodyId': 99999,
         'ahead_target_type': 'SMP223', 'category': '', 'in_scope': False,
         'morph_failed': True, 'source_type': 's-CPDN3C',
         'target_type': 'SMP223'},
        {'source_bodyId': 1, 'ahead_target_bodyId': 11111,
         'ahead_target_type': 'SMP219', 'category': 'sibling',
         'in_scope': True, 'source_type': 's-CPDN3C',
         'target_type': 'SMP223'},
    ], 'fills': []}
    b, order = build_category_buckets(res, v, None, suspicious_cap=20)
    # Revision 3.12: ONE candidates root; the qualified token rides on the
    # leaf and is also the leaf sortKey.
    assert 'candidates' in b
    assert {56425, 65631} <= b['candidates']['ids']
    assert b['candidates']['leaf_token'][56425] == 'SMP223>s-CPDN3C'
    assert b['candidates']['leaf_token'][65631] == 'SMP223(no_source)'
    assert b['candidates']['sort_key'][56425].startswith('SMP223>')
    assert 'sibling' in b and 11111 in b['sibling']['ids']
    # out-of-scope row never rendered
    assert all(99999 not in rec['ids'] for rec in b.values())


def test_rev312_one_bucket_per_neuron_follows_dedup_rank():
    """The live single-bucket rule (Revision 3.6 inversion, enforced by
    `build_category_buckets` through DEDUP_RANK): a target that qualifies
    for two expansion bins renders in ONE of them, the structurally
    stronger one, and the weaker claim is vacated — in either row order.

    This replaces what the retired `assign_invader` used to prove about the
    pre-3.12 builder; the rule itself is live, so it needs a live test.
    """
    from comparison.mapping_validation_visualize import build_category_buckets

    def _rows(strong_first):
        sibling = {'source_bodyId': 1, 'ahead_target_bodyId': 50274,
                   'ahead_target_type': 'CB3508', 'category': 'sibling',
                   'candidate_annotation': 'CB3508', 'in_scope': True,
                   'source_type': 's-CPDN3C', 'target_type': 'SMP222'}
        cand = dict(sibling, category='candidates',
                    candidate_annotation='CB3508>s-CPDN3C')
        return [sibling, cand] if strong_first else [cand, sibling]

    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(
        source_dataset='flywire_FAFB_v783', target_dataset='male-cns:v1.0',
        query_types=['x'], visualize=False)
    for strong_first in (True, False):
        b, _ = build_category_buckets(
            {'suspicious': _rows(strong_first), 'deep': [], 'fills': []},
            v, None, suspicious_cap=20)
        assert 50274 in b['sibling']['ids'], strong_first
        # the weaker bin never keeps a ghost of the vacated claim
        assert 50274 not in b.get('candidates', {'ids': set()})['ids'], \
            strong_first
        assert sum(50274 in rec['ids'] for rec in b.values()) == 1, \
            strong_first


def test_rev312_classify_category_partition():
    """The S2 ordered first-match (Rev 3.12): uniqueness of a target
    across categories, the sibling/candidates membership split, and the
    mode gating of family/relative/examinees."""
    from comparison.mapping_validation import classify_category

    # tier: in this branch's pool
    assert classify_category(
        target_bid=1, branch_pool={1}, in_map={1, 2}, target_type='T',
        branch_target_type='T', in_map_types={'T'}, candidate_types=set(),
        connectivity_qualified=False, morph_ok=False, is_deep=False,
        tier='verified', mode='restrictive') == ('verified', True, False)

    # sibling: in-map target of another branch, admitted
    assert classify_category(
        target_bid=2, branch_pool={1}, in_map={1, 2}, target_type='T',
        branch_target_type='T', in_map_types={'T'}, candidate_types=set(),
        connectivity_qualified=True, morph_ok=True, is_deep=False,
        tier=None, mode='restrictive')[0] == 'sibling'
    # sibling NOT admitted (morph fail) -> out of scope, morph_failed
    assert classify_category(
        target_bid=2, branch_pool={1}, in_map={1, 2}, target_type='T',
        branch_target_type='T', in_map_types={'T'}, candidate_types=set(),
        connectivity_qualified=True, morph_ok=False, is_deep=False,
        tier=None, mode='restrictive') == ('', False, True)

    # candidates: out-of-map, connectivity + morph qualified
    assert classify_category(
        target_bid=3, branch_pool={1}, in_map={1}, target_type='X',
        branch_target_type='T', in_map_types={'T'}, candidate_types=set(),
        connectivity_qualified=True, morph_ok=True, is_deep=False,
        tier=None, mode='restrictive')[0] == 'candidates'
    # same target, connectivity-qualified but morph-failed -> out of scope
    assert classify_category(
        target_bid=3, branch_pool={1}, in_map={1}, target_type='X',
        branch_target_type='T', in_map_types={'T'}, candidate_types=set(),
        connectivity_qualified=True, morph_ok=False, is_deep=False,
        tier=None, mode='restrictive') == ('', False, True)
    # not connectivity-qualified, untyped region -> out of scope, no fail
    assert classify_category(
        target_bid=3, branch_pool={1}, in_map={1}, target_type='X',
        branch_target_type='T', in_map_types={'T'}, candidate_types=set(),
        connectivity_qualified=False, morph_ok=False, is_deep=False,
        tier=None, mode='restrictive') == ('', False, False)

    # family: branch-local, this branch's target type, family mode
    assert classify_category(
        target_bid=9, branch_pool={1}, in_map={1}, target_type='T',
        branch_target_type='T', in_map_types={'T'}, candidate_types=set(),
        connectivity_qualified=False, morph_ok=False, is_deep=False,
        tier=None, mode='family')[0] == 'family'
    # relative: a candidate type OUTSIDE the map
    assert classify_category(
        target_bid=9, branch_pool={1}, in_map={1}, target_type='CB4091',
        branch_target_type='T', in_map_types={'T'},
        candidate_types={'CB4091'}, connectivity_qualified=False,
        morph_ok=False, is_deep=False, tier=None, mode='family')[0] \
        == 'relative'
    # examinees: deep window, aggressive only
    assert classify_category(
        target_bid=9, branch_pool={1}, in_map={1}, target_type='X',
        branch_target_type='T', in_map_types={'T'}, candidate_types=set(),
        connectivity_qualified=False, morph_ok=True, is_deep=True,
        tier=None, mode='aggressive')[0] == 'examinees'
    assert classify_category(
        target_bid=9, branch_pool={1}, in_map={1}, target_type='X',
        branch_target_type='T', in_map_types={'T'}, candidate_types=set(),
        connectivity_qualified=False, morph_ok=True, is_deep=True,
        tier=None, mode='family') == ('', False, False)


def test_rev311_widen_retired_pool_unchanged():
    """Revision 3.12: chain-aware POOL widening is RETIRED.  `_refine_pair_branch`
    keeps only the selected chain's pools regardless of mode, so the tier
    is identical across restrictive/family/aggressive (alternative-chain
    members are labelled `family` by the category classifier instead)."""
    from comparison.mapping_validation import MappingValidator
    from comparison.mapping_validation import TypePair as TP
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='flywire_FAFB_v783',
                                    target_dataset='male-cns:v1.0',
                                    query_types=['s-CPDN3C'],
                                    visualize=False, morph_enabled=False,
                                    validation_mode='family')
    v.notes = []
    v._source_type_counts = {'s-CPDN3C': 12}
    v._source_add_counts = {}

    class FakeMapper:
        def get_mapping_decision(self, t, src, tgt):
            return {'status': 'mapped', 'target_type': 'SMP220',
                    'target_types': ['SMP220']}

        def get_type_bridges(self, parent, src, tgt, max_bridges=0):
            out = []
            for linker in ('CB1709', 'CB2438', 'CB3767'):
                out.append([
                    {'dataset': src, 'column': 'type', 'value': parent},
                    {'dataset': src, 'column': 'additional_type(s)',
                     'value': linker, 'via': parent},
                    {'dataset': tgt, 'column': 'flywireType',
                     'value': 'SMP220'}])
            return out

    v.mapper = FakeMapper()
    pair = TP('flywire_FAFB_v783', 's-CPDN3C', [1], 'male-cns:v1.0',
              'SMP220', [2])

    import ui.neuron_index as ni
    orig = ni.resolve_prioritized_bridge_pool

    def spy(src, tgt, chains, stype, ttype):
        groups = {('CB1709', 'S'): [11, 12, 13, 14, 15, 16],
                  ('CB1709', 'T'): [201, 202, 203, 204, 205, 206],
                  ('CB2438', 'S'): [21, 22, 23, 24],
                  ('CB2438', 'T'): [251, 252, 253, 254],
                  ('CB3767', 'S'): [31, 32],
                  ('CB3767', 'T'): [331, 332]}
        chain = chains[0]
        linker = chain[1]['value']
        return {'resolution_status': 'supported',
                'selected_chain': chain,
                'source_body_ids': groups[(linker, 'S')],
                'target_body_ids': groups[(linker, 'T')],
                'source_basis': 'linker rows',
                'target_basis': 'linker rows',
                'source_type_total': 12, 'target_type_total': 12}

    ni.resolve_prioritized_bridge_pool = spy
    try:
        ok = v._refine_pair_branch(pair)
    finally:
        ni.resolve_prioritized_bridge_pool = orig
    assert ok is True
    # only the selected chain (CB1709 here) is applied — no widening.
    assert pair.pool_widen_added_targets == []
    assert pair.pool_widen_added_sources == []
    assert pair.source_pool == sorted(set(pair.source_pool))
    assert pair.target_pool == sorted(set(pair.target_pool))
    assert pair.pool_basis == 'linker rows'


def test_refine_rejects_wrong_endpoint_chain():
    """TMV-1 (2026-09-25 audit): `resolve_prioritized_bridge_pool` falls back
    to ANY supported chain of the source type when none ends at the requested
    target; a chain reaching a different type never narrowed THIS pair, so
    refinement must fail open to the full pools instead of recording
    provenance for — or substituting neurons of — another type (measured
    real shape: MCNS AVLP460 -> AVLP460b whose chains end at AVLP460)."""
    from comparison.mapping_validation import MappingValidator
    from comparison.mapping_validation import TypePair as TP
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='male-cns:v1.0',
                                    target_dataset='flywire_FAFB_v783',
                                    query_types=['AVLP460'], visualize=False,
                                    morph_enabled=False)
    v.notes = []

    class FakeMapper:
        def get_type_bridges(self, parent, src, tgt, max_bridges=0):
            return [[{'dataset': src, 'column': 'type', 'value': parent},
                     {'dataset': tgt, 'column': 'type', 'value': parent}]]

    v.mapper = FakeMapper()
    import ui.neuron_index as ni
    orig = ni.resolve_prioritized_bridge_pool

    def wrong_endpoint(src, tgt, chains, stype, ttype):
        return {'resolution_status': 'supported',
                'selected_chain': chains[0],
                'source_body_ids': [1, 2],
                'target_body_ids': [99, 98],
                'source_basis': 'linker rows',
                'target_basis': 'linker rows'}

    ni.resolve_prioritized_bridge_pool = wrong_endpoint
    try:
        pair = TP('male-cns:v1.0', 'AVLP460', [1, 2, 3],
                  'flywire_FAFB_v783', 'AVLP460b', [7, 8])
        ok = v._refine_pair_branch(pair)
    finally:
        ni.resolve_prioritized_bridge_pool = orig
    assert ok is False
    assert pair.source_pool == [1, 2, 3]   # untouched — fail open
    assert pair.target_pool == [7, 8]
    assert pair.selected_chain == []        # no misleading provenance
    assert pair.pool_basis == 'full population'


def test_summary_verdicts_resync_after_demotion():
    """TMV-3: stage 5's AUC demotion rewrites row verdicts AFTER _summary
    froze its counters, so pair_summary.csv disagreed with
    validation_results.csv (matrix-measured: banc l-LNv -> l-LNv summary
    strong=2/verified=6 vs 8/8 plain-verified rows).  The resync patches the
    SHARED summary dicts from the same row objects the demotion mutated."""
    from comparison.mapping_validation import MappingValidator
    v = MappingValidator.__new__(MappingValidator)
    summary = {'query': 'circadian_clock', 'source_type': 'l-LNv',
               'target_type': 'l-LNv',
               'verdict_verified_strong': 2, 'verdict_verified': 2,
               'verdict_borderline': 0, 'verdict_unmatched': 0,
               'verdict_skipped': 0}
    rows = [{'verdict': 'verified_strong'},
            {'verdict': 'verified'},
            {'verdict': 'verified'},
            {'verdict': 'unmatched'}]
    # stage 5 demotes one strong row below the AUC threshold
    rows[0]['verdict'] = 'verified'
    rows[0]['flags'] = 'morph_below_threshold'
    per_pair_res = {('circadian_clock', 'l-LNv', 'l-LNv'):
                    {'summary': summary, 'rows': rows}}
    v._resync_summary_verdicts(per_pair_res)
    assert summary['verdict_verified_strong'] == 0
    assert summary['verdict_verified'] == 3
    assert summary['verdict_unmatched'] == 1
    assert summary['verdict_borderline'] == 0
    assert summary['verdict_skipped'] == 0


def test_morph_stores_recorded_per_pass_not_overwritten():
    """WIP-D (2026-09-25): a pooling run runs two morph passes, and the old
    flat `morph_stores` assignment let the pooling record erase the
    supervised one.  Both passes are now keyed side by side, from one
    memoized store identity (the walk stats every skeleton file — twice per
    pass was pure cost)."""
    import comparison.mapping_validation as mv
    from comparison.mapping_validation import MappingValidator
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='flywire_FAFB_v783',
                                    target_dataset='banc_v888',
                                    query_types=['x'], visualize=False)
    fp = {}
    v._fingerprint = lambda: fp
    calls = []
    orig = mv._morph_store_identity

    def counted(dataset, project_root=None):
        calls.append(dataset)
        return {'vector_cache': {'bytes': 1}, 'dataset': dataset}

    mv._morph_store_identity = counted
    try:
        v._record_morph_stores()                # supervised stage 5
        v._record_morph_stores('pooling')       # pooling qualification
    finally:
        mv._morph_store_identity = orig
    assert set(fp['morph_stores']) == {'supervised', 'pooling'}
    # one identity walk per DATASET for the whole run, not per pass
    assert sorted(calls) == ['banc_v888', 'flywire_FAFB_v783']


def test_mapper_gap_report_in_readme():
    """Mapper-gap evidence is never silently absent — after the slim
    README (template `_plan/tmvev-run-report-template.md`, D1), the
    block moved out of README.txt into the set_coverage payload + the
    per-run report (report.html §4 + user_warning_notes.txt)."""
    from collections import Counter
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(
        source_dataset='flywire_FAFB_v783', target_dataset='male-cns:v1.0',
        query_types=['x'], visualize=False)
    v.notes = ['log line']
    v._mapper_gap_types = Counter({'ME_unclear': 2, 'SMP999': 1})
    v._mapper_gap_untyped = 3
    # _write_readme writes to self.run_dir — redirect to tmp
    import tempfile
    from pathlib import Path as _Path
    tmp = _Path(tempfile.mkdtemp())
    v.run_dir = tmp
    summaries = [{'source_type': 'A', 'target_type': 'B',
                  'mapping_status': 'mapped', 'pool_basis': 'linker rows',
                  'source_pool': 1, 'target_pool': 1, 'matched': 1,
                  'gap': 0, 'gap_ratio': 0.0, 'gap_triggered': False,
                  'source_type_total': 0, 'target_type_total': 0}]
    v._write_readme(summaries, None)
    text = (v.run_dir / 'README.txt').read_text()
    # slim README contract: no mapper-gap block, the report carries it
    assert 'Mapper-gap evidence' not in text
    assert 'report.html' in text
    # the evidence lands in the set_coverage payload (additive key)
    payload = v._set_coverage_payload({'source': {'total_queried': 1},
                                       'target': {'holes': 0}})
    assert payload['mapper_gap']['types'] == {'ME_unclear': 2,
                                              'SMP999': 1}
    assert payload['mapper_gap']['untyped_rows'] == 3
    # and the report generator surfaces it (data → warnings → HTML)
    from comparison.mapping_validation_report import (
        collect_run_data,
        collect_warnings,
        build_report_document,
    )
    d = collect_run_data(tmp, mapper_gap_types=dict(v._mapper_gap_types),
                         mapper_gap_untyped=3)
    assert d['mapper_gap'] == {'ME_unclear': 2, 'SMP999': 1}
    warns = collect_warnings(d)
    assert any(w.startswith('[mapper-gap] ME_unclear: 2')
               for w in warns)
    assert any('(untyped): 3 row(s)' in w for w in warns)
    assert 'Mapper-gap evidence' in build_report_document(d)


def test_track_b_floor_calibration_with_mock_cache(monkeypatch):
    """Track B end-to-end with a mocked native vector cache: identical
    refs give baseline ~1.0, floor = baseline - margin; a candidate
    identical to the refs qualifies (pool_ref >= floor), an orthogonal
    one does not; the matched-only tier is preferred over the
    compromise tier; CSV audit columns exist on the rows."""
    import types
    import numpy as np
    import morphology
    from comparison.mapping_validation import MappingValidationConfig

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

    # vectors: refs 11/12 identical; candidate 21 identical; 22 orthogonal
    unit = np.zeros(8); unit[0] = 1.0
    orth = np.zeros(8); orth[7] = 1.0
    vecs = {11: unit, 12: unit, 21: unit, 22: orth}

    class FakeCache:
        def vectors_for(self, ids, compute_missing=True):
            X = np.stack([vecs[i] for i in ids])
            return X, [True] * len(ids), None

        def load(self):
            return {'whiten': np.eye(8)}

    monkeypatch.setattr(morphology, 'find_similar_dataset_cache_v2',
                        lambda ds, verbose=False, **k: FakeCache())
    monkeypatch.setattr(morphology, 'apply_whitening',
                        lambda w, X: X)
    monkeypatch.setattr(morphology, 'v2_similarity_matrix',
                        lambda A, B, weights: A @ B.T)
    monkeypatch.setattr(morphology, 'DEFAULT_V2_BLOCK_WEIGHTS',
                        {'all': 1.0}, raising=False)

    val_rows = [{'source_bodyId': 1, 'target_bodyId': 11,
                 'verdict': 'verified_strong', 'flags': ''}]
    sus = [
        # invader 21: matched-tier candidate (v2 vs query 0.2 < 0.25
        # threshold, but pool-ref 1.0 passes the binding floor)
        {'source_bodyId': 1, 'source_type': 'T', 'target_type': 'T',
         'ahead_target_bodyId': 21,
         'ahead_target_type': 'TX', 'invader_class': 'unmapped'},
        # invader 22: orthogonal -> pool-ref 0.0 -> fails
        {'source_bodyId': 1, 'source_type': 'T', 'target_type': 'T',
         'ahead_target_bodyId': 22,
         'ahead_target_type': 'TX', 'invader_class': 'unmapped'},
    ]
    # pool: 11 matched, 12 verified -> tier 'matched' (matched >= 2)
    # two matched neurons -> the matched-only tier (>= 2 rule)
    pool_detail = [
        {'target_bodyId': 11, 'category': 'matched',
         'best_source_bodyId': 1, 'source_type': 'T', 'target_type': 'T'},
        {'target_bodyId': 12, 'category': 'matched',
         'best_source_bodyId': 1, 'source_type': 'T', 'target_type': 'T'},
    ]
    v.annotate_invaders(sus, [], {})   # populates the audit columns
    stub_skeleton_store_all_cached(monkeypatch)
    out = v.run_morphology(val_rows, sus, [], pool_detail, deep_rows=[])
    assert out['pool_ref_tiers']['T->T'] == 'matched+verified'
    assert out['pool_ref_baselines']['T->T'] == pytest.approx(1.0)
    assert out['pool_ref_floors']['T->T'] == pytest.approx(0.95)
    by_bid = {r['ahead_target_bodyId']: r for r in sus}
    assert by_bid[21]['morph_pool_ref'] == pytest.approx(1.0)
    assert by_bid[22]['morph_pool_ref'] == pytest.approx(0.0)
    # CSV audit columns present on the rows (they become CSV headers)
    # annotation + morph columns are annotate/run_morphology-owned
    # (the size columns are validate_pair-owned, covered by the
    # deep-window and caliber tests)
    need = {'invader_class', 'invader_label', 'sibling_pool_of',
            'backward_status', 'backward_maps_to', 'fafb_home_count',
            'backward_home_real', 'alt_chain_of_parent',
            'morph_pool_ref', 'pool_ref_tier'}
    assert need <= set(sus[0].keys())


def test_backward_decision_hollow_home():
    """A mapper 'mapped' decision with a 0-count source home is hollow
    and must not earn backward_home_real (the CB4091 case)."""
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='flywire_FAFB_v783',
                                    target_dataset='male-cns:v1.0',
                                    query_types=['x'], visualize=False,
                                    morph_enabled=False)
    v.notes = []

    class FakeMapper:
        def get_mapping_decision(self, t, src, tgt):
            return {'status': 'mapped', 'target_type': t,
                    'target_types': [t]}

    v.mapper = FakeMapper()
    v._source_type_counts = {'SMP368': 3, 'CB4091': 0}
    v._source_add_counts = {}
    hollow = v._backward_decision('CB4091')
    real = v._backward_decision('SMP368')
    assert hollow['home_real'] is False and hollow['home_count'] == 0
    assert real['home_real'] is True and real['home_count'] == 3


# ---------------------------------------------------------------------------
# Revision 3.7 — frame disclosure in the calibration record
# ---------------------------------------------------------------------------


def _nesting_fixture(mode):
    """One source, one pool best, four borderline rows, twelve near-zero rows
    that only the wide window reaches, and TWO homologs that are borderline
    only in rank_union.

    The shape is the point. Under a SHARED per-source budget the wide window
    spends it on the near-zero rows before the rank_union loop is ever read,
    which is how a real run lost 110 (male-cns) / 262 (BANC) borderline rows
    to the wider mode. And if a row's BAND follows whichever metric's loop
    reaches it first, the wide loop can charge a rank_union-borderline row to
    the deep band and the second loop can no longer count it — that is 430,
    jaccard rank 6 and rank_union rank 4.
    """
    import pandas as pd
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(
        source_dataset='dsA', target_dataset='dsB', query_types=['T'],
        morph_enabled=False, visualize=False, validation_mode=mode,
        candidate_window=25, deep_cap=10)
    v.mapper = None
    v.notes = []

    class FakeProfiler:
        def get_profile(self, bid, dataset):
            return None

        def get_types_for_bodyids(self, bids, dataset):
            return {b: 'TX' for b in bids}

    v.profiler = FakeProfiler()
    pair = TypePair('dsA', 'T', [1], 'dsB', 'T', [11])
    rows = [{'target_bid': 11, 'jaccard': 0.90, 'jaccard_rank': 1,
             'rank_union': 0.90, 'rank_union_rank': 1}]
    # 301-304: tight by jaccard (ranks 2-5)
    for i, b in enumerate((301, 302, 303, 304)):
        rows.append({'target_bid': b, 'jaccard': 0.80 - 0.02 * i,
                     'jaccard_rank': 2 + i, 'rank_union': 0.30 - 0.02 * i,
                     'rank_union_rank': 12 + i})
    # 430: deep by jaccard AND early in that window, borderline by rank_union
    rows.append({'target_bid': 430, 'jaccard': 0.735, 'jaccard_rank': 6,
                 'rank_union': 0.35, 'rank_union_rank': 4})
    # 305-308: deep by jaccard
    for i, b in enumerate((305, 306, 307, 308)):
        rows.append({'target_bid': b, 'jaccard': 0.40 - 0.02 * i,
                     'jaccard_rank': 7 + i, 'rank_union': 0.20 - 0.02 * i,
                     'rank_union_rank': 16 + i})
    # 600-611: only inside the WIDE window (jaccard ranks 11-22), and each is
    # near-zero enough to be backdrop material
    for i, b in enumerate(range(600, 612)):
        rows.append({'target_bid': b, 'jaccard': 0.05 - 0.001 * i,
                     'jaccard_rank': 11 + i, 'rank_union': 0.05 - 0.001 * i,
                     'rank_union_rank': 30 + i})
    rows.append({'target_bid': 501, 'jaccard': 0.02, 'jaccard_rank': 23,
                 'rank_union': 0.02, 'rank_union_rank': 42})
    rows.append({'target_bid': 502, 'jaccard': 0.015, 'jaccard_rank': 24,
                 'rank_union': 0.015, 'rank_union_rank': 43})
    # 420: rank 3 by rank_union, last by jaccard — reachable only through
    # the second metric's loop
    rows.append({'target_bid': 420, 'jaccard': 0.008, 'jaccard_rank': 25,
                 'rank_union': 0.60, 'rank_union_rank': 3})
    # 440 against 620-631: one borderline row TIED on jaccard with twelve deep
    # rows, and the weakest rank_union of the tie. Chain order puts it last, so
    # a single loop that stops the moment its deep budget is full never reaches
    # it — which is exactly how the post-fix 2026-09-24 run still lost 63 of
    # family's 189 borderline pairs. Reading the bands as separate passes is
    # what makes it reach.
    rows.append({'target_bid': 440, 'jaccard': 0.15, 'jaccard_rank': 5,
                 'rank_union': 0.05, 'rank_union_rank': 60})
    for i, b in enumerate(range(620, 632)):
        rows.append({'target_bid': b, 'jaccard': 0.15,
                     'jaccard_rank': 6 + i, 'rank_union': 0.30 - 0.01 * i,
                     'rank_union_rank': 44 + i})
    scans = {1: pd.DataFrame(rows)}
    return v.validate_pair(
        pair, scans,
        {11: 'T', 420: 'TX', 430: 'TX', 440: 'TX', 501: 'TX', 502: 'TX',
         **{b: 'TX' for b in list(range(301, 309)) + list(range(600, 633))}})


def test_widening_the_mode_never_displaces_a_borderline_row():
    """The modes NEST, so aggressive may only ADD to what family reported.

    Two ways this broke, both measured on the 2026-09-24 `circadian_clock`
    ladder: the borderline and deep bands shared ONE per-source budget, so
    aggressive spent it on its 25-wide jaccard window and the rank_union loop
    was never read (110 of 189 pairs displaced on male-cns, 262 of 408 on
    BANC, their `relative` rows and their layered gap-fill entries with them —
    153 rows → 63); and a row's band followed whichever metric reached it
    first, so the wide loop could charge a rank_union-borderline row to the
    deep band. Here 420 is the first victim and 430 the second.
    """
    fam = _nesting_fixture('family')
    agg = _nesting_fixture('aggressive')
    fam_rows = {(d['ahead_target_bodyId'], d['candidate_source'])
                for d in fam['deep']}
    agg_rows = {(d['ahead_target_bodyId'], d['candidate_source'])
                for d in agg['deep']}
    # family reads only the borderline window: the four tight rows, the row
    # tied at its boundary, and the two caught through their rank_union
    assert {b for b, _ in fam_rows} == {301, 302, 303, 304, 440, 430, 420}
    assert {src for _, src in fam_rows} == {'top_window'}
    # aggressive keeps every one of them, IN THE SAME BAND
    assert fam_rows <= agg_rows, f'displaced: {sorted(fam_rows - agg_rows)}'
    for b in (440, 430, 420):
        assert (b, 'top_window') in agg_rows, f'{b} lost its band'
    # and everything it adds is deep-band, under its own budget
    added = {b for b, src in agg_rows if src == 'deep_window'}
    assert added and added.isdisjoint({b for b, _ in fam_rows})
    assert sum(1 for _b, src in agg_rows if src == 'deep_window') == 10
    assert len(fam['deep']) == 7
    assert len(agg['deep']) == 17
    assert fam['summary']['deep_candidates'] == 7
    assert agg['summary']['deep_candidates'] == 17


def test_the_null_backdrop_is_the_same_sample_in_every_mode():
    """The no-homology backdrop gates every null-kind bar, so which rows a
    mode's own window happened to consume must not decide it.

    It used to: the sample excluded `seen_deep`, which accumulates the
    window, so the p95 moved 0.143561 → 0.143123 → 0.150580 across the ladder
    on male-cns and re-graded any row sitting inside that drift (one BANC pair
    holds similarity 0.2642521 in all three runs and reads `candidates` /
    out-of-scope / `candidates` as its bar moves).  600-611 are the same
    test: near-zero connectivity, and aggressive's wide window also reaches
    them.
    """
    fam = _nesting_fixture('family')
    agg = _nesting_fixture('aggressive')
    pick = lambda res: sorted((r['source_bodyId'], r['ahead_target_bodyId'])
                              for r in res['null'])
    assert pick(fam) == pick(agg)
    # every backdrop slot is filled from the same eligible set in both modes
    assert {r['ahead_target_bodyId'] for r in fam['null']} == {
        420, 501, 502, 600, 601}
    assert {r['ahead_target_bodyId'] for r in agg['null']} == {
        420, 501, 502, 600, 601}


def test_deep_window_off_by_default():
    """Rev 3.9 scope: without aggressive_expansion the deep window is
    inert — no deep rows even when below-pool homologs exist."""
    import pandas as pd
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(
        source_dataset='dsA', target_dataset='dsB', query_types=['T'],
        morph_enabled=False, visualize=False)
    v.mapper = None
    v.notes = []

    class FakeProfiler:
        def get_profile(self, bid, dataset):
            return None

        def get_types_for_bodyids(self, bids, dataset):
            return {b: 'TX' for b in bids}

    v.profiler = FakeProfiler()
    pair = TypePair('dsA', 'T', [1], 'dsB', 'T', [11])
    scans = {1: pd.DataFrame([
        {'target_bid': 11, 'rank_union': 0.90, 'rank_union_rank': 1,
         'jaccard': 0.90, 'jaccard_rank': 1},
        {'target_bid': 302, 'rank_union': 0.55, 'rank_union_rank': 4,
         'jaccard': 0.50, 'jaccard_rank': 4},
    ])}
    res = v.validate_pair(pair, scans, {11: 'T', 302: 'TX'})
    assert res['deep'] == []


def test_the_discovery_window_carries_candidates_a_perfect_pool_hides():
    """The CB4091 shape (r16 -> r18, measured on real data).

    When a pool member holds the global top-1 on BOTH metrics, nothing can
    beat the pool's best position, so the invader window is empty by
    construction — r18 published jaccard rank 1 on 190 of 223 rows.  The
    candidate feed ran through that window, and rule 5 of
    `classify_category` seeds the whole `relative` bin from the candidate
    TYPES, so a well-validated branch reported the least: examinees
    168 -> 118, candidate types 5 -> 3, `relatives.csv` 39 rows -> 1.

    Family MODE therefore reads a WINDOW instead: the top-`rank_top_k` of
    EACH metric.  302 sits below the pool best on both and is caught on the
    jaccard side; 303 is only inside the rank_union window, which is what
    makes the feed metric-parallel rather than a relabelled single-metric
    scan.
    """
    import pandas as pd
    from comparison.mapping_validation import classify_category
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(
        source_dataset='dsA', target_dataset='dsB', query_types=['T'],
        morph_enabled=False, visualize=False, validation_mode='family',
        rank_top_k=5)
    v.mapper = None
    v.notes = []

    class FakeProfiler:
        def get_profile(self, bid, dataset):
            return None

        def get_types_for_bodyids(self, bids, dataset):
            return {b: 'TX' for b in bids}

    v.profiler = FakeProfiler()
    pair = TypePair('dsA', 'T', [1], 'dsB', 'T', [11])
    scans = {1: pd.DataFrame([
        {'target_bid': 11, 'rank_union': 0.90, 'rank_union_rank': 1,
         'jaccard': 0.90, 'jaccard_rank': 1},
        {'target_bid': 302, 'rank_union': 0.10, 'rank_union_rank': 6,
         'jaccard': 0.24, 'jaccard_rank': 3},
        {'target_bid': 303, 'rank_union': 0.30, 'rank_union_rank': 2,
         'jaccard': 0.05, 'jaccard_rank': 9},
    ])}
    res = v.validate_pair(pair, scans,
                          {11: 'T', 302: 'CB4091', 303: 'SMP223'})
    # the pool's best position is rank 1 on both metrics: no invader exists
    assert res['suspicious'] == []
    assert res['rows'][0]['verdict'] == 'verified_strong'
    by_bid = {d['ahead_target_bodyId']: d for d in res['deep']}
    assert sorted(by_bid) == [302, 303]
    assert by_bid[302]['ahead_metric'] == 'jaccard'
    assert by_bid[303]['ahead_metric'] == 'rank_union'
    assert {d['candidate_source'] for d in res['deep']} == {'top_window'}
    # ...and a window row is connectivity evidence, so it reaches
    # `candidates` (which is what seeds its type into `relative`).
    assert classify_category(
        target_bid=302, branch_pool={11}, in_map={11}, target_type='CB4091',
        branch_target_type='T', in_map_types={'T'}, candidate_types=set(),
        connectivity_qualified=True, morph_ok=True, is_deep=True,
        tier=None, mode='family') == ('candidates', True, False)
    # the pre-fix reading: not connectivity-qualified, and its type is not
    # yet a candidate type, so the row carried no bin at all.
    assert classify_category(
        target_bid=302, branch_pool={11}, in_map={11}, target_type='CB4091',
        branch_target_type='T', in_map_types={'T'}, candidate_types=set(),
        connectivity_qualified=False, morph_ok=True, is_deep=True,
        tier=None, mode='family') == ('', False, False)


def test_annotate_invaders_alternate_chain(monkeypatch):
    """An invader whose type is the final hop of a NON-selected chain of
    its own parent (SMP223 via CB3612 under s-CPDN3D, where the CB3508
    chain was selected) is labeled alternate-chain, not plain backward."""
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='flywire_FAFB_v783',
                                    target_dataset='male-cns:v1.0',
                                    query_types=['s-CPDN3D'],
                                    visualize=False, morph_enabled=False)
    v.notes = []

    class FakeMapper:
        def get_mapping_decision(self, t, src, tgt):
            return {'status': 'mapped', 'target_type': 's-CPDN3D',
                    'target_types': ['s-CPDN3D']}

        def get_type_bridges(self, parent, src, tgt, max_bridges=0):
            return [
                [{'dataset': src, 'column': 'type', 'value': parent},
                 {'dataset': src, 'column': 'additional_type(s)',
                  'value': 'CB3508', 'via': parent},
                 {'dataset': tgt, 'column': 'flywireType',
                  'value': 'SMP223'}],
                [{'dataset': src, 'column': 'type', 'value': parent},
                 {'dataset': src, 'column': 'additional_type(s)',
                  'value': 'CB3612', 'via': parent},
                 {'dataset': tgt, 'column': 'flywireType',
                  'value': 'SMP223'}],
            ]

    v.mapper = FakeMapper()
    v.pairs = []
    v._source_type_counts = {'s-CPDN3D': 37}
    v._source_add_counts = {}
    # the selected chain for the rendered branch: CB3508 -> SMP223
    pair = TypePair('flywire_FAFB_v783', 's-CPDN3D', [1],
                    'male-cns:v1.0', 'SMP223', [2])
    pair.linkers = [{'column': 'additional_type(s)', 'raw_value': 'CB3508',
                     'home': 'flywire_FAFB_v783'}]
    v.pairs = [pair]
    sus = [{'source_type': 's-CPDN3D', 'target_type': 'SMP222',
            'ahead_target_bodyId': 61430, 'ahead_target_type': 'SMP223'}]
    v.annotate_invaders(sus, [], {})
    assert sus[0]['invader_class'] == 'alternate-chain'
    assert 'CB3612' in sus[0]['invader_label']
    assert 'SMP223' in sus[0]['invader_label']
    assert sus[0]['alt_chain_of_parent'] == 'CB3612'


def test_run_morphology_discloses_score_frame(monkeypatch):
    import morphology
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(
        source_dataset='flywire_FAFB_v783', target_dataset='male-cns:v1.0',
        query_types=['T'], visualize=False)
    v.notes = []
    v.log = lambda m='': None
    monkeypatch.setattr(morphology, 'enrich_homolog_results',
                        lambda df, s, t, verbose=False: df.assign(
                            morph_v2_similarity=0.5, morph_nblast=0.4))
    stub_skeleton_store_all_cached(monkeypatch)
    out = v.run_morphology(
        [{'source_bodyId': 1, 'target_bodyId': 11,
          'verdict': 'verified_strong', 'flags': ''}], [], [], [])
    sf = out['score_frame']
    assert 'target' in sf['track_a'] and 'transformed' in sf['track_a']
    assert 'native' in sf['track_b'] and 'no transforms' in sf['track_b']
    assert 'source' in sf['visualization']


def test_overlay_source_metadata_aligns_rows_to_bids(tmp_path):
    """The render loop reads overlay metadata by neuron position; a
    table-ordered frame mislabeled legend leaves whenever the table order
    differed from the layer order (the R5 61430/50274 bug)."""
    from types import SimpleNamespace

    from visualize_skeleton import VisualizeSkeleton

    ds = 'male-cns_v1_0'
    table_dir = tmp_path / 'datasets' / ds
    table_dir.mkdir(parents=True)
    # table in SCRAMBLED order, with one unrelated neuron and without
    # one of the requested bids
    pd.DataFrame({
        'bodyId': ['92109', '18706', '99999', '18326', '18009'],
        'type': ['CL125'] * 5,
        'instance': ['i_92109', 'i_18706', 'i_x', 'i_18326', 'i_18009'],
    }).to_csv(table_dir / f'{ds}_allneurons_neuron_df.csv', index=False)

    fake_self = SimpleNamespace(script_path=str(tmp_path))
    bids = [18706, 92109, 18326, 18009, 55555]
    rows = VisualizeSkeleton._overlay_source_metadata(
        fake_self, 'male-cns:v1.0', bids)

    assert rows is not None
    # one row per requested bid, in the caller's order
    assert [int(b) for b in rows['bodyId']] == bids
    assert rows['instance'].tolist()[:4] == [
        'i_18706', 'i_92109', 'i_18326', 'i_18009']
    # a bid missing from the table keeps its row (NaN metadata) so the
    # positions of the following rows never shift
    assert pd.isna(rows['instance'].tolist()[4])
    assert pd.isna(rows['type'].tolist()[4])


# ---------------------------------------------------------------------------
# Issue 1 — per-bodyId trace identity + hover (backend helper)
# ---------------------------------------------------------------------------

def test_tree_leaf_trace_identity_renames_and_embeds_bodyid():
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    from visualize_skeleton import VisualizeSkeleton

    class _Vol:
        id = 720575940647731252  # FAFB-sized: int64-exact only

    trace = go.Scatter3d(x=[1], y=[2], z=[3], mode='lines')
    ok = VisualizeSkeleton._apply_tree_leaf_trace_identity(
        trace, '720575940647731252_CL125_L', [_Vol()], 0, '7205759406477')
    assert ok is True
    assert trace.name == '720575940647731252_CL125_L'
    assert list(trace.customdata) == [720575940647731252]
    assert 'bodyId 720575940647731252' in trace.hovertemplate
    # group-level legend anchoring is untouched
    assert trace.legendgroup is None


def test_tree_leaf_trace_identity_tolerates_bad_index_and_id():
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    from visualize_skeleton import VisualizeSkeleton

    trace = go.Scatter3d(x=[1], y=[2], z=[3], mode='lines')
    # source_index beyond neuron_vols -> rename without a bodyId hover
    ok = VisualizeSkeleton._apply_tree_leaf_trace_identity(
        trace, '999_T_X', [], 5, '999')
    assert ok is False
    assert trace.name == '999_T_X'
    assert trace.customdata is None
    # un-int-able neuron id -> no bodyId hover either
    trace2 = go.Scatter3d(x=[1], y=[2], z=[3], mode='lines')
    ok2 = VisualizeSkeleton._apply_tree_leaf_trace_identity(
        trace2, 'x_y_z', [object()], 0, None)
    assert ok2 is False
    assert trace2.name == 'x_y_z'


# ===================================================================
# Revision 3.12 — the category partition (plan §2b T11 acceptance gate)
# ===================================================================

def _rev312_validator(mode='restrictive'):
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(
        source_dataset='flywire_FAFB_v783', target_dataset='male-cns:v1.0',
        query_types=['s-CPDN3C', 's-CPDN3D'], visualize=False,
        morph_enabled=True, validation_mode=mode)
    v.notes = []
    v.pairs = []
    # morph qualification is decided by the floor when present; with no
    # floors and no thresholds the Track-A bar is None -> a row is
    # qualified only when given morph_v2_similarity >= bar.  Tests set
    # the bar explicitly.
    v._pool_ref_floors = {}
    v._candidate_thresholds = {}
    v._track_a_null_bar = 0.5
    return v


def _rev312_branch(rows=None, pool=None, cats=None, pool_detail=None):
    return {
        'rows': rows or [],
        'suspicious': [],
        'deep': [],
        'fills': [],
        'target_categories': cats or {},
        'pool_detail': pool_detail or [],
        '_pool_set': sorted(pool or []),
    }


def test_rev312_finalize_partition_complete_and_exclusive():
    """Invariant 1/9: every in-scope evidence row gets exactly one
    category; an out-of-scope row is flagged, never silently blank."""
    v = _rev312_validator('restrictive')
    per_pair = {
        ('s-CPDN3C', 'T1'): _rev312_branch(
            pool=[101, 102], cats={101: 'matched', 102: 'verified'},
            pool_detail=[
                {'source_type': 's-CPDN3C', 'target_type': 'T1',
                 'target_bodyId': 101, 'category': 'matched'},
                {'source_type': 's-CPDN3C', 'target_type': 'T1',
                 'target_bodyId': 102, 'category': 'verified'}]),
        ('s-CPDN3D', 'T2'): _rev312_branch(
            pool=[201], cats={201: 'borderline'},
            pool_detail=[
                {'source_type': 's-CPDN3D', 'target_type': 'T2',
                 'target_bodyId': 201, 'category': 'borderline'}]),
    }
    # 201 is an in-map target of the query seen in branch T1 -> sibling
    # (admitted: connectivity + morph); 301 is out-of-map, qualified ->
    # candidates; 401 is out-of-map, connectivity-qualified but morph-
    # failed -> out of scope.
    sus = [
        {'source_type': 's-CPDN3C', 'target_type': 'T1',
         'source_bodyId': 1, 'ahead_target_bodyId': 201,
         'ahead_target_type': 'T2', 'morph_v2_similarity': 0.9,
         'backward_maps_to': ''},
        {'source_type': 's-CPDN3C', 'target_type': 'T1',
         'source_bodyId': 1, 'ahead_target_bodyId': 301,
         'ahead_target_type': 'CB4091', 'morph_v2_similarity': 0.9,
         'backward_maps_to': 's-CPDN3C', 'backward_home_real': True},
        {'source_type': 's-CPDN3C', 'target_type': 'T1',
         'source_bodyId': 1, 'ahead_target_bodyId': 401,
         'ahead_target_type': 'R1-R6', 'morph_v2_similarity': 0.1,
         'backward_maps_to': ''},
    ]
    v.finalize_categories(per_pair, sus, [], [], per_pair[
        ('s-CPDN3C', 'T1')]['pool_detail'] + per_pair[
        ('s-CPDN3D', 'T2')]['pool_detail'])
    by_bid = {int(r['ahead_target_bodyId']): r for r in sus}
    assert by_bid[201]['category'] == 'sibling'
    assert by_bid[301]['category'] == 'candidates'
    assert by_bid[301]['candidate_annotation'] == 'CB4091>s-CPDN3C'
    assert by_bid[401]['category'] == ''
    assert by_bid[401]['in_scope'] is False
    assert by_bid[401]['morph_failed'] is True
    # candidates count toward both fills; sibling does not.
    assert by_bid[301]['counts_toward_restrictive_fill'] is True
    assert by_bid[201]['counts_toward_restrictive_fill'] is False


def test_rev312_family_relative_type_disjoint():
    """Invariant 3: family (branch target type) and relative (candidate
    types OUTSIDE the map) are disjoint by type; family mode only."""
    v = _rev312_validator('family')
    v._bodyids_of_type = lambda t: {'T1': [101, 150], 'CB4091':
                                    [301, 302], 'T2': [201]}.get(t, [])
    per_pair = {
        ('s-CPDN3C', 'T1'): _rev312_branch(
            pool=[101], cats={101: 'matched'},
            pool_detail=[{'source_type': 's-CPDN3C', 'target_type': 'T1',
                          'target_bodyId': 101, 'category': 'matched'}]),
        ('s-CPDN3D', 'T2'): _rev312_branch(
            pool=[201], cats={201: 'matched'},
            pool_detail=[{'source_type': 's-CPDN3D', 'target_type': 'T2',
                          'target_bodyId': 201, 'category': 'matched'}]),
    }
    sus = [{'source_type': 's-CPDN3C', 'target_type': 'T1',
            'source_bodyId': 1, 'ahead_target_bodyId': 301,
            'ahead_target_type': 'CB4091', 'morph_v2_similarity': 0.9,
            'backward_maps_to': ''}]
    rows = v.finalize_categories(
        per_pair, sus, [], [], per_pair[('s-CPDN3C', 'T1')]['pool_detail']
        + per_pair[('s-CPDN3D', 'T2')]['pool_detail'])
    fam = {(r['source_type'], r['target_type'], r['ahead_target_bodyId'])
           for r in v._family_rows}
    rel = {(r['source_type'], r['target_type'], r['ahead_target_bodyId'])
           for r in v._relative_rows}
    # family is branch-local: T1's out-map bodyId 150 appears only under
    # the T1 branch (branch-local family).
    assert ('s-CPDN3C', 'T1', 150) in fam
    # 101 is in-map -> never family; CB4091 (out-of-map) -> relative.
    assert all(b != 101 for (_s, _t, b) in fam)
    assert all(t == 'T1' for (_s, t, _b) in fam)
    assert ('s-CPDN3C', 'T1', 302) in rel or ('s-CPDN3D', 'T2', 302) in rel
    # T2 (a different in-map type) is family under the T2 branch, not
    # relative.
    assert all(b not in (301, 302) for (_s, _t, b) in
               {(s, t, b) for (s, t, b) in fam if b in (301, 302)})
    # rows returned are the dedup rows
    assert rows and all('dedup_category' in r for r in rows)


def test_rev312_mode_nesting_superset():
    """Invariant 4: restrictive-labeled set ⊆ family ⊆ aggressive, and a
    shared neuron keeps the identical category."""
    per_pair = {
        ('s', 'T1'): _rev312_branch(
            pool=[1], cats={1: 'matched'},
            pool_detail=[{'source_type': 's', 'target_type': 'T1',
                          'target_bodyId': 1, 'category': 'matched'}]),
    }

    def run(mode):
        v = _rev312_validator(mode)
        v._bodyids_of_type = lambda t: {'T1': [1, 2]}.get(t, [])
        sus = [{'source_type': 's', 'target_type': 'T1',
                'source_bodyId': 1, 'ahead_target_bodyId': 300,
                'ahead_target_type': 'X', 'morph_v2_similarity': 0.9,
                'backward_maps_to': ''}]
        v.finalize_categories(per_pair, sus, [], [], per_pair[
            ('s', 'T1')]['pool_detail'])
        cats = {int(r['ahead_target_bodyId']): r['category'] for r in sus}
        # include the family members too
        for r in v._family_rows:
            cats.setdefault(int(r['ahead_target_bodyId']), r['category'])
        return cats

    r = run('restrictive')
    f = run('family')
    a = run('aggressive')
    # shared neurons keep identical labels
    for bid, cat in r.items():
        assert f.get(bid) == cat, (bid, cat, f.get(bid))
        assert a.get(bid) == cat, (bid, cat, a.get(bid))
    # family adds the family member 2
    assert 2 not in r
    assert f.get(2) == 'family'
    assert a.get(2) == 'family'


def test_rev312_dedup_precedence_and_dup_flag():
    """Invariant 6: dedup precedence tier > sibling > family > candidates
    > relative (user 2026-09-17: family outranks candidates — an
    unmapped member of an already-mapped type keeps its type identity);
    (dup) flags non-sibling repeats only."""
    v = _rev312_validator('family')
    v._bodyids_of_type = lambda t: []
    # 201 is a sibling under T1 and a tier (matched) under T2 -> the tier
    # wins in the rollup.
    per_pair = {
        ('s-CPDN3C', 'T1'): _rev312_branch(
            pool=[101], cats={101: 'matched'},
            pool_detail=[{'source_type': 's-CPDN3C', 'target_type': 'T1',
                          'target_bodyId': 101, 'category': 'matched'}]),
        ('s-CPDN3D', 'T2'): _rev312_branch(
            pool=[201], cats={201: 'matched'},
            pool_detail=[{'source_type': 's-CPDN3D', 'target_type': 'T2',
                          'target_bodyId': 201, 'category': 'matched'}]),
    }
    sus = [{'source_type': 's-CPDN3C', 'target_type': 'T1',
            'source_bodyId': 1, 'ahead_target_bodyId': 201,
            'ahead_target_type': 'T2', 'morph_v2_similarity': 0.9,
            'backward_maps_to': ''}]
    rows = v.finalize_categories(
        per_pair, sus, [], [], [per_pair[('s-CPDN3C', 'T1')]['pool_detail'][0],
                                per_pair[('s-CPDN3D', 'T2')]['pool_detail'][0]])
    by = {r['target_bodyId']: r for r in rows}
    assert by[201]['dedup_category'] == 'matched'   # tier beats sibling
    assert by[201]['dup'] is False                  # tier row, not sibling-tagged


def test_rev312_dedup_candidates_outranks_family():
    """Restored precedence (user 2026-09-17): candidates > family in the
    dedup rollup — the precedence serves the gap-fill accounting, so a
    family-material bodyId that ALSO carries candidate evidence rolls up
    as `candidates` (restrictive fill). The family category is reported
    COMPLETE via family_material + family_candidates.csv (the family
    enumeration is independent of the dedup rank), so nothing is hidden."""
    v = _rev312_validator('family')
    # T1's out-of-map type population: 301 carries candidate evidence in
    # branch T1; the same bodyId is family material under T2 (its type).
    v._bodyids_of_type = lambda t: {'T1': [101, 102],
                                    'T2': [201, 301]}.get(t, [])
    per_pair = {
        ('s-CPDN3C', 'T1'): _rev312_branch(
            pool=[101, 102], cats={101: 'matched', 102: 'verified'},
            pool_detail=[
                {'source_type': 's-CPDN3C', 'target_type': 'T1',
                 'target_bodyId': 101, 'category': 'matched'},
                {'source_type': 's-CPDN3C', 'target_type': 'T1',
                 'target_bodyId': 102, 'category': 'verified'}]),
        ('s-CPDN3D', 'T2'): _rev312_branch(
            pool=[201], cats={201: 'matched'},
            pool_detail=[
                {'source_type': 's-CPDN3D', 'target_type': 'T2',
                 'target_bodyId': 201, 'category': 'matched'}]),
    }
    sus = [
        # 301: candidate evidence under T1 ...
        {'source_type': 's-CPDN3C', 'target_type': 'T1',
         'source_bodyId': 1, 'ahead_target_bodyId': 301,
         'ahead_target_type': 'T2', 'morph_v2_similarity': 0.9,
         'backward_maps_to': ''},
        # 302: candidate evidence only, never family material
        {'source_type': 's-CPDN3C', 'target_type': 'T1',
         'source_bodyId': 1, 'ahead_target_bodyId': 302,
         'ahead_target_type': 'CB4091', 'morph_v2_similarity': 0.9,
         'backward_maps_to': ''},
    ]
    rows = v.finalize_categories(
        per_pair, sus, [], [],
        [per_pair[('s-CPDN3C', 'T1')]['pool_detail'][0],
         per_pair[('s-CPDN3D', 'T2')]['pool_detail'][0]])
    by = {r['target_bodyId']: r for r in rows}
    # dual-evidence bodyId rolls up candidates (fill accounting) ...
    assert by[301]['dedup_category'] == 'candidates'
    assert by[301]['counts_toward_restrictive_fill'] is True
    assert by[301]['counts_toward_family_fill'] is True
    # ... while the family enumeration still lists it (map structure)
    fam_bids = {r['ahead_target_bodyId'] for r in v._family_rows}
    assert '301' in fam_bids or 301 in fam_bids
    # candidate-only bodyId stays candidates
    assert by[302]['dedup_category'] == 'candidates'
    assert by[302]['counts_toward_restrictive_fill'] is True


def test_rev312_csv_invariants_acceptance(tmp_path):
    """T11 acceptance gate on the EXPORTED CSVs: no blank category among
    in-scope rows; in-pool pairing classified both directions; counts are
    qualification-gated; the dedup export is bodyId-unique with the right
    precedence.  Synthetic data (the real male-cns target resolution needs
    a NeuPrint client unavailable offline)."""
    import pandas as pd
    v = _rev312_validator('family')
    v._bodyids_of_type = lambda t: {'T1': [101, 150],
                                    'CB4091': [301, 302]}.get(t, [])
    v.run_dir = tmp_path
    per_pair = {
        ('s-CPDN3C', 'T1'): _rev312_branch(
            pool=[101, 102], cats={101: 'matched', 102: 'verified'},
            pool_detail=[
                {'source_type': 's-CPDN3C', 'target_type': 'T1',
                 'target_bodyId': 101, 'category': 'matched'},
                {'source_type': 's-CPDN3C', 'target_type': 'T1',
                 'target_bodyId': 102, 'category': 'verified'}]),
    }
    sus = [
        # sibling: 102 is in-map (own branch) -> tier handled by pool;
        # use a cross-branch in-map target via a second branch
        {'source_type': 's-CPDN3C', 'target_type': 'T1',
         'source_bodyId': 1, 'ahead_target_bodyId': 301,
         'ahead_target_type': 'CB4091', 'morph_v2_similarity': 0.9,
         'backward_maps_to': 's-CPDN3C', 'size_filtered': False},
        {'source_type': 's-CPDN3C', 'target_type': 'T1',
         'source_bodyId': 1, 'ahead_target_bodyId': 401,
         'ahead_target_type': 'R1-R6', 'morph_v2_similarity': 0.1,
         'backward_maps_to': '', 'size_filtered': False},
    ]
    fills = [
        {'source_type': 's-CPDN3C', 'target_type': 'T1', 'side': 'source',
         'fill_class': 'in_pool', 'bodyId': 1, 'proposal_bodyId': 101,
         'proposal_type': 'T1'},
        {'source_type': 's-CPDN3C', 'target_type': 'T1', 'side': 'target',
         'fill_class': 'in_pool', 'bodyId': 102, 'proposal_bodyId': 5,
         'proposal_type': 's-CPDN3C'},
    ]
    dedup = v.finalize_categories(
        per_pair, sus, [], fills, per_pair[('s-CPDN3C', 'T1')]['pool_detail'])
    v._write_outputs([], sus, [{'source_type': 's-CPDN3C',
                                'target_type': 'T1', 'matched': 1,
                                'gap': 0, 'gap_ratio': 0.0,
                                'gap_triggered': False, 'source_pool': 1,
                                'target_pool': 2, 'mapping_status': 'mapped',
                                'pool_basis': 'linker rows'}],
                     fills, None,
                     per_pair[('s-CPDN3C', 'T1')]['pool_detail'],
                     dedup_rows=dedup)
    # examinees CSV: every in-scope row has a category; out-of-scope blank
    sc = pd.read_csv(run_file_path(tmp_path, 'examinees.csv'))
    insc = sc[sc['in_scope'].astype(bool)]
    assert (insc['category'].fillna('') != '').all()
    oos = sc[~sc['in_scope'].astype(bool)]
    assert (oos['morph_failed'].astype(bool)).all()
    # fill CSV: target-side rows are classified (not blank) — D1 fix
    gf = pd.read_csv(run_file_path(tmp_path, 'gap_fill_proposals.csv'))
    assert gf['category'].notna().all()
    assert (gf['category'].fillna('') != '').all()
    # dedup: bodyId-unique
    dd = pd.read_csv(run_file_path(tmp_path, 'gap_fill_dedup.csv'))
    assert dd['target_bodyId'].is_unique
    assert set(dd['dedup_category']) <= {
        'matched', 'verified', 'borderline', 'unmatched', 'sibling',
        'candidates', 'family', 'relative', 'examinees'}


def test_rev312_normalize_mode_enum():
    """T12: `--mode aggressive` must engage aggressive behavior (the old
    precedence let `pool_widen` shadow `aggressive_expansion`); the legacy
    booleans resolve to the most permissive mode they imply."""
    from comparison.mapping_validation import normalize_mode
    assert normalize_mode('restrictive') == 'restrictive'
    assert normalize_mode('family') == 'family'
    assert normalize_mode('aggressive') == 'aggressive'
    # legacy flag aliases resolve to the implied mode
    assert normalize_mode('restrictive', aggressive_expansion=True) \
        == 'aggressive'
    assert normalize_mode('restrictive', pool_widen=True) == 'family'
    # both flags: most permissive wins (never family-labelled-but-aggressive)
    assert normalize_mode('restrictive', pool_widen=True,
                          aggressive_expansion=True) == 'aggressive'
    # bad input no longer fails SILENTLY to restrictive: a run that reported
    # one mode while running another is the degradation this guards against
    # (contract change with `pooling`; see also
    # test_mapping_validation_pooling.test_unknown_mode_raises_...).
    with pytest.raises(ValueError):
        normalize_mode('nonsense')
    assert normalize_mode('') == 'restrictive'
    assert normalize_mode('pooling') == 'pooling'

    # mode_at_least gates behavior
    from comparison.mapping_validation import MappingValidationConfig
    cfg = MappingValidationConfig(source_dataset='x', target_dataset='y',
                                  query_types=['z'], validation_mode='family')
    assert cfg.mode_at_least('family') is True
    assert cfg.mode_at_least('aggressive') is False
    cfg2 = MappingValidationConfig(source_dataset='x', target_dataset='y',
                                   query_types=['z'], validation_mode='aggressive')
    assert cfg2.mode_at_least('aggressive') is True


def test_rev312_n_to_1_dedup_contributes_once():
    """T9: for an N-to-1 mapping (two source types -> one target type),
    the query-level dedup lists the shared target bodyId ONCE, while the
    per-branch pools legitimately carry it twice."""
    v = _rev312_validator('restrictive')
    v._bodyids_of_type = lambda t: []
    per_pair = {
        ('s-A', 'T'): _rev312_branch(
            pool=[500, 501], cats={500: 'matched', 501: 'verified'},
            pool_detail=[
                {'source_type': 's-A', 'target_type': 'T',
                 'target_bodyId': 500, 'category': 'matched'},
                {'source_type': 's-A', 'target_type': 'T',
                 'target_bodyId': 501, 'category': 'verified'}]),
        ('s-B', 'T'): _rev312_branch(
            pool=[500, 502], cats={500: 'verified', 502: 'borderline'},
            pool_detail=[
                {'source_type': 's-B', 'target_type': 'T',
                 'target_bodyId': 500, 'category': 'verified'},
                {'source_type': 's-B', 'target_type': 'T',
                 'target_bodyId': 502, 'category': 'borderline'}]),
    }
    pd_all = (per_pair[('s-A', 'T')]['pool_detail']
              + per_pair[('s-B', 'T')]['pool_detail'])
    rows = v.finalize_categories(per_pair, [], [], [], pd_all)
    by = {r['target_bodyId']: r for r in rows}
    # 500 appears in two branches; the dedup holds it once, at its
    # strongest tier (matched > verified).
    assert by[500]['dedup_category'] == 'matched'
    assert by[500]['n_branches'] == 2
    # target bodyIds are unique across the dedup export
    ids = [r['target_bodyId'] for r in rows]
    assert len(ids) == len(set(ids))


def test_rev312_dup_flag_propagated_to_rows():
    """The query-level `dup` flag is written back onto the per-branch rows
    (and surfaces as a `(dup)` suffix in the scene key), not just the dedup
    export."""
    v = _rev312_validator('restrictive')
    v._bodyids_of_type = lambda t: []
    per_pair = {
        ('s-A', 'T1'): _rev312_branch(
            pool=[1], cats={1: 'matched'},
            pool_detail=[{'source_type': 's-A', 'target_type': 'T1',
                          'target_bodyId': 1, 'category': 'matched'}]),
        ('s-B', 'T2'): _rev312_branch(
            pool=[2], cats={2: 'matched'},
            pool_detail=[{'source_type': 's-B', 'target_type': 'T2',
                          'target_bodyId': 2, 'category': 'matched'}]),
    }
    # the same out-of-map suspect appears in BOTH branches -> dup
    sus = [
        {'source_type': 's-A', 'target_type': 'T1', 'source_bodyId': 1,
         'ahead_target_bodyId': 900, 'ahead_target_type': 'CB4091',
         'morph_v2_similarity': 0.9, 'backward_maps_to': '',
         'backward_home_real': True},
        {'source_type': 's-B', 'target_type': 'T2', 'source_bodyId': 2,
         'ahead_target_bodyId': 900, 'ahead_target_type': 'CB4091',
         'morph_v2_similarity': 0.9, 'backward_maps_to': '',
         'backward_home_real': True},
    ]
    rows = v.finalize_categories(
        per_pair, sus, [], [],
        per_pair[('s-A', 'T1')]['pool_detail']
        + per_pair[('s-B', 'T2')]['pool_detail'])
    assert all(r['category'] == 'candidates' for r in sus)
    assert all(r['dup'] is True for r in sus)
    dd = {r['target_bodyId']: r for r in rows}
    assert dd[900]['dup'] is True
    # Revision 3.12: the (dup) tag rides on the LEAF (a standalone tag),
    # and orders the leaf after the non-dup siblings of its category.
    from comparison.mapping_validation_visualize import build_category_buckets
    b, _ = build_category_buckets({'suspicious': sus, 'fills': []}, v, None,
                                  suspicious_cap=20)
    assert 'candidates' in b
    assert b['candidates']['tags'][900] == '(dup)'
    assert b['candidates']['sort_key'][900].endswith('(dup)')


def test_rev312_leaf_token_priority_chain():
    """The per-bodyId leaf token is one ordered, mutually exclusive value:
    (out-map) for an in-map TYPE [bodyId-level out-of-map] wins over
    {T}>{src} [type-level, foreign type with a real backward home] which
    wins over {T}(no_source) [type-level, no route]; untyped with no type."""
    from comparison.mapping_validation import (
        candidate_annotation, MappingValidator, MappingValidationConfig)
    # in-map type -> (out-map), regardless of backward evidence
    assert candidate_annotation('SMP220', ['s-CPDN3C'], True, True,
                                type_in_map=True) == 'SMP220(out-map)'
    assert candidate_annotation('SMP220', [], True, False,
                                type_in_map=True) == 'SMP220(out-map)'
    # foreign type with a real backward home -> >src
    assert candidate_annotation('CB1011', ['s-CPDN3B'], True, True) \
        == 'CB1011>s-CPDN3B'
    # foreign type with no route (hollow/absent) -> (no_source)
    assert candidate_annotation('CB4091', ['CB4091'], True, False) \
        == 'CB4091(no_source)'
    assert candidate_annotation('SMP220', [], True, False) \
        == 'SMP220(no_source)'
    assert candidate_annotation(None, [], False, False) == 'untyped'
    # _leaf_token applies the in-map-type short-circuit, else reads the
    # precomputed row fields, else the mapper.
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='a', target_dataset='b',
                                    query_types=['q'], visualize=False)
    v.mapper = None
    v._in_map_types = {'SMP220'}
    assert v._leaf_token({'backward_maps_to': 's-CPDN3C',
                          'backward_home_real': True}, 'SMP220') \
        == 'SMP220(out-map)'
    assert v._leaf_token({'backward_maps_to': 's-CPDN3B',
                          'backward_home_real': True}, 'CB1011') \
        == 'CB1011>s-CPDN3B'
    assert v._leaf_token({}, 'CB1011') == 'CB1011(no_source)'
    assert v._leaf_token({}, None) == 'untyped'


def test_rev312_relative_family_csv_cover_whole_bin():
    """relatives.csv / family_candidates.csv must list the WHOLE bin: a
    deep evidence row classified relative/family (aggressive mode) is
    unioned with the enumerated members, deduplicated per branch+bodyId."""
    from comparison.mapping_validation import _dedup_rows_by_bid
    ev = [
        {'source_type': 's', 'target_type': 'T', 'ahead_target_bodyId': 1},
        {'source_type': 's', 'target_type': 'T', 'ahead_target_bodyId': 1},
        {'source_type': 's', 'target_type': 'T', 'ahead_target_bodyId': 2},
        {'source_type': 's2', 'target_type': 'T', 'ahead_target_bodyId': 1},
    ]
    out = _dedup_rows_by_bid(ev)
    keys = {(r['source_type'], r['target_type'],
             r['ahead_target_bodyId']) for r in out}
    assert len(out) == len(keys) == 3      # branch+bodyId unique
    assert out[0]['ahead_target_bodyId'] == 1   # first wins


def test_rev312_expansion_categories_have_colors():
    """Every expansion category must have an explicit color, otherwise the
    scene silently falls back to examinees red (the `relative` vs
    `relatives` key mismatch)."""
    from comparison.mapping_validation import EXPANSION_CATEGORIES
    from comparison.mapping_validation_visualize import CATEGORY_COLORS
    missing = [c for c in EXPANSION_CATEGORIES
               if c not in CATEGORY_COLORS]
    assert missing == [], f'expansion categories without a color: {missing}'


# --------------------------------------------------------------------------
# Stage-4 scene styling (the UI's collapsed Advanced Visualization panel)
# --------------------------------------------------------------------------
def test_scene_colors_default_to_the_pipeline_palette():
    """No overrides -> the exact built-in map, so a run that never touched the
    panel renders byte-identically to before the dicts existed."""
    from comparison.mapping_validation_visualize import (
        CATEGORY_COLORS, UNASSIGNED_COLOR, resolve_scene_colors)
    logs = []
    assert resolve_scene_colors(None, logs.append) == CATEGORY_COLORS
    assert resolve_scene_colors({}, logs.append) == CATEGORY_COLORS
    assert logs == []
    # `unassigned` is one entry the editor paints, not a loose constant.
    assert UNASSIGNED_COLOR == CATEGORY_COLORS['unassigned']


def test_scene_color_override_follows_its_aliases():
    """A recolor must not leave a legacy alias behind wearing the old color —
    that is how `relative` vs `relatives` once silently mis-rendered."""
    from comparison.mapping_validation_visualize import resolve_scene_colors
    out = resolve_scene_colors({'relative': 'red', 'candidates': '#0000ff',
                                'query': 'rebeccapurple'})
    assert out['relatives'] == out['relative'] == 'red'
    assert out['fill'] == out['candidates'] == '#0000ff'
    # out-map query is the same source population as query (COLOR_ALIASES).
    assert out['out-map query'] == out['query'] == 'rebeccapurple'
    # and an untouched bin keeps its own color.
    assert out['matched'] == '#17becf'


def test_unparseable_scene_color_keeps_the_default_and_logs_once():
    """Stage 4 swallows a raised exception by losing EVERY scene, so a bad
    color may only cost its own category — and must say so in the run log."""
    from comparison.mapping_validation_visualize import (
        CATEGORY_COLORS, resolve_scene_colors)
    logs = []
    out = resolve_scene_colors({'pooling': 'not-a-color',
                                'verified': '', 'borderline': None},
                               logs.append)
    assert out['pooling'] == CATEGORY_COLORS['pooling']
    assert out['verified'] == CATEGORY_COLORS['verified']
    assert out['borderline'] == CATEGORY_COLORS['borderline']
    assert len(logs) == 1 and 'pooling' in logs[0]


def test_unknown_scene_category_is_named_not_silently_ignored():
    """A typo parses as a color, merges, and is never looked up — the bin keeps
    its default and the edit is silently wasted. It must cost a log line."""
    from comparison.mapping_validation_visualize import (
        CATEGORY_COLORS, resolve_scene_colors)
    logs = []
    out = resolve_scene_colors({'matchd': '#ff0000', 'matched': '#00ff00'},
                               logs.append)
    assert out['matched'] == '#00ff00'                 # the real key still lands
    assert 'matchd' not in out                         # the typo never enters
    assert len(logs) == 1 and 'matchd' in logs[0]
    # and the rejection is not a side effect of having a log: the provenance
    # record calls this WITHOUT one, and an accepted typo there would publish a
    # color the run never wore
    assert 'matchd' not in resolve_scene_colors({'matchd': '#ff0000'})


def test_editable_categories_cover_every_key_the_scene_looks_up():
    """The UI's editable list plus COLOR_ALIASES must cover every category the
    render path can color, or a bin exists that the panel cannot reach."""
    from comparison.mapping_validation_visualize import (
        CATEGORY_COLORS, COLOR_ALIASES, COLOR_EDITABLE_CATEGORIES)
    aliased = {a for targets in COLOR_ALIASES.values() for a in targets}
    reachable = set(COLOR_EDITABLE_CATEGORIES) | aliased
    unreachable = [c for c in CATEGORY_COLORS if c not in reachable]
    # The three retained legacy keys are the exception, and only `backward` is
    # unaliased: the pre-3.12 builder that could color them is deleted, so they
    # sit in CATEGORY_COLORS as a vocabulary hedge rather than a live bin.
    assert unreachable == ['backward'], unreachable
    assert set(COLOR_EDITABLE_CATEGORIES) <= set(CATEGORY_COLORS)


def _scene_cfg(scene_viz=None):
    """A real config carrying only what the scene merge reads (which also
    proves the two new fields are accepted by the constructor the UI runner
    calls)."""
    return MappingValidationConfig(
        source_dataset='flywire_FAFB_v783', target_dataset='male-cns:v1.0',
        query_types=['s-CPDN3C'], scene_viz=scene_viz)


def test_scene_viz_cannot_unpin_what_the_scene_correctness_depends_on():
    """The legend tree, the coordinate template, the synapse skip and the layer
    identity are not offered to a caller — and a caller that sends them anyway
    is ignored rather than crashing the run's scenes."""
    from comparison.mapping_validation_visualize import (
        SCENE_PINNED_KWARGS, scene_render_kwargs)
    sent = {k: 'X' for k in SCENE_PINNED_KWARGS}
    sent.update({'legend_mode': 'layer', 'brain_mesh': 'BANC',
                 'skip_synapse': False, 'skeleton_mode': 'tube'})
    out = scene_render_kwargs(_scene_cfg(sent))
    # only the tube request survives, plus the fraction the tube guard injects
    assert out == {'skeleton_mode': 'tube',
                   'skeleton_mesh_simplification': 0.9}
    assert not set(out) & set(SCENE_PINNED_KWARGS)


def test_scene_viz_drops_keys_that_are_not_scene_knobs():
    """neuron_alpha has one owner (the config field), the scene count is Max
    scenes, and the panel's bookkeeping must not reach the renderer."""
    from comparison.mapping_validation_visualize import (
        SCENE_DROPPED_KEYS, scene_render_kwargs)
    out = scene_render_kwargs(_scene_cfg({k: 'X' for k in SCENE_DROPPED_KEYS}))
    assert out == {}


def test_scene_viz_resolves_the_method_default_simplification():
    """The shared panel sends None for "use the method default". Custom
    (injected) layers bypass the renderer's fetch-time default path, and a None
    that reaches _effective_render_simplification raises — which stage 4 would
    swallow by losing every scene. Resolve it at this boundary instead."""
    from comparison.mapping_validation_visualize import scene_render_kwargs
    out = scene_render_kwargs(_scene_cfg({
        'skeleton_mode': 'tube', 'skeleton_mesh_simplification': None}))
    assert out['skeleton_mesh_simplification'] == 0.9
    # an explicit fraction stands
    out = scene_render_kwargs(_scene_cfg({
        'skeleton_mesh_simplification': 0.4}))
    assert out['skeleton_mesh_simplification'] == 0.4


def test_scene_viz_tube_without_a_fraction_still_gets_one():
    """The renderer's own `skeleton_mesh_simplification` default is None, and a
    None reaching `_effective_render_simplification` for custom (injected)
    layers raises — which stage 4 swallows by losing every scene. A caller that
    asks for tube without a fraction (the CLI's own documented example) must
    therefore still be given the method default, and line mode must stay
    untouched so an unset run remains byte-identical."""
    from comparison.mapping_validation_visualize import scene_render_kwargs
    out = scene_render_kwargs(_scene_cfg({'skeleton_mode': 'tube'}))
    assert out['skeleton_mesh_simplification'] == 0.9
    # the fallback is the RENDERER's own default pipeline, not morphology's
    # 'fine' — an untouched scene passed neither key, so 'fast'/0.90 is what
    # was actually in force before any of these knobs existed.
    import dataclasses
    from visualize_skeleton import VisualizeSkeleton
    default_pipeline = next(
        f.default for f in dataclasses.fields(VisualizeSkeleton)
        if f.name == 'neuprint_skeleton_pipeline')
    from comparison.mapping_validation_visualize import _DEFAULT_SCENE_PIPELINE
    assert _DEFAULT_SCENE_PIPELINE == default_pipeline
    # an explicit fraction is never overwritten by the guard
    out = scene_render_kwargs(_scene_cfg({'skeleton_mode': 'tube',
                                          'skeleton_mesh_simplification': 0.1}))
    assert out['skeleton_mesh_simplification'] == 0.1
    # line mode builds no neuron mesh, so nothing is injected
    assert scene_render_kwargs(_scene_cfg({'skeleton_mode': 'line'})) == {
        'skeleton_mode': 'line'}
    assert 'skeleton_mesh_simplification' not in scene_render_kwargs(
        _scene_cfg({'skeleton_mode': 'line'}))


def test_scene_viz_unset_sends_the_renderer_nothing():
    """The whole feature is inert by default: an unset run passes no extra
    kwargs at all, so the scene renders exactly as it did pre-panel."""
    from comparison.mapping_validation_visualize import scene_render_kwargs
    assert scene_render_kwargs(_scene_cfg(None)) == {}
    assert scene_render_kwargs(_scene_cfg({})) == {}


def test_scene_styling_record_publishes_what_the_run_wore():
    """`parameters.json` is a run's only provenance (it records no argv), so the
    scene look it names must be the one the pages actually wear: the panel's
    "method default" None resolved to its number, and a recolor visible on its
    legacy alias too. Recording the raw config would understate both."""
    def _cfg(**kw):
        return MappingValidationConfig(
            source_dataset='flywire_FAFB_v783', target_dataset='male-cns:v1.0',
            query_types=['t'], **kw)
    rec = scene_styling_record(_cfg(
        visualize=True,
        scene_viz={'skeleton_mode': 'tube',
                   'skeleton_mesh_simplification': None},
        scene_category_colors={'relative': 'red'}))
    assert rec['scene_viz']['skeleton_mesh_simplification'] == 0.9
    assert rec['scene_category_colors']['relative'] == 'red'
    assert rec['scene_category_colors']['relatives'] == 'red'
    # an untouched run still records the effective palette, not a bare None
    plain = scene_styling_record(_cfg(visualize=True))
    assert plain['scene_viz'] == {}
    assert plain['scene_category_colors']['matched'] == '#17becf'
    # scenes off: there was no look to wear, so the raw config is the record
    off = scene_styling_record(_cfg(visualize=False))
    assert off == {'scene_viz': None, 'scene_category_colors': None}


def test_rev312_mapper_gap_counts_follow_annotation_finalization():
    """The mapper-gap report (README `Mapper-gap evidence`) must count
    `(no_source)`/untyped CANDIDATES from the FINALIZED annotations —
    regression: the counter used to run before `candidate_annotation`
    was assigned, so it always saw empty strings and never fired."""
    v = _rev312_validator('restrictive')
    v._bodyids_of_type = lambda t: []
    sus = [{'source_type': 's', 'target_type': 'T1',
            'ahead_target_bodyId': 900, 'ahead_target_type': 'CB4091',
            'morph_v2_similarity': 0.9,
            'backward_maps_to': '', 'backward_home_real': False}]
    per_pair = {
        ('s', 'T1'): _rev312_branch(
            pool=[101], cats={101: 'matched'},
            pool_detail=[{'source_type': 's', 'target_type': 'T1',
                          'target_bodyId': 101, 'category': 'matched'}]),
    }
    v.finalize_categories(per_pair, sus, [], [],
                          per_pair[('s', 'T1')]['pool_detail'])
    assert sus[0]['candidate_annotation'] == 'CB4091(no_source)'
    assert dict(v._mapper_gap_types) == {'CB4091': 1}
    assert v._mapper_gap_untyped == 0


def test_rev312_family_leaf_token_is_out_map():
    """Family members are out-map bodyIds of the branch's OWN (in-map)
    target type, so their per-bodyId token is {T}(out-map) — a bodyId-level
    out-of-map, which wins over the type-level backward token even when the
    type does map back (SMP220 does)."""
    v = _rev312_validator('family')
    v._bodyids_of_type = lambda t: {'T1': [101, 150]}.get(t, [])
    per_pair = {
        ('s', 'T1'): _rev312_branch(
            pool=[101], cats={101: 'matched'},
            pool_detail=[{'source_type': 's', 'target_type': 'T1',
                          'target_bodyId': 101, 'category': 'matched'}]),
    }
    v.finalize_categories(per_pair, [], [], [],
                          per_pair[('s', 'T1')]['pool_detail'])
    fam = {r['ahead_target_bodyId']: r for r in v._family_rows}
    assert 150 in fam
    assert fam[150]['category'] == 'family'
    assert fam[150]['candidate_annotation'] == 'T1(out-map)'


def test_rev312_family_evidence_row_token():
    """A deep/evidence row classified `family` carries {T}(out-map) — its
    type is in-map — so deep_candidates.csv never has a family row with a
    blank token, and an in-map type is never described by a type-level
    token."""
    v = _rev312_validator('aggressive')
    v._bodyids_of_type = lambda t: []
    deep = [{'source_type': 's', 'target_type': 'T1',
             'ahead_target_bodyId': 900, 'ahead_target_type': 'T1',
             'morph_v2_similarity': 0.9,
             'backward_maps_to': 's-CPDN3C', 'backward_home_real': True}]
    per_pair = {
        ('s', 'T1'): _rev312_branch(
            pool=[1], cats={1: 'matched'},
            pool_detail=[{'source_type': 's', 'target_type': 'T1',
                          'target_bodyId': 1, 'category': 'matched'}]),
    }
    v.finalize_categories(per_pair, [], deep, [],
                          per_pair[('s', 'T1')]['pool_detail'])
    assert deep[0]['category'] == 'family'
    assert deep[0]['candidate_annotation'] == 'T1(out-map)'


def test_rev312_in_map_type_elsewhere_is_out_map_not_src():
    """The exact 110764/65631 case: a deep row whose type is an in-map type
    of ANOTHER branch (so it is an `examinees` row here) still gets {T}(out-map),
    NOT {T}>{src} — an in-map type is never described by a type-level
    token."""
    v = _rev312_validator('aggressive')
    v._bodyids_of_type = lambda t: []
    deep = [{'source_type': 's', 'target_type': 'T2',
             'ahead_target_bodyId': 900, 'ahead_target_type': 'T1',
             'morph_v2_similarity': 0.9,
             'backward_maps_to': 's-CPDN3C', 'backward_home_real': True}]
    per_pair = {
        ('s', 'T1'): _rev312_branch(
            pool=[1], cats={1: 'matched'},
            pool_detail=[{'source_type': 's', 'target_type': 'T1',
                          'target_bodyId': 1, 'category': 'matched'}]),
        ('s', 'T2'): _rev312_branch(
            pool=[2], cats={2: 'matched'},
            pool_detail=[{'source_type': 's', 'target_type': 'T2',
                          'target_bodyId': 2, 'category': 'matched'}]),
    }
    v.finalize_categories(per_pair, [], deep, [],
                          per_pair[('s', 'T1')]['pool_detail']
                          + per_pair[('s', 'T2')]['pool_detail'])
    # T1 is an in-map type (another branch) -> not sibling (not admitted),
    # and out-of-map by bodyId -> examinees, but the type is in-map.
    assert deep[0]['category'] == 'examinees'
    assert deep[0]['candidate_annotation'] == 'T1(out-map)'


def test_gap_fill_levels_assignment_and_hole_closer():
    """Floors v3 layered gap-fill report: candidates map to high/medium/low
    by bar_kind, family -> type_gated, relative -> advice; tier rows are
    claims and never appear; family_material candidates are hole closers."""
    from comparison.mapping_validation import build_gap_fill_levels
    dedup = [
        {'target_bodyId': 1, 'dedup_category': 'candidates', 'dup': False,
         'target_type': 'SMP220'},
        {'target_bodyId': 2, 'dedup_category': 'candidates', 'dup': False,
         'target_type': 'SMP220'},
        {'target_bodyId': 3, 'dedup_category': 'candidates', 'dup': False,
         'target_type': 'CB3508'},
        {'target_bodyId': 4, 'dedup_category': 'family', 'dup': False,
         'target_type': 'SMP219'},
        {'target_bodyId': 5, 'dedup_category': 'relative', 'dup': False,
         'target_type': 'CB4091'},
        {'target_bodyId': 6, 'dedup_category': 'matched', 'dup': False,
         'target_type': 'DN1a'},
    ]
    evidence = [
        {'ahead_target_bodyId': 1, 'bar_kind': 'native', 'bar_value': 0.71},
        {'ahead_target_bodyId': 2, 'bar_kind': 'track_a_backup',
         'bar_value': 0.55},
        {'ahead_target_bodyId': 3, 'bar_kind': 'null', 'bar_value': 0.61},
    ]
    rows, counts = build_gap_fill_levels(dedup, evidence, family_material=[3])
    by_bid = {r['target_bodyId']: r for r in rows}
    assert set(by_bid) == {1, 2, 3, 4, 5}          # tier row 6 excluded
    assert by_bid[1]['level'] == 'high'
    assert by_bid[2]['level'] == 'medium'
    assert by_bid[3]['level'] == 'low'
    assert by_bid[3]['note'] == 'hole closer'
    assert by_bid[4]['level'] == 'type_gated'
    assert by_bid[4]['evidence'] == 'type_membership'
    assert by_bid[5]['level'] == 'advice'
    assert counts == {'high': 1, 'medium': 1, 'low': 1,
                      'type_gated': 1, 'advice': 1}
    # ordering follows the level rank (high first, advice last)
    assert [r['level'] for r in rows] == ['high', 'medium', 'low',
                                          'type_gated', 'advice']


def test_compute_out_map_sources_source_side_gap():
    """The out-map query branch renders the UNCLAIMED source-side gap:
    annotated FAFB bodyIds that no branch pool claims (the
    pool-refinement residue).  In-pool sources — assigned or not — are
    claimed and never appear here."""
    from comparison.mapping_validation_visualize import compute_out_map_sources

    class Pair:
        parent_source_pool = [10, 11, 12, 13, 14]
        source_pool = [10, 12, 14]          # refined: 3 of 5 claimed

    res = {'pairs': [(10, 100)], 'rows': []}
    out = compute_out_map_sources([(Pair(), res)])
    # claimed = {10, 12, 14}; unclaimed = 11, 13 (never scanned)
    assert out == [11, 13]


def test_rev312_finalize_pool_categories_query_arity():
    """Regression (2026-09-17 reportcheck end-to-end run): the pool-row
    tier lookup inside finalize_categories still used a legacy
    (source, target) tuple after tiers migrated to (query, source,
    target) — every pool row silently collapsed to `unmatched`, which
    cascaded into gap_fill_dedup.csv (52/113/39 -> 204 unmatched) and
    the report's L2 provenance.  With real-run (query, ...) arity the
    pool rows must keep their tier categories."""
    v = _rev312_validator('restrictive')
    Q = 'circadian_clock'
    per_pair = {
        (Q, 's-CPDN3C', 'T1'): _rev312_branch(
            pool=[101, 102], cats={101: 'matched', 102: 'verified'},
            pool_detail=[
                {'query': Q, 'source_type': 's-CPDN3C',
                 'target_type': 'T1', 'target_bodyId': 101,
                 'category': 'matched'},
                {'query': Q, 'source_type': 's-CPDN3C',
                 'target_type': 'T1', 'target_bodyId': 102,
                 'category': 'verified'}]),
        (Q, 's-CPDN3D', 'T2'): _rev312_branch(
            pool=[201], cats={201: 'borderline'},
            pool_detail=[
                {'query': Q, 'source_type': 's-CPDN3D',
                 'target_type': 'T2', 'target_bodyId': 201,
                 'category': 'borderline'}]),
    }
    pool_detail = (per_pair[(Q, 's-CPDN3C', 'T1')]['pool_detail']
                   + per_pair[(Q, 's-CPDN3D', 'T2')]['pool_detail'])
    v.finalize_categories(per_pair, [], [], [], pool_detail)
    by_bid = {int(d['target_bodyId']): d['category'] for d in pool_detail}
    assert by_bid == {101: 'matched', 102: 'verified', 201: 'borderline'}


# ---------------------------------------------------------------------------
# §15.3: the per-side bridge basis
# ---------------------------------------------------------------------------

def _per_side_validator():
    class _Mapper:
        def get_type_bridges(self, *a, **k):
            return []

    v = MappingValidator.__new__(MappingValidator)
    v.cfg = MappingValidationConfig(source_dataset='src', target_dataset='tgt',
                                    query_types=['T'], visualize=False,
                                    morph_enabled=False)
    v.mapper = _Mapper()
    v.notes = []
    v.log = lambda *a, **k: None
    return v


def _per_side_pool(refinement):
    """The FAFB->BANC shape: the SELECTED chain is a target-side-only hop,
    so it refines the target pool exactly and leaves the source pool at the
    whole source-type population."""
    return {
        'resolution_status': 'supported',
        'selected_chain': [{'dataset': 'tgt', 'column': 'fafb_cell_type',
                            'value': 'X1'}],
        'source_body_ids': [1, 2, 3, 4],
        'target_body_ids': [11, 12],
        'source_basis': 'full population',
        'target_basis': 'linker rows',
        'per_linker': [{'column': 'fafb_cell_type', 'value': 'X1',
                        'home': 'tgt'}],
        'source_type_total': 4, 'target_type_total': 2,
        'source_side_refinement': refinement,
    }


def _per_side_pair():
    pair = TypePair('src', 'T1', [1, 2, 3, 4], 'tgt', 'X1', [11, 12])
    pair.parent_source_pool = [1, 2, 3, 4]
    pair.parent_target_pool = [11, 12]
    return pair


def test_the_source_side_takes_the_chain_that_names_it(monkeypatch):
    """One pool per branch, narrowed by the chain that carries source-side
    evidence for the SAME endpoint (plan §15.3; r22 measured 756 wide
    source slots for 242 neurons because the prioritizer's winner names
    only the target side)."""
    import ui.neuron_index as ni
    alt = [{'dataset': 'src', 'column': 'type', 'value': 'T1'},
           {'dataset': 'src', 'column': 'additional_type(s)', 'value': 'X1'},
           {'dataset': 'tgt', 'column': 'type', 'value': 'X1'}]
    monkeypatch.setattr(ni, 'resolve_prioritized_bridge_pool',
                        lambda *a, **k: _per_side_pool({
                            'rank': 2, 'chain': alt,
                            'source_basis': 'linker rows',
                            'source_body_ids': [2, 3],
                            'per_linker': [
                                {'column': 'additional_type(s)',
                                 'value': 'X1', 'home': 'src'}]}))
    pair = _per_side_pair()
    assert _per_side_validator()._refine_pair_branch(pair) is True
    assert pair.source_pool == [2, 3]              # narrowed
    assert pair.target_pool == [11, 12]            # the selected chain's own
    assert pair.pool_basis == 'linker rows'
    assert pair.target_pool_basis == 'linker rows'
    assert pair.selected_chain[0]['column'] == 'fafb_cell_type'
    assert pair.source_chain == alt               # who supplied the source
    assert pair.source_chain_text.endswith('tgt:type=X1')
    assert {(l['home'], l['column']) for l in pair.linkers} == {
        ('tgt', 'fafb_cell_type'), ('src', 'additional_type(s)')}


def test_the_per_side_rule_narrows_never_widens_never_replaces(monkeypatch):
    """A candidate that is not a subset of the selected pool is refused:
    the rule may describe the same neurons with better evidence, but it
    must not change membership or grow the pool."""
    import ui.neuron_index as ni
    alt = [{'dataset': 'tgt', 'column': 'type', 'value': 'X1'}]
    for ids, expect in (([1, 2, 3, 4, 5], [1, 2, 3, 4]),   # widens -> refused
                        ([7, 8], [1, 2, 3, 4]),            # replaces -> refused
                        ([1, 2, 3, 4], [1, 2, 3, 4])):     # equal set -> basis
        monkeypatch.setattr(ni, 'resolve_prioritized_bridge_pool',
                            lambda *a, ids=ids, **k: _per_side_pool({
                                'rank': 2, 'chain': alt,
                                'source_basis': 'linker rows',
                                'source_body_ids': ids,
                                'per_linker': []}))
        pair = _per_side_pair()
        _per_side_validator()._refine_pair_branch(pair)
        assert pair.source_pool == expect, ids
    assert pair.pool_basis == 'linker rows'       # the equal-set case ran last


def test_a_branch_stays_wide_when_no_chain_names_the_source(monkeypatch):
    """The name-asserted population (r22: APDN3 / l-LNv / DN1a / DN1pA /
    DN1pB — no FAFB `additional_type(s)` rows at all) keeps today's
    behaviour: full pool, `source_chain` == `selected_chain`."""
    import ui.neuron_index as ni
    monkeypatch.setattr(ni, 'resolve_prioritized_bridge_pool',
                        lambda *a, **k: _per_side_pool(None))
    pair = _per_side_pair()
    assert _per_side_validator()._refine_pair_branch(pair) is True
    assert pair.source_pool == [1, 2, 3, 4]
    assert pair.pool_basis == 'full population'
    assert pair.source_chain == pair.selected_chain


def test_the_refinement_of_a_different_endpoint_is_ignored(monkeypatch):
    """The alternative chain must reach THIS branch's target type: r21's
    same-name routes show a value-level mismatch is possible, and taking
    another branch's evidence would import its membership."""
    import ui.neuron_index as ni
    other = [{'dataset': 'tgt', 'column': 'type', 'value': 'X2'}]
    monkeypatch.setattr(ni, 'resolve_prioritized_bridge_pool',
                        lambda *a, **k: _per_side_pool({
                            'rank': 2, 'chain': other,
                            'source_basis': 'linker rows',
                            'source_body_ids': [2],
                            'per_linker': []}))
    pair = _per_side_pair()
    _per_side_validator()._refine_pair_branch(pair)
    assert pair.source_pool == [1, 2, 3, 4]
    assert pair.pool_basis == 'full population'


def test_every_row_backed_basis_is_measurable_not_just_one_string():
    """Consumers that compared the basis with the literal 'linker rows' read
    any other row-backed basis as a full population, so a side resolved
    through the release relation silently lost its disjointness measurement
    and read as an unresolved fan-out."""
    from comparison.mapping_validation import (
        annotate_pair_branches, measure_branch_disjointness)
    a = _per_side_pair()
    a.pool_basis = 'release relation participants'
    b = _per_side_pair()
    b.target_type, b.source_pool = 'X2', [5, 6]
    b.pool_basis = 'linker rows + alternative chains'
    assert measure_branch_disjointness([a, b]) is True
    annotate_pair_branches([a, b])
    assert 'resolved_by_linkers' in a.branch_annotation
    # a genuinely wide branch still makes the overlap unmeasurable
    wide = _per_side_pair()
    wide.target_type, wide.source_pool, wide.pool_basis = 'X3', [7], \
        'full population'
    assert measure_branch_disjointness([a, b, wide]) is None


def test_holes_are_the_unreached_part_of_family_material():
    """A hole is an OUT-map bodyId of an in-map TYPE (user 2026-09-22).

    The report's Targets-tab hover called it an "annotated in-map neuron",
    which inverts the category model: an unmapped bodyId of a type already
    in the map is out-map at bodyId level — the `{T}(out-map)` token, the
    `family` bin's population. Holes are that population minus whatever a
    fill candidate reached, so the per-type identity below is what the
    prose has to keep matching.
    """
    pair = TypePair('dsA', 'T1', [1], 'dsB', 'X', [11, 99, 97])
    pair.parent_source_pool = [1]
    pair.parent_target_pool = [11, 99, 97]
    per_pair_res = {('T1', 'X'): {
        'pairs': [(1, 11)],
        'target_categories': {11: 'matched'},
        'fills': [],
    }}
    evidence = [{'source_type': 'T1', 'target_type': 'X',
                 'ahead_target_bodyId': 99, 'ahead_target_type': 'X',
                 'category': 'candidates', 'in_scope': True}]
    m = compute_set_coverage([pair], per_pair_res, [],
                             evidence_rows=evidence)['target']
    per = m['per_type']['X']
    assert per['mapped_population'] == 3 and per['in_pool'] == 1
    # 99 is closed by the candidate; only 97 is still a hole
    assert per['hole_body_ids'] == [97]
    assert (per['mapped_population'] - per['in_pool']
            == per['reached_as_candidates_only'] + per['holes'])
    # and holes are a subset of the family material, never a different
    # population
    assert set(per['hole_body_ids']) <= set(m['family_material'])
    assert m['family_material'] == [97, 99]


def test_scene_planner_renders_every_parent_and_names_a_cap_drop():
    """`max_scenes = 0` is the default; a positive cap says what it dropped.

    Branch review is the point of a run, so the old default of 12 silently
    uncapped-out the smallest-pool parents (9 of 21 on a circadian run).
    The first cut of the warning unpacked the parent key wrong
    (`((query, source_type), branches)` items) and raised inside stage 4,
    which the pipeline swallowed as "visualization failed" — zero scenes.
    Ordering is by total source pool, largest first.
    """
    from comparison.mapping_validation_visualize import plan_scene_parents

    class _P:
        def __init__(self, n):
            self.source_pool = list(range(n))

    parents = {('q', 'BIG'): [(_P(10), {})],
               ('q', 'MID'): [(_P(5), {})],
               ('q', 'SML'): [(_P(2), {}), (_P(1), {})]}
    kept, dropped = plan_scene_parents(parents, 0)
    assert [key[1] for key, _ in kept] == ['BIG', 'MID', 'SML']
    assert dropped == []
    kept, dropped = plan_scene_parents(parents, 2)
    assert [key[1] for key, _ in kept] == ['BIG', 'MID']
    assert dropped == ['SML']


def test_an_empty_run_still_says_which_tree_it_ran_on(monkeypatch, tmp_path):
    """A run that resolves no pairs used to reach `parameters.json` with NO
    `input_fingerprint` at all: provenance was written by the stage-2 scan,
    which such a run never reaches. Measured twice on the MCNS->FAFB attempts
    of 2026-09-25/26 — both published an untraceable run folder, which is
    backwards, because a run that failed to start is exactly the one a reader
    has to place. The origin is recorded at stage 1 now, and a worktree that
    moves mid-run keeps both revs instead of overwriting the first.

    The matrix itself is the reason the second half matters: six runs against
    the live shared tree published four different revs, and one of them moved
    while it ran."""
    import comparison.mapping_validation as mv
    from types import SimpleNamespace

    class Stub:
        _fingerprint = mv.MappingValidator._fingerprint
        _record_run_origin = mv.MappingValidator._record_run_origin
        _record_scan_universe = mv.MappingValidator._record_scan_universe

        def __init__(self):
            self.lines = []

        def log(self, msg=''):
            self.lines.append(str(msg))

    monkeypatch.setattr(mv, '_git_rev', lambda: 'aaaa111')
    monkeypatch.setattr(mv, '_git_dirty', lambda: False)
    monkeypatch.setattr(mv, '_store_identity', lambda path: {})
    v = Stub()
    v.cfg = SimpleNamespace(source_dataset='s', target_dataset='t')
    v.profiler = SimpleNamespace(
        _get_cache_parquet_path=lambda ds: tmp_path / ds)
    v.mapper = object()

    v._record_run_origin()
    assert v.input_fingerprint == {'git_rev': 'aaaa111',
                                   'git_dirty': False}, \
        'a no-pair run must still carry the tree it ran on'

    # a concurrent session commits into the live tree while this run works
    monkeypatch.setattr(mv, '_git_rev', lambda: 'bbbb222')
    monkeypatch.setattr(mv, '_git_dirty', lambda: True)
    v._record_scan_universe([], [])
    fp = v.input_fingerprint
    assert fp['git_rev'] == 'bbbb222' and fp['git_dirty'] is True, \
        'the primary keys name the tree that actually scored'
    assert fp['git_rev_at_start'] == 'aaaa111'
    assert fp['git_dirty_at_start'] is False
    assert any('worktree moved' in line for line in v.lines), \
        'a moving tree is a finding, not a silent overwrite'


# ---------------------------------------------------------------------------
# I-3 (2026-09-28): the window cut is chain-deterministic.  A perfect tie
# block at the per-source budget cut resolves by the chain's bodyId term,
# so two modes reading the SAME scans keep byte-identical tight bands and
# a wider mode can only ADD rows.  The 09-28 "displaced rows" ladder FAIL
# was scan-input drift between cells (the universe grew 102144 -> 102145
# mid-matrix), not a positional cut — this test pins the cut itself.
# ---------------------------------------------------------------------------

def test_window_cut_resolves_tie_blocks_by_the_chain_not_frame_position():
    # pool members rank ahead; then ONE 30-way tie block (identical
    # zero-overlap profiles: jaccard 0.0, one shared rank_union score), so
    # the deep_cap=10 budget cuts INSIDE the tie block
    targets = {901: expanded_vector(
                   make_profile(901, {'A': 9, 'B': 5}, {'P': 4}), None),
               902: expanded_vector(
                   make_profile(902, {'A': 7, 'B': 3}, {'Q': 4}), None)}
    for bid in range(1001, 1031):
        targets[bid] = expanded_vector(
            make_profile(bid, {'Z': 5}, {'Y': 5, 'W': 5}), None)
    src = make_profile(1, SRC_UP, SRC_DN)

    _, _, fam = run_validator([1], [901, 902], targets, {1: src},
                              validation_mode='family')
    _, _, agg = run_validator([1], [901, 902], targets, {1: src},
                              validation_mode='aggressive')

    def band(res, name):
        return {int(r['ahead_target_bodyId']) for r in res['deep']
                if r['candidate_source'] == name}

    fam_tight, agg_tight = band(fam, 'top_window'), band(agg, 'top_window')
    # identical inputs -> the tight band is identical across modes, and a
    # tie block resolves to the chain's bodyId term, never frame position
    assert fam_tight == agg_tight
    assert fam_tight == set(range(1001, 1011))
    # the deep band never re-reads the borderline ranks (rank <= top_k),
    # so it cannot displace what the tight band kept
    assert band(agg, 'deep_window') == set()


def test_dedup_precedence_candidates_outrank_family():
    # I-2 (2026-09-28): the Fill tab's column header and code comment once
    # claimed "family claims outrank candidates" — DEDUP_RANK says the
    # opposite (candidates 4 > family 3) and the measured artifacts agree
    # (family-material ids with candidates rows read `candidates`).  Pin
    # the rollup so the wording can never drift back.
    v = MappingValidator.__new__(MappingValidator)

    def family_row(bid, cat):
        return {'ahead_target_bodyId': bid, 'category': cat,
                'ahead_target_type': 'X', 'source_type': 'A',
                'target_type': 'X'}

    only_family = v._build_dedup_rows(
        [], [], [family_row(555, 'family')], [])
    assert only_family[0]['dedup_category'] == 'family'

    both = v._build_dedup_rows(
        [({'category': 'candidates', 'source_type': 'A',
           'target_type': 'Y'}, 555, 'X')],
        [], [family_row(555, 'family')], [])
    assert both[0]['dedup_category'] == 'candidates'
