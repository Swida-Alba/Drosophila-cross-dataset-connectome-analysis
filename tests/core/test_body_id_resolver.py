"""Unit tests for the bodyId-level resolution backend
(plan-bodyid-level-granularity-in-type-mapper.md, Revisions 2-5).

Offline tests on synthetic ConnectivityProfile objects — no network, no
caches.  Covers:

- moved-primitive parity (reference scorer vs fast scorer)
- pool-scoped 1-to-N assignment: clean split, tie -> low_confidence,
  float64-range bodyIds, missing profile flag, non-positive gate
- side-aware matching ('require' reproduces the R2.2 probe pattern;
  'off' collapses to the unconstrained all-A outcome; side_unknown /
  side_fallback flags)
- explicit pool overrides + empty-pool groups + ProfilesUnavailable

The mapper is a stub whose ``standardize_partner_types`` is the identity
(type_mapper=None path semantics); profile construction is a stub
profiler, so the on-demand ConnectivityProfiler is never touched here.
Real-data coverage (live mapper + the motivating split) lives in the
mapper-integration tests below and scripts.
"""

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.body_id_resolver import (  # noqa: E402
    AssignmentResult,
    BodyIdResolver,
    BodyIdResolverConfig,
    ProfilesUnavailable,
    _SideStats,
    chain_key,
    expanded_vector,
    order_by_chain,
    scan_source,
    score_one_candidate,
    score_one_candidate_fast,
)
from comparison.connectivity_profiler import ConnectivityProfile  # noqa: E402


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------

class StubMapper:
    """Identity standardization (the type_mapper=None raw path)."""

    def standardize_partner_types(self, partner_types, source_dataset):
        return dict(partner_types)


class StubProfiler:
    """Duck-typed profiler over in-memory profiles/pools."""

    def __init__(self, profiles=None, pools=None, missing=frozenset()):
        self.profiles = profiles or {}   # {(ds, bid): ConnectivityProfile}
        self.pools = pools or {}         # {(ds, type): [ids]}
        self.missing = set(missing)      # {(ds, bid)} -> DataNotAvailable

    def get_bodyids_for_type(self, name, dataset):
        return list(self.pools.get((dataset, name), []))

    def get_profile(self, bid, dataset):
        key = (dataset, int(bid))
        if key in self.missing:
            from comparison.connectivity_profiler import DataNotAvailableError
            raise DataNotAvailableError(f'no connection data for {dataset}')
        if key not in self.profiles:
            return ConnectivityProfile(neuron_id=int(bid), dataset=dataset)
        return self.profiles[key]


def _prof(ds, bid, up, dn):
    return ConnectivityProfile(neuron_id=bid, dataset=ds,
                               upstream_partners=dict(up),
                               downstream_partners=dict(dn))


def _resolver(profiles=None, pools=None, config=None, missing=frozenset()):
    return BodyIdResolver(mapper=StubMapper(),
                          profiler=StubProfiler(profiles, pools, missing),
                          config=config or BodyIdResolverConfig())


# ---------------------------------------------------------------------------
# moved-primitive parity
# ---------------------------------------------------------------------------

def test_scoring_parity_reference_vs_fast():
    vec_a = {'A': 10.0, 'B': 8.0, 'P': 6.0, 'C': 4.0, 'Q': 3.0, 'D': 2.0}
    vec_b = {'A': 9.0, 'B': 2.0, 'P': 6.0, 'C': 1.0, 'R': 5.0, 'D': 3.0}
    ref = score_one_candidate(vec_a, vec_b)
    fast = score_one_candidate_fast(_SideStats(vec_a), _SideStats(vec_b))
    assert ref is not None and fast is not None
    for key in ('jaccard', 'rank_union', 'cosine', 'weighted_jaccard'):
        a, b = ref[key], fast[key]
        if a is None or (isinstance(a, float) and math.isnan(a)):
            assert isinstance(b, float) and math.isnan(b)
        else:
            assert a == pytest.approx(b, abs=1e-12)


def test_the_single_row_chain_key_and_the_frame_chain_order_agree():
    """Every S4 site that compares rows instead of sorting a frame goes
    through :func:`chain_key` — this is the one definition of "best", so the
    row form and the frame form must not be able to drift.

    The frame is built to contain the case the chain exists for: 11 and 12
    tie at jaccard 0.75 (so they share ``jaccard_rank`` 1) and only
    rank_union separates them.  A window taken on the rank column alone
    would delegate that choice to row order."""
    src = {'T1:1': 9.0, 'T1:2': 8.0, 'T2:1': 7.0, 'T3:1': 6.0}
    targets = {11: {'T1:1': 9.0, 'T1:2': 8.0, 'T2:1': 7.0},
               12: {'T1:1': 9.0, 'T1:2': 8.0, 'T3:1': 6.0},
               13: {'T1:1': 9.0, 'T4:1': 5.0}}
    stats = {b: _SideStats(v) for b, v in targets.items()}
    df = scan_source(src, stats, sorted(stats))
    assert list(df['jaccard_rank']) == [1.0, 1.0, 3.0]
    # dense positions, unlike the collapsed rank: this is what "top-N" means
    assert list(df['chain_pos']) == [1, 2, 3]
    assert list(order_by_chain(df)['target_bid']) == [11, 12, 13]
    assert [r['target_bid'] for r in
            sorted(df.to_dict('records'), key=chain_key)] == [11, 12, 13]
    assert chain_key(df.iloc[0]) < chain_key(df.iloc[1])


def test_moved_names_importable_from_mapping_validation():
    # the single-backend re-export shim keeps the historical surface
    from comparison.mapping_validation import (  # noqa: F401
        _SideStats as MV_SideStats,
        _pearson as MV_pearson,
        _rankdata_average as MV_rankdata,
        build_target_vectors as MV_btv,
        expanded_vector as MV_ev,
        load_caliber_map as MV_lcm,
        load_hemisphere_map as MV_lhm,
        passes_target_quality_gate as MV_ptqg,
        prep_target_stats as MV_pts,
        scan_source as MV_ss,
        score_one_candidate as MV_soc,
        score_one_candidate_fast as MV_socf,
    )
    from comparison import body_id_resolver as b
    assert MV_SideStats is b._SideStats
    assert MV_socf is b.score_one_candidate_fast
    assert MV_ss is b.scan_source
    assert MV_btv is b.build_target_vectors
    assert MV_ev is b.expanded_vector
    assert MV_lcm is b.load_caliber_map
    assert MV_lhm is b.load_hemisphere_map
    assert MV_pearson is b._pearson
    assert MV_rankdata is b._rankdata_average
    assert MV_ptqg is b.passes_target_quality_gate
    assert MV_pts is b.prep_target_stats
    assert MV_soc is b.score_one_candidate


# ---------------------------------------------------------------------------
# assign_bodyids — core semantics
# ---------------------------------------------------------------------------

# partner-type space; branch A (TA) and branch B (TB) have distinct
# connectivity signatures
TA = {'p1': 10.0, 'p2': 8.0, 'p3': 2.0}
TB = {'p1': 2.0, 'p2': 3.0, 'p5': 12.0}
DN_TA = {'p4': 5.0}
DN_TB = {'p4': 1.0, 'p5': 4.0}

S1 = {'p1': 9.0, 'p2': 7.0, 'p3': 3.0}      # close to TA
S2 = {'p1': 3.0, 'p2': 2.0, 'p5': 11.0}     # close to TB


def _split_profiles():
    return {
        ('ds_s', 1): _prof('ds_s', 1, S1, DN_TA),
        ('ds_s', 2): _prof('ds_s', 2, S2, DN_TB),
        ('ds_t', 11): _prof('ds_t', 11, TA, DN_TA),
        ('ds_t', 22): _prof('ds_t', 22, TB, DN_TB),
    }


def test_clean_split():
    r = _resolver(_split_profiles(), {('ds_t', 'TA'): [11],
                                      ('ds_t', 'TB'): [22]})
    res = r.assign_bodyids([1, 2], 'ds_s',
                           {'A': {'ds_t': 'TA'}, 'B': {'ds_t': 'TB'}})
    assert isinstance(res, AssignmentResult)
    assert res.assignments == {1: 'A', 2: 'B'}
    assert res.unassigned == []
    assert res.status_counts['assigned'] == 2
    # branch metadata: best member + pool used are recorded per group
    m = res.scores[1]['A']
    assert m['best_member'] == 11 and 11 in m['branch_pool_used']
    assert res.group_stats['A']['pool'] == {'ds_t': [11]}


def test_ambiguous_tie_gets_low_confidence_flag():
    # TA and TB identical -> any source scores equally against both;
    # the deterministic group-id order wins with margin 0 < min_margin
    profiles = {
        ('ds_s', 1): _prof('ds_s', 1, {'p1': 9.0, 'p2': 7.0, 'p4': 3.0}, {}),
        ('ds_t', 11): _prof('ds_t', 11, TA, DN_TA),
        ('ds_t', 22): _prof('ds_t', 22, TA, DN_TA),
    }
    r = _resolver(profiles, {('ds_t', 'TA'): [11], ('ds_t', 'TB'): [22]})
    res = r.assign_bodyids([1], 'ds_s',
                           {'A': {'ds_t': 'TA'}, 'B': {'ds_t': 'TB'}})
    assert res.assignments == {1: 'A'}  # deterministic tie order
    assert 'low_confidence' in res.flags[1]
    assert res.status_counts['assigned_low_confidence'] == 1


def test_float64_range_bodyids_stay_exact():
    big_s = 720575940619074049
    big_a = 720575940625254636
    big_b = 720575940627933336
    profiles = {
        ('ds_s', big_s): _prof('ds_s', big_s, S1, DN_TA),
        ('ds_t', big_a): _prof('ds_t', big_a, TA, DN_TA),
        ('ds_t', big_b): _prof('ds_t', big_b, TB, DN_TB),
    }
    r = _resolver(profiles, {('ds_t', 'TA'): [big_a],
                             ('ds_t', 'TB'): [big_b]})
    res = r.assign_bodyids([big_s], 'ds_s',
                           {'A': {'ds_t': 'TA'}, 'B': {'ds_t': 'TB'}})
    assert list(res.assignments.keys()) == [big_s]   # exact int, no float
    assert res.assignments[big_s] == 'A'
    assert res.scores[big_s]['A']['best_member'] == big_a


def test_missing_and_empty_profiles_flagged():
    profiles = {
        ('ds_s', 1): _prof('ds_s', 1, S1, DN_TA),
        ('ds_s', 5): _prof('ds_s', 5, {}, {}),            # empty profile
        ('ds_s', 6): _prof('ds_s', 6, {'zz': 4.0}, {}),   # no overlap
        ('ds_t', 11): _prof('ds_t', 11, TA, DN_TA),
        ('ds_t', 22): _prof('ds_t', 22, TB, DN_TB),
    }
    r = _resolver(profiles, {('ds_t', 'TA'): [11], ('ds_t', 'TB'): [22]})
    res = r.assign_bodyids([1, 5, 6], 'ds_s',
                           {'A': {'ds_t': 'TA'}, 'B': {'ds_t': 'TB'}})
    assert 1 in res.assignments
    assert 5 in res.unassigned and 'no_profile' in res.flags[5]
    # no shared partner types -> NaN rank_union -> positivity gate
    assert 6 in res.unassigned and 'non_positive' in res.flags[6]
    assert res.status_counts['unassigned_no_profile'] == 1
    assert res.status_counts['unassigned_non_positive'] == 1


def test_empty_pool_group_and_explicit_pool_override():
    # B's derived pool is empty (type absent in ds_t); an explicit pools
    # override wins over derivation
    r = _resolver(_split_profiles())
    res = r.assign_bodyids(
        [1, 2], 'ds_s',
        {'A': {'ds_t': 'TA'}, 'B': {'ds_t': 'TB'}},
        pools={'A': {'ds_t': [11]}, 'B': {'ds_t': [22]}})
    assert res.assignments == {1: 'A', 2: 'B'}
    assert res.group_stats['B']['pool'] == {'ds_t': [22]}


def test_all_target_profiles_missing_raises_profiles_unavailable():
    r = _resolver({('ds_s', 1): _prof('ds_s', 1, S1, DN_TA)},
                  pools={('ds_t', 'TA'): [11], ('ds_t', 'TB'): [22]},
                  missing={('ds_t', 11), ('ds_t', 22)})
    with pytest.raises(ProfilesUnavailable):
        r.assign_bodyids([1], 'ds_s',
                         {'A': {'ds_t': 'TA'}, 'B': {'ds_t': 'TB'}})


def test_source_profile_missing_connections_raises_profiles_unavailable():
    r = _resolver({('ds_t', 11): _prof('ds_t', 11, TA, DN_TA)},
                  missing={('ds_s', 1)})
    with pytest.raises(ProfilesUnavailable):
        r.assign_bodyids([1], 'ds_s',
                         {'A': {'ds_t': 'TA'}, 'B': {'ds_t': 'TB'}})


# ---------------------------------------------------------------------------
# side-aware matching (plan R2.2)
# ---------------------------------------------------------------------------

# Synthetic reproduction of the motivating probe: T_AR is the globally
# best match for BOTH sources (like the real FAFB right 5th-LNv), so the
# unconstrained argmax sends everything to branch A; only the
# same-side constraint recovers the biological 2+2 split.
T_AR = {'p1': 10.0, 'p2': 8.0, 'p3': 2.0}
T_AL = {'p1': 1.0, 'p2': 1.0, 'p5': 2.0}       # weak match for everything
T_BR = {'p1': 2.0, 'p2': 3.0, 'p5': 4.0}
T_BL = {'p1': 10.0, 'p2': 8.0, 'p3': 2.0, 'p5': 20.0}
S_R = {'p1': 9.0, 'p2': 7.0, 'p3': 3.0}
# S_L matches T_AR best globally (the R2.2 probe phenomenon) but still
# matches its same-side T_BL well above the gates
S_L = {'p1': 10.0, 'p2': 8.0, 'p3': 2.0, 'p5': 6.0}


def _side_profiles():
    return {
        ('ds_s', 101): _prof('ds_s', 101, S_R, {}),
        ('ds_s', 102): _prof('ds_s', 102, S_L, {}),
        ('ds_t', 201): _prof('ds_t', 201, T_AR, {}),
        ('ds_t', 202): _prof('ds_t', 202, T_AL, {}),
        ('ds_t', 203): _prof('ds_t', 203, T_BR, {}),
        ('ds_t', 204): _prof('ds_t', 204, T_BL, {}),
    }


@pytest.fixture()
def side_maps(monkeypatch):
    import comparison.body_id_resolver as b
    mapping = {'ds_s': {101: 'R', 102: 'L'},
               'ds_t': {201: 'R', 202: 'L', 203: 'R', 204: 'L'}}
    monkeypatch.setattr(b, 'load_hemisphere_map',
                        lambda ds, project_root=None: mapping.get(ds, {}))


def _side_groups():
    return ({'A': {'ds_t': 'TA'}, 'B': {'ds_t': 'TB'}},
            {'A': {'ds_t': [201, 202]}, 'B': {'ds_t': [203, 204]}})


def test_side_require_recovers_the_split(side_maps):
    r = _resolver(_side_profiles())
    groups, pools = _side_groups()
    res = r.assign_bodyids([101, 102], 'ds_s', groups, pools=pools)
    assert res.assignments == {101: 'A', 102: 'B'}
    # only same-side members were scored per source
    assert res.scores[101]['A']['branch_pool_used'] == [201]
    assert res.scores[102]['B']['branch_pool_used'] == [204]
    assert res.status_counts['assigned'] == 2


def test_side_off_collapses_to_the_unconstrained_winner(side_maps):
    r = _resolver(_side_profiles(),
                  config=BodyIdResolverConfig(side_matching='off'))
    groups, pools = _side_groups()
    res = r.assign_bodyids([101, 102], 'ds_s', groups, pools=pools)
    # unconstrained: the globally best member (201, branch A) wins both
    assert res.assignments == {101: 'A', 102: 'A'}


def test_side_unknown_source_never_constrains(side_maps):
    profiles = _side_profiles()
    profiles[('ds_s', 103)] = _prof('ds_s', 103, S_R, {})
    r = _resolver(profiles)
    groups, pools = _side_groups()
    res = r.assign_bodyids([103], 'ds_s', groups, pools=pools)
    assert 'side_unknown' in res.flags[103]
    assert res.scores[103]['A']['branch_pool_used'] == [201, 202]


def test_side_require_falls_back_when_no_group_has_same_side(
        side_maps, monkeypatch):
    import comparison.body_id_resolver as b
    r = _resolver(_side_profiles())
    groups, pools = _side_groups()
    # all target sides unknown -> no same-side candidate anywhere
    monkey_map = {'ds_s': {101: 'R', 102: 'L'}, 'ds_t': {}}
    monkeypatch.setattr(b, 'load_hemisphere_map',
                        lambda ds, project_root=None: monkey_map.get(ds, {}))
    res = r.assign_bodyids([101], 'ds_s', groups, pools=pools)
    assert 'side_fallback' in res.flags[101]
    assert res.scores[101]['A']['branch_pool_used'] == [201, 202]


def test_side_prefer_uses_same_side_when_present(side_maps):
    r = _resolver(_side_profiles(),
                  config=BodyIdResolverConfig(side_matching='prefer'))
    groups, pools = _side_groups()
    res = r.assign_bodyids([101, 102], 'ds_s', groups, pools=pools)
    assert res.assignments == {101: 'A', 102: 'B'}
    assert res.scores[102]['B']['branch_pool_used'] == [204]


def test_side_maps_not_loaded_when_off(monkeypatch):
    import comparison.body_id_resolver as b
    called = {'n': 0}

    def counting(ds, project_root=None):
        called['n'] += 1
        return {}

    monkeypatch.setattr(b, 'load_hemisphere_map', counting)
    r = _resolver(_side_profiles(),
                  config=BodyIdResolverConfig(side_matching='off'))
    groups, pools = _side_groups()
    r.assign_bodyids([101], 'ds_s', groups, pools=pools)
    assert called['n'] == 0


# ---------------------------------------------------------------------------
# mapper integration (lazy property + derive_split_groups)
# ---------------------------------------------------------------------------


@pytest.mark.requires_data("male-cns:v1.0")
def test_same_name_asymmetry_flag_and_population_accessor():
    """User directive 2026-09-14: same-name = EXACT name; extreme
    population asymmetry (>=10x) is carried as a suggested-check flag —
    evidence for the user, never a gate.

    Data-gated on the male-cns table the mapper loads (round-7 report:
    without it `get_mapping_support` returns None and the direct keyed
    access raised `KeyError: 'population_asymmetry'` on the Windows
    host)."""
    from comparison.cross_dataset_type_mapper import get_type_mapper
    m = get_type_mapper()
    flagged = m.get_mapping_support('TmY18', 'male-cns:v1.0', 'TmY18',
                                    'flywire_FAFB_v783')
    if flagged is None:
        pytest.skip('type-mapper crosswalk unavailable on this host '
                    '(male-cns tables not initialized)')
    asym = flagged['population_asymmetry']
    assert asym['n_source'] == 1367 and asym['n_target'] == 1
    assert asym['flag'] == 'extreme population asymmetry'
    # exact-name principle: different names never get the identity record
    assert m.get_mapping_support('5thsLNv_LNd6', 'male-cns:v1.0',
                                 '5th-LNv', 'flywire_FAFB_v783') is None
    # public bookkeeping accessor
    assert m.get_type_population('TmY18', 'male-cns:v1.0') == 1367
    assert m.get_type_population('TmY18', 'flywire_FAFB_v783') == 1
    assert m._format_bridge_support(flagged).endswith('(1367 vs 1; suggested check)')


def test_the_caliber_and_side_loaders_read_real_dataset_column_names(tmp_path):
    """BANC v888 spells its caliber column `Volume (nm^3)` and its side column
    `Soma side` ('left'/'right', 92% of 188,508 rows).

    Reading only `size`/`size_nm`, and only a bare `side`/`hemisphere`/`sides`,
    returned {} and all-'?' for that target — which silently disabled the
    spatial-caliber noise filter and the hemisphere-asymmetry gap trigger in
    every FAFB/MCNS -> BANC run (found preparing the FAFB -> BANC circadian
    run; MCNS/FAFB output is unchanged, which is why no run ever noticed).
    The loaders therefore match on an alphanumeric column key AND read by
    label: `itertuples` renames a column containing a space, so the previous
    `getattr(row, col)` form could not have worked even with the alias.
    """
    import pandas as pd
    from comparison.body_id_resolver import (load_caliber_map,
                                             load_hemisphere_map)
    from utils.naming_utils import canonical_dataset_name

    folder = canonical_dataset_name('banc_v888').replace(':', '_').replace(
        '.', '_')
    d = tmp_path / 'datasets' / folder
    d.mkdir(parents=True)
    pd.DataFrame([
        {'bodyId': 11, 'Volume (nm^3)': 5.0e9, 'Soma side': 'left',
         'instance': 'X_1'},
        {'bodyId': 12, 'Volume (nm^3)': 1.0e8, 'Soma side': 'Right',
         'instance': 'Unknown'},
        {'bodyId': 13, 'Volume (nm^3)': None, 'Soma side': None,
         'instance': 'Y_R'},
    ]).to_csv(d / f'{folder}_allneurons_neuron_df.csv', index=False)

    assert load_caliber_map('banc_v888', project_root=tmp_path) == {
        11: 5.0e9, 12: 1.0e8, 13: 0.0}
    assert load_hemisphere_map('banc_v888', project_root=tmp_path) == {
        11: 'L', 12: 'R', 13: 'R'}
