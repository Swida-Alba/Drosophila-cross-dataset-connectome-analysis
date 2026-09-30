"""Coverage tests for comparison.connectivity_profiler.

Hermetic: no network access. All data access methods are monkeypatched and
synthetic DataFrames/dicts are injected. All file I/O goes through tmp_path.
"""

import numpy as np
import pandas as pd
import pytest

import comparison.connectivity_profiler as cp_module
from comparison.connectivity_profiler import (
    ConnectivityProfile,
    ConnectivityProfiler,
    ConnectivityStatus,
    DataNotAvailableError,
    FuzzyMatchConfig,
    ProfilerConfig,
    clear_profiler_conn_cache,
    compute_ranks,
    get_profiler_conn_cache_info,
    normalize_partner_type,
)

DS = 'flywire_FAFB_v783'


def _profile(bid, ds=DS, upstream=None, downstream=None, **kwargs):
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
    return ConnectivityProfile(**params)


@pytest.fixture
def profiler(tmp_path):
    return ConnectivityProfiler(
        [DS], config=ProfilerConfig(), cache_dir=str(tmp_path), verbose=False
    )


# ---------------------------------------------------------------------------
# ConnectivityStatus enum
# ---------------------------------------------------------------------------

def test_connectivity_status_flags():
    assert ConnectivityStatus.COMPLETE.is_valid_for_comparison()
    assert ConnectivityStatus.INCOMPLETE.is_valid_for_comparison()
    assert not ConnectivityStatus.NONE.is_valid_for_comparison()
    assert not ConnectivityStatus.ORPHAN.is_valid_for_comparison()
    assert ConnectivityStatus.RARE.requires_warning()
    assert not ConnectivityStatus.COMPLETE.requires_warning()
    assert ConnectivityStatus.COMPLETE.is_complete()
    assert not ConnectivityStatus.INCOMPLETE.is_complete()
    assert 'Complete' in ConnectivityStatus.get_description(ConnectivityStatus.COMPLETE)
    assert ConnectivityStatus.get_description(None) == 'Unknown'


# ---------------------------------------------------------------------------
# Module-level connection cache helpers
# ---------------------------------------------------------------------------

def test_module_conn_cache_info_and_clear():
    clear_profiler_conn_cache()
    assert get_profiler_conn_cache_info() == {}
    cp_module._PROFILER_CONN_CACHE['test_ds'] = {
        'conn_df': pd.DataFrame({'a': [1, 2, 3]}),
        'type_lookup': {'x': 'y'},
    }
    cp_module._PROFILER_CONN_CACHE['empty_ds'] = {'conn_df': None, 'type_lookup': {}}
    info = get_profiler_conn_cache_info()
    assert info['test_ds']['conn_df_rows'] == 3
    assert info['test_ds']['type_lookup_size'] == 1
    assert info['empty_ds']['conn_df_rows'] == 0
    clear_profiler_conn_cache()
    assert get_profiler_conn_cache_info() == {}


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = ProfilerConfig()
    assert cfg.top_k_bodyid == 5
    assert cfg.top_m_type == 0
    assert cfg.normalize_method == 'rank'
    assert cfg.use_cache is True
    assert cfg.expand_untyped_2hop is True
    assert isinstance(cfg.fuzzy_match, FuzzyMatchConfig)
    fuzzy = FuzzyMatchConfig()
    assert fuzzy.enabled is False
    assert fuzzy.custom_mappings == {}


# ---------------------------------------------------------------------------
# normalize_partner_type
# ---------------------------------------------------------------------------

def test_normalize_partner_type_disabled():
    cfg = FuzzyMatchConfig(enabled=False)
    assert normalize_partner_type('T4a_R', cfg) == 'T4a_R'
    assert normalize_partner_type('', cfg) == ''
    assert normalize_partner_type(None, cfg) == ''
    # NaN is a missing value: fixed to normalize to '' (was the string 'nan')
    assert normalize_partner_type(np.nan, cfg) == ''


def test_normalize_partner_type_case_insensitive():
    cfg = FuzzyMatchConfig(enabled=True, case_insensitive=True)
    assert normalize_partner_type('T4A', cfg) == 't4a'


def test_normalize_partner_type_strip_lr():
    cfg = FuzzyMatchConfig(enabled=True, strip_lr_suffix=True)
    assert normalize_partner_type('T4a_R', cfg) == 'T4a'
    assert normalize_partner_type('T4a-left', cfg) == 'T4a'
    assert normalize_partner_type('T4a', cfg) == 'T4a'


def test_normalize_partner_type_strip_numeric_and_like():
    cfg = FuzzyMatchConfig(enabled=True, strip_numeric_suffix=True)
    assert normalize_partner_type('aMe12_1', cfg) == 'aMe12'
    assert normalize_partner_type('aMe12-2', cfg) == 'aMe12'
    cfg_like = FuzzyMatchConfig(enabled=True, strip_like_suffix=True)
    assert normalize_partner_type('T4a-like', cfg_like) == 'T4a'
    assert normalize_partner_type('T4a_LIKE', cfg_like) == 'T4a'


def test_normalize_partner_type_custom_mappings():
    cfg = FuzzyMatchConfig(enabled=True, custom_mappings={'T4a_R': 'T4a'})
    assert normalize_partner_type('T4a_R', cfg) == 'T4a'
    # Custom mapping applied AFTER normalization as well
    cfg2 = FuzzyMatchConfig(enabled=True, case_insensitive=True,
                            custom_mappings={'t4a': 'T4'})
    assert normalize_partner_type('T4A', cfg2) == 'T4'


# ---------------------------------------------------------------------------
# compute_ranks
# ---------------------------------------------------------------------------

def test_compute_ranks():
    assert compute_ranks({}) == {}
    assert compute_ranks({'a': 1.0}) == {'a': 1}
    ranks = compute_ranks({'a': 10.0, 'b': 10.0, 'c': 5.0, 'd': 1.0})
    assert ranks['a'] == 1
    assert ranks['b'] == 1
    assert ranks['c'] == 3
    assert ranks['d'] == 4


# ---------------------------------------------------------------------------
# ConnectivityProfile: status classification
# ---------------------------------------------------------------------------

def test_profile_status_none():
    p = _profile('n1')
    assert p.connectivity_status is ConnectivityStatus.NONE
    assert p.connectivity_status_str == 'none'
    assert p.is_weak_connectivity is True
    assert p.is_valid_for_comparison() is False


def test_profile_status_unidirectional():
    # explicit counts drive classification
    p2 = ConnectivityProfile(
        neuron_id='n2', dataset=DS,
        upstream_partners={'A': 1.0}, downstream_partners={},
        actual_upstream_count=0, actual_downstream_count=10,
        top_k_bodyid_used=5, top_m_type_target=0,
    )
    assert p2.connectivity_status is ConnectivityStatus.UNIDIRECTIONAL


def test_profile_status_rare():
    p = _profile('n3', upstream={'A': 1.0},
                 actual_upstream_count=3, actual_downstream_count=10,
                 top_k_bodyid_used=5, top_m_type_target=0)
    assert p.connectivity_status is ConnectivityStatus.RARE
    assert p.is_weak_connectivity is False  # RARE is valid (with warning)


def test_profile_status_incomplete():
    p = _profile('n4', upstream={'A': 1.0},
                 actual_upstream_count=6, actual_downstream_count=6,
                 top_k_bodyid_used=10, top_m_type_target=0)
    assert p.connectivity_status is ConnectivityStatus.INCOMPLETE


def test_profile_status_incomplete_expansion_and_sparse():
    p = _profile('n5', upstream={'A': 1.0},
                 actual_upstream_count=10, actual_downstream_count=10,
                 top_k_bodyid_used=10, top_m_type_target=5,
                 unique_types_upstream=2, unique_types_downstream=10)
    assert p.connectivity_status is ConnectivityStatus.INCOMPLETE_EXPANSION
    assert p.is_sparse is True


def test_profile_status_complete():
    p = _profile('n6', upstream={'A': 1.0},
                 actual_upstream_count=10, actual_downstream_count=10,
                 top_k_bodyid_used=10, top_m_type_target=0)
    assert p.connectivity_status is ConnectivityStatus.COMPLETE
    assert 'Status' in p.summary()


# ---------------------------------------------------------------------------
# ConnectivityProfile: serialization
# ---------------------------------------------------------------------------

def test_profile_to_dict_from_dict_roundtrip():
    p = ConnectivityProfile(
        neuron_id=123, dataset=DS, neuron_type='TypeX',
        upstream_partners={'A': 10.0}, downstream_partners={'B': 5.0},
        upstream_ranks={'A': 1}, downstream_ranks={'B': 1},
        total_upstream_weight=10.0, total_downstream_weight=5.0,
        partner_type_mapping_upstream={1: 'A'},
        partner_type_mapping_downstream={9: 'B'},
        untyped_upstream_bodyids={2: 3.0},
        untyped_downstream_bodyids={3: 1.0},
        untyped_upstream_2hop={2: {'Mi1': 4.0}},
        untyped_downstream_2hop={3: {'Tm1': 2.0}},
        untyped_upstream_2hop_ranks={2: {'Mi1': 1}},
        untyped_downstream_2hop_ranks={3: {'Tm1': 1}},
        typed_upstream_bodyids={1: 10.0},
        typed_downstream_bodyids={9: 5.0},
    )
    d = p.to_dict()
    assert d['neuron_id'] == '123'
    assert d['connectivity_status'] == p.connectivity_status_str
    p2 = ConnectivityProfile.from_dict(d)
    assert p2.partner_type_mapping_upstream == {1: 'A'}
    assert p2.untyped_upstream_bodyids == {2: 3.0}
    assert p2.untyped_upstream_2hop == {2: {'Mi1': 4.0}}
    assert p2.untyped_upstream_2hop_ranks == {2: {'Mi1': 1}}
    assert p2.typed_upstream_bodyids == {1: 10.0}
    assert p2.downstream_partners == {'B': 5.0}

    # from_dict tolerates missing optional keys
    minimal = {'neuron_id': 'x', 'dataset': DS}
    p3 = ConnectivityProfile.from_dict(minimal)
    assert p3.neuron_type is None


# ---------------------------------------------------------------------------
# ConnectivityProfile: accessors and vectors
# ---------------------------------------------------------------------------

def test_profile_accessors():
    p = _profile('acc', upstream={'A': 3.0, 'C': 1.0},
                 downstream={'B': 2.0, 'C': 4.0},
                 partner_type_mapping_upstream={1: 'A', 2: 'A', 3: 'C'})
    assert p.get_all_partner_types() == {'A', 'B', 'C'}
    assert p.get_partner_weight('A', 'upstream') == 3.0
    assert p.get_partner_weight('A', 'downstream') == 0.0
    assert p.get_partner_weight('C', 'both') == 5.0
    assert p.get_partner_bodyids('A', 'upstream') == [1, 2]
    assert p.get_partner_bodyids('A', 'downstream') == []

    props = p.get_proportions('upstream')
    assert props['A'] == pytest.approx(0.75)
    assert p.get_proportions('downstream')['C'] == pytest.approx(2.0 / 3.0)
    combined = p.get_proportions('both')
    assert sum(combined.values()) == pytest.approx(1.0)
    empty = _profile('empty')
    assert empty.get_proportions('upstream') == {}


def test_profile_weight_and_rank_vectors():
    p = _profile('vec', upstream={'A': 3.0, 'B': 4.0},
                 downstream={'A': 1.0})
    vocab = ['A', 'B', 'C']
    v = p.to_weight_vector(vocab, 'upstream', normalize=False)
    assert list(v) == [3.0, 4.0, 0.0]
    v_norm = p.to_weight_vector(vocab, 'upstream', normalize=True)
    assert np.linalg.norm(v_norm) == pytest.approx(1.0)
    v_both = p.to_weight_vector(vocab, 'both', normalize=False)
    assert list(v_both) == [4.0, 4.0, 0.0]
    # zero vector stays zero (no divide-by-zero)
    assert list(p.to_weight_vector(vocab, 'upstream', normalize=True) * 0) == [0.0, 0.0, 0.0]
    assert list(_profile('z').to_weight_vector(vocab, 'upstream', normalize=True)) == [0.0, 0.0, 0.0]

    rv = p.to_rank_vector(vocab, 'upstream')
    assert list(rv) == [2.0, 1.0, 3.0]  # default rank = K+1 = 3
    rv_d = p.to_rank_vector(vocab, 'upstream', default_rank=9)
    assert rv_d[2] == 9
    rv_both = p.to_rank_vector(vocab, 'both')
    assert rv_both.shape == (3,)


def test_profile_hybrid_rank_vector():
    p = ConnectivityProfile(
        neuron_id='h', dataset=DS,
        upstream_partners={'A': 5.0}, upstream_ranks={'A': 1},
        total_upstream_weight=5.0,
        untyped_upstream_bodyids={7: 2.0},
        untyped_upstream_2hop={7: {'X': 4.0}},
        untyped_upstream_2hop_ranks={7: {'X': 1}},
    )
    vocab = ['A', 'X', 'Z']
    cross = p.to_hybrid_rank_vector(vocab, direction='upstream', mode='cross_dataset')
    # A -> rank 1; X -> hop2 rank 1 + len(typed)=1 -> 2; Z -> default (k=2 -> 3)
    assert list(cross) == [1.0, 2.0, 3.0]
    intra = p.to_hybrid_rank_vector(vocab, direction='upstream', mode='intra_dataset')
    assert list(intra) == [1.0, 3.0, 3.0]


def test_profile_untyped_bodyid_ranks():
    p = ConnectivityProfile(
        neuron_id='u', dataset=DS,
        untyped_upstream_bodyids={10: 5.0, 20: 9.0},
    )
    assert p.get_untyped_bodyid_ranks('upstream') == {20: 1, 10: 2}
    assert p.get_untyped_bodyid_ranks('downstream') == {}


# ---------------------------------------------------------------------------
# ConnectivityProfile: static vocabulary / matrix helpers
# ---------------------------------------------------------------------------

def test_build_shared_vocabulary():
    p1 = _profile('a', upstream={'X': 1.0, 'Y': 2.0})
    p2 = _profile('b', upstream={'Y': 1.0, 'Z': 1.0}, downstream={'Y': 1.0})
    vocab = ConnectivityProfile.build_shared_vocabulary({'a': p1, 'b': p2}, 'upstream')
    assert vocab == ['X', 'Y', 'Z']
    vocab2 = ConnectivityProfile.build_shared_vocabulary(
        {'a': p1, 'b': p2}, 'upstream', min_occurrence=2)
    assert vocab2 == ['Y']
    vocab_both = ConnectivityProfile.build_shared_vocabulary({'b': p2}, 'both')
    assert 'Y' in vocab_both


def test_build_hybrid_vocabulary_includes_2hop():
    p = ConnectivityProfile(
        neuron_id='h', dataset=DS,
        upstream_partners={'A': 1.0},
        untyped_upstream_2hop={5: {'H2': 2.0}},
    )
    vocab = ConnectivityProfile.build_hybrid_vocabulary({'h': p}, 'upstream')
    assert vocab == ['A', 'H2']
    vocab_no2 = ConnectivityProfile.build_hybrid_vocabulary(
        {'h': p}, 'upstream', include_2hop=False)
    assert vocab_no2 == ['A']


def test_profiles_to_matrices():
    p1 = _profile('a', upstream={'X': 3.0, 'Y': 4.0})
    p2 = _profile('b', upstream={'Y': 1.0})
    matrix, ids, vocab = ConnectivityProfile.profiles_to_weight_matrix(
        {'a': p1, 'b': p2}, direction='upstream')
    assert matrix.shape == (2, 2)
    assert ids == ['a', 'b']
    assert vocab == ['X', 'Y']
    rmat, rids, rvocab = ConnectivityProfile.profiles_to_rank_matrix(
        {'a': p1, 'b': p2}, direction='upstream')
    assert rmat.shape == (2, 2)


def test_aggregate_bodyid_profiles():
    with pytest.raises(ValueError):
        ConnectivityProfile.aggregate_bodyid_profiles([], 'T', DS)
    p1 = _profile(1, upstream={'A': 2.0, 'B': 1.0},
                  untyped_upstream_count=1, untyped_upstream_weight_fraction=0.5)
    p2 = _profile(2, upstream={'A': 2.0, 'C': 3.0})
    agg = ConnectivityProfile.aggregate_bodyid_profiles([p1, p2], 'TypeT', DS)
    assert agg.neuron_type == 'TypeT'
    assert agg.num_neurons_aggregated == 2
    assert agg.upstream_partners['A'] == 4.0
    assert agg.upstream_ranks['A'] == 1
    assert agg.upstream_ranks['C'] == 2
    assert agg.untyped_upstream_count == 1


# ---------------------------------------------------------------------------
# ConnectivityProfiler: _process_connections
# ---------------------------------------------------------------------------

def _conn_df(rows):
    return pd.DataFrame(rows, columns=['partner_type', 'weight', 'partner_bodyId'])


def test_process_connections_empty_and_missing_column(profiler):
    res = profiler._process_connections(pd.DataFrame(), 'upstream', 5)
    partners, ranks, untyped_count, untyped_frac, total, actual, unique, mapping, k_used, ub, tb = res
    assert partners == {} and ranks == {} and k_used == 5

    df = pd.DataFrame({'weight': [1.0, 2.0]})
    res = profiler._process_connections(df, 'upstream', 3)
    assert res[0] == {} and res[8] == 3


def test_process_connections_typed(profiler):
    df = _conn_df([('A', 10.0, 1), ('B', 8.0, 2), ('A', 2.0, 3), ('C', 1.0, 4)])
    (partners, ranks, untyped_count, untyped_frac, total, actual,
     unique, mapping, k_used, ub, tb) = profiler._process_connections(df, 'upstream', 5)
    assert partners['A'] == pytest.approx(12.0)
    assert partners['B'] == pytest.approx(8.0)
    assert partners['C'] == pytest.approx(1.0)
    assert ranks['A'] == 1 and ranks['B'] == 2 and ranks['C'] == 3
    assert untyped_count == 0 and untyped_frac == 0.0
    assert total == pytest.approx(21.0)
    assert actual == 3 and unique == 3
    assert mapping[2] == 'B'
    assert tb[1] == pytest.approx(10.0)
    assert ub == {}


def test_process_connections_untyped_only(profiler):
    df = _conn_df([(None, 5.0, 1), (np.nan, 3.0, 2), ('', 2.0, 3)])
    (partners, ranks, untyped_count, untyped_frac, total, actual,
     unique, mapping, k_used, ub, tb) = profiler._process_connections(df, 'upstream', 5)
    assert partners == {}
    assert untyped_count == 3
    assert untyped_frac == pytest.approx(1.0)
    assert actual == 0 and unique == 0
    assert set(ub.keys()) == {1, 2, 3}
    assert tb == {}


def test_process_connections_include_untyped(tmp_path):
    cfg = ProfilerConfig(include_untyped_partners=True)
    profiler = ConnectivityProfiler([DS], config=cfg, cache_dir=str(tmp_path), verbose=False)
    df = _conn_df([('A', 5.0, 1), (None, 3.0, 2)])
    partners = profiler._process_connections(df, 'upstream', 5)[0]
    assert 'untyped' in partners
    assert partners['untyped'] == pytest.approx(3.0)


def test_process_connections_fuzzy(tmp_path):
    cfg = ProfilerConfig(fuzzy_match=FuzzyMatchConfig(enabled=True, case_insensitive=True))
    profiler = ConnectivityProfiler([DS], config=cfg, cache_dir=str(tmp_path), verbose=False)
    df = _conn_df([('T4a', 5.0, 1), ('t4a', 3.0, 2), ('Mi1', 1.0, 3)])
    partners, ranks, *_ = profiler._process_connections(df, 'upstream', 5)
    assert set(partners.keys()) == {'t4a', 'mi1'}
    assert partners['t4a'] == pytest.approx(8.0)


def test_process_connections_dynamic_expansion(tmp_path):
    cfg = ProfilerConfig(top_k_bodyid=2, top_m_type=4)
    profiler = ConnectivityProfiler([DS], config=cfg, cache_dir=str(tmp_path), verbose=False)
    rows = [(f'T{i}', float(10 - i), i + 1) for i in range(6)]
    (partners, ranks, uc, uf, tw, actual, unique, mapping,
     k_used, ub, tb) = profiler._process_connections(_conn_df(rows), 'upstream', 2)
    assert unique == 6
    assert k_used == 7  # stepped +5 past hit position
    assert len(partners) == 6


def test_process_connections_expansion_insufficient_types(tmp_path):
    cfg = ProfilerConfig(top_k_bodyid=2, top_m_type=10, max_expansion_factor=3)
    profiler = ConnectivityProfiler([DS], config=cfg, cache_dir=str(tmp_path), verbose=False)
    rows = [('A', 5.0, 1), ('A', 4.0, 2), ('B', 3.0, 3), ('B', 2.0, 4)]
    res = profiler._process_connections(_conn_df(rows), 'upstream', 2)
    partners, k_used = res[0], res[8]
    assert set(partners.keys()) == {'A', 'B'}
    assert k_used == 4  # min(max_k=6, n_rows=4)


def test_process_connections_no_dynamic_expansion(tmp_path):
    cfg = ProfilerConfig(dynamic_expansion=False)
    profiler = ConnectivityProfiler([DS], config=cfg, cache_dir=str(tmp_path), verbose=False)
    rows = [(f'T{i}', float(10 - i), i + 1) for i in range(8)]
    res = profiler._process_connections(_conn_df(rows), 'upstream', 3)
    assert len(res[0]) == 3 and res[8] == 3


# ---------------------------------------------------------------------------
# ConnectivityProfiler: get_profile with monkeypatched data access
# ---------------------------------------------------------------------------

def _patch_queries(monkeypatch, profiler, up_df, down_df, hop2=None):
    monkeypatch.setattr(profiler, 'ensure_data_available',
                        lambda dataset, raise_on_missing=True: True)
    monkeypatch.setattr(profiler, '_query_connections_local',
                        lambda neuron, dataset: (up_df, down_df))
    if hop2 is not None:
        monkeypatch.setattr(
            profiler, '_fetch_2hop_partners',
            lambda bids, dataset, direction: {b: hop2 for b in bids})


def test_get_profile_with_2hop_and_cache(profiler, monkeypatch):
    up = _conn_df([('A', 10.0, 1), ('B', 8.0, 2), ('C', 6.0, 3), (None, 4.0, 4)])
    down = _conn_df([('D', 5.0, 5), ('E', 3.0, 6)])
    _patch_queries(monkeypatch, profiler, up, down, hop2=({'H1': 2.0}, {'H1': 1}))

    prof = profiler.get_profile('MyType', DS)
    assert prof.neuron_type == 'MyType'
    assert set(prof.upstream_partners) == {'A', 'B', 'C'}
    assert prof.untyped_upstream_bodyids == {4: 4.0}
    assert prof.untyped_upstream_2hop == {4: {'H1': 2.0}}
    assert prof.untyped_upstream_2hop_ranks == {4: {'H1': 1}}
    assert prof.downstream_partners['D'] == pytest.approx(5.0)
    assert prof.top_k_bodyid_used == 5

    # memory cache hit on second call
    assert profiler.get_profile('MyType', DS) is prof
    # force_refresh bypasses cache
    assert profiler.get_profile('MyType', DS, force_refresh=True) is not prof


def test_get_profile_bodyid_writes_batch_file(profiler, monkeypatch, tmp_path):
    up = _conn_df([('A', 10.0, 1)])
    down = _conn_df([('B', 2.0, 2)])
    _patch_queries(monkeypatch, profiler, up, down)
    prof = profiler.get_profile(720575, DS)
    assert prof.neuron_id == 720575
    batch_dir = profiler._get_profile_batch_dir(DS)
    assert batch_dir.exists()
    assert any(batch_dir.iterdir())


def test_load_from_cache_required_top_k(profiler):
    p = _profile('cached', top_k_bodyid_used=5)
    profiler._save_to_cache(p)
    assert profiler._load_from_cache('cached', DS) is p
    assert profiler._load_from_cache('cached', DS, required_top_k=10) is None


def test_load_from_cache_missing(profiler):
    assert profiler._load_from_cache(999999, DS) is None
    assert profiler._load_cache_dataframe('nonexistent_dataset') is None


def test_save_to_cache_str_neuron_id_memory_only(profiler, tmp_path):
    p = _profile('type_level')
    profiler._save_to_cache(p)
    assert profiler._memory_cache[('type_level', DS)] is p
    batch_dir = profiler._get_profile_batch_dir(DS)
    assert (not batch_dir.exists()) or not any(batch_dir.iterdir())


def test_deferred_cache_writes_and_flush(profiler, tmp_path):
    profiler._defer_cache_writes = True
    p_int = _profile(424242)
    profiler._save_to_cache(p_int)
    assert '424242' in profiler._pending_cache_writes[DS]
    profiler.flush_pending_cache_writes(silent=True)
    assert profiler._pending_cache_writes.get(DS, {}) == {}

    # type-level profiles queued but skipped on flush
    profiler._pending_cache_writes = {DS: {'str_id': _profile('str_id')}}
    profiler.flush_pending_cache_writes(silent=False)
    assert profiler._pending_cache_writes.get(DS, {}) == {}

    # flush with nothing pending returns immediately
    profiler._pending_cache_writes = {}
    profiler.flush_pending_cache_writes()


def test_profile_row_roundtrip(profiler):
    p = _profile(555, upstream={'A': 3.0},
                 partner_type_mapping_upstream={1: 'A'},
                 typed_upstream_bodyids={1: 3.0})
    row = profiler._profile_to_row(p)
    assert isinstance(row['upstream_partners'], str)
    restored = profiler._row_to_profile(pd.Series(row))
    assert restored.upstream_partners == {'A': 3.0}
    # Fixed: mapping keys are restored to ints after the JSON round-trip,
    # matching from_dict()'s contract.
    assert restored.partner_type_mapping_upstream == {1: 'A'}


# ---------------------------------------------------------------------------
# ConnectivityProfiler: data availability / clients
# ---------------------------------------------------------------------------

def test_ensure_data_available_local_missing(profiler):
    with pytest.raises(DataNotAvailableError):
        profiler.ensure_data_available('flywire_FAFB_missing_v9', raise_on_missing=True)
    assert profiler.ensure_data_available('flywire_FAFB_missing_v9',
                                          raise_on_missing=False) is False


def test_ensure_data_available_neuprint_paths(profiler, monkeypatch):
    # client None -> raise / False
    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: None)
    with pytest.raises(DataNotAvailableError):
        profiler.ensure_data_available('hemibrain:v1.2.1', raise_on_missing=True)
    assert profiler.ensure_data_available('hemibrain:v1.2.1', raise_on_missing=False) is False

    # client present -> True
    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: object())
    assert profiler.ensure_data_available('hemibrain:v1.2.1') is True

    # client creation error -> raise / False
    def _boom(d):
        raise RuntimeError('no credentials')
    monkeypatch.setattr(profiler, '_get_client_for_dataset', _boom)
    with pytest.raises(DataNotAvailableError):
        profiler.ensure_data_available('hemibrain:v1.2.1')
    assert profiler.ensure_data_available('hemibrain:v1.2.1', raise_on_missing=False) is False


def test_get_data_status(profiler, monkeypatch):
    status = profiler.get_data_status(['flywire_FAFB_missing_v9'])
    entry = status['flywire_FAFB_missing_v9']
    assert entry['available'] is False
    assert entry['type'] == 'local'
    assert entry['error']

    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: object())
    status2 = profiler.get_data_status(['hemibrain:v1.2.1'])
    assert status2['hemibrain:v1.2.1']['available'] is True
    assert status2['hemibrain:v1.2.1']['type'] == 'neuprint'


def test_normalize_neuprint_dataset_name(profiler):
    assert profiler._normalize_neuprint_dataset_name('hemibrain:v1.2.1') == 'hemibrain:v1.2.1'
    assert profiler._normalize_neuprint_dataset_name('hemibrain_v1_2_1') == 'hemibrain:v1.2.1'
    assert profiler._normalize_neuprint_dataset_name('male_cns_v0_9') == 'male-cns:v0.9'
    assert profiler._normalize_neuprint_dataset_name('plainname') == 'plainname'


def test_get_client_for_dataset(profiler):
    # local dataset -> None (cached)
    assert profiler._get_client_for_dataset('flywire_FAFB_v783') is None
    assert 'flywire_FAFB_v783' in profiler._clients
    # neuprint dataset with injected client
    fake = object()
    profiler.client = fake
    assert profiler._get_client_for_dataset('hemibrain:v1.2.1') is fake


def test_normalize_types_vectorized(profiler):
    cfg = FuzzyMatchConfig(enabled=True, strip_lr_suffix=True)
    out = profiler._normalize_types_vectorized(pd.Series(['T4a_R', 'T4a']), cfg)
    assert list(out) == ['T4a', 'T4a']


def test_config_hash_and_cache_paths(profiler, tmp_path):
    h1 = profiler._get_config_hash()
    h2 = profiler._get_config_hash()
    assert isinstance(h1, str) and h1 == h2
    parquet_path = profiler._get_cache_parquet_path(DS)
    assert str(parquet_path).startswith(str(tmp_path))
    assert str(parquet_path).endswith('.parquet')
    assert str(profiler._get_profile_batch_dir(DS)).startswith(str(tmp_path))


def test_clear_cache(profiler):
    profiler._save_to_cache(_profile('todelete'))
    assert ('todelete', DS) in profiler._memory_cache
    profiler.clear_cache(DS)
    assert ('todelete', DS) not in profiler._memory_cache


# ---------------------------------------------------------------------------
# Appended coverage tests - helpers / fixtures
# ---------------------------------------------------------------------------

import sys
import types as _types


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """Redirect cp_module.__file__ so Path(__file__)-relative datasets/ and
    neuron_indexes/ paths resolve inside tmp_path (hermetic, no repo data)."""
    fake_file = tmp_path / 'src' / 'comparison' / 'connectivity_profiler.py'
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text('')
    monkeypatch.setattr(cp_module, '__file__', str(fake_file))
    return tmp_path


def _conn_cache_df():
    """Small synthetic connection table (pre -> post)."""
    return pd.DataFrame({
        'bodyId_pre': [1, 1, 2, 3],
        'bodyId_post': [10, 11, 10, 12],
        'type_pre': ['A', 'A', 'B', 'C'],
        'type_post': ['X', 'X', 'X', 'Y'],
        'weight': [5.0, 3.0, 2.0, 1.0],
    })


def _conn_indexes():
    return {
        'bodyid_pre_index': {'1': [0, 1], '2': [2], '3': [3]},
        'bodyid_post_index': {'10': [0, 2], '11': [1], '12': [3]},
    }


def _inject_conn_cache(dataset, conn_df=None, with_indexes=True):
    safe = dataset.replace(':', '_').replace('.', '_')
    entry = {}
    if conn_df is not None:
        entry['conn_df'] = conn_df
    if with_indexes and conn_df is not None:
        entry.update(_conn_indexes())
    cp_module._PROFILER_CONN_CACHE[safe] = entry
    return safe


def _write_neurons_csv(fake_repo, dataset, columns):
    safe = dataset.replace(':', '_').replace('.', '_')
    ds_dir = fake_repo / 'datasets' / safe
    ds_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(
        ds_dir / f'{safe}_allneurons_neuron_df.csv', index=False)
    return ds_dir


# ---------------------------------------------------------------------------
# Profile cache persistence: consolidation / atomic save / read / stats
# ---------------------------------------------------------------------------

def test_consolidate_profile_batch_files_polars(profiler):
    # empty batch dir -> 0
    assert profiler._consolidate_profile_batch_files(DS) == 0

    profiles = {str(i): _profile(i, upstream={'A': float(i)})
                for i in (101, 102)}
    profiler._save_profiles_to_cache_batch(profiles, DS)
    batch_dir = profiler._get_profile_batch_dir(DS)
    assert list(batch_dir.glob('*.parquet'))

    count = profiler._consolidate_profile_batch_files(DS)
    assert count == 1  # one batch file holding 2 profiles
    assert profiler._get_cache_parquet_path(DS).exists()
    assert not batch_dir.exists()
    assert set(profiler._disk_cache_df[DS]['neuron_id']) == {101, 102}
    assert '101' in profiler._disk_cache_index[DS]

    # empty batch dir (no files) -> 0
    batch_dir.mkdir(parents=True, exist_ok=True)
    assert profiler._consolidate_profile_batch_files(DS) == 0

    # merge into existing main cache, keep batch files this time
    profiler._save_profile_to_batch_file(_profile(103))
    count2 = profiler._consolidate_profile_batch_files(DS, delete_after=False)
    assert count2 == 1
    assert len(profiler._disk_cache_df[DS]) == 3
    # duplicate neuron_id keeps last occurrence
    profiler._save_profile_to_batch_file(_profile(103, upstream={'Z': 9.0}))
    profiler._consolidate_profile_batch_files(DS)
    assert len(profiler._disk_cache_df[DS]) == 3


def test_consolidate_profile_batch_files_pandas_fallback(profiler, monkeypatch):
    monkeypatch.setitem(sys.modules, 'polars', None)
    profiles = {str(i): _profile(i) for i in (201, 202)}
    profiler._save_profiles_to_cache_batch(profiles, DS)
    count = profiler._consolidate_profile_batch_files(DS)
    assert count == 1
    assert profiler._get_cache_parquet_path(DS).exists()
    assert len(profiler._disk_cache_df[DS]) == 2


def test_save_cache_dataframe_atomic(profiler):
    rows = [profiler._profile_to_row(_profile(i)) for i in (1, 2)]
    df = pd.DataFrame(rows)
    profiler._save_cache_dataframe(df, DS)
    path = profiler._get_cache_parquet_path(DS)
    assert path.exists()
    assert not path.with_suffix('.parquet.tmp').exists()
    assert len(profiler._disk_cache_df[DS]) == 2
    profiler._disk_cache_df.pop(DS)
    reloaded = profiler._load_cache_dataframe(DS, force_reload=True)
    assert len(reloaded) == 2


def test_save_cache_dataframe_polars_large(profiler):
    df = pd.DataFrame({'neuron_id': list(range(5001)),
                       'dataset': [DS] * 5001})
    profiler._save_cache_dataframe(df, DS)
    assert profiler._get_cache_parquet_path(DS).exists()
    assert len(profiler._disk_cache_df[DS]) == 5001


def test_save_profiles_to_cache_batch_variants(profiler):
    # empty -> early return
    profiler._save_profiles_to_cache_batch({}, DS)
    # mixed int/str ids: memory gets both, disk batch file gets int only
    mixed = {'1': _profile(1), 'T': _profile('T')}
    profiler._save_profiles_to_cache_batch(mixed, DS, silent=False)
    assert profiler._memory_cache[('1', DS)].neuron_id == 1
    assert profiler._memory_cache[('T', DS)].neuron_id == 'T'
    assert len(list(profiler._get_profile_batch_dir(DS).glob('*.parquet'))) == 1
    # use_cache disabled -> memory only
    profiler.config = ProfilerConfig(use_cache=False)
    profiler._save_profiles_to_cache_batch({'2': _profile(2)}, DS, silent=True)
    assert ('2', DS) in profiler._memory_cache
    assert len(list(profiler._get_profile_batch_dir(DS).glob('*.parquet'))) == 1
    # type-level only -> no disk write
    profiler.config = ProfilerConfig()
    profiler._save_profiles_to_cache_batch({'U': _profile('U')}, DS)
    assert len(list(profiler._get_profile_batch_dir(DS).glob('*.parquet'))) == 1


def test_read_cache_and_stats(profiler):
    assert profiler.read_connectivity_profile_cache(DS) == {}
    stats0 = profiler.get_cache_stats(DS)
    assert stats0['total_profiles'] == 0
    assert stats0['cache_modified'] is None

    profiles = {str(i): _profile(i, top_k_bodyid_used=5) for i in (11, 12, 13)}
    profiler._save_profiles_to_cache_batch(profiles, DS)
    profiler.consolidate_profile_cache(DS)

    # slow path: all rows
    all_profiles = profiler.read_connectivity_profile_cache(DS)
    assert len(all_profiles) == 3
    # fast index path with selection + min_top_k pass
    sel = profiler.read_connectivity_profile_cache(
        DS, neuron_types=['11', '99'], min_top_k=3)
    assert set(sel.keys()) == {'11'}
    assert ('11', DS) in profiler._memory_cache
    # min_top_k filters everything out
    sel2 = profiler.read_connectivity_profile_cache(
        DS, neuron_types=['11'], min_top_k=50)
    assert sel2 == {}

    stats = profiler.get_cache_stats(DS)
    assert stats['total_profiles'] == 3
    assert stats['top_k_distribution'].get(5) == 3
    assert stats['cache_modified']
    assert stats['total_size_mb'] >= 0
    assert len(stats['neuron_types']) == 3


# ---------------------------------------------------------------------------
# Connection data access: neuprint query, FNC cache, disk load, local query
# ---------------------------------------------------------------------------

class _FakeNeuprintClient:
    def __init__(self, up_df, down_df, fail=False):
        self.up_df = up_df
        self.down_df = down_df
        self.fail = fail
        self.queries = []

    def fetch_custom(self, query):
        self.queries.append(query)
        if self.fail:
            raise RuntimeError('query failed')
        if 'pre.bodyId AS partner_bodyId' in query:
            return self.up_df
        return self.down_df


def test_query_connections_neuprint(profiler, monkeypatch):
    up = pd.DataFrame({'partner_bodyId': [1], 'partner_type': ['A'],
                       'neuron_bodyId': [5], 'weight': [3.0]})
    down = pd.DataFrame({'partner_bodyId': [2], 'partner_type': ['B'],
                         'neuron_bodyId': [5], 'weight': [2.0]})
    client = _FakeNeuprintClient(up, down)
    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: client)

    u, d = profiler._query_connections_neuprint('MyType', 'hemibrain:v1.2.1')
    assert len(u) == 1 and len(d) == 1
    # regex type query
    profiler._query_connections_neuprint('My.*', 'hemibrain:v1.2.1')
    assert any("=~" in q for q in client.queries)
    # bodyId and list queries
    profiler._query_connections_neuprint(123, 'hemibrain:v1.2.1')
    profiler._query_connections_neuprint([1, 2], 'hemibrain:v1.2.1')
    # unsupported neuron type
    with pytest.raises(ValueError):
        profiler._query_connections_neuprint(3.5, 'hemibrain:v1.2.1')
    # query failure -> empty frames
    client.fail = True
    u, d = profiler._query_connections_neuprint('X', 'hemibrain:v1.2.1')
    assert u.empty and d.empty
    # no client -> empty frames
    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: None)
    u, d = profiler._query_connections_neuprint('X', 'hemibrain:v1.2.1')
    assert u.empty and d.empty


def test_get_cached_conn_df_profiler_cache_hit(profiler):
    df = _conn_cache_df()
    safe = _inject_conn_cache(DS, conn_df=df, with_indexes=False)
    try:
        assert profiler._get_cached_conn_df(DS) is df
    finally:
        cp_module._PROFILER_CONN_CACHE.pop(safe, None)


def test_get_cached_conn_df_fnc_cache(profiler, monkeypatch):
    safe = DS.replace(':', '_').replace('.', '_')
    cp_module._PROFILER_CONN_CACHE.pop(safe, None)
    df = _conn_cache_df()
    fake_coana = _types.ModuleType('coana')
    fake_coana._FNC_CACHE = {
        safe: {
            'conn_df': df,
            'conn_index': {'1': [0, 1], '2': [2], '3': [3]},
            'conn_index_post': {'10': [0, 2], '11': [1], '12': [3]},
        }
    }
    monkeypatch.setitem(sys.modules, 'coana', fake_coana)
    try:
        out = profiler._get_cached_conn_df(DS)
        assert out is df
        entry = cp_module._PROFILER_CONN_CACHE[safe]
        assert entry['bodyid_post_index'] == {'10': [0, 2], '11': [1], '12': [3]}

        # variant without FNC post index -> profiler builds it
        cp_module._PROFILER_CONN_CACHE.pop(safe, None)
        fake_coana._FNC_CACHE[safe] = {'conn_df': df.copy(), 'conn_index': {}}
        profiler._get_cached_conn_df(DS)
        built = cp_module._PROFILER_CONN_CACHE[safe]['bodyid_post_index']
        assert sorted(built.keys()) == ['10', '11', '12']
    finally:
        cp_module._PROFILER_CONN_CACHE.pop(safe, None)


def test_get_cached_conn_df_disk_load(profiler, fake_repo):
    ds = 'testconn_v1'
    safe = ds
    ds_dir = fake_repo / 'datasets' / safe
    ds_dir.mkdir(parents=True)
    _conn_cache_df().to_csv(ds_dir / f'{safe}_merged_connections.csv',
                            index=False)
    cp_module._PROFILER_CONN_CACHE.pop(safe, None)
    try:
        df = profiler._get_cached_conn_df(ds)
        assert df is not None and len(df) == 4
        entry = cp_module._PROFILER_CONN_CACHE[safe]
        assert entry['bodyid_pre_index']['1'] == [0, 1]
        assert entry['bodyid_post_index']['10'] == [0, 2]
        # cached on second call
        assert profiler._get_cached_conn_df(ds) is df
    finally:
        cp_module._PROFILER_CONN_CACHE.pop(safe, None)


def test_get_cached_conn_df_disk_missing(profiler, fake_repo):
    ds = 'noconn_v1'
    cp_module._PROFILER_CONN_CACHE.pop(ds, None)
    try:
        assert profiler._get_cached_conn_df(ds) is None
        assert cp_module._PROFILER_CONN_CACHE[ds]['conn_df'] is None
    finally:
        cp_module._PROFILER_CONN_CACHE.pop(ds, None)


def test_query_connections_local(profiler):
    profiler.config.min_synapse_threshold = 0
    safe = _inject_conn_cache(DS, conn_df=_conn_cache_df())
    try:
        # int query: neuron 10 receives from 1 and 2
        up, down = profiler._query_connections_local(10, DS)
        assert sorted(up['partner_bodyId'].tolist()) == [1, 2]
        assert down.empty
        # neuron 1 sends to 10 and 11
        up, down = profiler._query_connections_local(1, DS)
        assert sorted(down['partner_bodyId'].tolist()) == [10, 11]
        assert up.empty
        # list query
        up, down = profiler._query_connections_local([10, 12], DS)
        assert len(up) == 3
        # exact type query: 'X' appears in type_post (rows 0,1,2)
        up, down = profiler._query_connections_local('X', DS)
        assert len(up) == 3
        assert down.empty
        # regex type query: 'A*' matches type_pre 'A' (rows 0,1)
        up, down = profiler._query_connections_local('A*', DS)
        assert up.empty
        assert len(down) == 2
        # unsupported type
        with pytest.raises(ValueError):
            profiler._query_connections_local(3.5, DS)
    finally:
        cp_module._PROFILER_CONN_CACHE.pop(safe, None)


def test_query_connections_local_no_data(profiler, fake_repo):
    ds = 'noconn2_v1'
    cp_module._PROFILER_CONN_CACHE.pop(ds, None)
    try:
        up, down = profiler._query_connections_local(1, ds)
        assert up.empty and down.empty
    finally:
        cp_module._PROFILER_CONN_CACHE.pop(ds, None)


def test_fetch_2hop_partners(profiler):
    profiler.config.min_synapse_threshold = 0
    safe = _inject_conn_cache(DS, conn_df=_conn_cache_df())
    try:
        res = profiler._fetch_2hop_partners([10, 99], DS, 'upstream')
        assert res[99] == ({}, {})
        weights, ranks = res[10]
        assert set(weights) == {'A', 'B'}
        assert weights['A'] == pytest.approx(5.0 / 7.0)
        assert set(ranks) == {'A', 'B'}
        # downstream direction
        res_d = profiler._fetch_2hop_partners([1], DS, 'downstream')
        w1, _ = res_d[1]
        assert w1 == {'X': 1.0}
        # empty input
        assert profiler._fetch_2hop_partners([], DS, 'upstream') == {}
    finally:
        cp_module._PROFILER_CONN_CACHE.pop(safe, None)


def test_fetch_2hop_partners_pandas_fallback(profiler, monkeypatch):
    profiler.config.min_synapse_threshold = 0
    safe = _inject_conn_cache(DS, conn_df=_conn_cache_df())
    monkeypatch.setitem(sys.modules, 'polars', None)
    try:
        res = profiler._fetch_2hop_partners([10, 99], DS, 'upstream')
        weights, _ = res[10]
        assert set(weights) == {'A', 'B'}
        assert res[99] == ({}, {})
    finally:
        cp_module._PROFILER_CONN_CACHE.pop(safe, None)


def test_fetch_2hop_partners_missing_columns(profiler):
    profiler.config.min_synapse_threshold = 0
    df = pd.DataFrame({'bodyId_pre': [1], 'bodyId_post': [10],
                       'weight': [1.0]})  # no type columns
    safe = _inject_conn_cache(DS, conn_df=df, with_indexes=True)
    try:
        res = profiler._fetch_2hop_partners([10], DS, 'upstream')
        assert res == {10: ({}, {})}
    finally:
        cp_module._PROFILER_CONN_CACHE.pop(safe, None)


def test_fetch_2hop_partners_no_conn_data(profiler, fake_repo):
    ds = 'noconn3_v1'
    cp_module._PROFILER_CONN_CACHE.pop(ds, None)
    try:
        assert profiler._fetch_2hop_partners([1], ds, 'upstream') == {}
    finally:
        cp_module._PROFILER_CONN_CACHE.pop(ds, None)


# ---------------------------------------------------------------------------
# _process_connections edge branches
# ---------------------------------------------------------------------------

def test_process_connections_untyped_bodyid_cast_fallback(profiler):
    df = pd.DataFrame({
        'partner_type': [None, 'A'],
        'weight': [4.0, 2.0],
        'partner_bodyId': ['notanumber', 7],
    })
    res = profiler._process_connections(df, 'upstream', 5)
    # untyped bodyId not castable to int -> dropped via fallback loop
    assert res[9] == {}
    assert res[10] == {7: 2.0}


def test_process_connections_pandas_fallback(profiler, monkeypatch):
    monkeypatch.setitem(sys.modules, 'polars', None)
    profiler.config = ProfilerConfig(include_untyped_partners=True)
    df = pd.DataFrame({
        'partner_type': ['A', 'A', None, 'B'],
        'weight': [5.0, 3.0, 2.0, 1.0],
        'partner_bodyId': [1, 2, 3, 4],
    })
    res = profiler._process_connections(df, 'upstream', 10)
    partners, type_mapping, typed_bodyids = res[0], res[7], res[10]
    assert partners['A'] == pytest.approx(8.0)
    assert 'untyped' in partners
    assert type_mapping == {1: 'A', 2: 'A', 4: 'B'}
    assert typed_bodyids == {1: 5.0, 2: 3.0, 4: 1.0}


# ---------------------------------------------------------------------------
# get_profiles_batch / _build_profile_from_cache_direct
# ---------------------------------------------------------------------------

def test_get_profiles_batch_skip_cache(profiler, monkeypatch):
    up = _conn_df([('A', 10.0, 1)])
    down = _conn_df([('B', 2.0, 2)])
    monkeypatch.setattr(profiler, '_get_cached_conn_df', lambda ds: up)
    monkeypatch.setattr(profiler, '_query_connections_local',
                        lambda n, ds: (up, down))
    res = profiler.get_profiles_batch(
        [1, 2], DS, skip_profile_cache=True, show_progress=False)
    assert set(res) == {1, 2}
    assert res[1].neuron_id == 1
    assert res[1].upstream_partners['A'] == pytest.approx(10.0)


def test_get_profiles_batch_via_get_profile_and_errors(profiler, monkeypatch):
    def fake_get_profile(neuron, dataset, force_refresh=False):
        if neuron == 'bad':
            raise RuntimeError('nope')
        return _profile(neuron)
    monkeypatch.setattr(profiler, '_get_cached_conn_df', lambda ds: None)
    monkeypatch.setattr(profiler, 'get_profile', fake_get_profile)
    res = profiler.get_profiles_batch(['x', 'bad'], DS, show_progress=False)
    assert set(res) == {'x'}

    # no local data for a local dataset -> _build_profile_from_cache_direct None
    monkeypatch.setattr(
        profiler, '_query_connections_local',
        lambda n, ds: (pd.DataFrame(), pd.DataFrame()))
    res2 = profiler.get_profiles_batch(
        [5], DS, skip_profile_cache=True, show_progress=False)
    assert res2 == {}


def test_build_profile_from_cache_direct_type_query(profiler, monkeypatch):
    up = pd.DataFrame({'partner_type': ['A'], 'weight': [3.0],
                       'partner_bodyId': [9], 'neuron_bodyId': [77]})
    down = pd.DataFrame({'partner_type': pd.Series(dtype=str),
                         'weight': pd.Series(dtype=float),
                         'partner_bodyId': pd.Series(dtype=int),
                         'neuron_bodyId': pd.Series(dtype=int)})
    monkeypatch.setattr(profiler, '_query_connections_local',
                        lambda n, ds: (up, down))
    p = profiler._build_profile_from_cache_direct('SomeType', DS)
    assert p.neuron_type == 'SomeType'
    assert p.num_neurons_aggregated == 1
    assert p.upstream_partners['A'] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Neuron-table lookups (local files via fake_repo, neuprint via fake client)
# ---------------------------------------------------------------------------

def test_get_bodyids_for_type_local(profiler, fake_repo):
    ds = 'flywire_FAFB_fake_v1'
    _write_neurons_csv(fake_repo, ds,
                       {'bodyId': [1, 2, 3], 'type': ['Mi1', 'Mi1', 'T4']})
    assert profiler.get_bodyids_for_type('Mi1', ds) == [1, 2]
    assert profiler.get_bodyids_for_type('Nope', ds) == []
    assert profiler.get_bodyids_for_type('Mi1', 'flywire_FAFB_absent_v9') == []


def test_get_bodyids_for_type_neuprint(profiler, monkeypatch):
    class C:
        def fetch_custom(self, q):
            return pd.DataFrame({'bodyId': ['5', '6']})
    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: C())
    assert profiler.get_bodyids_for_type('Mi1', 'hemibrain:v1.2.1') == [5, 6]
    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: None)
    assert profiler.get_bodyids_for_type('Mi1', 'hemibrain:v1.2.1') == []


def test_get_bodyids_for_type_partial_local_table_falls_back(profiler, monkeypatch):
    """Review 2026-09-16: a PARTIAL local export (e.g. a hemibrain snapshot
    missing the queried type entirely) must not shadow a reachable API for
    non-native datasets, while FAFB/BANC stay local-only on a miss."""
    import pandas as pd
    partial = pd.DataFrame({'bodyId': [1], 'type': ['T1']})
    monkeypatch.setattr(profiler, '_load_local_neuron_frame',
                        lambda d: partial)
    monkeypatch.setattr('comparison.connectivity_profiler.'
                        'is_local_connectome_dataset', lambda d: False)

    class ApiClient:
        def fetch_custom(self, q):
            return pd.DataFrame({'bodyId': ['7']})

    monkeypatch.setattr(profiler, '_get_client_for_dataset',
                        lambda d: ApiClient())
    # local miss -> API fallback
    assert profiler.get_bodyids_for_type('Missing', 'hemibrain:v1.2.1') == [7]

    class Boom:
        def fetch_custom(self, q):
            raise AssertionError('API must not be called on a local hit')

    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: Boom())
    # local hit still wins without touching the API
    assert profiler.get_bodyids_for_type('T1', 'hemibrain:v1.2.1') == [1]

    # a natively-local dataset never falls back
    monkeypatch.setattr('comparison.connectivity_profiler.'
                        'is_local_connectome_dataset', lambda d: True)
    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: Boom())
    assert profiler.get_bodyids_for_type('Missing', 'flywire_FAFB_v783') == []


def test_get_bodyids_for_type_prioritized_columns(profiler, fake_repo):
    """Same prioritized column search as the connection tabs: the ``type``
    column wins, but names living only in ``cell_type`` (FAFB's
    circadian_clock) still resolve."""
    ds = 'flywire_FAFB_fake_v1'
    _write_neurons_csv(fake_repo, ds, {
        'bodyId': [1, 2, 3, 4],
        'type': ['Mi1', 'Mi1', '', ''],
        'cell_type': ['other', 'other', 'circadian_clock', 'Mi1'],
    })
    # Name only present in cell_type resolves via the fallback columns
    assert profiler.get_bodyids_for_type('circadian_clock', ds) == [3]
    # Name present in both columns: the type column owns the query
    assert profiler.get_bodyids_for_type('Mi1', ds) == [1, 2]
    # Wildcard patterns follow the shared resolver contract too
    assert profiler.get_bodyids_for_type('circ.*', ds) == [3]
    # Frame is memoized: repeated lookups reuse the cached read
    assert profiler._local_neuron_frames


def test_progress_bars_disabled(monkeypatch):
    from comparison.connectivity_profiler import progress_bars_disabled

    assert progress_bars_disabled(False, True) is True
    assert progress_bars_disabled(True, False) is True

    class FakeTTY:
        def isatty(self):
            return True

    class FakePipe:
        def isatty(self):
            return False

    monkeypatch.setattr(cp_module.sys, 'stdout', FakeTTY())
    assert progress_bars_disabled(True, True) is False

    monkeypatch.setattr(cp_module.sys, 'stdout', FakePipe())
    assert progress_bars_disabled(True, True) is True

    # A stream without isatty() (captured output variants) is treated as
    # non-TTY rather than crashing the pipeline.
    monkeypatch.setattr(cp_module.sys, 'stdout', object())
    assert progress_bars_disabled(True, True) is True


def test_list_types_and_load_all(profiler, fake_repo, monkeypatch):
    ds = 'flywire_FAFB_fake_v1'
    _write_neurons_csv(fake_repo, ds,
                       {'bodyId': [1, 2, 3], 'type': ['Mi1', 'T4', None]})
    assert profiler.list_types(dataset=ds) == ['Mi1', 'T4']
    assert profiler.list_types('Mi.*', dataset=ds) == ['Mi1']
    # invalid regex treated as literal -> no match
    assert profiler.list_types('(', dataset=ds) == []
    # missing dataset folder -> []
    assert profiler.list_types(dataset='flywire_FAFB_absent_v9') == []

    # neuprint path
    class C:
        def fetch_custom(self, q):
            return pd.DataFrame({'type': ['B', 'A', None]})
    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: C())
    assert profiler._load_all_types('hemibrain:v1.2.1') == ['A', 'B']
    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: None)
    assert profiler._load_all_types('hemibrain:v1.2.1') == []
    # list_types with no datasets configured
    empty = ConnectivityProfiler([], config=ProfilerConfig(),
                                 cache_dir=str(fake_repo), verbose=False)
    assert empty.list_types() == []


def test_get_type_for_bodyid(profiler, fake_repo, monkeypatch):
    ds = 'flywire_FAFB_fake_v1'
    _write_neurons_csv(fake_repo, ds,
                       {'bodyId': [1, 2], 'type': ['Mi1', 'T4']})
    assert profiler.get_type_for_bodyid(2, ds) == 'T4'
    assert profiler.get_type_for_bodyid(99, ds) is None
    assert profiler.get_type_for_bodyid(1, 'flywire_FAFB_absent_v9') is None

    class C:
        def fetch_custom(self, q):
            return pd.DataFrame({'type': ['Mi1']})

    class Boom:
        def fetch_custom(self, q):
            raise RuntimeError('x')

    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: C())
    assert profiler.get_type_for_bodyid(1, 'hemibrain:v1.2.1') == 'Mi1'
    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: Boom())
    assert profiler.get_type_for_bodyid(1, 'hemibrain:v1.2.1') is None
    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: None)
    assert profiler.get_type_for_bodyid(1, 'hemibrain:v1.2.1') is None


def test_get_types_for_bodyids(profiler, fake_repo, monkeypatch):
    assert profiler.get_types_for_bodyids([], DS) == {}
    ds = 'flywire_FAFB_fake_v1'
    _write_neurons_csv(fake_repo, ds,
                       {'bodyId': [1, 2], 'type': ['Mi1', 'T4']})
    res = profiler.get_types_for_bodyids([1, 2, 3], ds)
    assert res == {1: 'Mi1', 2: 'T4', 3: None}
    assert profiler.get_types_for_bodyids([1], 'flywire_FAFB_absent_v9') == {1: None}

    class C:
        def fetch_custom(self, q):
            return pd.DataFrame({'bodyId': [1], 'type': ['Mi1']})

    class Boom:
        def fetch_custom(self, q):
            raise RuntimeError('x')

    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: C())
    assert profiler.get_types_for_bodyids([1, 2], 'hemibrain:v1.2.1') == {
        1: 'Mi1', 2: None}
    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: Boom())
    assert profiler.get_types_for_bodyids([1], 'hemibrain:v1.2.1') == {1: None}
    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: None)
    assert profiler.get_types_for_bodyids([1], 'hemibrain:v1.2.1') == {1: None}


def test_get_types_for_label(profiler, fake_repo):
    ds = 'flywire_FAFB_fake_v1'
    _write_neurons_csv(fake_repo, ds, {
        'bodyId': [1, 2, 3],
        'type': ['Mi1', 'Mi1', 'T4'],
        'cell_class': ['Vis', 'Vis', 'Vis'],
    })
    res = profiler.get_types_for_label('Vis', ds)
    assert res == {'Mi1': [1, 2], 'T4': [3]}
    assert profiler.get_types_for_label('Nope', ds) == {}
    # non-local dataset -> {}
    assert profiler.get_types_for_label('Vis', 'hemibrain:v1.2.1') == {}
    # missing local table -> {}
    assert profiler.get_types_for_label('Vis', 'flywire_FAFB_absent_v9') == {}


def test_get_type_profile_from_bodyids(profiler, monkeypatch):
    monkeypatch.setattr(profiler, 'get_bodyids_for_type', lambda t, d: [])
    p = profiler.get_type_profile_from_bodyids('Mi1', DS)
    assert p.upstream_partners == {}

    monkeypatch.setattr(profiler, 'get_bodyids_for_type', lambda t, d: [1, 2])

    def boom(neuron, dataset, force_refresh=False):
        raise RuntimeError('no')
    monkeypatch.setattr(profiler, 'get_profile', boom)
    p2 = profiler.get_type_profile_from_bodyids('Mi1', DS)
    assert p2.upstream_partners == {}

    profs = {
        1: _profile(1, upstream={'A': 3.0}, downstream={'B': 1.0}),
        2: _profile(2, upstream={'A': 1.0}, downstream={'C': 2.0}),
    }
    monkeypatch.setattr(
        profiler, 'get_profile',
        lambda n, d, force_refresh=False: profs[n])
    agg = profiler.get_type_profile_from_bodyids('Mi1', DS)
    assert agg.num_neurons_aggregated == 2
    assert agg.upstream_partners['A'] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Vectorized profiles + npz I/O
# ---------------------------------------------------------------------------

def test_vectorized_profiles_and_io(profiler, monkeypatch, tmp_path):
    profs = {
        'T1': _profile('T1', upstream={'A': 3.0, 'B': 1.0},
                       downstream={'X': 2.0}),
        'T2': _profile('T2', upstream={'A': 1.0, 'C': 1.0},
                       downstream={'X': 1.0}),
    }
    monkeypatch.setattr(
        profiler, 'get_profile',
        lambda n, d, force_refresh=False: profs[n])
    res_w = profiler.get_vectorized_profiles(
        ['T1', 'T2'], datasets=[DS], vector_type='weight')
    assert res_w['matrix'].shape[0] == 2
    assert 'A' in res_w['vocabulary']
    res_r = profiler.get_vectorized_profiles(
        ['T1', 'T2'], datasets=[DS], vector_type='rank',
        direction='upstream')
    assert res_r['matrix'].shape[0] == 2

    # failing profile fetch -> empty result
    def boom(n, d, force_refresh=False):
        raise RuntimeError('no')
    monkeypatch.setattr(profiler, 'get_profile', boom)
    res_e = profiler.get_vectorized_profiles(['T1'], datasets=[DS])
    assert res_e['matrix'].shape == (0, 0)
    assert res_e['profiles'] == {}

    # save/load round trip
    monkeypatch.setattr(
        profiler, 'get_profile',
        lambda n, d, force_refresh=False: profs[n])
    out = profiler.save_vectorized_profiles(
        tmp_path / 'vecs.npz', ['T1', 'T2'], datasets=[DS])
    loaded = ConnectivityProfiler.load_vectorized_profiles(out)
    assert loaded['matrix'].shape[0] == 2
    assert loaded['vector_type'] == 'weight'
    assert loaded['direction'] == 'both'


def test_clear_cache_all(profiler, tmp_path):
    profiler._save_to_cache(_profile(111))
    profiler.clear_cache()
    assert profiler._memory_cache == {}
    assert tmp_path.exists()


def test_get_available_types(profiler, fake_repo, monkeypatch):
    ds = 'flywire_FAFB_fake_v1'
    ds_dir = fake_repo / 'datasets' / ds
    ds_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'bodyId': [1, 2], 'type': ['Mi1', 'T4']}).to_csv(
        ds_dir / f'{ds}_neuron_df.csv', index=False)
    assert profiler.get_available_types(ds) == ['Mi1', 'T4']
    assert profiler.get_available_types('flywire_FAFB_absent_v9') is None
    monkeypatch.setattr(profiler, '_get_client_for_dataset', lambda d: None)
    assert profiler.get_available_types('hemibrain:v1.2.1') is None


# ---------------------------------------------------------------------------
# Round-7 cache build + homolog finders (loose/strict/hybrid)
# ---------------------------------------------------------------------------

def test_build_connectivity_profile_cache_config_bug(profiler, monkeypatch):
    # FIXED (2026-09-30 code audit): the config rebuild dropped the
    # invalid cache_dir/verbose kwargs and get_all_types became list_types,
    # so the documented entry point now runs instead of raising TypeError.
    # Pin that the rebuild succeeds and the type listing routes through
    # list_types.
    listed = []
    monkeypatch.setattr(profiler, 'list_types',
                        lambda pattern=None, dataset=None: listed.append(dataset) or ['T1'])
    monkeypatch.setattr(
        profiler, 'get_profile',
        lambda n, d, force_refresh=False: _profile(n))
    out = profiler.build_connectivity_profile_cache(DS, neuron_types=None)
    assert listed == [DS]
    assert set(out) == {'T1'}


def test_find_homologs_loose(profiler, monkeypatch):
    q = _profile('Q', upstream={'A': 3.0, 'B': 1.0}, downstream={'X': 2.0})
    t1 = _profile('T1', upstream={'A': 2.0, 'B': 2.0}, downstream={'X': 1.0})
    t2 = _profile('T2', upstream={'C': 1.0})
    profs = {('Q', 'src_ds'): q, ('T1', 'tgt_ds'): t1, ('T2', 'tgt_ds'): t2}
    monkeypatch.setattr(
        profiler, 'get_profile',
        lambda n, d, force_refresh=False: profs[(n, d)])
    monkeypatch.setattr(profiler, 'list_types',
                        lambda pattern=None, dataset=None: ['T1', 'T2'])
    df = profiler.find_homologs_loose('Q', 'src_ds', 'tgt_ds', top_n=5)
    assert not df.empty
    assert df.iloc[0]['target_type'] == 'T1'
    assert 'weighted_jaccard' in df.columns
    # weights-based variant
    df_w = profiler.find_homologs_loose(
        'Q', 'src_ds', 'tgt_ds', direction='upstream', use_ranks=False)
    assert not df_w.empty
    # no candidate types -> empty
    monkeypatch.setattr(profiler, 'list_types',
                        lambda pattern=None, dataset=None: [])
    assert profiler.find_homologs_loose('Q', 'src_ds', 'tgt_ds').empty


def test_find_homologs_strict(profiler, monkeypatch):
    q = _profile('Q', upstream={'A': 3.0, 'B': 2.0, 'C': 1.0},
                 untyped_upstream_2hop={9: {'H': 1.0}})
    t1 = _profile('T1', upstream={'A': 1.0, 'B': 3.0, 'C': 2.0},
                  untyped_upstream_2hop={8: {'H': 0.5, 'G': 0.5}})
    t2 = _profile('T2', upstream={'Z': 1.0})
    profs = {('Q', 'src_ds'): q, ('T1', 'tgt_ds'): t1, ('T2', 'tgt_ds'): t2}
    monkeypatch.setattr(
        profiler, 'get_profile',
        lambda n, d, force_refresh=False: profs[(n, d)])
    monkeypatch.setattr(profiler, 'list_types',
                        lambda pattern=None, dataset=None: ['T1', 'T2'])
    df = profiler.find_homologs_strict(
        'Q', 'src_ds', 'tgt_ds', top_n=5, min_common_partners=3)
    assert len(df) == 1
    row = df.iloc[0]
    assert row['target_type'] == 'T1'
    assert row['jaccard_typed'] == pytest.approx(1.0)
    assert row['jaccard_2hop'] == pytest.approx(0.5)
    assert 'combined_score' in df.columns
    # threshold too high -> nothing passes
    empty = profiler.find_homologs_strict(
        'Q', 'src_ds', 'tgt_ds', min_common_partners=10)
    assert empty.empty
    # no target types -> empty
    monkeypatch.setattr(profiler, 'list_types',
                        lambda pattern=None, dataset=None: [])
    assert profiler.find_homologs_strict('Q', 'src_ds', 'tgt_ds').empty


def test_get_hybrid_profile_vector(profiler, monkeypatch):
    p = _profile('H', upstream={'A': 1.0}, downstream={'B': 1.0},
                 untyped_upstream_2hop={5: {'Z': 1.0}})
    monkeypatch.setattr(
        profiler, 'get_profile',
        lambda n, d, force_refresh=False: p)
    hybrid = profiler.get_hybrid_profile_vector('H', DS)
    assert hybrid['upstream']['typed'] == {'A': 1.0}
    assert hybrid['upstream']['untyped_2hop'] == {5: {'Z': 1.0}}
    assert 'downstream' in hybrid
    up_only = profiler.get_hybrid_profile_vector('H', DS, direction='upstream')
    assert 'downstream' not in up_only


def test_consolidate_quarantines_unreadable_main_cache(profiler):
    """An unreadable main cache must never be overwritten or lost: it is
    quarantined aside, consolidation aborts, and batch files survive
    (regression: the old path dropped the unreadable cache, replaced it
    with batch-only rows, then deleted the batches)."""
    main_path = profiler._get_cache_parquet_path(DS)
    main_path.parent.mkdir(parents=True, exist_ok=True)
    main_path.write_bytes(b'this is not a parquet file')
    profiler._save_profile_to_batch_file(_profile(301))
    batch_files = list(profiler._get_profile_batch_dir(DS).glob('*.parquet'))
    assert batch_files

    assert profiler._consolidate_profile_batch_files(DS) == 0

    # original path no longer holds the corrupt bytes; a quarantine copy does
    assert not main_path.exists() or main_path.read_bytes() != b'this is not a parquet file'
    quarantined = list(main_path.parent.glob('connectivity_profiles.corrupt-*'))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b'this is not a parquet file'
    # batch files were kept for the next (post-recovery) consolidation
    assert list(profiler._get_profile_batch_dir(DS).glob('*.parquet'))


def test_consolidate_quarantine_pandas_fallback(profiler, monkeypatch):
    monkeypatch.setitem(sys.modules, 'polars', None)
    main_path = profiler._get_cache_parquet_path(DS)
    main_path.parent.mkdir(parents=True, exist_ok=True)
    main_path.write_bytes(b'corrupt')
    profiler._save_profile_to_batch_file(_profile(401))
    assert profiler._consolidate_profile_batch_files(DS) == 0
    assert list(main_path.parent.glob('connectivity_profiles.corrupt-*'))
    assert list(profiler._get_profile_batch_dir(DS).glob('*.parquet'))


def test_profile_cache_gates_on_config_hash(profiler):
    """A profile persisted under one min_synapse_threshold must not be
    reused after the config changes (F-XD-002: the cache used to gate only
    on top_k, silently reusing stale profiles)."""
    profiler._save_profile_to_batch_file(_profile(501))
    assert profiler._consolidate_profile_batch_files(DS) == 1
    assert profiler._load_from_cache(501, DS) is not None

    # Same reader, different config that shapes profile content -> stale.
    original = profiler.config.min_synapse_threshold
    profiler.config.min_synapse_threshold = original + 40
    assert profiler._load_from_cache(501, DS) is None
    # ...in the memory tier too (the stamp travels with the profile).
    key = (str(501), DS)
    assert key not in profiler._memory_cache or \
        profiler._load_from_cache(501, DS) is None

    # Restoring the config makes the same row valid again.
    profiler.config.min_synapse_threshold = original
    assert profiler._load_from_cache(501, DS) is not None
