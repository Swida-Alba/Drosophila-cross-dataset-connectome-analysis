"""plan-shortest-batched-discovery — Phase 0/B harness + equivalence gates.

Covers the plan's hard gates on a synthetic fixture designed around the
review-round findings:

* overlapping upstream cones across >= 3 discovered targets;
* a node (A) sitting at DIFFERENT reverse depths for two targets, so its
  incoming edge rows are fetched into two ``conn_layer`` frames and
  ``add_edge`` sums them — the weight-multiplicity case a pair-deduped
  store would silently break (review changelog 1);
* equal bottlenecks straddling batch boundaries (every path through Y/T4
  ties with the S2 cone paths at 10) — the nested ``heapq.merge`` order
  pin (review changelog 3);
* a target reached by every source within 2 hops (per-target early stop)
  and a target no source reaches whose chain keeps a frontier alive at
  the depth cap (``discovery_complete=False`` in both modes).

Hard gates (plan §5 Phase B / §6.1): batched runs must reproduce the
monolithic path SET, the merged-stream ROW ORDER, the exported
StrongestFirst stats, the fetch sequence, and the run diagnostics.
"""

import json
import os

import polars as pl
import pytest

import coana
from vispath_pkg.fast_graph_core import (
    FastGraph,
    drain_shortest_merge,
    merge_shortest_batch_emissions,
)
from shortest_discovery_store import (
    compose_batches,
    load_store_meta,
    resolve_batching,
)

from tests.core.test_pathfinding import _make_pipeline_fc, _PIPELINE_TYPES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# bodyId -> type for the synthetic nodes (the offline pipeline fixture
# types every connection endpoint through this table).
_SYNTH_TYPES = {
    'S1': 'TS1', 'S2': 'TS2', 'S3': 'TS3',
    'A': 'TA', 'B': 'TB', 'C': 'TC', 'X': 'TX', 'Y': 'TY', 'D': 'TD',
    'Z0': 'TZ', 'Z1': 'TZ', 'Z2': 'TZ', 'Z3': 'TZ',
    'T1': 'TT1', 'T2': 'TT2', 'T3': 'TT3', 'T4': 'TT4',
}

# (pre, post, weight) — see the module docstring for what each cone pins.
_SYNTH_EDGES = [
    # T1 cone: S1 -> A -> T1 (2 hops, bottleneck 7)
    ('S1', 'A', 7), ('A', 'T1', 12),
    # T2 cone: S2 -> B -> C -> T2 (3 hops, bottleneck 10) plus the shared
    # A -> X -> T2 branch (S1 -> A -> X -> T2, bottleneck 7)
    ('S2', 'B', 10), ('B', 'C', 10), ('C', 'T2', 10),
    ('A', 'X', 12), ('X', 'T2', 12),
    # T3: no source reaches it; the Z chain keeps its frontier alive at
    # the depth cap (discovery_complete=False)
    ('D', 'T3', 10), ('S3', 'D', 10),
    ('Z2', 'Z3', 10), ('Z3', 'T3', 10), ('Z1', 'Z2', 10), ('Z0', 'Z1', 10),
    # T4: every source reaches it in 2 hops -> per-target early stop;
    # S2->T4 and S3->T4 tie with the S2 cone paths at bottleneck 10
    ('S1', 'Y', 9), ('S2', 'Y', 10), ('S3', 'Y', 11), ('Y', 'T4', 20),
]

_SYNTH_SOURCES = ('S1', 'S2', 'S3')
_SYNTH_TARGETS = ('T1', 'T2', 'T3', 'T4')


@pytest.fixture()
def synth_types():
    """Extend the shared fixture type table for the synthetic nodes."""
    saved = dict(_PIPELINE_TYPES)
    _PIPELINE_TYPES.update(_SYNTH_TYPES)
    yield _PIPELINE_TYPES
    _PIPELINE_TYPES.clear()
    _PIPELINE_TYPES.update(saved)


def _make_fc(monkeypatch, tmp_path, *, budget=None, fixed=0,
             max_paths=None, retention=None, **kwargs):
    # Same offline guards as the shortest-path optimization tests: the
    # materialization path must never reach a real NeuPrint client.
    monkeypatch.setattr(
        coana.FindNeuronConnection, "_ensure_neuprint_client",
        lambda self: None)
    fc, fetch_calls, logs = _make_pipeline_fc(
        monkeypatch, tmp_path, _SYNTH_EDGES, max_interlayer=3,
        source_ids=_SYNTH_SOURCES, target_ids=_SYNTH_TARGETS, **kwargs)
    fc.skip_bodyId = False
    if budget is not None:
        fc.discovery_batch_budget = budget
    fc.target_batch_size = fixed
    if max_paths is not None:
        fc.max_paths_bodyid = max_paths
    if retention is not None:
        fc.discovery_store_retention = retention
    return fc, fetch_calls, logs


def _run_monolithic_vs_batched(monkeypatch, tmp_path, **kwargs):
    """Phase 0 helper: same query through the legacy and batched paths."""
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc_mono, calls_mono, _ = _make_fc(
        monkeypatch, tmp_path / 'mono', budget=0, **kwargs)
    fc_mono.FindShortestPath()

    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc_batch, calls_batch, _ = _make_fc(
        monkeypatch, tmp_path / 'batch', **kwargs)
    fc_batch.FindShortestPath()
    return fc_mono, fc_batch, calls_mono, calls_batch


def _paths_csv(fc):
    return os.path.join(
        fc.allpath_folder, 'src_to_tgt_allpaths_bodyId_paths.csv')


def _attrs(fc):
    with open(os.path.join(fc.allpath_folder, 'all_attributes.json'),
              encoding='utf-8') as handle:
        return json.load(handle)


def _notes(fc):
    path = os.path.join(fc.allpath_folder, 'user_warning_notes.txt')
    if not os.path.exists(path):
        return ''
    return open(path, encoding='utf-8').read()


# ---------------------------------------------------------------------------
# Phase 0 — enumerator refactor: payload mode + shared drain
# ---------------------------------------------------------------------------

_UNIT_EDGES = [
    ('S1', 'A', 5), ('A', 'T1', 8),          # S1->T1 2-hop, bn 5
    ('S2', 'B', 6), ('B', 'C', 9), ('C', 'T2', 7),   # 3-hop, bn 6
    ('S3', 'T2', 6),                                  # 1-hop, bn 6 (tie)
    ('S1', 'T4', 5),                                  # tie with S1->T1
]


def _unit_graph():
    G = FastGraph()
    for u, v, w in _UNIT_EDGES:
        G.add_edge(u, v, w)
    return G


def test_payload_merge_matches_monolithic():
    G = _unit_graph()
    targets = ['T1', 'T2', 'T4']
    sources = ['S1', 'S2', 'S3']
    mono_stats = {}
    mono = list(G.find_paths_shortest_strongest_first(
        targets, sources, 4, budget=1_000_000, stats=mono_stats))

    payload = list(G.find_paths_shortest_strongest_first(
        targets, sources, 4, budget=None, payload=True))
    # each payload list is descending by bottleneck
    bottlenecks = [item[0] for item in payload]
    assert bottlenecks == sorted(bottlenecks, reverse=True)

    # single-batch composition
    stats_one = {}
    one = list(merge_shortest_batch_emissions(
        [payload], 1_000_000, stats_one))
    assert one == mono
    assert stats_one == mono_stats

    # split AFTER a tie boundary (T2/S3 and T1/S1... ties at bn 6 and 5)
    stats_split = {}
    split_at = next(i for i, b in enumerate(bottlenecks)
                    if b < bottlenecks[0])
    split = list(merge_shortest_batch_emissions(
        [payload[:split_at], payload[split_at:]], 1_000_000, stats_split))
    assert split == mono
    assert stats_split == mono_stats


def test_payload_merge_budget_bite_parity():
    G = _unit_graph()
    targets = ['T1', 'T2', 'T4']
    sources = ['S1', 'S2', 'S3']
    mono_stats = {}
    mono = list(G.find_paths_shortest_strongest_first(
        targets, sources, 4, budget=3, stats=mono_stats))

    payload = list(G.find_paths_shortest_strongest_first(
        targets, sources, 4, budget=None, payload=True))
    split = len(payload) // 2
    stats_split = {}
    out = list(merge_shortest_batch_emissions(
        [payload[:split], payload[split:]], 3, stats_split))
    assert out == mono
    assert stats_split == mono_stats
    # budget 3 + the bottleneck-5 tie group drains to 4 paths, unbitten
    # (nothing strictly weaker remains) — the tau-drain contract
    assert stats_split['emitted'] == 4
    assert stats_split['tau'] == 5
    assert stats_split['budget_bitten'] is False


def test_payload_mode_refuses_budget():
    G = _unit_graph()
    with pytest.raises(ValueError):
        list(G.find_paths_shortest_strongest_first(
            ['T1'], ['S1'], 4, budget=5, payload=True))


def test_drain_is_the_enumerator_semantics():
    """The shared drain produces the exact tau tie-drain set: biting keeps
    every tie at the achieved tau and drops strictly weaker paths."""
    G = _unit_graph()
    targets = ['T1', 'T2', 'T4']
    sources = ['S1', 'S2', 'S3']
    full = list(G.find_paths_shortest_strongest_first(
        targets, sources, 4, budget=None, payload=True))
    by_bottleneck = {}
    for bottleneck, path, target in full:
        by_bottleneck.setdefault(bottleneck, []).append(path)

    stats = {}
    bitten = list(merge_shortest_batch_emissions(
        [full[:2], full[2:]], 1, stats))
    tau = stats['tau']
    expected = [p for b in sorted(by_bottleneck, reverse=True)
                if b >= tau for p in by_bottleneck[b]]
    assert bitten == expected
    assert stats['emitted'] == len(bitten) == len(expected)
    assert tau == min(by_bottleneck) or tau > min(by_bottleneck)
    assert stats['budget_bitten'] is (
        stats['strongest_dropped'] is not None)
    if stats['strongest_dropped'] is not None:
        assert stats['strongest_dropped'] < tau


# ---------------------------------------------------------------------------
# Batch composition
# ---------------------------------------------------------------------------

def test_compose_batches_contiguous_and_sorted():
    states = {'b': 10, 'a': 1, 'c': 5}
    batches = compose_batches(['a', 'b', 'c'], states, budget=11)
    # contiguous runs of sorted(targets, key=str): a(1)+b(10) fits 11, c
    # would exceed it
    assert batches == [['a', 'b'], ['c']]
    assert compose_batches(['a', 'b', 'c'], states, budget=16) == \
        [['a', 'b', 'c']]

    # a single oversized target is never split
    batches = compose_batches(['a', 'b'], {'a': 100, 'b': 1}, budget=10)
    assert batches == [['a'], ['b']]

    # fixed size wins when set
    batches = compose_batches(['a', 'b', 'c', 'd'], {}, fixed_size=2)
    assert batches == [['a', 'b'], ['c', 'd']]


def test_resolve_batching_defaults():
    class _Fc:
        pass
    fc = _Fc()
    # object.__new__-style instances (tests, ad-hoc callers) get the
    # pinned default: batching ON with the 2M budget.
    enabled, budget, fixed = resolve_batching(fc)
    assert enabled and budget == 2_000_000 and fixed == 0
    fc.discovery_batch_budget = 0
    assert resolve_batching(fc) == (False, 0, 0)
    fc.target_batch_size = 3
    assert resolve_batching(fc) == (True, 0, 3)


# ---------------------------------------------------------------------------
# Pipeline-level equivalence (Phase B hard gates)
# ---------------------------------------------------------------------------

def test_pipeline_batched_equals_monolithic(monkeypatch, tmp_path, synth_types):
    fc_mono, fc_batch, calls_mono, calls_batch = _run_monolithic_vs_batched(
        monkeypatch, tmp_path)

    mono_df = pl.read_csv(_paths_csv(fc_mono))
    batch_df = pl.read_csv(_paths_csv(fc_batch))
    # ROW-ORDER equality (the merged-stream gate), not just set equality
    assert mono_df.rows() == batch_df.rows()

    # same fetch sequence (union-layer discovery fetches identically)
    assert calls_mono == calls_batch

    # same exported StrongestFirst stats + trim policy
    keys = ('trim_policy', 'strongest_first_cutoff',
            'strongest_first_budget_bitten', 'tau_canonical',
            'strongest_dropped_bottleneck', 'max_paths_bodyid')
    attrs_mono = _attrs(fc_mono)
    attrs_batch = _attrs(fc_batch)
    for key in keys:
        assert attrs_mono.get(key) == attrs_batch.get(key), key

    # same warning notes (modulo the batch-count citation on the scope
    # note; the static file footer is excluded from the comparison)
    def _note_lines(text):
        return [line for line in text.splitlines()
                if line.startswith('- [')]

    batch_lines = [
        line.split(' Discovery ran in')[0]
        if 'Discovery ran in' in line else line
        for line in _note_lines(_notes(fc_batch))
    ]
    assert _note_lines(_notes(fc_mono)) == batch_lines
    assert any('target batch(es)' in line
               for line in _note_lines(_notes(fc_batch)))


def test_pipeline_layer_recurred_edge_exports_physical_weight(
        monkeypatch, tmp_path, synth_types):
    """Engine issue 2026-10-05 (I1): the cross-layer duplicate case. A
    sits at reverse depth 1 for T1 and depth 2 for T2, so S1->A is
    fetched in TWO conn layers — but those are per-target-BFS REFETCHES
    of ONE physical edge (weight 7 in both). The exported bodyId weights
    must show the PHYSICAL weight once ([7, 12], not the fetch-multiplied
    [14, 12] the old cross-layer add_edge sum produced), the monolithic
    and batched twins must agree, and the store keeps both layer rows
    verbatim (the audit surface for what was fetched)."""
    fc_mono, fc_batch, _, _ = _run_monolithic_vs_batched(
        monkeypatch, tmp_path)

    def s1_rows(fc):
        df = pl.read_csv(_paths_csv(fc))
        return {row[0]: row for row in df.iter_rows()}

    mono_rows = s1_rows(fc_mono)
    batch_rows = s1_rows(fc_batch)
    s1_a_t1 = 'S1->A->T1'
    assert s1_a_t1 in mono_rows and s1_a_t1 in batch_rows
    assert mono_rows[s1_a_t1] == batch_rows[s1_a_t1]
    # weight column carries the PHYSICAL S1->A weight exactly once
    weights = batch_rows[s1_a_t1][1]
    assert weights == '[7, 12]', weights
    assert batch_rows[s1_a_t1][4] == 7   # min_weight = physical too
    # and the store itself kept both layer rows
    store = os.path.join(fc_batch.allpath_folder,
                         'shortest_discovery_store', 'connections')
    s1_a_layers = []
    for name in sorted(os.listdir(store)):
        frame = pl.read_parquet(os.path.join(store, name))
        kept = frame.filter(
            (pl.col('bodyId_pre') == 'S1') & (pl.col('bodyId_post') == 'A'))
        if len(kept):
            s1_a_layers.append(name)
    assert len(s1_a_layers) == 2, s1_a_layers


def test_pipeline_fixed_size_batches_equal(
        monkeypatch, tmp_path, synth_types):
    """target_batch_size=1: every target is its own batch — all ties fall
    across batch boundaries, the harshest order-equivalence case."""
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc_mono, calls_mono, _ = _make_fc(monkeypatch, tmp_path / 'mono',
                                      budget=0)
    fc_mono.FindShortestPath()
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc_batch, calls_batch, _ = _make_fc(monkeypatch, tmp_path / 'batch',
                                        fixed=1)
    fc_batch.FindShortestPath()
    assert pl.read_csv(_paths_csv(fc_mono)).rows() == \
        pl.read_csv(_paths_csv(fc_batch)).rows()
    diag = _attrs(fc_batch)['shortest_discovery_diagnostics']
    assert diag['batching']['batch_count'] == 4  # T1..T4, one batch each
    assert [b['target_count'] for b in diag['batching']['batches']] == \
        [1, 1, 1, 1]


def test_pipeline_budget_bite_parity(monkeypatch, tmp_path, synth_types):
    fc_mono, fc_batch, _, _ = _run_monolithic_vs_batched(
        monkeypatch, tmp_path, max_paths=3)
    assert pl.read_csv(_paths_csv(fc_mono)).rows() == \
        pl.read_csv(_paths_csv(fc_batch)).rows()
    attrs_mono, attrs_batch = _attrs(fc_mono), _attrs(fc_batch)
    assert attrs_mono['tau_canonical'] == attrs_batch['tau_canonical']
    assert attrs_mono['strongest_first_budget_bitten'] == \
        attrs_batch['strongest_first_budget_bitten']
    assert attrs_batch['strongest_first_budget_bitten'] in (True, False)


def test_pipeline_diagnostics_and_store_meta(
        monkeypatch, tmp_path, synth_types):
    fc_mono, fc_batch, _, _ = _run_monolithic_vs_batched(
        monkeypatch, tmp_path)
    diag_mono = _attrs(fc_mono)['shortest_discovery_diagnostics']
    diag_batch = _attrs(fc_batch)['shortest_discovery_diagnostics']

    # same core diagnostics; the batched dict adds only 'batching'
    mono_core = {k: v for k, v in diag_mono.items() if k != 'batching'}
    batch_core = {k: v for k, v in diag_batch.items() if k != 'batching'}
    assert batch_core == mono_core
    # the depth-cap target keeps discovery incomplete in BOTH modes
    assert diag_batch['discovery_complete'] is False
    assert diag_mono['discovery_complete'] is False
    assert set(diag_batch['batching']) >= {
        'discovery_batch_budget', 'target_batch_size', 'store_folder',
        'batch_count', 'batches'}

    store = os.path.join(fc_batch.allpath_folder,
                         'shortest_discovery_store')
    meta = load_store_meta(store)
    assert meta['targets_found'] == ['T1', 'T2', 'T3', 'T4']
    assert meta['target_hop_limits']['T1'] == 2   # S1's own distance
    assert meta['target_hop_limits']['T2'] == 3   # S1 detour is farthest
    assert meta['target_hop_limits']['T3'] == 2   # S3 -> D -> T3
    assert meta['target_hop_limits']['T4'] == 2
    assert meta['discovery_complete'] is False
    assert meta['edge_filter_config']['min_synapse_num'] == 1
    pairs = pl.read_parquet(os.path.join(store, 'pairs.parquet'))
    got = {(row[0], row[1], row[2]) for row in pairs.iter_rows()}
    assert ('S1', 'T1', 2) in got
    assert ('S1', 'T2', 3) in got      # via A -> X
    assert ('S3', 'T3', 2) in got
    assert ('S3', 'T4', 2) in got


def test_pipeline_type_paths_equal(monkeypatch, tmp_path, synth_types):
    fc_mono, fc_batch, _, _ = _run_monolithic_vs_batched(
        monkeypatch, tmp_path)
    type_csv = 'src_to_tgt_allpaths_type.csv'
    mono = pl.read_csv(os.path.join(fc_mono.allpath_folder, type_csv))
    batch = pl.read_csv(os.path.join(fc_batch.allpath_folder, type_csv))
    assert mono.rows() == batch.rows()
    # the min-hop contract survives batching: no 2-hop S1->T2 row exists
    assert not any(
        row[0].startswith('TS1->') and row[0].endswith('->TT2')
        and row[7] == 2 for row in batch.iter_rows())


def test_saveas_rerun_wipes_stale_store(monkeypatch, tmp_path, synth_types):
    """Store scoping (plan §4): rerunning into the SAME saveas folder
    must never mix labels from the earlier, deeper run. A stale
    node_distances chunk from run 1 used to survive run 2 and silently
    overwrite the fresh distances on read (later layer wins per key),
    dropping the rerun's paths; the store is wiped at discovery start."""
    def _run(depth):
        coana._FINDALLPATH_GRAPH_CACHE.clear()
        fc, _, _ = _make_fc(monkeypatch, tmp_path / f'd{depth}', budget=None)
        fc.max_interlayer = depth
        fc.saveas = 'rerun'
        fc.save_folder = str(tmp_path / 'shared')
        fc.FindShortestPath()
        return fc

    deep = _run(3)
    store = os.path.join(deep.allpath_folder, 'shortest_discovery_store')
    deep_files = sorted(
        os.listdir(os.path.join(store, 'node_distances')))

    shallow_rerun = _run(1)
    # same folder, and the stale deep-layer chunks are gone
    assert shallow_rerun.allpath_folder == deep.allpath_folder
    assert sorted(os.listdir(
        os.path.join(store, 'node_distances'))) != deep_files
    meta = load_store_meta(store)
    label_files = sorted(os.listdir(
        os.path.join(store, 'node_distances')))
    max_layer = max(
        int(name.split('_L')[1].split('_')[0]) for name in label_files)
    assert max_layer <= meta['layer_count'] - 1

    # the rerun's paths equal a fresh-folder shallow run's paths
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fresh, _, _ = _make_fc(monkeypatch, tmp_path / 'fresh', budget=None)
    fresh.max_interlayer = 1
    fresh.FindShortestPath()
    assert pl.read_csv(_paths_csv(shallow_rerun)).rows() == \
        pl.read_csv(_paths_csv(fresh)).rows()


# ---------------------------------------------------------------------------
# Store retention (plan-shortest-store-retention)
# ---------------------------------------------------------------------------

def _run_retention(monkeypatch, tmp_path, retention):
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc, _, _ = _make_fc(monkeypatch, tmp_path / retention, retention=retention)
    fc.FindShortestPath()
    return fc


def test_store_retention_modes(monkeypatch, tmp_path, synth_types):
    fcs = {mode: _run_retention(monkeypatch, tmp_path, mode)
           for mode in ('keep', 'compact', 'prune')}

    # output invariance: exported paths identical across all three modes
    base_rows = pl.read_csv(_paths_csv(fcs['keep'])).rows()
    for mode in ('compact', 'prune'):
        assert pl.read_csv(_paths_csv(fcs[mode])).rows() == base_rows, mode

    # keep: full store (per-layer connection files + dag chunks)
    store_keep = os.path.join(fcs['keep'].allpath_folder,
                              'shortest_discovery_store')
    assert any(name.startswith('connections_L')
               for name in os.listdir(store_keep + '/connections'))
    assert os.listdir(store_keep + '/dag_edges')

    # compact: ONE 4-column connections file, dag_edges gone, labels and
    # pairs kept, meta marked, and the S1->A cross-layer multiplicity row
    # count preserved
    store_c = os.path.join(fcs['compact'].allpath_folder,
                           'shortest_discovery_store')
    assert sorted(os.listdir(store_c + '/connections')) == \
        ['connections_all.parquet']
    merged = pl.read_parquet(store_c + '/connections/connections_all.parquet')
    assert merged.columns == ['bodyId_pre', 'bodyId_post', 'weight',
                              'conn_layer']
    s1a = merged.filter((pl.col('bodyId_pre') == 'S1')
                        & (pl.col('bodyId_post') == 'A'))
    assert len(s1a) == 2, 'cross-layer multiplicity must survive compaction'
    assert not os.path.isdir(store_c + '/dag_edges')
    assert os.path.isdir(store_c + '/node_distances')
    assert os.path.exists(store_c + '/pairs.parquet')
    meta_c = load_store_meta(store_c)
    assert meta_c['retention']['mode'] == 'compact'
    assert meta_c['retention']['store_bytes_after'] \
        <= meta_c['retention']['store_bytes_before']
    assert meta_c['retention']['recipe']

    # prune: only the meta census survives
    store_p = os.path.join(fcs['prune'].allpath_folder,
                           'shortest_discovery_store')
    assert sorted(os.listdir(store_p)) == ['meta.json']
    meta_p = load_store_meta(store_p)
    assert meta_p['retention']['mode'] == 'prune'
    assert meta_p['retention']['store_bytes_before'] > 0

    # provenance: diagnostics + warning notes for the acting modes
    for mode in ('compact', 'prune'):
        diag = _attrs(fcs[mode])['shortest_discovery_diagnostics']
        assert diag['batching']['store_retention']['mode'] == mode
        assert '[discovery store]' in _notes(fcs[mode])
    assert '[discovery store]' not in _notes(fcs['keep'])


def test_store_retention_invalid_falls_back_to_keep(
        monkeypatch, tmp_path, synth_types):
    fc = _run_retention(monkeypatch, tmp_path, 'nonsense')
    store = os.path.join(fc.allpath_folder, 'shortest_discovery_store')
    assert os.path.isdir(store + '/dag_edges')
    assert '[discovery store]' in _notes(fc)
    assert 'unknown discovery_store_retention' in _notes(fc)


def test_pipeline_ratio_shortest_twins_equal(monkeypatch, tmp_path,
                                              synth_types):
    """Audit 2026-10-06: on ratio-basis shortest runs the BATCHED lane
    drained on synapse weights while the monolithic twin used
    weight_ratio — tau on the wrong scale, different path sets. The
    batched graphs now convert the physical synapse weights to the same
    F9 ratios (totals from _attach_weight_ratio_columns), so the twins
    must agree row-for-row on the ratio basis too."""
    coana._FINDALLPATH_GRAPH_CACHE.clear()

    def _run(where):
        fc, _, _ = _make_fc(monkeypatch, tmp_path / where)
        fc.weight_basis = 'connection_ratio'
        fc.min_ratio = 0.0005
        fc.FindShortestPath()
        return pl.read_csv(_paths_csv(fc)).rows()

    mono = _run('mono')
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    batched = _run('batch')
    assert mono == batched
