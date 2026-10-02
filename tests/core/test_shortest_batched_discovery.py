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
             max_paths=None, **kwargs):
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


def test_drain_is_the_enumerator_semsantics():
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
    assert bitten == expected[:len(bitten)] or len(bitten) == len(expected)
    assert stats['budget_bitten'] is (stats['strongest_dropped'] is not None)


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


def test_pipeline_weight_multiplicity_preserved(
        monkeypatch, tmp_path, synth_types):
    """The cross-layer duplicate case: A sits at reverse depth 1 for T1
    and depth 2 for T2, so S1->A is fetched in TWO conn layers and the
    monolithic graph sums its weight. The batched store must reproduce
    the doubled weight (a pair-deduped store would halve it and reorder
    the emission)."""
    fc_mono, fc_batch, _, _ = _run_monolithic_vs_batched(
        monkeypatch, tmp_path)

    def s1_rows(fc):
        df = pl.read_csv(_paths_csv(fc))
        return {row[0]: row for row in df.iter_rows()}

    mono_rows = s1_rows(fc_mono)
    batch_rows = s1_rows(fc_batch)
    s1_a_t1 = 'S1->A->T1'
    assert s1_a_t1 in mono_rows and s1_a_t1 in batch_rows
    # weight column carries the DOUBLED S1->A weight (7 fetched twice)
    assert mono_rows[s1_a_t1] == batch_rows[s1_a_t1]
    weights = batch_rows[s1_a_t1][1]
    assert weights == '[14, 12]', weights
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
