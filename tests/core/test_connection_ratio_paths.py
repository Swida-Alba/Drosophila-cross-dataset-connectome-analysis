"""Connection-ratio pathfinding lane — Phase 1 tests
(plan-connection-ratio-pathfinding §7; decisions §18 D-1..D-8).

Two synthetic universes:
- INTEGER universe: every post total = 100, weights are integer
  percents → ratio values are integers → the lane MUST equal a
  synapse-basis production enumeration set/order/tau for set (the §18
  E4 parity keystone).
- FRACTIONAL universe: non-uniform denominators, one untyped node (D),
  an external strong input (X→A, the F9 denominator invariant), an
  untyped enrolled source (S3), and a hemisphere-suffixed type map.
"""
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC = PROJECT_ROOT / 'src'
for p in (PROJECT_ROOT, SRC, PROJECT_ROOT / 'vispath-subproject' / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from connection_ratio_paths import (  # noqa: E402
    ConnectionRatioPathError,
    WEIGHT_BASIS,
    _apply_guards,
    _as_frame,
    _implied_cutoff,
    compute_connection_ratio_paths,
    compute_incoming_totals,
    derive_label_paths,
    discover_cone,
    fit_edge_budget_ratio,
    find_ratio_paths,
    load_connections,
    prepare_ratio_frame,
    ratio_threshold_provenance,
)
from type_level_refill import (  # noqa: E402
    _enumerate_paths,
    filter_edges_by_hemisphere,
    hop_prefilter,
)

# ---------------------------------------------------------------------------
# Universes
# ---------------------------------------------------------------------------
# INTEGER universe (§18 E4): posts A,B,C,D,E,T1,T2 all total exactly 100.
INT_EDGES = [
    ('S1', 'A', 60), ('S2', 'A', 30), ('S1', 'B', 20), ('S2', 'B', 35),
    ('A', 'C', 15), ('B', 'C', 30), ('S1', 'C', 40), ('A', 'D', 70),
    ('B', 'D', 10), ('C', 'T1', 40), ('D', 'T1', 25), ('E', 'T1', 30),
    ('C', 'T2', 15), ('D', 'T2', 45), ('E', 'T2', 35),
    ('B', 'E', 25), ('A', 'E', 15), ('D', 'E', 45),
] + [
    ('bg1', 'A', 10), ('bg2', 'B', 45), ('bg3', 'C', 15),
    ('bg4', 'D', 20), ('bg5', 'E', 15), ('bg6', 'T1', 5),
    ('bg7', 'T2', 5),
]
INT_SOURCES = ['S1', 'S2']
INT_TARGETS = ['T1', 'T2']
INT_BOUND = 3

# FRACTIONAL universe: X = external strong input (F9 invariant),
# D = untyped intermediate node, S3 = UNTYPED enrolled source (absent
# from the typed map; keeps enrollment, loses its edge under the guard).
FRAC_EDGES = [
    ('X', 'A', 50),                      # cone-external (X not enrolled)
    ('S1', 'A', 10), ('S2', 'A', 5),     # A total 65
    ('S1', 'B', 3), ('S2', 'B', 27),     # B total 30
    ('A', 'C', 6), ('B', 'C', 2),        # C total 8
    ('A', 'D', 4),                       # D total 4 (D untyped)
    ('C', 'T1', 1), ('B', 'T1', 2), ('S3', 'T1', 1),   # T1 total 4
    ('C', 'T2', 3), ('B', 'T2', 3), ('D', 'T2', 7),    # T2 total 13
]
FRAC_TYPE_MAP = {
    'S1': 'SRC', 'S2': 'SRC', 'A': 'MID', 'B': 'MID',
    'C': 'INNER', 'T1': 'SINK', 'T2': 'SINK', 'X': 'EXT',
}
FRAC_TYPED_MAP = dict(FRAC_TYPE_MAP)   # S3 and D stay untyped
FRAC_SOURCES = ['S1', 'S2', 'S3']
FRAC_TARGETS = ['T1', 'T2']
FRAC_BOUND = 3

FRAC_HEMI_MAP = {
    'S1': 'SRC_L', 'S2': 'SRC_R', 'A': 'MID_L', 'B': 'MID_R',
    'C': 'INNER_L', 'T1': 'SINK_L', 'T2': 'SINK_R', 'X': 'EXT_U',
    'S3': 'SRC2_U',
}


def brute_paths(edges, sources, targets, bound):
    """Independent ground truth: all simple source->target paths with
    <= bound edges, as (bottleneck, nodes)."""
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
    src, tgt = set(sources), set(targets)
    found = []

    def dfs(node, path, wpath):
        if len(path) >= 2 and node in tgt:
            found.append((min(wpath), tuple(path)))
        if len(path) - 1 >= bound:
            return
        for v, w in adj.get(node, ()):
            if v not in path:
                dfs(v, path + [v], wpath + [w])

    for s in sorted(src):
        dfs(s, [s], [])
    return found


def production_reference(edges, sources, targets, bound, t_r, budget=None):
    """Synapse-basis production semantics on the SAME thresholded
    universe: filter at t_r (ratios), close (hop_prefilter), enumerate
    with the production strongest-first engine."""
    totals = compute_incoming_totals(edges)
    kept = [(u, v, w / totals[v]) for u, v, w in edges
            if w / totals[v] >= t_r]
    closed, _ = hop_prefilter(kept, sources, targets, bound)
    return _enumerate_paths(closed, sources, targets, bound, budget=budget)


def run_frac(**kwargs):
    params = dict(min_ratio=0.1, max_interlayer=FRAC_BOUND,
                  type_map=dict(FRAC_TYPED_MAP), drop_untyped=True,
                  exclude_intra_type=False)
    params.update(kwargs)
    return find_ratio_paths(FRAC_EDGES, FRAC_SOURCES, FRAC_TARGETS, **params)


# ---------------------------------------------------------------------------
# 1. Ratio math + the F9 denominator invariant
# ---------------------------------------------------------------------------
def test_totals_hand_computed():
    totals = compute_incoming_totals(FRAC_EDGES)
    assert totals == {'A': 65.0, 'B': 30.0, 'C': 8.0, 'D': 4.0,
                      'T1': 4.0, 'T2': 13.0}


def test_f9_denominator_invariant():
    """A cone-external strong input (X->A, 50 syn) must NOT shrink any
    ratio: A's denominator is the FULL-table all-post mass."""
    rec = run_frac()
    edge = rec['edges']['S1->A']
    assert edge['ratio'] == pytest.approx(10 / 65, rel=1e-12)


def test_duplicate_pairs_summed():
    edges = FRAC_EDGES + [('S1', 'A', 5)]      # duplicate (S1, A)
    totals = compute_incoming_totals(edges)
    assert totals['A'] == 70.0                 # 65 + 5


# ---------------------------------------------------------------------------
# 2. Cone discovery == exact closure; hop_prefilter-stable
# ---------------------------------------------------------------------------
def _frame(edges):
    frame, totals = prepare_ratio_frame(_as_frame(edges))
    return frame, totals


def test_cone_equals_bruteforce_closure():
    frame, _t = _frame(FRAC_EDGES)
    for t_r in (0.0, 0.1, 0.25):
        cone, stats = discover_cone(
            frame, FRAC_SOURCES, FRAC_TARGETS, FRAC_BOUND, t_r,
            type_map=dict(FRAC_TYPED_MAP))
        triples = list(zip(cone['bodyId_pre'].to_list(),
                           cone['bodyId_post'].to_list(),
                           cone['ratio'].to_list()))
        closed, _ = hop_prefilter(triples, FRAC_SOURCES, FRAC_TARGETS,
                                  FRAC_BOUND)
        assert sorted(triples) == sorted(closed)   # already the fixpoint


def test_cone_loses_no_path_vs_bruteforce():
    """Every brute-force path on the guarded, thresholded universe is
    emitted (lossless closure + full enumeration)."""
    guarded = [(u, v, w) for u, v, w in FRAC_EDGES
               if u in FRAC_TYPED_MAP and v in FRAC_TYPED_MAP]
    ref = brute_paths(guarded, FRAC_SOURCES, FRAC_TARGETS,
                      FRAC_BOUND + 1)
    rec = run_frac(min_ratio=1e-9)          # keep every guarded edge
    got = {tuple(p['nodes']) for p in rec['paths']}
    assert got == {path for _bn, path in ref}


# ---------------------------------------------------------------------------
# 3. Integer-ratio parity (§18 D-3 keystone)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('t_r,budget,expected_n,expected_tau,bitten', [
    (0.2, None, 15, 0.2, False),
    (0.2, 3, 3, 0.35, True),
    (0.2, 7, 8, 0.3, True),
    (0.3, 5, 8, 0.3, False),
])
def test_integer_ratio_parity(t_r, budget, expected_n, expected_tau, bitten):
    rec = find_ratio_paths(
        INT_EDGES, INT_SOURCES, INT_TARGETS, min_ratio=t_r,
        max_interlayer=INT_BOUND, path_budget=budget or 1_000_000,
        type_map=None, drop_untyped=False)
    ref, _info, ref_stats = production_reference(
        INT_EDGES, INT_SOURCES, INT_TARGETS, INT_BOUND + 1, t_r,
        budget=budget)
    got = [(p['ratio_bottleneck'], tuple(p['nodes'])) for p in rec['paths']]
    assert got == ref                                    # set + ORDER
    assert len(rec['paths']) == expected_n
    prov = rec['provenance']
    assert prov['strongest_first_tau'] == pytest.approx(expected_tau)
    assert prov['strongest_retained_bottleneck'] == pytest.approx(0.45)  # P1
    assert prov['strongest_first_budget_bitten'] == bitten
    bns = [p['ratio_bottleneck'] for p in rec['paths']]
    assert all(bns[i] >= bns[i + 1] for i in range(len(bns) - 1))
    assert ref_stats['emitted'] == len(rec['paths'])


def test_unbounded_equals_bruteforce_dfs():
    rec = find_ratio_paths(
        INT_EDGES, INT_SOURCES, INT_TARGETS, min_ratio=0.2,
        max_interlayer=INT_BOUND, type_map=None, drop_untyped=False)
    totals = compute_incoming_totals(INT_EDGES)
    bf = brute_paths([(u, v, w / totals[v]) for u, v, w in INT_EDGES
                      if w / totals[v] >= 0.2],
                     INT_SOURCES, INT_TARGETS, INT_BOUND + 1)
    assert {(p['ratio_bottleneck'], tuple(p['nodes']))
            for p in rec['paths']} == set(bf)


# ---------------------------------------------------------------------------
# 4. Float-tier Edge Budget
# ---------------------------------------------------------------------------
def test_floor_equals_complete_at_floor_tier():
    budgeted = find_ratio_paths(
        INT_EDGES, INT_SOURCES, INT_TARGETS, min_ratio=0.2,
        max_interlayer=INT_BOUND, edge_budget=9, type_map=None,
        drop_untyped=False)
    floor = budgeted['provenance']['edge_weight_floor']
    assert floor is not None and floor > 0.2   # binding
    at_floor = find_ratio_paths(
        INT_EDGES, INT_SOURCES, INT_TARGETS, min_ratio=floor,
        max_interlayer=INT_BOUND, type_map=None, drop_untyped=False)
    assert ([(p['ratio_bottleneck'], tuple(p['nodes']))
             for p in budgeted['paths']]
            == [(p['ratio_bottleneck'], tuple(p['nodes']))
                for p in at_floor['paths']])
    assert not budgeted['provenance']['paths_complete']
    assert 'edge_budget' in budgeted['provenance'][
        'applied_threshold_source']


def _layered_universe(seed):
    """Dense 3-hop universe (S -> a* -> b* -> c* -> T) with background
    inputs so denominators vary — guarantees a non-trivial cone."""
    rng = random.Random(seed)
    a = [f'a{i}' for i in range(3)]
    b = [f'b{i}' for i in range(3)]
    c = [f'c{i}' for i in range(3)]
    edges = set()
    for x in a:
        edges.add(('n0', x, rng.randint(5, 30)))
        for y in rng.sample(b, 2):
            edges.add((x, y, rng.randint(5, 30)))
    for x in b:
        for y in rng.sample(c, 2):
            edges.add((x, y, rng.randint(5, 30)))
    for y in c:
        edges.add((y, 'n10', rng.randint(5, 30)))
    nxt = 0
    for v in a + b + c + ['n10']:
        for _k in range(rng.randint(1, 2)):
            nxt += 1
            edges.add((f'bg{nxt}', v, rng.randint(1, 15)))
    return sorted(edges)


def test_budget_fit_exact_or_stronger():
    import random
    checked = 0
    for seed in range(8):
        edges = _layered_universe(seed)
        frame, _t = _frame(edges)
        S, T, bound = ['n0'], ['n10'], 4
        cone, _stats = discover_cone(frame, S, T, bound, 0.02,
                                     type_map=None)
        assert cone.height >= 10
        budget = random.Random(seed + 100).randint(
            max(3, cone.height // 3), cone.height - 1)
        kept, stats = fit_edge_budget_ratio(cone, budget, S, T, bound)
        assert not stats.get('floor_skipped') and stats['floor'] is not None
        # exact: weakest tier whose closure fits (weakest-first scan)
        tiers = sorted({r for r in cone['ratio'].to_list()})
        exact = None
        for tier in tiers:
            masked = cone.filter(pl.col('ratio') >= tier)
            triples = list(zip(masked['bodyId_pre'].to_list(),
                               masked['bodyId_post'].to_list(),
                               masked['ratio'].to_list()))
            closed, _ = hop_prefilter(triples, S, T, bound)
            if len(closed) <= budget:
                exact = tier
                break
        assert exact is not None
        checked += 1
        assert stats['floor'] == exact or stats['floor'] > exact
    assert checked >= 5


def test_budget_fit_landing_and_budget_line():
    """Review-round regression: production w1 = the budget-th LARGEST
    weight (coana kth = total - budget ascending); the float landing =
    the weakest distinct tier STRICTLY above w1 and is the FIRST probe."""
    frame, _t = _frame(INT_EDGES)
    cone, _ = discover_cone(frame, INT_SOURCES, INT_TARGETS, INT_BOUND,
                            0.2, type_map=None)
    budget = 9
    desc = sorted(cone['ratio'].to_list(), reverse=True)
    exp_w1 = desc[budget - 1]                     # budget-th largest
    tiers = sorted(set(desc))
    exp_landing = min(t for t in tiers if t > exp_w1)
    _kept, stats = fit_edge_budget_ratio(
        cone, budget, INT_SOURCES, INT_TARGETS, INT_BOUND)
    assert stats['budget_line'] == pytest.approx(exp_w1)
    assert stats['landing'] == pytest.approx(exp_landing)
    assert stats['probe_trace'][0][0] == pytest.approx(exp_landing)
    # provenance mirrors production: edge_budget_landing == w1 (the
    # budget-line tier), NOT the probe-start landing
    rec = find_ratio_paths(
        INT_EDGES, INT_SOURCES, INT_TARGETS, min_ratio=0.2,
        max_interlayer=INT_BOUND, edge_budget=budget, type_map=None,
        drop_untyped=False)
    assert rec['provenance']['edge_budget_landing'] == pytest.approx(exp_w1)
    assert rec['provenance']['edge_weight_floor'] is not None


def test_derive_label_paths_rule():
    """§1.6: unique sequences, source/target anchored, every consecutive
    label pair present in the kept label-edge set; sorted output."""
    paths = [(0.5, ('S1', 'A', 'T1')), (0.4, ('S2', 'B', 'T1')),
             (0.3, ('S1', 'A', 'T1'))]              # duplicate sequence
    label = {'S1': 'SRC', 'S2': 'SRC2', 'A': 'MID', 'B': 'MID',
             'T1': 'SINK'}
    kept = {('SRC', 'MID'), ('MID', 'SINK'), ('SRC2', 'MID')}
    out = derive_label_paths(paths, lambda n: label[n], ['SRC', 'SRC2'],
                             ['SINK'], kept)
    # duplicate sequence deduped; sorted lexicographically
    assert out == [['SRC', 'MID', 'SINK'], ['SRC2', 'MID', 'SINK']]
    # anchored rejection: target label not queried
    out2 = derive_label_paths(paths, lambda n: label[n], ['SRC'], ['X'],
                              kept)
    assert out2 == []
    # edge-consistency rejection: a required label pair missing
    kept_missing = {('SRC', 'MID')}
    out3 = derive_label_paths(paths, lambda n: label[n], ['SRC', 'SRC2'],
                              ['SINK'], kept_missing)
    assert out3 == []


def test_budget_fit_skipped_when_cone_fits():
    frame, _t = _frame(FRAC_EDGES)
    cone, _ = discover_cone(frame, FRAC_SOURCES, FRAC_TARGETS, FRAC_BOUND,
                            0.1, type_map=dict(FRAC_TYPED_MAP))
    _kept, stats = fit_edge_budget_ratio(
        cone, cone.height + 100, FRAC_SOURCES, FRAC_TARGETS, FRAC_BOUND)
    assert stats['floor_skipped'] is True and stats['floor'] is None


# ---------------------------------------------------------------------------
# 5. Provenance contract
# ---------------------------------------------------------------------------
def test_provenance_keys_complete():
    prov = find_ratio_paths(
        INT_EDGES, INT_SOURCES, INT_TARGETS, min_ratio=0.2,
        max_interlayer=INT_BOUND, path_budget=3, type_map=None,
        drop_untyped=False)['provenance']
    for key in ('requested_threshold', 'applied_threshold',
                'applied_threshold_source', 'strongest_first_budget_bitten',
                'strongest_first_tau', 'tau_canonical',
                'strongest_dropped_bottleneck', 'edge_budget',
                'edge_budget_applied', 'edge_floor_binding',
                'edge_budget_landing', 'edge_weight_floor',
                'strongest_retained_bottleneck', 'paths_complete',
                'weight_basis', 'threshold_unit'):
        assert key in prov, key
    assert prov['weight_basis'] == WEIGHT_BASIS
    assert prov['threshold_unit'] == 'connection_ratio'


def test_canonical_is_minimal_reproducing_threshold():
    """tau_canonical is the weakest tier strictly above w2 — the MINIMAL
    threshold that reproduces the materialized output set."""
    budgeted = find_ratio_paths(
        INT_EDGES, INT_SOURCES, INT_TARGETS, min_ratio=0.2,
        max_interlayer=INT_BOUND, path_budget=3, type_map=None,
        drop_untyped=False)
    prov = budgeted['provenance']
    w2 = prov['strongest_dropped_bottleneck']
    assert w2 is not None and prov['tau_canonical'] > w2
    assert prov['tau_canonical_exact'] is True
    reproduced = find_ratio_paths(
        INT_EDGES, INT_SOURCES, INT_TARGETS,
        min_ratio=prov['tau_canonical'], max_interlayer=INT_BOUND,
        type_map=None, drop_untyped=False)
    assert ([(p['ratio_bottleneck'], tuple(p['nodes']))
             for p in budgeted['paths']]
            == [(p['ratio_bottleneck'], tuple(p['nodes']))
                for p in reproduced['paths']])


def test_degenerate_bottom_tier_gap_disclosed():
    prov = ratio_threshold_provenance(
        0.15, strongest_first_budget_bitten=True,
        strongest_dropped_bottleneck=0.15, distinct_tiers=[0.15, 0.3])
    assert prov['tau_canonical'] == pytest.approx(0.3)
    prov2 = ratio_threshold_provenance(
        0.1, strongest_first_budget_bitten=True,
        strongest_dropped_bottleneck=0.1, distinct_tiers=[0.1])
    assert prov2['tau_canonical'] == pytest.approx(0.1)
    assert prov2['tau_canonical_exact'] is False   # disclosed, not clamped


def test_complete_run_source_requested():
    prov = find_ratio_paths(
        INT_EDGES, INT_SOURCES, INT_TARGETS, min_ratio=0.2,
        max_interlayer=INT_BOUND, type_map=None,
        drop_untyped=False)['provenance']
    assert prov['applied_threshold_source'] == 'requested'
    assert prov['paths_complete'] is True
    assert prov['applied_threshold'] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 6. Admission guards (§2 stage 5)
# ---------------------------------------------------------------------------
def test_drop_untyped_excludes_untyped_intermediates():
    rec = run_frac()                                   # drop_untyped=True
    assert all('A->D' not in k and 'D->' not in k and 'S3->' not in k
               for k in rec['edges'])
    # A->D, D->T2 (untyped intermediate) + S3->T1 (untyped source)
    assert rec['discovery']['dropped_untyped'] == 3
    # but through typed middle nodes the paths survive
    assert any(tuple(p['nodes'])[1] == 'B' for p in rec['paths'])


def test_untyped_enrolled_source_keeps_enrollment():
    """S3 is enrolled but untyped: its edges are dropped (no refusal),
    and the run still completes for the typed sources."""
    rec = run_frac()
    assert rec['status'] == 'ok'
    assert all(p['nodes'][0] != 'S3' for p in rec['paths'])


def test_keep_untyped_restores_cone():
    rec = run_frac(drop_untyped=False)
    assert any('A->D' in k for k in rec['edges'])
    assert rec['discovery']['dropped_untyped'] == 0
    assert rec['discovery']['drop_untyped_inapplicable'] is False


def test_drop_untyped_inapplicable_disclosed():
    rec = run_frac(type_map=None)
    assert rec['discovery']['drop_untyped_inapplicable'] is True


def test_exclude_intra_type():
    edges = FRAC_EDGES + [('B2', 'A', 20)]     # B2->A is MID->MID (intra)
    m = dict(FRAC_TYPED_MAP)
    m['B2'] = 'MID'
    rec = find_ratio_paths(edges, FRAC_SOURCES, FRAC_TARGETS,
                           min_ratio=0.1, max_interlayer=FRAC_BOUND,
                           type_map=m, drop_untyped=False,
                           exclude_intra_type=True)
    assert rec['discovery']['dropped_intra_type'] == 1
    assert all('B2->A' not in k for k in rec['edges'])
    rec2 = find_ratio_paths(edges, FRAC_SOURCES, FRAC_TARGETS,
                            min_ratio=0.1, max_interlayer=FRAC_BOUND,
                            type_map=m, drop_untyped=False,
                            exclude_intra_type=False)
    assert rec2['discovery']['dropped_intra_type'] == 0


def test_hemi_guard_matches_reference_implementation():
    frame, _t = _frame(FRAC_EDGES)
    qual = frame.filter(pl.col('ratio') >= 0.0)
    got, stats = _apply_guards(
        qual, FRAC_SOURCES, FRAC_TARGETS, FRAC_HEMI_MAP,
        drop_untyped=False, exclude_intra=False, hemi_filter='left')
    kept_pairs = set(zip(got['bodyId_pre'].to_list(),
                         got['bodyId_post'].to_list()))
    triples = [(u, v, float(w)) for u, v, w in
               zip(qual['bodyId_pre'], qual['bodyId_post'],
                   qual['weight'])]
    ref_pairs = {(u, v) for u, v, _w in filter_edges_by_hemisphere(
        triples, FRAC_HEMI_MAP, 'left')}
    assert kept_pairs == ref_pairs
    assert stats['dropped_hemisphere'] == len(triples) - len(ref_pairs)


# ---------------------------------------------------------------------------
# 7. Type readouts: MASS recompute (§2 stage 8), never a fold
# ---------------------------------------------------------------------------
def test_type_ratio_is_mass_recompute_not_weighted_mean():
    """§2 stage 8 (user-ratified 10-03): per-pair INVOLVED-post
    denominators — the mediant inequality guarantees the aggregate
    clears t_r when every bodyId pair does."""
    # B3 is a THIRD MID member whose incoming (X->B3, 10) never receives
    # a SRC edge: it must NOT count toward the SRC->MID denominator
    # (the pre-fix full-type denominator would have been 95+10=105).
    edges = FRAC_EDGES + [('X', 'B3', 10)]
    m = dict(FRAC_TYPED_MAP)
    m['B3'] = 'MID'
    rec = find_ratio_paths(edges, FRAC_SOURCES, FRAC_TARGETS,
                           min_ratio=0.1, max_interlayer=FRAC_BOUND,
                           type_map=m, drop_untyped=True)
    rows = {(r['type_pre'], r['type_post']): r for r in rec['type_pairs']}
    # MID->INNER: emitted A->C (6) + B->C (2) = mass 8 over the single
    # involved INNER post C (total 8; D untyped drops out) -> 1.0. The
    # synapse-weighted mean of the pair ratios would be
    # (0.75*6 + 0.25*2)/8 = 0.625 — a DIFFERENT number: the pin.
    row = rows[('MID', 'INNER')]
    assert row['synapse_mass'] == 8.0
    assert row['involved_posts'] == 1
    assert row['total_incoming_type'] == 8.0
    assert row['connection_ratio'] == pytest.approx(1.0, rel=1e-12)
    assert row['connection_ratio'] != pytest.approx(0.625)
    # SRC->MID (round-9 full-type): mass 40 over ALL MID members
    # A(65) + B(30) + B3(10) = 105 — B3 counts now (single-connection
    # robustness), and the coverage column exposes the 2/3 support.
    src_row = rows[('SRC', 'MID')]
    assert src_row['involved_posts'] == 2
    assert src_row['type_coverage'] == '2/3'
    assert src_row['total_incoming_type'] == 105.0
    assert src_row['connection_ratio'] == pytest.approx(
        40 / 105, rel=1e-12)


def test_traversal_probability_folds():
    rec = run_frac(aggregate_method='product')
    rows = {(r['type_pre'], r['type_post']): r for r in rec['type_pairs']}
    totals = compute_incoming_totals(FRAC_EDGES)
    # MID->INNER channels: p=1-(1-p_A->C)(1-p_B->C)
    p_ac = min(1.0, (6 / totals['C']) / 0.3)
    p_bc = min(1.0, (2 / totals['C']) / 0.3)
    assert rows[('MID', 'INNER')][
        'traversal_probability'] == pytest.approx(
        1 - (1 - p_ac) * (1 - p_bc), rel=1e-12)
    rec_avg = run_frac(aggregate_method='average')
    row = {(r['type_pre'], r['type_post']): r
           for r in rec_avg['type_pairs']}[('MID', 'INNER')]
    assert row['traversal_probability'] == pytest.approx(
        (p_ac * 6 + p_bc * 2) / 8, rel=1e-12)
    rec_ratio = run_frac(aggregate_method='ratio')
    row = {(r['type_pre'], r['type_post']): r
           for r in rec_ratio['type_pairs']}[('MID', 'INNER')]
    assert row['traversal_probability'] == pytest.approx(
        min((8 / 12) / 0.3, 1.0), rel=1e-12)


def test_no_type_map_no_type_pairs():
    assert run_frac(type_map=None)['type_pairs'] == []


# ---------------------------------------------------------------------------
# 8. ratio -> synapse mapping (§13.2, D-7)
# ---------------------------------------------------------------------------
def test_implied_cutoff_clamps_to_one():
    assert _implied_cutoff(0.5, 1.0) == 1          # 0.5 -> clamp
    assert _implied_cutoff(0.2, 100.0) == 20       # exact
    assert _implied_cutoff(0.001, 1500.0) == 2     # 1.5 -> ceil 2


def test_ratio_to_synapse_disclosure():
    rec = run_frac(min_ratio=0.1)
    d = rec['ratio_to_synapse']
    assert d['kept_edge_synapse_range'] == [1.0, 27.0]
    # C total 8 -> ceil(0.8) = 1 (noop, 0.1*8 = 0.8 < 1); D excluded by
    # drop_untyped; T2 total 13 -> ceil(1.3) = 2.
    cutoffs = {r['bodyId_post']: r['implied_syn_cutoff']
               for r in d['posts_capped_detail']}
    assert cutoffs['C'] == 1 and cutoffs['T2'] == 2
    assert d['noop_cutoff_posts'] >= 1


# ---------------------------------------------------------------------------
# 9. Refusals + no_paths status
# ---------------------------------------------------------------------------
def test_refuses_bad_threshold():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ConnectionRatioPathError, match='min_ratio'):
            find_ratio_paths(FRAC_EDGES, FRAC_SOURCES, FRAC_TARGETS,
                             min_ratio=bad, max_interlayer=2)


def test_refuses_missing_enrollment_ids():
    with pytest.raises(ConnectionRatioPathError, match='absent'):
        find_ratio_paths(FRAC_EDGES, ['NOPE'], FRAC_TARGETS,
                         min_ratio=0.1, max_interlayer=2)


def test_refuses_bad_aggregate_method():
    with pytest.raises(ConnectionRatioPathError, match='aggregate_method'):
        run_frac(aggregate_method='median')


def test_loader_refuses_missing_columns(tmp_path):
    bad = tmp_path / 'bad.csv'
    bad.write_text('a,b\n1,2\n', encoding='utf-8')
    with pytest.raises(ConnectionRatioPathError, match='missing columns'):
        load_connections(bad)
    with pytest.raises(ConnectionRatioPathError, match='not found'):
        load_connections(tmp_path / 'absent.parquet')


def test_empty_cone_is_status_not_refusal():
    rec = run_frac(min_ratio=0.9)
    assert rec['status'] == 'no_paths'
    d = rec['ratio_to_synapse']
    assert d['cone_edges_at_zero_threshold'] > 0
    assert 'into_targets_ratio_max' in d and 'note' in d


# ---------------------------------------------------------------------------
# 10. Determinism + outputs + orchestrator
# ---------------------------------------------------------------------------
def test_outputs_byte_deterministic(tmp_path):
    neuron = pd.DataFrame({'bodyId': list(FRAC_TYPED_MAP),
                           'type': list(FRAC_TYPED_MAP.values())})
    a, b = tmp_path / 'a', tmp_path / 'b'
    for out in (a, b):
        compute_connection_ratio_paths(
            connections=FRAC_EDGES, sources=FRAC_SOURCES,
            targets=FRAC_TARGETS, min_ratio=0.1,
            max_interlayer=FRAC_BOUND, neuron_frame=neuron, out_dir=out)
    for f in sorted(p.relative_to(a) for p in a.rglob('*')
                    if p.is_file() and p.name != 'README.md'):
        assert (a / f).read_bytes() == (b / f).read_bytes(), f


def test_output_files_and_parameters(tmp_path):
    out = tmp_path / 'lane'
    rec = compute_connection_ratio_paths(
        connections=FRAC_EDGES, sources=FRAC_SOURCES, targets=FRAC_TARGETS,
        min_ratio=0.1, max_interlayer=FRAC_BOUND,
        neuron_frame=pd.DataFrame({'bodyId': list(FRAC_TYPED_MAP),
                                   'type': list(FRAC_TYPED_MAP.values())}),
        out_dir=out)
    assert rec['status'] == 'ok'
    assert (out / 'parameters.txt').read_text(encoding='utf-8').startswith(
        'weight basis: connection_ratio (bodyId)')
    assert 'min connection ratio: 0.1' in (out / 'parameters.txt').read_text(
        encoding='utf-8')
    for rel in ('ratio_paths_bodyId.csv',
                'data_details/ratio_edges_bodyId.csv',
                'data_details/ratio_type_pairs.csv',
                'data_details/ratio_synapse_map.csv',
                'data_details/ratio_provenance.json',
                'README.md'):
        assert (out / rel).exists(), rel
    paths = pd.read_csv(out / 'ratio_paths_bodyId.csv')
    assert {'rank', 'path', 'ratio_bottleneck', 'synapse_bottleneck',
            'ratios'} <= set(paths.columns)


def test_empty_outputs_are_header_only(tmp_path):
    out = tmp_path / 'lane_empty'
    compute_connection_ratio_paths(
        connections=FRAC_EDGES, sources=FRAC_SOURCES, targets=FRAC_TARGETS,
        min_ratio=0.9, max_interlayer=FRAC_BOUND, out_dir=out)
    csv = out / 'ratio_paths_bodyId.csv'
    assert csv.read_bytes().count(b'\n') == 1       # header only


def test_sibling_out_dir_naming(tmp_path):
    run = tmp_path / 'find-paths-complete_X_A_to_B_L3w1_20260101_000000'
    run.mkdir()
    rec = compute_connection_ratio_paths(
        connections=FRAC_EDGES, sources=FRAC_SOURCES, targets=FRAC_TARGETS,
        min_ratio=0.05, max_interlayer=2, run_dir=run)
    assert rec['out_dir'].startswith(str(run.parent / (run.name +
           '_ratio_L2r0_05_')))
    assert not any(run.iterdir())                    # run folder untouched


# ---------------------------------------------------------------------------
# 11. CLI (subprocess) + --in-run byte-untouched proof
# ---------------------------------------------------------------------------
def _write_conn_csv(tmp_path):
    path = tmp_path / 'conn.csv'
    pd.DataFrame(FRAC_EDGES,
                 columns=['bodyId_pre', 'bodyId_post', 'weight']
                 ).to_csv(path, index=False)
    return path


def test_cli_run(tmp_path):
    conn = _write_conn_csv(tmp_path)
    out = tmp_path / 'cli_out'
    res = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / 'scripts' /
                             'ConnectionRatioPaths.py'),
         '--connections', str(conn),
         '--sources', 'S1,S2,S3', '--targets', 'T1,T2',
         '--min-ratio', '0.1', '--max-interlayer', '3',
         '--out', str(out)],
        capture_output=True, text=True,
        cwd=str(PROJECT_ROOT))
    assert res.returncode == 0, res.stderr
    assert 'status: ok' in res.stdout
    assert (out / 'ratio_paths_bodyId.csv').exists()


def test_cli_in_run_untouched(tmp_path):
    conn = _write_conn_csv(tmp_path)
    run = tmp_path / 'find-paths-complete_X_A_to_B_L3w1_20260101_000000'
    (run / 'data_details').mkdir(parents=True)
    pd.DataFrame({'bodyId': FRAC_SOURCES}).to_csv(
        run / 'source_neurons.csv', index=False)
    pd.DataFrame({'bodyId': FRAC_TARGETS, 'Checked': [True, True]}
                 ).to_csv(run / 'target_neurons.csv', index=False)
    (run / 'parameters.txt').write_text(
        'min synapse number: 1\nmax interlayer: 3\nfilter by: bodyId\n'
        'separate hemispheres: False\nhemisphere filter: both\n'
        'aggregate method: product\n'
        'exclude intra-type connections: False\ndataset: X\n',
        encoding='utf-8')
    before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
    res = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / 'scripts' /
                             'ConnectionRatioPaths.py'),
         '--connections', str(conn), '--in-run', str(run),
         '--min-ratio', '0.1'],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    assert res.returncode == 0, res.stderr
    after = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
    assert before == after                              # READ-ONLY proof
    siblings = sorted(run.parent.glob(run.name + '_ratio_L3r0_1_*'))
    assert siblings and (siblings[-1] / 'ratio_paths_bodyId.csv').exists()


def test_cli_refusal_exit_code(tmp_path):
    conn = _write_conn_csv(tmp_path)
    res = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / 'scripts' /
                             'ConnectionRatioPaths.py'),
         '--connections', str(conn), '--sources', 'S1', '--targets', 'T1',
         '--min-ratio', '0', '--max-interlayer', '2',
         '--out', str(tmp_path / 'x')],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    assert res.returncode == 1
    assert 'REFUSED' in res.stderr


# ---------------------------------------------------------------------------
# 12. Import without the vispath subproject
# ---------------------------------------------------------------------------
def test_import_without_vispath():
    blocker = (
        'import sys\n'
        'class _B:\n'
        '    def find_module(self, name, path=None):\n'
        '        return self if name.split(".")[0] == "vispath_pkg" '
        'else None\n'
        '    def load_module(self, name):\n'
        '        raise ImportError("blocked")\n'
        'sys.meta_path.insert(0, _B())\n'
        'import connection_ratio_paths\n'
        'print("import ok")\n')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        str(p) for p in (SRC, PROJECT_ROOT))
    res = subprocess.run([sys.executable, '-c', blocker],
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    assert 'import ok' in res.stdout


# ---------------------------------------------------------------------------
# 13. Real-data smoke (data-gated, the §7.6 cross-check)
# ---------------------------------------------------------------------------
_FAFB_DIR = PROJECT_ROOT / 'datasets' / 'flywire_FAFB_v783'
_REAL_RUNS = (PROJECT_ROOT / 'local_data' / 'refill-experiments' /
              'real' / 'real_out')
_FAFB_GATED = pytest.mark.skipif(
    not (_FAFB_DIR / 'flywire_FAFB_v783_merged_connections.parquet').exists()
    or not any(_REAL_RUNS.rglob('source_neurons.csv')),
    reason='local FAFB v783 release or real enrollment not present')


@_FAFB_GATED
def test_real_fafb_smoke():
    enrollment = next(iter(sorted(
        _REAL_RUNS.rglob('source_neurons.csv')))).parent
    sources, targets = [], []
    src = pd.read_csv(enrollment / 'source_neurons.csv')
    tgt = pd.read_csv(enrollment / 'target_neurons.csv')
    sources = [str(x) for x in src['bodyId'].dropna()]
    targets = [str(x) for x in tgt.loc[
        tgt['Checked'].astype(str).str.lower().isin(('true', '1')),
        'bodyId'].dropna()]
    frame = load_connections(
        _FAFB_DIR / 'flywire_FAFB_v783_merged_connections.parquet')
    rec = find_ratio_paths(frame, sources, targets, min_ratio=0.001,
                           max_interlayer=2, path_budget=1_000_000,
                           drop_untyped=False)
    assert rec['status'] == 'ok'
    assert len(rec['paths']) == 1219         # bound = max_interlayer+1
    # §7.6: every emitted edge's ratio == an INDEPENDENTLY recomputed
    # full-table all-post denominator (the F9 invariant on real data).
    df = pl.read_parquet(
        _FAFB_DIR / 'flywire_FAFB_v783_merged_connections.parquet',
        columns=['bodyId_pre', 'bodyId_post', 'weight'])
    tot = df.group_by('bodyId_post').agg(
        pl.col('weight').cast(pl.Float64).sum().alias('t'))
    lut = dict(zip(tot['bodyId_post'].to_list(), tot['t'].to_list()))
    lut = {str(k): v for k, v in lut.items()}
    for edge in rec['edges'].values():
        v = edge['bodyId_post']
        assert edge['ratio'] == pytest.approx(
            edge['synapse_weight'] / lut[v], rel=1e-12)
    # W* = max emitted bottleneck (P1)
    assert rec['provenance']['strongest_retained_bottleneck'] == (
        max(p['ratio_bottleneck'] for p in rec['paths']))
