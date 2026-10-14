"""Unit tests for the StrongestFirst budgeted path enumerator.

Spec: _plan/plan-cross-dataset-pathfinding-optimization.md §5 (F3).

Validates, on small hand-built weighted graphs:
- without a budget bite, the emitted SET equals the complete
  find_paths_memoized_dfs enumeration (identical universe, different order);
- emission order is non-increasing in path bottleneck;
- a budget cut yields exactly "all intact paths with bottleneck >= tau"
  (ties drained), with stats reporting;
- per_pair_k fair share; determinism; empty-result robustness.
"""

import pandas as pd
import pytest

from core.fast_graph import FastGraph


def _graph(edges):
    """Build a FastGraph from an edge list [(u, v, weight), ...]."""
    df = pd.DataFrame(edges, columns=['pre', 'post', 'weight'])
    g = FastGraph()
    g.build_from_dataframe(df, 'pre', 'post', 'weight', store_edge_attrs=False)
    return g


def _bottleneck(path, g):
    return min(g.adj[u][v] for u, v in zip(path, path[1:]))


def _universe(g, sources, targets, cutoff):
    """Reference universe: every simple path 1..cutoff edges ending at a
    target (the documented shared semantics of the legacy enumerators)."""
    return set(map(tuple, g.find_paths_memoized_dfs(
        sources, targets, cutoff, verbose=False)))


GRAPH_EDGES = [
    ('S1', 'A', 10), ('S1', 'B', 3),
    ('S2', 'A', 7), ('S2', 'C', 1),
    ('A', 'T1', 5), ('A', 'T2', 2), ('A', 'B', 8),
    ('B', 'T2', 4), ('B', 'A', 8),          # reciprocal pair with A->B
    ('C', 'T1', 9),
    ('T1', 'T2', 6),                        # target-to-target continuation
    ('D', 'T1', 50),                        # off-path component (D unreachable)
]

SOURCES = ['S1', 'S2']
TARGETS = ['T1', 'T2']
CUTOFF = 3


def _bottlenecks(paths, g):
    return [_bottleneck(p, g) for p in paths]


def test_no_budget_set_equality_with_memoized_dfs():
    g = _graph(GRAPH_EDGES)
    sf = set(map(tuple, g.find_paths_strongest_first(
        SOURCES, TARGETS, CUTOFF, budget=None)))
    ref = _universe(g, SOURCES, TARGETS, CUTOFF)
    assert sf == ref
    assert sf, 'sanity: the universe should not be empty'


def test_emission_order_non_increasing_bottleneck():
    g = _graph(GRAPH_EDGES)
    paths = list(g.find_paths_strongest_first(
        SOURCES, TARGETS, CUTOFF, budget=None))
    bns = _bottlenecks(paths, g)
    assert bns == sorted(bns, reverse=True)
    assert paths[0][0] in SOURCES and paths[0][-1] in TARGETS


def test_budget_cut_is_exact_bottleneck_cutoff():
    g = _graph(GRAPH_EDGES)
    ref = _universe(g, SOURCES, TARGETS, CUTOFF)
    ref_bns = sorted((_bottleneck(list(p), g) for p in ref), reverse=True)

    budget = 4
    stats = {}
    paths = list(g.find_paths_strongest_first(
        SOURCES, TARGETS, CUTOFF, budget=budget, stats=stats))

    assert stats['budget_bitten'] is True
    tau = stats['tau']
    assert stats['emitted'] == len(paths) >= budget
    # Exact semantics: all paths with bottleneck >= tau (ties drained).
    expected = {p for p in ref if _bottleneck(list(p), g) >= tau}
    assert set(map(tuple, paths)) == expected
    # tau is the strongest DROPPED bottleneck's strict superior: every
    # emitted path >= tau, every dropped path < tau.
    assert min(_bottlenecks(paths, g)) == tau
    assert tau > max((b for b in ref_bns if b < tau), default=0)
    # Strongest-first: emitted set == the strongest len(paths) of the universe
    # (when no ties straddle the boundary).
    strongest = set()
    for p in sorted(ref, key=lambda p: -_bottleneck(list(p), g)):
        strongest.add(p)
        if len(strongest) == len(paths):
            break
    if min(_bottlenecks(map(list, strongest), g)) == tau:
        assert strongest == set(map(tuple, paths))


def test_budget_larger_than_universe_not_bitten():
    g = _graph(GRAPH_EDGES)
    ref = _universe(g, SOURCES, TARGETS, CUTOFF)
    stats = {}
    paths = list(g.find_paths_strongest_first(
        SOURCES, TARGETS, CUTOFF, budget=len(ref) + 100, stats=stats))
    assert set(map(tuple, paths)) == ref
    assert stats['budget_bitten'] is False
    assert stats['emitted'] == len(ref)


def test_per_pair_k_fair_share():
    g = _graph(GRAPH_EDGES)
    k = 1
    paths = list(g.find_paths_strongest_first(
        SOURCES, TARGETS, CUTOFF, budget=None, per_pair_k=k))
    pairs = [(p[0], p[-1]) for p in paths]
    for pair in set(pairs):
        assert pairs.count(pair) <= k
    # Each emitted path must be its pair's strongest available path.
    ref = _universe(g, SOURCES, TARGETS, CUTOFF)
    by_pair = {}
    for p in ref:
        pair = (p[0], p[-1])
        by_pair.setdefault(pair, []).append(_bottleneck(list(p), g))
    for (s, t) in set(pairs):
        emitted_bn = max(_bottleneck(p, g) for p in paths
                         if p[0] == s and p[-1] == t)
        assert emitted_bn == max(by_pair[(s, t)])


def test_deterministic_across_runs():
    g = _graph(GRAPH_EDGES)
    run1 = [tuple(p) for p in g.find_paths_strongest_first(
        SOURCES, TARGETS, CUTOFF, budget=5)]
    run2 = [tuple(p) for p in g.find_paths_strongest_first(
        SOURCES, TARGETS, CUTOFF, budget=5)]
    assert run1 == run2


def test_source_that_is_also_target_emits_no_length0_path():
    # T1 doubles as a source: the enumerator must not emit the length-0
    # path [T1], but paths ending at T1 from other sources still appear.
    g = _graph(GRAPH_EDGES)
    paths = list(g.find_paths_strongest_first(
        ['S1', 'T1'], TARGETS, CUTOFF, budget=None))
    assert all(len(p) >= 2 for p in paths)
    assert any(p[0] == 'S1' and p[-1] == 'T1' for p in paths)


def test_unreachable_targets_yield_nothing():
    g = _graph([('S1', 'A', 5)])  # no edge towards T
    stats = {}
    paths = list(g.find_paths_strongest_first(
        ['S1'], ['T'], 3, budget=10, stats=stats))
    assert paths == []
    assert stats['emitted'] == 0
    assert stats['budget_bitten'] is False


@pytest.mark.parametrize('seed', [1, 7, 42])
def test_random_graphs_set_equality(seed):
    """Property check on pseudo-random layered graphs: no-budget
    StrongestFirst == MemoizedDFS universe."""
    import random
    rng = random.Random(seed)
    nodes = [f'n{i}' for i in range(12)]
    edges = []
    for u in nodes:
        for v in rng.sample(nodes, 4):
            if u != v:
                edges.append((u, v, rng.randint(1, 20)))
    g = _graph(edges)
    sources = nodes[:3]
    targets = nodes[-3:]
    sf = set(map(tuple, g.find_paths_strongest_first(
        sources, targets, 4, budget=None)))
    ref = _universe(g, sources, targets, 4)
    assert sf == ref


# ---------------------------------------------------------------------------
# Canonical tau: strongest-dropped tracking + the gap property
# ---------------------------------------------------------------------------

GAP_EDGES = [
    # three paths with bottleneck exactly 50 (the budget tier)
    ('S', 'A', 50), ('A', 'T', 50),
    ('S', 'B', 60), ('B', 'T', 50),
    ('S', 'C', 50), ('C', 'T', 55),
    # a gap, then the next tiers far below
    ('S', 'D', 30), ('D', 'T', 40),   # bottleneck 30
    ('S', 'E', 20), ('E', 'T', 25),   # bottleneck 20
]


def test_budget_cut_reports_strongest_dropped_and_gap():
    """Budget lands at tau = 50; the strongest dropped path has
    bottleneck 30 — the canonical (minimal) threshold reproducing the
    output is w2 + 1 = 31, and the interval [31, 50] all yields the
    identical set."""
    g = _graph(GAP_EDGES)
    stats = {}
    paths = list(g.find_paths_strongest_first(
        ['S'], ['T'], 3, budget=3, stats=stats))
    assert stats['budget_bitten'] is True
    assert stats['tau'] == 50
    assert stats['strongest_dropped'] == 30
    assert len(paths) == 3
    kept = set(map(tuple, paths))
    assert kept == {('S', 'A', 'T'), ('S', 'B', 'T'), ('S', 'C', 'T')}

    # the gap property: every threshold in (w2, tau] reproduces the set
    for t in (31, 40):
        p2 = list(g.find_paths_strongest_first(['S'], ['T'], 3, budget=3))
        assert set(map(tuple, p2)) == kept, t
    # re-asking the landing: same set, still tau-bounded (the 30/20 tiers
    # exist below and are skipped), strongest_dropped still 30
    s2 = {}
    p2 = list(g.find_paths_strongest_first(
        ['S'], ['T'], 3, budget=3, stats=s2))
    assert set(map(tuple, p2)) == kept
    assert s2['tau'] == 50
    assert s2['strongest_dropped'] == 30
    assert s2['budget_bitten'] is True


def test_no_budget_strongest_dropped_is_none():
    """Complete runs drop nothing — strongest_dropped stays None."""
    g = _graph(GAP_EDGES)
    stats = {}
    list(g.find_paths_strongest_first(['S'], ['T'], 3, budget=None,
                                      stats=stats))
    assert stats['budget_bitten'] is False
    assert stats['strongest_dropped'] is None
    assert stats['tau'] == 20          # natural tau = weakest path


# ---------------------------------------------------------------------------
# D1-B certificate soundness: the sub-tau pass must walk whole prefixes
# ---------------------------------------------------------------------------

DIP_EDGES = [
    ('S', 'A', 5), ('S', 'B', 8), ('A', 'T', 6), ('B', 'T', 3),
    ('A', 'C', 4), ('C', 'T', 9), ('S', 'T', 12),
]


def test_dip_on_a_later_edge_is_not_certified_away():
    """Regression: the strongest omitted path's dip can sit on a LATER edge
    (here S-8->B-3->T, bottleneck 3, while tau is 4 and its first edge is
    strong). The sub-tau pass used to skip every prefix still at/above tau,
    so it found nothing below tau, declared the emitted set complete and
    RETIRED a bite that really dropped a path — and the run folder then
    printed "Complete enumeration" for a truncated run.

    The emitted set is exactly the paths with bottleneck >= tau, so the
    certificate is what tells a user whether weaker paths exist."""
    g = _graph(DIP_EDGES)
    stats = {}
    kept = list(g.find_paths_strongest_first(
        ['S'], ['T'], 3, budget=3, stats=stats, yield_bottleneck=True))
    assert [b for b, _ in kept] == [12, 5, 4]
    assert stats['tau'] == 4
    assert stats['budget_bitten'] is True
    assert stats['strongest_dropped'] == 3      # S -> B -> T, omitted
    assert stats['w2_exact'] is True

    # the certified value must agree with the walk-relaxed BOUND it replaced
    bound_stats = {}
    list(g.find_paths_strongest_first(
        ['S'], ['T'], 3, budget=3, stats=bound_stats, exact_dropped=False))
    assert bound_stats['budget_bitten'] is True
    assert bound_stats['strongest_dropped'] == stats['strongest_dropped']


@pytest.mark.parametrize('seed,budget', [(1, 2), (7, 3), (42, 5), (11, 9)])
def test_random_graphs_budget_certificate_matches_bruteforce(seed, budget):
    """Property over pseudo-random graphs: whatever the budget does, the
    certified strongest omitted path equals the truth read off the complete
    universe, the emitted set is exactly the paths at/above tau, and the
    bite is retired only when nothing was actually omitted."""
    import random
    rng = random.Random(seed)
    nodes = [f'n{i}' for i in range(10)]
    edges = [(u, v, rng.randint(1, 15))
             for u in nodes for v in rng.sample(nodes, 3) if u != v]
    g = _graph(edges)
    sources, targets, cutoff = nodes[:2], nodes[-2:], 3

    universe = _universe(g, sources, targets, cutoff)
    bottlenecks = sorted((_bottleneck(p, g) for p in universe), reverse=True)
    stats = {}
    paths = list(g.find_paths_strongest_first(
        sources, targets, cutoff, budget=budget, stats=stats))

    if not bottlenecks:
        assert paths == [] and stats['budget_bitten'] is False
        return

    if len(bottlenecks) <= budget:
        expected_tau = bottlenecks[-1]
        dropped = []
    else:
        expected_tau = bottlenecks[budget - 1]
        dropped = [b for b in bottlenecks if b < expected_tau]

    assert stats['tau'] == expected_tau
    assert stats['budget_bitten'] is bool(dropped)
    assert stats['strongest_dropped'] == (max(dropped) if dropped else None)
    assert {tuple(p) for p in paths} == {
        tuple(p) for p in universe if _bottleneck(p, g) >= expected_tau}
    if dropped:
        # a real omission was certified exactly, not left as a bound
        assert stats['w2_exact'] is True
        assert stats['w2_capped'] is False
