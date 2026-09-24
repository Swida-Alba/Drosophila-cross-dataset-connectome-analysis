#!/usr/bin/env python3
"""Preliminary validation of the budget-fit edge pruning design (§7.4 of
_plan/plan-pathfinding-doc-audit.md).

Standalone prototype on plain edge lists (pure stdlib, no pandas/polars):
mirrors the semantics of coana.prune_layers_hop_budget /
apply_edge_budget_floor (lossless hop-budget pass, quickselect landing,
floor + ONE lossless pass), then adds the budget-fit search (gallop +
bisection over weight tiers under a probe budget) and checks the design's
invariants against brute-force ground truth (full simple-path
enumeration) on small random graphs.

Validated invariants:
  1. closure is a fixpoint (idempotent) and lossless (brute force)
  2. C(t) non-increasing in t (threshold monotonicity, both modes)
  3. revival: an edge dead at a high threshold can be alive at a lower one
  4. one-shot replica: kept < cap strictly, slack >= 1 always (Lemma)
  5. probe 0 of the fit search == the one-shot output (single-pass mode)
  6. fit search == brute-force t* (gallop+bisect exact, non-truncated)
  7. cap invariant C(t*) <= cap and t* <= landing threshold
  8. no-floor discovery (over-flooring fix) on a planted case
  9. single-tier decline (mirror of coana's degenerate floor decline)
 10. literal ledger divergence (allowance inflates, threshold ratchets to bottom)
 11. medium-graph performance sanity (probe count, wall time)

Run:  python3 scripts/harness/verify_budget_fit_pruning.py
"""
import random
import sys
import time
from collections import deque

INF = float('inf')


# --------------------------------------------------------------------------
# core prototype
# --------------------------------------------------------------------------

def hop_pass(edges, sources, targets, bound):
    """One lossless hop-budget pass (mirror of prune_layers_hop_budget's mask):
    keep (u, v) iff distS(u) + 1 + distT(v) <= bound, distances within the
    current edge set. Returns (kept, dropped_count)."""
    adj, radj = {}, {}
    for u, v, _ in edges:
        adj.setdefault(u, set()).add(v)
        radj.setdefault(v, set()).add(u)

    def bfs(starts, graph):
        dist = {}
        dq = deque()
        for s in starts:
            if s not in dist:
                dist[s] = 0
                dq.append(s)
        while dq:
            x = dq.popleft()
            for y in graph.get(x, ()):
                if y not in dist:
                    dist[y] = dist[x] + 1
                    dq.append(y)
        return dist

    distS = bfs(set(sources), adj)
    distT = bfs(set(targets), radj)
    kept = [e for e in edges
            if distS.get(e[0], INF) + 1 + distT.get(e[1], INF) <= bound]
    return kept, len(edges) - len(kept)


def closure(edges, sources, targets, bound, max_passes=None):
    """Fixpoint closure. max_passes=1 reproduces the one-shot's single
    lossless pass; None runs to convergence (§7.1). Returns (kept, passes)."""
    cur = list(edges)
    passes = 0
    while max_passes is None or passes < max_passes:
        cur, dropped = hop_pass(cur, sources, targets, bound)
        passes += 1
        if dropped == 0:
            break
    return cur, passes


def landing_threshold(edges, cap):
    """Fix D quickselect landing: threshold keeping < cap original edges.
    None when the cone already fits the cap (real gate: total <= budget)."""
    if cap >= len(edges):
        return None
    w1 = sorted((e[2] for e in edges), reverse=True)[cap - 1]
    return w1 + 1


def oneshot_fix_d(edges, cap, sources, targets, bound):
    """Mirror of the current one-shot (apply_edge_budget_floor): floor at
    w1 + 1, drop, then ONE lossless pass; declines when degenerate."""
    t0 = landing_threshold(edges, cap)
    if t0 is None:
        return dict(status='no_floor_needed', threshold=None, kept=list(edges))
    floored = [e for e in edges if e[2] >= t0]
    if not floored:
        return dict(status='declined', threshold=None, kept=list(edges),
                    reason='landing admits nothing (single weight tier)')
    kept, _ = hop_pass(floored, sources, targets, bound)
    if not kept:
        return dict(status='declined', threshold=None, kept=list(edges),
                    reason='floor would empty the graph')
    return dict(status='floored', threshold=t0, kept=kept)


def fit_budget(edges, cap, sources, targets, bound,
               max_probes=8, max_passes=None):
    """Budget-fit search (§7.4 refined spec). A probe = mask w >= tier weight
    + closure; the landing threshold is canonicalized to the strongest tier
    weight it admits (identical cone: E(w1+1) == E(w')). Gallop down the
    tiers, then bisect between the last fitting and first overshooting tier
    under a probe budget. A zero-count probe still counts as 'fits' for
    bracketing (a weaker tier can revive the cone) — only the FINAL set must
    be non-empty, else the floor is declined. Result carries `probe_seconds`
    (wall time per probe, mask + closure)."""
    import time as _time
    distinct = sorted({e[2] for e in edges}, reverse=True)
    probes = []
    probe_seconds = []

    def probe(w):
        t0 = _time.perf_counter()
        sel = [e for e in edges if e[2] >= w]
        kept, passes = closure(sel, sources, targets, bound, max_passes)
        probe_seconds.append(_time.perf_counter() - t0)
        probes.append((w, len(kept), passes))
        return kept

    t_raw = landing_threshold(edges, cap)
    if t_raw is None:
        return dict(status='no_floor_needed', threshold=None,
                    kept=list(edges), probes=probes, probe_seconds=probe_seconds,
                    truncated=False)
    landing_w = next((w for w in sorted(distinct) if w >= t_raw), None)
    if landing_w is None:
        best_t, best_kept, lo, start = None, [], None, 0
    else:
        best_t, best_kept = landing_w, probe(landing_w)
        lo = distinct.index(landing_w)
        start = lo + 1
    hi, i, step = None, start, 1
    while i < len(distinct) and len(probes) < max_probes:
        kept = probe(distinct[i])
        if len(kept) <= cap:                     # empty fits: revival possible
            best_t, best_kept, lo = distinct[i], kept, i
            i += step
            step *= 2
        else:
            hi = i
            break
    if hi is None and i >= len(distinct) and lo is not None \
            and lo < len(distinct) - 1:
        hi = len(distinct)                       # virtual boundary: tail unseen
    while hi is not None and lo is not None and hi - lo > 1 \
            and len(probes) < max_probes:
        mid = (lo + hi) // 2
        kept = probe(distinct[mid])
        if len(kept) <= cap:
            best_t, best_kept, lo = distinct[mid], kept, mid
        else:
            hi = mid
    truncated = ((hi is not None and lo is not None and hi - lo > 1)
                 or (hi is None and lo is not None and lo < len(distinct) - 1))
    if lo is not None and lo == len(distinct) - 1 and best_kept:
        return dict(status='no_floor_needed', threshold=None, kept=best_kept,
                    probes=probes, probe_seconds=probe_seconds, truncated=False)
    if not best_kept:
        return dict(status='declined', threshold=None, kept=list(edges),
                    probes=probes, probe_seconds=probe_seconds,
                    truncated=truncated,
                    reason='no non-empty floor fits the cap')
    return dict(status='floored', threshold=best_t, kept=best_kept,
                probes=probes, probe_seconds=probe_seconds,
                truncated=truncated)


# --------------------------------------------------------------------------
# brute-force ground truth (small graphs only)
# --------------------------------------------------------------------------

def admissible_path_edges(edges, sources, targets, bound, t):
    """Union of edges over ALL simple paths (1..bound edges, weights >= t)
    from any source to any target; paths may pass through other targets.
    Matches the enumerators' path universe (e.g. find_paths_memoized_dfs)."""
    adj = {}
    for u, v, w in edges:
        if w >= t:
            adj.setdefault(u, []).append(v)
    tset = set(targets)
    found = set()
    path, visited = [], set()

    def dfs(u, hops):
        if hops >= 1 and u in tset:
            found.update(path)
        if hops == bound:
            return
        for v in adj.get(u, ()):
            if v not in visited:
                visited.add(v)
                path.append((u, v))
                dfs(v, hops + 1)
                path.pop()
                visited.remove(v)

    for s in sources:
        dfs(s, 0)
    return found


def brute_t_star(edges, cap, sources, targets, bound, max_passes=None):
    """Exhaustive scan: minimal weight threshold whose closed cone fits the
    cap with a non-empty result. None when no threshold is usable."""
    best = None
    for w in sorted({e[2] for e in edges}):
        sel = [e for e in edges if e[2] >= w]
        kept, _ = closure(sel, sources, targets, bound, max_passes)
        if kept and len(kept) <= cap:
            best = w                     # weaker tiers only get bigger
            break
    return best


# --------------------------------------------------------------------------
# random graph generator
# --------------------------------------------------------------------------

def random_graph(rng, n_nodes=20, w_max=6):
    """Layered-ish digraph with reciprocal/back edges, dead ends, and a
    small weight domain (heavy tie mass). Returns (edges, sources, targets)."""
    nodes = list(range(n_nodes))
    layer = {n: i * 4 // max(n_nodes, 1) for i, n in enumerate(nodes)}
    sources = rng.sample(nodes[:4], 2)
    targets = rng.sample(nodes[-4:], 2)
    pairs = set()
    edges = []

    def add(u, v):
        if u != v and (u, v) not in pairs:
            pairs.add((u, v))
            edges.append((u, v, rng.randint(1, w_max)))

    for u in nodes:
        for v in nodes:
            if layer[v] == layer[u] + 1 and rng.random() < 0.35:
                add(u, v)
            elif layer[v] == layer[u] + 2 and rng.random() < 0.10:
                add(u, v)
    for (u, v) in list(pairs):
        if rng.random() < 0.15:
            add(v, u)                     # reciprocal
    for _ in range(n_nodes // 2):         # noise / dead-end branches
        add(rng.choice(nodes), rng.choice(nodes))
    return edges, sources, targets


# --------------------------------------------------------------------------
# test harness
# --------------------------------------------------------------------------

FAILURES = []


def check(name, fn):
    try:
        detail = fn() or ''
        print(f'  PASS  {name} {detail}')
    except AssertionError as exc:
        FAILURES.append(name)
        print(f'  FAIL  {name}: {exc}')
    except Exception as exc:                       # noqa: BLE001
        FAILURES.append(name)
        print(f'  ERROR {name}: {type(exc).__name__}: {exc}')


def trial_graphs(n=250, seed=42):
    for i in range(n):
        rng = random.Random(seed + i)
        yield rng, *random_graph(rng)


# 1 -- closure fixpoint + losslessness (brute force) -----------------------

def test_closure_fixpoint_and_lossless():
    checked = skipped = 0
    for rng, edges, sources, targets in trial_graphs():
        bound = rng.randint(2, 4)
        for t in sorted({e[2] for e in edges}):
            sel = [e for e in edges if e[2] >= t]
            if not sel:
                continue
            kept, _ = closure(sel, sources, targets, bound)
            again, _ = hop_pass(kept, sources, targets, bound)
            assert len(again) == len(kept), 'closure not idempotent'
            try:
                adm = admissible_path_edges(edges, sources, targets, bound, t)
            except RecursionError:
                skipped += 1
                continue
            kept_set = {(u, v) for u, v, _ in kept}
            missing = adm - kept_set
            assert not missing, (
                f'losslessness violated at t={t}: {sorted(missing)[:4]}')
            checked += 1
    return f'({checked} graph/threshold combos, {skipped} skipped)'


# 2 -- C(t) non-increasing in t --------------------------------------------

def test_threshold_monotonicity():
    for max_passes in (1, None):
        for rng, edges, sources, targets in trial_graphs(80, seed=9000):
            bound = rng.randint(2, 4)
            prev = None
            for w in sorted({e[2] for e in edges}, reverse=True):
                sel = [e for e in edges if e[2] >= w]
                kept, _ = closure(sel, sources, targets, bound, max_passes)
                if prev is not None:
                    assert prev <= len(kept), (
                        f'C decreased as t weakened: mode={max_passes}, w={w}')
                prev = len(kept)
    return '(1-pass and fixpoint modes)'


# 3 -- revival: dead at a high threshold, alive at a lower one --------------

def test_revival():
    edges = [('S', 'A', 5), ('A', 'B', 5), ('B', 'T', 1)]
    sources, targets, bound = ['S'], ['T'], 3
    kept_hi, _ = closure([e for e in edges if e[2] >= 5], sources, targets, bound)
    kept_lo, _ = closure(edges, sources, targets, bound)
    assert ('A', 'B', 5) not in kept_hi, 'expected dead at t=5'
    assert ('A', 'B', 5) in kept_lo, 'expected revived at t=1'
    assert len(kept_lo) == 3
    return '(A->B dead at t=5, alive at t=1: L monotone in t)'


# 4 -- one-shot replica: strict cap fit, slack >= 1 -------------------------

def test_oneshot_strict_fit_and_slack():
    trials = 0
    for rng, edges, sources, targets in trial_graphs(150, seed=3000):
        bound = rng.randint(2, 4)
        cap = rng.randint(3, max(3, len(edges) - 1))
        res = oneshot_fix_d(edges, cap, sources, targets, bound)
        if res['status'] == 'floored':
            assert len(res['kept']) < cap, 'one-shot kept >= cap'
            assert cap - len(res['kept']) >= 1, 'slack must be >= 1'
            trials += 1
    assert trials > 50, f'too few floored trials ({trials})'
    return f'({trials} floored trials, slack >= 1 every time)'


# 5 -- probe 0 == one-shot output (single-pass probes) ----------------------

def test_probe0_equals_oneshot():
    compared = 0
    for rng, edges, sources, targets in trial_graphs(150, seed=3000):
        bound = rng.randint(2, 4)
        cap = rng.randint(3, max(3, len(edges) - 1))
        one = oneshot_fix_d(edges, cap, sources, targets, bound)
        if one['status'] != 'floored':
            continue
        fit = fit_budget(edges, cap, sources, targets, bound,
                         max_probes=1, max_passes=1)
        assert fit['status'] == 'floored', (
            f"probe0 status {fit['status']} != one-shot floored")
        assert sorted(fit['kept']) == sorted(one['kept']), 'probe0 kept set differs'
        compared += 1
    assert compared > 50
    return f'({compared} comparisons)'


# 6 -- fit search == brute-force t* -----------------------------------------

def test_fit_matches_bruteforce():
    matched = nofloor = declined = 0
    for max_passes in (1, None):
        for rng, edges, sources, targets in trial_graphs(120, seed=6000):
            bound = rng.randint(2, 4)
            cap = rng.randint(3, max(3, len(edges) - 1))
            fit = fit_budget(edges, cap, sources, targets, bound,
                             max_probes=64, max_passes=max_passes)
            bt = brute_t_star(edges, cap, sources, targets, bound, max_passes)
            if fit['status'] == 'declined':
                assert bt is None, f'brute found {bt}, fit declined'
                declined += 1
            elif fit['status'] == 'no_floor_needed':
                assert bt == min(e[2] for e in edges), (
                    f'no_floor but brute t*={bt}')
                nofloor += 1
            else:
                assert fit['threshold'] == bt, (
                    f"fit t*={fit['threshold']} != brute {bt}")
                matched += 1
    assert matched + nofloor + declined > 100
    return f'({matched} floored, {nofloor} no-floor, {declined} declined; 2 modes)'


# 7 -- cap invariant + never worse than the landing -------------------------

def test_cap_and_landing_invariants():
    for max_passes in (1, None):
        for rng, edges, sources, targets in trial_graphs(150, seed=6000):
            bound = rng.randint(2, 4)
            cap = rng.randint(3, max(3, len(edges) - 1))
            fit = fit_budget(edges, cap, sources, targets, bound,
                             max_probes=64, max_passes=max_passes)
            if fit['status'] == 'floored':
                assert len(fit['kept']) <= cap, 'cap violated'
            if max_passes == 1:
                # never worse than the one-shot: same closure mode, cone at
                # t* is a superset of the cone at the landing threshold.
                # (A DECLINED one-shot keeps the raw cone — more edges but
                # budget-violating — so it is no utilization baseline.)
                one = oneshot_fix_d(edges, cap, sources, targets, bound)
                if one['status'] == 'floored':
                    one_set = {(u, v) for u, v, _ in one['kept']}
                    fit_set = {(u, v) for u, v, _ in fit['kept']}
                    assert one_set <= fit_set, 'fit kept less than the one-shot'
    return '(both modes; superset-vs-one-shot in single-pass mode)'


# 8 -- no-floor discovery beats a lossy one-shot (planted) ------------------

def test_no_floor_beats_lossy_oneshot():
    edges = [('S', 'a', 6), ('a', 'T', 6), ('S', 'b', 5), ('b', 'c', 5),
             ('c', 'T', 5), ('S', 'd', 5), ('d', 'e', 6), ('e', 'T', 6),
             ('a', 'f', 5), ('f', 'T', 6), ('S', 'g', 5), ('g', 'T', 5),
             ('S', 'h', 6), ('h', 'i', 5), ('i', 'T', 6), ('b', 'j', 6),
             ('j', 'T', 5),                              # 17 live edges
             ('S', 'e2', 2), ('e2', 'T', 2),             # 2 live WEAK edges
             ('a', 'x1', 3), ('x1', 'x2', 3), ('b', 'x3', 2),
             ('x3', 'x4', 3), ('d', 'x5', 3), ('x5', 'x6', 2),
             ('g', 'x7', 3), ('x7', 'x8', 3), ('h', 'x9', 2),
             ('x9', 'x10', 3)]                           # 10 dead branches
    sources, targets, bound, cap = ['S'], ['T'], 3, 19
    assert len(edges) > cap
    one = oneshot_fix_d(edges, cap, sources, targets, bound)
    fit = fit_budget(edges, cap, sources, targets, bound, max_probes=16)
    assert fit['status'] == 'no_floor_needed', f"expected no-floor, got {fit['status']}"
    fit_set = {(u, v) for u, v, _ in fit['kept']}
    assert ('S', 'e2') in fit_set and ('e2', 'T') in fit_set, 'weak live path lost'
    if one['status'] == 'floored':
        one_set = {(u, v) for u, v, _ in one['kept']}
        assert ('S', 'e2') not in one_set, 'expected one-shot to be lossy here'
        assert len(fit['kept']) > len(one['kept'])
    return '(fit keeps 19 edges incl. the weak path; one-shot drops it)'


# 9 -- single-tier decline ---------------------------------------------------

def test_single_tier_decline():
    edges = [(i, i + 1, 3) for i in range(20)]
    sources, targets, bound, cap = [0], [20], 3, 5
    one = oneshot_fix_d(edges, cap, sources, targets, bound)
    fit = fit_budget(edges, cap, sources, targets, bound, max_probes=8)
    assert one['status'] == 'declined'
    assert fit['status'] == 'declined', f"expected decline, got {fit['status']}"
    assert len(fit['kept']) == len(edges), 'decline must keep the cone'
    return "(mirrors coana's degenerate-floor decline)"


# 10 -- literal ledger divergence -------------------------------------------

def test_literal_ledger_diverges():
    rng = random.Random(777)
    edges, sources, targets = random_graph(rng, n_nodes=24)
    bound = 3
    cap = max(3, len(edges) // 3)
    b, history = cap, []
    for _ in range(12):
        t = landing_threshold(edges, b)
        if t is None or t <= min(e[2] for e in edges):
            t = min(e[2] for e in edges)      # bottom: admit everything
        kept, _ = closure([e for e in edges if e[2] >= t],
                          sources, targets, bound, max_passes=1)
        r = b - len(kept)
        assert r >= 1, f'release {r} < 1: the strict-landing lemma failed'
        history.append((b, t, len(kept), r))
        b += r                                 # the user's literal ledger
    assert all(history[i + 1][0] > history[i][0] for i in range(len(history) - 1)), \
        'allowance must strictly inflate'
    assert all(history[i + 1][1] <= history[i][1] for i in range(len(history) - 1)), \
        'landing must ratchet down'
    assert history[-1][1] == min(e[2] for e in edges), 'did not reach the bottom tier'
    return f'(allowance {history[0][0]} -> {history[-1][0]}, threshold -> bottom)'


# 11 -- medium-graph performance sanity --------------------------------------

def test_medium_graph_performance():
    rng = random.Random(2024)
    layers, width, edges = 9, 600, []
    node = lambda l, i: (l, i)                    # noqa: E731
    for l in range(layers - 1):
        for i in range(width):
            for _ in range(4):
                edges.append((node(l, i), node(l + 1, rng.randrange(width)),
                              rng.randint(1, 60)))
            if rng.random() < 0.25:               # skip-layer edge
                j = min(l + 2, layers - 1)
                edges.append((node(l, i), node(j, rng.randrange(width)),
                              rng.randint(1, 60)))
            if rng.random() < 0.03:               # back edge (cycle)
                edges.append((node(l, i), node(max(l - 1, 0), rng.randrange(width)),
                              rng.randint(1, 60)))
    sources = [node(0, i) for i in range(25)]
    targets = [node(layers - 1, i) for i in range(25)]
    cap = 2500
    bound = 8        # 9-layer span: matches the depth-8 explosion scenario

    t0 = time.perf_counter()
    one = oneshot_fix_d(edges, cap, sources, targets, bound)
    t_one = time.perf_counter() - t0

    t0 = time.perf_counter()
    fit = fit_budget(edges, cap, sources, targets, bound,
                     max_probes=8, max_passes=1)
    t_fit = time.perf_counter() - t0

    assert fit['status'] == 'floored', (
        f"expected a usable floor on the medium graph, got {fit['status']}")
    assert len(fit['kept']) <= cap, 'cap violated on the medium graph'
    one_set = {(u, v) for u, v, _ in one['kept']}
    fit_set = {(u, v) for u, v, _ in fit['kept']}
    if one['status'] == 'floored':
        assert one_set <= fit_set, (
            'fit must keep a superset of the one-shot (same closure mode)')
    print(f'        graph: {len(edges):,} edges, cap {cap:,} | '
          f'one-shot {len(one["kept"]):,} kept in {t_one:.2f}s | '
          f'fit {fit["status"]} {len(fit["kept"]):,} kept in {t_fit:.2f}s '
          f'({len(fit["probes"])} probes)')
    assert t_fit < 30, 'fit search too slow'
    return f'({t_fit:.2f}s vs one-shot {t_one:.2f}s)'


# --------------------------------------------------------------------------

def main():
    print('budget-fit pruning — preliminary validation (§7.4)')
    tests = [
        ('closure fixpoint + lossless (brute force)', test_closure_fixpoint_and_lossless),
        ('C(t) monotone non-increasing in t', test_threshold_monotonicity),
        ('revival across thresholds', test_revival),
        ('one-shot strict fit + slack >= 1', test_oneshot_strict_fit_and_slack),
        ('probe 0 == one-shot (single-pass)', test_probe0_equals_oneshot),
        ('fit search == brute-force t*', test_fit_matches_bruteforce),
        ('cap invariant + t* <= landing', test_cap_and_landing_invariants),
        ('no-floor beats lossy one-shot', test_no_floor_beats_lossy_oneshot),
        ('single-tier decline', test_single_tier_decline),
        ('literal ledger diverges', test_literal_ledger_diverges),
        ('medium-graph performance', test_medium_graph_performance),
    ]
    for name, fn in tests:
        check(name, fn)
    print()
    if FAILURES:
        print(f'RESULT: {len(tests) - len(FAILURES)}/{len(tests)} passed, '
              f'{len(FAILURES)} FAILED: {FAILURES}')
        return 1
    print(f'RESULT: all {len(tests)} tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
