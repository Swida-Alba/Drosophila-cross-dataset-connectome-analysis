"""Shared strongest-selection core (§6c of the report-fixes plan).

ONE canonical definition of "strongest" for everything that bounds or
ranks paths in DROCAT:

- a path's strength is its BOTTLENECK (the minimum edge weight along it);
- ranking is by descending bottleneck, with deterministic tie rules;
- a budget selects exactly ``{paths : bottleneck >= tau}`` (ties at tau
  fully drained), so a budgeted run at threshold t is *equivalent to a
  complete run at the raised threshold tau*.

Consumers:
- ``fast_graph_core.find_paths_strongest_first`` (enumeration);
- ``vispath.VisualizePath`` edge/path selection (display);
- ``coana`` prune/notes reporting (strongest-retained bottleneck).

Pure functions only — no graph-object or I/O dependencies — so the
enumeration and display layers can share it without coupling.

Edge-weight semantics are per-run: synapse counts in the default basis,
F9 connection ratios under weight_basis='connection_ratio' — the
bottleneck/ranking logic itself is basis-agnostic (float-safe).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

INF = float('inf')


def bottleneck(weights: Sequence[float]) -> float:
    """Min edge weight along a path (the path's strength). Empty → 0."""
    if not weights:
        return 0.0
    return float(min(weights))


def interior_strength(weights: Sequence[float]) -> float:
    """Weakest NON-endpoint hop (display refinement: boundary hops are
    often weak purely because they touch the source/target boundary)."""
    if len(weights) > 2:
        return float(min(weights[1:-1]))
    return bottleneck(weights)


def widest_path_backward(
    adj: Dict,
    targets: Iterable,
    cutoff: int,
) -> List[Dict]:
    """Best achievable bottleneck from every node to any target.

    ``W[d][v]`` = max over paths ``v -> any target`` using at most ``d``
    edges of the min edge weight along that path; ``W[0][t] = inf`` for
    targets (the path may stop there). Nodes that cannot reach a target
    within ``d`` edges are absent from ``W[d]``.

    ``adj`` is a ``{u: {v: weight}}`` mapping. O(cutoff * E).
    """
    target_set = set(targets)
    W_prev = {t: INF for t in target_set if t in adj}
    W_all = [dict(W_prev)]
    for _ in range(int(cutoff)):
        W_cur = dict(W_prev)
        for u, neighbors in adj.items():
            best = W_cur.get(u, -INF)
            for v, w in neighbors.items():
                tail = W_prev.get(v)
                if tail is None:
                    continue
                cand = w if tail == INF else min(w, tail)
                if cand > best:
                    best = cand
            if best > -INF:
                W_cur[u] = best
        W_all.append(W_cur)
        W_prev = W_cur
    return W_all


def path_rank_key(
    weights: Sequence[float],
    path_probability: float = 0.0,
) -> Tuple[float, float, float]:
    """Ranking key for display selection: bottleneck primary, interior
    strength and path probability as tie-breakers."""
    return (bottleneck(weights), interior_strength(weights),
            float(path_probability))


def selection_threshold(kept_bottlenecks: Sequence[float]) -> Optional[float]:
    """τ of a selected set: the weakest bottleneck it still contains
    (None when nothing was kept)."""
    if not kept_bottlenecks:
        return None
    return float(min(kept_bottlenecks))


def drain_budget(
    ranked: Sequence[Tuple[float, Tuple]],
    budget: Optional[int],
    per_pair_k: Optional[int] = None,
) -> Tuple[List[int], Optional[float], bool]:
    """Greedy strongest-first selection over pre-ranked items.

    ``ranked`` must be sorted by descending bottleneck. Returns
    ``(selected_indices, tau, budget_bitten)``. With ``budget`` set, the
    selection drains every tie at τ so the kept set is exactly
    ``{bottleneck >= tau}``. ``per_pair_k`` additionally skips items whose
    (source, target) pair already has k selections (display fair-share;
    breaks the pure τ-prefix property).
    """
    selected: List[int] = []
    tau: Optional[float] = None
    tau_open = False
    truncated = False
    pair_counts: Dict[tuple, int] = {}

    for idx, (bn, pair, _item) in enumerate(ranked):
        if tau_open and bn < tau:
            truncated = True
            break
        if per_pair_k is not None:
            if pair_counts.get(pair, 0) >= per_pair_k:
                continue
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        selected.append(idx)
        tau = bn
        if budget is not None and len(selected) >= budget:
            tau_open = True

    if tau is None:
        tau = min((bn for bn, _p, _i in ranked), default=None)
    return selected, tau, bool(budget is not None and truncated)
