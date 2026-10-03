"""Query-scoped threshold-density curves for auto threshold alignment.

Plan ``_plan/plan-auto-threshold-density-alignment.md`` (§4.2, §4.4, §4.5,
§4.6). Pure functions only — no side effects, no dataset access — so the
whole alignment math is unit-testable without a run.

Units (locked by the plan, §3.4 / §4.3):

* **x** = per-connection Min Synapse Count (raw ``weight``), bounded per
  dataset by ``[w_start, w_star_measured]``;
* **y** = a DENSITY ``E(t)/N`` with a *t-independent* ``N`` — the number of
  annotated nodes the query enrolled (falling back to searched nodes).
  A bare ``E(t)`` and the ``N(t)``-dependent ratio are both rejected: the
  former is not comparable across datasets of different size and the latter
  is non-monotone (the BANC id-space artifact, §4.3b), which would make
  equal-density inversion ambiguous.
* the universe is the query's own **searched graph** at the applied
  threshold — a frozen cone; ``t > applied`` is a filter over it, NOT a
  re-pruned cone (§4.5b item 2).

Ratio basis (plan-connection-ratio-pathfinding Phase-3 remainder): when
the capture's meta carries ``weight_basis: connection_ratio`` the edge
arrays are F9 connection ratios (float), ``w_start`` is the applied
ratio threshold (no synapse floor of 3), and the ladders/inversions
operate on FLOAT tiers — every cast below is value-preserving (ints
stay ints for synapse captures, so existing exports are unchanged).

Both curves are counts of ``weight >= t`` over the persisted arrays:
``E(t)`` = bodyId edges (primary) and the enumerated path count
(diagnostic, hub-inflated — never drives alignment).
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# A pair/edge "present at t" means weight >= t; the plan's ladder/inversion
# all use that convention (matching the pathfinder's ``min_synapse_num``).
_WEIGHT_FLOOR = 3


def _as_float_array(values) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def density_curve(bottlenecks, edge_weights, t_grid) -> Dict[str, np.ndarray]:
    """Exact ``>= t`` counts over the persisted arrays (plan §4.2).

    Returns ``path_count`` and ``edge_count`` aligned with ``t_grid`` plus
    ``edge_weight_sum`` (suffix sums, a diagnostic). Counts are exact by
    binary search on the sorted arrays — never a type-level ``min_weight``
    slice (the plan's §3.2 3-200x over-count regression).
    """
    grid = np.asarray(list(t_grid), dtype=np.float64)
    sb = np.sort(_as_float_array(bottlenecks))
    se = np.sort(_as_float_array(edge_weights))

    def _suffix_counts(sorted_vals: np.ndarray) -> np.ndarray:
        if sorted_vals.size == 0:
            return np.zeros(grid.shape, dtype=np.int64)
        start = np.searchsorted(sorted_vals, grid, side='left')
        return (sorted_vals.size - start).astype(np.int64)

    path_count = _suffix_counts(sb)
    edge_count = _suffix_counts(se)
    if se.size == 0:
        edge_weight_sum = np.zeros(grid.shape, dtype=np.float64)
    else:
        suffix = np.concatenate([np.cumsum(se[::-1])[::-1], [0.0]])
        start = np.searchsorted(se, grid, side='left')
        edge_weight_sum = suffix[start]
    return {
        't_grid': grid,
        'path_count': path_count,
        'edge_count': edge_count,
        'edge_weight_sum': edge_weight_sum,
    }


def dataset_window(meta: Optional[Dict]) -> Tuple[int, Optional[int]]:
    """Return ``(w_start, w_star_measured)`` for one dataset (plan §4.4).

    ``w_start = max(3, applied)``; the ceiling is the MEASURED retained
    bottleneck (``max(path_bottlenecks)``), never the stored, possibly
    hop-unbounded ``strongest_retained_bottleneck``. ``None`` ceiling means
    no window (no paths).
    """
    meta = meta or {}
    ratio = str(meta.get('weight_basis', '')) == 'connection_ratio'

    def _cast(value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return v if ratio else int(v)

    applied = _cast(meta.get('applied'))
    w_start_raw = _cast(meta.get('w_start'))
    if w_start_raw is not None:
        w_start = w_start_raw
    elif applied is not None:
        w_start = applied if ratio else max(_WEIGHT_FLOOR, applied)
    else:
        w_start = 1e-12 if ratio else _WEIGHT_FLOOR
    w_star = meta.get('w_star_measured')
    if w_star is None or (isinstance(w_star, float) and not np.isfinite(w_star)):
        return w_start, None
    return w_start, _cast(w_star)


def density_denominator(meta: Optional[Dict],
                        n_source: Optional[int] = None) -> Optional[float]:
    """The t-independent ``N`` for ``E(t)/N`` (plan §4.3).

    Prefers the caller-computed ``curve_n`` (the active basis's node count:
    typed-only with the untyped drop on, typed+untyped with it off — debris
    is never counted), then annotated nodes in the searched graph, then all
    searched nodes. Returns ``None`` when no positive denominator exists
    (caller then emits ``NaN`` density, never a fake ratio).
    """
    meta = meta or {}
    for key in ('curve_n', 'n_annotated_nodes', 'n_nodes'):
        value = meta.get(key)
        try:
            value = float(value) if value is not None else None
        except (TypeError, ValueError):
            value = None
        if value and value > 0:
            return value
    if n_source:
        return float(n_source)
    return None


def normalized_edge_density(edge_counts, meta: Optional[Dict],
                            normalizer: str = 'per_node',
                            n_source: Optional[int] = None) -> np.ndarray:
    """Apply the active normalizer and name its unit (plan §4.3).

    ``per_node`` (primary) = ``E(t)/N`` with the fixed denominator above;
    ``raw`` = bare ``E(t)``; ``per_source`` = ``E(t)/n_source``;
    ``cone`` = ``E(t)/E(applied)`` (relevant-cone edges at applied).
    Diagnostics other than ``per_node`` are available for plots/tables but
    are never the alignment axis.
    """
    counts = np.asarray(list(edge_counts), dtype=np.float64)
    meta = meta or {}
    norm = (normalizer or 'per_node').strip().lower()
    if norm == 'raw':
        return counts
    if norm == 'per_source':
        denom = float(n_source) if n_source else None
        if not denom:
            denom = density_denominator(meta, None)
        return counts / denom if denom else np.full(counts.shape, np.nan)
    if norm == 'cone':
        n_edges = meta.get('n_edges')
        try:
            n_edges = float(n_edges) if n_edges is not None else None
        except (TypeError, ValueError):
            n_edges = None
        return counts / n_edges if n_edges else np.full(counts.shape, np.nan)
    denom = density_denominator(meta, n_source)
    return counts / denom if denom else np.full(counts.shape, np.nan)


def align_vertical(windows: Dict[str, Tuple[int, Optional[int]]],
                   K: int = 5) -> List[int]:
    """Same-threshold ladder over the intersection of the windows (§4.4).

    ``W_floor = max_i w_start_i`` (all datasets complete) and
    ``W_ceiling = min_i w_star_measured_i`` (all datasets still have paths).
    The ladder is the user's asymmetric ``W_floor + (1/2)^N * W_delta``
    (N = 1..K) plus both endpoints, de-duplicated and rounded (§4.4b item 3,
    §4.6). Empty when the windows do not intersect.
    """
    los, his = [], []
    float_mode = False
    for _ds, (lo, hi) in (windows or {}).items():
        if lo is None or hi is None:
            continue
        if isinstance(lo, float) and not float(lo).is_integer():
            float_mode = True
        if isinstance(hi, float) and not float(hi).is_integer():
            float_mode = True
        los.append(lo)
        his.append(hi)
    if not los or not his:
        return []

    def _snap(v):
        # synapse captures stay integers (byte-identical ladders); ratio
        # tiers stay exact floats.
        if float_mode or (isinstance(v, float) and not v.is_integer()):
            return float(v)
        return int(v)

    w_floor = max(los)
    w_ceiling = min(his)
    if w_ceiling < w_floor:
        return []
    if w_ceiling == w_floor:
        return [_snap(w_floor)]
    delta = w_ceiling - w_floor
    grid = {_snap(w_floor), _snap(w_ceiling)}
    for n in range(1, max(int(K), 0) + 1):
        point = w_floor + (0.5 ** n) * delta
        grid.add(float(point) if float_mode else int(round(point)))
    ordered = sorted(grid)
    # Rounding can collapse points; keep at most K+2 distinct tiers while
    # always retaining both endpoints (§4.6).
    if len(ordered) > K + 2:
        keep = {ordered[0], ordered[-1]}
        keep.update(ordered[1:-1][:K])
        ordered = sorted(keep)
    return ordered


def _invert_density(thresholds: Sequence[int],
                    density: Sequence[float],
                    level: float) -> Optional[int]:
    """Smallest integer t whose density is ``<= level`` (plan §4.5).

    The curve is a non-increasing step function of integer t, so a level
    maps to a plateau; the plan's convention is the leftmost point
    (uniform low bias, documented in the axis/notes). Clamped internally to
    the curve's own domain.
    """
    ts = np.asarray(list(thresholds), dtype=np.float64)
    dens = np.asarray(list(density), dtype=np.float64)
    valid = np.isfinite(dens)
    if ts.size == 0 or not valid.any() or not np.isfinite(level):
        return None
    ts_v, dens_v = ts[valid], dens[valid]
    order = np.argsort(ts_v, kind='stable')
    ts_s, dens_s = ts_v[order], dens_v[order]
    # dens non-increasing; -dens non-decreasing. First index with
    # dens <= level  <=>  first index with -dens >= -level.
    idx = int(np.searchsorted(-dens_s, -float(level), side='left'))
    if idx <= 0:
        point = float(ts_s[0])
    elif idx >= ts_s.size:
        point = float(ts_s[-1])
    else:
        point = float(ts_s[idx])
    # Synapse grids stay integers; ratio tiers stay exact.
    return int(point) if point.is_integer() else point


def align_horizontal(curves: Dict[str, Dict], levels: int = 4,
                     normalizer: str = 'per_node') -> List[Dict]:
    """Same-density rows over the jointly-applicable band (plan §4.5).

    ``curves`` maps dataset -> ``{'thresholds': [...], 'density': [...]}``
    already normalized with the active normalizer. The band is
    ``[d_lo, d_hi]`` = ``[max_i d_i(ceiling_i), min_i d_i(floor_i)]``;
    ``levels`` evenly spaced density levels are inverted per dataset and
    clamped to each window. Identical integer combinations are collapsed
    (§4.6); a single distinct row is tagged degenerate.
    """
    usable = {
        ds: c for ds, c in (curves or {}).items()
        if c and len(c.get('thresholds') or []) and
        np.isfinite(np.asarray(c.get('density'), dtype=float)).any()
    }
    if len(usable) < 1:
        return []

    endpoint_lo = []  # densest dataset's density at its own ceiling
    endpoint_hi = []  # sparsest dataset's density at its own floor
    for ds, c in usable.items():
        ts = np.asarray(c['thresholds'])
        dens = np.asarray(c['density'], dtype=float)
        order = np.argsort(ts, kind='stable')
        ts, dens = ts[order], dens[order]
        finite = np.isfinite(dens)
        if not finite.any():
            continue
        endpoint_lo.append(float(dens[finite][-1]))   # at highest t
        endpoint_hi.append(float(dens[finite][0]))    # at lowest t
    if not endpoint_lo or not endpoint_hi:
        return []
    d_lo = max(endpoint_lo)
    d_hi = min(endpoint_hi)
    if not np.isfinite(d_lo) or not np.isfinite(d_hi) or d_lo > d_hi:
        return []

    n_levels = max(int(levels), 1)
    level_values = ([d_lo] if n_levels == 1
                    else list(np.linspace(d_lo, d_hi, n_levels)))
    dataset_order = sorted(usable.keys())
    rows: List[Dict] = []
    seen = set()
    for level in level_values:
        cell = {}
        clamped = False
        for ds in dataset_order:
            ts = np.asarray(usable[ds]['thresholds'], dtype=float)
            dens = np.asarray(usable[ds]['density'], dtype=float)
            finite = np.isfinite(dens)
            if not finite.any():
                continue
            t_lo = float(np.min(ts[finite]))
            t_hi = float(np.max(ts[finite]))
            t = _invert_density(usable[ds]['thresholds'],
                                usable[ds]['density'], level)
            if t is None:
                continue
            if t < t_lo:
                t, clamped = t_lo, True
            elif t > t_hi:
                t, clamped = t_hi, True
            cell[ds] = int(t) if float(t).is_integer() else float(t)
        if len(cell) != len(dataset_order):
            continue
        signature = tuple((ds, cell[ds]) for ds in dataset_order)
        if signature in seen:
            continue
        seen.add(signature)
        rows.append({
            'mode': 'horizontal',
            'level_continuous': float(level),
            'level_normalized': float(level),
            'thresholds': cell,
            'clamped': bool(clamped),
            'degenerate': False,
        })
    if len(rows) == 1:
        rows[0]['degenerate'] = True
    return rows


def vertical_rows(ladder: Iterable[int],
                  dataset_order: Sequence[str]) -> List[Dict]:
    """One same-integer row per ladder point, tagged ``mode=vertical``.

    See §4.4b item 6 — routed through the combination schema so both modes
    share one code path and one output shape.
    """
    rows = []
    for t in ladder or []:
        value = int(t) if float(t).is_integer() else float(t)
        rows.append({
            'mode': 'vertical',
            'level_continuous': float(t),
            'level_normalized': None,
            'thresholds': {ds: value for ds in dataset_order},
            'clamped': False,
            'degenerate': False,
        })
    return rows
