"""Query-specific threshold alignment (threshold-alignment spec, Feature C).

A bisection PROBER over the lowest-threshold extract of each dataset finds
the best-matching threshold in every other dataset on demand. Per dataset,
ONE precompute on the extract (O(E) groupby of per-pair max weight, then
sort) answers "how many distinct type pairs survive at threshold t" for ANY
t in O(log P) via binary search — the count function is monotone
non-increasing under ``weight >= t`` filtering.

The canonical alignment criterion (spec §6.5): the number of DISTINCT
mapped type pairs with at least one edge of weight >= t. Edge-count
distance is the PRIMARY metric:

    d = |n_a - n_b| / max(n_a, n_b, 1)   (0 when both counts are 0)

Secondary metrics — Jaccard over the shared edge keys (sanity column,
meaningful only with aligned/mapped type names) and Spearman rank
similarity over per-dataset weights of the shared pairs (blank when fewer
than 3 shared edges).
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ALIGNMENT_TOLERANCE = 0.10
MIN_SHARED_EDGES_FOR_RANK = 3


def edge_count_distance(n_a: int, n_b: int) -> float:
    """Primary alignment metric: symmetric, scale-free edge-count distance."""
    denom = max(n_a, n_b, 1)
    return abs(n_a - n_b) / denom


def jaccard(set_a: Set, set_b: Set) -> Optional[float]:
    """|A ∩ B| / |A ∪ B|; None when both sets are empty (undefined)."""
    if not set_a and not set_b:
        return None
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union else None


def rank_similarity(weights_a: Dict, weights_b: Dict) -> Optional[float]:
    """Spearman rank correlation of per-dataset weights on SHARED keys.

    None when the shared-edge count is below ``MIN_SHARED_EDGES_FOR_RANK``
    (spec §6.5) or the correlation is undefined (constant ranks).
    """
    shared = weights_a.keys() & weights_b.keys()
    if len(shared) < MIN_SHARED_EDGES_FOR_RANK:
        return None
    a = pd.Series([weights_a[k] for k in sorted(shared)], dtype=float)
    b = pd.Series([weights_b[k] for k in sorted(shared)], dtype=float)
    if a.nunique() <= 1 or b.nunique() <= 1:
        return None
    try:
        corr = a.corr(b, method='spearman')
    except Exception:
        return None
    return float(corr) if corr == corr else None  # NaN -> None


class EdgeDensityProber:
    """Bisection prober over one dataset's lowest-threshold extract (§6.2).

    Precompute: ``pair_max`` = max weight per (type_pre, type_post) pair,
    sorted ascending. ``count(t)`` = number of pairs with max weight >= t
    = ``P - searchsorted(sorted_max, t)`` — monotone non-increasing in t.
    """

    def __init__(self, extract_df: pd.DataFrame,
                 pre_col: str = 'type_pre', post_col: str = 'type_post',
                 weight_col: str = 'weight'):
        if (extract_df is None or extract_df.empty
                or pre_col not in extract_df.columns
                or post_col not in extract_df.columns):
            self.pair_max = np.array([], dtype=float)
            self.pair_weights: Dict[Tuple, float] = {}
            return
        frame = extract_df[[pre_col, post_col, weight_col]].copy()
        frame[pre_col] = frame[pre_col].astype(str)
        frame[post_col] = frame[post_col].astype(str)
        grouped = frame.groupby([pre_col, post_col])[weight_col].max()
        self.pair_weights = {
            (pre, post): float(w) for (pre, post), w in grouped.items()
        }
        self.pair_max = grouped.to_numpy(dtype=float)
        self.pair_max.sort()

    @property
    def total_pairs(self) -> int:
        return int(len(self.pair_max))

    def count(self, t: float) -> int:
        """Surviving type pairs at threshold t (weight >= t). O(log P)."""
        if self.total_pairs == 0:
            return 0
        return int(self.total_pairs - np.searchsorted(self.pair_max, t, side='left'))

    def counts(self, thresholds) -> List[int]:
        return [self.count(t) for t in thresholds]

    def edge_set(self, t: float) -> Set[Tuple[str, str]]:
        """Distinct (pre, post) pairs present at threshold t."""
        return {pair for pair, w in self.pair_weights.items() if w >= t}

    def edge_weights(self, t: float) -> Dict[Tuple[str, str], float]:
        """Per-pair max weights for pairs present at threshold t."""
        return {pair: w for pair, w in self.pair_weights.items() if w >= t}

    def match_interval(self, target_count: int, cap: int = 30) -> Tuple[int, int]:
        """Threshold interval [lo, hi] whose count equals ``target_count``.

        On a plateau many thresholds share one count; returning the whole
        interval is more informative than an arbitrary single point. When
        no t achieves the count exactly, returns the single best t as a
        degenerate interval.
        """
        if self.total_pairs == 0:
            return (0, 0)
        cap = max(int(cap), 1)
        best_t, best_count = None, None
        for t in range(1, cap + 1):
            c = self.count(t)
            if best_count is None or abs(c - target_count) < abs(
                    best_count - target_count):
                best_t, best_count = t, c
        lo = hi = best_t if best_t is not None else 0
        if best_count == target_count:
            while lo > 1 and self.count(lo - 1) == target_count:
                lo -= 1
            while hi < cap and self.count(hi + 1) == target_count:
                hi += 1
        return (lo, hi)

    def best_match(self, anchor_count: int, cap: int = 30,
                   neighbor_radius: int = 2) -> Dict:
        """Find the extended-grid threshold whose pair count is closest to
        ``anchor_count`` (bisection + ±``neighbor_radius`` neighbors, §6.2).

        Returns a dict with ``best_t`` (None when the target is empty OR
        its density is entirely below the anchor's range, so no threshold
        can match), ``best_t_range`` (the plateau achieving the best count),
        ``count_at_best_t``, ``count_distance`` and a ``match_status`` in
        {exact, within_tolerance, outside_tolerance,
         target_density_below_range}.
        """
        result = {
            'best_t': None,
            'best_t_range': None,
            'count_at_best_t': 0,
            'count_distance': float(anchor_count) if anchor_count else 0.0,
            'match_status': 'target_density_below_range',
        }
        if self.total_pairs == 0:
            return result

        grid = range(1, max(int(cap), 1) + 1)
        # Bisection: largest t with count(t) >= anchor_count (count is
        # monotone non-increasing). The crossover point is then refined by
        # evaluating the ±neighbor_radius neighborhood for near-ties.
        lo_t, hi_t = 1, int(cap)
        # binary search over t for the crossover
        while lo_t < hi_t:
            mid = (lo_t + hi_t + 1) // 2
            if self.count(mid) >= anchor_count:
                lo_t = mid
            else:
                hi_t = mid - 1
        candidates = {lo_t}
        for delta in range(1, neighbor_radius + 1):
            candidates.add(lo_t - delta)
            candidates.add(lo_t + delta)
        best_t, best_dist, best_count = None, None, 0
        for t in sorted(c for c in candidates if c in grid):
            c = self.count(t)
            d = abs(c - anchor_count)
            if best_dist is None or d < best_dist:
                best_t, best_dist, best_count = t, d, c
        max_count = self.count(1)
        # No threshold in range can reach the anchor's density: report the
        # shortfall honestly instead of fabricating best_t=1.
        if anchor_count > max_count:
            result.update({
                'best_t': None,
                'best_t_range': None,
                'count_at_best_t': int(max_count),
                'count_distance': edge_count_distance(anchor_count, max_count),
                'match_status': 'target_density_below_range',
            })
            return result
        if best_t is None:
            return result
        lo, hi = self.match_interval(best_count, cap=cap)
        status = ('exact' if best_dist == 0 else
                  'within_tolerance'
                  if edge_count_distance(anchor_count, best_count)
                  <= ALIGNMENT_TOLERANCE else 'outside_tolerance')
        result.update({
            'best_t': best_t,
            'best_t_range': (lo, hi),
            'count_at_best_t': int(best_count),
            'count_distance': edge_count_distance(anchor_count, best_count),
            'match_status': status,
        })
        return result


def suggest_combination_rows(
    best_matches_df,
    reference_dataset: str,
    dataset_order: List[str],
    tolerance: float = ALIGNMENT_TOLERANCE,
) -> List[Dict]:
    """Turn an alignment best-match frame into threshold-combination rows.

    Plan §7C: each reference-dataset anchor (applied threshold) becomes one
    query row ``{ref: anchor_t, target: best_t, ...}`` when every target is
    within ``tolerance`` (or the row is flagged). Output rows use the
    ``threshold_combinations`` shape consumed by ``ComparisonParameters``.
    """
    if best_matches_df is None or getattr(best_matches_df, 'empty', True):
        return []
    df = best_matches_df
    if 'match_kind' in df.columns:
        df = df[df['match_kind'] == 'anchor']
    rows: List[Dict] = []
    seen_signatures = set()
    anchors = sorted({
        (int(r['anchor_threshold']))
        for _, r in df.iterrows()
        if r.get('reference_dataset') == reference_dataset
        and r.get('best_t') is not None
    })
    index = 1
    for anchor in anchors:
        sub = df[(df['reference_dataset'] == reference_dataset)
                 & (df['anchor_threshold'] == anchor)
                 & (df['best_t'].notna())]
        cell = {reference_dataset: int(anchor)}
        distances = []
        for _, r in sub.iterrows():
            target = r.get('target_dataset')
            if target in dataset_order and target not in cell:
                cell[target] = int(r['best_t'])
                if r.get('count_distance') is not None:
                    distances.append(float(r['count_distance']))
        if set(cell.keys()) != set(dataset_order):
            continue  # incomplete: no best match for some target
        signature = tuple(sorted(cell.items()))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        rows.append({
            'id': f'combo_{index:03d}',
            'label': (f'Aligned @{reference_dataset}={anchor}'
                      + ('' if not distances else
                         f" (max d={max(distances):.2f})")),
            'thresholds': cell,
            'aligned_within_tolerance': (
                bool(distances) and max(distances) <= tolerance),
        })
        index += 1
    return rows


def build_alignment_matrix(
    grid_points: List[Tuple[str, int]],
    probers: Dict[str, EdgeDensityProber],
    mapped: bool,
) -> pd.DataFrame:
    """Pairwise metrics over the TYPED-threshold grid points only (§6.3).

    One row per unordered grid-point pair with the primary edge-count
    distance, Jaccard (only meaningful with mapped type names) and rank
    similarity (blank below the shared-edge floor).
    """
    sets = {(ds, t): probers[ds].edge_set(t) for ds, t in grid_points}
    weights = {(ds, t): probers[ds].edge_weights(t) for ds, t in grid_points}

    rows = []
    for i in range(len(grid_points)):
        for j in range(i + 1, len(grid_points)):
            (ds_a, t_a), (ds_b, t_b) = grid_points[i], grid_points[j]
            n_a, n_b = len(sets[(ds_a, t_a)]), len(sets[(ds_b, t_b)])
            jac = jaccard(sets[(ds_a, t_a)], sets[(ds_b, t_b)]) if mapped else None
            rank = rank_similarity(weights[(ds_a, t_a)], weights[(ds_b, t_b)]) \
                if mapped else None
            rows.append({
                'dataset_a': ds_a, 'threshold_a': t_a,
                'dataset_b': ds_b, 'threshold_b': t_b,
                'edge_count_a': n_a,
                'edge_count_b': n_b,
                'edge_count_distance': round(edge_count_distance(n_a, n_b), 4),
                'within_tolerance': edge_count_distance(n_a, n_b) <= ALIGNMENT_TOLERANCE,
                'jaccard': round(jac, 4) if jac is not None else None,
                'rank_similarity': round(rank, 4) if rank is not None else None,
                'shared_edges': len(sets[(ds_a, t_a)] & sets[(ds_b, t_b)]),
            })
    return pd.DataFrame(rows)
