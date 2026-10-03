"""Connection-ratio pathfinding lane — standalone Phase 1
(plan-connection-ratio-pathfinding).

A parallel FindAllPath-analog lane in which the EDGE WEIGHT BASIS is the
bodyId-level connection ratio

    ratio(u -> v) = synapses(u -> v) / total_incoming(v)

with the F9 threshold-free denominator: ``total_incoming(v)`` is the
ALL-POST incoming synapse mass of ``v`` at ``min_weight=1`` over the
FULL dataset table (the same definition as the pipeline's
``connection_ratio`` readout columns — coana
``_fetch_total_incoming_weight``, docstring "ratio is a threshold-free
READOUT — callers computing ratios MUST pass min_weight=1 (all-post)").

Semantics (plan §2): threshold on ratio (``>=``), the two production
edge-admission guards (``drop_untyped`` DEFAULT ON — untyped neurons can
never be intermediate nodes, and an untyped ENROLLED source/target keeps
its enrollment but loses its incident edges; ``exclude_intra_type``
DEFAULT OFF), lossless hop-bound closure, an optional float-tier Edge
Budget floor, and the PRODUCTION StrongestFirst enumerator (descending
bottleneck = min ratio, budget drains all ties at tau) via
``type_level_refill._enumerate_paths`` — so tau tie-drain and
determinism are inherited from the same code path synapse runs use.

Design contracts pinned by tests (plan §7/§17/§18):
- Type-level ratio = MASS RECOMPUTE from synapse masses (§2 stage 8):
  emitted synapse mass / the post TYPE's all-post incoming mass. Never a
  fold of bodyId ratios (a weighted mean equals the mass ratio only
  under uniform denominators).
- W* (``strongest_retained_bottleneck``) = the P1 measured ceiling: the
  MAX bottleneck over emitted paths (coana :17358-17367 semantics);
  ``None`` when nothing was emitted.
- ``tau_canonical`` = the weakest distinct ratio tier STRICTLY above w2
  (the float analog of the integer ``w2 + 1``); the degenerate
  bottom-tier gap is disclosed, never clamped.
- Cone discovery + closure run in POLARS (a python-dict closure over
  real-scale qualifying sets measured 12-37 s in the exploration round;
  plan §18 D-5).
- ``implied_syn_cutoff = max(1, ceil(t_r * total_incoming))`` — for
  low-total posts the raw product is sub-synapse and would silently
  admit every edge; the no-op-post count is disclosed (§18 D-7).
- Empty cone after threshold = STATUS ``no_paths`` with a disclosure
  record (NOT a refusal): on hub-targeted enrollments the target-side
  ratio ceiling makes high-``t_r`` empty cones a legitimate answer, and
  ratio thresholds compound per hop (§18 D-6).

This module never imports coana/statvis; the only production surface it
touches is ``type_level_refill`` (hop_prefilter / _enumerate_paths /
_type_probability / build_effective_type_map / read_enrollment /
read_run_provenance) and, through it, the production FastGraph
enumerator. Phase 2 (in-pipeline ``weight_basis`` field, UI, exports) is
planned separately and touches no code here.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import polars as pl

from type_level_refill import (
    DEFAULT_DETAIL_CAP,
    _enumerate_paths,
    _type_probability,
    build_effective_type_map,
    hop_prefilter,
)

__all__ = [
    'ConnectionRatioPathError',
    'DEFAULT_PATH_BUDGET',
    'DEFAULT_MAX_PROBES',
    'PROB_RATIO_SCALE',
    'compute_incoming_totals',
    'prepare_ratio_frame',
    'discover_cone',
    'fit_edge_budget_ratio',
    'ratio_threshold_provenance',
    'derive_label_paths',
    'find_ratio_paths',
    'compute_connection_ratio_paths',
    'write_outputs',
    'load_connections',
]

DEFAULT_PATH_BUDGET = 1_000_000   # production: max_paths_bodyid or 1_000_000
DEFAULT_MAX_PROBES = 8            # production fit_edge_budget probe cap
PROB_RATIO_SCALE = 0.3            # traversal_probability = ratio / 0.3
WEIGHT_BASIS = 'connection_ratio_bodyid'
THRESHOLD_UNIT = 'connection_ratio'

STATUS_OK = 'ok'
STATUS_NO_PATHS = 'no_paths'

_PRE = 'bodyId_pre'
_POST = 'bodyId_post'


class ConnectionRatioPathError(Exception):
    """Refusal: the ratio lane cannot run honestly on these inputs."""


# ---------------------------------------------------------------------------
# Input normalization + ratio math (§4.1)
# ---------------------------------------------------------------------------

def _as_frame(edges) -> pl.DataFrame:
    """Normalize any supported edge source into a canonical polars frame
    ``[bodyId_pre, bodyId_post, weight(Float64)]`` with duplicate
    (pre, post) rows summed (the FastGraph ``add_edge`` convention)."""
    if isinstance(edges, pl.DataFrame):
        frame = edges
    elif isinstance(edges, pd.DataFrame):
        frame = pl.from_pandas(edges)
    else:
        rows = [(str(u), str(v), float(w)) for u, v, w in edges]
        frame = pl.DataFrame(
            rows, schema={_PRE: pl.Utf8, _POST: pl.Utf8, 'weight': pl.Float64},
            orient='row') if rows else pl.DataFrame(
            schema={_PRE: pl.Utf8, _POST: pl.Utf8, 'weight': pl.Float64})
    for col in (_PRE, _POST, 'weight'):
        if col not in frame.columns:
            raise ConnectionRatioPathError(
                f'edge source lacks the required column {col!r}.')
    frame = (frame
             .with_columns([pl.col(_PRE).cast(pl.Utf8),
                            pl.col(_POST).cast(pl.Utf8),
                            pl.col('weight').cast(pl.Float64)])
             .group_by([_PRE, _POST])
             .agg(pl.col('weight').sum())
             .sort([_PRE, _POST]))
    return frame


def compute_incoming_totals(edges) -> Dict[str, float]:
    """All-post incoming totals over the FULL edge list (the F9
    denominators). ``edges``: iterable of (pre, post, syn_w); duplicate
    (pre, post) pairs are summed into canonical weights first."""
    canon: Dict[Tuple[str, str], float] = {}
    for u, v, w in edges:
        key = (str(u), str(v))
        canon[key] = canon.get(key, 0.0) + float(w)
    totals: Dict[str, float] = defaultdict(float)
    for (_u, v), w in canon.items():
        totals[v] += w
    return dict(totals)


def prepare_ratio_frame(frame: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, float]]:
    """Join the all-post totals and the ``ratio`` column onto the
    canonical frame. The denominator is the FULL-table all-post mass —
    cone-external inputs never shrink it (the F9 invariant)."""
    tot = (frame.group_by(_POST)
           .agg(pl.col('weight').sum().alias('total')))
    frame = (frame.join(tot, on=_POST, how='left')
             .with_columns((pl.col('weight') / pl.col('total')).alias('ratio'))
             .sort([_PRE, _POST]))
    totals = dict(zip(tot[_POST].to_list(), tot['total'].to_list()))
    return frame, totals


# ---------------------------------------------------------------------------
# Cone discovery (§4.4) — polars frontier + exact-distance closure
# ---------------------------------------------------------------------------

def _hop_distances(frame: pl.DataFrame, starts: Sequence[str], from_col: str,
                   to_col: str, max_hops: int) -> pl.DataFrame:
    """(node, d) BFS table over ``frame`` edges, walking from_col -> to_col
    from ``starts``, up to ``max_hops`` hops. Only nodes present in the
    graph are seeded; determinism via unique()+sort."""
    starts = [str(s) for s in starts]
    nodes = frame.select(pl.col(from_col).alias('node')).unique()
    start_df = nodes.filter(pl.col('node').is_in(starts))
    dist = start_df.with_columns(pl.lit(0, dtype=pl.UInt32).alias('d'))
    if start_df.is_empty():
        return dist
    frontier = start_df
    for hop in range(1, int(max_hops) + 1):
        nxt = (frame.join(frontier, left_on=from_col, right_on='node',
                          how='inner')
               .select(pl.col(to_col).alias('node'))
               .unique()
               .join(dist, on='node', how='anti'))
        if nxt.is_empty():
            break
        nxt = nxt.with_columns(pl.lit(hop, dtype=pl.UInt32).alias('d'))
        dist = pl.concat([dist, nxt])
        frontier = nxt.select('node')
    return dist.sort('node')


def _label_code(label: Optional[str]) -> str:
    if label is None:
        return 'U'
    return (label.rsplit('_', 1)[-1]
            if label.endswith(('_L', '_R', '_U')) else 'U')


def _apply_guards(qual: pl.DataFrame, sources, targets, type_map,
                  *, drop_untyped: bool, exclude_intra: bool,
                  hemi_filter: str) -> Tuple[pl.DataFrame, Dict[str, int]]:
    """Edge-admission guards on the qualifying frame (plan §2 stage 5),
    mirroring the production fetch-time rules. Returns (frame, stats)."""
    stats = {'dropped_untyped': 0, 'dropped_intra_type': 0,
             'dropped_hemisphere': 0,
             'drop_untyped_inapplicable': bool(drop_untyped and not type_map)}
    if type_map:
        lut = pl.DataFrame({'node': list(type_map),
                            'label': list(type_map.values())})
        joined = (
            qual.join(lut.rename({'node': _PRE, 'label': 'label_pre'}),
                      on=_PRE, how='left')
               .join(lut.rename({'node': _POST, 'label': 'label_post'}),
                     on=_POST, how='left'))
        if drop_untyped:
            before = joined.height
            joined = joined.filter(
                pl.col('label_pre').is_not_null()
                & pl.col('label_post').is_not_null())
            stats['dropped_untyped'] = before - joined.height
        if exclude_intra:
            before = joined.height
            joined = joined.filter(
                pl.col('label_pre').is_null() | pl.col('label_post').is_null()
                | (pl.col('label_pre') != pl.col('label_post')))
            stats['dropped_intra_type'] = before - joined.height
        if hemi_filter in ('left', 'right'):
            keep = hemi_filter[0].upper()
            before = joined.height
            code_pre = (pl.when(pl.col('label_pre').is_null())
                        .then(pl.lit('U'))
                        .otherwise(pl.col('label_pre').map_elements(
                            _label_code, return_dtype=pl.Utf8)))
            code_post = (pl.when(pl.col('label_post').is_null())
                         .then(pl.lit('U'))
                         .otherwise(pl.col('label_post').map_elements(
                             _label_code, return_dtype=pl.Utf8)))
            joined = joined.filter(
                code_pre.is_in([keep, 'U']) & code_post.is_in([keep, 'U']))
            stats['dropped_hemisphere'] = before - joined.height
        out = joined.select([_PRE, _POST, 'weight', 'ratio', 'total'])
    else:
        out = qual
    return out, stats


def discover_cone(frame: pl.DataFrame, sources, targets, bound: int,
                  t_r: float, *, type_map=None, drop_untyped: bool = True,
                  exclude_intra_type: bool = False,
                  hemi_filter: str = 'both'
                  ) -> Tuple[pl.DataFrame, Dict[str, object]]:
    """Threshold + guards + lossless closure (plan §2 stages 2-5).

    Distances are computed on the t_r-filtered, guarded graph with exact
    BFS levels, so one application equals production's ITERATED closure
    fixpoint. Returns (cone_frame, discovery_stats)."""
    bound = int(bound)
    if bound < 1:
        raise ConnectionRatioPathError('max_interlayer must be >= 1.')
    sources = [str(s) for s in sources]
    targets = [str(t) for t in targets]
    qual = frame.filter(pl.col('ratio') >= float(t_r))
    qual, guard_stats = _apply_guards(
        qual, sources, targets, type_map,
        drop_untyped=drop_untyped, exclude_intra=exclude_intra_type,
        hemi_filter=hemi_filter)
    stats: Dict[str, object] = {
        'rows_full': frame.height, 'rows_qualifying': qual.height,
        **guard_stats}
    if qual.is_empty():
        stats['cone_rows'] = 0
        return qual, stats
    ds = _hop_distances(qual, sources, _PRE, _POST, bound)
    dt = _hop_distances(qual, targets, _POST, _PRE, bound)
    cone = (qual
            .join(ds.rename({'d': 'ds'}), left_on=_PRE, right_on='node',
                  how='inner')
            .join(dt.rename({'d': 'dt'}), left_on=_POST, right_on='node',
                  how='inner')
            .filter(pl.col('ds') + 1 + pl.col('dt') <= bound)
            .select([_PRE, _POST, 'weight', 'ratio', 'total'])
            .sort([_PRE, _POST]))
    stats['cone_rows'] = cone.height
    stats['cone_nodes'] = len(
        set(cone[_PRE].to_list()) | set(cone[_POST].to_list()))
    # The binding target-side ceiling (empty-cone diagnosis, §18 D-6).
    into_targets = qual.filter(pl.col(_POST).is_in(targets))
    stats['into_targets_ratio_max'] = (
        float(into_targets['ratio'].max()) if into_targets.height else None)
    out_from_sources = qual.filter(pl.col(_PRE).is_in(sources))
    stats['from_sources_ratio_max'] = (
        float(out_from_sources['ratio'].max()) if out_from_sources.height
        else None)
    return cone, stats


# ---------------------------------------------------------------------------
# Float-tier Edge Budget (§4.2) — production parity, one translation
# ---------------------------------------------------------------------------

def fit_edge_budget_ratio(cone: pl.DataFrame, budget, sources, targets,
                          bound: int, max_probes: int = DEFAULT_MAX_PROBES
                          ) -> Tuple[pl.DataFrame, Dict[str, object]]:
    """The weakest ratio tier whose single-pass hop-closed cone fits
    ``budget`` (mirrors coana ``fit_edge_budget``: distinct-tier ladder,
    hop closure per probe, gallop + bisect, ``max_probes`` cap) with ONE
    translation: the float landing is the weakest distinct tier STRICTLY
    above w1 (replacing the integer ``w1 + 1``). A floored lane is
    exactly an unfloored lane at that tier; when the probe cap runs out
    (``truncated``) the floor lands conservatively stronger than the
    weakest fitting tier — valid, disclosed."""
    stats: Dict[str, object] = {
        'applied': False, 'floor': None, 'landing': None, 'probes': 0,
        'truncated': False, 'floor_skipped': False, 'probe_trace': []}
    triples = list(zip(cone[_PRE].to_list(), cone[_POST].to_list(),
                       cone['ratio'].to_list()))
    rows = [(u, v, r, w) for (u, v, r), w in
            zip(triples, cone['weight'].to_list())]
    if not rows or not budget or int(budget) < 1:
        return cone, stats
    budget = int(budget)
    stats['edges_before'] = len(rows)
    if len(rows) <= budget:
        stats['floor_skipped'] = True
        return cone, stats

    tiers = sorted({r for _u, _v, r in triples}, reverse=True)

    def probe(tier: float) -> int:
        masked = [(u, v, r) for u, v, r in triples if r >= tier]
        closed, _dropped = hop_prefilter(masked, sources, targets, bound)
        stats['probes'] += 1
        stats['probe_trace'].append((tier, len(closed)))
        return len(closed)

    # Production w1 (coana fit_edge_budget: ``kth = total - budget`` on
    # the ascending partition) = the budget-th LARGEST weight — the
    # budget line, with ~budget rows at or above it.
    ordered = sorted((r for _u, _v, r in triples), reverse=True)
    w1 = ordered[budget - 1]
    stats['budget_line'] = w1
    # Float landing: the weakest distinct tier STRICTLY above w1
    # (production: next(w for w in sorted(distinct) if w >= w1 + 1)).
    landing_idx = None
    for i in range(len(tiers) - 1, -1, -1):   # tiers desc: weakest first
        if tiers[i] > w1:
            landing_idx = i
            break                            # FIRST hit = weakest above w1
    stats['landing'] = tiers[landing_idx] if landing_idx is not None else None

    best_idx, lo, hi = None, None, None
    start = landing_idx if landing_idx is not None else 0
    if start < len(tiers) and probe(tiers[start]) <= budget:
        best_idx, lo = start, start
        start += 1
    else:
        start = 0
    i, step = start, 1
    while i < len(tiers) and stats['probes'] < max_probes:
        if probe(tiers[i]) <= budget:
            best_idx, lo = i, i
            i += step
            step *= 2
        else:
            hi = i
            break
    if hi is None and lo is not None and lo < len(tiers) - 1:
        hi = len(tiers)
    while hi is not None and lo is not None and hi - lo > 1 \
            and stats['probes'] < max_probes:
        mid = (lo + hi) // 2
        if probe(tiers[mid]) <= budget:
            best_idx, lo = mid, mid
        else:
            hi = mid
    stats['truncated'] = bool(
        (hi is not None and lo is not None and hi - lo > 1)
        or (hi is None and lo is not None and lo < len(tiers) - 1))
    if best_idx is None:
        return cone, stats
    floor = tiers[best_idx]
    stats['applied'] = True
    stats['floor'] = floor
    kept = cone.filter(pl.col('ratio') >= floor).sort([_PRE, _POST])
    stats['edges_after'] = kept.height
    stats['dropped'] = stats['edges_before'] - kept.height
    return kept, stats


# ---------------------------------------------------------------------------
# Provenance (§4.3) — float-unit mirror, key-for-key
# ---------------------------------------------------------------------------

def ratio_threshold_provenance(
        requested_threshold: float, *,
        strongest_first_tau: Optional[float] = None,
        strongest_first_budget_bitten: bool = False,
        strongest_dropped_bottleneck: Optional[float] = None,
        strongest_retained_bottleneck: Optional[float] = None,
        edge_weight_floor: Optional[float] = None,
        edge_budget_landing: Optional[float] = None,
        edge_budget: Optional[int] = None,
        distinct_tiers: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Float-unit mirror of ``utils.threshold_state.applied_threshold_
    provenance`` with IDENTICAL key names (the int-locked original
    cannot represent ratio thresholds), plus ``weight_basis`` and
    ``threshold_unit``. ``tau_canonical`` = the weakest distinct tier
    STRICTLY above w2 (the float analog of ``w2 + 1``); when w2 is
    already the weakest tier the gap is disclosed via
    ``tau_canonical_exact: False`` — never clamped. W* = the P1 measured
    ceiling (max emitted bottleneck), None when nothing was emitted."""
    t_r = float(requested_threshold)
    bitten = bool(strongest_first_budget_bitten)
    floor = None if edge_weight_floor is None else float(edge_weight_floor)
    floor_binding = floor is not None and floor > t_r
    sources: List[str] = []
    if bitten:
        sources.append('strongest_first_budget')
    if floor_binding:
        sources.append('edge_budget')

    canonical: Optional[float] = None
    canonical_exact = True
    if not sources:
        applied, applied_source, paths_complete = t_r, 'requested', True
        if strongest_first_tau is not None:
            canonical = float(strongest_first_tau)
    else:
        applied_source = '+'.join(sources)
        paths_complete = False
        if strongest_dropped_bottleneck is not None:
            w2 = float(strongest_dropped_bottleneck)
            tiers = sorted({float(t) for t in distinct_tiers or ()})
            above = [t for t in tiers if t > w2]
            if above:
                canonical = above[0]
            else:
                canonical = w2
                canonical_exact = False   # degenerate bottom-tier gap
        elif strongest_first_tau is not None:
            canonical = float(strongest_first_tau)
        if canonical is None and floor_binding:
            canonical = floor
        if canonical is not None and floor_binding and canonical < floor:
            canonical = floor   # a floored graph cannot emit below w0
        applied = canonical if canonical is not None else t_r

    prov: Dict[str, object] = {
        'requested_threshold': t_r,
        'applied_threshold': applied,
        'applied_threshold_source': applied_source,
        'strongest_first_budget_bitten': bitten,
        'strongest_first_tau': strongest_first_tau,
        'tau_canonical': canonical,
        'tau_canonical_exact': canonical_exact,
        'strongest_dropped_bottleneck': strongest_dropped_bottleneck,
        'edge_budget': int(edge_budget) if edge_budget else None,
        'edge_budget_applied': floor_binding,
        'edge_floor_binding': floor_binding,
        'edge_budget_landing': edge_budget_landing,
        'edge_weight_floor': floor,
        'strongest_retained_bottleneck': strongest_retained_bottleneck,
        'paths_complete': paths_complete,
        'weight_basis': WEIGHT_BASIS,
        'threshold_unit': THRESHOLD_UNIT,
    }
    return prov


# ---------------------------------------------------------------------------
# Type-sequence derivation (§1.6 rule — no phantom label chains)
# ---------------------------------------------------------------------------

def derive_label_paths(paths, node_label, source_labels, target_labels,
                       kept_edges) -> List[List[str]]:
    """Unique label sequences mapped from bodyId paths; a sequence is
    accepted only when anchored at a queried source/target label and
    every consecutive label pair exists in ``kept_edges`` (the label-edge
    set). Mirrors coana's derivation (which never produces phantom label
    chains and preserves repeated-label routes)."""
    source_set = {str(x) for x in source_labels}
    target_set = {str(x) for x in target_labels}
    seen = set()
    for _bn, nodes in paths:
        seen.add(tuple(str(node_label(n)) for n in nodes))
    out: List[List[str]] = []
    for seq in sorted(seen):
        if seq[0] not in source_set or seq[-1] not in target_set:
            continue
        if all((seq[i], seq[i + 1]) in kept_edges
               for i in range(len(seq) - 1)):
            out.append(list(seq))
    return out


# ---------------------------------------------------------------------------
# The lane proper (§17 find_ratio_paths)
# ---------------------------------------------------------------------------

def find_ratio_paths(edges, sources, targets, *, min_ratio: float,
                     max_interlayer: int, edge_budget=None,
                     path_budget: int = DEFAULT_PATH_BUDGET,
                     type_map=None, aggregate_method: str = 'product',
                     drop_untyped: bool = True,
                     exclude_intra_type: bool = False,
                     hemi_filter: str = 'both',
                     max_probes: int = DEFAULT_MAX_PROBES) -> Dict[str, object]:
    """Run the ratio-basis lane (plan §2). ``edges`` = the FULL-dataset
    connection table (polars/pandas frame or (pre, post, syn_w) triples)
    at natural weights. ``max_interlayer`` uses the PRODUCTION meaning
    (intermediate layers; the enumeration hop bound is
    ``max_interlayer + 1``, coana's StrongestFirst cutoff). Returns the
    result record: status, paths, edges, type_pairs (mass-recomputed),
    provenance, ratio_to_synapse disclosure, discovery + budget stats."""
    t_r = float(min_ratio)
    if not (0.0 < t_r <= 1.0):
        raise ConnectionRatioPathError(
            f'min_ratio must be in (0, 1] — got {min_ratio!r}.')
    sources = [str(s) for s in sources]
    targets = [str(t) for t in targets]
    if not sources or not targets:
        raise ConnectionRatioPathError(
            'both sources and (Checked) targets must be non-empty.')
    if aggregate_method not in ('product', 'average', 'ratio'):
        raise ConnectionRatioPathError(
            f"aggregate_method must be product/average/ratio — got "
            f"{aggregate_method!r}.")
    # Production parity: max_interlayer counts INTERMEDIATE layers; the
    # enumerator cutoff (coana find_paths_strongest_first wiring) is
    # max_interlayer + 1 hops.
    bound = int(max_interlayer) + 1

    frame = _as_frame(edges)
    nodes = set(frame[_PRE].to_list()) | set(frame[_POST].to_list())
    missing = ([s for s in sources if s not in nodes]
               + [t for t in targets if t not in nodes])
    if missing:
        raise ConnectionRatioPathError(
            f'enrolled ids absent from the connection table: {missing}.')

    frame, totals = prepare_ratio_frame(frame)
    cone, discovery = discover_cone(
        frame, sources, targets, bound, t_r,
        type_map=type_map, drop_untyped=drop_untyped,
        exclude_intra_type=exclude_intra_type, hemi_filter=hemi_filter)

    budget_stats: Dict[str, object] = {'applied': False}
    if cone.height and edge_budget:
        cone, budget_stats = fit_edge_budget_ratio(
            cone, edge_budget, sources, targets, bound,
            max_probes=max_probes)

    rec: Dict[str, object] = {
        'min_ratio': t_r, 'max_interlayer': int(max_interlayer),
        'sources': sources, 'targets': targets,
        'edge_budget': (int(edge_budget) if edge_budget else None),
        'path_budget': int(path_budget),
        'aggregate_method': aggregate_method,
        'drop_untyped': bool(drop_untyped),
        'exclude_intra_type': bool(exclude_intra_type),
        'hemi_filter': hemi_filter,
        'discovery': discovery, 'edge_budget_stats': budget_stats,
        'totals': totals,
    }

    if cone.is_empty():
        rec['status'] = STATUS_NO_PATHS
        rec['paths'] = []
        rec['edges'] = {}
        rec['type_pairs'] = []
        rec['provenance'] = ratio_threshold_provenance(
            t_r, edge_weight_floor=budget_stats.get('floor'),
            edge_budget_landing=budget_stats.get('budget_line'),
            edge_budget=edge_budget,
            distinct_tiers=[])
        rec['ratio_to_synapse'] = _empty_cone_disclosure(
            frame, sources, targets, bound, t_r, discovery,
            type_map=type_map, drop_untyped=drop_untyped,
            exclude_intra_type=exclude_intra_type, hemi_filter=hemi_filter)
        return rec

    triples = list(zip(cone[_PRE].to_list(), cone[_POST].to_list(),
                       cone['ratio'].to_list()))
    syn_map = {(u, v): w for (u, v, _r), w in
               zip(triples, cone['weight'].to_list())}
    ratio_map = {(u, v): r for u, v, r in triples}
    enum_paths, info, enum_stats = _enumerate_paths(
        triples, sources, targets, bound,
        budget=int(path_budget) if path_budget else None)

    paths_out = []
    for rank, (bn, nodes) in enumerate(enum_paths):
        hops = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
        syn_ws = [syn_map[e] for e in hops]
        paths_out.append({
            'rank': rank + 1,
            'path': '->'.join(nodes),
            'nodes': list(nodes),
            'hops': len(nodes) - 1,
            'ratio_bottleneck': bn,
            'synapse_bottleneck': min(syn_ws),
            'synapse_sum': sum(syn_ws),
            'ratios': [ratio_map[e] for e in hops],
            'traversal_probability': min(1.0, bn / PROB_RATIO_SCALE),
        })

    w_star = max((bn for bn, _p in enum_paths), default=None)
    provenance = ratio_threshold_provenance(
        t_r,
        strongest_first_tau=enum_stats.get('tau'),
        strongest_first_budget_bitten=enum_stats.get('budget_bitten', False),
        strongest_dropped_bottleneck=enum_stats.get('strongest_dropped'),
        strongest_retained_bottleneck=w_star,
        edge_weight_floor=budget_stats.get('floor'),
        edge_budget_landing=budget_stats.get('budget_line'),
        edge_budget=edge_budget,
        distinct_tiers=sorted({r for _u, _v, r in triples}))

    edges_out = {}
    for (u, v), e in info.items():
        r = e['w']
        edges_out[f'{u}->{v}'] = {
            'bodyId_pre': u, 'bodyId_post': v,
            'synapse_weight': syn_map[(u, v)], 'ratio': r,
            'traversal_probability': min(1.0, r / PROB_RATIO_SCALE),
            'traversals': e['traversals'], 'hops': e['hops'],
        }

    type_pairs = _type_pair_readouts(
        info, syn_map, type_map, totals, aggregate_method)

    rec.update({
        'status': STATUS_OK,
        'paths': paths_out,
        'edges': edges_out,
        'type_pairs': type_pairs,
        'provenance': provenance,
        'enumeration_stats': enum_stats,
        'ratio_to_synapse': _ratio_to_synapse(cone, totals, t_r),
    })
    return rec


def _traversal_probability(ratio: float) -> float:
    return min(1.0, ratio / PROB_RATIO_SCALE)


def _type_pair_readouts(info, syn_map, type_map, totals,
                        aggregate_method) -> List[Dict[str, object]]:
    """Mass-recomputed type-pair readouts (§2 stage 8): emitted synapse
    mass / the post TYPE's all-post incoming mass — never a fold of
    bodyId ratios. Traversal probability folds by aggregate_method
    (product/average over the pair channels; 'ratio' = min(mass ratio /
    0.3, 1))."""
    if not type_map:
        return []
    members: Dict[str, List[str]] = defaultdict(list)
    for bid, label in type_map.items():
        members[label].append(bid)
    type_totals = {label: sum(totals.get(b, 0.0) for b in bids)
                   for label, bids in members.items()}

    agg: Dict[Tuple[str, str], Dict[str, object]] = {}
    for (u, v), _e in info.items():
        pair = (type_map.get(u), type_map.get(v))
        if pair[0] is None or pair[1] is None:
            continue   # emitted edges always carry labels under the guards
        slot = agg.setdefault(pair, {
            'type_pre': pair[0], 'type_post': pair[1],
            'synapse_mass': 0.0, 'bodyid_pairs': 0,
            'pair_prob_weights': []})
        w = syn_map[(u, v)]
        slot['synapse_mass'] += w
        slot['bodyid_pairs'] += 1
        r = w / totals[v] if totals.get(v) else None
        if r is not None:
            slot['pair_prob_weights'].append(
                (_traversal_probability(r), w))

    rows = []
    for pair in sorted(agg):
        slot = agg[pair]
        denom = type_totals.get(pair[1], 0.0)
        mass_ratio = (slot['synapse_mass'] / denom) if denom > 0 else None
        fold = _type_probability(slot['pair_prob_weights'],
                                 aggregate_method, mass_ratio)
        rows.append({
            'type_pre': pair[0], 'type_post': pair[1],
            'bodyid_pairs': slot['bodyid_pairs'],
            'synapse_mass': slot['synapse_mass'],
            'total_incoming_type': denom,
            'connection_ratio': mass_ratio,
            'traversal_probability': fold,
        })
    return rows


def _implied_cutoff(t_r: float, total: float) -> int:
    """§18 D-7: clamp to >= 1 — a sub-synapse cutoff admits every edge."""
    return max(1, math.ceil(t_r * total))


def _ratio_to_synapse(cone: pl.DataFrame, totals, t_r: float
                      ) -> Dict[str, object]:
    """The ratio->synapse mapping disclosure (§13.2): kept-edge synapse
    range, per-post implied cutoff stats, no-op-post count, and a capped
    per-post table (one group_by pass — never a per-post filter scan)."""
    kept_syn = cone['weight'].to_list()
    per_post = (cone.group_by(_POST)
                .agg([pl.len().alias('kept_in_edges'),
                      pl.col('weight').min().alias('kept_syn_min'),
                      pl.col('weight').max().alias('kept_syn_max')])
                .to_dicts())
    rows = []
    noop = 0
    for entry in per_post:
        v = entry[_POST]
        total = totals.get(v, 0.0)
        cut = _implied_cutoff(t_r, total)
        if t_r * total < 1.0:
            noop += 1
        rows.append({
            'bodyId_post': v, 'total_incoming': total,
            'implied_syn_cutoff': cut,
            'kept_in_edges': entry['kept_in_edges'],
            'kept_syn_min': entry['kept_syn_min'],
            'kept_syn_max': entry['kept_syn_max'],
        })
    rows.sort(key=lambda r: (-r['total_incoming'], r['bodyId_post']))
    cutoffs = [r['implied_syn_cutoff'] for r in rows]
    return {
        'kept_edge_synapse_range': [min(kept_syn), max(kept_syn)]
        if kept_syn else None,
        'implied_cutoff_min': min(cutoffs) if cutoffs else None,
        'implied_cutoff_median': (sorted(cutoffs)[len(cutoffs) // 2]
                                  if cutoffs else None),
        'implied_cutoff_max': max(cutoffs) if cutoffs else None,
        'noop_cutoff_posts': noop,
        'posts_total': len(rows),
        'posts_capped_detail': rows[:DEFAULT_DETAIL_CAP],
    }


def _empty_cone_disclosure(frame, sources, targets, bound, t_r, discovery,
                           *, type_map=None, drop_untyped: bool = True,
                           exclude_intra_type: bool = False,
                           hemi_filter: str = 'both'
                           ) -> Dict[str, object]:
    """§17: an empty cone is a STATUS, not a refusal — disclose the
    binding ceiling + the t_r=0 cone size (under the run's OWN guards,
    so the count is the guarded upper bound) + the compounding note."""
    note = ('ratio thresholds compound per hop: every hop must clear '
            't_r, so viable thresholds sit far below single-hop ratio '
            'quantiles (plan §18 D-6)')
    zero, _stats = discover_cone(
        frame, sources, targets, bound, 0.0, type_map=type_map,
        drop_untyped=drop_untyped, exclude_intra_type=exclude_intra_type,
        hemi_filter=hemi_filter)
    return {
        'note': note,
        'cone_edges_at_zero_threshold': zero.height,
        'into_targets_ratio_max': discovery.get('into_targets_ratio_max'),
        'from_sources_ratio_max': discovery.get('from_sources_ratio_max'),
    }


# ---------------------------------------------------------------------------
# Loader + orchestrator + writers (§5/§6)
# ---------------------------------------------------------------------------

def load_connections(path) -> pl.DataFrame:
    """Full-dataset connection source: parquet/csv with bodyId_pre/
    bodyId_post/weight (natural weights >= 1; the same universe the
    pipeline's connection cache stores)."""
    path = Path(path)
    if not path.exists():
        raise ConnectionRatioPathError(f'--connections {path}: not found.')
    if path.suffix.lower() == '.parquet':
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path, infer_schema_length=0)
    missing = {_PRE, _POST, 'weight'} - set(df.columns)
    if missing:
        raise ConnectionRatioPathError(
            f'--connections {path}: missing columns {sorted(missing)}.')
    return _as_frame(df.select([_PRE, _POST, 'weight']))


def _format_decimal_for_folder(value: float) -> str:
    """Folder-safe decimal (coana convention: '.' -> '_', '-' -> 'neg')."""
    return str(value).replace('-', 'neg').replace('.', '_')


def compute_connection_ratio_paths(*, connections, sources, targets,
                                   min_ratio: float, max_interlayer: int,
                                   neuron_frame=None, dataset: str = '',
                                   mapping_file=None, label_mapper=None,
                                   separate_hemispheres: bool = False,
                                   hemi_filter: str = 'both',
                                   edge_budget=None,
                                   path_budget: int = DEFAULT_PATH_BUDGET,
                                   aggregate_method: str = 'product',
                                   drop_untyped: bool = True,
                                   exclude_intra_type: bool = False,
                                   max_probes: int = DEFAULT_MAX_PROBES,
                                   out_dir=None, run_dir=None,
                                   source_note: str = ''
                                   ) -> Dict[str, object]:
    """Orchestrator: build the effective type map (when a neuron table is
    given), run :func:`find_ratio_paths`, and (by default) write the
    §6 outputs. ``run_dir`` (the ``--in-run`` mode) is READ-ONLY — the
    outputs go to ``out_dir`` or a SIBLING folder, never into the run."""
    type_map = None
    if neuron_frame is not None:
        type_map = build_effective_type_map(
            neuron_frame, dataset=dataset, mapping_file=mapping_file,
            label_mapper=label_mapper,
            separate_hemispheres=separate_hemispheres)
    rec = find_ratio_paths(
        connections, sources, targets, min_ratio=min_ratio,
        max_interlayer=max_interlayer, edge_budget=edge_budget,
        path_budget=path_budget, type_map=type_map,
        aggregate_method=aggregate_method, drop_untyped=drop_untyped,
        exclude_intra_type=exclude_intra_type, hemi_filter=hemi_filter,
        max_probes=max_probes)
    rec['dataset'] = dataset
    rec['type_map'] = type_map
    rec['separate_hemispheres'] = bool(separate_hemispheres)
    rec['source_note'] = source_note
    if out_dir is not None or run_dir is not None:
        out = Path(out_dir) if out_dir is not None else _sibling_out_dir(
            Path(run_dir), rec['min_ratio'], rec['max_interlayer'])
        rec['out_dir'] = str(out)
        write_outputs(out, rec)
    return rec


def _sibling_out_dir(run_dir: Path, t_r: float, bound: int) -> Path:
    stamp = time.strftime('%Y%m%d_%H%M%S')
    return run_dir.parent / (
        f"{run_dir.name}_ratio_L{bound}"
        f"r{_format_decimal_for_folder(t_r)}_{stamp}")


_OUTPUT_COLUMNS_PATHS = [
    'rank', 'path', 'hops', 'ratio_bottleneck', 'synapse_bottleneck',
    'synapse_sum', 'traversal_probability', 'ratios']
_OUTPUT_COLUMNS_EDGES = [
    'bodyId_pre', 'bodyId_post', 'synapse_weight', 'ratio',
    'traversal_probability', 'traversals', 'hops']
_OUTPUT_COLUMNS_TYPE_PAIRS = [
    'type_pre', 'type_post', 'bodyid_pairs', 'synapse_mass',
    'total_incoming_type', 'connection_ratio', 'traversal_probability']
_SYNAPSE_MAP_COLUMNS = [
    'bodyId_post', 'total_incoming', 'implied_syn_cutoff', 'kept_in_edges',
    'kept_syn_min', 'kept_syn_max']


def write_outputs(out_dir, rec: Dict[str, object]) -> None:
    """§6 outputs, byte-deterministic (sorted rows; a README only when
    absent). Empty frames write header-only CSVs, never zero-byte."""
    out = Path(out_dir)
    (out / 'data_details').mkdir(parents=True, exist_ok=True)

    prov = rec.get('provenance', {})
    lines = [
        "weight basis: connection_ratio (bodyId)",
        f"min connection ratio: {rec['min_ratio']}",
        f"max interlayer: {rec['max_interlayer']}",
        f"aggregate method: {rec['aggregate_method']}",
        f"drop untyped: {rec['drop_untyped']}",
        f"exclude intra-type connections: {rec['exclude_intra_type']}",
        f"separate hemispheres: {rec.get('separate_hemispheres', False)}",
        f"hemisphere filter: {rec['hemi_filter']}",
        f"dataset: {rec.get('dataset', '')}",
    ]
    if rec.get('edge_budget'):
        lines.append(f"edge budget: {rec['edge_budget']}")
    if rec.get('path_budget'):
        lines.append(f"path budget: {rec['path_budget']}")
    for key in ('requested_threshold', 'applied_threshold',
                'applied_threshold_source', 'strongest_first_budget_bitten',
                'strongest_first_tau', 'tau_canonical',
                'strongest_dropped_bottleneck', 'edge_budget',
                'edge_budget_applied', 'edge_budget_landing',
                'edge_weight_floor', 'strongest_retained_bottleneck',
                'paths_complete'):
        if key in prov and prov[key] is not None:
            lines.append(f"{key}: {prov[key]}")
    (out / 'parameters.txt').write_text('\n'.join(lines) + '\n',
                                        encoding='utf-8')

    pd.DataFrame(rec['paths'],
                 columns=_OUTPUT_COLUMNS_PATHS).to_csv(
        out / 'ratio_paths_bodyId.csv', index=False)

    edges = list(rec['edges'].values())
    edges.sort(key=lambda e: (e['bodyId_pre'], e['bodyId_post']))
    pd.DataFrame(edges, columns=_OUTPUT_COLUMNS_EDGES).to_csv(
        out / 'data_details' / 'ratio_edges_bodyId.csv', index=False)

    pd.DataFrame(rec['type_pairs'],
                 columns=_OUTPUT_COLUMNS_TYPE_PAIRS).to_csv(
        out / 'data_details' / 'ratio_type_pairs.csv', index=False)

    disclosure = rec.get('ratio_to_synapse', {})
    synapse_map_rows = disclosure.get('posts_capped_detail', [])
    pd.DataFrame(synapse_map_rows,
                 columns=_SYNAPSE_MAP_COLUMNS).to_csv(
        out / 'data_details' / 'ratio_synapse_map.csv', index=False)

    prov_json = {
        'provenance': prov,
        'discovery': _jsonify(rec.get('discovery', {})),
        'edge_budget_stats': _jsonify(rec.get('edge_budget_stats', {})),
        'enumeration_stats': _jsonify(rec.get('enumeration_stats', {})),
        'ratio_to_synapse': {
            k: v for k, v in disclosure.items()
            if k != 'posts_capped_detail'},
        'status': rec.get('status'),
        'source_note': rec.get('source_note', ''),
    }
    (out / 'data_details' / 'ratio_provenance.json').write_text(
        json.dumps(prov_json, indent=2, default=str, sort_keys=True),
        encoding='utf-8')

    readme = out / 'README.md'
    if not readme.exists():
        readme.write_text(_README_TEXT, encoding='utf-8')


def _jsonify(obj):
    """JSON-safe copy (floats stay floats; sets/tuples -> lists)."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in (sorted(obj) if isinstance(obj, set)
                                      else obj)]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    return obj


_README_TEXT = """# Connection-ratio pathfinding output (Phase 1 standalone)

Edge weight basis: bodyId-level connection ratio
`weight / total_incoming(post)` with the F9 all-post, threshold-free
denominator (min_weight=1 over the full dataset table). Thresholds,
budgets and bottlenecks are in RATIO units; every row also carries the
underlying synapse counts.

- `ratio_paths_bodyId.csv` — emitted paths, strongest-first by
  RATIO bottleneck (the lane's strength definition; `min_ratio` in the
  pipeline's allpaths exports is the same quantity as a readout).
- `data_details/ratio_edges_bodyId.csv` — per on-path edge: both bases,
  traversal counts/hops.
- `data_details/ratio_type_pairs.csv` — MASS-RECOMPUTED type ratios:
  emitted synapse mass / the post TYPE's all-post incoming mass.
- `data_details/ratio_synapse_map.csv` — the ratio->synapse mapping:
  per post, `implied_syn_cutoff = max(1, ceil(t_r * total_incoming))`
  (a sub-synapse cutoff would admit every edge; no-op posts disclosed
  in ratio_provenance.json).
- `data_details/ratio_provenance.json` — thresholds/budgets/tier traces.

See `_plan/plan-connection-ratio-pathfinding.md` (§2, §6, §13, §18).
"""
