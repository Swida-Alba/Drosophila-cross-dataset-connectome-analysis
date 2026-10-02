"""Standalone type-level connection-strength refill for budget-bitten
full-path runs (plan: ``_plan/plan-type-level-refill.md``, Phase 1).

When a FindAllPath ('all' mode) run's applied threshold exceeded the asked
threshold because the Edge Budget floored the discovery cone and/or the
StrongestFirst path budget bit, the exported type-level strength table
(``data_details/connection_type.csv``) counts only the bodyId pairs that
survived those cuts. This module computes the missing mass — the REFILL —
as standalone data records next to the run's own outputs, without touching
any existing file.

Semantics (user-ratified 2026-10-02; validated by the exploration round in
``local_data/refill-experiments/``):

1. **Refill only, no recovery** — only type pairs that already have an
   emitted row are refilled; other pairs are counted in the provenance
   census, never emitted.
2. **Hop-pruned-only edges excluded** — the refill counts bodyIds on
   PATHS, so a lossless distS/distT hop prefilter runs before enumeration
   (never removes a path edge; the criterion is documented in
   ``coana._hop_budget_pass_once``).
3. **Consistency guards** — the caller supplies the EFFECTIVE type map
   (label-mapped, hemisphere-suffixed exactly as the exported table shows)
   and the full connection source; the module additionally honors the
   run's ``exclude intra-type connections`` and refuses runs whose
   re-derived cut cannot reproduce the exported table
   (``table_reproduced`` — the hard post-hoc anchor).
4. **Re-exploration** — the refill re-runs pathfinding on the subgraph
   induced by all bodyIds of the involved types, at the asked threshold,
   with no budgets, toward ALL enrolled (Checked) targets. The emitted
   set is RE-DERIVED by enumerating that same induced graph at the run's
   effective cut (floor threshold w0, then the recorded StrongestFirst
   budget with its production tau tie-drain); refill = E*(asked) minus
   the re-derived emitted set. This is exact where a naive ``weight <
   w0`` split is not: a boundary edge with weight >= w0 whose only path
   died with the floor still belongs in the refill.

Post-hoc by design: everything is read from the run folder
(``parameters.txt`` provenance block, enrollment CSVs,
``connection_type.csv``) plus the caller-supplied connection source and
type map — the bodyId exports the pipeline suppresses by default are never
needed.

Public entry point: :func:`compute_type_level_refill`.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

REFILL_FOLDER_NAME = 'type_level_refill'
DEFAULT_DETAIL_CAP = 50_000
DEFAULT_REFILL_PATH_BUDGET = 2_000_000
_DISCLOSURE_SAMPLE_CAP = 20   # per-edge examples in the provenance JSON
PROB_RATIO_SCALE = 0.3   # traversal_probability = ratio / 0.3, capped at 1

STATUS_NO_REFILL = 'no_refill_needed'
STATUS_NO_EMITTED = 'no_emitted_rows'
STATUS_REFILLED = 'refilled'

_PROVENANCE_KEYS = (
    'requested_threshold', 'applied_threshold', 'applied_threshold_source',
    'strongest_first_budget', 'strongest_first_budget_bitten',
    'strongest_first_tau', 'tau_canonical',
    'strongest_dropped_bottleneck', 'edge_budget', 'edge_budget_applied',
    'edge_budget_landing', 'edge_weight_floor',
    'strongest_retained_bottleneck', 'paths_complete', 'replayed_from',
)


class TypeLevelRefillError(Exception):
    """Refusal: the refill cannot be computed honestly for this run."""


# ---------------------------------------------------------------------------
# Run-folder readers
# ---------------------------------------------------------------------------
def _parse_scalar(raw: Optional[str]):
    if raw is None:
        return None
    raw = raw.strip()
    if raw.lower() in ('n/a', 'none', 'not applied', 'not reached', ''):
        return None
    return raw


def _parse_int(raw: Optional[str]) -> Optional[int]:
    raw = _parse_scalar(raw)
    if raw is None:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def read_run_provenance(run_dir) -> dict:
    """Parse the applied-threshold provenance from ``parameters.txt``.

    Replay slice folders write ``edge_weight_floor`` twice (the provenance
    block plus the replay writer's own line); all occurrences must agree,
    otherwise the run is ambiguous and refused.
    """
    run_dir = Path(run_dir)
    txt = run_dir / 'parameters.txt'
    if not txt.exists():
        raise TypeLevelRefillError(
            f'{run_dir}: no parameters.txt — not a pathfinding run folder '
            f'(or an unsupported layout).')
    text = txt.read_text(encoding='utf-8', errors='replace')
    prov: Dict[str, str] = {}
    for key in _PROVENANCE_KEYS:
        matches = re.findall(rf'^{key}:\s*(.+)$', text, re.MULTILINE)
        values = {_parse_scalar(m) for m in matches}
        values.discard(None)
        if len(values) > 1:
            raise TypeLevelRefillError(
                f'{run_dir}: parameters.txt carries contradictory '
                f'{key} values {sorted(values)} — refusing.')
        if values:
            prov[key] = values.pop()
    m = re.search(r'^max interlayer:\s*(\S+)', text, re.MULTILINE)
    if m:
        prov['max_interlayer'] = _parse_int(m.group(1))
    m = re.search(r'^min synapse number:\s*(\S+)', text, re.MULTILINE)
    if m:
        prov['min_synapse'] = _parse_int(m.group(1))
    m = re.search(r'^filter by:\s*(\S+)', text, re.MULTILINE)
    if m:
        prov['filter_by'] = _parse_scalar(m.group(1))
    m = re.search(r'^separate hemispheres:\s*(\S+)', text, re.MULTILINE)
    if m:
        prov['separate_hemispheres'] = _parse_scalar(m.group(1))
    m = re.search(r'^exclude intra-type connections:\s*(\S+)', text,
                  re.MULTILINE)
    if m:
        prov['exclude_intra_type_connections'] = _parse_scalar(m.group(1))
    m = re.search(r'^aggregate method:\s*(\S+)', text, re.MULTILINE)
    if m:
        prov['aggregate_method'] = _parse_scalar(m.group(1))
    # The lossless/lossy pruning disclosure rides on all_attributes.json
    # (decision 2: hop-prune-only counts are disclosed, never refilled).
    attrs = run_dir / 'all_attributes.json'
    if attrs.exists():
        try:
            data = json.loads(attrs.read_text(encoding='utf-8'))
            record = data.get('graph_pruning_record')
            if isinstance(record, dict) and record:
                prov['graph_pruning_record'] = record
        except (OSError, ValueError):
            pass
    return prov


def read_enrollment(run_dir) -> Tuple[List[str], List[str]]:
    """Sources + enrolled (Checked) targets from the run's own enrollment
    CSVs (both carry ``bodyId``; the target table carries ``Checked``)."""
    run_dir = Path(run_dir)
    try:
        src = pd.read_csv(run_dir / 'source_neurons.csv', dtype=str)
        tgt = pd.read_csv(run_dir / 'target_neurons.csv', dtype=str)
    except FileNotFoundError as exc:
        raise TypeLevelRefillError(
            f'{run_dir}: missing enrollment CSVs ({exc.filename}).') from exc
    for frame, name in ((src, 'source_neurons.csv'),
                        (tgt, 'target_neurons.csv')):
        if 'bodyId' not in frame.columns:
            raise TypeLevelRefillError(
                f'{run_dir}: {name} lacks a bodyId column.')
    if 'Checked' not in tgt.columns:
        raise TypeLevelRefillError(
            f'{run_dir}: target_neurons.csv lacks a Checked column.')
    sources = src['bodyId'].dropna().tolist()
    targets = tgt.loc[
        tgt['Checked'].astype(str).str.lower().isin(('true', '1')),
        'bodyId'].dropna().tolist()
    return sources, targets


def read_emitted_pairs(run_dir) -> Dict[Tuple[str, str], int]:
    """Per-type-pair emitted weight, summed across ``conn_layer`` rows
    (the type-graph convention — ``add_edge`` sums duplicate pairs across
    layers; in 'all' mode each pair lives in exactly one layer table)."""
    run_dir = Path(run_dir)
    path = run_dir / 'data_details' / 'connection_type.csv'
    if not path.exists():
        raise TypeLevelRefillError(
            f'{run_dir}: no data_details/connection_type.csv — cannot '
            f'determine the emitted type pairs.')
    ct = pd.read_csv(path, dtype={'type_pre': str, 'type_post': str})
    if ct.empty or 'type_pre' not in ct.columns:
        return {}
    agg = ct.groupby(['type_pre', 'type_post'])['weight'].sum()
    return {tuple(k): int(v) for k, v in agg.items()}


# ---------------------------------------------------------------------------
# Graph machinery
# ---------------------------------------------------------------------------
def _load_fast_graph():
    """Lazy FastGraph import — the same contract as coana pathfinding
    (the module itself imports without the vispath subproject)."""
    from core.fast_graph import FastGraph
    return FastGraph


def hop_prefilter(edges: Sequence[Tuple[str, str, float]], sources,
                  targets, bound: int) -> Tuple[List, int]:
    """Lossless distS/distT hop-bound criterion on the induced edges:
    keep (u, v) iff distS(u) + 1 + distT(v) <= bound. Never removes an
    edge that lies on a simple source->target path within ``bound`` (the
    converse holds only for walks — coana._hop_budget_pass_once)."""
    adj: Dict[str, set] = defaultdict(set)
    radj: Dict[str, set] = defaultdict(set)
    for u, v, _w in edges:
        adj[u].add(v)
        radj[v].add(u)

    def bfs(starts, graph):
        dist = {}
        dq = deque()
        for s in starts:
            if s in graph:
                dist[s] = 0
                dq.append(s)
        while dq:
            u = dq.popleft()
            for v in graph.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        return dist

    BIG = 10 ** 9
    ds = bfs(set(sources), adj)
    dt = bfs(set(targets), radj)
    kept, dropped = [], 0
    for u, v, w in edges:
        a, b = ds.get(u, BIG), dt.get(v, BIG)
        if a < BIG and b < BIG and a + 1 + b <= bound:
            kept.append((u, v, w))
        else:
            dropped += 1
    return kept, dropped


def _enumerate_paths(edges, sources, targets, bound, budget=None):
    """Enumerate simple source->target paths on ``edges`` with the
    PRODUCTION strongest-first engine (descending bottleneck, budget with
    tau tie-drain — identical emission semantics to the pipeline's
    budgeted 'all'-mode core). Returns (paths, edge info, stats) where
    edge info = {(u, v): {'w', 'traversals', 'hops'}} over path edges and
    paths = [(bottleneck, tuple(nodes))] in emission order."""
    FastGraph = _load_fast_graph()
    df = pd.DataFrame(edges, columns=['bodyId_pre', 'bodyId_post', 'weight'])
    g = FastGraph()
    g.build_from_dataframe(df, 'bodyId_pre', 'bodyId_post', 'weight',
                           store_edge_attrs=False)
    weight = {(u, v): w for (u, v, w) in edges}
    stats: Dict[str, object] = {}
    paths: List[Tuple[float, Tuple[str, ...]]] = []
    traversals: Dict[Tuple[str, str], int] = defaultdict(int)
    hops: Dict[Tuple[str, str], set] = defaultdict(set)
    seen_sources = {s for s in map(str, sources) if s in g}
    seen_targets = {t for t in map(str, targets) if t in g}
    gen = g.find_paths_strongest_first(
        sorted(seen_sources), sorted(seen_targets), bound,
        budget=budget, stats=stats)
    for path in gen:
        nodes = tuple(str(n) for n in path)
        bn = min(weight[(nodes[i], nodes[i + 1])]
                 for i in range(len(nodes) - 1))
        paths.append((bn, nodes))
        for i in range(len(nodes) - 1):
            key = (nodes[i], nodes[i + 1])
            traversals[key] += 1
            hops[key].add(i)
    info = {k: {'w': weight[k], 'traversals': traversals[k],
                'hops': sorted(hops[k])} for k in traversals}
    return paths, info, stats


def build_effective_type_map(neuron_frame, *, dataset,
                             mapping_file=None, label_mapper=None,
                             separate_hemispheres=False) -> dict:
    """bodyId -> EFFECTIVE type label, mirroring the run's own type-level
    aggregation (statvis ``EnrichConnectionTable`` semantics):

    1. Label mapping first — a bodyId-level mapping wins when it differs
       from the raw id, else a type-level mapping when it differs from
       the raw type (``LabelMapper.get_label``; unmapped ids keep their
       raw labels through the identity fallback).
    2. Hemisphere suffix ``_L/_R/_U`` appended when the run separated
       hemispheres (coana ``_append_hemisphere_suffix_series`` rules:
       hemisphere > somaSide/soma side > rootSide columns, then the
       instance ``_L/_R`` suffix, defaulting to 'U'; labels that already
       carry a suffix are left alone).
    3. Untyped labels dropped (``utils.label_utils.is_untyped_type_label``)
       — they can never be involved labels.

    ``label_mapper`` (a live LabelMapper) wins over ``mapping_file``; a
    recorded mapping file that cannot be loaded raises
    :class:`TypeLevelRefillError` (refuse-on-doubt: a wrong label map
    would fail the table-reproduction anchor anyway, with a worse
    diagnostic).
    """
    import numpy as np

    frame = neuron_frame
    if 'bodyId' not in frame.columns:
        raise TypeLevelRefillError(
            'neuron table lacks a bodyId column.')
    if mapping_file is not None and label_mapper is None:
        try:
            from comparison.label_mapper import LabelMapper
            label_mapper = LabelMapper(overall_mapping_json=str(mapping_file))
        except Exception as exc:
            raise TypeLevelRefillError(
                f'cannot load the recorded mapping file '
                f'{mapping_file!r} ({exc}) — the refill cannot reproduce '
                f'the run\'s labels without it.') from exc

    body_ids = frame['bodyId'].astype(str)
    types = frame['type'].fillna('Unknown').astype(str) \
        if 'type' in frame.columns else pd.Series(
            ['Unknown'] * len(frame), index=frame.index)

    if label_mapper is not None:
        bid_map = {b: label_mapper.get_label(dataset, b)
                   for b in body_ids.unique()}
        mapped_body = body_ids.map(bid_map)
        mask_body = mapped_body.notna() & (mapped_body != body_ids)
        type_map_lut = {t: label_mapper.get_label(dataset, t)
                        for t in types[~mask_body].unique()}
        mapped_type = types.map(type_map_lut)
        mask_type = (~mask_body & (types != 'Unknown')
                     & (types != body_ids) & mapped_type.notna()
                     & (mapped_type != types))
        effective = types.copy()
        effective.loc[mask_body] = mapped_body[mask_body]
        effective.loc[mask_type] = mapped_type[mask_type]
    else:
        effective = types

    if separate_hemispheres:
        codes = _hemisphere_codes(frame)
        has_suffix = effective.str.endswith(('_L', '_R', '_U'))
        effective = effective.where(has_suffix, effective + '_' + codes)

    from utils.label_utils import is_untyped_type_label
    keep = ~effective.map(is_untyped_type_label)
    return dict(zip(body_ids[keep], effective[keep]))


def _hemisphere_codes(frame):
    """Hemisphere codes per the coana rules (vectorized mirror of
    ``_normalize_hemisphere_series`` + the instance-suffix fallback)."""
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    col = next((lowered[c] for c in
                ('hemisphere', 'soma side', 'somaside', 'rootside')
                if c in lowered), None)
    codes = pd.Series('U', index=frame.index, dtype=object)
    if col is not None:
        vals = frame[col].fillna('').astype(str).str.strip().str.lower()
        codes = vals.map(_HEMI_ALIASES).fillna('U')
    elif 'instance' in frame.columns:
        inst = frame['instance'].fillna('').astype(str)
        codes[inst.str.endswith('_R')] = 'R'
        codes[inst.str.endswith('_L')] = 'L'
    return codes


_HEMI_ALIASES = {
    'r': 'R', 'right': 'R', 'rhs': 'R', 'right hemisphere': 'R',
    'l': 'L', 'left': 'L', 'lhs': 'L', 'left hemisphere': 'L',
}


def filter_edges_by_hemisphere(edges, type_map, hemi_filter):
    """Fetch-time edge rule (coana ``_apply_hemisphere_suffix_to_conn_df``):
    'left'/'right' keep an edge only when BOTH endpoints are that side or
    unlabeled ('U'); 'both' keeps everything. The endpoint side is read
    from the (possibly suffixed) effective label."""
    if hemi_filter not in ('left', 'right'):
        return edges
    keep_code = hemi_filter[0].upper()

    def _code(body_id):
        label = type_map.get(str(body_id))
        if label is None:
            return 'U'
        return (label.rsplit('_', 1)[-1]
                if label.endswith(('_L', '_R', '_U')) else 'U')

    return (e for e in edges
            if _code(e[0]) in (keep_code, 'U')
            and _code(e[1]) in (keep_code, 'U'))


def _type_probability(pair_prob_weights: Iterable[Tuple[Optional[float],
                                                        float]],
                      method: str,
                      type_ratio: Optional[float]) -> Optional[float]:
    """Fold per-pair probabilities into the type-level
    traversal_probability (same semantics as statvis /
    coana.aggregate_method): 'product' compounds block probabilities
    (reliability/OR), 'average' is the WEIGHT-weighted mean of the pair
    probabilities, 'ratio' is min(type ratio / 0.3, 1)."""
    pairs = [(p, w) for p, w in pair_prob_weights if p is not None and w > 0]
    if not pairs:
        return None
    if method == 'average':
        total_w = sum(w for _p, w in pairs)
        return sum(p * w for p, w in pairs) / total_w
    if method == 'ratio':
        return (None if type_ratio is None
                else min(type_ratio / PROB_RATIO_SCALE, 1.0))
    # 'product' (default): reliability/OR model over the pair channels.
    block = 1.0
    for p, _w in pairs:
        block *= (1.0 - p)
    return 1.0 - block


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _load_shortest_store(run_dir: Path):
    """Load the shortest discovery store's structural pieces: per-layer
    connection frames (pair -> per-layer weights, preserving the
    cross-layer multiplicity the exported table counts), the store meta
    (targets found + hop limits) and the per-target distance maps."""
    store_dir = run_dir / 'shortest_discovery_store'
    meta_file = store_dir / 'meta.json'
    if not meta_file.exists():
        return None
    try:
        import polars as pl
        from shortest_discovery_store import (
            load_store_meta,
            load_target_distances,
        )
        meta = load_store_meta(store_dir)
        layer_weights = []      # [(layer, {(pre, post): w})]
        conn_dir = store_dir / 'connections'
        merged = conn_dir / 'connections_all.parquet'
        if merged.exists():
            frame = pl.read_parquet(merged)
            for (layer,), group in frame.group_by(
                    ['conn_layer'] if 'conn_layer' in frame.columns
                    else ['layer']):
                weights = {}
                for pre, post, w in zip(group['bodyId_pre'],
                                        group['bodyId_post'],
                                        group['weight']):
                    weights[(str(pre), str(post))] = float(w)
                layer_weights.append((str(layer), weights))
        else:
            for path in sorted(conn_dir.glob('*.parquet')):
                frame = pl.read_parquet(path)
                weights = {}
                for pre, post, w in zip(frame['bodyId_pre'],
                                        frame['bodyId_post'],
                                        frame['weight']):
                    weights[(str(pre), str(post))] = float(w)
                layer_weights.append((path.stem, weights))
        if not layer_weights:
            return None
        targets = list(meta.get('targets_found') or [])
        distances = load_target_distances(store_dir, targets)
        hop_limits = dict(meta.get('target_hop_limits') or {})
        return {
            'layer_weights': layer_weights,
            'targets': targets,
            'distances': distances,
            'hop_limits': hop_limits,
        }
    except Exception:
        return None


def _enumerate_shortest(layer_weights, node_set, sources, targets, bound,
                        budget, distances, hop_limits):
    """Shortest-mode re-exploration on the induced subgraph: per-layer
    weights summed exactly like the production batch graphs
    (``load_edge_weight_frame`` parity), enumeration seeded with the
    store's full-graph distance maps so per-pair min-hop semantics match
    the run."""
    FastGraph = _load_fast_graph()
    summed: Dict[Tuple[str, str], float] = defaultdict(float)
    for _layer, weights in layer_weights:
        for (u, v), w in weights.items():
            if u in node_set and v in node_set:
                summed[(u, v)] += w
    df = pd.DataFrame([(u, v, w) for (u, v), w in summed.items()],
                      columns=['bodyId_pre', 'bodyId_post', 'weight'])
    g = FastGraph()
    g.build_from_dataframe(df, 'bodyId_pre', 'bodyId_post', 'weight',
                           store_edge_attrs=False)
    stats: Dict[str, object] = {}
    target_list = sorted(t for t in targets if t in g)
    gen = g.find_paths_shortest_strongest_first(
        target_list, [s for s in map(str, sources) if s in g], bound,
        budget=budget, stats=stats, payload=False,
        target_cutoffs={t: hop_limits[t] for t in target_list
                        if t in hop_limits},
        target_distances={t: distances.get(t, {}) for t in target_list},
    )
    traversals: Dict[Tuple[str, str], int] = defaultdict(int)
    for path in gen:
        nodes = [str(n) for n in path]
        for i in range(len(nodes) - 1):
            traversals[(nodes[i], nodes[i + 1])] += 1
    weight_of = dict(summed)
    info = {k: {'w': weight_of[k], 'traversals': n, 'hops': [0]}
            for k, n in traversals.items()}
    return info, stats


def compute_type_level_refill(
    run_dir,
    *,
    edges: Sequence[Tuple[str, str, float]],
    type_map: Mapping[str, str],
    detail_cap: int = DEFAULT_DETAIL_CAP,
    path_budget: int = DEFAULT_REFILL_PATH_BUDGET,
    out_dir=None,
    write: bool = True,
    source_note: str = '',
    path_mode: Optional[str] = None,
) -> dict:
    """Compute (and by default write) the type-level refill records for
    one FindAllPath ('all' mode) run folder.

    Parameters
    ----------
    run_dir : run folder (``find-paths-complete_*``) with
        ``parameters.txt``, enrollment CSVs and
        ``data_details/connection_type.csv``.
    edges : the FULL connection source at natural weights —
        ``(bodyId_pre, bodyId_post, weight)`` triples covering every
        dataset connection (the F9 threshold-free denominators are
        computed from it at ``min_weight=1``).
    type_map : EFFECTIVE labels ``bodyId -> type`` exactly as the
        exported table shows them (label-mapped, hemisphere-suffixed by
        the caller; untyped ids should be absent or will never match).
    detail_cap : row cap for ``refill_bodyId_pairs.csv`` (deterministic
        weight-desc order); the type-pair summary is never capped.
    path_budget : enumeration guard for the asked-threshold
        re-exploration; a bitten guard is reported as
        ``refill_truncated`` (an honest partial refill, never a silent
        one). The cut enumeration that reproduces the exported table is
        never guarded — it enumerates exactly what the original run did.
    out_dir : output directory (defaults to
        ``<run_dir>/data_details/type_level_refill``).
    write : write the record files when True.

    Returns the provenance dict (also written as
    ``refill_provenance.json``). Raises :class:`TypeLevelRefillError` on
    any refusal (unsupported filters, unreadable artifacts, or a cut
    re-derivation that fails to reproduce the exported table).
    """
    run_dir = Path(run_dir)
    prov = read_run_provenance(run_dir)

    # Shortest mode (Phase 3): the target-rooted store supplies the layer
    # frames + distance maps; the edge budget never floors shortest mode,
    # so the only cut is the StrongestFirst budget drain.
    is_shortest = (path_mode == 'shortest'
                   or (path_mode is None
                       and run_dir.name.startswith('find-paths-shortest_')))
    store = _load_shortest_store(run_dir) if is_shortest else None
    if is_shortest and store is None:
        raise TypeLevelRefillError(
            f'{run_dir}: shortest-mode refill requires the run\'s '
            f'shortest_discovery_store/ (connections + node_distances) — '
            f'not present (retention pruned it, or a pre-batching run).')

    filter_by = prov.get('filter_by', 'bodyId')
    if filter_by and filter_by != 'bodyId':
        raise TypeLevelRefillError(
            f'{run_dir}: filter_by={filter_by!r} runs are out of scope '
            f'for the v1 refill (type-level filtering changes the '
            f'asked-threshold graph itself).')

    sources, targets = read_enrollment(run_dir)
    bound = (prov.get('max_interlayer')
             if prov.get('max_interlayer') is not None else None)
    if bound is None:
        raise TypeLevelRefillError(
            f'{run_dir}: parameters.txt lacks a readable max interlayer.')
    bound += 1

    asked = _parse_int(prov.get('requested_threshold'))
    if asked is None:
        asked = _parse_int(prov.get('min_synapse'))
    if asked is None:
        raise TypeLevelRefillError(
            f'{run_dir}: parameters.txt lacks requested_threshold.')
    # Replay canon folders stamp requested_threshold with the CANONICAL
    # value; the orchestrating request's true asked threshold rides on
    # the replayed_from line (validated in the exploration round).
    replayed_from = _parse_int(prov.get('replayed_from'))
    if replayed_from is not None and replayed_from < asked:
        asked = replayed_from
    applied = _parse_int(prov.get('applied_threshold'))
    applied = applied if applied is not None else asked
    w0 = _parse_int(prov.get('edge_weight_floor'))
    budget_n = _parse_int(prov.get('strongest_first_budget'))
    bitten = prov.get('strongest_first_budget_bitten', 'False') == 'True'
    exclude_intra = prov.get(
        'exclude_intra_type_connections', 'False') == 'True'
    aggregate_method = prov.get('aggregate_method') or 'product'

    rec: dict = {
        'run_dir': str(run_dir),
        'requested_threshold': asked,
        'requested_threshold_stamped':
            _parse_int(prov.get('requested_threshold')),
        'replayed_from': replayed_from,
        'applied_threshold': applied,
        'applied_threshold_source': prov.get('applied_threshold_source'),
        'edge_weight_floor': w0,
        'edge_budget_landing': _parse_int(prov.get('edge_budget_landing')),
        'strongest_first_budget': budget_n,
        'strongest_first_budget_bitten': bitten,
        'max_interlayer': prov.get('max_interlayer'),
        'separate_hemispheres': prov.get('separate_hemispheres'),
        'aggregate_method': aggregate_method,
        'graph_pruning_record': prov.get('graph_pruning_record'),
        'source_note': source_note,
    }

    # -- gate (I4): refill only when a budget raised the applied threshold
    if applied <= asked:
        rec['status'] = STATUS_NO_REFILL
        if write:
            _write_records(run_dir, out_dir, rec, None, None, detail_cap)
        return rec

    emitted = read_emitted_pairs(run_dir)
    if not emitted:
        rec['status'] = STATUS_NO_EMITTED
        if write:
            _write_records(run_dir, out_dir, rec, None, None, detail_cap)
        return rec
    rec['status'] = STATUS_REFILLED
    involved = {p for pair in emitted for p in pair}
    node_set = {str(b) for b, t in type_map.items() if t in involved}

    if is_shortest:
        # -- Phase 3: structure from the run's discovery store; `edges`
        # is only the F9 denominator source. Per-pair weights are the
        # per-layer sums (the exported table counts cross-layer
        # multiplicity exactly this way).
        estar, ask_stats = _enumerate_shortest(
            store['layer_weights'], node_set, sources, store['targets'],
            bound, path_budget, store['distances'], store['hop_limits'])
        refill_truncated = bool(ask_stats.get('budget_bitten'))
        _paths_asked = None
        cut_threshold = asked          # the edge budget never floors here
        cut_info, cut_stats = _enumerate_shortest(
            store['layer_weights'], node_set, sources, store['targets'],
            bound, (budget_n if bitten else None),
            store['distances'], store['hop_limits'])
        e_u = []
        for _layer, weights in store['layer_weights']:
            e_u.extend((u, v, w) for (u, v), w in weights.items()
                       if u in node_set and v in node_set)
        non_involved_excluded = sum(
            1 for _layer, weights in store['layer_weights']
            for (u, v) in weights
            if u not in node_set or v not in node_set)
        prefilter_dropped = 0
        e_pref = e_u
    else:
        # -- node set U + induced edges at the asked threshold (single
        # pass over the edge source — it may be a streamed 20M-row table)
        e_u: List[Tuple[str, str, float]] = []
        non_involved_excluded = 0
        for u, v, w in edges:
            u, v = str(u), str(v)
            if w < asked:
                continue
            if u not in node_set or v not in node_set:
                non_involved_excluded += 1
                continue
            if exclude_intra and type_map.get(u) == type_map.get(v):
                continue
            e_u.append((u, v, w))
        e_pref, prefilter_dropped = hop_prefilter(
            e_u, sources, targets, bound)

        # -- E*(asked): re-exploration on the induced subgraph (no
        # budgets, guarded only by the honest truncation flag)
        _paths_asked, estar, ask_stats = _enumerate_paths(
            e_pref, sources, targets, bound, budget=path_budget)
        refill_truncated = bool(ask_stats.get('budget_bitten'))

        # -- re-derive the emitted set at the run's effective cut
        cut_threshold = max(asked, w0 or asked)
        if cut_threshold > asked:
            e_cut = [(u, v, w) for (u, v, w) in e_pref
                     if w >= cut_threshold]
        else:
            e_cut = e_pref
        _cut_paths, cut_info, cut_stats = _enumerate_paths(
            e_cut, sources, targets, bound,
            budget=(budget_n if bitten else None))

    # -- anchor: the re-derived cut must reproduce the exported table
    edge_weight = {(u, v): w for (u, v, w) in e_pref}
    rederived: Dict[Tuple[str, str], float] = defaultdict(float)
    for (u, v), meta in cut_info.items():
        rederived[(type_map[u], type_map[v])] += meta['w']
    mismatch = {
        f'{k[0]}->{k[1]}': (int(rederived.get(k, 0)), int(v))
        for k, v in emitted.items()
        if int(rederived.get(k, 0)) != int(v)}
    extra = {f'{k[0]}->{k[1]}': int(v)
             for k, v in rederived.items() if k not in emitted}
    if mismatch or extra or not cut_info:
        detail = '; '.join(
            f'{k}: re-derived {a} vs exported {b}'
            for k, (a, b) in sorted(mismatch.items())[:8])
        raise TypeLevelRefillError(
            f'{run_dir}: cut re-derivation failed to reproduce '
            f'connection_type.csv ({len(mismatch)} mismatched, '
            f'{len(extra)} extra pairs{"; " + detail if detail else ""}) '
            f'— the supplied edges/type map do not match the run '
            f'(labels, hemisphere suffixes, filters or dataset state).')
    rec['table_reproduced'] = True

    # -- refill = E*(asked) - emitted_rederived, emitted pairs only
    refill_edges: Dict[Tuple[str, str], dict] = {}
    skipped_edges: List[Tuple[str, str, float]] = []
    boundary: List[Tuple[str, str, float]] = []
    for (u, v), meta in estar.items():
        pair = (type_map[u], type_map[v])
        if pair not in emitted:
            skipped_edges.append((u, v, meta['w']))
        elif (u, v) in cut_info:
            continue
        else:
            refill_edges[(u, v)] = meta
            if w0 is not None and meta['w'] >= w0:
                boundary.append((u, v, meta['w']))

    # -- per-type-pair aggregation + threshold-free F9 ratios
    totals: Dict[Tuple[str, str], float] = defaultdict(float)
    for (u, v), meta in refill_edges.items():
        totals[(type_map[u], type_map[v])] += meta['w']
    body_in: Dict[str, float] = defaultdict(float)
    for (u, v, w) in edges:               # all-post denominators (min_weight=1)
        body_in[str(u)] += w
    pair_ratio = {}
    for (u, v) in list(refill_edges) + list(cut_info):
        denom = body_in.get(v, 0.0)
        pair_ratio[(u, v)] = (
            edge_weight[(u, v)] / denom if denom > 0 else None)

    def _pair_prob(uv):
        r = pair_ratio.get(uv)
        return None if r is None else min(r / PROB_RATIO_SCALE, 1.0)

    type_in: Dict[str, float] = defaultdict(float)
    for (u, v, w) in edges:
        type_in[type_map.get(str(v), f'~{v}')] += w

    emitted_pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for (u, v) in cut_info:
        emitted_pair_counts[(type_map[u], type_map[v])] += 1

    rows = []
    for pair in sorted(emitted):
        emit_w = emitted[pair]
        ref_w = totals.get(pair, 0.0)
        denom = type_in.get(pair[1], 0.0)
        type_ratio = ((emit_w + ref_w) / denom) if denom > 0 else None
        union_pairs = [
            uv for uv in list(refill_edges) + list(cut_info)
            if (type_map[uv[0]], type_map[uv[1]]) == pair]
        rows.append({
            'type_pre': pair[0], 'type_post': pair[1],
            'emitted_weight': int(emit_w),
            'refill_weight': int(ref_w),
            'refilled_total': int(emit_w + ref_w),
            'emitted_pair_count': emitted_pair_counts.get(pair, 0),
            'refill_pair_count': sum(
                1 for uv in refill_edges
                if (type_map[uv[0]], type_map[uv[1]]) == pair),
            'refilled_connection_ratio': (
                round(type_ratio, 6) if type_ratio is not None else None),
            'refilled_traversal_probability': round(
                _type_probability(
                    ((_pair_prob(uv), edge_weight[uv]) for uv in union_pairs),
                    aggregate_method, type_ratio) or 0.0, 6),
            'split_status': ('rederived_floor+topn' if (w0 is not None
                                                         and bitten)
                             else 'rederived_floor' if w0 is not None
                             else 'rederived_topn'),
        })

    ordered = sorted(refill_edges.items(),
                     key=lambda kv: (-kv[1]['w'], kv[0][0], kv[0][1]))
    capped = ordered[:max(0, int(detail_cap))]
    detail = [{
        'bodyId_pre': u, 'bodyId_post': v,
        'type_pre': type_map[u], 'type_post': type_map[v],
        'weight': meta['w'],
        'traversal_count': meta['traversals'],
        'min_hop': meta['hops'][0], 'max_hop': meta['hops'][-1],
    } for (u, v), meta in capped]

    # -- disclosure hygiene (real-data lesson: the per-edge boundary list
    # alone reached 40 KB on an FAFB query) — counts + capped examples +
    # per-type-pair aggregates, never unbounded edge dumps.
    def _edge_str(uv_w):
        u, v, w = uv_w
        return f'{u}->{v}({w:g})'

    def _cap_sample(items):
        ordered_sample = sorted(items, key=lambda t: (-t[2], t[0], t[1]))
        return [_edge_str(t) for t in ordered_sample[:_DISCLOSURE_SAMPLE_CAP]]

    skipped_by_pair: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: {'count': 0, 'weight': 0.0})
    for u, v, w in skipped_edges:
        agg = skipped_by_pair[(type_map[u], type_map[v])]
        agg['count'] += 1
        agg['weight'] += w
    boundary_weight = sum(w for _u, _v, w in boundary)

    rec.update({
        'involved_labels': sorted(involved),
        'node_set_size': len(node_set),
        'induced_edges': len(e_u),
        'prefilter_dropped_in_U': prefilter_dropped,
        'estar_size': len(estar),
        'paths_asked': (len(_paths_asked) if _paths_asked is not None
                        else ask_stats.get('emitted')),
        'refill_path_budget': path_budget,
        'refill_truncated': refill_truncated,
        'cut_threshold': cut_threshold,
        'cut_paths_emitted': cut_stats.get('emitted'),
        'cut_tau': cut_stats.get('tau'),
        'emitted_rederived_size': len(cut_info),
        'table_reproduced': True,
        'refill_edges': len(refill_edges),
        'refill_weight_total': int(sum(m['w'] for m in
                                       refill_edges.values())),
        'detail_rows_written': len(detail),
        'detail_cap': int(detail_cap),
        'detail_capped': len(ordered) > len(capped),
        'boundary_refill_edge_count_at_or_above_w0': len(boundary),
        'boundary_refill_weight_at_or_above_w0': int(boundary_weight),
        'boundary_refill_examples_at_or_above_w0': _cap_sample(boundary),
        'excluded_edges_non_involved_endpoint': non_involved_excluded,
        'skipped_pairs_not_emitted_type_pair': {
            f'{k[0]}->{k[1]}': {'count': v['count'],
                                'weight': int(v['weight'])}
            for k, v in sorted(skipped_by_pair.items())},
        'skipped_edge_examples': _cap_sample(skipped_edges),
    })
    if write:
        _write_records(run_dir, out_dir, rec, rows, detail, detail_cap)
    return rec


# ---------------------------------------------------------------------------
# Record writers
# ---------------------------------------------------------------------------
_README_TEXT = """# Type-level refill records

Standalone refill of the type-level connection strength for a run whose
applied threshold exceeded the asked threshold (Edge Budget floor and/or
StrongestFirst path budget). Generated by `type_level_refill` (plan:
`_plan/plan-type-level-refill.md`); no run file was modified.

Files
-----
- `refill_type_pairs.csv` — one row per EMITTED type pair (zeros
  included). `refilled_total = emitted_weight + refill_weight` is the
  refilled strength; `refilled_connection_ratio` uses the threshold-free
  all-post denominator (F9). Refill mass counts only bodyId pairs on
  simple source->target paths within the run's hop bound on the subgraph
  induced by the involved types' neurons — a certified lower bound of the
  full complete-at-asked refill (paths that leave the induced subgraph
  are disclosed in the provenance, not refilled).
- `refill_bodyId_pairs.csv` — the refilled bodyId pairs (capped,
  deterministic weight-desc order; the summary above is never capped).
- `refill_provenance.json` — thresholds, budgets, node/edge counts,
  `table_reproduced` (the re-derived cut reproduced the exported
  `connection_type.csv` exactly — the consistency anchor), and the
  disclosure counters (`skipped_pairs_not_emitted_type_pair` census +
  capped examples, `excluded_edges_non_involved_endpoint`,
  `boundary_refill_edge_count/weight/examples_at_or_above_w0`,
  `refill_truncated`).

Join recipe
-----------
Join `refill_type_pairs.csv` on `(type_pre, type_post)` against the
run's `data_details/connection_type.csv` (sum that table's `weight`
across its `conn_layer` rows first — that sum is `emitted_weight`).
"""


_DETAIL_COLUMNS = (
    'bodyId_pre', 'bodyId_post', 'type_pre', 'type_post', 'weight',
    'traversal_count', 'min_hop', 'max_hop')


def _write_records(run_dir: Path, out_dir, rec: dict, rows, detail,
                   detail_cap: int):
    out = Path(out_dir) if out_dir else (
        run_dir / 'data_details' / REFILL_FOLDER_NAME)
    out.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        pd.DataFrame(rows).to_csv(out / 'refill_type_pairs.csv', index=False)
        # Header-only when the refill is empty (never a zero-byte file).
        frame = (pd.DataFrame(detail, columns=list(_DETAIL_COLUMNS))
                 if not detail else pd.DataFrame(detail))
        frame.to_csv(out / 'refill_bodyId_pairs.csv', index=False)
        if not (out / 'README.md').exists():
            (out / 'README.md').write_text(_README_TEXT, encoding='utf-8')
    (out / 'refill_provenance.json').write_text(
        json.dumps(rec, indent=2, sort_keys=True, default=str),
        encoding='utf-8')
