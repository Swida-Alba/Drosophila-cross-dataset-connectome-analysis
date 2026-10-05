"""Shortest-path batched discovery: labeled store + budgeted batches.

Implements ``_plan/plan-shortest-batched-discovery.md`` (Phases A + B).
The target-rooted backward discovery of the shortest path mode holds one
BFS per target (frontier / seen / distances / DAG edges) simultaneously —
queries with many broad targets scale that resident state linearly in the
target count (measured 2026-10-02: 242 circadian targets, 9,347,194 summed
distance states at depth 2; the depth-5 run was killed at ~450 GB RSS).
This module replaces the per-target resident maps with:

* Phase A — union-layer discovery: the union of all targets' reverse
  frontiers is fetched and scanned once per depth; per-target usage is
  streamed as label rows into a run-folder-local parquet store
  (``shortest_discovery_store/``). Per-target resident state is a
  membership bitmap over the union node index plus the current layer's
  frontier/labeled sets — nothing per-target-all-depths stays in memory.
* Phase B — budgeted batch enumeration: targets are composed into
  contiguous batches (capped by the summed per-target distance-state
  count); each batch builds a standalone graph from its DAG-edge slice +
  the store's edge weights and enumerates seeded from the store's
  distance maps; the per-batch emissions are merged in batch order and
  the global StrongestFirst budget/tau tie-drain applies exactly once.

Weight fidelity (plan §3.A / §4): ``connections/`` keeps the finalized
per-layer frames exactly as the monolithic pipeline produced them — one
file per ``conn_layer`` — because ``FastGraph.build_from_dataframe`` runs
per layer and ``add_edge`` SUMS duplicate ``(pre, post)`` rows across
layers. A post can sit at different reverse depths for two targets (its
distance differs per target), so the same edge row is legitimately fetched
into two layer frames; collapsing the store to one row per unique pair
would silently change every bottleneck computed after it. The store
therefore preserves ``conn_layer`` granularity end to end, and the batch
graphs sum per-layer weights exactly like the monolithic build.
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path

import polars as pl

# FastGraph / the shared drain are imported INSIDE batched_shortest_paths:
# coana imports this module unconditionally at import time, and the
# lazy-import contract (test_coana_imports_without_vispath_subprocess)
# requires coana to stay importable when the vispath-subproject is absent —
# enumeration is the only place the graph engine is actually needed.

# Pinned defaults (plan §0.4; user-approved 2026-10-02).
DEFAULT_DISCOVERY_BATCH_BUDGET = 2_000_000

# Store retention modes (plan-shortest-store-retention §2).
STORE_RETENTION_MODES = ('keep', 'compact', 'prune')

# Store layout (run-folder local, v1: never reused by a later run).
STORE_FOLDER_NAME = 'shortest_discovery_store'
NODE_DISTANCES_DIR = 'node_distances'
DAG_EDGES_DIR = 'dag_edges'
CONNECTIONS_DIR = 'connections'
PAIRS_FILE = 'pairs.parquet'
META_FILE = 'meta.json'
STORE_SCHEMA_VERSION = 1

# Label/dag rows are buffered per layer and flushed to a chunk file when
# the buffer passes this size (deterministic: same rows -> same chunks).
_CHUNK_ROWS = 2_000_000


# ---------------------------------------------------------------------------
# Knob resolution
# ---------------------------------------------------------------------------

def resolve_batching(fc):
    """Return (enabled, budget, fixed_size) from the analyzer's knobs.

    ``discovery_batch_budget`` (default 2,000,000; 0 = off) enables
    adaptive state-budget batching; ``target_batch_size`` (> 0) is the
    simple alternative composing fixed-size batches. Both zero = the
    legacy monolithic path (plan §3 fallback).
    """
    budget = getattr(fc, 'discovery_batch_budget',
                     DEFAULT_DISCOVERY_BATCH_BUDGET)
    try:
        budget = int(budget or 0)
    except (TypeError, ValueError):
        budget = 0
    fixed = getattr(fc, 'target_batch_size', 0)
    try:
        fixed = int(fixed or 0)
    except (TypeError, ValueError):
        fixed = 0
    return (budget > 0 or fixed > 0), budget, fixed


# ---------------------------------------------------------------------------
# Coverage early-stop (shortest): stop deepening backward discovery once
# the per-type source/target coverage requirements are both met.
# ---------------------------------------------------------------------------

def resolve_coverage_stop(fc):
    """Validate the coverage knobs and build the checkpoint config.

    Returns ``None`` when both sides are ``None`` (legacy behavior — the
    discovery twins take their fast path), otherwise
    ``{'source': req|None, 'target': req|None,
       'source_types': {type: set(bodyIds)},
       'target_types': {type: set(bodyIds)}}`` where ``req`` is ``0.0``
    (Any: >= 1 bodyId per queried type) or a fraction in (0, 1]. Raises
    ``ValueError`` on any other value (refuse-on-doubt: a mistyped
    threshold must never silently pass).
    """
    def _req(value, name):
        if value is None:
            return None
        if isinstance(value, bool):
            # bool is a float subclass (True == 1.0 == Full) — refuse
            # silently-coerced flags like every other threshold knob.
            raise ValueError(
                f'{name} must be None, 0.0 (any) or a fraction in (0, 1]; '
                f'got {value!r}')
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f'{name} must be None, 0.0 (any) or a fraction in (0, 1]; '
                f'got {value!r}') from None
        if value == 0.0:
            return 0.0
        if 0.0 < value <= 1.0:
            return value
        raise ValueError(
            f'{name} must be None, 0.0 (any) or a fraction in (0, 1]; '
            f'got {value!r}')

    source_req = _req(getattr(fc, 'shortest_source_coverage', None),
                      'shortest_source_coverage')
    target_req = _req(getattr(fc, 'shortest_target_coverage', None),
                      'shortest_target_coverage')
    if source_req is None and target_req is None:
        return None

    def _type_map(frame):
        # bodyId -> queried-type membership; the enrollment frames carry
        # the resolved type per bodyId. No type column (custom groups /
        # bodyId-only queries) -> one pooled bucket, which degenerates
        # gracefully to whole-enrollment coverage.
        mapping = {}
        if frame is None or 'bodyId' not in getattr(frame, 'columns', []):
            return mapping
        type_col = next((c for c in ('type', 'custom_group', 'group')
                         if c in frame.columns), None)
        for row_bid, row_type in zip(
                frame['bodyId'].astype(str),
                frame[type_col].astype(str) if type_col else ''):
            mapping.setdefault(str(row_type), set()).add(row_bid)
        return mapping

    return {
        'source': source_req,
        'target': target_req,
        'source_types': _type_map(getattr(fc, 'source_df', None)),
        'target_types': _type_map(getattr(fc, 'target_df', None)),
    }


def coverage_stop_met(config, reached_sources, reached_targets):
    """Both sides' per-type requirements met at this layer boundary?

    ``reached_sources``/``reached_targets`` are the bodyId sets connected
    so far (a source counts once it reaches >= 1 target; a target once
    >= 1 source reaches it). Returns ``(met, achieved)`` where
    ``achieved`` is the per-type reached/total census for disclosure.
    """
    import math

    def _side(req, type_map, reached):
        achieved = {}
        met = True
        for type_name, members in (type_map or {}).items():
            hit = len(members & reached)
            achieved[type_name] = f'{hit}/{len(members)}'
            if req is None:
                continue
            required = (1 if req == 0.0
                        else max(1, math.ceil(req * len(members))))
            if hit < required:
                met = False
        if not type_map and req is not None:
            # No enrollment information at all (missing frames): an
            # Any requirement falls back to "at least one bodyId
            # reached"; a fraction/Full requirement is UNVERIFIABLE and
            # must never satisfy the stop (refuse-on-doubt — a vacuous
            # met here would collapse the search to layer 1).
            met = bool(reached) if req == 0.0 else False
        return met, achieved

    source_met, source_ach = _side(
        config.get('source'), config.get('source_types'),
        reached_sources)
    target_met, target_ach = _side(
        config.get('target'), config.get('target_types'),
        reached_targets)
    achieved = {'source': source_ach, 'target': target_ach}
    return (source_met and target_met), achieved


# ---------------------------------------------------------------------------
# Compact per-target membership (plan §3.A)
# ---------------------------------------------------------------------------

class _TargetBitmaps:
    """Per-target seen-membership bitmaps over the union node index.

    One ``bytearray`` per target; node indexes come from a shared
    registry. ``n_nodes x n_targets`` BITS of resident state — at the
    measured L2 scale (44,225 union nodes x 242 targets) about 1.3 MB
    instead of the 9.3M-entry per-target distance dicts. Growth extends
    every bitmap in 4 KiB steps so repeated registration stays amortized.
    """

    _GROW_BYTES = 4096

    def __init__(self, n_targets):
        self._bitmaps = [bytearray() for _ in range(n_targets)]

    def ensure(self, n_bits):
        need = -(-n_bits // 8)  # ceil
        # round up in _GROW_BYTES steps
        need = ((need + self._GROW_BYTES - 1) // self._GROW_BYTES) \
            * self._GROW_BYTES
        if need <= 0:
            return
        for ba in self._bitmaps:
            if len(ba) < need:
                ba.extend(b'\x00' * (need - len(ba)))

    def set(self, target_idx, node_idx):
        ba = self._bitmaps[target_idx]
        ba[node_idx >> 3] |= 1 << (node_idx & 7)

    def has(self, target_idx, node_idx):
        ba = self._bitmaps[target_idx]
        byte = node_idx >> 3
        if byte >= len(ba):
            return False
        return bool((ba[byte] >> (node_idx & 7)) & 1)


class _ChunkedLabelWriter:
    """Buffered parquet writer for (target, node[, ...]) label rows.

    Rows are appended as tuples; ``flush(layer)`` writes the buffer to
    ``<kind>_L<layer>_c<chunk>.parquet`` under the kind's folder. Chunk
    boundaries are row-count based, so identical runs produce identical
    files (plan §6.4 determinism gate).
    """

    def __init__(self, store_dir, kind, schema):
        self.folder = Path(store_dir) / kind
        self.folder.mkdir(parents=True, exist_ok=True)
        self.kind = kind
        self.schema = schema
        self._rows = []
        self._chunk = 0
        self._layer = None
        self.files_written = 0

    def start_layer(self, layer):
        self._layer = int(layer)
        self._chunk = 0

    def append(self, row):
        self._rows.append(row)
        if len(self._rows) >= _CHUNK_ROWS:
            self._flush_chunk()

    def _flush_chunk(self):
        if not self._rows:
            return
        # Rows are written in arrival (scan) order — deliberate: arrival
        # order groups adjacent posts/predecessors, which parquet
        # compresses far better than a target-sorted layout (measured at
        # L4: dag_edges 383 MB arrival vs 1.3 GB target-sorted, for only
        # ~30% faster per-target reads via row-group pruning). Arrival
        # order is itself deterministic, so identical runs still produce
        # byte-identical stores.
        frame = pl.DataFrame(
            self._rows, schema=self.schema, orient='row')
        path = self.folder / (
            f'{self.kind}_L{self._layer}_c{self._chunk}.parquet')
        frame.write_parquet(path)
        self.files_written += 1
        self._chunk += 1
        self._rows = []

    def close(self):
        self._flush_chunk()


def _conn_layer_files(store_dir):
    folder = Path(store_dir) / CONNECTIONS_DIR
    if not folder.is_dir():
        return []
    out = []
    for path in folder.glob('connections_L*.parquet'):
        try:
            layer = int(path.stem.split('_L')[1])
        except (IndexError, ValueError):
            continue
        out.append((layer, path))
    return sorted(out)


def _label_files(store_dir, kind):
    """List a label kind's chunk files in (layer, chunk) order."""
    folder = Path(store_dir) / kind
    if not folder.is_dir():
        return []
    out = []
    for path in folder.glob(f'{kind}_L*.parquet'):
        stem = path.stem.split(f'{kind}_L')[1]
        try:
            layer_text, chunk_text = stem.split('_c')
            out.append((int(layer_text), int(chunk_text), path))
        except (IndexError, ValueError):
            continue
    return [item[2] for item in sorted(out)]


# ---------------------------------------------------------------------------
# Phase A — union-layer discovery + store write
# ---------------------------------------------------------------------------

def run_union_layer_discovery(fc, source_ID, target_ID, max_hops,
                              store_dir, coverage_stop=None):
    """Target-rooted backward discovery writing the labeled store.

    Behavioral twin of ``FindNeuronConnection._discover_shortest_backward``
    (same fetch sequence, same untyped filtering, same per-target labeling
    / DAG-edge rules, same early-stop and completeness semantics — plus
    the coverage early-stop when ``coverage_stop`` is supplied), with
    the per-target resident dicts replaced by registry-indexed bitmaps and
    the labels streamed to parquet. Returns the discovery metadata the
    pipeline and the batch phases consume (the monolithic return values
    that depend on per-target maps are produced later, from the store, by
    ``finalize_store_discovery``).
    """
    store_dir = Path(store_dir)
    # Store scoping (plan §4): the store is valid only for THIS run's
    # discovery. A rerun into the same folder (the ``saveas`` scripting
    # pattern) must never mix labels from an earlier, deeper run — a
    # stale ``node_distances_L4`` chunk beside a fresh ``L2`` one would
    # silently overwrite the fresh distances on read (the chunk files
    # merge per key, later layer wins). Wipe and rebuild.
    if store_dir.exists():
        shutil.rmtree(store_dir)
    (store_dir / NODE_DISTANCES_DIR).mkdir(parents=True, exist_ok=True)
    (store_dir / DAG_EDGES_DIR).mkdir(parents=True, exist_ok=True)
    (store_dir / CONNECTIONS_DIR).mkdir(parents=True, exist_ok=True)

    source_set = {str(value) for value in source_ID}
    # dict-deduped, order preserved — matches the monolithic per-target
    # dict keys when duplicate target ids are supplied.
    target_ids = list(dict.fromkeys(str(value) for value in target_ID))
    coverage_stop_record = None
    max_hops = max(1, int(max_hops))
    n_targets = len(target_ids)

    registry = {}
    node_names = []

    def register(node):
        idx = registry.get(node)
        if idx is None:
            idx = len(node_names)
            registry[node] = idx
            node_names.append(node)
            bitmaps.ensure(idx + 1)
        return idx

    dist_writer = _ChunkedLabelWriter(
        store_dir, NODE_DISTANCES_DIR,
        {'target': pl.Utf8, 'node': pl.Utf8, 'dist': pl.UInt32})
    dag_writer = _ChunkedLabelWriter(
        store_dir, DAG_EDGES_DIR,
        {'target': pl.Utf8, 'bodyId_pre': pl.Utf8, 'bodyId_post': pl.Utf8})

    bitmaps = _TargetBitmaps(n_targets)
    frontiers = [{register(target)} for target in target_ids]
    state_counts = [0] * n_targets
    found_sources = [set() for _ in range(n_targets)]
    source_label_rows = []  # (target, source, dist)
    frontier_sizes = []
    layer_count = 0  # reverse-layer slots incl. the depth-0 target layer
    discovery_complete = True

    # Targets are seen at distance 0 for themselves — the label row must
    # land in the store too (the monolithic distances_by_target always
    # carried the target's own 0 entry; the valid-edge pass and the
    # per-target state counts both rely on it).
    dist_writer.start_layer(0)
    dag_writer.start_layer(0)
    for ti, target in enumerate(target_ids):
        bitmaps.set(ti, registry[target])
        state_counts[ti] += 1
        dist_writer.append((target, target, 0))

    for reverse_depth in range(max_hops):
        frontier_idx = set()
        for frontier in frontiers:
            frontier_idx |= frontier
        if not frontier_idx:
            break
        frontier_posts = sorted(node_names[idx] for idx in frontier_idx)
        frontier_sizes.append(len(frontier_posts))

        fc._vprint(
            f'Backward layer {reverse_depth + 1}: querying incoming '
            f'connections to {len(frontier_posts):,} target-frontier '
            'neurons...',
            level='full',
        )
        conn_df = fc._fetch_path_connections_backward(
            frontier_posts, source_bodyIds=source_ID)

        layer_count += 1
        if conn_df is None or conn_df.empty:
            frontiers = [set() for _ in range(n_targets)]
            break
        conn_df = conn_df.copy()
        conn_df['bodyId_pre'] = conn_df['bodyId_pre'].astype(str)
        conn_df['bodyId_post'] = conn_df['bodyId_post'].astype(str)
        conn_df = conn_df[
            conn_df['bodyId_post'].isin(set(frontier_posts))
        ].copy()

        conn_df = fc._filter_untyped_pandas(
            conn_df, layer_label=f'{reverse_depth}->{reverse_depth + 1}')
        if conn_df.empty:
            frontiers = [set() for _ in range(n_targets)]
            break

        # The finalized per-layer frame IS the store's connections row
        # group for this layer — written verbatim (plan §3.A: keep the
        # conn_layer granularity; add_edge's cross-layer sums depend on
        # it downstream).
        conn_pl = fc._as_polars_conn_frame(conn_df).with_columns(
            pl.lit(f'{reverse_depth}->{reverse_depth + 1}').alias(
                'conn_layer'
            )
        )
        conn_pl.write_parquet(
            (store_dir / CONNECTIONS_DIR /
             f'connections_L{reverse_depth}.parquet'))

        # post -> targets whose current frontier contains it: one pass
        # over the fetched rows serves every target (the monolithic loop
        # re-filtered the frame once per target).
        post_map = {}
        for ti in range(n_targets):
            for idx in frontiers[ti]:
                post_map.setdefault(node_names[idx], []).append(ti)

        dist_writer.start_layer(reverse_depth + 1)
        dag_writer.start_layer(reverse_depth + 1)
        labeled_this_step = [set() for _ in range(n_targets)]
        next_frontiers = [set() for _ in range(n_targets)]

        for pre_str, post_str in zip(conn_df['bodyId_pre'].values,
                                      conn_df['bodyId_post'].values):
            target_list = post_map.get(post_str)
            if not target_list:
                continue
            pre_idx = register(pre_str)
            for ti in target_list:
                if bitmaps.has(ti, pre_idx):
                    # Already known for this target: still a DAG edge when
                    # the predecessor was labeled earlier in THIS layer's
                    # pass (its distance is exactly reverse_depth + 1).
                    if pre_idx in labeled_this_step[ti]:
                        dag_writer.append(
                            (target_ids[ti], pre_str, post_str))
                    continue
                bitmaps.set(ti, pre_idx)
                labeled_this_step[ti].add(pre_idx)
                next_frontiers[ti].add(pre_idx)
                state_counts[ti] += 1
                dist_writer.append(
                    (target_ids[ti], pre_str, reverse_depth + 1))
                dag_writer.append((target_ids[ti], pre_str, post_str))
                if pre_str in source_set:
                    found_sources[ti].add(pre_str)
                    source_label_rows.append(
                        (target_ids[ti], pre_str, reverse_depth + 1))

        for ti in range(n_targets):
            # Once every requested source has been encountered for this
            # target, deeper incoming branches cannot improve any of
            # those source-target distances; other targets continue.
            required = source_set - {target_ids[ti]}
            if required and found_sources[ti] >= required:
                next_frontiers[ti] = set()
            elif (coverage_stop is not None
                    and coverage_stop.get('target') is not None
                    and found_sources[ti]):
                # OPTIMIZATION (target-coverage regime): a target with
                # >=1 found source is COVERED and needs nothing more —
                # deeper layers could only find it MORE sources, the
                # exact computation the coverage scope cuts. Closing it
                # here shrinks the union frontier during straggler
                # windows (smaller backward fetches; the global stop
                # arrives no later). Its emitted pairs keep the sources
                # found as of this closure layer, exact per-pair hops.
                next_frontiers[ti] = set()

        frontiers = next_frontiers
        del conn_df
        gc.collect()

        # Coverage early-stop (shortest): at this layer boundary every
        # reached pair already carries its exact backward-BFS distance;
        # stopping here scopes the pair set without touching the
        # per-pair minimum-hop guarantee of the emitted pairs.
        if coverage_stop is not None:
            reached_sources = set()
            reached_targets = set()
            for ti in range(n_targets):
                if found_sources[ti]:
                    reached_targets.add(target_ids[ti])
                    reached_sources |= found_sources[ti]
            met, achieved = coverage_stop_met(
                coverage_stop, reached_sources, reached_targets)
            if met:
                coverage_stop_record = {
                    'stopped_at_layer': reverse_depth + 1,
                    'requirements': {
                        'source': coverage_stop.get('source'),
                        'target': coverage_stop.get('target'),
                    },
                    'achieved': achieved,
                    'route': 'store',
                }
                fc._vprint(
                    f'Coverage stop at backward layer {reverse_depth + 1}: '
                    'source/target coverage requirements met — deeper '
                    'pairs within the depth bound are not searched.',
                    level='full',
                )
                break

        if not any(frontiers):
            break
    else:
        if any(frontiers):
            discovery_complete = False

    dist_writer.close()
    dag_writer.close()

    per_target_source_distances = {}
    for target, source, dist in source_label_rows:
        per_target_source_distances.setdefault(target, []).append(dist)
    targets_found = []
    target_layers = {}
    target_hop_limits = {}
    for target in target_ids:
        distances = per_target_source_distances.get(target) or []
        if distances:
            # Per-pair contract (F-CORR-01): nearest source distance for
            # the enrollment layer, FARTHEST for the enumeration cutoff.
            target_layers[target] = min(distances)
            target_hop_limits[target] = max(distances)
            targets_found.append(target)

    meta = {
        'schema_version': STORE_SCHEMA_VERSION,
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'dataset': getattr(fc, 'dataset', None),
        'max_hops': max_hops,
        'source_bodyIds': sorted(source_set),
        'target_bodyIds': list(target_ids),
        'discovery_complete': discovery_complete,
        'frontier_sizes': frontier_sizes,
        'layer_count': layer_count + 1,  # + depth-0 target layer
        'conn_layer_files': len(_conn_layer_files(store_dir)),
        'per_target_distance_states': {
            target: state_counts[ti]
            for ti, target in enumerate(target_ids)
        },
        'targets_found': targets_found,
        'target_layers': target_layers,
        'target_hop_limits': target_hop_limits,
        'edge_filter_config': _edge_filter_config(fc),
        'coverage_stop': coverage_stop_record,
    }
    with open(store_dir / META_FILE, 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

    # pairs.parquet: every reachable (source, target) bodyId pair with its
    # own shortest distance; the per-pair DAG cap is the farthest source
    # distance per target (carried by target_hop_limits above too).
    if source_label_rows:
        pl.DataFrame(
            [(source, target, dist)
             for target, source, dist in source_label_rows],
            schema={'source': pl.Utf8, 'target': pl.Utf8,
                    'dist': pl.UInt32},
            orient='row',
        ).write_parquet(store_dir / PAIRS_FILE)

    return meta


def _edge_filter_config(fc):
    """The full edge-filter configuration the store is valid for.

    v1 stores are run-folder-local so this is audit-grade provenance, not
    a correctness gate (plan §3.A); min_synapse alone would understate
    what shaped the rows (type-level filters, untyped drop, hemisphere
    knobs, label mapping).
    """
    return {
        'filter_by': getattr(fc, 'filter_by', None),
        'min_synapse_num': getattr(fc, 'min_synapse_num', None),
        'min_ratio': getattr(fc, 'min_ratio', None),
        'min_traversal_probability':
            getattr(fc, 'min_traversal_probability', None),
        'aggregate_method': getattr(fc, 'aggregate_method', None),
        'exclude_intra_type_connections':
            getattr(fc, 'exclude_intra_type_connections', None),
        'drop_untyped': getattr(fc, 'drop_untyped', None),
        'separate_hemispheres': getattr(fc, 'separate_hemispheres', None),
        'hemisphere_filter': getattr(fc, 'hemisphere_filter', None),
        'keep_only_hemisphere_conserved_connections':
            getattr(fc,
                    'keep_only_hemisphere_conserved_connections', None),
        'label_mapper': type(fc.label_mapper).__name__
        if getattr(fc, 'label_mapper', None) else None,
    }


# ---------------------------------------------------------------------------
# Store finalization (valid-edge pass + frame/layer reconstruction)
# ---------------------------------------------------------------------------

def load_store_meta(store_dir):
    with open(Path(store_dir) / META_FILE, encoding='utf-8') as handle:
        return json.load(handle)


def load_target_distances(store_dir, targets):
    """Rehydrate ``{target: {node: dist}}`` for the given targets."""
    wanted = list(targets)
    if not wanted:
        return {}
    result = {target: {} for target in wanted}
    files = _label_files(store_dir, NODE_DISTANCES_DIR)
    if not files:
        return result
    frame = pl.scan_parquet([str(p) for p in files]).filter(
        pl.col('target').is_in(wanted)).collect()
    for target, node, dist in frame.iter_rows():
        result[target][node] = dist
    return result


def load_target_dag_edges(store_dir, targets):
    """Rehydrate ``{target: [(pre, post), ...]}`` for the given targets."""
    wanted = list(targets)
    if not wanted:
        return {}
    result = {target: [] for target in wanted}
    files = _label_files(store_dir, DAG_EDGES_DIR)
    if not files:
        return result
    frame = pl.scan_parquet([str(p) for p in files]).filter(
        pl.col('target').is_in(wanted)).collect()
    for target, pre, post in frame.iter_rows():
        result[target].append((pre, post))
    return result


def load_edge_weight_frame(store_dir):
    """Compact polars frame of ``{(pre, post): summed weight}``.

    The sum runs across ``conn_layer`` files exactly like the monolithic
    per-layer ``build_from_dataframe`` calls (add_edge sums duplicate
    pairs across layers), so batch-graph weights match the monolithic
    ``FastGraph.adj`` bit for bit — including the cross-layer multiplicity
    of a post that sits at two reverse depths for different targets.
    Kept as one polars row per unique pair (hundreds of MB at L4 scale,
    versus gigabytes for an equivalent Python dict); per-batch slices
    come from ``weights_for_pairs``.
    """
    scans = [
        pl.scan_parquet(path).select(
            ['bodyId_pre', 'bodyId_post', 'weight'])
        for _layer, path in _conn_layer_files(store_dir)
    ]
    if not scans:
        return pl.DataFrame(schema={'bodyId_pre': pl.Utf8,
                                    'bodyId_post': pl.Utf8,
                                    'weight': pl.Int64})
    return (pl.concat(scans)
            .group_by(['bodyId_pre', 'bodyId_post'])
            .agg(pl.col('weight').sum().alias('weight'))
            .sort(['bodyId_pre', 'bodyId_post'])
            .collect())


def weights_for_pairs(weight_frame, pairs):
    """``{(pre, post): weight}`` for just the requested edge pairs.

    A semi-join slice of the compact weight frame — the per-batch
    replacement for the former whole-store Python dict, so enumeration
    memory no longer carries every pair of the union graph.
    """
    wanted = sorted(set(pairs))
    if not wanted:
        return {}
    lookup = pl.DataFrame(
        wanted, schema={'bodyId_pre': pl.Utf8, 'bodyId_post': pl.Utf8},
        orient='row')
    sliced = weight_frame.join(
        lookup, on=['bodyId_pre', 'bodyId_post'], how='semi')
    return {(pre, post): weight
            for pre, post, weight in sliced.iter_rows()}


def finalize_store_discovery(store_dir, meta, source_ID):
    """Valid-edge/backreach pass + discovery-result reconstruction.

    Port of the monolithic post-BFS pass of
    ``_discover_shortest_backward``: for every discovered target, only
    edges on a shortest-DAG branch from a requested source to that target
    are valid; everything else is dropped from the returned connection
    tables so uninvolved branches stay out of the graph and the
    visualization. Runs one target at a time (batch-of-1 memory) from
    store slices — the per-target maps are never all resident.

    Returns the monolithic-shaped discovery result: filtered per-layer
    connection frames, filtered reverse layers, valid node set, and the
    DAG census for the run diagnostics.
    """
    source_set = {str(value) for value in source_ID}
    targets_found = meta['targets_found']
    target_hop_limits = meta['target_hop_limits']

    valid_edges = set()
    valid_nodes = set()
    for target in targets_found:
        t_dist = load_target_distances(store_dir, [target])[target]
        dag_edges = load_target_dag_edges(store_dir, [target])[target]
        if not dag_edges:
            continue
        target_hop_limit = target_hop_limits[target]
        reverse_predecessors = {}
        for pre, post in dag_edges:
            if (t_dist.get(pre, target_hop_limit + 1) <= target_hop_limit
                    and t_dist.get(post, target_hop_limit + 1)
                    <= target_hop_limit):
                reverse_predecessors.setdefault(post, set()).add(pre)

        source_nodes = {
            source for source in source_set
            if source != target
            and t_dist.get(source, target_hop_limit + 1) <= target_hop_limit
        }
        keep_nodes = set(source_nodes)
        for node, node_distance in sorted(
                t_dist.items(),
                key=lambda item: item[1], reverse=True):
            if node_distance > target_hop_limit:
                continue
            if node in keep_nodes:
                continue
            if any(
                    predecessor in keep_nodes
                    for predecessor in reverse_predecessors.get(node, ())):
                keep_nodes.add(node)

        if target in keep_nodes:
            valid_nodes.update(keep_nodes)
            for post, predecessors in reverse_predecessors.items():
                if post not in keep_nodes:
                    continue
                for pre in predecessors:
                    if pre in keep_nodes:
                        valid_edges.add((pre, post))

    valid_nodes.update(targets_found)

    # Filtered per-layer frames: same positional structure as the
    # monolithic all_connections list (one slot per reverse layer,
    # empties included) so _graph_edge_frames sees an identical input.
    all_connections = []
    if valid_edges:
        edge_frame = pl.DataFrame(
            list(valid_edges), schema=['bodyId_pre', 'bodyId_post'],
            orient='row')
        conn_files = dict(_conn_layer_files(store_dir))
        for layer in range(max(meta['layer_count'] - 1, 0)):
            path = conn_files.get(layer)
            if path is None:
                all_connections.append(pl.DataFrame())
                continue
            frame = pl.read_parquet(path)
            all_connections.append(frame.join(
                edge_frame, on=['bodyId_pre', 'bodyId_post'], how='semi'))
    else:
        conn_files = _conn_layer_files(store_dir)
        all_connections = [
            pl.read_parquet(path).clear() if path.exists() else pl.DataFrame()
            for _layer, path in conn_files
        ]
        if not all_connections:
            all_connections = [pl.DataFrame()]

    # Filtered reverse layers: one slot per discovery iteration exactly
    # like the monolithic reverse_layers list (layer 0 = targets, layer
    # d = nodes labeled at depth d for any target; a depth whose fetch
    # produced no NEW labels yields an empty slot, not a missing one).
    # Backward mode has no positional consumer today, but the return
    # contract stays byte-compatible with the monolithic shape.
    layer_neurons = [set(meta['target_bodyIds']) & valid_nodes]
    label_files = _label_files(store_dir, NODE_DISTANCES_DIR)
    by_dist = {}
    if label_files:
        frame = pl.scan_parquet([str(p) for p in label_files]).select(
            ['node', 'dist']).collect()
        for dist in frame['dist'].unique().to_list():
            by_dist[int(dist)] = set(
                frame.filter(pl.col('dist') == dist)['node'].to_list())
    for depth in range(1, meta['layer_count']):
        layer_neurons.append(by_dist.get(depth, set()) & valid_nodes)

    return {
        'all_connections': all_connections,
        'layer_neurons': layer_neurons,
        'all_neurons_in_network': valid_nodes,
        'targets_found': targets_found,
        'target_layers': meta['target_layers'],
        'target_hop_limits': target_hop_limits,
        'complete': meta['discovery_complete'],
        'dag_nodes': len(valid_nodes),
        'dag_edges': len(valid_edges),
    }


# ---------------------------------------------------------------------------
# Phase B — batch composition + batched enumeration
# ---------------------------------------------------------------------------

def compose_batches(targets, per_target_states, budget=0, fixed_size=0):
    """Compose contiguous target batches over ``sorted(targets, key=str)``.

    The enumerator orders per-target streams by ``sorted(targets, key=str)``
    (``find_paths_shortest_strongest_first``); batches must be contiguous
    runs of that SAME order so the nested heapq.merge of the batch
    emissions reproduces the flat monolithic tie-breaking (plan §3.B).
    With ``fixed_size > 0`` batches hold exactly that many targets;
    otherwise ``budget`` accumulates targets until the next one would
    push the batch's summed distance-state count past it — a single
    target whose own count exceeds the budget still becomes its own
    batch (the budget is advisory per batch, never a target splitter).
    """
    ordered = sorted(targets, key=str)
    batches = []
    current, current_states = [], 0
    for target in ordered:
        states = int(per_target_states.get(target, 0))
        if fixed_size > 0:
            if len(current) >= fixed_size:
                batches.append(current)
                current, current_states = [], 0
        else:
            if current and current_states + states > budget:
                batches.append(current)
                current, current_states = [], 0
        current.append(target)
        current_states += states
    if current:
        batches.append(current)
    return batches


def batch_records(batches, per_target_states):
    """Provenance rows: one entry per batch (targets, state sum)."""
    return [
        {
            'batch_index': index,
            'targets': list(batch),
            'target_count': len(batch),
            'distance_states': sum(
                int(per_target_states.get(target, 0)) for target in batch),
        }
        for index, batch in enumerate(batches)
    ]


def batched_shortest_paths(fc, store_dir, meta, sources, cutoff, budget,
                           stats=None, batch_states=None, fixed_size=0,
                           progress=None):
    """Budgeted batched enumeration over the discovery store.

    Per batch: slice the store's DAG edges + distance maps, build a
    standalone ``FastGraph`` whose edge weights are the per-layer-summed
    store weights (monolithic parity, see ``load_edge_weight_frame``),
    enumerate with ``budget=None, payload=True``, then free the graph.
    The batch emissions are merged in batch order and the global budget /
    tau tie-drain applies exactly once — via the same shared drain the
    monolithic enumerator uses — so stats and emission order are
    identical to a monolithic run (plan §3.B).
    """
    targets = meta['targets_found']
    per_target_states = meta['per_target_distance_states']
    hop_limits = meta['target_hop_limits']
    batches = compose_batches(
        targets, per_target_states, budget=batch_states, fixed_size=fixed_size)
    fc._shortest_batch_records = batch_records(batches, per_target_states)

    # Deferred import: see the module header (coana's lazy-import
    # contract without the vispath subproject).
    from vispath_pkg.fast_graph_core import (
        FastGraph,
        merge_shortest_batch_emissions,
    )

    weight_frame = None
    emissions = []
    for batch in batches:
        dag_edges = load_target_dag_edges(store_dir, batch)
        seeds = load_target_distances(store_dir, batch)
        if weight_frame is None:
            weight_frame = load_edge_weight_frame(store_dir)

        graph = FastGraph()
        batch_pairs = set()
        for edges in dag_edges.values():
            batch_pairs.update(edges)
        batch_weights = weights_for_pairs(weight_frame, batch_pairs)
        for pre, post in sorted(batch_pairs):
            weight = batch_weights.get((pre, post))
            if weight is not None:
                graph.add_edge(pre, post, weight)

        payload = list(graph.find_paths_shortest_strongest_first(
            batch, list(sources), cutoff,
            budget=None, payload=True,
            target_cutoffs={target: hop_limits[target]
                            for target in batch},
            target_distances=seeds,
        ))
        del graph
        gc.collect()
        emissions.append(payload)
        if progress is not None:
            progress(len(emissions), len(batches))

    yield from merge_shortest_batch_emissions(emissions, budget, stats)


# ---------------------------------------------------------------------------
# Store retention (plan-shortest-store-retention)
# ---------------------------------------------------------------------------

def _dir_size_bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def _compact_connections(store_dir):
    """Merge the per-layer connection frames into one 4-column file.

    Rows keep their exact (layer, chunk, row) order, so the cross-layer
    (pre, post, conn_layer) multiplicity that the graph's add_edge sums
    are built from survives verbatim; only the fetch-dependent label /
    roi / ratio columns are dropped (re-derivable solely by re-running
    discovery).
    """
    frames = []
    for _layer, path in _conn_layer_files(store_dir):
        frames.append(pl.read_parquet(
            path,
            columns=['bodyId_pre', 'bodyId_post', 'weight', 'conn_layer']))
    folder = Path(store_dir) / CONNECTIONS_DIR
    if not frames:
        return 0
    merged = pl.concat(frames)
    merged.write_parquet(folder / 'connections_all.parquet')
    for frame_path in folder.glob('connections_L*.parquet'):
        frame_path.unlink()
    return merged.height


def apply_store_retention(fc):
    """Post-run keep/compact/prune of the run's discovery store.

    Runs only after a successful batched shortest run (the caller hooks
    this between materialization and the final metadata re-stamp), so no
    artifact the run exports depends on the store afterwards. ``keep`` is
    the pinned default; ``compact`` merges the connection layers into one
    4-column file and deletes the (derivable) dag_edges chunks; ``prune``
    removes every data file and keeps meta.json as the census. The mode,
    sizes, and the compact dag-derivation recipe land in meta.json, in
    ``shortest_discovery_diagnostics.batching.store_retention``, and in a
    user-warning note.
    """
    store_dir = getattr(fc, '_shortest_store_dir', None)
    if not store_dir:
        return
    mode = str(getattr(fc, 'discovery_store_retention', 'keep')
               or 'keep').strip().lower()
    if mode not in STORE_RETENTION_MODES:
        fc._warn_notes.append(
            f'- [discovery store] unknown discovery_store_retention '
            f'{mode!r}; kept the store unchanged.')
        mode = 'keep'
    if mode == 'keep':
        return

    store_dir = Path(store_dir)
    meta_path = store_dir / META_FILE
    meta = json.loads(meta_path.read_text(encoding='utf-8')) \
        if meta_path.exists() else {}
    sizes_before = {
        kind: _dir_size_bytes(store_dir / kind)
        for kind in (CONNECTIONS_DIR, DAG_EDGES_DIR, NODE_DISTANCES_DIR,
                     'total')
    }
    sizes_before['total'] = _dir_size_bytes(store_dir)

    removed = []
    if mode == 'compact':
        rows = _compact_connections(store_dir)
        shutil.rmtree(store_dir / DAG_EDGES_DIR, ignore_errors=True)
        removed = ['dag_edges/ (derivable, see meta.retention.recipe)',
                   'connections label columns (merged to '
                   'connections_all.parquet, 4 columns, '
                   f'{rows:,} rows)']
    else:  # prune
        for kind in (CONNECTIONS_DIR, DAG_EDGES_DIR, NODE_DISTANCES_DIR):
            shutil.rmtree(store_dir / kind, ignore_errors=True)
        (store_dir / PAIRS_FILE).unlink(missing_ok=True)
        removed = ['connections/', 'dag_edges/', 'node_distances/',
                   'pairs.parquet']

    meta['retention'] = {
        'mode': mode,
        'applied': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'store_bytes_before': sizes_before['total'],
        'store_bytes_after': _dir_size_bytes(store_dir),
        'kind_bytes_before': {
            kind: size for kind, size in sizes_before.items()
            if kind != 'total'},
        'removed': removed,
        'recipe': (
            'dag_edges rows are re-derivable offline: for every '
            '(target, pre, post) with dist_target[pre] == '
            'dist_target[post] + 1 over node_distances joined to the '
            'compact connections on (pre, post).'
        ) if mode == 'compact' else None,
    }
    with open(meta_path, 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

    batching = getattr(fc, 'shortest_discovery_diagnostics',
                       {}).setdefault('batching', {})
    batching['store_retention'] = dict(meta['retention'])

    fc._warn_notes.append(
        f'- [discovery store] discovery_store_retention={mode!r}: '
        f'removed {"; ".join(removed)}. Kept: '
        + ('node_distances/, pairs.parquet, meta.json.'
           if mode == 'compact' else 'meta.json only (size census).')
        + ' Re-run the query to regenerate the full store.')
    fc._vprint(
        f'Discovery store {mode}: '
        f'{sizes_before["total"]:,} -> {_dir_size_bytes(store_dir):,} '
        f'bytes.', level='full')
