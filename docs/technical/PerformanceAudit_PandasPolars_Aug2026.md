# Performance & Pandas/Polars Interop Audit — August 2026

> **Drift note (2026-09-25):** the separate `statvis_polars` module named
> below was merged back into `src/statvis.py` (commit b9b03027) —
> `EnrichConnectionTablePolars`, the mtime-keyed `_NEURON_DF_CACHE`, and
> `process_paths_streaming` / `process_batch_polars` all live in
> `statvis.py` now. The measurements remain a valid August 2026 record.

## Summary

An audit of the longest-running code paths found that most of the wall-clock
time in pathfinding, enrichment and filtering comes from a small set of
repeated, row-wise Python operations and repeated disk I/O.  All five
`FastGraph` pathfinding algorithms are already vectorized/streamed; the slow
parts are the orchestration around them.

This document lists what was found, what was fixed, and what remains a
known scaling limit.

## Biggest offender: type-level incoming weights

`_fetch_total_incoming_weight_by_type()` (called once per network layer by
`FindAllPath`/`FindDirectConnections` type-mode filtering) previously:

1. Re-read `neuron_index.parquet` and rebuilt a `bodyId -> type` Python dict
   on **every** call;
2. Scanned the **entire** `connections.parquet` in row-group chunks;
3. Aggregated with a **Python loop over every connection row**
   (`for bid, weight in zip(...)`).

On a synthetic 1M-row dataset the old code took ~80 ms per call; on real
datasets (tens of millions of rows) this is minutes per call, multiplied by
the number of layers.

**Fix:** the full-dataset aggregate `type_post -> total_incoming_weight` is
independent of the requested post types, so it is now computed **once** per
(data source, `min_weight`) with fully vectorized Polars joins/group-by
(`_get_total_incoming_by_type_table()`), cached on the instance, and each
per-layer call only filters the cached table.

Measured on 1M rows: first vectorized call ~23 ms, every subsequent call
~0.5 ms (≈150× faster for repeated calls, 3.5× faster even for the first).

The bodyId-level sibling `_fetch_total_incoming_weight()` got the same
full-table caching treatment.

## Repeated full-neuron-table disk I/O

Three different code paths re-read the same
`*_allneurons_neuron_df.csv` (often hundreds of MB) for every call:

- `coana._fetch_from_dataset_or_api()` / `_fetch_neurons_by_types()`
  (called per layer for neuron metadata);
- `statvis_polars.EnrichConnectionTablePolars()` (called per layer);
- legacy `statvis.EnrichConnectionTable()` (pandas, called by legacy
  `FindPath`).

**Fix:** per-instance cache in `coana` and a small module-level
mtime-keyed cache in `statvis_polars` (max 4 entries).  The table is
read-only inside the enrichment pipeline, and the mtime in the cache key
keeps results correct if a dataset file is regenerated.

## O(M×N) label-map expansion

`build_bodyid_label_map()` tested each mapping ID with a full
`neuron_df.filter(bodyId == id)` scan, i.e. O(M mappings × N neurons) per
call, inside the per-layer enrichment.

**Fix:** precompute the `bodyId` set once and use O(1) membership checks
(O(M + N)).

## Row-wise `map_elements` with a Python dict

The `std_label` global-ratio branch in
`EnrichConnectionTablePolars.aggregate_connections()` mapped every
`type_post` through a Python lambda/dict.

**Fix:** replaced with a vectorized Polars left-join on the global weights
table.

## Row-wise pandas `apply(axis=1)` in hot filters

These were replaced with vectorized NumPy/Pandas operations:

- `_apply_hemisphere_suffix_to_conn_df()` / `_apply_hemisphere_suffix_to_neuron_df()`
  — four `apply(axis=1)` calls per fetched connection table replaced with
  vectorized series logic (`_hemi_code_series` / `_append_hemi_suffix_series`),
  preserving the exact old precedence semantics (first non-null column wins,
  even when it cannot be normalized).
- `_apply_bodyid_level_filters()` / `_apply_type_level_filters()` — ratio
  calculation via `np.divide(..., where=valid)` instead of per-row lambdas;
- `_apply_type_level_filters()` — type-pair membership via
  `pd.MultiIndex.isin()` instead of `apply(axis=1)`;
- legacy `statvis.EnrichConnectionTable()` — the full-neuron-table label
  mapping now performs one `get_label()` call per unique bodyId/type instead
  of two calls per row via `apply(axis=1)`; the ratio lambda was vectorized
  too.

## Pandas/Polars interop conflicts found and fixed

- `_count_cached_connections()` called `.empty` on `_conn_df_cache`, which
  is a Polars DataFrame — `AttributeError` on the in-memory path.  Now uses
  `_is_empty_df()` (works for both).
- The module-level shared cache (`_FNC_CACHE`) stores the connection table
  as **pandas** (other modules expect pandas), so the new vectorized
  helpers convert pandas -> Polars transparently when the in-memory cache is
  used.
- `_NEURON_DF_CACHE` / instance neuron-cache keys include the file mtime so
  a regenerated dataset never serves stale rows.

## Known scaling limits (recommendations, not changed)

- **Enumerating ALL simple paths is exponential in the worst case.**
  `FindAllPath` collects every path in `all_paths` before building the path
  tables.  For very dense connectomes with many layers this list can reach
  millions of entries and dominate memory.  Recommended next step: add an
  optional `max_paths` cap and/or stream paths into `process_batch_polars`
  in chunks (the chunked writer `process_paths_streaming` already exists in
  `statvis_polars`).
- `comparison_analyzer` still contains several `iterrows()` loops over
  edge/presence tables in the cross-dataset report path; those tables are
  typically small-to-medium, but the two BFS dead-end trims were already
  switched from `list.pop(0)` (O(n²)) to `deque.popleft()`.
- `neuronbridge_finder` applies per-row functions in display/aggregation
  code; worth profiling per use case before vectorizing, since input sizes
  vary widely.

## Validation

New regression suite: `tests/core/test_performance_fixes.py` (14 tests
at audit time; 15 as of 2026-09-25 — one test was added afterwards)
covers the vectorized incoming-weight aggregates against a reference
implementation, duplicate-ID semantics, label-map correctness, global-ratio
aggregation, hemisphere vectorization parity with the old scalar logic,
type-level ratio/pair filtering, the `.empty` conflict, and both neuron-CSV
caches.  Full pytest: 69 passed; the only failure is the known
sandbox-localhost restriction in `TestHTTPServer`.
