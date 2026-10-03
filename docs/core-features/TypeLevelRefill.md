# Type-Level Connection-Strength Refill

Refill records for pathfinding runs whose **applied threshold exceeded
the asked threshold** because the Edge Budget floored the discovery cone
and/or the StrongestFirst path budget bit — full-path mode since Phase 1,
**automatically generated in-run since Phase 2** (`auto_type_level_refill`,
default on), and **shortest mode since Phase 3** (via the run's
`shortest_discovery_store/`).
Plan of record: `_plan/plan-type-level-refill.md`.

## When it applies

A run folder is refillable when its provenance block
(`parameters.txt` / `all_attributes.json`) shows
`applied_threshold > requested_threshold` — i.e.
`applied_threshold_source` names `edge_budget`, `strongest_first_budget`,
or both. Complete runs (`applied == requested`) are gated off and get a
`no_refill_needed` provenance record only. Multi-threshold replay folders
are evaluated per slice: a canon slice's true asked threshold is read
from its `replayed_from` line (the stamped `requested_threshold` carries
the canonical value), and W4 re-enumerated slices carry their own fresh
provenance.

## Semantics

The exported `data_details/connection_type.csv` counts only bodyId pairs
on **enumerated paths** — budget-cut pairs are missing from existing rows.
The refill adds back exactly that mass:

- **Refill only, no recovery** — only type pairs that already have an
  emitted row; other pairs appear in the provenance census only.
- **Path-supporting strength only** — the refill re-runs pathfinding on
  the subgraph induced by all bodyIds of the involved types, at the asked
  threshold, no budgets, toward **all enrolled (Checked) targets**.
  Hop-pruned-only edges (no source→target path within the run's bound)
  are excluded by a lossless prefilter before enumeration.
- **Exact emitted-set reconstruction** — the emitted set is re-derived by
  enumerating that induced graph at the run's effective cut (the floor
  threshold w0, then the recorded StrongestFirst budget with the
  production tau tie-drain). This catches floor-boundary edges whose
  weight ≥ w0 but whose only path died with the floor. The re-derivation
  must reproduce the exported table exactly (`table_reproduced`) — on
  any mismatch the refill **refuses** rather than emit wrong numbers.
- **Certified lower bound** — asked-threshold paths that leave the
  involved-type subgraph cannot be seen; the provenance discloses the
  conservatism gap instead of silently closing it.

Ratios use the threshold-free all-post denominators (F9,
`min_weight=1`); `refilled_traversal_probability` folds the union pair
set with the run's `aggregate method` (`product` by default).

## Usage

**Automatic (Phase 2):** every materialized FindAllPath / FindShortestPath
folder with `applied > asked` gets the records in
`data_details/type_level_refill/` right after the run completes
(`auto_type_level_refill=False` disables; a `user_warning_notes.txt` line
discloses what was generated or why it was refused; the pair report's
provenance card shows the totals). Complete runs produce nothing.

**CLI (post-hoc):**

```bash
python scripts/TypeLevelRefill.py <run_folder> \
    --connections <dataset connections.parquet|csv> \
    --neuron-table <*_allneurons_neuron_df.csv> \
    [--out DIR | --in-run] [--detail-cap N] [--path-budget N]
```

The connection source must be the FULL dataset connections at natural
weights; the neuron table the same one the run used. Labels are resolved
exactly as the run's type-level aggregation did: a recorded custom
mapping file is loaded through `LabelMapper` (bodyId-level mapping wins
over type-level, unmapped labels keep their raw names), hemisphere
suffixes apply when the run separated hemispheres, untyped ids drop. A
recorded mapping file that no longer exists refuses the run rather than
guess. Output goes to `--out` (default `./type_level_refill_output/<run>/`)
— run folders are never modified. `--in-run` instead writes the records
into `<run>/data_details/type_level_refill/` (still additive only; the
run guide surfaces them when present).

**Shortest mode (Phase 3):** detected from the folder name; the refill
reads the run's `shortest_discovery_store/` (per-layer connection frames
+ per-target distance maps) for structure and the connection source only
for the F9 denominators. The cut re-derivation runs the SHORTEST
strongest-first enumerator with store-seeded distances, so per-pair
min-hop semantics and the tau tie-drain match the run. When the store was
pruned by retention, the refill refuses with a clear message.

In-process: `from type_level_refill import compute_type_level_refill`,
passing `edges=[(pre, post, weight), ...]` and the effective
`type_map={bodyId: label}` (or build it with
`build_effective_type_map(frame, dataset=..., mapping_file=...,
separate_hemispheres=...)`).

## Outputs

| File | Content |
|---|---|
| `refill_type_pairs.csv` | One row per emitted type pair (zeros included): `emitted_weight`, `refill_weight`, `refilled_total`, `emitted_pair_count`, `refill_pair_count`, `refilled_connection_ratio`, `refilled_traversal_probability`, `split_status`. Never capped. |
| `refill_bodyId_pairs.csv` | The refilled bodyId pairs with `traversal_count` and hop positions; capped by `--detail-cap` (default 50,000, deterministic weight-desc order). |
| `refill_provenance.json` | Thresholds/budgets (incl. the run's `graph_pruning_record`), induced-graph sizes, `table_reproduced`, `refill_truncated`, and the disclosure counters: `skipped_pairs_not_emitted_type_pair` (per-type-pair census + capped examples), `excluded_edges_non_involved_endpoint`, `boundary_refill_edge_count/weight/examples_at_or_above_w0`. |
| `README.md` | Semantics + join recipe (join on `(type_pre, type_post)` against `connection_type.csv` summed across its `conn_layer` rows). |

## Real-data verification (FAFB v783, aMe12 → PPL101, L2, asked 3)

- Edge Budget 3,000 → floor w0 = 5: refill 3,810 bodyId edges / 16,282
  synapses. Against a no-budget control run at the asked threshold, 222 of
  226 emitted type pairs land EXACTLY on the control's weights; the 4
  remaining pairs carry the disclosed conservatism gap (287 of 39,073
  synapses, 0.7% — asked-threshold paths leaving the involved-type
  subgraph). The boundary phenomenon is real at scale: 877 of the 3,810
  refilled edges (23%) have weight ≥ w0 — they survived the floored cone
  but their paths died with the floor; a naive weight-split would have
  missed them all.
- StrongestFirst budget 200 → τ = 16: refill 436 edges / 6,318 synapses;
  all 29 emitted type pairs land EXACTLY on the control's weights.
- `table_reproduced` held in every real and synthetic case — the
  label/hemisphere/filter reconstruction is verified by construction.

## Scope and refusals

- Full-path ('all') mode only — shortest mode is Phase 3.
- `filter_by='type'` runs are refused (type-level filtering changes the
  asked-threshold graph).
- Any cut re-derivation that fails to reproduce the exported table
  refuses with a diagnostic (mismatched labels/hemisphere suffixes/
  filters or a stale connection source).

## Truncation guard

The asked-threshold re-exploration enumerates simple paths on the induced
subgraph with its own guard (`refill_path_budget`, default 2,000,000
paths; `--path-budget` on the CLI). Very broad queries (e.g. all R-cells
at depth 5) can exceed it — the provenance then carries
`refill_truncated: true` and the refill is an honest partial set. Raise
the guard for a full refill; the cut enumeration that reproduces the
exported table is never guarded.


## Ratio-basis runs (Phase 3)

`weight_basis='connection_ratio'` runs refill like synapse runs: thresholds parse as FLOATS, the re-enumeration runs on ratio weights, and the records carry SYNPASE recovery masses (dual-basis: `refill_weight` is synapse mass, `refilled_connection_ratio` the F9 ratio). Streamed edge sources are materialized once (the ratio lane needs two passes). The post-hoc CLI works on ratio run folders unchanged.
