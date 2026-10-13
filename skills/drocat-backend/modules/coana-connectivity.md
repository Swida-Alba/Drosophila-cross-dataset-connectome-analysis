# coana — FindNeuronConnection

The core connectome query engine (module `src/coana.py`). All pathfinding and
direct-edge analysis runs through the single class `FindNeuronConnection`. This is
the most flexible building block: one instance can discover paths, direct edges,
shortest paths, a mutual-connection network, and connection caches.

## Constructor (key params)

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    dataset="male-cns:v0.9",            # or hemibrain:v1.2.1, flywire_FAFB_v783...
    sourceNeurons=["aMe12"],            # types / bodyIds / instances / regex
    targetNeurons=["PPL101"],
    output_dir="/abs/output/paths",
    min_synapse_num=3,
    min_ratio=0.0,
    min_traversal_probability=0.0,
    max_interlayer=2,
    filter_by="bodyId",                 # or "type"
    pathfinding="StrongestFirst",       # built-in default; DP/MemoizedDFS/DFS = unbounded complete runs (API)
    max_paths_bodyid=0,                 # auto -> internal 1,000,000 path budget (tau reported when it bites)
    graph_edge_limit_bodyid=0,          # Edge Budget off; set ~1_000_000 to cap the discovery cone
    visualize_before_reconstruct=False,
    search_columns="auto",              # "auto" | "type" | "instance" | "bodyId"
    network_layout="distributed",       # spring, circular, hierarchical
    use_cache=True,
    edgeN_limit=500,
    output_format="csv",                # or "xlsx"
    skip_bodyId=True,
    showfig=False,
    custom_source_name="",
    custom_target_name="",
    keyword_in_path_to_remove=["None"],
    cache_only=False,
    saveas="",
    separate_hemispheres=False,
    hemisphere_filter="all",
    keep_only_hemisphere_conserved_connections=False,
    symmetry_analysis=False,
    find_reciprocal=False,
    custom_mapping_file=None,           # custom grouping/mapping JSON
)
```

## Methods

| Method | Purpose |
| --- | --- |
| `InitializeNeuronInfo()` | Resolve source/target queries into `source_df`/`target_df`; ensures dataset metadata + neuron index. Call before a query. |
| `FindDirectConnections()` | Direct one-hop edges between source and target (`filter_by` applies). |
| `FindPath(find_bodyId_path=None)` | A single pathfinding strategy. |
| `FindAllPath(find_bodyId_path=True, forward_only=True, exclude_searched_neurons=None, use_graph_cache=True, find_reciprocal=False)` | All paths within `max_interlayer`. Recommended reproducible first pass. |
| `FindShortestPath(find_bodyId_path=True, forward_only=True, exclude_searched_neurons=None, use_graph_cache=True, find_reciprocal=False)` | Minimum hop-count paths per reachable pair. |
| `FindNetwork()` | Mutual direct-connection network (source == target == queried set). |
| `build_connection_cache(neuron_types=None, neuron_bodyIds=None, batch_size=100, force_rebuild=False, quiet=False, progress_callback=None, cancel_event=None, max_workers=None, status_callback=None)` | Build the connection cache for the dataset. |
| `build_connectivity_profile_cache(neuron_types=None, top_k=10, top_m=5, expand_2hop=True, max_neurons=None, force_refresh=False, progress_callback=None)` | Build the connectivity-profile cache. |

## Typical compositions

```python
fc = FindNeuronConnection(dataset="male-cns:v0.9", sourceNeurons=["aMe12"],
                          targetNeurons=["PPL101"], output_dir="/abs/output/paths",
                          max_interlayer=2, filter_by="bodyId", skip_bodyId=True,
                          output_format="csv", use_cache=True, showfig=False)
fc.InitializeNeuronInfo()
fc.FindAllPath(forward_only=True, find_reciprocal=False)

# direct edges in a second pass
fc2 = FindNeuronConnection(dataset="male-cns:v0.9", sourceNeurons=["aMe12"],
                           targetNeurons=["PPL101"], output_dir="/abs/output/edges",
                           min_synapse_num=3, output_format="csv", showfig=False)
fc2.InitializeNeuronInfo()
fc2.FindDirectConnections()
```

## Type-level refill (auto hook)

DUAL REFILL (2026-10-08) — two kinds, both written by
`_maybe_generate_type_level_refill` without touching run files:

- **refill-asked** — re-runs path building at the ASKED threshold within the
  involved-types induced subgraph. Gate: synapse-basis runs need the applied
  threshold to exceed the asked one (Edge Budget floor and/or
  StrongestFirst bite); ratio-basis runs ALWAYS fill (non-bitten ones get
  the `complete_at_asked` view with mass 0).
RATIO DOCTRINE (chain-derivation principle, 2026-10-10): every ratio
rendered on a chain row is derived from THAT chain — numerator = the
realized hop mass (the bodyId edges backing that hop of that chain),
denominator = the dataset-wide incoming (all inputs plain; >= threshold
adjusted). The refill exports `type_incoming_asked.csv` (its
>=threshold incoming per type) and the report's refill-asked Adj
divides the chains' realized masses by it; the FULL scope omits the Adj
column (floor-1 conditioning is a no-op). Rows whose post-type incoming
is unresolvable carry Ratios/min_ratio = None explicitly plus a warning
note — pair join values are never silently substituted. bodyId-level
tables have no adjusted variant (threshold-free per-pair ratio only).

- **refill-full** — the second round admitted at SYNAPSE >= 1 ONLY (no
  ratio tier, no synapse tier), re-valuing the SAME emitted chains; the
  full-connection scope, produced for EVERY all-mode run with emitted rows.
  Realized per chain by the LAYERED PER-POSITION BODY-SET FILTER
  (2026-10-13, universal at every applied threshold — forward body sets
  from the run's sources, backward sets to its targets, intersected per
  position; per-hop mass = kept-edge weight sum; no bodyId paths
  materialized, so `full.paths`/`full.bodyid_paths` are null and
  `full.truncated` is always false). Chains without an effective-label
  revisit are exact; label-revisiting chains carry the layered-walk
  superset semantics, disclosed as `full.chains_revisit` (+ examples)
  and a run warning note. Linear cost — the L8 MCNS ratio cell's refill
  fell ~115 min → ~3 min. Files: `refill_full_type_pairs.csv`
  (data_details) +
  `refill_full_paths_type.csv` / `refill_full_carrying_paths_type.csv` /
  `refilled_full_connection_type.csv` (mirror). The report's Refill tab
  carries a Refill-asked | Refill-full scope switch; the full scope's Adj
  values read the full pair table's plain ratio (at floor 1, conditioned
  == plain).

Asked view (both kinds share the machinery):

- `data_details/type_level_refill/` — the raw records:
  `refill_type_pairs.csv` (per emitted pair: emitted / refill /
  refilled_total + refilled ratios), `refill_bodyId_pairs.csv` (capped
  bodyId detail), `refill_provenance.json` (`table_reproduced` is the
  consistency anchor — the re-derived cut must reproduce
  `connection_type.csv` exactly or the module refuses),
  `README.md`.
- `<run>/type_level_refill/` — the standalone mirror (parallel to
  `find_reciprocal/`): `complete_at_asked_paths_type.csv` (EVERY
  asked-threshold chain with the emitted/refill/skipped per-hop split,
  stamped `complete_at_asked` in the provenance — the Refill tab's
  DEFAULT view) and `refill_paths_type.csv` (the refill-carrying paths
  in the exact `*_allpaths_type.csv` column order — the second view), `refill_connection_type.csv`
  (refill-only) / `refilled_connection_type.csv` (complete-at-asked) /
  `refilled_connection_bodyId.csv` in the connection-table formats, plus
  `refill_type_heatmap.html` / `refill_type_network.html` (+ bodyId twins)
  built by the visualization stage.
- Post-hoc: `python scripts/TypeLevelRefill.py <run_folder> --connections
  <full connections parquet> --neuron-table <allneurons csv> --in-run`
  (add `--out DIR` to keep the run folder byte-identical). The
  `[type-level refill]` entry in `user_warning_notes.txt` reports applied
  vs asked and the recovered totals.

The pair report's **Refill tab** renders this as a second Pair Explorer
(selects → per-pair panes → capped "Top paths per length"); the per-connection
pair table and bodyId detail stay exported files.

**Probability SUSPENDED (2026-10-06, unverifiable fidelity):** traversal/path-probability values are no longer exported, displayed, or used for ranking. — the refill records and mirrors carry no probability columns;
ratios (plain + adjusted) are the strength readouts.

## Notes

- InitializeNeuronInfo emits a `[DROCAT][neuron-match] source=… target=…` marker;
  a zero match means the query did not resolve (different from an empty result).
- Empty `sourceNeurons`/`targetNeurons` have special backend meaning — state the
  scope explicitly.
- `max_interlayer=0` is "no limit" only for all-neurons sets.
- Cache-like methods are used by the Settings tab and the connectivity-profile /
  similar tools to prepare data before analysis.
