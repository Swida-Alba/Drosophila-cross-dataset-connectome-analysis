# Shortest Paths (find_shortest)

Reproduce the **Shortest Paths** UI tab as a direct backend call. Minimum
hop-count pathfinding between neuron groups (shares `FindNeuronConnection` with
Complete Paths).

## Backend contract

- **tool_key:** `find_shortest`
- **import:** `from coana import FindNeuronConnection`
- **class:** `FindNeuronConnection` (var `fc`)
- **init:** `fc.InitializeNeuronInfo()`
- **method:** `fc.FindShortestPath(forward_only=True)` — the method's find_reciprocal parameter defers to the constructor field (2026-09-25); the explicit forwarding form is retired. Set the constructor flag and call plain
- **enumeration:** `FastGraph.find_paths_shortest_strongest_first` — per-target
  reverse BFS + maximin bottleneck DP + best-first strength-ordered emission
  (k-way merge). (`find_paths_shortest_backward` exists on FastGraph but is NOT
  used by this pipeline.)

## Parameters the UI builds

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    dataset="male-cns:v0.9",
    sourceNeurons=["aMe12"],
    targetNeurons=["PPL101"],
    output_dir="/absolute/output/shortest",
    min_synapse_num=3,
    min_ratio=0.0,
    min_traversal_probability=0.0,
    max_interlayer=5,              # Shortest Paths UI default (5, not the shared default 2)
    filter_by="bodyId",
    graph_edge_limit_bodyid=0,     # Edge Budget NEVER applies in shortest mode (graph never floored)
    max_paths_bodyid=0,            # Max Paths: path-output budget; auto -> 1M StrongestFirst budget (tau reported when it bites)
    discovery_batch_budget=2_000_000,  # memory control: per-batch BFS distance states; 0 = legacy single batch (results identical)
    target_batch_size=0,           # alternative: fixed targets per batch; 0 = use discovery_batch_budget
    discovery_store_retention="keep",  # post-run store handling: keep | compact (merge connection layers to 4 cols, drop derivable dag chunks) | prune (keep meta.json only)
    visualize_before_reconstruct=False,
    search_columns="auto",              # "auto" | "type" | "instance" | "bodyId"
    network_layout="distributed",
    use_cache=True,
    edgeN_limit=500,               # Visualization Edge Limit: drawing-only cap per HTML view
    output_format="csv",
    skip_bodyId=True,
    drop_untyped=True,             # Drop Untyped Neurons: label filter; see Notes
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
)
# optional: fc constructor param custom_mapping_file for a custom grouping/mapping JSON

fc.InitializeNeuronInfo()
fc.FindShortestPath(forward_only=True)
```

## Run

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 \
  --script archive/scripts_local/agent_FindShortest_<date>.py
```

## Outputs

- Per-pair minimum-hop path tables (CSV/XLSX) and summaries, plus the same
  threshold/bottleneck provenance block as Complete Paths (`parameters.txt`,
  `all_attributes.json`, `data_details/parameters.csv`) and
  `data_details/untyped_dropped_records.csv` when untyped rows were dropped.
- Pair report: written automatically when the run completes
  (`path_report.html` + `paths_pair_breakdown/*.csv` in the run folder;
  see tabs/find-path.md "Pair report"). Shortest is minimum-hop per
  bodyId pair, so a TYPE-level pair can still span several lengths — the
  per-length cap applies normally.

## Notes

- The UI keeps the early network preview disabled (`visualize_before_reconstruct=False`).
- Start with `max_interlayer=5` (this tab's default; the shared default elsewhere
  is 2) and `skip_bodyId=True`; verify counts before enabling bodyId-level output.
- **Drop Untyped Neurons** (`drop_untyped=True`, checked by default): the same
  label filter as Complete Paths — applied after label enrichment and before
  graph construction, so untyped neurons never appear as intermediate nodes.
  Records: `data_details/untyped_dropped_records.csv`; counts appended to
  `user_warning_notes.txt` (only when rows were dropped).
- Budgets: the StrongestFirst path budget (`max_paths_bodyid`) may bite and is
  reported as tau in the provenance block; the Edge Budget never applies —
  shortest mode is never floored.
- **Batched discovery** (default on; `discovery_batch_budget=2_000_000`,
  `0` = legacy single batch): discovery memory for many broad targets would
  otherwise scale with targets × union frontier × depth (a 242-target depth-5
  run was killed at ~450 GB RSS, 2026-10-02). Labels stream to
  `shortest_discovery_store/` (connections per `conn_layer`,
  `node_distances`, `dag_edges`, `pairs.parquet`, `meta.json` — keep for
  auditability; runs never reuse a foreign store) and enumeration runs per
  batch of targets bounded by the state budget. Results are identical to the
  legacy path; the realized composition is in
  `shortest_discovery_diagnostics.batching` (all_attributes.json). The Path
  Budget bounds only the result; the Edge Budget never applies here. A
  single broad target at deep L is still depth-bound — lower
  `max_interlayer` or raise `min_synapse_num` for those.
- **Store retention** (`discovery_store_retention`, default `keep`):
  `compact` merges the per-layer connection frames into one 4-column
  `connections_all.parquet` (cross-layer weight multiplicity preserved)
  and deletes the derivable `dag_edges/` chunks — roughly halves a deep
  run's store; `prune` removes all store data files and keeps `meta.json`
  as the size census. Applied only after a successful run; path outputs
  are byte-identical across modes; recorded in `meta.json`
  (`retention`), the run metadata, and `user_warning_notes.txt`.
- `min_ratio` and `min_traversal_probability` remain available as readout
  columns for compatibility, but their filters are disabled in Shortest Paths;
  they do not appear as `r[]p[]` filename notes.
- This is also the engine behind `inter_dataset`'s `path_mode="shortest"`.


## Connection-ratio basis

Shortest Paths supports `weight_basis='connection_ratio'` (Phase 3): the ratio threshold applies at the target-rooted discovery, per-pair min-hop paths rank by ratio bottleneck, folders use the `L{d}r{t}` suffix. Multi-threshold replay remains 'all'-mode only (as in synapse basis).
