# Complete Paths (find_path)

Reproduce the **Complete Paths** UI tab as a direct backend call. Multi-hop
pathfinding between two neuron groups.

## Backend contract

- **tool_key:** `find_path`
- **import:** `from coana import FindNeuronConnection`
- **class:** `FindNeuronConnection` (var `fc`)
- **init:** `fc.InitializeNeuronInfo()`
- **method:** `fc.FindAllPath(forward_only=True)` — the method's find_reciprocal parameter defers to the constructor field (2026-09-25); the explicit forwarding form is retired. Set the constructor flag and call plain
  (the UI uses `find_all_path`; `fc.FindPath()` is the single-strategy variant)

## Parameters the UI builds

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    dataset="male-cns:v0.9",
    sourceNeurons=["aMe12"],          # types/bodyIds/instances/regex
    targetNeurons=["PPL101"],
    output_dir="/absolute/output/paths",
    min_synapse_num=3,
    min_ratio=0.0,
    min_traversal_probability=0.0,
    max_interlayer=2,
    filter_by="bodyId",               # or "type"
    pathfinding="StrongestFirst",     # built-in default; DP/MemoizedDFS/DFS = unbounded complete runs (API)
    max_paths_bodyid=0,               # Max Paths: path-output budget; auto -> internal 1,000,000 (tau reported when it bites)
    graph_edge_limit_bodyid=0,        # Edge Budget: graph-level floor, 'all' mode only; 0 = off; ~1_000_000 prunes the cone
    visualize_before_reconstruct=False,
    search_columns="auto",              # "auto" | "type" | "instance" | "bodyId"
    network_layout="distributed",
    use_cache=True,
    edgeN_limit=500,                  # Visualization Edge Limit: drawing-only cap per HTML view
    output_format="csv",              # or "xlsx"
    skip_bodyId=True,                 # faster type-level first pass
    drop_untyped=True,                # Drop Untyped Neurons: label filter; see Notes
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
fc.FindAllPath(forward_only=True)
```

## Run

Save the above as a focused script (e.g. `archive/scripts_local/agent_FindPath_<date>.py`)
and run through the launcher:

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 \
  --script archive/scripts_local/agent_FindPath_<date>.py
```

Or run the template `scripts/FindPath.py` (edit its config block):

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script scripts/FindPath.py
```

## Outputs

- CSV/XLSX type-level (`path_type`) and bodyId-level (`path_bodyId`) path tables,
  path summaries, and optional network/heatmap HTML.
- Canonical visualization names inside the run folder, under both
  `visualization/` and `bodyId_visualization/`: `Network_<run>.html`,
  `Heatmap_<run>.html`, `Sankey_<run>.html`, and
  `visualization_data/<run>_data_*.csv`; early previews keep `network_early/`
  and `network_early_bodyId/`. When the drawing cap trims, companion CSVs
  (`type_paths_visualized.csv`, `bodyId_paths_visualized.csv`) record what was
  rendered.
- Threshold/bottleneck provenance is written to `parameters.txt`,
  `all_attributes.json`, and `data_details/parameters.csv` (see Notes).
- Use `visualize_before_reconstruct=False` (the UI default) for headless runs and
  hand off to PlotPath for the interactive network.

## Pair report (automatic)

Every UI run (and any folder, via the CLI) writes a per-source-target-pair
HTML report (`path_report.html`) plus two lossless breakdown CSVs
(`paths_pair_breakdown/pair_breakdown_paths.csv`,
`pair_breakdown_intermediates.csv`) INTO the run folder — Overview /
Global / Pair Explorer (Source+Target multi-selects rendering one UNION
  pane; capped
top-10-per-length table, interactive layered network coloring shared vs
unique intermediates) / Data. The runner's post-run hook generates it
before the Output Files panel scans, so the files appear as native run
outputs; direct-script runs can generate it with:

```bash
python scripts/PathsPairReport.py <run_dir> [--top-per-length 10] \
    [--matrix-rows 50] [--global-pairs 50] [--global-edges 60] \
    [--rank-by min_weight]
```

## Notes

- `max_interlayer=0` means "no limit" only when a source/target set is marked as
  all neurons; otherwise limit hops, then increase after a smaller run completes.
- **Drop Untyped Neurons** (`drop_untyped=True`, checked by default; configurable
  in Settings → Default Settings → Pathfinding & Output): drops edges touching
  untyped neurons (empty label, Unknown/None/NaN sentinel, or all-digit
  bodyId-fallback label; shared predicate
  `utils.label_utils.is_untyped_type_label`) after label enrichment and before
  graph construction, so untyped neurons never appear as intermediate nodes of
  returned paths or visualizations. Dropped rows go to
  `data_details/untyped_dropped_records.csv` (written only when rows were
  dropped); counts are appended to `user_warning_notes.txt`. `drop_untyped` is
  part of the FindAllPath graph-cache key.
- Filter roles: Min Synapse Count = threshold; Edge Budget
  (`graph_edge_limit_bodyid`) = graph-level floor, `'all'` mode only; Max Paths
  (`max_paths_bodyid`) = path-output budget (both modes); Drop Untyped =
  neuron-label filter; Visualization Edge Limit (`edgeN_limit`) = drawing-only
  cap on unique edges rendered per HTML view — it never changes fetch/graph/paths,
  and a single complete path may exceed it to stay intact; type- and
  bodyId-level visualizations share the same cap.
- `min_ratio` and `min_traversal_probability` are retained as compatibility
  parameters and exported readout columns, but their filters are disabled in
  Complete Paths; they do not add `r[]p[]` notes to the run folder name.
- Every run writes a threshold/bottleneck provenance block to `parameters.txt`,
  `all_attributes.json`, and `data_details/parameters.csv`: `requested_threshold`,
  `applied_threshold`, `applied_threshold_source` ('requested' |
  'strongest_first_budget' | 'edge_budget' | 'strongest_first_budget+edge_budget'),
  `strongest_first_budget` (auto 1,000,000) / `strongest_first_budget_bitten`,
  `strongest_first_tau` / `tau_canonical`, `strongest_dropped_bottleneck`,
  `edge_budget` / `edge_budget_applied` / `edge_budget_landing`,
  `edge_weight_floor`, `strongest_retained_bottleneck` (W*), and
  `paths_complete`. `parameters.txt` keeps the legacy `applied_tau` /
  `edge_weight_floor` alias lines.
- Empty `sourceNeurons`/`targetNeurons` have special meaning in the backend; state
  the intended scope before using them.
- Keep the UI closed and `showfig=False` until the output is validated.


## Threshold basis (connection-ratio mode)

The Threshold Basis selector swaps the edge-weight basis: *Synapse count* (default) or *Connection ratio* (`weight_basis='connection_ratio'`). Ratio mode sends `min_ratio` as THE threshold (float, 0<t≤1; real-data viable values ~5e-4..2e-3), forces the synapse floor to 1, ranks paths by the exported type-path `min_ratio` column (a REALIZED mass share — the weakest per-hop realized support, which can fall below the applied bodyId-level threshold), and writes `data_details/ratio_synapse_map.csv`. Per-hop Weights/Ratios on the type paths count only the bodyId edges backing that exact sequence at that position; pair-level totals/shares live in `data_details/connection_type.csv`. At Min Synapse Count > 1, synapse runs additionally export `connection_ratio_adj` (threshold-conditioned share). See `docs/core-features/ConnectionRatioPaths.md`.
