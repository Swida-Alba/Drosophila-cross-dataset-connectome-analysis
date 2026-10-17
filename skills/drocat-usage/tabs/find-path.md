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
- The type-level path table is written **strongest-first** (`length`
  ascending, then `min_weight` descending) by the same finalize pass that
  adds `coverage` and re-scopes the realized per-hop Weights/Ratios; the
  report and the visualizations read the first `pathN_to_show` rows as the
  top paths, so that order is load-bearing. A pass that cannot complete
  leaves streaming order and discloses it as a `[type-path sort]` line in
  `user_warning_notes.txt`.
- A multi-threshold sweep (`FindAllPathMultiThreshold`) writes one folder per
  tier, restamping the tier token in each folder NAME
  (`..._L3w2_<ts>` → `..._L3w5_<ts>`; `r{decimal}` on the ratio basis), and a
  `Save As` name without a tier token gets the slice in a `minsyn_{t}` /
  `minratio_{d}` subfolder with a note. See
  `skills/drocat-backend/modules/coana-connectivity.md`.
- Use `visualize_before_reconstruct=False` (the UI default) for headless runs and
  hand off to PlotPath for the interactive network.

## Pair report (automatic)

The report is written for the TOOL that produced the folder: the run-name
grammar (`find-paths-complete_…`, `find-paths-shortest_…`, `find-network_…`,
`cross-dataset_…`) decides the wording, and — since 2026-10-16 — a folder under
a custom `Save As` name is read from its own `path_mode` stamp instead of
falling back to generic 'Pathfinding' advice (a renamed Network run with no
result used to raise 'no *_allpaths_type.csv found' rather than write its
kind-aware empty state).

Every UI run (and any folder, via the CLI) writes a per-source-target-pair
HTML report (`path_report.html`) plus two lossless breakdown CSVs
(`paths_pair_breakdown/pair_breakdown_paths.csv`,
`pair_breakdown_intermediates.csv`) INTO the run folder — Overview /
Global / Path / Data, Analysis carrying this run's destinations
behind a mode row (`Complete` | `Fast (corridor)`) and a view row
(`Pair Explorer` | `Refill`), each rendering only when it has at least
two options; the Pair Explorer (Source+Target multi-selects rendering one UNION
  pane; capped
top-10-per-length table, interactive layered network coloring shared vs
unique intermediates) is one of them. The runner's post-run hook generates it
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
  bodyId-level visualizations share the same cap. The cap is stamped in the
  folder as `visualization edge limit: 200 (drawing only; tables keep every
  edge)` (or `no drawing cap`), so a page showing fewer pairs than the run
  emitted explains itself to a reader who has only the folder (2026-10-17).
- A source or target query that resolves to no neuron in the selected dataset
  RAISES and writes nothing — not even a folder (2026-10-17; before that it
  printed an error and returned, which a headless caller could not tell apart
  from a run that searched and found nothing, and an explicit `saveas` still
  created its folder eagerly, so the refusal left a named EMPTY directory
  behind). An empty RESULT is different and does write: the folders a threshold
  cut to nothing each state which threshold produced nothing. Check a name
  against the dataset in the Search Neurons tab; a `cache_only` run only
  resolves names already local.
- Network and Sankey pages colour edges by predicted transmitter, and a pair the
  dataset leaves unannotated gets the palette's **Unknown** gray plus an
  `Unknown Edges` entry in the group selector — never the base link colour, a
  light blue that reads as a transmitter (2026-10-17).
- Network and Sankey pages colour edges by predicted transmitter. A pair the
  dataset leaves unannotated is drawn in the palette's **Unknown** gray
  (`#95A5A6`) and listed as **Unknown Edges** in the group selector — never in
  the base link colour, a light blue that reads as a transmitter (2026-10-17).
  The bucket is emitted only when the run annotates at least one pair, so a
  dataset with no transmitter column keeps its uniform edge colour.
- `bodyId_visualization/` pages grey out Refilled and the adjusted ratio: both
  are measured per TYPE pair, never per bodyId. When the run wrote those records
  the page says so under its Connection Metric selector and points at
  `visualization/` and `type_level_refill/`; a run with no refill records gets
  no such sentence, because the caption would point at a table the folder lacks.
- `min_ratio` and `min_traversal_probability` are retained as compatibility
  knobs. **Probability SUSPENDED (2026-10-06, unverifiable fidelity):** traversal/path-probability values are no longer exported, displayed, or used for ranking.
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

The Threshold Basis selector swaps the edge-weight basis: *Synapse count* (default) or *Connection ratio* (`weight_basis='connection_ratio'`). Ratio mode sends `min_ratio` as THE threshold — the UI field is **Min Connection Ratio (%)**, typed in percent and divided by 100 in one shared helper (`ui/components/pathfinding_controls.ratio_payloads`), so a typed `1` is a 1 % floor on every tab (the engine value is the typed percent ÷ 100, so 1 % → `min_ratio=0.01`; valid typed range `0 < t <= 100` percent, and real-data viable floors sit around 0.05 %–0.2 % = 5e-4..2e-3 raw), forces the synapse floor to 1, ranks paths by the exported type-path `min_ratio` column (a REALIZED mass share — the weakest per-hop realized support, which can fall below the applied bodyId-level threshold), and writes `data_details/ratio_synapse_map.csv`. Per-hop Weights/Ratios on the type paths are realized-scope on EVERY weight basis (engine fix 2026-10-05): Weights count only the bodyId edges backing that exact sequence at that position, Ratios divide that realized mass by the post type's full-membership incoming mass (the same canonical denominator as `connection_type.csv`); pair-level totals/shares live in `data_details/connection_type.csv`. `connection_ratio_adj` (threshold-conditioned share) is exported on the connection tables; it equals `connection_ratio` when Min Synapse Count is 1. Combine Thresholds = Both (AND) makes an edge enter the graph only when it clears BOTH the synapse count and the ratio thresholds (folder token `L{d}w{t}r{t_r}`). See `docs/core-features/ConnectionRatioPaths.md`.
