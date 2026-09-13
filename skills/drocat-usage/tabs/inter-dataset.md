# Cross-Dataset Comparison (inter_dataset)

Reproduce the **Cross-Dataset Comparison** UI tab as a direct backend call. Runs
`ComparisonAnalyzer` over N datasets with shared source/target queries.

## Backend contract

- **tool_key:** `inter_dataset`
- **import:** `from comparison import ComparisonParameters, ComparisonAnalyzer`
- **wrap:** the UI builds a `ComparisonParameters` object, then
  `ComparisonAnalyzer(params, verbose=True)`, then
  `analyzer.run_comparison()` and `analyzer.export_results()`.
- **class:** `ComparisonAnalyzer` (var `analyzer`)

## Parameters the UI builds (ComparisonParameters)

```python
from comparison import ComparisonParameters, ComparisonAnalyzer

params = ComparisonParameters(
    datasets=["hemibrain:v1.2.1", "male-cns:v0.9"],
    source_neurons=["aMe12"],           # shared across all datasets
    target_neurons=["PPL101"],
    output_folder="/absolute/output/comparison",
    comparison_mode="path",             # or "edge" to preserve strong direct edges
    path_mode="all",                     # "all" | "shortest"
    max_interlayer=2,
    thresholds=[1, 3, 5, 10],
    threshold_mode="standard",          # default: same N in every dataset
    # Custom combination mode uses complete query rows across at least two
    # datasets, not independent schedules:
    # threshold_mode="combinations",
    # threshold_dataset_order=["hemibrain:v1.2.1", "male-cns:v0.9"],
    # threshold_combinations=[
    #   {"id": "combo_001", "thresholds":
    #       {"hemibrain:v1.2.1": 3, "male-cns:v0.9": 8}},
    # ],
    # Auto (default in the UI; needs >= 2 datasets and path mode 'all';
    # threshold chips are optional — an empty box uses the bootstrap floor
    # and entered chips raise the floor to the lowest chip) measures each
    # dataset's window from one bootstrap
    # enumeration and installs BOTH the per-threshold (vertical) and the
    # density-matched (horizontal) aligned rows:
    # threshold_mode="auto",
    # threshold_dataset_order=["hemibrain:v1.2.1", "male-cns:v0.9"],
    top_edges=500,
    graph_edge_limit_bodyid=0,          # Edge Budget off; set ~1_000_000 to cap the discovery cone
    edgeN_limit=500,
    pathfinding="StrongestFirst",       # built-in default; budgeted by max_paths_bodyid (tau on bite)
    max_paths_bodyid=0,                 # auto -> internal 1,000,000 path budget
    search_columns="auto",              # "auto" | "type" | "instance" | "bodyId"
    skip_bodyId=True,
    cache_only=False,
    auto_type_mapping=True,
    _min_ratio=0.0,
    _min_prob=0.0,
    _output_format="csv",
    parallel=True,
    max_workers=None,                   # None disables parallelism
    separate_hemispheres=False,
    keep_only_hemisphere_conserved_connections=False,
    symmetry_analysis=False,
    find_reciprocal=False,
    drop_untyped=True,                  # Drop Untyped Neurons (Advanced Settings); see Notes
)
# optional: params = ComparisonParameters(..., overall_mapping_json="/path/to/mapping.json")

analyzer = ComparisonAnalyzer(params, verbose=True)
analyzer.run_comparison()
analyzer.export_results()
```

## Run

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_Compare_<date>.py
```

## Outputs

- Per-dataset comparison tables (CSV/XLSX), threshold summaries, report, and
  conserved-path HTML views.

## Notes

- `comparison_mode="path"` uses the pathfinding engine (FindAllPath/FindShortestPath);
  `comparison_mode="edge"` preserves strong direct edges. The `path_mode` selects
  per-pair minimum-hop vs all-paths behavior.
- **Drop Untyped Neurons** (`drop_untyped=True`, checkbox in Advanced Settings):
  the shared predicate `utils.label_utils.is_untyped_type_label` (empty label,
  Unknown/None/NaN sentinel, all-digit bodyId-fallback label), applied by the
  analyzer AFTER standardized cross-dataset label mapping. The delegated
  per-dataset pathfinding runs execute with `drop_untyped=False`, so only the
  comparison-level filter fires and no per-dataset `data_details/` records are
  written. Dropped rows land in `comparison_results/untyped_dropped_records.csv`
  with an `untyped_side` column (`pre` / `post` / `pre+post`); counts are
  appended to `user_warning_notes.txt` only when rows were dropped.
- **Threshold modes:** the Core Parameters editor defaults to Auto
  (density-aligned; requires >= 2 datasets, see the constructor comment
  above). Standard applies N threshold chips to every selected dataset.
  Advanced
  threshold combinations use dataset columns and query rows; each row must
  have one threshold for every dataset. The row is the alignment/comparison
  identity, while the sorted cell union is only a deduplicated raw-run
  schedule. Raw `(dataset, threshold)` jobs shared by rows are reused.
- Each per-dataset threshold folder carries the threshold/bottleneck provenance
  block (requested vs applied threshold, `applied_threshold_source`,
  StrongestFirst budget/bite/tau, `tau_canonical`, `w2`, Edge Budget `w0`/`w1`,
  `strongest_retained_bottleneck` (W*), and `paths_complete`) in
  `parameters.txt`, `all_attributes.json`, and `data_details/parameters.csv`.
- The comparison root writes `effective_thresholds.json` for the UI notice and
  `comparison_results/pathfinding_provenance.csv` with one complete row per
  dataset/requested raw threshold. `threshold_scope` distinguishes scalar
  rows from query cells. In that row, `applied_threshold` is the
  canonical equivalent Min Synapse Count, `tau` is the StrongestFirst landing
  bound, `w0`/`w1` are the Edge Budget floor/landing tier, `w2` is the strongest
  dropped path bottleneck, and `W*` is the strongest retained bottleneck.
- Combination runs additionally write
  `comparison_results/threshold_combinations.csv`, one row per `query_id` and
  dataset. It records the requested cell, applied threshold, source, tau,
  StrongestFirst budget, Edge Budget `w0`/`w1`, `w2`, `W*`, and completeness;
  use it as the join key for query-specific comparison exports.
- `comparison_results/threshold_sensitivity.csv` and
  `comparison_results/unified_summary.csv` carry the same provenance fields so
  downstream analysis does not have to infer the applied threshold from edge
  counts. In combination mode, sensitivity rows also carry `query_id` and
  are query-cell diagnostics rather than adjacent-threshold retention rows.
  Shortest mode reports the StrongestFirst budget state but never applies the
  Edge Budget floor.
- Combination-mode similarities are written under
  `similarity_matrices/similarity_query_{query_id}.csv` and
  `similarity_by_query.csv`; use the query ID and per-dataset threshold
  columns rather than a scalar threshold union.
- Combination-mode `comparison_report.html` uses the same full report shell as
  Standard mode (summary charts, provenance, similarities, networks,
  edge/path matrices, conservation, overlap, and statistics) and iterates
  every query row. Query-specific matrix exports are named
  `edge_presence_matrix_query_{query_id}.csv` and
  `path_presence_matrix_query_{query_id}.csv` using a filesystem-safe slug.
  The original query ID is retained inside the file and in the manifest.
- Pathfinding comparisons do not apply ratio or traversal-probability
  filtering; their comparison visualization exports therefore do not create
  `by_ratio/` or `by_probability/` folders.
- `edge_density_per_threshold.csv`, `threshold_alignment_matrix.csv`, and
  `threshold_alignment_best_matches.csv` are raw-run schedule diagnostics
  (`threshold_scope=raw_run_schedule_diagnostic` in combination mode); use
  `threshold_combinations.csv` for the actual query rows.
- **Density curves** (every pathfinding mode):
  `comparison_results/density_curves.csv` and `density_windows.csv` give
  each queried dataset a cone-scoped curve over its own window
  `[w_start, w_star_measured]` — from the minimal available threshold to
  the measured `max(path bottlenecks)`. The edge basis follows
  `drop_untyped`: `bodyId_edges_typed` when on (N = typed searched nodes),
  `bodyId_edges_all_but_debris` when off (N = typed + untyped); segmentation
  debris (ids absent from the curated table) is always excluded. The raw
  capture lives in `dataset_data/{dataset}/_density/` (`density_edges.npz`,
  `density_path_bottlenecks.npy`, `density_meta.json`); the two-panel figure
  is `comparison_visualizations/density_alignment_threshold_curves.png`. In
  auto mode, `comparison_results/density_alignment_best_matches.csv`
  additionally holds the measured vertical/horizontal aligned rows, and
  `[auto threshold]` / `[density]` warnings mirror into the run guide. When
  both row modes coexist, read the vertical rows as the like-for-like spine
  (one identical threshold for every dataset) and the horizontal rows as
  the density-matched envelope (per-dataset thresholds equalizing E(t)/N,
  valid even where no shared complete threshold exists). Horizontal labels
  carry their explicit per-dataset thresholds; the comparison report
  renders the two analyses as separate sections (per-threshold vs
  combination style).
- `degree_*`, `top_edges_*`, and `unique_to_*` exports are Standard-only;
  Custom combination runs omit them rather than infer a union threshold.
- Use `auto_type_mapping=True` (and `overall_mapping_json`) when type names differ
  between datasets. Path/edge merging resolves through the shared validity-aware
  resolver: licensed renames merge under the canonical key, valid splits expand,
  and conflicts stay dataset-scoped (never merged by raw same-name). The run's
  `auto_type_mapping.json` records per-status counts and `raw_fallback_used`.
- Use `parallel=True` with a bounded `max_workers` for many datasets; start with
  `skip_bodyId=True` and `max_interlayer=2`.
