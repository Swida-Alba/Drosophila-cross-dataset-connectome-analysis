# Cross-Dataset Comparison (inter_dataset)

Reproduce the **Paths** tab (UI label "Paths", in the **Cross-Dataset** group;
tool key still `inter_dataset`) as a direct backend call. Runs
`ComparisonAnalyzer` over N datasets with shared source/target queries. For
bodyId-level validation of a type mapping, see the sibling
[type-validation.md](type-validation.md).

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
    # (Min Synapse Count 3 synapse basis; ratio tier 0.001 connection-ratio
    # basis) and entered chips raise the floor to the lowest chip) measures each
    # dataset's window from one bootstrap
    # enumeration and installs BOTH the per-threshold (vertical) and the
    # density-matched (horizontal) aligned rows:
    # threshold_mode="auto",
    # threshold_dataset_order=["hemibrain:v1.2.1", "male-cns:v0.9"],
    top_edges=500,
    graph_edge_limit_bodyid=1_000_000,  # Edge Budget (the UI default in 'all' mode; 0 = off)
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
    max_workers=4,                      # the UI default; None disables parallelism
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
- Pair report: written automatically at the run root when the run
  completes (`path_report.html` + `paths_pair_breakdown/*.csv`); each
  `dataset_data/<dataset>/minsyn_<N>/` delegate becomes a unit — the
  Global tab's pair × unit matrix and coverage histogram give the
  run-wide view next to `comparison_report.html`. Every delegate folder
  also receives its own single-unit `path_report.html` (see
  tabs/find-path.md "Pair report").

## Report layout

The backend default renders the original single-page report. The new
tabbed layout (opt-in via `report_layout='tabbed'`, still under
refinement) opens on **Overview** and organizes the rest into page tabs —
**Combos (per query)** (per-row dashboards with jump buttons),
**Type Mapping**, **Plots** (per-threshold and density-matched charts as
separate sub-tabs), **Matrices & Networks**, **Cross-views**, **Notes**.
`report_layout='legacy'` restores the single-page report; `'both'`
writes both files.

## Query rows and merge granularity (auto mode)

- Auto-mode query rows are named `threshold={N}` (vertical, one threshold
  everywhere) and `aligned_density={level}` (horizontal, density-matched;
  the explicit per-dataset thresholds stay in the label). Those ids are the
  report tab buttons, the CSV `query_id` values, and the per-query export
  filenames (e.g. `conserved_network_threshold_19_network.html`).
- With auto type mapping, the run builds a **merge policy** from the query
  chips: the report's rows are keyed by the chips' group labels instead of
  always folding toward male-cns names. All chips resolving into one
  dataset's naming anchor on that dataset; all-same-name chips use the
  shared name; mixed chips emit a `[type granularity]` note and keep the
  minimal inseparable leaves. A 1-to-N parent queried from its own side
  merges all branches into ONE row (weak auto-vote branches stay valid but
  are listed under `[BANC auto labels]`); a leaf-anchored chip covers only
  its own branch and the shared parent stays a separate whole row; a leaf
  claimed by two queried parents merges with neither (`[merge fan-in]` —
  with an explicit note that the queried parent's row is PARTIAL by
  design).
- Evidence surfaces: the report's Type Mapping section leads with ONE
  merged, column-aligned query-role table — section groups **Queried
  sources** / **Queried targets** / **Path intermediates** share one
  coloring schema (status colors; muted = resolves there but was not
  traversed) and one `#paths` column (paths starting at / ending at /
  traversing the type), with the full canonical grid as a collapsed
  appendix. A **color legend** between the table title and its intro line
  shows the six resolution colors plus the muted and `—` no-resolution
  marks (rendered from the same palette the cells use). The grid's
  **Source (priority)** column shows the ONE
  observation from the highest-priority dataset where the row resolves —
  male-cns → FAFB → other neuprint → BANC, a GLOBAL order, never the
  per-name merge anchor (remaining observations collapse into a
  tooltip); target cells show per-dataset resolved names — names
  differing from the canonical are colored —
  split-branch members share one color — and lists beyond three names
  collapse; the Source column names remaining observations inline
  (`(+2: FAFB, BANC)`); the ⚠️ auto-only badge attaches ONLY to the
  auto-label source (donor) dataset cell, never the canonical name; the
  same-name-first suspects open in a persistent hover popover (copiable)
  instead of an in-cell expander. A
  fan-in key (claimed by two parents) is pruned from both parents' rows,
  and the SAME prune applies globally: a split branch that the mapper
  crosswalk itself lists as a 1-to-N target of a different (even
  unqueried) home-namespace parent leaves the parent row AND the merge
  counts (`[merge fan-in] … global contest`). The
  **Resolution topology** card stays collapsed by default; the
  run folder gains `type_resolution_topology.json`;
  `comparison_results/type_resolution_union.csv` resolves the FULL union
  of types that appeared in any dataset into EVERY dataset (per-query
  absence verdicts: `below_threshold` with the max source-side edge
  weight, `not_recruited`, `no_edges`, `not_in_dataset`,
  `unmapped`/`conflict`, `resolved_absent`); the presence matrices and
  `unified_edge_comparison.csv` carry matching per-endpoint
  `source_status_<ds>` / `target_status_<ds>` columns, and the network
  edge hovers / node tooltips annotate the same verdicts;
  `auto_type_mapping.csv`
  gains additive `anchor_group`/`auto_only` columns (`anchor_group` only
  when every endpoint of the row belongs to that same group); merged
  neuron-count rows carry their raw composition in `group_members`. All
  warning blocks mirror into
  `user_warning_notes.txt`; the custom label mapper (LabelMapper /
  `overall_mapping_json`) overrides any merge decision. Round-4 note: the
  section leads with the merged query-role table (intermediates ranked
  by traversal count) and the run header carries a
  "🧠 Hemisphere-aware run" badge whenever `separate_hemispheres` is on;
  `ComparisonParameters.for_run_folder(run_dir)` + `skip_existing` resumes
  a run folder in place WITHOUT losing the hemisphere/symmetry/reciprocity
  flags (the from_dict round-trip drop was fixed 2026-09-16).

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
  dataset/requested raw threshold. (Edge mode: the edge data is exactly
  `weight >= requested`, so `applied_threshold` equals the request there —
  the tau/budget fields describe the side-effect path runs only, kept under
  `side_path_run`.) `threshold_scope` distinguishes scalar
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
- The report's Similarity Trends grid plots one metric row each — Jaccard,
  Edge Rank, Cosine, Spearman — across per-threshold and per-density
  columns. A metric row with no plottable value in any query renders a
  centered explanatory note instead of bare axes. Spearman gating is
  intentionally asymmetric: the query-keyed grid reads the CSV column
  gated at ≥10 shared edges, while the Standard-mode panel computes
  Spearman live UNGATED (≥3-shared floor) so tiny samples stay visible
  for manual judgement — each empty note names the rule that applied.
- The Similarity section presents four representatives by LEVEL — edge
  🔷 Jaccard + Cosine, path 🟣 Path Jaccard, graph 🔶 NetSimile-lite
  (cards colored by level) — with a per-pair detail table (coverage,
  edge/path top-20, gated Spearman ≥10 shared, hop/strength W1). Edge
  Rank / Path Rank / Pearson / RV / Ruzicka are legacy CSV-only columns.
- Standard-mode `comparison_visualizations/` writes FOUR trend PNGs —
  `jaccard_similarity_trend.png`, `top20_overlap_trend.png` (v2.2,
  replaces the retired Edge Rank trend), `netsimile_trend.png` (v2.2,
  replaces the retired Path Rank trend), and `cosine_similarity_trend.png`
  — each with a matching `visualization_data/` CSV. Combination mode
  writes NONE of them: query rows have a display order, not a numeric
  threshold schedule, so a trend axis would dress query order up as a
  threshold axis; the HTML report's query-keyed trends grid replaces them.
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
  `density_path_bottlenecks.npy`, `density_meta.json`); the comparison
  report embeds the two-panel figure as an interactive Plotly chart whose
  grey dashed vertical guides mark the vertical (per-threshold) rows'
  thresholds and whose grey dotted horizontal guides mark the horizontal
  (density-matched) rows' levels — hover a guide for its row id.
  `comparison_visualizations/density_alignment_threshold_curves.png` stays
  the static export of the same figure. A dataset without a density
  capture is EXCLUDED from the horizontal rows instead of vetoing them:
  those rows carry a `partial_datasets` marker, the run notes say
  `[density partial]`, and the report labels them. Every threshold
  instance also appends its console trace to
  `dataset_data/{dataset}/run_log.txt`, so a silent fetch failure is
  diagnosable from the run folder alone. In
  auto mode, `comparison_results/density_alignment_best_matches.csv`
  additionally holds the measured vertical/horizontal aligned rows, and
  `[auto threshold]` / `[density]` warnings mirror into the run guide. A
  partially-resolved auto mode says so instead of quietly returning a
  Standard report: an `[auto threshold] auto mode did NOT resolve` /
  `resolved PARTIALLY` block in the run notes, `auto_mode_status` in
  `run_manifest.json`, and a banner in the report's density section. In
  that case the density-matched rows are still exported but only the
  vertical spine runs as queries, and the aligned-rows card names the
  advisory ids. When
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
- The Cross-Dataset Comparison tab's **Type Mapping** panel carries the row-based
  bodyId-level evidence of every 1-to-N split: per-branch linker-refined
  bodyId pools and label-vote provenance (curated vs `auto:` — e.g. the
  BANC `aMe24` bridge: 1 auto vote vs `s-LNv_a`: 2), a collapsed
  "Per-type breakdown" expansion for multi-type previews, and
  `mapping_support` / `support_*` columns in the mapping exports. A
  separate informational split-verification view (from the
  validate-expand-visualize verification machinery) may summarize a
  per-neuron partition. Strictly informational — none of it changes the
  analysis selection or run exports.
- **Same-name-first suspects** (plan-ui-type-mapper-alignment): when a
  fan-out's candidate set contained the queried type's own name, the mapper
  selects that candidate and demotes the rest to *suspects*. The panel marks
  such a row with a `⚠ suspects (N)` badge in its own **Suspects** column —
  on the pair-card mapped-pairs table, the forward/backward Type coverage
  tables, and the per-type breakdown. **Hovering the badge** explains the flag
  and lists the rivals (rival · own 1-to-1 pair
  · votes · reverse target · rival pair status); a **collapsed `Suspects`
  expander** below the table carries the identical facts, so the hover is
  additive and never the only route to the evidence. The pair's Type coverage
  expansion title is data-driven (`1-to-N fan-out` / `N-to-1 fan-in` /
  `all 1-to-1`, plus `· ⚠ N suspect pair(s)`), and a pair kept unmapped for
  this reason says so in the
  Orphan list instead of looking unmapped for no reason. The
  "See available neurons" viewer (cross-dataset mapping ON) shows the same
  state — three-way annotations (curated relation / same-name-first
  selection / bare-name echo) plus a collapsed suspects expander. The full
  per-rival record is `auto_type_mapping_suspects.csv`; labels state
  observations (`own_1to1_pair` / `no_own_1to1_pair`), never verdicts, and
  nothing is merged — the custom label mapper is the inclusion path.
- Use `parallel=True` with a bounded `max_workers` for many datasets; start with
  `skip_bodyId=True` and `max_interlayer=2`.


## Connection-ratio thresholds (v1)

Threshold Basis = Connection ratio accepts FLOAT tiers (0<t≤1) in Standard, Combinations AND Auto; delegate folders use the `minratio_{decimal}` grammar; Auto's density bootstrap/alignment runs on the float ratio tiers (float64 arrays, distinct-tier ladders). Ratio tiers are dimensionless fractions of each post neuron's total input — like-for-like across datasets. Replay Paths applies under this basis too (float slices); Auto-extend Collapsed Thresholds (F7) runs a float k × τ ladder capped at the (0, 1] ceiling. Export notes: query manifests/views carry the float tiers verbatim (`threshold_combinations.csv`, `effective_thresholds.json`), and `type_resolution_union.csv` absence diagnoses compare the max CONNECTION RATIO (w ÷ the post's full-table total incoming) against the tier, with the strongest leg spelled out in synapses.
