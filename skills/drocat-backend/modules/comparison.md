# comparison — Cross-Dataset Comparison

Cross-dataset comparison module (`src/comparison/`). The primary entry points are
`ComparisonParameters` (a dataclass holding all settings) and
`ComparisonAnalyzer` (the orchestrator). `quick_compare` is the one-liner;
`CrossDatasetTypeMapper` and `LabelMapper` handle naming differences between
datasets.

## ComparisonParameters + ComparisonAnalyzer

```python
from comparison import ComparisonParameters, ComparisonAnalyzer, quick_compare

params = ComparisonParameters(
    datasets=["hemibrain:v1.2.1", "male-cns:v0.9"],
    source_neurons=["aMe12"],           # shared across all datasets
    target_neurons=["PPL101"],
    output_folder="/abs/output/comparison",
    comparison_mode="path",             # or "edge"
    path_mode="all",                     # "all" | "shortest"
    max_interlayer=2,
    thresholds=[1, 3, 5, 10],
    threshold_mode="standard",          # "standard" | "combinations"
    # Combination mode example (replace thresholds above):
    # threshold_dataset_order=["hemibrain:v1.2.1", "male-cns:v0.9"],
    # threshold_combinations=[
    #     {"id": "combo_001", "label": "density match",
    #      "thresholds": {"hemibrain:v1.2.1": 3, "male-cns:v0.9": 8}},
    # ],
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
    max_workers=None,
    separate_hemispheres=False,
    keep_only_hemisphere_conserved_connections=False,
    symmetry_analysis=False,
    find_reciprocal=False,
    drop_untyped=True,                  # Drop Untyped Neurons: applied post label-mapping
    overall_mapping_json=None,          # custom cross-dataset mapping
)

analyzer = ComparisonAnalyzer(params, verbose=True)
analyzer.run_comparison()               # returns result dict; runs path/edge analysis
analyzer.export_results()               # writes tables/reports
analyzer.generate_report()              # summary report
```

Quick one-liner:

```python
results = quick_compare(
    datasets=["hemibrain:v1.2.1", "male-cns:v0.9"],
    source_neurons=["MBON14.*_R"],
    target_neurons=["KCg-d.*_R"],
)
```

## Key methods on ComparisonAnalyzer

| Method | Purpose |
| --- | --- |
| `run_path_analysis(...)` | Path-based per-dataset analysis. |
| `run_edge_analysis(...)` | Edge-based (strong direct edges) per-dataset analysis. |
| `run_all_analyses(skip_existing=True)` | Run every configured analysis, skipping completed ones. |
| `run_comparison(skip_existing=True)` | Main orchestrator. |
| `generate_report(output_path=None)` | Summary report. |
| `export_results(output_dir=None)` | Write per-dataset tables, summaries, conserved-path HTML. |
| `generate_html_report(output_path=None)` | HTML report. |

## Type mapping across datasets

```python
from comparison.type_resolver import resolve_valid_targets, canonical_merge_key
from comparison import LabelMapper

# Preferred: the shared validity-aware resolver (what the panel, viewer,
# homolog finding, and profile comparison all use) — preserves status,
# fails closed on conflicts, expands valid splits.
res = resolve_valid_targets(mapper, 'MeVPLo2', 'male-cns:v1.0', 'flywire_FAFB_v783')

labeler = LabelMapper()                # explicit per-dataset override (wins)
```

`ComparisonParameters.for_run_folder(run_dir)` pins parameters to an
existing run folder (in-place resume / re-export — `from_dict` alone
re-timestamps the folder). `ComparisonParameters.auto_type_mapping=True` (plus `overall_mapping_json`) is the
usual way to resolve differing type names; `LabelMapper` is the manual override.
`CrossDatasetTypeMapper.get_mapped_type()` / `resolve_type_across_datasets()`
are **compatibility-only** (no status/provenance; `None` on splits/conflicts) —
new code should use `comparison.type_resolver`.

### Tabbed report (`comparison/report_tabbed.py`, opt-in)

`report_layout` ('legacy' default) transforms the legacy single-page
report into page tabs (Overview / Combos per query / Type Mapping /
Plots / Matrices & Networks / Cross-views / Notes) via
`build_tabbed_report(analyzer, legacy_html, run_dir)`; `'legacy'`
passthrough, `'both'` also writes `comparison_report_legacy.html`.
Combos page = per-query dashboard (KPIs, involved types with
anchor_group/auto_only flags, deep-link buttons); Plots page = the four
summary metrics SPLIT into per-threshold and density-matched charts
(built from `comparison_report_used_data/*_by_query.csv`). Tag-soup
safe: sections are boundary-anchored (wrapper + section-header divs),
panes div-rebalanced, and each page renders from an inert `<template>`
clone (scripts re-execute on activation = deferred vis/plotly draw).
`_generate_html_content` applies the layout; `_generate_legacy_html_content`
is the untouched reference generator.

### Query-anchored merge policy (`comparison/merge_policy.py`)

Auto-mode comparison runs build a per-run `MergePolicy` (cached as
`analyzer._merge_policy`, lazily via `_merge_policy_or_none()`) that owns
the alignment merge keys: chip-anchored group labels replace the canonical
namespace. Anchor selection is resolution-based (all chips into one
dataset's naming → that dataset; all clean same-name → shared naming; mixed
→ `[type granularity]` warning + minimal inseparable leaves). A "1"-side
chip merges all its 1-to-N branches (Decision 9: no vote threshold);
leaf-anchored chips cover only their own branch; a leaf claimed by two
groups merges with NEITHER (`[merge fan-in]` — with an explicit
parent-partial note when a queried parent chip is a claimant). Fan-in pairs
are also PRUNED from the claimant groups' members/branches, so the display
lane (type-mapping table, `topology_dict`, `names_by_dataset`) agrees with
`key_map` (2026-09-15: CL317 showed inside both aMe26/aMe9 rows while the
counts excluded it). The whole construction is
chip-order invariant (canonical group ids, sorted warnings). It is applied
at the metrics alignment via `label_mapper=` (a policy-synthesized
`LabelMapper`; user mappings win by construction) plus
`merge_policy=` (group-label keys + anchor-namespace fallback); the report
gains a Resolution-topology subsection, `type_resolution_topology.json` is
exported, and `auto_type_mapping.csv` gains additive
`anchor_group`/`auto_only` columns (`anchor_group` only when EVERY
non-empty endpoint of the row is in that same group;
`comparison.merge_policy.row_anchor_group`). Auto-only BANC edges
(`[BANC auto labels]` warning, opened by a per-direction count summary)
come from `merge_policy.auto_only_edges(mapper, types, datasets)` —
evidence only, mappings stay valid. Neuron counts are keyed by group
labels with a `group_members` composition column.

### Mapper boundary (row-based evidence is carried, not consumed)

The mapper produces the mapping plus the row-based bodyId-level evidence
that backs it (`_bridge_provenance`: BANC label votes with curated/auto
provenance, linker values, release root-id pairs, FAFB additional-type
linker rows).  It **passes** that evidence — `get_mapping_support(...)`
accessor, `support` key on `get_mapping_decision`, `mapping_support`
column in `auto_type_mapping.csv`, `support_*` columns in
`auto_type_mapping_per_bridge.csv` — and never consumes it to gate or
verify a mapping.  Direct same-name pairs carry a pooled-identity record
(`same name (all bodyIds pooled)`; full populations, fully in-map)
EXCEPT when the pair sits inside a 1-to-N/N-to-1 structure — there the
evidence bridge resolves the bodyId-level resolution per branch.
Verification = the validate-expand-visualize pipeline
(connectivity scoring, morphology); any "tight mapping" filter is
consumer-side over these evidence surfaces.

### BodyId-level verification backend (`comparison/body_id_resolver.py`)

Connectivity-scoring machinery — TM VEV verification, housed in the
mapper layer (plan
`_plan/plan-bodyid-level-granularity-in-type-mapper.md`, Rev 3–5):

- Moved primitives live here (`expanded_vector`, `score_one_candidate_fast`,
  `_SideStats`, `scan_source`, `build_target_vectors`, `prep_target_stats`,
  quality gate, caliber/hemisphere loaders); `mapping_validation` imports them
  back — never re-implement bodyId scoring elsewhere.
- Connectivity API (verification only, accessed directly by
  `mapping_validation` — NOT via the mapper):
  `BodyIdResolver.assign_bodyids(body_ids, source_ds, groups,
  pools=None) -> AssignmentResult` scores a conflicted type's resolved
  pool across its 1-to-N branches (side-aware; gates
  `min_score`/`min_margin`/positivity).  The former mapper-facing
  wrappers (`body_id_resolver` property, `derive_split_groups`,
  `resolve_members_across_datasets`) were REMOVED from the type mapper
  (2026-09-14).
- Cost model: on-demand per-neuron profiles ONLY for resolved pool members
  (no bulk dataset profiling, no global scans). Dataset-scale scans
  (`build_target_vectors`/`scan_source`) are validation-only in usage.
- Wired surfaces (row-based, informational only): the Type Mapping panel
  shows branch pools + vote provenance, a collapsed per-type breakdown,
  and an **Export branch bodyIds** download
  (`mapping_branch_bodyids_*.csv`); the 'See available neurons' viewer
  notes when a candidate's parent family splits by neuron.  **Never
  consumed by mapping decisions** — the mapper's granularity surfaces
  are the row-based ones (§ Mapper boundary above).

## Type-mapping validate-expand-visualize (`comparison/mapping_validation.py`)

BodyId-level validation of an auto type mapping (CLI
`scripts/RunMappingValidation.py`); see the dedicated agent skill
`type-mapping-validation` for the full run recipe. Rev 3.12 essentials:

- **Category partition**, one ordered first-match per branch: tier
  (`matched`/`verified`/`borderline`/`unmatched`) > `sibling` (in-map
  bodyId of another branch) > `candidates` (out-of-map, invader or gap
  fire, morph-qualified) > `family` (out-map bodyIds of THIS branch's
  target type) > `relative` (candidate-type mates outside the map) >
  `suspicious` (aggressive-only deep window). `unmatched` is the tier's
  final else; there is no target-side `skipped`.
- **Modes nest**: `restrictive ⊆ family ⊆ aggressive` (single enum;
  `normalize_mode`). Chain-aware POOL widening is retired — family mode
  adds bins, it does not touch the tier.
- **Per-bodyId leaf token** (ordered): `{T}(out-map)` (the type is an
  in-map type — bodyId-level out-of-map) > `{T}>{src}` > `{T}(no_source)`
  > `untyped`; `(dup)` is a standalone tag.
- **Out-of-scope** (connectivity-qualified but morph-failed) rows are
  exported with `in_scope=False` / `morph_failed=True`, never rendered.
- Outputs live in `local_data/mapping_validation/…` (see
  `docs/OUTPUT_FILES.md` §9).

## Untyped-neuron drop (drop_untyped)

`drop_untyped=True` (default) removes edges touching untyped neurons from the
cross-dataset results. The predicate is the shared
`utils.label_utils.is_untyped_type_label` (empty label, Unknown/None/NaN
sentinel, all-digit bodyId-fallback label), but the analyzer applies it AFTER
standardized cross-dataset label mapping — the later, authoritative timing.
The delegated per-dataset `FindNeuronConnection` runs therefore execute with
`drop_untyped=False`, so only the comparison-level filter fires and no
per-dataset `data_details/` records are written.

Outputs, only when rows were dropped:

- `comparison_results/untyped_dropped_records.csv` — dropped rows with
  `dataset`, `threshold`, the connection columns, and `untyped_side`
  (`pre` / `post` / `pre+post`).
- Per-run counts appended to the run root's `user_warning_notes.txt`.

## Threshold query model

`threshold_mode="standard"` expands each scalar in `thresholds` into one
same-threshold query for every selected dataset. Use
`threshold_mode="combinations"` with `threshold_combinations` when a query
needs different thresholds per dataset. Custom combination mode requires at
least two selected datasets. Each row must contain exactly one positive
threshold for every selected dataset; it is a comparison identity, not a
per-dataset schedule. The union of cell values is only the deduplicated
raw-run/cache schedule.

Combination rows retain stable `id`/`label` values. Alignment, similarity,
reports, and presence matrices are keyed by that query row and never infer a
scalar from the union. Raw `(dataset, threshold)` jobs shared by multiple
rows run once and are referenced from each row.

## Pathfinding threshold/bottleneck provenance

Every delegated pathfinding threshold folder writes the shared provenance
block to `parameters.txt`, `all_attributes.json`, and
`data_details/parameters.csv`: `requested_threshold`, canonical
`applied_threshold`, `applied_threshold_source`, the effective
`strongest_first_budget` and bite flag, landing `tau`, `tau_canonical`, `w2`
(`strongest_dropped_bottleneck`), Edge Budget `edge_budget`/`w0`/`w1`,
`strongest_retained_bottleneck` (`W*`), and `paths_complete`.

The comparison root additionally exports:

- `effective_thresholds.json` — the UI/run-guide notice with one `runs` row
  per dataset and requested threshold, plus `queries`/`combinations` with
  requested threshold maps and applied per-dataset provenance.
- `comparison_results/pathfinding_provenance.csv` — the complete machine-
  readable row set. Use `applied_threshold` for the canonical equivalent Min
  Synapse Count; `tau` is the StrongestFirst landing/collapse bound, not
  necessarily the minimal applied threshold.
- `comparison_results/threshold_combinations.csv` — the canonical query
  manifest. It joins `query_id`, dataset, requested/applied threshold,
  StrongestFirst budget/tau, Edge Budget `w0`/`w1`, `w2`, `W*`, completeness,
  and raw-run/alias provenance.
- `comparison_results/threshold_sensitivity.csv` and
  `comparison_results/unified_summary.csv` — summary tables that retain the
  same provenance fields, including `skipped`/`duplicate_of` for tau-collapsed
  thresholds and untyped-drop counts where applicable. In combination mode,
  sensitivity rows carry `query_id`/`query_label`; adjacent-threshold
  retention is not inferred across unrelated query rows.
- `similarity_matrices/similarity_query_{query_id}.csv` and
  `similarity_matrices/similarity_by_query.csv` — query-keyed similarity
  exports for advanced combinations.

The Edge Budget is a lossy graph floor in `all` mode only. Shortest mode can
be bounded by the StrongestFirst path budget and report tau, but its Edge
Budget is ignored and `edge_weight_floor` remains empty.

## Raw threshold-schedule diagnostics and unsupported analyses

- `edge_density_per_threshold.csv`, `threshold_alignment_matrix.csv`, and
  `threshold_alignment_best_matches.csv` describe the raw execution/cache
  schedule over the sorted union of `(dataset, threshold)` cells. In
  combination mode every row carries
  `threshold_scope=raw_run_schedule_diagnostic`; never present these files as
  the Custom query comparison axis — join `threshold_combinations.csv`
  instead.
- **Density curves** (every pathfinding mode):
  `comparison_results/density_curves.csv` + `density_windows.csv` give each
  queried dataset a cone-scoped curve over `[w_start, w_star_measured]`
  (minimal applied threshold → measured `max(path bottlenecks)`), with
  endpoints classified typed/untyped/debris via
  `is_untyped_type_label`. The edge basis follows `drop_untyped`
  (`bodyId_edges_typed` on, `bodyId_edges_all_but_debris` off; debris ids
  absent from the curated table are always excluded). Raw capture in
  `dataset_data/{dataset}/_density/`; auto mode additionally writes
  `density_alignment_best_matches.csv` (runnable vertical/horizontal
  aligned rows — verticals are the like-for-like spine, horizontals the
  density-matched envelope). A dataset without a capture is EXCLUDED from
  the horizontal rows (rows carry `partial_datasets`, run notes log
  `[density partial]`) instead of vetoing them. The report embeds the
  curves as an interactive Plotly card with V/H guides (vertical = the
  per-threshold rows' thresholds, horizontal = the density levels; hover
  shows the owning row id). Zero-outdegree index markers are revalidated
  against the server (batched count query, once per process) before being
  trusted — poisoned markers from a failed fetch self-heal on the next
  run. Per-dataset console traces persist to
  `dataset_data/{dataset}/run_log.txt`; the run manifest carries a
  `dataset_coverage` block and the report opens a coverage callout when a
  configured dataset produced no data.
  `scripts/maintenance/repair_neuron_index.py` repairs a poisoned index
  (effective base+sidecar view; flips True+0 completion markers; dry-run
  default). An EMPTY connection pull is never trusted as proof of zero
  outdegree — large all-empty batches are refused at marking time. With
  `separate_hemispheres=True` the metrics canonical-map strips
  `_L/_R/_U` suffixes for the merge-key lookup and re-applies them
  (`hemi_aware=` threaded through the metrics alignment), so L and R stay
  distinct rows that still merge across datasets.
- `degree_*.csv`, `top_edges_*.csv`, and `unique_to_*.csv` are Standard-only
  exports (their scalar APIs would silently choose a union threshold in
  combination mode). Derive per-query variants from the query-keyed presence
  matrices if needed.
- Pathfinding comparisons disable ratio/traversal-probability filtering in
  both modes, so comparison visualization exports never create `by_ratio/`
  or `by_probability/` folders.

## Notes

- `comparison_mode="path"` uses the pathfinding engine (FindAllPath /
  FindShortestPath); `comparison_mode="edge"` preserves strong direct edges.
- `path_mode` selects per-pair minimum-hop vs all-paths behavior.
- `parallel=True` with a bounded `max_workers` speeds many datasets; start with
  `skip_bodyId=True` and `max_interlayer=2`.
