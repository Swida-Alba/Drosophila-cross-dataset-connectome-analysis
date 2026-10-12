# comparison.profile_comparator — Profiling, Homologs, Similarity

Module `src/comparison/profile_comparator.py`. Three main classes:

- `ConnectivityProfileComparer` — build/compare connectivity profiles for a query
  set (drives the Connectivity tab → Comparison sub-tab).
- `HomologFinder` — homolog discovery by connectivity-profile similarity,
  cross-dataset or intra-dataset (Target = Source; drives the Connectivity
  tab → Find Homolog sub-tab).
- `ProfileComparator` — lower-level comparator producing `ComparisonResult` and
  using `DEFAULT_SCORE_WEIGHTS`.

## ConnectivityProfileComparer

```python
from comparison.profile_comparator import ConnectivityProfileComparer

comparer = ConnectivityProfileComparer(
    query=["aMe12", "aMe10"],           # or a single query for a custom group
    dataset="male-cns:v0.9",            # or list of datasets via `datasets`
    top_k=15,
    top_m=5,
    min_synapse_threshold=3,
    direction="both",
    output_dir="/abs/output/profiles",
    generate_heatmaps=True,
    show_figures=False,
    skip_bodyId_level=False,
    verbose=True,
    use_cache=True,
    aggregation_level="type",           # "type" | "bodyid" | "custom"
    ensure_cache_complete=False,
    custom_mapping_file=None,           # required when aggregation_level="custom"
    label_mapping_file=None,            # tier-0 preset; does NOT change the rows
    group_partners_by_mapping=False,    # fold the partner space into preset labels
)
results = comparer.run()
```

`custom_mapping_file` and `label_mapping_file` take the same LabelMapper preset
JSON. The first makes the preset's groups the ROWS (it forces
`aggregation_level="custom"`); the second only overlays label identity on
whatever rows the query already produces, and is the parameter the
cross-dataset (multi-`datasets`) lane needs to compare settled labels.
`group_partners_by_mapping` collapses the PARTNER side of every type-level
profile through the same preset (weights summed per label, ranks rebuilt); it
requires one of the two mapping files and is ignored on bodyId-level vectors.

## HomologFinder

```python
from comparison.profile_comparator import HomologFinder

finder = HomologFinder(
    source="aMe12",                     # per query (loop for many)
    source_dataset="male-cns:v0.9",
    target_dataset="hemibrain:v1.2.1",
    output_dir="/abs/output/homologs",
    top_n=30,
    top_k=15,
    top_m=5,
    min_shared_partners=3,
    vector_prune_fraction=0.0,
    similarity_metric="rank_union",     # or other metrics
    vector_prefiltering=True,
    include_untyped_partners=False,
    min_synapse_threshold=3,
    use_cache=True,
    saveas="",
    ensure_cache_complete=False,
    output_folder_prefix="",
    visualize_skeleton=False,
    visualize_top_n=0,
    visualization_settings={},
    use_auto_type_mapping=True,
    verbose=True,
)

results = finder.find_homologs_fast()   # fast adjacency search (UI default)
# results = finder.find_homologs()        # slower comprehensive search
# results = finder.find_homologs_multi(   # multi-query UI entry; optional
#     source=["aMe12", "aMe10"],
#     source_dataset="male-cns:v1.0", target_dataset="flywire_FAFB_v783",
#     morph_qualify=True,        # cross-dataset vector_v2 qualification of
#     morph_null_k=200,          # the visualized top-N against a seeded
#     morph_bar_offset=0.0)      # null bar; see morph-cross-dataset module
# results = finder.find_novel_homologs()  # novel-only homologs (Similar Neurons)
# results = finder.find_homologs_intra_dataset(...)
```

## ProfileComparator / ComparisonResult

`ProfileComparator` is a static-helpers class (no constructor arguments). The real
methods are `compare_profiles(...)` and `compare_profiles_simple(...)`, which return
`ComparisonResult` and use `DEFAULT_SCORE_WEIGHTS` unless overridden.

```python
from comparison.profile_comparator import ProfileComparator, ComparisonResult, DEFAULT_SCORE_WEIGHTS

result: ComparisonResult = ProfileComparator.compare_profiles(...)
# or ProfileComparator.compare_profiles_simple(...)
```

## Notes

- `ConnectivityProfileComparer.run()` gates on **neurons in scope (≥2), not
  rows**, and raises `ValueError` on refusal (returning an error dict would let
  the runner exit 0 and the panel report Completed with no files). At
  `aggregation_level='bodyid'` the main matrices ARE the bodyId comparison: they
  are filed under `bodyid_level/results/bodyid_similarity_*`, `type_avg_bodyid_*`
  is folded from those same scores, and neither `type_level/` nor
  `profiles/aggregated/` is created because nothing was pooled.
  With `custom_mapping_file`/`label_mapping_file`, bodyId-level axes read
  `{bodyId}_{label}({native})` on preset-resolved ids — the parenthetical
  is dropped when the label already is the native name; unmapped ids keep
  the legacy form.
  `skip_bodyId_level` cannot skip the bodyId pass on a run left with one
  comparison row while that row stays inside the 1000-bodyId auto budget (the
  pair loop is quadratic, so past it the skip stands and the log says so).
- Multi-dataset runs pair a queried item across datasets in
  `_build_anchor_profiles`: the item becomes an anchor only when its licensed
  label is a profile row in **≥ 2** datasets, so a label present in one volume
  still yields its intra rows but no cross row. Names resolve EXACTLY first —
  a tier-0 user label is a literal and is never regex-expanded, and a type
  name containing `+`, `.` or `(` keeps its own row instead of expanding into
  the family its spelling happens to `re.match`; only an item with no exact
  row anywhere is treated as a pattern. Every item that forms no anchor is
  recorded in `parameters.json → dropped_inter_anchors`, repeated in
  `README.txt`, logged, and shown as a report meta chip, so a partial
  cross-dataset matrix cannot read as a complete one.
- The pooled level's folder names its axes (`_aggregate_names()`):
  `type_level/type_similarity_*` at `type`, `group_level/group_similarity_*` at
  `custom`. The internal level key stays `'type'`; only the artifact names move.
- Morphology enrichment of homolog results (the former
  `morph_v2_similarity`/`morph_nblast` columns) is disabled; results carry
  connectivity scores only.
- `vector_prefiltering=True` speeds large searches via vector screens.
- `find_homologs_fast` is the reproducible first pass; escalate to `find_homologs`
  only when the fast adjacency search is insufficient.
