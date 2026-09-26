# morphology_comparison — Intra-Dataset Morphology Comparison

Module `src/morphology_comparison.py`. One class:

- `MorphologyProfileComparer` — N×N morphology comparison of 2+ queried
  neurons within ONE dataset (drives the Morphology tab → Comparison
  sub-tab with exactly one selected dataset; two or more datasets dispatch
  to `morph_cross_dataset.CrossDatasetMorphComparer` instead). Builds on
  `morphology.MorphologyComparer` / `SkeletonVectorCacheV2` without
  modifying them.

## MorphologyProfileComparer

```python
from morphology_comparison import MorphologyProfileComparer

comparer = MorphologyProfileComparer(
    dataset="male-cns:v1.0",             # one dataset; BANC deferred (similarity validation pending)
    query=["aMe12", "aMe10", "aMe.*"],   # types, bodyIds, or regex patterns
    method="vector_v2",                  # "vector_v2" | "nblast"
    aggregation_level="type",            # "type" | "bodyid" | "custom group"
    custom_mapping_file=None,            # LabelMapper preset; forces "custom"
    max_members_per_type=25,
    max_total_neurons=200,               # truncation cap; nblast refuses > 30 neurons
    output_dir="/abs/output/morph_cmp",  # default local_data/morphology_comparison/
    saveas="",
    generate_heatmaps=True,
    show_figures=False,
    use_cache=True,
    verbose=True,
    n_workers=8,
)
results = comparer.run()   # {"output_folder", "aggregation_level",
                           #  "rows_compared", "neurons_compared", "files"}
```

## Unified skeleton acquisition

Both comparison modes share one skeleton contract: local raw cache first,
online fetch (persisting into the shared cache) when ``fetch_online`` is
on, explicit skip notes when offline and uncached. With
``fetch_online=False`` and ``visualize=True``, the 3D scene renders only
members whose raw skeletons are locally cached — it never triggers the
declined online fetch (which would otherwise abort the whole scene for
NeuPrint datasets missing a token).

## Semantics


- **Resolution** returns `{row label: [member bodyIds]}`, and the row label
  is what `aggregation_level` selects. Exact type names win over pattern
  interpretation; numeric tokens are bodyIds resolved through
  `_load_neuron_type_map`; patterns (`aMe.*`) full-match dataset type names.
  At `type` a queried bodyId folds into its type; at `bodyid` each neuron is
  its own row (keyed by the tree-legend label from `_display_labels`, with a
  type suffix if two neurons would collide); at `custom` rows are the preset's
  source-side groups, read through the shared
  `naming_utils.load_labelmapper_source_groups` that the connectivity comparer
  uses too, so the two tabs cannot disagree about what a group is.
- **Pinning**: a bodyId named in the query is protected from both
  `max_members_per_type` and `max_total_neurons` — those caps remove other
  members only. `_check_population` refuses below two rows and names the level
  (a two-bodyId query folding to one type says to switch levels).
- **vector_v2**: warms the per-dataset `SkeletonVectorCacheV2` via
  `vectors_for`; with `fetch_online=True` (default) cache misses are
  fetched through the API — `fetch_skeletons_on_demand_batch` on NeuPrint,
  `load_local_release_skeletons` (FAFB repair caches → healed zip → CAVE)
  on local releases — and re-vectorized with the cache's own `_vectorize_neuron`,
  mirroring Find Similar's cache-direct contract. It then scores the
  standardized + ZCA-whitened rows with
  `v2_pairwise_matrix` (shape/spatial 0.30/0.70) — the exact Find Similar
  space. Neurons without local skeletons carry NaN cells and are reported
  in `members.csv` as `no vector`. The BANC branch of the loader chain is
  unreachable from here: BANC morphological comparison is deferred
  (vector-quality validation pending), so BANC datasets are rejected
  before any skeleton is fetched.
- **nblast**: reuses `MorphologyComparer`'s dotprops pipeline
  (`_dotprops_for_ids`), scores both orientations per pair and averages
  (the forward NBLAST score is asymmetric). Aggregate means exclude
  contralateral pairs (`_dataset_soma_side_map`); the bodyId matrix keeps
  every pair. Refuses populations over 30 **neurons**, judged on the capped
  population (`min(total, max_total_neurons)`) — so a small `max_total_neurons`
  is honoured rather than rejected — and the UI drops the bodyId level while
  NBLAST is selected, because there each row is one neuron.
- **Outputs**: `bodyid_level/bodyid_similarity_{method}.csv` always;
  `type_level/type_similarity_{method}.csv` at the type level and
  `group_level/group_similarity_{method}.csv` at the custom group level —
  `_aggregate_name()` keeps the folder/file/heatmap names honest about the
  axes, and at the bodyId level the aggregate is neither computed nor written
  (no empty directory is created). Plus `members.csv`,
  `visualization/heatmap_{type|group|bodyid}_{method}.html` (VisPath via the
  shared `report_kit`, plotly fallback), `report.html`, `parameters.json`,
  `README.txt`.
  `report.html` uses the same tabbed generator as the connectivity-
  profiling and cross-dataset morphology reports (hero header,
  Type-or-Group / BodyId level tabs, Ward-clustered cards with CSV + VisPath editor
  links, compared-neuron/parameter details, scene link; Plotly embedded
  so it renders offline; heatmaps open/render with square cells). Scale per method: vector_v2 diverging [-1, 1]
  (whitened cosine can be negative), NBLAST positive [0, 1].
  Completion log line: `[MorphologyProfileComparer] Output: <run folder>`
  (parsed by the UI runner).

## Test seam

`tests/core/test_morphology_comparison.py` exercises the whole pipeline
hermetically: a fake vector cache (synthetic 256-dim rows, identity
whitening), a stubbed `NBlaster` + fake dotprops, and a forced
VisPath-unavailable run that exercises the kit's plotly fallback. It pins
each aggregation level (bodyId rows, the suppressed type matrix, custom-group
rows), the pinning rule, and the NBLAST effective-population gate.
`tests/ui/test_morphology_comparison_tab.py` covers the UI wiring: the
NBLAST option guard, the grouping board's visibility, and that the intra-only
controls hide for two datasets.
