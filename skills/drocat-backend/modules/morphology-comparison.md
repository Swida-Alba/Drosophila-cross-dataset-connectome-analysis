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
    max_members_per_type=25,
    max_total_neurons=200,               # nblast hard-caps at 30
    output_dir="/abs/output/morph_cmp",  # default local_data/morphology_comparison/
    saveas="",
    generate_heatmaps=True,
    show_figures=False,
    use_cache=True,
    verbose=True,
    n_workers=8,
)
results = comparer.run()   # {"output_folder", "types_compared",
                           #  "neurons_compared", "files"}
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


- **Resolution**: exact type names win over pattern interpretation; numeric
  tokens are bodyIds resolved through `_load_neuron_type_map`; patterns
  (`aMe.*`) full-match against dataset type names. Each resolved type is one
  matrix row, members capped per type and in total.
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
  (the forward NBLAST score is asymmetric). Type means exclude
  contralateral pairs (`_dataset_soma_side_map`); the bodyId matrix keeps
  every pair. Capped at 30 total neurons.
- **Outputs**: `type_level/type_similarity_{method}.csv`,
  `bodyid_level/bodyid_similarity_{method}.csv`, `members.csv`,
  `visualization/heatmap_*.html` (VisPath, plotly fallback),
  `report.html`, `parameters.json`, `README.txt`. Completion log line:
  `[MorphologyProfileComparer] Output: <run folder>` (parsed by the UI
  runner).

## Test seam

`tests/core/test_morphology_comparison.py` exercises the whole pipeline
hermetically: a fake vector cache (synthetic 256-dim rows, identity
whitening), a stubbed `NBlaster` + fake dotprops, and a monkeypatched
heatmap fallback.
