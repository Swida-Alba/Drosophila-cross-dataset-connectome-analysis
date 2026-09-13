# morph_cross_dataset — Cross-Dataset Morphology (shared helper)

Module `src/comparison/morph_cross_dataset.py`. Shared implementation for
two features (vector_v2 only — NBLAST is never used cross-dataset):

- **Morph qualification** for Find Homolog: the visualized top-N of a
  connectivity search scored against the transformed query and gated by a
  per-query null bar.
- **Cross-Dataset comparison** (`CrossDatasetMorphComparer`): queried
  neurons compared pairwise across datasets (drives the Morphology tab →
  Comparison sub-tab when two or more datasets are selected).

Scoring always goes through the production
`morphology.compute_morph_similarity_vs_queries(..., compute_nblast=False)`
— the block-weighted whitened vector_v2 scorer in the target's render
space. Dataset scope: **FAFB / male-cns / BANC** only (every other dataset
has no bridging transform within the 2-hop accuracy guard and is refused
with an explicit reason). BANC carries a standing reliability warning
(mixed public skeleton sources) and needs a one-time population-artifact
bootstrap.

## Dataset scope and availability

```python
from comparison.morph_cross_dataset import (
    dataset_scope, check_pair_availability, validate_cross_dataset_sets)

dataset_scope("banc_v888")
# {'ok': True, 'family': 'BANC', 'native_space': 'BANC',
#  'render_space': 'BANC', 'warnings': [...], 'reason': ''}

check_pair_availability("flywire_FAFB_v783", "male-cns:v1.0")
# {'ok': True, 'hops': 2, 'chain': 'FLYWIRE -> JRCFIB2022M -> ...', ...}

validate_cross_dataset_sets(["flywire_FAFB_v783", "manc_v1_2_1"])
# raises ValueError: refused (dataset or pair outside the ≤2-hop guard)
```

## CrossDatasetMorphComparer

```python
from comparison.morph_cross_dataset import CrossDatasetMorphComparer

comparer = CrossDatasetMorphComparer(
    datasets=["flywire_FAFB_v783", "male-cns:v1.0"],
    query=["APDN3", "SLP249"],       # types/patterns resolved per dataset
    output_dir="local_data/morph_cross_dataset",
    max_members_per_type=25,         # members sampled per type and dataset
    null_k=200,                      # seeded random-target null per pair
    reference_template="male-cns:v1.0",  # scene frame (scores are per-pair)
    scene_members_per_type=3,
    visualize=True,                  # one overlay scene per run
    fetch_online=True,
    verbose=True,
)
result = comparer.run()   # {"output_folder", "files", "warnings"}
```

Outputs per run folder: `overview.csv` (per queried type: the best target-type cell with
its name + the pair baseline), `report.html`, `parameters.json`, `README.txt`,
`<SRC>_to_<TGT>/{bodyid_scores.csv,type_matrix.csv,null_baseline.json}`,
and `plot-3d_*/` overlay scenes in the reference render space.

Semantics:

- **Pure comparison** of the queried neurons — no candidate pool, no
  full-dataset pre-rank. Types resolve per dataset by same-name/pattern
  over the dataset's neuron frame; missing types surface as explicit
  empty rows.
- **Null baseline**: each ordered pair reports the p95/median of the same
  queries against `null_k` seeded random target neurons (dataset-level
  seed — the sample and its render-space vectors are shared across
  queries and runs via a sidecar under
  `cache/<ds>/find_similar/morphology/`).
- Scores are comparable within a dataset pair (one render frame), not
  across pairs; the report states each pair's frame.

## Morph qualification (Find Homolog)

`HomologFinder.find_homologs_multi(..., morph_qualify=True,
morph_null_k=200, morph_bar_offset=0.0)` pools the would-be visualized
pairs of every resolved type into one scoring call, then:

- target candidates are pre-fetched into the shared raw cache (offline-first,
  per-neuron isolation) so uncached NeuPrint/BANC targets still receive a
  morph verdict;
- bars each query at `null_p95 + bar_offset`;
- excludes failing candidates from the rendered scenes;
- annotates `bodyid_results.csv` (`morph_v2`, `morph_null_p95`,
  `morph_z`, `morph_qualified`) on scored rows only and writes
  `results/morph_qualification.json`;
- with the option off, output is byte-identical.

Reusable pieces (also used by the finder):

```python
from comparison.morph_cross_dataset import (
    select_visualized_pairs,      # mirror the scene's top-N selection
    qualify_visualized_pairs,     # score pairs + shared null, gate by bar
    merge_morph_columns,          # annotate a results frame
    filter_qualified_top_matches, # scene exclusion (returns excluded rows)
    MorphQualification,           # .is_qualified(src, tgt), .bar(src), .z_score
)
```

## Population-artifact bootstrap (BANC, male-cns v0.9)

```python
from comparison.morph_cross_dataset import (
    population_artifacts_ready, ensure_population_artifacts)

ensure_population_artifacts("banc_v888", sample_k=300)  # offline, one-time
# samples locally cached skeletons, vectorizes with the cache's own row
# recipe, persists skeleton__vectors_v2.parquet + meta_v2.json + whitener
# so find_similar_dataset_cache_v2(...).load() succeeds (full scorer
# parity). CrossDatasetMorphComparer runs it automatically for BANC.
```
