# morph_cross_dataset — Cross-Dataset Morphology (shared helper)

Module `src/comparison/morph_cross_dataset.py`. Shared implementation for
two features (vector_v2 only — NBLAST is never used cross-dataset):

- **Morph qualification** for Find Homolog: the visualized top-N of a
  connectivity search scored against the transformed query (identity for
  same-dataset runs) and gated by a per-query bar (null percentile or
  mapping-referenced floor).
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
    generate_heatmaps=True,          # standalone VisPath heatmaps per pair
    fetch_online=True,
    verbose=True,
)
result = comparer.run()   # {"output_folder", "files", "warnings"}
```

Outputs per run folder: `overview.csv` (per queried type: the best target-type cell with
its name + the pair baseline), `members_summary.csv` (compared member counts per
type/dataset), `report.html`, `parameters.json`, `README.txt`, per pair
`<SRC>_to_<TGT>/results/{morph_type_matrix.csv,morph_bodyid_matrix.csv,morph_bodyid_scores.csv,null_baseline.json}`,
per pair `<SRC>_to_<TGT>/visualization/heatmap_morph_<SRC>_to_<TGT>_{type,bodyid}.html`
(VisPath interactive, shared report_kit — open with square cells locked), and `plot-3d_*/` overlay scenes in the
reference render space. `report.html` uses the same tabbed generator as the
connectivity-profiling export (shared `comparison.report_kit`): scrollable
overview table with frame-asymmetry disclosure → pair tabs →
Type/BodyId level tabs → Ward-clustered Plotly cards (square cells via
explicit-width sizing for small matrices) with CSV + VisPath editor
links; Plotly.js is embedded so it renders offline. The bodyId-level matrix
carries the raw per-neuron scores under tree-legend axis labels
(`{bodyId}_{instance}` / `{bodyId}_{type}_{L|R}`). vector_v2 renders on the
diverging [-1, 1] scale (a whitened cosine can be negative).

Semantics:

- **Pure comparison** of the queried neurons — no candidate pool, no
  full-dataset pre-rank. Types resolve per dataset by same-name/pattern
  over the dataset's neuron frame; missing types surface as explicit
  empty rows.
- **Null baseline**: each ordered pair reports the p95/median of the same
  queries against `null_k` seeded random target neurons (dataset-level
  seed — the sample is shared across queries and runs).
- **Target-vector store**: the render-space 256-dim vectors live in one npz
  sidecar per (dataset, render space) under
  `cache/<ds>/find_similar/morphology/cross_dataset_targetvec_<space>.npz`,
  shared by the null sample AND the candidates, because preparing a target
  neuron costs 0.412 s (skeleton load + render transform + vectorize) and every
  repeat run used to pay it again. A row is only reused when its hemisphere is
  known with it (a stored vector without a side still forces the render, so
  `pair_side` cannot change), and two layers invalidate a row: a file signature
  over the population bounds + render space + the V2 cache version, and the
  backing skeleton's `(mtime_ns, size)` — so a healed or re-fetched skeleton
  cannot score stale geometry while the report claims the new one. Rows whose
  skeleton has no resolvable file (a FAFB target served from the release
  bundle) fall back on the file signature, and what a run loaded / dropped as
  stale / saved is published on `MorphQualification.vector_cache` rather than
  left felt. `--purge-sidecars` in
  `scripts/maintenance/purge_legacy_simp90_cache.py` clears both generations of
  the sidecar.
- **Offline runs** (`fetch_online=False`): FAFB sources still resolve
  network-free from the local release sources (repair caches, raw cache,
  healed zip — the CAVE extrusion pass is skipped), so strict-offline
  FAFB→X and X→FAFB directions score fully; BANC's public-release stage
  fetches online and stays gated (raw-cache fallback only). Bridged FAFB
  overlay layers render as-is: their neurons are injected pre-transformed
  and excluded from the target dataset's fetch/preparation phases (with
  one tree-legend leaf per member). Any scene failure fails soft and
  keeps the comparison.
- Scores are comparable within a dataset pair (one render frame), not
  across pairs; the report states each pair's frame. A→B and B→A are
  **not symmetric** — each direction transforms into a different render
  space and uses a different whitening basis, so scores and baselines
  differ. The overview table includes a disclosure note.

## Morph qualification (Find Homolog)

`HomologFinder.find_homologs_multi(..., morph_qualify=True,
morph_null_k=200, morph_bar_offset=0.0, morph_mode='null',
morph_level=95)` pools the would-be visualized pairs of every resolved
type into one scoring call, then:

- works cross-dataset (bridging transform) and same-dataset (identity
  chain) alike; the target must pass `dataset_scope` (FAFB /
  male-cns:v1.0 / BANC) either way;
- target candidates are pre-fetched into the shared raw cache (offline-first,
  per-neuron isolation) so uncached NeuPrint/BANC targets still receive a
  morph verdict;
- bars each query per `morph_mode`: `'null'` — `null_p{morph_level} +
  bar_offset` of the shared seeded null sample (query bodyIds are
  excluded from the sample on intra-dataset runs, where the queries
  live in the target universe); `'mapping_ref'` — the mapper's
  target-side branch pools (native pool floor binding at >= 2 refs,
  Track-A `B_b - Δ` backup, per-source null fallback; same-dataset
  pairs have no mapper bridges — the UI disables the option when
  Target = Source, and a forced same-dataset call falls back to the
  null bar with a logged note);
- excludes failing candidates from the rendered scenes;
- annotates `bodyid_results.csv` (`morph_v2`, `morph_null_p95`,
  `morph_z`, `morph_bar_kind`, `morph_bar`, `morph_null_level`,
  `morph_qualified`) on scored rows only and writes
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
