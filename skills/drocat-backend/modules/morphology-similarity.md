# morphology — MorphologyComparer

Morphological similarity search (module `src/morphology.py`). The main class is
`MorphologyComparer` (query-vs-all similarity); `SkeletonVectorCache` manages the
skeleton vector cache used by morphological and `candidate_source` paths. BANC
is deferred for similarity (the public L2/full skeletons still need
vector-quality validation — BANC 3D visualization works) and raises a clear
error instead of silently reusing the FAFB pipeline.

## MorphologyComparer — key params

```python
from morphology import MorphologyComparer

comparer = MorphologyComparer(
    query="aMe12",                      # str/int query (per query; loop for many)
    dataset="male-cns:v0.9",            # non-BANC only
    level="auto",                       # "auto" | "type" | "bodyid"
    method="vector_v2",                 # "vector_v2" | "nblast"
    candidate_cap=500,                  # screen candidates entering comparison
    candidate_source="auto",            # "auto" | "roi" | "combined" | "profile" | "cache"
    visualize_top_n=0,                  # >0 to render top-N skeletons
    visualize_by="type",                # or "bodyId"
    min_weight=3,                       # connection-cache screen threshold
    min_shared_partners=2,
    roi_filter=None,                    # or ["EB", "LH", "AL"]
    output_dir="/abs/output/similar_morph",
    saveas="",
    verbose=True,
    n_workers=8,
    use_cache=True,
    cache_fetched_skeletons=True,       # persist raw skeletons
)

results = comparer.find_similar()       # -> pd.DataFrame
```

The two **size knobs** shared by every candidate-source mode are `candidate_cap`
(bounds how many screened candidates enter the comparison) and `visualize_top_n`
(bounds rendering only). Results are never truncated — every compared candidate is
returned and written.

### method="vector_v2" — spatial/topological comparison (default)

`vector_v2` is the DEFAULT and only vector method (the legacy global-shape
`"vector"` method was removed). It keeps the search pipeline (screening,
pooling, output) and pairs the vector schema with a block-weighted scorer.
The 256-dim V2 vector (cache schema v4) = the V1 124 dims + 8 **shape
extras** (`sx_*`: node-radius stats, sibling-edge branch angles at branch
points, cable-length fractions by Strahler order) + a **spatial block**
(12-dim arbor ellipsoid from node-coordinate PCA + 96-dim cable-mass
histogram over a population-fixed bounding box + 16-dim 1-D cable-mass
profiles: 8 radial bins from the arbor centroid and 8 midline-distance
bins). The former 24-dim Laplacian topology block was REMOVED in v4: its
eigensolver caused an intermittent vectorization cost lottery (median
0.3 s, tail to ~400 s/neuron) while its 10% weight measurably changed
nothing in the user-facing ranking (production-run analysis, l-LNv +
aMe12 2026-08-29: dropping it leaves top-10 type overlap 9-10/10,
Spearman >= 0.96). ROI-distribution similarity is a candidate-SELECTION signal only
(`candidate_source=roi`/`combined`, reported per row as `roi_similarity`);
the former runtime-composed ROI-expansion scoring block (`v2_roi_weight`)
was removed — scoring is shape/spatial only, matching the benchmarked
configuration on every dataset.

Scoring is per-block cosine with weights `v2_block_weights` (default
shape 0.30 / spatial 0.70; effective .300/.700 after renormalization —
unbiased-sample sweep: spatial-heavy ratios beat balanced and
shape-heavy ones)
after **truncated ZCA whitening** of the standardized population:
whitening directions whose covariance eigenvalue is below 1% of the
largest is skipped (they carry only population noise — amplifying them
once collapsed every cosine toward 1), and the gain of kept directions is
capped (max 10x). The whitener is persisted beside the cache with a fit
version; identity below 64 rows. Block slices are clamped to the input
width, so V1-width snapshots (the warmed V1 counterpart cache) are scored
by their shape prefix.

**Spatial block**: right-hemisphere arbors are reflected onto the left at
vectorization time (lateral normalization, cache schema v2), so type-level
aggregation can no longer average L and R positions into a midline blur.
In ROI/connectivity-screen-first runs the spatial block score blends its
cosine (50/50) with a **mass-overlap term** — the min-sum intersection of
the query's and the candidate's raw Hellinger histograms — so a neuron
with a proportionally similar but much smaller overlap in the query's
region scores lower. The term is OFF in cache-direct runs (factorial:
-0.007 MRR).

**NBLAST (`method="nblast"`)**: NBLAST scores are computed BEFORE the type
aggregation, so `type_summary.csv` reflects NBLAST (the former flow
refined only the bodyId table, leaving the type list showing vector
scores). Candidates whose dotprops cannot be built keep their vector
prefilter score (same-type reference rows always do); when refinement is
skipped entirely the run logs a loud warning and records
`nblast_applied: false` in README.txt. NBLAST is not mirror-invariant
(contralateral same-type pairs score at chance), so NBLAST type means
aggregate IPSILATERAL pairs only; contralateral rows stay in results.csv,
and types with exclusively contralateral evidence are omitted from the
type ranking. The vector method lateral-normalizes and uses both sides.
NBLAST on FAFB scores the whole candidate pool from the healed zip
skeletons. Both methods run against
the V2 skeleton-vector cache (never the mesh cache). **BANC is excluded
from both methods**: morphological similarity on its public-release SWCs
is deferred pending vector-quality validation, so the backend raises a
deferred-feature error and the loader chain is never exercised for scoring.

**Two-pass type reevaluation** (`expand_top_types=20`, `expand_per_type=10`,
`0` disables): after the first scoring pass, the remaining members of the
top-ranked candidate types join the pool (ROI-screen-ranked, up to
`expand_per_type` per type), are scored with the same
standardization/whitening, and the type aggregation re-runs. Type-level
scores are then coverage-weighted continuously: the type statistic (mean,
or max under `type_agg="max"`) is multiplied by `sqrt(type_coverage)`
(`similarity = similarity_raw * sqrt(coverage)`) — no thresholds, no
exclusion; sparse types are damped, fully covered types pass through
untouched. Columns `type_coverage`, `similarity_raw`, and `similarity_max`
document the treatment in type_summary.csv; the bodyId-level results.csv
carries `type_coverage` per row. Per-block score columns (`sim_shape`, `sim_spatial`) are written to
results.csv and averaged into type_summary.csv.

**Connectivity-profile evidence backfill**: the `profile_similarity` column
is populated even when the ROI screen discovered the pool — first from the
shared-partner screen, then (v2) any remaining rows, including intra-type
query pairs, are scored with the Similarity>Connectivity machinery
(`ConnectivityProfiler` type profiles + `ProfileComparator.
weighted_cosine_similarity` on the shared local connection cache).

Caching: `SkeletonVectorCacheV2` shares the raw skeleton store with V1 but
keeps separate files (`find_similar/morphology/skeleton__vectors_v2.parquet`,
`meta_v2.json`, `whiten_v2.npy`), so V1 results are byte-for-byte unaffected.
V2 vectors use the **raw basis** (plan-raw-basis-vectorization.md): the
skeleton cache stores raw (level-0) skeletons — fetch pipelines default
`simplification=0` — and the V2 builder vectorizes raw trees only
(simplified files are skipped; their bodies are re-fetched raw on demand).
Simplification is applied at visualization/render time only. The V1
counterpart cache stays warm during online fetches. Legacy simp90 files
(quarantine: `scripts/maintenance/purge_legacy_simp90_cache.py`) are never
mixed into the cache basis; a basis/schema change bumps
`VECTOR_CACHE_V2_VERSION`, which invalidates and rebuilds the cache on the
next run.
Standalone heatmaps open with square cells locked (all exported similarity
heatmaps use square cells, matching the report cards).
Block weights and basis are recorded in the run README. The legacy V1
vector path is retired: both methods read the V2 skeleton-vector cache,
online fetches no longer warm a V1 counterpart, and NBLAST dotprops are
rebuilt per search in memory (never persisted).

```python
comparer = MorphologyComparer(
    query="aMe4", dataset="male-cns:v1.0", method="vector_v2",
    v2_block_weights={"shape": 0.3, "spatial": 0.7},
)
```

## SkeletonVectorCache

```python
from morphology import SkeletonVectorCache

cache = SkeletonVectorCache(dataset="male-cns:v0.9")
cache.build(fetch_missing=0)            # vectorize cached skeletons incrementally
cache.ensure(fetch_missing=0)           # build + load, ready to query
cache.coverage()                        # dict of coverage stats
vecs = cache.vectors_for(body_ids, compute_missing=True)
```

## Supporting functions

- `find_similar_raw_cache(dataset, ...)` — the raw skeleton cache helper.
- `find_similar_dataset_cache(dataset, ...)` — dataset-level cache helper.
- `find_similar_flywire_mesh_cache(...)` — FlyWire mesh cache helper.
- `load_local_release_skeletons(dataset, body_ids, ...)` — canonical local
  raw-skeleton loader: locally repaired caches (`cave_skeletons` /
  `extrusion_fixes`, status-driven) → raw cache (legacy frozen reads) →
  FAFB healed zip or standalone BANC public-release SWCs. Only FAFB performs
  the one-time extrusion check and the CAVE/local-prune repair; BANC never
  enters those stages. FAFB CAVE-skeletonized replacements live in the
  dedicated `skeletons/cave_skeletons/` store, locally pruned fixes in
  `skeletons/extrusion_fixes/`; the loader never writes new raw-store
  entries and the prepared mesh cache is never consulted.

## Notes

- `candidate_source` options are `["auto", "roi", "combined", "profile", "cache"]`;
  `"profile"`/`"combined"`/`"cache"` read the connection cache, `"roi"` uses ROI
  screening. Ensure cache coverage first (or set `cache_fetched_skeletons`).
- `visualize_top_n>0` triggers 3D rendering of the top matches — the UI feeds
  this through the shared skeleton settings.
- Multiple queries are a loop: one `MorphologyComparer` per query.
