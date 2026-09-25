# Vector v2 Similarity — Calculation Pipeline

The `vector_v2` method is the production default for morphological similarity
search (Find Similar and the vector prefilter of the NBLAST pipeline; homolog
finding's former `morph_v2_similarity` enrichment column is disabled). This
document specifies the
exact computation, from raw skeleton to final score, as implemented in
[`src/morphology.py`](../../src/morphology.py). A summary of all toolkit
scores lives in the [Score Calculation Guide](../core-features/ScoreCalculation_Guide.md);
benchmark evidence for the design decisions is in
`morphology_benchmark_NBLAST_vector/REPORT.md`.

**One-line formula**: each neuron is collapsed once into a fixed 256-dim
descriptor (shape + spatial blocks), lateral-normalized, z-scored with
persisted population statistics, ZCA-whitened, and compared by a
block-weighted cosine with weights **shape 0.30 / spatial 0.70**.

```
raw skeleton (swc.zst, shared cache)
        │
        ▼
[1] lateral normalize ── reflect right-hemisphere arbors onto the left
        │
        ▼
[2] vectorize ─────────── 256-dim descriptor (132 shape + 124 spatial)
        │
        ▼
[3] standardize ───────── (x − μ) / σ   (μ, σ persisted in meta_v2.json)
        │
        ▼
[4] ZCA whiten ─────────── x @ Wᵀ       (W persisted in whiten_v2.npz)
        │
        ▼
[5] score ──────────────── 0.30·cos(shape block) + 0.70·cos(spatial block),
                           renormalized over defined blocks
```

---

## Table of Contents

- [Configuration summary](#configuration-summary)
- [Stage 0 — Population artifacts](#stage-0--population-artifacts)
  - [Spatial bounds](#spatial-bounds)
  - [Standardization statistics (μ/σ)](#standardization-statistics-μσ)
  - [ZCA whitener (W)](#zca-whitener-w)
  - [When the stats change (cache lifecycle)](#when-the-stats-change-cache-lifecycle)
- [Stage 1 — Lateral normalization](#stage-1--lateral-normalization)
- [Stage 2 — Vectorization (256-dim)](#stage-2--vectorization-256-dim)
  - [Shape block, dims 0–131](#shape-block-dims-0131)
  - [Spatial block, dims 132–255](#spatial-block-dims-132255)
- [Stage 3 — Standardization](#stage-3--standardization)
- [Stage 4 — ZCA whitening](#stage-4--zca-whitening)
- [Stage 5 — Block-weighted cosine score](#stage-5--block-weighted-cosine-score)
  - [Per-block cosine](#per-block-cosine)
  - [Weighted total](#weighted-total)
  - [Optional: spatial mass-overlap term](#optional-spatial-mass-overlap-term)
  - [Optional: ROI-expansion block](#removed-roi-expansion-scoring-block)
- [Score properties](#score-properties)
- [Benchmark validation](#benchmark-validation)
- [Code reference](#code-reference)

---

## Configuration summary

| Setting | Value | Defined at |
| --- | --- | --- |
| Vector dimension | 256 (`VECTOR_V2_DIM`) | morphology.py:198 |
| Block slices | shape `[0, 132)`, spatial `[132, 256)` | morphology.py:202-203 |
| Default weights | `{"shape": 0.30, "spatial": 0.70}` (`DEFAULT_V2_BLOCK_WEIGHTS`) | morphology.py:217 |
| Cache schema version | `VECTOR_CACHE_V2_VERSION = 5` (4 = topology block removed; 5 = `bbox_xy_ratio` capped at 1e6) | `morphology.py`, `VECTOR_CACHE_V2_VERSION` |
| Whitening fit version | `WHITEN_FIT_VERSION = 3` | morphology.py:235 |
| Minimum rows for whitening | 64 (`MIN_ROWS_FOR_WHITENING`); below → identity | morphology.py:230 |
| Whitening truncation | `relative_floor = 1e-2`, `max_amplification = 10` | morphology.py:1708 |
| Histogram bins | 6×4×4 = 96 over the population bbox (`SPATIAL_HIST_BINS`) | morphology.py:161 |
| Profile bins | 8 radial + 8 midline (`SPATIAL_PROFILE_*`) | morphology.py:179-180 |
| Lateral normalization | always on for cached rows (`lateral_normalize=True`) | `SkeletonVectorCacheV2._meta_extra` marker |
| Mass-overlap term | **OFF** in cache-direct production; ON in screen-first runs | benchmark §10 |
| ROI-expansion scoring block | **removed** — ROI evidence is candidate-selection only (`candidate_source=roi`/`combined`, reported as `roi_similarity`) | 2026-08-31 |

---

## Stage 0 — Population artifacts

Three dataset-level artifacts are estimated once from the cached population
and persisted; every neuron of the dataset is processed identically against
them.

### Spatial bounds

`SkeletonVectorCacheV2.spatial_bounds()` (morphology.py:3341) estimates a
population bounding box `[lo(3), hi(3)]` from up to 200 locally cached
skeletons (2 % padding, outliers clip at the edges) and persists it in
`meta_v2.json`. Every neuron's histogram and midline profile are binned
against the same box, so bin *k* means the same brain region for all
neurons — absolute position in the template space is preserved. With no
local skeletons the histogram and midline blocks are zeros everywhere
(ellipsoid/radial features still apply).

### Standardization statistics (μ/σ)

Mean and per-dim std over the **raw** feature matrix of the whole cache,
written to `meta_v2.json` by `build()` (morphology.py:3731-3735).
Zero-variance dims fall back to σ = 1.

### ZCA whitener (W)

A 256×256 matrix fitted on the standardized population (see
[Stage 4](#stage-4--zca-whitening)), persisted in the `whiten_v2.npz`
sidecar together with `fit_version` (morphology.py:3414-3442). Below 64
population rows the whitener is the identity.

### When the stats change (cache lifecycle)

**Standardization is deliberately frozen between full rebuilds** — appending
neurons grows the population but never moves μ/σ or W:

| Event | μ/σ | Whitener W |
| --- | --- | --- |
| `append_vectors` (new rows) | untouched (morphology.py:2991) | untouched |
| Pending-merge checkpoint | filled once, only if meta has none (morphology.py:2169) | untouched |
| Full `build()` | **recomputed** over all rows | sidecar **deleted** → refit on next load (morphology.py:3737-3741) |
| Schema-stale cache (`_is_stale`, morphology.py:3328) | rebuild path → recomputed | refit |
| Population crosses 64 rows | unchanged | identity → fitted (one-time discontinuity) |
| `WHITEN_FIT_VERSION` bump | unchanged | all persisted whiteners refit |

`load()` (morphology.py:2908-2911) always standardizes with the persisted
meta μ/σ; computing stats from current rows is a fallback for caches with no
persisted stats. The design trade-off: score stability (results never
silently re-rank because rows were appended; neurons vectorized months apart
share one comparable space) at the cost of possible staleness if the
population's composition shifts a lot — an explicit `build()` refreshes.

---

## Stage 1 — Lateral normalization

`vectorize_neuron_v2(..., lateral_normalize=True)` (morphology.py:867).
If the mean node x of the arbor lies right of the population bbox midline
(`0.5·(x_lo + x_hi)`), the whole neuron is reflected: `x ← 2·mid_x − x`
(morphology.py:891-895). Reflection happens **before** the spatial features
are computed, so every stored vector represents its neuron on the left
hemisphere.

Consequences:

- L/R homologs (contralateral twins) produce near-identical vectors — the
  method is mirror-invariant by construction. This is why vector_v2
  retrieves contralateral same-type pairs at 0.80× its ipsilateral score
  where NBLAST collapses to 0.03× (benchmark §7).
- Type-level aggregation can no longer average L and R positions into a
  midline blur.
- The shape block and the radial/midline profiles are reflection-invariant
  and unaffected; only the ellipsoid's x-signed dims and the histogram's
  x-bin order actually change.

---

## Stage 2 — Vectorization (256-dim)

`vectorize_neuron_v2` returns one fixed-width vector. Column layout (the
slices every downstream consumer must agree on):

| Dims | Block | Contents |
| --- | --- | --- |
| 0–123 | V1 shape | 24 morphometrics (`MORPHOMETRIC_FEATURES`) + 100 persistence dims (`pv_0..99`) — identical to the V1 vector |
| 124–131 | Shape extras | 8 `sx_*` arbor-geometry dims |
| 132–143 | Spatial: ellipsoid | 12 `sp_*` dims |
| 144–239 | Spatial: histogram | 96 `sh_*` dims (6×4×4 over the population bbox) — this slice is `SPATIAL_HIST_SLICE` |
| 240–247 | Spatial: radial profile | 8 `rp_0..7` |
| 248–255 | Spatial: midline profile | 8 `md_0..7` |

The former 24-dim Laplacian **topology block was removed in schema v4**: its
eigensolver dominated vectorization time (intermittent seconds-to-minutes
stalls) while its 10 % weight changed nothing in the user-facing ranking
(simtest A/B: all four probe types still retrieved at rank 1 without it).
The freed dims were refilled with the `sx_*` and `rp_`/`md_*` features.

### Shape block, dims 0–131

- **V1 features (0–123)**: `vectorize_neuron` (morphology.py:998) — 24
  curated morphometrics (cable length, node/branch/leaf counts, bbox,
  path-length stats, tortuosity, Strahler stats, densities, ratios) plus a
  100-dim persistence image of the skeleton's death sequence. Reflection-
  invariant. `bbox_xy_ratio` is capped at 1e6: planar arbors with a zero
  y-span (R7/R8 photoreceptor terminals are flat by anatomy) would
  otherwise produce 1e13-scale division artifacts that dominate the
  population statistics of the whole dim (schema v5).
- **Shape extras (124–131)**: `compute_shape_extras` (morphology.py:699),
  O(k) from the node table:
  `sx_radius_mean/std/max`, `sx_radius_leaf_mean`,
  `sx_branch_angle_mean/std` (angles between sibling edges at each branch
  point), `sx_strahler_frac_1`, `sx_strahler_frac_ge4` (cable-length
  fraction in Strahler order 1 / ≥4). Zeros for mesh-representation caches
  (no radius/branch structure).

### Spatial block, dims 132–255

All cable-mass features weight each skeleton edge by its length
(`_cable_mass_points`, edge midpoints; meshes fall back to unit vertex
mass). Histogram-like blocks are L1-normalized (fractions of total cable)
then Hellinger-transformed (`√x`) so Euclidean geometry ≈ distribution
comparison under cosine.

- **Ellipsoid (132–143)**: `compute_spatial_ellipsoid` (morphology.py:586) —
  PCA of the node coordinates: centroid (3), axis spreads `√λ` descending
  (3), sign-stabilized principal axis (3), anisotropy `(a1−a2)/a1`,
  flatness `(a2−a3)/a1`, RMS radius. Captures where the arbor sits and how
  it expands.
- **Histogram (144–239)**: `compute_spatial_histogram_abs`
  (morphology.py:664) — cable mass binned by edge-midpoint position into a
  6×4×4 grid over the **population-fixed** bbox (C-order flat index), so
  bin identity = brain region identity across all neurons.
- **Radial profile (240–247)**: `rp_0..7` — cable mass over 8 equal bins of
  distance from the arbor's own centroid (per-neuron max range). A
  scale-free shape-of-distribution summary; mirror-invariant.
- **Midline profile (248–255)**: `md_0..7` — cable mass over 8 equal bins
  of absolute distance from the brain midline (bbox x-mid, range = x span).
  Region identity along the lateral axis; mirror-invariant.

---

## Stage 3 — Standardization

Persisted population statistics: `X_std = (x − μ) / σ` per dim, with
`σ → 1` where `σ ≤ 0` (morphology.py:2908-2911). Applied identically to
cached rows, freshly vectorized query rows, and the intra-type reference
matrix — every score lives in one space.

---

## Stage 4 — ZCA whitening

`fit_zca_whitener` (morphology.py:1708) fits, once per population:

```
Σ = cov(X_std) = V Λ Vᵀ
s_j = min(1/√λ_j, 10)   if λ_j ≥ floor        (floor = 1e-2 · λ_max)
s_j = 1                 otherwise
W   = V diag(s) Vᵀ
```

Applied as `apply_whitening(W, X) = X @ Wᵀ` (W is symmetric).

- **Why**: decorrelates the feature space so a few high-variance shape dims
  (raw size scales) can no longer dominate the cosine; each direction
  contributes by its discriminative spread, not its units.
- **Truncated**: high-variance directions are *compressed* (`s < 1`),
  mid-range ones whitened (`s ≈ 1`). Directions with `λ < 1e-2·λ_max` are
  population noise and pass through **unchanged** (`s = 1`) instead of being
  amplified by `1/√λ`; the `max_amplification = 10` cap bounds the same
  pathology for mid-low directions. (The naive eps-regularized variant
  scaled a near-constant direction by `1/√eps`, and that shared constant
  dominated every vector, collapsing all cosines toward 1.)
- **Fallback**: below 64 standardized population rows → identity whitening
  (plain z-scored block cosine).

---

## Stage 5 — Block-weighted cosine score

### Per-block cosine

`_block_cosine_one` (morphology.py:1761) computes, per block
`b ∈ {shape [0,132), spatial [132,256)}`:

```
cos_b(q, t) = (q_b · t_b) / (‖q_b‖ · ‖t_b‖)
```

with a **validity rule**: if either side's block is all-zero (‖·‖ ≤ 1e-12),
the pair is *invalid* for that block — the block is excluded from the
weighted mean rather than forced to 0. A neuron with no arbor evidence in a
region must not read as "maximally dissimilar there"; it carries no evidence
for that block.

### Weighted total

`v2_similarity_matrix` (morphology.py:1806), production cache-direct
configuration:

```
score(q, t) = ( w_shape · cos_shape + w_spatial · cos_spatial )
              / ( w_shape + w_spatial )            over VALID blocks only

w_shape = 0.30, w_spatial = 0.70        (DEFAULT_V2_BLOCK_WEIGHTS)
```

The denominator renormalizes by the weights of the blocks actually defined
for that pair, so a pair with only one valid block gets that block's cosine
as the score. Weights sum to 1.0 in the default config, making the
renormalization a no-op there, but the mechanism matters when the optional
blocks below participate.

### Optional: spatial mass-overlap term

When enabled (screen-first runs only; `spatial_overlap` argument), the
spatial cosine is blended 50/50 with a histogram intersection on the **raw
Hellinger histograms** before weighting:

```
overlap(q, t) = Σ_k min(q_hist[k], t_hist[k])        (96 dims)
cos_spatial' = 0.5 · cos_spatial + 0.5 · overlap
```

Rationale: cosine is proportion-blind — a neuron with the same
*distribution shape* but far less arbor in the query's region scores
identically; the intersection measures the shared fraction directly.

The 2×2 factorial (benchmark §10) measured this term at **−0.007 MRR** in
cache-direct scoring (an earlier +0.015 reading was a benchmark-loader bug
that built the overlap from 112 columns instead of the production 96), so
the production default keeps it **OFF**. The term remains ON in
ROI-screen-first runs, where it was designed for curated pools.

### Removed: ROI-expansion scoring block

An earlier version composed a runtime ROI block (Hellinger pre/post synapse
fractions over the primary ROIs, weight 0.20, NeuPrint screen-first runs)
and appended it via an ``extra_blocks`` argument. It was removed on
2026-08-31: the weight sweep and factorial that validated 3:7 were
two-block, cache-direct measurements, and ROI evidence is a
candidate-*selection* signal (`candidate_source=roi`/`combined`, surfaced
per row as ``roi_similarity``), never a scoring input. Scoring is
shape/spatial only on every dataset.

---

## Score properties

| Property | Value |
| --- | --- |
| Range | roughly [−1, 1]; identical vectors → 1.0; uncorrelated/orthogonal whitened blocks → ~0 |
| Symmetry | symmetric in (q, t) |
| Mirror invariance | yes, by Stage 1 (contralateral twins ≈ ipsilateral scores) |
| Cost per query | O(P·d) — one whitened matmul + two block cosines, **independent of arbor size** (measured median ~4.7 ms vs ~159 ms NBLAST on a 2,273-neuron pool) |
| Prep cost | one-time vectorization ~6 ms median per neuron |
| Storage | one 2,048 B row per neuron in `skeleton__vectors_v2.parquet` |
| Aggregation | type score = aggregation over member pairs (`similarity · √coverage` weighting in the two-pass type reevaluation); mirror-invariant, so both sides participate. NBLAST type means are ipsilateral-only instead (see Score Calculation Guide) |
| Homolog enrichment | the same scorer applied in the target scene's render space, using dedicated render-space population artifacts (`meta_v2_render.json` / `whiten_v2_render.npz`) derived once per dataset from a transformed population sample; identity spaces use the native artifacts directly |

---

## Benchmark validation

Evidence backing the current configuration
(`morphology_benchmark_NBLAST_vector/REPORT.md`, unbiased 1 % sample of
male-cns v1.0, 2,273 neurons / 379 queries):

- **vector_v2 beats NBLAST on every headline metric**: MRR 0.926 vs 0.894
  (Wilcoxon p = 0.041), Hit@1 0.895 vs 0.844, AP@20 0.681 vs 0.521, mean
  AUC 0.981 vs 0.903.
- **Weights 3:7** (shape:spatial) — MRR peaks at 0.9261 and declines
  monotonically on both sides; +0.0162 over 5:5 (p = 0.039).
- **No NBLAST refinement**: inside the vector top-500 pool, NBLAST ordering
  scores MRR 0.894 vs the prefilter's own 0.926 — refining lowers quality.
- **Mass-overlap OFF** in cache-direct (factorial: −0.007 MRR corrected).
- **Topology block removed** (schema v4): eigensolver cost without ranking
  effect.
- **Whitening matters**: the no-whitening, no-blocks V1-prefix baseline
  collapses (MRR 0.034), confirming the value of the V2 spatial blocks and
  the whitened space.

---

## Code reference

All in [`src/morphology.py`](../../src/morphology.py):

| Component | Location |
| --- | --- |
| Schema constants (dims, slices, weights, versions) | morphology.py:155-235 |
| `vectorize_neuron_v2` (lateral normalize + assemble) | morphology.py:867 |
| `compute_shape_extras` (`sx_*`) | morphology.py:699 |
| `compute_spatial_ellipsoid` (`sp_*`) | morphology.py:586 |
| `compute_spatial_histogram_abs` (`sh_*`) | morphology.py:664 |
| `compute_spatial_profile_extras` (`rp_*`/`md_*`) | morphology.py:798 |
| `SkeletonVectorCacheV2` (cache, bounds, whitener sidecar) | morphology.py:3240 |
| `fit_zca_whitener` / `apply_whitening` | morphology.py:1708 / 1752 |
| `_block_cosine_one` (per-block cosine + validity) | morphology.py:1761 |
| `v2_similarity_matrix` (final score) | morphology.py:1806 |
| `v2_pairwise_matrix` (intra-type reference) | morphology.py:1875 |
| `_mirror_spatial_block` (runtime mirror approximation) | morphology.py:1781 |
| Metric dispatcher (`_similarity_matrix` / `_similarity_matrix_with_blocks`) | `MorphologyComparer` |
| Homolog enrichment (disabled in homolog finding; standalone helper) | `render_v2_artifacts` / `compute_morph_similarity_vs_queries` |

Cache artifacts:

| File | Content |
| --- | --- |
| `cache/<dataset>/find_similar/morphology/skeleton__vectors_v2.parquet` | raw 256-dim rows (+ bodyId, type, instance, rep) |
| `cache/<dataset>/find_similar/morphology/meta_v2.json` | μ/σ, spatial bounds, schema version, `lateral_normalize` marker, row counts |
| `cache/<dataset>/find_similar/morphology/whiten_v2.npz` | `W` + `fit_version` |
| `cache/<dataset>/skeletons/raw_skeletons/` | shared swc.zst skeletons (the V1 and V2 caches share one skeleton store, outside `find_similar/morphology/`) |
