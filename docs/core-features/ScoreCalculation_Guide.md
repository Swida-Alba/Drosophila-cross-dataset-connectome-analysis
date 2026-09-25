# Score and Probability Calculation Guide

This document provides comprehensive explanations of all score and probability calculations used throughout the toolkit, including their formulas, biological rationale, and where they are applied.

---

## Table of Contents

- [Score and Probability Calculation Guide](#score-and-probability-calculation-guide)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [The Dₜ Graph Model](#the-dₜ-graph-model)
    - [Threshold Semantics](#threshold-selection-guide)
    - [Per-Cutoff Aggregates (ThresholdedConnectionMap)](#per-cutoff-aggregates-thresholdedconnectionmap)
  - [Connection-Level Metrics](#connection-level-metrics)
    - [Connection Ratio](#connection-ratio)
    - [Traversal Probability](#traversal-probability)
    - [Block Probability](#block-probability)
  - [Population-Level Aggregation](#population-level-aggregation)
    - [Type-to-Type Connection Ratio](#type-to-type-connection-ratio)
    - [Type-to-Type Traversal Probability](#type-to-type-traversal-probability)
      - [1. Product Method (Default)](#1-product-method-default)
      - [2. Average Method](#2-average-method)
      - [3. Ratio Method (Legacy)](#3-ratio-method-legacy)
      - [Method Comparison](#method-comparison)
  - [Path-Level Metrics](#path-level-metrics)
    - [Path Probability](#path-probability)
    - [Minimum Edge Weight](#minimum-edge-weight)
    - [Path Ranking](#path-ranking)
  - [Connectivity-Profile Similarity Metrics](#connectivity-profile-similarity-metrics)
  - [Morphological Similarity Metrics](#morphological-similarity-metrics)
  - [NeuronBridge Matching Scores](#neuronbridge-matching-scores)
    - [Match Score](#match-score)
    - [Coverage Ratio](#coverage-ratio)
    - [Weighted Score](#weighted-score)
    - [Cross-Dataset Score](#cross-dataset-score)
  - [Filter Parameters Summary](#filter-parameters-summary)
  - [Threshold Selection Guide](#threshold-selection-guide)
    - [Connection Ratio (`min_ratio`)](#connection-ratio-min_ratio)
    - [Traversal Probability (`min_traversal_probability`)](#traversal-probability-min_traversal_probability)
    - [Conversion Formulas](#conversion-formulas)
  - [Code References](#code-references)
    - [Core Calculation Locations](#core-calculation-locations)
    - [Parameter Definitions](#parameter-definitions)
    - [Related Documentation](#related-documentation)
  - [Summary](#summary)

---

## Overview

The toolkit uses several metrics to quantify connection strength and signal propagation probability in the connectome:

| Metric                    | Formula                      | Range     | Level |
| ------------------------- | ---------------------------- | --------- | ----- |
| **Connection Ratio**      | `weight / total_incoming`    | 0.0 - 1.0 | Edge  |
| **Traversal Probability** | `min(1.0, ratio / 0.3)`      | 0.0 - 1.0 | Edge  |
| **Type Traversal Prob**   | `1 - ∏(1 - p_pair)` (default)| 0.0 - 1.0 | Type  |
| **Path Probability**      | `∏(edge probabilities)`      | 0.0 - 1.0 | Path  |
| **Weighted Score**        | `avg_score × coverage_ratio` | 0.0 - ∞   | Line  |

All ratios and probabilities are computed on the **thresholded graph Dₜ**
defined below, so every metric at every level refers to one consistent
definition of "the network" for a given `min_synapse_num = t`.

---

## The Dₜ Graph Model

The connection cache stores the **complete graph** (every edge with weight ≥ 1).
A query with `min_synapse_num = t` works on the derived graph

$$D_t = \{ \text{edges with } w_{ij} \ge t \}$$

which is treated as an **independent dataset**:

- The **numerator** of every ratio is the weight of an edge (or the sum over a
  set of edges) **inside Dₜ**.
- The **denominator** (`total_incoming`) is computed **from Dₜ alone** — never
  from the full cache and never from a different threshold.
- Ratios at different thresholds are therefore never mixed: `ratio(A→B)` at
  `t=1` and at `t=3` describe two different graphs and are not comparable.

**Why edge-thresholding matters (type level).** When aggregating bodyId-level
connections to type pairs, the cutoff is applied to **edges first**:

```
D_t consistency:
  correct:   keep edges with weight >= t, THEN sum weights per type pair
  incorrect: sum weights per type pair first, THEN keep pairs with sum >= t
```

The second form mixes two definitions of the cutoff — an edge-thresholded
denominator (computed from Dₜ) with a pair-thresholded numerator (computed
from a different graph) — and lets weak edges "ride along" inside a strong
pair. All type-level aggregation in this toolkit follows the first form.

### Per-Cutoff Aggregates (ThresholdedConnectionMap)

[`src/connection_map.py`](../../src/connection_map.py) —
`ThresholdedConnectionMap` implements Dₜ as a first-class object:

- **One map per cutoff `t`.** Each map owns *both* aggregate tables —
  `bodyId_post → total_incoming_weight` and `type_post → total_incoming_weight`
  — plus the neuron-index lookup, so the bodyId-level and type-level
  denominators always come from the **same** thresholded graph.
- **Computed once, reused everywhere.** Aggregates are computed lazily with
  vectorized Polars and cached for the lifetime of the map; every per-layer
  call of a FindPath/FindAllPath run, every FindDirect enrichment, and every
  type-level filter shares the same cached tables (previously each call
  re-scanned the whole connections parquet).
- **Auto-invalidation.** Maps are keyed by a source signature (in-memory frame
  identity, or disk paths + mtimes). When the cache is rebuilt or replaced,
  stale maps are discarded and rebuilt on the next access.

In `FindNeuronConnection`, `_connection_map(min_weight)` returns the Dₜ map
for a cutoff (capped at 16 live maps, oldest evicted), and
`_get_total_incoming_by_{bodyid,type}_table(min_weight)` are thin wrappers over
it.

---

## Connection-Level Metrics

These metrics are calculated for each individual synaptic connection (edge) in the network.

### Connection Ratio

**Definition**: The fraction of a neuron's **total input from ALL sources in the dataset** provided by a specific upstream connection.

**Formula (Global Ratio)**:

$$\mathit{ratio}_{ij}^{(t)} = \frac{w_{ij}^{(t)}}{\sum_{\forall k \in \text{dataset}} w_{kj}^{(t)}}$$

Where:
- $w_{ij}^{(t)}$ = Number of synapses from neuron $i$ to neuron $j$ passing threshold $t$ (an edge of Dₜ)
- $\sum_{\forall k \in \text{dataset}} w_{kj}^{(t)}$ = Total incoming synapses to neuron $j$ from **ALL neurons in the dataset** at threshold $t$ (computed from Dₜ by the `ThresholdedConnectionMap`)

**Why Global Ratio?**

The ratio is calculated using the **entire connection cache**, not just the neurons in the current query:
- Denominator includes ALL incoming connections to B, from ANY source neuron
- This gives the true biological fraction "how much of B's total input comes from A"
- Without global calculation, ratios would be inflated when only a subset of source neurons is queried

**Interpretation**:
- Value of `0.01` means 1% of neuron $j$'s **total input** comes from neuron $i$
- Higher values indicate stronger relative input influence
- Range: 0.0 to 1.0

**Missing global denominator (fallback)**:
- Post neurons/types **absent from the global incoming table** (e.g. untyped
  neurons, which are grouped by their bodyId, or types the connection cache
  has no entry for) fall back to the **local total** over the connections in
  the current table instead of becoming 0/NaN.
- The local fallback keeps every edge's ratio in (0.0, 1.0] so path
  probabilities never collapse to 0, but it is **inflated** compared with the
  true global fraction - a fresh/complete connection cache or an API-fetched
  incoming table restores exact global ratios.

**Example**:
```
Query: aMe12 → KCg-d at threshold = 3 synapses

aMe12 → KCg-d: 49 synapses (the connection we're analyzing)
All other neurons → KCg-d: 4951 synapses (from connection cache, at t=3)
Total incoming to KCg-d: 49 + 4951 = 5000

connection_ratio(aMe12→KCg-d) = 49 / 5000 = 0.0098 (0.98%)

Note: If we only counted from provided source neurons, we'd get a misleading
ratio like 0.98 (98%) which doesn't represent true biological significance.
```

**Used in**:
- [`src/coana.py`](../../src/coana.py) - `_fetch_total_incoming_weight()`, `_apply_bodyid_level_filters()`, `_apply_type_level_filters()`
- [`src/statvis.py`](../../src/statvis.py) - `EnrichConnectionTable()` (preserves pre-calculated ratios)
- Edge filtering via `min_ratio` parameter

---

### Traversal Probability

**Definition**: The probability that a signal will traverse from one neuron to another, scaled by a biological threshold.

**Formula**:

$$p_{ij} = \min\left(1.0, \frac{\mathit{ratio}_{ij}^{(t)}}{0.3}\right)$$

**The 0.3 Biological Threshold**:

The 0.3 scaling factor represents a **30% input threshold**—connections providing ≥30% of a neuron's inputs are considered maximally effective for signal propagation. This is based on neurobiological observations that:
- Connections exceeding ~30% of post-synaptic input typically reliably activate the downstream neuron
- Below this threshold, activation probability scales roughly linearly with input fraction

**Examples**:

| Scenario    | Weight | Post | Ratio | Traversal Prob |
| ----------- | ------ | ---- | ----- | -------------- |
| Weak        | 2      | 1000 | 0.2%  | 0.67%          |
| Moderate    | 10     | 500  | 2%    | 6.7%           |
| Strong      | 50     | 200  | 25%   | 83.3%          |
| Very Strong | 100    | 200  | 50%   | 100% (capped)  |

**Used in**:
- [`src/coana.py`](../../src/coana.py) - `_apply_bodyid_level_filters()`, `_pair_level_probabilities()`
- [`src/statvis.py`](../../src/statvis.py) - `EnrichConnectionTable()`
- Edge filtering via `min_traversal_probability` parameter
- Sankey diagrams and network visualizations

---

### Block Probability

**Definition**: The probability that a signal will NOT traverse a connection.

**Formula**:

$$\mathit{block}_{ij} = 1 - \mathit{p}_{ij}$$

**Used in**:
- [`src/statvis.py`](../../src/statvis.py) - Path probability calculations
- Type-level enrichment output (`block_probability = 1 - p_AB`)

---

## Population-Level Aggregation

When analyzing connections between neuron **types** (populations), individual connections must be aggregated.

### Type-to-Type Connection Ratio

**Definition**: The aggregate connection strength between two neuron types, using total input from ALL sources in the dataset.

**Formula (Global Ratio)**:

$$\mathit{ratio}_{AB}^{(t)} = \frac{\sum_{i \in A, j \in B} w_{ij}^{(t)}}{\sum_{\forall k \in \text{dataset}, j \in B} w_{kj}^{(t)}}$$

Where:
- $A$ = Set of neurons of type A (presynaptic population)
- $B$ = Set of neurons of type B (postsynaptic population)
- Numerator = Total synapses from type A to type B **over the edges of Dₜ** (the
  cutoff is applied to edges first, then pairs are summed — see
  [The Dₜ Graph Model](#the-dₜ-graph-model))
- Denominator = Total incoming synapses to type B from **ALL neuron types in the dataset** at threshold $t$ (from the same Dₜ, cached by `ThresholdedConnectionMap`)

**Key Point**: The denominator includes ALL incoming connections to type B from any neuron type in the entire dataset (not just the source types in your query), giving the true fraction of type B's total input that comes from type A. Both terms come from the same Dₜ.

**Implementation** ([`src/coana.py`](../../src/coana.py)):

```python
# Step 1: Apply the cutoff to EDGES first (D_t), then sum weights per type pair
conn = conn[conn['weight'] >= min_weight]
weight_sum = conn.groupby([type_pre, type_post])['weight'].sum()

# Step 2: Get total incoming weight per type from the SAME D_t
# (ThresholdedConnectionMap caches this per cutoff - no re-scan)
total_incoming_per_type = self._get_total_incoming_by_type_table(min_weight)

# Step 3: Calculate global ratio
conn_type['connection_ratio'] = weight_sum / total_incoming_per_type
```

**Aggregation via matrix operations** ([`src/coana.py`](../../src/coana.py)):

```python
# For type-level pivot tables, use 'max' aggregation
mat_ratio = df.pivot(values='connection_ratio', index='type_pre', 
                     columns='type_post', aggregate_function='max')
```

---

### Type-to-Type Traversal Probability

**Definition**: The aggregate probability that a signal will propagate from any neuron of type A to any neuron of type B.

The type-level probability is derived from the **deduplicated bodyId-level
pairs** of Dₜ (each pair appears once, with its per-pair traversal
probability `p_ij = min(ratio_ij / 0.3, 1)`). How the pairs are combined is
controlled by `aggregate_method` — **identical in all three places** that
compute type-level probabilities:

1. `statvis.EnrichConnectionTable` (pandas engine)
2. `statvis.EnrichConnectionTablePolars` (Polars engine)
3. `coana._apply_type_level_filters` (type-level filtering, via the
   `FindNeuronConnection.aggregate_method` field, default `'product'`)

so the filter threshold, the enrichment output, and the path metrics always
use the same definition.

#### 1. Product Method (Default)

**Formula**:

$$p_{AB} = 1 - \prod_{i \in A, j \in B}\left(1 - p_{ij}^{(t)}\right)$$

**Rationale**: The type edge A→B is a **bundle of parallel channels** (the
individual neuron pairs). A signal can take any channel, so the edge
"transmits" if **at least one** pair transmits — a reliability/OR model.
The more pairs (or the stronger each pair), the closer the type probability
gets to 1, even when no single pair exceeds the 30% threshold.

**Why this is the default**: it is the only model that composes correctly
with path analysis. A 2-hop path A→X→Y through type edges is only as good as
the weakest *type* edge, and the product model reflects that parallel
channels make type edges more reliable than their individual pairs.

**Example**:
```
Type edge A→B with 3 bodyId pairs (t=3):
  pair p: 0.50, 0.33, 0.67   (each capped at 1.0)

p_AB = 1 - (1-0.50)(1-0.33)(1-0.67)
     = 1 - (0.50)(0.67)(0.33)
     = 1 - 0.110 = 0.89
```

#### 2. Average Method

**Formula**:

$$p_{AB} = \frac{\sum_{i \in A, j \in B} w_{ij}^{(t)} \cdot p_{ij}^{(t)}}{\sum_{i \in A, j \in B} w_{ij}^{(t)}}$$

The **weight-weighted mean** of the pair probabilities: pairs with more
synapses dominate the type value. A descriptive average — useful when you
want one number per type pair for display, but it does not model parallel
reliability.

#### 3. Ratio Method (Legacy)

**Formula**:

$$p_{AB} = \min\left(1.0, \frac{\mathit{ratio}_{AB}^{(t)}}{0.3}\right)$$

The old uniform model: the type-level ratio scaled by 0.3, exactly like the
bodyId level. Kept for backward compatibility (`aggregate_method='ratio'`).

#### Method Comparison

| Method    | Typical value vs pairs | Use case                              |
| --------- | ---------------------- | ------------------------------------- |
| `product` | ≥ any single pair     | Path analysis, filtering (default)    |
| `average` | between pair values   | Descriptive type-level summaries      |
| `ratio`   | same as bodyId scale  | Legacy behavior, API compatibility    |

**Implementation**:

- [`src/coana.py`](../../src/coana.py) - `_apply_type_level_filters()` (type-level filtering, `aggregate_method` field), `_pair_level_probabilities()`
- [`src/statvis.py`](../../src/statvis.py) - `_type_probability_series()`, `EnrichConnectionTable()` (pandas engine)
- [`src/statvis.py`](../../src/statvis.py) - `EnrichConnectionTablePolars()` (Polars engine, `aggregate_connections()`)

---

## Path-Level Metrics

For multi-hop pathways (A → B → C → D), path-level metrics aggregate edge metrics along the entire path.

### Path Probability

**Definition**: The probability that a signal successfully traverses an entire multi-hop pathway.

**Formula**:

$$P_\mathrm{path} = \prod_{k=1}^{n} p_{k,k+1}$$

Where the path has $n$ edges with individual traversal probabilities $p_{k,k+1}$.
For type-level paths the edge probabilities come from the type-level table
(`conn_type`), which uses the compound product aggregation by default — so a
multi-pair type edge is *stronger* than any of its pairs, and the path
probability composes accordingly.

**Example**:
```
Path: A → B → C → D
Edge probabilities: [0.8, 0.6, 0.9]
Path probability: 0.8 × 0.6 × 0.9 = 0.432 (43.2%)
```

**Implementation** ([`src/statvis.py`](../../src/statvis.py#L4881)):

```python
row = {
    'path_prob': np.prod(probs) if probs else 0,
    'min_weight': min(weights) if weights else 0,
    'min_ratio': min(ratios) if ratios else 0,
    'length': len(path) - 1
}
```

**Note on Path Length**: Longer paths have exponentially lower probabilities:
- 3 edges at 0.5 each: $0.5^3 = 0.125$
- 6 edges at 0.5 each: $0.5^6 = 0.016$

---

### Minimum Edge Weight

**Definition**: The "bottleneck" of a path—the weakest connection along the route.

**Formula**:

$$w_\mathrm{min} = \min_{k=1}^{n} w_{k,k+1}$$

**Used for**: Identifying the limiting factor in signal transmission.

---

### Path Ranking

Paths are ranked by `path_prob` (descending) for output files and visualizations:

**Implementation** ([`src/coana.py`](../../src/coana.py)):

```python
if 'path_prob' in df_paths.columns:
    sort_cols.append('path_prob')
elif 'path_probability' in df_paths.columns:
    sort_cols.append('path_probability')
```

The `show_top_n_paths` parameter limits output to the highest-probability paths.

---

## Connectivity-Profile Similarity Metrics

These metrics score a pair of connectivity profiles (a dict of
partner-type → weight, restricted to the top-K partners per direction). They
are the same six metrics used by the **Connectivity → Find Similar** and
**Connectivity → Comparison** tabs,
and are computed for every pair regardless of the chosen sort metric.
In the tables below `A` and `B` are the two profiles' partner-type sets, `w_a`/`w_b`
are the partner weights, and `ρ` is the Spearman rank correlation of two
weight vectors.

| Metric                | Formula | Range |
| --------------------- | ------- | ----- |
| `jaccard`             | \|A ∩ B\| / \|A ∪ B\| (partner-type sets) | 0–1 |
| `weighted_jaccard`    | Σ min(w_a, w_b) / Σ max(w_a, w_b) over the union (Ruzicka) | 0–1 |
| `cosine`              | (A·B) / (‖A‖·‖B‖) over the union (missing partners = 0) | 0–1 |
| `rank_corr`           | ρ = Spearman on **shared** partners (raw, sign retained) | −1 to 1 |
| `rank_union`          | ρ over the **union** of both partner sets (missing = 0) | −1 to 1 |

**Notes**

- `rank_corr` and `rank_union` return `NaN` when fewer than 3 comparable
  partners exist (Spearman is undefined), and when a weight vector is
  constant. Both are reported raw (−1 to 1) so the sign retains meaning.
- The `combined` score and all `(ρ + 1) / 2` normalized rank variants
  (`rank_norm`, `rank_union_norm`, `rank_corr_norm`) and the derived
  `confidence` labels are no longer produced.
- For a direction-aware comparison (`upstream`/`downstream`), the metric is
  computed per direction and averaged for `both`.
- Homolog finding no longer computes morphology enrichment — the former
  `morph_v2_similarity`/`morph_nblast` result columns are removed
  (cross-dataset scores were weak discriminators and the per-run
  computation was expensive). `morphology.enrich_homolog_results` remains
  available for standalone use; `morph_v2_similarity` there is the
  production vector_v2 score (standardized + ZCA-whitened per-block
  cosine, shape/spatial 0.30/0.70), identical to the Morphology → Find
  Similar scorer
  (it had replaced the removed `morph_cosine`/`morph_pearson` columns,
  whose plain full-vector cosine/Pearson on unstandardized features was
  magnitude-dominated).

## Morphological Similarity Metrics

| Metric                | Formula | Range |
| --------------------- | ------- | ----- |
| `similarity`          | vector_v2: per-block cosine (shape/spatial 0.30/0.70) on standardized + ZCA-whitened vectors; NBLAST: normalized pair score | 0–1 |
| `roi_similarity`      | ROI-distribution **screen** score (candidate selection only, never in the morphology score): cosine of input/output synapse-distribution vectors over the primary ROIs, mirrored across the midline | 0–1 |
| `profile_similarity`  | shared-count / max-shared-count (normalized shared-partner count) | 0–1 |
| `intra_type_similarity` | mean pairwise similarity of the type's members (the intra-type reference) | 0–1 |

For the vector method (vector_v2) each neuron is reduced to a 256-dim
vector: a 132-dim shape block (24 morphometrics + 100-dim persistence + 8
arbor-geometry extras such as node-radius stats and branch angles) and a
124-dim spatial block (arbor ellipsoid + cable-mass histogram over the
population bounding box + radial and midline-distance cable-mass
profiles). The former 24-dim Laplacian topology block was removed in cache
schema v4 — its eigensolver dominated vectorization time while its 10%
weight changed nothing in the user-facing ranking. Scoring is per-block
cosine on z-scored (and ZCA-whitened) vectors with block weights — default
shape 0.30 / spatial 0.70, effective .300/.700 after
renormalization — so
cosine scores stay scale-fair and region-fair regardless of the local
population. The full stage-by-stage pipeline (feature layout, lateral
normalization, ZCA truncation details, cache lifecycle of the persisted
μ/σ/whitener, optional mass-overlap term) is specified in
[Vector v2 Similarity — Calculation Pipeline](../technical/VECTOR_V2_PIPELINE.md).

**ROI evidence is selection-only.** The former runtime-composed
ROI-expansion scoring block (weight 0.20, NeuPrint screen-first runs) was
removed: scoring is shape/spatial only, exactly the configuration the
morphology benchmark validated. ROI-distribution similarity continues to
drive candidate-pool selection (`candidate_source=roi`/`combined`) and is
reported per row as `roi_similarity`.

**Retired V1 legacy.** The legacy cosine/Pearson `metric` option applied
only to the retired V1 vector path; vector_v2 and NBLAST scores are
unaffected by it, and the Morphology → Find Similar UI no longer exposes it.

**Contralateral (mirror) pairs.** NBLAST matches arbors in absolute
coordinates, so contralateral same-type pairs score at chance (morphology
benchmark: median −0.03 vs 0.89 ipsilateral; partner reciprocal rank 0.13
vs 0.84). NBLAST type means therefore aggregate **ipsilateral pairs
only**: rows pairing opposite known sides (`somaSide` on NeuPrint
datasets, `hemisphere` on FlyWire) are excluded from `type_summary.csv`
(kept in `results.csv` for inspection), types with exclusively
contralateral evidence are omitted from the type ranking, and
`n_bodyids` counts ipsilateral plus unknown-side members. vector_v2
lateral-normalizes at vectorization (mirror twins become near-identical
vectors), so its type means use both sides and are unchanged.

**Dotprops are never cached.** NBLAST dotprops (`navis.make_dotprops`,
k=20) are rebuilt per search from the persisted raw skeletons and live in
memory only — no dotprops file is written to disk, so a schema, dataset,
or parameter change can never leave stale dotprops behind. Only raw
skeletons and vector rows persist.

---

## NeuronBridge Matching Scores

These scores are used for EM↔LM mapping via NeuronBridge.

### Match Score

**Definition**: The raw morphological similarity score from NeuronBridge's Color Depth MIP (CDM) search algorithm.

**Provided by**: NeuronBridge API (not calculated locally)

**Range**: Typically 0 - 500,000+ (higher = better match)

**Types**:
- `cds_score`: Color Depth MIP Search score
- `pppm_score`: Point Pattern Matching score
- `combined_rank`: Rank when using both algorithms

---

### Coverage Ratio

**Definition**: The fraction of queried neurons that a driver line labels.

**Formula**:

$$\mathit{coverage} = \frac{\mathit{matches}}{\mathit{total}}$$

Where:
- `match_count` = Number of unique queried neurons this line matches
- `total_query_neurons` = Total number of neurons in the query

**Example**:
```
Query: ["aMe12", "MBON01", "KC"] (3 neurons)
Line SS01234 matches: aMe12, MBON01 (2 neurons)

coverage_ratio = 2 / 3 = 0.667 (66.7%)
```

**Implementation** ([`src/neuronbridge_finder.py`](../../src/neuronbridge_finder.py#L7073)):

```python
line_stats['coverage_ratio'] = line_stats['match_count'] / total_query_neurons
```

---

### Weighted Score

**Definition**: A composite score prioritizing lines that label more of the queried neurons with high morphological similarity.

**Formula**:

$$\mathit{weighted} = \mathit{mean} \times \mathit{coverage}$$

Or equivalently:

$$\mathit{weighted} = \mathit{mean} \times \frac{\mathit{matches}}{\mathit{total}}$$

**Sorting Modes** (controlled by `sort_by` parameter):

1. **`sort_by='max'`** (default) - Sort by average score
   - Formula: Sort by `agg_mean_score` (descending)
   - Prioritizes: Lines with highest morphological similarity
   - Use when: Finding best morphological matches regardless of coverage

2. **`sort_by='completeness'`** - Sort by weighted score
   - Formula: Sort by `weighted_score` (descending)
   - Prioritizes: Lines labeling MORE queried neurons
   - Use when: Finding lines that label ALL queried neuron types

**Interpretation**:
- A line labeling ALL N queried neurons with score S gets `weighted_score = S`
- A line labeling only 1 of N neurons with score S gets `weighted_score = S/N`

**Example**:
```
Query: 3 neurons

sort_by='max':
  Line A: avg_score=100,000, matches 3/3 → ranks #1 (high score)
  Line B: avg_score=80,000, matches 3/3 → ranks #2

sort_by='completeness':
  Line B: weighted_score=80,000 × 1.0 = 80,000 → ranks #1 (labels all)
  Line A: weighted_score=100,000 × 0.33 = 33,000 → ranks #2 (if only 1/3)
```

**Implementation** ([`src/neuronbridge_finder.py`](../../src/neuronbridge_finder.py#L7191-L7193)):

```python
line_stats['coverage_ratio'] = line_stats['match_count'] / total_query_neurons
line_stats['weighted_score'] = line_stats['agg_mean_score'] * line_stats['coverage_ratio']

# Sorting
if sort_by == 'completeness':
    line_stats = line_stats.sort_values('weighted_score', ascending=False)
else:  # sort_by == 'max'
    line_stats = line_stats.sort_values('agg_mean_score', ascending=False)
```

**Used in**: `find_lines_batch()` output ranking

---

### Cross-Dataset Score

**Definition**: For multi-dataset queries, measures how consistently a line labels neurons across different datasets.

**Formulas**:

1. **Min Score per Dataset**: Worst-case performance across datasets
   $$\mathit{minScore} = \min_{d \in \mathit{datasets}} \max_{n \in d} score_{n}$$

2. **Cross-Dataset Score**: Average of max scores across datasets
   $$\mathit{crossScore} = \frac{1}{|D|} \sum_{d \in D} \max_{n \in d} score_{n}$$

**Used for**: Finding lines that reliably label homologous neurons in multiple connectome datasets.

---

## Filter Parameters Summary

| Parameter                   | Level        | Formula Check           | Description                     |
| --------------------------- | ------------ | ----------------------- | ------------------------------- |
| `min_synapse_num`           | Edge (Dₜ)    | `weight >= t`           | Minimum synapse count (defines Dₜ) |
| `min_ratio`                 | Edge/Type    | `connection_ratio >= X` | Minimum input fraction          |
| `min_traversal_probability` | Edge/Type    | `traversal_prob >= X`   | Minimum propagation probability |

**Filtering Order** ([`src/coana.py`](../../src/coana.py)):

```python
# Applied BEFORE pathfinding (edge-level pre-filtering)
1. Fetch connections (weight ≥ 1)
2. Calculate connection_ratio, traversal_probability (global D_t denominators)
3. BodyId level: filter weight >= min_synapse_num, ratio, probability
   Type level:   threshold EDGES to D_t first, aggregate pairs, then filter
                 type ratio / compound probability
4. Build graph with filtered edges
5. Find paths
```

**Type-level consistency**: when `filter_by='type'`, the synapse cutoff
(`min_synapse_num`) is applied to **edges before** the type-pair aggregation,
and the type-level probability filter uses the **same `aggregate_method`**
(`'product'` by default) as the enrichment output — so the set of connections
that pass the filter is exactly the set whose type pairs satisfy the
displayed probabilities.

---

## Threshold Selection Guide

### Connection Ratio (`min_ratio`)

| Threshold | Meaning         | Use Case             |
| --------- | --------------- | -------------------- |
| 0.0       | No filter       | Exploratory analysis |
| 0.001     | ≥0.1% of inputs | Very permissive      |
| 0.01      | ≥1% of inputs   | Moderate filtering   |
| 0.03      | ≥3% of inputs   | Significant inputs   |
| 0.05      | ≥5% of inputs   | Strong connections   |
| 0.1       | ≥10% of inputs  | Major inputs only    |

### Traversal Probability (`min_traversal_probability`)

The conversion table applies to the **edge-level** model. At the **type
level**, the default product aggregation makes probabilities larger than any
single pair (parallel channels), so a type-level threshold tends to keep more
edges than the same threshold at the bodyId level — choose the threshold
according to the level you filter on.

| Threshold | Equiv. Ratio | Recommendation            |
| --------- | ------------ | ------------------------- |
| 0.001     | 0.03%        | Conservative (most paths) |
| 0.01      | 0.3%         | Moderate (balanced)       |
| 0.05      | 1.5%         | Strict (strong paths)     |
| 0.1       | 3%           | Very strict               |

### Conversion Formulas

```python
# ratio → probability (edge level)
min_traversal_probability = min_ratio / 0.3

# probability → ratio (edge level)
min_ratio = min_traversal_probability * 0.3
```

---

## Code References

### Core Calculation Locations

| Metric                  | File                         | Function/Line                                                        |
| ----------------------- | ---------------------------- | -------------------------------------------------------------------- |
| Dₜ aggregates           | `src/connection_map.py`      | [`ThresholdedConnectionMap`](../../src/connection_map.py#L28)        |
| Dₜ map (per cutoff)     | `src/coana.py`               | [`_connection_map()`](../../src/coana.py#L4005)                      |
| `connection_ratio`      | `src/coana.py`               | [`_fetch_connections_with_cache()`](../../src/coana.py#L3621)        |
| Type-level filters      | `src/coana.py`               | [`_apply_type_level_filters()`](../../src/coana.py#L4363)            |
| Pair-level probabilities| `src/coana.py`               | [`_pair_level_probabilities()`](../../src/coana.py#L4324)            |
| Type prob aggregation   | `src/statvis.py`             | [`_type_probability_series()`](../../src/statvis.py#L5149)           |
| Enrichment (pandas)     | `src/statvis.py`             | [`EnrichConnectionTable()`](../../src/statvis.py#L5173)              |
| Enrichment (Polars)     | `src/statvis.py`             | [`EnrichConnectionTablePolars()`](../../src/statvis.py#L6770)         |
| `path_prob`             | `src/statvis.py`             | [`build_path_dataframe_from_paths()`](../../src/statvis.py#L4881)    |
| `weighted_score`        | `src/neuronbridge_finder.py` | [`find_lines_batch()`](../../src/neuronbridge_finder.py#L7191-L7193) |

### Parameter Definitions

| Parameter                    | File           | Line                                        |
| ---------------------------- | -------------- | ------------------------------------------- |
| `min_ratio`                  | `src/coana.py` | [L1268](../../src/coana.py#L1268)           |
| `min_traversal_probability`  | `src/coana.py` | [L1276](../../src/coana.py#L1276)           |
| `filter_by`                  | `src/coana.py` | [L1284](../../src/coana.py#L1284)           |
| `aggregate_method`           | `src/coana.py` | [L1293](../../src/coana.py#L1293)           |

### Related Documentation

- [Connection Ratio Filter](./ConnectionRatio_Filter.md) - Detailed `min_ratio` guide
- [Traversal Probability Edge Filter](./TraversalProbability_EdgeLevelFilter.md) - Edge-level filtering
- [NeuronBridge Guide](./NeuronBridge_Guide.md) - EM↔LM matching scores
- [PathFinding Methods](./PathFinding_Methods.md) - Algorithm selection

---

## Summary

| Context               | Primary Metric          | Formula                                |
| --------------------- | ----------------------- | -------------------------------------- |
| **Graph definition**  | Dₜ                      | edges with `weight >= t` (per-cutoff aggregates cached) |
| **Single connection** | `traversal_probability` | `min(1, weight/total_incoming/0.3)`    |
| **Type→Type direct**  | `connection_ratio`      | `Σweights(Dₜ) / Σtotal_incoming(Dₜ)`   |
| **Type→Type prob**    | `traversal_probability` | `1 - Π(1-p_pair)` (product, default)   |
| **Multi-hop path**    | `path_prob`             | `Π(p_edge)`                            |
| **Driver line rank**  | `weighted_score`        | `avg_score × coverage`                 |

---

*Last updated: August 2026*
