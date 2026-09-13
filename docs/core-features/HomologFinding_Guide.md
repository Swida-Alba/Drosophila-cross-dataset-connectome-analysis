# HomologFinding Guide

## Overview

> **UI location:** Connectivity tab → **Find Homolog** sub-tab (Target = Source for the intra-dataset search; the sub-tab was renamed from "Find Similar" — backend names unchanged).

The `HomologFinder` module provides connectivity profile-based homolog discovery across connectome datasets. It identifies neurons with similar connectivity patterns, which often indicates they are the same cell type in different animals or brain regions.

**Key Architecture**: HomologFinder is built entirely on top of `ConnectivityProfiler`. All profile building uses the **1-hop/2-hop hybrid approach** with **top-k/top-m dynamic expansion**.

## Architecture

```
HomologFinder (profile_comparator.py)
│
├── Uses: ConnectivityProfiler (connectivity_profiler.py)
│         └── Builds all profiles via get_profile()
│
├── Uses: ProfileComparator (static methods)
│         └── Computes similarity scores
│
├── Uses: VisualizeSkeleton (coana.py)
│         └── 3D skeleton visualization of top candidates
│
└── Uses: FindNeuronConnection (coana.py)
          └── Fetches neuron info when needed
```

### Dependency Chain

```
HomologFinder
    │
    ▼
ConnectivityProfiler
    │
    ├── ProfilerConfig (1-hop/2-hop, top-k/top-m settings)
    │
    ├── ConnectivityProfile (bodyId-level profiles)
    │
    └── Cache System (memory → disk-index → disk-df)
```

## Profile Construction

All profiles are built via `ConnectivityProfiler.get_profile()`:

```python
# HomologFinder internally does:
profile = self.profiler.get_profile(neuron, dataset)
```

### The 1-hop/2-hop Hybrid Approach

For each neuron:
1. **Query 1-hop partners** (direct synaptic connections)
2. **Separate typed vs untyped** partners
3. **For untyped 1-hop partners**: fetch their 2-hop typed partners
4. **Apply top-k/top-m expansion** to ensure minimum type diversity

### Configuration

```python
from src.comparison import HomologFinder

finder = HomologFinder(
    # Profile construction (passed to ConnectivityProfiler)
    top_k=15,                    # Top K partners per direction
    top_m=5,                     # Minimum unique types
    min_synapse_threshold=3,     # Minimum synapses
    
    # Module defaults
    source='aMe12',
    source_dataset='hemibrain:v1.2.1',
    target_dataset='flywire_FAFB_v783',
    
    # Visualization options
    visualize_skeleton=True,     # Enable skeleton visualization
    visualize_top_n=5,           # Visualize top 5 candidates
    
    # Output
    output_dir='/path/to/output',
    verbose=True
)
```

## Comparison and Output

HomologFinder **always computes bodyId-level comparisons first**, then aggregates to type-level summaries. Both levels of results are always saved.

### Always-Dual Output

Every homolog search automatically generates both:
- **BodyId-level results**: Individual neuron-to-neuron matches
- **Type-level summary**: Aggregated type-to-type statistics

```python
results = finder.find_homologs_fast(
    source='Mi1',
    top_n=20
)
# Returns bodyId-level DataFrame
# Also saves type_summary.csv when output_dir is set
```

### Output Files (Always Saved)

When `output_dir` is set, you always get:
- `bodyid_results.csv` - Sorted by source_bodyId, then rank_union
- `type_summary.csv` - Aggregated type-level summary (avg_ metric means)
- `homolog_results.csv` - Legacy format (sorted by the run's metric)
- `type_level_results.csv` - Pooled type-level ranking (top N per source type)
- `source_status_summary.json` - Source connectivity status breakdown

## Skeleton Visualization

HomologFinder can automatically visualize the 3D morphology of top homolog candidates using the `VisualizeSkeleton` module.

### Enable Visualization

```python
finder = HomologFinder(
    visualize_skeleton=True,   # Enable visualization
    visualize_top_n=5,         # Visualize top 5 candidates
    output_dir='/path/to/output'
)

results = finder.find_homologs_fast(
    source='Mi1',
    source_dataset='hemibrain:v1.2.1',
    target_dataset='hemibrain:v1.2.1'
)
# Skeleton visualizations saved to: output/visualizations/
```

### Visualization Output

When `visualize_skeleton=True`:

```
{output_dir}/{saveas}/
├── results/
│   └── ...
├── visualizations/
│   ├── rank_1_{bodyId}_{type}.html    # Interactive 3D skeleton
│   ├── rank_1_{bodyId}_{type}.png     # Static image export
│   ├── rank_2_{bodyId}_{type}.html
│   ├── rank_2_{bodyId}_{type}.png
│   └── ...
```

### Per-Method Visualization

You can also enable visualization per-call:

```python
results = finder.find_homologs_fast(
    source='LC4',
    source_dataset='hemibrain:v1.2.1',
    target_dataset='flywire_FAFB_v783',
    visualize_skeleton=True,
    visualize_top_n=10
)
```

### Cross-dataset overlay labels and bodyId legends

For a cross-dataset search, the query neuron is transformed into the target
coordinate space and included in the target `bodyId_level/` and `type_level/`
scenes as a `query_transformed_*` overlay. In interactive HTML with
`legend_mode='tree'`:

- The query overlay's bodyId rows use the source neuron's native `type`, not a
  mapped crosswalk field. For example, an MCNS `SMP227` row remains `SMP227`
  even when its `flywireType` value is `CB1449,CB2843`.
- BodyId-level custom groups with one neuron are shown as one direct row with
  count `1`; skeleton and soma companion traces do not create a redundant
  child leaf or inflate the count.
- Groups containing multiple neurons remain expandable, and their counts are
  the number of unique neuron items. Static exports retain the ordinary Plotly
  legend.

Auto Type Mapping still resolves connectivity partner types for candidate
finding and comparison through the shared validity-aware resolver
(`comparison/type_resolver.py`): licensed renames and valid splits resolve,
conflicts fail closed (never matched by raw same-name), and unmapped names
fall back to raw. It does not rewrite the source neuron's native label in the
transformed visualization overlay. The run writes `auto_type_mapping.json`
with the mapper provenance and per-status resolution counts (basis
`unique_type_resolutions`).

## Finding Methods

### find_homologs()

Comprehensive search that builds profiles for all target neurons.

```python
results = finder.find_homologs(
    source='Mi1',
    source_dataset='hemibrain:v1.2.1',
    target_dataset='flywire_FAFB_v783',
    top_n=20,
    metric='combined',
    visualize_skeleton=True,
    visualize_top_n=5
)
```

### find_homologs_fast()

Fast search using adjacency expansion for candidate discovery.

```python
results = finder.find_homologs_fast(
    source='aMe12',
    source_dataset='hemibrain:v1.2.1',
    target_dataset='hemibrain:v1.2.1',
    top_n=10,
    min_shared_partners=2,
    min_weight=3,
    visualize_skeleton=True,
    visualize_top_n=5
)
```

#### Fast Search Algorithm

```
Step 1: Load connection cache (for candidate discovery)
    │
Step 2: Find 1-hop partners of source (upstream + downstream)
    │
Step 3: Find 2-hop neighbors via adjacency expansion:
    │   - Downstream of upstream (types receiving from same inputs)
    │   - Upstream of downstream (types sending to same outputs)
    │
Step 4: Build profiles via ConnectivityProfiler.get_profile()
    │   (1-hop/2-hop hybrid with top-k/top-m expansion)
    │
Step 5: Compute similarity scores (bodyId-level)
    │
Step 6: Aggregate to type-level summary
    │
Step 7: Visualize top candidates (if enabled)
    │
Step 8: Return ranked results
```

### find_novel_homologs()

Shortcut for same-dataset discovery (excludes same-type matches).

```python
results = finder.find_novel_homologs(
    query='Dm9',
    dataset='hemibrain:v1.2.1',
    top_n=10,
    min_score=0.3
)
```

## Connectivity Status Hierarchy

HomologFinder uses a hierarchical classification to assess profile quality before comparison:

| Status                 | Criteria                         | Behavior                  | Log Message                                     |
| ---------------------- | -------------------------------- | ------------------------- | ----------------------------------------------- |
| `NONE`                 | 0 partners                       | **SKIPPED**               | Skipping source (0 partners)                    |
| `RARE`                 | < 5 partners                     | Included with **WARNING** | WARNING: source has RARE status                 |
| `INCOMPLETE`           | < top_k partners                 | Included with Warning     | Warning: source has INCOMPLETE status           |
| `INCOMPLETE_EXPANSION` | < top_m types                    | Included with Warning     | Warning: source has INCOMPLETE_EXPANSION status |
| `COMPLETE`             | ≥ top_k partners & ≥ top_m types | Included normally         | (normal processing)                             |

### Status Tracking in Output

When `output_dir` is set, the `source_status_summary.json` file tracks:

```json
{
  "skipped_sources": {
    "none": ["bodyId1", "bodyId2"],
    "total": 2
  },
  "warned_sources": {
    "rare": ["bodyId3", "bodyId4"],
    "total": 2
  },
  "included_for_comparison": 15,
  "status_hierarchy": {
    "none": "No connections (0 partners) - SKIPPED entirely",
    "rare": "Rare connectivity (<5 partners) - INCLUDED with WARNING",
    "incomplete": "Fewer than top_k partners - included with Warning",
    "incomplete_expansion": "Has top_k but fewer than top_m types - included with Warning",
    "complete": "Full profile meeting all criteria - included normally"
  }
}
```

## Similarity Metrics

| Metric               | Description               | Range   | Best For            |
| -------------------- | ------------------------- | ------- | ------------------- |
| `jaccard`            | Partner set overlap       | 0-1     | Quick screening     |
| `cosine`             | Weight vector similarity  | 0-1     | Weight importance   |
| `rank` / `rank_corr` | Spearman rank correlation | -1 to 1 | Rank order matching |
| `combined`           | Weighted average          | 0-1     | General use         |

### Custom Metric Weighting

You can specify custom weights for similarity metrics using a dictionary:

```python
finder = HomologFinder(
    similarity_metric={
        'rank_corr': 0.5,  # 50% weight on rank correlation
        'jaccard': 0.3,    # 30% weight on Jaccard
        'cosine': 0.2      # 20% weight on cosine
    }
)
```

When using a dict, results are sorted by the weighted combination of all specified metrics.

### Interpretation

| Score Range | Meaning                              |
| ----------- | ------------------------------------ |
| > 0.7       | Strong match - likely same cell type |
| 0.5 - 0.7   | Moderate match - possibly related    |
| < 0.5       | Weak match - likely different types  |

## Output Format

### bodyid_results.csv (Always Saved)

Sorted by `source_bodyId`, then `rank_corr` (descending):

```
| source_bodyId | source_type | target_bodyId | target_type | rank_corr | jaccard | cosine | source_status | target_status | adjacency_score |
| ------------- | ----------- | ------------- | ----------- | --------- | ------- | ------ | ------------- | ------------- | --------------- |
| 720575940...  | Mi1         | 720575940...  | Mi1         | 0.92      | 0.78    | 0.85   | COMPLETE      | COMPLETE      | 15              |
| 720575940...  | Mi1         | 720575940...  | Mi4         | 0.71      | 0.52    | 0.68   | COMPLETE      | INCOMPLETE    | 8               |
| 720575940...  | Mi1         | 720575940...  | Tm3         | 0.65      | 0.45    | 0.58   | RARE          | COMPLETE      | 12              |
```

**New Columns:**
- `source_status`: ConnectivityStatus of the source neuron (NONE, RARE, INCOMPLETE, INCOMPLETE_EXPANSION, COMPLETE)
- `target_status`: ConnectivityStatus of the target neuron
- `adjacency_score`: Number of shared partners found during adjacency expansion (candidate-finding phase, not comparison overlap)

### type_summary.csv (Always Saved)

Type-mean aggregated FROM the bodyId-level results (a bodyId-level
aggregation view, not a type-level profile comparison):

```
| query | source_dataset | target_dataset | source_type | target_type | n_bodyid_comparisons | avg_jaccard | avg_rank_union |
| ----- | -------------- | -------------- | ----------- | ----------- | -------------------- | ----------- | -------------- |
| Mi1   | hemibrain      | flywire_FAFB   | Mi1         | Mi1         | 12                   | 0.72        | 0.65           |
| Mi1   | hemibrain      | flywire_FAFB   | Mi1         | Mi4         | 12                   | 0.41        | 0.30           |
```

Targets that never resolved to a real cell type (untyped neurons whose
`type` column carries a coarse class label, e.g. FAFB `optic_lobes`) are
excluded from this table. `target_type_members` in
`type_level_results.csv` exposes the candidate type's member count so
hemilineage-scale annotations (e.g. Mi15, ~1000 members) are visible.

When scenes were rendered, `type_summary.csv` also carries `visualized`
(the type appears in a rendered scene) and `visualization_rank` (its order
across the rendered scenes — type-level scene first, then the bodyId-level
scene).

### Output layout notes

- Multi-type runs group per-type output under `by_type/<query_type>/`; a run
  that resolves to a single query type writes `results/`, `profiles/`,
  `overlaps/` and `visualization/` directly into the run folder (no
  `by_type/` nesting).
- Cross-dataset runs plot the query source inside the target brain as the
  `query_transformed_*` overlay layer of the `bodyId_level/` / `type_level/`
  scenes; `visualization/source_neurons/` is only a fallback scene in the
  source dataset's own template, drawn when the transform is unavailable.
- Individual per-neuron profile exports are disabled by default
  (`individual_profiles=False`); the batch scene HTML/PNG and the overlay
  layer cover the visual output.
- Morphological similarity of results against the transformed query neurons
  (the former `morph_v2_similarity`/`morph_nblast` columns and
  `results/morph_similarity.csv`) is **disabled** — cross-dataset scores
  were weak discriminators and the per-run computation was expensive. The
  library function `morphology.enrich_homolog_results` remains available
  for standalone use.
- The `type_level/` scene is driven by the pooled all-adjacency
  `type_level_results.csv` (real type-level ranking, `rank` order), not the
  bodyId-mean `type_summary.csv`. Each listed type renders with its FULL
  membership in the target dataset.

### type_level_results.csv (Always Saved)

True type-level homolog ranking: pooled all-adjacency type profiles (every
connection of each type's neurons, no top-k truncation) scored against every
typed target type. One row per source×target type with `target_type_members`
(the candidate type's member count — very large counts indicate coarse or
hemilineage-scale annotations, e.g. Mi15), `jaccard`, `weighted_jaccard`,
`cosine`, `rank_union` and a 1-based `rank` per source type (1 = best under
the run's metric). This is a separate computation from `type_summary.csv`,
not a re-aggregation of bodyId scores.

### homolog_results.csv (Legacy Format)

Sorted by the run's similarity metric (for backward compatibility).

## Result Saving

When `output_dir` is set, results are automatically saved with **both bodyId-level and type-level outputs**:

```
{output_dir}/{saveas}/
├── README.txt                    # Parameters and summary
├── results/
│   ├── bodyid_results.csv        # BodyId-level comparisons
│   │                              # Sorted by: source_bodyId, then rank_union
│   │                              # Columns: source_bodyId, source_type, 
│   │                              #          target_bodyId, target_type,
│   │                              #          rank_union, jaccard, cosine,
│   │                              #          source_status, target_status,
│   │                              #          adjacency_score, ...
│   ├── type_summary.csv          # Type-mean from bodyId level
│   │                              # Columns: source_type, target_type,
│   │                              #          n_bodyid_comparisons,
│   │                              #          avg_jaccard, avg_rank_union
│   │                              # Sorted by: similarity_metric (descending)
│   ├── type_level_results.csv    # Pooled type-level ranking (top N per source type)
│   ├── source_status_summary.json # ConnectivityStatus breakdown
│   │                              # Tracks skipped (NONE) and warned (RARE) sources
│   ├── homolog_results.csv       # Legacy format (sorted by the run's metric)
│   └── shuffle_test.json         # Shuffle test stats (if run_shuffle_test=True)
├── profiles/
│   ├── query/                    # Query neuron profile
│   │   └── source_bodyids.csv    # List of source bodyIds
│   └── matches/                  # Top match profiles
│       └── top_target_bodyids.csv # List of top target bodyIds
├── overlaps/                     # Partner overlap details
└── visualization/                # Skeleton visualizations (if enabled)
    ├── source_neurons/           # All source neurons together
    │   └── all_sources.html      # Multi-neuron visualization (legend_mode='single')
    ├── bodyid_level/             # Individual bodyId visualizations
    │   ├── {type}_{bodyId}.html  # Each source/target bodyId separately
    │   └── ...                   # neuron_alpha=0.6 for single neurons
    └── type_level/               # Type-aggregated visualizations
        ├── {type}.html           # All bodyIds of each type together
        └── ...
```

### Why Both Levels?

Since we always compute bodyId-level profiles first and then aggregate to type-level,
both outputs are generated at no extra computational cost:

- **bodyid_results.csv**: Shows exact neuron-to-neuron matches, useful for:
  - Identifying which specific neurons match best
  - Analyzing within-type variation
  - Validating individual neuron assignments

- **type_summary.csv**: Aggregated view, useful for:
  - Quick overview of type-to-type similarities
  - Identifying candidate homologous types
  - Statistical summary (avg, best, std, count)

## Usage Examples

### Example 1: Basic Homolog Search

```python
from src.comparison import HomologFinder

finder = HomologFinder(
    source='aMe12',
    source_dataset='hemibrain:v1.2.1',
    target_dataset='hemibrain:v1.2.1',
    output_dir='/path/to/output'
)

# Run search
results = finder.find_homologs_fast(top_n=20)
# Both bodyid_results.csv and type_summary.csv are saved
```

### Example 2: Cross-Dataset Comparison with Visualization

```python
results = finder.find_homologs_fast(
    source='LC4',
    source_dataset='hemibrain:v1.2.1',
    target_dataset='flywire_FAFB_v783',
    top_n=50,
    visualize_skeleton=True,
    visualize_top_n=10
)

# Filter strong matches
strong_matches = results[results['rank_corr'] > 0.7]

# Skeleton visualizations saved to output/visualizations/
```

### Example 3: Analyzing Type with Multiple Neurons

```python
# Find best matches for each neuron in type 'Mi1'
results = finder.find_homologs_fast(
    source='Mi1',
    top_n=5  # Top 5 per source bodyId
)

# Group by source neuron
for src_bid in results['source_bodyId'].unique():
    matches = results[results['source_bodyId'] == src_bid]
    print(f"\nMatches for {src_bid}:")
    print(matches[['target_type', 'rank_corr']].head())
```

### Example 4: Partner Overlap Analysis

```python
# Get detailed partner comparison
overlap = finder.get_partner_overlap(
    query='Mi1',
    source_dataset='hemibrain:v1.2.1',
    target_type='Mi1',
    target_dataset='flywire_FAFB_v783',
    direction='upstream'
)

print(overlap[['partner', 'source_weight', 'target_weight', 'overlap']])
```

## API Reference

### HomologFinder.__init__()

```python
HomologFinder(
    source: Optional[Union[str, int]] = None,
    source_dataset: Optional[str] = None,
    target_dataset: Optional[str] = None,
    output_dir: Optional[str] = None,
    saveas: Optional[str] = None,
    top_k: int = 15,
    top_m: int = 5,
    min_synapse_threshold: int = 3,
    use_cache: bool = True,
    visualize_skeleton: bool = False,
    visualize_top_n: int = 5,
    verbose: bool = True
)
```

| Parameter               | Type     | Default     | Description                            |
| ----------------------- | -------- | ----------- | -------------------------------------- |
| `source`                | str/int  | None        | Default source neuron (type or bodyId) |
| `source_dataset`        | str      | None        | Default source dataset                 |
| `target_dataset`        | str      | None        | Default target dataset                 |
| `output_dir`            | str      | None        | Output directory for results           |
| `saveas`                | str      | None        | Subfolder name for results             |
| `top_k`                 | int      | 15          | Top K partners per direction           |
| `top_m`                 | int      | 5           | Minimum unique types                   |
| `min_synapse_threshold` | int      | 3           | Minimum synapse count                  |
| `use_cache`             | bool     | True        | Use connection cache                   |
| `visualize_skeleton`    | bool     | False       | Enable skeleton visualization          |
| `visualize_top_n`       | int      | 5           | Number of top candidates to visualize  |
| `similarity_metric`     | str/dict | 'rank_corr' | Metric for sorting results (see below) |
| `verbose`               | bool     | True        | Enable verbose logging                 |

**similarity_metric Options:**
- String: `'rank_corr'`, `'jaccard'`, `'cosine'`, `'combined'`
- Dict: `{'rank_corr': 0.5, 'jaccard': 0.3, 'cosine': 0.2}` (weighted combination)

### find_homologs()

```python
finder.find_homologs(
    source: Optional[Union[str, int]] = None,
    source_dataset: Optional[str] = None,
    target_dataset: Optional[str] = None,
    top_n: int = 20,
    metric: str = 'combined',
    direction: str = 'both',
    min_score: float = 0.0,
    show_progress: bool = True,
    output_dir: Optional[str] = None,
    saveas: Optional[str] = None,
    visualize_skeleton: Optional[bool] = None,
    visualize_top_n: Optional[int] = None
) -> pd.DataFrame
```

### find_homologs_fast()

```python
finder.find_homologs_fast(
    source: Optional[Union[str, int]] = None,
    source_dataset: Optional[str] = None,
    target_dataset: Optional[str] = None,
    top_n: int = 20,
    min_shared_partners: int = 2,
    min_weight: int = 3,
    show_progress: bool = True,
    output_dir: Optional[str] = None,
    saveas: Optional[str] = None,
    include_partner_details: bool = True,
    top_n_details: int = 10,
    visualize_skeleton: Optional[bool] = None,
    visualize_top_n: Optional[int] = None
) -> pd.DataFrame
```

## Deprecated Methods

The following methods are **deprecated** and should not be used:

- `_build_profile_from_aggregates()` - Bypasses 1-hop/2-hop hybrid
- `_build_profile_from_bodyid_aggregates()` - Bypasses 1-hop/2-hop hybrid

All profile building now goes through `ConnectivityProfiler.get_profile()`.

## Best Practices

1. **Use `find_homologs_fast()` for large searches** - Much faster with adjacency expansion
2. **Check `is_same_type` column** - Distinguish same-type matches from novel discoveries
3. **Use output_dir for reproducibility** - Saves parameters and results
4. **Set appropriate top-k/top-m** - Higher values for cross-dataset, lower for same-dataset
5. **Enable visualization for key results** - Review skeleton morphology of top candidates
6. **Run random control tests** - Validate that results are meaningful (see below)

---

## Random Control Test

The random control test validates whether homolog finding results are statistically meaningful by comparing real results against randomized profiles.

### Why Use Random Control?

Homolog finding is based on connectivity profile similarity. But how do we know if high similarity scores are meaningful, or just due to chance? The random control test:

1. **Shuffles** the source profile (keeping weights, randomizing partner types)
2. **Runs** homolog finding on N shuffled profiles
3. **Compares** real scores against the shuffled distribution
4. **Reports** statistical significance (p-value, effect size)

### Usage

```python
from src.comparison import HomologFinder

finder = HomologFinder(
    source_dataset='hemibrain:v1.2.1',
    target_dataset='male-cns:v0.9',
    verbose=True
)

# Run control test with 100 shuffles
results = finder.run_random_control_test(
    source='Mi1',
    source_dataset='hemibrain:v1.2.1',
    target_dataset='male-cns:v0.9',
    n_shuffles=100,
    seed=42,  # For reproducibility
    output_dir='/path/to/output'
)

# Check significance
if results['is_significant']:
    print("✓ Homolog finding results are meaningful!")
    print(f"  P-value: {results['p_value']:.4f}")
    print(f"  Effect size: {results['effect_size']:.4f}")
else:
    print("✗ Results may be due to chance")
```

### Output

The control test returns a dictionary with:

| Key                    | Description                                        |
| ---------------------- | -------------------------------------------------- |
| `real_results`         | DataFrame of homolog results with real profile     |
| `real_mean_score`      | Mean similarity score from real profile            |
| `shuffled_mean_scores` | List of mean scores from each shuffle              |
| `p_value`              | Proportion of shuffles with higher score than real |
| `z_score`              | Standard deviations above shuffle mean             |
| `effect_size`          | Cohen's d effect size                              |
| `is_significant`       | True if p < 0.05                                   |
| `summary`              | Human-readable interpretation                      |

### Interpretation Guide

| P-value | Effect Size | Interpretation                        |
| ------- | ----------- | ------------------------------------- |
| < 0.001 | > 0.8       | Highly significant, large effect      |
| < 0.01  | > 0.5       | Significant, medium effect            |
| < 0.05  | > 0.2       | Marginally significant, small effect  |
| ≥ 0.05  | any         | Not significant, may be due to chance |

### Example Output

```
============================================================
RANDOM CONTROL TEST RESULTS
============================================================
Source neuron: Mi1
Source dataset: hemibrain:v1.2.1
Target dataset: male-cns:v0.9
Number of shuffles: 100

SCORES:
  Real profile mean rank_corr: 0.7234
  Shuffled mean (avg): 0.4521 ± 0.0832

STATISTICS:
  P-value (mean): 0.0000 ***
  Z-score: 3.26
  Effect size (Cohen's d): 3.26

INTERPRETATION:
  ✓ SIGNIFICANT with LARGE effect size
  → Homolog finding results are highly meaningful
============================================================
```

### Saved Files

When `output_dir` is specified:

```
{output_dir}/control_test_{source}_{timestamp}/
├── real_results.csv              # Homolog results with real profile
├── control_test_stats.csv        # Statistics summary
├── shuffled_score_distribution.csv  # All shuffle scores
└── summary.txt                   # Human-readable summary
```

### shuffle_profile() Method

You can also manually shuffle profiles:

```python
# Get original profile
profile = finder.profiler.get_profile('Mi1', 'hemibrain:v1.2.1')

# Create shuffled version
shuffled = finder.shuffle_profile(profile, seed=42)

# Shuffled profile has:
# - Same weights (same distribution)
# - Different partner type assignments (randomized)
# - Preserved rank structure
```

---

## See Also

- [ConnectivityProfiler_Guide.md](./ConnectivityProfiler_Guide.md) - Profile construction details
- [CrossDatasetComparison_Guide.md](./CrossDatasetComparison_Guide.md) - Cross-dataset workflows
- [CacheSystem_Guide_v4.md](./CacheSystem_Guide_v4.md) - Cache architecture

## Morph Qualification (cross-dataset, optional)

With **Morph Qualification** enabled (requires visualization and a
cross-dataset target among FAFB / male-cns / BANC), the visualized top-N
candidates get a morphological second opinion on top of the connectivity
ranking:

- the query skeleton is transformed into the target's render space and
  scored with the production **vector_v2** scorer (no NBLAST
  cross-dataset);
- the bar is **null-calibrated per query**: the p95 of `morph_null_k`
  (default 200) seeded random target neurons, plus a user-set offset —
  raw cross-dataset morph scores have a high, query-dependent baseline,
  so only the distance above the noise floor is meaningful;
- candidates below the bar are **excluded from the rendered scenes**; the
  results tables keep every row and gain `morph_v2`, `morph_null_p95`,
  `morph_z` and `morph_qualified` on the scored rows, with a
  `results/morph_qualification.json` provenance summary;
- with the option off, output is byte-identical to the classic behavior.

Shared implementation with the Morphology tab's Cross-Dataset comparison:
`src/comparison/morph_cross_dataset.py`.
