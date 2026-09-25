# Core Features Documentation

Documentation for the fundamental features of the DROCAT toolkit.

## Overview

This directory contains documentation for the core analytical capabilities:

### 🌉 EM↔LM Integration (NEW!)
- **✨ [NeuronBridge Integration](./NeuronBridge_Guide.md)**: Bidirectional mapping between EM reconstructions and LM driver lines
- **✨ [NeuronBridge Workflow Guide](./NeuronBridge_Workflow.md)**: Complete workflow with calling tree and recommendations ⭐
- **✨ [FlyLight Downloader](./FlyLight_Guide.md)**: Download FlyLight imagery from Janelia S3 bucket

### 🧬 Connectivity Analysis
- **✨ Module Calling Tree**: Visual architecture and dependency relationships
- **✨ Connectivity Profiler**: 1-hop/2-hop hybrid profile building approach
- **✨ Homolog Finding**: Find homologs across datasets using connectivity profiles
- **✨ Cross-Dataset Comparison**: Compare connectivity across multiple datasets
- **✨ Connectivity Profile Verification**: Verify neuron types using connectivity fingerprints
- **[Path Finding Methods](./PathFinding_Methods.md)**: Comparison of Bidirectional, DP, and DFS algorithms (with measured time/memory evaluation; see also [technical evaluation](../technical/PATHFINDING_ALGORITHM_EVALUATION.md))
- **Custom Groups**: Flexible neuron grouping for custom analysis
- **Cache System**: High-performance local data storage
- **Filtering**: Connection and neuron filtering options

---

## 🌉 NeuronBridge & FlyLight Integration (NEW!)

### [NeuronBridge Integration Guide](./NeuronBridge_Guide.md)

Bidirectional mapping between EM body IDs and LM driver lines using the NeuronBridge API.

**Key Features**:
- **EM → LM**: Find driver lines (GAL4, LexA, Split-GAL4) matching EM body IDs
- **LM → EM**: Find EM neurons matching driver line names
- **Match Types**: CDS (Color Depth Search), PPPM (PatchPerPixMatch), or combined ranking
- **Multi-Dataset**: Support for hemibrain, male-cns, FAFB, and standalone BANC
- **Batch Processing**: Process multiple queries with automatic aggregation
- **Image Downloads**: Integrated with FlyLight for imagery access

**Quick Example**:
```python
from src.neuronbridge_finder import NeuronBridgeFinder

# Initialize with custom settings
nbf = NeuronBridgeFinder(
    region='Brain',      # Filter for Brain region only
    match_type='both'    # Use both CDS and PPPM by default
)

# Find driver lines for an EM body ID
lines = nbf.id_to_lines(720575940621039145)  # Uses match_type='both'

# Find EM neurons for a driver line
neurons = nbf.line_to_neuron('LH173')  # Uses match_type='both', region='Brain'

# Override settings at method level if needed
results = nbf.find_lines_batch(
    queries='MBON01,MBON02',
    dataset='hemibrain:v1.2.1',
    match_type='cds',  # Override to use CDS only
    download_images='flylight',
    simple_mode=True
)

# VNC-specific search
nbf_vnc = NeuronBridgeFinder(region='VNC', match_type='pppm')
vnc_lines = nbf_vnc.id_to_lines(123456789)
```

### [FlyLight Downloader Guide](./FlyLight_Guide.md)

Direct access to FlyLight imagery from the Janelia S3 bucket and HTTP CDN.

**Key Features**:
- **Multiple Sources**: S3 bucket for R-lines/SS-lines, HTTP CDN for VT lines
- **Collection Filtering**: GAL4/LexA, SplitGAL4, MCFO, RawImages
- **File Types**: PNG, JPG, H5J (3D stacks), LSM (raw confocal), MP4 (videos), JSON (metadata)
- **Simple Mode**: Intelligent filename filtering to reduce download volume (up to 95% reduction)
- **Parallel Downloads**: Multi-threaded downloading for efficiency

**Quick Example**:
```python
from src.flylight_downloader import FlyLightDownloader, download_flylight_images

# Quick download
paths = download_flylight_images('SS01015', output_dir='./images', simple_mode=True)

# Advanced usage
downloader = FlyLightDownloader(
    collection_category='SplitGAL4',
    formats=['png', 'jpg'],
    image_types='mip',
    simple_mode=True
)
files = downloader.get_filtered_files('SS01015')  # 241 → 13 files
```

---

## ✨ Module Calling Tree

### [Module Calling Tree](./Module_Calling_Tree.md)
Visual architecture guide showing how all modules in the project interact.

**Key Topics**:
- Module dependency hierarchy
- Calling flows for each feature (Connection Query, Profile Building, Homolog Finding, Comparison)
- Import graph for all modules
- Key dependency rules and best practices

**Overview**:
```
User Entry Points
    │
    ├── FindNeuronConnection (coana.py)
    │       └── Connection queries, caching
    │
    ├── ConnectivityProfiler (connectivity_profiler.py)
    │       └── 1-hop/2-hop hybrid profile building
    │
    ├── HomologFinder (profile_comparator.py)
    │       └── Homolog discovery via profiles
    │
    └── VisualizePath (vispath-subproject)
            └── Network visualization (standalone)
```

---

## Path Finding Methods

### [Path Finding Methods](./PathFinding_Methods.md)
Detailed comparison of the available pathfinding algorithms in `FindAllPath`.

**Key Topics**:
- **StrongestFirst**: The built-in pipeline default since 2026-09-05 (the algorithm selector was removed).
- **Memoized DFS (forward)**: The fastest measured complete enumerator (API-only; no reversed-graph copy).
- **Optimized Backward Search (DP)**: Robust fallback; best at shallow depths.
- **DFS (backward memoized)**: Best for deep paths with few targets.
- **Meet-in-the-middle**: Fast at shallow depths; competitive for deep paths.
- **Bidirectional Search**: Shortest paths first, but highest memory.

**Comparison Table** (2026-08 measured; see [PATHFINDING_ALGORITHM_EVALUATION.md](../technical/PATHFINDING_ALGORITHM_EVALUATION.md)):
| Algorithm | Speed | Memory | Best Use Case |
| :--- | :--- | :--- | :--- |
| **Memoized DFS (forward)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **General purpose (2026-08 benchmark default; StrongestFirst is the pipeline default now)** |
| **DFS (backward)** | ⭐⭐⭐⭐ | ⭐⭐⭐ | Few targets, deep paths |
| **Meet-in-the-middle** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Shallow queries |
| **DP (Backward)** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Robust fallback, shallow |
| **Bidirectional** | ⭐⭐⭐ | ⭐ | Shortest-first, memory to spare |

---

## ✨ Connectivity Profiler

### [Connectivity Profiler Guide](./ConnectivityProfiler_Guide.md)
Core module for building connectivity profiles using the 1-hop/2-hop hybrid approach.

**Key Topics**:
- 1-hop/2-hop hybrid approach with top-k/top-m expansion
- ProfilerConfig settings and customization
- Cache system integration (memory → disk-index → disk-df)
- Fuzzy type matching for cross-dataset comparison
- Profile aggregation and normalization

**Architecture**:
```
ConnectivityProfiler
    │
    ├── ProfilerConfig
    │   - top_k_bodyid=15 (top K partners per direction)
    │   - top_m_type=5 (min unique types)
    │   - expand_untyped_2hop=True (fetch 2-hop for untyped)
    │
    └── ConnectivityProfile (output)
        - upstream_partners: Dict[str, float]
        - downstream_partners: Dict[str, float]
        - metadata (bodyId, type, dataset)
```

**Quick Start**:
```python
from src.comparison import ConnectivityProfiler, ProfilerConfig

# Configure profiler
config = ProfilerConfig(
    top_k_bodyid=15,        # Top 15 partners per direction
    top_m_type=5,           # Ensure at least 5 unique types
    expand_untyped_2hop=True,  # Fetch 2-hop for untyped partners
    min_synapse_threshold=3
)

# Create profiler
profiler = ConnectivityProfiler(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    config=config
)

# Get profile for a neuron
profile = profiler.get_profile('aMe12', 'hemibrain:v1.2.1')
print(profile.upstream_partners)
print(profile.downstream_partners)
```

**Confidence Thresholds**:
- **Jaccard**: >0.5 Very High, >0.3 High, >0.2 Medium, >0.1 Low
- **Rank Correlation**: ≥0.85 Very High, 0.7-0.85 High, 0.5-0.7 Medium, 0.3-0.5 Low

---

## ✨ Homolog Finding

### [Homolog Finding Guide](./HomologFinding_Guide.md)
Complete guide to finding homologous neurons across datasets using connectivity profiles.

**Key Topics**:
- Finding homologs by connectivity fingerprint comparison
- Fast discovery via adjacency expansion (2-hop neighbors)
- Loose mode (type-level) vs Strict mode (per-bodyId)
- Top-k/top-m profile construction rules
- Similarity metrics: Jaccard, cosine, rank correlation

**Quick Start**:
```python
from src.comparison import HomologFinder

# Initialize finder with top-k/top-m settings
finder = HomologFinder(
    top_k=15,   # Top 15 partners per direction
    top_m=5,    # Ensure at least 5 unique types
    use_cache=True
)

# Fast search via adjacency expansion
results = finder.find_homologs_fast(
    query='aMe12',
    source_dataset='flywire_FAFB_v783',
    target_dataset='hemibrain:v1.2.1',
    mode='loose',  # Type-level comparison
    top_n=20
)

# Comprehensive search
results = finder.find_homologs(
    query='Mi1',
    source_dataset='hemibrain:v1.2.1',
    target_dataset='flywire_FAFB_v783',
    mode='strict'  # Per-bodyId results for type queries
)
```

**Example Script**: [`examples/comparison/Example_HomologFinding.py`](../../archive/examples/comparison/Example_HomologFinding.py)

---

## ✨ Cross-Dataset Comparison (NEW)

### [Cross-Dataset Comparison Guide](./CrossDatasetComparison_Guide.md)
Complete guide to comparing neural connectivity across multiple connectome datasets.

**Key Topics**:
- Comparing hemibrain, male-cns, FAFB, and standalone BANC datasets
- Path-based vs edge-based comparison modes
- Conservation analysis and similarity metrics
- Interactive HTML reports with network visualizations
- Understanding output files and calculated parameters

**Quick Start**:
```python
from comparison import ComparisonParameters, ComparisonAnalyzer

params = ComparisonParameters(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9', 'flywire_FAFB_v783'],
    datasets_nickname=['hemi', 'mcns', 'fafb'],
    source_neurons=['aMe12'],
    target_neurons=['PPL101'],
    max_interlayer=1,
    thresholds=[1, 5, 10],
    comparison_mode='edge',  # or 'path'
    output_folder='/path/to/output',
)

analyzer = ComparisonAnalyzer(params, verbose=True)
results = analyzer.run_comparison()
```

**Example Script**: [`examples/Example_InterDatasetComparison.py`](../../archive/examples/comparison/Example_InterDatasetComparison.py)

---

## Path Finding

### [FindAllPath Documentation](./FindAllPath_Documentation.md)
Complete guide to graph-based pathfinding algorithms.

**Key Topics**:
- Finding multi-hop paths between neuron populations
- Depth and weight cutoffs
- Forward-only mode for biological realism
- Path validation and filtering

**Quick Start**:
```python
from findpath import FindAllPath

fap = FindAllPath(
    token='your_token',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['KC.*'],
    targetNeurons=['MBON03'],
    max_depth=3,
    min_weight=1,
    forward_only=True
)

paths_df = fap.find_all_paths()
```

### Related Documents
- [Custom Groups Feature](./CustomGroups_Feature.md)
- [Forward-Only Mode Guide](./ForwardOnly_Guide.md)
- [Multiple Connections Example](./FindAllPath_MultipleConnections_Example.md)
- [Caching Improvements](./FindAllPath_CachingImprovement.md)
- [Path Analysis Zero Weight Filter](./PathAnalysis_ZeroWeightFilter.md)

---

## Cache System

### [Cache System Guide](./CacheSystem_Guide.md)
Comprehensive guide to the local caching system that speeds up repeated
queries by serving them from local storage.

**Key Topics**:
- First-time setup and configuration
- Cache database architecture
- Storage optimization (40-60% smaller)
- Handling complete datasets

**Quick Start**:
```python
from findpath import FindNeuronConnection

fc = FindNeuronConnection(
    token='your_token',
    dataset='hemibrain:v1.2.1',
    output_dir='/path/to/cache',  # Enable caching
    use_cache=True
)
```

### Cache System Documents

#### Getting Started
- **[Quick Start](./CacheSystem_QuickStart.md)**: Get caching up and running in 5 minutes
- **[Complete Dataset](./CacheSystem_CompleteDataset.md)**: Handling type=None neurons efficiently

#### Technical Details
- **[Architecture v3](./CacheSystem_v3_DatabaseArchitecture.md)**: Database design and implementation
- **[Storage Optimization](./CacheSystem_StorageOptimization.md)**: Why cache is 40-60% smaller
- **[v4 Implementation](./CacheSystem_v4_Implementation.md)**: Latest improvements and pair-level caching
- **[v4 Complete](./CacheSystem_v4_Complete.md)**: Comprehensive v4 documentation
- **[v4 Pair-Level Proposal](./CacheSystem_v4_PairLevel_Proposal.md)**: Design rationale

### Performance Impact

| Operation | Without Cache | With Cache | Speedup |
|-----------|---------------|-----------|---------|
| First query (type-level) | ~2-5 seconds | ~2-5 seconds | 1x |
| Repeat query (same types) | ~2-5 seconds | ~0.05 seconds | **40-100x** |
| Repeat query (different types, cached) | ~2-5 seconds | ~0.1-0.2 seconds | **10-25x** |
| Large pathway analysis | Minutes | Seconds | **10-60x** |

---

## Performance Optimization

### Polars Integration

The toolkit uses Polars for high-performance data operations:

**Key Benefits**:
- 10-100x faster CSV writing
- 5-50x faster matrix generation
- 2-10x faster DataFrame operations

**Automatically Used In**:
- ComparisonAnalyzer output generation
- HomologFinder result saving
- Profile aggregation operations

### Internal Parallel Processing

Some internal operations use ThreadPoolExecutor for parallel processing:

**Profile Building** (in HomologFinder):
- Automatic worker count optimization
- Deferred cache writes for reduced I/O
- Memory-safe batch processing

See [Module Calling Tree](./Module_Calling_Tree.md) for architecture details.

---

## Filtering

### Connection Filtering

#### [Connection Ratio Filter](./ConnectionRatio_Filter.md)
Filter connections by the proportion of source neuron's output.

**Use case**: Find connections that represent significant portion of source neuron's output
```python
min_connection_ratio = 0.05  # 5% or more of source output
```

#### [Traversal Probability Filter](./TraversalProbability_EdgeLevelFilter.md)
Filter by likelihood of signal traversal through connection.

**Use case**: Focus on functionally significant connections
```python
min_traversal_probability = 0.01  # 1% or higher probability
```

#### [Filter Modes Fix](./FILTER_MODES_FIX.md)
Documentation of filter mode corrections and proper usage.

### Neuron Filtering

#### [FilterBy Feature](./FilterBy_Feature.md)
Filter neurons by various criteria before analysis.

**Criteria**:
- Instance vs type
- Status (Traced, Roughly traced, etc.)
- Size (synapse count)
- ROI (brain region)

**Quick Reference**: [FilterBy_QuickRef.md](./FilterBy_QuickRef.md)

---

## Path Finding Optimization

### Algorithm Improvements

#### [Pathfinding Optimization Summary](../technical/PATHFINDING_OPTIMIZATION_SUMMARY.md)
Complete summary of all pathfinding optimizations.

---

## Quick Reference

### Connection Filtering Options

The toolkit provides multiple filtering methods:

```python
# Synapse count threshold
min_synapse_num = 10

# Connection ratio (direct proportion)
min_ratio = 0.01  # weight/post

# Traversal probability (scaled)
min_traversal_probability = 0.001  # min(1, weight/(post*0.3))

# Filtering level
filter_by = 'bodyId'  # or 'type' for aggregation

# Exclude intra-type connections (NEW)
exclude_intra_type_connections = True  # Remove type_pre == type_post
```

**Intra-Type Exclusion Use Cases:**
- Cross-type connectivity analysis
- Cleaner network visualizations
- Inter-type communication studies
- See **[Example_ExcludeIntraType.py](../../archive/examples/basic/Example_ExcludeIntraType.py)**

### Path Finding Commands

```python
# Basic path finding
from findpath import FindAllPath

fap = FindAllPath(
    token='token',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['source_type'],
    targetNeurons=['target_type'],
    max_depth=3,
    min_weight=1,
    forward_only=True,
    use_cache=True
)

paths = fap.find_all_paths()
```

### Cache Setup

```python
# First time setup
fc = FindNeuronConnection(
    token='token',
    dataset='hemibrain:v1.2.1',
    output_dir='/path/to/cache',  # Creates cache here
    use_cache=True
)

# Cache is automatically used for all subsequent queries
```

### Filtering

```python
# Connection strength filters
min_synapse_num = 3
min_connection_ratio = 0.05
min_traversal_probability = 0.01

# Neuron filters
filterBy = {
    'status': 'Traced',
    'min_pre': 10,
    'min_post': 10,
    'rois': ['MB', 'LH']
}
```

---

## Workflow Integration

### Typical Analysis Pipeline

```
1. Setup cache (first time only)
   ↓
2. Find paths with filters:
   - min_weight for weak connections
   - forward_only for biological realism
   - Use cache for speedup
   ↓
3. Filter results:
   - By connection ratio
   - By traversal probability
   ↓
5. Visualize (see visualizations/)
```

### Performance Tips

1. **Always use cache**: repeated queries are served from local storage
   instead of the network API
2. **Use Polars optimization**: Automatic 10-100x faster CSV/matrix operations
3. **Use appropriate filters**: Reduce unnecessary computation
4. **Forward-only mode**: Faster and more biologically realistic
5. **Optimize depth**: Lower max_depth = faster results

---

## Troubleshooting

### Cache Issues
- **Cache not working**: Check `output_dir` path is writable
- **Cache too large**: Normal for complete datasets (~40-60GB for hemibrain)
- **Cache corrupted**: Delete cache directory and rebuild

### Path Finding Issues
- **Too many paths**: Increase `min_weight`, reduce `max_depth`
- **No paths found**: Check source/target names, reduce `min_weight`
- **Slow performance**: Enable cache and check Polars is being used

---

## Best Practices

### Cache Management
1. Use one cache directory per dataset
2. Let cache build naturally through queries
3. Don't manually edit cache files
4. Backup important caches

### Path Finding
1. Start with restrictive filters (high `min_weight`)
2. Gradually relax if needed
3. Use `forward_only=True` for biological accuracy
4. Monitor progress for long searches

### Filtering
1. Apply neuron filters before path finding
2. Use connection filters during path finding
3. Post-process results for final filtering
4. Document filter choices for reproducibility

---

## Related Documentation

- [Visualization Documentation](../visualizations/)
- [Technical Documentation](../technical/)
- [Main README](../../README.md)
- [Quick Start Guide](../QUICK_START.md)

---

## Examples

See `examples/` directory for complete workflows:
- `example_pathfinding.py`: Basic path finding
- `example_cache_setup.py`: Cache configuration
- `example_filtering.py`: Various filter combinations


### [Type-Mapping Validate-Expand-Visualize Guide](./TypeMappingValidateExpandVisualize_Guide.md)
User guide for the type-mapping validate-expand-visualize pipeline: validate an
automatic cross-dataset type mapping at the single-neuron level,
review disagreements in 3D scenes, and collect gap-fill/candidate
proposals (evidence only — the mapping is never rewritten). Categories
form a partition per branch (`matched`/`verified`/`borderline`/`unmatched`,
`sibling`, `candidates`, and — in the wider modes — `family`, `relative`,
`examinees`), and the three modes nest
(`restrictive ⊆ family ⊆ aggressive`).
