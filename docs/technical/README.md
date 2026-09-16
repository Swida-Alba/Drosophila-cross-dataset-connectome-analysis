# Technical Documentation

Advanced technical documentation for developers and power users.

## Overview

This directory contains detailed technical documentation about:
- Backend optimizations and algorithms
- Performance tuning and profiling
- Data structures and storage formats
- Implementation details

---

## Cross-Dataset Validation

### [Type-Mapping Validate-Expand-Visualize Pipeline](./TYPE_MAPPING_VALIDATE_EXPAND_VISUALIZE_PIPELINE.md)
Technical report for the bodyId-level type-mapping validation pipeline
(FAFB ↔ male-cns): branch-resolved pools, global connectivity scan,
tiered verdicts, the **Rev 3.12 category partition** (tier / `sibling` /
`candidates` / `family` / `relative` / `suspicious`, with the per-bodyId
`(out-map)` / `>{src}` / `(no_source)` leaf token), the three nested modes
(`restrictive ⊆ family ⊆ aggressive`), two-track morphology with a
binding native floor and null-calibrated bar, scene rendering, and the
full export contract.

---

## Performance Optimization

### [Deep Backend Optimizations](./DeepBackendOptimizations.md)
Comprehensive guide to backend performance improvements.

**Topics**:
- Query optimization strategies
- Memory management techniques
- Database indexing
- Batch processing

### [Heatmap Optimization Summary](./HeatmapOptimization_Summary.md)
Optimizations specific to heatmap generation and rendering.

**Key improvements**:
- Data processing pipeline
- Clustering algorithm efficiency
- Rendering performance
- Memory usage reduction

### [Pathfinding Pipeline](./PATHFINDING_PIPELINE.md)
**Start here** — comprehensive current-state report of the pathfinding
strategies: discovery, the lossless fixpoint pruning stack, the
budget-fit Edge Budget, StrongestFirst in both modes (simple paths and
min-hop paths), guarantees (τ-equivalence, cap honesty), complexity,
measured real-data performance, and design decisions.

### [StrongestFirst Pathfinding](./STRONGEST_FIRST_PATHFINDING.md)
Design record for the StrongestFirst family: the widest-path DP,
best-first enumeration, budget + τ drain semantics, W\* vs τ, and the
Fix C / Fix D history.

### [Pathfinding Optimization Summary](./PATHFINDING_OPTIMIZATION_SUMMARY.md)
Complete overview of path finding algorithm optimizations.

**Covered optimizations**:
- Depth-first search improvements
- Early termination strategies
- Memory-efficient recursion
- Progress tracking overhead reduction

### [Pathfinding Algorithm Evaluation](./PATHFINDING_ALGORITHM_EVALUATION.md)
Theoretical (complexity) and practical (measured time + memory) evaluation
of the complete-enumerator `FastGraph` pathfinding algorithms, with measured results on a
real hemibrain v1.2.1 connectome query and synthetic scaling graphs.

**Key findings**:
- All algorithms verified correct on 150 random graphs (750 runs, zero mismatches)
- Algorithm names now match the implementations (MemoizedDFS runs memoized
  DFS forward; MeetInMiddle runs meet-in-the-middle)
- Lazy reverse-adjacency index eliminated the ~250 MB reversed-graph
  copies: all algorithms except Bidirectional run in <10 MB at 2-4 layers
- MemoizedDFS is the fastest *complete* enumerator (StrongestFirst is the
  pipeline default); MeetInMiddle fastest at 2-3 layers,
  MemoizedDFS at 4, Bidirectional at 5 (but 18.9 -> 892 MB memory)
- Bitmask visited sets and degree-2 compression were measured and rejected
  (not faster on real connectomes)

### [Type Aggregation and bodyId Discovery](./TYPE_AGGREGATION_AND_BODYID_DISCOVERY.md)
Why FindAllPath finds paths at the bodyId level first and only then
aggregates the in-path pairs into type-level summaries — the *bundle
effect* (aggregating all per-bodyId pairs hides differences within a type
and reports phantom type paths), with the minimal synthetic failure mode,
real-data examples and comparative measurements across datasets/depths.

### [Performance Optimizations Dec 2025](./PERFORMANCE_OPTIMIZATIONS_DEC2025.md)
Details on Polars integration and `skip_bodyId` optimization.

**Key improvements**:
- Polars-based I/O (10-100x faster)
- Skip bodyId processing for type-level analysis
- Granular progress tracking

---

## Data Formats

### [Column Recognition Update](./COLUMN_RECOGNITION_UPDATE.md)
Automatic column name recognition in input data.

**Features**:
- Flexible column naming
- Auto-detection of metrics
- Backward compatibility
- Error handling

**Recognized column patterns**:
- Source: `source`, `sourcetype`, `source_type`
- Target: `target`, `targettype`, `target_type`
- Weight: `weight`, `synapse_count`, `synapses`
- Ratio: `ratio`, `connection_ratio`
- Probability: `probability`, `traversal_probability`

### [Negative Values Implementation](./NEGATIVE_VALUES_IMPLEMENTATION.md)
Support for negative connection values (inhibitory connections).

**Technical details**:
- Data validation
- Visualization handling
- Colorscale selection
- Edge case handling

**Quick Reference**: [NEGATIVE_VALUES_QUICKREF.md](./NEGATIVE_VALUES_QUICKREF.md)

### [Source Path Attribute](./SOURCE_PATH_ATTRIBUTE.md)
Documentation of source_path attribute for tracking data provenance.

### [Vector v2 Similarity — Calculation Pipeline](./VECTOR_V2_PIPELINE.md)
Stage-by-stage specification of the production `vector_v2` morphology score.

**Covers**:
- 256-dim feature layout (shape + spatial blocks, exact column slices)
- Lateral normalization and mirror invariance
- Persisted μ/σ standardization and truncated ZCA whitening
- Block-weighted cosine scoring (3:7), optional mass-overlap / ROI blocks
- Cache lifecycle: when the population stats change (and when they don't)
- Benchmark evidence and code reference map

---

## Performance Analysis

### [Dialog Performance Guide](./DIALOG_PERFORMANCE_GUIDE.md)
Profiling and optimizing interactive dialogs.

**Topics**:
- Sheet selection optimization
- File picker performance
- Memory usage profiling
- Lazy loading strategies

### [Dependency Summary](./DEPENDENCY_SUMMARY.md)
Complete list of package dependencies with version requirements and rationale.

**Key dependencies**:
- `neuprint-python`: NeuPrint API access
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `plotly`: Interactive visualizations
- `networkx`: Graph algorithms
- `scipy`: Scientific computing (clustering, etc.)

---

## Implementation Details

### Data Structures

**Connection Matrix Storage**:
- Sparse matrix format for memory efficiency
- Dense matrix for small datasets
- Automatic format selection based on sparsity

**Graph Representation**:
- NetworkX DiGraph for path finding
- Adjacency list for fast neighbor lookup
- Edge attributes stored in dictionary

**Cache Database Schema**:
- SQLite for metadata
- Parquet for bulk data
- Indexed queries for fast retrieval

### Algorithm Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Direct connection query | O(1) | O(n) |
| Path finding (DFS) | O(b^d) | O(d) |
| Hierarchical clustering | O(n² log n) | O(n²) |
| Force-directed layout | O(n²) | O(n) |
| Heatmap rendering | O(n²) | O(n²) |

Where:
- n = number of neurons
- b = average branching factor
- d = maximum depth

---

## Backend Architecture

### Module Organization

```
src/
├── findpath.py         # Path finding algorithms
├── statvis.py          # Statistical visualization (heatmap)
├── vispath.py          # Network and Sankey visualization
├── navis_related.py    # 3D skeleton rendering
├── cache_utils.py      # Cache management
└── utils/
    ├── graph_utils.py  # Graph algorithms
    ├── data_utils.py   # Data processing
    └── vis_utils.py    # Visualization helpers
```

### Data Flow

```
NeuPrint API → Cache Layer → Processing → Visualization
     ↓              ↓            ↓              ↓
  Raw data    Local SQLite   Analysis      HTML/PNG/SVG
                + Parquet     Filtering
```

### Caching Strategy

**Three-tier cache**:
1. **Memory cache**: Most recently used queries (LRU)
2. **Disk cache**: All previous queries (SQLite + Parquet)
3. **NeuPrint**: Original data source

**Cache invalidation**:
- Version-based (dataset version changes)
- Manual clear option
- Automatic corruption detection

---

## Optimization Techniques

### Query Optimization

**Batch Queries**:
```python
# Instead of multiple single queries
for neuron in neurons:
    query(neuron)  # N queries

# Use batch query
query_batch(neurons)  # 1 query
```

**Index Usage**:
- Indexed by bodyId for O(1) lookup
- Indexed by type for fast type-level queries
- Composite indexes for complex queries

### Memory Optimization

**Lazy Loading**:
- Load data only when needed
- Stream large datasets
- Release memory after processing

**Data Type Optimization**:
- Use appropriate dtypes (int32 vs int64)
- Categorical data for repeated strings
- Sparse matrices for sparse data

### Internal Parallel Processing

Some internal operations use ThreadPoolExecutor for parallel processing:

**Profile Building** (HomologFinder):
- ThreadPoolExecutor for building connectivity profiles
- Automatic worker count: `min(32, cpu_count + 4)`
- Deferred cache writes for reduced I/O contention

**Memory Sharing**:
- Read-only data shared across workers
- Write data collected after completion
- Batch processing for memory safety

---

## Profiling Tools

### Built-in Profiling

```python
# Enable profiling
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
fap.find_all_paths()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Your code
    pass
```

### Visualization Performance

Browser DevTools:
- Performance tab for rendering profiling
- Memory tab for memory leaks
- Network tab for data loading

---

## Testing

### Unit Tests

Located in `tests/` directory:
- `test_pathfinding.py`: Path finding algorithms
- `test_cache.py`: Cache operations
- `test_visualization.py`: Visualization generation
- `test_data_formats.py`: Input/output formats

### Performance Tests

Benchmark scripts:
- `bench_cache.py`: Cache performance
- `bench_pathfinding.py`: Path finding speed
- `bench_visualization.py`: Rendering performance

### Integration Tests

End-to-end workflows:
- Complete analysis pipeline
- Multiple visualization types
- Large dataset handling

---

## Debugging

### Common Issues

**Memory Leaks**:
- Check for circular references
- Release large objects explicitly
- Use weak references where appropriate

**Slow Performance**:
- Profile to find bottlenecks
- Check cache is enabled
- Verify Polars optimization is being used

**Data Corruption**:
- Validate input data formats
- Check for NaN/Inf values
- Verify data types

### Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hemibrain')

# In code
logger.debug('Processing neuron %s', neuron_id)
logger.info('Found %d paths', len(paths))
logger.warning('Slow query detected: %.2fs', elapsed)
logger.error('Failed to process: %s', error)
```

---

## Contributing

### Performance Improvements

When contributing optimizations:
1. Profile before and after
2. Measure actual speedup
3. Consider memory trade-offs
4. Document algorithm complexity
5. Add benchmark tests

### Code Quality

Standards:
- Type hints for function signatures
- Docstrings in NumPy format
- Unit tests for new features
- Performance tests for optimizations

---

## Advanced Topics

### Custom Algorithms

Implementing custom layout algorithms:
```python
def custom_layout(G, **kwargs):
    """
    Custom network layout algorithm.
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph
    **kwargs : dict
        Algorithm parameters
        
    Returns
    -------
    pos : dict
        Node positions {node: (x, y)}
    """
    pos = {}
    # Your algorithm here
    return pos
```

### Custom Visualizations

Creating new visualization types:
1. Extend base visualization class
2. Implement data processing
3. Generate Plotly figure
4. Add export options
5. Document usage

### Database Schema Extensions

Adding new cached data types:
1. Define table schema
2. Implement query methods
3. Add indexing strategy
4. Update cache version
5. Handle migrations

---

## Performance Benchmarks

### System Configuration
- CPU: Intel i7-9700K (8 cores)
- RAM: 32 GB DDR4
- Storage: NVMe SSD
- OS: macOS 12.6

### Benchmark Results

**Path Finding** (source→target, depth=3):
- 10 neurons: 0.5s
- 100 neurons: 8s
- 500 neurons: 95s
- 1000 neurons: 380s (with cache)

**Cache Performance** (representative single-query timings, not a
maintained benchmark):
- First query: 2-5s
- Cached query: 0.05-0.2s
- Speedup: 10-100x for the measured workload

**Polars Optimization**:
- CSV writing: 10-100x faster
- Matrix generation: 5-50x faster
- DataFrame operations: 2-10x faster

**Visualization Generation**:
- Heatmap (100×100): 0.8s
- Network (200 nodes): 1.2s
- Sankey (150 flows): 0.9s
- 3D Skeleton (10 neurons): 3.5s

---

## Future Optimizations

### Planned Improvements
- GPU acceleration for clustering
- Incremental cache updates
- Streaming visualization for large datasets
- WebGL rendering for networks

### Research Directions
- Machine learning for path prediction
- Approximate algorithms for large graphs
- Real-time collaborative visualization
- Cloud-based computation options

---

## Related Documentation

- [Core Features](../core-features/)
- [Visualization Documentation](../visualizations/)
- [Main README](../../README.md)
- Contributing Guide

---

## References

### Academic Papers
- Scheffer et al. (2020). "A connectome and analysis of the adult Drosophila central brain." eLife.
- Sugiyama et al. (1981). "Methods for visual understanding of hierarchical system structures."

### Libraries
- NeuPrint: https://neuprint.janelia.org/
- Plotly: https://plotly.com/python/
- NetworkX: https://networkx.org/
- NAVIS: https://navis.readthedocs.io/

### Tools
- Chrome DevTools: https://developer.chrome.com/docs/devtools/
- cProfile: https://docs.python.org/3/library/profile.html
- memory_profiler: https://pypi.org/project/memory-profiler/
