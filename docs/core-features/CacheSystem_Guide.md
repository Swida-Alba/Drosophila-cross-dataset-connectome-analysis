# Cache System Guide

> **⚠️ This guide is outdated.** The current cache system is v4.0.
> 
> **Please see [CacheSystem_Guide_v4.md](./CacheSystem_Guide_v4.md) for the current documentation.**
>
> Key improvements in v4.0:
> - Unified database architecture (`connections.parquet` + `neuron_index.parquet`)
> - In-memory caching with 178,000x speedup
> - O(1) dict-based lookups
> - Module-level cache sharing
> - Built-in cache building methods (`build_connection_cache()`, `build_connectivity_profile_cache()`)

---

# Cache System Guide (v3.0) - DEPRECATED

Complete guide to the local caching system for Neuprint connection data.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [How It Works](#how-it-works)
3. [Features](#features)
4. [Database Architecture](#database-architecture)
5. [Storage Optimization](#storage-optimization)
6. [Cache Management](#cache-management)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Enable Caching

```python
fc = FindNeuronConnection(
    token='your_token',
    dataset='optic-lobe:v1.1',
    sourceNeurons=['L3_R'],
    targetNeurons=['l-LNv_R'],
    min_synapse_num=10,
    use_cache=True  # Enable caching
)

fc.InitializeNeuronInfo()
fc.FindDirectConnections()  # Or FindAllPath()
```

### First-Time Setup

On first use with caching enabled, the system automatically downloads a complete neuron dataset:

```
📥 Complete dataset not found, downloading ALL neurons (including type=None)...
   This is a one-time download for cache enrichment.
Pulled 53847 neurons from optic-lobe:v1.1
✅ Complete dataset saved
```

This one-time download (15-30 seconds) ensures the cache can enrich all connections, even for neurons without assigned types.

---

## How It Works

### Fetching Strategy

The cache system uses a **"fetch once, filter many times"** approach:

1. **Fetch with minimum filter** (`min_weight=1`)
2. **Cache everything** locally
3. **Filter on demand** for any `min_synapse_num` threshold

### Example Workflow

```python
# First run: Fetches from Neuprint API with min_weight=1
fc = FindNeuronConnection(
    sourceNeurons=['L3_R'],
    min_synapse_num=10,  # Filter applied locally
    use_cache=True
)
# Fetches ALL connections (weight ≥ 1), saves to cache

# Second run: Loads from cache (MUCH faster!)
fc = FindNeuronConnection(
    sourceNeurons=['L3_R'],
    min_synapse_num=50,  # Different filter, same cache!
    use_cache=True
)
# Loads cached data, filters to weight ≥ 50 locally
```

### Performance Benefits

- **Dramatically faster repeated queries** — a cached query is a
  local read (tenths of a second) instead of a network API call (seconds)
- **Offline analysis** once data is cached
- **Flexible filtering** without re-fetching
- **No redundant API calls**

---

## Features

### ✅ What's Included in v3.0

1. **Automatic Caching**: Connection data saved automatically
2. **Smart Reuse**: Cached data works for different `min_synapse_num` values
3. **Neuron Registry**: Search cached neurons without API calls
4. **Database Architecture**: Efficient parquet-based storage
5. **Complete Dataset**: Handles ALL neurons (including `type=None`)
6. **Hash-Based Keys**: Stable cache keys independent of neuron count
7. **40-60% Storage Savings**: Stores only bodyId pairs, joins metadata from local files

### Neuron Search

Search cached neurons by type, instance, or bodyId:

```python
# Search by neuron type
l3_neurons = fc.search_cached_neurons('L3.*', 'type')

# Search by instance (e.g., right hemisphere)
right_neurons = fc.search_cached_neurons('.*_R$', 'instance')

# Search by exact bodyId
neuron = fc.search_cached_neurons(5813022222, 'bodyId')
```

**Output:**
```
   bodyId           type        instance
  5813022222        L3_R        L3_R
  5813022333        L3_R        L3_R
```

---

## Database Architecture

### File Structure

```
neuprint_cache/
  optic-lobe_v1.1/
    connections/
      upstream_a1b2c3.parquet       # Connection data
      query_d4e5f6_g7h8i9.parquet
    neuron_registry.parquet          # Searchable neuron database
    cache_index.parquet               # Query→file mapping
```

### 1. Neuron Registry (`neuron_registry.parquet`)

Stores all unique neurons ever cached for the dataset.

**Schema:**
```python
{
    'bodyId': int64,
    'type': str,
    'instance': str
}
```

**Purpose:**
- Enables local search without API calls
- Provides neuron metadata for enrichment
- Updated automatically when caching new connections

### 2. Cache Index (`cache_index.parquet`)

Maps queries to cache files with metadata.

**Schema:**
```python
{
    'cache_key': str,              # e.g., 'upstream_abc123'
    'upstream_bodyIds': str,       # JSON array
    'downstream_bodyIds': str,     # JSON array or null
    'created': str,                # ISO timestamp
    'connection_count': int64
}
```

**Purpose:**
- Fast lookup of existing cached queries
- Tracks which neurons are in each cache file
- Enables cache statistics

### 3. Connection Data (`connections/*.parquet`)

Actual connection data with stable hash-based filenames.

**Schema:** (Minimal - stores only essential data)
```python
{
    'bodyId_pre': int64,
    'bodyId_post': int64,
    'weight': int64,
    'roi': str  # Optional
}
```

**Pair-collapsed layout (2026-10-12):** consolidations emit the main
`connections.parquet` with **one row per (pre, post) pair** at the
pair's PHYSICAL weight (the sum of its per-ROI slices — slice rows are
how per-ROI caches record one connection). The per-ROI detail moves
verbatim to `connections_by_roi.parquet` beside it (the audit surface;
its presence marks the layout). Every consumer — D_t ratio
denominators, threshold filters, aggregation — reads the pair's
physical weight directly; legacy per-ROI caches remain readable (all
readers collapse per pair before thresholding). Re-consolidating an
existing cache (Settings → pull to completion) converts it.

**Note:** Type and instance are NOT stored in cache - joined from local datasets when loading!

---

## Storage Optimization

### Why Cache is 40-60% Smaller

**Traditional Approach (Full Storage):**
```python
# Store everything in cache
{
    'bodyId_pre': 5813022222,
    'type_pre': 'L3_R',           # Redundant!
    'instance_pre': 'L3_R',       # Redundant!
    'bodyId_post': 722817260,
    'type_post': 'l-LNv_R',       # Redundant!
    'instance_post': 'l-LNv_R',   # Redundant!
    'weight': 42,
    'roi': 'LO(R)'
}
```

**Optimized Approach (v3.0):**
```python
# Cache: Store only essentials
{
    'bodyId_pre': 5813022222,
    'bodyId_post': 722817260,
    'weight': 42,
    'roi': 'LO(R)'
}

# When loading: Join type/instance from local CSV
neuron_df = pd.read_csv('datasets/optic-lobe_v1_1_allneurons_neuron_df.csv')
connections = connections.merge(neuron_df[['bodyId', 'type', 'instance']], ...)
```

### Size Comparison

**Example: L3_R → l-LNv_R (15,234 connections)**

| Approach | Cache Size | Savings |
|----------|-----------|---------|
| Full storage (old) | 2.1 MB | - |
| Optimized (v3.0) | 0.9 MB | 57% smaller |

**Why it works:**
- Type/instance are the same for each bodyId
- Storing them in every connection row is redundant
- Local datasets provide this mapping
- No API calls needed for enrichment

### Complete Dataset Handling

The cache system ensures a complete local neuron dataset exists:

```
datasets/
  optic-lobe_v1_1_allneurons_neuron_df.csv    ← For cache (ALL neurons)
  optic-lobe_v1_1_alltypes_neuron_df.csv      ← For queries (typed only)
```

**Key Difference:**
- `alltypes`: Only neurons with assigned types (standard queries)
- `allneurons`: **ALL neurons including `type=None`** (for cache enrichment)

**Why both are needed:**
- User queries typically want typed neurons
- Cache must handle connections involving ANY neuron (including untyped)
- First-time cache use automatically downloads the complete dataset

---

## Cache Management

### View Cache Information

```python
fc.print_cache_info()
```

**Sample Output:**
```
======================================================================
CACHE INFORMATION: optic-lobe:v1.1
======================================================================
Cache location: neuprint_cache/optic-lobe_v1.1
Total cached queries: 5
Total neurons in registry: 1247

Total cache files: 5
Total cache size: 12.34 MB
Total cached connections: 156,892

Cache Key                           Upstream   Downstream   Connections  Created
----------------------------------- ---------- ------------ ------------ -------------------
upstream_a1b2c3d4e5f6               892        All          15,234       2024-01-15 10:30:45
query_d4e5f6_g7h8i9                 1          1            42           2024-01-15 11:22:10

NEURON REGISTRY SAMPLE:
   bodyId           type        instance
  5813022222        L3_R        L3_R
  722817260      l-LNv_R      l-LNv_R
... and 1245 more neurons
======================================================================
```

### Clear Cache

```python
# Clear all cache for current dataset
fc.clear_cache()

# Prompt:
# ⚠️  Clear all cache for optic-lobe:v1.1? (yes/no): yes

# Output:
# ✅ Cache cleared for optic-lobe:v1.1
```

### Interactive Management

```bash
python ManageCache.py
```

**Features:**
1. View cache information for all datasets
2. View cache for specific dataset
3. Search cached neurons
4. Clear cache for specific dataset
5. Clear all cache

---

## Troubleshooting

### Cache Not Working

**Problem:** Data still fetched from Neuprint despite `use_cache=True`

**Solutions:**
```python
# 1. Check cache status
fc.print_cache_info()

# 2. Verify cache is enabled
print(fc.use_cache)  # Should be True

# 3. Check if query is actually cached
# The cache may not have this specific upstream/downstream combination
```

### Search Returns Empty

**Problem:** `search_cached_neurons()` returns empty DataFrame

**Solutions:**
```python
# 1. Check registry exists
registry = fc._load_neuron_registry()
print(f'Registry has {len(registry)} neurons')

# 2. Try broader pattern
all_neurons = fc.search_cached_neurons('.*', 'type')
print(f'Found {len(all_neurons)} total neurons')

# 3. Check for exact matches
fc.search_cached_neurons('L3_R', 'type')  # Exact type name
```

### Complete Dataset Missing

**Problem:** Error about missing complete dataset

**Solution:**
```python
# The system should auto-download, but if it fails:
# 1. Check internet connection
# 2. Verify token is valid
# 3. Manually trigger download:
fc.use_cache = True
fc.InitializeNeuronInfo()  # Will download complete dataset
```

### Cache Size Too Large

**Problem:** Cache folder > 1 GB

**Solutions:**
```python
# 1. View cache statistics
fc.print_cache_info()

# 2. Clear cache for specific dataset
fc.clear_cache()

# 3. Or keep cache but filter more aggressively
fc = FindNeuronConnection(
    sourceNeurons=['L3_R'],
    min_synapse_num=100,  # Higher threshold
    use_cache=True
)
```

---

## Best Practices

### 1. Always Enable Cache for Repeated Analysis

```python
# ✅ Good - Cache enabled
fc = FindNeuronConnection(
    sourceNeurons=['L3_R'],
    use_cache=True
)

# ❌ Bad - Wastes time re-fetching
fc = FindNeuronConnection(
    sourceNeurons=['L3_R'],
    use_cache=False  # Every run hits API!
)
```

### 2. Use Search Before Fetching

```python
# Check if neurons are cached
cached = fc.search_cached_neurons('L3.*', 'type')

if not cached.empty:
    print(f'✅ Found {len(cached)} L3 neurons in cache')
else:
    print('⚠️ L3 neurons not cached, will fetch from API')
```

### 3. Fetch Once with Low Threshold

```python
# ✅ Good - Fetch once with min_weight=1 (automatic)
fc = FindNeuronConnection(
    sourceNeurons=['L3_R'],
    min_synapse_num=1,  # Cache handles this
    use_cache=True
)

# Then filter for different analyses:
strong = fc.filter_connections(min_weight=50)
medium = fc.filter_connections(min_weight=20)
weak = fc.filter_connections(min_weight=5)
```

### 4. Monitor Cache Size Periodically

```python
# Check cache size
fc.print_cache_info()

# If cache > 1 GB, consider clearing old data
if cache_size_mb > 1000:
    fc.clear_cache()
```

### 5. Clear Cache When Dataset Updates

```python
# When Neuprint releases a new dataset version (e.g. male-cns v0.9 -> v1.0):
# 1. Clear old cache
fc = FindNeuronConnection(dataset='male-cns:v0.9', use_cache=True)
fc.clear_cache()

# 2. Re-fetch with the new version
fc = FindNeuronConnection(dataset='male-cns:v1.0', use_cache=True)
```

---

## Summary

**Cache System v3.0** provides:

✅ **Automatic caching** with fetch-once, filter-many approach  
✅ **40-60% storage savings** by storing only bodyId pairs  
✅ **Neuron registry** for local search without API calls  
✅ **Complete dataset** handling (including `type=None` neurons)  
✅ **Stable cache keys** using SHA256 hashing  
✅ **Database architecture** with parquet format for speed  

This enables fast, offline analysis with minimal storage requirements.

---

**See Also:**
- Main README for quick start
- [Module Calling Tree](./Module_Calling_Tree.md) for architecture overview
- Configuration_Guide.md for filter parameters
