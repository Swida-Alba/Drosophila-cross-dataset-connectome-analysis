# Cache System Guide (v4.0)

Complete guide to the local caching system for Neuprint connection data.

> **Note:** This guide covers Cache System v4.0 with unified database architecture and in-memory caching (v4.1.7+).
> For the older v3.0 hash-based system, see [CacheSystem_v3_DatabaseArchitecture.md](./CacheSystem_v3_DatabaseArchitecture.md).

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [How It Works](#how-it-works)
3. [Architecture (v4.0)](#architecture-v40)
4. [In-Memory Caching (v4.1.7+)](#in-memory-caching-v417)
5. [Features](#features)
6. [Cache Building](#cache-building)
7. [Cache Management](#cache-management)
8. [Troubleshooting](#troubleshooting)

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

### Performance Results

| Operation | Time | Speedup |
|-----------|------|---------|
| First load (from disk) | ~126 ms | baseline |
| Second load (in-memory) | ~0.0007 ms | **178,000x** |
| Dict lookup per neuron | ~0.00002 ms | O(1) |

---

## How It Works

### Unified Database Approach (v4.0)

The cache uses a **"fetch once, store centrally, filter on demand"** approach:

1. **Fetch with minimum filter** (`min_weight=1`) from API
2. **Store in unified database** (`connections.parquet`)
3. **Track cached neurons** (`neuron_index.parquet`)
4. **Filter on demand** for any threshold locally

### Example Workflow

```python
# First run: Fetches from Neuprint API, stores in database
fc = FindNeuronConnection(
    sourceNeurons=['L3_R'],
    min_synapse_num=10,  # Filter applied locally
    use_cache=True
)
# Fetches ALL connections (weight ≥ 1), saves to unified database

# Second run: Loads from cache (instant!)
fc = FindNeuronConnection(
    sourceNeurons=['L3_R'],
    min_synapse_num=50,  # Different filter, same cache!
    use_cache=True
)
# Loads from memory cache, filters to weight ≥ 50 locally
```

---

## Architecture (v4.0)

### File Structure

```
cache/
├── hemibrain_v1_2_1/
│   ├── connections.parquet      # Unified connection database (gzip)
│   ├── neuron_index.parquet     # Tracks cached neurons
│   ├── available_rois.json      # Cached ROI list
│   ├── meshes/                  # ROI mesh cache
│   │   ├── AL_L.json
│   │   └── ...
│   ├── neurons/                 # Skeleton cache
│   └── synapses/                # Synapse cache
├── optic-lobe_v1_1/
│   ├── connections.parquet
│   └── neuron_index.parquet
└── flywire_FAFB_v783/
    ├── connections.parquet
    └── neuron_index.parquet
```

### Database Schemas

#### connections.parquet
| Column | Type | Description |
|--------|------|-------------|
| bodyId_pre | str | Upstream neuron ID |
| bodyId_post | str | Downstream neuron ID |
| weight | int64 | Synapse count |
| roi | str | Brain region (optional) |
| type_pre | str | Pre-synaptic neuron type |
| type_post | str | Post-synaptic neuron type |
| cached_date | datetime | When connection was cached |

#### neuron_index.parquet
| Column | Type | Description |
|--------|------|-------------|
| bodyId | str | Neuron bodyId |
| type | str | Neuron type |
| instance | str | Neuron instance |
| post | int | Post-synaptic site count |
| downstream_complete | bool | All downstream connections cached? |
| last_fetched | datetime | Last API fetch timestamp |
| connection_count | int64 | Number of downstream connections |

---

## In-Memory Caching (v4.1.7+)

### 3-Tier Cache Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      3-Tier Cache System                        │
├─────────────────────────────────────────────────────────────────┤
│  Tier 1: Memory Cache (DataFrame + Dict Index)                  │
│  ├── _conn_df_cache: DataFrame in memory                        │
│  ├── _conn_index: Dict[bodyId_pre → row indices]                │
│  └── Lookup: O(1), ~0.0007 ms                                   │
├─────────────────────────────────────────────────────────────────┤
│  Tier 2: Disk Cache (Parquet)                                   │
│  ├── connections.parquet (gzip compressed)                      │
│  └── First load: ~126 ms, cached thereafter                     │
├─────────────────────────────────────────────────────────────────┤
│  Tier 3: API Fetch (fallback)                                   │
│  └── ~900-2000 ms per neuron batch                              │
└─────────────────────────────────────────────────────────────────┘
```

### Module-Level Shared Cache

Multiple `FindNeuronConnection` instances share the same cache:

```python
# Global shared cache (module level)
_FNC_CACHE = {}  # {dataset: {'conn_df': DataFrame, 'conn_index': dict, ...}}

# Instance 1 loads from disk
fc1 = FindNeuronConnection(dataset='hemibrain:v1.2.1', use_cache=True)
# ~126 ms first load

# Instance 2 reuses cached DataFrame
fc2 = FindNeuronConnection(dataset='hemibrain:v1.2.1', use_cache=True)
# ~0.0007 ms (instant from shared cache)
```

### O(1) Dict Lookup

Instead of DataFrame filtering, uses dict for instant lookup:

```python
# Old approach (O(n) DataFrame filter)
cached = conn_df[conn_df['bodyId_pre'].isin(upstream_ids)]  # ~50-100 ms

# New approach (O(1) dict lookup)
row_indices = []
for bodyId in upstream_ids:
    if bodyId in self._conn_index:
        row_indices.extend(self._conn_index[bodyId])  # O(1)
cached = conn_df.iloc[row_indices]  # Direct row access
```

---

## Features

### ✅ What's Included in v4.0

1. **Unified Database**: Single `connections.parquet` per dataset
2. **Smart Reuse**: Cached data works for different filter values
3. **Incremental Updates**: Only fetches uncached neurons
4. **Completeness Tracking**: `downstream_complete` flag for perfect cache hits
5. **In-Memory Caching**: 178,000x speedup for repeated queries
6. **O(1) Lookups**: Dict-based index for instant neuron lookup
7. **Module-Level Sharing**: Multiple instances share cache
8. **40-60% Storage Savings**: Stores only essential columns

---

## Cache Building

### Build Connection Cache

Build cache for an entire dataset programmatically:

```python
from coana import FindNeuronConnection

# Initialize with cache enabled
fnc = FindNeuronConnection(
    dataset='hemibrain:v1.2.1',
    token='your_token',
    use_cache=True
)

# Build connection cache for all neurons
result = fnc.build_connection_cache(batch_size=100)
print(f"Cached {result['total_connections']} connections")
print(f"Cached {result['total_neurons']} neurons")
```

### Build Connectivity Profile Cache

Build profiles for homolog finding and cross-dataset comparison:

```python
# Build connectivity profile cache
result = fnc.build_connectivity_profile_cache(
    top_k=10,           # Store top 10 partners by weight
    top_m=5,            # Ensure at least 5 unique types
    expand_2hop=True    # Enable 2-hop expansion
)
print(f"Built {result['total_profiles']} profiles")
```

### Profile-cache validity (config hash)

Profile rows persist a `config_hash` (an 8-hex digest of the config that
shaped them: `min_synapse_threshold`, `top_k_bodyid`,
`include_untyped_partners`, and the fuzzy-match toggles). A cached profile is
reused only when the reader's config hash matches the row's — changing Min
Synapse Threshold (or the untyped/fuzzy toggles) rebuilds profiles instead of
silently scoring under the old filter. Rows written before 2026-09-25 carry
no hash and stay accepted (legacy caches are not invalidated wholesale).

If the main `connectivity_profiles.parquet` is ever unreadable (torn write,
disk error), consolidation does **not** overwrite or delete it: the file is
renamed aside as `connectivity_profiles.corrupt-<timestamp>.parquet` for
manual recovery, the failure is logged, and the batch files are kept.

### Command-Line Cache Building

```bash
# Build connection cache
python src/build_connection_cache.py hemibrain:v1.2.1 --token YOUR_TOKEN

# Build for specific neuron types
python src/build_connection_cache.py hemibrain:v1.2.1 --types Mi1 T4a aMe12

# Build connectivity profile cache
python src/build_connectivity_profile_cache.py hemibrain:v1.2.1

# Show cache statistics
python src/build_connection_cache.py --stats
python src/build_connectivity_profile_cache.py --stats
```

---

## Cache Management

### View Cache Status

```python
# Check cache status in verbose mode
fc = FindNeuronConnection(
    dataset='hemibrain:v1.2.1',
    use_cache=True,
    verbose_mode='full'
)
```

**Sample Output:**
```
  ✓ Loaded connection database (2.5 MB, 156,892 connections)
  ✓ Loaded neuron index (1,247 neurons, 892 complete)
  📂 Found 892/892 neurons in cache
     Retrieved 62,234 connections from database
```

### Check Cache Files

```bash
# View cache directory structure
ls -lh cache/hemibrain_v1_2_1/

# Check file sizes
du -sh cache/*/

# View neuron index content
python -c "import pandas as pd; print(pd.read_parquet('cache/hemibrain_v1_2_1/neuron_index.parquet').head())"
```

### Clear Cache

```bash
# Clear all cache for a dataset
rm -rf cache/hemibrain_v1_2_1/

# Clear only connections cache (keep ROI meshes, etc.)
rm cache/hemibrain_v1_2_1/connections.parquet
rm cache/hemibrain_v1_2_1/neuron_index.parquet
```

### Cache Sizes

Typical cache sizes per dataset:
| Dataset | Connections | Neuron Index | Total |
|---------|-------------|--------------|-------|
| hemibrain | ~50-100 MB | ~5 MB | ~55-105 MB |
| optic-lobe | ~20-50 MB | ~2 MB | ~22-52 MB |
| flywire_FAFB | ~200-500 MB | ~20 MB | ~220-520 MB |
| male-cns | ~100-200 MB | ~10 MB | ~110-210 MB |

---

## Troubleshooting

### Cache Not Working

**Problem:** Data still fetched from API despite `use_cache=True`

**Solutions:**
```python
# 1. Check if neurons are marked as complete
fc = FindNeuronConnection(dataset='hemibrain:v1.2.1', use_cache=True)
neuron_index = fc._load_neuron_index()
complete = neuron_index['downstream_complete'].astype(bool).sum()
print(f'{complete}/{len(neuron_index)} neurons fully cached')

# 2. Build cache explicitly
result = fc.build_connection_cache()
```

### Slow First Query

**Problem:** First query takes a long time

**Solution:** This is expected - first query loads from disk (~126ms). 
Subsequent queries use memory cache (~0.0007ms).

### Cache Size Too Large

**Problem:** Cache folder > 1 GB

**Solutions:**
```bash
# Check cache sizes
du -sh cache/*/

# Remove old/unused dataset caches
rm -rf cache/old_dataset_name/
```

### Memory Usage High

**Problem:** Too much RAM used by cache

**Solution:** The cache is shared at module level. To clear:
```python
from coana import _FNC_CACHE
_FNC_CACHE.clear()  # Clear all cached DataFrames
```

---

## Best Practices

### 1. Enable Cache for All Analyses

```python
# ✅ Good - Cache enabled
fc = FindNeuronConnection(
    sourceNeurons=['L3_R'],
    use_cache=True
)

# ❌ Bad - Re-fetches every time
fc = FindNeuronConnection(
    sourceNeurons=['L3_R'],
    use_cache=False
)
```

### 2. Pre-Build Cache for Large Analyses

```python
# Build cache once
fnc = FindNeuronConnection(dataset='hemibrain:v1.2.1', use_cache=True)
fnc.build_connection_cache()

# Then run multiple analyses (instant!)
for source_type in ['Mi1', 'L1', 'L2', 'L3', 'L4', 'L5']:
    fc = FindNeuronConnection(
        sourceNeurons=[source_type],
        targetNeurons=['T4a'],
        use_cache=True
    )
    fc.FindAllPath()
```

### 3. Share Cache Across Scripts

The module-level `_FNC_CACHE` is shared within a Python session:

```python
# Script 1 loads cache
fc1 = FindNeuronConnection(dataset='hemibrain:v1.2.1', use_cache=True)
fc1.FindDirectConnections()  # Loads from disk

# Script 2 (same session) reuses cache
fc2 = FindNeuronConnection(dataset='hemibrain:v1.2.1', use_cache=True)
fc2.FindAllPath()  # Instant from memory
```

---

## Summary

**Cache System v4.0** provides:

✅ **Unified database** with single `connections.parquet` per dataset  
✅ **In-memory caching** with 178,000x speedup for repeated queries  
✅ **O(1) dict lookups** for instant neuron retrieval  
✅ **Module-level sharing** across multiple instances  
✅ **Incremental updates** - only fetch uncached neurons  
✅ **40-60% storage savings** with optimized schema  
✅ **Cache building methods** for pre-populating cache  

---

## See Also

- [Cache Enhancements (v4.1.7)](CacheSystem_Guide.md) - In-memory caching details
- [Cache v4 Complete](./CacheSystem_v4_Complete.md) - Full v4.0 implementation
- [Build Scripts](../../src/build_connection_cache.py) - Cache building utilities

---

**Version:** 4.0  
**Last Updated:** December 2025
