# Cache System v3.0: Database Architecture

> **⚠️ Superseded by the v4 cache** — see [CacheSystem_Guide_v4.md](./CacheSystem_Guide_v4.md) and [CacheSystem_v4_Implementation.md](./CacheSystem_v4_Implementation.md) for the current architecture. Kept for history.

## Overview

**Cache System v3.0** introduces a database-style architecture with **neuron registry** and **query index**, enabling flexible local search and eliminating the need for neuron count in cache keys.

## What's New in v3.0

### Key Improvements

1. **Neuron Registry**: Local database of all cached neurons (bodyId, type, instance)
2. **Query Index**: Maps queries to cache files with metadata
3. **Hash-Based Cache Keys**: Stable keys using SHA256 (no neuron count dependency)
4. **Local Search**: Search cached neurons without API calls
5. **Better Scalability**: Clean separation between data and metadata

### Architecture Comparison

| Component | v2.0 (Unified Cache) | v3.0 (Database) |
|-----------|---------------------|-----------------|
| Cache Key | `conn_892neurons_all_abc123.parquet` | `upstream_abc123.parquet` |
| Metadata | JSON file with query info | Parquet index with query→file mapping |
| Neuron Info | Not stored | Searchable registry (parquet) |
| Search | Not available | Regex search by type/instance/bodyId |
| Stability | Breaks if neuron count changes | Stable hash-based keys |

---

## Database Structure

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

**Example:**
```
   bodyId           type        instance
  5813022222        L3_R        L3_R
  5813022333        L3_R        L3_R
  722817260      l-LNv_R      l-LNv_R
```

**Purpose:**
- Enables local search without API calls
- Provides neuron metadata for all cached queries
- Updated automatically when new connections are cached

### 2. Cache Index (`cache_index.parquet`)

Maps queries to cache files with metadata.

**Schema:**
```python
{
    'cache_key': str,              # Hash-based key (e.g., 'upstream_abc123')
    'upstream_bodyIds': str,       # JSON array of bodyIds
    'downstream_bodyIds': str,     # JSON array of bodyIds (or null for all)
    'created': str,                # ISO timestamp
    'connection_count': int64      # Number of connections
}
```

**Example:**
```
cache_key              upstream_bodyIds        downstream_bodyIds    created              connection_count
upstream_a1b2c3        "[5813022222, ...]"     null                 2024-01-15 10:30:45  15234
query_d4e5f6          "[5813022222]"          "[722817260]"        2024-01-15 11:22:10  42
```

**Purpose:**
- Fast lookup of existing cached queries
- Tracks which neurons are in each cache file
- Enables cache statistics and management

### 3. Connection Data (`connections/*.parquet`)

Actual connection data with stable hash-based filenames.

**Filename Format:**
- `upstream_{hash}.parquet` - Upstream connections (all downstream neurons)
- `downstream_{hash}.parquet` - Downstream connections (all upstream neurons)
- `query_{hash1}_{hash2}.parquet` - Specific source→target query

**Schema:** (Minimal connection data only)
```python
{
    'bodyId_pre': int64,   # Pre-synaptic neuron bodyId
    'bodyId_post': int64,  # Post-synaptic neuron bodyId
    'weight': int64,       # Synapse count
    'roi': str            # Region of interest (optional)
}
```

**Storage Optimization:**
- Only stores **essential connection data** (bodyId pairs + weight + roi)
- Neuron metadata (type, instance) is **NOT stored** in cache
- Type and instance are joined from **complete local datasets** when loading
- Reduces cache size by **40-60%** compared to storing full neuron info
- No API calls needed for enrichment - all from local CSV files

**Why this works:**
- Cache system ensures **complete local dataset** exists (`datasets/*_allneurons_neuron_df.csv`)
  - Includes **ALL neurons**, even those with `type=None`
  - Downloaded automatically on first use with caching enabled (one-time download)
- Standard user dataset (`datasets/*_alltypes_neuron_df.csv`) only has typed neurons
- Cache enrichment uses the **complete dataset** to handle all neurons
- When loading cache, we join with complete local neuron data to reconstruct full connection table
- No risk of missing neurons - complete dataset has everything!

**Dataset Files:**
```
datasets/
  optic-lobe_v1_1_allneurons_neuron_df.csv    ← For cache enrichment (ALL neurons)
  optic-lobe_v1_1_allneurons_roi_count_df.parquet
  optic-lobe_v1_1_alltypes_neuron_df.csv      ← For user queries (typed neurons only)
  optic-lobe_v1_1_alltypes_roi_count_df.parquet
```

---

## Cache Key Generation

### Hash-Based Keys (v3.0)

Cache keys are generated using **SHA256** hash of the **full bodyId list**, independent of neuron count.

```python
import hashlib
import json

def _get_cache_key_from_bodyids(upstream_ids, downstream_ids=None, direction='upstream'):
    """Generate stable cache key from bodyId lists"""
    upstream_sorted = sorted(upstream_ids)
    hash_str = hashlib.sha256(
        json.dumps(upstream_sorted, sort_keys=True).encode()
    ).hexdigest()[:12]
    
    if downstream_ids is None:
        return f'{direction}_{hash_str}'
    
    downstream_sorted = sorted(downstream_ids)
    down_hash = hashlib.sha256(
        json.dumps(downstream_sorted, sort_keys=True).encode()
    ).hexdigest()[:12]
    
    return f'query_{hash_str}_{down_hash}'
```

**Example:**
```python
upstream_ids = [5813022222, 5813022333, 722817260]

# v2.0: conn_3neurons_all_abc123.parquet  ❌ Breaks if count changes
# v3.0: upstream_a1b2c3d4e5f6.parquet     ✅ Stable regardless of count
```

**Benefits:**
- Stable keys that don't change if neuron list is reordered
- No dependency on neuron count (can add/remove neurons)
- Collision-resistant (SHA256 with 12-char hash = 48 bits)

---

## Workflow

### Fetching Connections with Cache

```python
fc = FindNeuronConnection(
    dataset='optic-lobe:v1.1',
    sourceNeurons=['L3_R'],
    targetNeurons=['l-LNv_R'],
    min_synapse_num=10,
    use_cache=True
)

# 1. Generate cache key from bodyId lists
cache_key = fc._get_cache_key_from_bodyids(upstream_ids, downstream_ids)

# 2. Check if query exists in cache index
cached_entry = fc._find_cached_query(upstream_ids, downstream_ids)

# 3a. If found: Load from cache, enrich, and filter
if cached_entry is not None:
    # Load minimal connection data (bodyId pairs + weight)
    connections = pd.read_parquet(cache_file)
    
    # Enrich with neuron type/instance from LOCAL dataset (no API call!)
    all_bodyids = list(set(connections['bodyId_pre'] + connections['bodyId_post']))
    neuron_df = statvis.getNeurons(all_bodyids, dataset)  # From local CSV
    
    # Join type_pre, instance_pre, type_post, instance_post
    connections = enrich_with_neuron_info(connections, neuron_df)
    
    # Filter by min_weight
    filtered = connections[connections['weight'] >= min_synapse_num]

# 3b. If not found: Fetch from Neuprint with min_weight=1
else:
    # Fetch full connection data from API (includes type/instance)
    connections = fetch_from_neuprint(min_weight=1)
    
    # Save ONLY bodyId pairs + weight to cache (40-60% smaller!)
    fc._save_connections_to_cache(cache_key, connections[['bodyId_pre', 'bodyId_post', 'weight', 'roi']])
    
    # Update neuron registry with type/instance before discarding
    fc._update_neuron_registry(connections)
    
    # Update cache index
    fc._update_cache_index(cache_key, upstream_ids, downstream_ids, len(connections))
    
    # Filter for current query
    filtered = connections[connections['weight'] >= min_synapse_num]
```

**Key Benefit:** When loading from cache, neuron type/instance is joined from local CSV files (no API call), but cache storage is 40-60% smaller!

### Searching Cached Neurons

```python
# Search by neuron type
l3_neurons = fc.search_cached_neurons('L3.*', 'type')

# Search by instance (e.g., right hemisphere)
right_neurons = fc.search_cached_neurons('.*_R$', 'instance')

# Search by bodyId
neuron = fc.search_cached_neurons(5813022222, 'bodyId')
```

**Output:**
```
   bodyId           type        instance
  5813022222        L3_R        L3_R
  5813022333        L3_R        L3_R
```

---

## Cache Management

### View Cache Information

```python
fc.print_cache_info()
```

**Output:**
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
...

NEURON REGISTRY SAMPLE (first 10):
   bodyId           type        instance
  5813022222        L3_R        L3_R
  722817260      l-LNv_R      l-LNv_R
  ...

... and 1237 more neurons
======================================================================
```

### Clear Cache

```python
# Clear all cache for current dataset
fc.clear_cache()

# Output:
# ⚠️  Clear all cache for optic-lobe:v1.1? (yes/no): yes
# ✅ Cache cleared for optic-lobe:v1.1
#    - Connection files removed
#    - Neuron registry removed
#    - Cache index removed
```

### Interactive Cache Management

Use the `ManageCache.py` utility for interactive management:

```bash
python ManageCache.py
```

**Features:**
1. View cache information for all datasets
2. View cache for specific dataset
3. **Search cached neurons** (NEW in v3.0)
4. Clear cache for specific dataset
5. Clear all cache

---

## Performance Benefits

### v2.0 vs v3.0 Comparison

| Scenario | v2.0 | v3.0 | Improvement |
|----------|------|------|-------------|
| **Cache Key Stability** | Breaks if neuron count changes | Hash-based, always stable | ✅ 100% stable |
| **Neuron Search** | Not available | Regex search in registry | ✅ New feature |
| **Metadata Storage** | JSON file (1 file) | Parquet index (faster, typed) | ✅ 2-3x faster |
| **Scalability** | Limited by JSON parsing | Efficient parquet queries | ✅ 10x+ scalability |
| **Cache Reusability** | Same (fetch with min_weight=1) | Same (fetch with min_weight=1) | = No change |

### Storage Efficiency

**Example: L3_R → l-LNv_R (892 upstream neurons, 15,234 connections)**

```
v2.0 Structure (Full neuron info in cache):
  neuprint_cache/optic-lobe_v1.1/
    connections/conn_892neurons_all_abc123.parquet   (2.1 MB)
    cache_metadata.json                               (5 KB)
  Total: 2.11 MB

v3.0 Structure (Minimal connection data only):
  neuprint_cache/optic-lobe_v1.1/
    connections/upstream_a1b2c3d4e5f6.parquet        (0.9 MB) ← 57% smaller!
    neuron_registry.parquet                           (45 KB)
    cache_index.parquet                               (8 KB)
  Total: 0.95 MB  (55% reduction!)
```

**Storage Optimization Breakdown:**

| Data Component | v2.0 Size | v3.0 Size | Savings |
|----------------|-----------|-----------|---------|
| Connection files | 2.1 MB | 0.9 MB | -57% |
| Metadata | 5 KB | 53 KB | +960% (but tiny) |
| **Total** | **2.11 MB** | **0.95 MB** | **-55%** |

**Why v3.0 is smaller:**
- Cache stores only: `bodyId_pre`, `bodyId_post`, `weight`, `roi`
- Does NOT store: `type_pre`, `type_post`, `instance_pre`, `instance_post` (redundant!)
- These fields are joined from local CSV files when loading (no API call)
- Typical savings: **40-60%** depending on neuron name lengths

**v3.0 advantages:**
- ✅ 40-60% smaller cache files
- ✅ Searchable neuron database (registry)
- ✅ Better scalability for many queries (index)
- ✅ Stable cache keys (hash-based)
- ✅ No extra API calls (local enrichment)

---

## Migration from v2.0

The cache system is **forward-compatible** - v3.0 can coexist with v2.0 cache files.

### Automatic Migration

When you run v3.0 code:
1. Old v2.0 cache files remain untouched
2. New queries create v3.0 structure (registry + index)
3. Both cache systems work independently

### Manual Migration (Optional)

To rebuild cache with v3.0 structure:

```python
# 1. Clear old cache
fc.clear_cache()

# 2. Re-fetch data (will create v3.0 structure)
fc = FindNeuronConnection(
    dataset='optic-lobe:v1.1',
    sourceNeurons=['L3_R'],
    targetNeurons=['l-LNv_R'],
    min_synapse_num=10,
    use_cache=True,
    token='your_token'
)
```

---

## Technical Implementation

### Key Functions

```python
# Neuron Registry
_get_neuron_registry_path()      # Path to registry file
_load_neuron_registry()          # Load registry DataFrame
_save_neuron_registry(df)        # Save registry to parquet
_update_neuron_registry(conns)   # Extract neurons from connections

# Cache Index
_get_cache_index_path()          # Path to index file
_load_cache_index()              # Load index DataFrame
_save_cache_index(df)            # Save index to parquet
_update_cache_index(...)         # Add/update index entry
_find_cached_query(...)          # Search index for matching query

# Cache Keys
_get_cache_key_from_bodyids(...) # Generate hash-based key

# User-Facing
search_cached_neurons(pattern, field)  # Search registry
print_cache_info()                      # Display cache statistics
clear_cache()                           # Clear all cache files
```

### Data Flow

```
User Query
    ↓
Generate bodyId lists (upstream, downstream)
    ↓
Generate cache key (SHA256 hash)
    ↓
Search cache index for matching query
    ↓
    ├─ Found ──→ Load from cache → Filter by min_weight
    │
    └─ Not Found ──→ Fetch from Neuprint (min_weight=1)
                       ↓
                     Save to cache file
                       ↓
                     Update neuron registry
                       ↓
                     Update cache index
                       ↓
                     Filter by min_weight
```

---

## Best Practices

### 1. Always Use `min_weight=1` for Fetching

The cache system automatically fetches with `min_weight=1` and filters locally:

```python
# ✅ Good - Cache handles filtering
fc = FindNeuronConnection(
    sourceNeurons=['L3_R'],
    min_synapse_num=10,
    use_cache=True
)

# ❌ Bad - Don't manually set min_weight
# The cache will still fetch with min_weight=1 internally
```

### 2. Use Search Before Fetching

Check if neurons are already cached:

```python
# Search cached neurons
l3_neurons = fc.search_cached_neurons('L3.*', 'type')

if not l3_neurons.empty:
    print(f'Found {len(l3_neurons)} L3 neurons in cache')
    # Use cached bodyIds
else:
    print('L3 neurons not cached, will fetch from Neuprint')
```

### 3. Monitor Cache Size

```python
fc.print_cache_info()
# Check "Total cache size" - consider clearing if > 1 GB
```

### 4. Clear Old Cache Periodically

```python
# Clear cache if data is outdated (e.g., new Neuprint version)
fc.clear_cache()
```

---

## Troubleshooting

### Cache Not Working

**Problem:** Data still fetched from Neuprint despite cache enabled

**Solution:**
```python
# Check cache status
fc.print_cache_info()

# Verify cache is enabled
print(fc.use_cache)  # Should be True
```

### Search Returns Empty Results

**Problem:** `search_cached_neurons()` returns empty DataFrame

**Solution:**
```python
# Check if registry exists
registry = fc._load_neuron_registry()
print(f'Registry has {len(registry)} neurons')

# Try broader pattern
fc.search_cached_neurons('.*', 'type')  # All neurons
```

### Cache Size Too Large

**Problem:** Cache folder > 1 GB

**Solution:**
```python
# View cache info
fc.print_cache_info()

# Clear specific dataset
fc.clear_cache()

# Or use ManageCache.py for interactive management
```

---

## API Reference

### `search_cached_neurons(pattern, field='type')`

Search the neuron registry by pattern.

**Parameters:**
- `pattern` (str or int): Search pattern (regex for str, exact match for int)
- `field` (str): Field to search - 'type', 'instance', or 'bodyId'

**Returns:**
- `pd.DataFrame`: Matching neurons (bodyId, type, instance)

**Examples:**
```python
# Search by type
l3_neurons = fc.search_cached_neurons('L3.*', 'type')

# Search by instance
right_neurons = fc.search_cached_neurons('.*_R$', 'instance')

# Search by bodyId
neuron = fc.search_cached_neurons(5813022222, 'bodyId')
```

### `print_cache_info()`

Display comprehensive cache statistics.

**Output:**
- Cache location
- Total cached queries
- Total neurons in registry
- Cache files and size
- Query details (upstream/downstream counts, connections, creation time)
- Sample of neuron registry

### `clear_cache(confirm=True)`

Clear all cached data for the current dataset.

**Parameters:**
- `confirm` (bool): Require user confirmation (default: True)

**Effect:**
- Removes all connection files
- Removes neuron registry
- Removes cache index
- Removes old metadata file (if exists)

---

## Summary

**Cache System v3.0** provides a robust database architecture for managing Neuprint connection data with:

✅ **Stable cache keys** using SHA256 hashing  
✅ **Neuron registry** for local search  
✅ **Query index** for fast lookup  
✅ **Better scalability** with parquet databases  
✅ **Backward compatibility** with v2.0 cache  

This enables flexible, efficient local caching and search without repeated API calls.

---

**Related Documentation:**
- [Cache System Quick Start](CacheSystem_QuickStart.md)
- Cache System v2.0 Technical Overview
- Cache System v1.0 Documentation
