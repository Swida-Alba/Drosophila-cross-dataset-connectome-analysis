# Cache System v4.0: Pair-Level Caching (Proposal)

> **⚠️ Design proposal, not the shipped system** — see [CacheSystem_v4_Implementation.md](./CacheSystem_v4_Implementation.md) and [CacheSystem_v4_Complete.md](./CacheSystem_v4_Complete.md) for what shipped. Kept for design history.

## Overview

**Proposed improvement**: Shift from **query-level caching** (caching entire neuron populations) to **pair-level caching** (caching individual neuron connections). This enables maximum cache reuse across different queries.

## Current Problem (v3.0)

### Query-Level Caching
```python
# Current behavior (Cache v3.0)
Run 1: Search [A, B, C] → API fetch → Save to: upstream_abc123.parquet
Run 2: Search [A, B]    → Different hash → API fetch → Save to: upstream_def456.parquet
Run 3: Search [B, C, D] → Different hash → API fetch → Save to: upstream_ghi789.parquet
```

**Issues:**
- ❌ **No cache reuse** even with overlapping neurons
- ❌ **Redundant storage** (A, B, C connections stored multiple times)
- ❌ **Wasted API calls** for already-cached neurons
- ❌ **Storage bloat** (3 files instead of 1 merged database)

### Example Inefficiency

```
Cache folder after 3 runs:
  upstream_abc123.parquet  → Contains: A→*, B→*, C→*  (15 MB)
  upstream_def456.parquet  → Contains: A→*, B→*      (10 MB) [DUPLICATE!]
  upstream_ghi789.parquet  → Contains: B→*, C→*, D→* (18 MB) [DUPLICATE!]
  
Total: 43 MB (but only ~23 MB unique data!)
```

## Proposed Solution (v4.0)

### Pair-Level Caching

```python
# Proposed behavior (Cache v4.0)
Run 1: Search [A, B, C] 
  → Check cache for A, B, C individually
  → All missing → API fetch all
  → Save: A→X, A→Y, B→Z, C→W as individual records

Run 2: Search [A, B]
  → Check cache for A, B individually  
  → A found ✅, B found ✅
  → No API call! Return cached data

Run 3: Search [B, C, D]
  → Check cache for B, C, D individually
  → B found ✅, C found ✅, D missing ❌
  → API fetch only D → Save D→* to cache
```

**Benefits:**
- ✅ **Maximum reuse**: Any subset of cached neurons can be retrieved
- ✅ **Minimal storage**: Each neuron pair stored only once
- ✅ **Optimal API usage**: Only fetch uncached neurons
- ✅ **Incremental caching**: Database grows with each new neuron

## Architecture Design

### Database Structure

#### 1. **Master Connection Database** (`connections.parquet`)

Single unified database with all cached connections:

```python
Schema:
{
    'bodyId_pre': int64,     # Upstream neuron
    'bodyId_post': int64,    # Downstream neuron  
    'weight': int64,         # Synapse count
    'roi': str,              # Brain region (optional)
    'cached_date': datetime  # When cached (for cleanup)
}

Indexed on: (bodyId_pre, bodyId_post) for fast lookup
```

**Example:**
```
bodyId_pre  bodyId_post  weight  roi      cached_date
5813022222  722817260    12      ME(R)    2024-01-15 10:30:45
5813022222  723917260    8       LO(R)    2024-01-15 10:30:45
5813022333  722817260    15      ME(R)    2024-01-15 10:30:45
...
```

#### 2. **Neuron Index** (`neuron_index.parquet`)

Track which neurons have been fully cached:

```python
Schema:
{
    'bodyId': int64,              # Neuron bodyId
    'type': str,                  # Neuron type
    'instance': str,              # Neuron instance
    'downstream_complete': bool,  # All downstream connections cached?
    'upstream_complete': bool,    # All upstream connections cached?
    'last_fetched': datetime,     # Last API fetch time
    'connection_count': int64     # Number of connections
}
```

**Example:**
```
bodyId      type    instance  downstream_complete  upstream_complete  last_fetched         connection_count
5813022222  L3_R    L3_R      True                 False              2024-01-15 10:30:45  234
5813022333  L3_R    L3_R      True                 False              2024-01-15 10:30:45  189
722817260   l-LNv_R l-LNv_R   False                True               2024-01-15 10:30:45  456
```

**Purpose:**
- Know which neurons are **fully cached** (all connections)
- Know which neurons are **partially cached** (only specific targets)
- Determine what needs to be fetched from API

### Core Logic

#### Query Resolution Algorithm

```python
def get_connections(upstream_bodyIds, downstream_bodyIds=None):
    """
    Smart cache query that maximizes reuse.
    """
    # Step 1: Separate cached vs uncached neurons
    cached_neurons = []
    uncached_neurons = []
    
    neuron_index = load_neuron_index()
    
    for bodyId in upstream_bodyIds:
        if bodyId in neuron_index['bodyId'].values:
            row = neuron_index[neuron_index['bodyId'] == bodyId].iloc[0]
            
            # Check if we have what we need
            if downstream_bodyIds is None:
                # Need all downstream
                if row['downstream_complete']:
                    cached_neurons.append(bodyId)
                else:
                    uncached_neurons.append(bodyId)
            else:
                # Need specific targets - check if all targets are in cache
                if all_targets_in_cache(bodyId, downstream_bodyIds):
                    cached_neurons.append(bodyId)
                else:
                    uncached_neurons.append(bodyId)
        else:
            uncached_neurons.append(bodyId)
    
    # Step 2: Load cached connections
    results = []
    
    if cached_neurons:
        print(f'  📂 Loading {len(cached_neurons)} neurons from cache...')
        cached_df = query_connection_database(
            bodyId_pre=cached_neurons,
            bodyId_post=downstream_bodyIds
        )
        results.append(cached_df)
    
    # Step 3: Fetch uncached neurons from API
    if uncached_neurons:
        print(f'  🌐 Fetching {len(uncached_neurons)} neurons from API...')
        api_df = fetch_from_api(uncached_neurons, downstream_bodyIds)
        
        # Save to database
        append_to_connection_database(api_df)
        update_neuron_index(uncached_neurons, downstream_bodyIds)
        
        results.append(api_df)
    
    # Step 4: Combine and return
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()
```

#### Database Query Functions

```python
def query_connection_database(bodyId_pre=None, bodyId_post=None, min_weight=1):
    """
    Query master connection database.
    Uses efficient index lookups.
    """
    conn_db = pd.read_parquet('connections.parquet')
    
    # Filter by upstream neurons
    if bodyId_pre is not None:
        conn_db = conn_db[conn_db['bodyId_pre'].isin(bodyId_pre)]
    
    # Filter by downstream neurons
    if bodyId_post is not None:
        conn_db = conn_db[conn_db['bodyId_post'].isin(bodyId_post)]
    
    # Filter by weight
    conn_db = conn_db[conn_db['weight'] >= min_weight]
    
    return conn_db

def append_to_connection_database(new_connections_df):
    """
    Append new connections to database.
    Handles duplicates (update weight if changed).
    """
    db_path = 'connections.parquet'
    
    if os.path.exists(db_path):
        existing_df = pd.read_parquet(db_path)
        
        # Merge on (bodyId_pre, bodyId_post)
        # Keep newer data for duplicates
        merged_df = pd.concat([existing_df, new_connections_df])
        merged_df = merged_df.drop_duplicates(
            subset=['bodyId_pre', 'bodyId_post'],
            keep='last'  # Keep newest
        )
    else:
        merged_df = new_connections_df
    
    # Add timestamp
    merged_df['cached_date'] = datetime.now()
    
    # Save with compression
    merged_df.to_parquet(db_path, index=False, compression='gzip')
```

## Performance Analysis

### Storage Comparison

**Current (v3.0) - Query-level:**
```
3 queries with overlapping neurons:
  Query 1: [A, B, C] → 10,000 connections → 1.2 MB
  Query 2: [A, B]    →  6,000 connections → 0.8 MB (60% duplicate!)
  Query 3: [B, C, D] → 12,000 connections → 1.5 MB (50% duplicate!)
  
Total: 3.5 MB (actual unique: ~2.0 MB)
Efficiency: 57%
```

**Proposed (v4.0) - Pair-level:**
```
Same 3 queries with pair-level caching:
  Database: 15,000 unique connections → 2.0 MB
  
Total: 2.0 MB
Efficiency: 100%
Savings: 43%
```

### API Call Comparison

**Scenario**: Search 100 different neuron combinations with 50% overlap

**Current (v3.0):**
```
API calls: 100 (every query is a cache miss for exact match)
Total neurons fetched: 10,000
```

**Proposed (v4.0):**
```
API calls: ~60 (only for new neurons)
Total neurons fetched: ~6,000
Savings: 40% API calls, 40% data transfer
```

### Query Speed

**Cache hit (fully cached):**
```
v3.0: Read 1 parquet file → 50-100ms
v4.0: Query indexed database → 30-80ms
Result: Similar or faster ✅
```

**Partial cache hit (v4.0 advantage):**
```
v3.0: Cache miss → Full API call → 2-10s
v4.0: Load 80% from cache + API call for 20% → 0.5-3s
Result: 3-5x faster ✅
```

## Migration Path

### Phase 1: Implement Alongside v3.0
- Keep existing query-level cache working
- Add new pair-level database
- New queries use v4.0, old cache still readable

### Phase 2: Migration Tool
- Provide `migrate_cache_v3_to_v4.py` script
- Reads all v3 parquet files
- Merges into single v4 database
- Removes duplicate pairs

### Phase 3: Deprecate v3.0
- Set v4.0 as default
- Keep v3.0 read support for 6 months
- Eventually remove old code

## Implementation Checklist

### Core Changes

- [ ] Create `connections.parquet` master database structure
- [ ] Create `neuron_index.parquet` with completion tracking
- [ ] Implement `query_connection_database()` with efficient indexing
- [ ] Implement `append_to_connection_database()` with deduplication
- [ ] Implement smart cache query logic (separate cached/uncached)
- [ ] Update `_get_layer()` to use new caching system
- [ ] Add database maintenance tools (compact, remove old entries)

### Supporting Features

- [ ] Cache statistics: show size, hit rate, duplicate ratio
- [ ] Migration tool: convert v3 → v4
- [ ] Cleanup tool: remove connections older than X months
- [ ] Index optimization: ensure fast lookups
- [ ] Incremental save: append without full reload

### Testing

- [ ] Test partial cache hits
- [ ] Test incremental neuron addition
- [ ] Benchmark query performance vs v3.0
- [ ] Test with large datasets (millions of connections)
- [ ] Verify storage savings
- [ ] Test concurrent access (multiprocessing safety)

## Example Usage

### Before (v3.0)

```python
# Run 1
conn = FindNeuronConnection(
    dataset='optic-lobe:v1.1',
    sourceNeurons=['L3', 'L4', 'L5'],  # Hash: abc123
    use_cache=True
)
conn.FindAllPath()
# Creates: upstream_abc123.parquet

# Run 2 (partial overlap)
conn2 = FindNeuronConnection(
    dataset='optic-lobe:v1.1',
    sourceNeurons=['L3', 'L4'],  # Hash: def456 (different!)
    use_cache=True
)
conn2.FindAllPath()
# Cache miss! Creates: upstream_def456.parquet
# API call for L3, L4 even though already cached!
```

### After (v4.0)

```python
# Run 1
conn = FindNeuronConnection(
    dataset='optic-lobe:v1.1',
    sourceNeurons=['L3', 'L4', 'L5'],
    use_cache=True
)
conn.FindAllPath()
# Saves: L3→*, L4→*, L5→* to connections.parquet
# Output: 🌐 Fetching 3 neurons from API...

# Run 2 (partial overlap)
conn2 = FindNeuronConnection(
    dataset='optic-lobe:v1.1',
    sourceNeurons=['L3', 'L4'],
    use_cache=True
)
conn2.FindAllPath()
# Queries database for L3, L4 connections
# Output: 📂 Loading 2 neurons from cache...
# No API call! ✅

# Run 3 (partial overlap + new neuron)
conn3 = FindNeuronConnection(
    dataset='optic-lobe:v1.1',
    sourceNeurons=['L4', 'L5', 'L6'],
    use_cache=True
)
conn3.FindAllPath()
# Output: 📂 Loading 2 neurons from cache...
#         🌐 Fetching 1 neuron from API...
# Only L6 fetched from API! ✅
```

## Technical Considerations

### Indexing Strategy

**PyArrow/Parquet Indexing:**
```python
# Write with index on (bodyId_pre, bodyId_post)
import pyarrow as pa
import pyarrow.parquet as pq

# Create schema with metadata
schema = pa.schema([
    ('bodyId_pre', pa.int64()),
    ('bodyId_post', pa.int64()),
    ('weight', pa.int64()),
    ('roi', pa.string())
])

# Write with row group partitioning for fast queries
pq.write_table(
    table,
    'connections.parquet',
    compression='gzip',
    row_group_size=100000,  # Optimize for common query sizes
    use_dictionary=['roi']  # Compress repeated ROI names
)
```

### Query Optimization

**Multi-level filtering:**
```python
# Fast path for small queries
if len(upstream_bodyIds) < 100:
    # Direct filter on DataFrame
    result = conn_db[conn_db['bodyId_pre'].isin(upstream_bodyIds)]
else:
    # Use ParquetDataset with row group filtering
    dataset = pq.ParquetDataset('connections.parquet')
    result = dataset.read(
        filters=[('bodyId_pre', 'in', upstream_bodyIds)]
    ).to_pandas()
```

### Concurrency Safety

**Multiprocessing considerations:**
```python
# Use file locking for writes
import fcntl

def append_to_database_safe(new_data):
    lock_file = 'connections.lock'
    
    with open(lock_file, 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # Exclusive lock
        try:
            append_to_connection_database(new_data)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)  # Release lock
```

## Future Enhancements

### 1. **ROI-Specific Caching**
Cache connections per brain region separately for even more granular reuse.

### 2. **Time-Based Expiry**
Automatically refresh old cached data (e.g., neurons cached >6 months ago).

### 3. **Lazy Loading**
Only load connection columns actually needed (e.g., skip 'roi' if not used).

### 4. **Distributed Cache**
Share cache database across team members (sync via cloud storage).

### 5. **Compression Optimization**
Use better compression for large databases (e.g., zstd instead of gzip).

## Conclusion

**Pair-level caching (v4.0)** solves the fundamental inefficiency of query-level caching by:

✅ **Maximizing reuse**: Any neuron subset can be cached/retrieved  
✅ **Minimizing storage**: Each connection stored once  
✅ **Optimizing API usage**: Only fetch truly new neurons  
✅ **Simplifying structure**: One database instead of many files  

**Impact on your workflow:**
- First run with [A, B, C]: Full API fetch (same as now)
- Second run with [A, B]: **Instant** (no API call!)
- Third run with [B, C, D]: Only fetch D (**3x faster**)

This is a significant architectural improvement that aligns with how users actually search connectome data: exploring overlapping neuron populations rather than exact repeated queries.

---

**Next steps**: Would you like me to implement this? I can start with the core database functions and gradually migrate from v3.0.
