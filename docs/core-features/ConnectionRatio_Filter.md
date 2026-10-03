# Connection Ratio Filter (`min_ratio`)

## Overview

Added a new filtering parameter `min_ratio` that filters connections based on the **direct connection ratio** (`weight/post`) without the 0.3 scaling factor used in `traversal_probability`.

## Update Date
October 25, 2025

## New Feature: `min_ratio`

### Definition

**Connection Ratio** = `weight / post`

Where:
- `weight` = Number of synapses from neuron i to neuron j
- `post` = Total number of post-synaptic inputs to neuron j

This is the **raw proportion** of the downstream neuron's inputs that come from the upstream neuron.

### Comparison with `traversal_probability`

| Metric | Formula | Range | Meaning |
|--------|---------|-------|---------|
| **connection_ratio** | `weight / post` | 0.0 to 1.0 | Raw fraction of inputs |
| **traversal_probability** | `(weight / post) / 0.3` | 0.0 to 1.0 (capped) | Scaled probability |

**Relationship:** `traversal_probability = min(1.0, connection_ratio / 0.3)`

### Examples

**Example 1: Strong connection**
```
Synapse count (weight): 50
Post-synaptic inputs: 200

connection_ratio = 50 / 200 = 0.25 (25%)
traversal_probability = 0.25 / 0.3 = 0.833 (83.3%)
```

**Example 2: Moderate connection**
```
Synapse count (weight): 10
Post-synaptic inputs: 500

connection_ratio = 10 / 500 = 0.02 (2%)
traversal_probability = 0.02 / 0.3 = 0.067 (6.7%)
```

**Example 3: Weak connection**
```
Synapse count (weight): 2
Post-synaptic inputs: 1000

connection_ratio = 2 / 1000 = 0.002 (0.2%)
traversal_probability = 0.002 / 0.3 = 0.0067 (0.67%)
```

**Example 4: Very strong connection (capped)**
```
Synapse count (weight): 100
Post-synaptic inputs: 200

connection_ratio = 100 / 200 = 0.5 (50%)
traversal_probability = min(1.0, 0.5 / 0.3) = 1.0 (100%, capped)
```

## Implementation

### New Parameter

```python
class FindNeuronConnection:
    min_ratio: float = 0.0
    '''
    minimum connection ratio (weight/post) to be considered as connection
    connection ratio is calculated as w_ij / W_j
    where w_ij is the number of synapses from neuron i to neuron j 
    and W_j is the total number of post-synaptic sites of neuron j
    This is the direct ratio without the 0.3 scaling factor used in traversal_probability
    '''
```

### Usage

```python
fc = FindNeuronConnection(
    dataset='optic-lobe:v1.1',
    sourceNeurons=['L3'],
    targetNeurons=['aMe12', 'aMe9'],
    min_synapse_num=5,          # Absolute synapse count filter
    min_ratio=0.01,             # Connection must be ≥1% of inputs (NEW!)
    min_traversal_probability=0.05,  # Probability filter
    max_interlayer=2
)
```

### Filtering Logic

Connections are filtered if they fail **any** of these criteria:

1. `weight >= min_synapse_num` (e.g., ≥5 synapses)
2. `connection_ratio >= min_ratio` (e.g., ≥1% of inputs)
3. `traversal_probability >= min_traversal_probability` (e.g., ≥5%)

**All three filters are applied as edge-level filters** before pathfinding.

> **Update (2026-10-03):** for *pathfinding* runs this doc's filter
> description is historical — F9 (2026-09-05) made `connection_ratio` /
> `traversal_probability` readout columns there (the filter mechanics
> survive in the shared fetch layer used by direct-connection queries).
> The ratio returned to pathfinding in a stronger form: as a WEIGHT
> BASIS — see `ConnectionRatioPaths.md` (`weight_basis=
> 'connection_ratio'`): the threshold, Edge Budget tiers and
> StrongestFirst bottlenecks themselves run in ratio units. The
> 0.01-0.1 tables below are direct-connection-era guidance; real-data
> pathfinding tiers sit far lower (~5e-4..2e-3 on FAFB).

## Data Output

### Saved Columns

All connection data now includes both metrics:

```python
# Connection DataFrame columns:
- bodyId_pre
- bodyId_post
- weight                    # Synapse count
- connection_ratio          # weight / post (NEW COLUMN NAME!)
- traversal_probability     # (weight / post) / 0.3
- type_pre
- type_post
- ... (other columns)
```

**Note:** The column previously named `ratio_post` is now `connection_ratio` for clarity.

### Excel Output

When you run `FindPath()` or `FindAllPath()`, the saved Excel files will include:

**Sheet: `connection_info`**
```
bodyId_pre | bodyId_post | weight | connection_ratio | traversal_probability | type_pre | type_post
-----------|-------------|--------|------------------|----------------------|----------|----------
5813022222 | 722817260   | 15     | 0.033            | 0.111                | L3_R     | aMe12_R
5813022333 | 723456789   | 8      | 0.016            | 0.053                | L3_R     | aMe9_R
```

This allows you to see:
- **weight**: Raw synapse count
- **connection_ratio**: Direct proportion (1% = 0.01, 10% = 0.10)
- **traversal_probability**: Scaled probability used in path analysis

## Use Cases

### Use Case 1: Focus on Major Inputs

```python
# Only keep connections that are ≥5% of downstream neuron's inputs
fc = FindNeuronConnection(
    ...,
    min_ratio=0.05,  # ≥5% of inputs
    min_traversal_probability=0.0,  # Don't use probability filter
)
```

**Effect:** Filters out minor connections, focuses on major input sources.

### Use Case 2: Absolute Threshold

```python
# Minimum 10 synapses AND ≥2% of inputs
fc = FindNeuronConnection(
    ...,
    min_synapse_num=10,
    min_ratio=0.02,  # ≥2% of inputs
)
```

**Effect:** Ensures connections are both strong (absolute) and significant (relative).

### Use Case 3: Probability-Based

```python
# Use traversal probability (with 0.3 scaling)
fc = FindNeuronConnection(
    ...,
    min_ratio=0.0,  # No direct ratio filter
    min_traversal_probability=0.05,  # ≥5% probability
)
```

**Effect:** Uses traditional probability-based filtering (equivalent to ratio ≥ 0.015).

### Use Case 4: Combined Filtering

```python
# Strict filtering on all three dimensions
fc = FindNeuronConnection(
    ...,
    min_synapse_num=10,     # ≥10 synapses
    min_ratio=0.03,          # ≥3% of inputs
    min_traversal_probability=0.1,  # ≥10% probability
)
```

**Effect:** Very strict - connection must pass all three filters.

## Choosing Thresholds

### `min_ratio` Recommendations

| Threshold | Meaning | Use Case |
|-----------|---------|----------|
| 0.0 | No filter | Exploratory analysis |
| 0.001 | ≥0.1% of inputs | Very permissive |
| 0.01 | ≥1% of inputs | Moderate filtering |
| 0.03 | ≥3% of inputs | Focus on significant inputs |
| 0.05 | ≥5% of inputs | Strong connections only |
| 0.1 | ≥10% of inputs | Major inputs only |

### Equivalent Thresholds

To convert between `min_ratio` and `min_traversal_probability`:

```python
# If you want connection_ratio ≥ X:
min_ratio = X
min_traversal_probability = 0  # Don't use

# If you want traversal_probability ≥ Y:
min_ratio = 0  # Don't use
min_traversal_probability = Y

# Equivalence (before capping at 1.0):
min_traversal_probability = min_ratio / 0.3
min_ratio = min_traversal_probability * 0.3
```

**Examples:**
```
min_ratio = 0.01  ≈  min_traversal_probability = 0.033
min_ratio = 0.03  ≈  min_traversal_probability = 0.1
min_ratio = 0.05  ≈  min_traversal_probability = 0.167
min_ratio = 0.1   ≈  min_traversal_probability = 0.333
```

## Technical Details

### Edge-Level Filtering

Like `min_synapse_num` and `min_traversal_probability`, `min_ratio` is applied **before pathfinding**:

```python
# Filtering sequence:
1. Fetch connections (weight ≥ 1)
2. Calculate connection_ratio for each edge
3. Filter: weight >= min_synapse_num
4. Filter: connection_ratio >= min_ratio
5. Filter: traversal_probability >= min_traversal_probability
6. Build graph with filtered edges
7. Find paths
```

### Modified Functions

**`_fetch_connections_with_cache()`**
```python
def _fetch_connections_with_cache(
    self, 
    upstream_bodyIds, 
    downstream_bodyIds=None, 
    min_weight=None,
    min_traversal_prob=None,
    min_conn_ratio=None  # ← NEW PARAMETER
):
```

**`_load_connections_from_cache()`**
```python
def _load_connections_from_cache(
    self, 
    cache_key, 
    min_weight=None, 
    min_traversal_prob=None,
    min_conn_ratio=None  # ← NEW PARAMETER
):
```

### Output Example

```
Layer 0->1: 892 neurons fetched
  📂 Loaded from cache: upstream_abc123.parquet
     Total: 15,234 connections, filtered to 8,456 (weight ≥ 5, ratio ≥ 0.01, prob ≥ 0.05)
     Enriched with neuron info from complete local dataset (no API call)
```

Shows all three filters applied!

## Column Name Change

### Before
```python
conn_df['ratio_post'] = conn_df.weight / conn_df.post
```

### After
```python
conn_df['connection_ratio'] = conn_df.weight / conn_df.post
```

**Reason:** More descriptive name that clearly indicates it's the connection strength ratio.

## Backward Compatibility

### Existing Code

Old code without `min_ratio` will work unchanged:

```python
# Old code (still works)
fc = FindNeuronConnection(
    dataset='optic-lobe:v1.1',
    sourceNeurons=['L3'],
    targetNeurons=['aMe12'],
    min_synapse_num=10,
    min_traversal_probability=0.001
)
```

**Behavior:** `min_ratio` defaults to 0.0 (no filtering), so existing code behavior is unchanged.

### Saved Data

If you have old Excel files with `ratio_post` column, they're compatible - it's the same value, just renamed to `connection_ratio` in new outputs.

## Why Both `min_ratio` and `min_traversal_probability`?

**Different perspectives on the same data:**

### `min_ratio` (Direct Proportion)
- **Question:** "What fraction of the downstream neuron's inputs come from this connection?"
- **Answer:** 0.05 = "5% of inputs"
- **Use:** When you want to understand relative input strength directly

### `min_traversal_probability` (Scaled Probability)
- **Question:** "What's the probability a signal traverses this connection?"
- **Answer:** 0.167 = "16.7% probability" (from 5% inputs / 0.3)
- **Use:** Traditional neuroscience probability interpretation

**You can use either, both, or neither** depending on your analysis goals!

## Examples

### Example 1: Only Synapse Count

```python
fc = FindNeuronConnection(
    ...,
    min_synapse_num=10,
    min_ratio=0.0,
    min_traversal_probability=0.0
)
```

Keeps: Any connection with ≥10 synapses (regardless of proportion)

### Example 2: Only Connection Ratio

```python
fc = FindNeuronConnection(
    ...,
    min_synapse_num=1,
    min_ratio=0.05,  # ≥5% of inputs
    min_traversal_probability=0.0
)
```

Keeps: Any connection that is ≥5% of downstream neuron's inputs (even if only 2-3 synapses)

### Example 3: Balanced Approach

```python
fc = FindNeuronConnection(
    ...,
    min_synapse_num=5,      # At least 5 synapses
    min_ratio=0.01,          # AND ≥1% of inputs
    min_traversal_probability=0.0
)
```

Keeps: Connections that are both numerous enough (5+) and significant enough (1%+)

## Performance Impact

**Negligible** - filtering happens during the same post-synaptic count fetch:

```python
# Single API call for both filters:
post_df, _ = fetch_neurons(post_bodyIds)

# Calculate both from same data:
connection_ratio = weight / post
traversal_probability = connection_ratio / 0.3
```

No additional API calls or computation overhead!

## Conclusion

**Key Points:**

1. ✅ New `min_ratio` parameter for direct proportion filtering
2. ✅ Works alongside `min_synapse_num` and `min_traversal_probability`
3. ✅ All three are edge-level filters (applied before pathfinding)
4. ✅ `connection_ratio` column saved in all output files
5. ✅ Renamed `ratio_post` → `connection_ratio` for clarity
6. ✅ No performance overhead (uses same API call)
7. ✅ Fully backward compatible (defaults to 0.0)

**Recommended Usage:**

- **Exploratory:** `min_ratio=0.0` (no filter)
- **Moderate:** `min_ratio=0.01` (≥1% of inputs)
- **Strict:** `min_ratio=0.05` (≥5% of inputs)

**Choose based on your question:**
- "How many synapses?" → Use `min_synapse_num`
- "What fraction of inputs?" → Use `min_ratio`
- "What's the probability?" → Use `min_traversal_probability`
- "All three!" → Use all three filters together!

---

This gives you fine-grained control over connection quality from three complementary perspectives: absolute strength, relative strength, and probabilistic interpretation.
