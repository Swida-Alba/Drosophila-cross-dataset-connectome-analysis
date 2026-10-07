# Multiple Connections Between Same Neuron Pair - Handling Example

> **SUSPENDED (2026-10-06, unverifiable fidelity):** probability values are no longer exported or displayed. The `traversal_probability` columns shown in the examples below no longer appear in exported files.


## Overview
The updated `FindAllPath()` method now correctly handles and preserves ALL connections between the same neuron pair when they exist in different layers and are used in valid paths.

## Example Network Scenario

### Network Structure
```
Source neurons: [S1, S2]
Target neuron: [T]

Connections:
Layer 0->1: S1 → A (weight: 10, traversal_prob: 0.8)
Layer 1->2: A → B (weight: 15, traversal_prob: 0.7)
Layer 2->3: B → A (weight: 12, traversal_prob: 0.6)  # Reciprocal: A returns!
Layer 3->4: A → T (weight: 20, traversal_prob: 0.9)  # A→T from Layer 3
Layer 0->1: S2 → A (weight: 8, traversal_prob: 0.5)
Layer 1->2: A → T (weight: 18, traversal_prob: 0.85) # A→T from Layer 1 (direct path)
```

### Visual Representation
```
     S1 ──┐
           ├──> A ──> B ──> A ──> T
     S2 ──┘      └──────────────> T
                 (direct)
```

### Key Observation
**Neuron pair (A, T) has TWO different connections:**
1. Layer 1->2: A → T (weight: 18) - Direct path
2. Layer 3->4: A → T (weight: 20) - After reciprocal loop

## Old Behavior (Problematic)

### What Would Happen
```python
# Old code only tracked edges without layer info
edges_in_paths = {(A, T)}  # Just one entry

# When filtering connections
# BOTH Layer 1->2 A→T AND Layer 3->4 A→T would be kept
# Even if only one is actually used in paths
```

**Problem**: Can't distinguish which layer's connection is actually on a valid path.

## New Behavior (Correct)

### Phase 3: Path Finding with Layer Tracking

**Step 1: Find all paths**
```python
Paths found:
Path 1: S1 → A → T           (length: 2, uses Layer 0->1 S1→A, Layer 1->2 A→T)
Path 2: S1 → A → B → A → T  (length: 4, uses Layer 0->1 S1→A, Layer 1->2 A→B, 
                              Layer 2->3 B→A, Layer 3->4 A→T)
Path 3: S2 → A → T           (length: 2, uses Layer 0->1 S2→A, Layer 1->2 A→T)
Path 4: S2 → A → B → A → T  (length: 4, uses Layer 0->1 S2→A, Layer 1->2 A→B,
                              Layer 2->3 B→A, Layer 3->4 A→T)
```

**Step 2: Track the unique edges used by valid paths**
```python
edges_in_paths = {
    (S1, A),   # Used in all paths
    (S2, A),   # Used in Path 3 & 4
    (A, B),    # Used in Path 2 & 4
    (B, A),    # Used in Path 2 & 4
    (A, T),    # Used in ALL paths (both the direct and the reciprocal route)
}
```

**Key Point**: The graph stores one merged edge `(A, T)`; the layer tables can
contain that edge in several layers. Every layer-table occurrence of an edge
that lies on a valid path is preserved.

**Step 3: Match path edges against the actual rows of each layer table**

Each layer table is filtered to the rows whose `(bodyId_pre, bodyId_post)`
pair appears in `edges_in_paths`. The matching uses the table rows themselves
— NOT the edge's position inside a path. A path position is only an
approximation of the fetch layer: a reciprocal edge (`B → A`) or an edge of a
neuron reachable via a longer route than its discovery layer can live in a
layer table that differs from the path index, and index-based matching would
silently drop those real connections.

```python
Layer 0->1 filtering:
  Keep: S1 → A ✓ (in edges_in_paths)
  Keep: S2 → A ✓ (in edges_in_paths)

Layer 1->2 filtering:
  Keep: A → B ✓ (in edges_in_paths)
  Keep: A → T ✓ (in edges_in_paths)

Layer 2->3 filtering:
  Keep: B → A ✓ (in edges_in_paths)

Layer 3->4 filtering:
  Keep: A → T ✓ (in edges_in_paths)
```

### Final Output

**Connections DataFrame (`conn_inpath`):**
```
conn_layer  bodyId_pre  bodyId_post  weight  traversal_probability
0->1        S1          A            10      0.8
0->1        S2          A            8       0.5
1->2        A           B            15      0.7
1->2        A           T            18      0.85    ← Connection 1
2->3        B           A            12      0.6
3->4        A           T            20      0.9     ← Connection 2
```

**BOTH A→T connections are preserved!** ✓

## Biological Significance

### Why This Matters

Different connections between the same neuron pair in different layers can represent:

1. **Different synaptic locations**
   - Layer 1->2 A→T might be a direct axonal projection
   - Layer 3->4 A→T might be from a different axonal branch after processing

2. **Different functional contexts**
   - Direct path (A→T in L1): Fast, simple signal
   - Reciprocal path (A→B→A→T in L3): Processed, modulated signal

3. **Different weights/probabilities**
   - Different synaptic strengths
   - Different transmission probabilities
   - Different biological properties

### Real-World Example: Drosophila Brain

```
Mushroom Body (MB) neurons can have:
- Direct projections to output neurons (short path)
- Recurrent connections through protocerebral bridge (long path)
- Both pathways are functional and biologically relevant!
```

## Implementation Details

### Code Structure

```python
# 1. Track the unique edges used by valid paths
edges_in_paths = set()

for path in all_paths:
    for i in range(len(path) - 1):
        pre_node = path[i]
        post_node = path[i+1]
        edges_in_paths.add((pre_node, post_node))

# 2. Match against the ACTUAL rows of every layer table
#    (see _match_path_edges_to_layers in src/coana.py)
valid_pairs_by_layer, matched_path_pairs = _match_path_edges_to_layers(
    edges_in_paths, all_connections
)

for layer_idx, conn_df in enumerate(all_connections):
    if conn_df.is_empty():
        continue
    valid_pairs = valid_pairs_by_layer[layer_idx]
    if not valid_pairs:
        continue
    valid_pairs_df = pl.DataFrame(
        list(valid_pairs), schema=['bodyId_pre', 'bodyId_post'], orient='row'
    )
    conn_filtered = conn_df.join(valid_pairs_df, on=['bodyId_pre', 'bodyId_post'], how='inner')
```

### Statistics Reported

```
Total paths found: 4
Neurons in valid paths: 5 (S1, S2, A, B, T)
Unique edges in valid paths: 5 ((S1,A), (S2,A), (A,B), (B,A), (A,T))
Layer-specific edges in valid paths: 6 ((0,S1,A), (0,S2,A), (1,A,B), (1,A,T), (2,B,A), (3,A,T))
```

**Notice**: 5 unique edges but 6 layer-specific edges because (A,T) appears in 2 layers!

## Comparison Table

| Scenario | Old Behavior | New Behavior |
|----------|-------------|--------------|
| Single connection (A→B) in one layer | Keep it ✓ | Keep it ✓ |
| Same connection (A→B) in multiple layers, all used | Keep all ✓ | Keep all ✓ |
| Same connection (A→B) in multiple layers, only one used | Keep all ✗ | Keep all occurrences of used edges ✓ (the graph search cannot tell which layer occurrence was traversed, so all occurrences on a valid path are kept) |
| Reciprocal/recurrent edges whose fetch layer differs from the path index | **Dropped ✗** | **Kept ✓** |
| Different paths use different layer connections | Can't distinguish | All layer occurrences preserved ✓ |
| Reciprocal connections creating multilayer paths | May drop real edges ✗ | Include every used edge ✓ |

## Benefits Summary

✅ **Accurate**: Only connections actually on valid paths are kept
✅ **Complete**: All layer-specific variants are preserved if used
✅ **Biologically meaningful**: Different connection contexts are distinguished
✅ **Traceable**: Can identify which layer each connection belongs to
✅ **Analyzable**: Users can study layer-specific connection properties

## Example Output

### Excel File Sheets

**connection_info sheet:**
```
Shows all individual connections with:
- conn_layer: Which layer the connection is from
- bodyId_pre, bodyId_post: Neuron IDs
- weight: Synapse count
- traversal_probability: Signal transmission likelihood
```

**connection_type sheet:**
```
Aggregates by neuron type:
- Multiple layer occurrences of same type pair are summed
- Still preserves layer information in conn_layer column
```

### Sankey Diagram

**Will show:**
- Neuron A in multiple vertical positions (layers)
- Multiple links from A to T at different vertical levels
- Each link colored/weighted appropriately

### Interactive Network

**Will display:**
- Neuron A appears at its first occurrence layer
- Edge A→T may have multiple visual representations if layout allows
- Hover information shows which layer each connection is from

## Conclusion

The new implementation correctly handles the complex case of multiple connections between the same neuron pair by:

1. **Tracking layer context** for each edge during pathfinding
2. **Filtering layer-specifically** when extracting connections
3. **Preserving all variants** that appear on valid paths
4. **Maintaining biological detail** about connection diversity

This provides a complete and accurate representation of the neural network structure, especially important for circuits with reciprocal connections and complex topologies.
