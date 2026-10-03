# FindAllPath() Method Documentation

## Overview
The `FindAllPath()` method finds **all paths** from source neurons to target neurons within the specified `max_interlayer`, keeping paths of different lengths while filtering out connections that don't lead to any target neurons.

## Key Differences from FindPath() and FindShortestPath()

### FindPath()
- Legacy forward layer-by-layer path routine.
- Stops discovery when all targets are found or the configured depth is
  reached, then enumerates paths in the discovered graph.
- Use `FindShortestPath()` when minimum-hop semantics are required.

### FindShortestPath()
- Starts from each target and follows incoming connections backward.
- Keeps the shortest paths independently for every exact
  `(source bodyId, target bodyId)` pair, including tied shortest paths.
- Aggregates the retained bodyId paths to type-level path rows afterward;
  it does not choose one minimum path for an entire type pair.
- The root-level `source_neurons.csv` and `target_neurons.csv` record source
  and target enrollment. The `user_warning_notes.txt` file explains when
  bodyId pairs may be hidden by type aggregation or when the depth bound
  prevents a global shortest-path claim.

### FindAllPath()
- Continues searching through **all layers** up to `max_interlayer`
- Returns **all paths** to target neurons, regardless of length
- Includes targets found at different layers (Layer 1, Layer 2, etc.)
- Filters out "dead-end" branches that don't connect to any target

## Algorithm

### FindAllPath Phase 1: Forward Search (Exploration)
1. Starting from source neurons, explore all downstream connections
2. Continue for all layers up to `max_interlayer`
3. Track which target neurons are found at each layer
4. Store all discovered connections in `conn_layers`

### FindShortestPath Phase 1: Target-rooted Search
1. Start at each requested target bodyId.
2. Query incoming connections layer by layer, stopping separately when the
   enrolled source bodyIds for that target have been reached or the exact
   hop bound is exhausted.
3. Retain only edges on shortest-DAG branches that reach requested sources;
   incoming branches from uninvolved sources are excluded.
4. Enumerate all tied shortest bodyId paths, then derive type-level paths from
   those concrete paths.

### FindAllPath Phase 2: Backward Filtering (Path Extraction)
1. Identify all target neurons found at any layer
2. Backtrack through layers to identify neurons that are part of paths to targets
3. Keep only connections where **both** pre- and post-synaptic neurons are in paths to targets
4. This preserves:
   - **All path lengths**: Targets found at Layer 1, Layer 2, Layer 3, etc.
   - **All alternative paths**: Multiple routes to the same target
   - **Convergent and divergent patterns**: One-to-many and many-to-one connections

### Why These Approaches Work
- **Forward search**: Ensures we explore all possible connections within the layer limit
- **Backward filtering**: Removes branches that don't lead anywhere (dead ends)
- **Target-rooted shortest search**: Avoids building the fan-out of sources that
  cannot participate in a requested target path.
- **Result**: Complete paths or exact per-bodyId shortest paths, depending on
  the selected method, with no uninvolved branches in the shortest graph.

## Output Structure

### Excel File Sheets
1. **parameters**: Analysis parameters
2. **source_neurons.csv** / **target_neurons.csv** at the run root: resolved
   enrollment details (`isInPath`, `Checked`, and `Layer`)
3. **source_neurons** / **target_neurons** sheets in Excel outputs
4. **total_weight_layer**: Total synaptic weight per layer
5. **connection_info**: All connections in paths (by bodyId)
6. **connection_type**: All connections in paths (by type)
7. **path_type**: All paths to targets (by neuron type)
8. **path_type_excluded**: Paths excluded by keyword filter
9. **path_bodyId**: All paths to targets (by bodyId) - if bodyId output is enabled
10. **layer_X**: Neuron information for each intermediate layer

For CSV runs, connection tables and supporting files are under
`data_details/`; the two enrollment CSVs remain at the run root.

### Sankey Diagrams
- **`visualization/Sankey_<run>.html`**: Shows all paths by neuron type
- **`bodyId_visualization/Sankey_<run>.html`**: Shows all paths by bodyId

The type-level and bodyId-level folders use the same `Network_<run>.html`,
`Heatmap_<run>.html`, and `Sankey_<run>.html` naming contract; the containing
folder identifies the visualization level. Early network previews use
`network_early/` and `network_early_bodyId/` with the same
`Network_<run>.html` prefix.

Both diagrams visualize the complete network including:
- All layers from source to furthest target
- All connections between layers
- Color-coded target neurons

## Console Output

```
Completed exploring all {N} layers (max_interlayer={M})
Target neurons found: {X} / {Y}

Filtering connections to keep only paths from source to target...
Neurons in paths to targets: {Z}

Path Network Statistics (source to target):
Total connections in paths: {A}
Total connection types in paths: {B}
Total neurons in paths: {C}
  Layer 0: {N0} neurons
  Layer 1: {N1} neurons
  Layer 2: {N2} neurons
  ...

Target neurons by layer:
  Layer 1: {T1} targets
  Layer 2: {T2} targets
  Layer 3: {T3} targets
  ...
```

## Use Cases

### When to Use FindAllPath()
1. **Comprehensive path analysis**: You want to see all possible routes, not just shortest
2. **Multi-layered targeting**: Targets are expected at different distances from source
3. **Alternative pathway discovery**: Finding backup routes or parallel pathways
4. **Network motif analysis**: Identifying patterns like feedforward loops, convergence, divergence

### When to Use FindShortestPath()
1. **Efficiency focus**: You only care about minimum-hop connections
2. **Large networks**: Target-rooted discovery avoids uninvolved source fan-out
3. **Exact pair analysis**: You need the shortest paths for each concrete
   source/target bodyId pair

## Example Scenario

Suppose you have:
- Source: PPL1 dopaminergic neurons
- Target: MBON output neurons
- max_interlayer: 3

**FindShortestPath()** would search backward from each MBON target and keep
the shortest path(s) separately for each source/target bodyId pair. If only
the type-level CSV is inspected, multiple target instances can collapse to the
same type sequence; use the bodyId output and the root enrollment files to
inspect the concrete pairs.

**FindAllPath()** would:
1. Search through all 3 layers
2. Find MBONs at Layer 1, Layer 2, and Layer 3
3. Keep all paths to each MBON
4. Remove neurons that connect to nothing downstream

Result: Complete picture of all PPL1→MBON pathways within 3 layers.

## Parameters

### Required
- Source neurons and target neurons must be initialized
- At least one target neuron must be found in the search
- For a globally reliable shortest-path result, increase the explored depth
  until the target-rooted search is not depth-capped. Otherwise the result is
  shortest only within the explored, threshold-filtered graph.

### Optional
- `find_bodyId_path=True`: Also generate bodyId-level paths (can be very large)
- `max_interlayer`: Maximum layers to search (default: 2)
- `min_synapse_num`: Minimum synapses for connection (default: 10)
- `min_ratio` / `min_traversal_probability`: retained for API compatibility
  and exported readout columns; both filters are disabled in Complete and
  Shortest Paths (default values are 0). They do not appear in the path-run
  folder name as `r[]p[]` notes.
- `drop_untyped`: Drop edges touching untyped neurons before the graph is
  built (type `bool`, default `True`). The shared predicate
  (`utils.label_utils.is_untyped_type_label`, also used by Cross-Dataset
  Comparison) treats a label as untyped when it is empty, an
  Unknown/None/NaN sentinel (case-insensitive), or the all-digit bodyId
  fallback. Applied after label enrichment and BEFORE graph construction in
  both modes, so an untyped neuron can never be an intermediate node of a
  returned path or visualization; an untyped source/target stays enrolled
  in `source_neurons.csv` / `target_neurons.csv` while its incident edges
  are removed. Dropped rows are exported to
  `data_details/untyped_dropped_records.csv` (only when something was
  dropped, with an `untyped_side` flag) and the counts appended to
  `user_warning_notes.txt`. This is a neuron-LABEL filter — distinct from
  graph budgets, path budgets and the drawing cap. `drop_untyped` is part
  of the FindAllPath graph-cache key, so runs with different values never
  reuse each other's cached graphs.

### Threshold provenance

Every pathfinding run (both modes) writes an applied-threshold provenance
block to `parameters.txt`, `all_attributes.json` and
`data_details/parameters.csv` with: `requested_threshold`,
`applied_threshold`, `applied_threshold_source` ('requested' |
'strongest_first_budget' | 'edge_budget' |
'strongest_first_budget+edge_budget'), `strongest_first_budget` (auto
1,000,000) and `strongest_first_budget_bitten`, `strongest_first_tau` (the
landing tau), `tau_canonical`, `strongest_dropped_bottleneck` (w2),
`edge_budget`, `edge_budget_applied`, `edge_budget_landing` (w1),
`edge_weight_floor` (w0), `strongest_retained_bottleneck` (W*), and
`paths_complete`. Backward-compatible `applied_tau` /
`edge_weight_floor` lines remain in `parameters.txt`.

## Performance Considerations

- Computation time increases with:
  - Number of layers (max_interlayer)
  - Network size (neurons and connections)
  - Number of paths to analyze
- Memory usage scales with the number of paths found
- Excel file size can be large for complex networks with many paths


## Connection-ratio basis

`weight_basis='connection_ratio'` swaps the edge-weight basis to the F9 connection ratio (synapses / the post neuron's all-post incoming mass at min_weight=1): `min_ratio` becomes the threshold (0 < t ≤ 1), the Edge Budget floors at float ratio tiers, and StrongestFirst bottlenecks are min-ratio (the allpaths `min_ratio` column). Synapse columns are unchanged everywhere; see `ConnectionRatioPaths.md`. Multi-threshold replay accepts float tiers; Shortest Paths supports the basis; the type-level refill covers ratio budgets.
