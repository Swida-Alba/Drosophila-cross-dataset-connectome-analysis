# Network (FindNetwork)

Build the **mutual direct-connection network among the queried neurons** —
equivalent to Direct Connections with source == target == the query set.
Backend: `FindNeuronConnection.FindNetwork`, using the FindAllPath pipeline
for enrichment, hemisphere-aware analysis, and visualization.

<div class="callout warn">
  <b>Limited scope:</b> the network contains only direct connections whose
  BOTH endpoints are in the queried set — intermediate neurons are never
  involved. For a more complete network that also involves intermediate
  neurons, use the <b>Find Path</b> tab with <b>Find Reciprocal
  Connections</b> enabled.
</div>

## Quick start

1. Add the **Query Neurons** (types, bodyIds, patterns, or an uploaded list).
2. Pick the **Threshold Basis** (Synapse count or Connection ratio) and set
   the threshold — ratio mode reads **Min Connection Ratio (%)**; Combine
   Thresholds (Any one / Both AND) decides whether both gates apply.
3. Click **Find Network**.

## What is computed

- One fetch of all downstream connections of the query set, filtered to
  edges whose post-synaptic partner is also in the set (both directions
  kept as directed edges).
- Enrichment with the FindAllPath semantics: connection ratios and
  traversal probabilities with **global** incoming-weight denominators,
  LabelMapper standardization, custom grouping when a mapping is selected.
- Hemisphere-aware analysis (same helpers as Find All Paths):
  **Hemisphere-aware** labels, **Symmetry Analysis** on the unfiltered
  table, and the **Keep Only Hemisphere-Conserved Edges** filter.
- **Pair report**: the run emits a 1-hop path table
  (`*_allpaths_type.csv`), so the same per-pair explorer report the
  pathfinding tools generate (`path_report.html` +
  `paths_pair_breakdown/`) is produced for network runs too.
- **Type-level refill**: ratio-basis runs always produce the
  complete-at-asked records (`data_details/type_level_refill/` + the
  `type_level_refill/` mirror); synapse-basis runs refill only when a
  budget bit, which a network never has.

## Outputs (nothing redundant)

The folder name uses the dataset abbreviation (`MCNS` = male-cns, `FAFB` =
flywire FAFB, ...) and a sanitized query name (e.g. `aMe.*` → `aMe`):

```
{output_dir}/find-network_MCNS_aMe_w3r0p0_20260815_142745/
  parameters.txt, all_attributes.json
  *_allpaths_type.csv               # 1-hop path table (pair-report input)
  source_neurons.csv, target_neurons.csv   # enrollment (refill reader)
  path_report.html, paths_pair_breakdown/   # per-pair explorer report
  type_level_refill/                # (ratio-mode runs; complete-at-asked)
  data_details/
    neurons.csv                     # all queried neurons
    connection_type.csv             # type-level mutual connections
    parameters.csv
    connection_info_bodyId.csv      # (when Skip BodyId is off)
    connection_custom_groups.csv    # (when a custom grouping is used)
    hemisphere_unconserved_edges.csv# (when the conserved filter is on)
  visualization/
    Network_*.html, Heatmap_*.html  # NO Sankey
    visualization_data/             # *_data_connections.csv,
                                    # *_data_original_paths.csv,
                                    # network_edges_input.csv
  user_warning_notes.txt            # (when any output-affecting option applied)
```

Visualizations are produced by the FindAllPath visualization backend
(VisualizePath) as **network + heatmap only** — the Sankey diagram and the
path/matrix exports of the path tools are intentionally omitted.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| Threshold Basis | Synapse count | Which weight basis thresholds the network (synapses or connection ratio) |
| Min Synapse Count | 3 | Minimum edge weight (ratio 'or' mode: floored to 1 by the backend) |
| Min Connection Ratio (%) | 1.0 | Ratio-mode threshold in percent (0.01 raw = 1%) |
| Combine Thresholds | Any one | 'or' (one threshold governs) or Both AND (both gates apply) |
| Search Columns | auto | Columns searched when resolving neuron names |
| Output Format | csv | csv or xlsx |
| Skip BodyId in Output | on | Skip the bodyId-level connection table |
| Hemisphere options | off | Same semantics as Find All Paths |

## Find Network vs Find Path + Find Reciprocal

| | Find Network | Find Path + Find Reciprocal |
|---|---|---|
| Node set | exactly the queried neurons | queried neurons + all discovered intermediates |
| Edges | direct connections within the query set | path edges, plus ALL direct connections among path-graph nodes |
| Depth | 1 hop only | up to Max Layers |
| Use when | you want the induced connectivity of a known neuron set | you want the neighborhood reachable between sources and targets |
