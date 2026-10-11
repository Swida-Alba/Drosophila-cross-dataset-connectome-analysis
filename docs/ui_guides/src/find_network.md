---
title: Network
tag: Connectome
---

Build the mutual direct-connection network among the queried neurons —
equivalent to Direct Connections with source == target == the query set.
Runs on the FindAllPath backend (enrichment, hemisphere-aware analysis,
network + heatmap visualizations without Sankey).

<div class="callout warn">
  <b>Limited scope:</b> the network contains only direct connections whose
  BOTH endpoints are in the queried set — intermediate neurons are never
  involved. For a more complete network that also involves intermediate
  neurons, use the <b>Complete Paths</b> tab with <b>Find Reciprocal
  Connections</b> enabled.
</div>

## Quick start

1. Pick the **dataset** and, if needed, an **Output Directory**
   (it inherits your saved default otherwise).
2. Add the **Query Neurons** (e.g. `aMe12`, `PPL101`, or an uploaded list).
3. Pick the **Threshold Basis** (Synapse count or Connection ratio)
   and set the threshold — ratio mode reads **Min Connection Ratio
   (%)**; **Combine Thresholds** (Any one / Both AND) decides
   whether both gates apply.
4. Optionally define **Custom Groups**
   (label-mapper panel) or set a **Custom Group Name** / **Custom
   Target Name** / **Save Folder
   Name** for the run.
5. Click **Find Network**.

## Parameters

<table>
  <tr><th>Parameter</th><th>Default</th><th>Meaning</th></tr>
  <tr><td>Threshold Basis</td><td>Synapse count</td><td>Which weight basis thresholds the network: raw synapse counts or the connection ratio (weight ÷ all-post incoming weight of the post neuron — comparable across datasets)</td></tr>
  <tr><td>Min Synapse Count</td><td>3</td><td>Minimum edge weight. Ratio 'or' mode: the backend floors this to 1 — the ratio threshold prunes</td></tr>
  <tr><td>Min Connection Ratio (%)</td><td>1.0</td><td>Ratio-mode threshold in percent (0.01 raw = 1%): keep edges carrying at least this share of the post neuron's total input. Visible in ratio mode or AND mode</td></tr>
  <tr><td>Combine Thresholds</td><td>Any one</td><td>'or': one threshold governs (synapse mode → synapse count, ratio mode → ratio). Both (AND): an edge must clear BOTH thresholds, each on its own scale</td></tr>
  <tr><td>Visualization Edge Limit</td><td>500</td><td>Drawing-only cap on rendered edges; never changes which connections are fetched or exported</td></tr>
  <tr><td>Network Layout</td><td>distributed</td><td>HTML graph layout (distributed / circular / shell / spring)</td></tr>
  <tr><td>Search Columns</td><td>auto</td><td>Columns searched when resolving neuron names (Advanced Settings)</td></tr>
  <tr><td>Use Cache / Cache Only / Drop Untyped Neurons</td><td>on / off / checked</td><td>Advanced Settings: reuse cached connections, force offline, and drop edges touching untyped neurons (same shared predicate as the path tabs)</td></tr>
  <tr><td>Output Format</td><td>csv</td><td>csv or xlsx</td></tr>
  <tr><td>Skip BodyId in Output</td><td>on</td><td>Skip the bodyId-level connection table</td></tr>
  <tr><td>Hemisphere options</td><td>off</td><td><b>Hemisphere-aware</b>, the <b>Hemisphere</b> filter (both/left/right; neurons without a hemisphere side are always included), <b>Symmetry Analysis</b> and <b>Keep Only Hemisphere-Conserved Edges</b> — the last three are auto-unchecked and greyed out unless Hemisphere-aware is on. Same semantics as Complete Paths — see
    <a href="find_path.html#hemisphere-analysis">Hemisphere analysis</a></td></tr>
</table>

## Score formulas

Connection scores are computed on the **thresholded graph**: on the
Synapse basis, edges with `weight ≥ Min Synapse Count`; on
the Connection-ratio basis, edges with `ratio ≥ Min Connection
Ratio` (the synapse count floors to 1 — the ratio is the
threshold). Ratios use the
global total incoming weight, so the denominator reflects the whole
dataset:

<table>
  <tr><th>Score</th><th>Formula</th><th>Range</th></tr>
  <tr><td><code>connection_ratio</code></td><td>w_ij / Σ_k w_kj over D_t (edge weight ÷ total input to the post neuron)</td><td>0–1</td></tr>
  <tr><td><code>traversal_probability</code></td><td><b>SUSPENDED (2026-10-06, unverifiable fidelity)</b> — no longer exported, displayed, or used for ranking</td><td>—</td></tr>
  <tr><td><code>block_probability</code></td><td><b>SUSPENDED (2026-10-06, unverifiable fidelity)</b> — no longer exported, displayed, or used for ranking</td><td>—</td></tr>
</table>

Full derivations live in
[Score Calculation Guide](../core-features/ScoreCalculation_Guide.md).

## Pair report and type-level refill

Every network run also produces the per-pair explorer report the
pathfinding tools generate: the run writes a 1-hop path-compatible
table (`*_allpaths_type.csv`), and the same post-run hook
builds `path_report.html` +
`paths_pair_breakdown/` from it — per-pair breakdown with
coverage hovers and a Refill tab.

The **type-level refill is always produced**: every network run
writes the full-scope refill records (the `type_level_refill/`
mirror) — every emitted chain re-valued at the synapse &ge; 1 full
connection scope. Ratio-basis runs additionally get the
complete-at-asked records; synapse-basis runs get refill-asked records
when a budget bit, which a network never has — so for networks the
refill is full-scope only.

## Output

The folder name uses the dataset abbreviation (`MCNS` =
male-cns, `FAFB` = flywire FAFB, ...) and a sanitized query
name (e.g. `aMe.*` → `aMe`):

```
{output_dir}/find-network_MCNS_aMe_w3r0p0_20260815_142745/
  parameters.txt, all_attributes.json
  *_allpaths_type.csv              # 1-hop path table (pair-report input)
  source_neurons.csv, target_neurons.csv   # enrollment (refill reader)
  path_report.html, paths_pair_breakdown/  # per-pair explorer report
  type_level_refill/               # (ratio-mode runs; complete-at-asked)
  data_details/
    neurons.csv                     # all queried neurons
    connection_type.csv             # type-level mutual connections
    parameters.csv
    connection_info_bodyId.csv      # (when Skip BodyId is off)
    connection_custom_groups.csv    # (when a custom grouping is used)
    hemisphere_unconserved_edges.csv (when the conserved filter is on)
  visualization/
    Network_*.html, Heatmap_*.html  # NO Sankey
    visualization_data/             # *_data_connections.csv,
                                    # *_data_original_paths.csv,
                                    # network_edges_input.csv
  user_warning_notes.txt            # (when any output-affecting option applied)
```

## Find Network vs Complete Paths + Find Reciprocal

<table>
  <tr><th></th><th>Find Network</th><th>Complete Paths + Find Reciprocal</th></tr>
  <tr><td>Node set</td><td>exactly the queried neurons</td><td>queried neurons + all discovered intermediates</td></tr>
  <tr><td>Edges</td><td>direct connections within the query set</td><td>path edges, plus ALL direct connections among path-graph nodes</td></tr>
  <tr><td>Depth</td><td>1 hop only</td><td>up to Max Layers</td></tr>
</table>

<div class="callout info">
  <b>Exported run guide:</b> every successful run also writes
  <code>_UserGuide_please_read_me.html</code> into the run folder — a
  self-contained explanation of every output file, every table column, and
  any warnings recorded for the run. Format (html / txt / markdown /
  disabled) is set in Settings → Default Settings → Run Guide Format.
</div>
