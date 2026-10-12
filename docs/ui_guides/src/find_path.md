---
title: Complete Paths
tag: Connectome
---

Discover multi-hop pathways between source and target neuron groups in a single dataset, e.g. <code>aMe12</code> → <code>aMe10</code>.

  
  
## Quick start

1. Add <b>Source Neurons</b> (<code>aMe12</code>, <code>aMe10</code>) and <b>Target Neurons</b> (<code>PPL101</code>, <code>PPL103</code>) as chips, paste, or upload a list.
2. Pick the <b>Dataset</b> and an <b>Output Directory</b>.
3. Keep <b>Max Intermediate Layers = 2</b> and click <b>Complete Paths</b>; watch the live log.

  
## Inputs &amp; examples

Every entry may be a type, bodyId, instance, or pattern. See <a href="input_formats.html">Input File Formats</a> for CSV/TSV/Excel.

  

<table>
  <tr><th>Example</th><th>What it does</th></tr>
  <tr><td><code>aMe12</code>, <code>aMe10</code> (source)</td><td>Two source types: all their bodyIds are starting points</td></tr>
  <tr><td><code>PPL101</code>, <code>PPL103</code> (target)</td><td>Two target types</td></tr>
  <tr><td><code>all_neurons</code> (source or target)</td><td>Loads the full (typed) neuron set on that side to fetch all adjacent neurons at the given thresholds. Replaces every other chip, and forces <b>Max Intermediate Layers = 0</b> (direct connections only). Source and target cannot both be <code>all_neurons</code>.</td></tr>
  <tr><td><code>KC.*</code> with Match by = Regex</td><td>Expands to all types starting with <code>KC</code> — pattern matching needs the <b>Match by = Regex</b> option (Starts with would match the literal string <code>KC.*</code>)</td></tr>
  <tr><td><code>720575940610453042</code></td><td>A single bodyId (kept as integer)</td></tr>
  <tr><td><code>aMe12</code> then <code>aMe10</code>, Enter after each</td><td>Two chips — the whole typed text becomes <b>one</b> chip (commas are kept inside a name, never treated as separators); commit each entry with Enter</td></tr>
</table>

<h2 id="all-neurons-queries">All-neurons queries (<code>all_neurons</code>)</h2>

Typing the special chip <code>all\_neurons</code> as the <b>source or target</b> loads the full (typed) neuron set on that side, so the run fetches <b>every adjacent neuron at the given thresholds</b> (Min Synapse Count):

- <code>all\_neurons</code> <b>replaces every other chip</b> in the same input.
- Source and target cannot both be <code>all\_neurons</code>.
- An <code>all\_neurons</code> side forces <b>Max Intermediate Layers = 0</b> (direct connections only) — the search stays bounded instead of exploring the whole graph. Use it to list all partners of a neuron group above a synapse threshold.

  

<div class="callout info">
  <b>Script/API parity:</b> the backend
  (<code>FindNeuronConnection.InitializeNeuronInfo</code>) enforces the
  same rules, so passing <code>sourceNeurons=['all_neurons']</code> in
  code behaves identically. All typed neurons are loaded from local data
  (works offline with Cache Only).
</div>

  
## Parameters

  

<table>
  <tr><th>Parameter</th><th>Default</th><th>Meaning</th><th>Example</th></tr>
  <tr><td>Max Intermediate Layers</td><td>2</td><td>Path length: 0 = direct, 2 = two intermediates</td><td>3 finds longer circuits but is slower</td></tr>
  <tr><td>Min Synapse Count</td><td>3</td><td>Connections below this weight are removed</td><td>10 keeps only strong edges</td></tr>
  <tr><td>Visualization Edge Limit</td><td>500</td><td>Drawing-only cap: maximum edges drawn per visualization (network / Sankey / heatmap); never changes fetching, the graph, or path outputs — a single complete path may exceed it to stay intact, and bodyId-level and type-level visualizations share the same cap</td><td>50 for very dense regions</td></tr>
  <tr><td>Edge Budget</td><td>1,000,000</td><td>Graph-level floor ('all' mode only). Fix D (budget-fit search): after the lossless prunes, a discovery cone exceeding this many bodyId edges is floored at the weakest weight tier whose lossless-closed cone still fits the cap — exactly equivalent to raising Min Synapse Count, reported as <code>edge_weight_floor</code> together with the residual slack and probe trace; when even the full closed cone fits, no floor is applied. 0 = off</td><td>0 to keep the full lossless-pruned cone</td></tr>
  <tr><td>Fast Mode (corridor lane)</td><td>auto</td><td>The second of the two analysis modes: <b>Complete mode</b> (this run) finds all intact paths &ge; the applied threshold; <b>Fast mode</b> is the route-scoped deep pass (<em>fourth pruning axis</em>): after the run, a separate lane re-enumerates at the <strong>ASKED</strong> threshold restricted to validated top-M* type-level corridors (per source-target type pair &times; hop length, phantom-validated, 100-path class budgets). Writes only <code>route_scope_paths/</code> beside the run (route_paths_type.csv = one row per validated walk with engine-style weights/both ratio flavors (plain + adjusted, like the Refill tab)/coverage plus applied_tau/route_ub/route_floor_h/plan_rank; route_paths_bodyId.csv; route_connection_type.csv; route_provenance.json; README) and <code>data_details/route_plan/plan.json</code> — <strong>never modifies this run's own outputs</strong>; a note appears in user_warning_notes.txt only if the lane refuses or fails. <code>auto</code> = runs after deep searches (Max Intermediate Layers &ge; 4); the live status line under the control shows whether it will fire at the current depth &times; mode. Values: <code>complete</code> (Complete mode only) / <code>auto</code> (Fast on deep runs) / <code>both</code> (always). Legacy saved values <code>off</code>/<code>on</code> still work (normalized). Fast also runs standalone on any existing run folder: <code>python scripts/CorridorDeepDive.py &lt;run_dir&gt;</code>. Ratio-basis runs are refused (corridor ranking is synapse-bottleneck). See the <a href="pathfinding_algorithms.html#corridor-lane">pathfinding algorithms guide</a>.</td><td><code>off</code> to skip the pass entirely</td></tr>
  <tr><td>Min Connection Ratio</td><td>hidden</td><td>F9: readout column only (weight / ALL-POST incoming weight, threshold-free); no longer filters</td><td>—</td></tr>
  <tr><td>Min Traversal Prob.</td><td>hidden</td><td>F9: readout column only (ratio/0.3, capped at 1.0); no longer filters</td><td>—</td></tr>
  <tr><td>Max Paths (BodyId)</td><td>0 = auto (1M)</td><td>Path-output budget. StrongestFirst path budget: when the search exceeds it, ALL paths above the achieved strength cutoff (tau) are kept and tau is reported in the run notes. StrongestFirst is the built-in algorithm (the Algorithm selector was removed). See the <a href="pathfinding_algorithms.html">pathfinding algorithms guide</a>.</td><td>Lower for bounded runs on dense queries</td></tr>
  <tr><td>Drop Untyped Neurons</td><td>checked</td><td>Neuron-label filter (shared predicate with Cross-Dataset Comparison, <code>utils.label_utils.is_untyped_type_label</code>): removes edges touching untyped neurons — empty, Unknown/None/NaN sentinel (case-insensitive), or all-digit bodyId-fallback type labels. Applied after label enrichment and BEFORE graph construction, so an untyped neuron can never be an intermediate node of a returned path or visualization (an untyped source/target stays enrolled but its incident edges are removed). Dropped rows: <code>data_details/untyped_dropped_records.csv</code> (written only when something was dropped, with an <code>untyped_side</code> flag); counts appended to <code>user_warning_notes.txt</code>. Distinct from the Edge Budget, Max Paths and the Visualization Edge Limit</td><td>uncheck to keep bodyId-fallback neurons</td></tr>
  <tr><td>Search Columns</td><td>auto</td><td>Columns searched when resolving source/target names: <code>auto</code> = all columns (bodyId → type → instance → flywireType/hemibrainType/…), or restrict to <code>type</code> / <code>instance</code> / <code>bodyId</code>. Needed when a name lives in a cross-dataset column (e.g. <code>MTe07</code> only exists in <code>flywireType</code> on male-cns v1.0).</td><td><code>auto</code></td></tr>
  <tr><td>Filter By</td><td>bodyId</td><td>bodyId (per neuron) or type (aggregated)</td><td>type for cleaner summaries</td></tr>
  <tr><td>Combine Thresholds</td><td>Any one (default)</td><td>How Min Synapse Count and Min Connection Ratio combine when both are set: <em>Any one</em> — they are alternatives (one threshold governs the graph; the hidden knob is a readout); <em>Both (AND)</em> — an edge enters the graph only when it clears BOTH thresholds, each on its own scale. Under AND both inputs stay visible and the run folder token carries both tiers (<code>L{d}w{t}r{t_r}</code>).</td><td>Both (AND) when strong-but-thin and thin-but-dense edges must both be excluded</td></tr>
  <tr><td>Output Format</td><td>csv</td><td>csv (fast) or xlsx</td><td>xlsx for Excel formatting</td></tr>
  <tr><td>Network Layout</td><td>distributed</td><td>HTML graph layout</td><td>spring for organic look</td></tr>
</table>

  
## Score formulas

Every connection and path is scored on the <b>thresholded graph</b> D\_t (edges with <code>weight ≥ Min Synapse Count</code>). Ratios use the global total incoming weight so the denominator reflects the whole dataset, not just the query:

<strong>Threshold Basis</strong>: choose <em>Synapse count</em> (default) or <em>Connection ratio</em>. Ratio mode thresholds edges by the fraction of the post neuron's total input (synapses / all-post incoming), ranks paths by the exported type-path <code>min\_ratio</code> (a <b>realized</b> mass share — see below), and writes <code>data\_details/ratio\_synapse\_map.csv</code> with the per-neuron synapse cutoffs. Viable ratio values sit far below the synapse scale (e.g. 0.0005 on FAFB).

  

<table>
  <tr><th>Score</th><th>Formula</th><th>Range</th></tr>
  <tr><td><code>connection_ratio</code></td><td>w_ij / Σ_k w_kj-&gt;j — edge weight ÷ the post neuron's total incoming weight, threshold-free (min_weight&nbsp;=&nbsp;1, all dataset sources)</td><td>0–1</td></tr>
  <tr><td><code>connection_ratio_adj</code></td><td>edge weight ÷ the post's incoming mass counting only edges ≥ Min Synapse Count (per neuron; member-summed at type level). Always ≥ connection_ratio and moves with the threshold by design; equals connection_ratio when Min Synapse Count is 1</td><td>0–1</td></tr>
  <tr><td><code>traversal_probability</code></td><td><b>SUSPENDED (2026-10-06, unverifiable fidelity)</b> — no longer exported, displayed, or used for ranking</td><td>—</td></tr>
  <tr><td><code>block_probability</code></td><td><b>SUSPENDED (2026-10-06, unverifiable fidelity)</b> — no longer exported, displayed, or used for ranking</td><td>—</td></tr>
  <tr><td><code>path_prob</code></td><td><b>SUSPENDED (2026-10-06, unverifiable fidelity)</b> — no longer exported, displayed, or used for ranking</td><td>—</td></tr>
  <tr><td><code>min_weight</code></td><td>weakest hop of the path — the <b>realized</b> synapse mass at that hop (the distinct bodyId edges backing it at that position)</td><td>integer</td></tr>
  <tr><td><code>min_ratio</code></td><td>smallest per-hop ratio — the <b>realized</b> share on connection-ratio runs (the hop's realized mass ÷ the post type's full-membership incoming mass; can fall below the applied bodyId-level threshold); the pair-level readout in synapse runs</td><td>0–1</td></tr>
</table>

<b>SUSPENDED (2026-10-06, unverifiable fidelity):</b> the aggregate traversal probabilities are no longer computed for display or export (historically the <b>product</b> method, <code>1 − Π(1 − p\_pair)</code>). Full derivations live in <a href="../core-features/ScoreCalculation_Guide.md">Score Calculation Guide</a>.

  
## Advanced options

- <b>Custom source/target names</b>: labels used in file and folder names (e.g. <code>aMe\_clock</code>), plus <b>Save Folder Name</b> for a custom output folder instead of the auto name — both live in the Output Options card.
- <b>Keywords to exclude</b>: paths containing e.g. <code>None</code> or <code>unknown</code> are removed.
- <b>Skip BodyId in Output</b>: only type-level results are saved.
- <b>Use Cache</b>: reuse cached connection data (on by default); <b>Cache Only (Offline)</b>: never contacts the server; requires a pre-built cache.
- <b>Show Figure</b>: open the interactive figures in the browser as they are generated.
- <b>Custom Grouping</b>: map neurons into custom groups before the comparison — see the <a href="label_mapper.html">label-mapper guide</a>.
- <b>Find Reciprocal Connections</b> (a standalone Core Parameters checkbox, off by default): also enumerate the reverse-direction paths and write the <code>find\_reciprocal/</code> outputs.
- <b>Hemisphere analysis</b>: separate L/R labels, a left/right filter, symmetry outputs, and keep only hemisphere-conserved edges. See <a href="#hemisphere-analysis">Hemisphere analysis</a> below.

  

<h2 id="hemisphere-analysis">Hemisphere analysis</h2>

Both Complete Paths and Shortest Paths can restrict or split the analysis by brain hemisphere. Hemisphere assignment is done automatically per neuron:

1. The <b>Soma side</b> / <b>hemisphere</b> column is used when present (NeuPrint neuron tables).
2. Otherwise the <b>instance</b> suffix is used: <code>\_R</code> → right, <code>\_L</code> → left.
3. Neurons with neither are unclassified (<code>U</code>) — they are <b>always included</b> in every option (both / left / right), so data without hemisphere notation is never silently dropped.

  
### Hemisphere-aware

Splits type/group aggregation into <code>\_L</code> / <code>\_R</code> / <code>U</code> labels so ipsilateral and contralateral edges can be told apart (e.g. <code>KCg-m\_L → MBON01\_L</code> vs <code>KCg-m\_R → MBON01\_R</code>). Enables <b>Symmetry Analysis</b> and <b>Keep Only Hemisphere-Conserved Edges</b>.

  
### Hemisphere filter — only left / only right / both

Restricts the query to one hemisphere: <b>both</b> (default) keeps everything; <b>left</b> keeps left + unclassified neurons; <b>right</b> keeps right + unclassified neurons. At the edge level, a connection is kept only when <b>both</b> endpoints belong to the selected hemisphere (unclassified endpoints are always kept). This works for every dataset — unlike the manual <code>.\*\_R</code> instance search, which only finds neurons that carry a hemisphere suffix in their instance name.

  

<div class="callout tip">
  <b>Manual suffix search still works</b> — typing <code>aMe12_R</code> or
  <code>.*_R</code> as a source/target pattern selects one side per type;
  the Hemisphere selector applies the same rule to every neuron
  automatically, including those without a suffix.
</div>

  
## Output &amp; examples

One timestamped folder per run, named after the dataset abbreviation, query and parameters (<code>MCNS</code> = male-cns, <code>FAFB</code> = flywire FAFB, ...):

  

<pre><code>{output_dir}/find-paths-complete_MCNS_aMe12_to_PPL101_L1w3_20260815_142520/
aMe12_to_PPL101_allpaths_type.csv  # paths (path, weights, ratios,
                                   #   min_weight, min_ratio, ...)
source_neurons.csv, target_neurons.csv # resolved enrollment/status details
all_attributes.json
parameters.txt, user_warning_notes.txt
data_details/
  conn_mat_type_weight|ratio|prob|nt.csv   # type-level matrices
  connection_type.csv                      # edges per type pair
  neurons_included.csv
  total_weight_layer.csv, parameters.csv
visualization/
  Network_*.html, Heatmap_*.html, Sankey_*.html
  <span class="small">(all three work fully offline — their plotly /
  cytoscape libraries are embedded in the files)</span>
  visualization_data/                     # data backing the HTML files
hemisphere_symmetry/...   # when Symmetry Analysis is on
find_reciprocal/...       # when Find Reciprocal Connections is on</code></pre>

<div class="example">
  <div class="example-title">Example folders</div>

<pre><code>find-paths-complete_MCNS_aMe12_to_PPL101_L1w3_20260815_142520/  &lt;!-- Complete Paths --&gt;
find-paths-shortest_MCNS_aMe12_to_PPL101_L1w3_20260815_142717/  &lt;!-- Shortest Paths --&gt;</code></pre>

</div>
<p class="small">With <b>Skip BodyId</b> off, bodyId-level tables
    (<code>..._allpaths_bodyId.csv</code>) are written next to the type-level
ones; with Output Format = xlsx, the tables become multi-sheet Excel
workbooks. The full reference is in <code>docs/OUTPUT_FILES.md</code>.</p>
<p class="small">Every run also writes an <b>applied-threshold
provenance</b> block to <code>parameters.txt</code>,
<code>all_attributes.json</code> and
<code>data_details/parameters.csv</code>: requested vs
<code>applied_threshold</code>, <code>applied_threshold_source</code>,
the StrongestFirst landing tau (<code>strongest_first_tau</code>),
<code>tau_canonical</code>, <code>strongest_dropped_bottleneck</code>
    (w2) / <code>strongest_retained_bottleneck</code> (W*), and the
Edge-Budget fields (<code>edge_budget</code>,
<code>edge_budget_landing</code>, <code>edge_weight_floor</code>) — so a
budgeted run states the exact strength cutoff its outputs represent.</p>

  
## Pair report (automatic)

Every run writes a per-source-target-pair report INTO its run folder when it completes: <code>path\_report.html</code> (Overview / Global / Pair Explorer / Data, with interactive offline networks and a per-pair Source/Target bodyId coverage columns, a per-length network tab strip; multi-selections render one union pane) plus two lossless CSVs under <code>paths\_pair\_breakdown/</code>. Every capped view in the report (top-10 paths per pair and length, 50 matrix rows) states its cap and links to the uncapped CSVs. Existing folders can (re-)generate it via <code>scripts/PathsPairReport.py &lt;run\_folder&gt;</code>; see <code>docs/OUTPUT\_FILES.md</code> §1 "Pair Report".

  
## Common issues &amp; tips

  

<div class="callout tip">
  <b>No paths found</b> - raise Max Intermediate Layers, lower Min Synapse,
  or check that targets exist in the dataset (log shows target counts).
</div>

<div class="callout warn">
  <b>Slow runs</b> - large networks (100k+ neurons) take minutes. Keep the
  query small first, then expand.
</div>

<div class="callout info">
  <b>Repeated runs</b> - the local cache gives 10-100x speedups; the second
  run with the same dataset is much faster.
</div>

<div class="callout info">
  <b>Exported run guide:</b> every successful run also writes
  <code>_UserGuide_please_read_me.html</code> into the run folder — a
  self-contained explanation of every output file, every table column, and
  any warnings recorded for the run. Format (html / txt / markdown /
  disabled) is set in Settings → Default Settings → Run Guide Format.
</div>
