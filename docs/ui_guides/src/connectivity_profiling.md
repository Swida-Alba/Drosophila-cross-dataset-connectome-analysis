---
title: Comparison
tag: Connectome
subtag: Connectivity tab
page_title: Connectivity · Comparison
---

Compare connectivity profiles of neuron types (or bodyIds / custom groups) using identical metrics — Jaccard, weighted Jaccard, cosine, raw rank correlation and rank-correlation union (same metric set as the sibling Find Homolog sub-tab) — within one dataset or across several datasets at once.


## Quick start

1. Enter one or more neurons (e.g. <code>aMe12</code>, <code>aMe10</code>, <code>aMe9</code>). The gate counts <b>neurons in scope, not matrix rows</b>: a single type is a legitimate run because its bodyId matrices compare its own neurons.
2. Pick one or more datasets (one dataset with multiple thresholds is also supported; two or more datasets enable the cross-dataset mode).
3. Click <b>Run Comparison</b>; open <b>report.html</b> in the output folder — the hub with the inter-dataset overview and link cards. Multi-dataset runs also write one <code>report_&lt;dataset&gt;.html</code> per dataset carrying that dataset's full heatmap set.


## Inputs &amp; examples

<table>
  <tr><th>Query</th><th>Meaning</th></tr>
  <tr><td><code>aMe12, aMe10, aMe9</code></td><td>Three types compared pairwise</td></tr>
  <tr><td><code>720575940610453042, aMe12</code></td><td>BodyId vs type (mixed allowed)</td></tr>
  <tr><td><code>KC.*</code> (Match by = Starts with)</td><td>All KC types compared — under <b>Aggregation Level = type</b> each matched KC type becomes its own independent row; under <b>custom group</b> the board's groups fill the rows and the neuron query above is <b>disabled</b> (an empty query box is not a missing input in that mode)</td></tr>
</table>


## Multi-dataset (cross-dataset) mode

Selecting <b>two or more datasets</b> profiles the SAME query in every dataset and compares it on two levels:

- <b>Intra-dataset</b> — N×N similarity of the queried types/bodyIds/ groups within each dataset (per-dataset folder).
- <b>Inter-dataset</b> — datasets × datasets similarity of the SAME queried neuron across datasets, using the homolog-finding backend algorithm (2-hop expanded partner types).

Different names of the same neuron across datasets are handled automatically via the <b>male-cns v1.0</b> neuron info (the type mapper), e.g. FAFB <code>MTe07</code> is standardized to <code>MeVPLo2</code>. <code>cross_dataset/mapping_summary.csv</code> lists the resolved name of every queried neuron per dataset and adds a numeric <code>same name</code> flag (1 when every dataset has the same non-empty name, otherwise 0).

An inter-dataset comparison exists only for a queried item that has a profile row in <b>at least two</b> datasets — that item is an <i>anchor</i>. Names resolve <b>exactly first</b>: with a LabelMapper preset loaded (<code>label_mapping_file</code>) a preset label is a literal, so a label containing <code>+</code>, <code>.</code> or <code>(</code> keeps its own row instead of being expanded into the family its spelling happens to match; only an item with no exact row anywhere is treated as a pattern. An item that forms no anchor is never dropped silently: it is listed in <code>parameters.json</code> under <code>dropped_inter_anchors</code> (with its kind, the datasets that did hold a row, and the reason), repeated as a WARNING block in <code>README.txt</code>, and shown as a chip in the report, so a partial cross-dataset matrix cannot be mistaken for a complete one.


## Profile construction

<table>
  <tr><th>Parameter</th><th>Default</th><th>Meaning</th></tr>
  <tr><td>Top K Partners</td><td>25</td><td>Top partners per direction by synapse weight</td></tr>
  <tr><td>Min Unique Types (M)</td><td>5</td><td>Minimum partner types; K expands automatically if fewer</td></tr>
  <tr><td>Min Synapse Threshold</td><td>3</td><td>Connection weight cutoff for profiles</td></tr>
  <tr><td>Aggregation Level<br><span class="small">(Profile Construction card)</span></td><td>type</td><td><b>type</b> (each matched type is one row; patterns expand to independent types), <b>bodyid</b> (every individual neuron is one row: nothing is pooled, so its matrices are filed under <code>bodyid_level/</code> and no <code>type_level/</code> folder or report tab is written), or <b>custom group</b> (rows from the <b>Custom Groups (LabelMapper)</b> card right below — define groups inline or reload them from history; see the <a href="label_mapper.html">label-mapper guide</a>; their matrix is filed as <code>group_level/group_similarity_*</code>, because its axes are groups). A one-dataset run at the bodyId level also gets <code>type_avg_bodyid_*</code>, folded from the same pair scores, so a per-type view is never lost.</td></tr>
  <tr><td>Custom Groups (LabelMapper)<br><span class="small">+ Apply Groups to Partner Keys</span></td><td>—</td><td>The grouping card is <b>always visible</b>, in every aggregation mode. In <b>custom group</b> mode each label is one comparison row and the neuron query is disabled. In <b>type</b>/<b>bodyid</b> modes the labels act as <b>tier-0 user labels</b>: a query chip naming a label resolves to that label's members per dataset, and label identities are never re-resolved through the auto type mapper. <b>Apply Groups to Partner Keys</b> additionally collapses the partner side of every profile through the groups (weights summed per label) so similarity is computed over the group feature space; bodyId-level matrices keep raw keys. Runs disclose the state in the report header (<span class="small">User labels N (tier 0) · Partner grouping ON</span>).</td></tr>
  <tr><td>Direction</td><td>both</td><td>upstream, downstream, or both</td></tr>
</table>


## Similarity metric formulas

Let A and B be two connectivity profiles (a dict of partner type → weight, restricted to top-K partners per direction), and let ρ denote the Spearman rank correlation of two weight vectors. The metrics are the same set used by the Find Homolog sub-tab; rank correlations are raw (−1 to 1) and there is no normalized 0–1 variant and no <code>combined</code> score. The <code>overall</code> matrix averages upstream and downstream connectivity.

<table>
  <tr><th>Metric</th><th>Formula</th><th>Range</th></tr>
  <tr><td><code>jaccard</code></td><td>|A ∩ B| / |A ∪ B| (partner-type sets)</td><td>0–1</td></tr>
  <tr><td><code>weighted_jaccard</code></td><td>Σ min(w_a, w_b) / Σ max(w_a, w_b) over the union (Ruzicka)</td><td>0–1</td></tr>
  <tr><td><code>cosine</code></td><td>(A·B) / (‖A‖·‖B‖) over the union (missing = 0)</td><td>0–1</td></tr>
  <tr><td><code>rank_corr</code></td><td>ρ = Spearman on <b>shared</b> partners (raw, NaN if &lt; 3 shared)</td><td>−1 to 1</td></tr>
  <tr><td><code>rank_union</code></td><td>ρ over the <b>union</b> of both partner sets (missing = 0)</td><td>−1 to 1</td></tr>
</table>
Diagonal (self) entries are 1.0 for every metric. Rank metrics return <code>NaN</code> for pairs with fewer than 3 comparable partners; those cells are rendered as 0 in the heatmaps while the CSV matrices keep the original values. In multi-dataset mode the cross-dataset matrices use the homolog-finding algorithm (2-hop expanded partner types) and rank correlation values are reported raw (−1 to 1).


## Advanced options

- <b>BodyId-Level Computation</b>: auto (skip over 1000 bodyIds), skip, or compute always. A run left with a <b>single</b> comparison row keeps the bodyId pass whatever this says — its pooled cell is that row pooled and compared with itself, i.e. 1.0 on every metric, so the bodyId matrices are the only information the run can carry. That override stops past the 1000-bodyId budget, because the pair loop is quadratic: one coarse label can hold tens of thousands of neurons, and the log then says to narrow the query or force <code>compute</code>.
- <b>Generate Heatmaps</b>: hierarchical (Ward) reordering of matrices, written as editable VisPath heatmaps plus Plotly heatmaps in the report. The heatmap pages draw dendrograms of the clustering (🌳 button; tree line-width and band-size sliders); the report's heatmap cards embed mini-dendrograms of the same ordering.
- <b>Show Figures</b>: open heatmaps in the browser.
- <b>Pre-build Full Dataset Cache</b>: OFF by default - leave it off; the tool uses connections already cached. Turning it on fetches every uncached neuron and can take hours.


## Output

The folder name uses the first dataset's abbreviation (<code>MCNS</code> = male-cns, <code>FAFB</code> = flywire FAFB, ...) and a sanitized query name (e.g. <code>aMe.*</code> → <code>aMe</code>):

<pre><code>{output_dir}/profiling_MCNS_aMe_20260815_143922/
report.html                        # hub: inter-dataset section + link cards
report_<dataset>.html              # full heatmap report per dataset (2+ datasets)
plotly.min.js                      # shared chart library — keep beside the reports
parameters.json, README.txt        # README lists only the folders written;
                                   # parameters.json also carries
                                   # dropped_inter_anchors (2+ datasets)
# ---- ONE dataset: level folders appear only for levels that ran ----
type_level/                        # type level only (custom groups write
  results/type_similarity_{metric}_{direction}.csv
  visualization/heatmap_type_{direction}_{metric}.html
group_level/                       # custom group level: same matrix, named
  results/group_similarity_{metric}_{direction}.csv   # for its group axes
  visualization/heatmap_group_{direction}_{metric}.html
bodyid_level/                      # every level except a skipped bodyId pass
  results/bodyid_similarity_{metric}_{direction}.csv
  results/type_avg_bodyid_similarity_{metric}_{direction}.csv
  visualization/heatmap_bodyid_*.html, heatmap_type_avg_*.html
profiles/
  individual/{bodyId}_{type}_profile.json
  aggregated/{type}_profile.json   # absent at the bodyId level (nothing pooled)
# ---- two or more datasets ----
intra_dataset/{dataset_folder}/    # per-dataset N×N comparisons
  results/similarity_{direction}_{metric}.csv             # type level
  results/bodyid_similarity_{metric}_{direction}.csv      # bodyId level
  results/type_avg_bodyid_similarity_{metric}_{direction}.csv
  visualization/heatmap_intra_*.html, heatmap_intra_type_avg_*.html
cross_dataset/                     # same neuron across datasets (2+ datasets)
  mapping_summary.csv              # resolved names + same-name flag (anchors
                                   # only — see dropped_inter_anchors)
  all_types/
    results/similarity_{direction}_{metric}.csv
    visualization/heatmap_inter_all_types_*.html
profiles/{dataset_folder}/
  aggregated/{type}_profile.json
  individual/{bodyId}_{type}_profile.json</code></pre>
BodyId-level rows and axes carry the tree-legend suffixes: <code>{bodyId}\_{instance}</code> (NeuPrint-style datasets) or <code>{bodyId}\_{type}_L/_R</code> (FAFB/BANC), falling back to the resolved comparison type when the neuron table has no metadata — no bodyId-level export reads as a bare numeric id. Queried bodyIds in the cross-dataset matrices carry their resolved type in the row/column labels. Individual profile JSONs include the neuron's <code>instance</code> alongside <code>bodyId</code>, <code>type</code>, and <code>dataset</code>.

<div class="callout warn">
  <b>First run on a new dataset</b> - profile extraction needs connection
  data. If the cache is empty the tool may take a long time or refuse the run
  for want of two neurons in scope; run a small Find Path/Direct query first
  to warm the cache.
</div>

<div class="callout info">
  <b>Exported run guide:</b> every successful run also writes
  <code>_UserGuide_please_read_me.html</code> into the run folder — a
  self-contained explanation of every output file, every table column, and
  any warnings recorded for the run. Format (html / txt / markdown /
  disabled) is set in Settings → Default Settings → Run Guide Format.
</div>
