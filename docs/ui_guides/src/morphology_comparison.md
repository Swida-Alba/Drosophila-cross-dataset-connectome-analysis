---
title: Comparison
tag: Connectome
subtag: Morphology tab
page_title: Morphology · Comparison
---

Compare the morphology of already-identified neurons <b>within one dataset</b>: enter custom group labels, neurons, types, taxonomy labels, or patterns and get the N×N similarity matrix at the bodyId level plus a type-level aggregation, with interactive heatmaps and a summary report.

<div class="callout warn">
  <b>One dataset = intra-dataset.</b> Skeletons are scored in one
  dataset's coordinate space against that dataset's caches. Select
  <b>two or more datasets</b> and the run keeps every selected volume's
  intra comparison <i>and</i> adds the
  <a href="cross_dataset_morphology.html">cross-dataset comparison</a>
  (queried neurons across FAFB / male-cns / BANC, null-baselined
  vector_v2) — all in <b>one run folder</b>, the same multi-dataset contract
  as Connectivity ▸ Comparison. The intra controls (Method, Aggregation
  Level, caps, cache) stay visible in multi-dataset mode and drive the intra
  half; the cross phase always uses native type rows.
</div>


## Quick start

1. Pick the <b>Datasets to compare</b> (multi-select; BANC runs with an explicit caveat — the public L2/full skeleton products' vector quality is still unvalidated, so treat BANC scores as provisional and spot-check surprising pairs in the 3D scene; BANC 3D visualization and pathfinding work).
2. Enter <b>one or more</b> query tokens: custom group labels from the grouping board, exact types, bodyIds, taxonomy labels (e.g. FAFB's <code>circadian_clock</code>), instance names, or patterns (e.g. <code>aMe.*</code>) — or pick <b>custom group</b> as the Aggregation Level and let the board's groups fill the rows directly.
3. Choose the <b>Aggregation Level</b> (type / bodyid / custom group) to decide what one matrix row is. The <b>Custom Groups (LabelMapper)</b> card is always visible — in <b>custom group</b> mode it fills the rows; in <b>type</b>/<b>bodyid</b> mode a loaded preset both expands a query chip that names one of its labels and acts as the tier-0 label overlay (see below).
4. Click <b>Run Comparison</b>; open <code>report.html</code> in the output folder.

<div class="callout info">
  <b>The gate is neurons, not rows.</b> A run needs at least <b>two
  neurons</b> in scope, so <b>one type</b> is a legitimate comparison: its
  aggregate cell is that type's own cohesion (the mean pairwise score among
  its members, not 1.0) and the bodyId matrix carries the pairwise detail.
  One query that resolves to a single neuron is refused, with the message
  naming what collapsed it — the query itself, or
  <b>Max Members per Type</b> being below 2 at the bodyId level.
</div>

<div class="callout info">
  <b>How each query token resolves.</b> Tokens resolve in the same order
  the Connectivity ▸ Comparison tab uses: <b>custom group label</b> (a name
  the grouping board defines resolves to that group's members in the
  selected dataset, and wins over a same-spelling native type — the priority
  every other lane already gives a label) → exact type name → bodyId →
  <b>taxonomy label</b> (a cell_type / class value such as FAFB's
  <code>circadian_clock</code> expands into one row per member type) →
  <b>instance name</b> (those neurons are pinned, like bodyId inputs) →
  pattern. Every expansion, pin, cap, and unmatched token is disclosed in
  the run folder's <code>user_warning_notes.txt</code> and in the run log,
  so a query that resolved through an expansion never looks like a
  literal-type run.
</div>


## Aggregation Level — what one matrix row is

- <b>type</b> (default): each queried type is one row; a bodyId input resolves to its type. The bodyId-level matrix is written alongside.
- <b>bodyid</b>: every individual neuron is its own row, so two neurons of the same type can be compared — the thing the type level folds away. Here the bodyId matrix <i>is</i> the comparison, so no type-level matrix or heatmap is produced, and the report carries no tab for it — a tab that can only ever read "not computed" invites the reader to look for a missing input rather than a deliberate level. Not offered while the method is NBLAST (see below).
- <b>custom group</b>: rows are the source-side groups of the grouping board (a saved preset or groups defined inline), resolved against the selected dataset; the neuron query above is disabled, not silently ignored. Same board and preset format as Connectivity ▸ Comparison. The aggregate matrix is written as <code>group_level/group_similarity\_{method}.csv</code> with <code>heatmap_group_*.html</code>, because its axes are groups, not types.


## Custom Groups (LabelMapper) — the tier-0 label overlay

The grouping card is visible in <b>every</b> mode, and a loaded preset (saved history or inline board — the same board and preset format as Connectivity ▸ Comparison) acts as the <b>tier-0 identity overlay</b> beyond the custom-group rows:

- a query chip naming a label resolves to that label's members for the dataset — the preset <i>is</i> the mapping;
- label identities are <b>never re-resolved</b> through the auto type mapper;
- bodyId axes read <code>{bodyId}\_{label}({native})</code> on resolved ids (see above for the dedup rule);
- <code>members.csv</code> gains a <code>label</code> column and <code>parameters.json</code> echoes the preset path.

See the <a href="label_mapper.html">label-mapper guide</a> for the preset format and the display role across the whole comparison family.


## How the comparison works

- The <b>bodyId-level matrix</b> is always the scored primitive: every individual pair, labelled <code>{bodyId}\_{instance}</code> (or <code>{bodyId}\_{type}_L/_R</code> on FAFB). With a LabelMapper preset loaded, resolved bodyIds read <code>{bodyId}\_{label}({native})</code> — the parenthetical is dropped when the label already is the native name (<code>{bodyId}\_{label}</code>, or <code>{bodyId}\_{label}\_{L|R}</code> when only the side differs); unmapped members keep the legacy form.
- The <b>aggregate matrix</b> (type or group level) is derived from it: each entry is the mean over the cross-member bodyId pairs, and the <b>diagonal is that row's intra-type/group cohesion</b> (mean pairwise among its own members, 1.0 for a single-member row).
- A neuron you queried <b>by bodyId is never dropped</b> by <i>Max Members per Type</i> or <i>Max Total Neurons</i>; those caps only remove other members. <code>members.csv</code> lists every compared neuron with the <code>row</code> it fed, its real <code>type</code>, and — when a preset is loaded — the <code>label</code> that bodyId resolves to (a native type can settle to two labels; a bodyId resolves to exactly one).


## Methods

<table>
  <tr><th>Method</th><th>Cost</th><th>Notes</th></tr>
  <tr><td><b>Vector (spatial)</b> — default</td><td>Fast (cached vectors)</td>
    <td>The Find Similar production score: per-block cosine (shape/spatial
      0.30/0.70) on standardized + ZCA-whitened vectors from the dataset's
      vector cache. Missing members are <b>fetched online by default</b> —
      NeuPrint raw SWC via the shared batch fetch, FAFB via the healed
      bundle → CAVE fallback — and persist into the shared skeleton +
      vector caches. Mirror twins are legitimate evidence (lateral
      normalization).</td></tr>
  <tr><td><b>NBLAST</b></td><td>Slow — warns on populations over <b>30
    neurons</b>, never refuses</td>
    <td>Canonical normalized NBLAST on raw-skeleton dotprops; both
      orientations of every pair are scored and averaged. Large populations
      are disclosed as slow in the run log and
      <code>user_warning_notes.txt</code> rather than gated. Aggregate means
      exclude contralateral pairs (mirror arbors score at chance); the
      bodyId matrix keeps every pair. Because every neuron enters scoring,
      the Aggregation Level drops <b>bodyid</b> while NBLAST is selected —
      there each row is one neuron, so a realistic query could not
      complete.</td></tr>
  <tr><td><b>Both</b></td><td>Vector cost + NBLAST cost</td>
    <td>One run, one folder, both scorer matrices
      (<code>*_similarity_vector_v2.csv</code> and
      <code>*_similarity_nblast.csv</code>) with two cards per level in the
      report and one heatmap per level and method. The morphology analogue
      of the connectivity comparison's metric set; <b>bodyid</b> is not
      offered because the NBLAST half would not complete.</td></tr>
</table>


## Parameters

<table>
  <tr><th>Parameter</th><th>Default</th><th>Meaning</th></tr>
  <tr><td>Aggregation Level</td><td>type</td><td>What one INTRA matrix row is:
    a <b>type</b> (bodyId inputs resolve to their type), an individual
    <b>neuron</b> (bodyid), or a <b>custom group</b> from the grouping board
    (its query box is disabled). In a multi-dataset run this level is used
    for every selected volume's intra matrices; the cross phase always keeps
    native type rows, because bodyIds are not comparable across datasets</td></tr>
  <tr><td>Max Members per Type</td><td>25</td><td>Members sampled per type
    for the pairwise scores (large types are truncated; the exact member
    list is written to <code>members.csv</code>). At the bodyId level it
    bounds the rows each queried type contributes; at the custom group
    level, the members a group keeps. Neurons queried by bodyId are never
    dropped by it</td></tr>
  <tr><td>Max Total Neurons</td><td>200</td><td>Safety cap on the compared
    population; truncation walks the rows in order and never drops a neuron
    you queried by bodyId. In a multi-dataset run it is applied <b>once</b>
    to the shared population, so the intra and cross phases compare exactly
    the same neurons. NBLAST scores every pair twice, so populations
    past 30 neurons (the size that will actually be scored — the cap, or
    more when your queried bodyIds alone exceed it) <b>warn</b> that the run
    may take very long — they are never refused</td></tr>
  <tr><td>Fetch Missing Skeletons Online<br><span class="small">(Advanced Settings)</span></td><td>on</td><td>Pull
    skeletons for cache misses through the API into the shared cache
    (turn off for a strictly offline comparison)</td></tr>
  <tr><td>Use Cache<br><span class="small">(Advanced Settings)</span></td><td>on</td><td>Serve
    neuron vectors from the dataset's local cache instead of recomputing
    them — the same switch the Connectivity ▸ Comparison card offers, so
    neither tab hides a cache decision behind a default. The cross-dataset
    comparison always uses the cache</td></tr>
  <tr><td>Generate Heatmaps</td><td>on</td><td>Interactive (VisPath)
    heatmaps, one per computed level</td></tr>
  <tr><td>3D Skeleton Visualization</td><td>off</td><td>Render the
    compared neurons as one skeleton layer per compared matrix row (line
    rendering by default; Advanced Visualization exposes the full render
    settings shared with Find Similar). With <b>Fetch Missing Skeletons
    Online</b> off, only members whose skeletons are already cached are
    rendered — the rest are skipped with a note instead of triggering an
    online fetch</td></tr>
  <tr><td>Show Figures</td><td>off</td><td>Open the heatmaps in the
    browser as they are generated</td></tr>
</table>


## 3D skeleton scene

With <b>3D Skeleton Visualization</b> enabled, the run also renders the compared population into <code>plot-3d\_{dataset}/</code> inside the run folder: one skeleton layer per compared matrix row (capped at 20 members per row for the render, with an in-page note when a layer is sampled), line skeletons on a template brain, no synapses. The scene is linked from <code>report.html</code>, and the legend tree lists every neuron leaf.


## Output

<pre><code>{output_dir}/morphology_comparison_MCNS_aMe12_aMe10_20260901_120000/
parameters.json
README.txt
members.csv                              # row/type/bodyId + status
user_warning_notes.txt                   # query-resolution disclosures (when present)
type_level/type_similarity_{method}.csv   # type×type — type level only
group_level/group_similarity_{method}.csv # group×group — custom group level
bodyid_level/bodyid_similarity_{method}.csv  # bodyId×bodyId matrix
visualization/heatmap_type_{method}.html     # or heatmap_group_… / heatmap_bodyid_…
visualization/heatmap_bodyid_{method}.html
plot-3d_{dataset}/                       # optional 3D skeleton scene
report.html</code></pre>

With two or more datasets the run writes one folder with both halves:

<pre><code>{output_dir}/morphology_comparison_MCNS_FAFB_BANC_aMe12_aMe10_20260901_120000/
intra_dataset/MCNS/                      # one self-contained intra run per volume
intra_dataset/FAFB/                      #   (both methods side by side when Method = Both)
intra_dataset/BANCv888/
cross_dataset/<SRC>_to_<TGT>/results/    # ordered pairs, null baselines
cross_dataset/overview.csv
cross_dataset/members_summary.csv
cross_dataset/report.html
report.html                              # hub linking both halves + key tables
parameters.json                          # combined provenance
README.txt
user_warning_notes.txt</code></pre>
<code>report.html</code> uses the same tabbed generator as the connectivity-profiling and cross-dataset morphology reports (shared <code>report_kit</code>): a hero header, Type/Group-level and BodyId-level tabs with Ward-clustered heatmap cards (hover for exact values, CSV and VisPath editor links), and compared-neuron / parameter details. With <b>Both</b> selected each level shows one card per method (vector_v2 and NBLAST) in the same tab. Every heatmap card's <b>cells render square</b> (1:1, at any window size) — the card itself is not squared, only the matrix cells. Plotly.js is embedded, so the report renders offline. Both similarity metrics span <b>[-1, 1]</b> — negative means less similar than chance (a whitened cosine and normalized NBLAST both genuinely score below zero) — so every heatmap renders on the diverging blue–white–red scale over the full [-1, 1] domain, whether or not a run happens to contain negative cells.

BodyId-level rows and axes are labeled <code>{bodyId}\_{instance}</code> (NeuPrint-style datasets, e.g. <code>11309_aMe4_L</code>) or <code>{bodyId}\_{type}_L/_R</code> (FAFB/BANC) — the same labels the 3D legend tree uses — so no export reads as a bare numeric id. <code>members.csv</code> maps every label back to its raw bodyId.

Neurons that stay unavailable even after the online fetch (no skeleton at the source) are reported in <code>members.csv</code> (<code>no vector</code> / <code>no dotprops</code>) and carry empty matrix cells instead of silently biasing the averages.

<div class="callout info">
  <b>Exported run guide:</b> every successful run also writes
  <code>_UserGuide_please_read_me.html</code> into the run folder — a
  self-contained explanation of every output file, every table column, and
  any warnings recorded for the run. Format (html / txt / markdown /
  disabled) is set in Settings → Default Settings → Run Guide Format.
</div>
