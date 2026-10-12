---
title: Find Similar Neurons
tag: Connectome
subtag: Morphology tab
page_title: Morphology · Find Similar
---

Search for neurons similar to a query by <b>morphology</b> (skeleton shape) within one dataset. This is the <b>Find Similar</b> sub-tab of the <a href="morphology.html">Morphology</a> tab; it is <b>intra-dataset only</b>. For connectivity-based similar search (including cross-dataset homolog search), use the <a href="find_homologs.html">Connectivity tab's Find Homolog sub-tab</a>.

  
## Morphological similarity

Two methods are available. Find Similar uses raw skeletons only. Raw skeletons are shared at <code>cache/{dataset}/skeletons/raw\_skeletons/</code>; Find Similar's vector parquet remains under <code>cache/{dataset}/find\_similar/</code>. The visualization simp90 cache is never used for comparison.

  

<table>
  <tr><th>Method</th><th>Cost</th><th>Notes</th></tr>
  <tr><td><b>Vector (spatial)</b> (recommended)</td><td>&lt;10 ms per query</td>
    <td>Each neuron is reduced to a 256-dim shape + spatial feature
      vector. Scores are per-block cosine (0.30/0.70) on standardized +
      ZCA-whitened vectors, cached at
      <code>find_similar/morphology/skeleton__vectors_v2.parquet</code>
      (the vector_v2 scoring cache; its standardization stats and ZCA
      whitening live in <code>meta_v2.json</code> /
      <code>whiten_v2.npz</code> beside it); if a
      row is missing it is fetched online and vectorized.</td></tr>
  <tr><td><b>NBLAST</b></td><td>~5–15 s for 100 prefiltered candidates</td>
    <td>Canonical Costa et al. 2016 pairwise score via navis on raw
      skeletons only. Dotprops are rebuilt per query and never cached —
      no dotprops file is ever written to disk.</td></tr>
</table>

Dotprops were benchmarked on the 8,839-node raw aMe12 skeleton: building them with <code>k=20</code> took about 28 ms, while a compressed serialized dotprops object was about 144 KB. A compressed raw skeleton was about 48 KB in the same benchmark, so dotprops caching would use more space while saving only tens of milliseconds per neuron. It is not enabled by default. NBLAST score caching was evaluated separately. A final pair-score cache would need a dataset revision, ordered bodyId pair, navis/NBLAST version, dotprop parameters, scale and scoring options in its key. In practice the reusable cost is raw skeleton/vector preparation; the current design keeps those assets and recomputes dotprops/scores so stale scores cannot survive a dataset or parameter change.

  
## Candidate source

<b>Level</b> controls the result granularity. <b>auto</b> (default) follows the query kind: a <b>type query returns type-to-type</b> rows (each row = one target type), a <b>bodyId query returns bodyId-to-bodyId</b> rows. <code>bodyid</code>/<code>type</code> force a level for multi-neuron or mixed queries.

<b>Candidate Source</b> selects how candidates are discovered before morphological scoring:

- <b>auto</b> (default): NeuPrint datasets (hemibrain, male-cns, ...) use the <b>ROI-distribution screen</b>: every neuron of the dataset is ranked by cosine similarity of its input/output synapse distributions over the dataset's <b>primary ROIs</b> (from <code>{dataset}\_allneurons\_roi\_count\_df.parquet</code>, or the legacy CSV from older pulls), comparing both orientations across the midline (bilateral mirror) so contralateral homologs match. This reaches <b>every</b> neuron — the connectivity screen below can never find neurons without shared partners (~8% of same-type members on male-cns). A one-time per-dataset matrix is cached (<code>cache/{dataset}/morphology/roi\_profiles.npz</code>); the top screen-ranked <b>types</b> (bounded by <b>Candidate Cap</b>) are kept and expanded to <b>all their member neurons</b>; raw skeletons are then fetched for the pool and shared for reuse. FAFB uses healed raw skeletons or the CAVE fallback. <b>BANC is not selectable for morphological find-similar</b>: the public L2/full skeletons still need vector-quality validation, so the feature is deferred (BANC 3D visualization and pathfinding work). Mesh and simp90 visualization files are never candidates for either method.
- <b>roi</b>: the ROI-distribution screen only. Requires the dataset's ROI-count table and metadata sidecar (missing sidecars are fetched and saved once automatically).
- <b>combined</b>: union of the ROI screen and the connectivity (shared-partner) screen — the widest candidate pool and the highest recall per fetched skeleton, at the cost of more transient skeleton fetches.
- <b>profile</b>: the connectivity screen only — candidates found directly in the connection cache (<code>cache/{dataset}/connections.parquet</code>: neurons sharing upstream/downstream partners with the query, ranked by shared-partner count).
- <b>cache</b>: full-morphology over the complete local raw skeleton population (download raw skeletons first — see below).

<b>Candidate Cap</b> (default 500) is the single knob bounding how many candidates are ever compared: the sorted candidate list of the selected screen is truncated to this many neurons, which become the scoring pool (it also serves as the NBLAST prefilter of <b>cache</b>-mode searches; pure <b>cache</b> vector searches compare the whole local population). Fetched raw skeletons are always persisted as compressed <code>.swc.zst</code> files in the shared dataset cache at <code>skeletons/raw\_skeletons/</code> for reuse by every workflow. The computed <b>raw vectors</b> are persisted into the vector cache — the lookup order is vector cache, shared raw skeleton file, then online fetch, and a fetched neuron is never vectorized twice. Already-cached raw skeletons are reused and the query neuron is fetched if missing. <b>ROI Filter</b> (shown when the dataset's connection cache carries synapse location data — male-cns has 114 ROIs like <code>AL(L)</code>, hemibrain has none) restricts the candidate rows of the connectivity screen (roi/combined discovery is dataset-wide by design). <b>results.csv and type\_summary.csv contain ALL compared candidates</b> (bounded by the cap), each row carrying the screen evidence — <code>roi\_similarity</code> (primary-ROI distribution cosine) and/or <code>profile\_similarity</code> (normalized shared-partner count) — alongside <code>similarity</code> (morphology); <b>Top N</b> applies to the optional 3D visualization only. If no candidates are found, the search falls back to the vector cache when available.

  
## Full-morphology mode &amp; skeleton download

Use <b>Settings → Dataset Cache → Download All Skeletons</b> to download a complete raw skeleton population. The button sits beside Pull Complete Connections, before Cancel. New bulk files use compressed <code>{bodyId}.swc.zst</code>; legacy <code>.swc.gz</code> and pickle files remain readable. Both the download and vector-cache update are resumable. After a raw download, build the vector cache and run with <b>Candidate Source = cache</b> to compare the query against the entire local skeleton population. The healed FAFB skeleton bundle remains a local raw source and is vectorized without copying its SWC entries.

  
## Skeleton vector caches

Two parquet caches live under <code>cache/{dataset}/find\_similar/morphology/</code>: <code>skeleton\_\_vectors\_v2.parquet</code> is the <b>vector\_v2 scoring cache</b> the Vector method scores from (with <code>meta\_v2.json</code> / <code>whiten\_v2.npz</code>), and <code>skeleton\_vectors.parquet</code> is the shared <b>pool-vector cache</b> (~1 KB/neuron, ~30–50x smaller than the raw skeletons) that feeds the connectivity-expanded candidate prefilter. Both build incrementally from already-cached skeletons (~15–85 ms/neuron) and are auto-triggered on the first query. Without a vector cache, the connectivity-expanded search z-scores the pool vectors with pool-computed statistics so cosine scores stay scale-fair.

  
## Similarity score formulas (morphology)

The three evidence columns are computed independently and are never mixed across candidate sources; the primary <code>similarity</code> column is the morphology score only.

  

<table>
  <tr><th>Score</th><th>Formula</th><th>Range</th></tr>
  <tr><td><code>similarity</code></td><td><b>vector_v2</b>: per-block cosine (shape/spatial, 0.30/0.70) on standardized + ZCA-whitened vectors (a whitened cosine — it can be negative, though real pairs are typically positive); <b>NBLAST</b>: normalized pair score</td><td>−1–1 for both (a whitened cosine and normalized NBLAST can both score below zero — less similar than chance)</td></tr>
  <tr><td><code>roi_similarity</code></td><td>ROI-distribution <b>screen</b> score (candidate selection only, never in the morphology score): cosine of input/output synapse-distribution vectors over the primary ROIs, mirrored across the midline</td><td>0–1</td></tr>
  <tr><td><code>profile_similarity</code></td><td>shared-count / max-shared-count (normalized shared-partner count)</td><td>0–1</td></tr>
  <tr><td><code>intra_type_similarity</code></td><td>mean pairwise similarity of the type's members (the intra-type reference)</td><td>0–1</td></tr>
</table>

For the vector method, each neuron is reduced to ~24 morphometrics plus a 100-dim persistence block; the pool is z-scored with pool-computed statistics so cosine scores stay scale-fair.

  
### Contralateral (mirror) handling

NBLAST matches arbors in absolute coordinates, so a neuron and its mirror twin sit far apart: contralateral same-type pairs score at chance (benchmark median &minus;0.03 vs 0.89 ipsilateral; partner reciprocal rank 0.13 vs 0.84). <b>NBLAST type means therefore use ipsilateral pairs only</b>: rows pairing opposite known sides (<code>somaSide</code>, or <code>hemisphere</code> on FlyWire) are excluded from <code>type\_summary.csv</code>, while <code>results.csv</code> keeps every pair row for inspection. Types whose only NBLAST evidence is contralateral are omitted from the type ranking, and <code>n\_bodyids</code> counts ipsilateral plus unknown-side members. The vector method is unaffected — it lateral-normalizes at vectorization, so mirror twins are legitimate evidence and both sides are aggregated.

  
## Intra-type reference data

Every search reports the query's intra-type similarity alongside the inter-type rows, so results are always comparable against a same-type baseline:

- <b>Type queries</b> return the query type itself as the first row (<code>is\_intra\_type=True</code>). Its <code>similarity</code> is the type's cohesion (mean score of its members against the query vector) and <code>intra\_type\_similarity</code> is the mean pairwise similarity of the type's members — the intra-type reference every other row is ranked against. <code>n\_bodyids</code> gives the member count (NBLAST: ipsilateral plus unknown-side members).
- <b>bodyId queries</b> tag same-type hits with <code>is\_same\_type=True</code> and carry the query type's <code>intra\_type\_similarity</code> on every result row.

In practice (hemibrain v1.2.1, verified 2026-08): aMe12's two members are each other's top hit (0.90 = the intra-type similarity), aMe5 bodyId queries return 10+ same-type members before any inter-type row, and sim(aMe12, aMe10) &gt; sim(aMe12, aMe5) in both directions. Note that single-cell types (e.g. SMP581) can tie or exceed the cohesion of morphologically diffuse types (aMe10, aMe5) — the intra row is a reference, not a guaranteed winner.

  
## 3D visualization of the found neurons

Find Similar can render the query and found neurons as interactive 3D skeleton views (same engine as the 3D Skeleton tab and the NeuronBridge FindNeuron visualization).

- <b>Visualize Top N Types / Neurons</b> (default 10; 0 = disabled): number of top morphology result types/rows to render.
- <b>Visualize By</b> (default <code>type</code>): <code>type</code> renders one layer per top type, containing all of its member skeletons found in the results (type-level queries resolve members from the vector cache, bounded by <b>BodyIds / Type</b>); <code>bodyId</code> renders one layer per top result neuron.

  

<div class="callout warning">
  Analysis-tab visualizations default to <b>Skeleton Mode = line</b> for
  speed. For high-quality morphology, open <b>Visualization → Skeleton</b>,
  or change the mode to <b>tube</b> and choose <b>fine</b> with 95%
  simplification.
</div>

When multiple queries are entered, the tab runs them sequentially and saves each query in its own run folder (keep the query to 50 neurons or fewer, and note the search works best on <b>male-cns:v1.0</b>). Morphological output is saved next to its results in <code>{run\_folder}/plot-3d\_{dataset\_folder}/</code>. Each scene starts with a query reference layer, followed by up to the requested top-N result layers. Each layer is colored per type and labeled <code>r{rank}\_{type}\_x{N}</code> (rank = similarity rank of the type, N = member count). The intra-type reference row is never rendered. The interactive HTML files are listed in the output panel; a visualization failure never discards the search results.

  
## Output

Each run is saved in its own timestamped folder with two always-written result tables:

  

<pre><code>{output_dir}/similar-morphology_flywire_FAFB_v783_aMe12_20260815_143601/
results.csv, type_summary.csv, README.txt</code></pre>

- <b>results.csv</b> always holds <b>bodyId-level</b> similarities (ranked <code>target\_bodyId</code> rows), whatever the query kind. A type query includes unique intra-type bodyId pairs for every resolved query member, followed by the top inter-type bodyId rows.
- <b>type\_summary.csv</b> always holds <b>type-level</b> similarities (one row per target type, the query type first as the intra-type reference with <code>is\_intra\_type=True</code>), whatever the query kind. Type searches rank the top-N types over the whole cached skeleton population (connection candidates + cached skeletons), so the result is not capped at the candidate types.

  

<div class="callout info">
  <b>Exported run guide:</b> every successful run also writes
  <code>_UserGuide_please_read_me.html</code> into the run folder — a
  self-contained explanation of every output file, every table column, and
  any warnings recorded for the run. Format (html / txt / markdown /
  disabled) is set in Settings → Default Settings → Run Guide Format.
</div>
