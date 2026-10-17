---
title: Settings
tag: System
tag_class: sys
---

Tokens, dataset availability, custom type mappings, default output directory, persistent tab defaults and dataset preparation.

  
  
## API tokens

  

<table>
  <tr><th>Token</th><th>Needed for</th><th>Where to get it</th></tr>
  <tr><td>NeuPrint Token</td><td>All NeuPrint datasets (hemibrain, male-cns, optic-lobe, manc, ...)</td><td>neuprint.janelia.org/account</td></tr>
  <tr><td>FlyWire Codex Token</td><td>Automatic FAFB data downloads (Settings &rarr; FAFB Dataset Downloads) and lazy per-neuron skeleton fetching</td><td>codex.flywire.ai/account (sign in with Google at the download page, then copy the token)</td></tr>
  <tr><td>CAVE Token (advanced)</td><td>Optional live-CAVE fallback only; no download or analysis needs it. Set via <code>config.json</code>/<code>config_local.json</code> or the <code>CAVE_TOKEN</code> env var — there is no Settings input for it (contributor-gated at codex.flywire.ai/auth_token)</td><td>config files / env only</td></tr>
</table>

Tokens are saved to <code>config\_local.json</code> (gitignored local override; template <code>config.example.json</code>) which also holds the versioned conda environment setting; the committed <code>config.json</code> is the clean fallback. For safety, the Settings page never pre-fills a saved secret into the browser. A <b>configured (kept hidden)</b> status means the saved value will be used; leave its field blank to keep it, or enter a new value to replace it.

Enable <b>Clear a saved value when its field is blank</b> only when you intentionally want to remove a token from the config file.

  

<div class="callout tip">
  <b>Test NeuPrint</b> verifies the token by counting neurons in
  hemibrain:v1.2.1.
</div>

  
## Dataset availability

Local dataset folders are shown as soon as the page opens. Click <b>Refresh</b> to re-check converted tables and query server metadata. Each dataset row shows its source badge (NeuPrint / FAFB / BANC), a neuron-count figure, and two status chip groups:

  

<table>
  <tr><th>Chip group</th><th>States</th><th>Meaning</th></tr>
  <tr><td>server</td><td>available / no token / timeout / unreachable / hidden / unknown</td><td>Result of the online metadata probe (token, reachability, or intentionally hidden from the selector)</td></tr>
  <tr><td>metadata · connectivity · viz</td><td>ready / on-demand / partial / missing</td><td>Local data readiness per layer: prepared tables, the connection cache, and visualization assets</td></tr>
</table>

  
## Dataset and connection cache

In <b>Dataset Cache</b>, choose a dataset and set the batch size and parallel worker count. <b>Pull Dataset Metadata</b> downloads/verifies only the neuron table, ROI table, and the materialized neuron index (<code>datasets/&lt;dataset&gt;/</code> + <code>neuron\_indexes/&lt;dataset&gt;/neuron\_index.parquet</code>); no connections are fetched, and already-present files are never re-downloaded. <b>Pull Complete Connections</b> fetches the full resumable connection cache (<code>cache/&lt;dataset&gt;/connections.parquet</code>), skipping neurons that are already cached (rows or verified zero-outdegree). <b>Download All Skeletons</b> sits below the dataset pull controls, right above its own progress bar, together with the <b>Cache Simplification</b> selector that chooses the percent of skeleton nodes removed when writing those cache files (0 = raw by default — the cache stores raw skeletons and simplification is applied at visualization time; higher values store pre-simplified trees), and a <b>Skeleton Batch Size</b> input (64 default). Progress is reported per completed skeleton, so the batch size does not affect progress granularity; larger batches only reduce the repeated per-batch metadata queries. The <b>Parallel workers</b> field above sets how many skeleton fetches run concurrently - raising it is the main speed lever (up to what the server tolerates). A running pull (dataset or skeletons) can be stopped with its <b>Cancel</b> button. FAFB is the one dataset <b>Download All Skeletons</b> cannot fetch: FAFB skeletons require the manual Codex download described in the preparation section below (BANC skeletons come from its public release bucket). The connection pull uses the batched builder and shared <code>cache/&lt;dataset&gt;/connections.parquet</code> cache, so it can resume an interrupted pull. Enable <b>Force rebuild</b> only when the existing connection cache needs to be replaced.

  
## Storage

The <b>Storage</b> card (below <b>Dataset Cache</b>) inspects and reclaims disk space. Nothing is removed automatically — the card scans on first open (and via <b>Scan Now</b>), and every removal previews the exact paths and bytes before you confirm. Actions are disabled while a dataset or skeleton pull is running.

<b>Scan coverage:</b> the scanner always covers the default output directory, the remembered per-tab output folders, every folder recorded under <b>Recorded output roots</b>, and the folders seen in the run history. A root is recorded automatically the first time a run exports to a non-default directory; type one more folder and press <b>Remember</b> to record it manually, and click the ✕ on a chip to stop scanning it (nothing is deleted by forgetting).

Every row shows a hover tooltip with the full path (cache rows list the exact files that a clear would remove), and the <b>folder\_open</b> button at the end of each row opens that folder in the system file manager.

<b>Caches</b> lists one row per cache class and dataset with its size, last-used time, and rebuild cost. Deleting the NB match cache goes through the finder's own clearer; the incoming-connections row removes the parquet <em>and</em> its completion-state file together (the state file can never be deleted alone, and is never left behind). Protected data — availability snapshots, the coverage snapshot, per-dataset <code>available\_rois.json</code>, and <code>cache/user\_mappings/</code> (your LabelMapper presets) — is never offered. BANC and FAFB connection caches rebuild offline from the local tables; NeuPrint ones refetch from the server. A clear takes full effect after an app restart (in-process caches may still serve cleared data until then). When <b>Cache Only (Offline)</b> is enabled and the selection contains connection data, a warning is shown — the clear still runs.

<b>Exported data</b> lists timestamped run folders under the default output directory (plus any folder typed into the extra-scan field). Registered NeuronBridge tools show a source-data vs deliverables split: <b>Prune Source Data</b> runs the Compact prune regardless of the runs' original output mode (deliverables stay, each folder's <code>cleanup\_audit.json</code> records the removals). <b>Delete Folder</b> removes whole run folders after a checkbox double-confirm. Folders belonging to an active run are locked. Removals of cache files and deleted folders are appended to <code>cache/storage\_audit.json</code>.

  
## Output settings

<b>Default Output Directory</b> is pre-filled in every panel; change it here and click <b>Save Default Directory</b>. Each run creates its own timestamped subfolder.

  
## App settings

Two toggles control the shared chip inputs (the analysis tabs, the mapping and layer panels, and the Advanced Layer Editor's neuron cells):

- <b>Input Auto-Suggestion</b> controls the type-ahead suggestions: dataset type / instance / bodyId names while typing. Disabling it restores plain chip inputs (the query-history list below is unaffected).
- <b>Show Query History</b> controls the Recent/Frequent list that appears when the field is empty. Disabling it hides the history list but keeps live type-ahead suggestions.

While a suggestion/history list is open, a pick adds its chip and then keeps that query's rows on screen, and the list marks them the same way: a name already in the input stays listed, tinted and ticked — so several entries can be handled from one list without retyping and the rows never shift under the pointer. Clicking a ticked row takes that chip back out, exactly like the chip's own <b>x</b>; on a history row the small <b>×</b> beside it is the other action, which drops the entry from the history and keeps the chip. Any click that closes the list (or ESC, or leaving the field) resets it to the history. <b>ArrowDown</b> moves into the open list and <b>ArrowUp</b> leaves it from the first row; <b>Enter</b> adds the highlighted entry — or removes it, when that row is ticked — and moves the highlight to the next one, so a run of picks works without the mouse. Both toggles are saved immediately and apply to every input.

  
## Default Settings

Tool-tab selections remain one-time: changing a control inside a tab applies only to that run. The <b>Default Settings</b> card defines the values every tab starts with. Adjust the controls, click <b>Save Defaults</b>, and the overrides are stored in <code>ui/local\_config.json</code> (under <code>user\_defaults</code>) so they survive restarts. <b>Reset</b> clears every override and restores the built-in values. Saved values are validated; an invalid entry in the config file falls back to the built-in default.

  

<table>
  <tr><th>Group</th><th>Examples</th><th>Wired into</th></tr>
  <tr><td>Dataset &amp; Search</td><td>Default Dataset, Similar-Search Target Dataset, Search Columns, Filter By</td><td>Dataset selectors in every tool tab, pathfinding and search tabs</td></tr>
  <tr><td>Connection Thresholds</td><td>Min Synapse Count, <b>Threshold Basis</b> (Synapse count / Connection ratio), <b>Combine Thresholds</b> (Any one / Both AND), Visualization Edge Limit. The single Min Connection Ratio / Min Traversal Probability FILTER entries are gone as global knobs (F9: they are readout columns) — but the ratio basis is a saved default like any other: here it chooses which basis a tab opens on, and the tab's <b>Min Connection Ratio (%)</b> field (typed in PERCENT, one shared conversion) is the threshold on a ratio run</td><td>Find Path, Find Shortest, Network, Cross-Dataset, Connectivity (Find Homolog + Comparison), Morphology</td></tr>
  <tr><td>Cache &amp; Data</td><td>Connection Cache (Use Cache), Cache Only, Skeleton Cache (Cache Neurons), Cache Synapses, Auto Type Mapping</td><td>Pathfinding tabs, similarity searches, skeleton rendering</td></tr>
  <tr><td>Pathfinding &amp; Output</td><td><b>Run Mode (Complete / Fast)</b> — the corridor lane switch, one control on all four pathfinding surfaces, whose <code>auto</code> fires at Max Intermediate Layers &ge; 4 (so the SAME saved default runs the lane on Shortest Paths, whose tab opens at depth 5, and not on Complete Paths at depth 2), Skip BodyId-Level Export, Output Format, Run Guide Format, Network Layout, Max Paths (BodyId) (path-output budget, reported tau), Edge Budget (graph-level floor, 'all' mode only), Drop Untyped Neurons (neuron-label filter, on by default), Show Figure, Max Intermediate Layers (the Pathfinding Algorithm entry was removed — StrongestFirst is built-in)</td><td>Pathfinding, Cross-Dataset and export surfaces</td></tr>
  <tr><td>3D Skeleton Rendering</td><td>Skeleton Mode, Simplification Method, Legend Mode, Background, Brain Mesh, Export Views</td><td>Visualization → Skeleton tab and the Advanced Visualization editor in analysis tabs</td></tr>
  <tr><td>Similarity &amp; Comparison Search</td><td>Top N/K/M, Similarity Metric, Fast Search, Vector Pre-filtering, Candidate Cap/Source, NeuronBridge controls</td><td>Connectivity (Find Homolog), Morphology (Find Similar), NeuronBridge tabs</td></tr>
</table>

<div class="callout tip">
  <b>Skeleton Cache default</b>
  Fetched skeletons persist as portable <code>.swc.zst</code> files in the
  shared dataset cache, so <b>Cache Neurons is on by default</b> on every
  skeleton surface (Visualization → Skeleton and the analysis tabs'
  visualization editors). Turning it off for a session is fine; saving a
  value in Default Settings pins the checkbox and disables the automatic
  on-default.
</div>

<div class="callout info">
  <b>Run Guide Format</b> (Pathfinding &amp; Output group) controls the
  exported per-run user guide: every successful run writes
  <code>_UserGuide_please_read_me</code> into its run folder, explaining
  each output file, each table column and any recorded warnings. Choose
  <b>html</b> (default), <b>txt</b> or <b>markdown</b> — or
  <b>disabled</b> to skip writing it. The environment variable
  <code>DROCAT_RUN_GUIDE_FORMAT</code> overrides this setting.
</div>

  
## Custom Type Mappings

The <b>Custom Type Mappings</b> card holds the same group editor the tool tabs mount as <b>Custom Mapping</b> / <b>Custom Grouping</b>: pick the <b>Target datasets</b>, then build aligned custom groups on the board (one row per label, one member input per selected dataset — see the <a href="label_mapper.html">label-mapper guide</a> for the editor, the history pulldown and the file format). The card adds an outlined <b>Save Mapping</b> action: it writes the stable <code>Custom Mapping</code> preset to <code>cache/user\_mappings.json</code> and exports it to <code>cache/user\_mappings/</code>. Saving does <b>not</b> attach the preset to future runs — no tab auto-loads saved presets; reload the exported JSON onto any board with the editor's <b>Load…</b> button.

  
## Dataset preparation

  

<div class="callout info">
  <b>Use the converter's directory contract exactly</b>
  NeuPrint tables are fetched automatically. FAFB is prepared from raw
  Codex downloads placed in <code>datasets/&lt;dataset&gt;/downloads/</code>;
  BANC metadata and connections are prepared from its public release
  products. Do not rename raw files or place them directly in the dataset
  root. The UI reports <b>✓ local</b> only after both the generated neuron
  table and merged connection table exist.
</div>

  
### NeuPrint

1. Get a NeuPrint token from <a href="https://neuprint.janelia.org/account" target="_blank">neuprint.janelia.org/account</a>.
2. Enter it above, click <b>Save Tokens</b>, then click <b>Test NeuPrint</b>.
3. The selected dataset is fetched and cached automatically on first use.

  
### FlyWire FAFB v783

<b>Automatic:</b> save a FlyWire Codex token in the API Tokens card, then use the <b>FAFB Dataset Downloads</b> card on the Settings tab — three levels (necessary data &middot; synapses &middot; skeleton bundle), resumable, and the converter runs automatically. Skeletons also fetch lazily per neuron from the Codex server during visualization.

The manual path: create <code>datasets/flywire\_FAFB\_v783/downloads/</code> from the project root and keep the original compressed filenames from the <a href="https://codex.flywire.ai/api/download?dataset=fafb" target="_blank">Codex FAFB download</a>.

<b>Required raw inputs for local connection analysis:</b>

- <code>classification.csv.gz</code>
- <code>connections\_princeton\_no\_threshold.csv.gz</code> (preferred), or <code>connections\_princeton.csv.gz</code> / <code>connections.csv.gz</code> as fallbacks

<b>Recommended metadata enrichment:</b> <code>names.csv.gz</code>, <code>coordinates.csv.gz</code>, <code>neurons.csv.gz</code>, <code>cell\_stats.csv.gz</code>, and <code>consolidated\_cell\_types.csv.gz</code>. The converter can proceed without these, but labels and metadata will be incomplete.

<b>Optional visualization inputs:</b> <code>fafb\_v783\_princeton\_synapse\_table.csv.gz</code> (local synapses) and <code>sk\_lod1\_783\_healed.zip</code> (local skeletons).

  

<pre><code>python src/FAFB_file_converter.py</code></pre>

Or select <code>flywire\_FAFB\_v783</code> in a tool and run it; the tool invokes the same local preparation automatically. Verify that the root contains <code>flywire\_FAFB\_v783\_allneurons\_neuron\_df.parquet</code> and <code>flywire\_FAFB\_v783\_merged\_connections.parquet</code>, then click <b>Refresh</b>.

  
### BANC v888 / v626

Choose exactly one identifier and do not mix versions. DROCAT prepares the metadata and connection tables automatically from the <a href="https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome" target="_blank">public BANC release bucket</a>; no FlyWire/Codex login or CAVE token is needed. The legacy files below are an optional offline fallback for users who already have a manual bundle:

- <code>neurons.csv.gz</code>
- <code>connections\_princeton.csv.gz</code>

Do not manually create or rename a <code>\*\_allneurons\_neuron\_df.csv</code> file; that is generated output. If the public bucket is unavailable, convert the legacy v626 bundle with:

  

<pre><code>python src/BANC_file_converter.py</code></pre>

For v888, run the converter with the selected dataset name:

  

<pre><code>python -c "import sys; sys.path.insert(0, 'src'); from BANC_file_converter import ensure_banc_data; d='banc_v888'; ensure_banc_data(d, 'datasets/' + d)"</code></pre>

Verify that the selected root contains both <code>&lt;dataset&gt;\_allneurons\_neuron\_df.parquet</code> and <code>&lt;dataset&gt;\_merged\_connections.parquet</code>, then click <b>Refresh</b>. BANC skeletons are fetched on demand from the public release bucket and cached as raw <code>.swc.zst</code>; no CAVE token is involved. Use the converted local files for pathfinding, networks, and tabular analysis.

  

<div class="callout warn">
  <b>CAVE token scope</b>
  A CAVE token is needed only for workflows that explicitly fetch FAFB data
  or skeletons through CAVE. It does not replace the local input files, and
  a token alone does not prepare a FAFB dataset.
</div>

  
## About

Version, GitHub repository and documentation links.
