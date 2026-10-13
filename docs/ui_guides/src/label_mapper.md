---
title: Custom Type Mappings
tag: Settings
tag_class: sys
---

Group neuron types across datasets into custom groups for cross-dataset comparison and pathfinding (LabelMapper format).


## UI workflow

Mappings are built <b>inline on each tab</b> — no scripting needed:

1. <b>Open the tab's mapping panel.</b> The <b>Cross-Dataset Comparison</b> tab mounts it as <b>Custom Mapping</b>; <b>Complete Paths</b>, <b>Network</b> and <b>Shortest Paths</b> mount it as <b>Custom Grouping</b>, and <b>Connectivity's Comparison sub-tab</b> as <b>Custom Grouping Preset</b> (the Settings tab's <b>Custom Type Mappings</b> card offers the same editor). Choose the <b>Target datasets</b> where offered, then edit the board — one row per custom group, one aligned member input per selected dataset, outlined chip fields, dataset-aware type-ahead with a Recent list in each member input (a pick keeps that query's rows on screen and ticks the names already added &mdash; in either list, and a ticked row takes its chip back out on the next click &mdash; and a Recent row hints at the column's own dataset when that name actually exists there), and a <b>See available neurons</b> lookup.
2. <b>Define your groups and run.</b> At run time the board is exported to <code>cache/user_mappings/\_inline/&lt;tab&gt;\_&lt;timestamp&gt;.json</code> and passed to the run — as <code>overall_mapping_json</code> on the Cross-Dataset tab, where it acts as a mapping <b>overlay</b> (your explicit source/target neuron queries stay, and only neurons matching the mapping are renamed to their custom groups; unmapped neurons keep their identifiers), or as <code>custom_mapping_file</code> on the path/network tabs, where mapped neurons are grouped at type level via <code>EnrichConnectionTable</code> (custom_group columns). <b>Residuals of collided native types</b> (2026-10-08; the producer set was fixed 2026-10-13): when a native type's population is only partially governed — some of its neurons are mapper members, others are not — or the type name is itself one of the mapper's labels, the type resolver star-tags the unenrolled neurons as <code>T\*</code> (native type + <code>*</code>), so they never silently merge into the governed population or into a same-spelling label. The key is terminal (never re-resolved, hemisphere suffix included) and deliberately undecorated, so what it stands for is published instead: the <code>[label mapper residuals]</code> line in <code>user_warning_notes.txt</code>, <code>label_mapper_residuals.csv</code> in the run folder and under <code>comparison_report_used_data/</code>, and the <code>label_mapper_residuals</code> record in <code>all_attributes.json</code>. Only lanes that can see the dataset's neuron table assign the key; readers consume it as exported, and every export keeps the raw native type beside it (<code>native_type_pre</code>/<code>native_type_post</code>). Without a local neuron table membership still applies and residuals are reported as <code>NOT EVALUATED</code>.
3. <b>Reuse a mapping later</b> via <b>Load…</b> (parses any mapping JSON back onto the board, so a run is exactly reproducible from its exported file) or the <b>history pulldown</b> (below).
4. <b>Optional — save a named copy in Settings.</b> The Settings tab's <b>Custom Type Mappings</b> card adds a <b>Save Mapping</b> action that writes the stable <code>Custom Mapping</code> preset to <code>cache/user_mappings.json</code> (gitignored) and exports it to <code>cache/user_mappings/</code>. Saving does <b>not</b> attach the preset to future runs — no tab auto-loads saved presets; bring the JSON back onto a board with <b>Load…</b> when you want it.

Runs always consume the exported file, so results stay reproducible (<code>label_map.json</code> / <code>parameters.json</code> record the mapping used).


## Inline groups &amp; history (query-local grouper)

Every tab with a <b>Custom Grouping</b> button opens a popup panel (same design as the neuron-index viewer). The panel pairs a lightweight group editor — one row per label: a group name plus one neuron box per dataset — with a <b>history pulldown</b> of previously-used groups. Opening the panel seeds one empty row for first editing. Groups are recorded into history automatically (when pushed to a query or run); nothing is named or saved by hand.

- <b>Same label, all datasets:</b> one label governs every dataset column on the board. On the Cross-Dataset tab the columns follow the selected datasets, and <b>label names are compulsory</b> (the shared label is what aligns datasets). On single-dataset tabs names are optional — blank rows auto-name to <code>Group_N</code>.
- <b>Uniform format:</b> at run time the board is exported to the same LabelMapper JSON schema as presets, at <code>cache/user_mappings/\_inline/&lt;tab&gt;\_&lt;timestamp&gt;.json</code>. Scripts load that file directly: <code>LabelMapper(overall_mapping_json="…")</code> — no conversion.
- <b>Reload:</b> <b>Load…</b> parses any mapping JSON (inline export, preset export, or hand-written) back onto the board, so a run is exactly reproducible from its exported file.
- <b>History pulldown:</b> the former "preset" control is now a history list. Groups are recorded automatically — when a group is pushed to a query or exported at run time — into <code>cache/user_mappings/group_history.json</code>. Selecting an entry loads that group's members onto the board for reuse or editing; no name is asked and nothing is saved manually. Each entry also has an <b>“x”</b> that removes it from the history after a confirmation popup. A label maps to every dataset it was ever defined for; defining it again with different members <b>covers</b> (replaces) the previous value for the datasets on the board. Auto-named <code>Group_N</code> rows are never recorded.
- <b>Use groups in the query:</b> each group row carries <i>Add to Source</i> / <i>Add to Target</i> (or <i>Add to Query</i>) buttons that append the group's <b>label</b> as a single chip to the tab's neuron inputs. The board's toolbar has a matching <i>Add all</i> button (the check-list icon beside <b>Load…</b>) that pushes every populated group at once, in board order, naming blank rows as <code>Group_N</code> the same way a single push does; a group with no members anywhere is skipped, and a board with nothing to add says so instead of touching the input. The backend expands that label into its member neurons, so a freshly defined group drives the run directly and the exported tables/visualizations aggregate under the group label. The buttons are <b>not offered for an input the tab has disabled</b> — in the Connectivity and Morphology Comparison sub-tabs, choosing <i>Aggregation Level = custom group</i> disables that query box because the rows come from the board itself, so pushing labels into it would change nothing. Rows re-render on every panel open, so the buttons follow the current level.
- <b>Which chips are groups:</b> a chip that names a group is ambiguous on its face — <code>DN1a</code> reads identically whether it means the native type or the group you just defined — so the query input marks the chips that currently resolve as groups: the group glyph, a cobalt left rule and a deeper fill, with a tooltip giving the member counts in the board's own column order (<i>Custom group DN1a — 4 in male-cns:v1.0, …</i>) including datasets the tab is not showing as columns. The mark tracks the board — rename a group or delete its row and the chip in the query box goes back to looking like a literal query, which is exactly what it then is. It also tracks <b>Match by</b>: at any mode other than <i>exact</i> the chip is rewritten into a regex before the backend sees it, so the label stops expanding and the mark is withheld rather than promising a resolution that will not happen. The <b>count badge</b> follows the same rule and is the one surface that also covers an <b>uploaded list</b> — a file of group labels never becomes chips, so it reads <i>21 groups</i> (or <i>26 neurons · 21 groups</i> when literal queries are mixed in) instead of calling them neurons.
- <b>Apply Groups to Partner Keys</b> (<b>Connectivity ▸ Comparison only</b>): the profiling comparer is the one grouping surface with a <i>partner side</i> — each neuron's profile is a map of partner keys to weights — so this control folds those keys through the groups (weights summed per label) and computes similarity over the group feature space instead of raw type names. It folds the <i>type-level</i> partner keys at whatever aggregation level the run ends up on (attaching a preset forces <i>custom group</i>), while bodyId-level matrices keep raw keys. The other grouping tabs have no partner side to collapse, which is why they offer no such control. It needs a mapper: with no preset attached the toggle is dropped and the run log says so.

<div class="callout info">
  <b>Division of labor:</b> the shared Custom Mapping panel writes aligned
  groups to the <i>source</i> and <i>target</i> roles.
  Empty boxes export as empty groups, which keeps cross-dataset validation
  happy when a label has members in only some datasets.
</div>


## Mapping file format (JSON)

The exported mapping uses LabelMapper's native JSON schema — a top-level <code>format</code> tag naming the grammar, one mapping per role, a <code>custom_label</code> list of group names, and one list of group lists per dataset:

<pre>{
"format": "drocat_custom_groups/v1",
"source_mapping": {
  "custom_label": ["aMe12"],
  "hemibrain:v1.2.1": [["aMe12"]],
  "male-cns:v0.9": [["aMe12"]],
  "flywire_FAFB_v783": [["720575940610453042"]]
},
"target_mapping": {
  "custom_label": ["KCg-s1"],
  "hemibrain:v1.2.1": [["KCg-s1"]],
  "male-cns:v0.9": [["KCg-s1"]],
  "flywire_FAFB_v783": [["KCg-s1"]]
}
}</pre>
<div class="callout info">
  <b>Only JSON</b> is supported for <code>overall_mapping_json</code>. A role
  may be omitted (e.g. source-only mappings); at least one mapping key must
  be present. The <code>format</code> key is required for a preset a run reads:
  it names the grammar the grouping board writes, and a preset that omits it —
  or declares a grammar this build does not read — is refused with a clear error
  rather than parsed on shape alone (CSV role files and in-memory dictionaries
  make no such claim and stay accepted). Exports from this board, and
  <code>LabelMapper.to_dict()</code>, always write the tag.
</div>


## CSV format (script use)

The CSV format is the most straightforward way to define mappings manually: a <code>custom_label</code> column plus one column per dataset. Cell values are specific neuron IDs or regex patterns (e.g. <code>MBON14.*_R</code> matches multiple neurons).

<table>
  <tr><th>custom_label</th><th>hemibrain:v1.2.1</th><th>male-cns:v0.9</th><th>flywire_FAFB_v783</th></tr>
  <tr><td>aMe12</td><td>aMe12</td><td>aMe12</td><td>720575940610453042</td></tr>
  <tr><td>MBON14</td><td>MBON14.*_R</td><td>MBON14.*_R</td><td>MBON14</td></tr>
  <tr><td>PPL101</td><td>PPL101</td><td>PPL101</td><td>720575940621886666</td></tr>
</table>


## Why use custom groupings?

- <b>Consistency</b>: compare "apples to apples" across datasets whose naming conventions differ (e.g. <code>aMe12</code> in hemibrain vs <code>aMe12-like</code> in male-cns).
- <b>Simplicity</b>: analysis code deals only with standard labels, not dataset-specific IDs (e.g. <code>720575940610453042</code>).
- <b>Scalability</b>: add new datasets by extending the mapping, without changing analysis code.


## Display role: matrix axes and label views

Beyond defining comparison rows, a loaded preset acts as the <b>tier-0 identity overlay</b> on every comparison surface — a label IS the identity and is never re-resolved through the auto type mapper:

- <b>BodyId axes</b> (Connectivity Comparison, Morphology Comparison, Cross-Dataset Morphology): a preset-resolved bodyId reads <code>{bodyId}\_{label}({native})</code> — e.g. <code>13531_5th-LNv(5thsLNv_LNd6_L)</code> — with the legacy suffix, side marker included, inside the parentheses. When the label already IS the native name the parenthetical is dropped (<code>{bodyId}\_{label}</code>, or <code>{bodyId}\_{label}\_{L|R}</code> when only the side differs). The bodyId stays the token before the first <code>_</code> (axis parsers key on the head); unmapped members keep the legacy form byte-identically.
- <b>Cross-dataset morphology</b> writes <code>morph_labeltype_matrix.csv</code> beside the native <code>morph_type_matrix.csv</code> (which stays native-keyed): one row per standard label, multi-label native types expanded, unmapped members keeping their native name. <code>overview.csv</code> gains a <code>label</code> column ('/'-joined for double-claim types).
- <b>Morphology Comparison</b> <code>members.csv</code> gains a <code>label</code> column beside the native <code>type</code> — per-bodyId resolution, because a native type can settle to two labels while a bodyId resolves to exactly one.

<div class="callout tip">
  <b>Reproducibility</b>: each run records the mapping file it consumed
  (<code>label_map.json</code> / <code>parameters.json</code>), so a mapping
  can be edited later without invalidating past results.
</div>
