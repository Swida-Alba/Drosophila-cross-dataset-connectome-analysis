---
title: Comparison — Cross-Dataset
tag: Connectome
subtag: Morphology tab
page_title: Morphology · Comparison (Cross-Dataset)
---

Compare the morphology of the <b>queried neurons across datasets</b>: enter query types, select two or more datasets, and get pairwise <b>vector_v2</b> similarity matrices per dataset pair, a seeded null baseline for context, and 3D overlay scenes with every dataset bridged into one reference template.

<div class="callout warn">
  <b>Dataset scope.</b> Only <b>FAFB</b>, <b>male-cns</b> and <b>BANC</b>
  participate — these are the datasets with a bridging transform inside
  the 2-hop accuracy guard (FAFB↔male-cns, FAFB↔BANC, male-cns↔BANC are
  1–2 transforms; everything else, e.g. hemibrain or MANC, needs 7–9 and
  is refused with a banner). <b>BANC comparisons carry a standing
  reliability warning</b>: public BANC skeleton products mix L2 / full /
  µm sources. First BANC use bootstraps its morphology population
  artifacts from the locally cached skeletons (one-time, offline).
</div>


## Quick start

1. Select <b>two or more datasets</b> (the list shows every dataset; pairs without a bridging transform are refused at run time with the red banner).
2. Enter the <b>Neurons to Compare</b> — types or patterns (e.g. <code>APDN3</code>, <code>aMe.*</code>). With <b>Auto Type Mapping</b> on (the default), types are resolved through the shared validity-aware resolver — curated renames (e.g. APDN3 → SLP249) and valid splits resolve across datasets, and conflicted types fail closed; a type missing in one dataset shows up as an explicit empty row in the report. Turn the mapping off to fall back to raw same-name/pattern resolution.
3. Optionally pick a <b>Reference Template</b> for the overlay scenes (defaults to the first selected dataset).
4. Click <b>Run Comparison</b> (the same button as the intra-dataset mode — two or more selected datasets switch it to this mode).


## Cross-dataset options

- <b>Auto Type Mapping</b> (on): validity-aware cross-dataset type resolution, as described above.
- <b>Null Sample Size</b> (default 200, range 10–1000): how many seeded random target neurons per query feed the null baseline.
- <b>3D Overlay Scenes</b> (on): render the bridged overlay scenes.
- <b>Fetch Missing Skeletons Online</b> (on): fetch skeletons that are not in the local cache.
- <b>Custom Groups (LabelMapper)</b>: the always-visible grouping card doubles as the cross run's <b>tier-0 label overlay</b> (a loaded preset or inline board). Labels are identities and are never re-resolved through the auto type mapper — see <a href="label_mapper.html">the label-mapper guide</a>.


## How scoring works

- For each ordered dataset pair <i>A → B</i>, the queried neurons' members of <i>A</i> are transformed into <i>B</i>'s render space and scored against the members of <i>B</i> with the production <b>vector_v2</b> scorer (block-weighted whitened cosine — the same scorer Find Similar uses). <b>NBLAST is not used cross-dataset.</b>
- Raw cross-dataset scores have a high, query-dependent baseline. Every pair therefore reports a <b>null baseline</b>: the p95/median of the same queries scored against ~200 seeded random target neurons. The sample is shared across queries and runs, so repeat runs are fast. Member pairs above the baseline p95 are flagged in <code>morph_bodyid_scores.csv</code>, and the raw per-neuron scores are exported as the <code>morph_bodyid_matrix.csv</code> bodyId × bodyId matrix. vector_v2 is a whitened cosine that can be negative, so the report renders it on a diverging blue–white–red scale.
- Scores are comparable <i>within</i> a dataset pair (one frame), not across pairs — the report states each pair's frame (frame disclosure). In particular, A&rarr;B and B&rarr;A are <b>not symmetric</b>: each direction transforms into a different render space and uses a different whitening basis, so the numeric scores and baselines differ. The overview table discloses this.
- Per type and dataset, at most <b>Max Members per Type</b> neurons (default 25) enter the scoring; the 3D scenes render at most <b>Scene Members per Type</b> (default 3) per dataset for legibility.
- With a <b>LabelMapper preset</b> loaded, the run also writes the label-aligned type view: the SAME member pairs folded by standard label instead of native type name. Native type names are dataset-scoped homonyms that can carry one, several, or no labels — the native matrix stays beside it as the ground truth, and <code>parameters.json</code> discloses the exact axes semantics.


## Output

<pre><code>{output_dir}/morph_cross_{datasets}_{queries}_{timestamp}/
overview.csv                    # queried type × pair: best target-type match
                                #   + a `label` column when a LabelMapper
                                #   preset was loaded ('/'-joined when a
                                #   native type settles to two labels)
members_summary.csv             # compared member counts per type/dataset
parameters.json / README.txt    # run config, warnings, resolution notes
report.html                     # tabbed report (same generator as the
                                # connectivity-profiling report: pair tabs,
                                # Type/BodyId levels, Ward-clustered cards,
                                # Plotly embedded — renders offline)
&lt;SRC&gt;_to_&lt;TGT&gt;/                 # one folder per dataset pair
  results/
    morph_type_matrix.csv       # type × type mean vector_v2 (STAYS native-keyed)
    morph_labeltype_matrix.csv  # LabelMapper view beside the native one
                                #   (one row per standard label; multi-label
                                #   native types expand; unmapped members
                                #   keep their native name) — only written
                                #   when a preset was loaded
    morph_bodyid_matrix.csv     # bodyId × bodyId vector_v2 (tree-legend
                                #   labels; with a preset, resolved ids
                                #   read {bodyId}_{label}({native}) — the
                                #   parenthetical is dropped when the label
                                #   already is the native name)
    morph_bodyid_scores.csv     # long-form scores + above_baseline flag
    null_baseline.json          # seeded null reference per query
  visualization/
    heatmap_morph_&lt;SRC&gt;_to_&lt;TGT&gt;_{type,bodyid}.html
                                # interactive VisPath heatmaps
plot-3d_crossmorph_*/           # overlay scenes (reference template)</code></pre>

<div class="callout info">
  <b>Comparison, not discovery.</b> With two or more datasets this mode compares the neurons you
  queried. To search a whole dataset morphologically, use the
  <b>Find Similar</b> sub-tab (intra-dataset); for cross-dataset homolog
  discovery, use the <a href="find_homologs.html">Connectivity tab's
  Find Homolog</a> search with its optional morph qualification.
</div>
