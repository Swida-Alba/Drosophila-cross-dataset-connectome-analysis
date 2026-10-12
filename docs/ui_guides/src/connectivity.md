---
title: Connectivity
tag: Connectome
---

The Connectivity tab holds the two connection-based similarity workflows, switched with the buttons at the top of the panel:

- <b>Find Homolog</b> — search for neurons similar to a query by <b>connectivity profile</b> (synaptic partner structure). Works across datasets (homolog search) or within one dataset (Target = Source). <a href="find_homologs.html">Find Homolog guide</a>
- <b>Comparison</b> — compare the connectivity profiles of already-identified neurons (2+ queries) within one dataset or across several datasets at once. <a href="connectivity_profiling.html">Comparison guide</a>

  

<div class="callout info">
  <b>Cross-dataset vs intra-dataset:</b> connectivity comparison supports
  <b>multiple datasets</b> — the same query is profiled in every selected
  dataset and compared on the intra- and inter-dataset level. Morphological
  comparison (the <a href="morphology.html">Morphology</a> tab) also supports
  cross-dataset runs — its Comparison sub-tab switches to cross-dataset mode
  when two or more datasets are selected.
</div>

  
## Which sub-tab do I need?

  

<table>
  <tr><th>Goal</th><th>Sub-tab</th></tr>
  <tr><td>Find candidates for a neuron's homolog in another dataset</td><td><b>Find Homolog</b> (Source ≠ Target)</td></tr>
  <tr><td>Find similar neurons within one dataset by connectivity</td><td><b>Find Homolog</b> (Target = Source)</td></tr>
  <tr><td>Compare profiles of 2+ known neurons, one or more datasets</td><td><b>Comparison</b></td></tr>
</table>

  
## Output

Both sub-tabs save each run in its own timestamped folder under the selected output directory and write the exported run guide (<code>\_UserGuide\_please\_read\_me.html</code>) next to the results. See the per-sub-tab guides for the exact file trees.
