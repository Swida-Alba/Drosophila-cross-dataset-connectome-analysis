---
title: Morphology
tag: Connectome
---

The Morphology tab holds the skeleton-based similarity workflows, switched with the buttons at the top of the panel:

- <b>Find Similar</b> — query-vs-all <b>morphological</b> similarity search within one dataset (Vector (spatial) or NBLAST method). <a href="find_similar.html">Find Similar guide</a>
- <b>Comparison</b> — N×N morphology comparison of already-identified neurons, within one dataset <b>or across datasets</b> (the intra-dataset run chooses its matrix row as a type, an individual neuron, or a custom group; bodyId-level matrices, heatmaps, report). <a href="morphology_comparison.html">Comparison guide</a>

  

<div class="callout info">
  <b>Cross-dataset vs intra-dataset:</b> Find Similar is intra-dataset.
  The <b>Comparison</b> sub-tab switches to <b>cross-dataset morphology</b>
  when two or more datasets are selected: it scores cross-dataset pairs with
  the shared raw-basis vector engine (per-dataset caches; each pair scores
  in the target dataset's render space) and writes one merged report — see
  the
  <a href="cross_dataset_morphology.html">Cross-Dataset guide</a>.
</div>

  
## Which sub-tab do I need?

  

<table>
  <tr><th>Goal</th><th>Sub-tab</th></tr>
  <tr><td>Find neurons with similar skeleton shape to a query</td><td><b>Find Similar</b></td></tr>
  <tr><td>Compare shapes of 2+ known neurons (same dataset)</td><td><b>Comparison</b> (one dataset)</td></tr>
  <tr><td>Compare neuron shapes across two or more datasets (homologs)</td><td><b>Comparison</b> (2+ datasets → cross-dataset mode)</td></tr>
</table>

  
## Output

Find Similar saves each run in its own timestamped folder (<code>similar-morphology\_{dataset}\_{query}\_{timestamp}</code>) under the selected output directory and writes the exported run guide (<code>\_UserGuide\_please\_read\_me.html</code>) next to the results. See the <a href="find_similar.html">Find Similar guide</a> for the full file tree and scoring details.
