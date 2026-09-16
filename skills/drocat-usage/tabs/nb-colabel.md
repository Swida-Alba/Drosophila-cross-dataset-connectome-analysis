# Co-Labeling Analysis (nb_colabel)

Reproduce the **Co-Labeling Analysis** UI tab. Uses `NeuronBridgeFinder` to
compute line co-labeling overlap, similarity, specificity, and expression matrices
across datasets. No token required.

## Backend contract

- **tool_key:** `nb_colabel`
- **import:** `from neuronbridge_finder import NeuronBridgeFinder`
- **class:** `NeuronBridgeFinder` (var `finder`)
- **method:** `finder.analyze_colabeling(**method_params)`

## Parameters the UI builds

```python
from neuronbridge_finder import NeuronBridgeFinder

finder = NeuronBridgeFinder(verbose=True)

result = finder.analyze_colabeling(
    lines=["SS00001", "SS00002"],       # or a list of names
    match_type="both",                 # "cds" | "pppm" | "both"
    output_dir="/absolute/output/nb_colabel",
    similarity_methods=["jaccard", "weighted_jaccard"],
    generate_report=True,
    visualize=True,                     # generate heatmaps
    visualize_top_n=0,                  # >0 to render 3D skeleton of top neurons
    top_n_neurons=10,
    min_score=0.0,
    min_type_avg_score=0.0,
    sort_by="score",
    background_color="#ffffff",
    pdf_images_per_page=(3, 2),
    datasets_to_visualize="male-cns:v0.9",  # scalar dataset, or "all"
    visualize_by="bodyId",
    visualization_settings={},          # skeleton settings when visualizing
)
```

## Run

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_Colabel_<date>.py
```

## Output detail (Full / Compact)

`keep_per_match_csv=True` (default, Full) keeps every file. Compact
(`keep_per_match_csv=False`) drops `line_labeled_neurons/` and
`distribution_data_by_neuron.csv` once the similarity matrices and report
are written — audited in the run's `cleanup_audit.json`. Matrices,
expression data, by-type distributions, and the HTML report always stay.

## Outputs

- Expression matrices, co-labeling heatmaps, similarity/specificity tables, and a
  report (PDF/PPTX when requested).

## Notes

- `similarity_methods` selects metrics; an empty list is an error in the UI.
- `visualize=True` writes the co-labeling heatmap; `visualize_top_n` adds 3D
  skeleton views (requires skeleton settings).
