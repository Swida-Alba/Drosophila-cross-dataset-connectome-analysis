# Find EM Neurons (nb_find_neuron)

Reproduce the **Find EM Neurons** UI tab (LM → EM). Uses `NeuronBridgeFinder` to
map driver-line names to candidate EM neurons, with optional 3D skeleton
visualization. No token required.

## Backend contract

- **tool_key:** `nb_find_neuron`
- **import:** `from neuronbridge_finder import NeuronBridgeFinder`
- **class:** `NeuronBridgeFinder` (var `finder`)
- **method:** `finder.find_neurons_batch(**method_params)`

## Parameters the UI builds

```python
from neuronbridge_finder import NeuronBridgeFinder

finder = NeuronBridgeFinder(verbose=True)

result = finder.find_neurons_batch(
    line_names=["SS00001"],             # or a list of lines
    output_dir="/absolute/output/nb_neuron",
    match_type="both",                 # "cds" | "pppm" | "both"
    top_n=10,
    min_score=0.0,
    visualize_top_n=0,                  # >0 to render top-N skeleton views
    generate_individual_profiles=None,  # ["pdf"] if profiles requested
    visualize_by="bodyId",
    visualization_settings={},          # skeleton settings when visualizing
    sort_by="score",
    pdf_images_per_page=(3, 2),
    background_color="#ffffff",
)
```

## Run

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_Neuron_<date>.py
```

## Output detail (Full / Compact)

`keep_per_match_csv=True` (default, Full) keeps every file. Compact
(`keep_per_match_csv=False`) drops the bodyId-level match tables
(`all_neurons.csv`, `{line}_neurons.csv`, `by_dataset/*_neurons.csv`) once
the per-dataset type summaries are written — audited in the run's
`cleanup_audit.json`. Type summaries, `{line}_type_mapped.csv`, the
distribution plot, and `plot-3d_*` renders always stay. Per-dataset files
live in `by_dataset/`.

## Outputs

- Ranked neuron matches (CSV) and optional 3D skeleton views / PDF profiles.

## Notes

- Start with `top_n` small and no `visualize_top_n`/profiles; add GUI-heavy output
  only after the ranked CSV is validated.
- `generate_individual_profiles=["pdf"]` writes per-line profile summaries.
