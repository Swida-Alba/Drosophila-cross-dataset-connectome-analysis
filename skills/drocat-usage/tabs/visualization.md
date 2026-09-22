# 3D Skeleton + Path Network (visualization)

Reproduce the **Visualization** UI tab. It drives **two** tools: `plot3d_skeleton`
(3D neuron morphology via `VisualizeSkeleton`) and `plot_path` (interactive
network/Sankey/heatmap HTML via `VisualizePath`). They can be run independently;
in practice you run a pathfinding analysis first, then feed its output to
`plot_path`, and/or render neurons with `plot3d_skeleton`.

## Backend contract

| Tool | tool_key | import · class | method |
| --- | --- | --- | --- |
| 3D skeleton | `plot3d_skeleton` | `from visualize_skeleton import VisualizeSkeleton` · `VisualizeSkeleton` | `vs.plot_neurons()` |
| Path network | `plot_path` | `from vispath_pkg import VisualizePath` · `VisualizePath` | `vp.visualize()` |

## 3D skeleton — VisualizeSkeleton.plot_neurons

```python
from visualize_skeleton import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset="male-cns:v0.9",
    neuron_layers=["aMe12", "aMe10", "PPL101"],
    search_columns="auto",              # "auto" | "type" | "instance" | "bodyId"
    hemisphere="both",
    custom_layer_names=[],              # optional layer labels
    output_dir="/absolute/output/skeleton",
    output_format="csv",                # merged synapse export: "csv" | "xlsx"
    skeleton_mode="line",               # start with line for large queries
    brain_mesh="native",
    vnc_mesh=None,
    legend_mode="layer",                # "single" | "type" | "tree" | "layer"
    neuron_alpha=1.0,
    neuron_colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
    synapse_colors=["#50E3C2"],
    background_color="#ffffff",
    skip_synapse=False,
    min_synapse_num=3,
    synapse_size="real",                # "real" or numeric; uniform_synapse_size bool
    uniform_synapse_size=False,
    synapse_alpha=1.0,
    synapse_mode="scatter",             # "scatter" | "sphere" | "cone" | "tetrahedron"
    mesh_roi=["EB", "LH", "AL"],
    mesh_color=["#4A90E2", "#50E3C2", "#B8E986"],
    mesh_alpha=0.1,
    cache_neurons=True,
    cache_synapses=True,
    smooth_skeleton=True,
    show_soma=False,
    show_connectors=False,
    export_method="webdriver",          # or "kaleido"
    export_scale=2,
    export_views=False,
    show_fig=False,
    brain_mesh_color=None,
    neuprint_skeleton_pipeline="fast",  # "fast" | "fine" | "artistic" | "direct" | ...
    skeleton_mesh_simplification=0.0,
    progress_total=3,                   # optional UI-injected progress protocol
)

vs.plot_neurons()
# optional exports (only after the base HTML succeeds):
vs.plot_individuals(pdf_images_per_page=(3, 2), views=["front"], summary_format=["pdf"])
vs.export_video(fps=30, degree_per_frame=1.0, rotate="horizontal",
                export_gif=True, gif_scale=0.2)
```

## Path network — VisualizePath.visualize

```python
from vispath_pkg import VisualizePath

vp = VisualizePath(
    path_file="/absolute/output/paths/path_data.csv",   # from a FindPath/complete-paths result
    output_folder="/absolute/output/network",
    source_color="#4A90E2",
    intermediate_color="#50E3C2",
    target_color="#B8E986",
    link_color="rgba(74,144,226,0.3)",
    network_layout="hierarchical",      # spring, circular, distributed
    showfig=False,
    generate_empty_network=False,       # True → blank editable Cytoscape canvas
    progress_total=4,                   # optional UI-injected progress protocol
)
connections, graph = vp.visualize()
```

Empty editable canvas:

```python
vp = VisualizePath(
    path_file=None,
    output_folder="/absolute/output/empty_network",
    generate_empty_network=True,
    network_layout="hierarchical",
    showfig=True,
)
vp.visualize()
```

## Run

```bash
# 3D skeleton
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_Skeleton_<date>.py
# path network
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 --script archive/scripts_local/agent_PathNet_<date>.py
```

## Outputs

- 3D skeleton: interactive Plotly HTML (plus optional PNG views, PDF/PPTX
  individual profiles, GIF/video).
- Path network: Cytoscape network, Sankey, and heatmap HTML plus connection tables;
  the empty mode reports an editable canvas HTML.

## Notes

- BANC (`banc_v626`/`banc_v888`) is supported (native BANC space, public
  bucket — no token). Skeleton chain is unified (888 L2 -> 888 full ->
  v626 pcg; no source selector). Defaults: Synapse Mode `skip` (opt-in
  downloads a ~3.9 GB table once; markers sit on pre-sites only),
  normalized tube radii (240 nm median target). Saved HTML/PNGs open on the
  calibrated BANC frontal view.
- `mesh_roi`/brain/VNC meshes are dataset-specific; verify ROI availability
  (`vs.list_available_rois()`). Mesh transforms also depend on the dataset.
- Keep `show_fig=False` and `export_views=False` for unattended runs; enable
  video/WebDriver/PDF exports only after the base HTML is valid.
- For `plot_path` use a real FindPath artifact (verified schema), not a hand-written
  file, unless the schema has been checked. `network_layout="distributed"` or
  `"hierarchical"` is best for large graphs.
