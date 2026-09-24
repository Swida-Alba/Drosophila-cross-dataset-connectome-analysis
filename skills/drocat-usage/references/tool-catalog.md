# v4.5.0 direct tool catalog

Use this file after selecting an operation in `SKILL.md`. The example scripts
are executable configuration templates; the classes are the stable programmatic
API for a focused agent-created run script.

## Connectome queries

### Direct edges — `FindDirect.py`

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    dataset="male-cns:v0.9",
    sourceNeurons=["aMe12"],
    targetNeurons=["PPL101"],
    output_dir="/absolute/output/edges",
    min_synapse_num=3,
    min_ratio=0.0,
    min_traversal_probability=0.0,
    filter_by="bodyId",          # or "type"
    exclude_intra_type_connections=False,
    use_cache=True,
    output_format="csv",         # or "xlsx"
    network_layout="distributed",
    edgeN_limit=500,
    showfig=False,
)
fc.InitializeNeuronInfo()
fc.FindDirectConnections()
```

Use a list of types, bodyIds, instances, or regular expressions. Empty lists
have special meaning in the backend; state the intended scope before using
them.

### Multi-hop paths — `FindPath.py`

```python
fc = FindNeuronConnection(
    dataset="hemibrain:v1.2.1",
    sourceNeurons=["KC.*"],
    targetNeurons=["MBON03"],
    output_dir="/absolute/output/paths",
    min_synapse_num=3,
    max_interlayer=2,
    pathfinding="StrongestFirst",  # built-in default; DP/MemoizedDFS/DFS = unbounded complete runs (API)
    max_paths_bodyid=0,            # Max Paths: path-output budget; auto -> internal 1,000,000 (tau reported when it bites)
    graph_edge_limit_bodyid=0,     # Edge Budget: graph-level floor; 0 = off
    filter_by="bodyId",
    skip_bodyId=True,              # faster type-level first pass
    drop_untyped=True,             # Drop Untyped Neurons: label filter applied before graph build
    keyword_in_path_to_remove=["None"],
    use_cache=True,
    output_format="csv",
    edgeN_limit=500,               # Visualization Edge Limit: drawing-only cap per HTML view
    showfig=False,
)
fc.InitializeNeuronInfo()
fc.FindAllPath(forward_only=True)
```

Use `FindPath()` for a single path strategy only when the user asks for it.
`FindAllPath(forward_only=True)` is the usual reproducible first pass. Increase
`max_interlayer` only after a smaller run completes; path counts can grow
rapidly.

Every pathfinding run writes a threshold/bottleneck provenance block
(`requested_threshold`, `applied_threshold`, `applied_threshold_source`,
StrongestFirst budget/tau, Edge Budget floor, `strongest_retained_bottleneck`
(W*), `paths_complete`) to `parameters.txt`, `all_attributes.json`, and
`data_details/parameters.csv`; with `drop_untyped=True` (the default) dropped
rows go to `data_details/untyped_dropped_records.csv`, counts to
`user_warning_notes.txt`. Visualization output uses the canonical names
`Network_<run>.html`, `Heatmap_<run>.html`, `Sankey_<run>.html`, plus
`visualization_data/` CSVs, inside both `visualization/` and
`bodyId_visualization/` (early previews: `network_early/`,
`network_early_bodyId/`).

## Visualizations

### Path/Sankey/network HTML — `PlotPath.py`

`VisualizePath` lives in `vispath-subproject/src/vispath_pkg/vispath.py` and is
not a separate pip package requirement.

```python
from vispath_pkg import VisualizePath

vp = VisualizePath(
    path_file="/absolute/output/paths/path_data.csv",
    output_folder="/absolute/output/network",
    network_layout="hierarchical",  # spring, circular, distributed
    source_color="#4A90E2",
    intermediate_color="#50E3C2",
    target_color="#B8E986",
    link_color="rgba(74,144,226,0.3)",
    showfig=False,
    output_format="csv",
)
connections, graph = vp.visualize()
```

The input may be CSV or Excel. For Excel, use `sheet_name="path_type"` or
`sheet_name="path_bodyId"` when autodetection is ambiguous. Use a completed
FindPath artifact rather than a hand-written file unless the schema has been
verified.

### Empty editable network canvas

```python
vp = VisualizePath(
    path_file=None,
    output_folder="/absolute/output/empty_network",
    generate_empty_network=True,
    network_layout="hierarchical",
    showfig=True,       # opens the HTML in a new browser tab
)
vp.visualize()
```

The HTML is a Cytoscape canvas with Edit Mode controls. Keep `showfig=False`
for a headless run and open the returned `*_network.html` file manually.

### 3D neuron morphology — `plot3dSkeleton.py`

```python
from visualize_skeleton import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset="male-cns:v0.9",
    output_dir="/absolute/output/skeleton",
    neuron_layers=["aMe12", "aMe10", "PPL101"],
    neuron_colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
    skeleton_mode="line",       # start with line for large queries
    skip_synapse=True,
    mesh_roi=["EB", "LH", "AL"],
    mesh_color=["#4A90E2", "#50E3C2", "#B8E986"],
    mesh_alpha=0.1,
    brain_mesh="native",
    show_fig=False,
    export_views=False,
)
vs.plot_neurons()
```

Colors are assigned in displayed order. Each resolved ROI mesh has an
independent legend entry. Add `plot_individuals(...)` or `export_video(...)`
only after the base HTML succeeds.

## Comparison and profile tools

### Cross-dataset comparison — `InterDatasetComparator.py`

```python
from comparison import ComparisonParameters, ComparisonAnalyzer

params = ComparisonParameters(
    datasets=["hemibrain:v1.2.1", "male-cns:v0.9"],
    source_neurons=["aMe12"],
    target_neurons=["PPL101"],
    max_interlayer=2,
    thresholds=[1, 3, 5, 10],
    comparison_mode="path",     # use "edge" to preserve strong direct edges
    top_edges=500,
    skip_bodyId=True,
    drop_untyped=True,          # Drop Untyped Neurons: applied post label-mapping
    output_folder="/absolute/output/comparison",
)
analyzer = ComparisonAnalyzer(params, verbose=True)
analyzer.run_comparison()
analyzer.generate_report()
analyzer.export_results()
```

Use `cache_only=True` for deliberately offline/deprecated datasets only after
checking local cache coverage. Use a `LabelMapper` when names differ between
datasets. `drop_untyped=True` (the default) drops edges touching untyped
neurons AFTER standardized cross-dataset label mapping; dropped rows land in
`comparison_results/untyped_dropped_records.csv` (counts appended to
`user_warning_notes.txt`), and each per-dataset threshold folder carries the
threshold/bottleneck provenance block.

### Connectivity profiles — `ConnectivityProfiling.py`

```python
from comparison.profile_comparator import ConnectivityProfileComparer

comparer = ConnectivityProfileComparer(
    query=["aMe12", "aMe10", "aMe9"],
    dataset="male-cns:v0.9",
    top_k=15,
    top_m=5,
    direction="both",
    output_dir="/absolute/output/profiles",
    generate_heatmaps=True,
    show_figures=False,
    skip_bodyId_level=False,
)
results = comparer.run()
```

### Homologs — `FindHomologs.py`

```python
from comparison.profile_comparator import HomologFinder

finder = HomologFinder(
    source="aMe12",
    source_dataset="male-cns:v0.9",
    target_dataset="hemibrain:v1.2.1",
    output_dir="/absolute/output/homologs",
    similarity_metric="jaccard",   # sorts only — every metric is computed;
                                   # jaccard/100 are the ctor + UI defaults
    top_n=100,
    vector_prefiltering=True,
    visualize_skeleton=False,
)
results = finder.find_homologs_fast()
```

Use `find_homologs()` for a slower comprehensive search when the fast adjacency
search is insufficient.

## NeuronBridge and FlyLight

`NeuronBridgeFinder` needs no token. The most useful methods are:

- `find_lines_batch(queries=..., dataset=..., output_dir=...)` for EM → LM;
- `find_neurons_batch(line_names=..., output_dir=...)` for LM → EM;
- `analyze_colabeling(lines=..., output_dir=...)` for line overlap and specificity.

`FlyLightDownloader.download(...)` accepts a line name/list, `output_dir`,
`max_files`, `dry_run`, filters, and optional PDF/PPTX summary generation.
Start with `dry_run=True` and a small `max_files` value before downloading many
images.
