# Cross-tab combinations

Some workflows need more than one UI-tab tool chained together. This reference
shows the common recipes. Each step is one tab's backend tool (see [`tabs/`](../tabs/));
pass the output artifact of one step into the next.

## FindPath → PlotPath (pathfinding → network HTML)

```python
# Step 1: Complete Paths (tab: find-path)
from coana import FindNeuronConnection
fc = FindNeuronConnection(dataset="male-cns:v0.9", sourceNeurons=["aMe12"],
                          targetNeurons=["PPL101"], output_dir="/abs/output/paths",
                          min_synapse_num=3, max_interlayer=2, filter_by="bodyId",
                          skip_bodyId=True, output_format="csv", use_cache=True,
                          showfig=False)
fc.InitializeNeuronInfo()
fc.FindAllPath(forward_only=True, find_reciprocal=False)

# Step 2: PlotPath (tab: visualization) — pick the newest path CSV
from vispath_pkg import VisualizePath
vp = VisualizePath(path_file="/abs/output/paths/path_data.csv",
                   output_folder="/abs/output/network",
                   network_layout="hierarchical", showfig=False)
vp.visualize()
```

## Morphology → connectivity find-similar → 3D skeleton (chained tabs)

```python
# Step 1: morphological similarity (Morphology tab → Find Similar)
from morphology import MorphologyComparer
comparer = MorphologyComparer(dataset="male-cns:v0.9", query="aMe12",
                              output_dir="/abs/output/similar_morph",
                              candidate_source="auto", use_cache=True,
                              visualize_top_n=5, verbose=True)
results = comparer.find_similar()

# Step 2: connectivity-profile similar search (Connectivity tab → Find Homolog,
# Target = Source for the intra-dataset search)
from comparison.profile_comparator import HomologFinder
finder = HomologFinder(source_dataset="male-cns:v0.9", target_dataset="male-cns:v0.9",
                       source="aMe12", output_dir="/abs/output/similar_profile",
                       use_cache=True, verbose=True)
finder.find_homologs_fast()

# Step 3: render top candidates (tab: visualization)
from visualize_skeleton import VisualizeSkeleton
vs = VisualizeSkeleton(dataset="male-cns:v0.9",
                       neuron_layers=[["aMe12"], ["aMe10", "aMe9"]],
                       output_dir="/abs/output/skeleton",
                       skeleton_mode="line", show_fig=False, skip_synapse=True)
vs.plot_neurons()
```

## Cross-dataset comparison → PlotPath

```python
# Step 1: comparison (tab: inter-dataset) → conserved-path CSV
from comparison import ComparisonParameters, ComparisonAnalyzer
params = ComparisonParameters(datasets=["hemibrain:v1.2.1", "male-cns:v0.9"],
                              source_neurons=["aMe12"], target_neurons=["PPL101"],
                              output_folder="/abs/output/comparison",
                              max_interlayer=2, thresholds=[1, 3, 5, 10],
                              skip_bodyId=True)
analyzer = ComparisonAnalyzer(params, verbose=True)
analyzer.run_comparison()
analyzer.export_results()

# Step 2: render the conserved path (tab: visualization)
from vispath_pkg import VisualizePath
vp = VisualizePath(path_file="/abs/output/comparison/<conserved>.csv",
                   output_folder="/abs/output/comparison/net",
                   network_layout="distributed", showfig=False)
vp.visualize()
```

## Shortest paths → network (find_shortest chained to find_network)

```python
from coana import FindNeuronConnection
fc = FindNeuronConnection(dataset="male-cns:v0.9", sourceNeurons=["aMe12"],
                          targetNeurons=["PPL101"], output_dir="/abs/output/shortest",
                          min_synapse_num=3, max_interlayer=2, filter_by="bodyId",
                          skip_bodyId=True, output_format="csv", showfig=False)
fc.InitializeNeuronInfo()
fc.FindShortestPath(forward_only=True, find_reciprocal=False)
# then a FindNetwork query over the same group for the enriched graph
fc2 = FindNeuronConnection(dataset="male-cns:v0.9", sourceNeurons=["aMe12"],
                           targetNeurons=["aMe12"], output_dir="/abs/output/network",
                           min_synapse_num=3, output_format="csv", showfig=False)
fc2.InitializeNeuronInfo()
fc2.FindNetwork()
```

## NeuronBridge lines → FlyLight images (nb_find_lines → FlyLight)

```python
# Step 1: find driver lines (tab: nb-find-lines)
from neuronbridge_finder import NeuronBridgeFinder
finder = NeuronBridgeFinder(verbose=True)
finder.find_lines_batch(queries=["aMe12"], dataset="male-cns:v0.9",
                        output_dir="/abs/output/nb_lines", match_type="both")

# Step 2: download FlyLight images for the top lines (tab: flylight)
from flylight_downloader import FlyLightDownloader
dl = FlyLightDownloader(output_dir="/abs/output/flylight", formats=["png"],
                        image_types=["mip"], max_workers=4, verbose="pbar")
dl.download(line_name=["SS00001"], output_dir="/abs/output/flylight", max_files=20,
            generate_summary=["pdf"])
```

## Guidance

- Keep each step's output in a persistent folder; pass the actual produced path
  (find the newest CSV rather than guessing a filename).
- Prefer `showfig=False` until all artifacts are validated.
- When chaining across datasets (comparison), ensure type mapping / labels are
  consistent (`auto_type_mapping` / `LabelMapper`).
- Without the UI, there is no automatic file handoff — record the path of each
  step's output and pass it to the next.
