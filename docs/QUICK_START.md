# Quick Start Guide

This guide provides a quick overview of how to perform common tasks using the Drosophila Connectome Analysis Toolkit (DROCAT) v4.5.0.

## Prerequisites

1. **Install the toolkit** - See [Installation Guide](INSTALLATION.md)
2. **Set up authentication** - Add your API tokens directly to `config.json` (`tokens` section) - a GitHub-pulled copy is yours to modify; the gitignored `config_local.json` only fills entries left empty there:
   ```json
   {
     "tokens": {
       "neuprint": "your_neuprint_token_here",
       "cave": "your_cave_token_here"
     },
     "envs": {
       "4.5.0": ""
     }
   }
   ```
   
   **Token Requirements:**
   - `NEUPRINT_TOKEN`: **Required** for NeuPrint datasets (hemibrain, male-cns, MANC, optic-lobe)
   - `CAVE_TOKEN`: **Optional for local conversion**; needed when a FAFB workflow uses CAVE API fetching or skeleton fallback (BANC API is unsupported)
   - NeuronBridge API requires **no authentication**
   
   📖 **[Authentication Setup](INSTALLATION.md#4-authentication)**

## Agent-first direct mode (no UI)

After installation, ask an agent to use
[`skills/drocat-usage/SKILL.md`](../skills/drocat-usage/SKILL.md) for direct script
execution. It can run the focused backend script, inspect outputs, and patch a
small source contract without loading the whole repository:

> Fetch https://raw.githubusercontent.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/v4.5.0/skills/drocat-usage/SKILL.md and follow it to install and use the DROCAT v4.5.0 direct-analysis skill for this repository, then finish the requested analysis end-to-end without opening the UI.

```bash
python skills/drocat-usage/scripts/run_direct.py \
  --conda-env drocat-4.5.0 \
  --script scripts/FindPath.py \
  --dry-run
```

Remove `--dry-run` after reviewing the resolved command.

## 1. Connection Path Finding

Find all paths between source and target neurons up to a specified number of hops (layers).

**Script:** `scripts/FindPath.py`
**Class:** `FindNeuronConnection` (in `src/coana.py`)

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    # Token automatically loaded from config.json
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['L2'],         # Supports bodyId, type, instance with regex
    targetNeurons=['l-LNv'],
    min_synapse_num=3,
    max_interlayer=3,
    output_dir='./output_data'
)

fc.InitializeNeuronInfo()
fc.FindPath()
```

**v4.4.0**: Priority-based search (bodyId → type → instance), accepts both int and string bodyIds!

**v4.5.0**: `FindAllPath()` runs on the **StrongestFirst** pipeline — paths are
emitted strongest-first under a path budget (`max_paths_bodyid`, 0 = auto →
1,000,000) with a reported cutoff **τ** (a budgeted run at τ equals a complete
run at `min_synapse_num = τ`), and the discovery cone can be capped with the
Edge Budget (`graph_edge_limit_bodyid`). Few-target queries rarely need these
knobs — the built-in lossless pruning already does most of the work.

## 2. Direct Connections

Find direct connections between a set of neurons (1-hop).

**Script:** `scripts/FindDirect.py`
**Class:** `FindNeuronConnection`

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    # Token automatically loaded from config.json
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['aMe12'],
    targetNeurons=['KCg-d'],
    min_synapse_num=1,
    output_dir='./output_data'
)

fc.InitializeNeuronInfo()
fc.FindDirectConnections()
```

## 2.5. Local FAFB/BANC Analysis (NEW! local-cache fast path)

Use the matching local-release source for fast FAFB or standalone BANC analysis.

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    dataset='flywire_FAFB_v783',  # Automatically uses local cache if available
    sourceNeurons=['CB0038'],      # FAFB/local-release root IDs
    targetNeurons=['LPLC2'],
    min_synapse_num=3,
    use_cache=True,               # Enable local caching
    output_dir='./output_data'
)

fc.InitializeNeuronInfo()
fc.FindAllPath()
```

📖 **[FAFB Integration Guide](FAFB_INTEGRATION.md)** and
**[BANC Integration Guide](BANC_INTEGRATION.md)** - setup instructions for the
two local-release sources

## 3. Network Visualization (Sankey, Heatmap)

Visualize the paths found in step 1 using Sankey diagrams, heatmaps, and network graphs.

**Script:** `scripts/PlotPath.py`
**Class:** `VisualizePath` (in `vispath-subproject/src/vispath_pkg/vispath.py`)

**v4.4.0**: NT edge groups, custom groups, improved default opacity (50% edges, 100% nodes)!

```python
from vispath_pkg import VisualizePath

# Initialize visualization with path data
vp = VisualizePath(
    path_file='./output_data/allpaths_.../path_data.xlsx', # Path to your FindPath output file
    sheet_name=0, # or 'path_bodyId' or 'path_type'
    output_folder='./visualization_output',
    source_color='#1f77b4',
    intermediate_color='#2ca02c',
    target_color='#d62728',
    network_layout='hierarchical', # 'hierarchical', 'spring', 'circular', 'distributed'
    showfig=True
)

# Generate all visualizations (Sankey, Network, Heatmap)
conn_df, G = vp.visualize()
```

📖 **[Network Features Guide](visualizations/VisualizePath_Network_Features.md)** - NT grouping, custom groups, export/import

## 4. 3D Skeleton Visualization

Visualize neurons and their synapses in 3D space.

**Script:** `scripts/plot3dSkeleton.py`
**Class:** `VisualizeSkeleton` (in `src/visualize_skeleton.py`)

```python
from visualize_skeleton import VisualizeSkeleton

vs = VisualizeSkeleton(
    token='YOUR_NEUPRINT_TOKEN',
    dataset='hemibrain:v1.2.1',
    neuron_layers=['VA1d_adPN', 'LHCENT3', 'MBON01'], # Layers to plot
    min_synapse_num=10,
    skeleton_mode='tube',
    synapse_mode='cone',
    brain_mesh='native',
    output_dir='./output_data'
)

vs.plot_neurons()
```

## 5. Cross-Dataset Comparison

Compare connectivity pathways across different datasets (e.g., Hemibrain vs. FAFB vs. Male CNS).

**Script:** `scripts/InterDatasetComparator.py`
**Classes:** `ComparisonParameters`, `ComparisonAnalyzer` (in `src/comparison/`)

```python
from comparison.comparison_parameters import ComparisonParameters
from comparison.comparison_analyzer import ComparisonAnalyzer

# Define parameters
params = ComparisonParameters(
    datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],
    source_neurons=['MBON14.*_R'],
    target_neurons=['KCg-d.*_R'],
    max_interlayer=2,
    thresholds=[1, 3, 5, 10],
    output_folder='./comparison_output'
)

# Run analysis
analyzer = ComparisonAnalyzer(params)
analyzer.run_comparison()
analyzer.generate_report()
```

## 6. Homolog Finding

Find potential homologs of a neuron in another dataset based on connectivity profiles.

**Script:** `scripts/FindHomologs.py`
**Class:** `HomologFinder` (in `src/comparison/profile_comparator.py`)

```python
from comparison.profile_comparator import HomologFinder

finder = HomologFinder(
    token='YOUR_NEUPRINT_TOKEN',
    source='aMe12',
    source_dataset='banc_v626',
    target_dataset='flywire_FAFB_v783',
    output_dir='./homolog_results',
    similarity_metric='jaccard', # or 'cosine', 'rank_union'
    visualize_skeleton=True      # Visualize top candidates
)

# Run fast search
results = finder.find_homologs_fast()
```

For detailed information on the output files generated by these scripts, please refer to [OUTPUT_FILES.md](OUTPUT_FILES.md).

## 7. Empty editable network canvas

Create a blank Cytoscape HTML canvas for direct node/edge drawing:

```python
from vispath_pkg import VisualizePath

vp = VisualizePath(
    path_file=None,
    output_folder='./local_data/empty_network',
    generate_empty_network=True,
    showfig=True,  # open the generated HTML in a new browser tab
)
vp.visualize()
```

Use `showfig=False` in unattended agent runs and open the returned
`*_network.html` file manually. See
[`skills/drocat-usage/SKILL.md`](../skills/drocat-usage/SKILL.md) for direct-run
recipes and output validation.
