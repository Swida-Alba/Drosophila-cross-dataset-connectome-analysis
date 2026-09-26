# VisualizePath - Neural Pathway Visualization Toolkit

A standalone visualization toolkit for neural pathways discovered through connectome analysis.

## Overview

VisualizePath provides interactive visualization capabilities for neural pathways, including:
- **Sankey diagrams** - Flow-based pathway visualization
- **Interactive heatmaps** - Connection matrices with hierarchical clustering
- **Network graphs** - Interactive network visualization with Cytoscape.js
- **Excel export** - Data export for further analysis

## Features

✅ **Fully Standalone** - All visualization features work without neuroscience dependencies  
✅ **Interactive Sankey Diagrams** - Flow-based pathway visualization with edge filtering  
✅ **Network Graphs** - Multiple layout algorithms (hierarchical, spring, circular)  
✅ **Connection Heatmaps** - With hierarchical clustering and metric toggle  
✅ **Flexible Input** - CSV/Excel edge-lists, path-based formats, or connection matrices  
✅ **Custom Styling** - Node colors, edge colors, fonts, sizes  
✅ **NT (Neurotransmitter) Support** - Color-coded edges by NT type with group selection  
✅ **Custom Groups** - Create custom groups from selected elements  
✅ **Export/Import** - Save and restore complete graph states including custom groups  

## Installation

### Recommended: Using Conda Environment

```bash
# Create environment with Python 3.11
conda create -n vispath python=3.11 -y
conda activate vispath

# Install the package
cd vispath-subproject
pip install -e .
```

### As a standalone package:

```bash
pip install git+https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis.git#subdirectory=vispath-subproject
```

### As part of the main project:

```bash
pip install git+https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis.git
```

### Development installation:

```bash
cd vispath-subproject
pip install -e .
```

## Quick Start

```python
from vispath_pkg import VisualizePath

# Create visualizer
vp = VisualizePath(
    path_file='paths.xlsx',
    output_folder='./output',
    showfig=True
)

# Generate all visualizations (heatmap + Sankey + network)
vp.visualize()

# Or generate only specific visualizations
vp.visualize(plot_heatmap=True, plot_Sankey=False, plot_network=True)

# Connection matrix input example
import pandas as pd
import numpy as np

# Create a connection matrix
matrix = pd.DataFrame(
    np.random.poisson(3, (5, 5)),
    index=['A', 'B', 'C', 'D', 'E'],
    columns=['A', 'B', 'C', 'D', 'E']
)

vp = VisualizePath(matrix, showfig=True)
vp.visualize()  # Automatically detects matrix format
```

## Dependencies

### Core:
- numpy>=1.20.0,<2.0.0 (constrained for pandas compatibility)
- pandas>=1.3.0,<2.0.0
- polars>=1.0.0 (efficient DataFrames)
- plotly>=5.0.0
- scipy>=1.7.0
- openpyxl>=3.0.0

**Important:** numpy is constrained to <2.0.0 for binary compatibility with pandas 1.x.

**Note:** NetworkX is no longer required. Graph operations use the built-in `FastGraph` module for improved performance and reduced dependencies.

### Optional:
- PyQt5>=5.15.0 (for fast file dialogs)

## Data Formats

VisualizePath supports three input formats:

### 1. Connection Matrix (Numeric matrix):

**Format:**
- 2D numeric DataFrame with rows as sources and columns as targets
- Index and column names represent node names
- Cell values represent connection weights
- Can be square (NxN) or rectangular (NxM)

**Requirements:**
- At least 2x2 matrix
- All values must be numeric
- Zero or NaN values are treated as no connection

**Example:**

|          | Neuron_1 | Neuron_2 | Neuron_3 |
| -------- | -------- | -------- | -------- |
| Neuron_A | 10       | 0        | 5        |
| Neuron_B | 15       | 8        | 0        |
| Neuron_C | 0        | 12       | 3        |

**Auto-generation:** If index/columns are numeric, node names are auto-generated as `N0`, `N1`, etc.

**Note:** Connection matrices are automatically converted to edge-list format internally.

### 2. Path-based format (Multi-hop paths):

**Required columns:**
- `path_block` (str): Path as "A -> B -> C -> D" format
- `weights` (list): Weights for each hop, e.g., `[10, 20, 15]`

**Optional columns:**
- `connection_ratios` (list): Connection ratios for each hop
- `traversal_probabilities` (list): Traversal probabilities for each hop
- `layer` (int): Layer number for the path

**Example:**

| path_block  | weights | connection_ratios | traversal_probabilities |
| ----------- | ------- | ----------------- | ----------------------- |
| A -> B -> C | [10, 5] | [0.5, 0.3]        | [0.8, 0.6]              |
| A -> D -> C | [15, 8] | [0.6, 0.4]        | [0.9, 0.7]              |

### 3. Edge-list format (Direct connections):

**Required columns (flexible naming):**
- Source: `source`, `from`, `pre`, or `*_pre` (e.g., `bodyId_pre`, `type_pre`)
- Target: `target`, `to`, `post`, or `*_post` (e.g., `bodyId_post`, `type_post`)
- Weight: `weight`, `weights`, `synapse_count`, or `count`

**Optional columns:**
- `color`: Edge color in hex (#RRGGBB) or rgba format
- Any numeric column (e.g., `ratio`, `probability`) will be preserved

**Example:**

| source | target | weight | ratio | probability |
| ------ | ------ | ------ | ----- | ----------- |
| A      | B      | 10     | 0.5   | 0.8         |
| B      | C      | 5      | 0.3   | 0.6         |
| D      | C      | 8      | 0.4   | 0.7         |

**Note:** Edge-list format is automatically converted to path-based format internally.

## Output

- `{basename}_Sankey.html` - Interactive Sankey diagram
- `{basename}_heatmap.html` - Interactive heatmap
- `{basename}_network.html` - Interactive network graph
- `{basename}_data.xlsx` - Processed data

## Network Features

### NT (Neurotransmitter) Edge Groups
- Edges are automatically grouped by neurotransmitter type (ACH, GABA, GLUT, DA, SER, OCT)
- Select NT groups from the dropdown to color/style all edges of that type
- NT type is shown in hover tooltip with color coding

### Custom Groups
- Select nodes/edges and create custom groups for batch editing
- Groups persist across export/import cycles
- Access via "Custom Groups" section in the left panel

### Group Selection
| Group Type         | Elements                            | Default Opacity |
| ------------------ | ----------------------------------- | --------------- |
| Source Nodes       | All source (input) neurons          | 100%            |
| Intermediate Nodes | All intermediate neurons            | 100%            |
| Target Nodes       | All target neurons                  | 70%             |
| Positive Edges     | All positive weight edges           | 50%             |
| Negative Edges     | All negative weight edges           | 100%            |
| NT Groups          | ACH, GABA, GLUT, DA, SER, OCT edges | Inherited       |
| Custom Groups      | User-defined selections             | Custom          |

### Keyboard Shortcuts
| Key          | Action                                  |
| ------------ | --------------------------------------- |
| H            | Hide selected nodes                     |
| E            | Hide selected edges                     |
| L            | Toggle label position                   |
| Right-click  | Hide node/edge (or delete in edit mode) |
| Double-click | Highlight connected edges               |

### Export/Import
- **Export Graph**: Saves nodes, edges, positions, styles, group settings, and custom groups
- **Import Graph**: Restores complete graph state including custom groups
- **Export Layout**: Saves the full view state (positions, colors + alpha, groups, filters, controls) as a v2 JSON; legacy positions-only files still import
- **Save/Load**: The same full view state in browser localStorage, auto-restored when the file reopens

## Documentation

For full documentation, see:
- [VisualizePath Quick Reference](../docs/visualizations/VisualizePath_QuickRef.md)
- [Network Features Guide](../docs/visualizations/VisualizePath_Network_Features.md)
- [Main Project README](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis)
Swida Alba & Copilot

## Part of

[DROCAT](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis)
