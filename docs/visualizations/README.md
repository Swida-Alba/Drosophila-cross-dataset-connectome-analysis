# Visualization Documentation Overview

Comprehensive guides for all visualization types in the DROCAT toolkit.

## Quick Reference

| Guide                                                         | Description                                          |
| ------------------------------------------------------------- | ---------------------------------------------------- |
| [VisualizePath Quick Reference](./VisualizePath_QuickRef.md)  | Quick start guide for VisualizePath                  |
| [**Interaction Guide**](./VisualizePath_Interaction_Guide.md) | **Complete keyboard/mouse/button reference**         |
| [Network Features Guide](./VisualizePath_Network_Features.md) | Complete guide to network features, groups, NT edges |
| [Heatmap Guide](./Heatmap_Guide.md)                           | Interactive connection matrix visualization          |
| [Network Guide](./Network_Guide.md)                           | Basic network visualization                          |
| [Sankey Guide](./Sankey_Guide.md)                             | Flow-based diagram guide                             |
| [3D Skeleton Guide](./3D_Skeleton_Guide.md)                   | 3D anatomical visualization                          |
| [Skeleton Pipeline Report](../technical/SKELETON_DATA_PIPELINE.md) | Skeleton fetch/cache/simplification + templates/ROIs |
| [Skeleton Pipeline Diagram](./skeleton_pipeline.html)         | Editorial data-flow rendering of the pipeline        |

## Recent Updates (2026)

- **[Interaction Guide](./VisualizePath_Interaction_Guide.md)** - Complete interactive controls reference:
  - Keyboard shortcuts (H, E, L, Shift+Click, Double-Click)
  - Mouse actions for all visualization types
  - Edit Mode operations
  - Troubleshooting common issues
- **[Network Features Guide](./VisualizePath_Network_Features.md)** - Complete network documentation including:
  - NT (Neurotransmitter) edge groups with color coding
  - Custom group creation and management
  - Export/import with group preservation
  - Enhanced group selection controls
  - Fixed reset colors functionality

## Feature Reference

### Network Features
| Feature                  | Documentation                                                                                 |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| **Interactive Controls** | [Interaction Guide](./VisualizePath_Interaction_Guide.md)                                     |
| NT Edge Groups           | [Network Features Guide](./VisualizePath_Network_Features.md#nt-neurotransmitter-edge-groups) |
| Custom Groups            | [Network Features Guide](./VisualizePath_Network_Features.md#custom-groups)                   |
| Export/Import            | [Network Features Guide](./VisualizePath_Network_Features.md#exportimport)                    |
| Edit Mode                | [Network Features Guide](./VisualizePath_Network_Features.md#edit-mode)                       |
| Layout Algorithms        | [Layout Algorithms Quick Ref](./LayoutAlgorithms_QuickRef.md)                                 |
| Edge Filtering           | [Network Edge Filter](./NETWORK_EDGE_FILTER.md)                                               |

### Heatmap Features
| Feature         | Documentation                                            |
| --------------- | -------------------------------------------------------- |
| Clustering      | [Heatmap Clustering](./HEATMAP_CLUSTERING_QUICKREF.md)   |
| Custom Ordering | [Custom Ordering](./CUSTOM_HEATMAP_ORDERING_QuickRef.md) |
| UI Improvements | [Heatmap UI](./HEATMAP_UI_IMPROVEMENT.md)                |

### Sankey Features
| Feature              | Documentation                                         |
| -------------------- | ----------------------------------------------------- |
| Interactive Controls | [Sankey Controls](./SANKEY_CONTROLS_QUICKREF.md)      |
| Multi-Selection      | [Sankey Multi-Selection](./SANKEY_MULTI_SELECTION.md) |

## Available Visualizations

### 1. [Heatmap Visualization](./Heatmap_Guide.md)
**Purpose**: Interactive connection matrix visualization  
**Best for**: Comparing connection patterns, hierarchical clustering, quantitative analysis  
**Output**: Interactive HTML with color-coded matrix  
**Key features**:
- Multiple scaling methods (linear, log, sqrt)
- Hierarchical clustering with 4 algorithms
- 15+ color schemes including custom scales
- Interactive drag-and-drop reordering
- Cell value display with auto-contrast
- Real-time filtering and customization

**When to use**:
- Analyzing pairwise connectivity between neuron populations
- Finding connection patterns and clusters
- Comparing connection strengths across groups
- Publication-ready matrices with statistical rigor

---

### 2. [Network Visualization](./Network_Guide.md)
**Purpose**: Interactive graph-based connectivity display  
**Best for**: Exploring topology, pathways, and network structure  
**Output**: Interactive HTML with draggable nodes and edges  
**Key features**:
- 4+ layout algorithms (hierarchical, force-directed, circular, distributed)
- Real-time node and edge customization
- **NT (Neurotransmitter) edge groups** - Color-code by ACH, GABA, GLUT, etc.
- **Custom groups** - Create groups from selected elements
- Edit mode for manual graph construction
- Multiple connection metrics (weight, ratio, probability)
- Edge width scaling and filtering
- Export/import graph state including custom groups

**When to use**:
- Visualizing neural pathways and circuits
- Exploring network topology
- Identifying hubs and bottlenecks
- Analyzing neurotransmitter distribution
- Interactive presentations and demonstrations

---

### 3. [Sankey Diagram](./Sankey_Guide.md)
**Purpose**: Flow-based visualization of connection magnitude  
**Best for**: Understanding information flow and layer structure  
**Output**: Interactive HTML with flowing links  
**Key features**:
- Flow width proportional to connection strength
- Multiple layout modes (snap, freeform)
- Automatic layer detection
- Node ordering optimization
- Multi-selection and custom coloring
- Metric toggling (synapse count, ratio, probability)

**When to use**:
- Showing signal flow through neural layers
- Emphasizing connection magnitude
- Visualizing layered pathway architectures
- Comparing flow patterns across conditions

---

### 4. [3D Skeleton Visualization](./3D_Skeleton_Guide.md)
**Purpose**: Three-dimensional anatomical rendering  
**Best for**: Spatial understanding, morphology analysis, anatomical context  
**Output**: Interactive 3D HTML or static images  
**Key features**:
- Accurate neuron skeleton morphology
- Brain region (ROI) meshes with transparency
- Synapse locations in 3D space
- Multiple camera views and rotation
- Distance measurements and volume calculations
- Animation support

**When to use**:
- Understanding spatial relationships between neurons
- Visualizing neuron morphology
- Correlating connectivity with brain anatomy
- Creating publication figures with anatomical context

---

## Choosing the Right Visualization

### By Analysis Goal

| Goal                             | Primary Visualization     | Secondary   |
| -------------------------------- | ------------------------- | ----------- |
| **Compare connection strengths** | Heatmap                   | -           |
| **Find connection patterns**     | Heatmap (with clustering) | Network     |
| **Explore pathway structure**    | Network                   | Sankey      |
| **Show information flow**        | Sankey                    | Network     |
| **Understand anatomy**           | 3D Skeleton               | -           |
| **Present to non-experts**       | Sankey or Network         | 3D Skeleton |
| **Statistical analysis**         | Heatmap                   | -           |
| **Interactive exploration**      | Network                   | Heatmap     |
| **Publication figure**           | Any (depends on message)  | -           |

### By Data Type

| Data Type                         | Recommended Visualization           | Input Format                   |
| --------------------------------- | ----------------------------------- | ------------------------------ |
| **Dense connectivity matrix**     | Heatmap                             | Connection matrix or edge-list |
| **Sparse pathways**               | Network or Sankey                   | Edge-list or path-based        |
| **Multi-hop paths**               | Sankey (flow) or Network (topology) | Path-based                     |
| **Single neuron morphology**      | 3D Skeleton                         | N/A                            |
| **Population-level connectivity** | Heatmap or Sankey                   | Connection matrix              |
| **Circuit motifs**                | Network                             | Edge-list                      |
| **Layered feedforward**           | Sankey                              | Path-based                     | ### By Audience |

| Audience                 | Best Choice           | Reason                                |
| ------------------------ | --------------------- | ------------------------------------- |
| **Researchers**          | Network or Heatmap    | Detailed, interactive exploration     |
| **General audience**     | Sankey or 3D Skeleton | Intuitive flow/spatial representation |
| **Statistical analysis** | Heatmap               | Quantitative, matrix-based            |
| **Presentations**        | Network or Sankey     | Interactive, visually engaging        |
| **Publications**         | Any                   | All support high-quality export       |

---

## Workflow Integration

### Complete Analysis Pipeline

```
1. Path Finding (FindAllPath)
   ↓
2. Generate all visualizations:
   - Heatmap: Quantitative overview
   - Network: Topology exploration  
   - Sankey: Flow visualization
   ↓
3. Iterative refinement:
   - Identify patterns in heatmap
   - Explore details in network
   - Understand flow in Sankey
   ↓
4. Export publication figures
```

### Typical Workflow Example

```python
from vispath_pkg import VisualizePath
import pandas as pd

# Example 1: Path-based analysis
vp = VisualizePath(
    path_file='my_paths.xlsx',
    output_folder='./visualizations',
    showfig=True
)
# Generate all visualizations
vp.visualize()
# Generates:
# - network_*.html (interactive network)
# - sankey_*.html (flow diagram)
# - heatmap_*.html (connection matrix)

# Example 2: Connection matrix input
matrix_df = pd.read_csv('connectivity_matrix.csv', index_col=0)
vp = VisualizePath(matrix_df, showfig=True)
vp.visualize()
# Automatically detects matrix format and generates all visualizations

# Example 3: Generate only specific visualizations
vp = VisualizePath('edge_list.csv')
vp.visualize(
    plot_heatmap=True,   # Generate heatmap
    plot_Sankey=False,   # Skip Sankey
    plot_network=True    # Generate network
)
# Only generates heatmap and network, skips Sankey diagram

# Example 4: Edge-list with flexible column names
edge_df = pd.DataFrame({
    'bodyId_pre': [123, 456, 789],
    'bodyId_post': [456, 789, 123],
    'weight': [10, 15, 8]
})
vp = VisualizePath(edge_df)
vp.visualize()  # Automatically recognizes edge-list format
```

### Cross-Referencing Visualizations

1. **Start with Heatmap**: Identify interesting connection patterns
2. **Switch to Network**: Explore topology of highlighted connections
3. **Use Sankey**: Understand flow magnitude through identified paths
4. **Verify in 3D**: Check spatial plausibility

---

## Common Features Across Visualizations

All visualizations support:

### Export Options
- **PNG**: Raster images with adjustable resolution
- **SVG**: Vector graphics for publications
- **HTML**: Interactive, shareable visualizations
- **Data**: Export current state for reproducibility

### Customization
- **Colors**: Comprehensive color controls
- **Size**: Adjustable dimensions for different uses
- **Labels**: Customizable text and font sizes
- **Filters**: Hide/show elements based on criteria

### Interactivity
- **Selection**: Click or Shift+click for multi-select
- **Hover**: Detailed information on hover
- **Zoom/Pan**: Explore large datasets
- **Settings**: Save/load custom configurations

### Performance
- Optimized for datasets up to 1000 neurons
- Progressive loading for large datasets
- Browser-based (no installation needed)
- Works offline after initial generation

---

## Quick Reference Card

### Heatmap
```python
from statvis import plot_stat
plot_stat(conn_df, metric='weight', showfig=True)
```
**Keys**: Scale buttons, Clustering toggle, Custom colors

### Network  
```python
from vispath import VisualizePath
vp = VisualizePath(path_file='data.xlsx', network_layout='hierarchical')
```
**Keys**: H (hide), L (labels), Shift (multi-select)

### Sankey
```python
vp = VisualizePath(path_file='data.xlsx')  # Auto-generates Sankey
```
**Keys**: Drag nodes, Metric buttons, Layout dropdown

### 3D Skeleton
```python
from navis_related import plot_navis_3d
plot_navis_3d(bodyIds=[...], show_rois=True)
```
**Keys**: Drag (rotate), Scroll (zoom), Click (select)

---

## Advanced Topics

### Combining Visualizations
- [Multi-panel figures](#)
- [Synchronized interactions](#)
- [Comparative analysis](#)

### Customization
- [Color schemes and themes](./CUSTOM_COLORS_GUIDE.md)
- [Layout algorithms](./AdvancedLayoutAlgorithms.md)
- [Export optimization](#)

### Performance
- [Large dataset handling](#)
- [Memory optimization](#)
- [Browser compatibility](#)

---

## Troubleshooting

### Common Issues

| Issue                  | Visualization   | Solution                                        |
| ---------------------- | --------------- | ----------------------------------------------- |
| Slow performance       | All             | Reduce data size, filter weak connections       |
| Overlapping labels     | Network, Sankey, Heatmap | Increase plot size, adjust font size (a heatmap fits its own canvas when 🏷️ Show Labels reveals the axis names) |
| Heatmap axis labels missing | Heatmap  | A map over 100 rows/columns opens with the tick labels hidden — press 🏷️ **Show Labels** (see `docs/TROUBLESHOOTING.md`) |
| Colors not visible     | Heatmap         | Adjust colorscale, check data range             |
| Export fails           | All             | Check browser permissions, try different format |
| Clustering unavailable | Heatmap         | Matrix too small or contains NaN                |
| Nodes overlap          | Network         | Try different layout algorithm                  |

### Getting Help

1. Check individual visualization guides
2. Review [troubleshooting sections](#) in each guide
3. Examine [examples directory](../../archive/examples/)
4. Check [GitHub issues](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/issues)

---

## Examples

See `examples/` directory for:
- `example_heatmap.py`: Complete heatmap workflow
- `example_network.py`: Network visualization with customization
- `example_sankey.py`: Sankey diagram generation
- `example_3d.py`: 3D skeleton rendering
- `example_combined.py`: Multi-visualization analysis

---

## Next Steps

1. **New users**: Start with [Network Guide](./Network_Guide.md) for intuitive introduction
2. **Quantitative analysis**: See [Heatmap Guide](./Heatmap_Guide.md)
3. **Flow analysis**: Read [Sankey Guide](./Sankey_Guide.md)
4. **Anatomical context**: Explore [3D Skeleton Guide](./3D_Skeleton_Guide.md)

---

## Related Documentation

- [Main README](../../README.md)
- [Installation Guide](../INSTALLATION.md)
- [Quick Start Guide](../QUICK_START.md)
- [Core Features](../core-features/)
