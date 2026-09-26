# Sankey Diagram Guide

Flow-based visualization showing the magnitude of connections between neuron layers.

## Overview

Sankey diagrams visualize neural pathways as flowing connections where:
- **Nodes** represent neurons or neuron types
- **Links** are connections with width proportional to strength
- **Flow direction** shows signal progression (left to right)
- **Colors** distinguish source, intermediate, and target populations

## Example Input Data

### Required Format

Your edge list should contain these columns:

| Column | Description | Example Values | Required |
|--------|-------------|----------------|----------|
| `source` or `from` | Source neuron | 'KC_alpha', 'MBON01' | ✅ Yes |
| `target` or `to` | Target neuron | 'MBON03', 'DAN_a' | ✅ Yes |
| `weight` | Connection strength | 10, 25, 150 | ✅ Yes |
| `ratio` | Connection ratio | 0.05, 0.15, 0.8 | ❌ Optional |
| `probability` | Traversal probability | 0.1, 0.5, 0.9 | ❌ Optional |

### Example Input Files

**From FindAllPath results** (recommended):
- Path: Output from `scripts/FindPath.py`
- File: `connection_data/*/allpaths_*/path_info.xlsx` (use sheet 'path_type')
- Example: [`examples/example_paths.csv`](../../archive/examples/data/example_paths.csv)

**Simple edge-list format**:
```csv
source,target,weight
KC_a,MBON01,25
KC_a,MBON03,10
KC_b,MBON01,50
MBON01,DAN_a,15
MBON03,DAN_b,8
```

**Multi-layer pathways**:
```csv
source,target,weight,layer
KC_alpha,LHN1,25,1
KC_alpha,LHN2,15,1
LHN1,MBON03,30,2
LHN2,MBON03,12,2
MBON03,DAN,20,3
```

**Example files provided**:
- [`examples/example_neuron_network.csv`](../../archive/examples/data/example_neuron_network.csv) - Multi-layer network
- [`examples/simple_network_data.csv`](../../archive/examples/data/simple_network_data.csv) - Basic format
- Output from [`scripts/PlotPath.py`](../../scripts/PlotPath.py)

**Note**: Same input files work for both Network and Sankey visualizations.

## Quick Start

### Basic Usage

```python
from vispath import VisualizePath

# Create Sankey diagram from path data
vp = VisualizePath(
    path_file='path_results.xlsx',
    output_folder='./visualizations',
    showfig=True
)

conn_df, G = vp.visualize()
```

The Sankey diagram is automatically generated as `sankey_*.html` alongside the network visualization.

## Key Features

### 1. Connection Metrics

Toggle between different strength measures:

- **Synapse Count (weight)**: Total synapses in connection
- **Connection Ratio**: Proportion of source neuron's output
- **Traversal Probability**: Likelihood of signal propagation

**Usage**: Click metric buttons at top of interface

**Effect**: Link widths update to reflect selected metric

**Not always synapses**: the metric label is supplied by the caller
(`edge_weight_label` / `metric_option_label`), so a diagram of cross-dataset
**type mapping** reads **Neuron count** and is titled for the mapping
(`Type mapping Sankey — <columns>`) rather than "pathway connections". The
same callers can drop the `(L<n>)` hop suffix from node labels
(`sankey_label_layers=False`) when a column is a dataset rather than a
pathway depth — which is the case for every type-mapping Sankey. Defaults
(pathway data) are unchanged.

**Opens offline**: the Sankey embeds the Plotly bundle in the file itself
(`include_plotlyjs=True`) rather than referencing `cdn.plot.ly`, so a
downloaded `_Sankey.html` renders with no network. The file is ~4.9 MB
instead of ~70 KB — the deliberate trade for an artifact reviewers open away
from their own machine. The Cytoscape **network** views still load their
libraries from a CDN, so those need network on first open.

### 2. Layout Modes

Control how the diagram is arranged:

#### Snap Layout (Default)
- **Behavior**: Nodes snap to vertical layers
- **Best for**: Clearly layered pathways
- **Alignment**: Automatic vertical positioning
- **Use case**: Source → Intermediate → Target flows

#### Freeform Layout
- **Behavior**: Nodes can be positioned anywhere
- **Best for**: Custom arrangements
- **Control**: Full manual positioning
- **Use case**: Emphasizing specific pathways

**Switching**: Use "Layout Mode" dropdown

### 3. Node Ordering

Control vertical order within layers:

#### Automatic Ordering
- **Top-to-Bottom**: Sorts by node name (alphabetical)
- **Bottom-to-Top**: Reverse alphabetical
- **Optimal**: Minimizes link crossings (algorithm-based)

#### Manual Ordering
- **Drag nodes** vertically within their layer
- **Snap to grid** for alignment
- **Relative positioning** maintained

**Usage**: Select ordering mode from dropdown, or enable freeform and drag

### 4. Interactive Controls

#### Node Manipulation
- **Click and drag**: Move node (freeform) or reorder (snap)
- **Hover**: Shows connection details
- **Click**: Highlights connected paths

#### Link Interaction
- **Hover**: Displays exact connection strength
- **Click**: Highlights full pathway
- **Multi-select**: Shift+click multiple links

#### Zoom & Pan
- **Scroll**: Zoom in/out
- **Drag background**: Pan view
- **Double-click**: Reset view

### 5. Color Customization

#### Node Colors (by role)
- **Source nodes**: First color in scheme
- **Intermediate nodes**: Second color
- **Target nodes**: Third color

#### Link Colors
- **Colored by source**: Links match source node color
- **Colored by target**: Links match target node color
- **Custom**: Set unique color per link

**Opacity Control**: Adjust transparency (0-100%) for nodes and links separately

### 6. Selection & Filtering

#### Selection
- **Single**: Click node or link
- **Multiple**: Shift+click to add
- **Clear**: Click background

#### Color Selected
1. Select elements
2. Choose color from palette
3. Apply to selection

#### Hide/Show
- **Hide selected**: Right-click or press H
- **Show hidden**: Toggle visibility button
- **Filter by weight**: Hide weak connections (see below)

### 7. Connection Filtering

Hide connections based on strength:

**Input format**: Comma-separated conditions
- `0`: Hide zero values
- `<5`: Hide less than 5
- `>100`: Hide greater than 100
- `>=10, <=50`: Show only range 10-50

**Examples**:
- `<3`: Hide weak connections (less than 3 synapses)
- `0, <10`: Hide zero and weak
- `>200`: Show only strong connections

**Effect**: Hidden links disappear, diagram reflows

### 8. Label Customization

#### Font Size
- **Default**: 10px
- **Range**: 6-20px
- **Control**: Font size slider
- **Affects**: All node labels

#### Label Position
- **Inside node**: Centered in node box
- **Outside node**: Next to node (auto-positioned)
- **Toggle**: Press L or use button

#### Show/Hide Labels
- **All labels**: Toggle button
- **Selected only**: Hide others temporarily
- **Custom**: Edit individual labels in freeform mode

### 9. Export Options

#### PNG Export
1. Set **Image Scale** (1-10x)
2. Click **PNG** button
3. Downloads as `sankey_YYYYMMDD_HHMMSS.png`

**Recommended scales**:
- Screen viewing: 1-2x
- Presentations: 2-3x
- Publications: 3-5x

#### SVG Export
1. Click **SVG** button
2. Downloads vector format
3. Infinite scaling, editable in Illustrator/Inkscape

#### Data Export
- Current state saved in Excel file
- Includes filtered connections
- Node positions preserved

### 10. Multi-Selection Colors

Apply colors to multiple elements at once:

1. **Select multiple** (Shift+click)
2. **Choose color** from palette
3. **Adjust opacity** with slider
4. **Click "Apply to Selected"**

**Use cases**:
- Highlight specific pathways
- Color-code functional groups
- Emphasize important connections

## Advanced Usage

### Custom Colors at Creation

Define colors programmatically:

```python
vp = VisualizePath(
    path_file='data.xlsx',
    source_color='#1f77b4',          # Blue
    intermediate_color='#2ca02c',    # Green
    target_color='#d62728',          # Red
    link_color='rgba(100,100,100,0.4)',  # Gray, 40% opacity
)
```

### Layered Pathway Visualization

For complex multi-hop pathways:

```python
# Data with multiple intermediate layers
# Format: source -> inter1 -> inter2 -> target

vp = VisualizePath(
    path_file='multilayer_paths.xlsx',
    network_layout='hierarchical',  # Also affects Sankey
)
```

**Result**: Sankey automatically detects layers and arranges nodes accordingly

### Type-Level vs Instance-Level

**Type-level** (neuron types):
```python
# Data aggregated by type
df = pd.DataFrame({
    'source': ['KC_alpha', 'KC_beta'],
    'target': ['MBON03', 'MBON03'],
    'weight': [150, 200]
})
```

**Instance-level** (individual neurons):
```python
# Individual neuron bodyIds
df = pd.DataFrame({
    'source': ['KC_alpha_001', 'KC_alpha_002'],
    'target': ['MBON03_001', 'MBON03_001'],
    'weight': [15, 12]
})
```

**Note**: Type-level is clearer for overview, instance-level for detailed analysis

### Filtering Before Visualization

```python
# Only include strong connections
strong_connections = df[df['weight'] > 10]

vp = VisualizePath(
    path_file=strong_connections,
    output_folder='./strong_only'
)
```

### Combining with Network View

Both visualizations are generated simultaneously:
- **Sankey**: Shows flow magnitude and layer structure
- **Network**: Shows detailed topology and graph structure

**Workflow**:
1. Use **Sankey** to identify major pathways
2. Use **Network** to explore detailed connectivity
3. Cross-reference between views

## Layout Algorithm Details

### Snap Layout

**Algorithm**: Layer-based positioning
- **X-coordinate**: Fixed by layer (source, intermediate, target)
- **Y-coordinate**: Optimized to minimize link crossings
- **Node height**: Proportional to total connection strength
- **Link width**: Proportional to connection weight

**Optimization**:
- Iterative crossing minimization
- Considers both incoming and outgoing connections
- Balances node spacing

### Freeform Layout

**Algorithm**: User-controlled
- **Initial positions**: Based on snap layout
- **Drag constraints**: None (full canvas)
- **Snap-to-grid**: Optional (10px grid)
- **Link routing**: Bezier curves, auto-adjusted

### Optimal Ordering

**Algorithm**: Barycenter heuristic
- **Barycenter**: Weighted average of connected node positions
- **Iterations**: Multiple passes until convergence
- **Objective**: Minimize weighted crossing count

**Formula**: 
```
barycenter(node) = Σ(connected_position * weight) / Σ(weight)
```

## Color Schemes

### Predefined Themes

**Cool Theme** (Default):
- Source: Blue (#4A90E2)
- Intermediate: Cyan (#50E3C2)
- Target: Light Green (#B8E986)

**Warm Theme**:
- Source: Red (#FF6B6B)
- Intermediate: Orange (#FFA500)
- Target: Gold (#FFD700)

**Purple Theme**:
- Source: Purple (#9C27B0)
- Intermediate: Lavender (#BA68C8)
- Target: Deep Pink (#FF1493)

**See PlotPath.py for complete list**

### Custom Theme Definition

```python
# Define your own color scheme
vp = VisualizePath(
    source_color='#your_hex_code',
    intermediate_color='#your_hex_code',
    target_color='#your_hex_code',
    link_color='rgba(r, g, b, alpha)',
)
```

**Tip**: Use colorbrewer2.org or coolors.co for palette generation

## Tips & Best Practices

### 1. Choosing Metrics
- **Synapse count**: Overall connectivity strength
- **Connection ratio**: Relative importance to source neuron
- **Traversal probability**: Functional significance

### 2. Layout Selection
- **Start with Snap** for automatic arrangement
- **Use Optimal ordering** to reduce clutter
- **Switch to Freeform** for custom emphasis

### 3. Visual Clarity
- **Hide weak connections** (<5 synapses) to reduce clutter
- **Use lower opacity** (30-40%) for links in dense diagrams
- **Increase font size** for better label readability
- **Color by pathway** to highlight specific routes

### 4. Publication Figures
- **Export SVG** for vector graphics
- **Use consistent colors** across figures
- **Snap layout with optimal ordering** is most publication-ready
- **Scale 3-4x for PNG** if vector format not accepted

### 5. Large Pathway Sets
- **Filter by probability** to show only likely paths
- **Aggregate by type** instead of individual neurons
- **Use hierarchical levels** to group similar neurons
- **Export multiple views** (filtered vs complete)

### 6. Comparison Across Conditions
- **Use same color scheme** for all diagrams
- **Keep metric consistent** (all synapse count or all ratio)
- **Export at same dimensions** for side-by-side comparison
- **Maintain same node ordering** if possible

## Performance Optimization

- **Reduce node count**: Aggregate to types
- **Filter weak connections**: Hide <5 synapses
- **Limit layers**: 3-4 layers optimal
- **Close unused panels**: Better rendering performance
- **Use Snap layout**: Faster than freeform for large diagrams

## Keyboard Shortcuts

- `H`: Hide/show selected elements
- `L`: Toggle label position
- `Shift`: Hold for multi-selection
- `Escape`: Clear selection
- `Ctrl/Cmd + Z`: Undo last move (freeform mode)

## Troubleshooting

**Links cross excessively**: Try "Optimal" ordering

**Nodes too small**: Increase plot height or reduce node count

**Can't move nodes**: Check that freeform mode is enabled

**Colors don't apply**: Make sure elements are selected before applying

**Export is blank**: Check browser compatibility, try different format

**Slow performance**: Reduce number of nodes/links, filter weak connections

**Labels overlap**: Adjust font size or use outside positioning

## Integration with Other Visualizations

### Sankey + Network
- Sankey shows **flow magnitude** and **layer structure**
- Network shows **detailed topology** and **alternative pathways**
- Use together for comprehensive understanding

### Sankey + Heatmap
- Sankey shows **pathway flow**
- Heatmap shows **pairwise connectivity matrix**
- Heatmap better for comparing connection patterns

### Sankey + 3D Skeleton
- Sankey shows **functional connectivity**
- 3D shows **anatomical positions**
- Correlate flow with spatial locations

## Related Documentation

- [Network Visualization Guide](./Network_Guide.md)
- [Custom Colors Guide](./CUSTOM_COLORS_GUIDE.md)
- [Sankey Interactive Controls](./SANKEY_INTERACTIVE_CONTROLS.md)
- [Multi-Selection Colors](./MULTI_SELECTION_COLOR_FEATURE.md)
