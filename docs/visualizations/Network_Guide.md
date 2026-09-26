# Network Visualization Guide

Interactive network graph visualization for exploring neural pathways and connectivity structures.

## Overview

The network visualization displays connectivity as an interactive graph where:
- **Nodes** represent neurons (colored by role: source, intermediate, target)
- **Edges** represent connections (thickness = synapse count, ratio, or probability)
- **Layout algorithms** arrange nodes to reveal structure
- **Interactive controls** enable exploration, editing, and customization

## Example Input Data

### Required Format

Your edge list should contain these columns:

| Column | Description | Example Values | Required |
|--------|-------------|----------------|----------|
| `source`, `from`, or `*_pre` | Source neuron | 'KC_alpha', 'PN1', 'L3_R' | ✅ Yes |
| `target`, `to`, or `*_post` | Target neuron | 'MBON03', 'LN1', 'Mi1_R' | ✅ Yes |
| `weight` or `synapse_count` | Connection strength | 10, 25, 150 | ✅ Yes |
| `ratio` | Connection ratio | 0.05, 0.15, 0.8 | ❌ Optional |
| `probability` | Traversal probability | 0.1, 0.5, 0.9 | ❌ Optional |

### Example Input Files

**From FindAllPath results**:
- Path: Output from `scripts/FindPath.py`
- Excel: `connection_data/*/allpaths_*/path_info.xlsx` (sheets: 'path_type' or 'path_bodyId')
- Example: [`examples/example_paths.csv`](../../archive/examples/data/example_paths.csv)

**Simple edge-list formats**:
```csv
# Minimal format
source,target,weight
KC_a,MBON01,25
KC_a,MBON03,10
MBON01,DAN_a,50

# With bodyId format (auto-detected)
bodyId_pre,bodyId_post,weight
123456789,987654321,15
987654321,111222333,8

# With neuron types
type_pre,type_post,weight
KC_alpha,MBON03,25
KC_beta,MBON03,12
```

**Example files provided**:
- [`examples/example_neuron_network.csv`](../../archive/examples/data/example_neuron_network.csv) - Simple neuron network
- [`examples/example_bodyid_network.csv`](../../archive/examples/data/example_bodyid_network.csv) - BodyId-level network
- [`examples/simple_network_data.csv`](../../archive/examples/data/simple_network_data.csv) - Minimal 3-column format

**From Custom Analysis**:
```python
import pandas as pd

# Simple 3-column format
edges = pd.DataFrame({
    'source': ['A', 'B', 'C'],
    'target': ['B', 'C', 'D'],
    'weight': [10, 20, 15]
})

# Pass directly to VisualizePath
vp = VisualizePath(path_file=edges)
```

## Quick Start

### Basic Usage

```python
from vispath import VisualizePath

# Create network from path analysis results
vp = VisualizePath(
    path_file='path_results.xlsx',    # Excel, CSV, or DataFrame
    sheet_name=None,                   # Auto-select sheet
    output_folder='./visualizations',
    network_layout='hierarchical',     # Layout algorithm
    showfig=True                       # Open in browser
)

# Generate visualizations
conn_df, G = vp.visualize()
```

## Key Features

### 1. Layout Algorithms

Choose the best layout for your network structure:

#### Hierarchical (Dagre)
- **Best for**: Layered pathways, feedforward networks
- **Algorithm**: Sugiyama's algorithm with edge crossing minimization
- **Direction**: Top-to-bottom flow
- **Use case**: Source → Intermediate → Target pathways

#### Force-Directed (Spring)
- **Best for**: General networks, community detection
- **Algorithm**: Force-atlas with attraction/repulsion
- **Behavior**: Connected nodes attract, others repel
- **Use case**: Exploring network communities

#### Circular
- **Best for**: Small networks, showing all connections
- **Algorithm**: Evenly spaced around circle
- **Benefit**: No overlapping nodes
- **Use case**: Dense connectivity, comparison view

#### Distributed
- **Best for**: Multiple disconnected components
- **Algorithm**: Separates components, uses force-directed within
- **Benefit**: Prevents component overlap
- **Use case**: Networks with isolated subgraphs

**Switching Layouts**: Use the layout dropdown at the top of the interface

### 2. Node Customization

#### Node Size
- **Default**: 40px
- **Range**: 20-80px
- **Control**: Node Size slider
- **Scales**: All nodes uniformly

#### Node Colors
- **Source nodes**: First color in scheme (e.g., blue)
- **Intermediate nodes**: Second color (e.g., cyan)
- **Target nodes**: Third color (e.g., green)
- **Customization**: See color settings panel

#### Font Size
- **Default**: 12px
- **Range**: 8-20px
- **Control**: Font Size slider
- **Affects**: Node labels

### 3. Edge Customization

#### Edge Width Scaling Methods

Control how connection strength maps to visual thickness:

- **Linear**: Direct proportional mapping
- **Logarithmic (log₂)**: Compress large differences
- **Logarithmic (log₁₀)**: Alternative compression
- **Square Root**: Moderate compression
- **None (Constant)**: All edges same width

**Formula Examples**:
- Linear: `width = weight * factor`
- Log₂: `width = log₂(weight + 1) * factor`
- Sqrt: `width = √weight * factor`

#### Edge Width Control
- **Default**: 3px
- **Range**: 0.5-30px (configurable)
- **Control**: Edge Width slider
- **Applies**: After scaling transformation

#### Arrow Size
- **Default**: 9px
- **Range**: 3-20px
- **Control**: Arrow Size slider
- **Affects**: Arrowhead at connection end

#### Connection Metrics

Toggle between different connection measures:

- **Synapse Count**: Total number of synapses
- **Connection Ratio**: Proportion of source neuron output
- **Traversal Probability**: Likelihood of signal traversal

**Note**: Edge widths update automatically when switching metrics

**Note**: the weight's unit label and the "weight" option text come from the
caller (`edge_weight_label` / `metric_option_label`), so the type-mapping
networks read **Neuron count** while pathway networks keep **Synapse Count**.
The Layout Algorithm selector also pre-selects the layout the document was
actually rendered with — including the programmatic `mapping` preset the
type-mapping views use — so the dropdown matches the canvas on open.

### 4. Edge Filtering

Hide edges based on weight values:

**Syntax**: Comma-separated rules
- Exact: `0`
- Less than: `<5`
- Greater than: `>100`
- Less/equal: `<=20`
- Greater/equal: `>=10`

**Examples**:
- `0, <5`: Hide zero and weak connections
- `>100`: Show only strong connections
- `<10, >50`: Hide medium-range connections

### 5. Opacity Controls

Adjust transparency for nodes and edges:

#### Node Opacity (by type)
- **Source nodes**: 0-100%
- **Intermediate nodes**: 0-100%
- **Target nodes**: 0-100%

#### Edge Opacity
- **Positive edges**: 0-100%
- **Negative edges**: 0-100% (if present)

**Use cases**:
- Reduce opacity to see overlapping elements
- Highlight specific neuron types
- Fade background connections

### 6. Color Settings

#### Individual Element Customization
1. **Select** elements (click node/edge, Shift+click for multiple)
2. **Choose color** from color picker
3. **Adjust opacity** with slider
4. **Apply** to update selected elements

#### Type-Based Colors
Set colors for all nodes of a type:
- **Source Color**: All source nodes
- **Intermediate Color**: All intermediate nodes
- **Target Color**: All target nodes
- **Edge Colors**: Positive and negative edges

**Color Format**: Hex codes (e.g., `#4A90E2`) or RGB

### 7. Interactive Controls

#### Selection
- **Single click**: Select node or edge
- **Shift+click**: Add to selection (multi-select)
- **Click background**: Clear selection
- **Double-click node**: Highlight neighborhood

#### Navigation
- **Drag canvas**: Pan view
- **Scroll**: Zoom in/out
- **Click and drag node**: Move (in edit mode)

#### Visibility
- **Press H**: Toggle hidden nodes
- **Press E**: Toggle hidden edges
- **Right-click**: Hide element
- **Press L**: Toggle label position

#### Node Labels
- **Default**: Below node
- **Toggle**: Press L to switch to inside/outside
- **Hide**: Adjust font size to 0 or opacity to 0

### 8. Edit Mode

Make structural changes to the network:

#### Enable Edit Mode
Click "✏️ Enable Edit Mode" button (changes to "🔒 Disable Edit Mode")

#### Add Nodes
1. Click "➕ Node" button
2. Node appears at center
3. Drag to desired position
4. **Double-click** to edit properties:
   - Label
   - Type (source/intermediate/target)
   - Color

#### Add Connections
1. **Click source node** and **drag to target**
2. Release to create connection
3. Default weight = 10
4. **Double-click edge** to edit properties

#### Delete Elements
- **Right-click** on node/edge
- Or select and click "🗑️ Delete" button

#### Modify Properties
- **Double-click** any element
- Edit label, weight, colors
- Changes save automatically

### 9. Export & Import

#### Export PNG
1. Set **Image Scale** (1-10x for resolution)
2. Click **PNG** button
3. Downloads as `network_YYYYMMDD_HHMMSS.png`

**Tip**: Use 2-3x scale for presentations, 4-5x for publications

#### Export SVG
1. Click **SVG** button
2. Downloads as `network_YYYYMMDD_HHMMSS.svg`
3. Vector format: infinite scaling without quality loss

#### Export Graph Data
1. Click **📊 Graph** button
2. Saves complete network structure as JSON
3. Includes:
   - All nodes with positions and properties
   - All edges with weights and colors
   - Current layout state

#### Import Graph
1. Click **📂 Import** button
2. Select previously exported JSON file
3. Reconstructs exact network state

#### Export/Import Layout Only
- **📤 Export Layout**: Save the full view state (positions, colors + alpha, groups, filters, controls); legacy positions map included
- **📥 Import Layout**: Load a layout file — v2 full-state files restore everything, legacy positions-only files update positions
- **Use case**: Share layouts between similar networks

### 10. Network Statistics

Displayed at bottom of interface:
- **Node count**: Total number of neurons
- **Connection count**: Total number of edges
- **Graph metrics**: Automatically calculated (degree, centrality, etc.)

## Advanced Usage

### Custom Color Schemes

Define colors programmatically:

```python
vp = VisualizePath(
    path_file='data.xlsx',
    source_color='#FF6B6B',           # Red
    intermediate_color='#FFA500',     # Orange
    target_color='#FFD700',           # Gold
    link_color='rgba(255,107,107,0.3)',  # Transparent red
    showfig=True
)
```

**Predefined Themes** (see PlotPath.py):
- Cool Theme (Blue-Cyan-Green)
- Warm Theme (Red-Orange-Gold)
- Purple Theme
- Earth Theme
- Ocean Theme
- Monochrome
- High Contrast
- Pastel
- Neon

### Edge Width Configuration

Control edge width range:

```python
vp = VisualizePath(
    path_file='data.xlsx',
    edge_width_scale='log',      # Scaling method
    min_edge_width=0.5,          # Minimum width (px)
    max_edge_width=30,           # Maximum width (px)
    edge_width_factor=1.5,       # Multiplier
)
```

### Node Size Configuration

```python
vp = VisualizePath(
    path_file='data.xlsx',
    min_node_size=20,
    max_node_size=80,
)
```

### Negative Edge Values

For inhibitory connections:

```python
# Data with negative weights
edges = [
    ('A', 'B', 10),    # Excitatory
    ('B', 'C', -5),    # Inhibitory
]

# Visualize with different colors
vp = VisualizePath(
    edge_color='#4CAF50',           # Green for positive
    negative_edge_color='#F44336',  # Red for negative
)
```

### DataFrame Input

Pass data directly without file:

```python
import pandas as pd

# Create edge list
df = pd.DataFrame({
    'source': ['A', 'B', 'C'],
    'target': ['B', 'C', 'D'],
    'weight': [10, 15, 8]
})

vp = VisualizePath(
    path_file=df,  # Pass DataFrame directly
    output_folder='./output'
)
```

### Multi-Sheet Excel Files

```python
# Auto-select sheet (prompts user)
vp = VisualizePath(
    path_file='results.xlsx',
    sheet_name=None
)

# Or specify sheet
vp = VisualizePath(
    path_file='results.xlsx',
    sheet_name='path_type'  # or 'path_bodyId'
)
```

## Layout Algorithm Details

### Hierarchical (Dagre)

**Algorithm**: Sugiyama's algorithm
- **Step 1**: Assign nodes to layers
- **Step 2**: Minimize edge crossings
- **Step 3**: Assign x-coordinates
- **Step 4**: Draw edges with minimal bends

**Best for**:
- Directed acyclic graphs (DAGs)
- Clear source → target flow
- Layered connectivity

**Parameters** (not exposed):
- Node separation: 50px
- Rank separation: 100px
- Edge weight influence on layering

### Force-Directed (Force-Atlas)

**Algorithm**: Physical simulation
- **Attraction**: Connected nodes pull together
- **Repulsion**: All nodes push apart
- **Iterations**: 300 steps to reach equilibrium

**Best for**:
- Undirected or bidirectional networks
- Community detection
- General topology exploration

### Circular

**Algorithm**: Even angular distribution
- Places nodes at equal angles
- Radius scales with node count
- Deterministic ordering (alphabetical)

**Best for**:
- Small networks (<20 nodes)
- Seeing all connections
- Symmetric visualization

### Distributed

**Algorithm**: Component separation + force-directed
- **Step 1**: Identify disconnected components
- **Step 2**: Layout each component separately
- **Step 3**: Pack components with spacing

**Best for**:
- Multiple pathways
- Disconnected subnetworks
- Complex datasets

## Tips & Best Practices

### 1. Choosing Layout
- **Start with hierarchical** for pathways
- **Switch to force-directed** if too cluttered
- **Use circular** for small, complete networks
- **Try distributed** if components overlap

### 2. Visual Clarity
- Use **log scaling** for wide weight ranges (1-1000)
- **Reduce opacity** for dense networks
- **Hide weak edges** to reduce clutter
- **Increase node size** for better label readability

### 3. Publication Figures
- Export **SVG** for vector graphics
- Use **2-3x PNG scale** for raster images
- Choose **consistent colors** across figures
- **Hierarchical layout** is most publication-ready

### 4. Exploration Workflow
1. Start with default hierarchical layout
2. Adjust edge width scale (try log)
3. Hide weak connections (<5 synapses)
4. Adjust colors for contrast
5. Export layout for reproducibility

### 5. Large Networks (>100 nodes)
- Hide weak edges aggressively
- Use force-directed layout
- Reduce node size
- Increase canvas height
- Consider filtering data before visualization

## Performance Optimization

- **Disable labels** for >200 nodes (set font size = 0)
- **Hide edges** selectively
- **Reduce opacity** instead of hiding (faster)
- **Use constant edge width** for >500 edges
- **Close edit mode** when not editing

## Keyboard Shortcuts

- `H`: Hide/show selected nodes
- `E`: Hide/show selected edges
- `L`: Toggle label position
- `Shift`: Hold for multi-selection
- `Delete`: Remove selected (in edit mode)

## Troubleshooting

**Nodes overlap**: Try force-directed or distributed layout

**Can't see edges**: Check edge opacity, adjust edge width

**Layout looks strange**: Click layout dropdown to refresh

**Export fails**: Check browser permissions, try different format

**Edit mode not working**: Ensure edit mode is enabled (orange button)

**Slow performance**: Reduce node count, hide weak edges, disable labels

## Related Documentation

- [VisualizePath Quick Reference](./VisualizePath_QuickRef.md)
- [Custom Colors Guide](./CUSTOM_COLORS_GUIDE.md)
- [Enhanced Edge-List Format](./Enhanced_EdgeList_Format.md)
- [Layout Persistence](./LAYOUT_PERSISTENCE_FEATURE.md)

## New Feature: Reciprocal Edge Offset & Mode Toggle (Nov 2025)

### Parallel Reciprocal Edges
- **Reciprocal connections** (A→B and B→A) are now drawn as parallel straight lines with a user-adjustable perpendicular offset.
- **Offset Slider**: Adjusts the separation between reciprocal edges (0–40px). Keeps arrows visually distinct and prevents overlap.
- **Arrowhead Placement**: Arrowheads are positioned outside node boundaries for clarity.

### Curved vs. Straight Edge Modes
- **Mode Toggle Button**: Switch between 'Straight' (parallel lines) and 'Curved' (bezier) edge styles for reciprocal connections.
    - Located beside the reciprocal offset slider.
    - Button color: Green for Straight, Orange for Curved.
- **Slider Disablement**: When in Curved mode, the offset slider is disabled (grayed out) since curved edges do not use offset.
- **Live Update**: Changing the mode or slider value updates the network instantly.

#### How to Use
1. **Enable Reciprocal Offset Controls**: Controls appear when reciprocal edge mode is active.
2. **Adjust Offset**: Use the slider to set the distance between parallel reciprocal edges (only in Straight mode).
3. **Toggle Mode**: Click the button to switch between Straight and Curved reciprocal edge styles.
4. **Observe**: Edges update immediately; curved mode uses standard bezier curves with no offset.

#### Technical Notes
- Offset is computed relative to a canonical direction (lexicographically ordered node IDs) to ensure reciprocal edges are always separated on opposite sides.
- Curved mode disables all offset and edge-length adjustments for reciprocal edges.
- Node size changes automatically update arrowhead positions to remain outside node boundaries.
