# VisualizePath Interaction Guide

Complete reference for keyboard shortcuts, mouse interactions, buttons, and interactive adjustments in VisualizePath visualizations (Network, Sankey, Heatmap).

## Table of Contents

- [Quick Reference](#quick-reference)
- [Network Visualization](#network-visualization-controls)
- [Sankey Diagram](#sankey-diagram-controls)
- [Heatmap](#heatmap-controls)
- [Edit Mode](#edit-mode-network-only)
- [Troubleshooting](#troubleshooting-interactive-issues)
- [Best Practices](#best-practices)

---

## Quick Reference

### Keyboard Shortcuts (Network)

| Key            | Action                                            | Notes                                     |
| -------------- | ------------------------------------------------- | ----------------------------------------- |
| `H`            | Hide selected nodes (+ connected edges)           | Select nodes first with click/Shift+click |
| `E`            | Hide selected edges                               | Select edges first                        |
| `L`            | Toggle label position (center ↔ outside ↔ hidden) | Cycles through positions                  |
| `Shift+Click`  | Multi-select nodes/edges                          | Add to current selection                  |
| `Double-Click` | Highlight all edges connected to node             | Good for exploring neighborhoods          |

### Mouse Actions

| Action                | Normal Mode           | Edit Mode            |
| --------------------- | --------------------- | -------------------- |
| **Click node**        | Select node           | Select node          |
| **Click edge**        | Select edge           | Select edge          |
| **Drag node**         | Reposition node       | Reposition node      |
| **Right-click node**  | Hide node + edges     | Delete node          |
| **Right-click edge**  | Hide edge             | Delete edge          |
| **Double-click node** | Highlight connections | Edit node properties |
| **Double-click edge** | —                     | Edit edge properties |
| **Click background**  | Clear selection       | Clear selection      |
| **Scroll wheel**      | Zoom in/out           | Zoom in/out          |
| **Drag background**   | Pan view              | Pan view             |
| **Hover**             | Show tooltip          | Show tooltip         |

---

## Network Visualization Controls

The network visualization (Cytoscape.js-based) provides the most comprehensive interactive features.

### Layout Panel (📐 Layout)

| Button                          | Function                          | Details                                           |
| ------------------------------- | --------------------------------- | ------------------------------------------------- |
| **🔄 Reset**                     | Re-run layout algorithm           | Animated, respects current algorithm selection    |
| **⛶ Fit**                       | Fit all visible nodes to viewport | Adds 80px padding                                 |
| **👁️ Hide Labels / Show Labels** | Toggle node labels                | Cycles visibility state                           |
| **👁️ Show All**                  | Restore all hidden nodes/edges    | Only appears when elements are hidden             |
| **🔄 Refresh Edges**             | Re-apply edge styling             | Fixes parallel/reciprocal edge display            |
| **💾 Save**                      | Save view state to browser storage | Full state: positions, colors + alpha, edges, groups, filters, toggles, all controls |
| **📂 Load**                      | Load saved view state           | Restores complete state; also auto-restored on reopen |

### Layout Algorithm Selector

The dropdown offers multiple layout algorithms with star ratings:

| Algorithm         | Rating | Description                                                 | Best For                             |
| ----------------- | ------ | ----------------------------------------------------------- | ------------------------------------ |
| **Dagre**         | ⭐⭐⭐⭐⭐  | Sugiyama's algorithm for optimal edge crossing minimization | Hierarchical pathways, DAGs          |
| **KLay**          | ⭐⭐⭐⭐   | Layer-based with advanced crossing reduction                | Complex hierarchical graphs          |
| **Breadth-First** | ⭐⭐⭐    | Simple hierarchy from graph traversal                       | Quick hierarchical view              |
| **fCoSE**         | ⭐⭐⭐⭐⭐  | Fast compound spring embedder with quality                  | Force-directed with clusters         |
| **CoSE-Bilkent**  | ⭐⭐⭐⭐   | High-quality force-directed                                 | Publication-quality layouts          |
| **CoSE**          | ⭐⭐⭐    | Standard compound spring embedder                           | General force-directed               |
| **Circular**      | ⭐⭐     | All nodes in a circle                                       | Small networks, showing connectivity |
| **Grid**          | ⭐⭐     | Nodes in matrix arrangement                                 | Very small, ordered networks         |
| **Concentric**    | ⭐⭐     | Nested circles by hierarchy                                 | Showing layered structure            |

**Algorithm Parameters** (internal):
- Dagre: `rankDir='TB'`, `nodeSep=50`, `edgeSep=20`, `rankSep=100`, `ranker='network-simplex'`
- fCoSE: `quality='proof'`, `idealEdgeLength=100`, `numIter=2500`
- CoSE-Bilkent: `quality='proof'`, `idealEdgeLength=100`, `numIter=2500`

### Reciprocal Edge Controls

For networks with bidirectional connections:

| Control                      | Function                                                    |
| ---------------------------- | ----------------------------------------------------------- |
| **Straight/Curved toggle**   | Switch between straight parallel edges and curved bezier    |
| **Reciprocal Offset slider** | Adjust spacing between parallel edges (0-40px, default 5px) |

### Edge Controls (Three-Column Panel)

**Column 1: Edge Filtering**

| Control                 | Options                                                | Description                        |
| ----------------------- | ------------------------------------------------------ | ---------------------------------- |
| **Connection Metric**   | Synapse Count, Connection Ratio, Traversal Probability | What edge width represents         |
| **Edge Width Scale**    | Linear, Log₂, Log₁₀, √, None                           | Scaling method for width           |
| **Hide Edges (weight)** | Text input                                             | Filter edges by weight expressions |

**Edge Filter Syntax**:
```
0           → Hide zero-weight edges
<5          → Hide edges with weight < 5
>100        → Hide edges with weight > 100
>=10        → Hide edges with weight >= 10
<=20        → Hide edges with weight <= 20
0, <10      → Hide zero AND weak connections
<=20, >=50  → Hide edges outside 20-50 range
```

**Column 2: Size Sliders**

| Slider         | Range    | Default | Affects                             |
| -------------- | -------- | ------- | ----------------------------------- |
| **Font Size**  | 8-20px   | 12px    | Node label text                     |
| **Node Size**  | 20-80px  | 40px    | All nodes uniformly                 |
| **Edge Width** | 0.5-30px | 3px     | Base edge thickness multiplier      |
| **Arrow Size** | 3-20px   | 9px     | Arrowhead scale (normalized to 9px) |

**Column 3: Export**

| Button       | Output          | Description                              |
| ------------ | --------------- | ---------------------------------------- |
| **PNG**      | `network_*.png` | Raster image with transparent background |
| **SVG**      | `network_*.svg` | Vector image (infinite scaling)          |
| **📊 Graph**  | JSON file       | Complete network structure with all data |
| **📂 Import** | —               | Load previously exported JSON            |
| **📤 Export Layout** | JSON file | Full view state (v2) + legacy positions map |
| **📥 Import Layout** | —         | Apply layout JSON (v2 full state or legacy positions) |

**Image Scale**: 1-10× (default 2×). Scale > 4× may fail in some browsers.

### Color Palette Panel (Right Sidebar)

The color palette is a fixed sidebar on the right side of the network.

**Edit Mode Section**:
- Toggle button to enable/disable edit mode
- Add Node / Delete buttons (only in edit mode)

**View Controls**:
| Button               | Function                                       |
| -------------------- | ---------------------------------------------- |
| **👻 Hide Orphans**   | Toggle visibility of nodes with no connections |
| **🔄 Refresh Layout** | Re-run layout after hiding/filtering           |

**Color Settings**:

1. **Selected Element(s)**: Color picker for currently selected nodes/edges
   - Color picker + hex display
   - Opacity slider (0-100%)
   - "Apply to Selected" button
   - "Clear Selection" button

2. **Edit by Group**: Batch color changes
   | Group                      | Affects                        |
   | -------------------------- | ------------------------------ |
   | Source Nodes               | All source-type nodes          |
   | Intermediate Nodes         | All intermediate-type nodes    |
   | Target Nodes               | All target-type nodes          |
   | All Nodes                  | Every node                     |
   | Positive Edges             | Edges with weight > 0          |
   | Negative Edges             | Edges with weight < 0          |
   | All Edges                  | Every edge                     |
   | NT Edges (ACH, GABA, etc.) | Edges by neurotransmitter type |

3. **Custom Groups**: Create your own groups
   - Enter group name
   - Select elements
   - Click "➕ Create" to save group
   - Select from dropdown to reuse

4. **Quick Actions**:
   - Select Source / Intermediate / Target / All Edges buttons
   - "🔄 Reset All Colors" button

### Hover Info Box (Bottom-Left)

Shows contextual information:

**For Nodes**:
```
Node: [label]
Type: [source/intermediate/target]
Color: [hex color]
```

**For Edges**:
```
Connection: [source] → [target]
Weight: [value] synapses ← Current (if selected metric)
Ratio: [value]
Probability: [value]
NT: [type] (colored by NT)
─────────────
[Custom labels if present]
```

---

## Sankey Diagram Controls

The Sankey diagram shows flow-based pathway visualization.

### Control Panel Layout

**Row 1: Node Colors**
| Control            | Default         | Description        |
| ------------------ | --------------- | ------------------ |
| Source Color       | Blue (#1f77b4)  | First-layer nodes  |
| Intermediate Color | Green (#2ca02c) | Middle-layer nodes |
| Target Color       | Red (#d62728)   | Last-layer nodes   |

**Row 2: Edge/Link Settings**
| Control      | Range             | Default        | Description                |
| ------------ | ----------------- | -------------- | -------------------------- |
| Edge Color   | Color picker      | Gray (#646464) | Base link color            |
| Edge Opacity | 0-100%            | 50%            | Link transparency          |
| Metric       | Weight/Ratio/Prob | Weight         | What link width represents |
| Node Width   | 1-50              | 5              | Sankey node thickness      |
| Font Size    | 8-20px            | 12px           | Label text size            |

**Row 3: NT Group Colors** (if NT data present)
| Control                   | Default          | Description                     |
| ------------------------- | ---------------- | ------------------------------- |
| ☐ Color by NT             | Off              | Enable NT-based edge coloring   |
| Excitatory (ACh)          | Orange (#F39C12) | Cholinergic connections         |
| Inhibitory (GABA, Glut)   | Green (#27AE60)  | GABAergic and glutamatergic connections (GLUT treated as inhibitory) |
| Modulatory (DA, 5-HT, OA) | Purple (#9B59B6) | Dopamine, Serotonin, Octopamine |
| Unknown NT                | Gray (#95A5A6)   | Unknown neurotransmitter        |

**Row 4: Action Buttons**
| Button        | Function                           |
| ------------- | ---------------------------------- |
| **Apply**     | Apply all color changes            |
| **Reset**     | Reset to default colors            |
| **Show/Hide** | Toggle visibility panel            |
| **🔍 +**       | Zoom in 20%                        |
| **🔍 -**       | Zoom out 20%                       |
| **⟲**         | Reset zoom to 100%                 |
| **🏷️ Labels**  | Toggle node labels                 |
| **📸 PNG**     | Export as PNG (with scale setting) |
| **🎨 SVG**     | Export as SVG                      |

**Zoom Range**: 30% – 300%

### Visibility Panel

Click "Show/Hide" to open the visibility panel:

- **Nodes list**: Click node name to toggle visibility
- **Edges list**: Click edge (A → B) to toggle visibility
- **Show All** button: Restore all hidden elements
- **Hide All** button: Hide everything (for selective reveal)

Hidden items show with strikethrough styling.

### Hover Tooltips

**On Links**:
```
[Source] → [Target]
Weight: [value]
Ratio: [value]
Probability: [value]
NT: [type]
```

**On Nodes**:
- Node label with layer information (e.g., "Mi1 (L1)")

---

## Heatmap Controls

The heatmap shows a connection matrix with interactive features.

### Matrix Display

- **Rows**: Source neurons (nodes that send connections)
- **Columns**: Target neurons (nodes that receive connections)
- **Cell color**: Connection strength (darker = stronger)

### Interactive Features

| Action                  | Result                                   |
| ----------------------- | ---------------------------------------- |
| **Hover cell**          | Show tooltip with source, target, weight |
| **Click row header**    | Sort columns by that row's values        |
| **Click column header** | Sort rows by that column's values        |

### Metric Toggle

Switch between different metrics:
- **Weight**: Synapse count
- **Ratio**: Connection ratio
- **Probability**: Traversal probability

### Color Scale Options

| Scale           | Formula    | Best For                           |
| --------------- | ---------- | ---------------------------------- |
| **Linear**      | value      | Even weight distribution           |
| **Logarithmic** | log(value) | Large weight variation (100-10000) |
| **Square Root** | √value     | Moderate variation                 |

### Node Ordering

Default order: Source-only nodes → Intermediate nodes → Target-only nodes

Custom order via Python:
```python
vp = VisualizePath(
    'data.xlsx',
    heatmap_row_order=['PN1', 'PN2', 'LHN1'],  # Custom row order
    heatmap_col_order=['LHN1', 'LHN2', 'MBON1']  # Custom column order
)
```

---

## Edit Mode (Network Only)

Edit Mode allows structural changes to the network.

### Enabling Edit Mode

1. Click **✏️ Enable Edit Mode** in the Color Palette panel
2. Button changes to **🔒 Disable Edit Mode**
3. Additional controls appear

### Edit Mode Actions

| Action                   | Method                                                    |
| ------------------------ | --------------------------------------------------------- |
| **Add node**             | Click "➕ Node" button, enter ID and type                  |
| **Add edge**             | Click source node, drag to target node                    |
| **Delete node**          | Right-click node, or select + click "🗑️ Delete"            |
| **Delete edge**          | Right-click edge, or select + click "🗑️ Delete"            |
| **Edit node properties** | Double-click node → dialog for label, type, color         |
| **Edit edge properties** | Double-click edge → dialog for weight, ratio, probability |

### Adding Edges in Edit Mode

1. Click on the source node (turns highlighted)
2. Click on the target node
3. Edge is created with default weight = 1
4. Double-click to edit weight

### Edit Tips

- New nodes appear at viewport center
- Use "🔄 Refresh Layout" after major changes
- Export graph to save edits (edits are not saved to original file)

---

## Troubleshooting Interactive Issues

### Layout Problems

| Issue                        | Solution                                            |
| ---------------------------- | --------------------------------------------------- |
| Nodes overlap after loading  | Press F5 to refresh, or click 🔄 Reset               |
| Layout computation slow      | Use Dagre for large networks (fastest hierarchical) |
| Wrong layout after switching | Click 🔄 Reset to recompute                          |
| Nodes drift to corner        | Click ⛶ Fit to re-center                            |

### Display Problems

| Issue                      | Solution                                       |
| -------------------------- | ---------------------------------------------- |
| Labels not visible         | Press L to toggle, or increase Font Size       |
| Edges too thin/thick       | Adjust Edge Width slider                       |
| Can't see weak connections | Change Edge Width Scale to Log₂                |
| Parallel edges overlap     | Enable Straight mode, adjust Reciprocal Offset |

### Interaction Problems

| Issue               | Solution                                   |
| ------------------- | ------------------------------------------ |
| Can't drag nodes    | Disable Edit Mode if accidentally enabled  |
| Can't select edges  | Click directly on edge line, not near it   |
| Colors not applying | Click "Apply" button after selecting color |
| Filter not working  | Check syntax (e.g., `<5` not `< 5`)        |

### Export Problems

| Issue             | Solution                           |
| ----------------- | ---------------------------------- |
| PNG blank         | Wait for full render, reduce scale |
| PNG partial       | Zoom out to show all content first |
| SVG too large     | Expected for large networks        |
| JSON import fails | Ensure same network structure      |

### Browser-Specific Issues

| Browser | Issue                      | Solution              |
| ------- | -------------------------- | --------------------- |
| Safari  | SVG export may fail        | Use Chrome/Firefox    |
| Firefox | Large PNG may crash        | Reduce scale to 3×    |
| All     | Memory limit at high scale | Use 4× max for safety |

---

## Best Practices

### For Presentations

```python
# Generate with presentation-friendly defaults
vp = VisualizePath(
    'data.xlsx',
    network_layout='dagre',  # Clear hierarchy
    showfig=True
)
```

In the browser:
1. Set Font Size to 14-16px
2. Use high contrast colors (dark on light)
3. Export at 2-3× scale
4. Use PNG for slides, SVG for zoom

### For Publications

```python
# Generate with publication defaults
vp = VisualizePath(
    'data.xlsx',
    network_layout='dagre',
    edge_width_scale='log',
    edge_width_factor=1.5
)
```

In the browser:
1. Export SVG for vector quality
2. Or PNG at 4-5× scale (300 DPI equivalent)
3. Ensure labels are legible at print size
4. Use colorblind-friendly palettes

### For Exploration

1. Start with **fCoSE** layout to see natural clusters
2. Use **multi-select** (Shift+click) to compare groups
3. **Hide weak edges** (`<10` in filter) to focus on strong connections
4. **Double-click** nodes to highlight their neighborhoods
5. Try different metrics (Weight → Ratio → Probability)

### For Large Networks (>500 edges)

```python
# Visualization-only edge cap for performance; pathfinding is unchanged
vp = VisualizePath(
    'data.xlsx',
    edgeN_limit=500,   # Draw at most 500 strongest edges
    edge_width_scale='log'  # Better visual separation
)
```

In the browser:
1. Use edge filter to hide weak connections
2. Use Dagre layout (fastest)
3. Consider hiding labels for overview
4. Export at lower scale (2×)

---

## Related Documentation

| Document                                                       | Description                      |
| -------------------------------------------------------------- | -------------------------------- |
| [Network_Guide.md](./Network_Guide.md)                         | Full network visualization guide |
| [Sankey_Guide.md](./Sankey_Guide.md)                           | Full Sankey diagram guide        |
| [Heatmap_Guide.md](./Heatmap_Guide.md)                         | Full heatmap guide               |
| [VisualizePath_QuickRef.md](./VisualizePath_QuickRef.md)       | Parameter quick reference        |
| [CUSTOM_COLORS_GUIDE.md](./CUSTOM_COLORS_GUIDE.md)             | Color customization              |
| [EDGE_WIDTH_SCALING.md](./EDGE_WIDTH_SCALING.md)               | Edge width options               |
| [LayoutAlgorithms_QuickRef.md](./LayoutAlgorithms_QuickRef.md) | Layout algorithm details         |
| [NETWORK_EXPORT_IMPORT.md](./NETWORK_EXPORT_IMPORT.md)         | Export/import features           |
