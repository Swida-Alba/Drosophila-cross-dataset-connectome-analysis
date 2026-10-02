# VisualizePath Network Features Guide

Comprehensive guide to the interactive network visualization features in VisualizePath.

## Table of Contents

1. [Input Formats](#input-formats)
2. [Group Selection](#group-selection)
3. [NT (Neurotransmitter) Edge Groups](#nt-neurotransmitter-edge-groups)
4. [Custom Groups](#custom-groups)
5. [Color & Opacity Controls](#color--opacity-controls)
6. [Export/Import](#exportimport)
7. [Keyboard Shortcuts](#keyboard-shortcuts)
8. [Edit Mode](#edit-mode)

---

## Input Formats

### Connection Matrix (NxM numeric DataFrame)
```python
import pandas as pd
import numpy as np

matrix = pd.DataFrame(
    np.random.poisson(3, (5, 5)),
    index=['A', 'B', 'C', 'D', 'E'],
    columns=['A', 'B', 'C', 'D', 'E']
)
vp = VisualizePath(matrix)
vp.visualize()
```

### Path-Based Format
| Column                    | Type | Required | Description                             |
| ------------------------- | ---- | -------- | --------------------------------------- |
| `path_block`              | str  | ✅        | Path as "A -> B -> C" format            |
| `weights`                 | list | ✅        | Weights for each hop                    |
| `nt_types`                | list | ❌        | NT type for each edge (ACH, GABA, etc.) |
| `connection_ratios`       | list | ❌        | Ratio values for each hop               |
| `traversal_probabilities` | list | ❌        | Probability for each hop                |

### Edge-List Format
| Column                      | Type    | Required | Description           |
| --------------------------- | ------- | -------- | --------------------- |
| `source` / `from` / `*_pre` | str     | ✅        | Source node           |
| `target` / `to` / `*_post`  | str     | ✅        | Target node           |
| `weight`                    | numeric | ✅        | Connection weight     |
| `color` / `edge_color`      | str     | ❌        | Per-edge color (hex or rgba) |
| `nt_type`                   | str     | ❌        | Neurotransmitter type |
| `source_group`              | str     | ❌        | Node classification of source (`source` / `intermediate` / `target`) |
| `target_group`              | str     | ❌        | Node classification of target (`source` / `intermediate` / `target`) |
| `ratio`                     | numeric | ❌        | Mapped to `connection_ratios` metric |
| `probability`               | numeric | ❌        | Mapped to `traversal_probabilities` metric |
| `nt_group` / `custom_groups`| str     | ❌        | Informational; accepted and ignored (NT grouping is re-derived from `nt_type`) |

Any other numeric column is picked up as an additional edge metric.

**Expanded edge list (CSV export round-trip).** The CSV written by the
**📋 Edge List CSV** export button uses exactly these columns:
`source, target, weight, color, nt_type, nt_group, source_group,
target_group, custom_groups, ratio, probability`. Re-passing that CSV to
`VisualizePath` rebuilds the same network - identical edges, weights, metric
columns, NT types and the source/intermediate/target node grouping. Without
`source_group` / `target_group`, a plain edge list would classify every node
as *source* (each row is a 1-hop path whose first node is a source).

---

## Group Selection

### Built-in Groups

Access via the **Edit by Group** dropdown in the left panel:

| Group                  | Description                  |
| ---------------------- | ---------------------------- |
| **Source Nodes**       | All source/input neurons     |
| **Intermediate Nodes** | All intermediate neurons     |
| **Target Nodes**       | All target/output neurons    |
| **All Nodes**          | All nodes regardless of type |
| **Positive Edges**     | Edges with positive weights  |
| **Negative Edges**     | Edges with negative weights  |
| **All Edges**          | All edges regardless of sign |

### Quick Selection Buttons

- **Select Source** - Selects all source nodes
- **Select Intermed.** - Selects all intermediate nodes
- **Select Target** - Selects all target nodes
- **Select All Edges** - Selects all edges

### How to Use Groups

1. Select a group from the dropdown
2. Adjust color using the color picker
3. Adjust opacity using the slider (0-100%)
4. Click **Apply to Group** to apply changes

---

## NT (Neurotransmitter) Edge Groups

### Supported NT Types

| NT Type  | Full Name     | Default Color      |
| -------- | ------------- | ------------------ |
| **ACH**  | Acetylcholine | 🟠 #F39C12 (Orange) |
| **GABA** | GABA          | 🟢 #27AE60 (Green)  |
| **GLUT** | Glutamate     | 🔴 #E74C3C (Red)    |
| **DA**   | Dopamine      | 🟣 #9B59B6 (Purple) |
| **SER**  | Serotonin     | 🔵 #3498DB (Blue)   |
| **OCT**  | Octopamine    | 🩵 #1ABC9C (Teal)   |

### How NT Groups Work

1. **Data Requirement**: Include `nt_type` column in edge-list or `nt_types` in path-based format
2. **Automatic Grouping**: NT groups appear in the dropdown under "NT Edges"
3. **Hover Display**: NT type shows in edge hover tooltip with color
4. **Batch Editing**: Select an NT group to change color/opacity of all edges of that type

### Input Data Example

```python
# Edge-list with NT type
edges_df = pd.DataFrame({
    'source': ['A', 'B', 'C'],
    'target': ['B', 'C', 'D'],
    'weight': [100, 50, 30],
    'nt_type': ['ACH', 'GABA', 'GLUT']
})

# Path-based with NT types
paths_df = pd.DataFrame({
    'path_block': ['A -> B -> C'],
    'weights': [[100, 50]],
    'nt_types': [['ACH', 'GABA']]
})
```

> **Per-hop alignment**: the `nt_types` list is read **positionally**, one
> entry per hop. A hop with no NT annotation must occupy an **empty slot**
> (`''`) so later hops' NT cannot shift onto the wrong edge — e.g.
> `['', 'ACH']` for a 2-hop path whose first connection is unannotated.
> The DROCAT producer (statvis) emits exactly this format; lists shorter
> than the hop count (legacy runs) render the missing tail as unknown.

---

## Custom Groups

### Creating Custom Groups

1. **Select elements**: Click nodes/edges while holding Shift, or use group selection buttons
2. **Enter group name**: Type a name in the "Group Name" field (optional - auto-generates if empty)
3. **Click Create**: Creates a new group containing the selected elements
4. **Access group**: Find it in the dropdown under "Custom Groups"

### Managing Custom Groups

- **Select group**: Choose from dropdown to select all elements in the group
- **Edit colors**: Apply color/opacity changes to the entire group
- **Delete group**: Select a custom group and click the Delete button

### Custom Group Features

- Groups persist through export/import cycles
- Each group stores: element IDs, color, opacity
- Groups appear in the main dropdown for easy access

---

## Color & Opacity Controls

### Default Opacity Values

| Element                | Default Opacity | Notes                        |
| ---------------------- | --------------- | ---------------------------- |
| **Source Nodes**       | 100%            | Fully opaque                 |
| **Intermediate Nodes** | 100%            | Fully opaque                 |
| **Target Nodes**       | 70%             | Slightly transparent         |
| **Positive Edges**     | 50%             | Semi-transparent for clarity |
| **Negative Edges**     | 100%            | Fully opaque                 |

These defaults can be overridden by passing custom colors with alpha values (e.g., `'rgba(100,150,240,0.3)'`).

### Individual Element Coloring

1. **Click an element** to select it
2. **Adjust color** using the Individual color picker
3. **Adjust opacity** using the slider
4. **Click Apply to Selected** to apply changes

### Multi-Selection Coloring

1. **Hold Shift + Click** to select multiple elements
2. Or use **Select All Edges** / group selection buttons
3. Changes apply to all selected elements

### Reset All Colors

Click **🔄 Reset All Colors** to restore all elements to their initialization state:
- **Nodes**: Return to type-based colors at 100% opacity (source/intermediate/target)
- **Positive edges**: Return to 50% opacity (configurable via `link_color`)
- **Negative edges**: Return to 100% opacity
- **Custom group colors**: Cleared

---

## Export/Import

### Export Graph (JSON)

Exports complete graph state including:
- ✅ Node positions (layout)
- ✅ Node/edge styles (colors, opacity)
- ✅ Hidden/filtered state
- ✅ Edge filter settings
- ✅ View settings (edge width, arrow size, font size)
- ✅ Group default colors (including NT groups)
- ✅ Custom groups with element IDs

**How to export:**
1. Click **📊 Graph** button in Export section
2. JSON file downloads automatically

### Import Graph

Restores complete graph state from exported JSON:
- **Replace mode**: Clears current graph and loads imported data
- **Merge mode**: Adds imported elements to existing graph

**How to import:**
1. Click **📂 Import** button
2. Select previously exported JSON file
3. Choose Replace or Merge

### Export Layout Only

Exports the full view state (v2): node positions plus colors and alpha,
edge base appearance, group definitions, filters, toggles and all style
controls. A legacy positions-only map is included for older imports:
1. Click **📤 Export Layout** button
2. Import elsewhere to restore the whole look (or just positions with legacy builds)

### Export Edge List (CSV)

Exports every edge of the current graph as an expanded edge-list CSV:
1. Click **📋 Edge List CSV** button in Export section
2. Columns: `source, target, weight, color, nt_type, nt_group, source_group,
   target_group, custom_groups, ratio, probability`
3. Includes Edit-Mode changes (edited weights, recolored edges,
   added/deleted edges) and the complete table even when the view is filtered

The file is a valid [Edge-List Format](#edge-list-format) input: re-import it
in the Net-Viz tab (or pass it to `VisualizePath`) to rebuild the same
network, including the source/intermediate/target node grouping via the
`source_group` / `target_group` columns.

---

## Keyboard Shortcuts

| Key               | Action                                    |
| ----------------- | ----------------------------------------- |
| **H**             | Hide selected nodes (and connected edges) |
| **E**             | Hide selected edges                       |
| **L**             | Toggle label position (center/outside)    |
| **Shift + Click** | Add to selection                          |
| **ESC**           | Clear selection                           |

### Mouse Actions

| Action                | Effect                                |
| --------------------- | ------------------------------------- |
| **Click node/edge**   | Select element                        |
| **Right-click**       | Hide element (or delete in edit mode) |
| **Double-click node** | Highlight connected edges             |
| **Drag node**         | Reposition node                       |
| **Scroll**            | Zoom in/out                           |
| **Drag background**   | Pan view                              |

---

## Edit Mode

Enable Edit Mode to modify the graph structure:

### Toggle Edit Mode

Click **✏️ Enable Edit Mode** button to enter edit mode.

### Edit Mode Features

| Action                | Effect                                       |
| --------------------- | -------------------------------------------- |
| **Double-click node** | Edit node label and type                     |
| **Double-click edge** | Edit edge weight                             |
| **Right-click**       | Delete element (instead of hide)             |
| **Add Edge**          | Click "Draw Edge" then click source → target |
| **Add Node**          | Creates new node at center                   |

### Exit Edit Mode

Click **🔒 Disable Edit Mode** to return to normal mode.

---

## Best Practices

### For Large Networks

1. Use **Hide Orphans** to reduce clutter
2. Use **Edge Filter** to show only significant connections
3. Export layout after arranging to preserve positions

### For Publication

1. Select groups and apply consistent colors
2. Adjust opacity for overlapping edges
3. Use **Export PNG** for high-resolution images

### For Analysis

1. Create custom groups for neurons of interest
2. Use NT groups to analyze neurotransmitter distribution
3. Export/import to preserve analysis state between sessions

---

## Troubleshooting

### NT Groups Not Appearing

- Ensure data has `nt_type` column (edge-list) or `nt_types` (path-based)
- NT values should match: ACH, GABA, GLUT, DA, SER, OCT

### Custom Group Lost After Refresh

- Groups are stored in the HTML, not in browser storage
- Use Export/Import to save and restore groups

### Colors Not Resetting Properly

- Click **Reset All Colors** to restore original defaults
- If issues persist, refresh the page

---

## Related Documentation

- [VisualizePath Quick Reference](VisualizePath_QuickRef.md)
- [Network Export/Import Guide](NETWORK_EXPORT_IMPORT.md)
- [Layout Algorithms](LayoutAlgorithms_QuickRef.md)
- [Custom Colors Guide](CUSTOM_COLORS_GUIDE.md)
