# Network Export/Import Guide (v2.0)

## Overview

The interactive network visualization supports comprehensive export and import functionality that preserves not just the graph structure, but also all visualization settings including edge filters, scaling methods, and UI control values.

## Export Format Versions

### Version 2.0 (Current)
- **Full graph data:** Nodes, edges, positions, colors
- **Complete settings:** Edge filter, scaling, slider values
- **Timestamp:** Export date and time
- **Metadata:** Node and edge counts

### Version 1.0 (Legacy)
- Graph data and positions only
- No settings preservation
- Still importable (settings ignored)

## Quick Start

### Export a Graph

1. Customize your network (filters, layout, colors, sizes)
2. Click **📊 Graph** button in Export section
3. File downloads: `network_graph_YYYY-MM-DD.json`
4. Settings are automatically saved

### Import a Graph

1. Click **📂 Import** button
2. Select a previously exported JSON file
3. Choose mode:
   - **OK (Replace):** Replace entire graph
   - **Cancel (Merge):** Merge with existing graph
4. All settings restore automatically

## Export Features

### What Gets Exported

#### 1. Graph Structure
```json
{
  "nodes": [
    {
      "data": {
        "id": "A",
        "type": "source",
        "label": "Neuron A"
      },
      "position": {"x": 100, "y": 200},
      "style": {
        "background-color": "#3498db",
        "opacity": 1
      }
    }
  ],
  "edges": [
    {
      "data": {
        "source": "A",
        "target": "B",
        "weight": 50,
        "original_weight": 50,
        "connection_ratio": 0.25,
        "traversal_prob": 0.8
      },
      "style": {
        "line-color": "#2ecc71",
        "opacity": 0.8
      }
    }
  ]
}
```

#### 2. Visualization Settings (v2.0)
```json
{
  "settings": {
    "edgeFilter": {
      "inputValue": "<5, >100",
      "ignoredValues": [],
      "expressions": [
        {"operator": "<", "threshold": 5},
        {"operator": ">", "threshold": 100}
      ]
    },
    "edgeWidthScaling": {
      "method": "log_2",
      "width": 3.5
    },
    "arrowSize": 12,
    "fontSize": 16,
    "nodeSize": 45
  }
}
```

#### 3. Metadata
```json
{
  "metadata": {
    "nodeCount": 50,
    "edgeCount": 120
  }
}
```

### Export Button Location

**Export section** (Column 3, top-right):
- **📊 Graph** - Export full graph with settings
- **📤 Export Layout** - Export the full view state (see Layout Export below)

### File Naming

Exports are automatically named with timestamps:
```
network_graph_2025-11-05.json
```

You can rename after download to something more descriptive:
```
hemibrain_KC_network_filtered.json
```

## Import Features

### Import Modes

#### Replace Mode (Recommended)
- **When to use:** Loading a complete saved network
- **What happens:**
  1. Current graph is completely cleared
  2. All nodes and edges from file are added
  3. Positions are restored exactly
  4. All settings are applied
- **Best for:** Restoring saved work, loading templates

#### Merge Mode (Advanced)
- **When to use:** Adding nodes/edges to current graph
- **What happens:**
  1. Current graph remains
  2. New nodes are added (existing nodes update positions)
  3. New edges are added (duplicates skipped)
  4. Settings still restored
- **Best for:** Combining multiple graphs, adding reference nodes

### Import Process

1. **File Selection**
   ```
   Click 📂 Import → Select JSON file
   ```

2. **Confirmation Dialog**
   ```
   Import 50 nodes and 120 edges.
   
   Click OK to REPLACE current graph
   Click Cancel to MERGE with current graph
   ```

3. **Settings Restoration**
   - Edge filter input populated and applied
   - Edge width scaling method selected
   - All slider values restored
   - Update functions called to apply changes

4. **Success Message**
   ```
   ✓ Imported with settings: 50 nodes, 120 edges
   ```

### Settings Restoration Details

The import automatically restores:

| Setting | UI Element | Function Called |
|---------|-----------|----------------|
| Edge Filter | "Hide Edges (weight)" input | `updateIgnoredEdges()` |
| Scaling Method | "Edge Width Scale" dropdown | `updateEdgeWidths()` |
| Edge Width | Edge Width slider | `updateEdgeWidth()` |
| Arrow Size | Arrow Size slider | `updateArrowSize()` |
| Font Size | Font Size slider | `updateFontSize()` |
| Node Size | Node Size slider | `updateNodeSize()` |
| Node Shape | Node Shape dropdown | `updateNodeShape()` |
| Per-node geometry | — (restored directly) | width/height/shape/z-order overrides in `nodeGeometry` |
| Per-edge width | — (restored directly) | manual widths in `edgeGeometry` (+ `customSize` marker) |

## Layout Export/Import

Exports the full view state (v2): positions plus per-element colors and
alpha (body-only — the node fill and edge line fade, label text keeps full
opacity), edge base appearance, group definitions and memberships, filters,
toggles and every style control, **plus the selection-panel overrides**:
`nodeGeometry` (explicit width/height, per-node shape, z-order — only
elements with an actual override are listed) and `edgeGeometry` (manually
sized edges). An element absent from these lists follows the global
controls, exactly like the live renderer. The legacy positions-only
`layout` map is still included so older builds can import the file.

### Export Layout Only

**Button:** **📤 Export Layout** (Import & Export section)

**File size:** ~5-50 KB (scales with node/edge counts)

**Format (v2):**
```json
{
  "version": "2.0",
  "type": "network-state",
  "layout": {
    "A": {"x": 100, "y": 200},
    "B": {"x": 150, "y": 250}
  },
  "state": {
    "version": 2,
    "positions": [{"id": "A", "position": {"x": 100, "y": 200}}, ...],
    "nodeGeometry": [{"id": "A", "width": 130, "height": 64,
                       "shape": "ellipse", "zIndex": "10000"}, ...],
    "edgeGeometry": [{"id": "e7", "width": 9}, ...],
    "colors": [{"id": "A", "color": "#ff0000", "opacity": 0.4}, ...],
    "edgeStyles": [...],
    "groupDefaults": {...}, "customGroups": {...},
    "filter": {...}, "hideToggles": {...}, "globalStyles": {...}
  }
}
```

**Use cases:**
- Save the complete customized view after manual arrangement
- Share the full look (colors, alpha, filters) with collaborators
- Back up positions without sharing data (legacy consumers read `layout`)

### Import Layout Only

**Button:** **📥 Import Layout** (Layout Persistence section)

**Process:**
1. Uploads layout JSON file
2. v2 files (`state` key): applies the full view state
3. Legacy v1 files (`layout` only): matches node IDs and updates positions
4. Reports nodes missing from the file (v1 path)

**Example output (v1 path):**
```
✓ Applied layout to 48/50 nodes
  Missing nodes: X, Y
```

See [CUSTOM_HEATMAP_ORDERING.md](CUSTOM_HEATMAP_ORDERING.md) for similar layout concepts.

## Complete Workflow Examples

### Workflow 1: Save and Restore Analysis

**Scenario:** Working on complex network analysis over multiple sessions

**Steps:**

1. **Day 1 - Setup**
   ```
   • Load raw data
   • Apply filter: <10
   • Arrange nodes manually
   • Adjust edge width scaling to logarithmic
   • Set font size to 14 for presentation
   • Export: hemibrain_analysis_day1.json
   ```

2. **Day 2 - Continue**
   ```
   • Import: hemibrain_analysis_day1.json
   • Choose Replace mode
   • Everything restored instantly!
   • Continue analysis...
   • Export: hemibrain_analysis_day2.json
   ```

3. **Day 3 - Present**
   ```
   • Import: hemibrain_analysis_day2.json
   • Adjust for presentation (larger fonts)
   • Take screenshots
   ```

### Workflow 2: Template Creation

**Scenario:** Create reusable layouts for similar datasets

**Steps:**

1. **Create Template**
   ```
   • Design ideal layout with sample data
   • Set optimal filter thresholds
   • Configure all visual settings
   • Export: network_template_v1.json
   ```

2. **Apply to New Data**
   ```
   • Generate new network from different dataset
   • Import: network_template_v1.json (Merge mode)
   • Matching nodes snap to template positions
   • Settings apply to all edges
   • Only arrange new nodes
   ```

### Workflow 3: Compare Filtered Views

**Scenario:** Analyze network at different connection strengths

**Steps:**

1. **Create Multiple Exports**
   ```
   • Set filter: <5    → Export: weak_only.json
   • Set filter: 5-20  → Export: medium_only.json
   • Set filter: >20   → Export: strong_only.json
   ```

2. **Compare**
   ```
   • Import weak_only.json → Analyze
   • Import medium_only.json → Analyze
   • Import strong_only.json → Analyze
   ```

3. **Present Findings**
   ```
   • Load each view in sequence
   • All settings consistent (font, colors, etc.)
   • Easy to compare differences
   ```

### Workflow 4: Collaboration

**Scenario:** Share analysis with collaborators

**Steps:**

1. **Prepare Export**
   ```
   • Finalize layout and filters
   • Export: shared_analysis_v1.json
   • Send file to collaborator
   ```

2. **Collaborator Imports**
   ```
   • Opens their VisualizePath
   • Import: shared_analysis_v1.json
   • Sees EXACT same view
   • Can verify findings
   • Make modifications
   ```

3. **Iterate**
   ```
   • Collaborator exports modified version
   • Send back: shared_analysis_v2.json
   • Original author imports and reviews
   ```

## Technical Details

### Null-Safe Export

The export uses optional chaining (`?.`) to handle missing UI elements:

```javascript
edgeWidthScaling: {
  method: document.getElementById('edgeWidthScale')?.value || 'linear',
  width: parseFloat(document.getElementById('edgeWidthSlider')?.value || 3)
}
```

This prevents crashes if elements don't exist in older versions.

### Import Compatibility

| Import Version | Export v1.0 | Export v2.0 |
|---------------|-------------|-------------|
| **Current (v2.0)** | ✅ Graph only | ✅ Graph + Settings |
| **Legacy (v1.0)** | ✅ Graph only | ⚠️ Ignores settings |

### Edge Matching in Merge Mode

Edges are matched by source-target pair:

```javascript
const existingEdge = cy.edges(`[source = "${source}"][target = "${target}"]`);
if (existingEdge.length === 0) {
  // Add new edge
}
```

Duplicate edges are skipped automatically.

### Settings Application Order

Import applies settings in this order:

1. **Graph structure** (nodes, edges, positions)
2. **Edge filter** (parses and applies immediately)
3. **Scaling method** (triggers edge width recalculation)
4. **Edge width** (applies to all edges)
5. **Arrow size** (visual update)
6. **Font size** (label update)
7. **Node size** (node dimensions)

This ensures dependencies are handled correctly.

## File Management Tips

### Naming Conventions

Use descriptive names that include:
- **Project:** `hemibrain_KC_network.json`
- **Filter state:** `hemibrain_strong_only.json`
- **Date/version:** `analysis_2025-11-05_v3.json`
- **Purpose:** `presentation_final.json`

### Organization

Create folder structure:
```
my_project/
├── raw_exports/
│   ├── network_2025-11-05.json
│   └── network_2025-11-06.json
├── templates/
│   ├── standard_layout.json
│   └── presentation_style.json
├── filtered_views/
│   ├── weak_connections.json
│   ├── medium_connections.json
│   └── strong_connections.json
└── final/
    └── publication_ready.json
```

### Version Control

Track graph exports in Git:
```bash
git add exports/network_analysis_v1.json
git commit -m "Add filtered network with <10 threshold"
```

JSON is text-based and diffs well in version control.

### Backup Strategy

Keep multiple versions:
1. **Auto-exports:** Keep timestamped versions
2. **Milestones:** Name significant versions
3. **Pre-modification:** Export before major changes
4. **Cloud backup:** Store in Dropbox/Google Drive

## Troubleshooting

### Export Doesn't Download

**Symptoms:** Clicking 📊 Graph button does nothing

**Solutions:**
1. Open browser console (F12) - check for errors
2. Check browser download settings
3. Try different browser
4. Verify disk space available

**Common Errors:**
```javascript
Error: Cannot read property 'value' of null
```
→ Missing UI element, update code to use `?.` operator

### Import Fails

**Symptoms:** Error message after selecting file

**Solutions:**
1. Verify JSON is valid (use JSONLint.com)
2. Check version in file (`"version": "2.0"`)
3. Ensure nodes array exists
4. Ensure edges array exists

**Error Messages:**
```
Invalid graph file format
```
→ Missing required fields (nodes/edges)

### Settings Not Restored

**Symptoms:** Graph imports but settings reset

**Solutions:**
1. Check file has `"settings"` object
2. Verify version is "2.0" not "1.0"
3. Check element IDs match in HTML
4. Look for console warnings

**Console Output:**
```javascript
Warning: Element 'edgeWidthScale' not found, using default
```
→ UI element ID mismatch

### Positions Wrong After Import

**Symptoms:** Nodes in wrong locations

**Solutions:**
1. Ensure using Replace mode (not Merge)
2. Check position data exists in JSON
3. Verify node IDs match
4. Try manual layout adjustment

### Large File Size

**Symptoms:** Export file is many megabytes

**Solutions:**
1. Use Layout Export for a positions-only legacy file (or trim the v2 `state` key)
2. Reduce graph size before export
3. Check for duplicate data in JSON
4. Compress JSON (gzip)

**Size Guidelines:**
- Small network (10-50 nodes): 10-50 KB
- Medium network (50-200 nodes): 50-200 KB
- Large network (200-1000 nodes): 200-500 KB
- Very large (1000+ nodes): 500 KB - 5 MB

## Best Practices

### 1. Export Early, Export Often

Save your work regularly:
- Before major changes
- After achieving good layout
- At end of session
- Before presentations

### 2. Use Meaningful Names

Don't rely on timestamps alone:
```
❌ network_graph_2025-11-05.json
✅ hemibrain_KC_strong_connections_2025-11-05.json
```

### 3. Document Settings

Add note in filename or separate README:
```
File: hemibrain_filtered.json
Filter: <10 (weak connections removed)
Scaling: Logarithmic (base 2)
Purpose: Focus on major pathways
```

### 4. Test Import Before Sharing

Before sending to collaborators:
1. Export graph
2. Clear browser
3. Import graph
4. Verify everything looks correct

### 5. Keep Raw Data Separate

Don't use exports as data storage:
- Keep original CSV/Excel files
- Use exports for visualization state only
- Can always regenerate from raw data

### 6. Combine with Layout Export

For maximum flexibility:
```
📊 Graph export → Complete graph structure + core settings
📤 Layout export → Full view state (v2), importable elsewhere
```

Export both for different use cases.

## Keyboard Shortcuts

While in network view:
- **Ctrl+S / Cmd+S:** (Browser save) - Save current page HTML
- **Ctrl+Shift+S:** (Browser save as) - Save with different name

Note: These save the HTML page, not the JSON export. Use export buttons for proper JSON files.

## Related Documentation

- [NETWORK_EDGE_FILTER.md](NETWORK_EDGE_FILTER.md) - Edge filtering system (saved in exports)
- [NETWORK_INTERACTIVE_EDITING.md](Interactive_Network_Dragging_Guide.md) - Manual graph editing
- [CUSTOM_HEATMAP_ORDERING.md](CUSTOM_HEATMAP_ORDERING.md) - Similar ordering concepts
- [VisualizePath Guide](../README.md) - Main documentation

## API Reference

### exportGraph()

Exports complete graph with settings.

**Trigger:** Click 📊 Graph button

**Returns:** Downloads JSON file

**Format:** See "Export Format Versions" above

### importGraph()

Triggers file selection for import.

**Trigger:** Click 📂 Import button

**Opens:** File selection dialog

### loadGraphFile(event)

Loads and processes imported graph file.

**Parameters:**
- `event` - File input change event

**Actions:**
1. Read file content
2. Parse JSON
3. Prompt for Replace/Merge mode
4. Add nodes and edges
5. Restore settings
6. Fit graph to view

### exportLayout()

Exports the full view state (v2, `type: 'network-state'`); the legacy
positions-only `layout` map is kept for older imports.

**Trigger:** Click 📤 Export Layout button

**File size:** ~5-50 KB

**Format:**
```json
{"version": "2.0", "type": "network-state", "layout": {"A": {"x": 100, "y": 200}, ...}, "state": {...}}
```

### loadLayoutFile(event)

Imports and applies a layout file. v2 files (`state` key) restore the full
view state via `applyNetworkState`; legacy v1 files (positions-only) update
node positions for matching IDs.

**Parameters:**
- `event` - File input change event

**Actions:**
1. Read layout file
2. v2: apply the captured state; v1: match node IDs and update positions
4. Report missing nodes

## Version History

- **v2.0** (Nov 2025) - Added settings export/import, edge filter integration
- **v1.0** (Oct 2025) - Initial graph export/import, layout-only export
