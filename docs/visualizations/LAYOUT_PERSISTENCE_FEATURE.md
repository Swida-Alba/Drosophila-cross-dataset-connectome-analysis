# Layout Persistence Feature

## Overview

The network visualizations now include a **Layout Persistence** feature that allows you to save and restore all your custom adjustments to the visualization. Reopening the same HTML file restores the last saved state automatically (with a **Reset** action on the toast that drops the saved state and reloads the generated defaults).

## Features

### 💾 Save Layout
- Saves the current state to **browser localStorage**
- Persists across browser sessions (stays saved even after closing browser)
- Automatically uses unique storage key based on filename

### 📂 Load Layout
- Restores previously saved layout from localStorage
- Restores all positions, appearance, settings (see the full list below)

### 🔄 Auto-Restore on Open
- When the page opens and a saved state exists for this file, it is re-applied automatically
- The confirmation toast offers a **Reset** action: drops the saved state and reloads the page, restoring the generated defaults

### 📤 Export Layout
- Downloads the full view state as a `.json` file (v2 format)
- Can be shared with collaborators
- Can be backed up or version-controlled

### 📥 Import Layout
- Upload a previously exported layout file
- v2 files apply the full state; legacy positions-only files (v1) still import

## What Gets Saved

The following state is preserved:

### Node Properties
- **Positions** (x, y coordinates)
- **Colors** (custom color assignments)
- **Alpha/opacity** — body-only: the fill's transparency set through the color panels; label text keeps full opacity
- **Visibility** (hidden/shown state)
- **Custom group memberships** (assigned_group)

### Edge Properties
- **Visibility** (hidden/shown state)
- **Base color** (line/arrow color)
- **Base alpha/opacity** — line + arrows only; edge weight-label text is never faded by alpha

### Group Definitions
- **Group defaults** (color + opacity per group, including edited NT/dataset groups)
- **Custom group definitions** (label, color, opacity)

### View State
- **Zoom level**
- **Pan position** (viewport center)
- **Label visibility** (on/off)
- **Label position** (center/outside)
- **Edge weight labels** (on/off)
- **Background color**
- **Label font color**
- **Hemisphere mirror** (on/off)

### Control Settings
- **Edge width** slider value
- **Edge width scale** method (linear/log/sqrt/none)
- **Arrow size** slider value
- **Font size** slider value
- **Node size** slider value
- **Edge label font size** slider value
- **Connection metric** (weight/ratio/probability)
- **Edge filter** expression
- **Hide toggles** (orphans / self-loops / dead ends)
- **Reciprocal edge mode** (curved/straight) + offset
- **Spacing trackers** (horizontal/vertical gaps) and **rotation** — values only; the saved positions carry the arrangement

### Metadata
- **Timestamp** (when saved)
- **Graph name** (for unique identification)
- **Version** (2)

## Usage

### Basic Workflow

1. **Adjust your visualization**
   - Move nodes around to desired positions
   - Change colors and alpha using the color panels
   - Hide/show nodes and edges, filter edges as needed
   - Adjust font sizes, edge widths, etc.

2. **Save your work**
   - Click **💾 Save** button
   - Your layout is now persisted in browser storage

3. **Reload anytime**
   - Open the same HTML file — the saved state is restored automatically
   - Or click **📂 Load** to re-apply it explicitly
   - The toast's **Reset** action drops the save and reloads the defaults

### Sharing Layouts

1. **Export to file**
   - Click **📤 Export Layout** button
   - Downloads `network_layout_<date>.json` with the full view state
   - Send this file to collaborators

2. **Import from file**
   - Collaborator clicks **📥 Import Layout** button
   - Selects your `.json` file
   - The full state is applied (legacy positions-only files import too)

## Technical Details

### Storage

- Uses browser's **localStorage** API
- Storage key: `cytoscape_layout_<filename>#<generation-timestamp>` (each generated HTML copy has independent saves; stale keys are evicted, newest 20 kept)
- Data format: JSON
- Size limit: ~5-10 MB (browser dependent, typically sufficient for hundreds of nodes)

### Browser Compatibility

- ✅ Chrome, Edge, Firefox, Safari (modern versions)
- ✅ Works offline (no internet needed after initial page load)
- ❌ Incognito/Private mode (localStorage cleared on exit)

### File Format

JSON structure (v2 — `layout` is the legacy positions-only map kept for older imports):
```json
{
  "version": 2,
  "positions": [{"id": "neuron1", "position": {"x": 100, "y": 200}}, ...],
  "colors": [{"id": "neuron1", "color": "#ff0000", "opacity": 0.4}, ...],
  "edgeStyles": [{"id": "edge1", "baseColor": "#ff0000", "baseOpacity": 0.3}, ...],
  "visibility": [{"id": "neuron1", "visible": true, "hidden": false}, ...],
  "edgeVisibility": [{"id": "edge1", "visible": true, "hidden": false}, ...],
  "assignedGroups": {"neuron1": "my-group"},
  "groupDefaults": {"source": {"color": "#ff5722", "opacity": 100}, ...},
  "customGroups": {"my-group": {"label": "...", "color": "...", "opacity": 100, ...}},
  "background": "#ffffff",
  "labelFontColor": "",
  "labelsVisible": true,
  "labelPosition": "center",
  "edgeWeightLabels": false,
  "hemisphereMirrorEnabled": false,
  "reciprocal": {"enabled": false, "offset": 5},
  "filter": {"inputValue": "", "ignoredValues": [], "expressions": []},
  "hideToggles": {"orphans": false, "selfLoops": false, "deadEnds": false},
  "globalStyles": {"nodeSize": 40, "edgeWidth": 3, "fontSize": 12,
                    "edgeLabelFontSize": 9, "arrowSize": 9,
                    "edgeWidthScale": "log_e", "metric": "weight",
                    "spacingX": 100, "spacingY": 100, "rotation": 0},
  "edgeWidth": "3", "edgeWidthScale": "log_2", "arrowSize": "9",
  "fontSize": "12", "nodeSize": "40",
  "zoom": 1.5,
  "pan": {"x": 0, "y": 0},
  "timestamp": "2026-09-27T10:30:00.000Z",
  "graphName": "network_selected_paths"
}
```

Every field is applied defensively: payloads (or files) written by older builds that lack the newer keys still load.

### Best Practices

### 1. Save Frequently
- Save after major layout changes
- Save before experimenting with new adjustments
- Export important layouts as backup

### 2. Name Your Exports
- Default name: `network_layout_<date>.json`
- Rename exported files with descriptive names
- Example: `network_layout_L3_to_MeVPMe_final_2026.json`

### 3. Version Control
- Export layouts before making major changes
- Keep multiple versions with date suffixes
- Store in project folder for team collaboration

### 4. Browser Considerations
- Each browser has separate localStorage
- Different browsers won't share saved layouts
- Use Export/Import to transfer between browsers

### 5. Backup Important Work
- Export to JSON file for permanent backup
- localStorage can be cleared by browser settings
- JSON files are portable and version-controllable

## Troubleshooting

### Layout not loading?
- Check the toast for errors
- Verify you're opening the same HTML file (storage keys are per generated copy)
- Try exporting and re-importing

### Restored a layout by mistake?
- Click **Reset** on the "Restored saved layout" toast — it drops the save and reloads the generated defaults
- Or undo individual steps with the undo system where applicable

### Positions slightly off?
- May occur if window size changed significantly
- Click "Fit to Screen" to recenter
- Re-save after adjusting

### Can't save in Incognito mode?
- localStorage disabled in private browsing
- Use Export instead to save to file
- Open in regular browser window

### Shared layout looks different?
- Different screen sizes may affect initial view
- Collaborator should click "Fit to Screen"
- Relative positions will be preserved

## Implementation Status

- ✅ **Network Visualization** - Fully implemented
- ⏳ **Sankey Diagram** - Planned (will be added after testing network version)

## Feedback

If you encounter issues or have suggestions for this feature, please open an issue on the GitHub repository.
