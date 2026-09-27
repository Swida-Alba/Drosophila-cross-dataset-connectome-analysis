# Network Features Quick Reference

## Edge Filter ⚡

**Location:** Edge Controls → "Hide Edges (weight)" input

**Syntax:**
- Exact: `1, 5, 10` → Hide specific weights
- Less: `<5` → Hide weight < 5
- Greater: `>100` → Hide weight > 100
- Combined: `<5, >100` → Hide multiple ranges

**All Operators:** `<`, `>`, `<=`, `>=`, `==`, `!=`

**Status:** Shows "X shown, Y hidden" in bottom-left

**Clear:** Delete all text → All edges reappear

---

## Export/Import 💾

### Full Graph Export

**Button:** 📊 Graph

**Saves:**
- All nodes & edges
- Positions & colors
- **Edge filter settings** ⭐
- **All slider values** ⭐
- Scaling method

**File:** `network_graph_YYYY-MM-DD.json`

### Full Graph Import

**Button:** 📂 Import

**Modes:**
- OK → **Replace** entire graph
- Cancel → **Merge** with current

**Restores:**
- Complete graph structure
- All visualization settings
- Edge filter (auto-applied)

### Layout Export

**Button:** 📤 Export Layout

**Saves:** Full view state (v2) — positions, colors + alpha, groups, filters, toggles, controls; includes a legacy positions-only map (~5-50 KB)

**Use:** Move the whole customized view to another machine or collaborator (legacy builds read the positions map)

### Layout Import

**Button:** 📥 Import Layout

**Action:** Apply a layout file — v2 restores the full state; legacy files update positions for matching nodes

---

## Interactive Editing ✏️

**Button:** ✏️ Enable Edit Mode

**Features:**
- **Add nodes:** ➕ Node button
- **Draw edges:** Click source → target
- **Edit properties:** Double-click node/edge
- **Delete:** Right-click or 🗑️ Delete button
- **Select:** Click to select, Shift+Click for multi-select

**Save:** Use 📊 Graph export to save changes

---

## Visual Controls 🎨

### Edge Controls
- **Display:** Weight / Ratio / Probability
- **Width Scale:** Linear / Log / Sqrt / None
- **Width:** Slider (1-20 px)
- **Arrow Size:** Slider (3-20 px)
- **Filter:** Hide by weight ⭐

### Font & Node
- **Font Size:** Slider (6-30 px)
- **Node Size:** Slider (10-100 px)
- **Node Shape:** Dropdown (Circle / Round Square / Sharp Square)

### Layout
- **Methods:** Hierarchical, Force, Circle, Grid, Concentric
- **Zoom:** Pinch / Scroll
- **Pan:** Click & drag background
- **Reset:** Fit graph button

---

## Color Palette 🎨

**Location:** Color Palette panel (toggle button)

**Individual Nodes:**
- Select node(s)
- Choose color & opacity
- Click "Apply to Selected"

**By Node Type:**
- Source / Intermediate / Target colors
- Per-type opacity sliders
- Positive / Negative edge colors

---

## Keyboard Shortcuts ⌨️

**Network View:**
- Click & Drag → Pan
- Scroll → Zoom
- Shift + Click → Multi-select
- Escape → Clear selection

**Edit Mode:**
- Double-Click Node → Edit properties
- Double-Click Edge → Edit weight/ratio
- Right-Click → Delete element

**Filter Input:**
- Type → Apply immediately
- Backspace/Delete → Clear filter
- Ctrl/Cmd + A → Select all

---

## Console Debugging 🔍

Open DevTools (F12) to see:

**Filter Operations:**
```
Parsed ignored edges: ["<5", ">100"]
Applying edge filter...
Edge filter complete: 45 shown, 5 hidden
```

**Export/Import:**
```
Exporting graph with settings...
✓ Exported 50 nodes, 120 edges
Restoring settings...
✓ Import complete
```

---

## Common Workflows

### 1. Quick Filter Analysis
```
1. Type: <10
2. Observe: Weak edges hide
3. Type: <20
4. Compare: More edges hide
5. Clear: All edges return
```

### 2. Save & Resume Work
```
1. Arrange network
2. Set filter: <5
3. Adjust settings
4. Click 💾 Save
5. [Later] Reopen the same HTML file
6. Everything restored automatically (📂 Load re-applies it;
   the toast's Reset drops the save and reloads defaults)
```

### 3. Create Template
```
1. Design ideal layout
2. Set optimal filters
3. Export 📊 → template.json
4. [New dataset] Import 📂 → template.json
5. Settings apply to new data
```

### 4. Compare Views
```
1. Filter: <10  → Export: weak.json
2. Filter: 10-50 → Export: medium.json
3. Filter: >50  → Export: strong.json
4. Import each to compare
```

---

## Tips & Tricks

### Edge Filter
- Start broad (`<50`), refine down
- Use status display to verify
- Combine operators: `<5, >100`
- Clear quickly: Ctrl+A, Delete

### Export/Import
- Export often (before big changes)
- Use descriptive names
- Test import before sharing
- Keep raw data separate

### Layout
- Export layout separately for reuse
- Manual arrangement preserved
- Merge mode for combining graphs
- Replace mode for clean restore

### Performance
- Hide edges to improve render speed
- Use scaling to emphasize patterns
- Filter before exporting screenshots
- Clear selection to reset view

---

## File Formats

### Full Export (v2.0)
```json
{
  "version": "2.0",
  "nodes": [...],
  "edges": [...],
  "settings": {
    "edgeFilter": {"inputValue": "<5", ...},
    "edgeWidthScaling": {"method": "log_2", ...},
    "arrowSize": 12,
    "fontSize": 16,
    "nodeSize": 40
  }
}
```

### Layout Export
```json
{
  "layout": {
    "A": {"x": 100, "y": 200},
    "B": {"x": 150, "y": 250}
  }
}
```

---

## Status Messages

**Bottom-Left Display:**

| Message | Meaning |
|---------|---------|
| `🔍 Edge filter: 45 shown, 5 hidden` | Filter active |
| `🔍 Edge filter: All edges visible` | No filter |
| `✓ Exported graph with settings: ...` | Export success |
| `✓ Imported with settings: ...` | Import success |
| `✓ Applied layout to 48/50 nodes` | Layout applied |
| `✏️ Edit Mode: Active` | Editing enabled |

---

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Filter not working | Check console, verify syntax |
| Export fails | Open console, check disk space |
| Import doesn't restore | Verify v2.0 format, check settings |
| Can't select nodes | Exit Edit Mode |
| Edges too thick/thin | Adjust Edge Width slider |
| Can't see labels | Increase Font Size |
| Lost my layout | Import last export |
| Wrong edges hidden | Check operator: `<5` hides LESS than 5 |

---

## Further Reading

**Detailed Guides:**
- [NETWORK_EDGE_FILTER.md](NETWORK_EDGE_FILTER.md) - Complete filter documentation
- [NETWORK_EXPORT_IMPORT.md](NETWORK_EXPORT_IMPORT.md) - Export/import workflows
- [NETWORK_INTERACTIVE_EDITING.md](Interactive_Network_Dragging_Guide.md) - Edit mode guide

**Related:**
- [EDGE_WIDTH_SCALING.md](EDGE_WIDTH_SCALING.md) - Scaling methods
- [CUSTOM_COLORS_GUIDE.md](CUSTOM_COLORS_GUIDE.md) - Color customization
- [AdvancedLayoutAlgorithms.md](AdvancedLayoutAlgorithms.md) - Layout options

**Main:**
- [README.md](../README.md) - Full VisualizePath documentation
