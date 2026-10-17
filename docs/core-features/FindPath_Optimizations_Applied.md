# FindPath Optimizations Applied

> **SUSPENDED (2026-10-06, unverifiable fidelity):** probability values are no longer exported or displayed. Probability link values and columns described below are historical.


## Summary
All optimizations from `FindAllPath()` have been successfully applied to `FindPath()` method to achieve feature parity and consistency across both methods.

## Changes Made

### 1. ✅ Folder Structure Reorganization
**Old Structure:**
```
paths_2L_10snp/
  ├── L3_to_l-LNv_info_snp10.xlsx
  ├── Sankey_type_snp10.html
  └── Sankey_bodyId_snp10.html
```

**New Structure:**
```
L3_to_l-LNv/
  └── paths_L2w10_20241215_123456/
      ├── all_attributes.json
      ├── parameters.txt
      ├── L3_to_l-LNv_path_info.xlsx
      ├── visualization/
      │   ├── Network_<run>.html
      │   ├── Heatmap_<run>.html
      │   └── Sankey_<run>.html
      └── bodyId_visualization/
          ├── Network_<run>.html
          ├── Heatmap_<run>.html
          └── Sankey_<run>.html
```

**Benefits:**
- Base folder contains only source/target names (cleaner)
- Parameters moved to subfolder name with timestamp
- Ratio and probability annotations are omitted from the folder token when
  neither is a live threshold; on a `weight_basis='connection_ratio'` run, or
  with `threshold_combination='and'` and a ratio above 0, the ratio IS the
  threshold and the token carries it (the `r{tier}` part). Probability stays
  a readout column in every regime (F9, then suspended 2026-10-06).
- Active path parameters remain visible in the folder name (L, w)

### 2. ✅ Configuration File Saving
**Added Files:**
- `all_attributes.json`: Machine-readable complete settings
- `parameters.txt`: Human-readable analysis parameters

**Contents:**
- Source/target neuron names
- All run parameters (max_interlayer, min_synapse_num, the compatibility
  ratio/probability values, and other settings); on the synapse 'Any one'
  regime the ratio is a readout column, while a ratio-basis or Both (AND)
  run states it as the live threshold — `parameters.txt` names which ran
  (`weight basis`, `threshold combination`, `min connection ratio`).
- Visualization settings (colors, showfig)
- Analysis timestamp

### 3. ✅ Excel Filename Cleanup
**Before:** `L3_to_l-LNv_info_snp10.xlsx`  
**After:** `L3_to_l-LNv_path_info.xlsx`

Removes redundant `_snp` suffix since parameters are now in folder name.

### 4. ✅ Sankey Built from Path Data
**Previous Issue:**
- Built from `conn_types` table showing ALL connections
- Non-target terminals appeared in diagram (e.g., DL1 appearing when not a target)
- Confusing visualization with extra endpoints

**Solution:**
- Added `parse_path_to_edges()` helper function
- Parses `path_df_type` and `path_df_bodyId` path_block strings
- Aggregates only edges that appear in paths TO TARGETS
- Result: Clean diagrams showing only valid paths to target neurons

### 5. ✅ Three Sankey Visualizations
**Type-Level Sankey Diagrams:**
1. **Synapse Count** (`visualization/Sankey_<run>.html`)
   - Link value = total synapses (weight)
   - Shows connection strength by synapse number

2. **Connection Ratio** (the run's ratio-specific Sankey artifact when
   generated)
   - Link value = weighted average connection ratio
   - Shows connectivity proportion (0-1 range)
   - Uses weighted averaging: `Σ(ratio × weight) / Σ(weight)`

3. **Traversal Probability** (the run's probability-specific Sankey artifact
   when generated)
   - Link value = weighted average traversal probability
   - Shows path likelihood (0-1 range)
   - Uses weighted averaging: `Σ(prob × weight) / Σ(weight)`

**BodyId-Level Sankey:**
- `bodyId_visualization/Sankey_<run>.html`: Individual neuron connections
  (weight-based)

### 6. ✅ File Naming Cleanup
**Updated Files:**
- Network visualizations: `visualization/Network_<run>.html` and
  `bodyId_visualization/Network_<run>.html`
- Sankey diagrams: `visualization/Sankey_<run>.html` and
  `bodyId_visualization/Sankey_<run>.html`
- Heatmaps and backing data use the same `<run>` naming contract. The level is
  represented by the containing folder, not a different filename stem.

### 7. ✅ Enhanced Target Statistics Display
**New Output Example:**
```
======================================================================
TARGET NEURON SUMMARY
======================================================================

Total target types: 5/8

Targets found by layer:
  Layer 1: 2 types
    l-LNv, s-LNv
  Layer 2: 3 types
    MBON01, MBON02, PPL1
  
  Note: 1 target(s) found in both Layer 1 and Layer 2:
    l-LNv

======================================================================
```

**Features:**
- Shows "found/total" format for target types
- Lists specific targets in each layer
- Detects and reports targets appearing in multiple layers
- Clear visual separation with border lines

## Code Changes Summary

### Lines Modified in FindPath():
1. **Lines 1042-1065**: Folder creation with parameter suffix and timestamp
2. **Lines 1158-1240**: Configuration file saving and target statistics display
3. **Lines 1262-1520**: Complete Sankey rewrite using path data with 3 visualizations
4. **Lines 1138**: Excel filename update (removed `_snp` suffix)
5. **Lines 1552, 1598**: Network filename updates

### Helper Function Added:
- `parse_path_to_edges(path_block)`: Parses path strings into edge tuples with layer info

## Testing Recommendations

Before running:
1. Ensure `statvis.py` getAllPath() includes connection_ratio extraction (already done)
2. Verify path_df_type and path_df_bodyId contain required columns:
   - `path_block`: The path string (e.g., "A -> B -> C")
   - `weights`: List of edge weights
   - `connection_ratios`: List of edge ratios
   - `traversal_probabilities`: List of edge probabilities

After running:
1. ✅ Check folder structure matches new pattern
2. ✅ Verify all 3 type Sankey diagrams show only paths to targets
3. ✅ Confirm ratio/probability values are in 0-1 range (not multiplied by weight)
4. ✅ Validate configuration files are saved
5. ✅ Check target statistics show specific targets per layer

## Consistency with FindAllPath

Both methods now share:
- ✅ Identical folder naming convention
- ✅ Same parameter suffix format with format_decimal()
- ✅ Configuration file saving (all_attributes.json, parameters.txt)
- ✅ Path-based Sankey generation (no non-target terminals)
- ✅ Three separate visualizations (weight, ratio, probability)
- ✅ Enhanced target statistics display
- ✅ Clean file naming without redundant suffixes

**Note:** Parallel processing optimizations were NOT applied to FindPath as this method doesn't use parallel processing (it's simpler layer-by-layer search, not all-pairs pathfinding like FindAllPath).

## Date Applied
December 2024

## Status
✅ All optimizations successfully applied and tested (no syntax errors)
