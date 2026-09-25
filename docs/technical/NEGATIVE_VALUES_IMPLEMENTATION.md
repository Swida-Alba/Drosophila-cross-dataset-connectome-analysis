# Negative Value Handling - Implementation Summary

## Overview
This document describes the implementation of negative value handling across all visualization types in the connectome analysis toolkit. The implementation ensures that negative edge weights are correctly displayed with appropriate visual indicators.

## Design Decisions

### Color Scheme
- **Positive edges**: Gray (`rgba(100, 100, 100, 0.4)`)
- **Negative edges**: Light Blue (`rgba(74, 144, 226, 0.4)`)

### Width Handling
- All edge widths use **absolute values** for proper display
- Negative sign is preserved in hover labels and data exports

## Implementation Details

### 1. Network Visualization (`vispath.py`)

#### Edge Data Preparation (Lines 2818-2857)
```python
edges_data.append({
    'data': {
        'weight': abs_weight,        # Positive value for Cytoscape
        'original_weight': weight,   # Preserve original sign
        'is_negative': is_negative   # Flag for styling
    }
})
```

**Why this approach?**
- Cytoscape.js doesn't support ternary operators in style definitions
- Storing positive values avoids blank display issues
- Separate flag enables selector-based styling

#### Edge Styling (Lines 3522-3527)
```javascript
// Selector-based style for negative edges
{
    selector: 'edge[is_negative = true]',
    style: {
        'line-color': '#4A90E2',        // Light blue
        'target-arrow-color': '#4A90E2'
    }
}
```

#### Hover Labels (Lines 3760-3777)
```javascript
// Show negative sign in hover
const displayWeight = data.is_negative ? -data.weight : data.weight;
html += `Weight: ${displayWeight.toLocaleString()} synapses`;
```

### 2. Sankey Diagram (`vispath.py`)

#### Custom Hover Labels (Lines 1849-1859)
```python
hover_text = f"{source_name} → {target_name}<br>"
hover_text += f"Weight: {orig_weight:,}"  # Shows negative sign
if ratios[i] != 0:
    hover_text += f"<br>Ratio: {ratios[i]:.3f}"
if probs[i] != 0:
    hover_text += f"<br>Probability: {probs[i]:.3f}"
```

**Features:**
- Shows source and target neuron names
- Displays original weight with negative sign
- Includes ratio and probability metrics when available

#### Edge Colors (Lines 1863-1875)
```python
if orig_weight < 0:
    edge_colors_updated.append('rgba(74, 144, 226, 0.4)')  # Light blue
else:
    edge_colors_updated.append('rgba(100, 100, 100, 0.4)')  # Gray
```

#### Legend (Lines 1900-1933)
- Automatically displays when negative values are present
- Shows color indicators for positive and negative edges

### 3. Heatmap Visualization (`statvis.py`)

#### Signed Transforms (Lines 910-926)
```python
def signed_transform(values, transform_func):
    """Apply transform preserving sign: sign(v) × transform(|v|)"""
    signs = np.sign(values)
    abs_values = np.abs(values)
    # Handle zeros
    abs_values = np.where(abs_values == 0, 1e-10, abs_values)
    return signs * transform_func(abs_values)
```

**Supported transforms:**
- `log2_signed`: `sign(v) × log2(|v|)`
- `log10_signed`: `sign(v) × log10(|v|)`
- `sqrt_signed`: `sign(v) × √|v|`

#### JavaScript Transforms (Lines 1570-1597)
```javascript
// Client-side transform for interactive updates
function transformValue(val, scale) {
    if (scale === 'log2' || scale === 'log10' || scale === 'sqrt') {
        const sign = Math.sign(val);
        const absVal = Math.abs(val);
        const base = (absVal === 0) ? 1e-10 : absVal;
        
        if (scale === 'log2') return sign * Math.log2(base);
        if (scale === 'log10') return sign * Math.log10(base);
        if (scale === 'sqrt') return sign * Math.sqrt(base);
    }
    return val;  // Linear scale
}
```

### 4. SankeyDirect Function (`statvis.py`)

#### Custom Hover Labels (Lines 3341-3363)
```python
hover_text = f"{source_names[source_i]} → {target_names[target_j]}<br>"
hover_text += f"Weight: {value:,}"
hover_labels.append(hover_text)
```

#### Implementation (Lines 3411-3416)
```python
link = dict(
    source = source_list,
    target = target_list,
    value = value_list,
    color = color_list,
    customdata = hover_labels,
    hovertemplate = '%{customdata}<extra></extra>'
)
```

## Testing

### Test Data
- **File**: `test_data/test_paths_with_negatives.csv`
- **Pathways**: 15 total
- **Connections**: 28 (18 positive, 10 negative)
- **Weight Range**: -105 to +220

### Test Script
- **File**: `scripts/PlotPath_TestNegatives.py` (removed; the contract
  is now covered by `tests/core/test_audit_fixes.py::test_negative_values_are_folder_safe`)
- **Output**: `test_negative_output/` directory
- **Validations**:
  - Network displays correctly (not blank)
  - Negative edges show in light blue
  - Hover labels show negative sign
  - Sankey shows source/target in hover
  - Heatmap handles all transform types

### Expected Output
```
Creating interactive network visualization...
  ℹ️  Found negative edge weights - using absolute values for width, light blue color for negative edges

Creating layered Sankey diagram...
  ℹ️  Found negative edge weights - using absolute values for link width, light blue for negative edges
```

## Usage Examples

### PlotPath with Negative Values
```python
from src.vispath import VisualizePath

# Load data with negative weights
vp = VisualizePath('test_data/test_paths_with_negatives.csv', 
                   format='csv', 
                   output_folder='output')

# Generate all visualizations
vp.plot(
    show_network=True,   # Light blue for negative edges
    show_sankey=True,    # Light blue for negative links
    show_heatmap=True    # Signed transforms for negative values
)
```

### FindDirect with Negative Ratios
```python
from src.coana import FindDirect

# Query might return negative connection ratios
fd = FindDirect(
    body_id=['12345', '67890'],
    dataset='hemibrain:v1.2.1'
)

# Visualizations automatically handle negatives
fd.SankeyDirect()  # Shows light blue links for negative ratios
```

## Technical Considerations

### Why Not Use Negative Values Directly in Cytoscape?
- Cytoscape.js uses edge width/opacity for visual weight
- Negative widths would cause display errors
- Selector-based styling is more robust than inline ternary operators

### Why Preserve Original Values?
- Important for scientific interpretation
- Negative connections may indicate inhibitory relationships
- Data exports should maintain original signs

### Why Signed Transforms?
- Log/sqrt of negative numbers produces NaN
- Signed transforms preserve both magnitude and sign
- Formula: `sign(v) × transform(|v|)`

## Future Enhancements

### Potential Improvements
1. **Custom color schemes**: Allow users to define colors for positive/negative
2. **Threshold-based styling**: Different colors for different magnitude ranges
3. **Bidirectional edges**: Support for both positive and negative connections between same pair
4. **Export options**: Include negative handling info in metadata

### Known Limitations
1. **Custom edge colors**: Negative color overrides custom colors
2. **Very small values**: Values near zero may not display well in log transforms
3. **Performance**: Color assignment done per-edge (could be optimized with vectorization)

## Related Documentation
- [Edge Width Scaling](../visualizations/EDGE_WIDTH_SCALING.md)
- [Custom Colors Guide](../visualizations/CUSTOM_COLORS_GUIDE.md)
- [Enhanced EdgeList Format](../visualizations/Enhanced_EdgeList_Format.md)

## Version History
- **v1.0** (2024): Initial implementation of negative value handling
  - Alternative approach using absolute values + display modification
  - Selector-based styling for network
  - Custom hover labels for Sankey with source/target info
  - Signed transforms for heatmaps
  - Light blue color scheme for negative edges
