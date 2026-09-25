# Enhanced Edge-List Format Guide

## Overview

The `VisualizePath` class now supports a flexible edge-list input format with automatic column recognition and metric detection. This enhancement allows users to work with various data formats without rigid column naming requirements.

## Key Features

### 1. Flexible Column Recognition

The system automatically detects columns for:

- **Source nodes**: 
  - Exact matches: `source`, `from`, `pre`
  - Pattern matches: Any column ending with `_pre` (e.g., `bodyId_pre`, `type_pre`, `neuron_pre`)

- **Target nodes**:
  - Exact matches: `target`, `to`, `post`
  - Pattern matches: Any column ending with `_post` (e.g., `bodyId_post`, `type_post`, `neuron_post`)

- **Edge weights**:
  - Recognized names: `weight`, `weights`, `synapse_count`, `count`

### 2. Automatic Metric Detection

All numeric columns (excluding source, target, and color columns) are automatically detected as additional metrics:

- **Standard metric mapping**:
  - `ratio` → `connection_ratios` (enables toggle support)
  - `connection_ratio` → `connection_ratios`
  - `probability` → `traversal_probabilities` (enables toggle support)
  - `prob` → `traversal_probabilities`
  - `trav_prob` → `traversal_probabilities`
  - `traversal_probability` → `traversal_probabilities`

- **Custom metrics**:
  - Any other numeric column is preserved as a custom metric
  - Custom metrics are available in exported data files
  - Examples: `strength`, `confidence`, `reliability`

### 3. Supported Input Formats

- pandas DataFrame (in-memory)
- CSV files (`.csv`)
- Excel files (`.xlsx`, `.xls`)

### 4. Expanded Export Columns (CSV Round-Trip)

The **📋 Edge List CSV** export in the network HTML writes an expanded edge
list with these exact columns:

```
source, target, weight, color, nt_type, nt_group, source_group,
target_group, custom_groups, ratio, probability
```

Re-importing that CSV (Net-Viz tab or `VisualizePath` directly) rebuilds the
same network:

| Column | Handling |
| ------ | -------- |
| `source` / `target` / `weight` | Edges of the network |
| `color` | Per-edge color (hex or rgba) |
| `nt_type` | Neurotransmitter type; drives NT edge groups (empty = unknown) |
| `source_group` / `target_group` | Restores the `source` / `intermediate` / `target` classification of each endpoint node. Without them every node of a plain edge list is classified as *source* |
| `ratio` / `probability` | Mapped to `connection_ratios` / `traversal_probabilities` (toggleable) |
| `nt_group` / `custom_groups` | Informational; accepted and ignored (NT grouping is re-derived from `nt_type`) |

## Usage Examples

### Example 1: Basic Format

```python
import pandas as pd
from vispath import VisualizePath

# Simple edge-list with source/target/weight
data = {
    'source': ['A', 'B', 'C'],
    'target': ['B', 'C', 'D'],
    'weight': [10, 15, 20]
}

df = pd.DataFrame(data)

vis = VisualizePath(
    path_file=df,
    output_folder='./output'
)

vis.create_network()
vis.create_sankey()
vis.create_heatmap()
```

### Example 2: BodyId Format

```python
# Using bodyId_pre/bodyId_post naming convention
data = {
    'bodyId_pre': [123456, 234567],
    'bodyId_post': [234567, 345678],
    'weight': [25, 30]
}

df = pd.DataFrame(data)
vis = VisualizePath(path_file=df, output_folder='./output')
```

### Example 3: With Ratio and Probability

```python
# Additional metric columns auto-detected
data = {
    'source': ['KC_a', 'KC_b'],
    'target': ['MBON_a', 'MBON_b'],
    'weight': [100, 80],
    'ratio': [0.67, 0.57],  # Auto-mapped to connection_ratios
    'probability': [0.95, 0.92]  # Auto-mapped to traversal_probabilities
}

df = pd.DataFrame(data)
vis = VisualizePath(path_file=df, output_folder='./output')

# Ratio and probability can be toggled in interactive visualizations
vis.create_network()  # Toggle controls included
vis.create_sankey()   # Toggle controls included
```

### Example 4: Custom Metrics

```python
# Custom metric columns preserved
data = {
    'neuron_pre': ['DA1_PN', 'VA1d_PN'],
    'neuron_post': ['LHON1', 'LHON2'],
    'synapse_count': [150, 120],
    'strength': [0.85, 0.72],      # Custom metric
    'confidence': [0.95, 0.88]      # Custom metric
}

df = pd.DataFrame(data)
vis = VisualizePath(path_file=df, output_folder='./output')
```

### Example 5: CSV File Input

```python
# Load directly from CSV file
vis = VisualizePath(
    path_file='path/to/edges.csv',
    output_folder='./output'
)
```

### Example 6: Excel File Input

```python
# Load directly from Excel file
vis = VisualizePath(
    path_file='path/to/edges.xlsx',
    output_folder='./output'
)
```

## Technical Details

### Column Detection Logic

1. **Source column**: Search for exact matches (`source`, `from`, `pre`) first, then pattern match (`*_pre`)
2. **Target column**: Search for exact matches (`target`, `to`, `post`) first, then pattern match (`*_post`)
3. **Weight column**: Search for known weight column names
4. **Numeric columns**: Scan all remaining columns for numeric data types

### Metric Mapping Process

```python
# Pseudo-code showing the process
numeric_cols = []
for col in dataframe.columns:
    if col not in [source, target, color]:
        if is_numeric(col):
            numeric_cols.append(col)

# Map to standard names for toggle support
for col in numeric_cols:
    if col.lower() == 'ratio' or col.lower() == 'connection_ratio':
        dataframe['connection_ratios'] = dataframe[col]
    elif col.lower() in ['probability', 'prob', 'trav_prob']:
        dataframe['traversal_probabilities'] = dataframe[col]
    else:
        # Preserve as custom metric
        dataframe[col] = dataframe[col]
```

### Output Features

When metrics are detected:
- **Interactive toggles**: Standard metrics (ratio, probability) can be toggled in visualizations
- **Edge styling**: Metrics affect edge width, color, and labels
- **Data export**: All metrics (standard and custom) are preserved in exported Excel files

## Testing

The historical test suite (`tests/test_enhanced_edgelist.py`) was
removed; the living equivalent is
[`tests/ui/test_edge_list_editor.py`](../../tests/ui/test_edge_list_editor.py):

- Test 1: Standard format (source/target/weight)
- Test 2: Pre/Post format
- Test 3: BodyId format (bodyId_pre/bodyId_post)
- Test 4: With ratio and probability columns
- Test 5: Type format with multiple metrics
- Test 6: Custom metric columns
- Test 7: From/To format
- Test 8: CSV file input
- Test 9: Excel file input

Run tests:
```bash
python -m pytest tests/ui/test_edge_list_editor.py
```

## Examples

Full example script available in `examples/Example_SimpleEdgeList.py` demonstrating:

1. Simple edge-list with source/target/weight
2. BodyId edge-list format
3. From/To edge-list format
4. Edge-list with ratio and probability columns
5. Edge-list with custom metric columns
6. Loading from CSV file
7. Loading from Excel file

Run examples:
```bash
python examples/Example_SimpleEdgeList.py
```

## Migration Guide

### From Rigid Format

**Before** (rigid naming):
```python
# Required exact column names
data = {
    'source': ['A', 'B'],
    'target': ['B', 'C'],
    'weight': [10, 15]
}
```

**After** (flexible naming):
```python
# Any of these work:
data1 = {'pre': ['A'], 'post': ['B'], 'count': [10]}
data2 = {'from': ['A'], 'to': ['B'], 'weights': [10]}
data3 = {'bodyId_pre': [123], 'bodyId_post': [234], 'synapse_count': [10]}
```

### Adding Metrics

**Before** (limited support):
```python
# Manual column creation required
vis = VisualizePath(path_file=df, output_folder='./output')
```

**After** (automatic detection):
```python
# Just add numeric columns - they're auto-detected
data = {
    'source': ['A', 'B'],
    'target': ['B', 'C'],
    'weight': [10, 15],
    'ratio': [0.5, 0.7],          # Auto-mapped
    'probability': [0.9, 0.85],   # Auto-mapped
    'custom_metric': [1.2, 1.5]   # Preserved
}
```

## Benefits

1. **Flexibility**: Work with various data formats without preprocessing
2. **Automatic detection**: No manual column mapping required
3. **Extended metrics**: Support for unlimited custom metrics
4. **Toggle support**: Standard metrics get interactive toggles
5. **Data preservation**: All metrics preserved in exports
6. **Backward compatible**: Existing code continues to work

## Limitations

- Source and target columns must exist (cannot be missing)
- Weight column must be numeric
- Color column (if present) is excluded from metric detection
- Custom metrics do not automatically get toggle controls (only standard mapped metrics)

## Future Enhancements

Potential improvements:
- User-defined metric mapping
- Custom toggle controls for any metric
- Metric validation and warnings
- Auto-detection of directed vs. undirected graphs
- Support for weighted vs. unweighted graphs

## See Also

- [CacheSystem_QuickStart.md](../core-features/CacheSystem_Guide.md) - Getting started with VisualizePath
- [FilterBy_Feature.md](../core-features/FilterBy_Feature.md) - Filtering pathway data
- [EDGE_WIDTH_SCALING.md](EDGE_WIDTH_SCALING.md) - Edge visualization controls
- [examples/Example_SimpleEdgeList.py](../../archive/examples/visualization/input_formats/Example_SimpleEdgeList.py) - Complete examples
- [tests/ui/test_edge_list_editor.py](../../tests/ui/test_edge_list_editor.py) - Test suite (the edge-list editor)

## Version History

- **v2.1.0+**: Enhanced edge-list format with flexible column recognition and automatic metric detection
- **v2.0.x**: Basic edge-list support with rigid column names
- **v1.x**: Path-based format only
