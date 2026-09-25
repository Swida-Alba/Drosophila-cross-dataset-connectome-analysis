# Custom Row and Column Ordering in Heatmaps

## Overview

The `VisualizePath` class supports custom ordering of rows and columns in heatmaps through **class attributes** and **method parameters**, allowing you to arrange nodes according to biological hierarchy, functional groups, or any custom organization.

## Three Ways to Set Custom Ordering

### 1. Via Class Initialization (Recommended for Consistent Ordering)

Set ordering when creating the VisualizePath object:

```python
from vispath import VisualizePath

vis = VisualizePath(
    path_file='data.csv',
    heatmap_row_order=['PN_A', 'PN_B', 'LHN_X'],
    heatmap_col_order=['LHN_X', 'MBON_1', 'MBON_2']
)

vis.visualize()  # All heatmaps will use custom ordering
```

### 2. Via Method Parameters (Recommended for One-Time Override)

Override ordering for a specific heatmap call:

```python
vis = VisualizePath('data.csv')
vis.build_network()

# Override for this specific heatmap
vis.create_heatmap(
    custom_row_order=['A', 'B', 'C'],
    custom_col_order=['X', 'Y', 'Z']
)
```

### 3. Via Attribute Modification (Recommended for Dynamic Ordering)

Modify attributes after initialization:

```python
vis = VisualizePath('data.csv')
vis.build_network()

# Get node info and create custom order dynamically
info = vis.get_heatmap_node_info()
vis.heatmap_row_order = info['source_only'] + info['intermediate']
vis.heatmap_col_order = sorted(info['target_only'])

vis.create_heatmap()  # Uses modified attributes
```

## Features

### Get Node Information

Before customizing order, you can inspect which nodes will appear in rows vs columns:

```python
from vispath import VisualizePath

vis = VisualizePath('data.csv')
vis.build_network()

# Print node information
vis.print_heatmap_node_info()

# Or get programmatically
info = vis.get_heatmap_node_info()
print("Row nodes:", info['row_nodes'])
print("Column nodes:", info['col_nodes'])
print("Dual-role nodes:", info['intermediate'])
```

**Output:**
```
================================================================================
HEATMAP NODE INFORMATION
================================================================================

Row nodes (sources): 5 total
  LHN_X, LHN_Y, PN_A, PN_B, PN_C

Column nodes (targets): 5 total
  LHN_X, LHN_Y, MBON_1, MBON_2, MBON_3

Node roles:
  Source-only:  3 nodes - PN_A, PN_B, PN_C
  Target-only:  3 nodes - MBON_1, MBON_2, MBON_3
  Both (intermediate): 2 nodes - LHN_X, LHN_Y

Note: Nodes in 'Both' category appear in BOTH rows AND columns
================================================================================
```

### Custom Order Specification

Specify the exact order you want nodes to appear:

```python
# Method 1: Set as attributes during initialization
vis = VisualizePath(
    path_file='data.csv',
    heatmap_row_order=['PN_A', 'PN_B', 'PN_C', 'LHN_X', 'LHN_Y'],
    heatmap_col_order=['LHN_X', 'LHN_Y', 'MBON_1', 'MBON_2', 'MBON_3']
)

# Method 2: Override with method parameters
vis.create_heatmap(
    custom_row_order=['PN_A', 'PN_B', 'PN_C'],
    custom_col_order=['MBON_1', 'MBON_2']
)
```

### Partial Custom Order

You can specify only some nodes - unspecified nodes will be appended at the end (sorted):

```python
# Only specify priority nodes
priority_rows = ['PN_A', 'LHN_X']  # Other nodes appended at end

heatmap_path = vis.create_heatmap(
    custom_row_order=priority_rows
)
```

Result: `['PN_A', 'LHN_X', 'LHN_Y', 'PN_B', 'PN_C']` (missing nodes sorted and appended)

## API Reference

### Class Attributes

**`heatmap_row_order`** : list of str or None
- Custom order for heatmap row nodes (sources)
- Set during initialization or modified later
- If None, uses alphabetical order

**`heatmap_col_order`** : list of str or None
- Custom order for heatmap column nodes (targets)
- Set during initialization or modified later
- If None, uses alphabetical order

### Methods

**`get_heatmap_node_info()`**

Returns a dictionary with node classification:

```python
{
    'row_nodes': ['LHN_X', 'LHN_Y', 'PN_A', 'PN_B', 'PN_C'],
    'col_nodes': ['LHN_X', 'LHN_Y', 'MBON_1', 'MBON_2', 'MBON_3'],
    'source_only': ['PN_A', 'PN_B', 'PN_C'],
    'target_only': ['MBON_1', 'MBON_2', 'MBON_3'],
    'intermediate': ['LHN_X', 'LHN_Y']
}
```

**`print_heatmap_node_info()`**

Prints formatted node information to console.

**`create_heatmap(custom_row_order=None, custom_col_order=None)`**

Create heatmap with optional custom ordering.

**Parameters:**
- `custom_row_order` (list of str, optional): Custom order for row nodes (overrides class attribute)
- `custom_col_order` (list of str, optional): Custom order for column nodes (overrides class attribute)

**Priority:** parameter > class attribute > default (sorted)

**Returns:**
- `str`: Path to generated HTML file

## Use Cases

### 1. Biological Hierarchy

Arrange nodes by their position in neural pathways:

```python
# Sensory neurons → Local neurons → Output neurons
custom_rows = ['PN1', 'PN2', 'LHN1', 'LHN2', 'MBON1']
custom_cols = ['LHN1', 'LHN2', 'MBON1', 'MBON2', 'Output']

vis.create_heatmap(custom_row_order=custom_rows, custom_col_order=custom_cols)
```

### 2. Functional Groups

Group nodes by function or brain region:

```python
# Group by brain region
visual_neurons = ['V1', 'V2', 'V3']
olfactory_neurons = ['PN_A', 'PN_B', 'PN_C']
output_neurons = ['MBON1', 'MBON2']

custom_order = visual_neurons + olfactory_neurons + output_neurons
vis.create_heatmap(custom_row_order=custom_order)
```

### 3. Comparative Analysis

Order nodes consistently across multiple heatmaps:

```python
# Use same order for all heatmaps
standard_order = ['A', 'B', 'C', 'D', 'E']

vis1.create_heatmap(custom_row_order=standard_order)
vis2.create_heatmap(custom_row_order=standard_order)
vis3.create_heatmap(custom_row_order=standard_order)
```

## Important Notes

1. **Node Availability**: Only nodes that actually exist in your data will be included
2. **Missing Nodes**: Nodes in `custom_order` but not in data are skipped
3. **Unspecified Nodes**: Nodes not in `custom_order` are appended at the end (sorted)
4. **Dual-Role Nodes**: Nodes that are both sources and targets appear in both rows AND columns
5. **Build Network First**: Must call `build_network()` before using custom ordering

## Example Workflow

```python
from vispath import VisualizePath

# 1. Load data
vis = VisualizePath('connections.csv', output_folder='./output')

# 2. Build network
vis.build_network()

# 3. Inspect nodes
vis.print_heatmap_node_info()

# 4. Define custom order
info = vis.get_heatmap_node_info()
custom_rows = ['PN_A', 'PN_B'] + sorted(info['intermediate'])

# 5. Create heatmap with custom order
heatmap = vis.create_heatmap(custom_row_order=custom_rows)

print(f"Heatmap created: {heatmap}")
```

## Testing

Run the test script to see examples:

(The one-off `scripts/test_custom_ordering.py` harness has been removed.)

This generates 5 heatmaps with different orderings:
1. Default (alphabetical)
2. Custom rows (reverse alphabetical)
3. Custom columns (reverse alphabetical)  
4. Custom both (biological hierarchy)
5. Partial custom (specified + remaining)

Compare the heatmaps to see how ordering affects visualization!
