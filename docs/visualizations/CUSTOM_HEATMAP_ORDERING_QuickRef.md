# Custom Heatmap Ordering - Quick Reference

## Three Ways to Set Ordering

### 1. Via Initialization (Recommended)
```python
vis = VisualizePath(
    path_file='data.csv',
    heatmap_row_order=['A', 'B', 'C'],
    heatmap_col_order=['X', 'Y', 'Z']
)
vis.visualize()  # Uses custom ordering
```

### 2. Via Method Parameters
```python
vis = VisualizePath('data.csv')
vis.build_network()
vis.create_heatmap(
    custom_row_order=['A', 'B', 'C'],
    custom_col_order=['X', 'Y', 'Z']
)
```

### 3. Via Attribute Modification
```python
vis = VisualizePath('data.csv')
vis.build_network()
vis.heatmap_row_order = ['A', 'B', 'C']
vis.heatmap_col_order = ['X', 'Y', 'Z']
vis.create_heatmap()  # Uses attributes
```

## Key Attributes & Methods

| Attribute/Method | Purpose |
|-----------------|---------|
| `heatmap_row_order` | Class attribute: custom row order |
| `heatmap_col_order` | Class attribute: custom column order |
| `get_heatmap_node_info()` | Returns dict with node classification |
| `print_heatmap_node_info()` | Prints node information |
| `create_heatmap(custom_row_order, custom_col_order)` | Creates heatmap (params override attributes) |

## Common Patterns

### Reverse Order
```python
info = vis.get_heatmap_node_info()
vis.create_heatmap(custom_row_order=list(reversed(info['row_nodes'])))
```

### Biological Hierarchy
```python
custom_order = sensory_neurons + local_neurons + output_neurons
vis.create_heatmap(custom_row_order=custom_order)
```

### Priority Nodes First
```python
priority = ['NodeA', 'NodeB']  # Rest appended automatically
vis.create_heatmap(custom_row_order=priority)
```

## Tips

✅ **DO:**
- Call `build_network()` first
- Use `get_heatmap_node_info()` to see available nodes
- Partial orders are OK (missing nodes auto-appended)

❌ **DON'T:**
- Include nodes not in your data (they're skipped)
- Forget that dual-role nodes appear in both rows AND columns

## Quick Test

(The one-off `scripts/test_custom_ordering.py` harness has been removed.)
