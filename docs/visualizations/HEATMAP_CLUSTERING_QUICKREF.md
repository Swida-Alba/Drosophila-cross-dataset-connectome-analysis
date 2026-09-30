# Heatmap Clustering - Quick Reference

## Quick Start

```python
from vispath import VisualizePath

# Create heatmap (clustering computed automatically)
vis = VisualizePath(path_file='connections.csv', output_folder='./output')
vis.create_heatmap()

# Open HTML file in browser
# Click "Clustered" button in "🔀 Ordering" section
```

## Button Location

```
Controls Panel
├── 📊 Metric (Weight/Ratio/Probability)
├── 🔀 Ordering ← **NEW!**
│   ├── [Original]    ← Input order
│   └── [Clustered]   ← Hierarchical clustering
├── 🔢 Scale (Linear/Log/Sqrt)
└── 🎨 Color Theme
```

## What It Does

**Original Order**:
- Shows neurons in the order they appear in your data
- Preserves manual ordering if intentional
- Default view

**Clustered Order**:
- Groups similar neurons together
- Uses hierarchical clustering (euclidean distance, Ward linkage)
- Reveals modular structure automatically
- Shows block-diagonal patterns for modules

## Visual Difference

### Original Order
```
Source neurons: A1, A2, B1, B2, C1, C2
Target neurons: X1, X2, Y1, Y2, Z1, Z2
Pattern: Mixed, hard to see structure
```

### Clustered Order
```
Source neurons: A1, A2, C1, C2, B1, B2
Target neurons: X1, Y1, Z1, X2, Y2, Z2
Pattern: Clear blocks, modules visible
```

## When to Use

### ✅ Use Clustering

- Exploring unknown connectivity
- Looking for functional modules
- Creating publication figures
- Comparing connectivity patterns
- Networks with >20 neurons

### ❌ Skip Clustering

- Data already meaningfully ordered
- Very small networks (<10 neurons)
- Random connectivity
- Need specific order preserved

## Common Patterns

### Block Diagonal
```
■ ■ · ·
■ ■ · ·
· · ■ ■
· · ■ ■
```
**Meaning**: Strong modular structure

### Upper Triangle
```
· ■ ■ ■
· · ■ ■
· · · ■
· · · ·
```
**Meaning**: Feed-forward network

### Symmetric
```
· ■ · ■
■ · ■ ·
· ■ · ■
■ · ■ ·
```
**Meaning**: Reciprocal connections

### Scattered
```
■ · ■ ·
· · ■ ■
■ ■ · ·
· ■ · ■
```
**Meaning**: Hub-and-spoke or random

## Tips

### Tip 1: Compare Orders
Toggle back and forth to see structure emerge:
1. Click "Original" - see raw data
2. Click "Clustered" - see organized pattern
3. Toggle to understand transformation

### Tip 2: Combine with Metrics
```
1. Click "Clustered"
2. Switch between Weight/Ratio/Probability
3. See if modules persist across metrics
```

### Tip 3: Use Log Scale
```
1. Click "Clustered"
2. Change scale to Log₂ or Log₁₀
3. Reveals weak connections within modules
```

### Tip 4: Export Clustered View
```
1. Click "Clustered"
2. Adjust plot size if needed
3. Use export button (camera icon)
4. Save as PNG for papers
```

## Technical Details

**Algorithm**: Hierarchical clustering with Ward linkage  
**Distance**: Euclidean  
**Complexity**: O(n² log n) for n neurons  
**Speed**: ~50ms for 100×100 matrix  
**Library**: scipy.cluster.hierarchy

## Troubleshooting

### Button doesn't appear?
→ Regenerate heatmap with updated code

### "Clustering not available" alert?
→ Data has constant rows or invalid values

### Clustered looks random?
→ Data may lack structure to cluster

### Too slow for large matrix?
→ Use filtered data or pre-cluster externally

## Examples

### Example 1: Modular Network
```python
# 3 groups of neurons
sources = ['A1','A2','A3', 'B1','B2','B3', 'C1','C2','C3']
targets = ['X1','X1','X1', 'X2','X2','X2', 'X3','X3','X3']
weights = [100, 95, 90,   80, 85, 82,   70, 75, 72]

df = pd.DataFrame({'source': sources, 'target': targets, 'weight': weights})
vis = VisualizePath(path_file=df, output_folder='./output')
vis.create_heatmap()
```
**Result**: 3 clear blocks in clustered view

### Example 2: With Multiple Metrics
```python
df = pd.DataFrame({
    'source': [...],
    'target': [...],
    'weight': [...],
    'ratio': [...],
    'probability': [...]
})
vis = VisualizePath(path_file=df, output_folder='./output')
vis.create_heatmap()

# Toggle between clustered + weight/ratio/probability
```

## See Also

- [HEATMAP_CLUSTERING_FEATURE.md](HEATMAP_CLUSTERING_FEATURE.md) - Complete documentation
- [Enhanced_EdgeList_Format.md](Enhanced_EdgeList_Format.md) - Input formats
- [VisualizeSelectedPaths_Guide.md](VisualizeSelectedPaths_Guide.md) - Full guide

## Keyboard Shortcuts

None currently. Click buttons to toggle.

## Browser Compatibility

✅ Chrome/Edge (recommended)  
✅ Firefox  
✅ Safari  
⚠️ IE11 (not supported)

---

**Quick tip**: Always try clustering first - you might discover structure you didn't know existed!
