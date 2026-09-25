# Negative Values - Quick Reference

## Summary
When edge weights contain negative values, the toolkit automatically handles them with special visual indicators.

## Visual Indicators

| Visualization | Positive Edges | Negative Edges |
|--------------|----------------|----------------|
| **Network** | Gray lines | Light blue lines |
| **Sankey** | Gray links | Light blue links |
| **Heatmap** | Normal colors | Signed transforms |

## Automatic Handling

### What Happens Automatically?
1. ✅ **Edge widths** use absolute values (no display errors)
2. ✅ **Negative sign** preserved in hover labels
3. ✅ **Color coding** distinguishes positive from negative
4. ✅ **Legend** appears when negatives are present
5. ✅ **Transforms** handle negatives correctly (log, sqrt)

### Console Messages
```
ℹ️  Found negative edge weights - using absolute values for width, light blue color for negative edges
```

## Hover Information

### Network Hover
```
Source → Target
Weight: -75 synapses
Ratio: 0.35
Probability: 0.42
```

### Sankey Hover
```
LHN_B → MBON_F
Weight: -105
Ratio: 0.28
Probability: 0.33
```

## Heatmap Transforms

All transform types work with negative values:

| Transform | Formula | Example |
|-----------|---------|---------|
| **Linear** | `v` | `-100` → `-100` |
| **Log2** | `sign(v) × log₂(|v| + 1)` | `-100` → `-6.66` |
| **Log10** | `sign(v) × log₁₀(|v| + 1)` | `-100` → `-2.00` |
| **Sqrt** | `sign(v) × √|v|` | `-100` → `-10.00` |

## Data Export

Excel files preserve original signs:
- **Weight columns**: Show negative values as-is
- **Connection tables**: Include negative weights
- **Ratio matrices**: May contain negative ratios

## Example Test Data

Test file: `test_data/test_paths_with_negatives.csv`

Sample negative connections:
```
PN_G → MBON_F: weight = -105
LHN_D → LHN_K: weight = -95
PN_A → LHN_E: weight = -90
PN_I → LHN_J: weight = -85
```

## Running Tests

```bash
python -m pytest tests/core/test_audit_fixes.py -k negative
```

(The one-off `scripts/PlotPath_TestNegatives.py` harness has been removed
together with its `test_negative_output/` artifacts and the
`test_data/` sample; the folder-safety contract lives in the pytest suite
above. That test covers negative PARAMETER formatting, not the
visualization styling.)

## Troubleshooting

### Network appears blank
- ✅ **Fixed**: Now uses absolute values in Cytoscape data
- ✅ Negative sign shown in hover labels

### Sankey hover shows only numbers
- ✅ **Fixed**: Now shows source, target, and metrics

### Heatmap shows NaN values
- ✅ **Fixed**: Signed transforms prevent NaN in log/sqrt scales

### Colors not showing correctly
- ✅ **Fixed**: Light blue for negative, gray for positive

## Color Scheme

```css
/* Positive edges */
rgba(100, 100, 100, 0.4)  /* Gray */

/* Negative edges */
rgba(74, 144, 226, 0.4)   /* Light Blue */
```

## Related Functions

| Function | Module | Handles Negatives? |
|----------|--------|-------------------|
| `VisualizePath.plot()` | `vispath_pkg.vispath` | ✅ Yes |
| `CreateHeatmap` (signed transforms) | `statvis` | ✅ Yes |
| `FindNeuronConnection.FindDirectConnections()` | `coana` | ➖ negatives not applicable (weights ≥ 0) |

`statvis.SankeyDirect` DROPS negative entries (`val > 0` filter) — the
negative-preserving Sankey path is vispath only.

## Need More Details?

See: [NEGATIVE_VALUES_IMPLEMENTATION.md](NEGATIVE_VALUES_IMPLEMENTATION.md)
