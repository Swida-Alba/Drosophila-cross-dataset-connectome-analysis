# Brain Template Transformation Fix (November 2024)

> **Historical record (November 2024).** `VisualizeSkeleton` has since moved out
> of `src/coana.py` into `src/visualize_skeleton.py`, so the
> `from coana import VisualizeSkeleton` snippets below no longer run. For
> current behavior and options see
> [3D_Skeleton_Guide.md](../visualizations/3D_Skeleton_Guide.md).

## Issue

Brain transformations were failing with error:
```
⚠️  Transforming skeletons failed: Target "JRC2018F" has no known bridging registrations. 
Did you mean "JRCFIB2018F" instead?
```

## Root Cause

The code was using the incorrect target template name `JRC2018F` instead of `JRCFIB2018F` for brain transformations with the `navis.xform_brain()` function.

## Fix Applied

Changed all transformation targets from `JRC2018F` to `JRCFIB2018F` throughout the codebase.

### Locations Fixed

1. **plot_skeleton()** - Skeleton transformations (2 locations)
   - Line 5644: `navis.xform_brain(neuron_vols, source='JRCFIB2018Fraw', target='JRCFIB2018F')`
   - Line 5652: Retry after transform download

2. **plot_synapses()** - Synapse coordinate transformations
   - Line 5723: `navis.xform_brain(xyz_df, source='JRCFIB2018Fraw', target='JRCFIB2018F')`

3. **plot_mesh()** - ROI mesh transformations (2 locations)
   - Line 6033: Transform downloaded meshes
   - Line 6050: Transform cached meshes

4. **_check_and_download_transforms()** - Template availability check
   - Line 5914: `flybrains.JRCFIB2018F` template check

5. **plot_mesh()** - Whole brain mesh loading (2 locations)
   - Line 6095: Initial load attempt
   - Line 6111: Retry after download

## Template Reference

The correct brain template hierarchy for hemibrain data:
- **Source**: `JRCFIB2018Fraw` - Raw hemibrain coordinate space
- **Target**: `JRCFIB2018F` - Full-brain coordinate space (corrected)
- **Brain Mesh**: `flybrains.JRCFIB2018F` - Full brain mesh template

## Verification

All transformations now use the correct template name:
```bash
grep -n "target='JRCFIB2018F'" src/coana.py
# Returns 5 matches (all transformation calls)

grep -n "flybrains.JRCFIB2018F" src/coana.py  
# Returns 4 matches (template references)
```

## Impact

✅ **Fixed**: Brain transformations now work correctly
✅ **Backward Compatible**: No API changes
✅ **All Features**: Skeleton, synapse, and mesh transformations all fixed

Users can now successfully use `brain_mesh='whole'` without transformation errors.

## Testing

To test the fix:
```python
from coana import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['EB'],
    brain_mesh='whole',  # Should now work correctly
    mesh_roi=['EB', 'PB']
)
vs.plot_neurons()
```

Expected output:
```
Transforming skeletons of layer 0...✓
plotting whole brain mesh...
✓ Whole brain mesh loaded successfully
```

---

**Date**: November 20, 2024  
**Status**: Fixed  
**Files Modified**: `src/coana.py`
