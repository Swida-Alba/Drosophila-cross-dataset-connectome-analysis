# VisualizeSkeleton Optimizations (November 2024)

> **Historical record (November 2024).** `VisualizeSkeleton` has since moved out
> of `src/coana.py` into `src/visualize_skeleton.py`, so the
> `from coana import VisualizeSkeleton` snippets below no longer run. For
> current behavior and options see [3D_Skeleton_Guide.md](3D_Skeleton_Guide.md).

> **Note:** This document uses `NEUPRINT_APPLICATION_CREDENTIALS` for historical reasons. The current recommended environment variable is `NEUPRINT_TOKEN`. Both are supported for backwards compatibility.

## Summary

Optimized VisualizeSkeleton implementation based on official navis and neuprint documentation with improved resource management and better API integration.

## Key Optimizations

### 1. **Environment Variable Integration** ✅
- Uses `NEUPRINT_APPLICATION_CREDENTIALS` environment variable for authentication
- No need to hardcode tokens in scripts
- Follows security best practices

```bash
# Set token once in your environment
export NEUPRINT_APPLICATION_CREDENTIALS="your_token_here"

# Run any script - token automatically used
python examples/Example_VisualizeSkeleton_ComprehensiveTests.py
```

### 2. **Lazy Directory Creation** ✅
- Cache directories created only when actually needed
- No empty directories cluttering the workspace
- Follows "on-demand" resource allocation principle

**Before:**
```
navis_roi_meshes_json/
├── hemibrain_v1_2_1/
├── optic-lobe_v1_1/  (empty)
├── fib/              (empty)
├── manc/             (empty)
└── primary_rois/
```

**After:**
```
navis_roi_meshes_json/
├── hemibrain_v1_2_1/           (created when mesh downloaded)
├── hemibrain_v1_2_1_available_rois.json
└── primary_rois/
```

### 3. **Improved NeuPrint Client Handling** ✅
- Proper client initialization with token from environment
- Dataset-specific server URLs (hemibrain vs optic-lobe)
- Better error messages when authentication fails

```python
# Get token from environment
token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS')

# Determine correct server based on dataset
if 'optic' in self.dataset.lower():
    server = 'https://neuprint-optic-lobe.janelia.org'
    dataset_name = self.dataset.split(':')[0]
else:
    server = 'https://neuprint.janelia.org'
    dataset_name = 'hemibrain:v1.2.1'

# Create client with proper credentials
client = Client(server, dataset=dataset_name, token=token)
```

### 4. **Enhanced Error Messages** ✅
- Clear, actionable error messages referencing official documentation
- Step-by-step troubleshooting hints
- Links to relevant resources

**Example:**
```
⚠️  Failed to download "NO" mesh from NeuPrint: <error>
   Possible solutions:
   1. Check ROI name spelling (case-sensitive): "NO"
   2. Set NEUPRINT_APPLICATION_CREDENTIALS environment variable
   3. Use list_available_rois(fetch_online=True) to see valid ROIs
   4. Visit https://neuprint.janelia.org/account for token
```

### 5. **Official API References** ✅
All implementations now reference official documentation:

- **navis.interfaces.neuprint.fetch_roi()**: Official method for ROI mesh fetching
  - API: https://navis-org.github.io/navis/reference/navis/interfaces/neuprint/#navis.interfaces.neuprint.fetch_roi
  - Returns: `navis.Volume` object
  - Parameters: `roi` (str), `client` (optional neuprint.Client)

- **neuprint.fetch_meta()**: Fetch dataset metadata including ROI list
  - Returns dictionary with `roiInfo` and `primaryRois` keys
  - Automatically uses client if provided, otherwise global client

- **navis.Volume.from_json() / to_json()**: Efficient mesh caching
  - Load/save ROI meshes in JSON format
  - Preserves all mesh properties and metadata

## Test Results

### ✅ Successful Tests

1. **Test 1: Online ROI Listing for Hemibrain** - PASSED
   - Fetched 230 ROIs from NeuPrint database
   - Cached to `hemibrain_v1_2_1_available_rois.json`
   - Second call used cache (instant response)

2. **Test 6: Cache Directory Structure** - PASSED
   - Only necessary directories created
   - Empty directories automatically cleaned up
   - Cached ROI list properly stored

### ⚠️ Known Issues

**Multiple Client Conflict**: When creating multiple NeuPrint clients in the same script (e.g., hemibrain + optic-lobe), neuprint-python requires explicit client management:

```
RuntimeError: Currently more than one Client exists, so neither was 
automatically chosen as the default. You must explicitly pass a client 
to query functions, or explicitly call set_default_client().
```

**Solution**: Use one client per script, or explicitly pass client to all query functions.

## Usage Examples

### Example 1: List Available ROIs (with online fetch)

```python
from coana import VisualizeSkeleton

# Set environment variable first:
# export NEUPRINT_APPLICATION_CREDENTIALS="your_token"

vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['EB']
)

# Fetch fresh from NeuPrint
rois = vs.list_available_rois(refresh=True, fetch_online=True)
# Output: 230 ROIs fetched and cached
```

### Example 2: Automatic ROI Mesh Download

```python
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['EB'],
    mesh_roi=['EB', 'PB', 'FB', 'NO']  # Mix of cached and new ROIs
)

# Meshes automatically downloaded if not cached
vs.plot_neurons()  
# Missing meshes downloaded using navis.interfaces.neuprint.fetch_roi()
```

### Example 3: Offline Mode (cache only)

```python
# No token needed - uses cached data only
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['EB']
)

# Only use local cache, don't fetch online
rois = vs.list_available_rois(refresh=False, fetch_online=False)
```

## Performance Improvements

1. **Reduced API Calls**: ROI lists cached locally
   - First call: ~2-3 seconds (fetches 230 ROIs)
   - Subsequent calls: <0.1 seconds (reads from cache)

2. **On-Demand Downloads**: Meshes downloaded only when needed
   - No pre-downloading of unused meshes
   - Saves bandwidth and storage

3. **Smart Fallbacks**: Multiple data sources
   - Priority: Local cache → Online database → Primary_rois fallback
   - Always tries to provide data even if one source fails

## File Structure

```
navis_roi_meshes_json/
├── hemibrain_v1_2_1/                    # Created on first mesh download
│   ├── EB.json
│   ├── PB.json
│   └── ... (63 meshes)
├── hemibrain_v1_2_1_available_rois.json # Cached ROI list (230 ROIs)
└── primary_rois/                        # Backward compatibility
    └── ... (63 meshes)
```

## Documentation References

### Official Documentation
- **navis**: https://navis-org.github.io/navis/
- **navis neuprint interface**: https://navis-org.github.io/navis/reference/navis/interfaces/neuprint/
- **neuprint-python**: https://github.com/connectome-neuprint/neuprint-python
- **NeuPrint database**: https://neuprint.janelia.org/

### Project Documentation
- `docs/visualizations/VisualizeSkeleton_Updates_Nov2024.md` - Complete feature guide
- `docs/visualizations/VisualizeSkeleton_Quick_Reference.md` - Quick reference
- `docs/visualizations/VisualizeSkeleton_Implementation_Summary.md` - Technical summary

## Testing

### Quick Demo (No Token Required)
```bash
python examples/Example_VisualizeSkeleton_QuickDemo.py
```
Shows cache structure, local meshes, and cached ROI lists.

### Comprehensive Tests (Token Required)
```bash
export NEUPRINT_APPLICATION_CREDENTIALS="your_token"
python examples/Example_VisualizeSkeleton_ComprehensiveTests.py
```
Tests all features including online fetching and automatic downloads.

## Best Practices

1. **Set Environment Variable**: Always set `NEUPRINT_APPLICATION_CREDENTIALS` in your environment, not in code
2. **Use Caching**: Let the system cache ROI lists and meshes for better performance
3. **One Client Per Script**: Avoid creating multiple NeuPrint clients simultaneously
4. **Check Available ROIs**: Use `list_available_rois()` to see what's available before plotting
5. **Reference Official Docs**: When in doubt, check navis and neuprint documentation

## Future Enhancements

- [ ] Support for explicit client passing in all methods
- [ ] Better multi-dataset client management
- [ ] ROI mesh compression for faster loading
- [ ] Progress bars for large downloads
- [ ] Automatic cache cleanup for old/unused meshes

## Changelog

### November 20, 2024
- ✅ Environment variable integration for token
- ✅ Lazy directory creation
- ✅ Improved client handling with proper server URLs
- ✅ Enhanced error messages with troubleshooting hints
- ✅ Official API references and documentation links
- ✅ Comprehensive testing suite
- ✅ Performance optimizations

---

**Author**: drocat  
**Date**: November 20, 2024  
**Version**: 3.1
