# VisualizeSkeleton Multi-Dataset Support (November 2024)

> **Historical record (November 2024).** `VisualizeSkeleton` has since moved out
> of `src/coana.py` into `src/visualize_skeleton.py`, so the
> `from coana import VisualizeSkeleton` snippets below no longer run. For
> current behavior and options see [3D_Skeleton_Guide.md](3D_Skeleton_Guide.md).

## Overview

The `VisualizeSkeleton` class has been enhanced to support multiple NeuPrint datasets with intelligent ROI mesh caching, automatic ROI discovery, and smart brain transformation handling.

## Key Features

### 1. Dataset-Specific ROI Mesh Caching

ROI meshes are now organized by dataset for better scalability and organization:

```
navis_roi_meshes_json/
├── primary_rois/         # Backward compatibility (hemibrain)
├── hemibrain_v1_2_1/     # Hemibrain-specific meshes
├── optic-lobe_v1_1/      # Optic-lobe-specific meshes
├── fib/                  # FIB-specific meshes
└── manc/                 # MANC-specific meshes
```

**Benefits:**
- Each dataset maintains its own ROI mesh cache
- Automatic fallback to `primary_rois/` for backward compatibility
- Cleaner organization for multi-dataset projects
- Reduces conflicts between datasets with overlapping ROI names

**Implementation:**
The `_get_dataset_mesh_dir()` method automatically determines the correct cache directory based on the dataset parameter. If a dataset-specific directory doesn't exist, it falls back to `primary_rois/`.

### 2. ROI Discovery from NeuPrint

New `list_available_rois()` and `_get_available_rois()` methods enable automatic ROI discovery:

```python
from coana import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='optic-lobe:v1.1',
    neuron_layers=['LNd'],
)

# List all available ROIs for this dataset
available_rois = vs.list_available_rois()
print(f"Found {len(available_rois)} ROIs: {available_rois[:10]}...")
```

**Features:**
- Queries NeuPrint database for dataset-specific ROI list
- Caches results locally to minimize API calls
- Auto-refreshes with `list_available_rois(refresh=True)`
- Fallback to local mesh directory if API unavailable

**Cache Location:**
ROI lists are cached at:
```
navis_roi_meshes_json/{dataset}_available_rois.json
```

### 3. Multi-Dataset Brain Mesh Support with Transform Downloads

**NEW**: `brain_mesh` parameter now supports `'none'`, `'template'`, or `'whole'` with automatic dataset-specific template selection:

```python
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['EB'],
    brain_mesh='whole',  # Triggers transform download if needed
)
```

**Supported Datasets:**

| Dataset | Template (EM) | Whole Brain/VNC | Transform Required |
|---------|--------------|-----------------|-------------------|
| **hemibrain:v1.2.1** | JRCFIB2018F | JRC2018F (whole brain) | ✅ Yes (~10GB) |
| **optic-lobe:v1.1** | JRCFIB2018F | JRC2018F (whole brain) | ✅ Yes (~10GB) |
| **manc:v1.2.3** | MANC (native VNC) | — | ❌ No |
| **male-cns:v0.9** | JRCFIB2022M (native) | — | ❌ No |

**Brain Mesh Options:**

```python
# Option 1: No brain mesh (fastest)
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['KC.*'],
    brain_mesh='none'  # No background mesh, clearest view
)

# Option 2: Template mesh (native EM resolution, 0.5-2s)
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['KC.*'],
    brain_mesh='template'  # JRCFIB2018F for hemibrain, no downloads
)

# Option 3: Whole brain/VNC mesh (standard resolution, one-time download)
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['KC.*'],
    brain_mesh='whole'  # JRC2018F whole brain, ~500MB download
)

# MANC dataset (native VNC, no transforms)
vs = VisualizeSkeleton(
    dataset='manc:v1.2.3',
    neuron_layers=['DN.*'],
    brain_mesh='template'  # MANC template, instant
)
```

**Transform Download Process:**

When using `brain_mesh='whole'` for hemibrain/optic-lobe, the system automatically checks for required transforms:

1. Checks if transforms exist in `~/flybrain-data`
2. If not found, prompts user:
   ```
   ⚠ Brain transforms not found for hemibrain:v1.2.1
     Target template: JRC2018F (whole brain)
     Transform size: ~10GB total (8 files, one-time download)
     Download time: ~1-2 hours
     Storage location: ~/flybrain-data
   
   Download transforms? [y/N]:
   ```
3. Downloads ALL JRC transforms (~10GB) to `~/flybrain-data` if confirmed
   - Only JRC2018F_JRCFIB2018F.h5 (~1.3GB) is used for hemibrain/optic-lobe
   - Other files (~8.7GB) enable cross-dataset functionality
   - Selective download not supported by flybrains package
4. Transforms persist and are shared across all NeuPrint projects and datasets
5. Automatic fallback to `brain_mesh='template'` if declined

**Benefits:**
- **Dataset-aware:** Automatically selects correct templates per dataset
- **No surprises:** User confirmation before large downloads
- **Persistent:** One-time download, shared across projects
- **Graceful:** Automatic fallback if transforms unavailable
- **Native templates:** MANC and male-cns don't need transforms

### 4. Enhanced Error Handling and Fallbacks

All new features include comprehensive error handling:

- **Missing ROI meshes:** Clear warning with fallback to available meshes
- **Transform failures:** Automatic retry after download or fallback to `brain_mesh='none'`
- **API failures:** Fallback to local mesh directory
- **Missing packages:** Clear instructions for installation

## Usage Examples

### Example 1: Basic Multi-Dataset Usage

```python
import statvis as sv
from coana import VisualizeSkeleton

# Hemibrain dataset
sv.LogInHemibrain(token='YOUR_TOKEN', dataset='hemibrain:v1.2.1')

vs_hemi = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['EB', 'PB'],
    mesh_roi=['EB', 'PB', 'FB'],  # Central complex
)
vs_hemi.plot_neurons()

# Optic-lobe dataset
sv.LogInHemibrain(token='YOUR_TOKEN', dataset='optic-lobe:v1.1')

vs_optic = VisualizeSkeleton(
    dataset='optic-lobe:v1.1',
    neuron_layers=['LNd'],
    mesh_roi=['ME(R)', 'AME(R)'],  # Optic lobe regions
)
vs_optic.plot_neurons()
```

### Example 2: ROI Discovery

```python
# Discover available ROIs for a dataset
vs = VisualizeSkeleton(dataset='hemibrain:v1.2.1', neuron_layers=['EB'])

# First call: fetches from NeuPrint and caches locally
available_rois = vs.list_available_rois()

# Subsequent calls: uses cached data (fast)
rois_cached = vs.list_available_rois()

# Force refresh from API
rois_fresh = vs.list_available_rois(refresh=True)
```

### Example 3: Brain Transformation with Confirmation

```python
# First time: prompts for download confirmation
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['EB'],
    mesh_roi=['EB'],
    brain_mesh='whole',  # Requires JRC2018F transforms
)

# Output:
# ======================================================================
# ⚠️  Brain Transformation Required
# ======================================================================
# To use brain_mesh="whole", you need to download JRC2018F brain transforms.
# This is a one-time download of approximately 500MB (compressed).
# 
# The transforms will be cached locally at:
#   ~/.navis/transforms/
# 
# For more information, see:
#   https://github.com/navis-org/navis-flybrains
# ======================================================================
# Download transforms now? [y/N]: y

# Subsequent runs: uses cached transforms (no prompt)
vs.plot_neurons()
```

### Example 4: Backward Compatibility

Existing scripts continue to work without modification:

```python
# Old script (still works)
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['EB'],
    mesh_roi=['EB', 'PB'],
)
# Automatically uses hemibrain_v1_2_1/ or falls back to primary_rois/
vs.plot_neurons()
```

## API Reference

### New Methods

#### `_get_dataset_mesh_dir()`
Returns the dataset-specific mesh directory path with automatic fallback.

**Returns:** `str` - Path to mesh directory

#### `_get_available_rois(use_cache=True)`
Queries NeuPrint for available ROIs in the current dataset.

**Parameters:**
- `use_cache` (bool): Use cached ROI list if available

**Returns:** `list` - Sorted list of ROI names

#### `list_available_rois(refresh=False)`
User-facing method to list and display available ROIs.

**Parameters:**
- `refresh` (bool): Force refresh from API (ignore cache)

**Returns:** `list` - Sorted list of ROI names

#### `_check_and_download_transforms()`
Checks for JRC2018F transforms and prompts for download if needed.

**Returns:** `bool` - True if transforms available, False otherwise

### Updated Methods

#### `plot_mesh()`
Enhanced to use dataset-specific caching with intelligent fallback:

**Behavior:**
1. Checks dataset-specific cache directory first
2. Falls back to `primary_rois/` if not found
3. Handles transform confirmation for `brain_mesh='whole'`
4. Comprehensive error handling with user-friendly messages

**Documentation References:**
- [navis Volume API](https://navis.readthedocs.io/en/latest/source/api.html#navis.Volume)
- [flybrains templates](https://github.com/navis-org/navis-flybrains)
- [Mesh optimization guide](https://navis.readthedocs.io/en/latest/)

#### `plot_skeleton()`
Enhanced transform handling with automatic retry:

**Behavior:**
1. Attempts brain transformation
2. On failure, triggers `_check_and_download_transforms()`
3. Retries transformation after successful download
4. Falls back to `brain_mesh='none'` if still fails

## Best Practices

### 1. Mesh Optimization

For faster rendering with large brain meshes:

```python
import navis

# Load and simplify mesh
mesh = navis.Volume.from_json('path/to/roi.json')
simplified_mesh = mesh.simplify(factor=0.5)  # Reduce vertices by 50%
simplified_mesh.to_json('path/to/roi_simplified.json')
```

**References:**
- [navis mesh simplification](https://navis.readthedocs.io/en/latest/source/api.html#navis.Volume.simplify)
- [Mesh compression techniques](https://navis.readthedocs.io/en/latest/)

### 2. Brain Transformations

Use appropriate template spaces for your dataset:

```python
# Hemibrain: JRCFIB2018Fraw -> JRC2018F
neurons_transformed = navis.xform_brain(
    neurons, 
    source='JRCFIB2018Fraw', 
    target='JRC2018F'
)

# For other datasets, check available templates:
import flybrains
print(flybrains.available_templates())
```

**References:**
- [flybrains documentation](https://github.com/navis-org/navis-flybrains)
- [JRC2018F brain template](https://www.janelia.org/open-science/jrc-2018-brain-templates)

### 3. ROI Mesh Storage

Organize custom ROI meshes by dataset:

```bash
# Add custom meshes to dataset-specific directories
navis_roi_meshes_json/
├── hemibrain_v1_2_1/
│   ├── EB.json
│   └── custom_roi.json
└── optic-lobe_v1_1/
    └── custom_optic_roi.json
```

### 4. Performance Tips

- Use `skeleton_mode='line'` for faster rendering with large datasets
- Enable mesh simplification for brain meshes (see Mesh Optimization)
- Cache ROI lists locally to avoid repeated API calls
- Use `show_fig=False` for batch processing

## Migration Guide

### From Previous Versions

**No breaking changes!** All existing scripts will continue to work. The system automatically:
- Falls back to `primary_rois/` if dataset-specific directory doesn't exist
- Uses existing hemibrain meshes for backward compatibility
- Maintains all previous parameter behaviors

**Optional Enhancements:**
To benefit from new features, optionally:
1. Organize ROI meshes by dataset (see directory structure above)
2. Use `list_available_rois()` to discover ROIs dynamically
3. Set `brain_mesh='whole'` with confidence (confirmation prompt added)

## Troubleshooting

### Issue: ROI mesh not found

**Error:**
```
⚠️  mesh file custom_roi.json not found in navis_roi_meshes_json/hemibrain_v1_2_1/ or primary_rois/
```

**Solution:**
1. Check available ROIs: `vs.list_available_rois()`
2. Verify mesh file exists in correct directory
3. Use exact ROI name (case-sensitive)

### Issue: Transform download fails

**Error:**
```
⚠️  Failed to load whole brain mesh: ...
```

**Solution:**
1. Check internet connection
2. Verify flybrains installation: `pip install navis[flybrains]`
3. Manually download transforms: `flybrains.download_jrc_transforms()`
4. Check disk space (~500MB required)

### Issue: API timeout when fetching ROIs

**Error:**
```
⚠️  Failed to fetch available ROIs from NeuPrint: timeout
```

**Solution:**
1. Check NeuPrint service status: [neuprint.janelia.org](https://neuprint.janelia.org/)
2. Use cached data: `vs.list_available_rois()`  (uses cache by default)
3. Fallback to local meshes (automatic)

## See Also

- [VisualizePath Updates (November 2024)](./VisualizePath_Updates_Nov2025.md)
- [Main Documentation](../README.md)
- [navis documentation](https://navis.readthedocs.io/)
- [flybrains GitHub](https://github.com/navis-org/navis-flybrains)
- [NeuPrint documentation](https://neuprint.janelia.org/)

## Testing

See `examples/Example_VisualizeSkeleton_MultiDataset.py` for comprehensive test suite covering:
- Dataset-specific mesh caching
- ROI discovery from NeuPrint
- Brain transformation confirmation
- Backward compatibility
- Error handling

Run tests:
```bash
cd examples
python Example_VisualizeSkeleton_MultiDataset.py
```

---

**Last Updated:** November 2024  
**Version:** v3.1
