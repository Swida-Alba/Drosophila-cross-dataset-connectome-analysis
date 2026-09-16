# VisualizeSkeleton Quick Reference Card

## New Features (November 2024)

### 1. Multi-Dataset Support

```python
from coana import VisualizeSkeleton

# Hemibrain dataset
vs_hemi = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['EB'],
    mesh_roi=['EB', 'PB', 'FB']
)

# Optic-lobe dataset
vs_optic = VisualizeSkeleton(
    dataset='optic-lobe:v1.1',
    neuron_layers=['LNd'],
    mesh_roi=['ME(R)', 'AME(R)']
)
```

**✓ Automatic dataset-specific caching**  
**✓ No configuration needed**  
**✓ Backward compatible**

---

### 2. Discover Available ROIs

```python
# List all ROIs for current dataset
available_rois = vs.list_available_rois()

# Force refresh from NeuPrint
fresh_rois = vs.list_available_rois(refresh=True)
```

**✓ Queries NeuPrint database**  
**✓ Cached locally for speed**  
**✓ Fallback to local meshes**

---

### 3. Brain Transformation Confirmation

```python
# User will be prompted if transforms are missing
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['EB'],
    brain_mesh='whole'  # Requires JRC2018F transforms (~500MB)
)
```

**✓ One-time download**  
**✓ User confirmation required**  
**✓ Clear error messages**

---

## Directory Structure

```
navis_roi_meshes_json/
├── hemibrain_v1_2_1/     # Hemibrain ROI meshes
├── optic-lobe_v1_1/      # Optic-lobe ROI meshes
├── fib/                  # FIB ROI meshes
├── manc/                 # MANC ROI meshes
└── primary_rois/         # Backward compatibility
```

---

## Common Use Cases

### Load Specific ROIs
```python
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['EB', 'PB'],
    mesh_roi=['EB', 'PB', 'FB', 'NO'],  # Central complex
    mesh_color='gray',                   # Color for ROI meshes
    mesh_alpha=0.1,                      # Transparency (0.0-1.0; default: 0.1)
)
vs.plot_neurons()
```

### Custom ROI Mesh Colors
```python
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['KC.*'],
    mesh_roi=['MB(R)', 'CA(R)', 'PED(R)'],
    mesh_color=['red', 'green', 'blue'],       # Per-ROI colors
    mesh_alpha=0.15,                           # Uniform transparency
    brain_mesh_color='rgba(200,200,200,0.05)', # Separate brain mesh color
)
```

**Note:** `mesh_color`/`mesh_alpha` apply to ROI meshes only. Brain and VNC mesh colors are set separately via `brain_mesh_color` and `vnc_mesh_color`.

**Color format and alpha:** `neuron_colors`, `synapse_colors`, and
`mesh_color` accept named colors, `#RGB/#RGBA/#RRGGBB/#RRGGBBAA`, RGB(A)
tuples/lists with 0–255 channels or 0–1 floats, CSS `rgb()/rgba()`, and
`hsl()/hsla()`. Lists may mix formats. An embedded alpha overrides the
corresponding global opacity; entries without alpha inherit it.

**ROI Expansion:** Base names like `'AME'` auto-expand to `['AME(L)', 'AME(R)']`; each resolved mesh has its own legend entry.

---

### Quick ROI Selection with Keywords/Patterns

```python
# Single ROI (string input)
vs = VisualizeSkeleton(
    neuron_layers=['EB'],
    mesh_roi='EB',  # Auto-converted to ['EB']
)

# Load all primary brain regions
vs = VisualizeSkeleton(
    neuron_layers=['KC.*'],
    mesh_roi=['primary'],  # ~50-100 major ROIs
)

# Load ALL available ROIs
vs = VisualizeSkeleton(
    neuron_layers=['EB'],
    mesh_roi=['all'],  # Every ROI in the dataset
)

# Use regex patterns
vs = VisualizeSkeleton(
    neuron_layers=['LC.*'],
    mesh_roi=['ME.*', 'LO.*'],  # Match patterns
)

# Nested lists for color grouping
vs = VisualizeSkeleton(
    neuron_layers=['KC.*'],
    mesh_roi=['AME', ['aL', 'bL', 'gL'], 'EB'],  # Lobes share color
    mesh_color=['red', 'green', 'blue'],
)
```

| Pattern       | Matches                                             |
| ------------- | --------------------------------------------------- |
| `'primary'`   | Major brain regions (MB, AL, LH, optic lobes, etc.) |
| `'all'`       | All available ROIs for the dataset                  |
| `'ME.*'`      | All ROIs starting with 'ME'                         |
| `'.*\\(R\\)'` | All right-hemisphere ROIs                           |

**FAFB Note:** FAFB/FlyWire uses male-cns ROI meshes (auto-transformed).  
**Finding ROIs:** Check `cache/{dataset}/available_rois.json` or use `vs.list_available_rois()`.

---

### Legend Mode Options

```python
vs = VisualizeSkeleton(
    neuron_layers=['KC.*', 'MBON.*'],
    legend_mode='tree',    # 'layer', 'type', 'tree', or 'single'
)
```

| Mode       | Description                                      | Colors             |
| ---------- | ------------------------------------------------ | ------------------ |
| `'layer'`  | One legend entry per layer (all neurons grouped) | Layer colors       |
| `'type'`   | Separate entry per neuron type (toggleable)      | Keeps layer colors |
| `'tree'`   | Collapsible group/type/bodyId legend in HTML     | Keeps layer colors |
| `'single'` | Individual entry per neuron (toggleable)         | Keeps layer colors |

**Tree legend details:** In interactive HTML, custom layer names create a
`group → type → bodyId/instance` hierarchy. A singleton custom bodyId group is
shown as one direct row with count `1`; a skeleton and its soma mesh do not
inflate that count or create a redundant child leaf. Multi-neuron groups keep
their expandable child rows. For cross-dataset query overlays, the row uses
the source neuron's native `type` (for example, `SMP227`) rather than a mapped
`flywireType` value such as `CB*`. Static exports retain the ordinary Plotly
legend.

---

### Per-Neuron Colors via CSV

Use `layer_map_csv` with optional `color` column:

```csv
layer,id_type_instance,color
PN,1234567890,red
PN,1234567891,#00FF00
KC,KC_gamma,blue
```

```python
vs = VisualizeSkeleton(
    layer_map_csv='my_neurons.csv',
    legend_mode='type',
)
```

---

### Work with Multiple Datasets
```python
# Hemibrain
vs1 = VisualizeSkeleton(dataset='hemibrain:v1.2.1', neuron_layers=['EB'])
vs1.plot_neurons()

# Optic-lobe  
vs2 = VisualizeSkeleton(dataset='optic-lobe:v1.1', neuron_layers=['LNd'])
vs2.plot_neurons()
```

### Enable Whole Brain Mesh
```python
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['EB'],
    brain_mesh='whole',  # Requires transform confirmation
    brain_mesh_color='rgba(116, 168, 214, 0.04)'
)
vs.plot_neurons()
```

---

## Troubleshooting

### ROI Not Found
```
⚠️  mesh file custom_roi.json not found
```
**Fix:** Check available ROIs with `vs.list_available_rois()`

### Transform Download Failed
```
⚠️  Failed to load whole brain mesh
```
**Fix:** 
1. Install flybrains: `pip install navis[flybrains]`
2. Check internet connection
3. Manually download: `flybrains.download_jrc_transforms()`

### API Timeout
```
⚠️  Failed to fetch available ROIs from NeuPrint: timeout
```
**Fix:** Uses cached/local data automatically (no action needed)

---

## Performance Tips

✓ **Use cached ROI lists:** Avoid `refresh=True` unless needed  
✓ **Simplify meshes:** `mesh.simplify(factor=0.5)` for faster rendering  
✓ **Use line mode:** `skeleton_mode='line'` for large datasets  
✓ **Disable auto-open:** `show_fig=False` for batch processing  

---

## Documentation

- **Full Guide:** [docs/visualizations/VisualizeSkeleton_Updates_Nov2024.md](../visualizations/VisualizeSkeleton_Updates_Nov2024.md)
- **Examples:** [examples/Example_VisualizeSkeleton_MultiDataset.py](../../archive/examples/visualization/skeleton/Example_VisualizeSkeleton_MultiDataset.py)
- **Main README:** [README.md](../../README.md)

---

## External References

- [navis documentation](https://navis.readthedocs.io/)
- [flybrains GitHub](https://github.com/navis-org/navis-flybrains)
- [NeuPrint](https://neuprint.janelia.org/)

---

**Version:** v3.1 | **Last Updated:** November 2024
