# 3D Skeleton Visualization Guide

Three-dimensional rendering of neuron morphologies and spatial connectivity in the Drosophila brain.

## Overview

The 3D skeleton visualization displays:
- **Neuron morphologies** as 3D skeletons with accurate spatial coordinates
- **Brain regions** (ROIs) as semi-transparent meshes
- **Connections** between neurons with spatial context
- **Interactive exploration** with rotation, zoom, and selection

## Example Input Data

### Required Input

3D skeleton visualization requires **neuron bodyIds** from NeuPrint:

| Input Type              | Description                      | Example                     | Source                   |
| ----------------------- | -------------------------------- | --------------------------- | ------------------------ |
| **bodyId**              | Unique neuron identifier         | 123456789, 987654321        | NeuPrint database        |
| **Neuron types**        | Type name (converted to bodyIds) | 'KC_alpha', 'MBON03'        | Fetched via NeuPrint API |
| **Layer specification** | Multi-layer pathway              | ['KC.*', 'MBON.*', 'DAN.*'] | From path analysis       |

### Example Input Files

**From FindAllPath results**:
- Path: `connection_data/*/allpaths_*/path_bodyId_info.xlsx`
- Extract bodyIds from 'path_bodyId' sheet
- Example script: [`scripts/plot3dSkeleton.py`](../../scripts/plot3dSkeleton.py)

**Direct bodyId list**:
```python
# Option 1: Direct bodyId list
bodyIds = [
    123456789,  # KC neuron
    987654321,  # MBON neuron
    111222333   # DAN neuron
]

# Option 2: From neuron types
neuron_types = ['KC_alpha', 'MBON03', 'PPL1-01']
# Will be converted to bodyIds automatically

# Option 3: From path layers (recommended)
neuron_layers = ['KC.*', 'MBON03', 'DAN']
# Or as string: 'KC.* -> MBON03 -> DAN'
```

**Example datasets**:
- Sample bodyIds: See [`examples/README.md`](../../archive/examples/README.md) for neuron lists
- Type-level data: `datasets/hemibrain_v1_2_1_alltypes_neuron_df.csv`
- Path results: Output from `scripts/FindPath.py` contains bodyId information

### Getting BodyIds

**From neuron types**:
```python
from neuprint import fetch_neurons, Client

client = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1', token='your_token')
neurons_df, roi_counts = fetch_neurons(NC(type='KC_alpha'))
bodyIds = neurons_df['bodyId'].tolist()
```

**From path analysis results**:
```python
import pandas as pd

# Load path results
paths = pd.read_excel('path_results.xlsx', sheet_name='path_bodyId')

# Extract unique bodyIds
all_bodyIds = set()
for col in paths.columns:
    if 'bodyId' in col or col in ['source', 'target']:
        all_bodyIds.update(paths[col].dropna().unique())
```

## Quick Start

### Basic Usage

```python
import statvis as sv
from coana import VisualizeSkeleton

# Login to NeuPrint
sv.LogInHemibrain(token='your_token', dataset='hemibrain:v1.2.1')

# Visualize neurons with 3D skeletons
vs = VisualizeSkeleton(
    neuron_layers=['KC.*', 'MBON03'],  # Neuron types to visualize
    brain_mesh='native',                # 'native', 'BANC', 'FAFB', 'male-cns', or 'none'
    mesh_roi=['MB(R)', 'CA(R)'],       # ROI meshes to display
    neuron_alpha=0.2,                   # Neuron transparency (legend shows full opacity)
    synapse_size=3,                     # Synapse marker size
    skeleton_mode='tube',               # 'tube' or 'line'
    legend_mode='layer',                # 'layer', 'type', 'tree', or 'single'
    freeze_view=True,                   # Viewer pages keep their 3D framing on legend toggles
    expand_colors='interpolation',      # Color expansion: 'interpolation' or 'darken'
    show_fig=True
)

vs.plot_neurons()

# Export rotating video
vs.export_video(fps=30, degree_per_frame=1.0, rotate='horizontal')
```

### Using CSV Layer Maps

Define layers and optionally per-neuron colors in a CSV file:

```csv
layer,id_type_instance,color
Inputs,1234567890,red
Inputs,1234567891,#FF4444
Outputs,KC_gamma_s1,blue
Outputs,KC_gamma_s2,
```

```python
vs = VisualizeSkeleton(
    layer_map_csv='my_neurons.csv',  # CSV with layer, id_type_instance, optional color
    legend_mode='type',              # Each type gets separate legend entry
    # neuron_layers is overridden by layer_map_csv
)
```

The `color` column is optional. If omitted or empty, neurons use the default layer color.

### Video Export

Export rotating 3D visualization to MP4 video:

```python
# Basic video export (after plot_neurons())
vs.export_video(fps=30, degree_per_frame=1.0)  # 12 sec video at 30fps

# Faster rotation (shorter video)
vs.export_video(fps=30, degree_per_frame=2.0)  # 6 sec video

# Vertical rotation (tumbling motion)
vs.export_video(fps=30, rotate='vertical')

# High quality export
vs.export_video(fps=30, scale=4, width=1920, height=1080)

# Reuse cached frames (fast regeneration)
vs.export_video(fps=60, use_existing_images=True)
```

**Standalone Export (from existing HTML file):**

```python
from visualize_skeleton import export_video_from_html

# No VisualizeSkeleton initialization needed
export_video_from_html('/path/to/my_neurons.html', fps=30)

# With options
export_video_from_html(
    '/path/to/my_neurons.html',
    fps=30,
    degree_per_frame=1.0,
    rotate='horizontal',
    scale=4
)
```

**Re-exporting a stored page:** reading a page back is supported and is how
an old run is re-exported without re-fetching skeletons. Pass an existing page
to `vs.export_video(html_file='/path/to/my_neurons.html')`, or use the
standalone `export_video_from_html()` above. Both rebuild a
`plotly.graph_objects.Figure` through `figure_from_plotly_html()`, which reads
the trace data, layout and config out of the page's own `Plotly.newPlot(...)`
call via `figure_payload_from_html()` — the pinned plotly 6.4 has no
`plotly.io.read_html()`, so DROCAT parses the page itself (from the end of the
file, because the embedded plotly.js bundle contains the same literal). Point at
the run folder or the canonical `{saveas}.html`: a `{saveas}_simplified.html`
copy is resolved back to the canonical page (`resolve_viewer_page`), and the
profile re-exporter reads `visualization_manifest.json` for the cameras and
trace roles the page itself does not carry.

`export_video_from_html()` also takes `export_method`:

```python
# One Chrome session for the whole turn instead of one kaleido render per frame
export_video_from_html('.../scene.html', export_method='webdriver', timeout=120)

# Everything re-exported lands beside the page, in its own folder
from visualize_skeleton import reexport_output_dir
reexport_output_dir('.../scene.html')      # '.../scene_reexport'
```

With `export_method='webdriver'` the frames are rendered by
`_render_video_frames_via_session`, which loads the page once into a single
`WebDriverExportSession` and then only moves the camera (`_rotation_camera`,
using the orbit radius the page itself baked and the handedness its manifest
recorded, `+1` for MANC scenes via `_page_z_sign`), so a scene too large for
kaleido can be re-exported at all; if no browser can be started it prints a
warning and falls back to kaleido, rendering one frame per `write_image()` call.
`background_color=None` composites the frames on the
background the run recorded in its manifest (white when there is none), and
`timeout` is the WebDriver page-load timeout (ignored by kaleido).
`reexport_output_dir()` names the default destination for a re-export: the
`<page stem>_reexport/` folder beside the source page, so the run itself is
never rewritten and re-running the same settings reuses the frames already on
disk.

**Video Export Parameters:**

| Parameter             | Default      | Description                                    |
| --------------------- | ------------ | ---------------------------------------------- |
| `fps`                 | 30           | Frames per second                              |
| `degree_per_frame`    | 1.0          | Rotation angle per frame (1.0 → 360 frames)    |
| `rotate`              | 'horizontal' | Rotation direction: 'horizontal' or 'vertical' |
| `scale`               | 2            | Resolution multiplier                          |
| `use_existing_images` | True         | Reuse cached frame images if available         |
| `export_method`       | class attribute / `'kaleido'` for `export_video_from_html()` | Frame engine: `'webdriver'` (one session) or `'kaleido'` |
| `timeout`             | 120 (`export_video_from_html()` only) | WebDriver page-load timeout in seconds |
| `background_color`    | run manifest, else `'white'` (`export_video_from_html()` / `export_individuals_from_html()`) | Background the webdriver frames are composited on |

**Output Files:**
- `{folder}/pics_{fps}fps_{plane}/` - Cached frame images
- `{folder}/{name}_video_{h|v}_forward.mp4` - Forward rotation
- `{folder}/{name}_video_{h|v}_backward.mp4` - Reverse rotation
- `{folder}/{name}_video_{h|v}_{forward|backward}.gif` - GIF copies, when `export_gif=True`

### Re-export from the UI

The Skeleton tab carries a **Re-export from a Stored Page** card below the
render controls, for re-profiling or re-filming a run without touching the
dataset again (`ui/components/skeleton_reexport.py`, run as the
`plot3d_reexport` tool). The **Stored 3D Skeleton Page** picker lists the
top-level viewer HTML of every `plot-3d_*` run folder under the configured
output roots and recent runs (newest first, capped at 40 entries,
`_simplified.html` copies never offered); **Refresh list** rescans. A run from
before `visualization_manifest.json` existed still re-exports — its profiles are
grouped from the page's own stamped identity, but its cameras then come from the
built-in default view set rather than the run's own dataset.

| Knob | Effect |
| --- | --- |
| **Re-export Profiles** | Renders one PNG per profile group into `<stem>_reexport/individual_profiles/`. |
| **Re-export Granularity** | The same `granularity` the renderer offers (`legend` / `layer` / `type` / `body`); a page can only be split as finely as its own legend identity allows. |
| **Re-export Views** | Camera views rendered per profile group. |
| **Re-export Video** | Adds a full turn to `<stem>_reexport/`: `pics_{fps}fps_{plane}/`, `{stem}_video_{h\|v}_{forward\|backward}.mp4` and GIFs. Existing frames are reused when the settings match. |
| **Video FPS / Degrees / Frame / Rotate / Also GIF / GIF Scale** | Frame rate, rotation step per frame, rotation plane, GIF conversion and GIF downscale — the same values `export_video_from_html()` takes. |
| **Frame Method** | `webdriver` (default here) renders the whole turn in one Chrome session; `kaleido` renders frame by frame. A `webdriver` run that cannot start a browser falls back to kaleido by itself. |
| **Frame Scale** | Resolution multiplier for the profile PNGs and the video frames. |
| **Frame Auto-crop** | Trims the background off profiles and, for the video, applies one consistent crop across all frames. |

**Re-export from Page** runs it as an ordinary subprocess, so the output panel,
progress steps and log behave exactly as during a render. Nothing in the source
run folder is modified: every artifact is written into the sibling
`<stem>_reexport/` folder.

---

## Export Methods: WebDriver vs Kaleido

VisualizeSkeleton supports two export engines for PNG/video generation, each with different trade-offs:

### Method Comparison

| Feature           | **Kaleido**                       | **WebDriver** (default)             |
| ----------------- | --------------------------------- | ----------------------------------- |
| **Speed**         | 🐢 Slower (one-by-one rendering) | ⚡ Fast (browser-based screenshot ) |
| **Quality**       | ✅ Excellent                     | ✅ Excellent                        |
| **Max HTML Size** | ~100 MB                         | ~200 MB+                           |
| **WebGL Support** | ❌ Limited (rasterization)       | ✅ Full (native WebGL)              |
| **Dependencies**  | `kaleido` only                  | `selenium` + Chrome                |
| **Reliability**   | ✅ Very stable                   | ⚠️ Requires Chrome version match    |
| **Large Figures** | ❌ May timeout                   | ✅ Better handling                  |

### When to Use Each Method

`VisualizeSkeleton.export_method` defaults to `'webdriver'` (the UI agrees), and
an export that cannot start a browser falls back to kaleido automatically. The
standalone `export_video_from_html()` re-exporter is the exception: it defaults
to `'kaleido'` and only uses the browser when asked (see
[Re-export from the UI](#re-export-from-the-ui)).

**Use `export_method='kaleido'` when:**
- HTML file size < 100 MB
- You want minimal dependencies
- Chrome or Selenium is unavailable
- Simple 3D scenes without complex WebGL

**Use `export_method='webdriver'` when:**
- HTML file size > 100 MB
- Kaleido times out on complex figures
- You need native WebGL rendering quality
- Exporting rotating videos (efficient frame generation)

### Configuration Example

```python
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['MBON.*'],
    
    # Export method selection
    export_method='webdriver',      # 'kaleido' or 'webdriver'
    
    # WebDriver-specific settings
    webdriver_render_wait=0.3,      # Seconds between frames (None = auto-calibrate)
    
    # Shared export settings
    export_scale=3,                 # Resolution multiplier (1-5)
    export_timeout=60,              # Timeout per frame (kaleido only)
    
    # Auto-simplification threshold
    html_size_cap=150,              # MB threshold for auto-simplification
                                    # None = auto (100MB kaleido, 200MB webdriver)
)
```

### Auto-Simplification

For large figures, VisualizeSkeleton automatically simplifies meshes before export:

| Export Method | Default Threshold | Behavior                                       |
| ------------- | ----------------- | ---------------------------------------------- |
| Kaleido       | 100 MB            | Simplify if HTML > threshold, retry on timeout |
| WebDriver     | 200 MB            | Simplify if HTML > threshold                   |

**Manual override:**
```python
# Disable auto-simplification (may cause timeouts)
vs = VisualizeSkeleton(..., html_size_cap=9999)

# Force aggressive simplification
vs = VisualizeSkeleton(..., html_size_cap=50)
```

When simplification occurs, a `{name}_simplified.html` file is saved for reference.

---

## Legend and Color Configuration

### Legend Display Modes

Control how neurons appear in the legend using `legend_mode`:

```python
vs = VisualizeSkeleton(
    neuron_layers=['KC.*', 'MBON.*', 'DAN.*'],
    
    # Legend mode options:
    legend_mode='layer',   # Default: one entry per layer (grouped)
    # legend_mode='type',  # Group by neuron type within layers
    # legend_mode='tree',  # Interactive group/type/bodyId legend in HTML
    # legend_mode='single', # Each neuron gets its own entry
)
```

**Legend Modes:**

| Mode       | Description                                              | Best For                              |
| ---------- | -------------------------------------------------------- | ------------------------------------- |
| `'layer'`  | One legend entry per layer (all neurons grouped)         | Clean overview, many neurons          |
| `'type'`   | Separate entry for each neuron type (keeps layer colors) | Mixed types per layer, toggling types |
| `'tree'`   | Collapsible interactive group/type/bodyId legend in HTML | Multi-layer and bodyId inspection     |
| `'single'` | Individual entry for each neuron ({bodyId}_{layer_name}) | Tracking specific neurons, few IDs    |

**Note:** In `'type'` and `'single'` modes, neurons **keep their layer colors** (from `neuron_colors`) but get separate legend entries. This allows you to toggle visibility of individual types/neurons while maintaining consistent coloring within each layer.

### Interactive tree legend

`legend_mode='tree'` adds a collapsible legend panel to interactive Plotly
HTML exports. Static PNG, video, and other non-HTML exports keep the normal
Plotly legend. The panel uses the following hierarchy:

- Without custom layer names: `type → bodyId/instance`.
- With `custom_layer_names` or `layer_map_csv`: `group → type → bodyId/instance`.

Counts represent unique neuron items, not raw Plotly traces. This matters when
one neuron emits both a skeleton trace and a soma mesh trace: a custom group
containing that one body is shown as a single direct row with count `1` and no
redundant child leaf. Groups containing multiple neurons remain expandable.

For a cross-dataset overlay, such as a homolog search's
`query_transformed_*` layer, the bodyId row uses the source neuron's native
`type` label. Crosswalk fields such as MCNS `flywireType` are used for mapping
and comparison, not as a replacement display label. Thus an MCNS `SMP227`
query whose mapped `flywireType` is `CB1449,CB2843` remains labeled `SMP227`
in the target FAFB scene.

### Freeze view / autoscale

`freeze_view=True` (the default) keeps the viewer's 3D framing constant when
traces are shown or hidden. Plotly autoranges the three scene axes over the
*visible* traces on every redraw, so toggling a native legend row or a
tree-panel eye moved the axes, not the camera. The permanent pages
(`{saveas}.html` and its `{saveas}_simplified.html` copy) therefore pin the
axes to the padded extents of **every** trace — hidden layers included, each
axis padded by 1/32 of its span, which is Plotly's own 3D autorange margin —
so a toggle only changes what is drawn.

- **Freeze/Fit button** in the page's top-left corner, or the **`F`** key,
  switches between the frozen framing and Plotly's autoscaling; double-clicking
  the scene (Plotly's reset gesture) re-applies the freeze.
- The pin is injected page JavaScript only. It exits early under
  `navigator.webdriver`, so PNG / video / individual-profile exports still
  autoscale, and the frozen page's own camera presets keep working.
- Because Plotly's own autorange sees only the visible traces, a scene with a
  far-away hidden layer (a VNC layer inside a brain view) now opens slightly
  wider than before. Set `freeze_view=False` to restore the old framing exactly.

### Per-Neuron Colors via CSV

When using `layer_map_csv`, you can specify per-neuron colors using an optional `color` column:

```csv
layer,id_type_instance,color
PN_layer,1234567890,red
PN_layer,1234567891,#00FF00
KC_layer,KC_gamma_s1,blue
KC_layer,KC_gamma_s2,rgba(255,128,0,0.9)
```

```python
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    layer_map_csv='my_neurons.csv',   # CSV with optional 'color' column
    legend_mode='type',               # Each type gets separate legend entry
)
```

The `color` column accepts any valid color format: named colors, hex codes, RGB tuples, or rgba strings. Colors override the default layer color for specific neurons.

### Color Expansion Methods

When you have more layers than available colors, VisualizeSkeleton automatically expands the color palette using `expand_colors`:

```python
vs = VisualizeSkeleton(
    neuron_layers=['layer1', 'layer2', ..., 'layer20'],  # 20 layers
    neuron_colors=('#1f77b4', '#ff7f0e', '#2ca02c'),     # Only 3 colors
    
    # Color expansion method:
    expand_colors='interpolation',  # Default: create smooth color gradient
    # expand_colors='darken',       # Alternative: recycle with darkening
    # expand_colors='cycle',        # Alternative: simple color cycling
)
```

**Color Expansion Methods:**

| Method            | Behavior                                                             | Brightness Range | Best For                      |
| ----------------- | -------------------------------------------------------------------- | ---------------- | ----------------------------- |
| `'interpolation'` | Creates smooth colormap from base palette, samples new colors evenly | 100% (full)      | Many layers, distinct colors  |
| `'darken'`        | Recycles base colors with progressive darkening                      | 100% → 70%       | Fewer layers, consistent hues |
| `'cycle'`         | Simple repetition of base colors without modification                | 100% (full)      | Color consistency, few cycles |

**Interpolation Mode (Default):**
- Builds a continuous colormap from your base palette
- Samples colors evenly across the spectrum
- Ensures visually distinct colors even with 20+ layers
- Example: 3 base colors → 20 smoothly varying colors

**Darken Mode:**
- Cycles through base palette, darkening by 30% per cycle
- Maintains color family (e.g., all blues, but darker)
- Maximum darkening: 70% brightness (minimum)
- Example with 3 colors for 9 layers:
  - Colors 1-3: 100% brightness (original)
  - Colors 4-6: 85% brightness
  - Colors 7-9: 70% brightness

**Cycle Mode:**
- Simply repeats the base color palette
- No modification to brightness or hue
- Colors repeat exactly: color1, color2, color3, color1, color2, color3...
- Best when you want consistent color families and have relatively few cycles

### Legend Color Swatches

**Automatic Opacity Handling:**
- Legend color swatches always show **full opacity** colors for clarity
- Neurons in the 3D plot use the specified `neuron_alpha` transparency
- This ensures legend remains visible and color-identifiable even with low alpha

Example:
```python
vs = VisualizeSkeleton(
    neuron_layers=['MBON.*', 'DAN.*'],
    neuron_alpha=0.2,  # Very transparent neurons in plot
    # Legend swatches will still show solid, vivid colors
)
```

**Custom Color Palettes:**
```python
# Define your own color palette to avoid auto-expansion
custom_colors = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5'
]

vs = VisualizeSkeleton(
    neuron_layers=['layer1', 'layer2', ..., 'layer10'],
    neuron_colors=custom_colors,  # No expansion needed
)
```

---

## WebDriver Setup & Requirements

### Dependencies

WebDriver export requires:

1. **Python packages:**
   ```bash
   pip install selenium webdriver-manager
   ```

2. **Google Chrome browser** (version 109 or later)
   - The `--headless=new` mode requires Chrome 109+
   - Download: https://www.google.com/chrome/

### Platform-Specific Setup

#### macOS

```bash
# Install via Homebrew (recommended)
brew install --cask google-chrome

# Or download directly from google.com/chrome

# Verify Chrome version (must be 109+)
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --version
```

**ChromeDriver is managed automatically** by `webdriver-manager`. It:
- Detects your Chrome version
- Downloads matching ChromeDriver
- Caches it at `~/.wdm/drivers/chromedriver/`

#### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install google-chrome-stable

# Or download .deb from google.com/chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb

# Verify version
google-chrome --version
```

#### Windows

1. Download Chrome from https://www.google.com/chrome/
2. Install normally
3. ChromeDriver is auto-managed by `webdriver-manager`

**Note:** On Windows, Chrome is typically found at:
- `C:\Program Files\Google\Chrome\Application\chrome.exe`
- `C:\Program Files (x86)\Google\Chrome\Application\chrome.exe`

### Troubleshooting WebDriver

#### Error: "Could not initialize Chrome WebDriver"

**Causes and solutions:**

1. **Chrome not installed:**
   ```
   Install Google Chrome from google.com/chrome
   ```

2. **Chrome version too old:**
   ```
   Update Chrome to version 109 or later
   Current minimum: Chrome 109 (for --headless=new WebGL support)
   ```

3. **ChromeDriver version mismatch:**
   ```bash
   # Clear cached ChromeDriver
   rm -rf ~/.wdm/drivers/chromedriver/
   
   # Re-run export (will download fresh ChromeDriver)
   ```

4. **webdriver-manager not installed:**
   ```bash
   pip install webdriver-manager
   ```

5. **Network error downloading ChromeDriver:**
   ```
   Check internet connection
   Retry the export
   ```

#### Error: "WebGL not supported"

This typically means Chrome is running in legacy headless mode. The code now uses `--headless=new` which supports WebGL on all platforms.

#### Fallback to Kaleido

If WebDriver fails, you can always fall back:
```python
vs = VisualizeSkeleton(..., export_method='kaleido')
```

### Chrome Version Check

To verify your Chrome version meets requirements:

```bash
# macOS
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --version

# Linux
google-chrome --version

# Windows (PowerShell)
(Get-Item "C:\Program Files\Google\Chrome\Application\chrome.exe").VersionInfo.FileVersion
```

**Minimum required: Chrome 109** (released January 2023)

---

### Prerequisites

**Required packages**:
```bash
pip install navis
pip install flybrains  # For brain templates and transforms
pip install plotly
pip install neuprint-python
```

**For WebDriver export:**
```bash
pip install selenium webdriver-manager
# + Google Chrome browser (version 109+)
```

**Data requirements**:
- Valid NeuPrint authentication token ([get yours here](https://neuprint.janelia.org/account))
- Neuron bodyIds or types from chosen dataset
- Internet connection (first time, to fetch skeleton data)
- **Optional**: ~10GB disk space for whole-brain transforms (one-time download, ~1-2 hours)

### Multi-Dataset Support (NEW)

**Supported Datasets:**
- **hemibrain:v1.2.1** / **optic-lobe:v1.1**: Adult brain
- **manc:v1.2.3**: Male VNC (ventral nerve cord)
- **male-cns:v0.9**: Full CNS (brain + VNC combined)
- **flywire_FAFB_v783**: FAFB local release (FlyWire/Codex or CAVE-backed)
- **banc_v626** / **banc_v888**: Standalone BANC brain + nerve cord releases (public bucket, no token; skeletons fetched on demand through the unified L2 → full → v626-pcg chain)

**Brain Mesh Options:**

```python
# Option 1: No brain mesh (fastest, clearest for single neurons)
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['KC.*'],
    brain_mesh='none'  # No background mesh
)

# Option 2: Template mesh (native EM resolution, 0.5-2s load time)
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['KC.*'],
    brain_mesh='native'  # JRCFIB2018F for hemibrain, MANC for manc
)

# Cross-template scenes: render everything (neurons + outline) in the
# selected template's coordinates:
vs = VisualizeSkeleton(
    dataset='male-cns:v1.0',
    neuron_layers=['l-LNv'],
    brain_mesh='FAFB',  # whole scene moved to FLYWIRE/FAFB coordinates
)
# 'BANC' and 'male-cns' work the same way from any dataset whose template
# space bridges to theirs (FLYWIRE / JRCFIB2022M / BANC) within two hops;
# unreachable selections keep the native scene with a warning.
```

**Dataset-Specific Templates (`brain_mesh='native'`):**

| Dataset              | Native outline         | Transform Required |
| -------------------- | ---------------------- | ------------------ |
| **hemibrain:v1.2.1** | JRCFIB2018F (EM)       | ❌ No (affine)      |
| **optic-lobe:v1.1**  | JRCFIB2022M (EM)       | ❌ No (affine)      |
| **manc:v1.2.3**      | MANC (native VNC)      | ❌ No               |
| **male-cns:v1.0**    | JRCFIB2022M (brain)    | ❌ No (affine)      |
| **flywire_FAFB**     | FLYWIRE (native)       | ❌ No               |
| **banc_v888**        | BANC (brain)           | ❌ No               |

**Retired mode:** the former `brain_mesh='whole'` (which moved the entire
scene into JRC2018F and required a ~10GB H5 transform download) was
replaced by the explicit `FAFB`/`BANC`/`male-cns` outline selections; legacy
'whole'/'template' values in saved settings are normalized automatically.

```
⚠ Brain transforms not found for hemibrain:v1.2.1
  Target template: JRC2018F (whole brain)
  Transform size: ~10GB total (8 files, one-time download)
  Download time: ~1-2 hours
  Storage location: ~/flybrain-data

Download transforms? [y/N]:
```

**Important:** Only **JRC2018F_JRCFIB2018F.h5 (~1.3 GB)** is needed for hemibrain/optic-lobe whole-brain visualization. However, the flybrains package downloads ALL JRC transforms (8 files, ~10GB total) as a bundle - selective download is not supported.

**Actual transform path used:**
```
JRCFIB2018Fraw → JRCFIB2018F → JRCFIB2018Fum → JRC2018F
```

The other 7 files (~8.7GB) enable cross-dataset registration and support for other brain templates (FAFB, JFRC2013, etc.).

Type `y` to download. Transforms persist and are shared across all NeuPrint projects.

## Key Features

### 1. Neuron Skeleton Rendering

#### Morphology Display
- **Skeleton structure**: Node-and-edge representation
- **Compartments**: Soma, dendrites, axon (if labeled)
- **Spatial accuracy**: Real coordinates from EM reconstruction
- **Color coding**: By neuron, by type, or by compartment

#### Multiple Neurons
```python
# Visualize multiple neurons together
plot_navis_3d(
    bodyIds=[
        123456789,  # Neuron 1
        987654321,  # Neuron 2
        111222333   # Neuron 3
    ],
    color_by='neuron'  # Color each neuron differently
)
```

### 2. Brain Region (ROI) Meshes

Display anatomical context with 3D brain region meshes:

#### ROI Selection
```python
# Show specific ROIs
plot_navis_3d(
    bodyIds=[123456789],
    show_rois=True,
    roi_list=['MB_CA', 'MB_PED', 'SLP_R'],  # Mushroom body regions
    roi_opacity=0.2  # Semi-transparent
)
```

#### ROI Mesh Colors and Transparency

**Using VisualizeSkeleton:**
```python
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['KC.*'],
    mesh_roi=['MB(R)', 'CA(R)', 'PED(R)'],
    
    # Color options for ROI meshes
    mesh_color='gray',                  # Named color (uses mesh_alpha)
    mesh_color=(100, 100, 100),         # RGB tuple (uses mesh_alpha)
    mesh_color='rgba(100,100,100,0.2)', # RGBA with explicit alpha (overrides mesh_alpha)
    mesh_color=['red', 'green', 'blue'],# Per-ROI colors
    
    # Transparency for ROI meshes
    mesh_alpha=0.1,                     # 0.0=transparent, 1.0=opaque (default: 0.1)
)
```

**Note:** `mesh_color` and `mesh_alpha` apply only to ROI meshes (specified in `mesh_roi`).
Brain envelope and VNC outline meshes use separate color settings:
- `brain_mesh_color`: Color for the brain mesh (default: 'auto', #74A8D6 at 4% opacity)
- `vnc_mesh_color`: Color for the VNC mesh (default: 'auto', #74A8D6 at 4% opacity)

**Alpha Override Behavior:**
- If `mesh_color` contains explicit alpha (e.g., RGBA tuple, `rgba()` string, or hex with alpha), those alpha values override `mesh_alpha`
- Per-ROI alpha: Use `mesh_color` with embedded alpha for different transparencies per ROI:
  ```python
  mesh_color=['rgba(255,0,0,0.1)', 'rgba(0,255,0,0.2)', 'rgba(0,0,255,0.05)']
  ```

#### Color Formats and Mixed Alpha Palettes

The `neuron_colors`, `synapse_colors`, and `mesh_color` parameters accept the
same color formats, and a single list may mix them:

- Named colors such as `'red'`, `'royalblue'`, and `'transparent'`
- Hex colors: `#RGB`, `#RGBA`, `#RRGGBB`, and `#RRGGBBAA`
- RGB/RGBA tuples or lists using 0–255 channels, or normalized 0–1 floats
- CSS `rgb()`/`rgba()` (comma or space/slash syntax, including percentages)
- CSS `hsl()`/`hsla()`; alpha may be written as 0–1, a percentage, or 0–255

An explicit alpha belongs only to that color entry and overrides the matching
global opacity (`neuron_alpha`, `synapse_alpha`, or `mesh_alpha`). A color
without alpha inherits the global opacity, so mixed input is valid:

```python
neuron_colors=['#ff0000', 'rgba(0, 255, 0, 0.25)', (0.0, 0.0, 1.0)]
synapse_colors=['rgb(255 165 0 / 70%)', (0.2, 0.4, 1.0, 0.4)]
mesh_color=['#888888', 'hsl(200 70% 50% / 20%)']
```

In the UI, use **Color format** for free-form values and enable **Alpha
override** when the numeric alpha control should be embedded in the selected
entry. Leaving that checkbox off keeps a color alpha-less so it follows the
global opacity.

**Automatic ROI Expansion:**
When you specify a base ROI name like `'AME'`, it's automatically expanded to bilateral variants `['AME(L)', 'AME(R)']` if available:
- Colors are expanded to match: `mesh_color='red'` → both sides get red
- **Independent legends**: Every resolved mesh has its own legend entry (for
  example, `brain region [AME(L)]` and `brain region [AME(R)]`)
- Toggle either side independently without hiding the other ROI mesh

```python
# These are equivalent:
mesh_roi=['AME'],              # Auto-expands to ['AME(L)', 'AME(R)']
mesh_roi=['AME(L)', 'AME(R)'], # Explicit bilateral specification
```

**Case-Sensitive ROI Names (macOS/Windows):**
Some ROI names differ only in case (e.g., `'AL'` = Antennal Lobe, `'aL'` = alpha Lobe). On case-insensitive filesystems, these are handled automatically with encoded filenames to prevent collisions.

**Special Keywords and Regex Patterns:**
Use keywords or regex patterns for convenient ROI selection:

```python
# Single ROI (string input automatically converted to list)
mesh_roi='EB'

# 'primary': Load all primary brain regions (major ROIs)
mesh_roi=['primary']

# 'all': Load ALL available ROIs for the dataset
mesh_roi=['all']

# Regex patterns: Match ROIs by pattern
mesh_roi=['ME.*']          # All ROIs starting with 'ME'
mesh_roi=['.*\\(R\\)']     # All right-hemisphere ROIs
mesh_roi=['MB.*', 'AL.*']  # Multiple patterns

# Mixed: Combine literal names, keywords, and patterns
mesh_roi=['EB', 'FB', 'ME.*']  # Central complex + medulla regions

# Nested lists for color grouping (ROIs share a color but remain separate traces)
mesh_roi=['AME', ['aL', 'bL', 'gL', "a'L", "b'L"], 'EB']
mesh_color=['red', 'green', 'blue']  # 'green' applies to all nested ROIs
```

| Keyword/Pattern | Description                                     | Example Output                            |
| --------------- | ----------------------------------------------- | ----------------------------------------- |
| `'primary'`     | Major brain regions (MB, AL, optic lobes, etc.) | ~50-100 ROIs depending on dataset         |
| `'all'`         | Every available ROI in the dataset              | All ROIs from `available_rois.json`       |
| `'ME.*'`        | Regex: ROIs starting with 'ME'                  | `ME(L)`, `ME(R)`, `ME_glomerulus(L)`, ... |
| `'.*\\(L\\)'`   | Regex: All left-hemisphere ROIs                 | `AL(L)`, `MB(L)`, `LH(L)`, ...            |

**Note:** Regex patterns use Python's `re` module. The pattern is automatically anchored (`^...$`).

**FAFB/FlyWire Note:**
FAFB does not have native ROI meshes. When visualizing FAFB data,
ROI meshes from male-cns are automatically transformed to FAFB coordinates.
This allows ROI context visualization but may have minor alignment differences.

#### Finding Available ROIs

- **Programmatically**: Use `vs.list_available_rois()` method
- **Cached list**: Check `cache/{dataset}/available_rois.json`
  - hemibrain: `cache/hemibrain_v1_2_1/available_rois.json`
  - male-cns: `cache/male-cns_v0_9/available_rois.json`
  - FAFB: Uses `cache/male-cns_v0_9/available_rois.json` (transformed)

#### Available ROIs
Major brain regions include:
- **Mushroom Body**: MB_CA (calyx), MB_PED (pedunculus), MB_VL (vertical lobe)
- **Central Complex**: FB (fan-shaped body), EB (ellipsoid body), PB (protocerebral bridge)
- **Optic Lobes**: ME (medulla), LO (lobula), LOP (lobula plate)
- **Antennal Lobe**: AL_R, AL_L
- **Lateral Horn**: LH_R, LH_L

**Complete list**: Available in `navis_roi_meshes_json/primary_rois/` directory

### 3. Synapse Visualization

Show synaptic connections in 3D space:

#### Presynaptic Sites (Outputs)
- **Color**: Red (default)
- **Size**: Proportional to partner count
- **Location**: Exact spatial coordinates

#### Postsynaptic Sites (Inputs)
- **Color**: Blue (default)
- **Size**: Proportional to partner count
- **Location**: Exact spatial coordinates

#### Connection Lines
```python
# Show connections between neurons
plot_navis_3d(
    bodyIds=[source_id, target_id],
    show_connectors=True,
    show_connections=True,  # Draw lines between connected synapses
    connection_color='green',
    connection_width=2
)
```

### 4. Color Schemes

#### By Neuron
Each neuron gets a unique color:
```python
color_by='neuron'  # Default
```

#### By Type
Neurons of same type share color:
```python
plot_navis_3d(
    bodyIds=neuron_list,
    color_by='type',
    neuron_types=['KC', 'MBON', 'DAN']  # Provide type info
)
```

#### By Compartment
Different colors for soma, dendrites, axon:
```python
color_by='compartment'
```

#### Custom Colors
Specify exact colors:
```python
plot_navis_3d(
    bodyIds=[id1, id2, id3],
    custom_colors=['#FF0000', '#00FF00', '#0000FF']  # Red, Green, Blue
)
```

### 5. Interactive Controls

#### Rotation
- **Click and drag**: Rotate view in any direction
- **Shift + drag**: Pan view
- **Scroll**: Zoom in/out

#### Selection
- **Click neuron**: Highlight and show info
- **Click synapse**: Show partner details
- **Click ROI**: Display region name and volume

#### Visibility Toggle
- **Neurons**: Toggle individual neurons on/off
- **ROIs**: Toggle brain regions
- **Synapses**: Show/hide connectors
- **Connections**: Show/hide connection lines

### 6. Camera Views

Preset viewing angles:

#### Anterior View
Front view of the brain:
```python
camera_view='anterior'
```

#### Posterior View
Back view:
```python
camera_view='posterior'
```

#### Dorsal View
Top-down view:
```python
camera_view='dorsal'
```

#### Ventral View
Bottom-up view:
```python
camera_view='ventral'
```

#### Custom View
Define exact camera position:
```python
camera_position={
    'eye': {'x': 1000, 'y': 1000, 'z': 1000},
    'center': {'x': 0, 'y': 0, 'z': 0},
    'up': {'x': 0, 'y': 0, 'z': 1}
}
```

### 7. Export Options

#### HTML Interactive
Default output format:
```python
output_format='html'  # Interactive 3D in browser
```

#### Static Image
```python
# Export as PNG
plot_navis_3d(
    bodyIds=[123456789],
    output_format='png',
    image_width=1920,
    image_height=1080,
    camera_view='anterior'
)
```

#### Animation
Create rotating view:
```python
# Generate rotation animation
plot_navis_3d(
    bodyIds=[123456789],
    output_format='gif',
    rotation_frames=360,  # Full rotation
    rotation_axis='z'     # Rotate around z-axis
)
```

#### Individual Profiles

`vs.plot_individuals(...)` renders every group on its own (all other
morphology traces hidden) into `individual_profiles/`, plus the PDF/PPTX grid
in the run folder. `granularity` picks how finely the scene is cut up:

| Granularity | One profile per                              | Notes                                                    |
| ----------- | -------------------------------------------- | -------------------------------------------------------- |
| `'legend'`  | Native legend entry (`legendgroup`)          | Default; the historical behaviour                        |
| `'layer'`   | Visualization layer / custom group           | Coarsest level; one row per `neuron_layers` entry        |
| `'type'`    | Neuron type                                  | Matches the tree legend's type level                     |
| `'body'`    | bodyId leaf                                  | The same rows the `legend_mode='tree'` panel exposes     |

At the `'layer'` level, two layers whose smart names collide are split by
their stamped `layer_index` into `{name}__L{index}` profiles rather than
merging; traces without the stamp (pre-stamp pages) stay under the plain
name.

The levels read the identity the renderer stamps on every trace
(`meta.drocatTrace`), so they work in any `legend_mode`. Companion soma meshes
and pre/post connector sites follow their owner's profile; brain/VNC/ROI
meshes, synapse markers and the legend-only swatches stay background in every
profile. The accepted values are the module constant `PROFILE_GRANULARITIES`
(`('legend', 'layer', 'type', 'body')`); anything else raises, and a trace the
classifier cannot place is reported in the run log instead of disappearing.

```python
vs.plot_individuals(output_format=['png', 'html'], granularity='body')
```

`views` takes the canonical camera names (`front` / `back` / `top` / `bottom` /
`left` / `right`) — the same list both UI dropdowns offer — plus `all` for every
view the scene's own camera table offers. A view name the camera table does not
carry is reported instead of being rendered from the page's current framing.

Every profile is one full-scene render, so the job is capped at
`MAX_INDIVIDUAL_PROFILES` (300) renders — profile groups × views. A scene over
budget renders the groups that fit and logs which ones were skipped rather than
grinding away for hours; to get the rest, separate the plots into smaller runs
(fewer neurons or layers per render), drop to a coarser `granularity`, or
request fewer views. The same cap covers `export_individuals_from_html()`.

The same page can be re-profiled later at another level, without re-running the
fetches: `export_individuals_from_html(page, granularity='body')` reads the
stored page back, classifies its traces and renders the PNGs through one
WebDriver session — see [Re-export from the UI](#re-export-from-the-ui).

### 8. Performance Options

#### Level of Detail
Control skeleton complexity:

```python
# High detail (slow)
skeleton_detail='high'  # All nodes and edges

# Medium detail (balanced)
skeleton_detail='medium'  # Simplified skeleton

# Low detail (fast)
skeleton_detail='low'  # Major branches only
```

#### ROI Mesh Quality
```python
# Lower quality for faster rendering
roi_mesh_quality='low'   # Fewer triangles
roi_mesh_quality='medium'
roi_mesh_quality='high'  # Maximum detail
```

### 9. Measurement Tools

#### Distance Measurement
Measure spatial distances:
```python
# Enable measurement mode
enable_measurements=True
```

**Usage**: Click two points to measure distance in micrometers

#### Volume Calculation
Calculate neuron volume:
```python
calculate_volumes=True  # Adds volume info to neuron labels
```

#### Path Length
Calculate cable length:
```python
calculate_path_lengths=True  # Total skeleton length
```

### 10. Advanced Visualization

#### Density Maps
Show synaptic density in 3D:
```python
plot_density_map=True
density_radius=10  # Micrometers
density_type='presynaptic'  # or 'postsynaptic'
```

#### Connectivity Matrix Overlay
Show connection strength as colors:
```python
plot_navis_3d(
    bodyIds=neuron_list,
    connectivity_matrix=conn_matrix,  # Pandas DataFrame
    color_by_connectivity=True
)
```

## Advanced Usage

### Pathway Visualization

Visualize complete pathways in 3D:

```python
# Get pathway neurons from path analysis
pathway_df = pd.read_excel('path_results.xlsx')
all_neurons = list(set(pathway_df['source'].tolist() + 
                      pathway_df['target'].tolist()))

# Extract bodyIds (if type names)
from neuprint import fetch_neurons
bodyIds = fetch_neurons(all_neurons, dataset='hemibrain:v1.2.1')

# Visualize pathway in 3D
plot_navis_3d(
    bodyIds=bodyIds,
    show_rois=True,
    show_connectors=True,
    color_by='type',
    roi_list=['MB_CA', 'MB_PED', 'MB_VL']  # Relevant ROIs
)
```

### Comparison Views

Compare neuron morphologies:

```python
# Create side-by-side views
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Two neurons in separate panels
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
)

# Add neuron 1 to left panel
# Add neuron 2 to right panel
# (detailed implementation in examples/)
```

### Synapse-Specific Filtering

Show only specific synapse types:

```python
# Only show strong connections (>10 synapses)
plot_navis_3d(
    bodyIds=[source_id, target_id],
    show_connectors=True,
    synapse_threshold=10,  # Minimum synapse count
    synapse_type='presynaptic'  # or 'postsynaptic'
)
```

### Custom Mesh Import

Use custom brain region meshes:

```python
# Load custom mesh
custom_mesh = load_mesh('custom_roi.obj')

plot_navis_3d(
    bodyIds=[123456789],
    custom_meshes=[custom_mesh],
    mesh_colors=['rgba(255,0,0,0.2)']
)
```

### Time-Series Animation

Show neuron growth or changes:

```python
# Multiple time points
neuron_versions = [
    bodyIds_t0,  # Time point 0
    bodyIds_t1,  # Time point 1
    bodyIds_t2   # Time point 2
]

# Create animation
create_temporal_animation(
    neuron_versions,
    frame_duration=1000,  # ms per frame
    output='neuron_growth.html'
)
```

## Coordinate Systems

### Hemibrain Coordinates
- **X-axis**: Anterior (front) to Posterior (back)
- **Y-axis**: Left to Right
- **Z-axis**: Ventral (bottom) to Dorsal (top)
- **Units**: Micrometers (μm)
- **Origin**: Defined by FlyEM/DVID system

### Transformations
Convert between coordinate systems:
```python
from navis import xform_brain

# Transform to JRC2018 template
neurons_transformed = xform_brain(
    neurons,
    source='FAFB14',
    target='JRC2018F'
)
```

## Performance Tips

### Large Neuron Sets
- **Limit to <20 neurons** for interactive performance
- **Use lower detail** for faster rendering
- **Hide connectors** if not needed
- **Reduce ROI mesh quality**

### High-Resolution Exports
- **Increase image dimensions** (2K, 4K)
- **Use static format** (PNG) instead of HTML
- **Render without connectors** for cleaner images
- **Set specific camera view** to avoid re-positioning

### Memory Management
- **Process neurons in batches**
- **Clear cache** between visualizations
- **Use lazy loading** for large datasets

## Troubleshooting

**Skeletons don't appear**: Check bodyIds are valid and dataset is correct

**ROIs not loading**: Ensure `navis-flybrains` is installed and ROI names are correct

**Slow performance**: Reduce neuron count, lower skeleton detail, hide connectors

**Export fails**: Check output directory permissions, try different format

**Colors wrong**: Verify color specification format (hex or rgba)

**Memory error**: Reduce number of neurons, disable connectors, lower mesh quality

## Integration with Other Visualizations

### 3D + Network
- **Network**: Shows connectivity topology
- **3D**: Shows spatial arrangement
- **Together**: Correlate topology with anatomy

### 3D + Heatmap
- **Heatmap**: Quantifies connection strengths
- **3D**: Visualizes spatial positions
- **Workflow**: Identify strong connections in heatmap → visualize in 3D

### 3D + Sankey
- **Sankey**: Shows flow and magnitude
- **3D**: Shows physical pathways
- **Use case**: Understand how flow maps to brain regions

## Example Workflows

### 1. Mushroom Body Pathway
```python
# Kenyon Cells → MBONs
kc_ids = [...]  # KC bodyIds
mbon_ids = [...]  # MBON bodyIds

plot_navis_3d(
    bodyIds=kc_ids + mbon_ids,
    show_rois=True,
    roi_list=['MB_CA', 'MB_PED', 'MB_VL', 'MB_ML'],
    show_connectors=True,
    color_by='type'
)
```

### 2. Visual Pathway
```python
# Optic lobe to central brain
plot_navis_3d(
    bodyIds=visual_neurons,
    show_rois=True,
    roi_list=['ME', 'LO', 'LOP', 'AOTU', 'BU'],
    camera_view='lateral'
)
```

### 3. Comparison Study
```python
# Compare two neuron types
plot_navis_3d(
    bodyIds=type1_ids,
    output_file='type1_3d.html'
)

plot_navis_3d(
    bodyIds=type2_ids,
    output_file='type2_3d.html'
)
```

## Related Documentation

- [Advanced Visualization Features](./AdvancedVisualizationFeatures.md)
- [NAVIS Integration](https://navis.readthedocs.io/)
- [FlyBrains Resources](https://github.com/navis-org/navis-flybrains)
- [NeuPrint Documentation](https://neuprint.janelia.org/help)

## External Resources

- **NAVIS**: https://navis.readthedocs.io/
- **FlyWire**: https://flywire.ai/
- **Virtual Fly Brain**: https://www.virtualflybrain.org/
- **FlyEM**: https://www.janelia.org/project-team/flyem
