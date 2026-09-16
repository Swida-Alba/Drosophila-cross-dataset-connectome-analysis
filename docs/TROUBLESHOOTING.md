# Troubleshooting Guide

Comprehensive troubleshooting guide for the Drosophila Connectome Analysis Toolkit (DROCAT).

---

## Table of Contents

- [Installation Issues](#installation-issues)
  - [Package Dependency Errors](#package-dependency-errors)
  - [Platform-Specific Issues](#platform-specific-issues)
- [Server & Connection Issues](#server--connection-issues)
- [Authentication & API Issues](#authentication--api-issues)
- [FlyWire Data Download Issues](#flywire-data-download-issues)
- [3D Visualization Issues](#3d-visualization-issues)
  - [WebDriver Export Errors](#webdriver-export-errors)
  - [Kaleido Export Errors](#kaleido-export-errors)
  - [Brain Mesh & Transform Issues](#brain-mesh--transform-issues)
  - [Memory & Performance Issues](#memory--performance-issues)
- [Network Visualization Issues](#network-visualization-issues)
- [Data Loading Issues](#data-loading-issues)
- [Cross-Platform Compatibility](#cross-platform-compatibility)
- [Advanced Settings](#advanced-settings)
- [Reporting Issues](#reporting-issues)

---

## Installation Issues

### Package Dependency Errors

#### numpy/pandas Binary Compatibility Error

**Symptom:**
```
numpy.dtype size changed, may indicate binary incompatibility
```

**Solution:**
```bash
# Ensure numpy <2.0.0 is installed
pip install 'numpy<2.0.0' --force-reinstall
```

This constraint is already in `requirements.txt` and `pyproject.toml`.

---

#### Pandas Version Conflict

**Symptom:**
```
ImportError: cannot import name 'X' from 'pandas'
```

**Solution:**
```bash
pip install "pandas<2"
```

This project requires pandas < 2.0.0 for compatibility with NeuPrint and visualization libraries.

---

#### Legacy memray / Pydantic Conflict

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement memray
```

**Cause:** An older DROCAT environment still contains the upstream `neuronbridge-python` distribution. It pulls Memray/Ray and constrains Pydantic to a version that conflicts with the current UI.

**Solution:** Rerun the one-click installer. For a manual repair:
```bash
python -m pip uninstall -y neuronbridge-python
python -m pip install -r requirements-windows.txt -r ui/requirements.txt
python -m pip check
```

DROCAT now bundles its NeuronBridge API client; do not reinstall the upstream package.

---

#### NeuronBridge Client Initialization Error

**Symptom:**
```
Failed to initialize NeuronBridge client
```

**Cause:** Network connectivity issues, proxy/VPN interference, or NeuronBridge service unavailable.

**Solutions:**

1. **Check network connection** - Ensure you have internet access
2. **Disable VPN/proxy** - Try without VPN if you're using one
3. **Retry** - The service may be temporarily unavailable
4. **Check NeuronBridge status** - Visit https://neuronbridge.janelia.org/ to verify service is online

---

#### General Import Errors

**Symptom:**
```
ModuleNotFoundError: No module named 'X'
```

**Solution:**
```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

---

#### "Could Not Find a Version That Satisfies the Requirement" ("from versions: none")

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement xlsxwriter<4,>=3.1.0 (from versions: none)
ERROR: No matching distribution found for xlsxwriter<4,>=3.1.0
ERROR conda.cli.main_run:execute(49): `conda run python -m pip install ...` failed.
```

**Cause:** Transient PyPI connectivity degradation (slow downloads, repeatedly
interrupted downloads, incomplete or empty index-metadata responses). "from
versions: none" is what pip reports when its index query for a package returns
an empty version list - it is not a packaging problem, and a different package
may fail on each run until the network recovers.

**Solution:** Simply re-run the one-click installer. It is idempotent and
reuses the created environment plus the project-local pip cache
(`<repo>/cache/pip`), so a re-run is cheap. Manual repair with pip's own
resilience flags:
```bash
python -m pip install --retries 5 --timeout 60 -r requirements-windows.txt -r ui/requirements.txt
```

---

### Platform-Specific Issues

#### PyQt5 Installation Issues

**macOS:**
```bash
pip install --upgrade pip
pip install PyQt5
```

**Linux:**
```bash
sudo apt-get install python3-pyqt5
# or
pip install PyQt5
```

**Windows:**
```bash
pip install PyQt5
```

---

#### Leftover Conda Env Directory After `conda env remove` (Windows)

**Symptom:**
```
DirectoryNotACondaEnvironmentError
```

**Cause:** After `conda env remove -n drocat-4.5.0` (cleanup), the environment
directory (`C:\Users\<user>\anaconda3\envs\drocat-4.5.0`) can remain on disk
as a stub containing only a `Scripts` folder. A second `conda env remove` then
fails with `DirectoryNotACondaEnvironmentError`.

**Solution:** Delete the leftover directory manually, then re-create the env:
```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\anaconda3\envs\drocat-4.5.0"
```

---

#### UnicodeEncodeError / Garbled Symbols on Non-UTF-8 Windows Consoles

**Symptom:**
```
UnicodeEncodeError: 'gbk' codec can't encode character '\u26a0' in position 0
```
raised from a `print` while running the library or a script directly in a
terminal whose code page is not UTF-8 (e.g. zh-CN GBK, cp1252) — typically
with output redirected to a file or pipe. The one-click UI launcher is
unaffected: `windows_DROCAT.bat` sets `PYTHONIOENCODING=utf-8` and the
library reconfigures its own streams at import.

**Solution:** Update to a revision that includes the console-encoding guard
(`src/utils/console_encoding.py`). For older revisions, set the variable
yourself before running Python:
```powershell
$env:PYTHONIOENCODING = "utf-8"
```

---

#### "Cache ... is incomplete" Refusal in Cache-Only Mode

**Symptom:**
```
Cache-only mode requested but the local connection cache is incomplete ...
N of M neurons flagged complete in the neuron index have fewer cached
connection rows than recorded
```
or
```
Cache-only run needs N neuron(s) that are not in the local cache ...
```

**Cause:** The local cache does not fully cover the query. Earlier versions
continued anyway and returned partial results presented as a complete
analysis (in the 2026-09-12 Windows test, a truncated cache returned 9,469
paths where the full cache returns 49,871).

**Solution:** Run once with Cache-Only disabled (server reachable, token
configured) so the cache is completed, then re-run offline. If you knowingly
accept partial results, pass `allow_incomplete_cache=True` (library/CLI
parameter): the run then warns and stamps its output folders with an
`INCOMPLETE_CACHE.txt` marker.

---

#### Missing tkinter (Linux)

**Symptom:**
```
ModuleNotFoundError: No module named 'tkinter'
```

**Solution:**
```bash
sudo apt-get install python3-tk
```

---

## Server & Connection Issues

### HTTP Errors (500, 502, 503, 504)

**Symptom:**
```
HTTPError: 500 Server Error
HTTPError: 502 Bad Gateway
HTTPError: 503 Service Unavailable
HTTPError: 504 Gateway Timeout
```

**Cause:** A NeuPrint or FAFB/CAVE server is temporarily unavailable or overloaded.
The standalone BANC path uses the public release bucket and is independent of
FlyWire/CAVE; for BANC, check the release-bucket connection instead.

**Solutions:**

1. **Wait and retry:** Server issues are usually temporary. Wait a few minutes and re-run your code.

2. **Check server status:**
   - NeuPrint: https://neuprint.janelia.org/
   - FlyWire: https://flywire.ai/

3. **Reduce query size:** Large queries may timeout. Try smaller batches:
   ```python
   # Instead of querying all at once
   for batch in batches:
       results = fetch_neurons(batch)
   ```

---

### Connection Timeout

**Symptom:**
```
ConnectionError: Connection timed out
TimeoutError: Request timed out
```

**Solutions:**

1. **Check internet connection**

2. **Retry the operation:** Network glitches are common. Simply re-run the code.

3. **Use local caching:** Once data is cached locally, you won't need network access:
   ```python
   vs = VisualizeSkeleton(
       ...,
       cache_neurons=True,
       cache_synapses=True
   )
   ```

---

### API Rate Limiting

**Symptom:**
```
HTTPError: 429 Too Many Requests
```

**Solution:** Wait a few minutes before retrying. Consider using local caching to reduce API calls.

---

## Authentication & API Issues

### NeuPrint Token Issues

**Symptom:**
```
AuthError: Invalid token
```

**Solutions:**

1. **Put the token in config_local.json** in project root (recommended, gitignored):
   ```json
   {
     "tokens": {
       "neuprint": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
       "cave": ""
     },
     "envs": {
       "4.5.0": ""
     }
   }
   ```
   (or set it in the UI Settings tab, which writes the same file)

2. **Or set environment variable:**
   ```bash
   export NEUPRINT_TOKEN='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'
   ```

3. **Get a new token** from https://neuprint.janelia.org/account

**Note:** For backwards compatibility, `NEUPRINT_APPLICATION_CREDENTIALS` is also supported.

---

### FAFB CAVE Authentication

**Symptom:**
```
CAVEclient authentication failed
```

**Solution:**
```bash
# Generate CAVE secret
python -c "from caveclient import CAVEclient; CAVEclient(datastack_name='flywire_fafb_production')"
```

Follow the browser prompt to authenticate.

### FAFB Visualization Re-downloads the Same Neurons from CAVE Every Run

**Symptom:** Every FAFB visualization run logs
`Skeletonizing N FAFB skeleton(s) from CAVE meshes...` for the same bodies
(usually extrusion-flagged neurons such as l-LNv), adding minutes per run.

**Cause:** CAVE-skeletonized replacement trees are cached in the dedicated
`cache/{dataset}/skeletons/cave_skeletons/` store. If that directory was
deleted, or the bodies were previously only "repaired" by the removed
prepared-mesh cache (versions before the FAFB tree unification), the store is
empty and the resolver fetches online again.

**Solution:** none needed — the fetch is a one-time ~5-20s cost per neuron
and the replacement store makes every later run fully offline. To pre-warm
specific neurons without rendering:

```python
from coana import VisualizeSkeleton
VisualizeSkeleton.fix_fafb_extrusions([720575940624086675])
```

If runs still re-fetch with the store populated, check the repair status
ledger `cache/{dataset}/extrusion_check_results.parquet`: a body whose
`repair_status` is `api_failed`/`local_fallback` is deliberately retried
until a CAVE fetch succeeds.

---

## FAFB and BANC Release Preparation Issues

### FAFB Dataset Setup

FlyWire FAFB data requires manual download from the Codex website.

**Symptom:**
```
❌ Missing required file: classification.csv.gz
CRITICAL ERROR: FAFB data preparation failed.
```

**Solution:**

1. **Download required files** from: https://codex.flywire.ai/api/download?dataset=fafb

2. **Save files to:** `datasets/flywire_FAFB_v783/downloads/`

3. **Required files:**
   | File                                        | Description           | Required             |
   | ------------------------------------------- | --------------------- | -------------------- |
   | `classification.csv.gz`                     | Neuron Classification | ✅ Yes                |
   | `connections_princeton_no_threshold.csv.gz` | Connectivity Data     | ✅ Yes                |
   | `names.csv.gz`                              | Neuron Names          | ✅ Yes                |
   | `coordinates.csv.gz`                        | Soma Coordinates      | ✅ Yes                |
   | `neurons.csv.gz`                            | Neurotransmitters     | ✅ Yes                |
   | `cell_stats.csv.gz`                         | Cell Statistics       | ✅ Yes                |
   | `consolidated_cell_types.csv.gz`            | Cell Types            | ✅ Yes                |
   | `fafb_v783_princeton_synapse_table.csv.gz`  | Synapse Coordinates   | For 3D visualization |
   | `sk_lod1_783_healed.zip`                    | Skeletons             | For 3D visualization |

4. **Run the converter:**
   ```bash
   python src/FAFB_file_converter.py
   ```

---

### BANC Dataset Setup

**Symptom:**
```
❌ Missing required file: neurons.csv.gz
❌ Missing required file: connections_princeton.csv.gz
```

**Solution:**

1. **Select** `banc_v626` or `banc_v888`. BANC metadata and connections are
   fetched from the public release bucket; no Codex login or CAVE token is
   needed.

2. **If automatic preparation is unavailable**, place the release files in
   `datasets/banc_v626/downloads/` (or the selected version's folder).

3. **Required files for the local converter:**
   | File                           | Description       | Required |
   | ------------------------------ | ----------------- | -------- |
   | `neurons.csv.gz`               | Neuron Data       | ✅ Yes    |
   | `connections_princeton.csv.gz` | Connectivity Data | ✅ Yes    |

4. **Run the converter when using local input files:**
   ```bash
   python src/BANC_file_converter.py
   ```

---

### Skeleton File Not Found

**Symptom:**
```
Warning: FlyWire skeleton zip not found
Visualization might fail or be incomplete.
```

**Solution:**
1. Download `sk_lod1_783_healed.zip` from the Codex website
2. Place in `datasets/flywire_FAFB_v783/` directory
3. The file will be automatically detected on next run

---

### NBLAST Search Shows Fewer Types or Smaller n_bodyids Than Expected

**Symptom:** With `method=nblast`, some candidate types are missing from
`type_summary.csv` (present in an earlier vector run), and `n_bodyids` is
smaller than the type's scored member count. The log contains
`NBLAST: N contralateral pair rows excluded from type means`.

**Explanation:** This is intentional. NBLAST matches arbors in absolute
coordinates, so contralateral (mirror) same-type pairs score at chance —
benchmark median −0.03 vs 0.89 ipsilateral. Type means aggregate
ipsilateral pairs only; types whose only NBLAST evidence is contralateral
are omitted from the ranking. Every pair row (including contralateral
ones) stays in `results.csv` for inspection.

**Solution:** None needed. To rank types using mirror-twin evidence as
well, run the same search with `method=vector_v2` — it lateral-normalizes
at vectorization and aggregates both sides.

---

## 3D Visualization Issues

### WebDriver Export Errors

#### "Could not initialize Chrome WebDriver"

**Causes and solutions:**

1. **Chrome not installed:**
   - Install Google Chrome from https://www.google.com/chrome/

2. **Chrome version too old:**
   - Update Chrome to version 109 or later
   - Minimum: Chrome 109 (for `--headless=new` WebGL support)

3. **ChromeDriver version mismatch:**
   ```bash
   # Clear cached ChromeDriver
   rm -rf ~/.wdm/drivers/chromedriver/
   
   # Re-run export (will download fresh ChromeDriver)
   ```

4. **Missing Python packages:**
   ```bash
   pip install selenium webdriver-manager
   ```

5. **Network error downloading ChromeDriver:**
   - Check internet connection
   - Retry the export

**Verify Chrome version:**
```bash
# macOS
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --version

# Linux
google-chrome --version

# Windows (PowerShell)
(Get-Item "C:\Program Files\Google\Chrome\Application\chrome.exe").VersionInfo.FileVersion
```

---

#### "WebGL not supported"

**Cause:** Chrome running in legacy headless mode.

**Solution:** The code now uses `--headless=new` which supports WebGL. Ensure Chrome 109+ is installed.

---

#### Blank/Black Images from WebDriver Export

**Symptom:** Exported PNGs are completely blank or black.

**Cause:** WebGL drawing buffer was cleared before capture.

**Solution:** This is automatically handled by the code which calls `scene.render()` before capture. If still experiencing issues:
```python
# Increase render wait time
vs = VisualizeSkeleton(
    ...,
    webdriver_render_wait=0.5  # Default is auto-calibrated
)
```

---

#### WebDriver Fallback

If WebDriver continues to fail:
```python
vs = VisualizeSkeleton(
    ...,
    export_method='kaleido'  # Use kaleido instead
)
```

---

### Export Method Performance Guide

Based on comprehensive benchmarking (5 trials per condition, scales 1-10), here are the recommended export methods:

#### Benchmark Results Summary

| Scale | kaleido | webdriver | webdriver-fast | **Fastest**    |
| ----- | ------- | --------- | -------------- | -------------- |
| 1     | 2.1s    | 0.7s      | **0.3s**       | webdriver-fast |
| 2     | 1.6s    | 0.9s      | **0.5s**       | webdriver-fast |
| 3     | 2.2s    | 1.1s      | **0.9s**       | webdriver-fast |
| 4     | 2.5s    | **1.3s**  | 1.5s           | webdriver      |
| 5     | 3.1s    | **1.4s**  | 2.2s           | webdriver      |
| 6     | 2.4s    | **1.6s**  | 2.5s           | webdriver      |
| 7     | 2.7s    | **1.7s**  | 2.3s           | webdriver      |


#### Recommendations by Use Case

| Use Case                       | Recommended Method             | Reason                                      |
| ------------------------------ | ------------------------------ | ------------------------------------------- |
| **Single PNG, scale 1-3**      | `webdriver-fast`               | 2-7x faster than kaleido                    |
| **Single PNG, scale 4-7**      | `webdriver`                    | Most consistent, 1.5-2x faster than kaleido |
| **Video export (many frames)** | `webdriver-fast` with scale ≤3 | Best throughput for batch processing        |
| **Maximum reliability**        | `kaleido`                      | Most stable, but 1.5-3x slower              |

#### Configuration Examples

```python
# Fast preview export (scale=1-3)
vs = VisualizeSkeleton(
    ...,
    export_method='webdriver-fast',
    export_scale=2
)

# High quality export (scale=4-7)
vs = VisualizeSkeleton(
    ...,
    export_method='webdriver',
    export_scale=5
)

# Video export (optimize for throughput)
vs = VisualizeSkeleton(
    ...,
    export_method='webdriver-fast',
    export_scale=3  # Good balance of quality and speed
)
vs.export_video(fps=30)

# Maximum reliability (fallback)
vs = VisualizeSkeleton(
    ...,
    export_method='kaleido',
    export_timeout=120
)
```

---

### Kaleido Export Errors

#### Export Timeout

**Symptom:**
```
Timeout waiting for kaleido to export image
```

**Solutions:**

1. **Increase timeout:**
   ```python
   vs = VisualizeSkeleton(..., export_timeout=120)
   ```

2. **Reduce figure complexity:**
   ```python
   vs = VisualizeSkeleton(
       ...,
       skeleton_mesh_simplification=0.95,  # More aggressive simplification
       export_scale=2  # Lower resolution
   )
   ```

3. **Use WebDriver instead:**
   ```python
   vs = VisualizeSkeleton(..., export_method='webdriver')
   ```

---

#### Large HTML File Errors

**Symptom:** Export fails on HTML files > 100MB.

**Solution:** Adjust auto-simplification threshold:
```python
vs = VisualizeSkeleton(
    ...,
    html_size_cap=150,  # Increase threshold (default: 100MB for kaleido, 200MB for webdriver)
)
```

---

### Brain Mesh & Transform Issues

#### "Target 'JRC2018F' has no known bridging registrations"

**Cause:** Incorrect template name in old code versions.

**Solution:** This is fixed in current version. If using old code:
- Use `JRCFIB2018F` instead of `JRC2018F`
- Update to latest version

---

#### Transform Download Issues

**Symptom:**
```
⚠️ Brain transforms not found for hemibrain:v1.2.1
```

**Solutions:**

1. **Accept the download prompt** (requires ~10GB disk space, 1-2 hours)

2. **Use template mesh instead:**
   ```python
   vs = VisualizeSkeleton(..., brain_mesh='template')  # Fast, no download
   ```

3. **Disable brain mesh:**
   ```python
   vs = VisualizeSkeleton(..., brain_mesh='none')
   ```

---

### Memory & Performance Issues

#### WebDriver Request Content-Length Error (Historical)

**Symptom:**
```
WebDriverException: Message: request content-length too big or unknown
```

**Status:** ✅ Fixed in current version.

**Solution (Automatic):** The current version always uses a local HTTP server to serve HTML files,
which works reliably with files of any size (tested up to 200MB+).

---

#### Chrome Tab Crashes During Video Export (Historical)

**Symptom:**
```
selenium.common.exceptions.WebDriverException: Message: tab crashed
```

**Status:** ✅ Fixed in current version.

**Solution (Automatic):** The current version uses `canvas.toDataURL()` for screenshot capture,
which is memory-efficient and doesn't cause Chrome to crash. Combined with automatic memory
cleanup intervals for large files, video exports now complete reliably.

---

#### Video Export Aborts Mid-Way

**Symptom:** Video or PNG export starts successfully but aborts somewhere in the middle
(e.g., after 50-100 frames), not at the beginning.

**Cause:** The default `webdriver_render_wait=0` may be too fast for some systems or
complex figures, causing rendering to fall behind.

**Solution:** Increase the render wait time:
```python
vs = VisualizeSkeleton(
    ...,
    webdriver_render_wait=0.3,  # Add delay between frames (try 0.3-0.5)
)
```

Alternatively, use `None` for auto-calibration:
```python
vs = VisualizeSkeleton(
    ...,
    webdriver_render_wait=None,  # Auto-calibrate based on machine performance
)
```

---

#### Memory Error / Out of Memory

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Reduce neuron count:**
   ```python
   vs = VisualizeSkeleton(
       ...,
       neuron_layers=['TypeA'],  # Instead of ['TypeA', 'TypeB', 'TypeC', ...]
   )
   ```

2. **Increase mesh simplification:**
   ```python
   vs = VisualizeSkeleton(
       ...,
       skeleton_mesh_simplification=0.98,  # More aggressive (default: 0.95)
   )
   ```

3. **Disable synapses:**
   ```python
   vs = VisualizeSkeleton(..., skip_synapse=True)
   ```

4. **Reduce export scale:**
   ```python
   vs = VisualizeSkeleton(..., export_scale=2)  # Instead of 5
   ```

---

#### Slow Performance

**Tips for faster visualization:**

1. **Use local caching:**
   ```python
   vs = VisualizeSkeleton(
       ...,
       cache_neurons=True,
       cache_synapses=True
   )
   ```

2. **Use simplified skeletons:**
   ```python
   vs = VisualizeSkeleton(
       ...,
       skeleton_mode='line'  # Instead of 'tube'
   )
   ```

3. **Limit to essential ROIs:**
   ```python
   vs = VisualizeSkeleton(
       ...,
       mesh_roi=['MB(R)']  # Only essential ROIs
   )
   ```

---

## Network Visualization Issues

### Network Appears Blank

**Symptom:** Network canvas shows nothing after generation.

**Cause (historical):** JavaScript type mismatch with Python booleans.

**Status:** ✅ Fixed in current version.

**Workaround (if using old version):**
- Update to latest version
- Regenerate the network HTML

---

### Layout Issues After Loading

**Symptom:** Nodes positioned incorrectly, overlapping, or layout looks broken after opening HTML file.

**Solutions:**

1. **Refresh the page**
   - Press `F5` or `Ctrl+R` (macOS: `Cmd+R`)
   - This forces the browser to re-render and re-calculate the layout

2. **Click the Reset Layout button**
   - Click the 🔄 **Reset** button in the Layout panel
   - This re-runs the layout algorithm with animation

3. **Try a different layout algorithm**
   - Use the dropdown selector to switch layouts (e.g., Dagre → fCoSE)
   - Some layouts work better for certain network structures

---

### Labels Not Visible

**Symptom:** Node labels are missing or invisible.

**Solutions:**

1. **Toggle label visibility**
   - Press `L` key to toggle label position (center ↔ outside)
   - Or click 👁️ **Hide Labels** / **Show Labels** button

2. **Increase font size**
   - Use the Font Size slider (minimum 8px, default 12px)

3. **Check opacity settings**
   - In Color Palette panel, ensure node opacity > 0%

---

### Hidden Elements Won't Return

**Symptom:** Pressed "Show All" but some nodes/edges still missing.

**Solutions:**

1. **Click Show All button**: Look for 👁️ **Show All Nodes** button
2. **Clear edge filter**: Empty the "Hide Edges (weight)" input field
3. **Refresh page**: As a last resort, reload the HTML file

---

### Negative Weight Display Issues

**Symptom:** Negative weights not displaying correctly.

**Status:** ✅ Fixed in current version.
- Negative edges now show in light blue
- Positive edges show in gray
- Absolute values used for edge width calculation
- Original signed values shown in hover labels

---

### Heatmap NaN Values

**Symptom:** Heatmap shows NaN values with log/sqrt scales.

**Status:** ✅ Fixed in current version.
- Signed transforms now prevent NaN in logarithmic scales

---

## Data Loading Issues

### "No neurons found matching criteria"

**Symptom:**
```
Warning: No neurons found matching criteria in local FAFB data.
```

**Solutions:**

1. **Check neuron type spelling:**
   ```python
   # Use exact type name from dataset
   neuron_layers=['MTe50']  # Correct
   neuron_layers=['MTE50']  # Wrong - case sensitive
   ```

2. **Use regex for flexible matching:**
   ```python
   neuron_layers=['MTe.*']  # Matches MTe50, MTe51, etc.
   ```

3. **Verify dataset:**
   ```python
   # Ensure neuron type exists in your dataset
   vs = VisualizeSkeleton(
       dataset='hemibrain:v1.2.1',  # Check this matches your data
       ...
   )
   ```

---

### KeyError: 'bodyId'

**Symptom:**
```
KeyError: 'bodyId'
```

**Cause:** Empty DataFrame or missing column in layer_map_csv.

**Solution:** Verify your CSV file has the required columns:
```csv
bodyId,type
720575940624086675,MTe50
720575940612345678,MTe51
```

---

### Connection Cache Not Found

**Symptom:**
```
[ERROR] Connection cache not found
```

**Solution:**
1. Build the connection cache first:
   ```python
   from src.build_connection_cache import build_connection_cache
   build_connection_cache(dataset='hemibrain:v1.2.1')
   ```

2. Or enable automatic caching:
   ```python
   vs = VisualizeSkeleton(..., cache_neurons=True)
   ```

---

### Failed to Fetch Mesh/Skeleton/Synapses

**Symptom:**
```
✗ Failed to fetch mesh: {body_id}: {error}
✗ Failed to fetch skeleton: {body_id}: {error}
✗ Failed to fetch synapses: {body_id}: {error}
```

**Cause:** Network issues or invalid bodyId.

**Solutions:**
1. **Retry:** Network errors are often transient
2. **Verify bodyId:** Ensure the neuron exists in the dataset
3. **Check authentication:** Verify your tokens are valid
4. **Use local cache:** If available, the cached version will be used automatically

---

### k3d/ipywidgets Not Installed

**Symptom:**
```
⚠️ k3d not installed. Please install it with `pip install k3d`
⚠️ ipywidgets not installed. Cannot save k3d plot to HTML.
```

**Solution:**
```bash
pip install k3d ipywidgets
```

Note: k3d is optional and only needed for Jupyter notebook interactive 3D plots.

---

### trimesh Not Installed

**Symptom:**
```
⚠️ trimesh not installed. Cannot export 3D model.
```

**Solution:**
```bash
pip install trimesh
```

---

### Clustering Failed

**Symptom:**
```
Clustering failed: {error}
```

**Cause:** Too few data points or invalid data for clustering.

**Solution:** Ensure you have enough neurons/connections for meaningful clustering. This warning can often be safely ignored.

---

## Cross-Platform Compatibility

### macOS

| Issue             | Solution                                                       |
| ----------------- | -------------------------------------------------------------- |
| tkinter not found | Included with Python by default on macOS                       |
| PyQt5 slow        | `pip install PyQt5` for faster dialogs                         |
| Chrome path       | `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` |

### Linux

| Issue             | Solution                                |
| ----------------- | --------------------------------------- |
| tkinter not found | `sudo apt-get install python3-tk`       |
| PyQt5 not found   | `sudo apt-get install python3-pyqt5`    |
| Chrome not found  | `sudo apt install google-chrome-stable` |
| Display issues    | Set `DISPLAY` environment variable      |

### Windows

| Issue            | Solution                                                |
| ---------------- | ------------------------------------------------------- |
| legacy memray error | Rerun `archive/install/install.bat` to remove the obsolete client   |
| Chrome path      | `C:\Program Files\Google\Chrome\Application\chrome.exe` |
| Long path errors | Enable long paths in Windows settings                   |

---

## Advanced Settings

### Reading Docstrings for Parameter Options

For fine control and advanced settings, **read the docstrings** in the source code. Each class and function has detailed parameter documentation.

**In Python/IPython:**
```python
from visualize_skeleton import VisualizeSkeleton

# View all parameters and their options
help(VisualizeSkeleton)

# Or in Jupyter/IPython
VisualizeSkeleton?
```

**Key classes with extensive options:**
- `VisualizeSkeleton` - 3D visualization with 50+ parameters
- `FindNeuronConnection` - Pathfinding with filtering options
- `NeuronBridgeFinder` - EM↔LM mapping options
- `InterDatasetComparator` - Cross-dataset comparison settings

**Example docstring parameters:**
```python
# From VisualizeSkeleton docstring:
skeleton_mesh_simplification: float = 0.95
'''
Mesh simplification ratio for skeleton rendering.
- 0.95: Reduce to 5% of original faces (default, fast)
- 0.9: Reduce to 10% (more detail)
- 0.5: Reduce to 50% (high detail, slow)
'''
```

---

## Reporting Issues

### How to Report a Bug

If you encounter an issue not covered in this guide:

1. **Check existing issues:** https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/issues

2. **Email the maintainer:** krleng(at)pku.edu.cn
   - Replace `(at)` with `@`

3. **Include in your report:**
   - Python version (`python --version`)
   - Package versions (`pip list | grep -E "navis|plotly|selenium|pandas|numpy"`)
   - Full error traceback
   - Minimal code to reproduce the issue
   - Dataset and neuron types involved

---

## Getting Help

### Debug Information

When reporting issues, include:

1. **Python version:**
   ```bash
   python --version
   ```

2. **Package versions:**
   ```bash
   pip list | grep -E "navis|plotly|selenium|pandas|numpy"
   ```

3. **Chrome version:**
   ```bash
   google-chrome --version  # or equivalent for your OS
   ```

4. **Full error traceback**

5. **Minimal reproducible example**

### Resources

- **GitHub Issues:** Report bugs and request features
- **Email Support:** krleng(at)pku.edu.cn (replace `(at)` with `@`)
- **Documentation:** [docs/README.md](README.md)
- **NeuPrint Help:** https://neuprint.janelia.org/help
- **NAVIS Documentation:** https://navis.readthedocs.io/
- **FlyWire Codex:** https://codex.flywire.ai/

---

## Related Documentation

- [Installation Guide](INSTALLATION.md)
- [3D Skeleton Guide](visualizations/3D_Skeleton_Guide.md)
- [Output Files Reference](OUTPUT_FILES.md)
- [FAFB Integration](FAFB_INTEGRATION.md)
- [BANC Integration](BANC_INTEGRATION.md)
- [Brain Template Fix](bugfixes/Brain_Template_Fix_Nov2024.md)
- [Network Blank Fix](visualizations/NETWORK_BLANK_FIX.md)

---

*Last updated: January 2026*
