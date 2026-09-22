# FAFB (FlyWire) Data Integration

This project supports the FAFB (Full Adult Fly Brain) dataset from FlyWire using a local file-based approach for high performance.

## Data Preparation

We provide a dedicated script to prepare the FAFB data for analysis. This script handles file organization, format conversion (to Parquet), and data enrichment.

### 1. Download Data

Download the following files from [FlyWire Codex Downloads](https://codex.flywire.ai/api/download?dataset=fafb):

**Required for local connection analysis:**
*   `classification.csv.gz` (Neuron Classification) - **~1 MB**
*   `connections_princeton_no_threshold.csv.gz` (Connectivity) - **~263 MB**
    *   *Note: `connections_princeton.csv.gz` or `connections.csv.gz` are also accepted as fallbacks.*

**Recommended metadata enrichment (optional):**
*   `consolidated_cell_types.csv.gz` (Consolidated Cell Types) - **~1 MB**
*   `names.csv.gz` (Neuron Names) - **~1 MB**
*   `coordinates.csv.gz` (Soma Coordinates) - **~5 MB**
*   `neurons.csv.gz` (Neurotransmitters) - **~2 MB**
*   `cell_stats.csv.gz` (Cell Statistics) - **~2.5 MB**

The converter warns when enrichment files are missing and continues with
incomplete neuron metadata. It stops only when the classification or
connectivity input is missing.

**Optional Visualization Files:**
*   `sk_lod1_783_healed.zip` (Skeletons - needed for local 3D skeleton visualization) - **~13 GB**
*   `fafb_v783_princeton_synapse_table.csv.gz` (Synapses - needed for local synapse-table visualization) - **~2.5 GB**

**Storage Summary:**
*   **Minimal Analysis:** ~300 MB download → ~200 MB final size
*   **With Synapses:** +2.5 GB download → +0.9 GB final size
*   **With Skeletons:** +13 GB download → +13 GB final size
*   **Full Dataset:** ~16 GB total storage required

The converted synapse table keeps only what the visualization reader
consumes: pre/post site coordinates and root ids. The cleft-centre
columns (`ctr_*`), `size` and `neuropil` are dropped at conversion and
are absent from tables converted by recent DROCAT versions; tables
converted earlier are slimmed **and** losslessly re-encoded in place
automatically the next time the synapse visualization or converter
touches them (~0.1 GB reclaimed on top of the dropped columns). The
stored layout is measured-lossless: coordinates narrow to int32
(guarded by footer statistics) and take `BYTE_STREAM_SPLIT`, the string
id columns take `DELTA_LENGTH_BYTE_ARRAY`, and a `DROCAT.lossless`
footer marker makes the upgrade a one-time pass — values read back are
bit-identical. The source CSV (`fafb_v783_princeton_synapse_table.csv.gz`)
retains the dropped columns if they are ever needed again (see
`docs/audits/SYNAPSE_TABLE_COLUMN_AUDIT_2026-09-10.md`).

### 2. Run the Converter

You can trigger the data preparation in two ways:

**Option A: Run the converter directly (Recommended)**
```bash
# Run from the project root
python src/FAFB_file_converter.py
```

**Option B: Run any analysis script**
Simply selecting `flywire_FAFB_v783` in a tool (or initializing
`FindNeuronConnection` with that dataset) will automatically check for and
convert the data if needed.

The script will check for the required files and guide you if anything is missing.

**What the script does:**
1.  Creates the `datasets/flywire_FAFB_v783` directory structure.
2.  Checks the `datasets/flywire_FAFB_v783/downloads` folder for source files.
3.  If files are missing, it prints a list of what to download and where to put them.
4.  Converts CSVs to optimized **Parquet** files for fast loading.
5.  Merges metadata (names, coordinates, types) into a single neuron dataframe.
6.  Moves the skeleton zip file to the correct location.

### 3. Cleanup (Removable Files)

After the script successfully completes (look for "✓ Conversion complete" messages), you can safely delete the entire `downloads` folder to save space.

**Removable Folder:**
*   `datasets/flywire_FAFB_v783/downloads/` (The entire folder can be deleted)

**Do NOT delete:**
*   The generated `.parquet` files in `datasets/flywire_FAFB_v783/`.
*   The `sk_lod1_783_healed.zip` file in `datasets/flywire_FAFB_v783/` (The converter moves this file out of downloads, so it is safe).

## Usage

### Path Finding

Use `FindNeuronConnection` with the FAFB dataset name.

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from coana import FindNeuronConnection

fnc = FindNeuronConnection(
    dataset='flywire_FAFB_v783',
    sourceNeurons=['l-LNv.*'],  # Regex patterns supported
    targetNeurons=['s-LNv.*'],
    max_interlayer=2,
    min_synapse_num=5,
    verbose_mode='simple',
)

fnc.InitializeNeuronInfo()
fnc.FindAllPath()
```

### Visualization

Visualize skeletons using `VisualizeSkeleton` (or the shorthand `Vis3S`).

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from visualize_skeleton import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    token='',  # Not needed for FAFB local data
    output_dir='/path/to/output',
    neuron_layers=['l-LNv'],  # Neuron types or bodyIds
    skip_synapse=True,
    neuron_alpha=0.2,
    min_synapse_num=3,
    skeleton_mode='tube',
    legend_mode='layer',
    show_fig=True,
    brain_mesh='native',  # Use native FAFB coordinates
    cache_neurons=True,
)

vs.plot_neurons()
```

### CAVE API Fetching (force_API_fetching)

To route FAFB skeleton resolution through CAVE instead of the local ZIP file,
set `force_API_fetching=True`. With `cache_neurons=True`, existing
`cave_skeletons` entries may be reused; set `cache_neurons=False` for an
online-only fetch that neither reads nor writes the replacement store:

```python
from visualize_skeleton import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    output_dir='/path/to/output',
    neuron_layers=['l-LNv'],
    skip_synapse=True,
    neuron_alpha=0.2,
    skeleton_mode='tube',
    legend_mode='layer',
    show_fig=True,
    brain_mesh='native',
    cache_neurons=True,
    force_API_fetching=True,  # Use the CAVE resolution path
)

vs.plot_neurons()
```

**Key Features:**
- **Source priority (every render mode)**: previously repaired bodies served from the local repair caches (`cave_skeletons` / `extrusion_fixes`, status-driven) → shared raw SWC cache (`.swc.zst`, legacy frozen reads) → healed skeleton zip → one-time extrusion check. Missing or extrusion-affected bodies use the dedicated CAVE replacement store when available, then CAVE skeletonization. Every FAFB source is a TreeNeuron, so all pipelines render from the same geometry.
- **CAVE skeletonization**: missing or extrusion-flagged bodies are replaced by wavefront-skeletonizing the raw CAVE mesh (no pre-decimation — measured to match the healed zip's node density within ~11%). The level-0 tree is cached in the dedicated `cache/{dataset}/skeletons/cave_skeletons/` store with a `# DROCAT source: cave_mesh_wavefront` header, so the replacement never overwrites the raw store. Later runs can skip the network for bodies present in that store.
- **force_API_fetching=True**: routes every body through CAVE resolution. With `cache_neurons=True`, an existing CAVE replacement may be reused; with `cache_neurons=False`, the fetch is online-only.
- **Automatic Fallback**: bodies missing from every local source fall through to CAVE skeletonization automatically.
- **Updated Data**: Use `force_API_fetching=True, cache_neurons=False` when you need to bypass the local CAVE replacement store and request current data from the API.

### Fixing Skeleton Extrusion Issues

The downloaded FAFB skeleton ZIP (`sk_lod1_783_healed.zip`) may contain neurons with extrusion artifacts (mesh errors that appear as spikes or protrusions). These typically occur around the soma (cell body) region when aggressive mesh simplification is applied.

#### Understanding Extrusions

Extrusion artifacts happen because:
1. The soma region has high vertex density in the original mesh
2. When simplification is applied uniformly (e.g., 0.95 = remove 95% of faces), the soma's fine structure collapses
3. This creates "spiky" protrusions extending from the cell body

#### Solution 1: Automatic Extrusion Detection and Fix

Enable `auto_fix_extrusions=True` to automatically detect and replace problematic skeletons:

```python
from visualize_skeleton import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    neuron_layers=['MTe50', 'MTe51', 'MTe54'],
    skeleton_mesh_simplification=0.95,
    auto_fix_extrusions=True,  # Automatically detect and fix extrusions
    cache_neurons=True,
    show_fig=True,
)
vs.plot_neurons()
```

**How auto_fix_extrusions works:**
1. When loading skeletons from the raw cache or healed zip, each is converted to a simplified mesh
2. Edge length analysis detects abnormal "spiky" geometry (edge ratio > 10x median)
3. Problematic neurons are replaced by wavefront-skeletonizing their raw CAVE mesh (one-time ~5-20s per neuron, then cached)
4. If a CAVE fetch fails, the long parent→child edge is mapped back to the local
   tree and only that child subtree is pruned when the cut is safe; the pruned
   fix is persisted to `cache/{dataset}/skeletons/extrusion_fixes/` with a
   `# DROCAT source: local_extrusion_fix` header
5. **Extrusion check results are cached** in `cache/{dataset}/extrusion_check_results.parquet`
6. On subsequent runs, only new neurons are checked (previously checked neurons use cached results)
7. CAVE replacement trees are cached in the dedicated `cave_skeletons/` store (never overwriting the
   raw store); bodies recorded `api_repaired` are served from that store, and bodies recorded
   `local_fallback` from `cave_skeletons/` then `extrusion_fixes/`, on later runs without another
   network round-trip. Statuses recorded `api_failed` retry CAVE on the next run.

**Performance notes:**
- First run may take longer due to mesh analysis for extrusion detection
- Subsequent runs are fast because check results are cached in parquet format
- CAVE skeletonization is a one-time cost per replaced neuron; the replacement store makes later runs offline for those cached replacements
- Set `auto_fix_extrusions=False` if you need faster loading and can tolerate artifacts

#### Solution 2: Soma-Aware Simplification (Built-in)

The visualization system includes **soma-aware simplification** that applies gentler simplification to the soma region:

```python
from visualize_skeleton import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    neuron_layers=[720575940624086675],
    skeleton_mesh_simplification=0.95,     # Aggressive on skeleton branches
    soma_mesh_simplification=0.8,          # Gentler on cell body (default)
    soma_region_radius=20000,              # 20µm radius around soma (default)
    cache_neurons=True,
    show_fig=True,
)
vs.plot_neurons()
```

**Default render settings:**
- Skeleton simplification: 0.95 (keep 5% of faces)
- Soma simplification: 0.8 (keep 20% of faces) 
- Soma region radius: 20,000nm (20µm)

#### Solution 3: Detect and Fix Extrusions Manually

Use the built-in detection tools to identify problematic neurons:

```python
from visualize_skeleton import VisualizeSkeleton

# Check a specific neuron for extrusions
result = VisualizeSkeleton.check_fafb_skeleton_for_extrusions(
    720575940624086675,
    simplification=0.95,
    verbose=True
)

if result['has_extrusions']:
    print(f"Extrusions detected! Severity: {result['severity']}")
    print(f"Recommendation: {result['recommendation']}")
    
    # Fix it by fetching from CAVE API
    VisualizeSkeleton.fix_fafb_extrusions([720575940624086675])
```

Or use **auto-fix** mode:

```python
# Check AND automatically fix if extrusions found
result = VisualizeSkeleton.check_fafb_skeleton_for_extrusions(
    720575940624086675,
    auto_fix=True,  # Automatically fetch from API if needed
    verbose=True
)
print(f"Auto-fixed: {result['auto_fixed']}")
```

#### Solution 4: Manual API Fetching

For more control, you can manually route selected skeletons through the CAVE
API path. Use `cache_neurons=False` when the request must bypass an existing
local CAVE replacement; that online-only mode does not write a cache:

```python
from visualize_skeleton import VisualizeSkeleton

# Method 1: Fix specific neurons by fetching them via API
# Route the selected neurons through the CAVE resolution path
vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    neuron_layers=[720575940596125868, 720575940597856265],  # Problematic bodyIds
    force_API_fetching=True,  # Use CAVE; cache_neurons=True may reuse a stored replacement
    show_fig=False,  # Just cache the fixed skeletons
    cache_neurons=True,
)
vs.plot_neurons()  # This caches the API-fetched skeletons

# Method 2: Once fixed, run your full visualization
# Stored CAVE replacements are reused for the corresponding repaired bodies
vs2 = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    neuron_layers=['l-LNv', 's-LNv'],  # Mix of neurons
    force_API_fetching=False,  # Use raw/ZIP data; repaired bodies use cave_skeletons
    show_fig=True,
    brain_mesh='native',
)
vs2.plot_neurons()
```

#### Extrusion Detection API

The `detect_mesh_extrusions()` method provides detailed analysis:

```python
from visualize_skeleton import VisualizeSkeleton

# If you have a mesh object already
result = VisualizeSkeleton.detect_mesh_extrusions(
    mesh,
    soma_pos=[100, 200, 300],  # Soma position (optional)
    soma_radius=20000,         # 20µm radius
    verbose=True
)

# Result dictionary contains:
# - 'has_extrusions': bool
# - 'severity': 'none', 'mild', 'moderate', or 'severe'
# - 'extrusion_count': number of problematic vertices
# - 'edge_length_ratio': max/median edge ratio (>3 indicates issues)
# - 'soma_region_issues': bool (extrusions near cell body)
# - 'recommendation': suggested action string
```

**How Extrusion Fixes Work:**
1. CAVE replacement skeletons are cached in the dedicated
   `cache/{dataset}/skeletons/cave_skeletons/` store; legacy API-cache
   pickles are read only as a migration fallback
2. Extrusion check results are cached per neuron in
   `cache/{dataset}/extrusion_check_results.parquet`
3. TreeNeuron sources (bundle / raw SWC cache) run the extrusion check every
   render; flagged neurons are replaced through the CAVE replacement path
4. This allows you to selectively fix problematic neurons without re-downloading the entire 13GB ZIP
5. Fixed neurons persist across sessions via the cache

**Note on force_API_fetching Behavior:**
- **VisualizeSkeleton**: Uses raw/ZIP sources when
  `force_API_fetching=False`; repaired bodies use `cave_skeletons` and
  `force_API_fetching=True` routes all bodies through CAVE resolution
- **FindNeuronConnection**: Uses API only when `force_API_fetching=True` (for consistency with local data)

**Requirements:**
- CAVE token for API fetching (obtain from https://codex.flywire.ai/auth_token)
- Set token in `config.json` (or the gitignored `config_local.json` fallback) or as environment variable `CAVE_TOKEN`

**Note:** BANC is a separate public-release source and does not support
`force_API_fetching`; see the [BANC Integration Guide](BANC_INTEGRATION.md).

## Find Similar (Morphology) on FAFB

- **vector_v2** uses the skeleton-vector cache built from the healed zip
  (`find_similar/morphology/skeleton__vectors_v2.parquet`). The prepared-mesh
  cache is the V1 counterpart and is not used by vector_v2 scoring.
- **Skeleton loading follows the canonical FlyWire chain**
  (`morphology.load_local_release_skeletons`): previously repaired bodies
  from the local repair caches (`cave_skeletons` / `extrusion_fixes`,
  status-driven) → raw `.swc.zst` cache (legacy frozen reads) → healed zip →
  one-time extrusion check with cached results (flagged neurons are replaced
  through the `cave_skeletons` store, then CAVE; a CAVE failure persists a
  locally pruned fix to `extrusion_fixes`) → token-gated CAVE
  skeletonization for anything still missing. The prepared mesh cache is
  never consulted — scoring is TreeNeuron-native.
- **NBLAST** scores the entire candidate pool from skeletons — the local raw
  store first, healed-zip fallback — so pool coverage does not depend on
  vector-cache rows.
- **NBLAST is not mirror-invariant**: contralateral same-type pairs score at
  chance. Type-level means therefore aggregate ipsilateral pairs only,
  classified via the neuron index's `hemisphere` column (`somaSide` on
  NeuPrint datasets); `results.csv` keeps every pair row. The vector method
  lateral-normalizes at vectorization and uses both sides.
- **Dotprops are rebuilt per search in memory and never written to disk** —
  only raw skeletons and vector rows persist, so a schema or parameter change
  can never leave stale dotprops behind.
- A vector cache written by an older schema version is detected (schema
  version + lateral-normalization marker) and rebuilt automatically on the
  next search; the ZCA whitener is refit when its fit version differs.

## Notes

*   **Root IDs**: FAFB root IDs are very large integers. The system handles them as strings internally to avoid precision loss, but you can pass them as integers in your scripts.
*   **Caching**: The caching system currently produces warnings for FAFB IDs due to their size, but this does not affect the analysis results.
*   **Skeletons**: Skeleton visualization requires either the `sk_lod1_783_healed.zip` file, a recognized local skeleton cache, or CAVE API access. Repaired CAVE trees are stored under `cache/{dataset}/skeletons/cave_skeletons/` and are used offline on later runs.
*   **Extrusion Issues**: The downloaded `sk_lod1_783_healed.zip` may contain neurons with extrusion artifacts (mesh errors appearing as spikes). Use `VisualizeSkeleton.fix_fafb_extrusions([bodyId1, bodyId2, ...])` to fetch and cache replacement skeletons from CAVE. If CAVE is unavailable during automatic repair, the visualizer prunes a safely localized bad node subtree in memory without overwriting the raw ZIP/cache source.
