# FAFB and Standalone BANC Dataset Usage Guide

This toolkit supports analysis of FAFB and standalone BANC datasets using
separate **local-release workflows**. FAFB raw files come from the
Codex/FlyWire portal (automatically downloadable via Settings → FAFB Data
Downloads once a FlyWire Codex token is saved) and are converted from
`datasets/<dataset>/downloads/`.
BANC metadata and connections are prepared on demand from its public release
bucket, and BANC skeletons are fetched from that same bucket. FAFB alone has
an optional CAVE API path for workflows that explicitly request remote
fetching; BANC never uses CAVE.

## 1. Data Preparation

### FAFB: local-release preparation

FAFB raw files come from the Codex/FlyWire portal. See the [FAFB Integration
Guide](FAFB_INTEGRATION.md) for the full file list.

**Automatic path:** save a FlyWire Codex API token (Settings → API Tokens)
and use Settings → FAFB Dataset Downloads — three levels (necessary data /
synapses / skeleton bundle), resumable, with the converter run for you.
Skeletons also fetch lazily per neuron from the Codex server during
visualization. The manual steps below are the offline fallback.

#### Step 1: Download FAFB data files

Download the required FAFB CSV files and optional healed skeleton bundle from
the Codex Download Page (or an equivalent authorized source).

#### Step 2: Place FAFB files in the dataset directory

Create the directory structure in your project folder and place the downloaded
files there:

```
datasets/
  └── flywire_FAFB_v783/
      └── downloads/
            ├── classification.csv.gz
            ├── names.csv.gz
            ├── coordinates.csv.gz
            ├── neurons.csv.gz
            ├── cell_stats.csv.gz
            ├── consolidated_cell_types.csv.gz
            ├── connections_princeton_no_threshold.csv.gz
            ├── fafb_v783_princeton_synapse_table.csv.gz  (optional)
            └── sk_lod1_783_healed.zip                    (optional)
```

#### Step 3: Convert FAFB files

The converter can be run manually, or it runs automatically when the dataset
is first used:

```bash
python src/FAFB_file_converter.py
```

The FAFB converter will:
1.  Read the CSV files from the `downloads` folder.
2.  Merge and enrich the neuron metadata.
3.  Convert the data into optimized Parquet files (`.parquet`) for fast loading.
4.  Save the processed files in the dataset folder.

### BANC: Automatic public-release preparation

BANC (`banc_v626` or `banc_v888`) does not use the Codex/FlyWire download
workflow. On first use, DROCAT downloads the selected BANC metadata and
connection product from the public `banc_public_gcs` release bucket and builds
the local Parquet tables automatically. No login, CAVE token, or manual
`neurons.csv.gz`/`connections_princeton.csv.gz` download is required.

Skeletons are fetched on demand from the same public bucket and cached as
`.swc.zst` files under `cache/<dataset>/skeletons/raw_skeletons/` (derived
release caches, if a workflow builds one, live beside that `skeletons/`
directory). See the [BANC Integration
Guide](BANC_INTEGRATION.md) for the release layout and source details.

If an offline legacy BANC CSV bundle is already available, the converter still
supports it as a compatibility fallback. Place the files in the matching
`datasets/<dataset>/downloads/` directory and run:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from BANC_file_converter import ensure_banc_data; d='banc_v888'; ensure_banc_data(d, 'datasets/' + d)"
```

This fallback reads the existing CSVs and writes the canonical BANC Parquet
tables; it is not part of the normal public-bucket setup.

## 2. Using FAFB/BANC Data in Analysis

Once the appropriate local source is available, you can use FAFB and BANC in
the same analysis APIs. BANC's tables and skeletons remain owned by its public
release workflow; it is not routed through NeuPrint or CAVE.

### Example: Finding Connections

```python
from coana import FindNeuronConnection

# Initialize connection finder
# bodyIds verified against the prepared banc_v888 table (aMe12 -> PPL101,
# the same pair exercised by tests/e2e/test_banc_ame12_ppl_e2e.py)
fc = FindNeuronConnection(
    token='dummy_token',  # Token is ignored for local files
    dataset='banc_v888', # or 'banc_v626' or 'flywire_FAFB_v783'
    sourceNeurons=['720575941596944935'],  # aMe12 (BANC bodyId)
    targetNeurons=['720575941416009108'],  # PPL101
    min_synapse_num=5
)

# Run analysis
fc.InitializeNeuronInfo()
fc.FindDirectConnections()
```

### Example Script

Select the dataset in the UI or in a direct `FindNeuronConnection` script. For
FAFB, the first run checks the matching `downloads/` folder and converts the
raw files if the generated tables are absent. For BANC, the first run prepares
the metadata and connection tables from the public release bucket.

### Example: Visualizing Skeletons

If you downloaded the `sk_lod1_783_healed.zip` file for FAFB — or saved a
FlyWire Codex token, in which case skeletons fetch lazily per neuron — you
can visualize 3D skeletons.

```python
from visualize_skeleton import VisualizeSkeleton

vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    neuron_layers=['720575940596125868'],  # FAFB root ID present in the local table
    brain_mesh='native',    # Uses the FAFB (FLYWIRE) template
    FAFB_template_correction=True # Default: True. Corrects the slight tilt of the FAFB template.
)

vs.plot_neurons()
```

For BANC, select `banc_v626` or `banc_v888`; missing skeletons are fetched on
demand from the public release bucket and cached as `.swc.zst` files in the
BANC cache namespace. No manual skeleton download or CAVE token is involved.

### FAFB Tilt Correction
The FAFB/FlyWire template mesh has a slight tilt relative to the standard view axes. By default (`FAFB_template_correction=True`), `VisualizeSkeleton` applies a rotation correction to align the brain:
- **Z-axis rotation**: -4 degrees (corrects left-right tilt in front view)
- **Y-axis rotation**: -3 degrees (corrects tilt in top view)

This ensures that the brain appears straight in standard views (Front, Top, etc.). If you need the original raw coordinates (e.g., for alignment with other raw FAFB data), you can set `FAFB_template_correction=False`.

## 3. Important Notes

-   **Storage**: FAFB raw CSVs and bundles can be large. BANC's metadata and connection products are cached under the dataset `downloads/` folder after automatic preparation; BANC skeletons are cached as compressed SWCs under `cache/`.
-   **Updates**: For FAFB, replace the files in the `downloads` folder and rerun the converter. For BANC, remove the generated local tables and cached release products before rerunning automatic preparation.
-   **IDs**: FAFB and BANC use long integer Root IDs (e.g., `720575940...`). Ensure you use these IDs in your queries.
