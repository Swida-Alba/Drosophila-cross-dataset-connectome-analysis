# Basic Usage Guide (v4.5.0)

This guide demonstrates the core functionality of DROCAT using the **practical example scripts** in the [`scripts/`](../../scripts/FindPath.py) folder. These are the most commonly used workflows.

For agent-driven execution without the web UI, use the repository's
[`drocat-usage` skill](../../skills/drocat-usage/SKILL.md). It provides a
focused tool catalog, a safe direct launcher, output-validation recipes, and
guidance for targeted script/backend edits.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Script Overview](#script-overview)
- [FindDirect.py - Direct Connections](#finddirectpy---direct-connections)
- [FindPath.py - Multi-Hop Pathways](#findpathpy---multi-hop-pathways)
- [NeuronBridge_FindLines.py - Find Driver Lines](#neuronbridge_findlinespy---find-driver-lines)
- [FlyLight_fetcher.py - Download Images](#flylight_fetcherpy---download-images)
- [Common Parameters Reference](#common-parameters-reference)
- [Output Files](#output-files)

---

## Prerequisites

### 1. Set Up Your Environment

```bash
# Create and activate environment
conda create -n drocat-4.5.0 python=3.11 -y
conda activate drocat-4.5.0

# Install dependencies
pip install -r requirements.txt
```

### 2. Get Your NeuPrint Token

1. Visit [NeuPrint](https://neuprint.janelia.org/)
2. Log in with your Google account
3. Go to **Account** → Copy your **Auth Token**

Set the token as environment variable (recommended):

```bash
export NEUPRINT_TOKEN="your_token_here"
```

Or edit the `token=''` line in each script.

---

## Script Overview

All example scripts are in the [`scripts/`](../../scripts/FindPath.py) folder:

| Script                                                                   | Purpose                          | Typical Use                                 |
| ------------------------------------------------------------------------ | -------------------------------- | ------------------------------------------- |
| **[FindDirect.py](../../scripts/FindDirect.py)**                         | Direct synaptic connections      | "What connects to what?"                    |
| **[FindPath.py](../../scripts/FindPath.py)**                             | Multi-hop pathways               | "How does signal flow A→B?"                 |
| **[NeuronBridge_FindLines.py](../../scripts/NeuronBridge_FindLines.py)** | Find driver lines for EM neurons | "Which GAL4 lines label my neurons?"        |
| **[FlyLight_fetcher.py](../../scripts/FlyLight_fetcher.py)**             | Download driver line images      | "Get images for line SS01015"               |
| **[FindHomologs.py](../../scripts/FindHomologs.py)**                     | Cross-dataset homolog finding    | "Find equivalent neurons in other datasets" |
| **[InterDatasetComparator.py](../../scripts/InterDatasetComparator.py)** | Cross-dataset comparison         | "Compare connections across datasets"       |

**To run any script:**

```bash
cd scripts/
python FindDirect.py  # or any other script
```

---

## FindDirect.py - Direct Connections

**Location**: [`scripts/FindDirect.py`](../../scripts/FindDirect.py)

Find **direct synaptic connections** between neurons.

### Key Configuration

Edit these parameters in the script:

```python
neurons_network = ['aMe.*']  # Regex pattern for neurons

fc = FindNeuronConnection(
    token='',                         # Your NeuPrint token
    output_dir='../local_data/connection_data',
    dataset='male-cns:v0.9',         # Which dataset
    sourceNeurons=neurons_network,    # Source neurons
    targetNeurons=neurons_network,    # Target neurons
    
    # Connection filters
    min_synapse_num=1,               # Minimum synapses per connection
    min_ratio=0.0,                   # Minimum connection ratio (weight/post)
    min_traversal_probability=0.0,   # Minimum propagation probability
    filter_by='bodyId',              # 'bodyId' or 'type' level filtering
    
    # Options
    exclude_intra_type_connections=False,  # Exclude same-type connections
    use_cache=True,                  # Enable caching for speed
    output_format='csv',             # 'xlsx' or 'csv'
    showfig=False,                   # Display interactive plots
)

fc.InitializeNeuronInfo()
fc.FindDirectConnections()
```

### Common Patterns

**Find all connections within a neuron type:**
```python
neurons_network = ['MBON.*']
sourceNeurons = neurons_network
targetNeurons = neurons_network
exclude_intra_type_connections = False
```

**Find connections between two specific types:**
```python
sourceNeurons = ['KC.*']
targetNeurons = ['MBON03']
min_synapse_num = 10  # Only strong connections
```

**Filter by connection strength:**
```python
min_ratio = 0.01                    # At least 1% of target's inputs
min_traversal_probability = 0.03    # At least 3% propagation probability
```

### Output Files

```
local_data/connection_data/
├── {date}_{source}_{target}_direct_connections.csv
├── {date}_{source}_{target}_network.html
├── {date}_{source}_{target}_heatmap.html
└── {date}_{source}_{target}_summary.xlsx
```

> 📖 **See [Score Calculation Guide](./ScoreCalculation_Guide.md)** for details on `connection_ratio` and `traversal_probability`.

---

## FindPath.py - Multi-Hop Pathways

**Location**: [`scripts/FindPath.py`](../../scripts/FindPath.py)

Find **multi-hop pathways** connecting neurons through intermediate layers.

### Key Configuration

```python
fc = FindNeuronConnection(
    token='',
    output_dir='../local_data/connection_data',
    dataset='flywire_FAFB_v783',     # NeuPrint or FAFB dataset
    
    # Neurons
    sourceNeurons=['CB0038'],
    targetNeurons=['LPLC2'],
    custom_source_name='Fdg',        # Custom label for sources
    custom_target_name='',
    
    # Connection filters
    min_synapse_num=3,
    min_ratio=0.0,
    min_traversal_probability=0,
    filter_by='bodyId',
    
    # Pathfinding options
    max_interlayer=4,                # Maximum intermediate layers
    keyword_in_path_to_remove=['None'],  # Exclude these types from paths
    pathfinding='StrongestFirst',  # Built-in 'all'-mode algorithm (bounded by Max Paths with a reported tau)
    drop_untyped=True,             # Neuron-label filter: drop edges touching
                                   #   untyped neurons before the graph is built
    
    # Performance options
    skip_bodyId=True,                # Skip bodyId-level for speed
    use_cache=True,
    output_format='csv',
    edgeN_limit=500,                 # Visualization Edge Limit (network/Sankey/heatmap)
)

fc.InitializeNeuronInfo()
fc.FindAllPath(forward_only=True)   # forward_only=True for faster search
```

### Pathfinding Algorithms

StrongestFirst is the built-in pipeline algorithm (default since
2026-09-05; bounded by Max Paths with a reported tau). The complete
enumerators below are API-only alternatives (set
`max_paths_bodyid=0` for unbounded runs):

| Algorithm         | When to Use                | Speed   | Memory |
| ----------------- | -------------------------- | ------- | ------ |
| `'StrongestFirst'`| **All depths** (default)    | Bounded | Bounded |
| `'MemoizedDFS'`   | Complete runs (API)         | Fastest | Moderate |
| `'DP'`            | **Sparse graphs**          | Medium  | Lowest |
| `'DFS'`           | Standard traversal         | Medium  | Medium |
| `'Backtracking'`  | Extreme memory constraints | Slowest | Lowest |

> 📖 **See [PathFinding Methods](./PathFinding_Methods.md)** for detailed algorithm comparison.

### Common Patterns

**Find all paths between PN and MBON:**
```python
sourceNeurons = ['.*PN.*']
targetNeurons = ['MBON.*']
max_interlayer = 2
keyword_in_path_to_remove = ['None', 'APL']  # Exclude certain types
```

**Fast type-level analysis:**
```python
skip_bodyId = True              # Skip individual neuron analysis
filter_by = 'type'              # Filter at type level
output_format = 'csv'           # CSV is faster than Excel
```

**Deep circuit tracing:**
```python
max_interlayer = 4              # 4 intermediate layers
pathfinding = 'StrongestFirst'  # Built-in; strongest paths first with a reported tau
min_synapse_num = 3             # Filter weak connections
```

### Output Files

```
local_data/connection_data/
├── {date}_{source}_{target}_paths_type.csv        # Type-level paths
├── {date}_{source}_{target}_paths_bodyId.csv      # Individual paths (if not skipped)
├── {date}_{source}_{target}_sankey_type.html      # Flow diagram
├── {date}_{source}_{target}_network.html          # Interactive network
└── {date}_{source}_{target}_summary.xlsx          # Complete summary
```

---

## NeuronBridge_FindLines.py - Find Driver Lines

**Location**: [`scripts/NeuronBridge_FindLines.py`](../../scripts/NeuronBridge_FindLines.py)

Find **GAL4/Split-GAL4 driver lines** matching your EM neurons via NeuronBridge.

### Key Configuration

```python
# Query neurons
query = 'aMe12'                 # Type name, bodyId, or pattern
# query = ['aMe12', 'MBON01']   # Multiple types
# query = 636798093             # BodyId

# Dataset(s) to search
dataset = ['male-cns:v0.9', 'hemibrain:v1.2.1', 'flywire_FAFB_v783']
# dataset = None                # Search ALL datasets

# Match algorithm
match_type = 'cds'              # 'cds', 'pppm', or 'both'

# Output
output_dir = '../local_data/neuronbridge_finding'
verbose = True
use_cache = True

# Image download options
region = 'Brain'                           # 'Brain', 'VNC', or 'All'
download_img_for_top_n_lines = 20         # Download for top 20 lines
sort_by = 'max'                            # 'max' (by agg_mean_score) or 'completeness' (by weighted_score)
image_formats = ['png', 'jpg']
max_download_images_per_line = 6
flylight_category = ['GAL4/LEXA', 'SplitGAL4', 'MCFO']
simple_mode = True                         # Filter to essential images
separate_splitgal4 = True                  # Separate GAL4 from Split-GAL4

# Summary generation
summary_format = ['pdf', 'pptx']           # Generate PDF and PPTX summaries

# Run
finder = NeuronBridgeFinder(
    use_cache=use_cache,
    verbose=verbose,
    separate_splitgal4=separate_splitgal4,
    match_type=match_type,
    region=region,
)

results = finder.find_lines_batch(
    queries=query,
    dataset=dataset,
    sort_by=sort_by,
    output_dir=output_dir,
    download_img_for_top_n_lines=download_img_for_top_n_lines,
    image_formats=image_formats,
    max_download_images_per_line=max_download_images_per_line,
    flylight_category=flylight_category,
    simple_mode=simple_mode,
    summary_format=summary_format,
)
```

### Understanding the Scoring

**Two sorting modes:**

1. **`sort_by='max'`** (default) - Ranks by **average score**
   - Prioritizes lines with highest morphological similarity
   - Formula: Sort by `agg_mean_score`
   - Use when: Finding best morphological matches regardless of coverage

2. **`sort_by='completeness'`** - Ranks by **weighted score**
   - Prioritizes lines that label MORE of your queried neurons
   - Formula: `weighted_score = agg_mean_score × coverage_ratio`
   - Use when: Finding lines that label ALL queried neuron types

**Example:**
```
Query: 3 neuron types (aMe12, MBON01, KC)

sort_by='max':
  Line A: avg_score=50000, labels 1/3 types → ranks #1 (high score)
  Line B: avg_score=40000, labels 3/3 types → ranks #2

sort_by='completeness':
  Line B: weighted_score=40000×1.0=40000 → ranks #1 (labels all types)
  Line A: weighted_score=50000×0.33=16666 → ranks #2
```

> 📖 **See [NeuronBridge Guide](./NeuronBridge_Guide.md)** for complete documentation.
> 
> 📖 **See [Score Calculation Guide](./ScoreCalculation_Guide.md#neuronbridge-matching-scores)** for score formulas.

### Output Files

```
local_data/neuronbridge_finding/NB-find-lines_{query}_{date}/
├── line_summary.csv              # All lines ranked by weighted_score
├── gal4_lexa_summary.csv         # GAL4/LexA lines only
├── split_gal4_summary.csv        # Split-GAL4 lines only
├── {query}_lines.csv             # Per-query row-level matches
├── images/                       # Downloaded images
│   ├── SS01015/                  # One folder per line
│   └── VT037867/
├── images_summary.pdf            # PDF summary (pages ordered by rank)
└── images_summary.pptx           # PowerPoint summary (slides ordered by rank)
```

---

## FlyLight_fetcher.py - Download Images

**Location**: [`scripts/FlyLight_fetcher.py`](../../scripts/FlyLight_fetcher.py)

Download **FlyLight imagery** for specific driver lines.

### Key Configuration

```python
# Line name(s)
line_name = ['SS01015', 'VT037867']  # Single or multiple lines

# File formats
formats = ['jpg', 'png']             # 'png', 'jpg', 'h5j', 'lsm', 'mp4', 'json', 'all'

# Collection category
collection_category = ['GAL4/LexA', 'SplitGAL4']  # Which collections to search

# Region filter
region = 'Brain'                     # 'Brain', 'VNC', or 'All'

# Output
output_dir = '../local_data/flylight'
max_files = 6                        # Limit per line
max_workers = 4                      # Parallel downloads

# Summary generation
generate_summary = ['pdf', 'pptx']   # Generate image summaries
summary_images_per_page = (3, 2)     # Grid layout (columns, rows)
add_timestamp = True                 # Add timestamp to folder names

# Initialize downloader
downloader = FlyLightDownloader(verbose=True)

# Download
downloader.download(
    line_name=line_name,
    formats=formats,
    collection_category=collection_category,
    region=region,
    output_dir=output_dir,
    max_files=max_files,
    max_workers=max_workers,
    generate_summary=generate_summary,
    images_per_page=summary_images_per_page,
    add_timestamp=add_timestamp,
)
```

### Automatic Source Detection

The downloader automatically detects the line type and uses the appropriate source:

- **R-lines** (e.g., R10A06): S3 bucket + HTTP CDN
- **VT lines** (e.g., VT037867): HTTP CDN only
- **Split-GAL4** (e.g., SS00731): S3 bucket
- **Other Gen1** (e.g., GMR_*): S3 bucket

### Output Files

```
local_data/flylight/SS01015_{timestamp}/
├── SS01015-20x-GAL4-Brain-f-001.jpg
├── SS01015-20x-GAL4-Brain-f-002.jpg
├── SS01015-cdm-f.png
├── images_summary.pdf
└── images_summary.pptx
```

> 📖 **See [FlyLight Guide](./FlyLight_Guide.md)** for complete documentation.

---

## Common Parameters Reference

### Dataset Selection

| Dataset             | Type     | Description                 |
| ------------------- | -------- | --------------------------- |
| `hemibrain:v1.2.1`  | NeuPrint | Adult fly brain (central)   |
| `male-cns:v0.9`     | NeuPrint | Full male CNS (brain + VNC) |
| `optic-lobe:v1.1`   | NeuPrint | Optic lobe detailed         |
| `manc:v1.2.3`       | NeuPrint | Male ventral nerve cord     |
| `flywire_FAFB_v783` | Local    | FlyWire female adult brain  |
| `banc_v888` | Local    | Standalone BANC brain + nerve cord (public release, current) |
| `banc_v626` | Local    | Standalone BANC brain + nerve cord (public release, older) |

### Neuron Selection (Regex Support)

```python
# Exact match
sourceNeurons = ['MBON01']

# Regex patterns
sourceNeurons = ['KC.*']           # All Kenyon cells
sourceNeurons = ['.*PN.*']         # All projection neurons
sourceNeurons = ['L[1-5]']         # L1 through L5
sourceNeurons = ['aMe.*']          # All aMe neurons

# Multiple patterns
sourceNeurons = ['MBON01', 'MBON03', 'KC.*']
```

#### Search Priority

The system searches for neurons using the following priority order:

| Priority | Column        | Match Type                  | Example                          |
| -------- | ------------- | --------------------------- | -------------------------------- |
| 1        | **bodyId**    | Exact (numeric, int or str) | `[123456789]` or `['123456789']` |
| 2        | **type**      | Exact                       | `['MBON01']`                     |
| 3        | **type**      | Regex                       | `['KC.*']`                       |
| 4        | **instance**  | Exact                       | `['MBON01(R)']`                  |
| 5        | **instance**  | Regex                       | `['.*_R']`                       |
| 6        | **bodyId**    | Regex                       | `['720575.*']`                   |
| 7        | Other columns | Exact/Regex                 | fallback                         |

**Notes:**
- Numeric inputs (int or numeric strings like `'123456789'`) are matched against `bodyId` first
- Both `[123456789]` (int) and `['123456789']` (string) are supported for bodyId lookup
- Regex patterns (containing `*`, `.`, `^`, `$`, `[`) are matched via pattern
- The search returns all neurons matching the **first column that produces results**
- All comparisons are string-based internally for consistency

### Hemisphere Handling (FindNeuronConnection)

| Parameter              | Default | Description                                                                                                                                                  |
| ---------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `separate_hemispheres` | False   | When True, type and custom-group labels are suffixed with `_L/_R/_U` based on hemisphere annotations. All aggregation is split by hemisphere.                |
| `symmetry_analysis`    | True    | When True, generates hemisphere symmetry outputs (ipsilateral vs contralateral) under `hemisphere_symmetry/`. Auto-enabled when `separate_hemispheres=True`. |

**Behavior notes:**
- `_L`/`_R` suffixes in query patterns are treated as hemisphere filters.
- When hemisphere info is missing, `_U` is used.
- Hemisphere symmetry outputs are saved per-threshold in `hemisphere_symmetry/` under the run folder.

### Connection Filters

| Parameter                   | Description                            | Default    | Typical Values |
| --------------------------- | -------------------------------------- | ---------- | -------------- |
| `min_synapse_num`           | Minimum synapse count                  | 1          | 3-10           |
| `min_ratio`                 | Minimum connection ratio (weight/post) | 0.0        | 0.01-0.1       |
| `min_traversal_probability` | Minimum traversal probability          | 0.0        | 0.01-0.1       |
| `filter_by`                 | Filter level: `'bodyId'` or `'type'`   | `'bodyId'` | -              |

**Filter relationships:**
```python
# Convert between filters
min_traversal_probability = min_ratio / 0.3
min_ratio = min_traversal_probability * 0.3
```

> 📖 **See [Score Calculation Guide](./ScoreCalculation_Guide.md)** for detailed explanations of these metrics.

### Pathfinding Options

| Parameter                   | Description                    | Default         | Values            |
| --------------------------- | ------------------------------ | --------------- | ----------------- |
| `max_interlayer`            | Maximum intermediate layers    | 1 (API) / 2 (UI) | 1-6             |
| `pathfinding`               | Algorithm choice               | `'StrongestFirst'` | See table below   |
| `drop_untyped`              | Neuron-label filter: drop edges touching untyped neurons (empty / Unknown / NaN / bodyId-fallback labels) before the graph is built | `True`          | `False` to keep   |
| `show_top_n_paths`          | Limit output paths (-1 = all)  | -1              | 100-1000          |
| `keyword_in_path_to_remove` | Exclude paths with these types | `[]`            | `['None', 'APL']` |
| `skip_bodyId`               | Skip bodyId-level analysis     | `False`         | `True` for speed  |

**Filter distinctions** (these knobs bound different things):

- **Min Synapse Count** = the threshold.
- **Edge Budget** (`graph_edge_limit_bodyid`) = graph-level floor — 'all'
  mode only, exactly equivalent to raising the threshold (shortest mode is
  never floored).
- **Max Paths (BodyId)** (`max_paths_bodyid`) = path-output budget (keeps
  everything above a reported tau).
- **Drop Untyped Neurons** (`drop_untyped`) = neuron-label filter.
- **Visualization Edge Limit** (`edgeN_limit`) = drawing-only cap — never
  changes fetching, the graph, or path outputs.

**Algorithm Selection:**

| Algorithm          | Best For         | Parameter Value   |
| ------------------ | ---------------- | ----------------- |
| All depths (default) | StrongestFirst  | `'StrongestFirst'` |
| Deep paths (L≥5)   | StrongestFirst (bounded)   | `'StrongestFirst'`   |
| Sparse graphs      | Backward pruning | `'DP'`            |
| Memory constrained | Backtracking     | `'Backtracking'`  |

> 📖 **See [PathFinding Methods](./PathFinding_Methods.md)** for algorithm details.

### Performance Options

| Parameter             | Description                     | Impact          |
| --------------------- | ------------------------------- | --------------- |
| `use_cache=True`      | Cache API results locally       | Much faster repeat queries (local reads) |
| `skip_bodyId=True`    | Skip individual neuron analysis | 50% faster      |
| `output_format='csv'` | Use CSV instead of Excel        | 2-5x faster     |
| `filter_by='type'`    | Filter at type level            | Fewer edges     |

---

## Output Files

### FindDirect Output

```
local_data/connection_data/YYYYMMDD_HHMMSS_{source}_{target}/
├── direct_connections.csv           # Edge list with all metrics
├── connection_matrix_weight.csv     # Synapse count matrix
├── connection_matrix_ratio.csv      # Connection ratio matrix
├── network.html                     # Interactive network graph
├── heatmap.html                     # Connection heatmap
└── summary.xlsx                     # Excel with multiple sheets
```

### FindPath Output

```
local_data/connection_data/YYYYMMDD_HHMMSS_{source}_{target}/
├── paths_type.csv                   # Type-level paths with path_prob
├── paths_bodyId.csv                 # Individual neuron paths
├── sankey_type.html                 # Sankey flow diagram
├── network.html                     # Interactive network
├── heatmap_layer_0_1.html           # Layer 0→1 heatmap
├── heatmap_layer_1_2.html           # Layer 1→2 heatmap
└── summary.xlsx                     # Complete summary
```

### NeuronBridge Output

```
local_data/neuronbridge_finding/NB-find-lines_{query}_{timestamp}/
├── line_summary.csv                 # Ranked by weighted_score
├── gal4_lexa_summary.csv            # GAL4/LexA only
├── split_gal4_summary.csv           # Split-GAL4 only
├── {query}_lines.csv                # Per-query row-level matches
├── images/                          # Downloaded images
│   ├── SS01015/
│   │   ├── SS01015-20x-f-001.jpg
│   │   └── SS01015-cdm-f.png
│   └── VT037867/
├── images_summary.pdf               # PDF summary (ranked order)
└── images_summary.pptx              # PowerPoint summary (ranked order)
```

### CSV File Columns

**direct_connections.csv:**
- `bodyId_pre`, `type_pre`, `instance_pre`: Source neuron
- `bodyId_post`, `type_post`, `instance_post`: Target neuron
- `weight`: Synapse count
- `connection_ratio`: weight/post
- `traversal_probability`: min(1, ratio/0.3)
- `roi`: Region of interest

**paths_type.csv:**
- `path`: Type sequence (e.g., "PN→KC→MBON")
- `path_prob`: Product of edge probabilities
- `min_weight`: Bottleneck synapse count
- `min_ratio`: Weakest connection ratio
- `length`: Number of hops

**line_summary.csv:**
- `line`: Driver line name
- `agg_mean_score`: Average NeuronBridge score
- `match_count`: Number of unique queried neurons labeled
- `coverage_ratio`: match_count / total_query_neurons
- `weighted_score`: agg_mean_score × coverage_ratio
- `matched_bodyIds`: Comma-separated bodyIds
- `matched_types`: Comma-separated types

> 📖 **See [Output Files Reference](../OUTPUT_FILES.md)** for complete documentation.

---

## Script Customization Tips

### Running Scripts Programmatically

Instead of editing the script files, you can import and run them programmatically:

```python
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path.cwd().parent / 'src'))

from coana import FindNeuronConnection

# Your custom parameters
fc = FindNeuronConnection(
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['MBON01'],
    targetNeurons=['MBON03'],
    min_synapse_num=5,
)

fc.InitializeNeuronInfo()
fc.FindDirectConnections()
```

### Reading Neuron Lists from Files

```python
import pandas as pd

# From Excel
source_list = pd.read_excel('sourceNeurons.xlsx', header=None).iloc[:,0].tolist()

# From CSV
target_list = pd.read_csv('targetNeurons.csv', header=None).iloc[:,0].tolist()

fc = FindNeuronConnection(
    sourceNeurons=source_list,
    targetNeurons=target_list,
    custom_source_name='MySourceNeurons',  # Custom label for outputs
    custom_target_name='MyTargetNeurons',
)
```

### Batch Processing Multiple Queries

```python
queries = ['aMe12', 'MBON01', 'KC']

for query in queries:
    fc = FindNeuronConnection(
        sourceNeurons=[query],
        targetNeurons=['.*'],  # All neurons
        output_dir=f'./results/{query}',
    )
    fc.InitializeNeuronInfo()
    fc.FindDirectConnections()
```

---

## Next Steps

After mastering the basic scripts:

1. **[NeuronBridge Workflow](./NeuronBridge_Workflow.md)** - Complete EM→LM mapping workflow
2. **[Cross-Dataset Comparison](./CrossDatasetComparison_Guide.md)** - Compare circuits across datasets
3. **[3D Visualization](../visualizations/3D_Skeleton_Guide.md)** - Visualize neuron morphology
4. **[Score Calculations](./ScoreCalculation_Guide.md)** - Deep dive into all metrics
5. **[PathFinding Methods](./PathFinding_Methods.md)** - Algorithm selection guide

---

*All example scripts are in the [`scripts/`](../../scripts/FindPath.py) folder - copy and customize them for your analyses!* Reference

### Dataset Selection

| Dataset            | Description               | Example            |
| ------------------ | ------------------------- | ------------------ |
| `hemibrain:v1.2.1` | Adult fly brain (central) | Most commonly used |
| `male-cns:v0.9`    | Full male CNS             | Brain + VNC        |
| `optic-lobe:v1.1`  | Optic lobe detailed       | Visual system      |
| `manc:v1.2.3`      | Male VNC                  | Ventral nerve cord |

### Neuron Selection (Regex Support)

```python
# Exact match
sourceNeurons=['MBON01']

# Regex patterns
sourceNeurons=['KC.*']           # All Kenyon cells
sourceNeurons=['.*PN.*']         # All projection neurons
sourceNeurons=['L[1-5]']         # L1 through L5

# Multiple patterns
sourceNeurons=['MBON01', 'MBON03', 'KC.*']
```

> 📖 **Search Priority**: The system searches columns in order: `bodyId` (exact) → `type` (exact/regex) → `instance` (exact/regex) → other columns. See [Search Priority](#search-priority) section above for details.

### Connection Filters

| Parameter                   | Description                            | Default    |
| --------------------------- | -------------------------------------- | ---------- |
| `min_synapse_num`           | Minimum synapse count                  | 1          |
| `min_ratio`                 | Minimum connection ratio (weight/post) | 0.0        |
| `min_traversal_probability` | Minimum traversal probability          | 0.0        |
| `filter_by`                 | Filter level: `'bodyId'` or `'type'`   | `'bodyId'` |

> 📖 **See [Score Calculation Guide](./ScoreCalculation_Guide.md)** for detailed explanations of these metrics.

### Pathfinding Options

| Parameter                   | Description                                                 | Default         |
| --------------------------- | ----------------------------------------------------------- | --------------- |
| `max_interlayer`            | Maximum intermediate layers                                 | 1               |
| `pathfinding`               | Algorithm: `'StrongestFirst'` (built-in), `'MemoizedDFS'`, `'Bidirectional'`, `'DP'`, etc. | `'StrongestFirst'` |
| `drop_untyped`              | Drop edges touching untyped neurons (neuron-label filter)   | `True`          |
| `show_top_n_paths`          | Limit output paths (-1 = all)                               | -1              |
| `keyword_in_path_to_remove` | Exclude paths containing these types                        | `[]`            |

> 📖 **See [PathFinding Methods](./PathFinding_Methods.md)** for algorithm details.

---

## Output Files

### CSV Files

| File                     | Content                                   |
| ------------------------ | ----------------------------------------- |
| `direct_connections.csv` | Edge list with weight, ratio, probability |
| `paths_type.csv`         | Paths aggregated by neuron type           |
| `paths_bodyId.csv`       | Paths at individual neuron level          |

### Visualizations

| File             | Description                        |
| ---------------- | ---------------------------------- |
| `network.html`   | Interactive force-directed network |
| `sankey_*.html`  | Sankey flow diagrams               |
| `heatmap_*.html` | Connection strength heatmaps       |

### Excel Summary

The `summary.xlsx` file contains multiple sheets:
- **Connections**: All edges with metrics
- **Source_Neurons**: Source neuron info
- **Target_Neurons**: Target neuron info
- **Paths**: Path enumeration (for FindPath)

> 📖 **See [Output Files Reference](../OUTPUT_FILES.md)** for complete documentation.

---

## Next Steps

After mastering the basics:

1. **[NeuronBridge Integration](./NeuronBridge_Guide.md)** - Find driver lines for your neurons
2. **[Cross-Dataset Comparison](./CrossDatasetComparison_Guide.md)** - Compare across connectomes
3. **[3D Visualization](../visualizations/3D_Skeleton_Guide.md)** - Visualize neuron morphology
4. **[Score Calculations](./ScoreCalculation_Guide.md)** - Understand all metrics

---

*See the [examples/](../../archive/examples/) folder for complete working examples.*
