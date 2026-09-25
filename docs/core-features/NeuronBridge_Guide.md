# NeuronBridge Integration Guide

The NeuronBridge module provides programmatic access to the [NeuronBridge](https://neuronbridge.janelia.org/) database, enabling **bidirectional** mapping between electron microscopy (EM) reconstructions and light microscopy (LM) driver lines.

> 📖 **See also**: [NeuronBridge Workflow Guide](./NeuronBridge_Workflow.md) for step-by-step tutorials and decision trees.

## Table of Contents

- [Overview](#overview)
- [Two Directions: EM→LM and LM→EM](#two-directions-emlm-and-lmem)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [NeuronBridgeFinder Class](#neuronbridgefinder-class)
  - [Initialization Parameters](#initialization-parameters)
  - [Line Type Separation](#line-type-separation)
  - [Specificity and Selectivity Analysis](#specificity-and-selectivity-analysis)
  - [Co-Labeling Analysis](#co-labeling-analysis)
  - [Core Methods](#core-methods)
  - [Batch Processing Methods](#batch-processing-methods)
  - [Image Download Methods](#image-download-methods)
- [3D Skeleton Visualization](#3d-skeleton-visualization)
- [Match Types](#match-types)
- [Dataset Support](#dataset-support)
- [Examples](#examples)
- [Integration with FlyLight Downloader](#integration-with-flylight-downloader)
- [Troubleshooting](#troubleshooting)

---

## Overview

NeuronBridge is a web application that uses Color Depth MIP (CDM) search to find morphological matches between:
- **EM reconstructions** from hemibrain, male-cns, FlyWire FAFB, and other datasets
- **LM driver lines** (GAL4, LexA, Split-GAL4) from the FlyLight project

This module wraps the NeuronBridge API to provide:
- **EM → LM mapping**: Find driver lines matching a given EM body ID (`find_lines_batch`)
- **LM → EM mapping**: Find EM neurons matching a driver line name (`find_neurons_batch`)
- **Batch processing**: Process multiple queries with automatic result aggregation
- **Image downloads**: Download CDM images from NeuronBridge or full imagery from FlyLight

---

## Two Directions: EM→LM and LM→EM

| Direction   | Use Case                                       | Script                       | Key Method             |
| ----------- | ---------------------------------------------- | ---------------------------- | ---------------------- |
| **EM → LM** | "What driver lines label my EM neurons?"       | `NeuronBridge_FindLines.py`  | `find_lines_batch()`   |
| **LM → EM** | "What EM neurons does this driver line label?" | `NeuronBridge_FindNeuron.py` | `find_neurons_batch()` |

### Quick Comparison

| Aspect            | EM→LM (FindLines)   | LM→EM (FindNeuron)     |
| ----------------- | ------------------- | ---------------------- |
| **Input**         | Neuron type/bodyId  | Driver line name       |
| **Output**        | Ranked driver lines | Matched EM neurons     |
| **Key Metric**    | `weighted_score`    | `score` per neuron     |
| **Visualization** | FlyLight images     | 3D skeletons           |
| **Primary Use**   | Design experiments  | Validate line coverage |

---

## Installation

The NeuronBridge module uses DROCAT's bundled cross-platform client. Install the normal release requirements:

```bash
pip install -r requirements.txt
```

Do not install the upstream `neuronbridge-python` distribution; its Pydantic constraint conflicts with the v4.5 UI stack.

**Authentication:** NeuronBridge API is publicly accessible and requires **no authentication token**. 

However, if you want to use local dataset features (type lookups, enrichment with neuron metadata), you can provide a NeuPrint token. Note that accessing NeuPrint datasets directly (e.g., via `FindPath.py`) **requires** a NEUPRINT_TOKEN - it's only the NeuronBridge API itself that doesn't need authentication.

For FlyLight image downloads, optionally install boto3 for faster S3 access:

```bash
pip install boto3
```

---

## Quick Start

### Basic Usage

```python
from src.neuronbridge_finder import NeuronBridgeFinder

# Initialize the finder
nbf = NeuronBridgeFinder(verbose=True)

# Find driver lines matching an EM body ID
lines_df = nbf.id_to_lines(720575940621039145)
print(lines_df[['line', 'score', 'library']].head())

# Find EM neurons matching a driver line
neurons_df = nbf.line_to_neuron('LH173')
print(neurons_df[['bodyId', 'dataset', 'score']].head())
```

### Batch Processing with Image Download

```python
from src.neuronbridge_finder import NeuronBridgeFinder

nbf = NeuronBridgeFinder()

# Find lines for multiple neurons and download images
results = nbf.find_lines_batch(
    queries='MBON01,aMe12',
    dataset='hemibrain:v1.2.1',
    match_type='both',
    output_dir='./neuronbridge_results',
    download_images='flylight',
    download_top_n_img=10,
    simple_mode=True
)
```

---

## NeuronBridgeFinder Class

### Initialization Parameters

```python
@dataclass
class NeuronBridgeFinder:
    datasets_path: Optional[str] = None
    use_cache: bool = True
    cache_folder: Optional[str] = None
    verbose: bool = True
    separate_splitgal4: bool = False
    neuprint_token: Optional[str] = None
    neuprint_server: str = 'https://neuprint.janelia.org'
    match_type: str = 'cds'
    region: str = 'All'
    max_api_images_per_line: int = 10
```

| Parameter                 | Type            | Default                          | Description                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------- | --------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `datasets_path`           | `str` or `None` | Auto-detect                      | Path to the datasets folder containing neuron_df CSV files. Used for enriching results with local neuron metadata.                                                                                                                                                                                                                                                                                      |
| `use_cache`               | `bool`          | `True`                           | Whether to cache API results locally. Records are stored as compressed, NeuronBridge-versioned Parquet; legacy CSV records are ignored.                                                                                                                                                                                                                                                            |
| `cache_folder`            | `str` or `None` | Auto-detect                      | Folder for cached results. Default: `cache/neuronbridge/` in the project directory.                                                                                                                                                                                                                                                                                                                     |
| `verbose`                 | `bool`          | `True`                           | Print progress messages during operations.                                                                                                                                                                                                                                                                                                                                                              |
| `separate_splitgal4`      | `bool`          | `False`                          | **NEW**: If True, separate results into GAL4/LexA and Split-GAL4 categories. When enabled, `download_top_n_img` applies separately to each category (see [Line Type Separation](#line-type-separation)).                                                                                                                                                                                                |
| `neuprint_token`          | `str` or `None` | `None`                           | **OPTIONAL** (for NeuronBridge only): NeuPrint API token for local dataset enrichment features. Not required for basic NeuronBridge API calls. If not set, checks `NEUPRINT_TOKEN` environment variable. **Note:** While optional for NeuronBridge, this token is **required** for direct NeuPrint data acces and NeuronBridge data enrichment. Get your token at: https://neuprint.janelia.org/account |
| `neuprint_server`         | `str`           | `'https://neuprint.janelia.org'` | **NEW**: NeuPrint server URL.                                                                                                                                                                                                                                                                                                                                                                           |
| `match_type`              | `str`           | `'cds'`                          | **NEW**: Default match algorithm for all operations: `'cds'` (Color Depth MIP Search), `'pppm'` (Pattern Matching), or `'both'`. Can be overridden at method level.                                                                                                                                                                                                                                     |
| `region`                  | `str`           | `'All'`                          | **NEW**: Filter images by anatomical region: `'Brain'`, `'VNC'`, or `'All'`. Filters out images from non-matching regions to reduce processing time and improve specificity.                                                                                                                                                                                                                            |
| `max_api_images_per_line` | `int`           | `10`                             | **NEW**: Maximum LM images to process per driver line for API calls. Use `-1` for unlimited. Images are pre-filtered by match_type availability before limiting.                                                                                                                                                                                                                                        |

**Setting up NeuPrint Token** (optional - for enhanced features only):

```python
# Option 1: Pass token directly (optional)
nbf = NeuronBridgeFinder(neuprint_token='your_token_here')

# Option 2: Set environment variable (optional, recommended if needed)
# In terminal: export NEUPRINT_TOKEN="your_token_here"
nbf = NeuronBridgeFinder()

# Option 3: No token - basic NeuronBridge features work fine without it
nbf = NeuronBridgeFinder()
```

**Important:** The NeuPrint token is optional for NeuronBridge API calls (finding lines, finding neurons, image downloads), but **required** for direct NeuPrint dataset access (e.g., pathfinding with `FindPath.py` or `FindDirect.py`). The token enables local dataset enrichment features in NeuronBridge, but the NeuronBridge API itself works without authentication.

**Region Filtering** (filter images by anatomical area):

```python
# Filter for Brain region only (exclude VNC images)
nbf = NeuronBridgeFinder(region='Brain')

# Filter for VNC region only (exclude Brain images)
nbf = NeuronBridgeFinder(region='VNC')

# Process all regions (default)
nbf = NeuronBridgeFinder(region='All')

# Combine with match_type for complete configuration
nbf = NeuronBridgeFinder(region='Brain', match_type='both')
```

**Image Limit Configuration** (control API calls per line):

```python
# Process only top 10 images per line (default, recommended)
nbf = NeuronBridgeFinder(max_api_images_per_line=10)

# Process more images for comprehensive results (slower)
nbf = NeuronBridgeFinder(max_api_images_per_line=50)

# Process all available images (slowest, for exhaustive searches)
nbf = NeuronBridgeFinder(max_api_images_per_line=-1)

# Combine settings for optimized searching
nbf = NeuronBridgeFinder(
    region='Brain',
    match_type='cds',
    max_api_images_per_line=20
)
```

**Match Type Configuration** (set default algorithm):

```python
# Use CDS matching by default (faster, color-based)
nbf = NeuronBridgeFinder(match_type='cds')  # Default

# Use PPPM matching by default (pattern-based)
nbf = NeuronBridgeFinder(match_type='pppm')

# Use both algorithms by default (combines results with rank-based scoring)
nbf = NeuronBridgeFinder(match_type='both')

# Override at method level if needed
results = nbf.find_neurons_batch('VT037867', match_type='cds')  # Override
```

### Line Type Separation

When `separate_splitgal4=True`, driver lines are classified into two categories based on their name prefixes:

**GAL4/LexA lines** (traditional enhancer trap lines):
- VT lines (Vienna Tile): `VT037867`, `VT001234`
- R lines (Rubin lab): `R10A06`, `R65E11`
- GMR lines: `GMR_01A01`

**Split-GAL4 lines** (intersectional genetic approach):
- SS lines (Split Screen): `SS01015`, `SS00734`
- LH lines (Lateral Horn): `LH173`, `LH1234`
- MB lines (Mushroom Body): `MB011B`, `MB543`
- IS lines: `IS49879`, `IS83928`
- OL lines (Optic Lobe): `OL0042B`
- LC lines (Lobula Columnar): `LC16`
- And others: `LLPC`, `LPC`, `JRC_SS`, `BJD_SS`

**Usage Example**:
```python
# Enable separate mode on instance
nbf = NeuronBridgeFinder(separate_splitgal4=True)

# With download_top_n_img=5, downloads:
# - Top 5 GAL4/LexA lines
# - Top 5 Split-GAL4 lines
# = 10 lines total
results = nbf.find_lines_batch(
    queries='MBON01',
    download_images='flylight',
    download_top_n_img=5,
    output_dir='./output'
)

# Results include 'line_type' column
print(results[results['line_type'] == 'gal4_lexa']['line'].unique()[:5])
print(results[results['line_type'] == 'split_gal4']['line'].unique()[:5])
```

**Separate Output Files** (when `separate_splitgal4=True`):
```
output/NB-find-lines_{query}_{timestamp}/
├── {query}_lines.csv          # Per-query results (row-level matches)
├── line_summary.csv           # Aggregated stats (all lines)
├── gal4_lexa_summary.csv      # GAL4/LexA summary
├── split_gal4_summary.csv     # Split-GAL4 summary
├── parameters.json
└── images/                    # (when image download is enabled)
```

---

### Specificity and Selectivity Analysis

For detailed analysis of how specific each driver line is to your target neuron types, use the **Co-Labeling Analysis** feature separately via `NeuronBridge_Colabel.py` or the `analyze_colabeling()` method.

**Why Separate Analysis?**
- Finding driver lines (`find_lines_batch`) focuses on discovering which lines match your neurons
- Specificity/selectivity analysis requires querying each line's labeled neurons (API-intensive)
- Separating these workflows gives you more control and faster results for initial searches

**Recommended Workflow:**
1. **Find Lines First**: Use `NeuronBridge_FindLines.py` to discover matching driver lines
2. **Analyze Specificity**: Use `NeuronBridge_Colabel.py` with your top candidate lines for detailed analysis

See [Co-Labeling Analysis](#co-labeling-analysis) section below for complete documentation.

---

### Co-Labeling Analysis

Analyze co-labeling patterns among driver lines to understand how they overlap in their neuron labeling patterns, and assess specificity and selectivity.

**Script**: `scripts/NeuronBridge_Colabel.py`

```python
from src.neuronbridge_finder import NeuronBridgeFinder

nbf = NeuronBridgeFinder()

# Analyze co-labeling between multiple lines
results = nbf.analyze_colabeling(
    lines=['LH173', 'VT037867', 'SS00731', 'R10A06'],
    match_type='cds',
    top_n_neurons=100,
    similarity_methods=['jaccard', 'weighted_jaccard'],
    output_dir='./colabel_analysis',
    generate_report=True,
    min_score=30000,
    min_type_avg_score=30000
)
```

**Parameters**:

| Parameter            | Type            | Default                           | Description                                                                              |
| -------------------- | --------------- | --------------------------------- | ---------------------------------------------------------------------------------------- |
| `lines`              | `str` or `list` | Required                          | Driver lines to analyze (at least 2 required)                                            |
| `match_type`         | `str`           | `'cds'`                           | Match algorithm for neuron lookup                                                        |
| `top_n_neurons`      | `int`           | `50`                              | Top N ranked neurons retrieved per line                                                   |
| `min_score`          | `float`         | `30000.0`                         | Filters expression/labeling matrices after top-N retrieval; raw top-N files are retained |
| `min_type_avg_score` | `float`         | `10000.0`                         | Additional type filtering for similarity matrices only                                  |
| `similarity_methods` | `list`          | `['jaccard', 'weighted_jaccard']` | Similarity methods for co-labeling                                                       |
| `output_dir`         | `str`           | `None`                            | Output directory for results                                                             |
| `generate_report`    | `bool`          | `True`                            | Generate HTML analysis report                                                            |
| `visualize`          | `bool`          | `True`                            | Generate heatmap visualizations                                                          |

**Similarity Methods**:
- `'jaccard'`: Binary Jaccard similarity (|A ∩ B| / |A ∪ B|) based on type presence/absence
- `'weighted_jaccard'`: Score-weighted Jaccard that accounts for match confidence
- `'rank_correlation'`: Spearman correlation of type rankings based on scores

**Returns Dictionary**:
```python
{
    'expression_matrix': pd.DataFrame,      # Type × Line score matrix
    'colabeling_matrices': {                # Similarity matrices per method
        'jaccard': pd.DataFrame,
        'weighted_jaccard': pd.DataFrame
    },
    'line_neurons': {                       # Per-line neuron details
        'LH173': pd.DataFrame,
        'VT037867': pd.DataFrame,
        ...
    },
    'line_summary': pd.DataFrame,           # Summary statistics per line
    'report_path': str                      # Path to HTML report
}
```

**Line Summary Columns**:

| Column             | Description                                                       |
| ------------------ | ----------------------------------------------------------------- |
| `line`             | Driver line name                                                  |
| `n_neurons`        | Total number of neurons matched                                   |
| `n_types`          | Number of unique types labeled                                    |
| `mean_score`       | Mean NeuronBridge match score                                     |
| `max_score`        | Maximum match score (50000 = perfect match)                       |
| `n_neurons_HMS`    | Neurons above Half Max Score (score ≥ max_score/2)                |
| `n_types_HMS`      | Unique types above Half Max Score                                 |
| `n_neurons_MS`     | Neurons at Max Score (within 0.1% tolerance)                      |
| `n_types_MS`       | Unique types at Max Score                                         |
| `Qf`               | Quality Factor = max_score / n_types_HMS. Higher = more selective |
| `colabel_sparsity` | Uniqueness of labeling pattern (0-1, higher = more unique)        |

**Quality Metrics Explained**:
- **Half Max Score (HMS)** metrics show how many neurons/types have high-confidence matches (score ≥ 50% of max)
- **Max Score (MS)** metrics show how many neurons/types achieve the best possible match
- **Qf (Quality Factor)** indicates line selectivity: a high Qf means high match quality concentrated in few types (more selective), while a low Qf indicates the line labels many types at high confidence (less selective)

**Output Files**:
```
output/NB-colabeling_LH173_VT037867_SS00731_etc_20241223_123456/
├── expression_matrix.csv               # Type × Line score matrix (dataset-prefixed types)
├── expression_matrix.html              # Interactive heatmap (per-dataset)
├── expression_matrix_merged.csv        # Type × Line matrix (types merged across datasets)
├── expression_matrix_merged.html       # Interactive heatmap (merged types)
├── colabeling_matrix_jaccard.csv       # Binary similarity matrix
├── colabeling_matrix_jaccard.html      # Interactive heatmap
├── colabeling_matrix_weighted_jaccard.csv   # Weighted similarity
├── colabeling_matrix_weighted_jaccard.html  # Interactive heatmap
├── line_summary.csv                    # Summary statistics
├── colabeling_report.html              # Comprehensive HTML report
├── labeling_distribution_by_type.html  # Score distribution by type
├── labeling_distribution_by_neuron.html # Score distribution by neuron
└── line_labeled_neurons/               # Per-line neuron details
    ├── LH173_neurons.csv
    ├── VT037867_neurons.csv
    └── ...
```

**Expression Matrix Variants**:
- **Original** (`expression_matrix.*`): Types prefixed with dataset (e.g., `MCNS_aMe12`, `FAFB_aMe12`)
- **Merged** (`expression_matrix_merged.*`): Types combined across datasets (e.g., `aMe12` = max of MCNS + FAFB)

**Use Cases**:
1. **Find complementary lines**: Lines with low co-labeling similarity label different neuron populations
2. **Identify redundant lines**: Lines with high similarity label similar neurons
3. **Design experiments**: Choose lines that together cover your target neurons without redundancy
4. **Assess specificity**: Lines with high sparsity label unique neurons

**Script**: `scripts/NeuronBridge_Colabel.py`

---

### Core Methods

#### `id_to_lines(body_id, match_type='cds', expected_dataset=None)`

Find driver lines matching a given EM body ID.

```python
lines_df = nbf.id_to_lines(
    body_id=720575940621039145,
    match_type='cds',
    expected_dataset='hemibrain:v1.2.1'
)
```

| Parameter          | Type            | Default  | Description                                                                                                                      |
| ------------------ | --------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `body_id`          | `int`           | Required | The EM body ID to search for.                                                                                                    |
| `match_type`       | `str`           | `'cds'`  | Match algorithm: `'cds'` (Color Depth Search), `'pppm'` (PatchPerPixMatch), or `'both'`.                                         |
| `expected_dataset` | `str` or `None` | `None`   | Expected dataset name (e.g., `'male-cns:v0.9'`). Filters results to match this dataset when body ID exists in multiple datasets. |

**Returns**: `pd.DataFrame` with columns:
- `line`: Driver line name (e.g., 'LH173', 'SS01015')
- `library`: Source library (e.g., 'FlyLight Gen1 MCFO')
- `score`: Match score (higher is better)
- `image_id`: NeuronBridge image identifier
- `match_type`: Algorithm used ('cds' or 'pppm')
- For `match_type='both'`: Additional columns `cds_score`, `pppm_score`, `cds_rank`, `pppm_rank`, `combined_rank`

---

#### `line_to_neuron(line_name, match_type='cds')`

Find EM neurons matching a driver line name.

```python
neurons_df = nbf.line_to_neuron(
    line_name='LH173',
    match_type='cds'
)
```

| Parameter    | Type  | Default  | Description                                                     |
| ------------ | ----- | -------- | --------------------------------------------------------------- |
| `line_name`  | `str` | Required | The driver line name to search for (e.g., 'LH173', 'VT037867'). |
| `match_type` | `str` | `'cds'`  | Match algorithm: `'cds'`, `'pppm'`, or `'both'`.                |
| `top_n`      | `int` | `-1`     | Maximum matches to return. `-1` for all.                        |

**Returns**: `pd.DataFrame` with columns:
- `bodyId`: EM body ID
- `dataset`: Dataset name (e.g., 'hemibrain:v1.2.1')
- `instance`: Neuron instance name
- `type`: Neuron type
- `status`: Reconstruction status
- `score`: Match score
- `image_id`: NeuronBridge image identifier
- `lm_sample`: LM sample identifier
- `match_type`: Algorithm used
- `library`: Source library

---

#### `neuron_to_lines(query, dataset=None, match_type='cds')`

Find driver lines matching neurons specified by bodyId, instance, or type.

```python
# Search by neuron type
results = nbf.neuron_to_lines(
    query='MBON01',
    dataset='hemibrain:v1.2.1',
    match_type='both'
)

# Search by body ID
results = nbf.neuron_to_lines(query=720575940621039145)

# Search multiple datasets
results = nbf.neuron_to_lines(
    query='aMe12',
    dataset=['hemibrain:v1.2.1', 'flywire_FAFB_v783']
)
```

| Parameter    | Type                     | Default  | Description                                                                                                 |
| ------------ | ------------------------ | -------- | ----------------------------------------------------------------------------------------------------------- |
| `query`      | `str`, `int`, or `list`  | Required | Query to search: body ID (int), neuron type/instance (str), or list of queries. Supports regex patterns.    |
| `dataset`    | `str`, `list`, or `None` | `None`   | Dataset(s) to search. Can be a single string, list of datasets, or `None` to search all available datasets. |
| `match_type` | `str`                    | `'cds'`  | Match algorithm: `'cds'`, `'pppm'`, or `'both'`.                                                            |

**Returns**: `Dict[str, pd.DataFrame]` mapping body IDs to DataFrames of matching lines.

---

### Batch Processing Methods

#### `find_lines_batch(...)`

Find driver lines for multiple EM neurons with automatic saving and optional image download.

```python
results = nbf.find_lines_batch(
    queries='MBON01,aMe12,720575940621039145',
    dataset='hemibrain:v1.2.1',
    match_type='both',
    output_dir='./output',
    download_images='flylight',
    download_img_for_top_n_lines=20,
    image_formats=['png', 'jpg'],
    image_types='all',
    max_download_images_per_line=10,
    flylight_category=['GAL4/LEXA', 'SplitGAL4'],
    organize_by_region=False,
    simple_mode=True,
    pdf_images_per_page=(5, 3),
    pdf_landscape=True
)
```

| Parameter                      | Type                     | Default                      | Description                                                                                   |
| ------------------------------ | ------------------------ | ---------------------------- | --------------------------------------------------------------------------------------------- |
| `queries`                      | `str`, `int`, or `list`  | Required                     | Neuron query(s). Can be comma-separated string, body ID, or list.                             |
| `dataset`                      | `str`, `list`, or `None` | `None`                       | Dataset(s) to search. Use list for multiple datasets.                                         |
| `match_type`                   | `str` or `None`          | `None`                       | Match algorithm: `'cds'`, `'pppm'`, or `'both'`. If `None`, uses instance-level `match_type`. |
| `output_dir`                   | `str` or `None`          | `None`                       | Directory to save results. Creates timestamped subfolder.                                     |
| `download_images`              | `str` or `None`          | `'flylight'`                 | Image source: `'neuronbridge'`, `'flylight'`, `'both'`, or `None`.                            |
| `download_img_for_top_n_lines` | `int` or `None`          | `10`                         | Download images for top N lines only (by aggregate score/rank).                               |
| `image_formats`                | `str` or `list`          | `['png', 'jpg']`             | File formats to download.                                                                     |
| `image_types`                  | `str` or `list`          | `'all'`                      | Image types: `'mip'`, `'cdm'`, `'aligned'`, `'all'`, etc.                                     |
| `max_download_images_per_line` | `int` or `None`          | `20`                         | Maximum images to download per line.                                                          |
| `flylight_category`            | `str` or `list`          | `['GAL4/LEXA', 'SplitGAL4']` | FlyLight collection category. Missing lines fall back to MCFO, then RawImages, then NeuronBridge. |
| `organize_by_region`           | `bool`                   | `False`                      | Organize images into Brain/VNC subfolders.                                                    |
| `simple_mode`                  | `bool`                   | `False`                      | Apply filename filtering to reduce download volume (see [Simple Mode](#simple-mode)).         |
| `pdf_images_per_page`          | `tuple`                  | `(5, 3)`                     | (columns, rows) - images per page in PDF summary.                                             |
| `pdf_landscape`                | `bool`                   | `True`                       | Use landscape orientation for PDF.                                                            |

**Returns**: `pd.DataFrame` with combined results including:
- All columns from `id_to_lines()`
- `source_query`: Original query string
- `source_bodyId`: Matching body ID
- `source_dataset`: Source dataset
- `matched_bodyIds`: Comma-separated list of all body IDs matching each line

**Output Files** (when `output_dir` is specified):
```
output/NB-find-lines_aMe12_20241223_123456/
├── aMe12_lines.csv            # Individual query results (row-level matches)
├── line_summary.csv           # Aggregated stats per line, SORTED BY weighted_score
├── gal4_lexa_summary.csv      # GAL4/LexA summary, SORTED BY weighted_score
├── split_gal4_summary.csv     # Split-GAL4 summary, SORTED BY weighted_score
├── parameters.json            # Analysis parameters
├── images_summary.pdf         # PDF summary (pages ordered by weighted_score ranking)
└── images/                    # Downloaded images
    └── {line_name}/
        └── *.png, *.jpg
```

> The combined row-level tables (`all_lines.csv`, `gal4_lexa_lines.csv`,
> `split_gal4_lines.csv`) and the type-level exports (`{query}_types.csv`,
> `top_types_heatmap.png`) are no longer written — they were extremely large
> for whole-GAL4-line queries. `line_summary.csv` is the main output; the
> per-query `*_lines.csv` files retain every row-level match.

**Line Summary Columns** (`*_summary.csv`):

Summary files are sorted by `weighted_score` descending, prioritizing lines that label MORE of the queried neurons:

| Column                  | Description                                                             |
| ----------------------- | ----------------------------------------------------------------------- |
| `line`                  | Driver line name (e.g., VT000770, SS00001)                              |
| `agg_mean_score`        | Average NeuronBridge match score across all matched neurons             |
| `agg_max_score`         | Maximum NeuronBridge match score for this line                          |
| `match_count`           | Number of UNIQUE bodyIds labeled by this line                           |
| `matched_bodyIds`       | Comma-separated list of unique bodyIds                                  |
| `matched_types`         | Comma-separated list of unique neuron types labeled                     |
| `coverage_ratio`        | match_count / total_query_neurons (fraction of queried neurons labeled) |
| `weighted_score`        | **agg_mean_score × coverage_ratio** (PRIMARY SORTING KEY)               |
| `datasets_labeled`      | Number of datasets where this line labels queried neurons               |
| `matched_datasets`      | Comma-separated list of matched dataset names                           |
| `min_score_per_dataset` | Minimum of max scores across datasets                                   |
| `cross_dataset_score`   | Mean of max scores across datasets                                      |
| `line_type`             | 'gal4_lexa' or 'split_gal4' (when separate_splitgal4=True)              |

**Weighted Score Calculation**:

```
weighted_score = agg_mean_score × (match_count / total_query_neurons)
```

This scoring prioritizes lines that:
1. Have high average matching scores (good morphological match)
2. Label MORE of the queried neurons (high coverage)

**Example**: When querying 'aMe12' across 3 datasets with 15 total neurons:
- Line A: agg_mean_score=45000, match_count=15 → weighted_score=45000×(15/15)=45000
- Line B: agg_mean_score=50000, match_count=2 → weighted_score=50000×(2/15)=6666

Line A ranks higher because it labels ALL queried neurons, even though Line B has a higher raw score.

**Multi-Type Query Behavior**:

When querying multiple types together (e.g., 'aMe12,MBON01'), the program finds lines that label ALL queried neuron types. The weighted_score ensures lines labeling more types rank higher.

⚠️ **IMPORTANT**: If you want to find lines labeling DIFFERENT groups separately, DO NOT query them together. Run separate queries instead:
- Query 1: 'aMe12' → finds best lines for aMe12
- Query 2: 'MBON01' → finds best lines for MBON01

Querying 'aMe12,MBON01' together finds lines labeling BOTH types.

**💡 Tip**: For specificity/selectivity analysis of found lines, use `NeuronBridge_Colabel.py` with your top candidate lines. See [Co-Labeling Analysis](#co-labeling-analysis) section.

**Type-level summaries**: the standalone `{query}_types.csv` /
`top_types_heatmap.png` exports were retired with the row-level combined
tables. Type-level aggregation now lives in the Find EM Neurons mode
(`{line}_{dataset_folder}_types.csv`) and the Co-Labeling expression
matrices; `line_summary.csv` carries the per-line `matched_types` column
for Find Lines runs.

---

#### Coverage-Routed, Name-Expanded Find Lines (v4.5.0+)

The Find Driver Lines tab (and `src/neuronbridge_query_expansion.py`) wraps
`find_lines_batch` with two layers that fix a real gap: **names resolve only
in local dataset tables, and NeuronBridge hosts EM bodies only for a subset
of DROCAT's datasets, at specific releases.** Before v4.5.0 a `male-cns:v1.0`
query (the default dataset!) returned zero rows silently — the same bodyIds
are hosted only under male-cns **v0.9** libraries, and BANC / optic-lobe are
not hosted at all.

**Coverage service** (`src/neuronbridge_coverage.py`): samples a few typed
bodyIds per dataset from the local tables, probes NeuronBridge's
`metadata/by_body/{id}.json`, and classifies each dataset as `exact`,
`aligned` (hosted at another release: `male-cns:v1.0` → `v0.9`,
`manc:v1.2.3` → `v1.2.1`), `unavailable` (BANC, optic-lobe), or `unknown`
(offline / no local table — never disables anything). The snapshot persists
at `cache/neuronbridge/coverage_snapshot.json` and refreshes automatically
when NeuronBridge's data version changes (or after 7 days). Disabling
unavailable datasets in the UI is advisory: runs targeting them warn and
proceed.

**Expansion orchestrator** (`ExpandedLineFinder.run(...)`): per query chip,
resolve cross-dataset equivalents with the row-based type resolver
(`resolve_valid_targets` → `expansion_targets` — splits expand to all branch
names, conflicts to none, unmapped keeps the raw name), then run one
`find_lines_batch` per chip at the hosted releases. The male-cns v1.0 → v0.9
alignment needs no extra code: it IS the mapper's curated shared-name alias.

```python
from neuronbridge_query_expansion import ExpandedLineFinder

finder = ExpandedLineFinder(verbose=True, max_workers=8)
result = finder.run(
    queries=["DNp01"],                       # male-cns v1.0 name → 3888 rows via v0.9
    dataset=["male-cns:v1.0", "banc_v626"],  # user selection (BANC unavailable → warning)
    expand_names=True,
    coverage_datasets=["male-cns:v1.0", "banc_v626"],
    match_type="cds", download_images=None, summary_format=None,
    output_dir="./output",
)
```

**Run folder** (`NB-find-lines-expanded_*`): flat for single-chip runs; one
`chip_{query}/` folder per chip for multi-chip runs — plus root reports:
`expansion_map.csv` (chip → expanded name → hosted release → mapping
status), `expansion_summary.json` (original selection + coverage routing +
cleanup audit), and `user_warning_notes.txt` (coverage warnings carry a
`coverage:` prefix). See [Output Files](../OUTPUT_FILES.md) §9a-bis.

**Output detail — Full vs Compact:** Full (default) keeps every file.
Compact keeps the newest N expanded runs' match tables (default N=1 —
the latest query stays inspectable), prunes older Compact runs' tables,
and removes the downloaded images once the PDF/PPTX exists — audited in
`cleanup_audit.json`. A zero-byte summary/PDF (a crashed writer) never
licenses deletion, and a run whose output folder is not on disk prunes nothing —
the pass returns an empty audit instead of creating a folder. The NeuronBridge match cache is OFF by
default (`use_cache=False`; Settings → NeuronBridge Match Cache re-enables
it) because NB queries are large and rarely reused — so a removed match
table regenerates only by re-running the query. The same knob exists on
Find EM Neurons (drops `all_neurons.csv` / `{line}_neurons.csv` /
`by_dataset/*_neurons.csv` after the type summaries are written) and
Co-Labeling (drops `line_labeled_neurons/` and
`distribution_data_by_neuron.csv` after the matrices/report are written).

---

#### `find_neurons_batch(line_names, top_n=50, match_type=None, output_dir=None, min_score=30000)`

Find EM neurons for multiple driver lines.

```python
results = nbf.find_neurons_batch(
    line_names='LH173,VT037867,SS01015',
    top_n=50,
    match_type='both',
    output_dir='./output'
)
```

| Parameter    | Type            | Default  | Description                                                   |
| ------------ | --------------- | -------- | ------------------------------------------------------------- |
| `line_names` | `str` or `list` | Required | Driver line name(s). Can be comma-separated.                  |
| `top_n`      | `int`           | `50`     | Maximum ranked matches per line. The score cutoff does not reduce this list. |
| `min_score`  | `float`         | `30000`  | Score cutoff used for score-based annotations; raw top-N matches are retained. |
| `match_type` | `str` or `None` | `None`   | Match algorithm. If `None`, uses instance-level `match_type`. |
| `output_dir` | `str` or `None` | `None`   | Directory to save results.                                    |

---

### Image Download Methods

#### `download_line_images(line_names, output_dir, source='neuronbridge', ...)`

Download images for driver lines from NeuronBridge or FlyLight.

```python
files = nbf.download_line_images(
    line_names='LH173,SS01015',
    output_dir='./images',
    source='flylight',
    formats=['png', 'jpg'],
    image_types='mip',
    max_files=10
)
```

| Parameter     | Type            | Default          | Description                                     |
| ------------- | --------------- | ---------------- | ----------------------------------------------- |
| `line_names`  | `str` or `list` | Required         | Driver line name(s).                            |
| `output_dir`  | `str`           | Required         | Directory to save images.                       |
| `source`      | `str`           | `'neuronbridge'` | Image source: `'neuronbridge'` or `'flylight'`. |
| `formats`     | `str` or `list` | `'png'`          | File formats.                                   |
| `image_types` | `str` or `list` | `'cdm'`          | Image types.                                    |
| `max_files`   | `int` or `None` | `None`           | Maximum files per line.                         |

---

## 3D Skeleton Visualization

The `find_neurons_batch()` method can generate interactive 3D skeleton visualizations of matched EM neurons.

### Basic Usage

```python
results = nbf.find_neurons_batch(
    queries='LH173,SS01015',
    match_type='cds',
    output_dir='./output',
    visualize_top_n=30,           # Visualize top 30 types per dataset
    visualize_by='bodyId',        # Group by individual neurons
    generate_individual_profiles=True  # Create per-neuron PNGs + PDF
)
```

### Visualization Parameters

| Parameter                      | Type       | Default   | Description                                              |
| ------------------------------ | ---------- | --------- | -------------------------------------------------------- |
| `visualize_top_n`              | `int`      | `0`       | Number of top types/bodyIds to visualize (0 = disabled). |
| `visualize_by`                 | `str`      | `'type'`  | Grouping mode: `'type'` or `'bodyId'`.                   |
| `generate_individual_profiles` | `bool`     | `False`   | Generate per-neuron PNG profiles and PDF summary.        |
| `pdf_images_per_page`          | `tuple`    | `(3, 2)`  | PDF layout as (columns, rows).                           |
| `background_color`             | `str`      | `'white'` | Background color: 'white', 'black', or CSS color.        |
| `type_filter`                  | `dict`     | `None`    | Filter types by name pattern (see below).                |
| `datasets_to_visualize`        | `str/list` | `'all'`   | Constrain which datasets to visualize.                   |

### Type Filtering

Filter which neuron types to visualize by name pattern. The filter gets ALL types first, applies the filter, then takes top N from filtered results. **Original ranks are preserved** in the output labels.

```python
# Filter types containing 'DN'
results = nbf.find_neurons_batch(
    'SS29633',
    visualize_top_n=12,
    type_filter={'contains': 'DN'}
)

# Filter types starting with 'AN' or 'DN'
results = nbf.find_neurons_batch(
    'SS29633',
    visualize_top_n=12,
    type_filter={'startswith': ['AN', 'DN']}  # OR logic within key
)

# Combine multiple filters (AND logic across keys)
results = nbf.find_neurons_batch(
    'SS29633',
    visualize_top_n=12,
    type_filter={'contains': 'B', 'startswith': 'AN'}  # Must match BOTH
)

# Use regex for complex patterns
results = nbf.find_neurons_batch(
    'SS29633',
    visualize_top_n=12,
    type_filter={'regex': r'^[AD]N\d+'}
)
```

**Filter Types:**
| Key          | Description                   | Example                     |
| ------------ | ----------------------------- | --------------------------- |
| `contains`   | Type name contains pattern    | `{'contains': 'DN'}`        |
| `startswith` | Type name starts with pattern | `{'startswith': 'AN'}`      |
| `endswith`   | Type name ends with pattern   | `{'endswith': '_R'}`        |
| `regex`      | Match regex pattern           | `{'regex': r'^DN[a-z]\d+'}` |

**Filter Logic:**
- Multiple values within same key: **OR** (match any)
- Multiple keys: **AND** (must match all)

### Dataset Filtering

Constrain which datasets to visualize:

```python
# Visualize only hemibrain and MANC
results = nbf.find_neurons_batch(
    'LH173',
    visualize_top_n=20,
    datasets_to_visualize=['hemibrain:v1.2.1', 'manc:v1.0']
)

# Visualize only male-cns
results = nbf.find_neurons_batch(
    'SS29633',
    visualize_top_n=12,
    datasets_to_visualize='male-cns:v0.9'
)
```

### Grouping Modes

**`visualize_by='type'`** (default):
- Neurons are merged by type (shows combined morphology)
- Layer labels: Type names (e.g., `MBON01`, `aMe12`)
- Good for comparing overall type morphologies

**`visualize_by='bodyId'`**:
- Individual neurons shown separately, grouped by type
- Layer labels: `r{rank}_{type}_x{N}` format
  - `rank`: Type ranking (1 = highest average score)
  - `type`: Neuron type name
  - `N`: Number of neurons of this type
  - Example: `r1_MBON01_x5` (rank 1 type with 5 neurons)
- Neurons sorted by actual rank, not alphabetically

### Output Files

```
output/findneurons_20241223_123456/
├── {line}_neurons.csv         # Matched neurons for each line
├── all_neurons.csv            # Combined results
├── by_dataset/                # Per-dataset matches + type summaries
└── plot-3d_{dataset}/          # Per-dataset visualization folder
    ├── {dataset}.html         # Interactive 3D skeleton viewer
    ├── parameters.txt         # Visualization settings record
    ├── exported_views/        # Static PNG exports
    │   ├── {dataset}_front.png
    │   ├── {dataset}_back.png
    │   ├── {dataset}_top.png
    │   ├── {dataset}_bottom.png
    │   ├── {dataset}_left.png
    │   └── {dataset}_right.png
    └── individual_profiles/   # Per-neuron profiles (if enabled)
        ├── r{rank}_{type}_x{N}.png  # Individual neuron images
        └── profile_summary.pdf      # Combined PDF with all neurons
```

### Visualization Features

**Automatic Mesh Simplification**:
- Large meshes are automatically simplified (95% face reduction)
- Minimum 100 faces preserved to maintain shape
- Significantly reduces HTML file size (from 1.5GB to ~50MB)
- Logging shows simplification progress and results

**Natural Sorting in PDF**:
- PDF pages sorted naturally: r1, r2, ..., r9, r10 (not r1, r10, r11...)
- Makes browsing through neurons more intuitive

**Skeleton Modes**:
- Tube mode (default): Renders radius information for thick neurites
- Line mode: Used automatically when >50 neurons (faster rendering)

### Example: Visualize Top Matches

```python
from src.neuronbridge_finder import NeuronBridgeFinder

nbf = NeuronBridgeFinder(verbose=True)

# Find neurons for LH173 and visualize top 20 by bodyId
results = nbf.find_neurons_batch(
    queries='LH173',
    match_type='cds',
    output_dir='./lh173_analysis',
    visualize_top_n=20,
    visualize_by='bodyId',
    generate_individual_profiles=True,
    pdf_images_per_page=(4, 3)  # 12 images per page
)

# Results include neurons grouped by type in 3D viewer
# Individual profiles saved to plot-3d_{dataset}/individual_profiles/
```

### Example: Filter and Visualize Specific Types

```python
from src.neuronbridge_finder import NeuronBridgeFinder

nbf = NeuronBridgeFinder(verbose=True)

# Find neurons and filter to only visualize descending neurons (DN)
results = nbf.find_neurons_batch(
    queries='SS29633',
    match_type='cds',
    output_dir='./ss29633_analysis',
    visualize_top_n=12,
    type_filter={'startswith': 'DN'},  # Only DN types
    datasets_to_visualize='male-cns:v0.9'  # Only MANC
)

# Or filter ascending neurons (AN) with preserved original ranks
# If AN07B072_a was rank 4 in full results, it stays r4 in visualization
results = nbf.find_neurons_batch(
    queries='SS29633',
    match_type='cds',
    output_dir='./ss29633_an_analysis',
    visualize_top_n=12,
    type_filter={'startswith': 'AN'}
)
```

---

## Match Types

### CDS (Color Depth Search)
- **Algorithm**: Compares color depth MIP images pixel-by-pixel
- **Speed**: Fast (~seconds per query)
- **Best for**: Quick initial searches, finding similar morphologies

### PPPM (PatchPerPixMatch)
- **Algorithm**: Advanced patch-based matching with geometric constraints
- **Speed**: Slower but more accurate
- **Best for**: Refined matching, reducing false positives

### Both (Combined Ranking)
When using `match_type='both'`, results are ranked by combined score:
1. Each line gets a CDS rank (1-based, by descending score)
2. Each line gets a PPPM rank (1-based, by descending score)
3. Combined rank = CDS rank + PPPM rank
4. Results sorted by combined rank (lower is better)

This ensures lines that rank well in both algorithms appear first.

---

## Dataset Support

### Supported EM Datasets

| Dataset             | Format   | Description             |
| ------------------- | -------- | ----------------------- |
| `hemibrain:v1.2.1`  | NeuPrint | Hemibrain connectome    |
| `male-cns:v0.9`     | NeuPrint | Male CNS (brain + VNC)  |
| `flywire_FAFB_v783` | FAFB     | FAFB full brain         |
| `banc_v888` | BANC     | Standalone BANC brain + nerve cord (public release, current) |
| `banc_v626` | BANC     | Standalone BANC brain + nerve cord (public release, older) |
| `vnc:v0.5`          | NeuPrint | VNC (older version)     |
| `manc:v1.2.1`       | NeuPrint | MANC dataset            |

### Dataset Name Normalization

The module automatically normalizes dataset names for comparison:
- `flywire_FAFB_v783` ↔ `flywire_fafb:v783`
- `hemibrain_v1_2_1` ↔ `hemibrain:v1.2.1`
- `male-cns_v0_9` ↔ `male-cns:v0.9`
- `flywire_BANC_v626` / `flywire_BANC_v888` ↔ `banc_v626` / `banc_v888` (NeuronBridge library names `FlyWire_BANC*` map to the standalone local releases)

This allows seamless matching between local folder names and NeuronBridge API responses.

---

## Examples

### Example 1: Find Lines for a Neuron Type

```python
from src.neuronbridge_finder import NeuronBridgeFinder

nbf = NeuronBridgeFinder()

# Find all body IDs of type MBON01 in hemibrain
# Then find matching driver lines for each
results = nbf.neuron_to_lines(
    query='MBON01',
    dataset='hemibrain:v1.2.1',
    match_type='both'
)

for body_id, lines_df in results.items():
    print(f"\nBody ID {body_id}:")
    print(lines_df[['line', 'score', 'combined_rank']].head(5))
```

### Example 2: Cross-Dataset Search

```python
# Search for aMe12 neurons in both hemibrain and FlyWire FAFB
results = nbf.neuron_to_lines(
    query='aMe12',
    dataset=['hemibrain:v1.2.1', 'flywire_FAFB_v783'],
    match_type='cds'
)
```

### Example 3: Batch Processing with FlyLight Downloads

```python
# Find lines for multiple queries and download representative images
results = nbf.find_lines_batch(
    queries='MBON01,MBON02,MBON03',
    dataset='hemibrain:v1.2.1',
    match_type='both',
    output_dir='./mbon_analysis',
    download_images='flylight',
    download_top_n_img=5,
    flylight_category='SplitGAL4',
    simple_mode=True  # Only download 20x multichannel MIPs
)

print(f"Found {len(results)} total matches")
print(f"Unique lines: {results['line'].nunique()}")
```

### Example 4: Find EM Neurons for a Split-GAL4 Line

```python
# Find which neurons are labeled by SS01015
neurons = nbf.line_to_neuron('SS01015', match_type='both')

# Group by dataset
for dataset in neurons['dataset'].unique():
    subset = neurons[neurons['dataset'] == dataset]
    print(f"\n{dataset}: {len(subset)} neurons")
    print(subset[['bodyId', 'type', 'score']].head())
```

### Example 5: Separate GAL4/LexA from Split-GAL4 Lines

```python
from src.neuronbridge_finder import NeuronBridgeFinder

# Enable separate_splitgal4 mode
nbf = NeuronBridgeFinder(separate_splitgal4=True)

# Find lines for MBON01
# With download_top_n_img=5, downloads:
#   - Top 5 GAL4/LexA lines (VT*, R*, GMR*)
#   - Top 5 Split-GAL4 lines (SS*, LH*, MB*, etc.)
results = nbf.find_lines_batch(
    queries='MBON01',
    dataset='hemibrain:v1.2.1',
    match_type='cds',
    output_dir='./mbon_separated',
    download_images='flylight',
    download_top_n_img=5,  # 5 GAL4/LexA + 5 Split-GAL4 = 10 lines
    simple_mode=True
)

# Results include 'line_type' column
print("\nLine type distribution:")
print(results.groupby('line_type')['line'].nunique())

# Top GAL4/LexA matches
gal4_lines = results[results['line_type'] == 'gal4_lexa']
print("\nTop GAL4/LexA lines:")
print(gal4_lines.groupby('line')['score'].max().nlargest(5))

# Top Split-GAL4 matches
split_lines = results[results['line_type'] == 'split_gal4']
print("\nTop Split-GAL4 lines:")
print(split_lines.groupby('line')['score'].max().nlargest(5))
```

**Output:**
```
Line type distribution:
line_type
gal4_lexa      834
split_gal4     785
Name: line, dtype: int64

Top GAL4/LexA lines:
line
R65E11    0.89
R20H11    0.87
R14C08    0.85
...

Top Split-GAL4 lines:
line
IS49879    0.92
IS83928    0.90
MB011B     0.88
...
```

---

## Simple Mode

Simple mode (`simple_mode=True`) reduces download volume by applying filename-based filtering:

| Collection   | Filter                                                | Effect                        |
| ------------ | ----------------------------------------------------- | ----------------------------- |
| Split-GAL4   | `20x` AND `multichannel`, excluding `image1`/`image2` | ~95% reduction                |
| VT GAL4      | Files with `total` in filename                        | ~90% reduction                |
| Gen1 R-lines | CDM and MIP files only                                | Keeps representative images   |
| MCFO         | Keep all files                                        | Full stochastic labeling data |

This is useful when you only need representative images rather than all available imagery.

---

## Integration with FlyLight Downloader

The NeuronBridgeFinder integrates with the [FlyLight Downloader](./FlyLight_Guide.md) for downloading full-resolution imagery:

```python
# NeuronBridge provides CDM images (color depth masks)
# FlyLight provides full MIP images, aligned stacks, and metadata

results = nbf.find_lines_batch(
    queries='MBON01',
    download_images='flylight',  # Uses FlyLightDownloader internally
    flylight_category=['GAL4/LEXA', 'SplitGAL4'],
    simple_mode=True
)
```

For direct FlyLight access, see the [FlyLight Downloader Guide](./FlyLight_Guide.md).

### NeuronBridge cache layout

The current cache lives under `cache/neuronbridge/parquet/<neuronbridge-version>/`:

- `id_to_lines/` contains one table per canonical body/dataset/match key. Region and `max_api_images_per_line` are not part of this key because they do not affect EM-to-LM lookups.
- `image_cache/` contains one compact table per LM image and algorithm (`cds` or `pppm`). The unused duplicate `both` tables are no longer created.
- `line_image_mapping.json` stores image metadata beside those tables in the same versioned directory.
- `manifest.json` records the cache format and schema version.

`line_to_neuron()` derives its result from the per-image tables, so `top_n` is applied at query time and cannot make a smaller request poison a later full request. The cache is Parquet-only and isolated by the NeuronBridge API version. After changing cache formats or versions, clear the cache and let it rebuild:

```python
nbf.clear_cache()
```

---

## Troubleshooting

### Common Issues

**Q: API returns no results for a known body ID**
- Check if the body ID exists in NeuronBridge (not all neurons have matches)
- Verify the expected dataset matches the actual dataset in NeuronBridge

**Q: Dataset mismatch errors**
- Use the `dataset` parameter to specify which dataset to search
- The module validates body IDs against their actual dataset in NeuronBridge

**Q: Slow performance**
- Enable caching with `use_cache=True`
- Use `download_top_n_img` to limit image downloads
- Consider using `simple_mode=True` for FlyLight downloads

**Q: NeuronBridge client import fails**
```bash
pip install -e . --no-deps
python -c "import neuronbridge_client; print('NeuronBridge client OK')"
```

### Clearing Cache

```python
nbf = NeuronBridgeFinder()

# Clear all cache
nbf.clear_cache()

# Clear specific cache type
nbf.clear_cache(cache_type='id_to_lines')
nbf.clear_cache(cache_type='line_to_neuron')
nbf.clear_cache(cache_type='image_cache')
```

---

## See Also

- [FlyLight Downloader Guide](./FlyLight_Guide.md) - Direct FlyLight imagery access
- [NeuronBridge Website](https://neuronbridge.janelia.org/) - API and web service
