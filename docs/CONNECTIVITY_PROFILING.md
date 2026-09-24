# Connectivity Profiling Guide

## Overview

> **UI location:** Connectivity tab → **Comparison** sub-tab (run folders keep the `profiling_` prefix).

The `ConnectivityProfiling.py` script compares connectivity profiles within one dataset or across multiple selected datasets, enabling analysis of neural circuit similarity patterns. It supports comparison at multiple levels (bodyId, type) with interactive heatmap visualization.

## Key Features

- **Flexible Query Input**: Simple list, nested list with custom names, or CSV file
- **Nested List Format**: `[['GroupName', [id1, id2]], ['Group2', [id3, 'type']]]`
- **CSV File Support**: `group_map_csv` parameter (like VisualizeSkeleton's `layer_map_csv`)
- **Aggregation Levels**: Compare at bodyId-level or type-level (mean pooling)
- **ALL Metrics Output**: Automatically computes jaccard, cosine, rank_corr, rank_corr_union
- **Separate Heatmaps**: One heatmap file per metric (not combined)
- **Directional Analysis**: Separate upstream/downstream or overall (upstream + downstream) profile comparison
- **Multi-Dataset Overview**: Inter-dataset heatmaps use neuron/type rows and dataset-pair columns
- **Report Rendering**: Reports redraw heatmaps with Plotly and link to the local VisPath HTML for editing
- **Interactive Visualization**: Heatmaps via VisualizePath with native Ward clustering
- **Profile Saving**: Saves individual and aggregated connectivity profiles
- **Auto-Generated Output**: Folder named `profiling_{query_name}_{timestamp}`

## Quick Start

### Basic Usage

```python
# In scripts/ConnectivityProfiling.py, modify the configuration section:

from comparison.profile_comparator import ConnectivityProfileComparer

comparer = ConnectivityProfileComparer(
    query=['Mi1', 'Tm3', 'aMe12', 'L2'],  # Neuron types or bodyIds
    dataset='male-cns:v0.9',
    aggregation_level='type',
    output_dir='../local_data/'
)

results = comparer.run()
```

Then run:
```bash
cd scripts
python ConnectivityProfiling.py
```

## Input Options

### Option 1: Simple List

Compare profiles from a simple list of types or bodyIds:

```python
comparer = ConnectivityProfileComparer(
    query=['Mi1', 'Tm3', 'aMe12', 'L2', 'Dm9'],
    dataset='male-cns:v0.9',
    aggregation_level='type',  # Aggregate by type (mean pooling)
)
```

### Option 2: Nested List with Custom Group Names

Compare profiles with custom-named groups (like VisualizeSkeleton's `neuron_layers`):

```python
comparer = ConnectivityProfileComparer(
    query=[
        ['Clock Neurons', ['DN1pA', 'DN1pB', 'DN2']],
        ['Visual Neurons', ['Mi1', 'Tm3', 'aMe12']],
        ['Motor', ['MN1', 'MN2']],
    ],
    dataset='male-cns:v0.9',
    aggregation_level='type',
)
```

Each group is `['GroupName', [list_of_ids_or_types]]`. The group names become the profile labels.

### Option 3: CSV File for Group Mapping

Use a CSV file to define groups (like VisualizeSkeleton's `layer_map_csv`):

```python
comparer = ConnectivityProfileComparer(
    query=[],  # Will be overridden by CSV
    dataset='male-cns:v0.9',
    group_map_csv='my_groups.csv',  # CSV file path
)
```

**CSV Format:**
```csv
group,id_type_instance
Clock,DN1pA
Clock,DN1pB
Clock,DN2
Visual,Mi1
Visual,Tm3
Motor,MN1
Motor,MN2
```

### Option 4: BodyIds Directly

Compare individual neuron profiles:

```python
comparer = ConnectivityProfileComparer(
    query=[720575940610453042, 720575940610453043, 720575940610453044],
    dataset='male-cns:v0.9',
    aggregation_level='bodyid',
)
```

## Configuration Parameters

### Dataset Configuration

| Parameter | Description        | Default           |
| --------- | ------------------ | ----------------- |
| `dataset` | Dataset identifier | `'male-cns:v0.9'` |

### Query Input

| Parameter       | Description                                                          | Default  |
| --------------- | -------------------------------------------------------------------- | -------- |
| `query`         | List of types/bodyIds, nested list with names, or empty if using CSV | Required |
| `group_map_csv` | Path to CSV file for group mapping                                   | `None`   |

### Profile Construction

| Parameter               | Description                      | Default |
| ----------------------- | -------------------------------- | ------- |
| `top_k`                 | Top K partners per direction     | `15`    |
| `top_m`                 | Minimum unique types to ensure   | `5`     |
| `min_synapse_threshold` | Minimum synapses for connections | `3`     |

### Comparison Parameters

| Parameter           | Description                               | Default  |
| ------------------- | ----------------------------------------- | -------- |
| `aggregation_level` | `'bodyid'` or `'type'`                    | `'type'` |
| `direction`         | `'upstream'`, `'downstream'`, or `'both'` | `'both'` |

**Note:** ALL similarity metrics are computed automatically:
- `jaccard`: Set-based overlap (0-1)
- `cosine`: Weight vector similarity (0-1)
- `rank_corr`: Spearman correlation (-1 to 1)
- `rank_corr_union`: Raw Spearman correlation on the partner union (-1 to 1; sign meaningful, 0 = no relation)

### Output Configuration

| Parameter           | Description                | Default            |
| ------------------- | -------------------------- | ------------------ |
| `output_dir`        | Base directory for results | `'../local_data/'` |
| `generate_heatmaps` | Generate visualizations    | `True`             |
| `show_figures`      | Open in browser            | `False`            |

Output folder is auto-generated as: `{output_dir}/profiling_{query_name}_{timestamp}/`

In the UI (Connectivity tab → Comparison sub-tab), the **Output Directory** is passed directly to the
backend and to the output-file browser. It inherits the Settings default unless
the tab has its own saved override.
| `TOP_K` | Top K partners per direction | `15` |
| `TOP_M` | Minimum unique types to ensure | `5` |
| `MIN_SYNAPSE_THRESHOLD` | Minimum synapses for connections | `3` |

### Comparison Parameters

| Parameter           | Description                               | Default  |
| ------------------- | ----------------------------------------- | -------- |
| `AGGREGATION_LEVEL` | `'bodyid'` or `'type'`                    | `'type'` |
| `DIRECTION`         | `'upstream'`, `'downstream'`, or `'both'` | `'both'` |

**Note:** ALL similarity metrics are computed automatically:
- `jaccard`: Set-based overlap (0-1)
- `cosine`: Weight vector similarity (0-1)
- `rank_corr`: Spearman correlation (-1 to 1)
- `rank_corr_union`: Raw Spearman correlation on the partner union (-1 to 1; sign meaningful, 0 = no relation)

### Output Configuration

| Parameter           | Description                | Default            |
| ------------------- | -------------------------- | ------------------ |
| `OUTPUT_DIR`        | Base directory for results | `'../local_data/'` |
| `GENERATE_HEATMAPS` | Generate visualizations    | `True`             |
| `SHOW_FIGURES`      | Open in browser            | `False`            |

Output folder is auto-generated as: `{OUTPUT_DIR}/profiling_{query_name}_{timestamp}/`

## Similarity Metrics

### Jaccard Similarity

Set-based overlap of partner types (ignores weights):

$$\text{Jaccard} = \frac{|A \cap B|}{|A \cup B|}$$

### Cosine Similarity

Weight vector similarity:

$$\text{Cosine} = \frac{A \cdot B}{\|A\| \times \|B\|}$$

### Rank Correlation (`rank_corr`)

Spearman correlation of partner rankings (raw value, -1 to 1):

$$\text{RankCorr} = \rho_{spearman}$$

### Rank Correlation Union (`rank_corr_union`)

Raw Spearman correlation on the partner union (missing = 0), NOT normalized —
the sign is meaningful (positive = concordant, negative = discordant) and
0 means no monotonic relation:

$$\text{RankCorr}_{union} = \rho_{spearman}(union)$$

## Output Structure

```
{output_dir}/profiling_{query_name}_{timestamp}/
├── parameters.json
├── README.txt
├── results/
│   ├── similarity_jaccard_combined.csv
│   ├── similarity_jaccard_upstream.csv
│   ├── similarity_jaccard_downstream.csv
│   ├── similarity_cosine_combined.csv
│   ├── similarity_cosine_upstream.csv
│   ├── similarity_cosine_downstream.csv
│   ├── similarity_rank_corr_combined.csv
│   ├── similarity_rank_corr_upstream.csv
│   ├── similarity_rank_corr_downstream.csv
│   ├── similarity_rank_corr_union_combined.csv
│   ├── similarity_rank_corr_union_upstream.csv
│   └── similarity_rank_corr_union_downstream.csv
├── profiles/
│   ├── profiles_summary.json     # Summary metadata
│   ├── individual/               # Raw connectivity profiles per bodyId
│   │   └── {type}_{bodyId}_profile.json
│   └── aggregated/               # Type-aggregated profiles
│       └── {type}_profile.json
└── visualization/
    ├── heatmap_combined_jaccard.html
    ├── heatmap_combined_cosine.html
    ├── heatmap_combined_rank_corr.html
    ├── heatmap_combined_rank_corr_union.html
    ├── heatmap_upstream_jaccard.html
    ├── heatmap_upstream_cosine.html
    ├── heatmap_upstream_rank_corr.html
    ├── heatmap_upstream_rank_corr_union.html
    ├── heatmap_downstream_jaccard.html
    ├── heatmap_downstream_cosine.html
    ├── heatmap_downstream_rank_corr.html
    └── heatmap_downstream_rank_corr_union.html
```

## Visualization Features

The editable heatmap files are generated using `VisualizePath.VisConnMatInteractive`, providing:

- **Clustering toggle**: Switch between Original and Clustered ordering (Ward by default)
- **Clustering method selection**: Ward, Average, Complete, Single linkage
- **Scale options**: Linear, Log₂, Log₁₀, √ scales
- **Color scale presets**: Multiple color schemes
- **Interactive features**: Zoom, pan, export to SVG/PNG

For connectivity profiling, heatmaps default to **clustered ordering** (`init_clustered=True`) to highlight similar profiles.

**Note:** Each metric gets its own separate heatmap file, making it easy to compare different similarity measures side-by-side.

The generated `report.html` redraws those matrices with Plotly instead of embedding the VisPath pages. The report uses separate **Intra-dataset** and **Inter-dataset** tabs, dataset tabs, and Overall/Upstream/Downstream sub-tabs. Each direction shows six metric cards in a three-column grid (two rows); cell labels are hidden and exact values are available on hover. The report follows VisPath's Ward/Euclidean clustered ordering and includes a local **Open VisPath heatmap for editing** link. For multi-dataset runs, the overview files under `cross_dataset/all_types/` have one row per queried neuron/type and one column per dataset pair. The `cross_dataset/mapping_summary.csv` file also includes a numeric `same name` column (`1` when every dataset has the same non-empty resolved name, otherwise `0`).

## Intra-Type vs Inter-Type Analysis

For detailed bodyId-level analysis within and across types:

```python
comparer = ConnectivityProfileComparer(
    dataset='male-cns:v0.9',
    query=['Mi1', 'Tm3'],  # Must be neuron types (not bodyIds)
    output_dir='./results'
)

# Run intra/inter type comparison (automatically uses query neuron types)
intra_inter = comparer.compare_intra_inter_type()

# Access results
print(intra_inter['intra_type'].head())  # Same-type bodyId pairs
print(intra_inter['inter_type'].head())  # Cross-type bodyId pairs
```

## Interactive Heatmap Features

The generated HTML heatmaps include:

- **Clustering Toggle**: Switch between original and Ward-clustered ordering
- **Clustering Methods**: Ward, Average, Complete, Single linkage
- **Color Scales**: Multiple colorscales (Viridis, Plasma, Blues, etc.)
- **Font Size Control**: Adjustable label font size
- **Label Toggle**: Show/hide labels for large matrices
- **Hover Info**: Detailed values on hover
- **Export Options**: SVG and PNG export

## Example Workflows

### Workflow 1: Compare Visual System Types

```python
comparer = ConnectivityProfileComparer(
    query=['Mi1', 'Tm3', 'Tm1', 'Tm2', 'L2', 'L3', 'Dm9'],
    dataset='male-cns:v0.9',
    aggregation_level='type',
    direction='both',
)
results = comparer.run()
```

### Workflow 2: Custom Group Comparison

```python
comparer = ConnectivityProfileComparer(
    query=[
        ['Lamina', ['L1', 'L2', 'L3', 'L4', 'L5']],
        ['Medulla T-cells', ['Tm1', 'Tm2', 'Tm3', 'Tm9']],
        ['Mi-cells', ['Mi1', 'Mi4', 'Mi9']],
    ],
    dataset='male-cns:v0.9',
    aggregation_level='type',
)
results = comparer.run()
```

### Workflow 3: From CSV File

Create `neuron_groups.csv`:
```csv
group,id_type_instance
Lamina,L1
Lamina,L2
Lamina,L3
Medulla,Mi1
Medulla,Tm1
Medulla,Tm3
```

```python
comparer = ConnectivityProfileComparer(
    query=[],
    dataset='male-cns:v0.9',
    group_map_csv='neuron_groups.csv',
)
results = comparer.run()
```

### Workflow 4: Analyze Homolog Candidates

```python
# Compare specific bodyIds that may be homologs
comparer = ConnectivityProfileComparer(
    query=[bid1, bid2, bid3, bid4, bid5],
    dataset='male-cns:v0.9',
    aggregation_level='bodyid',
)
results = comparer.run()
# Check rank_corr_union heatmap for similarity patterns
```

## Troubleshooting

### Profile Extraction Fails

Ensure the dataset has pre-built connection cache:
```bash
# Build connection cache first (the dataset is a POSITIONAL argument)
python src/build_connection_cache.py male-cns:v0.9
```

### Not Enough Profiles

Check that:
1. Neuron types exist in the dataset
2. BodyIds are valid for the dataset
3. group_map_csv file path is correct and has required columns

### Memory Issues with Large Comparisons

For many profiles (>100), consider:
- Using type-level aggregation instead of bodyId
- Processing in batches
- Reducing `top_k` parameter

## Related Documentation

- [Profile Comparator Module](./core-features/ConnectivityProfiler_Guide.md)
- [Connectivity Profiler Module](./core-features/ConnectivityProfiler_Guide.md)
- [FindHomologs Script](./core-features/HomologFinding_Guide.md)
- [VisualizeSkeleton (similar grouping pattern)](./visualizations/3D_Skeleton_Guide.md)
