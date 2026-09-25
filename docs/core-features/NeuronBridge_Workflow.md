# NeuronBridge Workflow Guide

This guide provides comprehensive workflows for using NeuronBridge tools for **bidirectional** EM↔LM mapping.

## Table of Contents

- [Overview](#overview)
- [Two Workflows: EM→LM vs LM→EM](#two-workflows-emlm-vs-lmem)
- [Workflow A: Find Driver Lines (EM→LM)](#workflow-a-find-driver-lines-emlm)
- [Workflow B: Find Neurons (LM→EM)](#workflow-b-find-neurons-lmem)
- [Analyze Co-Labeling](#analyze-co-labeling)
- [Download FlyLight Images](#download-flylight-images)
- [Weighted Score Optimization](#weighted-score-optimization)
- [Multi-Type Query Strategy](#multi-type-query-strategy)
- [Output Files Reference](#output-files-reference)
- [Recommendations](#recommendations)

---

## Overview

The NeuronBridge integration provides **bidirectional** mapping between electron microscopy (EM) reconstructions and light microscopy (LM) driver lines:

| Direction | Use Case | Script | Method |
|-----------|----------|--------|--------|
| **EM → LM** | Find driver lines labeling your EM neurons | `NeuronBridge_FindLines.py` | `find_lines_batch()` |
| **LM → EM** | Find EM neurons matching a driver line | `NeuronBridge_FindNeuron.py` | `find_neurons_batch()` |
| **Analysis** | Analyze co-labeling and specificity | `NeuronBridge_Colabel.py` | `analyze_colabeling()` |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     NeuronBridge Analysis Pipeline                           │
└─────────────────────────────────────────────────────────────────────────────┘

                          ┌───────────────────┐
                          │   NeuronBridge    │
                          │       API         │
                          │  • CDS Search     │
                          │  • PPPM Search    │
                          └─────────┬─────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           │                                                  │
           ▼                                                  ▼
┌─────────────────────┐                          ┌─────────────────────┐
│  WORKFLOW A: EM→LM  │                          │  WORKFLOW B: LM→EM  │
│  ─────────────────  │                          │  ─────────────────  │
│  Input: EM neurons  │                          │  Input: Driver line │
│  (type/bodyId)      │                          │  (GAL4/Split-GAL4)  │
│                     │                          │                     │
│  Output: Matching   │                          │  Output: Matching   │
│  driver lines       │                          │  EM neurons         │
└──────────┬──────────┘                          └──────────┬──────────┘
           │                                                  │
           ▼                                                  ▼
┌─────────────────────┐                          ┌─────────────────────┐
│  • line_summary.csv │                          │  • all_neurons.csv  │
│  • weighted_score   │                          │  • 3D visualization │
│  • coverage_ratio   │                          │  • PDF profiles     │
└──────────┬──────────┘                          └─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  CO-LABELING        │
│  ANALYSIS           │
│  ─────────────────  │
│  • Specificity      │
│  • Selectivity      │
│  • Overlap matrix   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  FlyLight Images    │
│  • MIP PNGs         │
│  • CDM Masks        │
│  • Stack Files      │
└─────────────────────┘
```

---

## Two Workflows: EM→LM vs LM→EM

### When to Use Each Workflow

| Scenario | Workflow | Script |
|----------|----------|--------|
| "I have EM neurons, what driver lines label them?" | **EM→LM** | `NeuronBridge_FindLines.py` |
| "I have a driver line, what neurons does it label?" | **LM→EM** | `NeuronBridge_FindNeuron.py` |
| "Compare specificity of multiple driver lines" | **Co-Labeling** | `NeuronBridge_Colabel.py` |

### Quick Comparison

| Aspect | EM→LM (FindLines) | LM→EM (FindNeuron) |
|--------|-------------------|---------------------|
| **Input** | Neuron type/bodyId | Driver line name |
| **Output** | Ranked driver lines | Matched EM neurons |
| **Key Metric** | `weighted_score` | `score` per neuron |
| **Visualization** | FlyLight images | 3D skeletons |
| **Use Case** | Design experiments | Validate line coverage |

---

## Workflow A: Find Driver Lines (EM→LM)

Use this workflow to find GAL4/Split-GAL4 driver lines that label your EM neurons.

### Module Calling Tree

```
NeuronBridge_FindLines.py
│
├── NeuronBridgeFinder.find_lines_batch()
│   │
│   ├── Query Resolution
│   │   ├── Body ID lookup (int → NeuronBridge search)
│   │   ├── Type/Instance lookup (str → dataset query → body IDs)
│   │   └── LabelMapper support (cross-dataset unified naming)
│   │
│   ├── NeuronBridge API Calls
│   │   ├── id_to_lines() - CDS/PPPM search per body ID
│   │   └── Multi-dataset aggregation
│   │
│   ├── Result Processing
│   │   ├── weighted_score calculation
│   │   ├── coverage_ratio computation
│   │   └── Sorting by weighted_score
│   │
│   └── Output Generation
│       ├── {query}_lines.csv (per-query row-level matches)
│       ├── line_summary.csv (aggregated, sorted)
│       ├── gal4_lexa_summary.csv (GAL4/LexA only)
│       └── split_gal4_summary.csv (Split-GAL4 only)
│
└── Optional: Image Download
    ├── FlyLight S3/CDN access
    └── PDF summary generation
```

### Basic Usage

```python
# scripts/NeuronBridge_FindLines.py

# Single type query
query = 'aMe12'

# Multiple types together (finds lines labeling BOTH)
query = 'aMe12,MBON01'

# Multiple datasets
dataset = ['male-cns:v0.9', 'hemibrain:v1.2.1', 'flywire_FAFB_v783']
```

### Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `query` | Neuron type(s) or body ID(s) | Use type names for broader search |
| `dataset` | Dataset(s) to search | Use multiple for cross-dataset validation |
| `match_type` | 'cds', 'pppm', or 'both' | 'cds' for most cases |
| `separate_splitgal4` | Separate GAL4/LexA from Split-GAL4 | `True` recommended |
| `download_img_for_top_n_lines` | Images for top N lines | 10-20 |

### Output: Line Summary CSV Columns

| Column | Description |
|--------|-------------|
| `line` | Driver line name (e.g., VT000770) |
| `agg_mean_score` | Average NeuronBridge score |
| `match_count` | Number of unique neurons labeled |
| `coverage_ratio` | match_count / total_query_neurons |
| `weighted_score` | **agg_mean_score × coverage_ratio** |
| `matched_types` | Neuron types labeled by this line |

---

## Workflow B: Find Neurons (LM→EM)

Use this workflow to find EM neurons that match a given driver line's morphology.

### Module Calling Tree

```
NeuronBridge_FindNeuron.py
│
├── NeuronBridgeFinder.find_neurons_batch()
│   │
│   ├── Line Name Parsing
│   │   └── Support for comma-separated or list input
│   │
│   ├── NeuronBridge API Calls
│   │   ├── line_to_neuron() - search by line name
│   │   └── Multi-image aggregation per line
│   │
│   ├── Dataset Detection & Enrichment
│   │   ├── Match bodyIds to local neuron_df
│   │   └── Add type, instance, status info
│   │
│   └── Output Generation
│       ├── {line}_neurons.csv (per-line results)
│       ├── {line}_{dataset}_neurons.csv (dataset-categorized)
│       ├── {line}_{dataset}_types.csv (type summary)
│       └── all_neurons.csv (combined)
│
└── Optional: 3D Visualization
    ├── VisualizeSkeleton module
    ├── Automatic mesh simplification
    ├── PNG profile export
    └── PDF summary with natural sorting
```

### Basic Usage

```python
# scripts/NeuronBridge_FindNeuron.py

# Single driver line
lines = 'VT037867'

# Multiple lines
lines = 'LH173,VT037867,SS00731'

# Configuration
match_type = 'cds'      # 'cds', 'pppm', or 'both'
region = 'Brain'        # 'Brain', 'VNC', or 'All'
visualize_top_n = 50    # Visualize top 50 types
visualize_by = 'type'   # 'type' or 'bodyId'
```

### Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `lines` | Driver line name(s) | Single line or comma-separated |
| `match_type` | 'cds', 'pppm', or 'both' | 'cds' for speed |
| `region` | 'Brain', 'VNC', or 'All' | Filter by region |
| `top_n` | Max ranked matches per line; applied before score cutoff | 50 |
| `visualize_top_n` | 3D viz for top N types | 20-50 |
| `visualize_by` | 'type' or 'bodyId' | 'type' for grouping |

### Output: Neurons CSV Columns

| Column | Description |
|--------|-------------|
| `bodyId` | EM body ID |
| `dataset` | Source dataset (hemibrain, male-cns, etc.) |
| `type` | Neuron type |
| `instance` | Neuron instance name |
| `score` | NeuronBridge match score |
| `source_line` | Driver line name |

### 3D Visualization Features

When `visualize_top_n > 0`:
- **Grouped by type**: Neurons merged by type with `r{rank}_{type}_x{N}` labels
- **Automatic simplification**: 95% mesh reduction for large visualizations
- **Multi-view export**: Front, back, top, bottom, left, right PNG exports
- **PDF summary**: Natural-sorted individual neuron profiles

```
plot-3d_{dataset}/
├── {dataset}.html              # Interactive 3D viewer
├── exported_views/             # PNG exports
│   ├── front.png
│   ├── back.png
│   └── ...
├── individual_profiles/        # Per-neuron PNGs
│   ├── r1_aMe12_x5.png
│   ├── r2_MBON01_x3.png
│   └── ...
└── individual_profiles.pdf     # Summary PDF
```

---

## Analyze Co-Labeling

After finding candidate lines (from either workflow), analyze their specificity and overlap.

### When to Use

- Compare multiple driver lines from FindLines results
- Assess which lines are most specific to your target neurons
- Design experiments with minimal overlap

### Module Calling Tree

```
NeuronBridge_Colabel.py
│
├── NeuronBridgeFinder.analyze_colabeling()
│   │
│   ├── line_to_neuron() per line
│   │   └── Get all EM neurons labeled by each line
│   │
│   ├── Expression Matrix Building
│   │   └── Types × Lines score matrix
│   │
│   ├── Co-Labeling Analysis
│   │   ├── Jaccard similarity
│   │   ├── Weighted Jaccard
│   │   └── Rank correlation
│   │
│   └── Visualization
│       ├── labeling_distribution.html (mountain plots)
│       ├── expression_matrix.html (heatmap)
│       ├── colabeling_matrix.html (similarity heatmap)
│       └── 3D skeleton visualization (optional)
│
└── Output Files
    ├── expression_matrix.csv
    ├── labeling_info.csv
    ├── colabeling_matrix_*.csv
    └── line_labeled_neurons/*.csv
```

### Input: Top Lines from FindLines

```python
# scripts/NeuronBridge_Colabel.py

lines = [
    'VT037867',  # Top GAL4 lines from FindLines
    'SS01015',   # Top Split-GAL4 lines
    # ... more lines
]
```

### Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `lines` | List of driver lines to analyze | Use top 10-20 from FindLines |
| `min_score` | Expression/labeling matrix and similarity cutoff, applied after top-N retrieval | 30000 |
| `min_type_avg_score` | Additional type filtering for similarity matrices only | 10000 |
| `visualize_top_n` | 3D visualization of top N types | 10-20 |

> **Note**: Top-N retrieval is performed before score filtering, so the raw per-line tables always contain the requested number of ranked matches, including rows below the cutoff. `min_score` filters the expression/labeling matrices and the similarity inputs; `min_type_avg_score` applies only to similarity-matrix type filtering. Runs that use these cutoffs record the distinction in `user_warning_notes.txt`.

---

## Download FlyLight Images

Images are automatically downloaded during `find_lines_batch()` if configured.

### FlyLight Category Priority

```python
flylight_category = ['GAL4/LEXA', 'SplitGAL4', 'MCFO']
```

Images are searched in order:
1. GAL4/LexA collection
2. Split-GAL4 collection
3. MCFO collection (fallback)

### Simple Mode (Recommended)

```python
simple_mode = True  # Reduce download volume
# - Split-GAL4: only '20x' AND 'multichannel' files
# - GAL4/LexA: only 'total' files
```

---

## Weighted Score Optimization

### The Problem

Traditional scoring ranks lines by raw NeuronBridge score, which may favor lines that only label a subset of queried neurons with very high scores.

### The Solution: Weighted Score

```
weighted_score = agg_mean_score × coverage_ratio
coverage_ratio = match_count / total_query_neurons
```

### Example

When querying 'aMe12' across 2 datasets with 8 total neurons:

| Line | agg_mean_score | match_count | coverage_ratio | weighted_score |
|------|----------------|-------------|----------------|----------------|
| VT000770 | 43,925 | 8 | 1.0 | **43,925** ✓ |
| SS55729 | 50,000 | 7 | 0.875 | 43,750 |

**VT000770 ranks higher** because it labels ALL queried neurons, even though SS55729 has a higher raw score.

---

## Multi-Type Query Strategy

### Query Together vs. Separately

| Goal | Query Method | Example |
|------|--------------|---------|
| Lines labeling **BOTH** types | Query together | `'aMe12,MBON01'` |
| Best lines for **EACH** type | Query separately | Run 'aMe12' then 'MBON01' |

### ⚠️ Important Warning

If you want to find lines labeling **different groups independently**, DO NOT query them together:

```python
# ❌ WRONG: Finds lines labeling BOTH aMe12 AND MBON01
query = 'aMe12,MBON01'

# ✅ CORRECT: Run separate queries
# Query 1: Find best lines for aMe12
query = 'aMe12'

# Query 2: Find best lines for MBON01
query = 'MBON01'
```

---

## Output Files Reference

### FindLines Output Structure (EM→LM)

```
NB-find-lines_{query}_{timestamp}/
├── {query}_lines.csv           # Per-query row-level matches
├── line_summary.csv           # Aggregated, sorted by weighted_score
├── gal4_lexa_summary.csv      # GAL4/LexA only
├── split_gal4_summary.csv     # Split-GAL4 only
├── images/                    # Downloaded images
│   └── {line_name}/
│       └── *.png, *.jpg
└── images_summary.pdf         # PDF summary
```

### FindNeuron Output Structure (LM→EM)

```
NB-find-neurons_{lines}_{timestamp}/
├── {line}_neurons.csv          # All neurons for this line
├── {line}_{dataset}_neurons.csv # Dataset-categorized neurons
├── {line}_{dataset}_types.csv   # Type summary with counts
├── all_neurons.csv             # Combined results
├── parameters.json             # Reproducibility parameters
└── plot-3d_{dataset}/           # 3D visualizations
    ├── {dataset}.html          # Interactive viewer
    ├── exported_views/         # PNG exports
    └── individual_profiles/    # Per-neuron profiles + PDF
```

### Colabeling Output Structure

```
NB-colabeling_{lines}_{timestamp}/
├── expression_matrix.csv      # Type × Line scores (dataset-prefixed)
├── expression_matrix_merged.csv  # Types merged across datasets
├── expression_matrix*.html    # Interactive heatmaps
├── labeling_info.csv          # All labeled types with dataset
├── colabeling_matrix_*.csv    # Similarity matrices
├── colabeling_matrix_*.html   # Interactive heatmaps
├── labeling_distribution_*.html  # Mountain plots
├── line_summary.csv           # Per-line statistics (see below)
├── colabeling_report.html     # Comprehensive report
├── line_labeled_neurons/      # Per-line neuron details
│   ├── {line}_neurons.csv
│   └── {line}_{dataset}_types.csv
└── plot-3d_{dataset}/          # 3D visualizations
```

### Line Summary Columns (Colabeling)

| Column | Description |
|--------|-------------|
| `line` | Driver line name |
| `n_neurons` | Total neurons matched |
| `n_types` | Unique types labeled |
| `max_score` | Maximum match score (50000 = perfect) |
| `n_neurons_HMS` | Neurons ≥ Half Max Score |
| `n_types_HMS` | Types ≥ Half Max Score |
| `n_neurons_MS` | Neurons at Max Score (within 0.1%) |
| `n_types_MS` | Types at Max Score |
| `Qf` | Quality Factor = max_score / n_types_HMS |
| `colabel_sparsity` | Labeling uniqueness (0-1) |

**Interpretation**: Higher `Qf` = more selective line (few high-scoring types). Lower `Qf` = broader labeling (many high-scoring types).

---

## Recommendations

### Workflow Selection Guide

```
Start
│
├── What do you have?
│   │
│   ├── EM neurons (type/bodyId)
│   │   └── Use Workflow A: FindLines (EM→LM)
│   │       └── Get ranked driver lines by weighted_score
│   │
│   └── Driver line name
│       └── Use Workflow B: FindNeuron (LM→EM)
│           └── Get matched EM neurons with 3D viz
│
├── Need specificity analysis?
│   │
│   ├── Yes → Run NeuronBridge_Colabel.py
│   │       └── Compare multiple candidate lines
│   │
│   └── No → Use direct results
│
└── What type of line do you need?
    │
    ├── GAL4/LexA (broad expression)
    │   └── Focus on gal4_lexa_summary.csv
    │
    └── Split-GAL4 (sparse expression)
        └── Focus on split_gal4_summary.csv
```

### Parameter Recommendations

| Use Case | Parameters |
|----------|------------|
| Initial EM→LM screening | `match_type='cds'`, `min_score=30000` |
| Comprehensive line search | `match_type='both'`, multiple datasets |
| LM→EM validation | `region='Brain'`, `visualize_top_n=50` |
| High specificity analysis | `min_score=40000`, `min_type_avg_score=40000` |
| Quick visualization | `visualize_top_n=10`, `simple_mode=True` |

---

## Related Documentation

- **[NeuronBridge Integration Guide](./NeuronBridge_Guide.md)** - Complete API reference
- **[FlyLight Downloader Guide](./FlyLight_Guide.md)** - Image download details
- **[Co-Labeling Similarity Methods](../COLABELING_SIMILARITY_METHODS.md)** - Similarity metrics explained
- **[3D Skeleton Visualization Guide](../visualizations/3D_Skeleton_Guide.md)** - 3D visualization options

---

## Quick Reference

### Complete Calling Tree

```
User Input
│
├── EM neurons (type/bodyId)     ──────┐
│   │                                   │
│   ▼                                   ▼
│ ┌──────────────────────────┐   ┌──────────────────────────┐
│ │  NeuronBridge_FindLines  │   │  NeuronBridge_FindNeuron │
│ │  ────────────────────────│   │  ────────────────────────│
│ │  find_lines_batch()      │   │  find_neurons_batch()    │
│ │  • resolve query→bodyIds │   │  • parse line names      │
│ │  • search NeuronBridge   │   │  • search NeuronBridge   │
│ │  • calc weighted_score   │   │  • enrich with metadata  │
│ │  • rank by coverage×score│   │  • 3D visualization      │
│ └────────────┬─────────────┘   └──────────────────────────┘
│              │                           ▲
│              ▼                           │
│ ┌──────────────────────────┐             │
│ │   Top Candidate Lines    │─────────────┘
│ │   (weighted_score)       │    (validate with LM→EM)
│ └────────────┬─────────────┘
│              │
│              ▼
│ ┌──────────────────────────┐
│ │  NeuronBridge_Colabel    │
│ │  ────────────────────────│
│ │  analyze_colabeling()    │
│ │  • get labeled neurons   │
│ │  • build expression mat  │
│ │  • compute similarity    │
│ │  • visualize overlap     │
│ └────────────┬─────────────┘
│              │
│              ▼
│ ┌──────────────────────────┐
│ │   Final Selection        │
│ │   • Specificity report   │
│ │   • 3D visualization     │
│ │   • Downloaded images    │
│ └──────────────────────────┘
└── Driver line name ──────────────────────────────────────────┘
```
