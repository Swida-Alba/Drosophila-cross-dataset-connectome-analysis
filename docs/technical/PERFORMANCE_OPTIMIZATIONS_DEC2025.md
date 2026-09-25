# Performance Optimizations (December 2025)

**Date:** December 7, 2025
**Version:** 4.1

> **Drift note (2026-09-25):** two mechanisms below were later redefined.
> (1) `skip_bodyId` semantics changed on 2026-09-04 — bodyId paths are
> still enumerated in memory and type-level outputs are DERIVED from
> them; the flag now suppresses bodyId-level *exports* only (and gates
> interlayer info *saving*, not fetching). (2) The aggregation-internal
> tqdm bars described in §3 were later disabled — only the path-enrichment
> bar remains active in `statvis.py`. The measurements remain a valid
> record of the December 2025 optimization round.

## Overview

This document details the performance optimizations introduced in version 4.1 to handle large-scale connectome datasets (e.g., FlyWire, whole-brain comparisons). The primary focus was on reducing I/O bottlenecks and memory usage during the data saving and aggregation phases.

---

## 1. Polars Integration for I/O and Matrix Generation

### Problem
The previous implementation used **Pandas** for saving CSV files and generating connection matrices (pivoting). For large datasets (e.g., 6M+ rows), Pandas operations were becoming a significant bottleneck:
- **CSV Writing**: `to_csv` is relatively slow for large DataFrames.
- **Pivoting**: `pivot_table` is computationally expensive and memory-intensive.

### Solution
We integrated the **Polars** library (`polars`) to handle these specific tasks. Polars is a high-performance DataFrame library written in Rust.

**Implementation Details:**
- **Zero-Copy Conversion**: DataFrames are converted from Pandas to Polars efficiently.
- **Fast Pivoting**: Polars' `pivot` operation is significantly faster than Pandas.
- **Multi-threaded CSV Writing**: Polars writes CSV files in parallel, utilizing all available CPU cores.

**Performance Impact:**
- **Matrix Generation**: 10-100x faster (seconds vs minutes).
- **CSV Saving**: 5-10x faster.

**Code Location:** `src/coana.py` -> `_save_matrices_to_csv`

---

## 2. Skip BodyId Processing (`skip_bodyId`)

### Problem
In many cross-dataset comparison workflows, researchers are primarily interested in **neuron type-level** connectivity (e.g., "Does cell type A connect to cell type B?"). However, the system was always calculating and saving **bodyId-level** data (individual neuron connections), which is:
- **Resource Intensive**: Requires processing millions of individual edges.
- **Disk Heavy**: Generates massive CSV files (GBs) that are often unused.
- **Time Consuming**: Adds significant overhead to the analysis pipeline.

### Solution
Introduced a new parameter `skip_bodyId` to `ComparisonParameters` and `FindNeuronConnection`.

**How it works** *(as introduced in v4.1; see the drift note for the
current semantics)*:
When `skip_bodyId=True`:
1. **Skips Saving**: `connection_info_bodyId.csv` and bodyId-level matrices are not generated.
2. **Skips Enrichment**: Detailed path analysis at the bodyId level is bypassed.
3. **Skips Interlayer Fetching**: Intermediate neuron details are not fetched if not needed for type analysis.

**Usage:**
```python
params = ComparisonParameters(
    # ...
    skip_bodyId=True
)
```

**Performance Impact:**
- **Execution Time**: Reduces analysis time by 40-60% for large datasets.
- **Disk Usage**: Saves gigabytes of storage space.

---

## 3. Granular Progress Tracking

### Problem
During heavy aggregation steps (e.g., grouping millions of edges by neuron type), the system appeared to "hang" with no feedback, leading users to believe the process had crashed.

### Solution
Added `tqdm` progress bars to the internal aggregation functions in `src/statvis.py`.

**Features:**
- Shows real-time progress of grouping and aggregation operations.
- Provides visual confirmation that the system is active.

**Code Location:** `src/statvis.py` -> `EnrichConnectionTable`
