# Module Calling Tree

This document illustrates the mutual calls and dependencies between all core modules in the project and the vispath-subproject.

## Overview Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           USER ENTRY POINTS                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│  FindNeuronConnection    ComparisonAnalyzer    HomologFinder    VisualizePath │
│        (coana)          (comparison_analyzer) (profile_comparator)  (vispath) │
└────────────┬─────────────────────┬───────────────────┬──────────────────┬────┘
             │                     │                   │                  │
             ▼                     ▼                   ▼                  ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                            CORE MODULES LAYER                                   │
├─────────────────────┬─────────────────────────┬─────────────────────────────────┤
│ ConnectivityProfiler│   ProfileComparator     │                                 │
│ (connectivity_      │   (profile_comparator)  │                                 │
│  profiler.py)       │                         │                                 │
└─────────┬───────────┴───────────┬─────────────┴────────────────┬───────────────┘
          │                       │                              │
          ▼                       ▼                              ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                          DATA ACCESS LAYER                                      │
├────────────────────────────┬───────────────────────┬───────────────────────────┤
│    FindNeuronConnection    │      DataLoader       │    File Converters        │
│    (_FNC_CACHE)           │    (comparison)       │  (FAFB, BANC)             │
└────────────────────────────┴───────────────────────┴───────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                           │
├────────────────────┬─────────────────────┬───────────────────┬─────────────────┤
│   Neuprint API     │   Local Cache       │  Parquet Files    │  CSV Datasets   │
│  (hemibrain, etc.) │   (./cache/)        │  (FAFB, BANC)     │  (./datasets/)  │
└────────────────────┴─────────────────────┴───────────────────┴─────────────────┘
```

---

## Detailed Module Dependencies

### 1. Main Project (`src/`)

```
src/
├── coana.py                    # FindNeuronConnection - Main connectome analysis
│   ├── imports:
│   │   ├── vispath_pkg.VisualizePath    # Visualization
│   │   ├── statvis                       # Statistical visualization
│   │   ├── FAFB_file_converter           # FAFB data conversion
│   │   └── BANC_file_converter           # BANC data conversion
│   └── provides:
│       ├── FindNeuronConnection          # Main class for connection queries
│       ├── _FNC_CACHE                    # Module-level connection cache
│       └── clear_fnc_cache()             # Cache management
│
├── statvis.py                  # Statistical visualization utilities
│   ├── imports: (external only - matplotlib, plotly, etc.)
│   └── provides: Statistical plots and analysis
│
├── FAFB_file_converter.py      # FAFB dataset converter
│   └── provides: Conversion utilities for FAFB parquet files
│
├── BANC_file_converter.py      # BANC dataset converter
│   └── provides: Conversion utilities for BANC parquet files
│
├── comparison/                 # Cross-dataset comparison module
│   ├── connectivity_profiler.py     # 1-hop/2-hop hybrid profiler (CORE)
│   ├── profile_comparator.py        # Profile comparison + HomologFinder
│   ├── comparison_analyzer.py       # Main comparison orchestrator
│   ├── data_loader.py               # Data loading utilities
│   ├── dataset_config.py            # Dataset configuration
│   ├── comparison_parameters.py     # Comparison parameters
│   ├── label_mapper.py              # Cross-dataset label mapping
│   ├── metrics.py                   # Comparison metrics
│   ├── visualizations.py            # Comparison visualizations
│   ├── profile_visualizations.py    # Profile-specific visualizations
│   └── html_report_generator.py     # Report generation
│
└── utils/
    └── api_utils.py            # API utilities (retry, escaping)
```

### 2. VisPath Subproject (`vispath-subproject/`)

```
vispath-subproject/
├── src/
│   └── vispath_pkg/
│       └── vispath.py          # VisualizePath - Network visualization
│           ├── imports: (external only - networkx, pandas, plotly)
│           └── provides:
│               ├── VisualizePath             # Main visualization class
│               ├── visualize_paths()         # Quick visualization function
│               ├── visualize_heatmap()       # Heatmap visualization
│               └── VisConnMatInteractive     # Interactive connection matrix
│
└── (standalone package - no internal project dependencies)
```

---

## Calling Tree by Feature

### A. Connection Query Flow

```
User Query: "Find connections from MBON14 to KC"
                    │
                    ▼
    ┌───────────────────────────────┐
    │   FindNeuronConnection        │
    │   (coana.py)                  │
    └───────────────┬───────────────┘
                    │
          ┌────────┴────────┐
          ▼                 ▼
   ┌─────────────┐   ┌──────────────┐
   │ _FNC_CACHE  │   │ Neuprint API │
   │ (in-memory) │   │ or Parquet   │
   └──────┬──────┘   └──────┬───────┘
          │                 │
          └────────┬────────┘
                   ▼
    ┌───────────────────────────────┐
    │   Connection DataFrame        │
    │   (conn_df, neuron_df)        │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │   VisualizePath               │
    │   (vispath_pkg)               │
    └───────────────────────────────┘
```

### B. Connectivity Profile Building Flow (1-hop/2-hop Hybrid)

```
User Query: "Get profile for neuron EPG_R"
                    │
                    ▼
    ┌───────────────────────────────┐
    │   ConnectivityProfiler        │
    │   (connectivity_profiler.py)  │
    │                               │
    │   ProfilerConfig:             │
    │   - top_k_bodyid=15           │
    │   - top_m_type=5              │
    │   - expand_untyped_2hop=True  │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │ Step 1: Query 1-hop partners  │
    │ (Neuprint API / Cache)        │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │ Step 2: Separate typed/untyped│
    │ - Typed: Keep top-k by weight │
    │ - Untyped: Mark for 2-hop     │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │ Step 3: Expand untyped 2-hop  │
    │ Query 2-hop partners of       │
    │ untyped 1-hop neurons         │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │ Step 4: Aggregate & normalize │
    │ - top-m unique types          │
    │ - Normalize weights           │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │   ConnectivityProfile         │
    │   (bodyId-level profile)      │
    └───────────────────────────────┘
```

### C. Homolog Finding Flow

```
User Query: "Find homologs for EPG in male-cns"
                    │
                    ▼
    ┌───────────────────────────────┐
    │   HomologFinder               │
    │   (profile_comparator.py)     │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │   Initialize with datasets    │
    │   ┌────────────────────────┐  │
    │   │ ConnectivityProfiler   │  │
    │   │ (internal instance)    │  │
    │   └────────────────────────┘  │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │ Step 1: Build source profiles │
    │ profiler.get_profile(type,    │
    │                      dataset) │
    │ ───────────────────────────── │
    │ → 1-hop/2-hop hybrid          │
    │ → bodyId-level profiles       │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │ Step 2: Get target candidates │
    │ (from target dataset)         │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │ Step 3: Build target profiles │
    │ profiler.get_profile(...)     │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │ Step 4: Compare profiles      │
    │ (bodyId-level comparison)     │
    │ - Jaccard similarity          │
    │ - Rank correlation            │
    │ - Combined score              │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │ Step 5: Aggregate & Save      │
    │ - bodyid_results.csv          │
    │ - type_summary.csv            │
    │ - homolog_results.csv         │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │ Step 6: Visualize Skeletons   │
    │ (if visualize_skeleton=True)  │
    │ ┌───────────────────────────┐ │
    │ │ VisualizeSkeleton (coana) │ │
    │ │ - HTML interactive plot   │ │
    │ │ - PNG export              │ │
    │ └───────────────────────────┘ │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │   Homolog Candidates          │
    │   (ranked by similarity)      │
    │   + Skeleton visualizations   │
    └───────────────────────────────┘
```

### D. Cross-Dataset Comparison Flow

```
User Query: "Compare connectivity across datasets"
                    │
                    ▼
    ┌───────────────────────────────┐
    │   ComparisonAnalyzer          │
    │   (comparison_analyzer.py)    │
    └───────────────┬───────────────┘
                    │
          ┌────────┴────────┐
          ▼                 ▼
    ┌──────────────┐  ┌──────────────────────────┐
    │ DataLoader   │  │ ConnectivityProfiler     │
    │              │  │                          │
    └──────┬───────┘  └────────────┬─────────────┘
           │                       │
           │         ┌─────────────┴─────────────┐
           │         │                           │
           ▼         ▼                           ▼
    ┌──────────────────────┐         ┌────────────────────┐
    │ FindNeuronConnection │         │ ProfileComparator  │
    │ (per dataset)        │         │ (similarity calc)  │
    └──────────────────────┘         └────────────────────┘
                                              │
                                              ▼
                                  ┌────────────────────┐
                                  │ ComparisonVisualizer│
                                  │ ProfileVisualizer   │
                                  └────────────────────┘
```

---

## Module Import Graph

```
                                    External
                                   Libraries
                                       │
    ┌──────────────────────────────────┼──────────────────────────────────┐
    │                                  │                                  │
    ▼                                  ▼                                  ▼
┌────────┐                      ┌──────────────┐                    ┌──────────┐
│neuprint│                      │ pandas/numpy │                    │ networkx │
│  API   │                      │ scipy        │                    │ plotly   │
└────┬───┘                      └──────┬───────┘                    └────┬─────┘
     │                                 │                                 │
     │                                 │                                 │
     ▼                                 ▼                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              PROJECT MODULES                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                          vispath-subproject                             │ │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │ │
│  │  │ VisualizePath (vispath.py)                                        │  │ │
│  │  │   - Network graph visualization                                   │  │ │
│  │  │   - Heatmap generation                                            │  │ │
│  │  │   - Interactive connection matrices                               │  │ │
│  │  │   (NO internal project dependencies - standalone)                 │  │ │
│  │  └───────────────────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                     ▲                                        │
│                                     │ imports                                │
│                                     │                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                             src/coana.py                                │ │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │ │
│  │  │ FindNeuronConnection                                              │  │ │
│  │  │   - Primary connection query interface                            │  │ │
│  │  │   - Multi-dataset support (hemibrain, FAFB, BANC, male-cns)       │  │ │
│  │  │   - Connection caching (_FNC_CACHE)                               │  │ │
│  │  │   - Skeleton visualization                                        │  │ │
│  │  └───────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────┬────────────────────────────────────┘ │
│                                       │ imports                              │
│                         ┌─────────────┴─────────────┐                        │
│                         ▼                           ▼                        │
│  ┌──────────────────────────────┐    ┌──────────────────────────────┐        │
│  │ FAFB_file_converter.py       │    │ BANC_file_converter.py       │        │
│  │ statvis.py                   │    │                              │        │
│  └──────────────────────────────┘    └──────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        src/comparison/                                  │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │ │
│  │  │ connectivity_profiler.py (CORE PROFILER)                        │    │ │
│  │  │   - 1-hop/2-hop hybrid approach                                 │    │ │
│  │  │   - ProfilerConfig, FuzzyMatchConfig                            │    │ │
│  │  │   - ConnectivityProfile dataclass                               │    │ │
│  │  │   - _PROFILER_CONN_CACHE (module-level cache)                   │    │ │
│  │  └────────────────────────────────┬────────────────────────────────┘    │ │
│  │                                   │                                     │ │
│  │                          ┌────────┴────────┐                            │ │
│  │                          ▼                 ▼                            │ │
│  │  ┌───────────────────────────┐  ┌───────────────────────────┐           │ │
│  │  │ profile_comparator.py     │  │                            │          │ │
│  │  │   - ProfileComparator     │  │                            │          │ │
│  │  │   - HomologFinder         │  │                            │          │ │
│  │  │   - ComparisonResult      │  └───────────────────────────┘           │ │
│  │  └────────────────────┬──────┘                                          │ │
│  │                       │                                                 │ │
│  │                       ▼                                                 │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │ │
│  │  │ comparison_analyzer.py                                          │    │ │
│  │  │   - ComparisonAnalyzer (main orchestrator)                      │    │ │
│  │  │   - quick_compare() function                                    │    │ │
│  │  │   - Uses: DataLoader, LabelMapper, ComparisonMetrics            │    │ │
│  │  └────────────────────────────────┬────────────────────────────────┘    │ │
│  │                                   │                                     │ │
│  │                                   ▼                                     │ │
│  │  ┌───────────────────────────┐  ┌───────────────────────────┐           │ │
│  │  │ visualizations.py         │  │ profile_visualizations.py │           │ │
│  │  │   - ComparisonVisualizer  │  │   - ProfileVisualizer     │           │ │
│  │  └───────────────────────────┘  └───────────────────────────┘           │ │
│  │                                                                         │ │
│  │  ┌───────────────────────────┐  ┌───────────────────────────┐           │ │
│  │  │ Support modules:          │  │ Configuration:            │           │ │
│  │  │   - data_loader.py        │  │   - dataset_config.py     │           │ │
│  │  │   - label_mapper.py       │  │   - comparison_params.py  │           │ │
│  │  │   - metrics.py            │  │                           │           │ │
│  │  └───────────────────────────┘  └───────────────────────────┘           │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Dependency Rules

### 1. ConnectivityProfiler is the Foundation

All profile-based operations **MUST** go through `ConnectivityProfiler`:

```python
# CORRECT: Always use ConnectivityProfiler for profiles
from src.comparison import ConnectivityProfiler, ProfilerConfig

config = ProfilerConfig(expand_untyped_2hop=True)
profiler = ConnectivityProfiler(datasets=['hemibrain:v1.2.1'], config=config)
profile = profiler.get_profile('EPG', 'hemibrain:v1.2.1')

# CORRECT: HomologFinder uses internal profiler
finder = HomologFinder(datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'])
# finder.profiler is automatically a ConnectivityProfiler instance
```

### 2. HomologFinder Depends on ConnectivityProfiler

```
HomologFinder
    │
    ├── self.profiler: ConnectivityProfiler
    │       └── 1-hop/2-hop hybrid profile building
    │
    ├── ProfileComparator (static methods)
    │       └── Similarity calculations
    │
    └── VisualizeSkeleton (optional)
            └── 3D skeleton visualization of top candidates
```

### 3. VisualizePath is Standalone

VisualizePath has **no internal project dependencies**. It receives data from other modules:

```python
# coana.py passes data to VisualizePath
vp = VisualizePath(
    all_connections=connections_df,  # Data from FindNeuronConnection
    all_paths=paths_df,
    ...
)
vp.visualize()
```

### 4. Cache Hierarchy

```
_FNC_CACHE (coana.py)
    └── Module-level cache for FindNeuronConnection data
        └── Shared across all FNC instances
        
_PROFILER_CONN_CACHE (connectivity_profiler.py)  
    └── Module-level cache for ConnectivityProfiler data
        └── Shared across all profiler instances
        
Disk Cache (./cache/)
    └── Persistent parquet/json files
        └── Used by all modules
```

---

## Quick Reference: Which Module to Use

| Task | Primary Module | Uses |
|------|----------------|------|
| Query connections | `FindNeuronConnection` (coana) | Neuprint API, Cache |
| Build connectivity profile | `ConnectivityProfiler` | 1-hop/2-hop hybrid |
| Find homologs | `HomologFinder` | ConnectivityProfiler, VisualizeSkeleton |

| Compare datasets | `ComparisonAnalyzer` | DataLoader, ConnectivityProfiler |
| Visualize network | `VisualizePath` | (receives data) |
| Visualize 3D skeleton | `VisualizeSkeleton` (coana) | Skeleton mesh data |
| Visualize profiles | `ProfileVisualizer` | ConnectivityProfile |

---

## Detailed Module Workflows

### FindNeuronConnection (coana.py) Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FindNeuronConnection Workflow                            │
│                                                                             │
│  INITIALIZATION                                                             │
│  ══════════════                                                             │
│  1. __init__(dataset, token=None)                                           │
│     ├─ Set dataset (hemibrain:v1.2.1, male-cns:v0.9, etc.)                  │
│     ├─ Setup authentication (NeuPrint token or local files)                │
│     └─ Initialize _FNC_CACHE module-level connection cache                 │
│                                                                             │
│  CONNECTION QUERY                                                           │
│  ════════════════                                                           │
│  2. get_connections(source, target)                                         │
│     ├─ Check _FNC_CACHE for cached connections                             │
│     ├─ If miss: query data source                                           │
│     │   ├─ NeuPrint API (hemibrain, male-cns)                              │
│     │   ├─ Parquet files (FAFB, BANC via converters)                       │
│     │   └─ Local cache (./cache/{dataset}/)                                │
│     ├─ _build_conn_index() using Polars groupby (O(1) lookups)             │
│     └─ Return (conn_df, neuron_df)                                          │
│                                                                             │
│  INDEX BUILDING (Polars-optimized)                                          │
│  ═══════════════════════════════════                                        │
│  3. _build_conn_index(conn_df)                                              │
│     ├─ Convert to Polars LazyFrame                                         │
│     ├─ group_by('bodyId_pre').agg(pl.col('index'))  →  bodyid_pre_index    │
│     ├─ group_by('bodyId_post').agg(pl.col('index')) →  bodyid_post_index   │
│     └─ O(1) lookup for any bodyId's connections                            │
│                                                                             │
│  CACHE MANAGEMENT                                                           │
│  ════════════════                                                           │
│  4. build_connection_cache(dataset)                                         │
│     ├─ Fetch ALL connections for dataset                                   │
│     ├─ Save as Parquet (./cache/{dataset}/connections.parquet)             │
│     ├─ Build and save indexes                                               │
│     └─ Enable fast local queries                                            │
│                                                                             │
│  VISUALIZATION                                                              │
│  ═════════════                                                              │
│  5. visualize_skeleton(bodyids)                                             │
│     ├─ Fetch skeleton meshes                                                │
│     ├─ Create interactive 3D HTML plot                                     │
│     └─ Optional PNG export                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ConnectivityProfiler Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ConnectivityProfiler Workflow                            │
│                                                                             │
│  CONFIGURATION                                                              │
│  ═════════════                                                              │
│  ProfilerConfig:                                                            │
│  ├─ top_k_bodyid=15        (max partners per bodyId)                       │
│  ├─ top_m_type=5           (min unique types after aggregation)            │
│  ├─ dynamic_expansion=True (expand until M types reached)                  │
│  ├─ max_expansion_factor=3 (max expansion: k * factor)                     │
│  ├─ expand_untyped_2hop=True (2-hop for untyped partners)                  │
│  └─ fuzzy_match: FuzzyMatchConfig (type normalization)                     │
│                                                                             │
│  PROFILE BUILDING FLOW                                                      │
│  ═════════════════════                                                      │
│                                                                             │
│  get_profile(neuron, dataset)                                               │
│     │                                                                       │
│     ├─[1] Cache Check                                                       │
│     │   └─ _load_from_cache() → Return if hit                              │
│     │                                                                       │
│     ├─[2] Query Connections                                                 │
│     │   ├─ _query_connections_local() [FAST - uses indexes]                │
│     │   │   ├─ _get_cached_conn_df(dataset)                                │
│     │   │   ├─ Use bodyid_pre_index / bodyid_post_index → O(1)             │
│     │   │   └─ Return (upstream_df, downstream_df)                         │
│     │   │                                                                   │
│     │   └─ _query_connections_neuprint() [FALLBACK - API call]             │
│     │                                                                       │
│     ├─[3] Process Connections (_process_connections)                        │
│     │   ├─ Separate typed vs untyped partners                              │
│     │   ├─ Apply fuzzy_match normalization                                 │
│     │   ├─ Dynamic expansion (until M types)                               │
│     │   ├─ Aggregate by type: groupby('partner_type').sum()                │
│     │   │   └─ [POLARS OPTIMIZATION TARGET]                                │
│     │   ├─ Compute ranks (higher weight = lower rank)                      │
│     │   └─ Build bodyId→type mapping                                       │
│     │                                                                       │
│     ├─[4] 2-Hop Expansion (if enabled)                                      │
│     │   └─ _fetch_2hop_partners(untyped_bodyids)                           │
│     │       ├─ Query connections for each untyped partner                  │
│     │       ├─ Aggregate typed 2-hop partners                              │
│     │       └─ Return {bodyId: (weights, ranks)}                           │
│     │                                                                       │
│     ├─[5] Create ConnectivityProfile                                        │
│     │   ├─ upstream_partners: Dict[type, weight]                           │
│     │   ├─ downstream_partners: Dict[type, weight]                         │
│     │   ├─ upstream_ranks: Dict[type, rank]                                │
│     │   ├─ downstream_ranks: Dict[type, rank]                              │
│     │   ├─ untyped_*_2hop: Dict[bodyId, Dict[type, weight]]               │
│     │   └─ typed_*_bodyids: Dict[bodyId, weight]                           │
│     │                                                                       │
│     └─[6] Save to Cache                                                     │
│         └─ _save_to_cache(profile)                                          │
│                                                                             │
│  BATCH PROCESSING (Polars-optimized)                                        │
│  ════════════════════════════════════                                       │
│  _load_cache_dataframe(dataset)                                             │
│     ├─ Uses Polars for fast Parquet reading                                │
│     ├─ Builds indexes with group_by (2.5x faster than Pandas)              │
│     └─ Enables O(1) lookups                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### HomologFinder (profile_comparator.py) Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HomologFinder Workflow                               │
│                                                                             │
│  INITIALIZATION                                                             │
│  ══════════════                                                             │
│  HomologFinder(                                                             │
│      source_dataset='hemibrain:v1.2.1',                                     │
│      target_dataset='male-cns:v0.9',                                        │
│      token='YOUR_TOKEN',  # Passed to profiler                             │
│      verbose=True                                                           │
│  )                                                                          │
│  └─ Creates internal ConnectivityProfiler instance                         │
│                                                                             │
│  HOMOLOG FINDING FLOW                                                       │
│  ═════════════════════                                                      │
│                                                                             │
│  find_homologs(source_type, target_types)                                   │
│     │                                                                       │
│     ├─[1] Ensure Data Available                                             │
│     │   └─ _ensure_connection_cache_complete(dataset)                      │
│     │       ├─ Check for connections.parquet                               │
│     │       ├─ If missing: fetch via API and save                          │
│     │       └─ Per-batch saving (memory-safe)                              │
│     │                                                                       │
│     ├─[2] Get Source BodyIds                                                │
│     │   └─ _get_bodyids_for_type(source_type, source_dataset)              │
│     │                                                                       │
│     ├─[3] Build Source Profiles (MEMORY-SAFE)                               │
│     │   └─ _build_profiles_memory_safe(source_bodyids, source_dataset)     │
│     │       ├─ _build_profiles_batch() with ThreadPoolExecutor             │
│     │       │   ├─ max_workers = min(32, cpu_count + 4)                    │
│     │       │   ├─ Deferred cache writes (_defer_cache_writes=True)        │
│     │       │   ├─ Batch saves every _batch_size profiles                  │
│     │       │   └─ tqdm progress bar                                       │
│     │       └─ Release connection data → gc.collect()                      │
│     │                                                                       │
│     ├─[4] Get Target Candidates                                             │
│     │   └─ _get_bodyids_for_type(target_types, target_dataset)             │
│     │                                                                       │
│     ├─[5] Build Target Profiles (MEMORY-SAFE)                               │
│     │   └─ _build_profiles_memory_safe(target_bodyids, target_dataset)     │
│     │       └─ Same workflow as source profiles                            │
│     │                                                                       │
│     ├─[6] Compare Profiles                                                  │
│     │   └─ _compare_profiles_batch()                                        │
│     │       ├─ For each (source_bid, target_bid) pair:                     │
│     │       │   ├─ Jaccard similarity (type overlap)                       │
│     │       │   ├─ Rank correlation (Spearman on shared types)             │
│     │       │   └─ Combined score                                          │
│     │       └─ Return ranked candidates                                    │
│     │                                                                       │
│     ├─[7] Aggregate Results                                                 │
│     │   ├─ bodyid_results.csv (all pairwise scores)                        │
│     │   ├─ type_summary.csv (aggregated by type)                           │
│     │   └─ homolog_results.csv (top candidates)                            │
│     │                                                                       │
│     └─[8] Visualize (if enabled)                                            │
│         └─ VisualizeSkeleton for top candidates                            │
│                                                                             │
│  PARALLEL PROCESSING DETAILS                                                │
│  ════════════════════════════                                               │
│  _build_profiles_batch():                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  ThreadPoolExecutor (max_workers = min(32, cpu_count + 4))            │ │
│  │                                                                       │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐     ┌─────────┐              │ │
│  │  │ Thread1 │  │ Thread2 │  │ Thread3 │ ... │ ThreadN │              │ │
│  │  │ profile │  │ profile │  │ profile │     │ profile │              │ │
│  │  │  bid_1  │  │  bid_2  │  │  bid_3  │     │  bid_N  │              │ │
│  │  └────┬────┘  └────┬────┘  └────┬────┘     └────┬────┘              │ │
│  │       │            │            │               │                   │ │
│  │       └────────────┴────────────┴───────────────┘                   │ │
│  │                           │                                         │ │
│  │                 ┌─────────▼─────────┐                               │ │
│  │                 │ as_completed()    │                               │ │
│  │                 │ + batch_size save │                               │ │
│  │                 └───────────────────┘                               │ │
│  │                                                                       │ │
│  │  Deferred Cache Writes:                                              │ │
│  │  ├─ Profiles collected in _pending_cache_writes                      │ │
│  │  ├─ flush_pending_cache_writes() at batch boundaries                 │ │
│  │  └─ Reduces I/O contention in parallel                              │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ComparisonAnalyzer (comparison_analyzer.py) Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ComparisonAnalyzer Workflow                             │
│                                                                             │
│  OVERVIEW                                                                   │
│  ════════                                                                   │
│  High-level orchestrator for cross-dataset comparison.                      │
│  Coordinates: DataLoader, ConnectivityProfiler, ProfileComparator          │
│                                                                             │
│  WORKFLOW                                                                   │
│  ════════                                                                   │
│  ComparisonAnalyzer(datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'])        │
│     │                                                                       │
│     ├─[1] Initialize Components                                             │
│     │   ├─ DataLoader: loads connection data                               │
│     │   ├─ LabelMapper: maps types across datasets                         │
│     │   ├─ ConnectivityProfiler: builds profiles                           │
│     │   └─ ComparisonMetrics: calculates similarity                        │
│     │                                                                       │
│     ├─[2] compare(neuron_types)                                             │
│     │   ├─ Build profiles for each type in each dataset                    │
│     │   ├─ Calculate pairwise similarities                                 │
│     │   └─ Generate comparison report                                       │
│     │                                                                       │
│     └─[3] visualize()                                                       │
│         ├─ ComparisonVisualizer: summary plots                             │
│         └─ ProfileVisualizer: per-type profile plots                       │
│                                                                             │
│  quick_compare() Function                                                   │
│  ════════════════════════                                                   │
│  Convenience wrapper for one-shot comparisons:                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ from src.comparison import quick_compare                            │   │
│  │                                                                     │   │
│  │ results = quick_compare(                                            │   │
│  │     neuron_type='EPG',                                              │   │
│  │     datasets=['hemibrain:v1.2.1', 'male-cns:v0.9'],                 │   │
│  │     visualize=True                                                  │   │
│  │ )                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Optimization Notes

### Polars Integration (v3.1.1+)

The following methods use Polars for 2-3x speedup:

| Module | Method | Optimization |
|--------|--------|--------------|
| `coana.py` | `_build_conn_index()` | Polars `group_by` for index building |
| `coana.py` | `_load_connection_db()` | Polars for batch Parquet loading |
| `connectivity_profiler.py` | `_load_cache_dataframe()` | Polars for cache loading |
| `connectivity_profiler.py` | `_build_indexes()` | Polars `group_by` (2.5x faster) |
| `connectivity_profiler.py` | `_process_connections()` | Polars for aggregation |

### Pre-built Indexes

Connection indexes enable O(1) lookup:
```python
# Index structure (built once, reused)
bodyid_pre_index:  {bodyId → [row_indices]}  # connections FROM this neuron
bodyid_post_index: {bodyId → [row_indices]}  # connections TO this neuron

# O(1) query (vs O(n) DataFrame filter)
rows = bodyid_pre_index.get(bodyId, [])
connections = conn_df.iloc[rows]
```

### Parallel Processing Guidelines

```python
# ThreadPoolExecutor settings in HomologFinder
max_workers = min(32, (os.cpu_count() or 1) + 4)

# Batch size for cache writes (reduce I/O contention)
_batch_size = 100  # profiles per cache flush

# Deferred cache writes (collect, then flush)
_defer_cache_writes = True
flush_pending_cache_writes()
```

---

## Version History

- **v3.1.1**: Polars integration for 2-3x speedup in index building and aggregation
- **v3.1**: Unified profile building through ConnectivityProfiler (1-hop/2-hop hybrid only)
- **v3.0**: Added HomologFinder, ProfileComparator
- **v2.x**: Original coana module with FindNeuronConnection
