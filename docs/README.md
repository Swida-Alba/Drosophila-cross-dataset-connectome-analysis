# Documentation Index

Welcome to the Drosophila Connectome Analysis Toolkit (DROCAT) documentation!

<a id="v450-changelog"></a>

## 🆕 Recent Updates (August 2026) - V4.5.0

### 🤖 Direct agent analysis without the UI
- **[DROCAT Direct Analysis Skill](../skills/drocat-usage/SKILL.md)**: Run,
  inspect, and safely modify backend scripts without loading the entire
  repository into an agent context.
- **[Quick Start](./QUICK_START.md#agent-first-direct-mode-no-ui)**: Use the
  repository-safe direct launcher.

Agent requests should be completion-oriented: ask the agent to run the
requested operation, validate its outputs, and report the final artifacts in
the same session. The [direct-analysis skill](../skills/drocat-usage/SKILL.md)
defines this completion contract.

## 🆕 Recent Updates (January 2026) - V4.4.0

### 🚀 Local FAFB + Standalone BANC Dataset Support (RECOMMENDED)
- **Local-first architecture**: Store FAFB locally and prepare BANC from its own public release bucket so repeated queries avoid network round-trips entirely
- **Mixed mode**: Seamlessly combines local cache + API fallback  
- **Zero API latency**: Instant queries for cached neurons
- **Automatic caching**: Build your cache once, reuse forever
- **📖 [FAFB Integration Guide](./FAFB_INTEGRATION.md)** | **[Cache System](./core-features/CacheSystem_Guide.md)**

### 🔍 Priority-Based Neuron Search
- **Smart search order**: bodyId → type → instance with automatic fallback
- **Flexible input**: Accept both int and string bodyIds: `[123456789]` or `['123456789']`
- **Regex support**: Use patterns like `['KC.*']`, `['.*PN.*']` across all columns
- **Consistent matching**: String-based comparison internally for reliability
- **📖 [Basic Usage Guide](./core-features/BasicUsage_Guide.md#search-priority)**

### 🎨 NT Visualization & Grouping
- **NT edge groups**: ACH, GABA, GLUT, DA, SER, OCT - select and style by neurotransmitter
- **Custom groups**: Create and save custom element groups for batch editing
- **Hover labels**: NT type displayed in edge tooltips with color coding
- **Export/Import**: Save complete graph states including custom groups and NT settings
- **Default opacity**: 50% for edges (vs 20%), 100% for nodes (vs 50%) - better visibility
- **📖 [Network Features Guide](./visualizations/VisualizePath_Network_Features.md)**

### 🔐 Authentication Improvements
- **config.json wins per key**: tokens in `config.json` are used first (the file a GitHub-pulled copy edits directly); the gitignored `config_local.json` fills empty entries - use it to keep secrets out of git
- **Automatic loading**: No need to pass tokens manually in scripts
- **Secure storage**: Keep credentials out of version control
- **📖 [Authentication Setup](./INSTALLATION.md#4-authentication)**

## 🆕 Recent Updates (December 2025) - V4.3

### 🌉 NeuronBridge & FlyLight Integration (NEW!)

**Comprehensive EM↔LM mapping and FlyLight imagery access:**

- **📖 [NeuronBridge Integration Guide](./core-features/NeuronBridge_Guide.md)** - Complete guide with examples
  - Bidirectional search: EM body ID → LM driver lines, and vice versa
  - CDS and PPPM matching with combined ranking
  - Multi-dataset support (hemibrain, male-cns, FAFB, standalone BANC)
  - Batch processing with automatic result aggregation
  - Image download integration with FlyLight
  - **Specificity/Selectivity Analysis**: Use `NeuronBridge_Colabel.py` for detailed line analysis
  
- **📖 [FlyLight Downloader Guide](./core-features/FlyLight_Guide.md)** - Complete guide with examples
  - Access to Janelia FlyLight S3 bucket and HTTP CDN
  - Support for R-lines, SS-lines (Split-GAL4), VT lines, and MCFO
  - Collection filtering (GAL4/LexA, SplitGAL4, MCFO, RawImages)
  - `simple_mode` for intelligent download volume reduction
  - Multiple file formats (PNG, H5J, LSM, MP4, JSON)

- **New Scripts**: `NeuronBridge_FindNeuron.py`, `NeuronBridge_FindLines.py`, `NeuronBridge_Colabel.py`
- **New Modules**: `src/neuronbridge_finder.py`, `src/flylight_downloader.py`

### HomologFinder Improvements
- **Hierarchical ConnectivityStatus**: Profile quality classified into 5 levels (NONE, RARE, INCOMPLETE, INCOMPLETE_EXPANSION, COMPLETE)
- **RARE Source Handling**: RARE sources (< 5 partners) now included with WARNING instead of being skipped
- **Status Tracking**: New `source_status_summary.json` and status columns in results
- **Renamed Column**: `shared_partner_count` → `adjacency_score` (clarifies this is from candidate-finding)
- **Dict-Based similarity_metric**: Now accepts weighted metric combinations
- **Improved Output Folder Structure**: New `source_bodyids.csv` and `top_target_bodyids.csv` files
- **Vector Prefiltering (fast mode)**: Candidate pool is first reduced to top 5% by adjacency score, then filtered to cosine>0 before full scoring

### Performance Optimizations
- **Polars Integration**: 10-100x faster CSV saving and matrix generation using Polars library
- **Skip BodyId Processing**: New `skip_bodyId=True` parameter in `ComparisonParameters` to bypass resource-intensive bodyId-level data saving and calculations for large-scale type-level analyses
- **Progress Tracking**: Added granular progress bars for heavy aggregation steps

### Other Updates
- **✨ NEW: [Module Calling Tree](./core-features/Module_Calling_Tree.md)** - Visual guide to module dependencies and calling relationships
- **✨ NEW: [Connectivity Profiler Guide](./core-features/ConnectivityProfiler_Guide.md)** - 1-hop/2-hop hybrid profile building approach
- **✨ [Homolog Finding Guide](./core-features/HomologFinding_Guide.md)** - Find homologous neurons across datasets using connectivity profiles
- **✨ [Cross-Dataset Comparison Guide](./core-features/CrossDatasetComparison_Guide.md)** - Compare connectivity across hemibrain, male-cns, FlyWire, and more
- **[BANC Integration](./BANC_INTEGRATION.md)** - Standalone BANC public-release support (no FlyWire/CAVE dependency).
- **[Visualization Guides](visualizations/README.md)** - Visualization docs and update summaries.
- **[VisualizePath Updates Nov 2025](./visualizations/VisualizePath_Updates_Nov2025.md)** - Detailed guide on connection matrix support and new features.
- **[vispath-subproject README](../vispath-subproject/README.md)** - Corrected data format documentation
- **[vispath-subproject Installation](../vispath-subproject/INSTALLATION.md)** - Added data format examples

## 📚 Documentation Structure

Documentation is organized into three main categories:

### 🌉 EM↔LM Integration (NEW!)
- **[NeuronBridge Guide](./core-features/NeuronBridge_Guide.md)**: Find matching EM neurons ↔ LM driver lines
- **[NeuronBridge Workflow Guide](./core-features/NeuronBridge_Workflow.md)**: Complete workflow with calling tree and recommendations ⭐
- **[NeuronBridge Co-Labeling Analysis](./core-features/NeuronBridge_Guide.md#co-labeling-analysis)**: Analyze specificity, selectivity, and overlap patterns between driver lines
- **[FlyLight Guide](./core-features/FlyLight_Guide.md)**: Download FlyLight imagery by driver line

### 🎨 [Visualizations](./visualizations/)
Comprehensive guides for all visualization types:
- **[Heatmap](./visualizations/Heatmap_Guide.md)**: Interactive connection matrices
- **[Network](./visualizations/Network_Guide.md)**: Graph-based connectivity visualization
- **[Sankey](./visualizations/Sankey_Guide.md)**: Flow diagrams showing connection magnitude
- **[3D Skeleton](./visualizations/3D_Skeleton_Guide.md)**: Three-dimensional neuron morphology

### 🔧 [Core Features](./core-features/)
Essential functionality documentation:
- **✨ NeuronBridge Integration**: EM↔LM mapping (NEW)
- **✨ [NeuronBridge Workflow](./core-features/NeuronBridge_Workflow.md)**: Complete workflow with calling tree ⭐
- **✨ FlyLight Downloader**: FlyLight imagery access (NEW)
- **✨ Module Calling Tree**: Visual architecture and dependency guide
- **✨ Connectivity Profiler**: 1-hop/2-hop hybrid profile building
- **✨ Homolog Finding**: Find homologous neurons across datasets
- **✨ Cross-Dataset Comparison**: Compare connectivity across multiple datasets
- **✨ [Graph Similarity Metrics](./core-features/GraphSimilarityMetrics_Documentation.md)**: Detailed metric explanations (Jaccard, Cosine, Edge Rank, etc.)
- **✨ Connectivity Profile Verification**: Verify neuron types using connectivity fingerprints
- **✨ [Connectivity Profiling](./CONNECTIVITY_PROFILING.md)**: Intra-dataset profile comparison with heatmap visualization (NEW)
- **Path Finding**: Multi-hop connection discovery
- **Custom Groups**: Flexible neuron grouping for custom analysis
- **Cache System**: High-performance local storage for repeat queries
- **Filtering**: Connection and neuron filtering options

### ⚙️ [Technical](./technical/)
Advanced technical documentation:
- **Performance Optimization**: Algorithm improvements and profiling
- **Data Formats**: Input/output specifications
- **Backend Architecture**: System design and implementation
- **Debugging**: Troubleshooting and development guides

---

## Quick Navigation

### For New Users
1. Start with [Installation Guide](./INSTALLATION.md)
2. Read [Quick Start Guide](./QUICK_START.md)
3. Explore [Visualization Overview](./visualizations/README.md)
4. Check [Troubleshooting Guide](./TROUBLESHOOTING.md) if you encounter issues

### For Researchers
- **[NeuronBridge Integration](./core-features/NeuronBridge_Guide.md)** - Find driver lines for EM neurons (NEW)
- **[NeuronBridge Workflow](./core-features/NeuronBridge_Workflow.md)** - Complete workflow with calling tree ⭐
- **[FlyLight Downloader](./core-features/FlyLight_Guide.md)** - Download LM imagery (NEW)
- **[Homolog Finding](./core-features/HomologFinding_Guide.md)** - Find homologs across datasets
- **[Cross-Dataset Comparison](./core-features/CrossDatasetComparison_Guide.md)** - Compare connectivity patterns
- **[Graph Similarity Metrics](./core-features/GraphSimilarityMetrics_Documentation.md)** - Understanding similarity metrics
- **[Connectivity Profiler Guide](./core-features/ConnectivityProfiler_Guide.md)** - Build connectivity profiles (1-hop/2-hop hybrid)
- [Heatmap Guide](./visualizations/Heatmap_Guide.md) - Quantitative analysis
- [Network Guide](./visualizations/Network_Guide.md) - Topology exploration
- [Path Finding](./core-features/README.md#path-finding) - Multi-hop connections

### For Developers
- [Technical Documentation](./technical/README.md) - System architecture
- [Core Features](./core-features/README.md) - API and algorithms
- [Archive](./archive/DOCUMENTATION_STRUCTURE.md) - Historical changes and fixes

---

## Key Documents

### Getting Started
- **[Installation](./INSTALLATION.md)**: Setup instructions and requirements
- **[Quick Start](./QUICK_START.md)**: Get running in 5 minutes
- **[Troubleshooting](./TROUBLESHOOTING.md)**: Common issues and solutions
- **[Output Files Reference](./OUTPUT_FILES.md)**: Detailed explanation of all generated files
- **[Artifact Citation Index](./ARTIFACT_INDEX.md)**: Cite a produced file → its topic, the `file:line` that writes it, the code that reads it, and the skill that covers it
- **[FlyWire-FAFB Integration](./FAFB_INTEGRATION.md)**: Setup guide for FAFB dataset
- **[BANC Integration](./BANC_INTEGRATION.md)**: Setup guide for the standalone BANC public release

### Visualization Guides
- **[Heatmap Guide](./visualizations/Heatmap_Guide.md)**: Complete heatmap documentation
- **[Network Guide](./visualizations/Network_Guide.md)**: Network visualization reference
- **[Sankey Guide](./visualizations/Sankey_Guide.md)**: Flow diagram documentation
- **[3D Skeleton Guide](./visualizations/3D_Skeleton_Guide.md)**: 3D rendering guide

### Core Functionality
- **✨ [Module Calling Tree](./core-features/Module_Calling_Tree.md)**: Visual architecture and dependency guide
- **✨ [Connectivity Profiler Guide](./core-features/ConnectivityProfiler_Guide.md)**: 1-hop/2-hop hybrid profile building
- **✨ [Homolog Finding](./core-features/HomologFinding_Guide.md)**: Find homologs across datasets
- **✨ [Cross-Dataset Comparison](./core-features/CrossDatasetComparison_Guide.md)**: Multi-dataset analysis
- **✨ [Connectivity Profiler Guide](./core-features/ConnectivityProfiler_Guide.md)**: Build cross-dataset connectivity profiles
- **✨ [Connectivity Profiling](./CONNECTIVITY_PROFILING.md)**: Intra-dataset profile comparison with heatmaps (NEW)
- **[Cache System Guide](./core-features/CacheSystem_Guide.md)**: Caching for fast repeat queries
- **[Path Finding](./core-features/FindAllPath_Documentation.md)**: Multi-hop path discovery
- **[Custom Groups](./core-features/CustomGroups_Feature.md)**: Flexible neuron grouping

---

## Documentation by Task

### Finding Connections
| Task                        | Documentation                                                                        |
| --------------------------- | ------------------------------------------------------------------------------------ |
| Direct connections          | [Main README](./core-features/BasicUsage_Guide.md)                                             |
| Multi-hop paths             | [FindAllPath](./core-features/FindAllPath_Documentation.md)                          |
| **Compare across datasets** | **[Cross-Dataset Comparison](./core-features/CrossDatasetComparison_Guide.md)**      |
| **Validate a type mapping (bodyId level)** | The **Cross-Dataset › Type Validation** tab — see [TypeMappingValidateExpandVisualize_Guide.md](./core-features/TypeMappingValidateExpandVisualize_Guide.md) §1b |
| **Build profiles**          | **[Connectivity Profiler Guide](./core-features/ConnectivityProfiler_Guide.md)**     |
| **Find homologs**           | **[Homolog Finding](./core-features/HomologFinding_Guide.md)**                       |
| **Intra-dataset profiling** | **[Connectivity Profiling](./CONNECTIVITY_PROFILING.md)**                            |
| Custom neuron groups        | [Custom Groups](./core-features/CustomGroups_Feature.md)                             |
| Forward-only paths          | [Forward-Only Guide](./core-features/ForwardOnly_Guide.md)                           |
| Filter connections          | [Connection Filters](./core-features/README.md#filtering)                            |

### Visualizing Data
| Task              | Documentation                                              |
| ----------------- | ---------------------------------------------------------- |
| Connection matrix | [Heatmap Guide](./visualizations/Heatmap_Guide.md)         |
| Network graph     | [Network Guide](./visualizations/Network_Guide.md)         |
| Flow diagram      | [Sankey Guide](./visualizations/Sankey_Guide.md)           |
| 3D anatomy        | [3D Skeleton Guide](./visualizations/3D_Skeleton_Guide.md) |

### Optimizing Performance
| Task             | Documentation                                                         |
| ---------------- | --------------------------------------------------------------------- |
| Enable caching   | [Cache Quick Start](./core-features/CacheSystem_QuickStart.md)        |
| Optimize queries | [Deep Backend Optimizations](./technical/DeepBackendOptimizations.md) |

### Customizing Output
| Task               | Documentation                                                              |
| ------------------ | -------------------------------------------------------------------------- |
| Custom colors      | [Custom Colors Guide](./visualizations/CUSTOM_COLORS_GUIDE.md)             |
| Layout algorithms  | [Advanced Layout Algorithms](./visualizations/AdvancedLayoutAlgorithms.md) |
| Heatmap clustering | [Heatmap Clustering](./visualizations/HEATMAP_CLUSTERING_FEATURE.md)       |
| Edge width control | [Edge Width Scaling](./visualizations/EDGE_WIDTH_SCALING.md)               |

---

## Common Workflows

### Workflow 1: Basic Connection Analysis
```
1. Find direct connections (FindDirect.py)
2. Visualize as heatmap
3. Export results
```
**Documentation**: [README](./core-features/BasicUsage_Guide.md) → [Heatmap Guide](./visualizations/Heatmap_Guide.md)

### Workflow 2: Pathway Discovery
```
1. Find multi-hop paths (FindAllPath)
2. Visualize as network + Sankey
3. Apply clustering to heatmap
4. Export layouts
```
**Documentation**: [FindAllPath](./core-features/FindAllPath_Documentation.md) → [Network Guide](./visualizations/Network_Guide.md)

### Workflow 3: High-Performance Analysis
```
1. Setup cache (first time)
2. Run large-scale queries
3. Generate visualizations
```
**Documentation**: [Cache Setup](./core-features/CacheSystem_QuickStart.md) → [Performance Optimizations](./technical/PERFORMANCE_OPTIMIZATIONS_DEC2025.md)

### Workflow 4: Publication Figures
```
1. Generate visualizations
2. Customize colors and layout
3. Export high-resolution PNG/SVG
4. Integrate with other analysis
```
**Documentation**: [Visualization Overview](./visualizations/README.md) → Individual visualization guides

---

## Visualizer Changelog

### Latest Features
- ✨ Multiple clustering algorithms for heatmaps (Ward, Average, Complete, Single)
- ✨ Enhanced edge-list format with flexible column recognition
- ✨ Expanded canvas for better visualization space
- ✨ Reorganized UI controls for better usability

### Recent Improvements
- ⚡ Cache v4 with pair-level optimization
- ⚡ Polars integration for 10-100x faster CSV/matrix operations
- 🎨 Enhanced color customization for all visualizations
- 📊 Better heatmap clustering with algorithm selection

---

## Archive

Historical documentation and fixes are in the [archive](./archive/DOCUMENTATION_STRUCTURE.md) directory:
- Bug fixes and corrections
- Implementation summaries
- Reorganization history
- Deprecated features

---

## External Resources

### Data Sources
- **NeuPrint**: https://neuprint.janelia.org/
- **FlyEM**: https://www.janelia.org/project-team/flyem
- **Hemibrain Paper**: Scheffer et al. (2020), eLife

### Related Tools
- **NAVIS**: https://navis.readthedocs.io/
- **neuPrint Python**: https://connectome-neuprint.github.io/neuprint-python/
- **FlyWire**: https://flywire.ai/

### Visualization Libraries
- **Plotly**: https://plotly.com/python/
- **Cytoscape.js**: https://js.cytoscape.org/
- **NetworkX**: https://networkx.org/

---

## Getting Help

### Documentation Issues
- Check the specific guide for your task
- Look in [troubleshooting sections](#)
- Review [examples directory](../archive/examples/)

### Technical Support
- [GitHub Issues](https://github.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/issues)
- Check [Technical Documentation](./technical/README.md)
- Review [FAQ](#) (coming soon)

### Contributing
- See [Contributing Guide](../README.md#contributing)
- Read [Technical Documentation](./technical/README.md)
- Follow coding standards

---

## Documentation Maintenance

### Last Updated
- Major reorganization: 2025-11-05
- Visualization guides created: 2025-11-05
- Test cleanup: 2025-11-05

### Version
Documentation version 4.0 (matches software version)

### Feedback
Documentation feedback welcome via GitHub Issues or pull requests.

---

## Quick Links

**Essential Reading**:
- [Installation](./INSTALLATION.md)
- [Quick Start](./QUICK_START.md)
- [Visualization Overview](./visualizations/README.md)

**Visualization Guides**:
- [Heatmap](./visualizations/Heatmap_Guide.md) | [Network](./visualizations/Network_Guide.md) | [Sankey](./visualizations/Sankey_Guide.md) | [3D](./visualizations/3D_Skeleton_Guide.md)

**Core Features**:
- [Cache System](./core-features/CacheSystem_Guide.md) | [Path Finding](./core-features/FindAllPath_Documentation.md) | [Module Calling Tree](./core-features/Module_Calling_Tree.md)

**Technical**:
- [Optimization](./technical/DeepBackendOptimizations.md) | [Architecture](./technical/README.md) | [Data Formats](./technical/COLUMN_RECOGNITION_UPDATE.md)
