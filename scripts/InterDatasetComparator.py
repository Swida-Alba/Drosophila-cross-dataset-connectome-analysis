"""
Example: Cross-Dataset Comparison Analysis

This example demonstrates the complete workflow for comparing connectivity 
across multiple datasets using the comparison module.

Features demonstrated:
- ComparisonParameters as primary entry point
- Two-dataset comparison (hemibrain vs male-cns)
- Intermediate layer analysis (max_interlayer=1)
- Automatic export of all comparison results
- Interactive HTML report with Cytoscape.js network visualization
- Automatic matplotlib visualizations
- PDF export capability
- Comparison modes: 'path' (path-based) vs 'edge' (edge-based)
- **NEW**: Direct comparison of specific neurons (direct_comparison)
- **NEW**: Connectivity profile comparison as SEPARATE step (connectivity_profile_comparison)
- **NEW**: Standalone conserved path visualization with multi-dataset synapse labels
- **NEW**: Unified JSON label mapping (source, target, intermediate in one file)


Comparison Modes:
  - 'path' (default): Edges are discovered through path traversal. May miss 
    strong edges if they're on paths with weak intermediate connections.
  - 'edge': Edges are evaluated independently by weight. Provides direct 
    comparison of synaptic connectivity but loses path-level circuit structure.

Direct Comparison:
  - Compares specific neurons by type name or bodyId
  - Uses ProfileComparator.direct_comparison() under the hood
  - Supports LabelMapper for cross-dataset type name resolution
  - Example: analyzer.direct_comparison('aMe12', 'aMe12')

Connectivity Profile Comparison:
  - Compares all neuron types from comparison results across datasets
  - Uses multiple similarity metrics (Jaccard, Cosine, Rank correlation)
  - Generates confidence scores for type assignments
  Parameters can be configured in ComparisonParameters or passed directly.

Conserved Path Visualization:
  - Visualizes edges conserved across ALL datasets using VisualizePath module
  - Trims dead-end nodes that don't connect source to target
  - Hover labels show synapse strengths from each dataset
  - Example: analyzer.visualize_conserved_paths(threshold=5, trim_dead_ends=True)

Label Mapping:
  - Supports unified JSON file containing source, target, and intermediate mappings
  - Example: LabelMapper(overall_mapping_json='path/to/all_mappings.json')
  - Also supports separate CSV/JSON files or direct dictionary input
"""
# DROCAT edit-to-run TEMPLATE, not a CLI. It mirrors the UI's
# "inter_dataset" tool (registry
# ui/runner.py TOOL_REGISTRY; payload built in ui/tabs/inter_dataset.py), and its
# values below are pinned to ui/config.py DEFAULTS so an unedited
# run reproduces what the tab sends. Copy to
# archive/scripts_local/ before a real scientific run
# (skills/drocat-usage/SKILL.md:66); the relative paths assume a
# working directory of scripts/.

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from comparison import (
    ComparisonAnalyzer,
    ComparisonParameters,
    LabelMapper,
)


def run_comprehensive_comparison():
    """
    Comprehensive comparison example demonstrating all features.
    
    This example:
    1. Compares hemibrain and male-cns datasets
    2. Analyzes aMe12 → PPL101 connectivity with 1 intermediate layer
    3. Tests multiple thresholds (1, 3, 5, 10)
    4. Exports all comparison results
    5. Generates interactive HTML report with Cytoscape.js network
    6. Generates matplotlib visualizations
    7. Runs connectivity profile verification (optional, separate step)
    """
    
    # =========================================================================
    # Step 1: Create ComparisonParameters
    # =========================================================================
    print("\n📋 Step 1: Creating ComparisonParameters...")
    
    label_map = LabelMapper(
        source_mapping_dict={
            'flywire_FAFB_v783': ['CB0890'],
            'male-cns:v0.9': ['GNG458'],
        },
        source_labels=['Foxglove'],
        target_mapping_dict={
            'flywire_FAFB_v783': ['LPLC2'],
            'male-cns:v0.9': ['LPLC2'],
        },
        target_labels=['LPLC2'],
    )
    
    # Unified JSON mapping file (contains source, target, and/or intermediate mappings)
    # Structure:
    # {
    #   "source_mapping": { "custom_label": [...], "dataset1": [...], ... },
    #   "target_mapping": { ... },
    #   "intermediate_mapping": { ... }
    # }
    # label_map = LabelMapper(
    #   overall_mapping_json='../local_data/dataset_comparison/comparison_results_20251227_213819_Fdg-LPLC2-5hops/label_map.json'
    # )
    
    params = ComparisonParameters(
        # Token automatically loaded from config_local.json (recommended) or set token='' here
        # Output settings
        output_folder='../local_data/dataset_comparison',
        saveas=None,  # Auto-generate timestamp folder
        
        # Datasets to compare
        datasets=['male-cns:v0.9', 'flywire_FAFB_v783'],
        datasets_nickname=['MCNS', 'FAFB'],
        
        # source_neurons=['aMe.*'],
        source_neurons=label_map,
        
        # target_neurons=['PPL1.*'],
        target_neurons=label_map,
        
        max_interlayer=2,
        
        # Multiple thresholds to analyze sensitivity
        thresholds=[3, 5, 10, 15, 20],
        # thresholds = list(range(1,21)),
        
        # Automatic type mapping across datasets (experimental)
        auto_type_mapping=False,
        
        # Top edges/paths to retain in comparison report tables
        top_edges=500,
        
        # Comparison mode: 'path' (path-based filtering) or 'edge' (edge-based filtering)
        # - 'path': Discovers edges through paths; may miss strong edges on weak paths
        # - 'edge': Evaluates each edge independently by weight
        comparison_mode='path',  # Change to 'edge' to use edge-based comparison
        
        skip_bodyId=True, # Skip bodyId-level results for speed and local storage
        
        # Performance Settings (path finding algorithm)
        # -----------------------------------------------------------------
        pathfinding='MemoizedDFS', # 'MemoizedDFS' (default, fastest), 'DFS' (backward), 'MeetInMiddle', 'DP', 'Bidirectional'
        
        # Offline/Cache Mode
        # -----------------------------------------------------------------
        # cache_only=True: Use ONLY local cache, never connect to NeuPrint server.
        #   Useful for deprecated datasets (e.g., male-cns:v0.9) or offline work.
        #   Requires prior cache population. Raises error if cache is insufficient.
        # cache_only=False (default): Try server first, auto-fallback to cache if
        #   server is unavailable but cache exists.
        cache_only=True,  # Enable for deprecated datasets like male-cns:v0.9
    )
    
    analyzer = ComparisonAnalyzer(params, verbose=True)
    results = analyzer.run_comparison()
    analyzer.generate_report()
    analyzer.export_results()
    
    ### ====== optional: standalone visualization of conserved paths ======
    
    # Visualize conserved edges using VisualizePath with dead-ends trimmed
    # Hover labels show synapse strengths from all datasets
    # Generates one network visualization per threshold in conserved_paths/ subfolder
    analyzer.visualize_conserved_paths_all_thresholds(
        trim_dead_ends=True,  # Remove nodes not on source→target paths
        showfig=False,        # Set to True to open in browser
    )
    
    # Or generate for a single threshold:
    # analyzer.visualize_conserved_paths(threshold=5, trim_dead_ends=True)
    
    ### ====== optional comparison of connectivity profiles ======
    
    comparison_results = analyzer.connectivity_profile_comparison()
    
    ### ============================================================
    
    return analyzer


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    import time
    # time it
    
    start_time = time.time()
    try:
        analyzer = run_comprehensive_comparison()
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n⏱️ Total elapsed time: {elapsed/60:.2f} minutes")
