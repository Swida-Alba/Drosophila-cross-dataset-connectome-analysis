#!/usr/bin/env python
"""
ConnectivityProfiling - Intra-dataset connectivity profile comparison and visualization

This script compares connectivity profiles within a single dataset, supporting:
- Type-level comparisons (aggregated profiles via mean pooling)
- BodyId-level comparisons (individual neuron profiles)
- Type-avg-bodyId comparisons (type similarities from averaged bodyId pairs)
- Interactive heatmap visualization with native clustering

Key Features:
    - Flexible query input: simple list, nested list with custom names, or CSV file
    - ALL similarity metrics output: jaccard, cosine, rank_corr, rank_corr_union
    - Separate heatmap files for EACH metric
    - Separate upstream/downstream and combined analysis
    - Interactive heatmap via VisualizePath with Ward clustering
    - Saves individual and aggregated connectivity profiles

Query Input Formats:
    1. Simple list: ['Mi1', 'Tm3', 720575940610453042]
    2. Nested list with custom group names (like VisualizeSkeleton's neuron_layers):
        [['DN1p', ['DN1pA', 'DN1pB']], ['DN2', ['DN2']], ['l-LNv', [12345]]]
    3. CSV file via group_map_csv parameter (like VisualizeSkeleton's layer_map_csv):
        CSV format: columns 'group' and 'id_type_instance'

Profile Construction (consistent with ConnectivityProfiler):
    - top_k: Top K partners per direction by synapse weight (default: 15)
    - top_m: Minimum unique partner types to ensure (default: 5)
    - Dynamic expansion: If top_k yields < top_m types, expand K

Output Structure:
    {output_dir}/profiling_{query_name}_{timestamp}/
    ├── parameters.json
    ├── README.txt
    ├── profiles/
    │   ├── individual/          # Individual bodyId connectivity profiles
    │   └── aggregated/          # Type-aggregated profiles
    ├── type_level/
    │   ├── results/             # Type-aggregated similarity matrices
    │   └── visualization/       # Type-level heatmaps
    └── bodyid_level/
        ├── results/
        │   ├── bodyid_similarity_{metric}_{direction}.csv
        │   └── type_avg_bodyid_similarity_{metric}_{direction}.csv
        └── visualization/
            ├── heatmap_bodyid_{direction}_{metric}.html
            └── heatmap_type_avg_{direction}_{metric}.html

"""

import sys
from pathlib import Path

# Add repo src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from comparison.profile_comparator import ConnectivityProfileComparer

if __name__ == "__main__":
    
    # Optional: get types matching a pattern
    # type_list, _, _ = get_types('aMe.*', dataset='male-cns:v0.9')
    # print(type_list)
    
    # Create comparer instance with all settings
    
    comparer = ConnectivityProfileComparer(
        # --- Query (types, bodyIds, or patterns) ---
        # query=type_list,  # List of neuron types/bodyIds/patterns
        query=['aMe12', 'aMe10', 'aMe9'],
        
        # --- Dataset ---
        # `dataset` is the intra-dataset form; the ctor also takes
        # `datasets=[...]` for the cross-dataset N×M comparison the
        # Connectivity tab builds (ui/tabs/connectivity.py:579), with
        # `aggregation_level` ('type' default) and `use_cache` (True).
        dataset='male-cns:v0.9',
        # dataset='flywire_FAFB_v783',
        # datasets=['male-cns:v0.9', 'hemibrain:v1.2.1'],   # cross-dataset

        # --- Profile Construction ---
        # 25 is what the Connectivity tab sends (ui/config.py DEFAULTS);
        # the class itself defaults to 15, so the old value here profiled
        # fewer partners per direction than the same query in the UI.
        top_k=25,  # Top K partners per direction
        top_m=5,   # Minimum unique types to ensure
        
        # --- Comparison ---
        direction='both',  # 'upstream', 'downstream', or 'both'
        # Note: ALL metrics are computed automatically (jaccard, cosine, rank_corr, rank_corr_union)
        # Note: Both type-level AND bodyId-level comparisons are always performed
        
        # --- Output ---
        output_dir='../local_data/connectivity_profiling',
        
        # --- Visualization ---
        generate_heatmaps=True,
        show_figures=False,
        
        # --- Other ---
        verbose=True,
        
        skip_bodyId_level=False,  # Set to True to skip bodyId-level analysis for speed
    )
    
    # Run comparison
    results = comparer.run()
