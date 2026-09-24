#!/usr/bin/env python3
"""
NeuronBridge_Colabel.py - Co-Labeling Analysis for Driver Lines

This script performs comprehensive co-labeling analysis to understand how different
GAL4/Split-GAL4 driver lines overlap in their neuron labeling patterns.

Usage:
    Edit the parameters in the script and run directly:
    python NeuronBridge_Colabel.py

Key Features:
    - Analyze labeling overlap between multiple driver lines
    - Build expression matrix (Types × Lines) with match scores
    - Compute co-labeling similarity matrices (Jaccard, Weighted Jaccard)
    - Calculate line specificity (sparsity) metrics
    - Generate interactive heatmap visualizations
    - Create comprehensive HTML analysis report
    - 3D skeleton visualization of top co-labeled neuron types

Output Files:
    - expression_matrix.csv: Full Type × Line matrix with match scores
    - expression_matrix_viz.csv: Truncated/filtered matrix for visualization
    - expression_matrix.html: Interactive heatmap visualization
    - expression_matrix_merged.csv: Full merged matrix (types across datasets)
    - expression_matrix_merged_viz.csv: Truncated/filtered merged matrix
    - expression_matrix_merged.html: Interactive merged heatmap
    - colabeling_matrix_{method}.csv: Line × Line similarity matrix
    - colabeling_matrix_{method}.html: Interactive similarity heatmap
    - line_labeled_neurons/{line}_neurons.csv: Per-line neuron details
    - line_labeled_neurons/{line}_{dataset}_neurons.csv: Dataset-split neurons
    - line_labeled_neurons/{line}_{dataset}_types.csv: Type summary per dataset
    - line_summary.csv: Summary statistics per line
    - colabeling_report.html: Comprehensive analysis report
    - plot-3d_{dataset}/: 3D visualization folder (if visualize_top_n > 0)
      - {dataset}.html: Interactive 3D skeleton visualization
      - exported_views/: PNG exports (front, back, top, bottom, left, right)
      - individual_profiles/: Per-neuron PNG profiles + PDF summary

Metrics Explained:
    - Expression Matrix: NeuronBridge match scores for each type-line pair
    - Jaccard Similarity: |A ∩ B| / |A ∪ B| - binary type overlap
    - Weighted Jaccard: Score-weighted type overlap (accounts for match confidence)
    - Sparsity: 1 - (fraction of lines with similar labeling pattern)
      High sparsity = more unique/specific line

Use Cases:
    1. Find complementary lines that label different neuron populations
    2. Identify redundant lines that label similar neurons
    3. Assess specificity of driver lines for experimental planning
    4. Design intersectional genetics experiments (Split-GAL4)

Author: DROCAT Project
"""
# DROCAT edit-to-run TEMPLATE, not a CLI. It mirrors the UI's
# "nb_colabel" tool (registry
# ui/runner.py TOOL_REGISTRY; payload built in ui/tabs/nb_colabel.py), and its
# values below are pinned to ui/config.py DEFAULTS so an unedited
# run reproduces what the tab sends. Copy to
# archive/scripts_local/ before a real scientific run
# (skills/drocat-usage/SKILL.md:66); the relative paths assume a
# working directory of scripts/.

import sys
import time
from pathlib import Path

# Add repo src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from neuronbridge_finder import NeuronBridgeFinder

if __name__ == "__main__":
    t0 = time.time()
    # ==========================================================================
    # CONFIGURATION - Edit these parameters
    # ==========================================================================
    
    # Driver lines to analyze - can be:
    #   - Multiple as string: 'LH173,VT037867,SS00731'
    #   - Multiple as list: ['LH173', 'VT037867', 'SS00731']
    # 
    # At least 2 lines are required for co-labeling analysis
    lines = [
        'VT037867',
        'VT037866',
        'SS01015',
    ]
    
    # Match algorithm: 'cds' (Color Depth Search), 'pppm', or 'both'
    match_type = 'cds'
    
    # Number of top ranked neuron matches to consider per line. The score
    # cutoff is applied after this top-N selection.
    top_n_neurons = 50
    
    # Similarity methods for co-labeling matrix
    # Options: 'jaccard', 'weighted_jaccard', 'rank_correlation'
    # - 'jaccard': Binary presence/absence of types (simpler)
    # - 'weighted_jaccard': Accounts for match scores (recommended)
    # - 'rank_correlation': Spearman correlation of type rankings
    similarity_methods = ['jaccard', 'weighted_jaccard']
    
    # Output directory (set to None for no file output)
    output_dir = '../local_data/neuronbridge_finding'
    
    # ==========================================================================
    # 3D SKELETON VISUALIZATION - Visualize top N co-labeled types
    # ==========================================================================
    
    # Visualize top N types per dataset using 3D skeleton (0 = disabled)
    # This creates interactive HTML visualizations showing the top N neuron types
    # Output folder: plot-3d_{dataset}/ (no timestamp, overwrites previous)
    visualize_top_n = 5
    
    # Generate individual profile PNGs for each neuron type
    # Creates a subfolder 'individual_profiles' with one PNG per type + PDF summary
    # PDF uses natural sorting: r1, r2, ..., r9, r10 (not r1, r10, r11...)
    generate_individual_profiles = True
    
    # PDF layout for individual profiles (columns, rows)
    # Default: (3, 2) = 3 columns x 2 rows = 6 images per page
    pdf_images_per_page = (3, 2)
    
    # Background color for 3D visualization
    # Options: 'white', 'black', or any CSS color (e.g., '#f0f0f0', 'lightgray')
    # Default: 'white'
    background_color = 'white'
    
    # ==========================================================================
    # VISUALIZATION FILTERING - Filter which types/datasets to visualize
    # ==========================================================================
    
    # Filter neuron types for visualization by name pattern
    # Uses a dict with filter type as key and pattern(s) as value
    # Filter types: 'contains', 'startswith', 'endswith', 'regex'
    # - Multiple patterns within same filter key: OR logic (match any)
    # - Across different filter keys: AND logic (must match all)
    # - The rank r{N} preserves original ranking BEFORE filtering
    # - Gets ALL types first, filters, then takes top N from filtered results
    # Examples:
    #   type_filter = {'contains': 'DN'}           # Types containing 'DN'
    #   type_filter = {'startswith': ['DN', 'IN']} # Types starting with 'DN' or 'IN'
    #   type_filter = {'endswith': '_R'}           # Types ending with '_R'
    #   type_filter = {'regex': r'DN[a-z]\d+'}     # Types matching regex pattern
    #   type_filter = {'contains': 'DN', 'endswith': '_R'}  # Both conditions (AND)
    # Set to None or {} to disable filtering (visualize all top types)
    # Note: This filter applies to:
    #   - 3D skeleton visualization (types shown)
    #   - expression_matrix_viz.csv and expression_matrix_merged_viz.csv
    #   - Expression matrix HTML heatmaps
    type_filter = None
    
    # Constrain which datasets to visualize
    # Options:
    #   - 'all' or None: Visualize all datasets found in results (default)
    #   - List of dataset names: ['hemibrain:v1.2.1', 'manc:v1.0']
    #   - Single dataset: 'hemibrain:v1.2.1'
    # Dataset names should match exactly as they appear in the results
    datasets_to_visualize = None
    
    # ==========================================================================
    # SCORE FILTERING - Filter low-confidence neurons and types
    # ==========================================================================
    
    # Minimum score threshold for expression/labeling matrices and similarity
    # calculations. Raw top-N per-line tables retain lower-scoring matches.
    # Default: 30000. Set to 0 to disable the score cutoff.
    min_score = 30000
    
    # Minimum average score threshold for types in SIMILARITY matrix (clustering)
    # Types with average score < threshold may be excluded from clustering
    # Note: Expression matrix is filtered by min_score, not by this threshold
    # Default: 10000. Set to 0 to include all types in clustering
    min_type_avg_score = 10000
    
    # ==========================================================================
    # ADVANCED OPTIONS
    # ==========================================================================
    
    # Path to datasets folder for neuron enrichment (None = auto-detect)
    datasets_path = None
    
    # Enable result caching (speeds up repeated analyses)
    use_cache = True
    
    # Verbose output
    verbose = True
    
    # Anatomical region filter: 'Brain', 'VNC', or 'All'
    region = 'Brain'
    
    # ==========================================================================
    # EXECUTION - No need to edit below this line
    # ==========================================================================
    
    # Initialize finder
    finder = NeuronBridgeFinder(
        datasets_path=datasets_path,
        use_cache=use_cache,
        verbose=verbose,
        match_type=match_type,
        region=region,
    )
    
    # Run co-labeling analysis
    results = finder.analyze_colabeling(
        lines=lines,
        match_type=match_type,
        top_n_neurons=top_n_neurons,
        similarity_methods=similarity_methods,
        output_dir=output_dir,
        visualize_top_n=visualize_top_n,
        generate_individual_profiles=generate_individual_profiles,
        pdf_images_per_page=pdf_images_per_page,
        min_score=min_score,
        min_type_avg_score=min_type_avg_score,
        background_color=background_color,
        type_filter=type_filter,
        datasets_to_visualize=datasets_to_visualize,
    )
    
    # Display summary
    if results:
        print(f"\n{'='*60}")
        print("📊 Results Summary")
        print('='*60)
        
        # Expression matrix info
        if results.get('expression_matrix') is not None:
            expr = results['expression_matrix']
            print(f"\n   Expression Matrix: {expr.shape[0]} types × {expr.shape[1]} lines")
            print(f"   Non-zero entries: {(expr > 0).sum().sum()}")
        
        # Co-labeling matrices
        if results.get('colabeling_matrices'):
            print(f"\n   Co-labeling Matrices:")
            for method, matrix in results['colabeling_matrices'].items():
                # Count significant pairs (similarity > 0.1)
                n_pairs = ((matrix > 0.1) & (matrix < 1.0)).sum().sum() // 2
                mean_sim = matrix.values[matrix.values < 1.0].mean()
                print(f"      {method}: {n_pairs} significant pairs, mean similarity={mean_sim:.3f}")
        
        # Line summary
        if results.get('line_summary') is not None:
            summary = results['line_summary']
            print(f"\n   Line Statistics:")
            print(f"      Total neurons across all lines: {summary['n_neurons'].sum()}")
            print(f"      Avg neurons per line: {summary['n_neurons'].mean():.1f}")
            print(f"      Avg types per line: {summary['n_types'].mean():.1f}")
            
            # Most specific line
            if 'colabel_sparsity' in summary.columns:
                most_specific = summary.loc[summary['colabel_sparsity'].idxmax()]
                print(f"\n   Most Specific Line: {most_specific['line']}")
                print(f"      Sparsity: {most_specific['colabel_sparsity']:.3f}")
                print(f"      Types: {most_specific['n_types']}")
        
        # Report path
        if results.get('report_path'):
            print(f"\n   📝 Full Report: {results['report_path']}")
    
    print("\n✅ Done!")
    t1 = time.time()
    elapsed = t1 - t0
    print(f"⏱️ Elapsed time: {elapsed/60:.2f} minutes\n")
