#!/usr/bin/env python3
"""
NeuronBridge_FindLines.py - Find LM driver lines matching EM neurons

This script searches NeuronBridge to find GAL4 driver lines that express
neurons matching the morphology of a given EM body ID or neuron type.

Usage:
    Edit the parameters in the script and run directly:
    python NeuronBridge_FindLines.py

Key Features:
    - Search by body ID (integer), type name, or instance name
    - Multiple queries supported as comma-separated string or list
    - Regex patterns supported for type/instance matching
    - LabelMapper support for cross-dataset unified naming
    - Match algorithms: 'cds' (Color Depth Search), 'pppm', or 'both'
    - Multi-dataset search: when dataset=None, searches all available datasets
    - Cross-dataset scoring: calculates min_score_per_dataset for multi-dataset ranking
    - Line type separation: GAL4/LexA vs Split-GAL4 with separate top_n limits
    - Optional image download from NeuronBridge or FlyLight
    - Results saved to CSV with driver line information

Output Files:
    - {query}_lines.csv: Matched driver lines with scores per query
    - all_lines.csv: Combined results from all queries (row-level matches)
    - line_summary.csv: Aggregated stats per line, SORTED BY weighted_score
    - gal4_lexa_summary.csv: GAL4/LexA lines only, SORTED BY weighted_score
    - split_gal4_summary.csv: Split-GAL4 lines only, SORTED BY weighted_score
    - images/: Downloaded images (if download_images enabled)
    - images_summary.pdf: PDF summary of downloaded images (pages ordered by weighted_score ranking)

Output Columns in *_summary.csv (sorted by weighted_score descending):
    Core columns:
    - line: Driver line name (e.g., VT000770, SS00001)
    - agg_mean_score: Average NeuronBridge match score across all matched neurons
    - agg_max_score: Maximum NeuronBridge match score for this line
    - match_count: Number of UNIQUE bodyIds labeled by this line
    - matched_bodyIds: Comma-separated list of unique bodyIds
    - matched_types: Comma-separated list of unique neuron types labeled
    
    Scoring columns:
    - coverage_ratio: match_count / total_query_neurons (fraction of queried neurons labeled)
    - weighted_score: agg_mean_score × coverage_ratio (PRIMARY SORTING KEY)
                     Higher = better line for labeling ALL queried neurons
    
    Multi-dataset columns (when querying multiple datasets):
    - datasets_labeled: Number of datasets where this line labels queried neurons
    - matched_datasets: Comma-separated list of matched dataset names
    - min_score_per_dataset: Minimum of max scores across datasets
    - cross_dataset_score: Mean of max scores across datasets
    
    Line type column (when separate_splitgal4=True):
    - line_type: 'gal4_lexa' or 'split_gal4'

Weighted Score Calculation:
    weighted_score = agg_mean_score × (match_count / total_query_neurons)
    
    This scoring prioritizes lines that:
    1. Have high average matching scores (good morphological match)
    2. Label MORE of the queried neurons (high coverage)
    
    Example: When querying 'aMe12' across 3 datasets with 15 total neurons:
    - Line A: agg_mean_score=45000, match_count=15 → weighted_score=45000×(15/15)=45000
    - Line B: agg_mean_score=50000, match_count=2 → weighted_score=50000×(2/15)=6666
    Line A ranks higher because it labels ALL queried neurons, even though Line B
    has a higher raw score for the neurons it does label.

Multi-Type Query Behavior:
    When querying multiple types together (e.g., 'aMe12,MBON01'), the program
    finds lines that label ALL queried neuron types. The weighted_score ensures
    lines labeling more types rank higher.
    
    ⚠️ IMPORTANT: If you want to find lines labeling DIFFERENT groups of neurons
    separately, DO NOT query them together. Instead, run separate queries:
    - Query 1: 'aMe12' → finds best lines for aMe12
    - Query 2: 'MBON01' → finds best lines for MBON01
    
    Querying 'aMe12,MBON01' together finds lines labeling BOTH types.

Specificity/Selectivity Analysis:
    For detailed analysis of how specific each driver line is to your target
    neuron types, use NeuronBridge_Colabel.py after finding lines.
    This provides:
    - Co-labeling matrices showing overlap between lines
    - Expression matrices showing which types each line labels
    - Specificity score distributions
    - Detailed per-line neuron breakdowns

Author: DROCAT Project
"""
# DROCAT edit-to-run TEMPLATE, not a CLI. It mirrors the UI's
# "nb_find_lines / nb_find_lines_expanded" tool (registry
# ui/runner.py TOOL_REGISTRY; payload built in ui/tabs/nb_find_lines.py), and its
# values below are pinned to ui/config.py DEFAULTS so an unedited
# run reproduces what the tab sends. Copy to
# archive/scripts_local/ before a real scientific run
# (skills/drocat-usage/SKILL.md:66); the relative paths assume a
# working directory of scripts/.

import os
import sys
import time
import pandas as pd
from pathlib import Path

# Add repo src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from neuronbridge_finder import NeuronBridgeFinder

# Optional: Import LabelMapper for cross-dataset unified naming
try:
    from comparison.label_mapper import LabelMapper
    HAS_LABELMAPPER = True
except ImportError:
    HAS_LABELMAPPER = False
    LabelMapper = None

if __name__ == "__main__":
    t0 = time.time()
    # ==========================================================================
    # CONFIGURATION - Edit these parameters
    # ==========================================================================
    
    # Neuron query - can be:
    #   - Body ID (integer): 636798093
    #   - Type name: 'aMe12'
    #   - Instance name: 'aMe12_R'
    #   - Regex pattern: 'MBON.*'
    #   - Multiple as string: 'aMe12,MBON01,636798093'
    #   - Multiple as list: ['aMe12', 'MBON01', 636798093]
    #   - LabelMapper object for cross-dataset unified naming (see below)
    
    # ==========================================================================
    # LABELMAPPER SUPPORT (Optional - for cross-dataset unified naming)
    # ==========================================================================
    # Use LabelMapper to unify neuron naming across datasets. This allows you to:
    #   - Define equivalent neuron types across different datasets
    #   - Search for the same logical neuron type using dataset-specific names
    #   - Get unified results with standardized type names
    #
    # Example usage:
    #   mapper = LabelMapper(overall_mapping_json='path/to/mapping.json')
    #   query = mapper  # Pass mapper directly as query
    #   dataset = None  # Will be auto-derived from mapper
    #
    # Example mapping JSON format:
    # {
    #     "source_mapping": {
    #         "custom_label": ["aMe12_grp"],
    #         "hemibrain:v1.2.1": [["aMe12", "aMe12_R"]],
    #         "male-cns:v0.9": [["aMe12", "aMe12-like"]]
    #     }
    # }
    #
    # Uncomment below to use LabelMapper:
    # if HAS_LABELMAPPER:
    #     mapper = LabelMapper(overall_mapping_json='path/to/your/mapping.json')
    #     query = mapper
    #     dataset = None  # Auto-derived from mapper
    query = 'aMe12'
    
    # Dataset to search in (for type/instance lookups)
    # Options: 'hemibrain:v1.2.1', 'male-cns:v0.9', 'flywire_FAFB_v783', etc.
    # Set to None to search ALL available datasets (recommended for broad search)
    # Note: When using LabelMapper, dataset is auto-derived from the mapper
    dataset = ['male-cns:v0.9', 'hemibrain:v1.2.1', 'flywire_FAFB_v783']
    
    # Match algorithm: 'cds' (Color Depth Search), 'pppm', or 'both'
    match_type = 'cds'
    
    # Output directory (set to None for stdout only)
    output_dir = '../local_data/neuronbridge_finding'
    
    # Enable result caching
    use_cache = True
    
    # Verbose output
    verbose = True
    
    # Maximum number of parallel workers for processing neurons
    # Higher values speed up processing but may hit API rate limits
    # Default: 8, set to 1 for sequential processing
    max_workers = 8
    
    # ==========================================================================
    # IMAGE DOWNLOAD OPTIONS (Optional)
    # ==========================================================================
    
    # Region of interest for downloading images
    # 'Brain', 'VNC', or 'All' for both (default is 'All')
    region = 'Brain'
    
    # Download images only for top N lines (by aggregate score/rank)
    # Set to None to download for all lines
    download_img_for_top_n_lines = 20
    
    # Sort by 'max' score or 'completeness' 
    sort_by = 'max'
    
    # File formats to download
    # For neuronbridge: 'png', 'jpg'
    # For flylight: 'png', 'jpg', 'h5j', 'mp4', 'all'
    image_formats = ['png','jpg']
    
    # Maximum images to download per line (None = no limit)
    max_download_images_per_line = 6
    
    # FlyLight collection category (only used when download_images='flylight' or 'both')
    # Options (case-insensitive): 'GAL4/LEXA', 'SplitGAL4', 'MCFO', 'RawImages', 'All'
    # Can also be a list: ['GAL4/LEXA', 'SplitGAL4']
    # None = search all collections
    #
    # ** IMPORTANT: Categories are searched in PRIORITY ORDER **
    # Images are collected from each category sequentially until max_download_images_per_line
    # is reached. For example, with category=['GAL4/LEXA', 'SplitGAL4', 'MCFO']:
    #   1. First search GAL4/LEXA collection
    #   2. If not enough images, search SplitGAL4 collection  
    #   3. If still not enough, search MCFO collection
    #
    # ** MCFO FALLBACK: ** If a line has NO images in the specified categories,
    # 'MCFO' is automatically searched as a fallback (NeuronBridge CDM images
    # are typically in the MCFO collection).
    flylight_category = ['GAL4/LEXA', 'SplitGAL4', 'MCFO']
    
    # Simple mode for FlyLight downloads - reduces download volume by filtering filenames
    # If True:
    #   - Split-GAL4 collections: only download files with '20x' AND 'multichannel' in filename
    #   - GAL4/LexA collections: only download files with 'total' in filename
    # If False: download all matching files (default)
    simple_mode = True
    
    # Separate GAL4/LexA from Split-GAL4 lines (affects download and results)
    # If True:
    #   - Results will include a 'line_type' column ('gal4_lexa
    #     or 'split_gal4')
    #   - download_img_for_top_n_lines applies separately to each category
    # If False: all lines treated together (default)
    separate_splitgal4 = True
    
    # Summary format for image aggregation
    # Options:
    #   - 'pdf' or ['pdf']: Generate PDF only
    #   - 'pptx' or ['pptx']: Generate PPTX only
    #   - ['pdf', 'pptx']: Generate both formats
    #   - False or None: Disable generation
    summary_format = ['pdf', 'pptx']
    
    # Background color for PDF/PPTX summary files
    # Options: 'black', 'white', '#RRGGBB' hex color, or (R, G, B) tuple (0-255)
    # Default: 'black' - uses dark background with white text for better contrast
    summary_background_color = 'black'
    
    # ==========================================================================
    # EXECUTION - No need to edit below this line
    # ==========================================================================
    
    # Initialize finder
    finder = NeuronBridgeFinder(
        use_cache=use_cache,
        verbose=verbose,
        separate_splitgal4=separate_splitgal4,
        match_type=match_type,                    # Set default match algorithm
        region=region,                            # Set region filter
        max_workers=max_workers,                  # Parallel processing workers
    )
    
    # Run batch search (with optional image download)
    # match_type parameter is now optional - uses instance setting by default
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
        summary_background_color=summary_background_color,
    )
    
    # Display summary
    if not results.empty:
        print(f"\n{'='*60}")
        print(f"📊 Summary")
        print('='*60)
        print(f"   Total matches: {len(results)}")
        
        if 'source_query' in results.columns:
            print(f"\n   By source query:")
            for q, count in results['source_query'].value_counts().items():
                print(f"      {q}: {count}")
        
        if 'source_dataset' in results.columns:
            print(f"\n   By source dataset:")
            for ds, count in results['source_dataset'].value_counts().items():
                print(f"      {ds}: {count}")
        
        if 'source_type' in results.columns:
            unique_types = results['source_type'].dropna().unique()
            if len(unique_types) > 0:
                print(f"\n   Matched neuron types: {len(unique_types)}")
                for t in sorted(unique_types)[:10]:
                    if t:  # Skip empty strings
                        print(f"      {t}")
                if len(unique_types) > 10:
                    print(f"      ... and {len(unique_types) - 10} more")
        
        # Show top results from summary file (sorted by weighted_score)
        print(f"\n   Top lines (sorted by weighted_score):")
        
        # Try to load from summary file for better display
        summary_file = os.path.join(output_dir, 'line_summary.csv') if output_dir else None
        summary_df = None
        if summary_file and os.path.exists(summary_file):
            summary_df = pd.read_csv(summary_file)
        
        if summary_df is not None and not summary_df.empty and 'weighted_score' in summary_df.columns:
            # Use summary file with weighted_score
            display_cols = ['line', 'weighted_score', 'coverage_ratio', 'match_count', 'matched_types']
            display_cols = [c for c in display_cols if c in summary_df.columns]
            if display_cols:
                print(summary_df[display_cols].head(10).to_string(index=False))
            else:
                print(summary_df.head(10).to_string(index=False))
        elif 'min_score_per_dataset' in results.columns:
            display_cols = ['line', 'score', 'min_score_per_dataset', 'datasets_labeled', 'source_type']
            display_cols = [c for c in display_cols if c in results.columns]
            if display_cols:
                # Group by line for display
                print(results[display_cols].drop_duplicates('line').head(10).to_string(index=False))
        else:
            display_cols = ['line', 'library', 'score', 'source_query', 'source_type']
            display_cols = [c for c in display_cols if c in results.columns]
            if display_cols:
                print(results[display_cols].drop_duplicates('line').head(10).to_string(index=False))
            else:
                print(results.drop_duplicates('line').head(10).to_string(index=False))
        
        # Notice about specificity/selectivity analysis
        print(f"\n{'='*60}")
        print("💡 Tip: For specificity/selectivity analysis of found lines,")
        print("   use NeuronBridge_Colabel.py with the top lines from these results.")
        print("   This provides co-labeling matrices, expression analysis, and")
        print("   detailed breakdowns of which neuron types each line labels.")
        print('='*60)
    
    print("\n✅ Done!")
    t1 = time.time()
    elapsed = t1 - t0
    print(f"⏱️ Elapsed time: {elapsed/60:.2f} minutes\n")
