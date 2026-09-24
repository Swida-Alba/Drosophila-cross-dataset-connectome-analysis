#!/usr/bin/env python
"""
Example: Connectivity Profile-Based Homolog Finding

This example demonstrates how to use the HomologFinder module to find potential
homologs of neurons across different connectome datasets.

Key Features:
    - Module-level defaults: Set source, source_dataset, target_dataset once
    - Fast search: Adjacency expansion for efficient candidate discovery
    - 1-hop/2-hop hybrid: Uses ConnectivityProfiler for 2-hop expansion of untyped
    - Automatic saving: Both bodyId-level and type-level results always saved
    - Skeleton visualization: Optionally visualize top candidates with VisualizeSkeleton

Profile Construction Rules (consistent with ConnectivityProfiler):
    - top_k: Top K partners per direction by synapse weight (default: 25)
    - top_m: Minimum unique partner types to ensure (default: 5)
    - Dynamic expansion: If top_k yields < top_m types, expand K
    - expand_untyped_2hop: Fetch 2-hop typed partners for untyped 1-hop (default: True)

Output Files (always saved when output_dir is set):
    - bodyid_results.csv: BodyId-level comparisons (sorted by source_bodyId, jaccard)
    - type_summary.csv: Type-level aggregated summary (avg/best/std metrics)
    - type_level_results.csv: Pooled all-adjacency type-level ranking
    - homolog_results.csv: Legacy format (rank_union-ordered)
    - visualizations/: Skeleton visualizations (if visualize_skeleton=True)

Finding Methods:
    - find_homologs(): Comprehensive search (builds all target profiles)
    - find_homologs_fast(): Fast search via adjacency expansion

Author: Example script for drocat
"""
# DROCAT edit-to-run TEMPLATE, not a CLI. It mirrors the UI's
# "find_homologs" tool (registry
# ui/runner.py TOOL_REGISTRY; payload built in ui/tabs/connectivity.py), and its
# values below are pinned to ui/config.py DEFAULTS so an unedited
# run reproduces what the tab sends. Copy to
# archive/scripts_local/ before a real scientific run
# (skills/drocat-usage/SKILL.md:66); the relative paths assume a
# working directory of scripts/.

import sys
from pathlib import Path

# Add repo src/ to path to force loading the in-repo comparison module (avoids picking up any older installed version)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from comparison.profile_comparator import HomologFinder

if __name__ == "__main__":
    # Token automatically loaded from config_local.json (recommended) or set token='' here
    finder = HomologFinder(
        source='aMe12', # type or bodyId
        source_dataset='male-cns:v0.9',
        target_dataset='hemibrain:v1.2.1',
        
        output_dir='../local_data/homolog_finding',
        # Off by default, like the ctor (profile_comparator.py:2580): this
        # opens one browser page per visualized candidate, and a template that
        # turns it on ships five tabs to a run that came for a CSV.
        visualize_skeleton=False,  # True to plot the top candidates in 3D
        visualize_top_n=5,         # Number of candidates to visualize
        verbose=True,
        # jaccard/100 are the ctor and UI defaults
        # (ui/config.py:546 SIMILARITY_METRICS leads with jaccard); every
        # metric is computed regardless, this only sorts the table.
        similarity_metric='jaccard',
        top_n=100,
        vector_prefiltering=True,
    )
    
    # Run using defaults - no arguments needed!
    results1 = finder.find_homologs_fast()
    
    # results2 = finder.direct_comparison(
    #     neurons_a='aMe12',
    #     neurons_b='aMe12',
    #     dataset_a='flywire_FAFB_v783',
    #     dataset_b='male-cns:v0.9',
    #     )
    # results2 = finder.find_homologs()
