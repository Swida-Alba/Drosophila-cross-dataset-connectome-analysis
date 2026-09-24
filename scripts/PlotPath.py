# DROCAT edit-to-run TEMPLATE, not a CLI. It mirrors the UI's
# "plot_path" tool (registry
# ui/runner.py TOOL_REGISTRY; payload built in ui/tabs/visualization.py), and its
# values below are pinned to ui/config.py DEFAULTS so an unedited
# run reproduces what the tab sends. Copy to
# archive/scripts_local/ before a real scientific run
# (skills/drocat-usage/SKILL.md:66); the relative paths assume a
# working directory of scripts/.
import sys
from pathlib import Path
import warnings

# Add vispath-subproject to Python path for local development
vispath_pkg_path = Path(__file__).parent.parent / 'vispath-subproject' / 'src'
if vispath_pkg_path.exists():
    sys.path.insert(0, str(vispath_pkg_path))

warnings.filterwarnings("ignore")

from vispath_pkg import VisualizePath

if __name__ == '__main__':
    
    # =========================================================================
    # COLOR SCHEMES - Choose one by uncommenting
    # =========================================================================
    
    # Default (Blue-Green-Red)
    # source_color = '#1f77b4'                        # Source: Blue
    # intermediate_color = 'rgba(44, 160, 44, 1.0)'   # Intermediate: Green with 20% opacity
    # target_color = '#d62728'                        # Target: Red
    # link_color = 'rgba(100,100,100,0.6)'            # Edges: Gray with 30% opacity
    
    # Warm Theme (Red-Orange-Gold)
    # source_color = '#FF6B6B'                      # Source: Red
    # intermediate_color = '#FFA500'                # Intermediate: Orange
    # target_color = '#FFD700'                      # Target: Gold
    # link_color = 'rgba(255,107,107,0.3)'         # Edges: Red with 30% opacity
    
    # Cool Theme (Blue-Cyan-Green)
    source_color = '#4A90E2'              # Source: Blue
    intermediate_color = '#50E3C2'        # Intermediate: Cyan
    target_color = '#B8E986'              # Target: Light Green
    link_color = 'rgba(74,144,226,0.3)'  # Links: Blue
    
    # Purple Theme (Purple-Lavender-Pink)
    # source_color = '#9C27B0'              # Source: Purple
    # intermediate_color = '#BA68C8'        # Intermediate: Lavender
    # target_color = '#FF1493'              # Target: Deep Pink
    # link_color = 'rgba(156,39,176,0.3)'  # Links: Purple
    
    # Earth Theme (Brown-Olive-Orange)
    # source_color = '#8B4513'              # Source: Brown
    # intermediate_color = '#808000'        # Intermediate: Olive
    # target_color = '#FF8C00'              # Target: Dark Orange
    # link_color = 'rgba(139,69,19,0.3)'   # Links: Brown
    
    # Ocean Theme (Navy-Teal-Aqua)
    # source_color = '#000080'              # Source: Navy
    # intermediate_color = '#008080'        # Intermediate: Teal
    # target_color = '#00FFFF'              # Target: Aqua
    # link_color = 'rgba(0,128,128,0.3)'   # Links: Teal
    
    # Monochrome (Dark-Medium-Light Gray)
    # source_color = '#333333'              # Source: Dark Gray
    # intermediate_color = '#888888'        # Intermediate: Medium Gray
    # target_color = '#CCCCCC'              # Target: Light Gray
    # link_color = 'rgba(100,100,100,0.3)' # Links: Gray
    
    # High Contrast (Black-Gray-White)
    # source_color = '#000000'              # Source: Black
    # intermediate_color = '#666666'        # Intermediate: Gray
    # target_color = '#FFFFFF'              # Target: White (with border)
    # link_color = 'rgba(0,0,0,0.3)'       # Links: Black
    
    # Pastel Theme (Soft colors)
    # source_color = '#FFB3BA'              # Source: Pastel Pink
    # intermediate_color = '#BAFFC9'        # Intermediate: Pastel Green
    # target_color = '#BAE1FF'              # Target: Pastel Blue
    # link_color = 'rgba(255,179,186,0.3)' # Links: Pastel Pink
    
    # Neon Theme (Bright colors)
    # source_color = '#FF00FF'              # Source: Magenta
    # intermediate_color = '#00FF00'        # Intermediate: Lime
    # target_color = '#00FFFF'              # Target: Cyan
    # link_color = 'rgba(255,0,255,0.3)'   # Links: Magenta
    
    # =========================================================================
    # MAIN VISUALIZATION
    # =========================================================================
    
    # Point to your FindAllPath results
    # Change this path to match your actual output folder
    # 
    # NEW FEATURES:
    # - Supports both .xlsx and .csv files
    # - If path_file is None, empty, or not found, a file picker dialog will open
    # - For Excel files, if sheet_name is None, you'll be prompted to select a sheet
    # - For CSV files, sheet_name is ignored
    # 
    # Examples:
    # path_file = None  # Opens file picker
    # path_file = 'my_paths.csv'  # Load CSV directly (sheet_name ignored)
    # path_file = 'my_paths.xlsx'  # Load Excel, auto-select sheet if not specified
    
    # path_file = '../local_data/connection_data/aMe12_R_to_PPL103_R/allpaths_L3w10r0_01p0_20251027_213354/aMe12_R_to_PPL103_R_allpaths_info.xlsx'
    path_file = 'load_custom_data.xlsx'
    
    # Create visualizations with custom colors
    vp = VisualizePath(
        path_file=path_file,
        sheet_name=0,             # or 'path_bodyId' for bodyId-level paths (None = auto-select)
        output_folder=None,                 # None = auto-creates 'selected_paths' folder
        source_color=source_color,          # Custom source node color
        intermediate_color=intermediate_color,  # Custom intermediate node color
        target_color=target_color,          # Custom target node color
        link_color=link_color,              # Custom link color
        highlight_color='#FFFF80',      # Light yellow for highlighted nodes/edges
        network_layout='hierarchical',      # 'hierarchical', 'spring', 'circular', 'distributed'
        edge_width_scale='sqrt',            # Edge width scaling method
        showfig=True,                        # Open visualizations in browser
        max_edge_width=30,
        generate_empty_network=True, # Set to True to generate empty network for manual editing, ignore path_file
        straight_reciprocal_edges=False,
    )
    
    # Generate all visualizations
    conn_df, G = vp.visualize()
    
    print(f"\n✓ Visualization complete!")
    print(f"✓ Created {len(conn_df) if conn_df is not None else 0} connections")
    print(f"✓ Network has {G.number_of_nodes() if G is not None else 0} nodes and {G.number_of_edges() if G is not None else 0} edges")
    
    # =========================================================================
    # OPTIONAL: Filter paths before visualization
    # =========================================================================
    
    # Uncomment below to filter high-quality paths
    """
    import pandas as pd
    s
    # Read all paths
    all_paths = pd.read_excel(path_file, sheet_name='path_type')
    print(f"Total paths: {len(all_paths)}")
    
    # Filter: Keep only high-probability, short paths
    high_quality = all_paths[
        (all_paths['traversal_probability'] > 0.5) &
        (all_paths['inter_layer_num'] <= 2)
    ]
    print(f"High-quality paths: {len(high_quality)}")
    
    # Visualize filtered paths
    vp_filtered = VisualizePath(
        path_file=high_quality,  # Pass DataFrame directly!
        output_folder='./high_quality_paths',
        network_layout='hierarchical',
        showfig=True
    )
    
    conn_df_filtered, G_filtered = vp_filtered.visualize()
    """
    
    # =========================================================================
    # OPTIONAL: Custom colors
    # =========================================================================
    
    # Uncomment below for custom color scheme
    """
    vp_custom = VisualizePath(
        path_file=path_file,
        output_folder='./custom_colors',
        node_color=['#FF6B6B', '#FFA500'],  # Red-orange theme
        target_color='#FFD700',              # Gold
        network_layout='spring',
        showfig=True
    )
    
    vp_custom.visualize()
    """
    
    print("\n" + "="*80)
    print("Output files created:")
    print("  • sankey_selected_paths.html - Flow-based diagram")
    print("  • network_selected_paths.html - Interactive network (drag, hide, hover)")
    print("  • selected_paths_connections.xlsx - Connection data")
    print("="*80)
    print("\nColor scheme used:")
    print(f"  • Source nodes: {source_color}")
    print(f"  • Intermediate nodes: {intermediate_color}")
    print(f"  • Target nodes: {target_color}")
    print(f"  • Links: {link_color}")
    print("\nTip: Uncomment different color themes at the top of this file to try them!")
    print("="*80)
